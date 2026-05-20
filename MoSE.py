import autogen
import json
import heapq
import re
from autogen import Cache
import numpy as np
from sentence_transformers import util
from collections import Counter
from prompt import get_prompt


def _no_temperature_models():
    return ("gpt-5", "o1", "o3", "o4")

def make_llm_config(config_list, cache_seed, max_completion_tokens):
    model = config_list[0].get("model", "") if config_list else ""
    cfg = {
        "config_list": config_list,
        "cache_seed": cache_seed,
        "max_completion_tokens": max_completion_tokens,
    }
    if not any(model.startswith(prefix) for prefix in _no_temperature_models()):
        cfg["temperature"] = 0.0
    return cfg


VALID_STRATEGIES = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
]


MIXTURE_AGENT_PERSPECTIVES = [
    "Select the most context-grounded strategy mixture supported by the retrieved examples.",
    "Select an alternative valid strategy mixture that differs from the first while remaining supported by the retrieved examples.",
    "Select a conservative strategy mixture that avoids unsupported advice and generic responses.",
]


def normalize_strategy_name(strategy):
    if not strategy:
        return None
    cleaned = re.sub(r"[^a-zA-Z\- ]+", " ", strategy).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    for valid_strategy in VALID_STRATEGIES:
        if cleaned == valid_strategy.lower():
            return valid_strategy
    return None


def extract_valid_strategies(text, max_strategies=3):
    matches = []
    for strategy in VALID_STRATEGIES:
        for match in re.finditer(re.escape(strategy), text or "", re.IGNORECASE):
            matches.append((match.start(), strategy))

    strategies = []
    for _, strategy in sorted(matches, key=lambda item: item[0]):
        if strategy not in strategies:
            strategies.append(strategy)
        if len(strategies) >= max_strategies:
            break
    return strategies


def fallback_strategies_from_examples(examples, max_strategies=3):
    raw_strategies = re.findall(r"\[([^\]]+)\]", examples or "")
    normalized = [normalize_strategy_name(strategy) for strategy in raw_strategies]
    normalized = [strategy for strategy in normalized if strategy]
    if not normalized:
        return []
    counter = Counter(normalized)
    return [strategy for strategy, _ in counter.most_common(max_strategies)]


def build_strategy_mixture(
    strategies,
    raw_response="",
    mixing_plan="",
    agent_name="",
    agent_perspective="",
    candidate_id=None,
):
    unique_strategies = []
    for strategy in strategies:
        normalized = normalize_strategy_name(strategy)
        if normalized and normalized not in unique_strategies:
            unique_strategies.append(normalized)

    weights_by_count = {
        1: [1.0],
        2: [0.7, 0.3],
        3: [0.6, 0.25, 0.15],
    }
    weights = weights_by_count.get(len(unique_strategies), [])

    mixture = []
    for idx, strategy in enumerate(unique_strategies):
        mixture.append({
            "strategy": strategy,
            "role": "primary" if idx == 0 else "secondary",
            "weight": weights[idx] if idx < len(weights) else 0.0,
        })

    return {
        "candidate_id": candidate_id,
        "agent_name": agent_name,
        "agent_perspective": agent_perspective,
        "strategies": mixture,
        "mixing_plan": mixing_plan.strip(),
        "raw": raw_response,
    }


def format_strategy_mixture(strategy_mixture):
    if isinstance(strategy_mixture, list):
        strategy_mixture = build_strategy_mixture(strategy_mixture)

    strategies = strategy_mixture.get("strategies", []) if strategy_mixture else []
    primary = next((item["strategy"] for item in strategies if item.get("role") == "primary"), "None")
    secondary = [item["strategy"] for item in strategies if item.get("role") == "secondary"]
    mixing_plan = strategy_mixture.get("mixing_plan", "") if strategy_mixture else ""
    agent_name = strategy_mixture.get("agent_name", "") if strategy_mixture else ""
    agent_perspective = strategy_mixture.get("agent_perspective", "") if strategy_mixture else ""

    secondary_text = ", ".join(secondary) if secondary else "None"
    if not mixing_plan:
        mixing_plan = "Use the primary strategy as the main response style and secondary strategies only when they naturally support it."

    lines = []
    if agent_name:
        lines.append(f"Agent: {agent_name}")
    if agent_perspective:
        lines.append(f"Agent Perspective: {agent_perspective}")
    lines.extend([
        f"Primary Strategy: {primary}",
        f"Secondary Strategies: {secondary_text}",
        f"Mixing Plan: {mixing_plan}",
    ])
    return "\n".join(lines)


def get_mixture_strategy_names(strategy_mixture):
    strategies = strategy_mixture.get("strategies", []) if strategy_mixture else []
    return [item["strategy"] for item in strategies if item.get("strategy")]


def parse_best_candidate_id(raw_response, fallback_id):
    match = re.search(
        r"(?:Best|Selected) Candidate:\s*(?:Candidate\s*)?([A-Za-z0-9_-]+)",
        raw_response or "",
        re.IGNORECASE,
    )
    if not match:
        return fallback_id
    return match.group(1).strip()


def format_retrieved_examples(pairs, max_examples=3, strategies=None):
    if max_examples <= 0:
        return "None"

    strategy_list = [normalize_strategy_name(s) for s in (strategies or [])]
    strategy_list = [s for s in strategy_list if s]
    strategy_set = set(strategy_list)

    # at least one example per strategy
    per_strategy = {s: None for s in strategy_list}
    for post, strategy_response in pairs:
        match = re.match(r"\[([^\]]+)\]", strategy_response)
        pair_strategy = normalize_strategy_name(match.group(1)) if match else None
        if pair_strategy in per_strategy and per_strategy[pair_strategy] is None:
            per_strategy[pair_strategy] = (post, strategy_response)
        if all(v is not None for v in per_strategy.values()):
            break

    selected = [v for v in per_strategy.values() if v is not None]

    # fill remaining slots with additional strategy-matched examples
    if len(selected) < max_examples:
        for post, strategy_response in pairs:
            if len(selected) >= max_examples:
                break
            item = (post, strategy_response)
            if item not in selected:
                match = re.match(r"\[([^\]]+)\]", strategy_response)
                pair_strategy = normalize_strategy_name(match.group(1)) if match else None
                if strategy_set and pair_strategy in strategy_set:
                    selected.append(item)

    # fill remaining slots with any example
    if len(selected) < max_examples:
        for post, strategy_response in pairs:
            if len(selected) >= max_examples:
                break
            item = (post, strategy_response)
            if item not in selected:
                selected.append(item)

    if not selected:
        return "None"

    return "\n\n".join(
        f"{idx}. {post}\n{strategy_response}"
        for idx, (post, strategy_response) in enumerate(selected, start=1)
    )


def parse_mixing_plan(raw_response):
    match = re.search(r"Mixing Plan:\s*(.*)", raw_response or "", re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    plan = match.group(1).strip()
    plan = re.split(r"\n\s*(?:Reasoning|Primary Strategy|Secondary Strategies):", plan, maxsplit=1, flags=re.IGNORECASE)[0]
    return plan.strip()


def _extract_expert_blocks(raw_response):
    pattern = re.compile(
        r"^\s*Expert\s+(\d+)\s*:?\s*\n(.*?)(?=^\s*Expert\s+\d+\s*:?\s*\n|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    blocks = []
    for match in pattern.finditer(raw_response or ""):
        expert_idx = int(match.group(1))
        block_text = match.group(2).strip()
        if not block_text:
            continue
        blocks.append((expert_idx, block_text))
    return blocks


def _parse_strategies_from_block(block_text, examples, max_strategies):
    primary = normalize_strategy_name(
        _extract_prefixed_value(block_text, "Primary Strategy:", "")
    )
    secondary_text = _extract_prefixed_value(block_text, "Secondary Strategies:", "")
    secondary = extract_valid_strategies(
        secondary_text,
        max_strategies=max(0, max_strategies - 1),
    )

    strategies = []
    if primary:
        strategies.append(primary)
    for strategy in secondary:
        if strategy not in strategies:
            strategies.append(strategy)

    if not strategies:
        strategies = extract_valid_strategies(block_text, max_strategies=max_strategies)
    if not strategies:
        strategies = fallback_strategies_from_examples(examples, max_strategies=max_strategies)
    return strategies


def is_complex(prompt, config_list, cache_path_root):
    agent = autogen.ConversableAgent(
        name='Assistant',
        system_message="You are a psychological counseling expert.",
        llm_config=make_llm_config(config_list, 2024, 100),
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        response = agent.generate_reply(
            messages=[{'content': prompt, 'role': 'user'}],
            cache=cache,
        )

    flag = True if "yes" in response.lower() else False
    return flag


def single_agent_response(prompt, config_list, cache_path_root):
    agent = autogen.ConversableAgent(
        name='Assistant',
        system_message="You are a psychological counseling expert.",
        llm_config={
            **make_llm_config(config_list, 2024, 100),
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        response = agent.generate_reply(
            messages=[{'content': prompt, 'role': 'user'}],
            cache=cache,
        )

    try:
        response = re.findall(r'Response:\s*(.*)', response)[0].strip()
    except:
        response = "None"
    return response


def _strip_think_block(response):
    if "</think>" in response:
        return response.split("</think>", 1)[1].strip()
    return response.strip()


def _extract_prefixed_value(response, prefix, default_value):
    match = re.search(rf"^{re.escape(prefix)}\s*(.*)$", response or "", re.MULTILINE)
    if not match:
        return default_value
    value = match.group(1).strip()
    return value if value else default_value


def _build_reason_text(label, value, reasoning):
    return f"{label}: {value}\nReasoning: {reasoning}"


def get_joint_analysis(prompt, config_list, cache_path_root):
    agent = autogen.ConversableAgent(
        name='Joint Perception Agent',
        system_message="You are a psychological counseling expert.",
        llm_config={
            **make_llm_config(config_list, 2024, 700),
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        raw_response = agent.generate_reply(
            messages=[{'content': prompt, 'role': 'user'}],
            cache=cache,
        )

    response = _strip_think_block(raw_response)
    emotion = _extract_prefixed_value(response, "Emotion:", "Negative")
    emotion_reasoning = _extract_prefixed_value(response, "Emotion Reasoning:", "Not mention")
    cause = _extract_prefixed_value(response, "Event:", "Not mention")
    cause_reasoning = _extract_prefixed_value(response, "Event Reasoning:", "Not mention")
    intention = _extract_prefixed_value(response, "Intention:", "Not mention")
    intention_reasoning = _extract_prefixed_value(response, "Intention Reasoning:", "Not mention")

    return {
        "emotion": emotion,
        "cause": cause,
        "intention": intention,
        "emo_and_reason": _build_reason_text("Emotion", emotion, emotion_reasoning),
        "cau_and_reason": _build_reason_text("Event", cause, cause_reasoning),
        "int_and_reason": _build_reason_text("Intention", intention, intention_reasoning),
        "raw": response,
    }


# ── Strategy Count Agent ───────────────────────────────────────────────────────

def parse_strategy_count(response_text):
    match = re.search(r'Count:\s*([1-9])', str(response_text), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 1


def count_strategy_frequency(pairs):
    counts = Counter()
    order = []
    for _, tagged_response in pairs:
        strategy = tagged_response.split("]", 1)[0].strip("[").strip()
        counts[strategy] += 1
        if strategy not in order:
            order.append(strategy)
    ranked = sorted(order, key=lambda name: (-counts[name], order.index(name)))
    return counts, ranked


def select_strategy_count(emo_and_reason, cau_and_reason, int_and_reason, context, available_strategies, config_list, cache_path_root):
    available_text = ", ".join(available_strategies) if available_strategies else ", ".join(VALID_STRATEGIES)
    prompt = f'''### You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation and inferred the user's emotional state, the event that led to the emotion, and the user's intention.

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

### Event
{cau_and_reason}

### Intention
{int_and_reason}

### Available strategies from retrieved examples
{available_text}

Decide how many strategies should be mixed in a single response. Choose:
- Count 1 when one strategy is clearly sufficient.
- Count 2 when the response should blend two complementary goals, such as comforting plus exploration, or validation plus action.
- Count 3 when the dialogue needs a richer mixture across emotional validation, exploration, and action.
- Count 4 when the user's situation is complex and multiple emotional and practical dimensions must all be addressed.
- Count 5 only when the dialogue is highly complex and requires a comprehensive response covering emotional support, validation, exploration, and action across distinct needs.

Consider the user's current emotional complexity, the counseling stage, and whether a focused or comprehensive strategy mixture is needed.

Your answer must follow this format:
Count: [1/2/3/4/5]
Reasoning: [reasoning]
'''

    agent = autogen.ConversableAgent(
        name='Strategy Count Agent',
        system_message="You are a psychological counseling expert.",
        llm_config={
            **make_llm_config(config_list, 2024, 200),
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        response = agent.generate_reply(
            messages=[{'content': prompt, 'role': 'user'}],
            cache=cache,
        )

    requested_count = parse_strategy_count(response)
    effective_count = max(1, min(requested_count, max(1, len(available_strategies))))
    return requested_count, effective_count, response


# ── Strategy selection ─────────────────────────────────────────────────────────

def select_strategy_mixture_panel(
    emo_and_reason,
    cau_and_reason,
    int_and_reason,
    context,
    examples,
    config_list,
    cache_path_root,
    max_strategies=3,
    agent_num=3,
):
    max_strategies = max(1, min(max_strategies, 3))  # cap at 3
    agent_num = max(1, agent_num)
    perspectives = []
    for idx in range(agent_num):
        perspectives.append(
            f"Expert {idx + 1}: {MIXTURE_AGENT_PERSPECTIVES[idx % len(MIXTURE_AGENT_PERSPECTIVES)]}"
        )

    n_secondary = max_strategies - 1
    if n_secondary == 0:
        secondary_format = "None"
    else:
        slots = ", ".join(f"[strategy{i+1}]" for i in range(n_secondary))
        secondary_format = f"{slots} or None"

    prompt = get_prompt("select_strategy_mixture_panel").format(
        expert_perspectives="\n".join(perspectives),
        context=context,
        emo_and_reason=emo_and_reason,
        cau_and_reason=cau_and_reason,
        int_and_reason=int_and_reason,
        examples=examples,
        max_strategies=max_strategies,
        max_secondary_strategies=n_secondary,
        agent_num=agent_num,
        secondary_format=secondary_format,
    )

    agent = autogen.ConversableAgent(
        name="Strategy Mixture Panel",
        system_message="You are a psychological counseling expert panel selecting compact strategy mixtures.",
        llm_config={
            **make_llm_config(config_list, 2024, 480),
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        raw_response = agent.generate_reply(
            messages=[{'content': prompt, 'role': 'user'}],
            cache=cache,
        )

    parsed_blocks = {expert_idx: block for expert_idx, block in _extract_expert_blocks(raw_response)}
    candidates = []
    for idx in range(agent_num):
        expert_idx = idx + 1
        perspective = MIXTURE_AGENT_PERSPECTIVES[idx % len(MIXTURE_AGENT_PERSPECTIVES)]
        block_text = parsed_blocks.get(expert_idx, "")
        strategies = _parse_strategies_from_block(
            block_text,
            examples=examples,
            max_strategies=max_strategies,
        )
        mixing_plan = parse_mixing_plan(block_text)
        candidate = build_strategy_mixture(
            strategies,
            raw_response=block_text or raw_response,
            mixing_plan=mixing_plan,
            agent_name=f"Strategy Mixture Agent {expert_idx}",
            agent_perspective=perspective,
            candidate_id=f"M{expert_idx}",
        )
        if candidate.get("strategies"):
            candidates.append(candidate)
    return candidates


def select_strategy_mixture_candidates(
    emo_and_reason,
    cau_and_reason,
    int_and_reason,
    context,
    examples,
    config_list,
    cache_path_root,
    max_strategies=3,
    agent_num=3,
):
    return select_strategy_mixture_panel(
        emo_and_reason=emo_and_reason,
        cau_and_reason=cau_and_reason,
        int_and_reason=int_and_reason,
        context=context,
        examples=examples,
        config_list=config_list,
        cache_path_root=cache_path_root,
        max_strategies=max_strategies,
        agent_num=agent_num,
    )


def get_strategy(
    emo_and_reason,
    cau_and_reason,
    int_and_reason,
    context,
    post,
    quadruple,
    model,
    config_list,
    cache_path_root="",
    n=10,
    max_strategies=3,
    agent_num=3,
):
    post_embedding = model.encode(post)
    can_embeddings = [np.array(json.loads(q.split('__SEP__')[3]), dtype=np.float32) for q in quadruple]
    similarities = util.pytorch_cos_sim(post_embedding, can_embeddings)[0].tolist()
    top_indices = heapq.nlargest(n, range(len(similarities)), key=lambda i: similarities[i])
    pairs = []
    for i in top_indices:
        post, response, strategy, _ = quadruple[i].split('__SEP__')
        pairs.append((post, f"[{strategy}] {response}"))

    examples = "\n\n".join([f"{pair[0]}\n{pair[1]}" for pair in pairs])

    # Count Agent: determine how many strategies to mix (1~5 allowed, capped at 3 by panel)
    _, available_strategies = count_strategy_frequency(pairs)
    requested_count, effective_count, count_reasoning = select_strategy_count(
        emo_and_reason, cau_and_reason, int_and_reason, context,
        available_strategies, config_list, cache_path_root,
    )
    effective_count = min(effective_count, max_strategies)
    panel_count = max(1, min(effective_count, 3))  # what panel actually uses

    strategy_mixtures = select_strategy_mixture_candidates(
        emo_and_reason=emo_and_reason,
        cau_and_reason=cau_and_reason,
        int_and_reason=int_and_reason,
        context=context,
        examples=examples,
        config_list=config_list,
        cache_path_root=cache_path_root,
        max_strategies=effective_count,
        agent_num=agent_num,
    )
    strategies = []
    for strategy_mixture in strategy_mixtures:
        for strategy in get_mixture_strategy_names(strategy_mixture):
            if strategy not in strategies:
                strategies.append(strategy)

    count_info = {
        "requested_count": requested_count,
        "effective_count": panel_count,
        "count_reasoning": count_reasoning,
    }
    return strategies, pairs, strategy_mixtures, count_info


def response_with_strategy_mixture(
    context,
    emo_and_reason,
    cau_and_reason,
    int_and_reason,
    strategy_mixture,
    top_examples,
    config_list,
    cache_path_root,
):
    prompt = get_prompt("response_with_strategy_mixture").format(
        context=context,
        emo_and_reason=emo_and_reason,
        cau_and_reason=cau_and_reason,
        int_and_reason=int_and_reason,
        strategy_mixture=format_strategy_mixture(strategy_mixture),
        top_examples=top_examples or "None",
    )

    agent = autogen.ConversableAgent(
        name='Assistant',
        system_message="You are a psychological counseling expert.",
        llm_config={
            **make_llm_config(config_list, 2024, 120),
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        raw_response = agent.generate_reply(
            messages=[{'content': prompt, 'role': 'user'}],
            cache=cache,
        )

    try:
        response = re.findall(r'Response:\s*(.*)', raw_response)[0].strip()
    except:
        response = raw_response.strip() if raw_response.strip() else "None"

    return response, raw_response


def refine_with_strategy_mixture(
    context,
    emo_and_reason,
    cau_and_reason,
    int_and_reason,
    strategy_mixture,
    draft_response,
    top_examples,
    config_list,
    cache_path_root,
    candidate_responses_text="None",
):
    prompt = get_prompt("refine_with_strategy_mixture").format(
        context=context,
        emo_and_reason=emo_and_reason,
        cau_and_reason=cau_and_reason,
        int_and_reason=int_and_reason,
        strategy_mixture=format_strategy_mixture(strategy_mixture),
        draft_response=draft_response,
        top_examples=top_examples or "None",
        candidate_responses=candidate_responses_text or "None",
    )

    agent = autogen.ConversableAgent(
        name="Mixture Refiner",
        system_message="You are a psychological counseling expert who refines one strategy-mixture response using similar examples.",
        llm_config={
            **make_llm_config(config_list, 2024, 140),
        },
        human_input_mode='NEVER'
    )

    with Cache.disk(cache_path_root=cache_path_root) as cache:
        raw_response = agent.generate_reply(
            messages=[{'content': prompt, 'role': 'user'}],
            cache=cache,
        )

    try:
        response = re.findall(r'Response:\s*(.*)', raw_response)[0].strip()
    except:
        response = raw_response.strip() if raw_response.strip() else "None"

    strategies = strategy_mixture.get("strategies", []) if strategy_mixture else []
    primary_strategy = strategies[0]["strategy"] if strategies else "None"
    selected_candidate_id = parse_best_candidate_id(raw_response, None)
    return primary_strategy, response, raw_response, selected_candidate_id
