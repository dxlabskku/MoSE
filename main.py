import torch
import json
import random
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os
import autogen
import argparse
import re
from prompt import get_prompt
from MoSE import (
    is_complex,
    single_agent_response,
    get_joint_analysis,
    get_strategy,
    response_with_strategy_mixture,
    refine_with_strategy_mixture,
    format_retrieved_examples,
    get_mixture_strategy_names,
)


from datetime import datetime



def get_embeddings(model, targets):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    targets = [target[0] for target in targets]
    embeddings = model.encode(targets).tolist()
    return embeddings


def is_lightweight_closing_turn(post):
    text = " ".join(str(post or "").lower().split())
    if not text:
        return False, "empty_post"

    tokens = re.findall(r"[a-z']+", text)
    if len(tokens) > 14:
        return False, "too_long"

    if "?" in text:
        return False, "question_turn"

    risk_patterns = [
        r"\bsuicid",
        r"\bkill\b",
        r"\bdie\b",
        r"\bdeath\b",
        r"\bdead\b",
        r"\bhopeless\b",
        r"\bworthless\b",
        r"\bpanic\b",
        r"\bscared\b",
        r"\bafraid\b",
        r"\bcan't\b",
        r"\bcannot\b",
        r"\banymore\b",
        r"\bno one\b",
        r"\balone\b",
        r"\bhate\b",
        r"\bdone\b",
    ]
    if any(re.search(pattern, text) for pattern in risk_patterns):
        return False, "risk_or_distress_keyword"

    if re.search(r"\bthanks to\b", text):
        return False, "causal_thanks_to"

    strong_closing_patterns = [
        r"\bthank(s| you)?\b",
        r"\bappreciate\b",
        r"\bthat helps?\b",
        r"\bthis helps?\b",
        r"\bhelped\b",
        r"\bmakes sense\b",
        r"\bsounds good\b",
        r"\bgood idea\b",
        r"\bi'?ll try\b",
        r"\bi will try\b",
        r"\bi can try\b",
        r"\bi'?ll do\b",
        r"\bbye\b",
        r"\bgoodbye\b",
        r"\btalk later\b",
    ]
    if any(re.search(pattern, text) for pattern in strong_closing_patterns):
        return True, "short_closing_gratitude_or_acceptance"

    weak_agreement_patterns = [
        r"^ok(ay)?[.!]*$",
        r"^yeah[.!]*$",
        r"^yes[.!]*$",
        r"^yep[.!]*$",
        r"^sure[.!]*$",
        r"^right[.!]*$",
    ]
    if any(re.search(pattern, text) for pattern in weak_agreement_patterns):
        return True, "very_short_agreement"

    return False, "no_shortcut_pattern"


def clean_strategy(strategy):
    cleaned_strategy = []
    for s in strategy:
        canonical = None
        if s.lower() == "question":
            canonical = "Question"
        elif s.lower() == "restatement or paraphrasing":
            canonical = "Restatement or Paraphrasing"
        elif s.lower() == "reflection of feelings":
            canonical = "Reflection of feelings"
        elif s.lower() == "self-disclosure":
            canonical = "Self-disclosure"
        elif s.lower() == "affirmation and reassurance":
            canonical = "Affirmation and Reassurance"
        elif s.lower() == "providing suggestions":
            canonical = "Providing Suggestions"
        elif s.lower() == "information":
            canonical = "Information"
        elif s.lower() == "others":
            canonical = "Others"
        if canonical and canonical not in cleaned_strategy:
            cleaned_strategy.append(canonical)
    return cleaned_strategy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="dataset/ESConv.json")
    parser.add_argument("--model_path", type=str, default="all-roberta-large-v1")
    parser.add_argument("--llm_name", type=str, default="qwen2.5:32b")
    parser.add_argument("--cache_path_root", type=str, default=".cache_single_count")
    parser.add_argument("--save_path", type=str, default="results_single_count")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--retrieval_top_k", type=int, default=10)
    parser.add_argument("--mixture_agent_num", type=int, default=3)
    parser.add_argument("--max_mixture_strategies", type=int, default=3)
    parser.add_argument("--generation_top_k_examples", type=int, default=5)
    parser.add_argument("--refiner_top_k_examples", type=int, default=3)
    parser.add_argument(
        "--refiner_use_strategy_examples",
        dest="refiner_use_strategy_examples",
        action="store_true",
        help="Allow the mixture refiner to see top-k strategy reference examples.",
    )
    parser.add_argument(
        "--no_refiner_use_strategy_examples",
        dest="refiner_use_strategy_examples",
        action="store_false",
        help="Disable top-k strategy reference examples in the mixture refiner.",
    )
    parser.set_defaults(refiner_use_strategy_examples=True)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_quadruple(model, dataset):
    path = "./embeddings.txt"
    if not os.path.exists(path):
        with open(path, 'a') as txt:
            for sample in tqdm(dataset[100:]):
                targets = []
                dialog = sample['dialog']
                for count in range(len(dialog)-1):
                    if dialog[count]['speaker'] == "seeker" and dialog[count+1]['speaker'] == "supporter":
                        targets.append((dialog[count]['content'].strip(), dialog[count+1]['content'].strip(), dialog[count+1]['annotation']['strategy']))
                embeddings = get_embeddings(model, targets)
                for triple, embedding in zip(targets, embeddings):
                    post = triple[0]
                    response = triple[1]
                    strategy = triple[2]
                    line = f"{post}__SEP__{response}__SEP__{strategy}__SEP__{embedding}".replace("\n", "\\n") + "\n"
                    txt.write(line)
        with open(path, "r") as txt:
            quadruple = txt.readlines()
    else:
        with open(path, "r") as txt:
            quadruple = txt.readlines()

    return quadruple


def json2natural(history):
    natural_language = ""
    for u in history:
        content = u["content"].strip()
        role = u["role"].capitalize() if "role" in u.keys() else u["speaker"].capitalize()
        if role == "Supporter":
            role = "Assistant"
        if role == "Seeker":
            role = "User"

        natural_language += f"{role}: {content} "
    return natural_language.strip()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_path, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
    samples = dataset[:args.num_samples]
    #samples = dataset[:20]
    config_list = autogen.config_list_from_json(
        env_or_file="OAI_CONFIG_LIST",
        file_location=".",
        filter_dict={
            "model": [args.llm_name],
        }
    )

    model = SentenceTransformer(args.model_path)
    quadruple = get_quadruple(model, dataset)

    ret = []
    for sample in tqdm(samples):
        dialog = sample['dialog']
        count = 0
        history = []

        while True:
            save = {}
            if count == len(dialog):
                break

            if count != 0 and dialog[count]["speaker"] == "supporter":

                if (count < len(dialog) - 1 and dialog[count+1]['speaker'] != "supporter") or (count == len(dialog)-1):
                    save["strategy"] = dialog[count]['annotation']['strategy']
                    save["reference"] = dialog[count]['content'].strip()

                    context = json2natural(history)
                    save["context"] = context
                    post = history[-1]['content']

                    history.append({
                        "content": dialog[count]["content"].strip(),
                        "role": "user" if dialog[count]["speaker"] == "seeker" else "assistant"
                    })
                    count += 1

                elif count < len(dialog) - 1 and dialog[count+1]['speaker'] == "supporter":
                    save["strategy"] = f"{dialog[count]['annotation']['strategy']} and {dialog[count+1]['annotation']['strategy']}"
                    save["reference"] = dialog[count]['content'].strip() + ' ' + dialog[count+1]['content'].strip()

                    context = json2natural(history)
                    save["context"] = context
                    post = history[-1]['content']

                    history.append({
                        "content": dialog[count]["content"].strip(),
                        "role": "user" if dialog[count]["speaker"] == "seeker" else "assistant"
                    })
                    history.append({
                        "content": dialog[count+1]["content"].strip(),
                        "role": "user" if dialog[count+1]["speaker"] == "seeker" else "assistant"
                    })

                    count += 2

                shortcut_simple, shortcut_reason = is_lightweight_closing_turn(post)
                if count <= 5 or shortcut_simple or not is_complex(get_prompt("behavior_control").format(context=context), config_list=config_list, cache_path_root=args.cache_path_root):
                    response = single_agent_response(get_prompt("zero_shot").format(context=context), config_list=config_list, cache_path_root=args.cache_path_root)
                    save["response"] = response
                    save["pred_strategy"] = "None"
                    save["shortcut_simple"] = shortcut_simple
                    save["shortcut_reason"] = shortcut_reason if shortcut_simple else ("early_turn" if count <= 5 else "not_complex")

                else:
                    save["shortcut_simple"] = False
                    save["shortcut_reason"] = shortcut_reason
                    joint_analysis = get_joint_analysis(
                        get_prompt("get_joint_analysis").format(context=context),
                        config_list,
                        args.cache_path_root,
                    )
                    emotion = joint_analysis["emotion"]
                    cause = joint_analysis["cause"]
                    intention = joint_analysis["intention"]
                    emo_and_reason = joint_analysis["emo_and_reason"]
                    cau_and_reason = joint_analysis["cau_and_reason"]
                    int_and_reason = joint_analysis["int_and_reason"]
                    pred_strategy, pairs, strategy_mixtures, count_info = get_strategy(
                        emo_and_reason,
                        cau_and_reason,
                        int_and_reason,
                        context,
                        post,
                        quadruple,
                        model,
                        config_list,
                        cache_path_root=args.cache_path_root,
                        n=args.retrieval_top_k,
                        max_strategies=args.max_mixture_strategies,
                        agent_num=args.mixture_agent_num,
                    )
                    pred_strategy = clean_strategy(pred_strategy)
                    for strategy_mixture in strategy_mixtures:
                        strategy_mixture["strategies"] = [
                            item for item in strategy_mixture.get("strategies", [])
                            if item.get("strategy") in pred_strategy
                        ]
                    strategy_mixtures = [
                        strategy_mixture for strategy_mixture in strategy_mixtures
                        if strategy_mixture.get("strategies")
                    ]
                    save["emotion"], save["cause"], save["intention"] = emotion, cause, intention
                    save["emo_and_reason"] = emo_and_reason
                    save["cau_and_reason"] = cau_and_reason
                    save["int_and_reason"] = int_and_reason
                    save["joint_analysis_raw"] = joint_analysis["raw"]
                    save["analysis_mode"] = "joint"
                    save["strategy_count_requested"] = count_info["requested_count"]
                    save["strategy_count_effective"] = count_info["effective_count"]
                    save["strategy_count_reasoning"] = count_info["count_reasoning"]
                    save["strategy_selector_raw"] = [strategy_mixture.get("raw", "") for strategy_mixture in strategy_mixtures]
                    if len(pred_strategy) == 0 or len(strategy_mixtures) == 0:
                        response = single_agent_response(get_prompt("zero_shot").format(context=context), config_list=config_list, cache_path_root=args.cache_path_root)
                        save["response"] = response
                        save["pred_strategy"] = "None"
                    else:
                        refiner_examples = format_retrieved_examples(
                            pairs,
                            max_examples=args.refiner_top_k_examples,
                            strategies=pred_strategy,
                        ) if args.refiner_use_strategy_examples else "None"

                        candidate_responses = []
                        for strategy_mixture in strategy_mixtures:
                            mixture_strategies = get_mixture_strategy_names(strategy_mixture)
                            generation_examples = format_retrieved_examples(
                                pairs,
                                max_examples=args.generation_top_k_examples,
                                strategies=mixture_strategies,
                            )
                            draft_response, draft_raw = response_with_strategy_mixture(
                                context=context,
                                emo_and_reason=emo_and_reason,
                                cau_and_reason=cau_and_reason,
                                int_and_reason=int_and_reason,
                                strategy_mixture=strategy_mixture,
                                top_examples=generation_examples,
                                config_list=config_list,
                                cache_path_root=args.cache_path_root,
                            )
                            candidate_responses.append({
                                "candidate_id": strategy_mixture.get("candidate_id") or f"M{len(candidate_responses) + 1}",
                                "agent_name": strategy_mixture.get("agent_name", ""),
                                "agent_perspective": strategy_mixture.get("agent_perspective", ""),
                                "strategy_mixture": strategy_mixture,
                                "pred_strategy_mixture": mixture_strategies,
                                "response": draft_response,
                                "raw": draft_raw,
                                "generation_top_examples": generation_examples,
                            })

                        candidate_responses_text = "\n\n".join(
                            [
                                f"Candidate {candidate['candidate_id']}\n"
                                f"Agent: {candidate.get('agent_name', '')}\n"
                                f"Strategies: {', '.join(candidate.get('pred_strategy_mixture', []))}\n"
                                f"Response: {candidate.get('response', 'None')}"
                                for candidate in candidate_responses
                            ]
                        )
                        default_candidate = candidate_responses[0]
                        selected_strategy_mixture = default_candidate["strategy_mixture"]
                        draft_response = default_candidate["response"]

                        primary_strategy, response, refiner_raw, refiner_selected_candidate_id = refine_with_strategy_mixture(
                            context=context,
                            emo_and_reason=emo_and_reason,
                            cau_and_reason=cau_and_reason,
                            int_and_reason=int_and_reason,
                            strategy_mixture=selected_strategy_mixture,
                            draft_response=draft_response,
                            top_examples=refiner_examples,
                            config_list=config_list,
                            cache_path_root=args.cache_path_root,
                            candidate_responses_text=candidate_responses_text,
                        )
                        selected_candidate = next(
                            (
                                candidate for candidate in candidate_responses
                                if candidate["candidate_id"] == refiner_selected_candidate_id
                            ),
                            default_candidate,
                        )
                        selected_strategy_mixture = selected_candidate["strategy_mixture"]
                        primary_strategy = get_mixture_strategy_names(selected_strategy_mixture)[0]

                        save["ori_response"] = selected_candidate.get("response", draft_response)
                        save["response_generation_raw"] = selected_candidate.get("raw", "")
                        save["pred_strategy"] = primary_strategy
                        save["pred_strategy_mixture"] = get_mixture_strategy_names(selected_strategy_mixture)
                        save["all_pred_strategies"] = pred_strategy
                        save["strategy_mixture"] = selected_strategy_mixture
                        save["strategy_mixture_candidates"] = strategy_mixtures
                        save["candidate_responses"] = candidate_responses
                        save["selected_candidate_id"] = selected_candidate.get("candidate_id")
                        save["mixture_judge_raw"] = ""
                        save["selection_source"] = "refiner"
                        save["candidate_responses_text"] = candidate_responses_text
                        save["generation_top_examples"] = selected_candidate.get("generation_top_examples", "")
                        save["refiner_top_examples"] = refiner_examples
                        save["mixture_used"] = len(strategy_mixtures) > 1 or len(save["pred_strategy_mixture"]) > 1
                        save["mixture_gate_reason"] = "multi_agent_mixture_candidates"
                        save["refiner_raw"] = refiner_raw
                        save["refiner_use_strategy_examples"] = args.refiner_use_strategy_examples
                        save["self_reflection_used"] = True
                        save["response"] = response

                ret.append(save)

            else:

                history.append({
                    "content": dialog[count]["content"].strip(),
                    "role": "user" if dialog[count]["speaker"] == "seeker" else "assistant"
                })
                count += 1

    result_path = os.path.join(save_dir, "results.json")
    with open(result_path, 'w') as f:
        json.dump(ret, f, indent=4)
