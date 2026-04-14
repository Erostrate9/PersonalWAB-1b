import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from PUMA.graph.builder import GraphBuilder
from PUMA.graph.retriever import GraphRetriever
from PUMA.graph.reward import select_graph_dpo_pair
from PUMA.graph.serializer import build_graph_param_prompt, serialize_subgraph
from utils import PARAM_PROMPT, build_taskspe_memory, prettify_product_info, retrieve_top_k_memories, truncate_text


llama_tokenizer = None
sim_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
sim_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
sim_model.eval()
if torch.cuda.is_available():
    sim_model.to("cuda")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare parameter-generation data for DPO")
    parser.add_argument("--task_file", type=str, required=True, help="Path to the task file")
    parser.add_argument("--user_history_file", type=str, required=True, help="Path to the user history file")
    parser.add_argument("--dpo_data_file", type=str, default=None, help="Path to flat DPO candidate-score file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output results")
    parser.add_argument("--mem_token_length", type=int, default=768, help="Maximum token length of memory or graph evidence")
    parser.add_argument("--mem_length", type=int, default=100, help="Maximum flat-memory retrieval count")
    parser.add_argument("--graph_mode", action="store_true", help="Use graph-conditioned retrieval and composite rewards")
    parser.add_argument("--user_profile_file", type=str, default=None, help="Path to the user profile file")
    parser.add_argument("--all_products_file", type=str, default=None, help="Path to all products file for recommendation scoring")
    parser.add_argument("--reward_margin", type=float, default=0.05, help="Minimum chosen vs rejected composite reward gap")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Tokenizer used for truncation")
    return parser.parse_args()


def task_to_tool(task_type: str) -> str:
    if task_type == "search":
        return "search_product_by_query"
    if task_type == "recommend":
        return "get_recommendations_by_history"
    return "add_product_review"


def get_chosen_reject(options: Dict[str, float]) -> Tuple[Optional[str], Optional[str]]:
    if not options:
        return None, None
    max_score = max(options.values())
    min_score = min(options.values())
    chosen = None
    reject = None
    for key, score in options.items():
        if score == max_score and chosen is None:
            chosen = key
        if score == min_score:
            reject = key
    return chosen, reject


def build_flat_prompt(task: Dict, user_history: Dict[str, list], mem_length: int, mem_token_length: int) -> Tuple[str, str]:
    input_text = task["task"]
    task_type = task["type"]
    timestamp = task["timestamp"]
    product_info = task["target"]["product_info"]
    history = [item for item in user_history[task["user_id"]] if item["review"]["timestamp"] < timestamp]
    history = retrieve_top_k_memories(input_text, history, sim_model, sim_tokenizer, k=mem_length) if history else []

    mem = build_taskspe_memory(history, task_type)
    prefix_text = PARAM_PROMPT.replace("<Instruction>", input_text + (prettify_product_info(product_info) if task_type == "review" else ""))
    memory_text = "|".join(mem)
    tool_text = task_to_tool(task_type)
    memory_text_truncated = truncate_text(llama_tokenizer, memory_text, mem_token_length)
    prompt = prefix_text.replace("<Memory>", memory_text_truncated).replace("<Tool>", tool_text)
    return prompt, memory_text_truncated


def build_graph_prompt(
    task: Dict,
    user_history: Dict[str, list],
    user_profiles: Dict[str, dict],
    graph_builder: GraphBuilder,
    mem_token_length: int,
) -> Tuple[str, object, str]:
    filtered_history = [
        item
        for item in user_history.get(task["user_id"], [])
        if item["review"]["timestamp"] < task["timestamp"]
    ]
    graph = graph_builder.build_user_graph(
        user_id=task["user_id"],
        history=filtered_history,
        user_profile=user_profiles.get(task["user_id"], {}),
    )
    retriever = GraphRetriever(graph)
    subgraph = retriever.retrieve_subgraph(
        instruction=task["task"],
        task_type=task["type"],
        timestamp=task["timestamp"],
        product_info=task["target"]["product_info"] if task["type"] == "review" else None,
    )
    graph_text = serialize_subgraph(subgraph)
    graph_text = truncate_text(llama_tokenizer, graph_text, mem_token_length)
    prompt = build_graph_param_prompt(
        instruction=task["task"],
        tool_name=task_to_tool(task["type"]),
        graph_text=graph_text,
        product_info=task["target"]["product_info"] if task["type"] == "review" else None,
    )
    return prompt, subgraph, graph_text


def main():
    args = parse_args()
    global llama_tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

    task_file = json.load(open(args.task_file, "r"))
    user_history = json.load(open(args.user_history_file, "r"))
    dpo_data = json.load(open(args.dpo_data_file, "r")) if args.dpo_data_file and not args.graph_mode else {}
    user_profiles = json.load(open(args.user_profile_file, "r")) if args.graph_mode and args.user_profile_file else {}
    all_products = json.load(open(args.all_products_file, "r")) if args.graph_mode and args.all_products_file else {}

    graph_builder = GraphBuilder()
    llama_data = {"train": [], "test": []}
    skipped = 0

    for split, tasks in task_file.items():
        for task in tqdm(tasks, desc=f"Preparing {split}"):
            if args.graph_mode:
                prompt, subgraph, graph_text = build_graph_prompt(
                    task=task,
                    user_history=user_history,
                    user_profiles=user_profiles,
                    graph_builder=graph_builder,
                    mem_token_length=args.mem_token_length,
                )
                pair = select_graph_dpo_pair(
                    task=task,
                    subgraph=subgraph,
                    all_products=all_products,
                    min_margin=args.reward_margin,
                )
                if pair is None:
                    skipped += 1
                    continue
                llama_data[split].append(
                    {
                        "instruction": task["task"],
                        "prompt": prompt,
                        "chosen": pair["chosen"],
                        "rejected": pair["rejected"],
                        "graph_mode": True,
                        "graph_text": graph_text,
                        "graph": subgraph.to_dict(),
                        "reward_margin": pair["margin"],
                        "candidate_rewards": pair["candidates"],
                        "tool": task_to_tool(task["type"]),
                        "task_type": task["type"],
                    }
                )
            else:
                prompt, memory_text = build_flat_prompt(
                    task=task,
                    user_history=user_history,
                    mem_length=args.mem_length,
                    mem_token_length=args.mem_token_length,
                )
                chosen, rejected = get_chosen_reject(dpo_data.get(task["task"], {}))
                if chosen is None or rejected is None:
                    skipped += 1
                    continue
                llama_data[split].append(
                    {
                        "instruction": task["task"],
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "graph_mode": False,
                        "mem": memory_text,
                        "task_type": task["type"],
                    }
                )

    if skipped:
        print(f"Skipped {skipped} tasks because no valid DPO pair met the selection criteria.")

    with open(args.output_file, "w") as f:
        json.dump(llama_data, f, indent=2)


if __name__ == "__main__":
    main()
