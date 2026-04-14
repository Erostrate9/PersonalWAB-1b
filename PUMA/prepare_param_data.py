import argparse
import json
import os
import sys
from typing import Dict, Optional

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from PUMA.graph.builder import GraphBuilder
from PUMA.graph.retriever import GraphRetriever
from PUMA.graph.reward import synthesize_recommendation_sequence, synthesize_search_query
from PUMA.graph.serializer import build_graph_param_prompt, serialize_subgraph
from utils import (
    PARAM_PROMPT,
    build_taskspe_memory,
    generate_search_query,
    prettify_product_info,
    retrieve_top_k_memories,
    truncate_text,
)

llama_tokenizer = None
sim_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
sim_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
sim_model.eval()
if torch.cuda.is_available():
    sim_model.to("cuda")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare parameter-generation data for SFT")
    parser.add_argument("--task_file", type=str, required=True, help="Path to the task file")
    parser.add_argument("--user_history_file", type=str, required=True, help="Path to the user history file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output results")
    parser.add_argument("--mem_token_length", type=int, default=768, help="Maximum token length of memory or graph evidence")
    parser.add_argument("--mem_length", type=int, default=100, help="Maximum flat-memory retrieval count")
    parser.add_argument("--graph_mode", action="store_true", help="Use graph-conditioned retrieval and labels")
    parser.add_argument("--user_profile_file", type=str, default=None, help="Path to the user profile file")
    parser.add_argument("--recommendation_length", type=int, default=10, help="Maximum recommendation sequence length")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Tokenizer used for truncation")
    return parser.parse_args()


def task_to_tool(task_type: str) -> str:
    if task_type == "search":
        return "search_product_by_query"
    if task_type == "recommend":
        return "get_recommendations_by_history"
    return "add_product_review"


def build_flat_sample(task: Dict, split: str, user_history: Dict[str, list], mem_length: int, mem_token_length: int) -> Dict:
    input_text = task["task"]
    user_id = task["user_id"]
    task_type = task["type"]
    timestamp = task["timestamp"]
    product_info = task["target"]["product_info"]

    original_history = [item for item in user_history[user_id] if item["review"]["timestamp"] < timestamp]
    history = retrieve_top_k_memories(input_text, original_history, sim_model, sim_tokenizer, k=mem_length) if original_history else []
    mem = build_taskspe_memory(history, task_type)

    prefix_text = PARAM_PROMPT.replace("<Instruction>", input_text + (prettify_product_info(product_info) if task_type == "review" else ""))
    memory_text = "|".join(mem)
    tool_text = task_to_tool(task_type)
    memory_text_truncated = truncate_text(llama_tokenizer, memory_text, mem_token_length)
    prompt = prefix_text.replace("<Memory>", memory_text_truncated).replace("<Tool>", tool_text)

    if task_type == "search":
        target = generate_search_query(input_text, mem) if split == "train" else product_info["title"]
    elif task_type == "recommend":
        same_category_asins = [
            history_item["product_info"]["parent_asin"]
            for history_item in original_history
            if history_item["product_info"]["main_category"] == product_info["main_category"]
        ]
        target = ", ".join(same_category_asins[-30:]) if same_category_asins else product_info["parent_asin"]
    else:
        target = task["target"]["review"]["text"]

    return {
        "instruction": input_text,
        "prompt": prompt,
        "target": target,
        "mem": memory_text,
        "tool": tool_text,
        "task_type": task_type,
        "graph_mode": False,
    }


def build_graph_sample(
    task: Dict,
    user_history: Dict[str, list],
    user_profiles: Dict[str, dict],
    graph_builder: GraphBuilder,
    mem_token_length: int,
    recommendation_length: int,
) -> Dict:
    task_type = task["type"]
    product_info = task["target"]["product_info"]
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
        task_type=task_type,
        timestamp=task["timestamp"],
        product_info=product_info if task_type == "review" else None,
    )
    graph_text = serialize_subgraph(subgraph)
    truncated_graph_text = truncate_text(llama_tokenizer, graph_text, mem_token_length)
    tool_text = task_to_tool(task_type)
    prompt = build_graph_param_prompt(
        instruction=task["task"],
        tool_name=tool_text,
        graph_text=truncated_graph_text,
        product_info=product_info if task_type == "review" else None,
    )

    if task_type == "search":
        target = synthesize_search_query(task["task"], subgraph)
    elif task_type == "recommend":
        target = synthesize_recommendation_sequence(subgraph, max_items=recommendation_length) or product_info["parent_asin"]
    else:
        target = task["target"]["review"]["text"]

    return {
        "instruction": task["task"],
        "prompt": prompt,
        "target": target,
        "mem": truncated_graph_text,
        "tool": tool_text,
        "task_type": task_type,
        "graph_mode": True,
        "graph_text": truncated_graph_text,
        "graph": subgraph.to_dict(),
    }


def main():
    args = parse_args()
    global llama_tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

    task_file = json.load(open(args.task_file, "r"))
    user_history = json.load(open(args.user_history_file, "r"))
    user_profiles = json.load(open(args.user_profile_file, "r")) if args.graph_mode and args.user_profile_file else {}
    graph_builder = GraphBuilder()
    llama_data = {"train": [], "test": []}

    for split, tasks in task_file.items():
        for task in tqdm(tasks, desc=f"Preparing {split}"):
            if args.graph_mode:
                sample = build_graph_sample(
                    task=task,
                    user_history=user_history,
                    user_profiles=user_profiles,
                    graph_builder=graph_builder,
                    mem_token_length=args.mem_token_length,
                    recommendation_length=args.recommendation_length,
                )
            else:
                sample = build_flat_sample(
                    task=task,
                    split=split,
                    user_history=user_history,
                    mem_length=args.mem_length,
                    mem_token_length=args.mem_token_length,
                )
            llama_data[split].append(sample)

    with open(args.output_file, "w") as f:
        json.dump(llama_data, f, indent=2)


if __name__ == "__main__":
    main()
