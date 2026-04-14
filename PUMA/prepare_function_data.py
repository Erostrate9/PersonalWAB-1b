import argparse
import json
import os
import sys
from typing import Dict

from tqdm import tqdm
from transformers import AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from PUMA.graph.builder import GraphBuilder
from PUMA.graph.retriever import GraphRetriever
from PUMA.graph.serializer import build_graph_function_prompt, serialize_subgraph
from utils import FUNCTION_PROMPT, truncate_text


llama_tokenizer = None


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare function-selection data for SFT")
    parser.add_argument("--task_file", type=str, required=True, help="Path to the task file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output results")
    parser.add_argument("--graph_mode", action="store_true", help="Build graph-conditioned prompts")
    parser.add_argument("--user_history_file", type=str, default=None, help="Path to the user history file")
    parser.add_argument("--user_profile_file", type=str, default=None, help="Path to the user profile file")
    parser.add_argument("--mem_token_length", type=int, default=512, help="Maximum token length of graph evidence")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Tokenizer used for truncation")
    return parser.parse_args()


def task_to_tool(task_type: str) -> str:
    if task_type == "search":
        return "search_product_by_query"
    if task_type == "recommend":
        return "get_recommendations_by_history"
    return "add_product_review"


def build_graph_prompt(
    task: Dict,
    user_history: Dict[str, list],
    user_profiles: Dict[str, dict],
    graph_builder: GraphBuilder,
    mem_token_length: int,
) -> Dict:
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
    truncated_graph_text = truncate_text(llama_tokenizer, graph_text, mem_token_length)
    return {
        "graph_text": truncated_graph_text,
        "graph": subgraph.to_dict(),
    }


def main():
    args = parse_args()
    global llama_tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

    tasks_by_split = json.load(open(args.task_file, "r"))
    user_history = json.load(open(args.user_history_file, "r")) if args.graph_mode else {}
    user_profiles = json.load(open(args.user_profile_file, "r")) if args.graph_mode and args.user_profile_file else {}

    if args.graph_mode and not args.user_history_file:
        raise ValueError("--user_history_file is required when --graph_mode is enabled.")

    graph_builder = GraphBuilder()
    llama_data = {"train": [], "test": []}

    for split, tasks in tasks_by_split.items():
        for task in tqdm(tasks, desc=f"Preparing {split}"):
            tool_text = task_to_tool(task["type"])
            sample = {
                "instruction": task["task"],
                "target": tool_text,
                "task_type": task["type"],
                "tool": tool_text,
            }

            if args.graph_mode:
                graph_payload = build_graph_prompt(
                    task=task,
                    user_history=user_history,
                    user_profiles=user_profiles,
                    graph_builder=graph_builder,
                    mem_token_length=args.mem_token_length,
                )
                sample["prompt"] = build_graph_function_prompt(task["task"], graph_payload["graph_text"])
                sample["graph_mode"] = True
                sample["graph_text"] = graph_payload["graph_text"]
                sample["graph"] = graph_payload["graph"]
            else:
                sample["prompt"] = FUNCTION_PROMPT.replace("<Instruction>", task["task"])

            llama_data[split].append(sample)

    with open(args.output_file, "w") as f:
        json.dump(llama_data, f, indent=2)


if __name__ == "__main__":
    main()
