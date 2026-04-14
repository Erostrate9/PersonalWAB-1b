import json
import glob
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
from PersonalWAB.envs.pwab.functions.get_recommendations_by_history import get_recommendations_by_history
from PersonalWAB.envs.pwab.functions.search_product_by_query import search_product_by_query
from tabulate import tabulate
from PUMA.graph.builder import GraphBuilder
from PUMA.graph.retriever import GraphRetriever
from PUMA.graph.reward import composite_reward
from PUMA.utils import normalize_tool_prediction


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PWA tasks")
    parser.add_argument('--evaluate_dpo', type=str, default='False', help='Whether to evaluate DPO')
    parser.add_argument('--task_file', type=str, default='PersonalWAB/envs/pwab/data/user_instructions.json', help='Path to task file')
    parser.add_argument('--param_file', type=str, default='PUMA/output/res/', help='Path to tool input file')
    parser.add_argument('--function_file', type=str, default='PUMA/output/', help='Path to tool selected file')
    parser.add_argument('--all_products', type=str, default='data/Reviews/all_products.json', help='Path to all products file')
    parser.add_argument('--dpo_output', type=str, default='PUMA/data/dpo_data.json', help='Path to DPO output file')
    parser.add_argument('--graph_mode', action='store_true', help='Use graph-aware reward decomposition')
    parser.add_argument('--user_history_file', type=str, default='PersonalWAB/envs/pwab/data/user_history_part_*.json', help='Path or glob to user history file(s) when graph_mode is enabled')
    parser.add_argument('--user_profile_file', type=str, default='PersonalWAB/envs/pwab/data/user_profiles.json', help='Path to user profile file when graph_mode is enabled')
    parser.add_argument('--tokenizer_name', type=str, default='meta-llama/Llama-3.2-1B-Instruct', help='Tokenizer used for truncation-only operations')
    return parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
if torch.cuda.is_available():
    model.to('cuda')

def compute_similarity(target_review, agent_review):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    sentences = [target_review, agent_review]
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    if torch.cuda.is_available():
        encoded_input.to('cuda')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    similarity = F.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0).item()
    del model_output
    del sentence_embeddings
    torch.cuda.empty_cache()
    return similarity

def trucate_text(text, max_length):
    tokenized_memory = llama_tokenizer(text, return_tensors=None, truncation=True, max_length=max_length)
    truncated_memory_ids = tokenized_memory["input_ids"]
    memory_text_truncated = llama_tokenizer.decode(truncated_memory_ids, skip_special_tokens=True)
    return memory_text_truncated

args = parse_args()

tasks = json.load(open(args.task_file))
tool_input = json.load(open(args.param_file))
tool_selected = json.load(open(args.function_file))
all_products = json.load(open(args.all_products))


def load_json_or_glob(path):
    if '*' not in path:
        return json.load(open(path))
    merged = {}
    for json_file in sorted(glob.glob(path)):
        merged.update(json.load(open(json_file)))
    return merged


user_history = load_json_or_glob(args.user_history_file) if args.graph_mode else {}
user_profiles = json.load(open(args.user_profile_file)) if args.graph_mode else {}
graph_builder = GraphBuilder() if args.graph_mode else None

final_results = {'search':[], 'recommend':[], 'review':[]}
tool_accuracy = {'search':[], 'recommend':[], 'review':[]}
graph_metrics = {
    'search': {'task': [], 'faith': [], 'valid': [], 'hop': [], 'len': [], 'composite': []},
    'recommend': {'task': [], 'faith': [], 'valid': [], 'hop': [], 'len': [], 'composite': []},
    'review': {'task': [], 'faith': [], 'valid': [], 'hop': [], 'len': [], 'composite': []},
}
missing_predictions = {'search': 0, 'recommend': 0, 'review': 0}

llama_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
if llama_tokenizer.pad_token is None:
    llama_tokenizer.pad_token = llama_tokenizer.eos_token


def build_subgraph(task):
    filtered_history = [
        item
        for item in user_history.get(task['user_id'], [])
        if item['review']['timestamp'] < task['timestamp']
    ]
    graph = graph_builder.build_user_graph(
        user_id=task['user_id'],
        history=filtered_history,
        user_profile=user_profiles.get(task['user_id'], {}),
    )
    retriever = GraphRetriever(graph)
    return retriever.retrieve_subgraph(
        instruction=task['task'],
        task_type=task['type'],
        timestamp=task['timestamp'],
        product_info=task['target']['product_info'] if task['type'] == 'review' else None,
    )

if args.evaluate_dpo == 'True':
    all_res = {}
    for task in tqdm(tasks['train']):
        task_type = task['type']
        instructions = task['task']
        target_asin = task['target']['product_info']['parent_asin']
        cur_res = {}
        subgraph = build_subgraph(task) if args.graph_mode else None
        if task_type == 'search':
            query = tool_input[instructions]
            for q in query:
                if args.graph_mode:
                    cur_res[q] = composite_reward(task, q, subgraph, all_products=all_products)
                else:
                    res = search_product_by_query(data={}, query=q)
                    score = 0
                    for i in range(len(res)):
                        if target_asin in res[i]:
                            score = 1 - i/len(res)
                            break      
                    cur_res[q] = score
        elif task_type == 'recommend':
            history = tool_input[instructions]
            for h in history:
                if args.graph_mode:
                    cur_res[h] = composite_reward(task, h, subgraph, all_products=all_products)
                else:
                    h_ = [item.strip() for item in h.split(',')]
                    h_ = list(set(h_))
                    res = get_recommendations_by_history(data={'all_products':all_products}, product_sequence=h_)
                    score = 0
                    for i in range(len(res)):
                        if target_asin in res[i]:
                            score = 1 - i/len(res)
                            break
                    cur_res[h] = score
        else:
            review = tool_input[instructions]
            for r in review:
                if args.graph_mode:
                    cur_res[r] = composite_reward(task, r, subgraph, all_products=all_products)
                else:
                    target_review = task['target']['review']['text']
                    agent_review = r
                    similarity = compute_similarity(target_review, agent_review)
                    cur_res[r] = similarity
        all_res[instructions] = cur_res
    with open(args.dpo_output, 'w') as f:
        json.dump(all_res, f, indent=2)
else:
    for task in tqdm(tasks['test']):
        if task['task'] not in tool_selected:
            gt_task_type = task['type']
            tool_accuracy[gt_task_type].append(0)
            final_results[gt_task_type].append(0)
            missing_predictions[gt_task_type] += 1
            continue
        tool = normalize_tool_prediction(tool_selected[task['task']][0])
        if tool == 'search_product_by_query':
            task_type = 'search'
        elif tool == 'get_recommendations_by_history':
            task_type = 'recommend'
        else:
            task_type = 'review'
        gt_task_type = task['type']
        if task_type == gt_task_type:
            tool_accuracy[gt_task_type].append(1)
        else:
            tool_accuracy[gt_task_type].append(0)
            final_results[gt_task_type].append(0)
            continue
        instructions = task['task']
        if instructions not in tool_input:
            final_results[gt_task_type].append(0)
            missing_predictions[gt_task_type] += 1
            continue
        target_asin = task['target']['product_info']['parent_asin']
        score = 0
        best_graph_reward = None
        subgraph = build_subgraph(task) if args.graph_mode else None
        if task_type == 'search':
            query = tool_input[instructions]
            for q in query:
                res = search_product_by_query(data={}, query=q)
                for i in range(len(res)):
                    if target_asin in res[i]:
                        score = 1 - i/len(res)
                        break      
                if args.graph_mode:
                    reward = composite_reward(task, q, subgraph, all_products=all_products)
                    if best_graph_reward is None or reward['composite'] > best_graph_reward['composite']:
                        best_graph_reward = reward
        elif task_type == 'recommend':
            history = tool_input[instructions]
            for h in history:
                h_ = [item.strip() for item in h.split(',')]
                h_ = list(set(h_))
                res = get_recommendations_by_history(data={'all_products':all_products}, product_sequence=h_)
                for i in range(len(res)):
                    if target_asin in res[i]:
                        score = 1 - i/len(res)
                        break
                if args.graph_mode:
                    reward = composite_reward(task, h, subgraph, all_products=all_products)
                    if best_graph_reward is None or reward['composite'] > best_graph_reward['composite']:
                        best_graph_reward = reward
        else:
            review = tool_input[instructions]
            for r in review:
                target_review = task['target']['review']['text']
                agent_review = r
                similarity = compute_similarity(target_review, agent_review)
                score = similarity
                if args.graph_mode:
                    reward = composite_reward(task, r, subgraph, all_products=all_products)
                    if best_graph_reward is None or reward['composite'] > best_graph_reward['composite']:
                        best_graph_reward = reward
        final_results[gt_task_type].append(score)
        if args.graph_mode and best_graph_reward is not None:
            for metric_name, metric_value in best_graph_reward.items():
                graph_metrics[gt_task_type][metric_name].append(metric_value)

    combined_data = [
        ['Search', len(final_results['search']), sum(tool_accuracy['search']) / len(tool_accuracy['search']), sum(final_results['search']) / len(final_results['search'])],
        ['Recommend', len(final_results['recommend']), sum(tool_accuracy['recommend']) / len(tool_accuracy['recommend']), sum(final_results['recommend']) / len(final_results['recommend'])],
        ['Review', len(final_results['review']), sum(tool_accuracy['review']) / len(tool_accuracy['review']), sum(final_results['review']) / len(final_results['review'])],
        ['Overall', len(final_results['search'] + final_results['recommend'] + final_results['review']),
        sum(tool_accuracy['search'] + tool_accuracy['recommend'] + tool_accuracy['review']) / len(tool_accuracy['search'] + tool_accuracy['recommend'] + tool_accuracy['review']),
        sum(final_results['search'] + final_results['recommend'] + final_results['review']) / len(final_results['search'] + final_results['recommend'] + final_results['review'])]
    ]

    headers = ['Task Type', 'Total', 'Tool Accuracy Avg', 'Result Avg']
    print(tabulate(combined_data, headers=headers, tablefmt='grid'))
    print('Missing predictions:', missing_predictions)

    if args.graph_mode:
        graph_rows = []
        for task_name in ['search', 'recommend', 'review']:
            metrics = graph_metrics[task_name]
            graph_rows.append([
                task_name.title(),
                sum(metrics['task']) / len(metrics['task']) if metrics['task'] else 0,
                sum(metrics['faith']) / len(metrics['faith']) if metrics['faith'] else 0,
                sum(metrics['valid']) / len(metrics['valid']) if metrics['valid'] else 0,
                sum(metrics['hop']) / len(metrics['hop']) if metrics['hop'] else 0,
                sum(metrics['len']) / len(metrics['len']) if metrics['len'] else 0,
                sum(metrics['composite']) / len(metrics['composite']) if metrics['composite'] else 0,
            ])
        print(tabulate(
            graph_rows,
            headers=['Task Type', 'Task Reward', 'Faithfulness', 'Validity', 'Hop Bonus', 'Length Penalty', 'Composite'],
            tablefmt='grid',
        ))
