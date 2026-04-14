from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from PersonalWAB.envs.pwab.functions.get_recommendations_by_history import get_recommendations_by_history
from PersonalWAB.envs.pwab.functions.search_product_by_query import search_product_by_query

from .index import STOPWORDS, tokenize_text
from .schema import RetrievedSubgraph


_review_tokenizer = None
_review_model = None


def _load_review_encoder():
    global _review_model, _review_tokenizer
    if _review_model is None or _review_tokenizer is None:
        _review_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        _review_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        _review_model.eval()
        if torch.cuda.is_available():
            _review_model.to("cuda")
    return _review_tokenizer, _review_model


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def compute_review_similarity(target_review: str, candidate_review: str) -> float:
    tokenizer, model = _load_review_encoder()
    encoded_input = tokenizer([target_review, candidate_review], padding=True, truncation=True, return_tensors="pt")
    if torch.cuda.is_available():
        encoded_input = encoded_input.to("cuda")
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = _mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return float(F.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0).item())


def parse_recommendation_sequence(candidate: str) -> List[str]:
    items = []
    for part in candidate.split(","):
        token = part.strip()
        if token and token not in items:
            items.append(token)
    return items


def clean_query(candidate: str) -> str:
    return " ".join(candidate.strip().split())


def supported_terms(subgraph: RetrievedSubgraph) -> set[str]:
    return set(subgraph.metadata.get("support_terms", []))


def graph_faithfulness_score(task_type: str, candidate: str, subgraph: RetrievedSubgraph) -> float:
    support = supported_terms(subgraph)
    if task_type == "recommend":
        items = parse_recommendation_sequence(candidate)
        if not items:
            return 0.0
        supported_asins = set(subgraph.metadata.get("positive_products", [])) | set(subgraph.metadata.get("recent_positive_asins", []))
        supported_count = sum(1 for item in items if item in supported_asins)
        return supported_count / len(items)

    candidate_tokens = [token for token in tokenize_text(candidate) if token not in STOPWORDS]
    if not candidate_tokens:
        return 0.0
    overlap = sum(1 for token in candidate_tokens if token in support)
    if task_type == "review":
        style_terms = tokenize_text(" ".join([
            *subgraph.metadata.get("preferred_features", []),
            *subgraph.metadata.get("preferred_categories", []),
        ]))
        overlap += 0.5 * sum(1 for token in candidate_tokens if token in style_terms)
    return min(1.0, overlap / max(1, len(candidate_tokens)))


def validity_score(task_type: str, candidate: str) -> float:
    candidate = candidate.strip()
    if not candidate:
        return 0.0
    if task_type == "recommend":
        items = parse_recommendation_sequence(candidate)
        if not items:
            return 0.0
        unique_ratio = len(set(items)) / len(items)
        return 0.5 + 0.5 * unique_ratio
    if task_type == "search":
        return 1.0 if len(candidate.split()) >= 2 else 0.5
    return 1.0 if len(candidate) >= 20 else 0.5


def multi_hop_bonus(task_type: str, candidate: str, subgraph: RetrievedSubgraph) -> float:
    candidate_tokens = set(tokenize_text(candidate))
    if task_type == "recommend":
        path_asins = {
            path.metadata.get("asin")
            for path in subgraph.preference_paths
            if path.metadata.get("asin")
        }
        items = set(parse_recommendation_sequence(candidate))
        return 1.0 if items.intersection(path_asins) else 0.0

    matched_paths = 0
    for path in subgraph.preference_paths:
        path_tokens = set(tokenize_text(path.text))
        if candidate_tokens.intersection(path_tokens):
            matched_paths += 1
    if not subgraph.preference_paths:
        return 0.0
    return min(1.0, matched_paths / len(subgraph.preference_paths))


def length_penalty(task_type: str, candidate: str) -> float:
    if task_type == "recommend":
        items = parse_recommendation_sequence(candidate)
        return max(0.0, (len(items) - 10) / 10)
    token_count = len(candidate.split())
    if task_type == "search":
        return max(0.0, (token_count - 14) / 10)
    return max(0.0, (token_count - 120) / 60)


def task_reward(
    task: Dict[str, Any],
    task_type: str,
    candidate: str,
    all_products: Optional[Dict[str, Any]] = None,
) -> float:
    target_asin = task["target"]["product_info"]["parent_asin"]
    if task_type == "search":
        results = search_product_by_query(data={}, query=clean_query(candidate))
        for idx, result in enumerate(results):
            if target_asin in result:
                return 1.0 - (idx / max(1, len(results)))
        return 0.0

    if task_type == "recommend":
        sequence = parse_recommendation_sequence(candidate)
        if not sequence:
            return 0.0
        results = get_recommendations_by_history(
            data={"all_products": all_products or {}},
            product_sequence=sequence,
        )
        for idx, result in enumerate(results):
            if target_asin in result:
                return 1.0 - (idx / max(1, len(results)))
        return 0.0

    target_review = task["target"]["review"]["text"]
    return compute_review_similarity(target_review, candidate)


def composite_reward(
    task: Dict[str, Any],
    candidate: str,
    subgraph: RetrievedSubgraph,
    all_products: Optional[Dict[str, Any]] = None,
    alpha: float = 1.0,
    beta: float = 0.6,
    gamma: float = 0.3,
    delta: float = 0.2,
    eta: float = 0.15,
) -> Dict[str, float]:
    task_type = task["type"]
    rewards = {
        "task": task_reward(task, task_type, candidate, all_products=all_products),
        "faith": graph_faithfulness_score(task_type, candidate, subgraph),
        "valid": validity_score(task_type, candidate),
        "hop": multi_hop_bonus(task_type, candidate, subgraph),
        "len": length_penalty(task_type, candidate),
    }
    rewards["composite"] = (
        alpha * rewards["task"]
        + beta * rewards["faith"]
        + gamma * rewards["valid"]
        + delta * rewards["hop"]
        - eta * rewards["len"]
    )
    return rewards


def synthesize_search_query(instruction: str, subgraph: RetrievedSubgraph, max_terms: int = 12) -> str:
    selected_terms: List[str] = []
    instruction_terms = tokenize_text(instruction)
    seen = set()

    for token in instruction_terms:
        if token in seen:
            continue
        seen.add(token)
        selected_terms.append(token)
        if len(selected_terms) >= max_terms - 2:
            break

    support_tokens = set(subgraph.metadata.get("support_terms", []))
    anchor_texts = subgraph.metadata.get("anchor_node_texts", [])
    for anchor_text in anchor_texts:
        anchor_tokens = tokenize_text(anchor_text)
        if not anchor_tokens:
            continue
        overlap = set(anchor_tokens).intersection(instruction_terms)
        if not overlap:
            continue
        for token in anchor_tokens:
            if token in seen:
                continue
            seen.add(token)
            selected_terms.append(token)
            if len(selected_terms) >= max_terms:
                break
        if len(selected_terms) >= max_terms:
            break

    for brand in subgraph.metadata.get("preferred_brands", []):
        brand_tokens = tokenize_text(brand)
        if not brand_tokens:
            continue
        if not set(brand_tokens).intersection(support_tokens):
            continue
        if len(selected_terms) + len(brand_tokens) > max_terms:
            break
        for token in brand_tokens:
            if token in seen:
                continue
            seen.add(token)
            selected_terms.append(token)
        break

    return " ".join(selected_terms[:max_terms]).strip()


def synthesize_recommendation_sequence(
    subgraph: RetrievedSubgraph,
    max_items: int = 10,
) -> str:
    candidates = []
    for asin in subgraph.metadata.get("recent_positive_asins", []):
        if asin not in candidates:
            candidates.append(asin)
        if len(candidates) >= max_items:
            break
    if not candidates:
        for asin in subgraph.metadata.get("positive_products", []):
            if asin not in candidates:
                candidates.append(asin)
            if len(candidates) >= max_items:
                break
    return ", ".join(candidates[:max_items])


def build_review_negatives(gold_review: str) -> List[str]:
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", gold_review.strip()) if segment.strip()]
    negatives = []
    if sentences:
        negatives.append(sentences[0])
    negatives.append("This product was okay overall.")
    negatives.append("Good product.")
    return [candidate for candidate in negatives if candidate and candidate != gold_review]


def build_search_negatives(instruction: str, subgraph: RetrievedSubgraph) -> List[str]:
    negatives = []
    categories = subgraph.metadata.get("preferred_categories", [])
    brands = subgraph.metadata.get("avoided_brands", [])
    if categories:
        negatives.append(" ".join(categories[:1]))
    if brands:
        negatives.append(" ".join(brands[:1] + categories[:1]))
    negatives.append("best popular products")
    return [candidate for candidate in negatives if candidate]


def build_recommendation_negatives(subgraph: RetrievedSubgraph) -> List[str]:
    negatives = []
    recent_negative = subgraph.metadata.get("recent_negative_asins", [])
    recent_all = subgraph.metadata.get("recent_asins", [])
    if recent_negative:
        negatives.append(", ".join(recent_negative[:5]))
    if recent_all:
        negatives.append(", ".join(recent_all[-5:]))
    positives = subgraph.metadata.get("recent_positive_asins", [])
    if positives:
        negatives.append(", ".join(positives[:2]))
    return [candidate for candidate in negatives if candidate]


def build_graph_candidates(task: Dict[str, Any], subgraph: RetrievedSubgraph) -> List[str]:
    task_type = task["type"]
    if task_type == "search":
        primary = synthesize_search_query(task["task"], subgraph)
        return [primary] + build_search_negatives(task["task"], subgraph)
    if task_type == "recommend":
        primary = synthesize_recommendation_sequence(subgraph)
        return [primary] + build_recommendation_negatives(subgraph)
    gold_review = task["target"]["review"]["text"]
    return [gold_review] + build_review_negatives(gold_review)


def select_graph_dpo_pair(
    task: Dict[str, Any],
    subgraph: RetrievedSubgraph,
    all_products: Optional[Dict[str, Any]] = None,
    min_margin: float = 0.05,
) -> Optional[Dict[str, Any]]:
    candidates = []
    seen = set()
    for candidate in build_graph_candidates(task, subgraph):
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        reward = composite_reward(task, normalized, subgraph, all_products=all_products)
        candidates.append({"candidate": normalized, "reward": reward})
    if len(candidates) < 2:
        return None
    candidates.sort(key=lambda item: item["reward"]["composite"], reverse=True)
    chosen = candidates[0]
    rejected = candidates[-1]
    margin = chosen["reward"]["composite"] - rejected["reward"]["composite"]
    if margin < min_margin:
        return None
    return {
        "chosen": chosen["candidate"],
        "rejected": rejected["candidate"],
        "margin": margin,
        "candidates": candidates,
    }
