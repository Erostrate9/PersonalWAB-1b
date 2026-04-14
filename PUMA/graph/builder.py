from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .schema import GraphEdge, GraphNode, UserGraph


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("_", " ").strip().split())


def normalize_key(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("$", "").replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def bucket_price(price: Any) -> str:
    numeric_price = safe_float(price)
    if numeric_price is None:
        return "unknown"
    if numeric_price < 25:
        return "budget"
    if numeric_price < 100:
        return "midrange"
    return "premium"


def coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_text(item) for item in value if normalize_text(item)]
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            key_text = normalize_text(key)
            value_text = normalize_text(item)
            if key_text and value_text:
                items.append(f"{key_text}: {value_text}")
            elif key_text:
                items.append(key_text)
            elif value_text:
                items.append(value_text)
        return items
    text = normalize_text(value)
    return [text] if text else []


def extract_brand(product_info: Dict[str, Any]) -> str:
    details = product_info.get("details") or {}
    brand = details.get("Brand") or details.get("Manufacturer") or product_info.get("store")
    return normalize_text(brand)


def extract_store(product_info: Dict[str, Any]) -> str:
    return normalize_text(product_info.get("store"))


def extract_categories(product_info: Dict[str, Any]) -> List[str]:
    categories = [normalize_text(product_info.get("main_category"))]
    categories.extend(coerce_list(product_info.get("categories")))
    return [item for item in categories if item]


def extract_feature_terms(product_info: Dict[str, Any], max_items: int = 10) -> List[str]:
    blocked_features = {
        "imported",
        "plastic",
        "white",
        "black",
        "machine wash",
        "new",
        "small",
        "medium",
        "large",
    }
    features = coerce_list(product_info.get("features"))
    details = product_info.get("details") or {}
    for key in ("Color", "Material", "Style", "Size", "Flavor", "Scent"):
        value = normalize_text(details.get(key))
        if value:
            features.append(value)

    deduped = []
    seen = set()
    for feature in features:
        normalized = normalize_text(feature)
        if not normalized:
            continue
        key = normalized.lower()
        if key in blocked_features:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
        if len(deduped) >= max_items:
            break
    return deduped


def product_summary_text(product_info: Dict[str, Any]) -> str:
    chunks = [
        normalize_text(product_info.get("title")),
        normalize_text(product_info.get("main_category")),
        extract_brand(product_info),
        extract_store(product_info),
        " ".join(extract_feature_terms(product_info, max_items=6)),
        normalize_text(product_info.get("description")),
    ]
    return " | ".join([chunk for chunk in chunks if chunk])


def review_summary_text(review: Dict[str, Any]) -> str:
    parts = [
        f"rating {review.get('rating')}",
        normalize_text(review.get("title")),
        normalize_text(review.get("text")),
    ]
    return " | ".join([part for part in parts if part and part != "rating None"])


def rating_signal(rating: Any) -> float:
    if rating is None:
        return 0.0
    try:
        numeric_rating = float(rating)
    except (TypeError, ValueError):
        return 0.0
    return max(-1.0, min(1.0, (numeric_rating - 3.0) / 2.0))


class GraphBuilder:
    def __init__(self) -> None:
        self._edge_counter = 0

    def _next_edge_id(self, prefix: str) -> str:
        self._edge_counter += 1
        return f"{prefix}:{self._edge_counter}"

    def _add_node_if_missing(
        self,
        graph: UserGraph,
        node_id: str,
        node_type: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if node_id in graph.nodes:
            return
        graph.add_node(GraphNode(node_id=node_id, node_type=node_type, text=text, metadata=metadata or {}))

    def _add_edge(
        self,
        graph: UserGraph,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0,
        timestamp: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        edge_id = self._next_edge_id(relation)
        graph.add_edge(
            GraphEdge(
                edge_id=edge_id,
                source=source,
                target=target,
                relation=relation,
                weight=weight,
                timestamp=timestamp,
                metadata=metadata or {},
            )
        )
        return edge_id

    def build_user_graph(
        self,
        user_id: str,
        history: List[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> UserGraph:
        graph = UserGraph(user_id=user_id, user_profile=user_profile or {})
        user_node_id = f"user:{user_id}"
        self._add_node_if_missing(
            graph,
            user_node_id,
            "user",
            f"user {user_id}",
            metadata={"user_id": user_id},
        )

        profile_payload = user_profile.get("user_profile", {}) if user_profile else {}
        for key, value in profile_payload.items():
            profile_text = normalize_text(value)
            if not profile_text:
                continue
            profile_node_id = f"profile:{normalize_key(key)}:{normalize_key(profile_text)}"
            self._add_node_if_missing(
                graph,
                profile_node_id,
                "profile_attribute",
                f"{key}: {profile_text}",
                metadata={"attribute": key, "value": profile_text},
            )
            self._add_edge(
                graph,
                user_node_id,
                profile_node_id,
                "has_profile_attribute",
                metadata={"attribute": key},
            )

        preference_scores: Dict[str, Dict[str, Dict[str, Any]]] = {
            "category": {},
            "brand": {},
            "feature": {},
            "store": {},
            "price_bucket": {},
            "product": {},
        }

        sorted_history = sorted(history, key=lambda item: item["review"]["timestamp"])
        for idx, item in enumerate(sorted_history):
            product_info = item["product_info"]
            review = item["review"]
            asin = normalize_text(product_info.get("parent_asin") or review.get("parent_asin"))
            timestamp = review.get("timestamp")
            signal = rating_signal(review.get("rating"))

            product_node_id = f"product:{asin}"
            self._add_node_if_missing(
                graph,
                product_node_id,
                "product",
                product_summary_text(product_info),
                metadata={
                    "asin": asin,
                    "title": normalize_text(product_info.get("title")),
                    "main_category": normalize_text(product_info.get("main_category")),
                    "brand": extract_brand(product_info),
                    "store": extract_store(product_info),
                    "price": product_info.get("price"),
                    "price_bucket": bucket_price(product_info.get("price")),
                    "features": extract_feature_terms(product_info),
                },
            )

            event_node_id = f"event:{user_id}:{idx}:{timestamp}"
            self._add_node_if_missing(
                graph,
                event_node_id,
                "interaction_event",
                f"{review_summary_text(review)} | {product_summary_text(product_info)}",
                metadata={
                    "timestamp": timestamp,
                    "rating": review.get("rating"),
                    "signal": signal,
                    "asin": asin,
                },
            )

            review_node_id = f"review:{user_id}:{idx}:{timestamp}"
            self._add_node_if_missing(
                graph,
                review_node_id,
                "review",
                review_summary_text(review),
                metadata={
                    "timestamp": timestamp,
                    "rating": review.get("rating"),
                    "title": normalize_text(review.get("title")),
                    "text": normalize_text(review.get("text")),
                },
            )

            user_event_edge_id = self._add_edge(
                graph,
                user_node_id,
                event_node_id,
                "performed_interaction",
                weight=max(signal, 0.0) + 1.0,
                timestamp=timestamp,
                metadata={"asin": asin},
            )
            event_product_edge_id = self._add_edge(
                graph,
                event_node_id,
                product_node_id,
                "interacted_with_product",
                weight=max(signal, 0.0) + 1.0,
                timestamp=timestamp,
            )
            event_review_edge_id = self._add_edge(
                graph,
                event_node_id,
                review_node_id,
                "has_review",
                weight=1.0,
                timestamp=timestamp,
            )

            relation = "positive_product_preference" if signal >= 0 else "negative_product_preference"
            user_product_edge_id = self._add_edge(
                graph,
                user_node_id,
                product_node_id,
                relation,
                weight=abs(signal),
                timestamp=timestamp,
                metadata={"rating": review.get("rating"), "timestamp": timestamp},
            )

            categories = extract_categories(product_info)
            brand = extract_brand(product_info)
            store = extract_store(product_info)
            features = extract_feature_terms(product_info)
            price_bucket = bucket_price(product_info.get("price"))

            relation_edge_ids = {
                "user_event": user_event_edge_id,
                "event_product": event_product_edge_id,
                "event_review": event_review_edge_id,
                "user_product": user_product_edge_id,
            }

            category_node_ids = []
            for category in categories:
                category_node_id = f"category:{normalize_key(category)}"
                category_node_ids.append(category_node_id)
                self._add_node_if_missing(
                    graph,
                    category_node_id,
                    "category",
                    category,
                    metadata={"name": category},
                )
                relation_edge_ids.setdefault("categories", []).append(
                    self._add_edge(
                        graph,
                        product_node_id,
                        category_node_id,
                        "in_category",
                        timestamp=timestamp,
                    )
                )
                self._record_preference(
                    preference_scores,
                    "category",
                    category,
                    signal,
                    item,
                    event_node_id,
                    product_node_id,
                    category_node_id,
                )

            brand_node_id = None
            if brand:
                brand_node_id = f"brand:{normalize_key(brand)}"
                self._add_node_if_missing(
                    graph,
                    brand_node_id,
                    "brand",
                    brand,
                    metadata={"name": brand},
                )
                relation_edge_ids["brand"] = self._add_edge(
                    graph,
                    product_node_id,
                    brand_node_id,
                    "made_by_brand",
                    timestamp=timestamp,
                )
                self._record_preference(
                    preference_scores,
                    "brand",
                    brand,
                    signal,
                    item,
                    event_node_id,
                    product_node_id,
                    brand_node_id,
                )

            store_node_id = None
            if store:
                store_node_id = f"store:{normalize_key(store)}"
                self._add_node_if_missing(
                    graph,
                    store_node_id,
                    "store",
                    store,
                    metadata={"name": store},
                )
                relation_edge_ids["store"] = self._add_edge(
                    graph,
                    product_node_id,
                    store_node_id,
                    "sold_by_store",
                    timestamp=timestamp,
                )
                self._record_preference(
                    preference_scores,
                    "store",
                    store,
                    signal,
                    item,
                    event_node_id,
                    product_node_id,
                    store_node_id,
                )

            feature_node_ids = []
            for feature in features:
                feature_node_id = f"feature:{normalize_key(feature)}"
                feature_node_ids.append(feature_node_id)
                self._add_node_if_missing(
                    graph,
                    feature_node_id,
                    "feature",
                    feature,
                    metadata={"name": feature},
                )
                relation_edge_ids.setdefault("features", []).append(
                    self._add_edge(
                        graph,
                        product_node_id,
                        feature_node_id,
                        "has_feature",
                        weight=0.8,
                        timestamp=timestamp,
                    )
                )
                self._record_preference(
                    preference_scores,
                    "feature",
                    feature,
                    signal,
                    item,
                    event_node_id,
                    product_node_id,
                    feature_node_id,
                )

            price_bucket_node_id = f"price_bucket:{normalize_key(price_bucket)}"
            self._add_node_if_missing(
                graph,
                price_bucket_node_id,
                "price_bucket",
                price_bucket,
                metadata={"bucket": price_bucket},
            )
            relation_edge_ids["price_bucket"] = self._add_edge(
                graph,
                product_node_id,
                price_bucket_node_id,
                "in_price_bucket",
                timestamp=timestamp,
            )
            self._record_preference(
                preference_scores,
                "price_bucket",
                price_bucket,
                signal,
                item,
                event_node_id,
                product_node_id,
                price_bucket_node_id,
            )

            self._record_preference(
                preference_scores,
                "product",
                asin,
                signal,
                item,
                event_node_id,
                product_node_id,
                product_node_id,
            )

            graph.interaction_records.append(
                {
                    "event_node_id": event_node_id,
                    "product_node_id": product_node_id,
                    "review_node_id": review_node_id,
                    "timestamp": timestamp,
                    "rating": review.get("rating"),
                    "signal": signal,
                    "item": item,
                    "asin": asin,
                    "brand": brand,
                    "store": store,
                    "categories": categories,
                    "features": features,
                    "price_bucket": price_bucket,
                    "edge_ids": relation_edge_ids,
                    "node_ids": {
                        "product": product_node_id,
                        "event": event_node_id,
                        "review": review_node_id,
                        "categories": category_node_ids,
                        "brand": brand_node_id,
                        "store": store_node_id,
                        "features": feature_node_ids,
                        "price_bucket": price_bucket_node_id,
                    },
                }
            )

        for preference_type, items in preference_scores.items():
            for preference_name, payload in items.items():
                if payload["score"] == 0:
                    continue
                node_id = payload["node_id"]
                relation = f"{preference_type}_affinity"
                self._add_edge(
                    graph,
                    user_node_id,
                    node_id,
                    relation,
                    weight=payload["score"],
                    metadata={
                        "evidence_event_ids": payload["event_ids"],
                        "count": payload["count"],
                        "name": preference_name,
                    },
                )

        graph.preference_scores = preference_scores
        graph.metadata["history_size"] = len(sorted_history)
        graph.metadata["latest_timestamp"] = (
            max(record["timestamp"] for record in graph.interaction_records)
            if graph.interaction_records
            else None
        )
        return graph

    def _record_preference(
        self,
        preference_scores: Dict[str, Dict[str, Dict[str, Any]]],
        preference_type: str,
        name: str,
        signal: float,
        item: Dict[str, Any],
        event_node_id: str,
        source_node_id: str,
        target_node_id: str,
    ) -> None:
        key = normalize_text(name)
        if not key:
            return
        payload = preference_scores[preference_type].setdefault(
            key,
            {
                "name": key,
                "score": 0.0,
                "count": 0,
                "event_ids": [],
                "source_node_id": source_node_id,
                "node_id": target_node_id,
                "latest_timestamp": item["review"].get("timestamp"),
            },
        )
        payload["score"] += signal
        payload["count"] += 1
        payload["event_ids"].append(event_node_id)
        payload["latest_timestamp"] = max(
            payload["latest_timestamp"] or item["review"].get("timestamp"),
            item["review"].get("timestamp"),
        )
