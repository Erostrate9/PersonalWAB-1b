from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .builder import extract_brand, extract_categories, extract_feature_terms, normalize_key, normalize_text
from .index import LexicalGraphIndex, tokenize_text
from .schema import EvidenceLine, GraphPath, RetrievedSubgraph, UserGraph


DEFAULT_LIMITS = {
    "priors": 4,
    "recent_evidence": 4,
    "paths": 4,
    "anchors": 4,
}


class GraphRetriever:
    def __init__(self, graph: UserGraph):
        self.graph = graph
        self.index = LexicalGraphIndex(graph.nodes)
        self.user_node_id = f"user:{graph.user_id}"

    def retrieve_subgraph(
        self,
        instruction: str,
        task_type: str,
        timestamp: int,
        product_info: Optional[Dict[str, Any]] = None,
        limits: Optional[Dict[str, int]] = None,
    ) -> RetrievedSubgraph:
        merged_limits = {**DEFAULT_LIMITS, **(limits or {})}
        query_tokens = tokenize_text(instruction)
        ranked_records = self._rank_records(query_tokens, task_type, product_info)
        user_priors = self._build_user_priors(task_type, merged_limits["priors"])
        recent_evidence = self._build_recent_evidence(ranked_records, merged_limits["recent_evidence"])
        preference_paths = self._build_preference_paths(
            ranked_records,
            task_type,
            product_info,
            merged_limits["paths"],
        )
        task_anchors = self._build_task_anchors(
            instruction,
            task_type,
            product_info,
            merged_limits["anchors"],
        )

        line_items = user_priors + recent_evidence + task_anchors
        node_ids = set()
        edge_ids = set()
        for line in line_items:
            node_ids.update(line.node_ids)
            edge_ids.update(line.edge_ids)
        for path in preference_paths:
            node_ids.update(path.node_ids)
            edge_ids.update(path.edge_ids)

        support_terms = self._collect_support_terms(line_items, preference_paths)
        positive_products = [
            record["asin"]
            for record in self.graph.interaction_records
            if record.get("signal", 0.0) > 0
        ]
        negative_products = [
            record["asin"]
            for record in self.graph.interaction_records
            if record.get("signal", 0.0) < 0
        ]
        seed_node_ids = [record["product_node_id"] for record, _ in ranked_records[:3]]
        return RetrievedSubgraph(
            user_id=self.graph.user_id,
            task_type=task_type,
            timestamp=timestamp,
            instruction=instruction,
            user_priors=user_priors,
            recent_evidence=recent_evidence,
            preference_paths=preference_paths,
            task_anchors=task_anchors,
            seed_node_ids=seed_node_ids,
            node_ids=sorted(node_ids),
            edge_ids=sorted(edge_ids),
            metadata={
                "support_terms": sorted(support_terms),
                "positive_products": positive_products,
                "negative_products": negative_products,
                "recent_asins": [record["asin"] for record, _ in ranked_records[:10]],
                "recent_positive_asins": [
                    record["asin"]
                    for record, _ in ranked_records
                    if record.get("signal", 0.0) > 0
                ][:10],
                "recent_negative_asins": [
                    record["asin"]
                    for record, _ in ranked_records
                    if record.get("signal", 0.0) < 0
                ][:10],
                "preferred_brands": self._top_preference_names("brand", positive=True, limit=5),
                "avoided_brands": self._top_preference_names("brand", positive=False, limit=5),
                "preferred_categories": self._top_preference_names("category", positive=True, limit=5),
                "preferred_features": self._top_preference_names("feature", positive=True, limit=6),
                "preferred_price_buckets": self._top_preference_names("price_bucket", positive=True, limit=3),
                "anchor_node_texts": [
                    self.graph.nodes[line.node_ids[0]].text
                    for line in task_anchors
                    if line.node_ids and line.node_ids[0] in self.graph.nodes
                ],
                "history_size": len(self.graph.interaction_records),
            },
        )

    def _rank_records(
        self,
        query_tokens: Sequence[str],
        task_type: str,
        product_info: Optional[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], float]]:
        timestamps = [record["timestamp"] for record in self.graph.interaction_records]
        min_timestamp = min(timestamps) if timestamps else 0
        max_timestamp = max(timestamps) if timestamps else 0

        target_categories = set(extract_categories(product_info)) if product_info else set()
        target_brand = extract_brand(product_info) if product_info else ""
        target_features = set(extract_feature_terms(product_info)) if product_info else set()

        ranked = []
        for record in self.graph.interaction_records:
            product_node_id = record["product_node_id"]
            textual_score = self.index.score_tokens(query_tokens, product_node_id)
            attribute_overlap = self._attribute_overlap_score(
                query_tokens=query_tokens,
                categories=record.get("categories", []),
                brand=record.get("brand"),
                features=record.get("features", []),
                store=record.get("store"),
            )
            recency_score = self._normalize(
                record["timestamp"],
                min_timestamp,
                max_timestamp,
            )
            sentiment_score = max(record.get("signal", 0.0), 0.0)
            category_match = 1.0 if target_categories.intersection(record.get("categories", [])) else 0.0
            brand_match = 1.0 if target_brand and target_brand == record.get("brand") else 0.0
            feature_match = 1.0 if target_features.intersection(record.get("features", [])) else 0.0

            score = textual_score + attribute_overlap
            if task_type == "search":
                score += 0.4 * recency_score + 0.5 * sentiment_score + 0.6 * category_match + 0.4 * brand_match
            elif task_type == "recommend":
                score += 0.8 * recency_score + 1.2 * sentiment_score + 0.8 * category_match + 0.6 * brand_match + 0.4 * feature_match
            else:
                score += 0.9 * recency_score + 0.7 * sentiment_score + 0.7 * category_match + 0.3 * brand_match + 0.5 * feature_match

            if not query_tokens:
                score += recency_score + sentiment_score

            ranked.append((record, score))

        ranked.sort(key=lambda item: (item[1], item[0]["timestamp"]), reverse=True)
        return ranked

    def _build_user_priors(self, task_type: str, limit: int) -> List[EvidenceLine]:
        lines: List[EvidenceLine] = []
        profile = self.graph.user_profile.get("user_profile", {})
        preferred_brands = self._top_preference_names("brand", positive=True, limit=3)
        preferred_categories = self._top_preference_names("category", positive=True, limit=3)
        preferred_features = self._top_preference_names("feature", positive=True, limit=4)
        avoided_brands = self._top_preference_names("brand", positive=False, limit=2)

        if profile.get("Price Sensitivity"):
            lines.append(
                EvidenceLine(
                    line_id="prior:price_sensitivity",
                    block="user_priors",
                    text=f"Price sensitivity: {profile['Price Sensitivity']}.",
                    node_ids=self._profile_node_ids("Price Sensitivity", profile["Price Sensitivity"]),
                    edge_ids=self._profile_edge_ids("Price Sensitivity", profile["Price Sensitivity"]),
                )
            )

        if task_type == "review" and (profile.get("Tone and Style") or profile.get("Focus Aspect")):
            tone = profile.get("Tone and Style", "").strip()
            focus = profile.get("Focus Aspect", "").strip()
            style_parts = []
            if tone:
                style_parts.append(f"tone/style: {tone}")
            if focus:
                style_parts.append(f"focus aspects: {focus}")
            lines.append(
                EvidenceLine(
                    line_id="prior:review_style",
                    block="user_priors",
                    text=f"Review priors: {'; '.join(style_parts)}.",
                    node_ids=self._profile_node_ids("Tone and Style", tone) + self._profile_node_ids("Focus Aspect", focus),
                    edge_ids=self._profile_edge_ids("Tone and Style", tone) + self._profile_edge_ids("Focus Aspect", focus),
                )
            )

        if preferred_categories:
            lines.append(
                EvidenceLine(
                    line_id="prior:categories",
                    block="user_priors",
                    text=f"Top positive categories: {', '.join(preferred_categories)}.",
                    node_ids=[self._category_node_id(name) for name in preferred_categories],
                    edge_ids=self._user_affinity_edge_ids("category", preferred_categories),
                )
            )

        if preferred_brands and task_type in {"search", "recommend"}:
            lines.append(
                EvidenceLine(
                    line_id="prior:brands",
                    block="user_priors",
                    text=f"Top positive brands: {', '.join(preferred_brands)}.",
                    node_ids=[self._brand_node_id(name) for name in preferred_brands],
                    edge_ids=self._user_affinity_edge_ids("brand", preferred_brands),
                )
            )

        if preferred_features and task_type in {"search", "review"}:
            lines.append(
                EvidenceLine(
                    line_id="prior:features",
                    block="user_priors",
                    text=f"Frequently liked features: {', '.join(preferred_features)}.",
                    node_ids=[self._feature_node_id(name) for name in preferred_features],
                    edge_ids=self._user_affinity_edge_ids("feature", preferred_features),
                )
            )

        if avoided_brands and task_type == "recommend":
            lines.append(
                EvidenceLine(
                    line_id="prior:avoid_brands",
                    block="user_priors",
                    text=f"Lower-rated brands in history: {', '.join(avoided_brands)}.",
                    node_ids=[self._brand_node_id(name) for name in avoided_brands],
                    edge_ids=self._user_affinity_edge_ids("brand", avoided_brands),
                )
            )

        return lines[:limit]

    def _build_recent_evidence(
        self,
        ranked_records: List[Tuple[Dict[str, Any], float]],
        limit: int,
    ) -> List[EvidenceLine]:
        lines: List[EvidenceLine] = []
        for idx, (record, score) in enumerate(ranked_records[:limit]):
            product = record["item"]["product_info"]
            title = normalize_text(product.get("title"))
            category = normalize_text(product.get("main_category"))
            brand = normalize_text(record.get("brand"))
            features = ", ".join(record.get("features", [])[:3])
            rating = record.get("rating")
            text = (
                f"{record['timestamp']}: rated {rating} for \"{title}\""
                f" | category: {category or 'unknown'}"
                f" | brand: {brand or 'unknown'}"
            )
            if features:
                text += f" | liked attributes: {features}"
            lines.append(
                EvidenceLine(
                    line_id=f"recent:{idx}",
                    block="recent_evidence",
                    text=text,
                    node_ids=self._record_node_ids(record),
                    edge_ids=self._record_edge_ids(record),
                    score=score,
                    metadata={"asin": record["asin"], "rating": rating},
                )
            )
        return lines

    def _build_preference_paths(
        self,
        ranked_records: List[Tuple[Dict[str, Any], float]],
        task_type: str,
        product_info: Optional[Dict[str, Any]],
        limit: int,
    ) -> List[GraphPath]:
        paths: List[GraphPath] = []
        seen_texts = set()
        for idx, (record, score) in enumerate(ranked_records):
            if len(paths) >= limit:
                break
            attribute_type, attribute_name, attribute_node_id = self._select_best_path_attribute(record, task_type, product_info)
            if not attribute_name or not attribute_node_id:
                continue
            affinity_edge_ids = self._user_affinity_edge_ids(attribute_type, [attribute_name])
            relation_edge_ids = self._attribute_relation_edge_ids(record, attribute_node_id)
            path_text = (
                f"User -> {attribute_type} affinity \"{attribute_name}\" -> product \""
                f"{record['item']['product_info']['title']}\" -> rating {record['rating']} evidence at {record['timestamp']}."
            )
            if path_text in seen_texts:
                continue
            seen_texts.add(path_text)
            paths.append(
                GraphPath(
                    path_id=f"path:{idx}",
                    text=path_text,
                    node_ids=[self.user_node_id, attribute_node_id, record["product_node_id"], record["review_node_id"]],
                    edge_ids=affinity_edge_ids + relation_edge_ids + self._record_edge_ids(record),
                    score=score,
                    hop_count=3,
                    metadata={"attribute_type": attribute_type, "attribute_name": attribute_name, "asin": record["asin"]},
                )
            )
        return paths

    def _build_task_anchors(
        self,
        instruction: str,
        task_type: str,
        product_info: Optional[Dict[str, Any]],
        limit: int,
    ) -> List[EvidenceLine]:
        candidate_ids = [
            node_id
            for node_id, node in self.graph.nodes.items()
            if node.node_type in {"category", "brand", "feature", "store", "product", "price_bucket"}
        ]
        ranked_nodes = self.index.rank_nodes(instruction, candidate_ids=candidate_ids, limit=max(limit, 6))
        lines: List[EvidenceLine] = []

        if product_info:
            anchor_chunks = []
            title = normalize_text(product_info.get("title"))
            if title:
                anchor_chunks.append(f"target title: {title}")
            category = normalize_text(product_info.get("main_category"))
            if category:
                anchor_chunks.append(f"target category: {category}")
            brand = extract_brand(product_info)
            if brand:
                anchor_chunks.append(f"target brand: {brand}")
            if anchor_chunks:
                lines.append(
                    EvidenceLine(
                        line_id="anchor:target_product",
                        block="task_anchors",
                        text="Task anchor from target product: " + "; ".join(anchor_chunks) + ".",
                        node_ids=[],
                        edge_ids=[],
                    )
                )

        for idx, (node_id, score) in enumerate(ranked_nodes[:limit]):
            node = self.graph.nodes[node_id]
            lines.append(
                EvidenceLine(
                    line_id=f"anchor:{idx}",
                    block="task_anchors",
                    text=f"Instruction-aligned {node.node_type}: {node.text}.",
                    node_ids=[node_id],
                    edge_ids=self._connected_edge_ids(node_id),
                    score=score,
                    metadata={"node_type": node.node_type},
                )
            )

        return lines[:limit]

    def _collect_support_terms(
        self,
        lines: Iterable[EvidenceLine],
        paths: Iterable[GraphPath],
    ) -> List[str]:
        support_terms = set()
        for line in lines:
            support_terms.update(tokenize_text(line.text))
        for path in paths:
            support_terms.update(tokenize_text(path.text))
        return list(support_terms)

    def _attribute_overlap_score(
        self,
        query_tokens: Sequence[str],
        categories: Sequence[str],
        brand: Optional[str],
        features: Sequence[str],
        store: Optional[str],
    ) -> float:
        overlap = 0.0
        category_terms = tokenize_text(" ".join(categories))
        brand_terms = tokenize_text(brand or "")
        feature_terms = tokenize_text(" ".join(features))
        store_terms = tokenize_text(store or "")
        query_set = set(query_tokens)
        overlap += 1.2 * len(query_set.intersection(category_terms))
        overlap += 1.4 * len(query_set.intersection(brand_terms))
        overlap += 1.0 * len(query_set.intersection(feature_terms))
        overlap += 0.6 * len(query_set.intersection(store_terms))
        return overlap

    def _top_preference_names(self, preference_type: str, positive: bool, limit: int) -> List[str]:
        items = list(self.graph.preference_scores.get(preference_type, {}).values())
        if positive:
            items = [item for item in items if item["score"] > 0]
            items.sort(key=lambda item: (item["score"], item["count"], item["latest_timestamp"]), reverse=True)
        else:
            items = [item for item in items if item["score"] < 0]
            items.sort(key=lambda item: (item["score"], -item["count"], item["latest_timestamp"]))
        return [item["name"] for item in items[:limit]]

    def _record_node_ids(self, record: Dict[str, Any]) -> List[str]:
        node_ids = [
            record["event_node_id"],
            record["product_node_id"],
            record["review_node_id"],
        ]
        for key in ("categories", "features"):
            node_ids.extend(record["node_ids"].get(key, []))
        for key in ("brand", "store", "price_bucket"):
            value = record["node_ids"].get(key)
            if value:
                node_ids.append(value)
        return node_ids

    def _record_edge_ids(self, record: Dict[str, Any]) -> List[str]:
        edge_ids: List[str] = []
        for value in record["edge_ids"].values():
            if isinstance(value, list):
                edge_ids.extend(value)
            elif value:
                edge_ids.append(value)
        return edge_ids

    def _select_best_path_attribute(
        self,
        record: Dict[str, Any],
        task_type: str,
        product_info: Optional[Dict[str, Any]],
    ) -> Tuple[str, Optional[str], Optional[str]]:
        if task_type == "review":
            preferred_features = self._top_preference_names("feature", positive=True, limit=6)
            for feature in record.get("features", []):
                if feature in preferred_features:
                    return "feature", feature, self._feature_node_id(feature)

        preferred_categories = self._top_preference_names("category", positive=True, limit=6)
        for category in record.get("categories", []):
            if category in preferred_categories:
                return "category", category, self._category_node_id(category)

        preferred_brands = self._top_preference_names("brand", positive=True, limit=6)
        brand = record.get("brand")
        if brand and brand in preferred_brands:
            return "brand", brand, self._brand_node_id(brand)

        if task_type in {"search", "review"}:
            preferred_features = self._top_preference_names("feature", positive=True, limit=6)
            for feature in record.get("features", []):
                if feature in preferred_features:
                    return "feature", feature, self._feature_node_id(feature)

        return "", None, None

    def _attribute_relation_edge_ids(self, record: Dict[str, Any], attribute_node_id: str) -> List[str]:
        attribute_edge_ids = []
        for edge_id in self.graph.outgoing.get(record["product_node_id"], []):
            edge = self.graph.edges[edge_id]
            if edge.target == attribute_node_id:
                attribute_edge_ids.append(edge_id)
        return attribute_edge_ids

    def _connected_edge_ids(self, node_id: str) -> List[str]:
        return list(self.graph.outgoing.get(node_id, []))[:4] + list(self.graph.incoming.get(node_id, []))[:4]

    def _profile_node_ids(self, attribute: str, value: str) -> List[str]:
        if not value:
            return []
        node_id = f"profile:{normalize_key(attribute)}:{normalize_key(value)}"
        return [node_id] if node_id in self.graph.nodes else []

    def _profile_edge_ids(self, attribute: str, value: str) -> List[str]:
        if not value:
            return []
        node_ids = self._profile_node_ids(attribute, value)
        edge_ids = []
        for node_id in node_ids:
            for edge_id in self.graph.outgoing.get(self.user_node_id, []):
                edge = self.graph.edges[edge_id]
                if edge.target == node_id and edge.relation == "has_profile_attribute":
                    edge_ids.append(edge_id)
        return edge_ids

    def _user_affinity_edge_ids(self, preference_type: str, names: Sequence[str]) -> List[str]:
        relation = f"{preference_type}_affinity"
        name_set = set(names)
        edge_ids = []
        for edge_id in self.graph.outgoing.get(self.user_node_id, []):
            edge = self.graph.edges[edge_id]
            if edge.relation != relation:
                continue
            target_node = self.graph.nodes.get(edge.target)
            if target_node and target_node.text in name_set:
                edge_ids.append(edge_id)
        return edge_ids

    def _category_node_id(self, name: str) -> str:
        return self._node_id_by_type_and_text("category", name)

    def _brand_node_id(self, name: str) -> str:
        return self._node_id_by_type_and_text("brand", name)

    def _feature_node_id(self, name: str) -> str:
        return self._node_id_by_type_and_text("feature", name)

    def _node_id_by_type_and_text(self, node_type: str, text: str) -> str:
        normalized = normalize_text(text).lower()
        for node_id, node in self.graph.nodes.items():
            if node.node_type == node_type and node.text.lower() == normalized:
                return node_id
        return ""

    def _normalize(self, value: float, minimum: float, maximum: float) -> float:
        if maximum <= minimum:
            return 1.0
        return (value - minimum) / (maximum - minimum)
