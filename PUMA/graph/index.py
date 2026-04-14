from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from .schema import GraphNode


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a",
    "about",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "around",
    "as",
    "at",
    "be",
    "best",
    "but",
    "by",
    "for",
    "find",
    "from",
    "get",
    "great",
    "have",
    "help",
    "i",
    "im",
    "in",
    "into",
    "is",
    "it",
    "keep",
    "like",
    "looking",
    "make",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "product",
    "products",
    "really",
    "recommend",
    "recommendation",
    "reviews",
    "search",
    "some",
    "something",
    "super",
    "that",
    "the",
    "them",
    "there",
    "these",
    "this",
    "to",
    "up",
    "user",
    "want",
    "with",
}


def tokenize_text(text: str) -> List[str]:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text or "")]
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


class LexicalGraphIndex:
    def __init__(self, nodes: Dict[str, GraphNode]):
        self.nodes = nodes
        self.node_tokens: Dict[str, Counter] = {}
        self.document_frequency: Counter = Counter()
        self._build()

    def _build(self) -> None:
        for node_id, node in self.nodes.items():
            tokens = tokenize_text(node.text)
            counts = Counter(tokens)
            self.node_tokens[node_id] = counts
            for token in counts:
                self.document_frequency[token] += 1

    def score_tokens(self, query_tokens: Sequence[str], node_id: str) -> float:
        node_counts = self.node_tokens.get(node_id)
        if not node_counts:
            return 0.0
        total_docs = max(1, len(self.node_tokens))
        score = 0.0
        for token in query_tokens:
            if token not in node_counts:
                continue
            doc_freq = max(1, self.document_frequency.get(token, 1))
            idf = math.log((1 + total_docs) / doc_freq)
            score += node_counts[token] * (1.0 + idf)
        return score

    def score_text(self, query: str, node_id: str) -> float:
        return self.score_tokens(tokenize_text(query), node_id)

    def rank_nodes(
        self,
        query: str,
        candidate_ids: Iterable[str] | None = None,
        limit: int = 10,
    ) -> List[Tuple[str, float]]:
        query_tokens = tokenize_text(query)
        candidate_ids = list(candidate_ids) if candidate_ids is not None else list(self.nodes)
        scored = []
        for node_id in candidate_ids:
            score = self.score_tokens(query_tokens, node_id)
            if score > 0:
                scored.append((node_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]
