from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEdge:
    edge_id: str
    source: str
    target: str
    relation: str
    weight: float = 1.0
    timestamp: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphPath:
    path_id: str
    text: str
    node_ids: List[str] = field(default_factory=list)
    edge_ids: List[str] = field(default_factory=list)
    score: float = 0.0
    hop_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceLine:
    line_id: str
    block: str
    text: str
    node_ids: List[str] = field(default_factory=list)
    edge_ids: List[str] = field(default_factory=list)
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedSubgraph:
    user_id: str
    task_type: str
    timestamp: int
    instruction: str
    user_priors: List[EvidenceLine] = field(default_factory=list)
    recent_evidence: List[EvidenceLine] = field(default_factory=list)
    preference_paths: List[GraphPath] = field(default_factory=list)
    task_anchors: List[EvidenceLine] = field(default_factory=list)
    seed_node_ids: List[str] = field(default_factory=list)
    node_ids: List[str] = field(default_factory=list)
    edge_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def all_evidence(self) -> List[EvidenceLine]:
        path_lines = [
            EvidenceLine(
                line_id=path.path_id,
                block="preference_paths",
                text=path.text,
                node_ids=path.node_ids,
                edge_ids=path.edge_ids,
                score=path.score,
                metadata={"hop_count": path.hop_count, **path.metadata},
            )
            for path in self.preference_paths
        ]
        return self.user_priors + self.recent_evidence + path_lines + self.task_anchors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "task_type": self.task_type,
            "timestamp": self.timestamp,
            "instruction": self.instruction,
            "user_priors": [line.to_dict() for line in self.user_priors],
            "recent_evidence": [line.to_dict() for line in self.recent_evidence],
            "preference_paths": [path.to_dict() for path in self.preference_paths],
            "task_anchors": [line.to_dict() for line in self.task_anchors],
            "seed_node_ids": list(self.seed_node_ids),
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "metadata": self.metadata,
        }


@dataclass
class UserGraph:
    user_id: str
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: Dict[str, GraphEdge] = field(default_factory=dict)
    outgoing: Dict[str, List[str]] = field(default_factory=dict)
    incoming: Dict[str, List[str]] = field(default_factory=dict)
    interaction_records: List[Dict[str, Any]] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    preference_scores: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node
        self.outgoing.setdefault(node.node_id, [])
        self.incoming.setdefault(node.node_id, [])

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges[edge.edge_id] = edge
        self.outgoing.setdefault(edge.source, []).append(edge.edge_id)
        self.incoming.setdefault(edge.target, []).append(edge.edge_id)

    def get_node(self, node_id: str) -> GraphNode:
        return self.nodes[node_id]

    def neighbors(self, node_id: str) -> List[GraphNode]:
        neighbor_ids = []
        for edge_id in self.outgoing.get(node_id, []):
            neighbor_ids.append(self.edges[edge_id].target)
        return [self.nodes[target_id] for target_id in neighbor_ids if target_id in self.nodes]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": {edge_id: edge.to_dict() for edge_id, edge in self.edges.items()},
            "interaction_records": self.interaction_records,
            "user_profile": self.user_profile,
            "preference_scores": self.preference_scores,
            "metadata": self.metadata,
        }
