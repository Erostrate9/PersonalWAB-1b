from __future__ import annotations

from typing import Any, Dict, List, Optional

from .builder import extract_brand, normalize_text
from .schema import RetrievedSubgraph


GRAPH_PARAM_PROMPT = """Below is an instruction that describes a task. Generate the tool parameter that appropriately completes the request.
### Instruction: <Instruction>

Retrieved Graph Evidence:
<GraphEvidence>

Tool: <Tool>

Output Contract:
Return only the tool parameter. Do not include explanation, JSON wrappers, or tool names.

### Tool Parameter:
"""


GRAPH_FUNCTION_PROMPT = """Below is an instruction that describes a task. Choose a tool that appropriately completes the request.
### Instruction: <Instruction>

Retrieved Graph Sketch:
<GraphEvidence>

Choose exactly one tool from:
- search_product_by_query
- get_recommendations_by_history
- add_product_review

### Tool:
"""


def render_product_anchor(product_info: Optional[Dict[str, Any]]) -> str:
    if not product_info:
        return ""
    chunks = []
    title = normalize_text(product_info.get("title"))
    category = normalize_text(product_info.get("main_category"))
    brand = extract_brand(product_info)
    price = product_info.get("price")
    if title:
        chunks.append(f"Title: {title}")
    if category:
        chunks.append(f"Main Category: {category}")
    if brand:
        chunks.append(f"Brand: {brand}")
    if price is not None:
        chunks.append(f"Price: {price}")
    return "\n".join(chunks)


def serialize_subgraph(
    subgraph: RetrievedSubgraph,
    include_headers: bool = True,
    block_limits: Optional[Dict[str, int]] = None,
) -> str:
    block_limits = block_limits or {}
    sections: List[str] = []

    def add_block(header: str, lines: List[str]) -> None:
        if not lines:
            return
        if include_headers:
            sections.append(header)
        sections.extend(lines)

    priors = [f"- {line.text}" for line in subgraph.user_priors[: block_limits.get("user_priors", len(subgraph.user_priors))]]
    recent = [f"- {line.text}" for line in subgraph.recent_evidence[: block_limits.get("recent_evidence", len(subgraph.recent_evidence))]]
    paths = [f"- {path.text}" for path in subgraph.preference_paths[: block_limits.get("preference_paths", len(subgraph.preference_paths))]]
    anchors = [f"- {line.text}" for line in subgraph.task_anchors[: block_limits.get("task_anchors", len(subgraph.task_anchors))]]

    add_block("[User Priors]", priors)
    add_block("[Recent Supporting Evidence]", recent)
    add_block("[Preference Paths]", paths)
    add_block("[Task Anchors]", anchors)
    return "\n".join(sections).strip()


def build_graph_function_prompt(instruction: str, graph_text: str) -> str:
    graph_text = graph_text or "No graph evidence available."
    return GRAPH_FUNCTION_PROMPT.replace("<Instruction>", instruction).replace("<GraphEvidence>", graph_text)


def build_graph_param_prompt(
    instruction: str,
    tool_name: str,
    graph_text: str,
    product_info: Optional[Dict[str, Any]] = None,
) -> str:
    product_anchor = render_product_anchor(product_info)
    full_instruction = instruction
    if product_anchor:
        full_instruction = f"{instruction}\n\nTarget Product:\n{product_anchor}"
    graph_text = graph_text or "No graph evidence available."
    return (
        GRAPH_PARAM_PROMPT.replace("<Instruction>", full_instruction)
        .replace("<GraphEvidence>", graph_text)
        .replace("<Tool>", tool_name)
    )
