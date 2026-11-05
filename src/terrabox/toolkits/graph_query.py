"""N070 read-only annotation graph toolkit; no shared service imports."""
from __future__ import annotations

from typing import Any

from ..core.graph_query_data import GraphQueryError, configured_snapshot
from ..core.registry import ToolSpec


_COMMON = {
    "dataset_id": {"type": "string", "description": "Optional expected dataset ID from previous tool provenance; a mismatch is an error."},
    "snapshot_id": {"type": "string", "description": "Optional expected sha256: snapshot ID to pin the evidence version across calls."},
}
_PROPERTIES = {
    "get_item": {
        "item_id": {"type": "string", "description": "Exact existing annotation ID, e.g. N003. Use search_graph for names first."},
    },
    "query_nodes": {
        "partition_id": {"type": "string", "description": "Existing area ID. Invalid IDs fail without expanding the query."},
        "facility_id": {"type": "string", "description": "Existing facility ID; selects target members within that exact area and facility."},
        "node_type": {"type": "string", "description": "Optional annotation hierarchy level. Only facility or target are valid; other values fail core validation. Leave absent unless the task requests a level. Independent of target_type: target includes every business category at that level. For all target members, set node_type=target and leave target_type_contains absent."},
        "target_type_contains": {"type": "string", "description": "Optional business-category filter, separate from node_type. Set only when the user explicitly requests that category. One exact code from allowed_target_types; target here is one category, not all target-level records. Label keywords belong in search_graph."},
        "risk_level": {"type": "string", "description": "Optional risk filter. Only unknown is valid; other values fail core validation. Leave absent unless explicitly requested. Existing annotations have no assessed hazard risk."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 80},
    },
    "search_graph": {
        "query": {"type": "string", "minLength": 1},
        "source_types": {"type": "array", "items": {"type": "string", "enum": ["node", "partition", "edge"]}, "minItems": 1},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
    },
}


def _execute(operation: str, arguments: dict, context: Any, account: Any) -> dict[str, Any]:
    snapshot = None
    try:
        if not isinstance(arguments, dict):
            raise GraphQueryError("invalid_arguments", "Tool arguments must be an object.")
        unknown = sorted(set(arguments) - set(_PROPERTIES[operation]) - set(_COMMON))
        if unknown:
            raise GraphQueryError("unknown_arguments", "Unsupported arguments are rejected, never silently dropped.", arguments=unknown)
        snapshot = configured_snapshot()
        for key, actual in (("dataset_id", snapshot.dataset_id), ("snapshot_id", snapshot.provenance["snapshot_id"])):
            if key in arguments and arguments[key] != actual:
                raise GraphQueryError("snapshot_mismatch", f"Expected {key} does not match the configured snapshot.", actual=actual)
        return getattr(snapshot, operation)(arguments)
    except GraphQueryError as exc:
        result = {"success": False, "ok": False, "execution_blocked": True, "reason_code": exc.code,
                  "error": {"code": exc.code, "message": str(exc), **exc.details}}
        if snapshot:
            result["provenance"] = dict(snapshot.provenance)
        return result


def get_item_handler(arguments: dict, context: Any, account: Any) -> dict[str, Any]:
    return _execute("get_item", arguments, context, account)


def query_nodes_handler(arguments: dict, context: Any, account: Any) -> dict[str, Any]:
    return _execute("query_nodes", arguments, context, account)


def search_graph_handler(arguments: dict, context: Any, account: Any) -> dict[str, Any]:
    return _execute("search_graph", arguments, context, account)


def setup(registrar) -> None:
    registrar.toolkit(name="graph_query", description="Read-only existing annotation graph: details, filtered counts and keyword lookup with snapshot provenance.", version="0.1.0")
    tools = (
        ("get_item", "Get Graph Item", "Read one graph annotation by exact ID with parents and member records. A facility/area has members; a target is a leaf. Names must first be looked up, and ambiguous names never select the first item.", ["item_id"], get_item_handler),
        ("query_nodes", "Query Graph Nodes", "Count or filter existing graph annotations by exact area/facility IDs and hierarchy node_type. Business category target_type_contains is optional: never add it unless requested; node_type=target includes all target-level categories. count is the full matching record count; limit truncates only nodes. Zero matches is valid. Counts describe annotation records, never individual vehicles or newly detected objects.", [], query_nodes_handler),
        ("search_graph", "Search Graph", "Find candidate IDs by keyword in existing structured annotation fields. Returns matched_fields and source types. count is returned candidate count; total_matches is the full keyword match count. For a population count use query_nodes. Edge hits are hierarchy-derived links, not new observations.", ["query"], search_graph_handler),
    )
    for operation, name, description, required, handler in tools:
        registrar.tool(ToolSpec(slug=f"graph_query.{operation}", name=name, description=description,
                                parameters={"type": "object", "properties": {**_PROPERTIES[operation], **_COMMON},
                                            "required": required, "additionalProperties": False},
                                requires_connection=False), handler)
