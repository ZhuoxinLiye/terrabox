"""Read-only queries over one explicitly configured annotation snapshot.

This module uses only the standard library. It neither imports the original
Graph-Agent service nor loads documents, models, or editable graph state.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path(__file__).resolve().parents[4] / "graph-agent/data/nnss_annotations_5.21.1_json_aligned_to_csv.json"
SEMANTICS = (
    "Counts refer to existing annotation records, not detected physical objects. "
    "A vehicle_or_equipment_area record is an annotated area, not one vehicle. "
    "The graph does not establish flood cause, new inundation, or current real-world conditions."
)


class GraphQueryError(ValueError):
    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code = code
        self.details = details


def category_for(label: str, level: str) -> str:
    """Match the original store's label-derived category vocabulary (not a model)."""
    if level == "facility":
        return "facility"
    for words, code in (
        (("车辆", "车", "集结"), "vehicle_or_equipment_area"),
        (("机场", "机坪", "停机坪", "航空"), "aviation_area"),
        (("楼", "建筑", "宿舍", "办公"), "building"),
        (("堆场", "建材", "吊装"), "storage_or_industrial_area"),
        (("训练", "靶", "测试"), "training_or_test_area"),
    ):
        if any(word in label for word in words):
            return code
    return "target"


def checked_limit(value: Any, default: int, ceiling: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
        raise GraphQueryError("invalid_limit", f"limit must be an integer in [1, {ceiling}].")
    return value


class GraphSnapshot:
    def __init__(self, source_path: Path, source_bytes: bytes):
        self.path = source_path
        self.sha256 = hashlib.sha256(source_bytes).hexdigest()
        data = json.loads(source_bytes)
        if not isinstance(data, dict) or not isinstance(data.get("annotations"), list):
            raise GraphQueryError("unsupported_source_format", "Expected an annotations JSON snapshot.")
        self.dataset_id = str(data.get("datasetId") or data.get("dataset_id") or f"annotations:{self.sha256[:16]}")
        self.provenance = {
            "dataset_id": self.dataset_id, "snapshot_sha256": self.sha256,
            "snapshot_id": f"sha256:{self.sha256}", "source_path": str(source_path),
            "source_version": data.get("version") or data.get("schemaVersion"),
            "source_format": "graph_annotations_json", "read_only": True,
            "synthetic": bool(data.get("synthetic", False)),
            "evidence_kind": "existing_annotation_records",
            "adapter_version": "0.1.0",
            "category_method": "original_graph_store_label_rules_v1",
        }
        self.items: dict[str, dict[str, Any]] = {}
        for index, annotation in enumerate(data["annotations"]):
            if not isinstance(annotation, dict):
                raise GraphQueryError("invalid_source_record", "Annotation must be an object.", index=index)
            item_id = annotation.get("id")
            level = annotation.get("level")
            if not isinstance(item_id, str) or not item_id or level not in ("area", "facility", "target"):
                raise GraphQueryError("invalid_source_record", "Expected nonempty id and area/facility/target level.", index=index)
            if item_id in self.items:
                raise GraphQueryError("duplicate_source_id", "Annotation IDs must be unique.", item_id=item_id)
            label = annotation.get("label") or item_id
            if not isinstance(label, str) or any(
                annotation.get(key) is not None and not isinstance(annotation.get(key), str)
                for key in ("area", "facility", "note")
            ):
                raise GraphQueryError("invalid_source_record", "Annotation text fields must be strings.", index=index)
            self.items[item_id] = {
                "id": item_id, "item_type": level, "label": label,
                "node_type": level if level != "area" else None,
                "target_type": category_for(label, level) if level != "area" else None,
                "risk_level": "unknown", "area": annotation.get("area") or "",
                "facility": annotation.get("facility") or "", "note": annotation.get("note") or "",
                "source": {"annotation_id": item_id, "sourceLayer": annotation.get("sourceLayer"),
                           "snapshot_sha256": self.sha256},
                "annotation": annotation,
            }
        self.partitions = [r for r in self.items.values() if r["item_type"] == "area"]
        self.nodes = [r for r in self.items.values() if r["item_type"] != "area"]
        self.allowed_categories = sorted({r["target_type"] for r in self.nodes if r["item_type"] == "target"})
        for row in self.nodes:
            parents = [p["id"] for p in self.partitions if p["label"] == row["area"]]
            row["partition_id"] = parents[0] if len(parents) == 1 else None
            row["partition_candidates"] = parents
        for row in self.nodes:
            parents = [f["id"] for f in self.nodes if f["item_type"] == "facility"
                       and f["label"] == row["facility"] and f["partition_id"] == row["partition_id"]]
            row["facility_id"] = parents[0] if row["item_type"] == "target" and len(parents) == 1 else None
            row["facility_candidates"] = parents if row["item_type"] == "target" else []
            row["hierarchy_warnings"] = []
            if not row["partition_id"]:
                row["hierarchy_warnings"].append("ambiguous_area_parent" if len(row["partition_candidates"]) > 1 else "unresolved_area_parent")
            if row["item_type"] == "target" and row["facility"] and not row["facility_id"]:
                row["hierarchy_warnings"].append("ambiguous_facility_parent" if len(parents) > 1 else "unresolved_facility_parent")
        self.edges = []
        for row in self.nodes:
            for parent_key, relation in (("partition_id", "BELONGS_TO_PARTITION"), ("facility_id", "BELONGS_TO_FACILITY")):
                if row.get(parent_key):
                    edge_id = "derived:" + hashlib.sha256(f'{row["id"]}:{relation}:{row[parent_key]}'.encode()).hexdigest()[:16]
                    self.edges.append({"id": edge_id, "source": row["id"], "target": row[parent_key],
                                       "type": relation, "derived": True,
                                       "description": f'{row["label"]} {relation} {self.items[row[parent_key]]["label"]}'})

    def envelope(self, **result: Any) -> dict[str, Any]:
        return {"success": True, "ok": True, "execution_blocked": False,
                "provenance": dict(self.provenance), "count_unit": "annotation_records",
                "semantic_boundary": SEMANTICS, **result}

    @staticmethod
    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in row.items() if k not in ("annotation", "partition_candidates", "facility_candidates")}

    def resolve_id(self, value: Any, level: str | None = None) -> dict[str, Any]:
        if not isinstance(value, str) or not value.strip():
            raise GraphQueryError("invalid_id", "Provide a nonempty exact object ID.")
        row = self.items.get(value)
        if row is None:
            candidates = [self.compact(r) for r in self.items.values() if r["label"].casefold() == value.casefold()
                          and (level is None or r["item_type"] == level)]
            if candidates:
                code = "ambiguous_name" if len(candidates) > 1 else "exact_id_required"
                raise GraphQueryError(code, "Use an exact ID from candidates; names are never resolved by picking the first.", candidates=candidates)
            raise GraphQueryError("invalid_scope" if level else "item_not_found", "ID does not exist in this snapshot.", given=value)
        if level and row["item_type"] != level:
            raise GraphQueryError("invalid_scope", f"Expected a {level} ID.", given=value, actual_type=row["item_type"])
        return row

    def get_item(self, args: dict[str, Any]) -> dict[str, Any]:
        row = self.resolve_id(args.get("item_id"))
        members = []
        if row["item_type"] == "area":
            self._guard_area_links(row["id"])
            members = [r for r in self.nodes if r["partition_id"] == row["id"]]
        elif row["item_type"] == "facility":
            self._guard_facility_links(row["id"])
            members = [r for r in self.nodes if r["facility_id"] == row["id"]]
        parent_ids = [v for v in (row.get("partition_id"), row.get("facility_id")) if v]
        return self.envelope(item_id=row["id"], item_type=row["item_type"], item=self.compact(row),
                             annotation=row["annotation"], parents=[self.compact(self.items[x]) for x in parent_ids],
                             parent_candidates={"partition": row.get("partition_candidates", []),
                                                "facility": row.get("facility_candidates", [])},
                             nodes=[self.compact(r) for r in members[:80]], member_count=len(members),
                             returned_count=min(len(members), 80), truncated=len(members) > 80)

    def _guard_area_links(self, partition_id: str) -> None:
        ambiguous = [r["id"] for r in self.nodes if partition_id in r["partition_candidates"]
                     and len(r["partition_candidates"]) > 1]
        if ambiguous:
            raise GraphQueryError("ambiguous_parent_links", "Source area names are ambiguous; refusing an uncertain scope.", annotation_ids=ambiguous)

    def _guard_facility_links(self, facility_id: str) -> None:
        facility = self.items[facility_id]
        if not facility.get("partition_id"):
            raise GraphQueryError("unresolved_parent_scope", "Facility has no unique area parent; refusing an uncertain scope.", facility_id=facility_id)
        ambiguous = [r["id"] for r in self.nodes if facility_id in r["facility_candidates"]
                     and len(r["facility_candidates"]) > 1]
        if ambiguous:
            raise GraphQueryError("ambiguous_parent_links", "Source facility names are ambiguous within an area; refusing an uncertain scope.", annotation_ids=ambiguous)

    def query_nodes(self, args: dict[str, Any]) -> dict[str, Any]:
        limit = checked_limit(args.get("limit"), 80, 200)
        rows = list(self.nodes)
        filters = {k: v for k, v in args.items() if k not in ("limit", "dataset_id", "snapshot_id")}
        for key, level in (("partition_id", "area"), ("facility_id", "facility")):
            if key in args:
                scope = self.resolve_id(args[key], level)
                if level == "area":
                    self._guard_area_links(scope["id"])
                else:
                    self._guard_facility_links(scope["id"])
                    if args.get("partition_id") and scope.get("partition_id") != args["partition_id"]:
                        raise GraphQueryError("scope_mismatch", "Facility is outside the requested area; scope was not expanded.")
                rows = [r for r in rows if r.get(key) == scope["id"]]
        node_type = args.get("node_type")
        if "node_type" in args and node_type not in ("facility", "target"):
            raise GraphQueryError("invalid_node_type", "node_type must be facility or target.")
        if node_type:
            rows = [r for r in rows if r["node_type"] == node_type]
        category = args.get("target_type_contains")
        if "target_type_contains" in args:
            if category not in self.allowed_categories:
                raise GraphQueryError("unknown_target_type", "Use one exact category code; use search_graph for label keywords.", allowed_target_types=self.allowed_categories)
            rows = [r for r in rows if r["target_type"] == category]
        if "risk_level" in args:
            if args["risk_level"] != "unknown":
                raise GraphQueryError("unknown_risk_level", "This annotation snapshot has no assessed risk; only unknown is supported.")
            rows = [r for r in rows if r["risk_level"] == args["risk_level"]]
        return self.envelope(count=len(rows), returned_count=min(len(rows), limit), truncated=len(rows) > limit,
                             nodes=[self.compact(r) for r in rows[:limit]], filters_applied=filters,
                             by_node_type=dict(Counter(r["node_type"] for r in rows)),
                             by_target_type=dict(Counter(r["target_type"] for r in rows)),
                             count_semantics="total_matches_before_limit", allowed_target_types=self.allowed_categories)

    def search_graph(self, args: dict[str, Any]) -> dict[str, Any]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise GraphQueryError("invalid_query", "Provide a nonempty keyword.")
        limit = checked_limit(args.get("limit"), 20, 50)
        sources = args.get("source_types", ["node", "partition", "edge"])
        if not isinstance(sources, list) or not sources or any(x not in ("node", "partition", "edge") for x in sources):
            raise GraphQueryError("invalid_source_types", "source_types must be a nonempty array of node, partition, edge.")
        matches = []
        needle = query.strip().casefold()
        for source_type, rows, fields in (
            ("node", self.nodes, ("id", "label", "area", "facility", "note", "target_type")),
            ("partition", self.partitions, ("id", "label", "note")),
            ("edge", self.edges, ("id", "source", "target", "type", "description")),
        ):
            if source_type not in sources:
                continue
            for row in rows:
                matched = [f for f in fields if needle in str(row.get(f, "")).casefold()]
                if matched:
                    matches.append({"source_type": source_type, "id": row["id"], "matched_fields": matched,
                                    "item": self.compact(row)})
        return self.envelope(query=query.strip(), matches=matches[:limit], count=min(len(matches), limit),
                             returned_count=min(len(matches), limit), total_matches=len(matches), truncated=len(matches) > limit,
                             count_semantics="returned_candidates_not_node_population", source_types=sources,
                             lookup_semantics="case_insensitive_substring_in_structured_fields")


@lru_cache(maxsize=4)
def _load_snapshot(path: str, mtime_ns: int, size: int) -> GraphSnapshot:
    return GraphSnapshot(Path(path), Path(path).read_bytes())


def configured_snapshot() -> GraphSnapshot:
    """The caller cannot select arbitrary files: source is server configuration only."""
    path = Path(os.environ.get("EARTHMIND_GRAPH_SOURCE", str(DEFAULT_SOURCE))).resolve()
    try:
        stat = path.stat()
        return _load_snapshot(str(path), stat.st_mtime_ns, stat.st_size)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GraphQueryError("graph_source_unavailable", "Configured graph snapshot could not be read as JSON.", error_type=type(exc).__name__) from exc
