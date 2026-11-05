"""Deterministic synthetic contracts; these are not remote-sensing evaluations."""
import hashlib
import json
from pathlib import Path

import pytest

from terrabox.core.graph_query_data import GraphQueryError, GraphSnapshot
from terrabox.toolkits.graph_query import get_item_handler, query_nodes_handler, search_graph_handler, setup


FIXTURE = Path(__file__).parent / "fixtures/graph_query_synthetic.json"


@pytest.fixture(autouse=True)
def synthetic_source(monkeypatch):
    monkeypatch.setenv("EARTHMIND_GRAPH_SOURCE", str(FIXTURE))


def call(handler, arguments):
    return handler(arguments, {}, None)


def test_total_count_precedes_limit_and_distribution_uses_full_set():
    out = call(query_nodes_handler, {"facility_id": "S-F1", "node_type": "target", "limit": 1})
    assert out["success"] and out["count"] == 3 and out["returned_count"] == 1
    assert len(out["nodes"]) == 1 and out["truncated"]
    assert out["by_target_type"] == {"vehicle_or_equipment_area": 2, "building": 1}


def test_legal_zero_is_success_and_never_broadens_scope():
    out = call(query_nodes_handler, {"facility_id": "S-F3", "node_type": "target"})
    assert out["success"] and out["count"] == 0 and out["nodes"] == []
    assert not out["truncated"] and not out["execution_blocked"]


def test_target_level_does_not_implicitly_select_one_business_category():
    """Regression for the actual native task's erroneous extra category filter."""
    all_targets = call(query_nodes_handler, {"facility_id": "S-F1", "node_type": "target", "limit": 1})
    vehicles = call(query_nodes_handler, {"facility_id": "S-F1", "node_type": "target",
                                         "target_type_contains": "vehicle_or_equipment_area", "limit": 1})
    assert all_targets["count"] == 3 and vehicles["count"] == 2
    assert all_targets["filters_applied"] == {"facility_id": "S-F1", "node_type": "target"}
    assert all_targets["returned_count"] == vehicles["returned_count"] == 1
    assert all_targets["truncated"] and vehicles["truncated"]


@pytest.mark.parametrize("args,code", [
    ({"facility_id": "missing"}, "invalid_scope"),
    ({"partition_id": "missing"}, "invalid_scope"),
    ({"facility_id": "S-A"}, "invalid_scope"),
    ({"partition_id": "S-F1"}, "invalid_scope"),
    ({"partition_id": "S-B", "facility_id": "S-F1"}, "scope_mismatch"),
    ({"facility_id": "Twin Facility"}, "ambiguous_name"),
    ({"partition_id": "Synthetic Area A"}, "exact_id_required"),
    ({"partition_id": ""}, "invalid_id"),
    ({"node_type": "area"}, "invalid_node_type"),
    ({"target_type_contains": "车辆"}, "unknown_target_type"),
    ({"risk_level": "high"}, "unknown_risk_level"),
    ({"limit": 0}, "invalid_limit"),
    ({"limit": True}, "invalid_limit"),
    ({"limit": 201}, "invalid_limit"),
    ({"dataset_id": "other-dataset"}, "snapshot_mismatch"),
    ({"snapshot_id": "sha256:other"}, "snapshot_mismatch"),
    ({"source_path": "/etc/passwd"}, "unknown_arguments"),
    ({"sibling_of": "S-T1"}, "unknown_arguments"),
])
def test_invalid_query_parameters_are_blocked_without_results(args, code):
    out = call(query_nodes_handler, args)
    assert out["success"] is False and out["execution_blocked"] is True
    assert out["reason_code"] == code
    assert "count" not in out and "nodes" not in out


def test_same_named_facility_in_other_area_does_not_leak_members():
    one = call(query_nodes_handler, {"facility_id": "S-F1"})
    two = call(query_nodes_handler, {"facility_id": "S-F2"})
    assert one["count"] == 3 and two["count"] == 2
    assert {r["id"] for r in one["nodes"]}.isdisjoint(r["id"] for r in two["nodes"])


def test_details_contain_parent_chain_source_and_true_container_members():
    leaf = call(get_item_handler, {"item_id": "S-T1"})
    assert leaf["item_type"] == "target" and leaf["member_count"] == 0
    assert [r["id"] for r in leaf["parents"]] == ["S-A", "S-F1"]
    assert leaf["item"]["source"]["sourceLayer"] == "synthetic_layer"
    facility = call(get_item_handler, {"item_id": "S-F1"})
    area = call(get_item_handler, {"item_id": "S-A"})
    assert facility["member_count"] == 3 and area["member_count"] == 5
    assert all(r["partition_id"] == "S-A" for r in facility["nodes"])


def test_search_count_is_candidates_not_full_population_and_returns_ambiguity():
    out = call(search_graph_handler, {"query": "车辆区域甲", "source_types": ["node"], "limit": 1})
    assert out["count"] == 1 and out["returned_count"] == 1 and out["total_matches"] == 2
    assert out["truncated"] and "label" in out["matches"][0]["matched_fields"]
    ambiguity = call(get_item_handler, {"item_id": "车辆区域甲"})
    assert ambiguity["reason_code"] == "ambiguous_name"
    assert {r["id"] for r in ambiguity["error"]["candidates"]} == {"S-T1", "S-T4"}


@pytest.mark.parametrize("args,code", [
    ({"query": " "}, "invalid_query"),
    ({"query": "area", "source_types": []}, "invalid_source_types"),
    ({"query": "area", "source_types": ["documents"]}, "invalid_source_types"),
    ({"query": "area", "source_types": "node"}, "invalid_source_types"),
    ({"query": "area", "limit": 51}, "invalid_limit"),
])
def test_invalid_search_parameters_are_blocked(args, code):
    assert call(search_graph_handler, args)["reason_code"] == code


def test_valid_search_zero_and_derived_edge_origin():
    out = call(search_graph_handler, {"query": "does-not-exist", "source_types": ["node"]})
    assert out["success"] and out["total_matches"] == 0 and out["matches"] == []
    edges = call(search_graph_handler, {"query": "S-T1", "source_types": ["edge"]})
    assert edges["total_matches"] == 2
    assert all(m["item"]["derived"] for m in edges["matches"])


def test_invalid_item_and_missing_input_are_failure():
    assert call(get_item_handler, {"item_id": "missing"})["reason_code"] == "item_not_found"
    assert call(get_item_handler, {})["reason_code"] == "invalid_id"
    assert call(get_item_handler, None)["reason_code"] == "invalid_arguments"


def test_snapshot_pin_and_semantics_and_source_unchanged():
    before = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    out = call(query_nodes_handler, {"node_type": "target", "target_type_contains": "vehicle_or_equipment_area"})
    assert out["count"] == 3 and out["count_unit"] == "annotation_records"
    assert "not one vehicle" in out["semantic_boundary"]
    assert out["provenance"]["synthetic"] is True
    assert out["provenance"]["snapshot_sha256"] == before
    pinned = call(get_item_handler, {"item_id": "S-F1", "snapshot_id": out["provenance"]["snapshot_id"]})
    assert pinned["success"] and hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == before


def test_unavailable_source_is_structured_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("EARTHMIND_GRAPH_SOURCE", str(tmp_path / "missing.json"))
    out = call(query_nodes_handler, {})
    assert not out["success"] and out["reason_code"] == "graph_source_unavailable"


def test_duplicate_facility_within_area_is_not_silently_resolved(monkeypatch, tmp_path):
    data = json.loads(FIXTURE.read_bytes())
    data["annotations"].append({"id": "S-F4", "level": "facility", "label": "Twin Facility", "area": "Synthetic Area A"})
    path = tmp_path / "ambiguous-synthetic.json"
    path.write_text(json.dumps(data))
    monkeypatch.setenv("EARTHMIND_GRAPH_SOURCE", str(path))
    out = call(query_nodes_handler, {"facility_id": "S-F1"})
    assert out["reason_code"] == "ambiguous_parent_links" and "count" not in out


def test_duplicate_area_and_duplicate_ids_are_guarded(monkeypatch, tmp_path):
    data = json.loads(FIXTURE.read_bytes())
    data["annotations"].append({"id": "S-C", "level": "area", "label": "Synthetic Area A"})
    path = tmp_path / "ambiguous-area-synthetic.json"
    path.write_text(json.dumps(data))
    monkeypatch.setenv("EARTHMIND_GRAPH_SOURCE", str(path))
    assert call(query_nodes_handler, {"partition_id": "S-A"})["reason_code"] == "ambiguous_parent_links"
    data["annotations"].append(data["annotations"][0])
    with pytest.raises(GraphQueryError, match="IDs must be unique"):
        GraphSnapshot(path, json.dumps(data).encode())


def test_unresolved_facility_parent_never_becomes_global_scope(monkeypatch, tmp_path):
    data = json.loads(FIXTURE.read_bytes())
    data["annotations"].extend([
        {"id": "S-ORPHAN-F", "level": "facility", "label": "Orphan", "area": "Missing Area"},
        {"id": "S-ORPHAN-T", "level": "target", "label": "Orphan Target", "area": "Missing Area", "facility": "Orphan"},
    ])
    path = tmp_path / "unresolved-parent-synthetic.json"
    path.write_text(json.dumps(data))
    monkeypatch.setenv("EARTHMIND_GRAPH_SOURCE", str(path))
    assert call(query_nodes_handler, {"facility_id": "S-ORPHAN-F"})["reason_code"] == "unresolved_parent_scope"
    detail = call(get_item_handler, {"item_id": "S-ORPHAN-T"})
    assert "unresolved_area_parent" in detail["item"]["hierarchy_warnings"]


def test_three_tools_register_without_original_service_or_credentials():
    class Recorder:
        def __init__(self):
            self.tools = {}
        def toolkit(self, **kwargs):
            self.name = kwargs["name"]
        def tool(self, spec, handler):
            self.tools[spec.slug] = (spec, handler)
    reg = Recorder()
    setup(reg)
    assert reg.name == "graph_query"
    assert set(reg.tools) == {"graph_query.get_item", "graph_query.query_nodes", "graph_query.search_graph"}
    assert all(not spec.requires_connection for spec, handler in reg.tools.values())
