"""Synthetic transport/evidence checks. These do not run a detection model."""
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest
from PIL import Image

from terrabox.core import segearth_bridge as bridge
from terrabox.toolkits.segearth import handler, validate_arguments
from terrabox.toolkits.graph_query import query_nodes_handler, get_item_handler


@pytest.fixture
def remote(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    image = runtime / "uploads" / "synthetic.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (20, 10), "white").save(image)
    provenance = {"input_sha256": bridge.digest(image), "elapsed_seconds": 0.01,
                  "coordinate_system": "original image pixels", "source_model": "segearth"}
    rows = [{"label": "vehicle", "class_id": 4, "global_class_id": 5,
             "bbox_xyxy": [1, 1, 7, 5], "area_px": 15, "mean_logit": 2.5, "max_logit": 7.0}]
    roles = [{"class_id": 0, "label": "background", "role": "context"},
             {"class_id": 1, "label": "building", "role": "target"},
             {"class_id": 4, "label": "vehicle", "role": "target"}]
    mapping = {"classes": [{"local_class_id": r["class_id"], "label": r["label"], "global_class_id": {0: 0, 1: 1, 4: 5}[r["class_id"]]} for r in roles]}
    result = {"status": "completed", "detection_count": 1, "provenance": provenance,
              "class_counts": [{**r, "count": int(r["class_id"] == 4)} for r in roles if r["role"] == "target"]}
    files = {"segearth_candidates.json": json.dumps(rows).encode(), "category_roles.json": json.dumps(roles).encode(),
             "category_mapping.json": json.dumps(mapping).encode(), "provenance.json": json.dumps(provenance).encode(),
             "segearth_candidate_filtered_labeled_bbox.png": image.read_bytes()}
    state = {"posts": 0, "image_bytes": None, "status": 200, "result": result, "files": files}

    class Remote(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            state["posts"] += 1
            assert self.headers["Authorization"] == "Bearer synthetic-service-token"
            state["image_bytes"] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(state["status"])
            self.end_headers()
            payload = {**state["result"], "artifacts": {name: "/remote/outputs/" + name for name in state["files"]}}
            self.wfile.write(json.dumps(payload if state["status"] == 200 else {"error": "synthetic busy"}).encode())
        def do_GET(self):
            from urllib.parse import parse_qs, urlsplit
            name = Path(parse_qs(urlsplit(self.path).query)["path"][0]).name
            self.send_response(200)
            self.end_headers()
            self.wfile.write(state["files"][name])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Remote)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setenv("SEGEARTH_SERVICE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("SEGEARTH_SERVICE_TOKEN", "synthetic-service-token")
    monkeypatch.setenv("SEGEARTH_CONFIG_FILE", str(tmp_path / "absent.json"))
    monkeypatch.setenv("TERRABOX_RUNTIME_DIR", str(runtime))
    output = runtime / "outputs/test/exec/segearth"
    output.mkdir(parents=True)
    context = {"runtime_dir": str(runtime), "output_dir": str(output), "execution_id": "synthetic-exec"}
    yield image, context, state
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)


def test_actual_http_transfer_and_same_snapshot_query(remote):
    image, context, state = remote
    result = handler({"input_path": str(image), "request_id": "a" * 32}, context)
    assert state["posts"] == 1 and state["image_bytes"] == image.read_bytes()
    assert result["execution_id"] == "synthetic-exec"
    assert all(Path(p).is_file() for p in result["artifacts"].values())
    query = query_nodes_handler({"dataset_id": result["dataset_id"], "snapshot_id": result["snapshot_id"],
                                 "node_type": "target", "target_type_contains": "vehicle", "limit": 1}, {}, None)
    assert query["success"] and query["count"] == 1 and query["count_unit"] == "model_candidate_regions"
    assert query["provenance"]["category_method"] == "explicit_model_category"
    assert query["nodes"][0]["bbox_xyxy"] == [1, 1, 7, 5]
    assert query["nodes"][0]["detection"]["mean_logit"] == 2.5
    assert "confidence" not in query["nodes"][0]["detection"]
    detail = get_item_handler({"dataset_id": result["dataset_id"], "item_id": "D000001"}, {}, None)
    assert detail["parents"][0]["id"] == "IMAGE" and detail["annotation"]["detection"]["global_class_id"] == 5
    zero = query_nodes_handler({"dataset_id": result["dataset_id"], "target_type_contains": "building"}, {}, None)
    assert zero["success"] and zero["count"] == 0
    with pytest.raises(ValueError, match="already completed"):
        handler({"input_path": str(image), "request_id": "a" * 32}, context)
    assert state["posts"] == 1


@pytest.mark.parametrize("fault", ["input_hash", "count", "bounds", "context_class", "global_class", "provenance", "nan", "bool_class", "negative_area"])
def test_mismatched_evidence_is_not_published(remote, fault):
    image, context, state = remote
    if fault == "input_hash":
        state["result"]["provenance"]["input_sha256"] = "0" * 64
        state["files"]["provenance.json"] = json.dumps(state["result"]["provenance"]).encode()
    elif fault == "count": state["result"]["detection_count"] = 2
    elif fault == "provenance": state["files"]["provenance.json"] = b'{}'
    else:
        rows = json.loads(state["files"]["segearth_candidates.json"])
        if fault == "bounds": rows[0]["bbox_xyxy"] = [1, 1, 21, 5]
        elif fault == "context_class": rows[0].update(class_id=0, label="background", global_class_id=0)
        elif fault == "global_class": rows[0]["global_class_id"] = 4
        elif fault == "nan": rows[0]["mean_logit"] = float("nan")
        elif fault == "bool_class": rows[0]["class_id"] = True
        elif fault == "negative_area": rows[0]["area_px"] = -1
        state["files"]["segearth_candidates.json"] = json.dumps(rows).encode()
    with pytest.raises((ValueError, RuntimeError)):
        handler({"input_path": str(image)}, context)
    assert not list((Path(context["runtime_dir"]) / "detection-datasets").glob("*.json"))


def test_remote_busy_is_not_retried_or_published(remote):
    image, context, state = remote
    state["status"] = 429
    with pytest.raises(RuntimeError, match="HTTP 429"):
        handler({"input_path": str(image)}, context)
    assert state["posts"] == 1
    assert not list((Path(context["runtime_dir"]) / "detection-datasets").glob("*.json"))


def test_changed_snapshot_is_rejected(remote):
    image, context, _ = remote
    result = handler({"input_path": str(image)}, context)
    Path(result["artifacts"]["graph_annotations.json"]).write_text('{}')
    out = query_nodes_handler({"dataset_id": result["dataset_id"]}, {}, None)
    assert not out["success"] and out["reason_code"] == "detection_dataset_unavailable"


@pytest.mark.parametrize("value", ["../../etc/passwd", "a" * 31, 123])
def test_request_id_is_not_a_path(value):
    with pytest.raises(ValueError): validate_arguments({"input_path": "x.png", "request_id": value})


def test_changed_result_metadata_is_rejected(remote):
    image, context, _ = remote
    result = handler({"input_path": str(image)}, context)
    path = Path(result["output_dir"]) / "earthmind_result.json"
    data = json.loads(path.read_text())
    data["detection_count"] = 999
    path.write_text(json.dumps(data))
    out = query_nodes_handler({"dataset_id": result["dataset_id"]}, {}, None)
    assert not out["success"] and out["reason_code"] == "detection_dataset_unavailable"
