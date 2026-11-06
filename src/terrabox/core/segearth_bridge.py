"""Transfer image bytes to SegEarth and publish immutable detection evidence.

Model inference stays in the existing remote service. This module uses no
model framework; raw model scores remain logits, never confidence percentages.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

from .utils.runtime_paths import runtime_root

MAX_IMAGE_BYTES = 100 * 1024 * 1024
MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
REQUIRED = {"segearth_candidates.json", "provenance.json", "category_roles.json",
            "category_mapping.json", "segearth_candidate_filtered_labeled_bbox.png"}
SEMANTICS = (
    "Counts are SegEarth model candidate regions extracted from semantic segmentation, "
    "not verified physical object counts. Coordinates and area_px use original image pixels; "
    "there is no geographic CRS or square-metre measurement. mean_logit/max_logit are "
    "uncalibrated model scores, not probabilities. No facility membership is inferred."
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def child(path, root: Path) -> Path:
    result, root = Path(path).resolve(), root.resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError("Path must be inside the configured runtime directory")
    return result


def service_request(path: str, *, data=None, headers=None, timeout=None):
    config_path = Path(os.environ.get("SEGEARTH_CONFIG_FILE", str(Path.home() / ".config/agent-workspace/segearth.json")))
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    base = os.environ.get("SEGEARTH_SERVICE_URL", config.get("service_url", "")).rstrip("/")
    token = os.environ.get("SEGEARTH_SERVICE_TOKEN", "")
    if not token and config.get("service_token_file"):
        token = Path(config["service_token_file"]).read_text().strip()
    if not base or not token:
        raise RuntimeError("SegEarth remote service has not been configured")
    request = Request(base + path, data=data, headers={"Authorization": "Bearer " + token, **(headers or {})})
    try:
        return urlopen(request, timeout=timeout or float(os.environ.get("SEGEARTH_HTTP_TIMEOUT", "7500")))
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read(8192)).get("error", "Remote request failed")
        except (ValueError, UnicodeError):
            detail = "Remote request failed"
        raise RuntimeError(f"SegEarth HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"SegEarth connection failed ({type(exc).__name__}); no automatic retry") from exc


def service_json(path: str, **kwargs):
    with service_request(path, **kwargs) as response:
        return json.load(response)


def image_size(path: Path) -> tuple[int, int]:
    # Pillow is already used by Terrabox's image tools; it is not a model dependency.
    from PIL import Image
    with Image.open(path) as image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Invalid image dimensions")
        return width, height


def snapshot_from_result(result: dict, image: Path, candidates: list, roles: list, dataset_id: str) -> dict:
    width, height = image_size(image)
    provenance = result.get("provenance") or {}
    if provenance.get("input_sha256") != digest(image):
        raise ValueError("SegEarth result input SHA256 does not match the uploaded image")
    if result.get("status") != "completed" or type(result.get("detection_count")) is not int or result["detection_count"] != len(candidates):
        raise ValueError("Completed status and candidate count must match the actual result")
    if any(not isinstance(r, dict) or type(r.get("class_id")) is not int or r["class_id"] < 0
           or not isinstance(r.get("label"), str) or not r["label"].strip()
           or r.get("role") not in {"target", "context"} for r in roles):
        raise ValueError("Category roles must have integer IDs, labels and target/context roles")
    target_roles = {r["class_id"]: r for r in roles if r.get("role") == "target"}
    if len({r["class_id"] for r in roles}) != len(roles):
        raise ValueError("Category IDs must be unique")
    annotations = [{"id": "IMAGE", "level": "area", "label": "Detection image", "bbox_xyxy": [0, 0, width, height]}]
    counts = {key: 0 for key in target_roles}
    for index, row in enumerate(candidates, 1):
        if not isinstance(row, dict):
            raise ValueError("Each candidate must be an object")
        if type(row.get("class_id")) is not int or (row.get("global_class_id") is not None and type(row["global_class_id"]) is not int):
            raise ValueError("Candidate category IDs must be integers")
        role = target_roles.get(row.get("class_id"))
        if not role or row.get("label") != role["label"]:
            raise ValueError("Candidate category must match a declared target class")
        box = row.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
            raise ValueError("Candidate must have four finite pixel coordinates")
        x1, y1, x2, y2 = box
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError("Candidate bounding box exceeds the original image")
        for key in ("mean_logit", "max_logit", "area_px"):
            if key in row and (type(row[key]) not in (int, float) or not math.isfinite(row[key])):
                raise ValueError("Candidate scores/area must be finite numbers")
        if "area_px" in row and not 0 < row["area_px"] <= width * height:
            raise ValueError("Candidate pixel area must be positive and within the image")
        counts[row["class_id"]] += 1
        annotations.append({"id": f"D{index:06d}", "level": "target", "label": row["label"],
                            "area": "Detection image", "facility": "", "target_type": row["label"],
                            "sourceLayer": "segearth_candidates", "detection": row,
                            "bbox_xyxy": box, "note": "Model candidate region in original image pixels"})
    declared_rows = result.get("class_counts", [])
    if not isinstance(declared_rows, list) or any(not isinstance(r, dict)
            or type(r.get("class_id")) is not int or type(r.get("count")) is not int or r["count"] < 0
            or r["class_id"] not in target_roles or r.get("label") != target_roles[r["class_id"]]["label"]
            for r in declared_rows) or len(declared_rows) != len(target_roles):
        raise ValueError("Class counts must identify every declared target class exactly once")
    declared = {r["class_id"]: r["count"] for r in declared_rows}
    if declared != counts:
        raise ValueError("Class counts disagree with candidate records")
    return {"datasetId": dataset_id, "version": "segearth-detections-v1", "synthetic": False,
            "evidence_kind": "model_detection_candidates", "count_unit": "model_candidate_regions",
            "semantic_boundary": SEMANTICS,
            "allowed_categories": [r["label"] for r in target_roles.values()],
            "image": {"width": width, "height": height, "sha256": digest(image),
                      "coordinate_system": "original_image_pixels", "crs": None},
            "detection_provenance": provenance, "annotations": annotations}


def publish_result(result: dict, image: Path, output: Path, runtime: Path, request_id: str, *, origin: str) -> dict:
    """Publish only after input, counts, dimensions and categories agree."""
    candidates = json.loads((output / "segearth_candidates.json").read_text())
    roles = json.loads((output / "category_roles.json").read_text())
    if not isinstance(candidates, list) or not isinstance(roles, list):
        raise ValueError("Expected candidate and category-role arrays")
    if json.loads((output / "provenance.json").read_text()) != result.get("provenance"):
        raise ValueError("Artifact provenance differs from the service response")
    mapping = json.loads((output / "category_mapping.json").read_text())
    classes = {row["local_class_id"]: row for row in mapping["classes"]}
    for row in candidates:
        category = classes.get(row.get("class_id"))
        if not category or category["label"] != row.get("label") or category.get("global_class_id") != row.get("global_class_id"):
            raise ValueError("Local/global category mapping disagrees with candidates")
    dataset_id = "segearth:" + request_id
    snapshot = snapshot_from_result(result, image, candidates, roles, dataset_id)
    if image.suffix.lower() in {".tif", ".tiff"} and not (output / "original_preview.png").exists():
        from PIL import Image
        with Image.open(image) as original:
            original.convert("RGB").save(output / "original_preview.png")
    snapshot_path = output / "graph_annotations.json"
    atomic_json(snapshot_path, snapshot)
    hashes = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    local = {**result, "output_dir": str(output), "request_id": request_id,
             "dataset_id": dataset_id, "snapshot_id": "sha256:" + digest(snapshot_path),
             "image": {**snapshot["image"], "path": str(image)},
             "origin": origin, "registered_at": datetime.now(timezone.utc).isoformat(),
             "semantic_boundary": SEMANTICS, "artifact_sha256": hashes,
             "artifacts": {p.name: str(p) for p in output.iterdir() if p.is_file()}}
    atomic_json(output / "earthmind_result.json", local)
    index = runtime / "detection-datasets" / (request_id + ".json")
    if index.exists():
        raise ValueError("Detection dataset already exists; results are immutable")
    atomic_json(index, {"dataset_id": dataset_id, "snapshot_path": str(snapshot_path),
                        "snapshot_id": local["snapshot_id"], "result_path": str(output / "earthmind_result.json"),
                        "result_sha256": digest(output / "earthmind_result.json")})
    return local


def registered_result(dataset_id: str, runtime: Path | None = None) -> dict:
    runtime = runtime or runtime_root()
    if not re.fullmatch(r"segearth:[a-f0-9]{32}", dataset_id):
        raise ValueError("Invalid detection dataset ID")
    index = json.loads((runtime / "detection-datasets" / (dataset_id.split(":")[1] + ".json")).read_text())
    snapshot = child(index["snapshot_path"], runtime / "outputs")
    if index["dataset_id"] != dataset_id or "sha256:" + digest(snapshot) != index["snapshot_id"]:
        raise ValueError("Detection snapshot integrity check failed")
    result_path = child(index["result_path"], runtime / "outputs")
    if digest(result_path) != index.get("result_sha256"):
        raise ValueError("Detection result metadata integrity check failed")
    result = json.loads(result_path.read_text())
    if result["dataset_id"] != dataset_id or result["snapshot_id"] != index["snapshot_id"]:
        raise ValueError("Detection result and snapshot do not match")
    return result


def detect(arguments: dict, context: dict) -> dict:
    runtime = Path(context["runtime_dir"]).resolve()
    image = child(arguments["input_path"], runtime)
    if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES or not 0 < image.stat().st_size <= MAX_IMAGE_BYTES:
        raise ValueError("Expected a supported runtime image of at most 100 MB")
    image_size(image)
    output = child(context["output_dir"], runtime / "outputs") / ("segearth_" + uuid.uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    request_id = arguments.get("request_id") or uuid.uuid4().hex
    if (runtime / "detection-datasets" / (request_id + ".json")).exists():
        raise ValueError("Request already completed; select the saved result rather than rerunning it")
    payload = {"filename": image.name, "batch": arguments.get("window_batch_size", 2), "job": request_id}
    headers = {"Content-Type": "application/octet-stream"}
    if "prompt_categories" in arguments:
        headers["X-SegEarth-Prompts"] = quote(json.dumps(arguments["prompt_categories"], ensure_ascii=False), safe="~()*!.'-")
    # Send bytes rather than a machine-local path. No retry: disconnects may leave a remote job running.
    result = service_json("/api/detect?" + urlencode(payload), data=image.read_bytes(), headers=headers)
    if not isinstance(result, dict) or result.get("status") != "completed":
        raise RuntimeError("SegEarth returned an incomplete result")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict) or not REQUIRED <= set(artifacts):
        raise RuntimeError("SegEarth result is missing required artifacts")
    total = 0
    for name, remote_path in artifacts.items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", name) or name in {".", ".."} or not isinstance(remote_path, str):
            raise ValueError("Invalid artifact entry")
        with service_request("/api/file?" + urlencode({"path": remote_path})) as response, (output / name).open("xb") as target:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                total += len(block)
                if total > MAX_ARTIFACT_BYTES:
                    raise RuntimeError("SegEarth artifact transfer exceeds the configured 1 GiB limit")
                target.write(block)
    source = output / ("source_image" + image.suffix.lower())
    shutil.copyfile(image, source)
    # Keep the remote result unmodified for audit; expose only verified local paths to callers.
    atomic_json(output / "remote_result.json", result)
    local = publish_result({**result, "execution_id": context.get("execution_id")}, source, output, runtime, request_id, origin="remote_inference")
    return local
