"""Lightweight Terrabox plugin; model dependencies live in a separate service."""
import json
from pathlib import Path

_ROLE_PATH = Path(__file__).parent / "src/algorithm/category_routes.json"
if _ROLE_PATH.is_file():
    _ROLES = {c["label"].casefold(): c["role"] for c in json.loads(_ROLE_PATH.read_text(encoding="utf-8"))["categories"]}
else:
    # A lightweight wheel contains only this module, without the GPU algorithm tree.
    _ROLES = {label: "context" for label in ("background", "ground clutter", "material pile", "open yard",
                                               "shadow", "vehicle-like clutter", "container-like clutter")}


def category_role(category):
    return category.get("role", _ROLES.get(category["label"].strip().casefold(), "target"))

PARAMETERS = {
    "type": "object",
    "x-request-timeout-seconds": 7500,
    "properties": {
        "input_path": {"type": "string", "description": "Uploaded RGB image path in the Terrabox runtime directory."},
        "request_id": {"type": "string", "pattern": "^[a-f0-9]{32}$", "description": "Optional unique request ID for progress correlation; never reuse a completed request."},
        "window_batch_size": {"type": "integer", "minimum": 1, "default": 2},
        "prompt_categories": {"type": "array", "description": "Editable categories: label, prompts and optional role (target or context). Context competes during inference but is excluded from display and counts; background is added automatically.", "minItems": 1, "maxItems": 32, "items": {"type": "object", "properties": {"label": {"type": "string"}, "prompts": {"type": "array", "items": {"type": "string"}}, "role": {"type": "string", "enum": ["target", "context"]}}, "required": ["label", "prompts"], "additionalProperties": False}},
    },
    "required": ["input_path"],
    "additionalProperties": False,
}


def validate_arguments(arguments):
    if not isinstance(arguments, dict):
        raise ValueError("inputs must be an object")
    unknown = set(arguments) - set(PARAMETERS["properties"])
    if unknown:
        raise ValueError(f"Unknown inputs: {sorted(unknown)}")
    image = arguments.get("input_path")
    batch = arguments.get("window_batch_size", 2)
    if not isinstance(image, str) or not image.strip():
        raise ValueError("input_path must be a non-empty string")
    if type(batch) is not int or batch < 1:
        raise ValueError("window_batch_size must be a positive integer")
    result = {"input_path": image, "window_batch_size": batch}
    if "prompt_categories" in arguments:
        categories = arguments["prompt_categories"]
        if not isinstance(categories, list) or not 1 <= len(categories) <= 32:
            raise ValueError("Provide 1..32 prompt categories")
        normalized, seen = [], {"background"}
        for category in categories:
            if not isinstance(category, dict) or not {"label", "prompts"} <= set(category) or set(category) - {"label", "prompts", "role"}:
                raise ValueError("Each category needs label and prompts")
            if "role" in category and category["role"] not in ("target", "context"):
                raise ValueError("Category role must be target or context")
            label, prompts = category["label"], category["prompts"]
            if not isinstance(prompts, list) or not 1 <= len(prompts) <= 40:
                raise ValueError("Each category needs 1..40 prompts")
            for value in [label, *prompts]:
                if not isinstance(value, str) or not value.strip() or len(value) > 160 or any(c in value for c in ',\r\n\x00'):
                    raise ValueError("Labels/prompts must be 1..160 characters without commas or line breaks")
            if label.strip().casefold() in seen:
                raise ValueError("Category labels must be unique; background is reserved")
            seen.add(label.strip().casefold())
            normalized.append({"label": label.strip(), "prompts": [p.strip() for p in prompts]})
            if "role" in category:
                normalized[-1]["role"] = category["role"]
        result["prompt_categories"] = normalized
    if "request_id" in arguments:
        import re
        if not isinstance(arguments["request_id"], str) or not re.fullmatch(r"[a-f0-9]{32}", arguments["request_id"]):
            raise ValueError("request_id must be 32 lowercase hex characters")
        result["request_id"] = arguments["request_id"]
    return result


def handler(arguments: dict, context: dict, account=None) -> dict:
    from ..core.segearth_bridge import detect
    return detect(validate_arguments(arguments), context)


def setup(registrar):
    from ..core.registry import ToolSpec
    registrar.toolkit(name="segearth", description="SegEarth detection with remote image transfer, result preview and immutable graph snapshots", version="1.1.0")
    registrar.tool(ToolSpec(slug="segearth.detect", name="SegEarth target detection",
        description="Detect candidate regions in an uploaded RGB image using the configured SegEarth service. "
                    "Transfers actual image bytes across servers and returns local artifacts, dataset_id and snapshot_id for graph_query tools. "
                    "Counts describe model candidate regions; bbox_xyxy and area_px use original image pixels, with no geographic CRS. "
                    "Raw logits are not calibrated confidence. No automatic retry after timeout.",
        parameters=PARAMETERS, requires_connection=False), handler)
