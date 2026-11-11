"""Scientific preflight and water-label evidence for aligned rasters and single scenes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ..core.registry import ToolSpec


_REASON_ORDER = (
    "grid_mismatch",
    "shape_mismatch",
    "crs_mismatch",
    "transform_mismatch",
    "bounds_mismatch",
    "resolution_mismatch",
    "nodata_mismatch",
    "valid_mask_mismatch",
)


def _rasterio_numpy():
    try:
        import numpy as np
        import rasterio
    except ImportError as exc:
        raise ImportError("geo_water requires the Terrabox geo dependencies: pip install -e '.[geo]'") from exc
    return rasterio, np


def _normalise_class_table(value: Any) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Accept the public class-table forms and turn malformed input into data."""
    raw = value
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            return [], {"reason_code": "invalid_class_table", "details": f"invalid JSON: {exc.msg}"}
    if isinstance(raw, dict):
        # A wrapped table is convenient for JSON tool callers; a value->label
        # mapping is kept compatible with the earlier demo request shape.
        raw = raw.get("classes", raw)
        if isinstance(raw, dict):
            entries = []
            for key, item in raw.items():
                if isinstance(item, dict):
                    entry = dict(item)
                    entry.setdefault("value", key)
                else:
                    entry = {"value": key, "name": str(item)}
                entries.append(entry)
            raw = entries
    if not isinstance(raw, list):
        return [], {"reason_code": "invalid_class_table", "details": "expected an array or value-to-label object"}
    entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or "value" not in item:
            return [], {"reason_code": "invalid_class_table", "details": f"entry {index} must contain value and name"}
        try:
            numeric = int(item["value"])
        except (TypeError, ValueError, OverflowError):
            return [], {"reason_code": "invalid_class_table", "details": f"entry {index} has a non-integer value"}
        if isinstance(item["value"], bool) or (isinstance(item["value"], float) and item["value"] != numeric):
            return [], {"reason_code": "invalid_class_table", "details": f"entry {index} has a non-integer value"}
        if numeric in seen:
            return [], {"reason_code": "invalid_class_table", "details": f"duplicate class value {numeric}"}
        # ``name`` is the stable public field.  Treat a title-only entry as a
        # malformed table instead of silently accepting a display-only field;
        # this makes caller mistakes visible before raster values are checked.
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            return [], {"reason_code": "invalid_class_table", "details": f"entry {index} needs a non-empty name"}
        entry = dict(item)
        entry["value"] = numeric
        entry["name"] = name.strip()
        seen.add(numeric)
        entries.append(entry)
    return entries, None


def _profile(src) -> dict[str, Any]:
    return {
        "width": src.width,
        "height": src.height,
        "shape": [src.height, src.width],
        "count": src.count,
        "crs": src.crs.to_string() if src.crs else None,
        "transform": list(src.transform)[:6],
        "bounds": [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top],
        "resolution": list(src.res),
        "nodata": list(src.nodatavals),
        "dtype": list(src.dtypes),
    }


def _grid_id(profile: dict[str, Any]) -> str:
    canonical = {
        "shape": profile["shape"],
        "crs": profile["crs"],
        "transform": profile["transform"],
    }
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"grid:{digest[:16]}"


# Projections whose scale grows with latitude (area factor sec^2 of latitude):
# pixel counts times the nominal pixel area misstate ground area, often by tens
# of percent. Transverse Mercator (UTM) is not in this family.
_AREA_DISTORTING_METHODS = (
    "popular visualisation pseudo mercator",
    "mercator (variant a)",
    "mercator (variant b)",
    "mercator (variant c)",
    "mercator (spherical)",
    "mercator",
)

# Largest accepted deviation of the projection's areal scale from 1 anywhere on
# the raster. In-zone UTM stays within about 0.2%; Web Mercator and equidistant
# cylindrical grids are off by tens of percent at mid latitudes.
_MAX_AREAL_SCALE_ERROR = 0.01


def _areal_scale_range(projected, transform, shape) -> tuple[float, float] | None:
    """Projection areal scale (grid area / ground area) at the raster corners and centre.

    Returns None when the scale cannot be evaluated, which callers treat as unsupported.
    """
    from pyproj import Proj, Transformer

    rows, cols = shape if shape else (0, 0)
    pixels = {(0, 0), (cols, 0), (0, rows), (cols, rows), (cols / 2, rows / 2)}
    try:
        to_geographic = Transformer.from_crs(projected, projected.geodetic_crs, always_xy=True)
        factors = Proj(projected)
        scales = []
        for col, row in pixels:
            x = transform.a * col + transform.b * row + transform.c
            y = transform.d * col + transform.e * row + transform.f
            lon, lat = to_geographic.transform(x, y)
            scales.append(float(factors.get_factors(lon, lat).areal_scale))
    except Exception:  # noqa: BLE001 - any projection failure means the area cannot be vouched for
        return None
    if not scales or not all(math.isfinite(scale) and scale > 0 for scale in scales):
        return None
    return min(scales), max(scales)


def _area_support(crs, transform, shape=None) -> dict[str, Any]:
    """Nominal per-pixel ground area, or the reason it cannot be used.

    Requires a projected CRS with linear axis units whose areal scale stays
    within ``_MAX_AREAL_SCALE_ERROR`` of 1 over the raster (``shape`` is
    ``(rows, cols)``; without it only the origin is checked). Geographic
    (degree) grids are refused rather than squaring an angular pixel size.
    """
    if crs is None or not crs.is_projected:
        return {"reason_code": "unsupported_area_crs", "crs": crs.to_string() if crs else None}
    try:
        from pyproj import CRS
    except ImportError as exc:
        raise ImportError("geo_water area calculation requires pyproj from the Terrabox geo dependencies") from exc
    projected = CRS.from_user_input(crs)
    operation = projected.coordinate_operation
    method = operation.method_name if operation is not None else None
    if method and method.strip().lower() in _AREA_DISTORTING_METHODS:
        return {"reason_code": "area_distorting_projection", "crs": crs.to_string(), "projection_method": method}
    axes = projected.axis_info
    if len(axes) < 2 or not all(axis.unit_conversion_factor for axis in axes[:2]):
        return {"reason_code": "unsupported_area_units", "crs": crs.to_string(), "projection_method": method}
    pixel_area_m2 = abs(transform.a * transform.e - transform.b * transform.d)
    pixel_area_m2 *= axes[0].unit_conversion_factor * axes[1].unit_conversion_factor
    if not math.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        return {"reason_code": "invalid_pixel_area", "crs": crs.to_string(), "projection_method": method}
    scale = _areal_scale_range(projected, transform, shape)
    if scale is None:
        return {"reason_code": "unsupported_area_crs", "crs": crs.to_string(), "projection_method": method}
    if max(abs(scale[0] - 1), abs(scale[1] - 1)) > _MAX_AREAL_SCALE_ERROR:
        return {"reason_code": "area_distorting_projection", "crs": crs.to_string(), "projection_method": method,
                "areal_scale_min": scale[0], "areal_scale_max": scale[1],
                "areal_scale_tolerance": _MAX_AREAL_SCALE_ERROR}
    return {"reason_code": None, "crs": crs.to_string(), "projection_method": method, "pixel_area_m2": pixel_area_m2,
            "areal_scale_min": scale[0], "areal_scale_max": scale[1]}


def _same_nodata(first: list[Any], second: list[Any]) -> bool:
    """Nodata lists are equal when every value matches; NaN equals NaN (a common float nodata)."""
    if len(first) != len(second):
        return False
    for a, b in zip(first, second):
        both_nan = isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)
        if not both_nan and a != b:
            return False
    return True


def _pair_preflight(before_path: str, after_path: str, before_grid_id: str | None = None,
                    after_grid_id: str | None = None) -> dict[str, Any]:
    rasterio, np = _rasterio_numpy()
    with rasterio.open(before_path) as before, rasterio.open(after_path) as after:
        before_profile = _profile(before)
        after_profile = _profile(after)
        differences = []

        declared_ids = (before_grid_id, after_grid_id)
        if any(value is not None for value in declared_ids):
            if not all(isinstance(value, str) and value.strip() for value in declared_ids):
                differences.append("grid_mismatch")
            elif before_grid_id != after_grid_id:
                differences.append("grid_mismatch")

        comparisons = (
            ("shape", "shape_mismatch"),
            ("crs", "crs_mismatch"),
            ("transform", "transform_mismatch"),
            ("bounds", "bounds_mismatch"),
            ("resolution", "resolution_mismatch"),
            ("nodata", "nodata_mismatch"),
        )
        for field, reason in comparisons:
            if field == "nodata":
                if not _same_nodata(before_profile[field], after_profile[field]):
                    differences.append(reason)
            elif before_profile[field] != after_profile[field]:
                differences.append(reason)

        if before.count < 1 or after.count < 1:
            differences.append("profile_mismatch")
        elif before.width == after.width and before.height == after.height:
            before_valid = before.dataset_mask() > 0
            after_valid = after.dataset_mask() > 0
            if not np.array_equal(before_valid, after_valid):
                differences.append("valid_mask_mismatch")
            common_valid_pixels = int(np.count_nonzero(before_valid & after_valid))
        else:
            common_valid_pixels = None

    reason_code = next((reason for reason in _REASON_ORDER if reason in differences), None)
    if reason_code is None and differences:
        reason_code = differences[0]
    matched_grid_id = _grid_id(before_profile) if not differences else None
    return {
        "success": not differences,
        "execution_blocked": bool(differences),
        "reason_code": reason_code,
        "failed_checks": differences,
        "grid_id": matched_grid_id,
        "declared_grid_ids": {"before": before_grid_id, "after": after_grid_id},
        "common_valid_pixels": common_valid_pixels,
        "before": before_profile,
        "after": after_profile,
    }


def raster_profile_preflight_handler(arguments: dict, context: Any, account: Any) -> dict[str, Any]:
    """Compare complete raster grids and masks before pixel-wise analysis."""
    return _pair_preflight(
        arguments["before_path"],
        arguments["after_path"],
        arguments.get("before_grid_id"),
        arguments.get("after_grid_id"),
    )


def align_raster_handler(arguments: dict, context: Any, account: Any) -> dict[str, Any]:
    """Resample one raster onto a declared target grid for categorical analysis."""
    rasterio, np = _rasterio_numpy()
    from rasterio.warp import Resampling, reproject

    source_path = arguments["source_path"]
    target_path = arguments["target_path"]
    output_path = arguments["output_path"]
    method = str(arguments.get("resampling") or "nearest").lower()
    if method != "nearest":
        return _blocked("unsupported_resampling", resampling=method, supported_resampling=["nearest"])
    with rasterio.open(source_path) as source, rasterio.open(target_path) as target:
        if source.count != 1 or target.count != 1:
            return _blocked("unsupported_band_count", source_count=source.count, target_count=target.count)
        source_profile = _profile(source)
        target_profile = _profile(target)
        source_nodata = source.nodata
        destination_nodata = arguments.get("dst_nodata")
        if destination_nodata is None:
            destination_nodata = source_nodata if source_nodata is not None else 255
        dtype = np.dtype(source.dtypes[0])
        destination = np.full((target.height, target.width), destination_nodata, dtype=dtype)
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source_nodata,
            dst_transform=target.transform,
            dst_crs=target.crs,
            dst_nodata=destination_nodata,
            resampling=Resampling.nearest,
        )
        profile = target.profile.copy()
        profile.update(dtype=source.dtypes[0], count=1, nodata=destination_nodata, compress="deflate")
        with rasterio.open(output_path, "w", **profile) as output:
            output.write(destination, 1)
        output_profile = _profile(output)
    return {
        "success": True,
        "execution_blocked": False,
        "reason_code": None,
        "failed_checks": [],
        "semantic_class": "categorical_raster_resampled",
        "source_id": str(Path(source_path)),
        "target_id": str(Path(target_path)),
        "output_path": str(Path(output_path)),
        "source_grid_id": _grid_id(source_profile),
        "target_grid_id": _grid_id(target_profile),
        "grid_id": _grid_id(output_profile),
        "source": source_profile,
        "target": target_profile,
        "output": output_profile,
        "resampling": "nearest",
        "source_nodata": source_nodata,
        "output_nodata": destination_nodata,
        "provenance": {
            "method": "rasterio.warp.reproject",
            "resampling": "nearest",
            "target_grid_source": str(Path(target_path)),
        },
        "limitations": [
            "Nearest-neighbour preserves categorical codes but moves class boundaries to the target grid.",
            "The aligned raster is a derived product and must retain this provenance with downstream statistics.",
        ],
    }


def water_evidence_handler(arguments: dict, context: Any, account: Any) -> dict[str, Any]:
    """Summarize supplied 0/1 semantic water labels over common valid support."""
    rasterio, np = _rasterio_numpy()
    before_path = arguments["before_path"]
    after_path = arguments["after_path"]
    check = _pair_preflight(
        before_path,
        after_path,
        arguments.get("before_grid_id"),
        arguments.get("after_grid_id"),
    )
    if check["execution_blocked"]:
        return {**check, "semantic_class": "water_not_flood", "change_answerable": False}

    with rasterio.open(before_path) as before, rasterio.open(after_path) as after:
        before_data = before.read(1)
        after_data = after.read(1)
        valid = (before.dataset_mask() > 0) & (after.dataset_mask() > 0)
        before_valid = np.isfinite(before_data) & np.isin(before_data, (0, 1))
        after_valid = np.isfinite(after_data) & np.isin(after_data, (0, 1))
        valid &= before_valid & after_valid

        invalid_labels = int(np.count_nonzero(
            ((before.dataset_mask() > 0) & ~before_valid) | ((after.dataset_mask() > 0) & ~after_valid)
        ))
        if invalid_labels:
            return {
                "success": False,
                "execution_blocked": True,
                "reason_code": "invalid_water_labels",
                "failed_checks": ["invalid_water_labels"],
                "invalid_label_pixels": invalid_labels,
                "semantic_class": "water_not_flood",
                "change_answerable": False,
            }

        area = _area_support(before.crs, before.transform, before.shape)
        if area.get("reason_code"):
            return {
                "success": False,
                "execution_blocked": True,
                "reason_code": area["reason_code"],
                "failed_checks": [area["reason_code"]],
                "semantic_class": "water_not_flood",
                "change_answerable": False,
                "crs": area.get("crs"),
                "projection_method": area.get("projection_method"),
            }
        pixel_area_m2 = area["pixel_area_m2"]

        valid_pixels = int(np.count_nonzero(valid))
        total_pixels = int(valid.size)
        before_water = int(np.count_nonzero(valid & (before_data == 1)))
        after_water = int(np.count_nonzero(valid & (after_data == 1)))
        new_water = int(np.count_nonzero(valid & (before_data == 0) & (after_data == 1)))
        receded_water = int(np.count_nonzero(valid & (before_data == 1) & (after_data == 0)))
        common_valid_area = valid_pixels * pixel_area_m2
        before_profile = _profile(before)
        after_profile = _profile(after)

    return {
        "success": valid_pixels > 0,
        "execution_blocked": valid_pixels == 0,
        "reason_code": None if valid_pixels else "no_common_valid_pixels",
        "failed_checks": [] if valid_pixels else ["no_common_valid_pixels"],
        "semantic_class": "water_not_flood",
        "support": "common_valid_pair",
        "grid_id": check["grid_id"],
        "source_ids": {"before": str(Path(before_path)), "after": str(Path(after_path))},
        "crs": before_profile["crs"],
        "transform": before_profile["transform"],
        "shape": before_profile["shape"],
        "area_method": "affine_pixel_area_with_projected_crs_unit_conversion",
        "projection_method": area["projection_method"],
        "pixel_area_m2": pixel_area_m2,
        "valid_pixels": valid_pixels,
        "excluded_pixels": total_pixels - valid_pixels,
        "valid_area_m2": common_valid_area if valid_pixels else None,
        "before_water_pixels": before_water if valid_pixels else None,
        "after_water_pixels": after_water if valid_pixels else None,
        "before_water_area_m2": before_water * pixel_area_m2 if valid_pixels else None,
        "after_water_area_m2": after_water * pixel_area_m2 if valid_pixels else None,
        "new_water_area_m2": new_water * pixel_area_m2 if valid_pixels else None,
        "receded_water_area_m2": receded_water * pixel_area_m2 if valid_pixels else None,
        "net_water_change_m2": (after_water - before_water) * pixel_area_m2 if valid_pixels else None,
        "change_answerable": valid_pixels > 0,
        "limitations": [
            "Statistics describe supplied 0/1 water labels, not flood causation or segmentation accuracy.",
            "New-water labels do not establish inundation without validated observations and task-specific evidence.",
        ],
    }


def rasterize_reference_handler(arguments: dict, context: Any, account: Any) -> dict[str, Any]:
    """Rasterize a polygon reference on a target grid, excluding unmapped coverage."""
    import geopandas as gpd
    import pyogrio
    from rasterio.features import rasterize
    from rasterio.warp import transform_geom

    rasterio, np = _rasterio_numpy()
    bounds = arguments.get("coverage_bbox_wgs84")
    if isinstance(bounds, str):
        try:
            bounds = json.loads(bounds)
        except ValueError:
            return _blocked("invalid_reference_coverage")
    if not isinstance(bounds, list) or len(bounds) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bounds):
        return _blocked("invalid_reference_coverage")
    west, south, east, north = bounds
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        return _blocked("invalid_reference_coverage")
    vector_path, target_path, output_path = (arguments[k] for k in ("vector_path", "target_path", "output_path"))
    layers = [str(row[0]) for row in pyogrio.list_layers(vector_path)]
    layer = arguments.get("layer")
    if not layer and len(layers) != 1:
        return _blocked("ambiguous_reference_layer", layers=layers)
    layer = layer or layers[0]
    shapes = []
    source = gpd.read_file(vector_path, layer=layer, engine="pyogrio")
    with rasterio.open(target_path) as target:
        if not target.crs or not source.crs:
            return _blocked("missing_crs")
        profile = target.profile.copy()
        target_profile = _profile(target)
        for feature in source.iterfeatures():
            geom = feature.get("geometry")
            if not geom or geom["type"] not in ("Polygon", "MultiPolygon"):
                return _blocked("invalid_reference_geometry")
            shapes.append(transform_geom(source.crs, target.crs, geom))
        source_crs = str(source.crs)
        coverage = {"type": "Polygon", "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]]}
        coverage = transform_geom("EPSG:4326", target.crs, coverage)
        valid = rasterize([(coverage, 1)], out_shape=target.shape, transform=target.transform, fill=0, all_touched=False, dtype="uint8") > 0
        if not np.any(valid):
            return _blocked("no_reference_coverage")
        water = rasterize([(g, 1) for g in shapes], out_shape=target.shape, transform=target.transform, fill=0, all_touched=False, dtype="uint8") if shapes else np.zeros(target.shape, dtype="uint8")
        data = np.where(valid, water, 255).astype("uint8")
        profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)
    return {
        "success": True, "execution_blocked": False, "reason_code": None, "failed_checks": [],
        "semantic_class": "reference_product_not_ground_truth", "output_path": output_path,
        "grid_id": _grid_id(target_profile), "crs": target_profile["crs"],
        "shape": target_profile["shape"], "transform": target_profile["transform"],
        "valid_pixels": int(valid.sum()), "excluded_pixels": int(valid.size - valid.sum()),
        "reference_water_pixels": int(np.count_nonzero(valid & (water == 1))),
        "source_ids": {"vector": vector_path, "target": target_path}, "source_crs": source_crs,
        "coverage_bbox_wgs84": bounds, "layer": layer, "feature_count": len(shapes),
        "provenance": {"method": "geopandas/pyogrio + rasterio.warp.transform_geom + rasterio.features.rasterize", "all_touched": False,
                       "coverage_rule": "pixel centre inside declared coverage; outside is nodata 255", "labels": {"0": "outside reference polygons within mapped coverage", "1": "inside reference polygons", "255": "outside mapped coverage"}},
        "limitations": ["Reference coverage is a caller declaration; the file alone does not establish that outside polygons was observed dry.",
                        "An older/different-date flood polygon product is not same-date water ground truth."],
    }


def water_extent_handler(arguments: dict, context: Any, account: Any) -> dict[str, Any]:
    """Water area of one 0/1 semantic water-label raster over its valid pixels."""
    rasterio, np = _rasterio_numpy()
    label_path = arguments["label_path"]
    with rasterio.open(label_path) as src:
        data = src.read(1)
        dataset_valid = src.dataset_mask() > 0
        label_valid = np.isfinite(data) & np.isin(data, (0, 1))
        invalid_labels = int(np.count_nonzero(dataset_valid & ~label_valid))
        profile = _profile(src)
        area = _area_support(src.crs, src.transform, src.shape)
    blocked = {"semantic_class": "water_not_flood", "extent_answerable": False, "crs": profile["crs"]}
    if profile["count"] != 1:
        return {**blocked, "success": False, "execution_blocked": True,
                "reason_code": "unsupported_band_count", "failed_checks": ["unsupported_band_count"]}
    if invalid_labels:
        return {**blocked, "success": False, "execution_blocked": True, "reason_code": "invalid_water_labels",
                "failed_checks": ["invalid_water_labels"], "invalid_label_pixels": invalid_labels}
    if area.get("reason_code"):
        return {**blocked, "success": False, "execution_blocked": True, "reason_code": area["reason_code"],
                "failed_checks": [area["reason_code"]], "projection_method": area.get("projection_method")}
    valid = dataset_valid & label_valid
    valid_pixels = int(np.count_nonzero(valid))
    total_pixels = int(valid.size)
    if valid_pixels == 0:
        return {**blocked, "success": False, "execution_blocked": True,
                "reason_code": "no_valid_pixels", "failed_checks": ["no_valid_pixels"]}
    water_pixels = int(np.count_nonzero(valid & (data == 1)))
    pixel_area_m2 = area["pixel_area_m2"]
    return {
        "success": True,
        "execution_blocked": False,
        "reason_code": None,
        "failed_checks": [],
        "semantic_class": "water_not_flood",
        "support": "valid_pixels_of_scene",
        "source_id": str(Path(label_path)),
        "grid_id": _grid_id(profile),
        "crs": profile["crs"],
        "transform": profile["transform"],
        "resolution": profile["resolution"],
        "area_method": "affine_pixel_area_with_projected_crs_unit_conversion",
        "projection_method": area["projection_method"],
        "pixel_area_m2": pixel_area_m2,
        "valid_pixels": valid_pixels,
        "excluded_pixels": total_pixels - valid_pixels,
        "valid_area_m2": valid_pixels * pixel_area_m2,
        "water_pixels": water_pixels,
        "water_area_m2": water_pixels * pixel_area_m2,
        "water_fraction": water_pixels / valid_pixels,
        "extent_answerable": True,
        "limitations": [
            "Statistics describe supplied 0/1 water labels, not flood extent or segmentation accuracy.",
            "One scene shows water at one time; it does not show change or inundation.",
        ],
    }


def _same_grid(paths: dict[str, str]) -> dict[str, Any]:
    """Shape, CRS and transform of every raster must match; nodata may differ (labels vs classes)."""
    rasterio, _ = _rasterio_numpy()
    profiles = {}
    for role, path in paths.items():
        with rasterio.open(path) as src:
            profiles[role] = _profile(src)
    first = next(iter(profiles.values()))
    failed = []
    for field, reason in (("shape", "shape_mismatch"), ("crs", "crs_mismatch"), ("transform", "transform_mismatch")):
        if any(profile[field] != first[field] for profile in profiles.values()):
            failed.append(reason)
    failed += ["unsupported_band_count"] if any(profile["count"] != 1 for profile in profiles.values()) else []
    return {"failed_checks": failed, "profiles": profiles, "grid_id": _grid_id(first)}


def _blocked(reason: str, **extra: Any) -> dict[str, Any]:
    return {"success": False, "execution_blocked": True, "reason_code": reason, "failed_checks": [reason], **extra}


def _read_labels(src, np) -> tuple[Any, Any, int]:
    """0/1 water labels: data, valid mask, and the number of valid pixels that are not 0 or 1."""
    data = src.read(1)
    dataset_valid = src.dataset_mask() > 0
    label_valid = np.isfinite(data) & np.isin(data, (0, 1))
    return data, dataset_valid & label_valid, int(np.count_nonzero(dataset_valid & ~label_valid))


def water_by_class_handler(arguments: dict, context: Any, account: Any) -> dict[str, Any]:
    """Water (and, with a before mask, new water) area inside each class of an aligned class raster."""
    rasterio, np = _rasterio_numpy()
    paths = {"after": arguments["after_path"], "classes": arguments["class_path"]}
    if arguments.get("before_path"):
        paths["before"] = arguments["before_path"]
    classes, class_error = _normalise_class_table(arguments.get("classes"))
    semantic = {"semantic_class": "water_not_flood", "class_answerable": False}
    if class_error:
        return {**_blocked(class_error["reason_code"], details=class_error.get("details")), **semantic}
    declared = {int(entry["value"]): entry for entry in classes}
    nodata_values = {value for value, entry in declared.items() if entry.get("nodata")}
    grid = _same_grid(paths)
    if grid["failed_checks"]:
        reason = grid["failed_checks"][0]
        return {**_blocked(reason), **semantic, "failed_checks": grid["failed_checks"]}
    with rasterio.open(paths["classes"]) as src:
        class_data = src.read(1)
        class_valid = src.dataset_mask() > 0
        area = _area_support(src.crs, src.transform, src.shape)
        crs = src.crs.to_string() if src.crs else None
        transform = list(src.transform)[:6]
    if not declared:
        return {**_blocked("missing_class_table"), **semantic}
    present = set(np.unique(class_data[class_valid]).tolist())
    undeclared = sorted(int(v) for v in present - set(declared))
    if undeclared:
        return {**_blocked("undeclared_class_values", undeclared_values=undeclared[:20]), **semantic}
    if area.get("reason_code"):
        return {**_blocked(area["reason_code"], crs=area.get("crs"), projection_method=area.get("projection_method")),
                **semantic}
    valid = class_valid & ~np.isin(class_data, list(nodata_values))
    masks = {}
    for role in ("after", "before"):
        if role not in paths:
            continue
        with rasterio.open(paths[role]) as src:
            data, label_valid, invalid = _read_labels(src, np)
        if invalid:
            return {**_blocked("invalid_water_labels", invalid_label_pixels=invalid, role=role), **semantic}
        masks[role] = data
        valid &= label_valid
    valid_pixels = int(np.count_nonzero(valid))
    if valid_pixels == 0:
        return {**_blocked("no_valid_pixels"), **semantic}
    pixel_area_m2 = area["pixel_area_m2"]
    after_water = valid & (masks["after"] == 1)
    new_water = valid & (masks["before"] == 0) & (masks["after"] == 1) if "before" in masks else None
    rows = []
    for value, entry in sorted(declared.items()):
        if value in nodata_values:
            continue
        inside = valid & (class_data == value)
        pixels = int(np.count_nonzero(inside))
        if pixels == 0:
            continue
        row = {
            "value": value,
            "name": entry.get("name"),
            "title": entry.get("title") or entry.get("name"),
            "class_area_m2": pixels * pixel_area_m2,
            "water_area_m2": int(np.count_nonzero(inside & after_water)) * pixel_area_m2,
        }
        if new_water is not None:
            row["new_water_area_m2"] = int(np.count_nonzero(inside & new_water)) * pixel_area_m2
            row["new_water_fraction_of_class"] = row["new_water_area_m2"] / row["class_area_m2"]
        rows.append(row)
    total_new = int(np.count_nonzero(new_water)) * pixel_area_m2 if new_water is not None else None
    for row in rows:
        if new_water is not None:
            row["share_of_new_water"] = row["new_water_area_m2"] / total_new if total_new else None
    output_path = arguments.get("output_path")
    if output_path:
        # Class value where (new) water is, 0 elsewhere, 255 outside the valid support.
        target = new_water if new_water is not None else after_water
        layer = np.where(valid, np.where(target, class_data, 0), 255).astype("uint8")
        with rasterio.open(paths["classes"]) as src:
            profile = src.profile.copy()
        profile.update(dtype="uint8", count=1, nodata=255, compress="deflate")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(layer, 1)
    max_row = max(rows, key=lambda row: row.get("new_water_area_m2", row.get("water_area_m2", 0)), default=None)
    return {
        "success": True,
        "execution_blocked": False,
        "reason_code": None,
        "failed_checks": [],
        "semantic_class": "water_not_flood",
        "support": "valid_pixels_of_all_inputs",
        "grid_id": grid["grid_id"],
        "crs": crs,
        "transform": transform,
        "shape": grid["profiles"]["after"]["shape"],
        "source_ids": paths,
        "area_method": "affine_pixel_area_with_projected_crs_unit_conversion",
        "projection_method": area["projection_method"],
        "pixel_area_m2": pixel_area_m2,
        "valid_pixels": valid_pixels,
        "excluded_pixels": int(valid.size) - valid_pixels,
        "valid_area_m2": valid_pixels * pixel_area_m2,
        "water_area_m2": int(np.count_nonzero(after_water)) * pixel_area_m2,
        "new_water_area_m2": total_new,
        "class_table": classes,
        "class_count": len(rows),
        "total_new_water_area_m2": total_new,
        "max_new_water_class": max_row if total_new else None,
        "classes": rows,
        "output_path": output_path,
        "class_answerable": True,
        "limitations": [
            "Areas describe supplied 0/1 water labels intersected with a supplied class raster; both carry their own errors.",
            "Class rasters resampled onto the label grid (nearest neighbour) keep class codes but shift class edges.",
        ],
    }


def compare_with_reference_handler(arguments: dict, context: Any, account: Any) -> dict[str, Any]:
    """Agreement between a predicted 0/1 water raster and an aligned 0/1 reference raster."""
    rasterio, np = _rasterio_numpy()
    paths = {"predicted": arguments["predicted_path"], "reference": arguments["reference_path"]}
    grid = _same_grid(paths)
    semantic = {"semantic_class": "water_not_flood", "comparison_answerable": False}
    if grid["failed_checks"]:
        return {**_blocked(grid["failed_checks"][0]), **semantic, "failed_checks": grid["failed_checks"]}
    data, valid = {}, None
    for role, path in paths.items():
        with rasterio.open(path) as src:
            values, label_valid, invalid = _read_labels(src, np)
            if role == "predicted":
                area = _area_support(src.crs, src.transform, src.shape)
                crs = src.crs.to_string() if src.crs else None
                profile = src.profile.copy()
        if invalid:
            return {**_blocked("invalid_water_labels", invalid_label_pixels=invalid, role=role), **semantic}
        data[role] = values
        valid = label_valid if valid is None else valid & label_valid
    if area.get("reason_code"):
        return {**_blocked(area["reason_code"], crs=area.get("crs"), projection_method=area.get("projection_method")),
                **semantic}
    valid_pixels = int(np.count_nonzero(valid))
    if valid_pixels == 0:
        return {**_blocked("no_valid_pixels"), **semantic}
    predicted, reference = data["predicted"] == 1, data["reference"] == 1
    tp = int(np.count_nonzero(valid & predicted & reference))
    fp = int(np.count_nonzero(valid & predicted & ~reference))
    fn = int(np.count_nonzero(valid & ~predicted & reference))
    tn = valid_pixels - tp - fp - fn

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    output_path = arguments.get("output_path")
    if output_path:
        # 0 both non-water, 1 both water, 2 predicted only, 3 reference only, 255 invalid.
        layer = np.full(predicted.shape, 255, dtype="uint8")
        layer[valid] = 0
        layer[valid & predicted & reference] = 1
        layer[valid & predicted & ~reference] = 2
        layer[valid & ~predicted & reference] = 3
        profile.update(dtype="uint8", count=1, nodata=255, compress="deflate")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(layer, 1)
    pixel_area_m2 = area["pixel_area_m2"]
    return {
        "success": True,
        "execution_blocked": False,
        "reason_code": None,
        "failed_checks": [],
        "semantic_class": "water_not_flood",
        "support": "common_valid_pixels",
        "grid_id": grid["grid_id"],
        "crs": crs,
        "transform": grid["profiles"]["predicted"]["transform"],
        "shape": grid["profiles"]["predicted"]["shape"],
        "source_ids": paths,
        "area_method": "affine_pixel_area_with_projected_crs_unit_conversion",
        "projection_method": area["projection_method"],
        "pixel_area_m2": pixel_area_m2,
        "valid_pixels": valid_pixels,
        "valid_area_m2": valid_pixels * pixel_area_m2,
        "predicted_water_area_m2": (tp + fp) * pixel_area_m2,
        "reference_water_area_m2": (tp + fn) * pixel_area_m2,
        "area_difference_m2": (fp - fn) * pixel_area_m2,
        "agreement_area_m2": tp * pixel_area_m2,
        "predicted_only_area_m2": fp * pixel_area_m2,
        "reference_only_area_m2": fn * pixel_area_m2,
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "true_negative_pixels": tn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "overall_accuracy": (tp + tn) / valid_pixels,
        "output_path": output_path,
        "comparison_answerable": True,
        "limitations": [
            "Agreement with a reference product is not accuracy unless the reference is validated ground truth.",
            "Differences in acquisition date, sensor and mapping method between prediction and reference count as disagreement.",
        ],
    }


def setup(registrar) -> None:
    registrar.toolkit(
        name="geo_water",
        description="Scientific grid preflight and area statistics for semantic water-label rasters (paired or single scene, by class, against a reference).",
        version="0.1.0",
    )
    registrar.tool(
        ToolSpec(
            slug="geo_water.rasterize_reference", name="Rasterize Reference Polygons",
            description="Rasterize an existing polygon reference onto a target raster grid. Requires its declared WGS84 mapped coverage; outside coverage becomes nodata. This preserves reference-product semantics, not ground truth.",
            parameters={"type": "object", "properties": {
                "vector_path": {"type": "string"}, "target_path": {"type": "string"}, "output_path": {"type": "string"},
                "layer": {"type": "string"},
                "coverage_bbox_wgs84": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                                         "description": "Declared mapped coverage [west,south,east,north] in EPSG:4326; do not infer coverage from flood polygon bounds."},
            }, "required": ["vector_path", "target_path", "output_path", "coverage_bbox_wgs84"]},
            requires_connection=False,
        ), rasterize_reference_handler,
    )
    registrar.tool(
        ToolSpec(
            slug="geo_water.raster_profile_preflight",
            name="Raster Profile Preflight",
            description="Compare grid ID, shape, CRS, transform, bounds, resolution, nodata and valid masks before paired raster analysis.",
            parameters={
                "type": "object",
                "properties": {
                    "before_path": {"type": "string"},
                    "after_path": {"type": "string"},
                    "before_grid_id": {"type": "string"},
                    "after_grid_id": {"type": "string"},
                },
                "required": ["before_path", "after_path"],
            },
            requires_connection=False,
        ),
        raster_profile_preflight_handler,
    )
    registrar.tool(
        ToolSpec(
            slug="geo_water.align_raster",
            name="Align Categorical Raster",
            description="Resample one categorical raster onto a target raster grid with nearest-neighbour resampling and explicit provenance.",
            parameters={
                "type": "object",
                "properties": {
                    "source_path": {"type": "string"},
                    "target_path": {"type": "string"},
                    "output_path": {"type": "string"},
                    "resampling": {"type": "string", "enum": ["nearest"], "default": "nearest"},
                    "dst_nodata": {"type": ["number", "null"]},
                },
                "required": ["source_path", "target_path", "output_path"],
            },
            requires_connection=False,
        ),
        align_raster_handler,
    )
    registrar.tool(
        ToolSpec(
            slug="geo_water.water_evidence",
            name="Water Label Evidence",
            description="Compute paired water-label area only after grid preflight. Inputs must be aligned semantic rasters with values 0=non-water, 1=water and nodata masked.",
            parameters={
                "type": "object",
                "properties": {
                    "before_path": {"type": "string"},
                    "after_path": {"type": "string"},
                    "before_grid_id": {"type": "string"},
                    "after_grid_id": {"type": "string"},
                },
                "required": ["before_path", "after_path"],
            },
            requires_connection=False,
        ),
        water_evidence_handler,
    )
    registrar.tool(
        ToolSpec(
            slug="geo_water.water_extent",
            name="Water Label Extent",
            description="Compute water area of one semantic water-label raster (0=non-water, 1=water, nodata masked) over its valid pixels. Blocks geographic grids and area-distorting projections.",
            parameters={
                "type": "object",
                "properties": {"label_path": {"type": "string"}},
                "required": ["label_path"],
            },
            requires_connection=False,
        ),
        water_extent_handler,
    )
    registrar.tool(
        ToolSpec(
            slug="geo_water.water_by_class",
            name="Water Area by Class",
            description="Water area (and new-water area when a before mask is given) inside each class of a class raster (e.g. land cover) on the same grid as the 0/1 water labels. Blocks grid mismatch, undeclared class values, invalid labels, geographic grids and area-distorting projections.",
            parameters={
                "type": "object",
                "properties": {
                    "after_path": {"type": "string"},
                    "before_path": {"type": "string"},
                    "class_path": {"type": "string"},
                    "classes": {
                        "description": "Array of {value, name, optional nodata}; a value-to-label object is accepted for compatibility.",
                        "oneOf": [
                            {"type": "array", "items": {"type": "object", "required": ["value"]}},
                            {"type": "object"},
                            {"type": "string"},
                        ],
                    },
                    "output_path": {"type": "string"},
                },
                "required": ["after_path", "class_path", "classes"],
            },
            requires_connection=False,
        ),
        water_by_class_handler,
    )
    registrar.tool(
        ToolSpec(
            slug="geo_water.compare_with_reference",
            name="Compare Water With Reference",
            description="Agreement between a predicted 0/1 water raster and a 0/1 reference raster on the same grid: precision, recall, F1, IoU and areas of agreement, predicted-only and reference-only water. Blocks grid mismatch, invalid labels, geographic grids and area-distorting projections.",
            parameters={
                "type": "object",
                "properties": {
                    "predicted_path": {"type": "string"},
                    "reference_path": {"type": "string"},
                    "output_path": {"type": "string"},
                },
                "required": ["predicted_path", "reference_path"],
            },
            requires_connection=False,
        ),
        compare_with_reference_handler,
    )
