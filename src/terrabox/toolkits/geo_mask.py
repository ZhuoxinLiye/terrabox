"""General statistics on categorical mask rasters: class area, phase series, components, overlap.

A mask is whatever the caller supplies (a perception-model estimate, a reference label or a
threshold product); these tools only count pixels, so their precision never exceeds the mask's
own accuracy. Area needs a trustworthy pixel size:

* a raster on a projected metre grid (not Mercator) uses its own grid resolution;
* an image without any georeferencing (e.g. a PNG tile) needs a caller-declared ``pixel_size_m``,
  and the result is marked ``declared_pixel_size_unverified``;
* geographic (degree) and Mercator grids are refused instead of squaring an angular or
  latitude-stretched pixel size.

Pixels equal to the dataset nodata value are excluded from every denominator.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ..core.registry import ToolSpec

_MERCATOR_METHODS = (
    "popular visualisation pseudo mercator",
    "mercator (variant a)",
    "mercator (variant b)",
    "mercator (variant c)",
    "mercator (spherical)",
    "mercator",
)
_MAX_UNIQUE = 64
_MAX_ITEMS = 60


def _deps():
    try:
        import numpy as np
        import rasterio
    except ImportError as exc:
        raise ImportError("geo_mask requires the Terrabox geo dependencies: pip install -e '.[geo]'") from exc
    return rasterio, np


def _blocked(reason: str, **extra: Any) -> dict[str, Any]:
    return {"success": False, "execution_blocked": True, "reason_code": reason, "failed_checks": [reason], **extra}


class _Blocked(Exception):
    def __init__(self, reason: str, **extra: Any):
        super().__init__(reason)
        self.payload = _blocked(reason, **extra)


def _guard(handler):
    def wrapped(arguments: dict, context: Any = None, account: Any = None) -> dict[str, Any]:
        try:
            return handler(arguments or {})
        except _Blocked as refusal:
            return refusal.payload
    wrapped.__name__ = handler.__name__
    wrapped.__doc__ = handler.__doc__
    return wrapped


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _json_argument(value: Any, name: str) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if name == "class_values":
                try:
                    return [int(piece) for piece in text.replace(";", ",").split(",") if piece.strip()]
                except ValueError as exc:
                    raise _Blocked("invalid_class_values", details="expected integers such as [255] or 1,2") from exc
            raise _Blocked(f"invalid_{name}", details="expected JSON") from None
    return value


def _class_values(value: Any) -> list[int]:
    raw = _json_argument(value, "class_values")
    if raw is None or raw == []:
        raise _Blocked("missing_class_values", details="State which mask values count, e.g. [255] or [1].")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = [raw]
    if isinstance(raw, dict):
        raw = list(raw.keys())
    values = []
    for item in raw:
        try:
            number = int(item)
        except (TypeError, ValueError) as exc:
            raise _Blocked("invalid_class_values", details=f"{item!r} is not an integer class value") from exc
        if isinstance(item, float) and item != number:
            raise _Blocked("invalid_class_values", details=f"{item!r} is not an integer class value")
        values.append(number)
    return sorted(set(values))


def _pixel_size_argument(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        size = float(value)
    except (TypeError, ValueError) as exc:
        raise _Blocked("invalid_pixel_size", details="pixel_size_m must be a number of metres") from exc
    if not math.isfinite(size) or size <= 0:
        raise _Blocked("invalid_pixel_size", details="pixel_size_m must be positive", pixel_size_m=size)
    return size


def _read(path_value: Any, role: str, band: Any = 1):
    """Band array, valid mask and grid description of one mask raster."""
    rasterio, np = _deps()
    if not isinstance(path_value, str) or not path_value.strip():
        raise _Blocked("missing_mask_path", role=role)
    path = Path(path_value.strip())
    if not path.is_file():
        raise _Blocked("mask_not_found", role=role, path=str(path))
    try:
        band_index = int(band or 1)
    except (TypeError, ValueError) as exc:
        raise _Blocked("invalid_band", role=role, details="band must be a 1-based integer") from exc
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ungeoreferenced images are expected and handled below
            with rasterio.open(path) as src:
                if band_index < 1 or band_index > src.count:
                    raise _Blocked("invalid_band", role=role, band=band_index, band_count=src.count)
                data = src.read(band_index)
                nodata = src.nodatavals[band_index - 1]
                crs = src.crs
                transform = src.transform
                grid = {"width": src.width, "height": src.height, "band_count": src.count, "band": band_index,
                        "dtype": src.dtypes[band_index - 1], "crs": crs.to_string() if crs else None,
                        "transform": list(transform)[:6], "nodata": nodata}
    except _Blocked:
        raise
    except Exception as exc:
        raise _Blocked("mask_read_failed", role=role, path=str(path), details=str(exc)[:300]) from exc
    valid = np.ones(data.shape, dtype=bool)
    if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
        valid &= data != nodata
    if np.issubdtype(data.dtype, np.floating):
        valid &= np.isfinite(data)
    georeferenced = crs is not None and not transform.is_identity
    grid["georeferenced"] = bool(georeferenced)
    grid["file"] = _file_record(path)
    return data, valid, grid, crs, transform


def _pixel_area(grid: dict, crs, transform, declared: float | None) -> dict[str, Any]:
    """Per-pixel ground area and where it comes from, or a refusal."""
    if crs is None or transform.is_identity:
        if declared is None:
            raise _Blocked("missing_pixel_size", crs=None,
                           details="The raster has no georeferencing; supply pixel_size_m from the data description.")
        return {"pixel_size_m": declared, "pixel_area_m2": declared * declared,
                "area_basis": "declared_pixel_size_unverified",
                "area_basis_note": "Image has no georeferencing; ground area relies on the caller-declared pixel size."}
    from pyproj import CRS

    projected = CRS.from_user_input(crs)
    if not projected.is_projected:
        raise _Blocked("unsupported_area_crs", crs=projected.to_string(), problem="geographic_crs",
                       hint="Degrees cannot be squared into metres; reproject the mask to a projected metre grid first.")
    operation = projected.coordinate_operation
    method = (operation.method_name if operation is not None else "") or ""
    if method.strip().lower() in _MERCATOR_METHODS:
        raise _Blocked("unsupported_area_crs", crs=projected.to_string(), problem="area_distorting_projection",
                       projection_method=method, hint="Mercator pixel areas grow with latitude; reproject to UTM or an equal-area grid.")
    # A name list cannot cover every distorting grid (equidistant cylindrical, out-of-zone UTM, ...).
    from .geo_water import _MAX_AREAL_SCALE_ERROR, _areal_scale_range

    scale = _areal_scale_range(projected, transform, (grid["height"], grid["width"]))
    if scale is None or max(abs(scale[0] - 1), abs(scale[1] - 1)) > _MAX_AREAL_SCALE_ERROR:
        raise _Blocked("unsupported_area_crs", crs=projected.to_string(), problem="area_distorting_projection",
                       projection_method=method or None, areal_scale_range=list(scale) if scale else None,
                       areal_scale_tolerance=_MAX_AREAL_SCALE_ERROR,
                       hint="Grid area differs from ground area here; reproject to the local UTM zone or an equal-area grid.")
    unit = projected.axis_info[0].unit_conversion_factor if projected.axis_info else 1.0
    width_m, height_m = abs(transform.a) * unit, abs(transform.e) * unit
    if transform.b or transform.d:
        width_m = math.hypot(transform.a, transform.d) * unit
        height_m = math.hypot(transform.b, transform.e) * unit
    if declared is not None:
        nominal = math.sqrt(width_m * height_m)
        if abs(declared - nominal) > 0.01 * nominal:
            raise _Blocked("pixel_size_conflict", declared_pixel_size_m=declared, grid_pixel_size_m=[width_m, height_m],
                           crs=projected.to_string(),
                           details="Declared pixel size disagrees with the raster grid by more than 1%.")
    return {"pixel_size_m": math.sqrt(width_m * height_m), "pixel_width_m": width_m, "pixel_height_m": height_m,
            "pixel_area_m2": width_m * height_m, "area_basis": "projected_grid_resolution",
            "projection_method": method or None}


def _same_grid(first: dict, other: dict, label: str) -> None:
    if (first["height"], first["width"]) != (other["height"], other["width"]):
        raise _Blocked("shape_mismatch", item=label, expected_shape=[first["height"], first["width"]],
                       actual_shape=[other["height"], other["width"]],
                       details="Masks must share one pixel grid before they are compared.")
    if first["crs"] != other["crs"]:
        raise _Blocked("crs_mismatch", item=label, expected_crs=first["crs"], actual_crs=other["crs"])
    if first["georeferenced"] and any(abs(a - b) > 1e-6 * max(1.0, abs(a)) for a, b in zip(first["transform"], other["transform"])):
        raise _Blocked("transform_mismatch", item=label, expected_transform=first["transform"],
                       actual_transform=other["transform"])


def _class_stats(data, valid, values: list[int], area: dict, np) -> dict[str, Any]:
    selected = valid & np.isin(data, values)
    valid_pixels = int(np.count_nonzero(valid))
    pixels = int(np.count_nonzero(selected))
    return {"class_pixels": pixels, "class_area_m2": pixels * area["pixel_area_m2"],
            "valid_pixels": valid_pixels, "valid_area_m2": valid_pixels * area["pixel_area_m2"],
            "excluded_nodata_pixels": int(valid.size) - valid_pixels,
            "fraction_of_valid": (pixels / valid_pixels) if valid_pixels else None}


_OK = {"success": True, "execution_blocked": False, "reason_code": None, "failed_checks": []}
_LIMITS = ["Area is pixel count times pixel area on the supplied mask; mask errors carry through unchanged.",
           "A mask class is a label, not a cause: flooded/damaged/landslide attribution needs comparable before/after evidence."]


@_guard
def inspect_handler(arguments: dict) -> dict[str, Any]:
    """Grid, georeferencing and value histogram of one mask raster."""
    _, np = _deps()
    data, valid, grid, crs, transform = _read(arguments.get("mask_path"), "mask", arguments.get("band"))
    values, counts = np.unique(data[valid], return_counts=True)
    histogram = [{"value": (int(v) if float(v).is_integer() else float(v)), "pixels": int(c)}
                 for v, c in list(zip(values, counts))[:_MAX_UNIQUE]]
    area_support: dict[str, Any]
    try:
        area_support = {"area_computable_without_declared_pixel_size": True, **_pixel_area(grid, crs, transform, None)}
    except _Blocked as refusal:
        area_support = {"area_computable_without_declared_pixel_size": False,
                        "area_refusal_reason": refusal.payload["reason_code"],
                        "area_refusal_problem": refusal.payload.get("problem")}
    return {**_OK, **grid, "valid_pixels": int(np.count_nonzero(valid)), "total_pixels": int(valid.size),
            "distinct_value_count": int(values.size), "values_truncated": values.size > _MAX_UNIQUE,
            "value_histogram": histogram, **area_support,
            "note": "More than a few distinct values usually means this is an image, not a categorical mask."}


@_guard
def class_area_handler(arguments: dict) -> dict[str, Any]:
    """Pixel count, area and share of the selected class values in one mask."""
    _, np = _deps()
    values = _class_values(arguments.get("class_values"))
    data, valid, grid, crs, transform = _read(arguments.get("mask_path"), "mask", arguments.get("band"))
    area = _pixel_area(grid, crs, transform, _pixel_size_argument(arguments.get("pixel_size_m")))
    present = sorted(int(v) for v in np.unique(data[valid]) if float(v).is_integer())[:_MAX_UNIQUE]
    return {**_OK, "mask": grid, "class_values": values, **area, **_class_stats(data, valid, values, area, np),
            "values_present_in_mask": present, "class_values_absent_from_mask": [v for v in values if v not in present],
            "denominator": "valid (non-nodata) pixels of this mask", "limitations": _LIMITS}


@_guard
def class_area_series_handler(arguments: dict) -> dict[str, Any]:
    """Class area for an ordered list of masks on one grid, with peak, minimum and change ratios."""
    _, np = _deps()
    values = _class_values(arguments.get("class_values"))
    items = _json_argument(arguments.get("items"), "items")
    if isinstance(items, dict):
        items = [{"label": key, "mask_path": path} for key, path in items.items()]
    if not isinstance(items, list) or not items:
        raise _Blocked("invalid_items", details='expected [{"label": "t1", "mask_path": "/path/a.png"}, ...]')
    if len(items) > _MAX_ITEMS:
        raise _Blocked("too_many_items", item_count=len(items), max_items=_MAX_ITEMS)
    declared = _pixel_size_argument(arguments.get("pixel_size_m"))
    rows, first_grid, area, labels = [], None, None, set()
    for index, item in enumerate(items):
        if isinstance(item, str):
            item = {"label": Path(item).stem, "mask_path": item}
        if not isinstance(item, dict) or not (item.get("mask_path") or item.get("path")):
            raise _Blocked("invalid_items", details=f"item {index} needs label and mask_path")
        label = str(item.get("label") or item.get("timestamp") or item.get("phase") or f"item_{index + 1}")
        if label in labels:
            raise _Blocked("duplicate_item_label", label=label)
        labels.add(label)
        data, valid, grid, crs, transform = _read(item.get("mask_path") or item.get("path"), label, arguments.get("band"))
        if first_grid is None:
            first_grid = grid
            area = _pixel_area(grid, crs, transform, declared)
        else:
            _same_grid(first_grid, grid, label)
        rows.append({"label": label, "order": index + 1, "file": grid["file"], **_class_stats(data, valid, values, area, np)})
    first, last = rows[0], rows[-1]
    for row in rows:
        row["ratio_to_first"] = (row["class_area_m2"] / first["class_area_m2"]) if first["class_area_m2"] else None
        row["change_from_first_m2"] = row["class_area_m2"] - first["class_area_m2"]
    for previous, row in zip([None] + rows[:-1], rows):
        row["change_from_previous_m2"] = None if previous is None else row["class_area_m2"] - previous["class_area_m2"]
    peak = max(rows, key=lambda r: r["class_area_m2"])  # first maximum in the given order
    low = min(rows, key=lambda r: r["class_area_m2"])
    tied = [r["label"] for r in rows if r["class_area_m2"] == peak["class_area_m2"]]

    def brief(row):
        return {"label": row["label"], "class_area_m2": row["class_area_m2"], "class_pixels": row["class_pixels"]}

    return {**_OK, "class_values": values, **area,
            "grid": {k: first_grid[k] for k in ("width", "height", "crs", "georeferenced")},
            "item_count": len(rows), "order_semantics": "items keep the order given by the caller",
            "items": rows, "first": brief(first), "last": brief(last), "max": brief(peak), "min": brief(low),
            "labels_tied_at_max": tied,
            "max_to_first_ratio": (peak["class_area_m2"] / first["class_area_m2"]) if first["class_area_m2"] else None,
            "max_to_first_pct": (100.0 * peak["class_area_m2"] / first["class_area_m2"]) if first["class_area_m2"] else None,
            "last_to_first_ratio": (last["class_area_m2"] / first["class_area_m2"]) if first["class_area_m2"] else None,
            "last_minus_first_m2": last["class_area_m2"] - first["class_area_m2"],
            "mean_area_m2": sum(r["class_area_m2"] for r in rows) / len(rows),
            "ratio_semantics": "ratios divide by the first item's area; null when that area is zero (no zero-denominator ratio is invented)",
            "denominator": "valid (non-nodata) pixels of each mask", "limitations": _LIMITS + [
                "A series of independent masks shows label area per item; pixelwise change needs co-registered, comparable observations."]}


@_guard
def components_handler(arguments: dict) -> dict[str, Any]:
    """Connected regions of the selected classes: count, largest regions, area, perimeter and centroid."""
    rasterio, np = _deps()
    from rasterio import features
    from shapely.geometry import shape

    values = _class_values(arguments.get("class_values"))
    data, valid, grid, crs, transform = _read(arguments.get("mask_path"), "mask", arguments.get("band"))
    area = _pixel_area(grid, crs, transform, _pixel_size_argument(arguments.get("pixel_size_m")))
    connectivity = int(arguments.get("connectivity") or 8)
    if connectivity not in (4, 8):
        raise _Blocked("invalid_connectivity", connectivity=connectivity, supported=[4, 8])
    try:
        min_pixels = max(1, int(arguments.get("min_pixels") or 1))
        top_k = max(1, min(int(arguments.get("top_k") or 10), 100))
    except (TypeError, ValueError) as exc:
        raise _Blocked("invalid_argument", details="min_pixels and top_k must be integers") from exc
    selected = (valid & np.isin(data, values)).astype("uint8")
    pixel = area["pixel_size_m"]
    regions = []
    for geometry, _ in features.shapes(selected, mask=selected.astype(bool), connectivity=connectivity):
        polygon = shape(geometry)  # pixel coordinates: x = column, y = row
        pixels = int(round(polygon.area))
        if pixels < min_pixels:
            continue
        centroid = polygon.centroid
        minx, miny, maxx, maxy = polygon.bounds
        region = {"pixels": pixels, "area_m2": pixels * area["pixel_area_m2"],
                  "perimeter_m": float(polygon.length) * pixel,
                  "centroid_col_row": [float(centroid.x), float(centroid.y)],
                  "centroid_offset_m": [float(centroid.x) * pixel, float(centroid.y) * pixel],
                  "bbox_col_row": [int(minx), int(miny), int(maxx), int(maxy)], "holes": len(polygon.interiors)}
        if grid["georeferenced"]:
            x, y = transform * (centroid.x, centroid.y)
            region["centroid_map_xy"] = [float(x), float(y)]
        regions.append((region, polygon))
    regions.sort(key=lambda item: -item[0]["pixels"])
    for rank, (region, _) in enumerate(regions, 1):
        region["rank"] = rank
    output_path = arguments.get("output_path")
    written = None
    if isinstance(output_path, str) and output_path.strip() and regions:
        import geopandas as gpd
        from shapely.affinity import affine_transform

        if grid["georeferenced"]:
            matrix = [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f]
            out_crs = grid["crs"]
        else:  # local metre frame: x east of the left edge, y negative below the top edge
            matrix = [pixel, 0.0, 0.0, -pixel, 0.0, 0.0]
            out_crs = None
        frame = gpd.GeoDataFrame(
            {"rank": [r["rank"] for r, _ in regions], "pixels": [r["pixels"] for r, _ in regions],
             "area_m2": [r["area_m2"] for r, _ in regions], "perimeter_m": [r["perimeter_m"] for r, _ in regions]},
            geometry=[affine_transform(p, matrix) for _, p in regions], crs=out_crs)
        target = Path(output_path.strip())
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_file(target, driver="GeoJSON", engine="pyogrio")
        written = str(target)
    total_pixels = sum(r["pixels"] for r, _ in regions)
    return {**_OK, "mask": grid, "class_values": values, **area, "connectivity": connectivity, "min_pixels": min_pixels,
            "component_count": len(regions), "components_total_pixels": total_pixels,
            "components_total_area_m2": total_pixels * area["pixel_area_m2"],
            "largest": regions[0][0] if regions else None,
            "components": [r for r, _ in regions[:top_k]], "returned_count": min(len(regions), top_k),
            "truncated": len(regions) > top_k, "components_order": "largest first",
            "perimeter_semantics": "length of the pixel-edge outline including holes, in metres",
            "output_path": written,
            "output_coordinates": None if not written else ("raster CRS" if grid["georeferenced"]
                                                            else "local metres from the image top-left corner, no CRS"),
            "limitations": _LIMITS + ["A connected region of mask pixels is not necessarily one real-world object."]}


@_guard
def overlap_handler(arguments: dict) -> dict[str, Any]:
    """Overlap between the selected classes of two masks on one grid, in pixels, area and regions."""
    _, np = _deps()
    from scipy import ndimage

    values_a = _class_values(arguments.get("class_values_a"))
    values_b = _class_values(arguments.get("class_values_b"))
    data_a, valid_a, grid_a, crs, transform = _read(arguments.get("mask_a_path"), "mask_a", arguments.get("band_a"))
    data_b, valid_b, grid_b, _, _ = _read(arguments.get("mask_b_path"), "mask_b", arguments.get("band_b"))
    _same_grid(grid_a, grid_b, "mask_b")
    area = _pixel_area(grid_a, crs, transform, _pixel_size_argument(arguments.get("pixel_size_m")))
    valid = valid_a & valid_b
    a = valid & np.isin(data_a, values_a)
    b = valid & np.isin(data_b, values_b)
    both = a & b
    pa, pb, pboth = int(np.count_nonzero(a)), int(np.count_nonzero(b)), int(np.count_nonzero(both))
    connectivity = int(arguments.get("connectivity") or 8)
    if connectivity not in (4, 8):
        raise _Blocked("invalid_connectivity", connectivity=connectivity, supported=[4, 8])
    structure = np.ones((3, 3), dtype=bool) if connectivity == 8 else None
    labels_a, count_a = ndimage.label(a, structure=structure)
    touched = np.unique(labels_a[both])
    touched = touched[touched > 0]
    sizes = ndimage.sum_labels(a, labels_a, index=touched) if touched.size else np.asarray([])
    _, overlap_regions = ndimage.label(both, structure=structure)
    pixel_area = area["pixel_area_m2"]
    union = pa + pb - pboth
    valid_pixels = int(np.count_nonzero(valid))
    return {**_OK, "mask_a": grid_a, "mask_b": grid_b, "class_values_a": values_a, "class_values_b": values_b, **area,
            "connectivity": connectivity, "common_valid_pixels": valid_pixels,
            "common_valid_area_m2": valid_pixels * pixel_area,
            "a_pixels": pa, "a_area_m2": pa * pixel_area, "b_pixels": pb, "b_area_m2": pb * pixel_area,
            "overlap_pixels": pboth, "overlap_area_m2": pboth * pixel_area,
            "a_only_area_m2": (pa - pboth) * pixel_area, "b_only_area_m2": (pb - pboth) * pixel_area,
            "union_area_m2": union * pixel_area,
            "overlap_share_of_a": (pboth / pa) if pa else None, "overlap_share_of_b": (pboth / pb) if pb else None,
            "iou": (pboth / union) if union else None,
            "a_region_count": int(count_a), "a_regions_touching_b": int(touched.size),
            "a_regions_touching_b_full_area_m2": float(sizes.sum()) * pixel_area if touched.size else 0.0,
            "overlap_region_count": int(overlap_regions),
            "count_semantics": "a_regions_touching_b counts connected regions of mask A that share at least one pixel with mask B",
            "denominator": "pixels valid in both masks", "limitations": _LIMITS}


def setup(registrar) -> None:
    registrar.toolkit(
        name="geo_mask",
        description="General statistics for categorical mask rasters (model outputs or reference labels, GeoTIFF or plain PNG): value inspection, class area, area over an ordered series of masks, connected regions with perimeter, and overlap of two masks. Area requires a projected metre grid or an explicitly declared pixel size.",
        version="0.1.0",
    )

    def tool(slug, name, description, properties, required, handler):
        registrar.tool(ToolSpec(slug=slug, name=name, description=description,
                                parameters={"type": "object", "properties": properties, "required": required},
                                requires_connection=False), handler)

    path = {"type": "string", "description": "Absolute path of a categorical mask raster (GeoTIFF or PNG)."}
    classes = {"description": "Mask values that count as the class of interest, as a JSON array of integers, e.g. [255] or [2, 3].",
               "oneOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "string"}]}
    pixel = {"type": "number", "description": "Ground pixel size in metres. Required for images without georeferencing (e.g. PNG tiles); take it from the data description. Leave blank for rasters on a projected metre grid."}
    band = {"type": "integer", "description": "1-based band to read (default 1)."}
    connectivity = {"type": "integer", "default": 8, "description": "Pixel connectivity for regions: 8 (default, diagonal neighbours join) or 4."}
    tool("geo_mask.inspect", "Inspect Mask Raster",
         "Report size, bands, CRS/georeferencing, nodata and the distinct pixel values with counts of a mask raster. Run first to learn which values a mask contains and whether area needs a declared pixel size.",
         {"mask_path": path, "band": band}, ["mask_path"], inspect_handler)
    tool("geo_mask.class_area", "Mask Class Area",
         "Count pixels of the selected class values in one mask and convert to area in m2, with the share of valid pixels. Refuses geographic/Mercator grids and ungeoreferenced images without a declared pixel size.",
         {"mask_path": path, "class_values": classes, "pixel_size_m": pixel, "band": band},
         ["mask_path", "class_values"], class_area_handler)
    tool("geo_mask.class_area_series", "Mask Class Area Over A Series",
         "Class area for an ordered list of masks on the same pixel grid (e.g. phases t1..tn): per-item area, the maximum and minimum item, change from the previous and first item, and ratios to the first item. Use for peak phase, trend and peak-to-initial ratio questions.",
         {"items": {"description": 'Ordered JSON array of masks: [{"label": "p2", "mask_path": "/abs/a.png"}, {"label": "p3", "mask_path": "/abs/b.png"}].',
                    "oneOf": [{"type": "array", "items": {"type": "object"}}, {"type": "string"}]},
          "class_values": classes, "pixel_size_m": pixel, "band": band}, ["items", "class_values"], class_area_series_handler)
    tool("geo_mask.components", "Mask Connected Regions",
         "Vectorize the selected classes of a mask into connected regions: number of regions, and for the largest ones area (m2), outline perimeter (m), centroid and bounding box. Optionally writes the region polygons.",
         {"mask_path": path, "class_values": classes, "pixel_size_m": pixel, "connectivity": connectivity,
          "min_pixels": {"type": "integer", "description": "Ignore regions smaller than this many pixels (default 1)."},
          "top_k": {"type": "integer", "description": "How many of the largest regions to list (default 10, max 100)."},
          "band": band,
          "output_path": {"type": "string", "description": "Optional output GeoJSON file with the region polygons."}},
         ["mask_path", "class_values"], components_handler)
    tool("geo_mask.overlap", "Overlap Of Two Masks",
         "Overlap of class A in one mask with class B in another mask on the same pixel grid: overlap area (m2), each class's own area, shares, IoU, and how many connected regions of A touch B. Use for questions such as how much of one mapped class lies inside another.",
         {"mask_a_path": path, "class_values_a": classes, "mask_b_path": path, "class_values_b": classes,
          "pixel_size_m": pixel, "connectivity": connectivity, "band_a": band, "band_b": band},
         ["mask_a_path", "class_values_a", "mask_b_path", "class_values_b"], overlap_handler)
