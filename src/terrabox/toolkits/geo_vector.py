"""General local vector analysis: inspect, filter, measure, nearest, buffer, overlay, dissolve, Voronoi.

Every tool reads caller-supplied vector files (Shapefile, GeoJSON, GeoPackage) and never
downloads data. Distances, lengths and areas are computed only in a projected metric CRS:
the caller's ``metric_crs`` if it passes the gate, the source CRS if it is already a
non-Mercator projected metre grid, otherwise the UTM zone estimated from the data extent.
Degrees are never squared and Web-Mercator coordinates are never used as ground distances.
Counts are always complete; ``limit`` only shortens the returned record list.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
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
_OPS = ("eq", "ne", "ieq", "in", "contains", "icontains", "startswith", "gt", "ge", "lt", "le", "isnull", "notnull")
_OP_ALIASES = {"==": "eq", "=": "eq", "!=": "ne", "<>": "ne", ">": "gt", ">=": "ge", "<": "lt", "<=": "le",
               "like": "icontains", "not_null": "notnull", "is_null": "isnull", "equals": "eq"}
_HASH_LIMIT_BYTES = 256 * 1024 * 1024
_DEFAULT_LIMIT = 10
_MAX_LIMIT = 200


def _geo():
    try:
        import geopandas as gpd
        import numpy as np
        import pandas as pd
        import shapely
    except ImportError as exc:
        raise ImportError("geo_vector requires the Terrabox geo dependencies: pip install -e '.[geo]'") from exc
    return gpd, np, pd, shapely


def _blocked(reason: str, **extra: Any) -> dict[str, Any]:
    return {"success": False, "execution_blocked": True, "reason_code": reason, "failed_checks": [reason], **extra}


class _Blocked(Exception):
    """Carries a structured refusal out of nested helpers."""

    def __init__(self, reason: str, **extra: Any):
        super().__init__(reason)
        self.payload = _blocked(reason, **extra)


def _clean(value: Any) -> Any:
    """JSON-safe scalar/array conversion; NaN and NaT become None."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _clean(item())
        except (ValueError, TypeError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _clean(tolist())
    try:
        if value != value:  # NaN / NaT
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _source_record(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    record: dict[str, Any] = {"path": str(path), "bytes": size}
    if size <= _HASH_LIMIT_BYTES:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        record["sha256"] = digest.hexdigest()
    else:
        record["sha256"] = None
        record["sha256_skipped"] = "file larger than hash limit"
    return record


def _load(path_value: Any, role: str, layer: Any = None):
    gpd, _, _, _ = _geo()
    if not isinstance(path_value, str) or not path_value.strip():
        raise _Blocked("missing_vector_path", role=role)
    path = Path(path_value.strip())
    if not path.is_file():
        raise _Blocked("vector_not_found", role=role, path=str(path))
    try:
        frame = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    except Exception as exc:  # unreadable or not a vector dataset
        raise _Blocked("vector_read_failed", role=role, path=str(path), details=str(exc)[:300]) from exc
    frame = frame.reset_index(drop=True)
    frame["__feature_index"] = range(len(frame))
    return frame, _source_record(path)


def _parse_where(value: Any) -> list[dict[str, Any]]:
    """Accept a condition array, one condition object, a JSON string, or 'field op value [AND ...]' text."""
    if value is None or value == "" or value == []:
        return []
    raw = value
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            raw = _parse_where_text(text)
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise _Blocked("invalid_where", details="expected an array of {field, op, value} conditions")
    conditions = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or "field" not in item:
            raise _Blocked("invalid_where", details=f"condition {index} needs a field", supported_ops=list(_OPS))
        op = str(item.get("op", "eq")).strip().lower()
        op = _OP_ALIASES.get(op, op)
        if op not in _OPS:
            raise _Blocked("invalid_where", details=f"condition {index} has unsupported op {op!r}", supported_ops=list(_OPS))
        if op not in ("isnull", "notnull") and "value" not in item:
            raise _Blocked("invalid_where", details=f"condition {index} needs a value", supported_ops=list(_OPS))
        conditions.append({"field": str(item["field"]), "op": op, "value": item.get("value")})
    return conditions


_TEXT_CONDITION = re.compile(
    r"^\s*([A-Za-z_][\w.:]*)\s*(==|!=|<>|>=|<=|=|>|<|\s+(?:ieq|in|contains|icontains|startswith|like|eq|ne)\s+)\s*(.+?)\s*$",
    re.IGNORECASE,
)


def _parse_where_text(text: str) -> list[dict[str, Any]]:
    conditions = []
    for part in re.split(r"\s+AND\s+", text, flags=re.IGNORECASE):
        match = _TEXT_CONDITION.match(part)
        if not match:
            raise _Blocked("invalid_where", details=f"cannot parse condition {part!r}",
                           example='[{"field": "fclass", "op": "eq", "value": "park"}]')
        field, op, rest = match.group(1), match.group(2).strip().lower(), match.group(3).strip()
        if op == "in":
            items = [piece.strip().strip("'\"") for piece in rest.strip("()[]").split(",") if piece.strip()]
            parsed: Any = items
        else:
            parsed = rest.strip("'\"")
            if rest == parsed:  # unquoted: allow numbers
                try:
                    parsed = float(rest) if any(ch in rest for ch in ".eE") else int(rest)
                except ValueError:
                    parsed = rest
        conditions.append({"field": field, "op": op, "value": parsed})
    return conditions


def _apply_where(frame, conditions: list[dict[str, Any]], role: str):
    _, np, pd, _ = _geo()
    attribute_fields = [c for c in frame.columns if c not in ("geometry", "__feature_index")]
    mask = np.ones(len(frame), dtype=bool)
    for condition in conditions:
        field, op, value = condition["field"], condition["op"], condition["value"]
        if field not in attribute_fields:
            lowered = {name.lower(): name for name in attribute_fields}
            if field.lower() in lowered:
                field = lowered[field.lower()]
            else:
                raise _Blocked("unknown_field", role=role, field=field, available_fields=attribute_fields)
        series = frame[field]
        if op == "isnull":
            current = series.isna()
        elif op == "notnull":
            current = series.notna()
        elif op in ("gt", "ge", "lt", "le"):
            numeric = pd.to_numeric(series, errors="coerce")
            try:
                bound = float(value)
            except (TypeError, ValueError) as exc:
                raise _Blocked("invalid_where", role=role, details=f"{op} needs a numeric value for {field}") from exc
            current = {"gt": numeric > bound, "ge": numeric >= bound, "lt": numeric < bound, "le": numeric <= bound}[op]
            current = current.fillna(False)
        else:
            text = series.astype("string")
            if op == "in":
                values = value if isinstance(value, (list, tuple)) else [value]
                wanted = {str(v) for v in values}
                current = text.isin(wanted)
                numeric_wanted = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
                if numeric_wanted:
                    current = current | pd.to_numeric(series, errors="coerce").isin(numeric_wanted)
            elif op in ("eq", "ne"):
                current = text == str(value)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    current = current | (pd.to_numeric(series, errors="coerce") == float(value))
                if op == "ne":
                    current = ~current.fillna(False)
            elif op == "ieq":
                current = text.str.lower() == str(value).lower()
            elif op == "contains":
                current = text.str.contains(str(value), regex=False)
            elif op == "icontains":
                current = text.str.lower().str.contains(str(value).lower(), regex=False)
            else:  # startswith
                current = text.str.startswith(str(value))
            current = current.fillna(False)
        mask &= np.asarray(current, dtype=bool)
    return frame[mask]


def _value_examples(frame, conditions: list[dict[str, Any]], cap: int = 25) -> dict[str, Any]:
    """Distinct values of the filtered fields; lets a caller correct a filter that matched nothing."""
    examples: dict[str, Any] = {}
    for condition in conditions:
        field = condition["field"]
        if field in frame.columns and field not in examples:
            counts = frame[field].astype("string").value_counts(dropna=True)
            examples[field] = {"distinct_values": int(counts.size),
                               "most_common": {str(k): int(v) for k, v in counts.head(cap).items()}}
    return examples


def _crs_problem(crs) -> str | None:
    if crs is None:
        return "missing_crs"
    if not crs.is_projected:
        return "geographic_crs"
    operation = crs.coordinate_operation
    method = (operation.method_name if operation is not None else "") or ""
    if method.strip().lower() in _MERCATOR_METHODS:
        return "area_distorting_projection"
    for axis in crs.axis_info:
        if (axis.unit_name or "").lower() not in ("metre", "meter"):
            return "non_metre_units"
    return None


def _scale_problem(crs, frames: list) -> dict[str, Any] | None:
    """Why a projected metre CRS still misstates lengths or areas where the data lies, if it does."""
    from .geo_water import _MAX_AREAL_SCALE_ERROR, _scale_deviation

    points: list[tuple[float, float]] = []
    for frame in frames:
        if frame is None or len(frame) == 0 or frame.crs is None:
            continue
        west, south, east, north = (float(v) for v in frame.to_crs(4326).total_bounds)
        if not all(math.isfinite(v) for v in (west, south, east, north)):
            continue
        points += [(west, south), (west, north), (east, south), (east, north), ((west + east) / 2, (south + north) / 2)]
    if not points:
        return None
    scale = _scale_deviation(crs, points)
    if scale is None:
        return {"problem": "projection_scale_unknown"}
    worst = max(abs(value - 1) for value in scale.values())
    if worst > _MAX_AREAL_SCALE_ERROR:
        return {"problem": "area_distorting_projection", **scale, "scale_tolerance": _MAX_AREAL_SCALE_ERROR}
    return None


def _metric_crs(frames: list, requested: Any) -> tuple[Any, dict[str, Any]]:
    """The projected metre CRS used for every distance, length and area in one call."""
    from pyproj import CRS

    if requested not in (None, ""):
        try:
            crs = CRS.from_user_input(requested)
        except Exception as exc:
            raise _Blocked("invalid_metric_crs", metric_crs=str(requested), details=str(exc)[:200]) from exc
        problem = _crs_problem(crs)
        if problem:
            raise _Blocked("unsupported_metric_crs", metric_crs=crs.to_string(), problem=problem,
                           hint="Use a projected metre CRS such as the local UTM zone; leave blank to auto-select UTM.")
        distorted = _scale_problem(crs, frames)
        if distorted:
            raise _Blocked("unsupported_metric_crs", metric_crs=crs.to_string(), **distorted,
                           hint="This CRS distorts lengths or areas where the data lies; leave blank to auto-select UTM.")
        return crs, {"metric_crs": crs.to_string(), "metric_crs_selection": "caller_supplied"}
    for frame in frames:
        if frame is None or len(frame) == 0:
            continue
        if frame.crs is None:
            raise _Blocked("missing_crs", details="Input has no CRS; distances and areas cannot be computed.")
        source = CRS.from_user_input(frame.crs)
        problem = _crs_problem(source)
        if problem is None:
            problem = (_scale_problem(source, frames) or {}).get("problem")
        if problem is None:
            return source, {"metric_crs": source.to_string(), "metric_crs_selection": "source_crs_projected_metre"}
        utm = frame.estimate_utm_crs()
        return CRS.from_user_input(utm), {"metric_crs": CRS.from_user_input(utm).to_string(),
                                          "metric_crs_selection": "auto_utm_from_data_extent",
                                          "source_crs_problem": problem}
    return None, {"metric_crs": None, "metric_crs_selection": "not_needed_no_features"}


def _to_metric(frame, crs):
    if frame is None or crs is None or len(frame) == 0:
        return frame
    if frame.crs is None:
        raise _Blocked("missing_crs", details="Input has no CRS; distances and areas cannot be computed.")
    return frame.to_crs(crs)


def _limit(arguments: dict) -> int:
    try:
        limit = int(arguments.get("limit", _DEFAULT_LIMIT) or _DEFAULT_LIMIT)
    except (TypeError, ValueError):
        limit = _DEFAULT_LIMIT
    return max(0, min(limit, _MAX_LIMIT))


def _fields_argument(value: Any) -> list[str] | None:
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [piece.strip() for piece in value.split(",") if piece.strip()]
    return [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]


def _attributes(row, fields: list[str] | None, columns: list[str]) -> dict[str, Any]:
    names = [f for f in (fields or columns[:12]) if f in columns]
    return {name: _clean(row[name]) for name in names}


def _records(frame_metric, fields: list[str] | None, limit: int, extra: dict[str, list] | None = None) -> list[dict[str, Any]]:
    """Attribute records with metric measures; geometry itself is never dumped."""
    columns = [c for c in frame_metric.columns if c not in ("geometry", "__feature_index")]
    rows = []
    head = frame_metric.head(limit)
    lonlat = head.geometry.centroid.to_crs(4326) if len(head) and frame_metric.crs is not None else None
    for position, (_, row) in enumerate(head.iterrows()):
        geometry = row.geometry
        record = {"feature_index": int(row["__feature_index"]), "attributes": _attributes(row, fields, columns),
                  "geometry_type": geometry.geom_type if geometry is not None else None}
        if geometry is not None and not geometry.is_empty:
            if geometry.geom_type in ("Polygon", "MultiPolygon"):
                record["area_m2"] = float(geometry.area)
                record["perimeter_m"] = float(geometry.length)
            elif geometry.geom_type in ("LineString", "MultiLineString"):
                record["length_m"] = float(geometry.length)
            if lonlat is not None:
                point = lonlat.iloc[position]
                record["centroid_lon"], record["centroid_lat"] = float(point.x), float(point.y)
        for key, values in (extra or {}).items():
            record[key] = _clean(values[position])
        rows.append(record)
    return rows


def _write(frame, output_path: Any, source_crs=None) -> str | None:
    if not isinstance(output_path, str) or not output_path.strip():
        return None
    path = Path(output_path.strip())
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.drop(columns=[c for c in ("__feature_index",) if c in frame.columns])
    if source_crs is not None and out.crs is not None and len(out):
        out = out.to_crs(source_crs)
    driver = "GPKG" if path.suffix.lower() == ".gpkg" else "GeoJSON"
    out.to_file(path, driver=driver, engine="pyogrio")
    return str(path)


def _geometry_totals(frame_metric) -> dict[str, Any]:
    kinds = frame_metric.geometry.geom_type.value_counts().to_dict() if len(frame_metric) else {}
    polygons = frame_metric[frame_metric.geometry.geom_type.isin(["Polygon", "MultiPolygon"])] if len(frame_metric) else frame_metric
    lines = frame_metric[frame_metric.geometry.geom_type.isin(["LineString", "MultiLineString"])] if len(frame_metric) else frame_metric
    return {
        "geometry_types": {str(k): int(v) for k, v in kinds.items()},
        "total_area_m2": float(polygons.geometry.area.sum()) if len(polygons) else 0.0,
        "total_perimeter_m": float(polygons.geometry.length.sum()) if len(polygons) else 0.0,
        "total_length_m": float(lines.geometry.length.sum()) if len(lines) else 0.0,
    }


def _attribute_sums(frame, value: Any) -> dict[str, Any]:
    """Sum numeric attributes over the selected features (e.g. a per-feature area estimate)."""
    _, _, pd, _ = _geo()
    sums: dict[str, Any] = {}
    for name in _fields_argument(value) or []:
        if name not in frame.columns or name in ("geometry", "__feature_index"):
            raise _Blocked("unknown_field", role="sum_fields", field=name,
                           available_fields=[c for c in frame.columns if c not in ("geometry", "__feature_index")])
        numeric = pd.to_numeric(frame[name], errors="coerce")
        sums[name] = {"sum": float(numeric.sum()), "features_with_value": int(numeric.notna().sum())}
    return sums


def _guard(handler):
    def wrapped(arguments: dict, context: Any = None, account: Any = None) -> dict[str, Any]:
        try:
            return handler(arguments or {})
        except _Blocked as refusal:
            return refusal.payload
    wrapped.__name__ = handler.__name__
    wrapped.__doc__ = handler.__doc__
    return wrapped


_METHOD_NOTE = "planar geometry in the reported metric_crs; no degrees squared, no Web-Mercator distances"


@_guard
def inspect_handler(arguments: dict) -> dict[str, Any]:
    """Schema, CRS, extent and attribute value summary of one vector file."""
    frame, source = _load(arguments.get("vector_path"), "vector", arguments.get("layer"))
    columns = [c for c in frame.columns if c not in ("geometry", "__feature_index")]
    fields = []
    for name in columns:
        series = frame[name]
        counts = series.astype("string").value_counts(dropna=True)
        entry = {"name": name, "dtype": str(series.dtype), "non_null": int(series.notna().sum()),
                 "distinct_values": int(counts.size)}
        if counts.size > 8 and counts.size == entry["non_null"]:
            entry["all_values_distinct"] = True  # identifier or free-text field: a value table would only be noise
            entry["examples"] = [str(k)[:60] for k in series.dropna().astype("string").head(5)]
        elif counts.size <= 25:
            entry["value_counts"] = {str(k)[:60]: int(v) for k, v in counts.items()}
        else:
            entry["most_common"] = {str(k)[:60]: int(v) for k, v in counts.head(25).items()}
            entry["most_common_truncated"] = True
        fields.append(entry)
    crs = frame.crs
    problem = None
    if crs is not None:
        from pyproj import CRS
        problem = _crs_problem(CRS.from_user_input(crs))
    bounds_wgs84 = None
    if len(frame) and crs is not None:
        bounds_wgs84 = [float(v) for v in frame.to_crs(4326).total_bounds]
    return {
        "success": True, "execution_blocked": False, "reason_code": None, "failed_checks": [],
        "source_file": source, "feature_count": int(len(frame)),
        "crs": crs.to_string() if crs is not None else None,
        "crs_usable_for_metric_measurement": crs is not None and problem is None,
        "crs_problem": problem if crs is not None else "missing_crs",
        "geometry_types": {str(k): int(v) for k, v in frame.geometry.geom_type.value_counts().items()} if len(frame) else {},
        "bounds_source_crs": [float(v) for v in frame.total_bounds] if len(frame) else None,
        "bounds_wgs84": bounds_wgs84,
        "fields": fields,
        "limitations": ["Attribute values are whatever the file's producer recorded; a tag is not field verification."],
    }


@_guard
def select_handler(arguments: dict) -> dict[str, Any]:
    """Attribute filter with a complete count and an optional GeoJSON/GPKG output."""
    frame, source = _load(arguments.get("vector_path"), "vector", arguments.get("layer"))
    conditions = _parse_where(arguments.get("where"))
    selected = _apply_where(frame, conditions, "vector")
    limit = _limit(arguments)
    crs, crs_info = _metric_crs([selected], arguments.get("metric_crs"))
    metric = _to_metric(selected, crs)
    result = {
        "success": True, "execution_blocked": False, "reason_code": None, "failed_checks": [],
        "source_file": source, "where": conditions, "input_feature_count": int(len(frame)),
        "count": int(len(selected)), "returned_count": int(min(len(selected), limit)),
        "truncated": len(selected) > limit, **crs_info, **_geometry_totals(metric),
        "attribute_sums": _attribute_sums(selected, arguments.get("sum_fields")),
        "records": _records(metric, _fields_argument(arguments.get("fields")), limit),
        "output_path": _write(selected, arguments.get("output_path")) if len(selected) else None,
        "measurement_method": _METHOD_NOTE,
    }
    if not len(selected) and conditions:
        result["zero_match_field_values"] = _value_examples(frame, conditions)
    return result


@_guard
def measure_handler(arguments: dict) -> dict[str, Any]:
    """Count, area and length of the selected features, optionally grouped by one attribute."""
    frame, source = _load(arguments.get("vector_path"), "vector", arguments.get("layer"))
    conditions = _parse_where(arguments.get("where"))
    selected = _apply_where(frame, conditions, "vector")
    crs, crs_info = _metric_crs([selected], arguments.get("metric_crs"))
    metric = _to_metric(selected, crs)
    limit = _limit(arguments)
    result = {
        "success": True, "execution_blocked": False, "reason_code": None, "failed_checks": [],
        "source_file": source, "where": conditions, "input_feature_count": int(len(frame)),
        "count": int(len(selected)), **crs_info, **_geometry_totals(metric),
        "attribute_sums": _attribute_sums(selected, arguments.get("sum_fields")),
        "measurement_method": _METHOD_NOTE,
    }
    group_by = arguments.get("group_by")
    if isinstance(group_by, str) and group_by.strip():
        field = group_by.strip()
        if field not in frame.columns:
            raise _Blocked("unknown_field", role="group_by", field=field,
                           available_fields=[c for c in frame.columns if c not in ("geometry", "__feature_index")])
        groups = []
        if len(metric):
            for key, part in metric.groupby(metric[field].astype("string").fillna("<null>")):
                groups.append({"value": str(key), "count": int(len(part)), **{k: v for k, v in _geometry_totals(part).items()
                                                                              if k != "geometry_types"}})
        groups.sort(key=lambda g: (-g["count"], g["value"]))
        result["group_by"] = field
        result["group_count"] = len(groups)
        result["groups"] = groups[:_MAX_LIMIT]
        result["groups_truncated"] = len(groups) > _MAX_LIMIT
    else:
        ordered = metric
        if len(metric):
            size = metric.geometry.area.where(metric.geometry.area > 0, metric.geometry.length)
            ordered = metric.assign(__size=size).sort_values("__size", ascending=False).drop(columns="__size")
        result["records_order"] = "largest area (or length) first"
        result["records"] = _records(ordered, _fields_argument(arguments.get("fields")), limit)
        result["returned_count"] = int(min(len(selected), limit))
        result["truncated"] = len(selected) > limit
    if not len(selected) and conditions:
        result["zero_match_field_values"] = _value_examples(frame, conditions)
    return result


def _two_layers(arguments: dict, a: str, b: str):
    frame_a, source_a = _load(arguments.get(f"{a}_path"), a, arguments.get(f"{a}_layer"))
    cond_a = _parse_where(arguments.get(f"{a}_where"))
    selected_a = _apply_where(frame_a, cond_a, a)
    same_file = arguments.get(f"{b}_path") in (None, "") or arguments.get(f"{b}_path") == arguments.get(f"{a}_path")
    if same_file:
        frame_b, source_b = frame_a, source_a
    else:
        frame_b, source_b = _load(arguments.get(f"{b}_path"), b, arguments.get(f"{b}_layer"))
    cond_b = _parse_where(arguments.get(f"{b}_where"))
    selected_b = _apply_where(frame_b, cond_b, b)
    crs, crs_info = _metric_crs([selected_a, selected_b], arguments.get("metric_crs"))
    info = {
        f"{a}_file": source_a, f"{b}_file": source_b, f"{a}_where": cond_a, f"{b}_where": cond_b,
        f"{a}_count": int(len(selected_a)), f"{b}_count": int(len(selected_b)), **crs_info,
    }
    if not len(selected_a) and cond_a:
        info[f"{a}_zero_match_field_values"] = _value_examples(frame_a, cond_a)
    if not len(selected_b) and cond_b:
        info[f"{b}_zero_match_field_values"] = _value_examples(frame_b, cond_b)
    return selected_a, selected_b, _to_metric(selected_a, crs), _to_metric(selected_b, crs), info, same_file


@_guard
def nearest_handler(arguments: dict) -> dict[str, Any]:
    """Nearest target features for each source feature by minimum geometry-to-geometry distance."""
    _, np, _, shapely = _geo()
    _, _, source, target, info, same_file = _two_layers(arguments, "source", "target")
    try:
        k = max(1, min(int(arguments.get("k", 1) or 1), 20))
    except (TypeError, ValueError):
        k = 1
    max_sources = 25
    base = {"success": True, "execution_blocked": False, "reason_code": None, "failed_checks": [], **info, "k": k,
            "distance_semantics": "minimum distance between the two geometries (0 when they touch or overlap); "
                                  "centroid_distance_m is reported separately; neither is a road or travel distance",
            "measurement_method": _METHOD_NOTE}
    if not len(source) or not len(target):
        return {**base, "pairs": [], "pair_count": 0,
                "note": "No source or no target feature matched; a nearest neighbour is undefined, not zero distance."}
    if len(source) > max_sources:
        return _blocked("too_many_source_features", source_count=int(len(source)), max_sources=max_sources,
                        hint="Narrow source_where to the features of interest.", **info)
    fields = _fields_argument(arguments.get("fields"))
    source_columns = [c for c in source.columns if c not in ("geometry", "__feature_index")]
    target_columns = [c for c in target.columns if c not in ("geometry", "__feature_index")]
    target_geoms = np.asarray(target.geometry.values)
    target_index = target["__feature_index"].to_numpy()
    # One flat record per (source, neighbour): callers that summarise nested results keep every value.
    pairs = []
    for _, row in source.iterrows():
        distances = shapely.distance(row.geometry, target_geoms)
        if same_file:
            distances = np.where(target_index == row["__feature_index"], np.inf, distances)
        order = [int(i) for i in np.argsort(distances, kind="stable")[:k] if np.isfinite(distances[i])]
        ties = int(np.count_nonzero(np.isclose(distances, distances[order[0]], atol=1e-6))) if order else 0
        for rank, position in enumerate(order, 1):
            other = target.iloc[position]
            geometry = other.geometry
            pair = {"source_feature_index": int(row["__feature_index"]),
                    "source_attributes": _attributes(row, fields, source_columns),
                    "rank": rank,
                    "target_feature_index": int(other["__feature_index"]),
                    "target_attributes": _attributes(other, fields, target_columns),
                    "distance_m": float(distances[position]),
                    "centroid_distance_m": float(row.geometry.centroid.distance(geometry.centroid)),
                    "target_geometry_type": geometry.geom_type,
                    "targets_tied_at_minimum_distance": ties}
            if geometry.geom_type in ("Polygon", "MultiPolygon"):
                pair["target_area_m2"] = float(geometry.area)
                pair["target_perimeter_m"] = float(geometry.length)
            elif geometry.geom_type in ("LineString", "MultiLineString"):
                pair["target_length_m"] = float(geometry.length)
            pairs.append(pair)
    return {**base, "pairs": pairs, "pair_count": len(pairs),
            "pairs_order": "by source feature, nearest target first",
            "limitations": ["A polygon's enclosed area is not usable open space; attributes are unverified tags."]}


@_guard
def buffer_handler(arguments: dict) -> dict[str, Any]:
    """Metric buffer around the selected features, optionally dissolved into one coverage."""
    gpd, _, _, _ = _geo()
    frame, source = _load(arguments.get("vector_path"), "vector", arguments.get("layer"))
    conditions = _parse_where(arguments.get("where"))
    selected = _apply_where(frame, conditions, "vector")
    try:
        distance = float(arguments.get("distance_m"))
    except (TypeError, ValueError) as exc:
        raise _Blocked("invalid_distance", details="distance_m must be a number of metres") from exc
    if not math.isfinite(distance) or distance <= 0:
        raise _Blocked("invalid_distance", details="distance_m must be positive", distance_m=distance)
    crs, crs_info = _metric_crs([selected], arguments.get("metric_crs"))
    base = {"success": True, "execution_blocked": False, "reason_code": None, "failed_checks": [],
            "source_file": source, "where": conditions, "input_feature_count": int(len(frame)),
            "count": int(len(selected)), "distance_m": distance, **crs_info, "measurement_method": _METHOD_NOTE}
    if not len(selected):
        extra = {"zero_match_field_values": _value_examples(frame, conditions)} if conditions else {}
        return {**base, "buffer_feature_count": 0, "buffer_area_m2": 0.0, "output_path": None, **extra}
    metric = _to_metric(selected, crs)
    dissolve = arguments.get("dissolve", True) not in (False, "false", "False", 0, "0")
    buffered = metric.copy()
    buffered["geometry"] = metric.geometry.buffer(distance)
    if dissolve:
        buffered = gpd.GeoDataFrame({"buffer_distance_m": [distance], "source_feature_count": [len(metric)]},
                                    geometry=[buffered.geometry.union_all()], crs=metric.crs)
    return {**base, "dissolved": dissolve, "buffer_feature_count": int(len(buffered)),
            "buffer_area_m2": float(buffered.geometry.area.sum()),
            "output_path": _write(buffered, arguments.get("output_path"), frame.crs),
            "output_crs": frame.crs.to_string() if frame.crs is not None else None}


@_guard
def within_distance_handler(arguments: dict) -> dict[str, Any]:
    """Target features within a metric distance of any reference feature (buffer-and-intersect as one step)."""
    _, np, _, shapely = _geo()
    reference_src, target_src, reference, target, info, _ = _two_layers(arguments, "reference", "target")
    try:
        distance = float(arguments.get("distance_m"))
    except (TypeError, ValueError) as exc:
        raise _Blocked("invalid_distance", details="distance_m must be a number of metres") from exc
    if not math.isfinite(distance) or distance < 0:
        raise _Blocked("invalid_distance", details="distance_m must be zero or positive", distance_m=distance)
    mode = str(arguments.get("mode") or "intersects").lower()
    if mode not in ("intersects", "centroid", "within"):
        raise _Blocked("invalid_mode", mode=mode, supported_modes=["intersects", "centroid", "within"])
    limit = _limit(arguments)
    base = {"success": True, "execution_blocked": False, "reason_code": None, "failed_checks": [], **info,
            "distance_m": distance, "mode": mode,
            "mode_semantics": {"intersects": "any part of the target lies within the distance",
                               "centroid": "the target centroid lies within the distance",
                               "within": "the whole target lies within the distance"}[mode],
            "measurement_method": _METHOD_NOTE}
    if not len(reference) or not len(target):
        return {**base, "count": 0, "target_total_area_m2": 0.0, "records": [], "returned_count": 0, "truncated": False,
                "share_of_target_count": None, "output_path": None,
                "note": "No reference or no target feature matched the filters."}
    zone = reference.geometry.buffer(distance).union_all() if distance > 0 else reference.geometry.union_all()
    shapely.prepare(zone)
    geoms = np.asarray(target.geometry.values)
    if mode == "intersects":
        hits = shapely.intersects(zone, geoms)
    elif mode == "centroid":
        hits = shapely.intersects(zone, shapely.centroid(geoms))
    else:
        hits = shapely.contains(zone, geoms)
    selected = target[hits]
    nearest = shapely.distance(reference.geometry.union_all(), np.asarray(selected.geometry.values)) if len(selected) else []
    order = np.argsort(nearest, kind="stable")[:limit] if len(selected) else []
    ordered = selected.iloc[list(order)] if len(selected) else selected
    totals = _geometry_totals(selected)
    all_totals = _geometry_totals(target)
    output = None
    if len(selected) and arguments.get("output_path"):
        output = _write(target_src[hits], arguments.get("output_path"))
    return {**base, "count": int(len(selected)),
            "share_of_target_count": len(selected) / len(target),
            "target_total_area_m2": totals["total_area_m2"], "target_total_length_m": totals["total_length_m"],
            "all_targets_total_area_m2": all_totals["total_area_m2"],
            "share_of_target_area": (totals["total_area_m2"] / all_totals["total_area_m2"]) if all_totals["total_area_m2"] else None,
            "area_semantics": "full area of each matching target feature, not only the part inside the distance zone",
            "attribute_sums": _attribute_sums(selected, arguments.get("sum_fields")),
            "zone_area_m2": float(zone.area),
            "records_order": "closest to the reference first",
            "records": _records(ordered, _fields_argument(arguments.get("fields")), limit,
                                {"distance_to_reference_m": [float(nearest[i]) for i in order]}),
            "returned_count": int(min(len(selected), limit)), "truncated": len(selected) > limit,
            "output_path": output}


@_guard
def overlay_handler(arguments: dict) -> dict[str, Any]:
    """Intersection / difference / union of two selections with metric area and length of the result."""
    gpd, np, _, shapely = _geo()
    left_src, _, left, right, info, _ = _two_layers(arguments, "left", "right")
    how = str(arguments.get("how") or "intersection").lower()
    if how == "clip":
        how = "intersection"
    if how not in ("intersection", "difference", "union"):
        raise _Blocked("invalid_how", how=how, supported=["intersection", "difference", "union"])
    base = {"success": True, "execution_blocked": False, "reason_code": None, "failed_checks": [], **info, "how": how,
            "measurement_method": _METHOD_NOTE}
    left_totals = _geometry_totals(left) if len(left) else {"total_area_m2": 0.0, "total_length_m": 0.0}
    base["left_total_area_m2"] = left_totals["total_area_m2"]
    base["left_total_length_m"] = left_totals["total_length_m"]
    if not len(left) or (not len(right) and how == "intersection"):
        return {**base, "left_features_intersecting_right": 0, "result_feature_count": 0, "result_part_count": 0,
                "result_area_m2": 0.0, "result_length_m": 0.0, "share_of_left_area": None, "output_path": None,
                "note": "One side matched no feature; an empty overlay is reported, not an error."}
    right_union = right.geometry.union_all() if len(right) else None
    left_geoms = np.asarray(left.geometry.values)
    touching = shapely.intersects(right_union, left_geoms) if right_union is not None else np.zeros(len(left), dtype=bool)
    if how == "union":
        merged = shapely.union_all(list(left_geoms) + ([right_union] if right_union is not None else []))
        result = gpd.GeoDataFrame({"overlay": ["union"]}, geometry=[merged], crs=left.crs)
    else:
        if right_union is None:
            pieces = left_geoms
        elif how == "intersection":
            pieces = shapely.intersection(left_geoms, right_union)
        else:
            pieces = shapely.difference(left_geoms, right_union)
        keep = ~(shapely.is_empty(pieces) | shapely.is_missing(pieces))
        result = left[keep].copy()
        result["geometry"] = pieces[keep]
    totals = _geometry_totals(result) if len(result) else {"total_area_m2": 0.0, "total_length_m": 0.0, "geometry_types": {}}
    parts = int(result.geometry.explode(index_parts=False).shape[0]) if len(result) else 0
    return {**base, "left_features_intersecting_right": int(np.count_nonzero(touching)),
            "result_feature_count": int(len(result)), "result_part_count": parts,
            "result_geometry_types": totals.get("geometry_types", {}),
            "result_area_m2": totals["total_area_m2"], "result_length_m": totals["total_length_m"],
            "share_of_left_area": (totals["total_area_m2"] / left_totals["total_area_m2"])
            if how != "union" and left_totals["total_area_m2"] else None,
            "output_path": _write(result, arguments.get("output_path"), left_src.crs) if len(result) else None}


@_guard
def dissolve_handler(arguments: dict) -> dict[str, Any]:
    """Union the selected features (all together or per attribute value) and count the resulting parts."""
    gpd, _, _, _ = _geo()
    frame, source = _load(arguments.get("vector_path"), "vector", arguments.get("layer"))
    conditions = _parse_where(arguments.get("where"))
    selected = _apply_where(frame, conditions, "vector")
    crs, crs_info = _metric_crs([selected], arguments.get("metric_crs"))
    base = {"success": True, "execution_blocked": False, "reason_code": None, "failed_checks": [],
            "source_file": source, "where": conditions, "input_feature_count": int(len(frame)),
            "count": int(len(selected)), **crs_info, "measurement_method": _METHOD_NOTE}
    if not len(selected):
        extra = {"zero_match_field_values": _value_examples(frame, conditions)} if conditions else {}
        return {**base, "dissolved_feature_count": 0, "part_count": 0, "total_area_m2": 0.0, "output_path": None, **extra}
    metric = _to_metric(selected, crs)
    by = arguments.get("by")
    if isinstance(by, str) and by.strip():
        field = by.strip()
        if field not in metric.columns:
            raise _Blocked("unknown_field", role="by", field=field,
                           available_fields=[c for c in frame.columns if c not in ("geometry", "__feature_index")])
        dissolved = metric[[field, "geometry"]].dissolve(by=field, dropna=False).reset_index()
    else:
        field = None
        dissolved = gpd.GeoDataFrame({"source_feature_count": [len(metric)]},
                                     geometry=[metric.geometry.union_all()], crs=metric.crs)
    parts = dissolved.geometry.explode(index_parts=False)
    totals = _geometry_totals(dissolved)
    return {**base, "by": field, "dissolved_feature_count": int(len(dissolved)),
            "part_count": int(len(parts)),
            "part_count_semantics": "separate single-part geometries after the union; touching or overlapping inputs merge into one part",
            **{k: totals[k] for k in ("total_area_m2", "total_perimeter_m", "total_length_m")},
            "output_path": _write(dissolved, arguments.get("output_path"), frame.crs)}


@_guard
def voronoi_handler(arguments: dict) -> dict[str, Any]:
    """Voronoi (Thiessen) cells of the selected point features, clipped to an explicit envelope."""
    gpd, np, _, shapely = _geo()
    frame, source = _load(arguments.get("vector_path"), "points", arguments.get("layer"))
    conditions = _parse_where(arguments.get("where"))
    selected = _apply_where(frame, conditions, "points")
    crs, crs_info = _metric_crs([selected], arguments.get("metric_crs"))
    envelope_mode = str(arguments.get("envelope") or "bbox").lower()
    base = {"success": True, "execution_blocked": False, "reason_code": None, "failed_checks": [],
            "source_file": source, "where": conditions, "input_feature_count": int(len(frame)),
            "point_count": int(len(selected)), **crs_info, "envelope": envelope_mode,
            "measurement_method": _METHOD_NOTE}
    if not len(selected):
        extra = {"zero_match_field_values": _value_examples(frame, conditions)} if conditions else {}
        return {**base, "distinct_point_count": 0, "cell_count": 0, "output_path": None, **extra}
    metric = _to_metric(selected, crs)
    non_points = int((~metric.geometry.geom_type.isin(["Point"])).sum())
    points = metric.geometry.centroid if non_points else metric.geometry
    coords = np.round(np.column_stack([points.x.to_numpy(), points.y.to_numpy()]), 3)
    _, first = np.unique(coords, axis=0, return_index=True)
    keep = np.sort(first)
    distinct = metric.iloc[keep]
    distinct_points = points.iloc[keep]
    base.update({"distinct_point_count": int(len(distinct)),
                 "duplicate_points_dropped": int(len(metric) - len(distinct)),
                 "non_point_features_reduced_to_centroid": non_points})
    if len(distinct) < 2:
        return _blocked("too_few_points", **{k: v for k, v in base.items() if k not in ("success", "execution_blocked",
                                                                                           "reason_code", "failed_checks")},
                        details="A Voronoi tessellation needs at least two distinct points.")
    try:
        margin = float(arguments.get("envelope_buffer_m", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise _Blocked("invalid_distance", details="envelope_buffer_m must be a number of metres") from exc
    multipoint = shapely.multipoints(np.asarray(distinct_points.values))
    clip_path = arguments.get("clip_path")
    if isinstance(clip_path, str) and clip_path.strip():
        clip_frame, clip_source = _load(clip_path, "clip", None)
        if clip_frame.crs is None:
            raise _Blocked("missing_crs", role="clip")
        envelope = clip_frame.to_crs(metric.crs).geometry.union_all()
        base["clip_file"] = clip_source
        base["envelope"] = "clip_path"
    elif envelope_mode == "convex_hull":
        envelope = multipoint.convex_hull
    elif envelope_mode == "bbox":
        envelope = shapely.box(*multipoint.bounds)
    else:
        raise _Blocked("invalid_envelope", envelope=envelope_mode, supported=["bbox", "convex_hull", "clip_path"])
    if margin:
        envelope = envelope.buffer(margin)
    if envelope.is_empty or envelope.area <= 0:
        raise _Blocked("degenerate_envelope", details="Envelope has no area; add envelope_buffer_m or a clip_path.")
    extent = envelope.buffer(max(envelope.length, 1.0))
    cells = shapely.get_parts(shapely.voronoi_polygons(multipoint, extend_to=extent))
    tree = shapely.STRtree(cells)
    owner = tree.query(np.asarray(distinct_points.values), predicate="within")
    cell_for_point = {int(p): int(c) for p, c in zip(owner[0], owner[1])}
    rows, geometries = [], []
    for position in range(len(distinct)):
        cell_index = cell_for_point.get(position)
        if cell_index is None:
            continue
        clipped = cells[cell_index].intersection(envelope)
        if clipped.is_empty:
            continue
        rows.append(distinct.iloc[position].drop(labels=["geometry"]))
        geometries.append(clipped)
    result = gpd.GeoDataFrame(rows, geometry=geometries, crs=metric.crs) if rows else gpd.GeoDataFrame(geometry=[], crs=metric.crs)
    areas = result.geometry.area if len(result) else []
    limit = _limit(arguments)
    return {**base, "cell_count": int(len(result)),
            "cells_cover_envelope_area_m2": float(sum(areas)) if len(result) else 0.0,
            "envelope_area_m2": float(envelope.area), "envelope_buffer_m": margin,
            "cell_area_m2_min": float(min(areas)) if len(result) else None,
            "cell_area_m2_max": float(max(areas)) if len(result) else None,
            "records_order": "largest cell first",
            "records": _records(result.assign(__a=list(areas)).sort_values("__a", ascending=False).drop(columns="__a"),
                                _fields_argument(arguments.get("fields")), limit) if len(result) else [],
            "returned_count": int(min(len(result), limit)), "truncated": len(result) > limit,
            "output_path": _write(result, arguments.get("output_path"), frame.crs) if len(result) else None,
            "limitations": ["Cells are straight-line nearest-point regions inside the chosen envelope; "
                            "they are not travel-time or administrative service areas."]}


@_guard
def zone_counts_handler(arguments: dict) -> dict[str, Any]:
    """For each zone polygon, count (and measure) the features of another layer that fall in it."""
    _, np, _, shapely = _geo()
    _, _, zones, features, info, _ = _two_layers(arguments, "zone", "feature")
    predicate = str(arguments.get("predicate") or "intersects").lower()
    if predicate not in ("intersects", "within", "centroid_within"):
        raise _Blocked("invalid_predicate", predicate=predicate, supported=["intersects", "within", "centroid_within"])
    limit = _limit(arguments)
    base = {"success": True, "execution_blocked": False, "reason_code": None, "failed_checks": [], **info,
            "predicate": predicate, "measurement_method": _METHOD_NOTE}
    if not len(zones):
        return {**base, "zones": [], "zones_with_features": 0, "features_in_any_zone": 0,
                "note": "No zone feature matched the filter."}
    non_polygon = int((~zones.geometry.geom_type.isin(["Polygon", "MultiPolygon"])).sum())
    if non_polygon:
        raise _Blocked("zones_not_polygons", non_polygon_zone_count=non_polygon, **info)
    probe = features.geometry.centroid if predicate == "centroid_within" and len(features) else features.geometry
    probe_geoms = np.asarray(probe.values) if len(features) else np.asarray([])
    feature_areas = features.geometry.area.to_numpy() if len(features) else np.asarray([])
    feature_lengths = features.geometry.length.to_numpy() if len(features) else np.asarray([])
    is_line = features.geometry.geom_type.isin(["LineString", "MultiLineString"]).to_numpy() if len(features) else np.asarray([])
    fields = _fields_argument(arguments.get("fields"))
    columns = [c for c in zones.columns if c not in ("geometry", "__feature_index")]
    any_hit = np.zeros(len(features), dtype=bool)
    rows = []
    for _, zone in zones.iterrows():
        if len(features):
            geometry = zone.geometry
            shapely.prepare(geometry)
            hits = shapely.intersects(geometry, probe_geoms) if predicate != "within" else shapely.contains(geometry, probe_geoms)
        else:
            hits = np.zeros(0, dtype=bool)
        any_hit |= hits
        rows.append({"zone_feature_index": int(zone["__feature_index"]), "attributes": _attributes(zone, fields, columns),
                     "zone_area_m2": float(zone.geometry.area), "feature_count": int(np.count_nonzero(hits)),
                     "feature_area_m2": float(feature_areas[hits & ~is_line].sum()) if len(features) else 0.0,
                     "feature_length_m": float(feature_lengths[hits & is_line].sum()) if len(features) else 0.0})
    rows.sort(key=lambda r: (-r["feature_count"], r["zone_feature_index"]))
    return {**base, "zones_with_features": sum(1 for r in rows if r["feature_count"]),
            "features_in_any_zone": int(np.count_nonzero(any_hit)),
            "records_order": "most features first",
            "zones": rows[:limit], "returned_count": min(len(rows), limit), "truncated": len(rows) > limit}


_REGION_LIST_KEYS = ("candidates", "detections", "regions", "annotations", "objects", "items")


@_guard
def pixel_regions_to_vector_handler(arguments: dict) -> dict[str, Any]:
    """Georeference image-pixel regions (bounding boxes) with the grid of the raster they were found in."""
    gpd, np, _, shapely = _geo()
    import rasterio

    regions_value = arguments.get("regions_path")
    if not isinstance(regions_value, str) or not regions_value.strip():
        raise _Blocked("missing_regions_path")
    regions_path = Path(regions_value.strip())
    if not regions_path.is_file():
        raise _Blocked("regions_not_found", path=str(regions_path))
    try:
        payload = json.loads(regions_path.read_text())
    except (OSError, ValueError) as exc:
        raise _Blocked("regions_read_failed", path=str(regions_path), details=str(exc)[:200]) from exc
    rows = payload
    if isinstance(payload, dict):
        rows = next((payload[key] for key in _REGION_LIST_KEYS if isinstance(payload.get(key), list)), None)
    if not isinstance(rows, list):
        raise _Blocked("invalid_regions", details="expected a JSON array of regions, or an object holding one under "
                       + "/".join(_REGION_LIST_KEYS))
    raster_value = arguments.get("raster_path")
    if not isinstance(raster_value, str) or not raster_value.strip():
        raise _Blocked("missing_raster_path")
    raster_path = Path(raster_value.strip())
    if not raster_path.is_file():
        raise _Blocked("raster_not_found", path=str(raster_path))
    try:
        with rasterio.open(raster_path) as src:
            crs, transform, width, height = src.crs, src.transform, src.width, src.height
    except Exception as exc:
        raise _Blocked("raster_read_failed", path=str(raster_path), details=str(exc)[:200]) from exc
    if crs is None or transform.is_identity:
        raise _Blocked("raster_not_georeferenced", path=str(raster_path),
                       details="Pixel regions can only be placed on the ground with a georeferenced raster.")
    box_field = str(arguments.get("bbox_field") or "bbox_xyxy")
    label_field = str(arguments.get("label_field") or "label")
    wanted = set(_fields_argument(arguments.get("labels")) or [])
    geometries, attributes, skipped, label_counts = [], [], 0, {}
    for index, row in enumerate(rows):
        box = row.get(box_field) if isinstance(row, dict) else None
        if not (isinstance(box, (list, tuple)) and len(box) == 4
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box)):
            skipped += 1
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise _Blocked("bbox_outside_raster", region_index=index, bbox=list(box), raster_size=[width, height],
                           details="A region lies outside this raster; the regions belong to a different image grid.")
        label = row.get(label_field)
        label_counts[str(label)] = label_counts.get(str(label), 0) + 1
        if wanted and str(label) not in wanted:
            continue
        geometries.append(shapely.Polygon([transform * (x1, y1), transform * (x2, y1), transform * (x2, y2), transform * (x1, y2)]))
        area_px = row.get("area_px")
        attributes.append({"region_index": index, "label": None if label is None else str(label),
                           "bbox_area_px": (x2 - x1) * (y2 - y1),
                           "area_px": float(area_px) if isinstance(area_px, (int, float)) and not isinstance(area_px, bool) else None})
    base = {"success": True, "execution_blocked": False, "reason_code": None, "failed_checks": [],
            "regions_file": _source_record(regions_path), "raster_file": _source_record(raster_path),
            "raster_crs": crs.to_string(), "raster_size": [width, height], "region_count": len(rows),
            "regions_without_bbox_skipped": skipped, "label_counts_all_regions": label_counts,
            "labels_kept": sorted(wanted) if wanted else "all", "geometry_kind": "bounding_box_of_each_region"}
    if not geometries:
        return {**base, "converted_count": 0, "output_path": None, "metric_crs": None,
                "note": "No region with a usable bounding box matched; nothing was written."}
    frame = gpd.GeoDataFrame(attributes, geometry=geometries, crs=crs)
    frame["__feature_index"] = range(len(frame))
    metric_crs, crs_info = _metric_crs([frame], arguments.get("metric_crs"))
    box_area = frame.to_crs(metric_crs).geometry.area.to_numpy()
    frame["bbox_area_m2"] = box_area
    # Local ground area per pixel comes from the reprojected box, so Mercator or degree grids are never squared.
    pixel_area = np.asarray([a["area_px"] if a["area_px"] is not None else np.nan for a in attributes], dtype="float64")
    frame["region_area_m2"] = pixel_area * box_area / frame["bbox_area_px"].to_numpy()
    with_area = int(np.count_nonzero(np.isfinite(pixel_area)))
    return {**base, **crs_info, "converted_count": int(len(frame)),
            "total_bbox_area_m2": float(box_area.sum()),
            "regions_with_pixel_area": with_area,
            "total_region_area_m2": float(np.nansum(frame["region_area_m2"].to_numpy())) if with_area else None,
            "attributes_written": ["region_index", "label", "bbox_area_px", "area_px", "bbox_area_m2", "region_area_m2"],
            "area_semantics": "bbox_area_m2 is the box (an upper bound of the object); region_area_m2 scales the region's own "
                              "pixel count by the local ground area per pixel and is empty when no pixel count was supplied",
            "output_path": _write(frame, arguments.get("output_path")),
            "limitations": ["Regions are whatever the upstream detector or annotator produced; a box is not a footprint.",
                            "Placement is only as good as the raster's georeferencing and requires that the regions were "
                            "found on exactly this raster grid."]}


_WHERE_DOC = ('Attribute filter as a JSON array of conditions combined with AND, e.g. '
              '[{"field": "fclass", "op": "eq", "value": "park"}]. Ops: eq, ne, ieq (case-insensitive), in (value is an array), '
              'contains, icontains, startswith, gt, ge, lt, le, isnull, notnull. Leave blank for all features. '
              'Run inspect first to see real field names and values.')
_WHERE = {"description": _WHERE_DOC, "oneOf": [{"type": "array", "items": {"type": "object"}}, {"type": "object"}, {"type": "string"}]}
_METRIC = {"type": "string", "description": "Optional projected metre CRS for measurement, e.g. EPSG:32615. Leave blank: a suitable projected source CRS is used, otherwise the UTM zone of the data is selected automatically. Geographic and Mercator CRS are refused."}
_LIMIT = {"type": "integer", "description": "Maximum records returned (default 10, max 200). Counts and totals always cover all matches."}
_FIELDS = {"description": 'Attribute names to return per record as a JSON array, e.g. ["name", "fclass"]. Default: the first 12 fields.',
           "oneOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}]}
_OUT = {"type": "string", "description": "Optional output GeoJSON file for the resulting features; later steps can read it as a vector_path."}
_SUMS = {"description": 'Optional numeric attribute names to sum over the matching features, as a JSON array, e.g. ["region_area_m2"].',
         "oneOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}]}


def setup(registrar) -> None:
    registrar.toolkit(
        name="geo_vector",
        description="General analysis of local vector files (Shapefile, GeoJSON, GeoPackage): inspect, attribute filter, count, area/length, nearest neighbour, buffer, within-distance selection, overlay, dissolve, Voronoi, per-zone counts. Measurements use a projected metre CRS only.",
        version="0.1.0",
    )

    def tool(slug, name, description, properties, required, handler):
        registrar.tool(ToolSpec(slug=slug, name=name, description=description,
                                parameters={"type": "object", "properties": properties, "required": required},
                                requires_connection=False), handler)

    path = {"type": "string", "description": "Absolute path of a local vector file (.shp, .geojson, .gpkg)."}
    tool("geo_vector.inspect", "Inspect Vector File",
         "List feature count, CRS, geometry types, extent and every attribute field with its values or examples. Run this first to learn real field names and category values before filtering.",
         {"vector_path": path}, ["vector_path"], inspect_handler)
    tool("geo_vector.select", "Select Vector Features",
         "Filter features of one vector file by attributes. Returns the complete match count, total area/length in metres, and a limited list of matching records with attributes, area and centroid. A filter that matches nothing returns count 0 plus the real values of the filtered fields.",
         {"vector_path": path, "where": _WHERE, "fields": _FIELDS, "sum_fields": _SUMS, "limit": _LIMIT, "metric_crs": _METRIC,
          "output_path": _OUT}, ["vector_path"], select_handler)
    tool("geo_vector.measure", "Measure Vector Features",
         "Count the selected features and sum polygon area (m2), polygon perimeter (m) and line length (m) in a projected metre CRS; optionally grouped by one attribute. Use for totals such as network length or built-up area.",
         {"vector_path": path, "where": _WHERE, "group_by": {"type": "string", "description": "Optional attribute to group totals by."},
          "fields": _FIELDS, "sum_fields": _SUMS, "limit": _LIMIT, "metric_crs": _METRIC}, ["vector_path"], measure_handler)
    tool("geo_vector.nearest", "Nearest Vector Features",
         "For each selected source feature, find the k nearest target features by minimum geometry-to-geometry distance in metres (0 when they touch or overlap). Returns names/attributes, distance, centroid distance and the area or length of each nearest target. Source and target may be different files or two selections of one file.",
         {"source_path": path, "source_where": _WHERE,
          "target_path": {"type": "string", "description": "Target vector file; leave blank to search within the source file."},
          "target_where": _WHERE, "k": {"type": "integer", "description": "Neighbours per source feature (default 1, max 20)."},
          "fields": _FIELDS, "metric_crs": _METRIC}, ["source_path"], nearest_handler)
    tool("geo_vector.buffer", "Buffer Vector Features",
         "Buffer the selected features by a distance in metres (computed in a projected metre CRS) and write the buffer polygons; dissolved into one coverage by default.",
         {"vector_path": path, "where": _WHERE, "distance_m": {"type": "number", "description": "Buffer distance in metres (> 0)."},
          "dissolve": {"type": "boolean", "description": "Merge all buffers into one coverage (default true)."},
          "metric_crs": _METRIC, "output_path": _OUT}, ["vector_path", "distance_m"], buffer_handler)
    tool("geo_vector.within_distance", "Select Features Within Distance",
         "Count and list target features lying within distance_m metres of any selected reference feature (a buffer-and-intersect in one step). Returns complete count, share of all targets, total area/length of the matching targets, and the closest records.",
         {"reference_path": path, "reference_where": _WHERE,
          "target_path": {"type": "string", "description": "Target vector file; leave blank to use the reference file."},
          "target_where": _WHERE, "distance_m": {"type": "number", "description": "Distance in metres (0 = must touch or overlap)."},
          "mode": {"type": "string", "enum": ["intersects", "centroid", "within"], "default": "intersects",
                   "description": "intersects: any part within the distance; centroid: target centroid within; within: whole target within."},
          "fields": _FIELDS, "sum_fields": _SUMS, "limit": _LIMIT, "metric_crs": _METRIC, "output_path": _OUT},
         ["reference_path", "distance_m"], within_distance_handler)
    tool("geo_vector.pixel_regions_to_vector", "Georeference Pixel Regions",
         "Turn regions given in image pixels (a JSON list with a bounding box per region, e.g. detector candidates) into a georeferenced vector file, using the CRS and grid of the raster the regions were found on. Each region becomes its bounding-box polygon with label, pixel area and ground-area attributes, so the other geo_vector tools can filter, count and measure it. Refuses rasters without georeferencing and regions outside the raster.",
         {"regions_path": {"type": "string", "description": "Absolute path of a JSON file: an array of regions, or an object holding the array under candidates/detections/regions/annotations."},
          "raster_path": {"type": "string", "description": "Absolute path of the georeferenced raster (GeoTIFF) the regions were found on."},
          "labels": {"description": 'Optional labels to keep, as a JSON array, e.g. ["building"]. Default: all regions.',
                     "oneOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}]},
          "bbox_field": {"type": "string", "description": "Name of the per-region box field [x1, y1, x2, y2] in pixels (default bbox_xyxy)."},
          "label_field": {"type": "string", "description": "Name of the per-region label field (default label)."},
          "metric_crs": _METRIC, "output_path": _OUT}, ["regions_path", "raster_path"], pixel_regions_to_vector_handler)
    tool("geo_vector.overlay", "Overlay Vector Layers",
         "Geometric intersection (clip), difference or union of two selections. Reports how many left features intersect the right layer and the area (m2) and length (m) of the result, e.g. length of roads inside a zone or area of parcels inside a hazard polygon.",
         {"left_path": path, "left_where": _WHERE,
          "right_path": {"type": "string", "description": "Right vector file; leave blank to use the left file."},
          "right_where": _WHERE, "how": {"type": "string", "enum": ["intersection", "difference", "union"], "default": "intersection"},
          "metric_crs": _METRIC, "output_path": _OUT}, ["left_path"], overlay_handler)
    tool("geo_vector.dissolve", "Dissolve Vector Features",
         "Union the selected features into one geometry (or one per value of an attribute) and report how many separate parts remain plus total area. Touching or overlapping features merge into one part.",
         {"vector_path": path, "where": _WHERE, "by": {"type": "string", "description": "Optional attribute to dissolve by."},
          "metric_crs": _METRIC, "output_path": _OUT}, ["vector_path"], dissolve_handler)
    tool("geo_vector.voronoi", "Voronoi Cells From Points",
         "Build Voronoi (Thiessen) cells for the selected point features in a projected metre CRS, one cell per distinct point, clipped to an envelope (bounding box, convex hull, or a clip polygon file). Reports point count, duplicates dropped and cell count; the output file can be passed to dissolve or zone_counts.",
         {"vector_path": path, "where": _WHERE,
          "envelope": {"type": "string", "enum": ["bbox", "convex_hull"], "default": "bbox", "description": "Envelope used to bound the outer cells."},
          "envelope_buffer_m": {"type": "number", "description": "Optional margin in metres added around the envelope."},
          "clip_path": {"type": "string", "description": "Optional polygon file to clip cells to instead of the envelope."},
          "fields": _FIELDS, "limit": _LIMIT, "metric_crs": _METRIC, "output_path": _OUT}, ["vector_path"], voronoi_handler)
    tool("geo_vector.zone_counts", "Count Features Per Zone",
         "For each selected zone polygon, count the features of another selection that intersect it, lie within it, or have their centroid in it (point-in-polygon), with their summed area and length.",
         {"zone_path": path, "zone_where": _WHERE,
          "feature_path": {"type": "string", "description": "Vector file with the features to count; leave blank to use the zone file."},
          "feature_where": _WHERE,
          "predicate": {"type": "string", "enum": ["intersects", "within", "centroid_within"], "default": "intersects"},
          "fields": _FIELDS, "limit": _LIMIT, "metric_crs": _METRIC}, ["zone_path"], zone_counts_handler)
