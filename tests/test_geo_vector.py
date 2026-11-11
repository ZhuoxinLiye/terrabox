"""geo_vector contract tests.

Every geometry here is a SYNTHETIC fixture built in the test (squares, points and lines on a
UTM zone 15N grid). Nothing in this file is observed map data or a benchmark answer.
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, Point, box

from terrabox.toolkits.geo_vector import (
    buffer_handler,
    dissolve_handler,
    inspect_handler,
    measure_handler,
    nearest_handler,
    overlay_handler,
    pixel_regions_to_vector_handler,
    select_handler,
    voronoi_handler,
    within_distance_handler,
    zone_counts_handler,
)

UTM = "EPSG:32615"
X0, Y0 = 270000.0, 3290000.0  # arbitrary synthetic origin inside UTM 15N


def square(dx: float, dy: float, size: float = 100.0):
    return box(X0 + dx, Y0 + dy, X0 + dx + size, Y0 + dy + size)


class GeoVectorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        # Synthetic polygons: three "park" squares (two share an edge), one "water" square.
        zones = gpd.GeoDataFrame(
            {"name": ["A", "B", "C", "Pond"], "kind": ["park", "park", "park", "water"]},
            geometry=[square(0, 0), square(100, 0), square(400, 0), square(0, 400)], crs=UTM)
        self.zones_utm = self._save(zones, "zones_utm.geojson")
        self.zones_wgs = self._save(zones.to_crs(4326), "zones_wgs84.geojson")
        self.zones_mercator = self._save(zones.to_crs(3857), "zones_3857.geojson")
        # Synthetic buildings: two share a name; one overlaps park A; one is 50 m east of park C.
        buildings = gpd.GeoDataFrame(
            {"name": ["Hall", "Depot", "Depot", None], "levels": [2, 1, 3, 1]},
            geometry=[square(40, 40, 20), square(550, 0, 20), square(900, 900, 20), square(250, 30, 20)], crs=UTM)
        self.buildings = self._save(buildings.to_crs(4326), "buildings_wgs84.geojson")
        lines = gpd.GeoDataFrame(
            {"kind": ["rail", "rail", "road"]},
            geometry=[LineString([(X0, Y0), (X0 + 300, Y0)]), LineString([(X0, Y0 + 50), (X0, Y0 + 450)]),
                      LineString([(X0, Y0), (X0 + 1000, Y0 + 1000)])], crs=UTM)
        self.lines = self._save(lines.to_crs(4326), "lines_wgs84.geojson")
        points = gpd.GeoDataFrame(
            {"name": ["p1", "p2", "p3", "p4", "p4-duplicate", "shop"],
             "kind": ["clinic", "clinic", "clinic", "clinic", "clinic", "shop"]},
            geometry=[Point(X0, Y0), Point(X0 + 1000, Y0), Point(X0, Y0 + 1000), Point(X0 + 1000, Y0 + 1000),
                      Point(X0 + 1000, Y0 + 1000), Point(X0 + 50, Y0 + 50)], crs=UTM)
        self.points = self._save(points.to_crs(4326), "points_wgs84.geojson")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _save(self, frame, name: str) -> str:
        path = self.root / name
        frame.to_file(path, driver="GeoJSON", engine="pyogrio")
        return str(path)

    # --- inspection and filtering -------------------------------------------------
    def test_inspect_reports_fields_values_and_crs_usability(self):
        result = inspect_handler({"vector_path": self.zones_wgs}, None, None)
        self.assertTrue(result["success"])
        self.assertEqual(result["feature_count"], 4)
        self.assertEqual(result["crs"], "EPSG:4326")
        self.assertFalse(result["crs_usable_for_metric_measurement"])
        self.assertEqual(result["crs_problem"], "geographic_crs")
        kind = next(f for f in result["fields"] if f["name"] == "kind")
        self.assertEqual(kind["value_counts"], {"park": 3, "water": 1})
        self.assertEqual(len(result["source_file"]["sha256"]), 64)

    def test_inspect_keeps_identifier_fields_compact(self):
        frame = gpd.GeoDataFrame({"osm_id": list(range(1000, 1030)), "kind": ["a"] * 30},
                                 geometry=[Point(X0 + i, Y0) for i in range(30)], crs=UTM)
        result = inspect_handler({"vector_path": self._save(frame, "ids.geojson")}, None, None)
        ids = next(f for f in result["fields"] if f["name"] == "osm_id")
        self.assertTrue(ids["all_values_distinct"])
        self.assertEqual(len(ids["examples"]), 5)
        self.assertNotIn("value_counts", ids)

    def test_select_keeps_complete_count_when_limit_truncates(self):
        result = select_handler({"vector_path": self.zones_wgs, "where": [{"field": "kind", "op": "eq", "value": "park"}],
                                 "limit": 1, "fields": ["name"]}, None, None)
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["returned_count"], 1)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["metric_crs"], UTM)
        self.assertEqual(result["metric_crs_selection"], "auto_utm_from_data_extent")
        self.assertAlmostEqual(result["total_area_m2"], 30000.0, delta=30.0)
        self.assertEqual(list(result["records"][0]["attributes"]), ["name"])

    def test_select_zero_matches_is_a_legal_result_with_real_values(self):
        result = select_handler({"vector_path": self.zones_wgs, "where": 'kind = "Park"'}, None, None)
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["zero_match_field_values"]["kind"]["most_common"], {"park": 3, "water": 1})

    def test_select_duplicate_names_are_all_counted(self):
        result = select_handler({"vector_path": self.buildings, "where": "name = Depot"}, None, None)
        self.assertEqual(result["count"], 2)
        self.assertEqual([r["feature_index"] for r in result["records"]], [1, 2])

    def test_where_text_json_string_and_in_operator(self):
        as_text = select_handler({"vector_path": self.buildings, "where": "levels >= 2"}, None, None)
        as_json = select_handler({"vector_path": self.buildings,
                                  "where": '[{"field": "levels", "op": "ge", "value": 2}]'}, None, None)
        as_in = select_handler({"vector_path": self.zones_utm, "where": "kind in (park, water)"}, None, None)
        self.assertEqual(as_text["count"], 2)
        self.assertEqual(as_json["count"], 2)
        self.assertEqual(as_in["count"], 4)

    def test_unknown_field_and_bad_inputs_are_blocked_with_reason_codes(self):
        unknown = select_handler({"vector_path": self.zones_wgs, "where": "fclass = park"}, None, None)
        self.assertTrue(unknown["execution_blocked"])
        self.assertEqual(unknown["reason_code"], "unknown_field")
        self.assertIn("kind", unknown["available_fields"])
        missing = inspect_handler({"vector_path": str(self.root / "absent.shp")}, None, None)
        self.assertEqual(missing["reason_code"], "vector_not_found")
        bad_op = select_handler({"vector_path": self.zones_wgs,
                                 "where": [{"field": "kind", "op": "regex", "value": "p.*"}]}, None, None)
        self.assertEqual(bad_op["reason_code"], "invalid_where")
        # An unsupported negated list test is refused instead of being silently read as "ne".
        not_in = select_handler({"vector_path": self.zones_wgs,
                                 "where": [{"field": "kind", "op": "not in", "value": ["park"]}]}, None, None)
        self.assertEqual(not_in["reason_code"], "invalid_where")

    # --- measurement gate ----------------------------------------------------------
    def test_measure_uses_projected_source_crs_and_groups(self):
        result = measure_handler({"vector_path": self.zones_utm, "group_by": "kind"}, None, None)
        self.assertEqual(result["metric_crs_selection"], "source_crs_projected_metre")
        self.assertAlmostEqual(result["total_area_m2"], 40000.0, places=6)
        self.assertEqual({g["value"]: g["count"] for g in result["groups"]}, {"park": 3, "water": 1})

    def test_measure_never_uses_degrees_or_web_mercator(self):
        for path in (self.zones_wgs, self.zones_mercator):
            result = measure_handler({"vector_path": path}, None, None)
            self.assertEqual(result["metric_crs"], UTM)
            self.assertEqual(result["metric_crs_selection"], "auto_utm_from_data_extent")
            self.assertAlmostEqual(result["total_area_m2"], 40000.0, delta=40.0)
        for bad in ("EPSG:4326", "EPSG:3857"):
            refused = measure_handler({"vector_path": self.zones_wgs, "metric_crs": bad}, None, None)
            self.assertTrue(refused["execution_blocked"])
            self.assertEqual(refused["reason_code"], "unsupported_metric_crs")

    def test_measure_refuses_or_replaces_other_distorting_projected_crs(self):
        # Synthetic zones. Equidistant cylindrical is projected, in metres and not Mercator, but stretches
        # east-west distances away from the equator.
        refused = measure_handler({"vector_path": self.zones_wgs, "metric_crs": "EPSG:4087"}, None, None)
        self.assertTrue(refused["execution_blocked"])
        self.assertEqual(refused["reason_code"], "unsupported_metric_crs")
        self.assertEqual(refused["problem"], "area_distorting_projection")
        import geopandas as gpd

        stored = self._save(gpd.read_file(self.zones_wgs).to_crs(4087), "zones_4087.geojson")
        result = measure_handler({"vector_path": stored}, None, None)
        self.assertEqual(result["metric_crs"], UTM)
        self.assertEqual(result["metric_crs_selection"], "auto_utm_from_data_extent")
        self.assertAlmostEqual(result["total_area_m2"], 40000.0, delta=40.0)

    def test_measure_line_length(self):
        result = measure_handler({"vector_path": self.lines, "where": "kind = rail"}, None, None)
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["total_length_m"], 700.0, delta=0.7)
        self.assertEqual(result["total_area_m2"], 0.0)

    def test_missing_crs_is_blocked(self):
        frame = gpd.GeoDataFrame({"name": ["x"]}, geometry=[square(0, 0)], crs=None)
        path = self._save(frame, "no_crs.geojson")
        # GeoJSON readers assume WGS84, so use a format that keeps "no CRS".
        shp = self.root / "no_crs.shp"
        frame.to_file(shp, engine="pyogrio")
        result = measure_handler({"vector_path": str(shp)}, None, None)
        self.assertEqual(result["reason_code"], "missing_crs")
        self.assertTrue(Path(path).is_file())

    # --- spatial relations ---------------------------------------------------------
    def test_nearest_distance_is_geometry_to_geometry(self):
        result = nearest_handler({"source_path": self.buildings, "source_where": "name = Hall",
                                  "target_path": self.zones_wgs, "target_where": "kind = park",
                                  "fields": ["name"]}, None, None)
        pair = result["pairs"][0]
        self.assertEqual(pair["source_attributes"]["name"], "Hall")
        self.assertEqual(pair["target_attributes"]["name"], "A")
        self.assertEqual(pair["distance_m"], 0.0)  # building lies inside park A
        self.assertGreater(pair["centroid_distance_m"], 0.0)
        self.assertAlmostEqual(pair["target_area_m2"], 10000.0, delta=10.0)
        # Flat records: no value sits deeper than attributes -> field (callers compact deeper nesting away).
        self.assertTrue(all(not isinstance(v, (dict, list)) or k.endswith("_attributes") for k, v in pair.items()))

    def test_nearest_reports_gap_distance_and_k_neighbours(self):
        result = nearest_handler({"source_path": self.buildings, "source_where": [{"field": "levels", "op": "eq", "value": 1},
                                                                                 {"field": "name", "op": "eq", "value": "Depot"}],
                                  "target_path": self.zones_wgs, "k": 2, "fields": ["name"]}, None, None)
        pairs = result["pairs"]
        self.assertEqual([(p["rank"], p["target_attributes"]["name"]) for p in pairs], [(1, "C"), (2, "B")])
        self.assertAlmostEqual(pairs[0]["distance_m"], 50.0, delta=0.1)

    def test_nearest_within_one_file_excludes_the_feature_itself(self):
        result = nearest_handler({"source_path": self.zones_utm, "source_where": "name = C", "fields": ["name"]}, None, None)
        nearest = result["pairs"][0]
        self.assertEqual(nearest["target_attributes"]["name"], "B")
        self.assertAlmostEqual(nearest["distance_m"], 200.0, places=6)

    def test_nearest_with_no_source_is_not_zero_distance(self):
        result = nearest_handler({"source_path": self.buildings, "source_where": "name = Missing",
                                  "target_path": self.zones_wgs}, None, None)
        self.assertTrue(result["success"])
        self.assertEqual(result["pair_count"], 0)
        self.assertIn("name", result["source_zero_match_field_values"])

    def test_buffer_area_and_invalid_distance(self):
        out = self.root / "buffer.geojson"
        result = buffer_handler({"vector_path": self.points, "where": "name = p1", "distance_m": 100,
                                 "output_path": str(out)}, None, None)
        self.assertAlmostEqual(result["buffer_area_m2"], math.pi * 100 ** 2, delta=0.01 * math.pi * 100 ** 2)
        self.assertTrue(out.is_file())
        self.assertEqual(result["output_crs"], "EPSG:4326")
        refused = buffer_handler({"vector_path": self.points, "distance_m": -5}, None, None)
        self.assertEqual(refused["reason_code"], "invalid_distance")

    def test_within_distance_modes(self):
        base = {"reference_path": self.zones_wgs, "reference_where": "name = C", "target_path": self.buildings,
                "fields": ["name"]}
        near = within_distance_handler({**base, "distance_m": 60}, None, None)
        self.assertEqual(near["count"], 1)
        self.assertEqual(near["records"][0]["attributes"]["name"], "Depot")
        self.assertAlmostEqual(near["records"][0]["distance_to_reference_m"], 50.0, delta=0.1)
        self.assertAlmostEqual(near["share_of_target_count"], 0.25)
        whole = within_distance_handler({**base, "distance_m": 60, "mode": "within"}, None, None)
        self.assertEqual(whole["count"], 0)  # the 20 m building spans 50-70 m from the park
        none = within_distance_handler({**base, "distance_m": 10}, None, None)
        self.assertEqual(none["count"], 0)
        self.assertTrue(none["success"])

    def test_overlay_intersection_and_difference(self):
        clip = self._save(gpd.GeoDataFrame({"id": [1]}, geometry=[box(X0 + 50, Y0 - 10, X0 + 150, Y0 + 110)], crs=UTM), "clip.geojson")
        inter = overlay_handler({"left_path": self.zones_utm, "left_where": "kind = park", "right_path": clip}, None, None)
        self.assertEqual(inter["left_features_intersecting_right"], 2)
        self.assertAlmostEqual(inter["result_area_m2"], 10000.0, places=6)
        self.assertAlmostEqual(inter["share_of_left_area"], 1 / 3, places=6)
        diff = overlay_handler({"left_path": self.zones_utm, "left_where": "kind = park", "right_path": clip,
                                "how": "difference"}, None, None)
        self.assertAlmostEqual(diff["result_area_m2"], 20000.0, places=6)
        rail = overlay_handler({"left_path": self.lines, "left_where": "kind = rail", "right_path": clip}, None, None)
        self.assertAlmostEqual(rail["result_length_m"], 100.0, delta=0.2)

    def test_dissolve_merges_touching_features(self):
        out = self.root / "dissolved.geojson"
        result = dissolve_handler({"vector_path": self.zones_utm, "where": "kind = park", "output_path": str(out)}, None, None)
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["dissolved_feature_count"], 1)
        self.assertEqual(result["part_count"], 2)  # A and B share an edge, C is separate
        self.assertAlmostEqual(result["total_area_m2"], 30000.0, places=6)
        by_kind = dissolve_handler({"vector_path": self.zones_utm, "by": "kind"}, None, None)
        self.assertEqual(by_kind["dissolved_feature_count"], 2)

    def test_voronoi_cells_duplicates_and_dissolve_chain(self):
        out = self.root / "cells.geojson"
        result = voronoi_handler({"vector_path": self.points, "where": "kind = clinic", "output_path": str(out)}, None, None)
        self.assertEqual(result["point_count"], 5)
        self.assertEqual(result["distinct_point_count"], 4)
        self.assertEqual(result["duplicate_points_dropped"], 1)
        self.assertEqual(result["cell_count"], 4)
        self.assertAlmostEqual(result["cells_cover_envelope_area_m2"], result["envelope_area_m2"], delta=1.0)
        self.assertAlmostEqual(result["cell_area_m2_max"], 250000.0, delta=300.0)
        merged = dissolve_handler({"vector_path": str(out)}, None, None)
        self.assertEqual(merged["count"], 4)
        self.assertEqual(merged["part_count"], 1)

    def test_voronoi_needs_two_distinct_points(self):
        result = voronoi_handler({"vector_path": self.points, "where": "name startswith p4"}, None, None)
        self.assertTrue(result["execution_blocked"])
        self.assertEqual(result["reason_code"], "too_few_points")

    def test_zone_counts_point_in_polygon(self):
        result = zone_counts_handler({"zone_path": self.zones_wgs, "zone_where": "kind = park",
                                      "feature_path": self.points, "predicate": "centroid_within",
                                      "fields": ["name"]}, None, None)
        counts = {z["attributes"]["name"]: z["feature_count"] for z in result["zones"]}
        self.assertEqual(counts["A"], 2)  # p1 on the corner and the shop inside
        self.assertEqual(counts["C"], 0)
        self.assertEqual(result["features_in_any_zone"], 2)
        lines = zone_counts_handler({"zone_path": self.lines, "feature_path": self.points}, None, None)
        self.assertEqual(lines["reason_code"], "zones_not_polygons")

    # --- pixel regions (e.g. detector candidates) ----------------------------------------
    def _raster(self, name: str, *, crs=UTM, georeferenced=True) -> str:
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        path = self.root / name
        profile = {"driver": "GTiff", "height": 100, "width": 100, "count": 1, "dtype": "uint8"}
        if georeferenced:
            profile.update(crs=crs, transform=from_origin(X0, Y0 + 1000, 10, 10))
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.zeros((100, 100), dtype="uint8"), 1)
        return str(path)

    def _regions(self, rows) -> str:
        import json

        path = self.root / "regions.json"
        path.write_text(json.dumps({"candidates": rows}))
        return str(path)

    def test_pixel_regions_become_georeferenced_boxes_with_ground_area(self):
        # Synthetic candidate boxes on a synthetic 10 m grid; not detector output.
        regions = self._regions([
            {"label": "building", "bbox_xyxy": [0, 0, 4, 2], "area_px": 6},
            {"label": "building", "bbox_xyxy": [50, 50, 52, 52], "area_px": 4},
            {"label": "vehicle", "bbox_xyxy": [10, 10, 11, 11]},
            {"label": "building"},
        ])
        out = self.root / "boxes.geojson"
        result = pixel_regions_to_vector_handler({"regions_path": regions, "raster_path": self._raster("grid.tif"),
                                                  "labels": ["building"], "output_path": str(out)}, None, None)
        self.assertTrue(result["success"])
        self.assertEqual(result["region_count"], 4)
        self.assertEqual(result["regions_without_bbox_skipped"], 1)
        self.assertEqual(result["label_counts_all_regions"], {"building": 2, "vehicle": 1})
        self.assertEqual(result["converted_count"], 2)
        self.assertAlmostEqual(result["total_bbox_area_m2"], (8 + 4) * 100.0, places=3)
        self.assertAlmostEqual(result["total_region_area_m2"], (6 + 4) * 100.0, places=3)
        # The written boxes work with the other tools, including attribute sums.
        near = within_distance_handler({"reference_path": self.zones_utm, "reference_where": "name = A",
                                        "target_path": str(out), "distance_m": 1000,
                                        "sum_fields": ["region_area_m2"]}, None, None)
        self.assertEqual(near["count"], 2)
        self.assertAlmostEqual(near["attribute_sums"]["region_area_m2"]["sum"], 1000.0, places=3)
        unknown = measure_handler({"vector_path": str(out), "sum_fields": ["missing"]}, None, None)
        self.assertEqual(unknown["reason_code"], "unknown_field")

    def test_pixel_regions_refuse_wrong_or_ungeoreferenced_grids(self):
        regions = self._regions([{"label": "building", "bbox_xyxy": [90, 90, 140, 95]}])
        outside = pixel_regions_to_vector_handler({"regions_path": regions, "raster_path": self._raster("grid.tif")}, None, None)
        self.assertEqual(outside["reason_code"], "bbox_outside_raster")
        plain = pixel_regions_to_vector_handler({"regions_path": self._regions([{"label": "b", "bbox_xyxy": [0, 0, 1, 1]}]),
                                                 "raster_path": self._raster("plain.tif", georeferenced=False)}, None, None)
        self.assertEqual(plain["reason_code"], "raster_not_georeferenced")
        missing = pixel_regions_to_vector_handler({"regions_path": str(self.root / "none.json"),
                                                   "raster_path": self._raster("grid2.tif")}, None, None)
        self.assertEqual(missing["reason_code"], "regions_not_found")

    def test_pixel_regions_on_web_mercator_get_true_ground_area(self):
        from pyproj import Transformer

        x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(-95.4, 29.7)
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        path = self.root / "mercator.tif"
        with rasterio.open(path, "w", driver="GTiff", height=10, width=10, count=1, dtype="uint8", crs="EPSG:3857",
                           transform=from_origin(x, y, 100, 100)) as dst:
            dst.write(np.zeros((10, 10), dtype="uint8"), 1)
        result = pixel_regions_to_vector_handler({"regions_path": self._regions([{"label": "b", "bbox_xyxy": [0, 0, 10, 10],
                                                                                 "area_px": 100}]),
                                                  "raster_path": str(path)}, None, None)
        self.assertEqual(result["metric_crs_selection"], "auto_utm_from_data_extent")
        expected = (1000.0 * math.cos(math.radians(29.7))) ** 2  # Mercator metres shrink by cos(latitude) on the ground
        self.assertAlmostEqual(result["total_bbox_area_m2"], expected, delta=0.01 * expected)
        self.assertLess(result["total_region_area_m2"], 0.8 * 1000.0 ** 2)


if __name__ == "__main__":
    unittest.main()
