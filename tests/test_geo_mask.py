"""geo_mask contract tests.

All rasters are SYNTHETIC arrays written by the test (tiny 0/255 or class-code grids). They are
not satellite images, model outputs or benchmark masks.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from terrabox.toolkits.geo_mask import (
    class_area_handler,
    class_area_series_handler,
    components_handler,
    inspect_handler,
    overlap_handler,
)


class GeoMaskTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _png(self, name: str, values) -> str:
        path = self.root / name
        Image.fromarray(np.asarray(values, dtype=np.uint8)).save(path)
        return str(path)

    def _tif(self, name: str, values, *, crs="EPSG:32611", transform=None, nodata=None) -> str:
        data = np.asarray(values, dtype=np.uint8)
        path = self.root / name
        profile = {"driver": "GTiff", "height": data.shape[0], "width": data.shape[1], "count": 1, "dtype": "uint8",
                   "crs": crs, "transform": transform or from_origin(500000, 4100000, 10, 10), "nodata": nodata}
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
        return str(path)

    # --- area basis gate -----------------------------------------------------------
    def test_png_needs_declared_pixel_size(self):
        mask = self._png("mask.png", [[0, 255, 255], [0, 0, 255]])
        refused = class_area_handler({"mask_path": mask, "class_values": [255]}, None, None)
        self.assertTrue(refused["execution_blocked"])
        self.assertEqual(refused["reason_code"], "missing_pixel_size")
        result = class_area_handler({"mask_path": mask, "class_values": "[255]", "pixel_size_m": 0.5}, None, None)
        self.assertTrue(result["success"])
        self.assertEqual(result["class_pixels"], 3)
        self.assertAlmostEqual(result["class_area_m2"], 0.75)
        self.assertEqual(result["valid_pixels"], 6)
        self.assertAlmostEqual(result["fraction_of_valid"], 0.5)
        self.assertEqual(result["area_basis"], "declared_pixel_size_unverified")
        self.assertEqual(len(result["mask"]["file"]["sha256"]), 64)

    def test_projected_grid_uses_its_own_resolution_and_checks_declared_size(self):
        mask = self._tif("utm.tif", [[1, 1], [0, 2]])
        result = class_area_handler({"mask_path": mask, "class_values": [1, 2]}, None, None)
        self.assertEqual(result["area_basis"], "projected_grid_resolution")
        self.assertAlmostEqual(result["class_area_m2"], 300.0)
        agree = class_area_handler({"mask_path": mask, "class_values": [1], "pixel_size_m": 10}, None, None)
        self.assertAlmostEqual(agree["class_area_m2"], 200.0)
        conflict = class_area_handler({"mask_path": mask, "class_values": [1], "pixel_size_m": 0.8}, None, None)
        self.assertEqual(conflict["reason_code"], "pixel_size_conflict")

    def test_geographic_and_mercator_grids_are_refused_even_with_declared_size(self):
        degrees = self._tif("deg.tif", [[1, 0], [0, 0]], crs="EPSG:4326", transform=from_origin(-95.4, 29.7, 0.0001, 0.0001))
        mercator = self._tif("merc.tif", [[1, 0], [0, 0]], crs="EPSG:3857", transform=from_origin(-10620000, 3470000, 3, 3))
        for path, problem in ((degrees, "geographic_crs"), (mercator, "area_distorting_projection")):
            for declared in (None, 3.0):
                result = class_area_handler({"mask_path": path, "class_values": [1], "pixel_size_m": declared}, None, None)
                self.assertTrue(result["execution_blocked"])
                self.assertEqual(result["reason_code"], "unsupported_area_crs")
                self.assertEqual(result["problem"], problem)

    def test_other_distorting_projected_grids_are_refused(self):
        # Synthetic grid: World Equidistant Cylindrical near 41N is in metres and not a Mercator variant,
        # yet its grid area is about a third larger than ground area.
        cylindrical = self._tif("eqc.tif", [[1, 0], [0, 0]], crs="EPSG:4087", transform=from_origin(2450000, 4565000, 10, 10))
        result = class_area_handler({"mask_path": cylindrical, "class_values": [1]}, None, None)
        self.assertTrue(result["execution_blocked"])
        self.assertEqual(result["reason_code"], "unsupported_area_crs")
        self.assertEqual(result["problem"], "area_distorting_projection")
        self.assertGreater(result["areal_scale_range"][0], 1.2)

    def test_nodata_is_excluded_from_the_denominator(self):
        mask = self._tif("nodata.tif", [[1, 255], [0, 255]], nodata=255)
        result = class_area_handler({"mask_path": mask, "class_values": [1]}, None, None)
        self.assertEqual(result["valid_pixels"], 2)
        self.assertEqual(result["excluded_nodata_pixels"], 2)
        self.assertAlmostEqual(result["fraction_of_valid"], 0.5)

    def test_class_values_are_required_and_absent_values_reported(self):
        mask = self._png("mask.png", [[0, 255]])
        missing = class_area_handler({"mask_path": mask, "pixel_size_m": 1}, None, None)
        self.assertEqual(missing["reason_code"], "missing_class_values")
        absent = class_area_handler({"mask_path": mask, "class_values": [1], "pixel_size_m": 1}, None, None)
        self.assertTrue(absent["success"])
        self.assertEqual(absent["class_pixels"], 0)
        self.assertEqual(absent["class_values_absent_from_mask"], [1])
        self.assertEqual(absent["values_present_in_mask"], [0, 255])
        not_found = class_area_handler({"mask_path": str(self.root / "none.png"), "class_values": [1]}, None, None)
        self.assertEqual(not_found["reason_code"], "mask_not_found")

    def test_inspect_lists_values_and_area_support(self):
        mask = self._png("mask.png", [[0, 255, 255], [0, 0, 7]])
        result = inspect_handler({"mask_path": mask}, None, None)
        self.assertFalse(result["georeferenced"])
        self.assertFalse(result["area_computable_without_declared_pixel_size"])
        self.assertEqual(result["area_refusal_reason"], "missing_pixel_size")
        self.assertEqual({h["value"]: h["pixels"] for h in result["value_histogram"]}, {0: 3, 7: 1, 255: 2})

    # --- series ---------------------------------------------------------------------
    def test_series_peak_ratio_and_order(self):
        items = [{"label": "p2", "mask_path": self._png("p2.png", [[255, 0], [0, 0]])},
                 {"label": "p3", "mask_path": self._png("p3.png", [[255, 255], [255, 0]])},
                 {"label": "p4", "mask_path": self._png("p4.png", [[255, 255], [0, 0]])}]
        result = class_area_series_handler({"items": items, "class_values": [255], "pixel_size_m": 2}, None, None)
        self.assertTrue(result["success"])
        self.assertEqual([i["class_area_m2"] for i in result["items"]], [4.0, 12.0, 8.0])
        self.assertEqual(result["max"]["label"], "p3")
        self.assertEqual(result["min"]["label"], "p2")
        self.assertAlmostEqual(result["max_to_first_ratio"], 3.0)
        self.assertAlmostEqual(result["max_to_first_pct"], 300.0)
        self.assertAlmostEqual(result["last_minus_first_m2"], 4.0)
        self.assertEqual(result["items"][2]["change_from_previous_m2"], -4.0)
        as_text = class_area_series_handler({"items": __import__("json").dumps(items), "class_values": "255",
                                             "pixel_size_m": "2"}, None, None)
        self.assertEqual(as_text["max"]["label"], "p3")

    def test_series_zero_first_item_gives_no_invented_ratio(self):
        items = [{"label": "t1", "mask_path": self._png("t1.png", [[0, 0]])},
                 {"label": "t2", "mask_path": self._png("t2.png", [[255, 0]])}]
        result = class_area_series_handler({"items": items, "class_values": [255], "pixel_size_m": 1}, None, None)
        self.assertIsNone(result["max_to_first_ratio"])
        self.assertIsNone(result["items"][1]["ratio_to_first"])
        self.assertEqual(result["max"]["label"], "t2")

    def test_series_blocks_mismatched_grids_and_duplicate_labels(self):
        a = self._png("a.png", [[0, 255]])
        b = self._png("b.png", [[0, 255], [0, 0]])
        mismatch = class_area_series_handler({"items": [{"label": "a", "mask_path": a}, {"label": "b", "mask_path": b}],
                                              "class_values": [255], "pixel_size_m": 1}, None, None)
        self.assertEqual(mismatch["reason_code"], "shape_mismatch")
        duplicate = class_area_series_handler({"items": [{"label": "a", "mask_path": a}, {"label": "a", "mask_path": a}],
                                               "class_values": [255], "pixel_size_m": 1}, None, None)
        self.assertEqual(duplicate["reason_code"], "duplicate_item_label")

    # --- regions and overlap ---------------------------------------------------------
    def test_components_area_perimeter_and_connectivity(self):
        mask = self._png("blobs.png", [[255, 255, 255, 0, 0],
                                       [255, 255, 255, 0, 0],
                                       [0, 0, 0, 255, 0],
                                       [0, 0, 0, 0, 0],
                                       [255, 0, 0, 0, 0]])
        out = self.root / "regions.geojson"
        eight = components_handler({"mask_path": mask, "class_values": [255], "pixel_size_m": 2,
                                    "output_path": str(out)}, None, None)
        self.assertEqual(eight["component_count"], 2)  # the diagonal pixel joins the 2x3 block
        self.assertEqual(eight["largest"]["pixels"], 7)
        self.assertTrue(out.is_file())
        four = components_handler({"mask_path": mask, "class_values": [255], "pixel_size_m": 2, "connectivity": 4}, None, None)
        self.assertEqual(four["component_count"], 3)
        self.assertEqual(four["largest"]["pixels"], 6)
        self.assertAlmostEqual(four["largest"]["area_m2"], 24.0)
        self.assertAlmostEqual(four["largest"]["perimeter_m"], 20.0)  # 2x3 pixel rectangle, 2 m pixels
        self.assertEqual(four["largest"]["centroid_col_row"], [1.5, 1.0])
        filtered = components_handler({"mask_path": mask, "class_values": [255], "pixel_size_m": 2, "connectivity": 4,
                                       "min_pixels": 2}, None, None)
        self.assertEqual(filtered["component_count"], 1)

    def test_overlap_area_and_touching_regions(self):
        a = self._png("a.png", [[1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
        b = self._png("b.png", [[0, 255, 255, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        result = overlap_handler({"mask_a_path": a, "class_values_a": [1], "mask_b_path": b, "class_values_b": [255],
                                  "pixel_size_m": 10}, None, None)
        self.assertEqual(result["overlap_pixels"], 1)
        self.assertAlmostEqual(result["overlap_area_m2"], 100.0)
        self.assertAlmostEqual(result["a_area_m2"], 400.0)
        self.assertEqual(result["a_region_count"], 2)
        self.assertEqual(result["a_regions_touching_b"], 1)
        self.assertAlmostEqual(result["a_regions_touching_b_full_area_m2"], 200.0)
        self.assertAlmostEqual(result["iou"], 1 / 5)
        other = self._png("small.png", [[0, 255]])
        mismatch = overlap_handler({"mask_a_path": a, "class_values_a": [1], "mask_b_path": other, "class_values_b": [255],
                                    "pixel_size_m": 10}, None, None)
        self.assertEqual(mismatch["reason_code"], "shape_mismatch")


if __name__ == "__main__":
    unittest.main()
