"""
Sandbox: VLM accuracy test for identify_object_from_views().

Set the constants below, then run:
    python sandbox.py
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import open3d as o3d

# ── User-defined constants ────────────────────────────────────────────────────
CONFIG_FILE = "config.yml"
# INPUT_FILE = "/home/mohammedalsalihi/LidarData/hand_selected/outdoor_for_test.las"  # path to the processed pipeline LAS (must have a Clusters attribute)
INPUT_FILE = "/home/mohammedalsalihi/LidarData/for_mohammed/fh_outdoor_result/3_post_process.las"
ORIGINAL_FILE = "/home/mohammedalsalihi/LidarData/for_mohammed/fh_outdoor.las"  # path to the original full-resolution LAS; leave empty to skip restoration

CAR_CLUSTER_IDS: list[int] = [16, 67, 68, 116, 280, 737, 738, 751]
NON_CAR_CLUSTER_IDS: list[int] = [8, 10, 15, 17, 40, 42, 56, 92, 99, 226, 735, 755]

SCREENSHOTS_DIR = os.path.join(tempfile.gettempdir(), "sandbox_vlm_test")
CHUNK_SIZE = 10_000_000
VLM_VIEWS_COUNT = 8
# ─────────────────────────────────────────────────────────────────────────────

from modules.utils.config import SegmentorConfig
from modules.models.post_processing_tools import merge_chunks
from modules.models.cluster_tools import build_original_pcds_per_cluster
from modules.models.pcd_tools import apply_rgb_colors_to_pcd
from modules.models.rich_media_tools import take_rich_media_screenshots
from modules.models.vlm_tools import identify_object_from_views
from modules.utils import constants


def _extract_cluster_pcd(
    xyz: np.ndarray,
    merged_attrs: dict[str, np.ndarray],
    cluster_id: int,
) -> o3d.geometry.PointCloud | None:
    objects_clusters = np.asarray(merged_attrs.get("Clusters", []), dtype=np.int32)
    if objects_clusters.shape[0] != len(xyz):
        return None
    mask = objects_clusters == cluster_id
    if not np.any(mask):
        return None
    indices = np.where(mask)[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[indices])
    n = len(xyz)
    cluster_attrs = {k: np.asarray(v)[indices] for k, v in merged_attrs.items() if np.asarray(v).shape[0] == n}
    try:
        apply_rgb_colors_to_pcd(pcd, cluster_attrs)
    except Exception:
        pass
    return pcd


def _test_cluster(
    xyz: np.ndarray,
    merged_attrs: dict[str, np.ndarray],
    cluster_id: int,
    is_car: bool,
    original_pcds: dict[int, o3d.geometry.PointCloud],
) -> dict:
    expected_label = "car" if is_car else "non_car"
    print(f"  cluster {cluster_id:>4d}  (expected: {expected_label}) ...", end=" ", flush=True)

    processed_pcd = _extract_cluster_pcd(xyz, merged_attrs, cluster_id)
    if processed_pcd is None or len(processed_pcd.points) == 0:
        print("NOT FOUND in file")
        return {"cluster_id": cluster_id, "expected": expected_label, "ai_label": "N/A", "description": "", "correct": False}

    restored = original_pcds.get(cluster_id)
    screenshot_pcd = restored if (restored is not None and len(restored.points) > 0) else processed_pcd

    cluster_dir = os.path.join(SCREENSHOTS_DIR, str(cluster_id))
    os.makedirs(cluster_dir, exist_ok=True)
    view_paths = take_rich_media_screenshots(screenshot_pcd, cluster_dir, number_of_views=VLM_VIEWS_COUNT)

    record = identify_object_from_views(view_paths, cluster_id=cluster_id, chunk_index=0)

    predicted_car = record["classification"] == constants.PCB_CLASSIFICATION_CAR
    correct = predicted_car == is_car
    print(f"ai_label='{record['ai_label']}'  {'✓' if correct else '✗'}")

    return {
        "cluster_id": cluster_id,
        "expected": expected_label,
        "ai_label": record["ai_label"],
        "description": record.get("description", ""),
        "correct": correct,
    }


def _print_report(results: list[dict]) -> None:
    car_results = [r for r in results if r["expected"] == "car"]
    non_car_results = [r for r in results if r["expected"] == "non_car"]
    total = len(results)
    correct = sum(r["correct"] for r in results)
    car_correct = sum(r["correct"] for r in car_results)
    non_car_correct = sum(r["correct"] for r in non_car_results)

    print("\n" + "=" * 60)
    print("ACCURACY REPORT")
    print("=" * 60)
    if total:
        print(f"  Overall   : {correct}/{total} ({100 * correct / total:.1f}%)")
    if car_results:
        print(f"  Cars      : {car_correct}/{len(car_results)} ({100 * car_correct / len(car_results):.1f}%)")
    if non_car_results:
        print(f"  Non-cars  : {non_car_correct}/{len(non_car_results)} ({100 * non_car_correct / len(non_car_results):.1f}%)")

    print("\nDETAILS")
    print("-" * 60)
    # for r in results:
    #     status = "✓" if r["correct"] else "✗"
    #     print(f"  {status}  cluster {r['cluster_id']:>4d}  expected={r['expected']:<8s}  ai_label='{r['ai_label']}'")
    #     if r["description"]:
    #         print(f"            description: {r['description']}")
    print()


def main() -> None:
    if not INPUT_FILE:
        raise ValueError("Set INPUT_FILE to the path of your processed LAS file.")
    if not CAR_CLUSTER_IDS and not NON_CAR_CLUSTER_IDS:
        raise ValueError("Set at least one cluster ID in CAR_CLUSTER_IDS or NON_CAR_CLUSTER_IDS.")

    SegmentorConfig.initialize(CONFIG_FILE)

    print(f"Loading {INPUT_FILE} ...")
    xyz, merged_attrs = merge_chunks(INPUT_FILE, CHUNK_SIZE)
    print(f"Loaded {len(xyz):,} points.\n")

    all_clusters = [(cid, True) for cid in CAR_CLUSTER_IDS] + [(cid, False) for cid in NON_CAR_CLUSTER_IDS]
    print(f"Testing {len(all_clusters)} clusters ({len(CAR_CLUSTER_IDS)} cars, {len(NON_CAR_CLUSTER_IDS)} non-cars) ...\n")

    all_cluster_ids = [cid for cid, _ in all_clusters]
    objects_clusters = np.asarray(merged_attrs.get("Clusters", []), dtype=np.int32)

    original_pcds: dict[int, o3d.geometry.PointCloud] = {}
    if ORIGINAL_FILE:
        print(f"Restoring original PCDs from {ORIGINAL_FILE} ...")
        original_pcds = build_original_pcds_per_cluster(
            original_path=ORIGINAL_FILE,
            xyz=xyz,
            objects_clusters=objects_clusters,
            cluster_ids=all_cluster_ids,
        )
        print()

    results = [_test_cluster(xyz, merged_attrs, cid, is_car, original_pcds) for cid, is_car in all_clusters]
    _print_report(results)


if __name__ == "__main__":
    main()
