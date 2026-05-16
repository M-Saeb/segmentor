#!/usr/bin/env python3
"""
Result Validation Tool: manually validate clustering results from the Post Process stage.

Reads `3_post_process.las`, loops through every cluster in a GUI, lets the user
mark each one Correct or Incorrect, then writes `validation.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.models.post_processing_tools import merge_chunks  # noqa: E402
from modules.utils import constants  # noqa: E402
from modules.utils.log_utils import get_dual_logger  # noqa: E402

log = get_dual_logger(__name__)

_COLOR_GRAY = np.array([0.6, 0.6, 0.6], dtype=np.float64)
_COLOR_BLUE = np.array([0.0, 0.4, 1.0], dtype=np.float64)

_DEFAULT_CHUNK_SIZE = 500_000


class ResultValidationViewer:

    def __init__(self, input_path: str, output_path: str) -> None:
        self._input_path = input_path
        self._output_path = output_path

        self._xyz: np.ndarray | None = None
        self._attrs: dict[str, np.ndarray] | None = None
        self._cluster_ids: list[int] = []
        self._current_index: int = 0

        # Non-ceiling boolean mask (len == total points)
        self._visible_mask: np.ndarray | None = None

        # Validation counters
        self._total_points: int = 0
        self._total_clusters_count: int = 0
        self._false_points_count: int = 0
        self._false_clusters_count: int = 0

        # GUI widgets
        self._window: gui.Application | None = None
        self._preview_scene: gui.SceneWidget | None = None
        self._cluster_label: gui.Label | None = None
        self._camera_initialized: bool = False

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self, chunk_size: int) -> None:
        log.info("Loading %s ...", self._input_path)
        xyz, attrs = merge_chunks(self._input_path, chunk_size=chunk_size)
        self._xyz = xyz
        self._attrs = attrs
        self._total_points = len(xyz)

        classification = np.asarray(
            attrs.get("classification", np.zeros(len(xyz), dtype=np.int32)),
            dtype=np.int32,
        )
        self._visible_mask = classification != constants.PCB_CLASSIFICATION_CEILING

        clusters = np.asarray(
            attrs.get("Clusters", np.zeros(len(xyz), dtype=np.int32)),
            dtype=np.int32,
        )
        all_ids = np.unique(clusters[clusters > 0]).tolist()
        car_ids = [
            int(cid)
            for cid in all_ids
            if np.any(classification[clusters == int(cid)] == constants.PCB_CLASSIFICATION_CAR)
        ]
        self._cluster_ids = car_ids
        self._total_clusters_count = len(car_ids)

        log.info(
            "Loaded %d points, %d clusters",
            self._total_points,
            self._total_clusters_count,
        )

    # ------------------------------------------------------------------
    # Point cloud building
    # ------------------------------------------------------------------

    def _build_pcd_for_cluster(self, cluster_index: int) -> o3d.geometry.PointCloud:
        assert self._xyz is not None
        assert self._attrs is not None
        assert self._visible_mask is not None

        clusters = np.asarray(
            self._attrs.get("Clusters", np.zeros(len(self._xyz), dtype=np.int32)),
            dtype=np.int32,
        )
        current_id = self._cluster_ids[cluster_index]
        current_mask = clusters == int(current_id)

        vis_xyz = self._xyz[self._visible_mask]
        vis_current = current_mask[self._visible_mask]

        colors = np.where(vis_current[:, np.newaxis], _COLOR_BLUE, _COLOR_GRAY)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vis_xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _display_point_cloud(self, pcd: o3d.geometry.PointCloud) -> None:
        if self._preview_scene is None:
            return

        self._preview_scene.scene.clear_geometry()

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 3.0

        self._preview_scene.scene.add_geometry("pcd", pcd, mat)

        if not self._camera_initialized:
            bounds = pcd.get_axis_aligned_bounding_box()
            center = bounds.get_center()
            self._preview_scene.setup_camera(60.0, bounds, center)
            self._camera_initialized = True

    def _refresh_view(self) -> None:
        if self._current_index >= len(self._cluster_ids):
            return
        pcd = self._build_pcd_for_cluster(self._current_index)
        self._display_point_cloud(pcd)
        self._update_cluster_label()

    def _update_cluster_label(self) -> None:
        if self._cluster_label is None:
            return
        cid = self._cluster_ids[self._current_index]
        self._cluster_label.text = (
            f"Cluster {self._current_index + 1} / {self._total_clusters_count}"
            f"  (ID: {cid})"
        )

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_correct(self) -> None:
        if self._current_index >= len(self._cluster_ids):
            return
        log.info(
            "Cluster ID %d marked Correct", self._cluster_ids[self._current_index]
        )
        self._advance()

    def _on_incorrect(self) -> None:
        if self._current_index >= len(self._cluster_ids):
            return
        assert self._xyz is not None and self._attrs is not None

        cid = self._cluster_ids[self._current_index]
        clusters = np.asarray(
            self._attrs.get("Clusters", np.zeros(len(self._xyz), dtype=np.int32)),
            dtype=np.int32,
        )
        n_pts = int(np.sum(clusters == int(cid)))

        self._false_clusters_count += 1
        self._false_points_count += n_pts
        log.info("Cluster ID %d marked Incorrect (%d points)", cid, n_pts)
        self._advance()

    def _advance(self) -> None:
        self._current_index += 1
        if self._current_index >= len(self._cluster_ids):
            self._finish()
        else:
            app = gui.Application.instance
            app.post_to_main_thread(self._window, self._refresh_view)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finish(self) -> None:
        accuracy = (
            (self._false_points_count / self._total_points) * 100.0
            if self._total_points > 0
            else 0.0
        )
        result = {
            "total_points": self._total_points,
            "total_clusters_count": self._total_clusters_count,
            "false_points_count": self._false_points_count,
            "false_clusters_count": self._false_clusters_count,
            "accuracy": round(accuracy, 4),
        }

        os.makedirs(os.path.dirname(os.path.abspath(self._output_path)), exist_ok=True)
        with open(self._output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        log.info("Wrote validation results to %s", self._output_path)
        log.info("Summary: %s", result)

        app = gui.Application.instance

        def _close() -> None:
            if self._window is not None:
                try:
                    self._window.close()
                    app.quit()
                finally:
                    os._exit(0)

        app.post_to_main_thread(self._window, _close)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> None:
        self._load_data(chunk_size=chunk_size)

        if not self._cluster_ids:
            log.warning("No clusters found in input file. Writing empty validation.")
            self._finish()
            return

        app = gui.Application.instance
        app.initialize()

        self._window = app.create_window("Result Validation", 1400, 900)
        em = self._window.theme.font_size

        # --- Left panel ---
        panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))

        panel.add_child(gui.Label("Result Validation Tool"))

        self._cluster_label = gui.Label("–")
        panel.add_child(self._cluster_label)

        correct_btn = gui.Button("Correct")
        correct_btn.horizontal_padding_em = 1.0
        correct_btn.set_on_clicked(self._on_correct)
        panel.add_child(correct_btn)

        incorrect_btn = gui.Button("Incorrect")
        incorrect_btn.horizontal_padding_em = 1.0
        incorrect_btn.set_on_clicked(self._on_incorrect)
        panel.add_child(incorrect_btn)

        # --- Preview (right) ---
        self._preview_scene = gui.SceneWidget()
        self._preview_scene.scene = rendering.Open3DScene(self._window.renderer)
        self._preview_scene.scene.set_background([1.0, 1.0, 1.0, 1.0])

        self._window.add_child(self._preview_scene)
        self._window.add_child(panel)

        def _on_layout(theme: gui.Theme) -> None:
            r = self._window.content_rect
            panel_width = int(20 * em)
            panel.frame = gui.Rect(r.x, r.y, panel_width, r.height)
            self._preview_scene.frame = gui.Rect(
                r.x + panel_width,
                r.y,
                max(0, r.width - panel_width),
                r.height,
            )

        self._window.set_on_layout(_on_layout)

        self._refresh_view()

        log.info("Starting GUI — %d clusters to review", self._total_clusters_count)
        app.run()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manually validate clustering results from the Post Process stage. "
            "Input must be the 3_post_process.las file."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to 3_post_process.las",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path for validation.json (default: next to the input file)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=_DEFAULT_CHUNK_SIZE,
        help=f"Points per read chunk (default: {_DEFAULT_CHUNK_SIZE})",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        output_path = os.path.join(os.path.dirname(input_path), "validation.json")

    log.info("Input : %s", input_path)
    log.info("Output: %s", output_path)

    viewer = ResultValidationViewer(input_path=input_path, output_path=output_path)
    viewer.run(chunk_size=args.chunk_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
