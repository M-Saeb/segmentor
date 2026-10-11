import threading

import numpy as np
from open3d.visualization import gui, rendering

from modules.controllers.post_process import PostProcessingController
from modules.utils.config import SegmentorConfig
from modules.utils import constants
from modules.utils.log_utils import get_dual_logger
from modules.utils.pipeline import PipelineStage
from modules.viewers.pipeline.input_layout.objects_segmentation import InputLayoutObjectsSegmentation

log = get_dual_logger(__name__)


class PostProcessingViewer:
    def __init__(self, config_file: str):
        self.config = SegmentorConfig.initialize(config_file)
        self.controller = PostProcessingController()
        self._window = None
        self._layout = None
        self._preview_scene = None
        self._loading_dialog = None
        self._camera_initialized = False
        self._dbscan_stage: PipelineStage | None = None
        self.run_input_btn = None
        self.confirm_input_btn = None

    def _make_dbscan_stage(self) -> PipelineStage:
        obj_cfg = getattr(self.config, "objects_segmentation", {}) or {}
        return PipelineStage(
            name=constants.PIPELINE_OBJECTS_SEGMENTATION,
            func=lambda **kwargs: None,
            params={
                "eps": float(obj_cfg.get("eps", 0.05)),
                "min_points": int(obj_cfg.get("min_points", 10)),
            },
            enabled=True,
        )

    def _get_eps_min_points(self) -> tuple[float, int]:
        assert self._dbscan_stage is not None
        return (
            float(self._dbscan_stage.params["eps"]),
            int(self._dbscan_stage.params["min_points"]),
        )

    def create_gui(self):
        app = gui.Application.instance
        app.initialize()

        self._window = app.create_window("Post-processing", 1600, 900)
        em = self._window.theme.font_size

        self._layout = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        self._layout.add_child(gui.Label("Input (pipeline output)"))
        self._layout.add_child(gui.Label(self.controller.get_pipeline_las_path()))

        self._dbscan_stage = self._make_dbscan_stage()
        input_layout = InputLayoutObjectsSegmentation(em)
        input_layout.configure(self._dbscan_stage)
        self._layout.add_child(gui.Label("DBSCAN (post-processing)"))
        self._layout.add_child(input_layout.o3d_gui)

        self.run_input_btn = gui.Button("Run input")
        self.run_input_btn.horizontal_padding_em = 1.0
        self.run_input_btn.set_on_clicked(self._on_run_input_clicked)
        self._layout.add_child(self.run_input_btn)

        self.confirm_input_btn = gui.Button("Confirm input")
        self.confirm_input_btn.horizontal_padding_em = 1.0
        self.confirm_input_btn.set_on_clicked(self._on_confirm_input_clicked)
        self._layout.add_child(self.confirm_input_btn)

        self._preview_scene = gui.SceneWidget()
        self._preview_scene.scene = rendering.Open3DScene(self._window.renderer)
        self._preview_scene.scene.set_background([1, 1, 1, 1])

        self._window.add_child(self._preview_scene)
        self._window.add_child(self._layout)

        def _on_layout(theme):
            r = self._window.content_rect
            panel_width = int(22 * em)
            self._layout.frame = gui.Rect(r.x, r.y, panel_width, r.height)
            self._preview_scene.frame = gui.Rect(
                r.x + panel_width, r.y, max(0, r.width - panel_width), r.height
            )

        self._window.set_on_layout(_on_layout)

        self._show_loading("Loading merged LAS...")
        err_holder: list[str | None] = [None]

        def _load_worker():
            try:
                self.controller.load_merged_data()
            except Exception as e:
                log.exception("Failed to load pipeline LAS")
                err_holder[0] = str(e) or "Load failed."

            def _done():
                self._hide_loading()
                if err_holder[0]:
                    self._show_message("Error", err_holder[0])
                    return
                try:
                    pcd = self.controller.get_preview_pcd_initial()
                    self._display_point_cloud(pcd)
                except Exception:
                    log.exception("Initial preview failed")
                    self._show_message("Error", "Could not build preview.")

            app.post_to_main_thread(self._window, _done)

        threading.Thread(target=_load_worker, daemon=True).start()

        app.run()

    def _on_run_input_clicked(self):
        if self._window is None:
            return
        self._show_loading("Running DBSCAN...")
        app = gui.Application.instance
        eps, min_pts = self._get_eps_min_points()
        rng = np.random.default_rng()

        # Open3D cluster_dbscan uses the internal JobSystem; run on the GUI thread
        # (same pattern as pipeline viewer preview) to avoid "thread has not been adopted".
        def _run_on_main_thread():
            err_holder: list[str | None] = [None]
            pcd_holder: list = [None]
            try:
                pcd_holder[0] = self.controller.run_dbscan_preview(eps, min_pts, rng=rng)
            except Exception as e:
                log.exception("DBSCAN preview failed")
                err_holder[0] = str(e) or "DBSCAN failed."
            self._hide_loading()
            if err_holder[0]:
                self._show_message("Error", err_holder[0])
                return
            try:
                self._display_point_cloud(pcd_holder[0])
            except Exception:
                log.exception("Preview display failed")

        app.post_to_main_thread(self._window, _run_on_main_thread)

    def _on_confirm_input_clicked(self):
        if self._window is None:
            return
        self._show_loading("DBSCAN, VLM, writing output...")
        app = gui.Application.instance
        eps, min_pts = self._get_eps_min_points()

        # OffscreenRenderer (VLM screenshots) must run on the same thread as the GUI
        # Application; background threads trigger JobSystem PreconditionPanic.
        def _run_on_main_thread():
            err_holder: list[str | None] = [None]
            result_holder: list[tuple | None] = [None]
            try:
                result_holder[0] = self.controller.confirm_post_processing(eps, min_pts)
                log.info("Post-processing Done")
            except Exception as e:
                log.exception("Post-processing failed")
                err_holder[0] = str(e) or "Post-processing failed."
            self._hide_loading()
            if err_holder[0]:
                self._show_message("Error", err_holder[0])
                return
            n, las_path, traces_dir = result_holder[0]
            self._show_message(
                "Done",
                f"Wrote {n:,} points to:\n{las_path}\n\nVLM traces (images + data.yml):\n{traces_dir}",
            )

        app.post_to_main_thread(self._window, _run_on_main_thread)

    def _display_point_cloud(self, pcd):
        if self._preview_scene is None:
            return
        self._preview_scene.scene.clear_geometry()
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 3.0
        self._preview_scene.scene.add_geometry("pcd", pcd, material)
        if not self._camera_initialized and len(pcd.points) > 0:
            bounds = pcd.get_axis_aligned_bounding_box()
            center = bounds.get_center()
            self._preview_scene.setup_camera(60.0, bounds, center)
            self._camera_initialized = True

    def _show_loading(self, text: str = "Loading..."):
        if self._window is None or self._loading_dialog is not None:
            return
        em = self._window.theme.font_size
        dlg = gui.Dialog("Please wait")
        layout = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        layout.add_child(gui.Label(text))
        dlg.add_child(layout)
        self._loading_dialog = dlg
        self._window.show_dialog(dlg)

    def _hide_loading(self):
        if self._window is None or self._loading_dialog is None:
            return
        self._window.close_dialog()
        self._loading_dialog = None

    def _show_message(self, title: str, text: str):
        if self._window is None:
            return
        em = self._window.theme.font_size
        dlg = gui.Dialog(title)
        layout = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        layout.add_child(gui.Label(text))
        ok_btn = gui.Button("OK")

        def _close():
            self._window.close_dialog()

        ok_btn.set_on_clicked(_close)
        layout.add_child(ok_btn)
        dlg.add_child(layout)
        self._window.show_dialog(dlg)
