from open3d.visualization import gui, rendering
import threading
from modules.utils.config import SegmentorConfig
from modules.utils.log_utils import get_dual_logger
from modules.controller import Controller
from .input_layout import create_input_layout_for_stage

log = get_dual_logger(__name__)

class Viewer:

    def __init__(self, config: str):
        self._config = SegmentorConfig(config)
        self.controller = Controller(self._config)
        self._window = None
        self._layout = None  # left control panel (container)
        self._main_layout = None  # section for main buttons
        self._input_layout = None  # current per-stage layout (recreated when stage changes)
        self._stage_buttons_container = None  # Vert for Run / Confirm buttons
        self._params_proxy = None  # WidgetProxy to swap in current stage's param panel
        self.run_btn = None
        self.confirm_btn = None
        self._preview_scene = None
        self._loading_dialog = None
        self._camera_initialized = False


    def create_gui(self):

        app = gui.Application.instance
        app.initialize()

        # Create a larger main window so the interface is more spacious
        self._window = app.create_window("Point Cloud GUI", 1600, 900)

        em = self._window.theme.font_size

        # Vertical layout for controls (left panel)
        layout = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        self._layout = layout

        # --- Main buttons section (structure only for now) ---
        self._main_layout = gui.Vert(0.25 * em, gui.Margins(0, 0, 0, 0))
        rgb_colors_button = gui.Button("Apply RGB Colors")
        rgb_colors_button.horizontal_padding_em = 1.0
        rgb_colors_button.set_on_clicked(
            lambda: self.update_preview_pcd("RGB")
        )
        self._main_layout.add_child(rgb_colors_button)

        classifactions_colors_button = gui.Button("Apply Classification Colors")
        classifactions_colors_button.horizontal_padding_em = 1.0
        classifactions_colors_button.set_on_clicked(
            lambda: self.update_preview_pcd("Classification")
        )
        self._main_layout.add_child(classifactions_colors_button)

        self.run_btn = gui.Button(f"Run Stage")
        self.run_btn.horizontal_padding_em = 1.0
        self._main_layout.add_child(self.run_btn)

        self.confirm_btn = gui.Button(f"Confirm Stage")
        self.confirm_btn.horizontal_padding_em = 1.0
        self._main_layout.add_child(self.confirm_btn)

        finalize_button = gui.Button("Finalize & Run Full Processing")
        finalize_button.horizontal_padding_em = 1.0
        finalize_button.set_on_clicked(
            lambda: self._run_full_processing()
        )
        self._main_layout.add_child(finalize_button)


        self._layout.add_child(gui.Label("General"))
        self._layout.add_child(self._main_layout)

        self._params_proxy = gui.WidgetProxy()
        self._layout.add_child(gui.Label("Parameters"))
        self._layout.add_child(self._params_proxy)

        # Create a SceneWidget to display the point cloud inside the same window
        self._preview_scene = gui.SceneWidget()
        self._preview_scene.scene = rendering.Open3DScene(self._window.renderer)

        # Optional: set a neutral background color
        self._preview_scene.scene.set_background([1, 1, 1, 1])

        # Add both widgets directly to the window
        self._window.add_child(self._preview_scene)
        self._window.add_child(self._layout)

        # Explicitly control layout so widgets don't disappear / collapse
        def _on_layout(theme):
            r = self._window.content_rect

            # Reserve a fixed-width panel on the left for controls (slightly wider)
            panel_width = int(22 * em)

            self._layout.frame = gui.Rect(r.x, r.y, panel_width, r.height)
            self._preview_scene.frame = gui.Rect(
                r.x + panel_width, r.y, max(0, r.width - panel_width), r.height
            )

        self._window.set_on_layout(_on_layout)

        self.update_preview_pcd()

        self.switch_to_next_stage_for_setup_pipeline()

        app.run()

    def _stage_preview_btn_wrapper(self):
        """
        Run only the current setup pipeline stage on the preview chunk and
        refresh the display, without advancing to the next stage.
        """
        if self._window is None:
            return

        # Show loading dialog immediately on the GUI thread.
        self._show_loading("Loading...")

        app = gui.Application.instance

        def _worker():
            try:
                # Heavy computation runs in background thread.
                self.controller.run_current_stage_of_setup_pipeline()
            except Exception:
                log.exception("Error while running setup pipeline stage preview")

            # Once done, update GUI on the main thread.
            def _on_done():
                try:
                    self.update_preview_pcd()
                finally:
                    self._hide_loading()

            app.post_to_main_thread(self._window, _on_done)

        threading.Thread(target=_worker, daemon=True).start()

    def _stage_confirm_btn_wrapper(self):
        """
        Confirm the current stage (keeping the last preview result) and
        advance to the next stage, reconfiguring the UI accordingly.
        """
        log.info("confirm stage button clicked")
        if self._window is None:
            return

        # Mark the current stage as enabled before advancing so that only
        # confirmed stages are later included in the process pipeline.
        current_stage = self.controller.get_current_setup_pipeline_stage()
        if current_stage is not None:
            current_stage.enabled = True

        try:
            next_stage = self.controller.confirm_current_stage_and_advance()
        except Exception:
            log.exception("Error while confirming setup pipeline stage")
            return

        # Clear existing buttons and, if there is a next stage, configure UI
        # for it without advancing again.
        self.switch_to_next_stage_for_setup_pipeline(advance=False)

    def switch_to_next_stage_for_setup_pipeline(self, advance: bool = True):
        """
        Move to the next setup pipeline stage (optionally) and reconfigure the
        UI to show its parameters and actions.

        When `advance` is True, the controller also snapshots the current
        preview chunk so that repeated previews for the new stage always start
        from the same input.
        """
        if advance:
            self.controller.switch_to_next_setup_pipeline_stage()
        stage_info = self.controller.get_current_setup_pipeline_stage()
        if stage_info is None:
            return
        em = self._window.theme.font_size
        self._input_layout = create_input_layout_for_stage(em, stage_info.name)
        self._input_layout.configure(stage_info)  # type: ignore[arg-type]
        self._params_proxy.set_widget(self._input_layout.o3d_gui)
        self.run_btn.set_on_clicked(self._stage_preview_btn_wrapper)
        self.run_btn.text = f"Run {stage_info.name}"
        self.confirm_btn.set_on_clicked(self._stage_confirm_btn_wrapper)
        self.confirm_btn.text = f"Confirm {stage_info.name}"

    def _run_full_processing(self):
        """
        Build the process pipeline from the current setup pipeline parameters
        and run it over all chunks of the input file. Runs in a background
        thread so the UI stays responsive and a waiting dialog is shown.
        """
        if self._window is None:
            return

        self._show_loading("Generated Output Folder, This may take a while ...")
        app = gui.Application.instance
        error_holder = [None]  # message string if failed, None if success

        def _on_done():
            try:
                self._hide_loading()
                if error_holder[0] is not None:
                    self.show_message("Error", error_holder[0])
                else:
                    path = self.controller.result_generator.full_las_path
                    self.show_message("Done", f"Output written to {path}")
            except Exception:
                log.exception("Error in full processing completion callback")

        def _worker():
            try:
                self.controller.build_process_pipeline_from_setup()
                self.controller.run_process_pipeline_on_all_chunks()
            except Exception:
                log.exception("Error while running full processing")
                error_holder[0] = "Full processing failed. Check the log for details."
            app.post_to_main_thread(self._window, _on_done)

        threading.Thread(target=_worker, daemon=True).start()

    def show_message(self, title, text):

        log.info("_show_message() hit")
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

    def _show_loading(self, text: str = "Loading..."):
        """
        Show a simple modal 'loading' dialog while work is in progress.
        """
        if self._window is None:
            return
        if self._loading_dialog is not None:
            # Already showing
            return

        em = self._window.theme.font_size
        dlg = gui.Dialog("Please wait")
        layout = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        layout.add_child(gui.Label(text))
        dlg.add_child(layout)

        self._loading_dialog = dlg
        self._window.show_dialog(dlg)
        log.info("_show_loading() hit: %s", text)

    def _hide_loading(self):
        """
        Hide the loading dialog if it is currently shown.
        """
        if self._window is None:
            return
        if self._loading_dialog is None:
            return

        # close_dialog() closes the top-most dialog.
        self._window.close_dialog()
        self._loading_dialog = None

    def update_preview_pcd(self, color="Classification"):
        pcd = self.controller.get_preview_pcd(color)
        self._display_point_cloud(pcd)

    def _display_point_cloud(self, pcd):
        # Render the point cloud inside the existing SceneWidget instead of a new window
        if self._preview_scene is None:
            log.error("Scene widget is not initialized; cannot display point cloud.")
            return

        self._preview_scene.scene.clear_geometry()

        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 3.0

        self._preview_scene.scene.add_geometry("pcd", pcd, material)

        # Only set up the camera once; subsequent updates keep the current view
        if not self._camera_initialized:
            bounds = pcd.get_axis_aligned_bounding_box()
            center = bounds.get_center()
            self._preview_scene.setup_camera(60.0, bounds, center)
            self._camera_initialized = True
