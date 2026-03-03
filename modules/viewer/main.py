from open3d.visualization import gui, rendering
import threading
from modules.utils.config import SegmentorConfig
from modules.utils.log_utils import get_dual_logger
from modules.controller import  Controller
from .input_layout import InputLayout

log = get_dual_logger(__name__)

class Viewer:

    def __init__(self, config: str):
        self._config = SegmentorConfig(config)
        self.controller = Controller(self._config)
        self._window = None
        self._layout = None  # left control panel (container)
        self._main_layout = None  # section for main buttons
        self._input_layout = None
        self._preview_scene = None
        self._loading_dialog = None


    def create_gui(self):

        app = gui.Application.instance
        app.initialize()

        # Create a larger main window so the interface is more spacious
        self._window = app.create_window("Point Cloud GUI", 1600, 900)

        em = self._window.theme.font_size

        self._input_layout = InputLayout(em)

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

        finalize_button = gui.Button("Finalize & Run Full Processing")
        finalize_button.horizontal_padding_em = 1.0
        finalize_button.set_on_clicked(
            lambda: self._run_full_processing()
        )
        self._main_layout.add_child(finalize_button)

        self._layout.add_child(gui.Label("General"))
        self._layout.add_child(self._main_layout)

        # --- Custom buttons section ---
        self._layout.add_child(gui.Label("Custom buttons"))
        self._layout.add_child(self._input_layout.o3d_gui)


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
        log.info("preview stage button clicked")
        if self._window is None:
            return

        # Show loading dialog immediately on the GUI thread.
        self._show_loading("Running preview...")

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

        try:
            next_stage = self.controller.confirm_current_stage_and_advance()
        except Exception:
            log.exception("Error while confirming setup pipeline stage")
            return

        # Clear existing buttons and, if there is a next stage, configure UI
        # for it without advancing again.
        self.switch_to_next_stage_for_setup_pipeline(advance=False)

    def switch_to_next_stage_for_setup_pipeline(self, advance: bool = True):
        self._input_layout.reset_all_custom_btns()
        if advance:
            self.controller.switch_to_next_setup_pipeline_stage()
        stage = self.controller.get_current_setup_pipeline_stage()
        if stage is None:
            return
        # Reconfigure parameter controls for the newly selected stage.
        self._input_layout.configure_for_stage(stage)
        self._input_layout.set_custom_btn(
            title=f"Run {stage.name}",
            callback=self._stage_preview_btn_wrapper,
        )
        self._input_layout.set_custom_btn(
            title=f"Confirm {stage.name}",
            callback=self._stage_confirm_btn_wrapper,
        )

    def _run_full_processing(self):
        """
        Build the process pipeline from the current setup pipeline parameters
        and run it over all chunks of the input file.
        """
        self.controller.build_process_pipeline_from_setup()
        self.controller.run_process_pipeline_on_all_chunks()

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

        # Set up the camera to nicely frame the point cloud
        bounds = pcd.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        self._preview_scene.setup_camera(60.0, bounds, center)
