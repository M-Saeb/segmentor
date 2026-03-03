from open3d.visualization import gui, rendering
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
        self._scene = None


    def create_gui(self):

        app = gui.Application.instance
        app.initialize()

        self._window = app.create_window("Point Cloud GUI", 1200, 700)
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
            lambda: self.update_displayed_pcb("RGB")
        )
        self._main_layout.add_child(rgb_colors_button)

        classifactions_colors_button = gui.Button("Apply Classification Colors")
        classifactions_colors_button.horizontal_padding_em = 1.0
        classifactions_colors_button.set_on_clicked(
            lambda: self.update_displayed_pcb("Classification")
        )
        self._main_layout.add_child(classifactions_colors_button)

        finalize_button = gui.Button("Finalize & Run Full Processing")
        finalize_button.horizontal_padding_em = 1.0
        finalize_button.set_on_clicked(
            lambda: self._input_btn_wrapper(self._run_full_processing)
        )
        self._main_layout.add_child(finalize_button)

        self._layout.add_child(gui.Label("General"))
        self._layout.add_child(self._main_layout)

        # --- Custom buttons section ---
        self._layout.add_child(gui.Label("Custom buttons"))
        self._layout.add_child(self._input_layout.o3d_gui)

        self._input_layout.set_custom_btn(
            "Start read LAS file",
            callback=lambda: self._input_btn_wrapper(self._handle_reading_las_button)
        )

        # Create a SceneWidget to display the point cloud inside the same window
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self._window.renderer)

        # Optional: set a neutral background color
        self._scene.scene.set_background([1, 1, 1, 1])

        # Add both widgets directly to the window
        self._window.add_child(self._scene)
        self._window.add_child(self._layout)

        # Explicitly control layout so widgets don't disappear / collapse
        def _on_layout(theme):
            r = self._window.content_rect

            # Reserve a fixed-width panel on the left for controls
            panel_width = int(18 * em)

            self._layout.frame = gui.Rect(r.x, r.y, panel_width, r.height)
            self._scene.frame = gui.Rect(
                r.x + panel_width, r.y, max(0, r.width - panel_width), r.height
            )

        self._window.set_on_layout(_on_layout)

        app.run()

    def _input_btn_wrapper(self, callback):
        log.info("button clicked %s", str(callback))
        callback()
        self.update_displayed_pcb()
        

    def switch_to_next_pipeline_stage(self):
        self._input_layout.reset_all_custom_btns()
        self.controller.switch_to_next_pipeline_stage()
        stage = self.controller.get_current_pipeline_stage()
        if stage is None:
            return
        # Reconfigure parameter controls for the newly selected stage.
        self._input_layout.configure_for_stage(stage)
        self._input_layout.set_custom_btn(
            title=stage.name,
            callback=lambda: self._input_btn_wrapper(self.controller.run_current_pipeline)
        )

    def _run_full_processing(self):
        """
        Build the process pipeline from the current setup pipeline parameters
        and run it over all chunks of the input file.
        """
        self.controller.build_process_pipeline_from_setup()
        self.controller.run_process_pipeline_on_all_chunks()

    def show_message(self, title, text):

        print("_show_message() hit")
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

    def _get_path(self):
        # return "/home/mohammedalsalihi/LidarData/for_mohammed/office_og.las"
        return self._config.input_file

    def _handle_reading_las_button(self):
        path = self._get_path()
        if not path:
            log.info("LAS File path not found")
            self.show_message("No file", "Please select or enter a point cloud file path first.")
            return
        self.update_displayed_pcb()
        self.switch_to_next_pipeline_stage()

    def update_displayed_pcb(self, color="Classification"):
        pcd = self.controller.get_display_pcd(color)
        self._display_point_cloud(pcd)

    def _display_point_cloud(self, pcd):
        # Render the point cloud inside the existing SceneWidget instead of a new window
        if self._scene is None:
            log.error("Scene widget is not initialized; cannot display point cloud.")
            return

        self._scene.scene.clear_geometry()

        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 3.0

        self._scene.scene.add_geometry("pcd", pcd, material)

        # Set up the camera to nicely frame the point cloud
        bounds = pcd.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        self._scene.setup_camera(60.0, bounds, center)
