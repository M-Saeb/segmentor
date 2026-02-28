from open3d.visualization import gui
import open3d as o3d
from .utils.config import SegmentorConfig
from .utils.log_utils import get_dual_logger
from .controller import  Controller

log = get_dual_logger(__name__)

class Viewer:

    def __init__(self, config: str):
        self._config = SegmentorConfig(config)
        self.controller = Controller(self._config)
        self._window = None
        self._layout = None
        self._path_edit = None

    def create_gui(self):

        app = gui.Application.instance
        app.initialize()

        self._window = app.create_window("Point Cloud GUI", 1200, 700)
        em = self._window.theme.font_size

        # Vertical layout for controls
        layout = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        self._layout = layout

        # Row: label + text edit + browse button
        row = gui.Horiz(0.25 * em)
        row.add_child(gui.Label("Point cloud file:"))

        self._path_edit = gui.TextEdit()
        self._path_edit.placeholder_text = "Path to .las / .laz / .pcd ..."
        row.add_child(self._path_edit)

        browse_btn = gui.Button("Browse...")
        browse_btn.set_on_clicked(lambda: self._open_file_dialog())
        row.add_child(browse_btn)

        self._layout.add_child(row)

        # First action button
        start_reading_las_btn = gui.Button("Start read LAS file")
        start_reading_las_btn.horizontal_padding_em = 1.0
        start_reading_las_btn.set_on_clicked(self._handle_reading_las_button)
        self._layout.add_child(start_reading_las_btn)

        # Second action button
        second_btn = gui.Button("Run action 2")
        second_btn.horizontal_padding_em = 1.0
        second_btn.set_on_clicked(lambda: self._handle_second_button())
        self._layout.add_child(second_btn)

        self._window.add_child(self._layout)

        app.run()

    def _open_file_dialog(self):
        if self._window is None:
            return

        def _on_done(filename):
            # filename is empty string if user canceled
            self._window.close_dialog()
            if filename:
                self._path_edit.text_value = filename
        # Note: in this Open3D version the FileDialog constructor expects (mode, title, theme)
        # and callbacks are set via methods, not the constructor.
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select point cloud file", self._window.theme)
        dlg.set_on_done(_on_done)
        dlg.set_on_cancel(lambda: self._window.close_dialog())
        dlg.add_filter(".las", "LAS files")
        dlg.add_filter(".laz", "LAZ files")
        dlg.add_filter(".*", "All files")
        dlg.set_path("/home/mohammedalsalihi/LidarData/for_mohammed/")
        self._window.show_dialog(dlg)

    def _show_message(self, title, text):

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
        log.info("Clicked _handle_reading_las_button()")
        path = self._get_path()
        if not path:
            log.info("LAS File path not found")
            self._show_message("No file", "Please select or enter a point cloud file path first.")
            return

        pcd = self.controller.get_display_pcd()
        # Display the point cloud for the selected file
        self._display_point_cloud(pcd)

    def _display_point_cloud(self, pcd):
        o3d.visualization.draw_geometries([pcd])

    def _handle_second_button(self):
        print("=== 2 Clicked ===")
        # Lazily create and show a third button below the second one
        if self._layout is not None:
            third_btn = gui.Button("Run action 3")
            third_btn.horizontal_padding_em = 1.0

            def _on_third():
                self._show_message("Action 3", "Third button clicked.")

            third_btn.set_on_clicked(_on_third)
            self._layout.add_child(third_btn)

            # Ask Open3D to recompute layout
            self._window.set_needs_layout()

if __name__ == "__main__":
    view = Viewer()
    # Example callback that accepts the selected path
    view.create_gui()