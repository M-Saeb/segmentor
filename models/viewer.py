from open3d.visualization import gui
import open3d as o3d
import laspy
import numpy as np
from utils import apply_rgb_colors


class Viewer:

    def __init__(self):
        # Open3D GUI objects will be created in create_gui
        self._window = None
        self._path_edit = "/home/mohammedalsalihi/LidarData/for_mohammed/office_og.las"
        self._on_first_action = None
        self._on_second_action = None

    def create_gui(self, on_first_action=None, on_second_action=None):
        """
        Create and start an Open3D GUI that lets the user:
        - Choose or type a point cloud file path.
        - Click one of two buttons that each trigger a callback.

        Args:
            on_first_action (Callable[[str], None] | None): Optional callback for the first button.
            on_second_action (Callable[[str], None] | None): Optional callback for the second button.
        """

        self._on_first_action = on_first_action
        self._on_second_action = on_second_action

        app = gui.Application.instance
        app.initialize()

        self._window = app.create_window("Point Cloud GUI", 1200, 700)
        em = self._window.theme.font_size

        # Main horizontal layout: left controls, right 3D view
        layout = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))

        # Row: label + text edit + browse button
        row = gui.Horiz(0.25 * em)
        row.add_child(gui.Label("Point cloud file:"))

        self._path_edit = gui.TextEdit()
        self._path_edit.placeholder_text = "Path to .las / .laz / .pcd ..."
        row.add_child(self._path_edit)

        browse_btn = gui.Button("Browse...")
        browse_btn.set_on_clicked(lambda: self._open_file_dialog())
        row.add_child(browse_btn)

        layout.add_child(row)

        # First action button
        first_btn = gui.Button("Run action 1")
        first_btn.horizontal_padding_em = 1.0
        first_btn.set_on_clicked(lambda: self._handle_first_button())
        layout.add_child(first_btn)

        # Second action button
        second_btn = gui.Button("Run action 2")
        second_btn.horizontal_padding_em = 1.0
        second_btn.set_on_clicked(lambda: self._handle_second_button())
        layout.add_child(second_btn)

        self._window.add_child(layout)

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
        return self._path_edit.text_value.strip() if self._path_edit is not None else ""

    def _handle_first_button(self):
        print("=== Click ===")
        path = self._get_path()
        if not path:
            print("path not found")
            self._show_message("No file", "Please select or enter a point cloud file path first.")
            return

        # Display the point cloud for the selected file
        self._display_point_cloud(path)

        # Also invoke optional external callback if provided
        if callable(self._on_first_action):
            self._on_first_action(path)
        else:
            self._show_message("Action 1", f"Displayed point cloud for:\n{path}")

    def _display_point_cloud(self, path: str):
        """Load a LAS/LAZ file and display it using Open3D."""
        try:
            with laspy.open(path) as f:
                las = f.read()
        except Exception as exc:
            self._show_message("Load error", f"Failed to read file:\n{path}\n\n{exc}")
            return

        try:
            xyz = np.vstack((las.x, las.y, las.z)).T
        except Exception as exc:
            self._show_message("Data error", f"Failed to extract XYZ from file:\n{path}\n\n{exc}")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # Try to apply RGB colors if available
        points_attributes = {}
        try:
            for dim in las.point_format.dimension_names:
                if dim in ("x", "y", "z"):
                    continue
                points_attributes[dim] = getattr(las, dim)

            if all(c in points_attributes for c in ("red", "green", "blue")):
                pcd = apply_rgb_colors(pcd, points_attributes)
        except Exception:
            # If anything goes wrong, just show the geometry without colors
            pass

        o3d.visualization.draw_geometries([pcd])

    def _handle_second_button(self):
        path = self._get_path()
        if not path:
            self._show_message("No file", "Please select or enter a point cloud file path first.")
            return

        if callable(self._on_second_action):
            self._on_second_action(path)
        else:
            self._show_message("Action 2", f"Second action triggered for:\n{path}")

if __name__ == "__main__":
    view = Viewer()
    # Example callback that accepts the selected path
    random_function = lambda path: print("Selected path:", path)
    view.create_gui(random_function, random_function)