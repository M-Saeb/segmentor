import threading

from open3d.visualization import gui

from modules.controllers import PreProcessingController
from modules.utils.config import SegmentorConfig
from modules.utils.log_utils import get_dual_logger

log = get_dual_logger(__name__)

# Defaults matching scripts/pre_processing.py main() user_input
_DEFAULT_INPUT_CHUNK_SIZE = 10_000_000
_DEFAULT_OUTPUT_CHUNK_SIZE = 25_000
_DEFAULT_NB_NEIGHBORS = 20
_DEFAULT_STD_RATIO = 2.0
_DEFAULT_VOXEL_SIZE = 0.15


class PreProcessingViewer:
    def __init__(self, config_file: str):
        self.config = SegmentorConfig.initialize(config_file)
        self.controller = PreProcessingController()
        self._window = None
        self._layout = None
        self._loading_dialog = None
        self._input_chunk_edit = None
        self._output_chunk_edit = None
        self._nb_neighbors_edit = None
        self._std_ratio_edit = None
        self._voxel_edit = None
        self._path_label = None

    def _add_numeric_row(
        self,
        panel: gui.Vert,
        em: float,
        display_label: str,
        default: float,
    ) -> gui.NumberEdit:
        row = gui.Horiz(0.25 * em, gui.Margins(0, 0, 0, 0))
        label = gui.Label(display_label)
        field = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        field.set_value(float(default))
        row.add_child(label)
        row.add_child(field)
        panel.add_child(row)
        return field

    def _collect_user_input(self) -> dict:
        return {
            "input_chunk_size": int(round(self._input_chunk_edit.double_value)),
            "output_chunk_size": int(round(self._output_chunk_edit.double_value)),
            "filter_noise": {
                "nb_neighbors": int(round(self._nb_neighbors_edit.double_value)),
                "std_ratio": float(self._std_ratio_edit.double_value),
            },
            "downsample": {
                "voxel_size": float(self._voxel_edit.double_value),
            },
        }

    def _on_run_clicked(self):
        if self._window is None:
            return

        self._show_loading("Pre-processing, this may take a while...")
        app = gui.Application.instance
        error_holder: list[str | None] = [None]
        result_holder: list[str | None] = [None]

        try:
            user_input = self._collect_user_input()
        except Exception:
            log.exception("Invalid parameters")
            self._hide_loading()
            self._show_message("Error", "Invalid parameter values.")
            return

        def _worker():
            try:
                result_holder[0] = self.controller.run_from_user_input(user_input)
            except Exception as e:
                log.exception("Pre-processing failed")
                error_holder[0] = str(e) or "Pre-processing failed. Check the log for details."

            def _on_done():
                try:
                    self._hide_loading()
                    if error_holder[0] is not None:
                        self._show_message("Error", error_holder[0])
                    else:
                        self._show_message(
                            "Done",
                            f"Pre-processed data folder:\n{result_holder[0]}",
                        )
                except Exception:
                    log.exception("Error in pre-processing completion callback")

            app.post_to_main_thread(self._window, _on_done)

        threading.Thread(target=_worker, daemon=True).start()

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

    def _show_loading(self, text: str = "Loading..."):
        if self._window is None:
            return
        if self._loading_dialog is not None:
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

    def create_gui(self):
        app = gui.Application.instance
        app.initialize()

        self._window = app.create_window("Pre-processing", 720, 520)
        em = self._window.theme.font_size

        self._layout = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))

        self._layout.add_child(gui.Label("Input file (from config)"))
        self._path_label = gui.Label(self.config.input_file)
        self._layout.add_child(self._path_label)

        self._layout.add_child(gui.Label("Parameters"))
        self._input_chunk_edit = self._add_numeric_row(
            self._layout,
            em,
            "Input chunk size",
            float(_DEFAULT_INPUT_CHUNK_SIZE),
        )
        self._output_chunk_edit = self._add_numeric_row(
            self._layout,
            em,
            "Output chunk size",
            float(_DEFAULT_OUTPUT_CHUNK_SIZE),
        )
        self._nb_neighbors_edit = self._add_numeric_row(
            self._layout,
            em,
            "Filter noise: nb_neighbors",
            float(_DEFAULT_NB_NEIGHBORS),
        )
        self._std_ratio_edit = self._add_numeric_row(
            self._layout,
            em,
            "Filter noise: std_ratio",
            _DEFAULT_STD_RATIO,
        )
        self._voxel_edit = self._add_numeric_row(
            self._layout,
            em,
            "Downsample: voxel_size",
            _DEFAULT_VOXEL_SIZE,
        )

        run_btn = gui.Button("Run pre-processing")
        run_btn.horizontal_padding_em = 1.0
        run_btn.set_on_clicked(lambda: self._on_run_clicked())
        self._layout.add_child(run_btn)

        self._window.add_child(self._layout)

        def _on_layout(theme):
            r = self._window.content_rect
            self._layout.frame = r

        self._window.set_on_layout(_on_layout)

        app.run()
