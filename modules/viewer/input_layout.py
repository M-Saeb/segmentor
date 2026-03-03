from typing import Callable
from open3d.visualization import gui
from modules.utils.log_utils import get_dual_logger

log = get_dual_logger(__name__)

class InputLayout:
    def __init__(self, em):
        self.o3d_gui = gui.Vert(0.25 * em, gui.Margins(0, 0, 0, 0))
        self._custom_btns = []
        self._custom_number_fields = []
        for _ in range(3):
            btn = gui.Button("Template Button")
            btn.horizontal_padding_em = 1.0
            btn.visible = False
            self._custom_btns.append(btn)
            self.o3d_gui.add_child(btn)

            number_input = gui.NumberEdit(gui.NumberEdit.Type.INT)
            # number_input.horizontal_padding_em = 1.0
            number_input.visible = True
            self._custom_number_fields.append(number_input)
            self.o3d_gui.add_child(number_input)

    def set_custom_btn(self, title: str, callback: Callable):
        btn = self._get_next_free_custom_btn()
        btn.text = title
        btn.set_on_clicked(callback)
        btn.visible = True
        return btn

    def reset_all_custom_btns(self):
        for btn in self._custom_btns:
            btn.visible = False

    def _get_next_free_custom_btn(self):
        for btn in self._custom_btns:
            if not btn.visible:
                return btn

        raise RuntimeError("No Free Button Found")

