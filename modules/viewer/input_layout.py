from typing import Callable
from open3d.visualization import gui
from modules.utils.log_utils import get_dual_logger

log = get_dual_logger(__name__)

class InputLayout:
    def __init__(self, em):
        self._em = em
        self.o3d_gui = gui.Vert(0.25 * em, gui.Margins(0, 0, 0, 0))

        # Container for stage buttons (e.g., "Filter Noise").
        self._buttons_container = gui.Vert(0.25 * em, gui.Margins(0, 0, 0, 0))
        # Proxy area for parameter controls of the currently selected stage.
        # Using WidgetProxy allows us to swap out the entire parameter panel
        # without needing clear_children() support.
        self._params_proxy = gui.WidgetProxy()

        self._custom_btns = []
        for _ in range(3):
            btn = gui.Button("Template Button")
            btn.horizontal_padding_em = 1.0
            btn.visible = False
            self._custom_btns.append(btn)
            self._buttons_container.add_child(btn)

        self.o3d_gui.add_child(self._buttons_container)
        self.o3d_gui.add_child(self._params_proxy)

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

    def configure_for_stage(self, stage):
        """
        Configure the numeric controls to edit parameters for the given pipeline stage.
        """
        if not hasattr(stage, "params"):
            return

        # Decide which parameters to expose for this stage.
        param_key_order = None
        if getattr(stage, "name", "") in ("Filter Noise", "Post-wall Noise Filter"):
            param_key_order = ["nb_neighbors", "std_ratio"]
        elif getattr(stage, "name", "") == "Ground Segmentation":
            # Focus on the most relevant CSF parameters for interactive tuning.
            param_key_order = ["threshold_factor", "resolution", "rigidness"]
        elif getattr(stage, "name", "") == "Wall Segmentation":
            param_key_order = ["epsilon", "min_support_points", "max_normal_dev"]

        if param_key_order is not None:
            keys = [k for k in param_key_order if k in stage.params]
        else:
            # Fallback: expose all numeric params we can find.
            keys = [k for k, v in stage.params.items() if isinstance(v, (int, float))]

        # Build a fresh parameter panel for this stage.
        params_panel = gui.Vert(0.25 * self._em, gui.Margins(0, 0, 0, 0))

        for key in keys:
            original_value = stage.params[key]
            if not isinstance(original_value, (int, float)):
                continue

            original_type = int if isinstance(original_value, int) else float

            # Horizontal row: label + number field
            row = gui.Horiz(0.25 * 1.0, gui.Margins(0, 0, 0, 0))
            label = gui.Label(str(key))
            field = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
            field.set_value(float(original_value))

            def _make_callback(param_key, param_type):
                def _on_value_changed(new_value):
                    # Keep the underlying parameter type stable (int vs float).
                    if param_type is int:
                        stage.params[param_key] = int(new_value)
                    else:
                        stage.params[param_key] = float(new_value)

                return _on_value_changed

            field.set_on_value_changed(_make_callback(key, original_type))
            row.add_child(label)
            row.add_child(field)
            params_panel.add_child(row)

        # Even if there are no rows, set a (possibly empty) panel so the
        # previous stage's controls disappear.
        self._params_proxy.set_widget(params_panel)

