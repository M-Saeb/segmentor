from abc import ABC, abstractmethod
from typing import Any, Dict

from open3d.visualization import gui

from modules.utils.pipeline import PipelineStage
from modules.utils.log_utils import get_dual_logger

log = get_dual_logger(__name__)


class InputLayout(ABC):
    """
    Abstract base for per-stage parameter layouts. Subclasses define which
    params are visible and their display names, and build the parameter panel
    in configure(stage).
    """

    def __init__(self, em: float):
        self._em = em
        # Single vertical container that directly holds all parameter rows.
        self.o3d_gui = gui.Vert(0.25 * em, gui.Margins(0, 0, 0, 0))
        self.create_layout()

    @abstractmethod
    def create_layout(self) -> None:
        """Subclasses may add static structure to the layout. Default: no-op."""
        pass

    @abstractmethod
    def configure(self, stage: PipelineStage) -> None:
        """
        Build the parameter panel for the given pipeline stage. Subclasses
        should clear _params_container and add rows for each visible param.
        """
        pass

    def _add_numeric_param_row(
        self,
        panel: gui.Vert,
        param_key: str,
        display_label: str,
        value: int | float,
        stage: PipelineStage,
    ) -> None:
        """Add a single labeled numeric field that updates stage.params[param_key]."""
        original_type = int if isinstance(value, int) else float
        row = gui.Horiz(0.25 * 1.0, gui.Margins(0, 0, 0, 0))
        label = gui.Label(display_label)
        field = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        field.set_value(float(value))

        def _make_callback(key: str, param_type: type):
            def _on_value_changed(new_value: float):
                if param_type is int:
                    stage.params[key] = int(new_value)
                else:
                    stage.params[key] = float(new_value)

            return _on_value_changed

        field.set_on_value_changed(_make_callback(param_key, original_type))
        row.add_child(label)
        row.add_child(field)
        panel.add_child(row)

    def _build_params_from_config(
        self,
        stage: PipelineStage,
        param_config: Dict[str, Dict[str, Any]],
        key_order: list[str] | None = None,
    ) -> None:
        """
        Clear the parameter panel and add rows for each param in stage.params
        that is present in param_config with visible=True. Uses display labels
        from param_config. key_order defines the display order; if None, uses
        param_config key order.
        """
        if not hasattr(stage, "params"):
            return

        keys = key_order if key_order is not None else list(param_config.keys())
        for key in keys:
            if key not in stage.params or key not in param_config:
                continue
            meta = param_config[key]
            if not meta.get("visible", True):
                continue
            value = stage.params[key]
            if not isinstance(value, (int, float)):
                continue
            label = meta.get("label", key)
            self._add_numeric_param_row(self.o3d_gui, key, label, value, stage)


class DefaultInputLayout(InputLayout):
    """Fallback layout that shows all numeric params with raw key names."""

    def create_layout(self) -> None:
        pass

    def configure(self, stage: PipelineStage) -> None:
        if not hasattr(stage, "params"):
            return
        for key, value in stage.params.items():
            if not isinstance(value, (int, float)):
                continue
            self._add_numeric_param_row(self.o3d_gui, key, str(key), value, stage)
