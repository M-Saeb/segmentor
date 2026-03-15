from modules.controller.main import PipelineStage
from .base import InputLayout


PARAM_CONFIG = {
    "epsilon": {"label": "Epsilon", "visible": True},
    "min_support_points": {"label": "Min support points", "visible": True},
    "max_tilt_from_horizontal_deg": {"label": "Max tilt from horizontal (deg)", "visible": True},
    "max_iterations": {"label": "Max iterations", "visible": True},
    "min_plane_size_ratio": {"label": "Min plane size ratio", "visible": True},
    "min_height_above_ground": {"label": "Min height above ground", "visible": True},
    "max_height_above_ground": {"label": "Max height above ground", "visible": True},
}

KEY_ORDER = [
    "epsilon",
    "min_support_points",
    "max_tilt_from_horizontal_deg",
    "max_iterations",
    "min_plane_size_ratio",
    "min_height_above_ground",
    "max_height_above_ground",
]


class InputLayoutCeilingSegmentation(InputLayout):
    """Layout for the Ceiling Segmentation pipeline stage."""

    def create_layout(self) -> None:
        pass

    def configure(self, stage: PipelineStage) -> None:
        self._build_params_from_config(stage, PARAM_CONFIG, KEY_ORDER)
