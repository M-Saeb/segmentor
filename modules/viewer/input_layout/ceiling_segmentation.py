from modules.utils.pipeline import PipelineStage
from .base import InputLayout


PARAM_CONFIG = {
    "distance_threshold": {"label": "Distance threshold", "visible": True},
    "num_iterations": {"label": "Num iterations", "visible": True},
    "max_tilt_from_horizontal_deg": {"label": "Max tilt from horizontal (deg)", "visible": True},
    "min_height_above_ground": {"label": "Min height above ground", "visible": True},
    "max_height_above_ground": {"label": "Max height above ground", "visible": True},
}

KEY_ORDER = [
    "distance_threshold",
    "num_iterations",
    "max_tilt_from_horizontal_deg",
    "min_height_above_ground",
    "max_height_above_ground",
]


class InputLayoutCeilingSegmentation(InputLayout):
    """Layout for the Ceiling Segmentation pipeline stage."""

    def create_layout(self) -> None:
        pass

    def configure(self, stage: PipelineStage) -> None:
        self._build_params_from_config(stage, PARAM_CONFIG, KEY_ORDER)
