from modules.utils.pipeline import PipelineStage
from .base import InputLayout


PARAM_CONFIG = {
    "epsilon": {"label": "Epsilon", "visible": True},
    "min_support_points": {"label": "Min support points", "visible": True},
    "max_normal_dev": {"label": "Max normal deviation", "visible": True},
    "max_iterations": {"label": "Max iterations", "visible": False},
    "min_plane_size_ratio": {"label": "Min plane size ratio", "visible": False},
}

KEY_ORDER = ["epsilon", "min_support_points", "max_normal_dev"]


class InputLayoutWallSegmentation(InputLayout):
    """Layout for the Wall Segmentation pipeline stage."""

    def create_layout(self) -> None:
        pass

    def configure(self, stage: PipelineStage) -> None:
        self._build_params_from_config(stage, PARAM_CONFIG, KEY_ORDER)
