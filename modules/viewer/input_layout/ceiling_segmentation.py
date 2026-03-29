from modules.utils.pipeline import PipelineStage
from .base import InputLayout


PARAM_CONFIG = {
    "facing_up_threshold": {"label": "Facing up threshold", "visible": True},
    "min_inlier_density": {"label": "Min inlier density", "visible": True},
}

KEY_ORDER = ["facing_up_threshold", "min_inlier_density"]


class InputLayoutCeilingSegmentation(InputLayout):
    """Layout for the Ceiling Segmentation pipeline stage."""

    def create_layout(self) -> None:
        pass

    def configure(self, stage: PipelineStage) -> None:
        self._build_params_from_config(stage, PARAM_CONFIG, KEY_ORDER)
