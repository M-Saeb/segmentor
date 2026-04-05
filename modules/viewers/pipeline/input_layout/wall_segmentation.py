from modules.utils.pipeline import PipelineStage
from .base import InputLayout


PARAM_CONFIG = {
    "rotation_threshold": {"label": "Rotation threshold", "visible": True},
}

KEY_ORDER = ["rotation_threshold"]


class InputLayoutWallSegmentation(InputLayout):
    """Layout for the Wall Segmentation pipeline stage."""

    def create_layout(self) -> None:
        pass

    def configure(self, stage: PipelineStage) -> None:
        self._build_params_from_config(stage, PARAM_CONFIG, KEY_ORDER)
