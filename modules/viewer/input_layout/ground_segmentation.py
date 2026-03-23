from modules.utils.pipeline import PipelineStage
from .base import InputLayout


PARAM_CONFIG = {
    "threshold_factor": {"label": "Threshold factor", "visible": True},
    "resolution": {"label": "Resolution", "visible": True},
    "rigidness": {"label": "Rigidness", "visible": True},
    "max_iter": {"label": "Max iterations", "visible": False},
}

KEY_ORDER = ["threshold_factor", "resolution", "rigidness"]


class InputLayoutGroundSegmentation(InputLayout):
    """Layout for the Ground Segmentation pipeline stage."""

    def create_layout(self) -> None:
        pass

    def configure(self, stage: PipelineStage) -> None:
        self._build_params_from_config(stage, PARAM_CONFIG, KEY_ORDER)
