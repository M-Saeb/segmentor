from modules.utils.pipeline import PipelineStage
from .base import InputLayout


PARAM_CONFIG = {
    "eps": {"label": "Epsilon (eps)", "visible": True},
    "min_points": {"label": "Min points", "visible": True},
}

KEY_ORDER = ["eps", "min_points"]


class InputLayoutObjectsSegmentation(InputLayout):
    """Layout for the Objects Segmentation (DBSCAN) pipeline stage."""

    def create_layout(self) -> None:
        pass

    def configure(self, stage: PipelineStage) -> None:
        self._build_params_from_config(stage, PARAM_CONFIG, KEY_ORDER)
