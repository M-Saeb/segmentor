from modules.controller.main import PipelineStage
from .base import InputLayout


PARAM_CONFIG = {
    "nb_neighbors": {"label": "Neighbor count", "visible": True},
    "std_ratio": {"label": "Standard deviation ratio", "visible": True},
}

KEY_ORDER = ["nb_neighbors", "std_ratio"]


class InputLayoutFilterNoise(InputLayout):
    """Layout for the Filter Noise pipeline stage."""

    def create_layout(self) -> None:
        pass

    def configure(self, stage: PipelineStage) -> None:
        self._build_params_from_config(stage, PARAM_CONFIG, KEY_ORDER)
