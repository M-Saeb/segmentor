from .base import InputLayout


PARAM_CONFIG = {
    "nb_neighbors": {"label": "Neighbor count", "visible": True},
    "std_ratio": {"label": "Standard deviation ratio", "visible": True},
}

KEY_ORDER = ["nb_neighbors", "std_ratio"]


class BaseNoiseFilterLayout(InputLayout):
    """
    Shared layout for noise filter stages (initial and post-wall).
    """

    def create_layout(self) -> None:
        pass

    def configure(self, stage) -> None:
        self._build_params_from_config(stage, PARAM_CONFIG, KEY_ORDER)


class InputLayoutFilterNoise(BaseNoiseFilterLayout):
    """Layout for the Filter Noise pipeline stage."""
