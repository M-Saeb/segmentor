from .filter_noise import BaseNoiseFilterLayout


PARAM_CONFIG = {
    "nb_neighbors": {"label": "Neighbor count", "visible": True},
    "std_ratio": {"label": "Standard deviation ratio", "visible": True},
}

KEY_ORDER = ["nb_neighbors", "std_ratio"]


class InputLayoutFilterNoiseAfterWallSegmentation(BaseNoiseFilterLayout):
    """Layout for the Post-wall Noise Filter pipeline stage."""
