from modules.utils.pipeline import PipelineStage
from .base import InputLayout


CSF_PARAM_CONFIG = {
    "threshold_factor": {"label": "Threshold factor", "visible": True},
    "resolution": {"label": "Resolution", "visible": True},
    "rigidness": {"label": "Rigidness", "visible": True},
    "max_iter": {"label": "Max iterations", "visible": False},
}

CSF_KEY_ORDER = ["threshold_factor", "resolution", "rigidness"]

RANSAC_PARAM_CONFIG = {
    "distance_threshold": {"label": "Distance threshold", "visible": True},
    "num_iterations": {"label": "RANSAC iterations", "visible": True},
}

RANSAC_KEY_ORDER = ["distance_threshold", "num_iterations"]


class InputLayoutGroundSegmentation(InputLayout):
    """Layout for the Ground Segmentation pipeline stage."""

    def create_layout(self) -> None:
        pass

    def configure(self, stage: PipelineStage) -> None:
        method = "basic"
        if hasattr(stage, "params") and isinstance(stage.params, dict):
            method = stage.params.get("method", "basic")
        if method == "csf":
            self._build_params_from_config(stage, CSF_PARAM_CONFIG, CSF_KEY_ORDER)
        elif method == "ransac":
            self._build_params_from_config(stage, RANSAC_PARAM_CONFIG, RANSAC_KEY_ORDER)
        # basic: no numeric parameters
