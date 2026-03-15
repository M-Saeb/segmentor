from .base import DefaultInputLayout, InputLayout
from .ceiling_segmentation import InputLayoutCeilingSegmentation
from .filter_noise import InputLayoutFilterNoise
from .ground_segmentation import InputLayoutGroundSegmentation
from .post_wall_noise_filter import InputLayoutFilterNoiseAfterWallSegmentation
from .wall_segmentation import InputLayoutWallSegmentation

_STAGE_NAME_TO_LAYOUT = {
    "Filter Noise": InputLayoutFilterNoise,
    "Ground Segmentation": InputLayoutGroundSegmentation,
    "Ceiling Segmentation": InputLayoutCeilingSegmentation,
    "Wall Segmentation": InputLayoutWallSegmentation,
    "Post-wall Noise Filter": InputLayoutFilterNoiseAfterWallSegmentation,
}


def create_input_layout_for_stage(em: float, stage_name: str) -> InputLayout:
    """
    Return an InputLayout instance for the given pipeline stage name.
    Falls back to DefaultInputLayout for unknown stages.
    """
    layout_class = _STAGE_NAME_TO_LAYOUT.get(stage_name, DefaultInputLayout)
    return layout_class(em)


__all__ = [
    "InputLayout",
    "DefaultInputLayout",
    "InputLayoutFilterNoise",
    "InputLayoutGroundSegmentation",
    "InputLayoutCeilingSegmentation",
    "InputLayoutWallSegmentation",
    "InputLayoutFilterNoiseAfterWallSegmentation",
    "create_input_layout_for_stage",
]
