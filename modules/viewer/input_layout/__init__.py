from .base import DefaultInputLayout, InputLayout
from .ceiling_segmentation import InputLayoutCeilingSegmentation
from .ground_segmentation import InputLayoutGroundSegmentation
from .post_wall_noise_filter import InputLayoutFilterNoiseAfterWallSegmentation
from .wall_segmentation import InputLayoutWallSegmentation
from .objects_segmentation import InputLayoutObjectsSegmentation
from modules.utils import constants

_STAGE_NAME_TO_LAYOUT = {
    constants.PIPELINE_GROUND_SEGMENTATION: InputLayoutGroundSegmentation,
    constants.PIPELINE_CEILING_SEGMENTATION: InputLayoutCeilingSegmentation,
    constants.PIPELINE_WALL_SEGMENTATION: InputLayoutWallSegmentation,
    constants.PIPELINE_POST_SEGMENTATION_NOISE_FILTER: InputLayoutFilterNoiseAfterWallSegmentation,
    constants.PIPELINE_OBJECTS_SEGMENTATION: InputLayoutObjectsSegmentation,
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
    "InputLayoutGroundSegmentation",
    "InputLayoutCeilingSegmentation",
    "InputLayoutWallSegmentation",
    "InputLayoutFilterNoiseAfterWallSegmentation",
    "InputLayoutObjectsSegmentation",
    "create_input_layout_for_stage",
]
