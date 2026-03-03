import dataclasses
from typing import Callable, Dict, List
from modules.utils.config import SegmentorConfig
from modules.models.las_wrapper import LasWrapper
from modules.models import pcd_tools
from .pcb_previewer import PCBPreviewer

@dataclasses.dataclass
class PipelineStage:
    name: str
    func: Callable
    inputs: List[
        Dict[str, any]
    ] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class Pipeline:
    stages: List[PipelineStage] = dataclasses.field(default_factory=list)
    current_stage_index: int = -1 # starting from -1 to handle iteration

class Controller:
    def __init__(self, config: SegmentorConfig):
        self.config = config
        self.las_wrapper = LasWrapper(config)
        self.pipeline = self._create_pipeline()

    def _create_pipeline(self):
        new_pipeline = Pipeline()
        if self.config.filter_noise:
            new_pipeline.stages.append(
                PipelineStage(
                    name="Filter Noise",
                    func=lambda **kwargs: self._pipeline_functions_wrapper(
                        pcd_tools.filter_noise,
                        **kwargs
                    ),
                    inputs=[
                        {"nb_neighbors": 20},
                        {"std_ratio": 2.0}
                    ]
                )
            )
        if len(new_pipeline.stages) == 0:
            raise ValueError("No stages for the pipeline was created")

        return new_pipeline

    def _pipeline_functions_wrapper(self, callback, **kwargs):
        pcd, points_attributes = callback(
            *self.las_wrapper.get_current_chunk(),
            **kwargs
        )
        self.las_wrapper.set_current_chunk(pcd, points_attributes)

    def switch_to_next_pipeline_stage(self):
        self.pipeline.current_stage_index += 1

    def get_current_pipeline_stage(self):
        if self.pipeline.current_stage_index >= len(self.pipeline.stages):
            return None
        return self.pipeline.stages[self.pipeline.current_stage_index]

    def run_current_pipeline(self, **kwargs):
        current_stage = self.get_current_pipeline_stage()
        current_stage.func(**kwargs)


    def get_display_pcd(self, color="RGB"):
        pcd, points_attributes = self.las_wrapper.get_current_chunk()
        previewer = PCBPreviewer(pcd, points_attributes)
        previewer.apply_compress(
            max_points=self.config.previewer["maximum_points"]
        )
        if color == "RGB":
            previewer.apply_rgb_colors()
        elif color == "Classification":
            previewer.apply_classfication_colors()
        return previewer.pcd


if __name__ == "__main__":
    con = Controller()