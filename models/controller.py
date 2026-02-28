import dataclasses
from typing import Callable, Dict, List
from .utils.config import SegmentorConfig
from .utils.pcd_utils import cloth_simulation_filter
from .las_wrapper import LasWrapper
from .pcb_visualizer import PCBVisualizer

@dataclasses.dataclass
class PipelineStage:
    name: str
    func: Callable
    inputs: List[
        Dict[str, str]
    ] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class Pipeline:
    stages: List[PipelineStage] = dataclasses.field(default_factory=list)
    current_stage_index: int = 0

class Controller:
    def __init__(self, config: SegmentorConfig):
        self.pipeline = self._create_pipeline()
        self.config = config
        self.las_wrapper = LasWrapper(config)

    def _create_pipeline(self):
        new_pipeline = Pipeline()
        config_info = SegmentorConfig("./config.yml")
        if config_info.filter_noise:
            new_pipeline.stages.append(
                PipelineStage(
                    name="Filter Noise",
                    func=cloth_simulation_filter
                )
            )
        if len(new_pipeline.stages) == 0:
            raise ValueError("No stages for the pipeline was created")

        return new_pipeline

    def get_display_pcd(self):
        pcd, points_attributes = self.las_wrapper.get_current_chunk()
        visualizer = PCBVisualizer(pcd, points_attributes)
        visualizer.apply_rgb_colors()
        return visualizer.pcd


if __name__ == "__main__":
    con = Controller()