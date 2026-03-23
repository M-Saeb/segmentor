import dataclasses
from typing import Callable, Dict, List

@dataclasses.dataclass
class PipelineStage:
    name: str
    func: Callable
    params: Dict[str, object] = dataclasses.field(default_factory=dict)
    enabled: bool = False

@dataclasses.dataclass
class Pipeline:
    stages: List[PipelineStage] = dataclasses.field(default_factory=list)
    current_stage_index: int = -1  # starting from -1 to handle iteration
    kind: str = "setup"

    def to_dict(self):
        return {
            "kind": self.kind,
            "stages": [
                {
                    "name": stage.name,
                    "params": dict(stage.params),
                    "enabled": stage.enabled,
                }
                for stage in self.stages
            ],
        }