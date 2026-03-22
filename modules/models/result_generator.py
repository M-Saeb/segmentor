import os
import shutil
import yaml
from modules.utils import config
from modules.models.las_wrapper import LasWrapper
from modules.models.ai_object_detector import identified_objects_yaml_path
from modules.utils.log_utils import get_dual_logger

log = get_dual_logger(__name__)

class ResultGenerator:
    def __init__(self, config: config.SegmentorConfig):
        self.config = config
        self.las_wrapper = LasWrapper(config, auto_read_input_file=False)
        self.output_folder_path = self._make_output_folder()
        self.full_las_path = os.path.join(self.output_folder_path, "output.las")

    def reset_las_file(self):
        self.las_wrapper.initialize_output_file(self.full_las_path)

    def _make_output_folder(self):
        output_folder_path = self.config.result["output_folder_path"]
        log.info("Output folder path: %s", output_folder_path)
        if not os.path.exists(output_folder_path):
            os.makedirs(
                output_folder_path,
                exist_ok=True
            )
        else:
            log.info("Output folder already exists")
            if self.config.result["overwrite_output_folder"]:
                log.info("Overwriting output folder")
                shutil.rmtree(output_folder_path)
                os.makedirs(output_folder_path, exist_ok=True)
            else:
                log.info("Creating new output folder with increamental name")
                # create increamental folder name
                increment = 2
                while os.path.exists(output_folder_path + f"_{increment}"):
                    increment += 1
                output_folder_path = output_folder_path + f"_{increment}"
                os.makedirs(output_folder_path, exist_ok=False)

        return output_folder_path

    def path_for_identified_objects_yaml(self) -> str:
        """Path to identified_objects.yml under the run output folder."""
        return identified_objects_yaml_path(self.output_folder_path)

    def write_pipeline_yaml(self, pipeline_data: dict):
        """Write the process pipeline (kind and stages with params) to pipeline.yml in the output folder."""
        path = os.path.join(self.output_folder_path, "pipeline.yml")
        with open(path, "w") as f:
            yaml.safe_dump(pipeline_data, f, default_flow_style=False, sort_keys=False)
        log.info("Wrote pipeline YAML to: %s", path)

    def write_las_chunk_to_file(self, index, pcd, points_attributes):
        self.las_wrapper.write_pcb_chunk_to_file(index, pcd, points_attributes, self.full_las_path)