import os
import shutil
import yaml
from modules.utils import config
from modules.models.las_wrapper import LasWrapper
from modules.models.ai_object_detector import identified_objects_yaml_path
from modules.utils.log_utils import get_dual_logger

log = get_dual_logger(__name__)

def init_result_folder_path(cfg: config.SegmentorConfig) -> str:
    output_folder_path = cfg.result["output_folder_path"]
    log.info("Output folder path: %s", output_folder_path)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path, exist_ok=True)
        return output_folder_path

    log.info("Output folder already exists")
    if cfg.result["overwrite_output_folder"]:
        log.info("Overwriting output folder")
        shutil.rmtree(output_folder_path)
        os.makedirs(output_folder_path, exist_ok=True)
        return output_folder_path

    log.info("Creating new output folder with increamental name")
    increment = 2
    while os.path.exists(output_folder_path + f"_{increment}"):
        increment += 1
    output_folder_path = output_folder_path + f"_{increment}"
    os.makedirs(output_folder_path, exist_ok=False)
    return output_folder_path


def get_output_las_path(result_folder_path: str) -> str:
    return os.path.join(result_folder_path, "output.las")


def path_for_identified_objects_yaml(result_folder_path: str) -> str:
    """Path to identified_objects.yml under the run output folder."""
    return identified_objects_yaml_path(result_folder_path)


def create_pipeline_yaml_file(result_folder_path: str, pipeline_data: dict) -> str:
    """Write process pipeline (kind and stages with params) to pipeline.yml."""
    path = os.path.join(result_folder_path, "pipeline.yml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(pipeline_data, f, default_flow_style=False, sort_keys=False)
    log.info("Wrote pipeline YAML to: %s", path)
    return path


def create_las_output_file(cfg: config.SegmentorConfig, output_las_path: str) -> LasWrapper:
    """Initialize output LAS file and return writer wrapper."""
    las_writer = LasWrapper(cfg)
    las_writer.initialize_output_file(output_las_path)
    return las_writer


def write_las_chunk_to_output_file(
    las_writer: LasWrapper,
    output_las_path: str,
    index,
    pcd,
    points_attributes,
) -> None:
    las_writer.write_pcb_chunk_to_file(index, pcd, points_attributes, output_las_path)