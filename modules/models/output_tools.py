from typing import Union
import os
import yaml
from modules.utils import config
from modules.models.ai_object_detector import identified_objects_yaml_path
from modules.utils.log_utils import get_dual_logger
from .las_tools.las_wrapper import LasWrapper

log = get_dual_logger(__name__)

def init_result_folder_path(cfg: config.SegmentorConfig, use_existing_folder: bool = False) -> str:
    output_folder_path = cfg.result["output_folder_path"]
    log.info("Getting output folder path: %s", output_folder_path)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path, exist_ok=True)
        return output_folder_path

    # check if folder is empty
    if os.path.isdir(output_folder_path) and not os.listdir(output_folder_path):
        log.info("Output folder exists but it is empty, using it")
        return output_folder_path

    if use_existing_folder:
        log.info("Using existing output folder")
        return output_folder_path

    log.info("Creating new output folder with increamental name")
    increment = 2
    while os.path.exists(output_folder_path + f"_{increment}"):
        increment += 1
    output_folder_path = output_folder_path + f"_{increment}"
    os.makedirs(output_folder_path, exist_ok=False)
    return output_folder_path

def get_pre_processing_output_las_path(result_folder_path: Union[str, None]= None) -> str:
    if result_folder_path is None:
        result_folder_path = init_result_folder_path(use_existing_folder=True)
    return os.path.join(result_folder_path, "1_pre_processing.las")

def get_pre_processing_index_json_path(result_folder_path: Union[str, None]= None) -> str:
    if result_folder_path is None:
        result_folder_path = init_result_folder_path(use_existing_folder=True)
    return os.path.join(result_folder_path, "1_pre_processing_index.json")

def get_pipeline_output_las_path(result_folder_path: Union[str, None]= None) -> str:
    if result_folder_path is None:
        result_folder_path = init_result_folder_path(use_existing_folder=True)
    return os.path.join(result_folder_path, "2_pipeline.las")

def path_for_identified_objects_yaml(result_folder_path: str) -> str:
    """Path to identified_objects.yml under the run output folder."""
    return identified_objects_yaml_path(result_folder_path)

def create_pipeline_yaml_file(pipeline_data: dict, result_folder_path: Union[str, None]= None) -> str:
    if result_folder_path is None:
        result_folder_path = init_result_folder_path(use_existing_folder=True)
    """Write process pipeline (kind and stages with params) to pipeline.yml."""
    path = os.path.join(result_folder_path, "2_pipeline.yml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(pipeline_data, f, default_flow_style=False, sort_keys=False)
    log.info("Wrote pipeline YAML to: %s", path)
    return path

def create_las_output_file(cfg: config.SegmentorConfig, output_las_path: str) -> LasWrapper:
    """Initialize output LAS file and return writer wrapper."""
    las_writer = LasWrapper(cfg)
    las_writer.initialize_output_file(output_las_path)
    return las_writer
