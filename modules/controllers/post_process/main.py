import os

import numpy as np

from modules.models import output_tools
from modules.models.post_processing_core import (
    build_dbscan_preview_pcd,
    build_init_ground_preview_pcd,
    copy_merged_attributes,
    merge_chunks,
    run_dbscan_on_merged,
    run_post_process_pipeline,
)
from modules.utils.config import SegmentorConfig
from modules.utils.log_utils import get_dual_logger

log = get_dual_logger(__name__)


class PostProcessingController:
    """Loads merged `2_pipeline.las`, previews INIT/GROUND and DBSCAN, runs full post pipeline on confirm."""

    def __init__(self):
        self.config = SegmentorConfig.get_instance()
        self._result_folder = output_tools.init_result_folder_path(use_existing_folder=True)
        self._pipeline_las_path = output_tools.get_pipeline_output_las_path(self._result_folder)
        self._output_las_path = output_tools.get_post_process_output_las_path(self._result_folder)
        self._baseline_xyz: np.ndarray | None = None
        self._baseline_attrs: dict[str, np.ndarray] | None = None

    def get_pipeline_las_path(self) -> str:
        return self._pipeline_las_path

    def get_output_las_path(self) -> str:
        return self._output_las_path

    def get_result_folder(self) -> str:
        return self._result_folder

    def load_merged_data(self) -> None:
        if not os.path.isfile(self._pipeline_las_path):
            raise FileNotFoundError(f"Pipeline output not found: {self._pipeline_las_path}")
        xyz, attrs = merge_chunks(
            input_path=self._pipeline_las_path,
            chunk_size=int(self.config.chunk_size),
        )
        self._baseline_xyz = np.asarray(xyz, dtype=np.float64, copy=True)
        self._baseline_attrs = copy_merged_attributes(attrs)
        log.info("Loaded merged pipeline LAS: %s points", len(self._baseline_xyz))

    def has_loaded_data(self) -> bool:
        return self._baseline_xyz is not None and self._baseline_attrs is not None

    def get_preview_pcd_initial(self):
        if not self.has_loaded_data():
            raise RuntimeError("Call load_merged_data() first.")
        assert self._baseline_xyz is not None and self._baseline_attrs is not None
        return build_init_ground_preview_pcd(self._baseline_xyz, self._baseline_attrs)

    def run_dbscan_preview(self, eps: float, min_points: int, rng=None):
        if not self.has_loaded_data():
            raise RuntimeError("Call load_merged_data() first.")
        assert self._baseline_xyz is not None and self._baseline_attrs is not None
        working = copy_merged_attributes(self._baseline_attrs)
        _pcd, working = run_dbscan_on_merged(
            self._baseline_xyz,
            working,
            eps=float(eps),
            min_points=int(min_points),
        )
        return build_dbscan_preview_pcd(self._baseline_xyz, working, rng=rng)

    def confirm_post_processing(self, eps: float, min_points: int) -> tuple[int, str, str]:
        """
        DBSCAN + VLM + write `3_post_process.las`. Returns (point_count, output_las, objects_yml path).
        """
        original = os.path.abspath(str(getattr(self.config, "input_file", "")).strip())
        if not original or not os.path.isfile(original):
            raise ValueError(
                "config input_file must point to the original LAS used for restoration."
            )

        obj_cfg = getattr(self.config, "objects_segmentation", {}) or {}
        ollama_model = str(obj_cfg.get("ollama_model", "gemma3:4b"))
        vlm_traces_subdir = str(obj_cfg.get("vlm_traces_subdir", "vlm_object_traces"))

        xyz, _attrs = run_post_process_pipeline(
            input_path=self._pipeline_las_path,
            output_path=self._output_las_path,
            objects_folder_path=self._result_folder,
            chunk_size=int(self.config.chunk_size),
            eps=float(eps),
            min_points=int(min_points),
            ollama_model=ollama_model,
            vlm_traces_subdir=vlm_traces_subdir,
            original_input_path=original,
        )
        from modules.models.ai_object_detector import identified_objects_yaml_path

        yml = identified_objects_yaml_path(self._result_folder)
        return len(xyz), self._output_las_path, yml
