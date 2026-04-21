from typing import Dict

from modules.utils.config import SegmentorConfig
from modules.utils.pipeline import PipelineStage, Pipeline
from modules.models import LasPipelineWrapper
from modules.models import LasIteratorKDTree
from modules.models import output_tools
from modules.models.cityjson_export import write_pipeline_cityjson
from modules.models import pcd_tools
from modules.utils.log_utils import get_dual_logger
from modules.utils import constants
from .pcd_previewer import PCDPreviewer

log = get_dual_logger(__name__)


class PipelineController:
    def __init__(self, *, require_setup_pipeline: bool = True):
        self.config = SegmentorConfig.get_instance()
        self.las_pipeline_wrapper = LasPipelineWrapper(iterator_class=LasIteratorKDTree)
        if not self.las_pipeline_wrapper.iterate_chunk(0):
            log.warning("Could not initialize LasPipelineWrapper preview chunk; input may be empty.")
        self.output_folder_path = output_tools.init_result_folder_path(use_existing_folder=True)
        self.full_las_path = output_tools.get_pipeline_output_las_path(self.output_folder_path)
        self.output_las_writer = None
        if require_setup_pipeline:
            self.setup_pipeline = self._create_setup_pipeline()
        else:
            self.setup_pipeline = Pipeline(kind="setup")
        # Will be populated later when building the full processing pipeline
        self.process_pipeline: Pipeline | None = None

    def _create_setup_pipeline(self) -> Pipeline:
        """
        Build the interactive ("setup") pipeline used by the GUI on a preview
        chunk.

        The setup pipeline operates on a single preview chunk, allowing the user
        to tune parameters stage-by-stage (ground/ceiling/wall
        segmentation, etc.). Its structure is later cloned into a "process"
        pipeline that is applied to every chunk of the input file.
        """
        new_pipeline = Pipeline(kind="setup")

        ground_cfg = getattr(self.config, "ground_segmentation", {})
        if ground_cfg.get("enable", False):
            new_pipeline.stages.append(self._make_ground_segmentation_stage())

        ceiling_cfg = getattr(self.config, "ceiling_segmentation", {})
        if ceiling_cfg.get("enable", False):
            new_pipeline.stages.append(self._make_ceiling_segmentation_stage())

        wall_cfg = getattr(self.config, "wall_segmentation", {})
        if wall_cfg.get("enable", False):
            new_pipeline.stages.append(self._make_wall_segmentation_stage())

        if len(new_pipeline.stages) == 0:
            raise ValueError("No stages for the setup pipeline were created")

        return new_pipeline

    def _make_ground_segmentation_stage(self) -> PipelineStage:
        ground_cfg = getattr(self.config, "ground_segmentation", {})
        method = ground_cfg.get("method", "basic")
        csf_cfg = ground_cfg.get("csf_params", {})
        ransac_cfg = ground_cfg.get("ransac_params", {})

        params: Dict[str, object] = {"method": method}
        if method == "csf":
            params.update(
                {
                    "threshold_factor": csf_cfg.get("threshold_factor", 0.1),
                    "resolution": csf_cfg.get("resolution", 1.0),
                    "rigidness": csf_cfg.get("rigidness", 3),
                    "max_iter": csf_cfg.get("max_iter", 500),
                }
            )
        elif method == "ransac":
            params.update(
                {
                    "distance_threshold": ransac_cfg.get("distance_threshold", 0.05),
                    "num_iterations": int(ransac_cfg.get("num_iterations", 1000)),
                }
            )

        return PipelineStage(
            name=constants.PIPELINE_GROUND_SEGMENTATION,
            func=lambda **kwargs: self._pipeline_functions_wrapper(
                pcd_tools.segment_ground,
                **kwargs,
            ),
            params=params,
        )

    def _make_ceiling_segmentation_stage(self) -> PipelineStage:
        ceiling_cfg = getattr(self.config, "ceiling_segmentation", {})
        params: Dict[str, object] = {
            "facing_up_threshold": float(ceiling_cfg.get("facing_up_threshold", 0.95)),
            "min_inlier_density": float(ceiling_cfg.get("min_inlier_density", 15)),
        }

        return PipelineStage(
            name=constants.PIPELINE_CEILING_SEGMENTATION,
            func=lambda **kwargs: self._pipeline_functions_wrapper(
                pcd_tools.apply_ceiling_segmentation,
                **kwargs,
            ),
            params=params,
        )

    def _make_wall_segmentation_stage(self) -> PipelineStage:
        wall_cfg = getattr(self.config, "wall_segmentation", {})
        params: Dict[str, object] = {
            "rotation_threshold": float(wall_cfg.get("rotation_threshold", 0.9)),
        }

        return PipelineStage(
            name=constants.PIPELINE_WALL_SEGMENTATION,
            func=lambda **kwargs: self._pipeline_functions_wrapper(
                pcd_tools.apply_wall_segmentation,
                **kwargs,
            ),
            params=params,
        )

    def switch_to_next_setup_pipeline_stage(self):
        self.setup_pipeline.current_stage_index += 1
        current_stage = self.get_current_setup_pipeline_stage()
        if current_stage is None:
            # No more stages; clear any previous snapshot.
            self.las_pipeline_wrapper.delete_snapshot()
            return

        # Snapshot the current preview chunk as the immutable input state
        # for all preview runs of this stage.
        self.las_pipeline_wrapper.save_snapshot()

    def get_current_setup_pipeline_stage(self):
        if self.setup_pipeline.current_stage_index >= len(self.setup_pipeline.stages):
            return None
        return self.setup_pipeline.stages[self.setup_pipeline.current_stage_index]

    def iterate_preview_chunk(self, delta: int) -> bool:
        """
        Move the preview to another input chunk (forward or backward).

        On success, clears the stage snapshot and resets the setup pipeline to
        the first stage, since preview state is tied to the current chunk.
        """
        if not self.las_pipeline_wrapper.iterate_chunk(delta):
            return False
        self.las_pipeline_wrapper.delete_snapshot()
        self.setup_pipeline.current_stage_index = 0
        return True

    def set_flag_for_current_setup_stage(self, flag: bool):
        current_stage = self.get_current_setup_pipeline_stage()
        if current_stage is None:
            return
        current_stage.enabled = flag
        log.info(f"Set flag for stage {current_stage.name} to {flag}")

    def run_current_stage_of_setup_pipeline(self, **kwargs):
        current_stage = self.get_current_setup_pipeline_stage()
        if current_stage is None:
            return

        # Always start from the snapshotted input for this stage so that
        # repeated previews with different parameters do not stack effects.
        self.las_pipeline_wrapper.restore_snapshot()

        # Merge stored params with any explicit overrides
        call_kwargs = {**current_stage.params, **kwargs}
        current_stage.func(**call_kwargs)

    def confirm_current_stage_and_advance(self):
        """
        Finalize the current stage (keeping the last preview result as the
        global state), then advance to the next stage and initialize its
        input snapshot.
        """
        self.switch_to_next_setup_pipeline_stage()
        return self.get_current_setup_pipeline_stage()

    # ---- Process pipeline helpers ----
    def build_process_pipeline_from_setup(self) -> Pipeline:
        """
        Clone the current setup pipeline into a "process" pipeline that
        operates on arbitrary chunks (pcd, points_attributes) instead of only
        the preview chunk.

        Each setup stage is mapped to a processing-stage callable with its
        parameters frozen at their current values, so that the full input file
        can be processed consistently with the configuration chosen in the GUI.
        """
        process_pipeline = Pipeline(kind="process")

        for setup_stage in self.setup_pipeline.stages:
            # Only include stages that have been explicitly enabled.
            if not getattr(setup_stage, "enabled", False):
                continue
            process_pipeline.stages.append(self._make_process_stage(setup_stage))

        self.process_pipeline = process_pipeline
        return process_pipeline

    def load_process_pipeline_from_dict(self, data: dict) -> None:
        """
        Build `self.process_pipeline` from a serialized process pipeline dict
        (same shape as `Pipeline.to_dict()` / process YAML files).

        Intended for headless runs and future embedding in app config.
        """
        if not data or data.get("kind") != "process":
            raise ValueError('Pipeline data must be a dict with kind: "process"')

        raw_stages = data.get("stages")
        if raw_stages is None:
            raise ValueError("Pipeline data must contain a 'stages' list")

        process_pipeline = Pipeline(kind="process")

        for entry in raw_stages:
            if not isinstance(entry, dict):
                raise ValueError(f"Each stage must be a dict, got {type(entry)!r}")
            if not entry.get("enabled", True):
                continue
            name = entry.get("name")
            if not name or not isinstance(name, str):
                raise ValueError(f"Each enabled stage must have a non-empty string 'name': {entry!r}")
            if name not in constants.KNOWN_PROCESS_STAGE_NAMES:
                raise ValueError(
                    f"Unknown pipeline stage name {name!r}. "
                    f"Supported names: {sorted(constants.KNOWN_PROCESS_STAGE_NAMES)}"
                )
            params = dict(entry.get("params") or {})

            placeholder = PipelineStage(
                name=name,
                func=lambda **kwargs: None,
                params=params,
                enabled=True,
            )
            process_pipeline.stages.append(self._make_process_stage(placeholder))

        if not process_pipeline.stages:
            raise ValueError("No enabled stages in process pipeline data")

        self.process_pipeline = process_pipeline

    def _make_process_stage(self, setup_stage: PipelineStage) -> PipelineStage:
        """
        Map a setup stage to its processing equivalent while freezing parameters.

        The mapping is based on the human-readable stage `name` used in the
        setup pipeline; known names (e.g. "Filter Noise", "Ground Segmentation")
        are translated into dedicated processing helpers, while unknown names
        fall back to using the original callable directly.
        """
        name = setup_stage.name
        params = dict(setup_stage.params)

        if name == constants.PIPELINE_GROUND_SEGMENTATION:
            func = self._process_ground_segmentation_stage
        elif name == constants.PIPELINE_CEILING_SEGMENTATION:
            func = self._process_ceiling_segmentation_stage
        elif name == constants.PIPELINE_WALL_SEGMENTATION:
            func = self._process_wall_segmentation_stage
        else:
            # Fallback: reuse the same callable, assuming it accepts
            # (pcd, points_attributes, **params) for processing as well.
            log.warning(f"Falling back: no process stage for {name} with params {params}")
            func = setup_stage.func

        return PipelineStage(name=name, func=func, params=params, enabled=True)

    def _process_ground_segmentation_stage(self, pcd, points_attributes, **kwargs):
        return pcd_tools.segment_ground(
            pcd,
            points_attributes,
            **kwargs
        )

    def _process_ceiling_segmentation_stage(self, pcd, points_attributes, **kwargs):
        return pcd_tools.apply_ceiling_segmentation(
            pcd,
            points_attributes,
            **kwargs
        )

    def _process_wall_segmentation_stage(self, pcd, points_attributes, **kwargs):
        return pcd_tools.apply_wall_segmentation(
            pcd,
            points_attributes,
            **kwargs
        )

    def run_process_pipeline_on_all_chunks(self):
        """
        Execute the finalized process pipeline on every input chunk.

        This iterates over all LAS chunks, applies each process-stage function
        in order, and writes the resulting classified points out via
        `ResultGenerator`. A YAML description of the pipeline is written once
        before processing begins.

        The LAS iterator is rewound to the first chunk first so preview
        navigation (later chunk indices) does not skip the start of the file.
        """
        if self.process_pipeline is None:
            self.build_process_pipeline_from_setup()

        if self.process_pipeline is None or not self.process_pipeline.stages:
            return

        output_tools.create_pipeline_yaml_file(
            pipeline_data=self.process_pipeline.to_dict(),
            result_folder_path=self.output_folder_path,
        )

        self.output_las_writer = output_tools.create_las_output_file(
            output_las_path=self.full_las_path,
        )

        lw = self.las_pipeline_wrapper
        while lw.iterate_chunk(-1):
            pass

        for index, pcd, points_attributes in lw.iter_input_chunks():
            for stage in self.process_pipeline.stages:
                stage_kwargs = dict(stage.params)
                pcd, points_attributes = stage.func(
                    pcd=pcd,
                    points_attributes=points_attributes,
                    **stage_kwargs,
                )

            self.output_las_writer.write_pcb_chunk_to_file(
                index=index,
                pcd=pcd,
                points_attributes=points_attributes,
                full_path=self.full_las_path
            )
        log.info("===== Finished writing to: %s ======", self.full_las_path)

        export_cfg = getattr(self.config, "cityjson_export", None) or {}
        if export_cfg.get("enable", True):
            try:

                write_pipeline_cityjson(
                    result_folder_path=self.output_folder_path,
                    pipeline_las_path=self.full_las_path,
                    cfg=self.config,
                )
            except Exception:
                log.exception(
                    "CityJSON export failed; pipeline LAS output is still valid."
                )

    def _pipeline_functions_wrapper(self, callback, **kwargs):
        pcd, points_attributes = callback(
            *self.las_pipeline_wrapper.get_current_chunk(),
            **kwargs
        )
        self.las_pipeline_wrapper.set_current_chunk(pcd, points_attributes)

    def get_preview_pcd(self, color="RGB"):
        pcd, points_attributes = self.las_pipeline_wrapper.get_current_chunk()
        previewer = PCDPreviewer(pcd, points_attributes)
        previewer.apply_compression(
            max_points=self.config.previewer["maximum_points"]
        )
        if color == "RGB":
            previewer.apply_rgb_colors()
        elif color == "Classification":
            previewer.apply_classfication_colors()
        return previewer.pcd
