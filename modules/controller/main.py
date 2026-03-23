from typing import Dict, FrozenSet

from modules.utils.config import SegmentorConfig
from modules.utils.pipeline import PipelineStage, Pipeline
from modules.models.las_wrapper import LasWrapper
from modules.models.result_generator import ResultGenerator
from modules.models import pcd_tools
from modules.utils.log_utils import get_dual_logger
from .pcd_previewer import PCDPreviewer

log = get_dual_logger(__name__)

# Stage names that map to dedicated process callables in `_make_process_stage`.
KNOWN_PROCESS_STAGE_NAMES: FrozenSet[str] = frozenset(
    {
        "Filter Noise",
        "Ground Segmentation",
        "Ceiling Segmentation",
        "Wall Segmentation",
        "Post-wall Noise Filter",
        "Objects Segmentation",
    }
)


class Controller:
    def __init__(self, config: SegmentorConfig, *, require_setup_pipeline: bool = True):
        self.config = config
        self.las_wrapper = LasWrapper(config, auto_read_input_file=True)
        self.result_generator = ResultGenerator(config)
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
        to tune parameters stage-by-stage (noise filter, ground/ceiling/wall
        segmentation, object clustering, etc.). Its structure is later cloned into a "process"
        pipeline that is applied to every chunk of the input file.
        """
        new_pipeline = Pipeline(kind="setup")

        if getattr(self.config, "filter_noise", False):
            new_pipeline.stages.append(self._make_filter_noise_stage())

        ground_cfg = getattr(self.config, "ground_segmentation", {})
        if ground_cfg.get("enable", False):
            new_pipeline.stages.append(self._make_ground_segmentation_stage())

        ceiling_cfg = getattr(self.config, "ceiling_segmentation", {})
        if ceiling_cfg.get("enable", False):
            new_pipeline.stages.append(self._make_ceiling_segmentation_stage())

        wall_cfg = getattr(self.config, "wall_segmentation", {})
        if wall_cfg.get("enable", False):
            new_pipeline.stages.append(self._make_wall_segmentation_stage())
            if wall_cfg.get("filter_noise_after_segmenetation", False):
                new_pipeline.stages.append(self._make_post_wall_segmentation_noise_filter_stage())

        objects_cfg = getattr(self.config, "objects_segmentation", {})
        if objects_cfg.get("enable", False):
            new_pipeline.stages.append(self._make_objects_segmentation_stage())

        if len(new_pipeline.stages) == 0:
            raise ValueError("No stages for the setup pipeline were created")

        return new_pipeline

    def _make_filter_noise_stage(self) -> PipelineStage:
        # Defaults chosen for interactive preview; can be adjusted via GUI.
        default_nb_neighbors = 20
        default_std_ratio = 2.0
        return PipelineStage(
            name="Filter Noise",
            func=lambda **kwargs: self._pipeline_functions_wrapper(
                pcd_tools.filter_noise,
                **kwargs,
            ),
            params={
                "nb_neighbors": default_nb_neighbors,
                "std_ratio": default_std_ratio,
            },
        )

    def _make_ground_segmentation_stage(self) -> PipelineStage:
        ground_cfg = getattr(self.config, "ground_segmentation", {})
        method = ground_cfg.get("method", "basic")
        csf_cfg = ground_cfg.get("csf_params", {})

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

        return PipelineStage(
            name="Ground Segmentation",
            func=lambda **kwargs: self._pipeline_functions_wrapper(
                pcd_tools.segment_ground,
                **kwargs,
            ),
            params=params,
        )

    def _make_ceiling_segmentation_stage(self) -> PipelineStage:
        ceiling_cfg = getattr(self.config, "ceiling_segmentation", {})
        params: Dict[str, object] = {
            "epsilon": ceiling_cfg.get("epsilon", 0.01),
            "min_support_points": ceiling_cfg.get("min_support_points", 2000),
            "max_tilt_from_horizontal_deg": ceiling_cfg.get("max_tilt_from_horizontal_deg", 10.0),
            "max_iterations": ceiling_cfg.get("max_iterations", 1000),
            "min_plane_size_ratio": ceiling_cfg.get("min_plane_size_ratio", 0.01),
            "min_height_above_ground": ceiling_cfg.get("min_height_above_ground", 2.0),
            "max_height_above_ground": ceiling_cfg.get("max_height_above_ground", 10.0),
        }

        return PipelineStage(
            name="Ceiling Segmentation",
            func=lambda **kwargs: self._pipeline_functions_wrapper(
                pcd_tools.apply_ceiling_segmentation,
                **kwargs,
            ),
            params=params,
        )

    def _make_wall_segmentation_stage(self) -> PipelineStage:
        # Defaults mirror the algorithm defaults in pcd_tools.apply_wall_segmentation.
        params: Dict[str, object] = {
            "epsilon": 0.01,
            "min_support_points": 2000,
            "max_normal_dev": 10.0,
            "max_iterations": 1000,
            "min_plane_size_ratio": 0.01,
        }

        return PipelineStage(
            name="Wall Segmentation",
            func=lambda **kwargs: self._pipeline_functions_wrapper(
                pcd_tools.apply_wall_segmentation,
                **kwargs,
            ),
            params=params,
        )

    def _make_post_wall_segmentation_noise_filter_stage(self) -> PipelineStage:
        # Reuse the same defaults as the initial noise filter.
        default_nb_neighbors = 20
        default_std_ratio = 2.0
        return PipelineStage(
            name="Post-wall Noise Filter",
            func=lambda **kwargs: self._pipeline_functions_wrapper(
                pcd_tools.filter_noise_for_none_classified_points,
                **kwargs,
            ),
            params={
                "nb_neighbors": default_nb_neighbors,
                "std_ratio": default_std_ratio,
            },
        )

    def _make_objects_segmentation_stage(self) -> PipelineStage:
        obj_cfg = getattr(self.config, "objects_segmentation", {})
        params: Dict[str, object] = {
            "eps": obj_cfg.get("eps", 0.05),
            "min_points": int(obj_cfg.get("min_points", 10)),
            "output_folder_path": self.result_generator.output_folder_path,
            "chunk_index": 0,
            "ollama_model": obj_cfg.get("ollama_model", "gemma3:4b"),
            "vlm_traces_subdir": obj_cfg.get("vlm_traces_subdir", "vlm_object_traces"),
            # Setup preview: DBSCAN only; VLM runs in run_process_pipeline_on_all_chunks only.
            "enable_vlm": False,
        }
        return PipelineStage(
            name="Objects Segmentation",
            func=lambda **kwargs: self._pipeline_functions_wrapper(
                pcd_tools.segment_objects,
                **kwargs,
            ),
            params=params,
        )

    def switch_to_next_setup_pipeline_stage(self):
        self.setup_pipeline.current_stage_index += 1
        current_stage = self.get_current_setup_pipeline_stage()
        if current_stage is None:
            # No more stages; clear any previous snapshot.
            self.las_wrapper.delete_snapshot()
            return

        # Snapshot the current preview chunk as the immutable input state
        # for all preview runs of this stage.
        self.las_wrapper.save_snapshot()

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
        if not self.las_wrapper.iterate_chunk(delta):
            return False
        self.las_wrapper.delete_snapshot()
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
        self.las_wrapper.restore_snapshot()

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
            if name not in KNOWN_PROCESS_STAGE_NAMES:
                raise ValueError(
                    f"Unknown pipeline stage name {name!r}. "
                    f"Supported names: {sorted(KNOWN_PROCESS_STAGE_NAMES)}"
                )
            params = dict(entry.get("params") or {})
            if name == "Objects Segmentation":
                params = self._merge_objects_segmentation_pipeline_params(params)

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

    def _merge_objects_segmentation_pipeline_params(self, params: Dict[str, object]) -> Dict[str, object]:
        obj_cfg = getattr(self.config, "objects_segmentation", {})
        defaults: Dict[str, object] = {
            "eps": obj_cfg.get("eps", 0.05),
            "min_points": int(obj_cfg.get("min_points", 10)),
            "ollama_model": obj_cfg.get("ollama_model", "gemma3:4b"),
            "vlm_traces_subdir": obj_cfg.get("vlm_traces_subdir", "vlm_object_traces"),
        }
        merged = {**defaults, **params}
        merged["output_folder_path"] = self.result_generator.output_folder_path
        merged["chunk_index"] = 0
        return merged

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

        if name == "Filter Noise":
            func = self._process_filter_noise_stage
        elif name == "Ground Segmentation":
            func = self._process_ground_segmentation_stage
        elif name == "Ceiling Segmentation":
            func = self._process_ceiling_segmentation_stage
        elif name == "Wall Segmentation":
            func = self._process_wall_segmentation_stage
        elif name == "Post-wall Noise Filter":
            func = self._process_post_wall_noise_filter_stage
        elif name == "Objects Segmentation":
            func = self._process_objects_segmentation_stage
        else:
            # Fallback: reuse the same callable, assuming it accepts
            # (pcd, points_attributes, **params) for processing as well.
            log.warning(f"Falling back: no process stage for {name} with params {params}")
            func = setup_stage.func

        return PipelineStage(name=name, func=func, params=params, enabled=True)

    def _process_filter_noise_stage(self, pcd, points_attributes, **kwargs):
        return pcd_tools.filter_noise(pcd, points_attributes, **kwargs)

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

    def _process_post_wall_noise_filter_stage(self, pcd, points_attributes, **kwargs):
        return pcd_tools.filter_noise_for_none_classified_points(
            pcd,
            points_attributes,
            **kwargs
        )

    def _process_objects_segmentation_stage(self, pcd, points_attributes, **kwargs):
        kwargs = dict(kwargs)
        kwargs["enable_vlm"] = True
        return pcd_tools.segment_objects(
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
        """
        if self.process_pipeline is None:
            self.build_process_pipeline_from_setup()

        if self.process_pipeline is None or not self.process_pipeline.stages:
            return

        self.result_generator.write_pipeline_yaml(self.process_pipeline.to_dict())

        self.result_generator.reset_las_file()

        for index, pcd, points_attributes in self.las_wrapper.iter_input_chunks():
            for stage in self.process_pipeline.stages:
                stage_kwargs = dict(stage.params)
                if stage.name == "Objects Segmentation":
                    stage_kwargs["chunk_index"] = index
                pcd, points_attributes = stage.func(
                    pcd=pcd,
                    points_attributes=points_attributes,
                    **stage_kwargs,
                )

            self.result_generator.write_las_chunk_to_file(index, pcd, points_attributes)
        log.info("===== Finished writing to: %s ======", self.result_generator.full_las_path)

    def _pipeline_functions_wrapper(self, callback, **kwargs):
        pcd, points_attributes = callback(
            *self.las_wrapper.get_current_chunk(),
            **kwargs
        )
        self.las_wrapper.set_current_chunk(pcd, points_attributes)

    def get_preview_pcd(self, color="RGB"):
        pcd, points_attributes = self.las_wrapper.get_current_chunk()
        previewer = PCDPreviewer(pcd, points_attributes)
        previewer.apply_compression(
            max_points=self.config.previewer["maximum_points"]
        )
        if color == "RGB":
            previewer.apply_rgb_colors()
        elif color == "Classification":
            previewer.apply_classfication_colors()
        return previewer.pcd
