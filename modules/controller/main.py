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
    params: Dict[str, object] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class Pipeline:
    stages: List[PipelineStage] = dataclasses.field(default_factory=list)
    current_stage_index: int = -1  # starting from -1 to handle iteration
    kind: str = "setup"

class Controller:
    def __init__(self, config: SegmentorConfig):
        self.config = config
        self.las_wrapper = LasWrapper(config)
        # Setup pipeline works on the current preview chunk
        self.setup_pipeline = self._create_setup_pipeline()
        # Alias used by existing viewer code
        self.pipeline = self.setup_pipeline
        # Will be populated later when building the full processing pipeline
        self.process_pipeline: Pipeline | None = None

    def _create_setup_pipeline(self) -> Pipeline:
        """
        Build the interactive (setup) pipeline used by the GUI on a preview chunk.
        """
        new_pipeline = Pipeline(kind="setup")

        if getattr(self.config, "filter_noise", False):
            new_pipeline.stages.append(self._make_filter_noise_stage())

        ground_cfg = getattr(self.config, "ground_segmentation", {})
        if ground_cfg.get("enable", False):
            new_pipeline.stages.append(self._make_ground_segmentation_stage())

        wall_cfg = getattr(self.config, "wall_segmentation", {})
        if wall_cfg.get("enable", False):
            new_pipeline.stages.append(self._make_wall_segmentation_stage())
            if wall_cfg.get("filter_noise_after_segmenetation", False):
                new_pipeline.stages.append(self._make_post_wall_noise_filter_stage())

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
                self.las_wrapper.segment_ground_with_params,
                **kwargs,
            ),
            params=params,
        )

    def _make_wall_segmentation_stage(self) -> PipelineStage:
        # Defaults mirror the algorithm defaults in pcd_tools.apply_ransac_wall_detection.
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
                self.las_wrapper.segment_vertical_surfaces_with_params,
                **kwargs,
            ),
            params=params,
        )

    def _run_filter_noise_for_unclassified(self, **kwargs):
        """
        Helper used in the setup pipeline to filter noise only on unclassified points.
        Uses a dummy index (0) since we only operate on the preview chunk here.
        """
        pcd, points_attributes = self.las_wrapper.get_current_chunk()
        pcd, points_attributes = self.las_wrapper.filter_noise_for_none_classified_points(
            index=0,
            pcd=pcd,
            points_attributes=points_attributes,
            **kwargs,
        )
        self.las_wrapper.set_current_chunk(pcd, points_attributes)

    def _make_post_wall_noise_filter_stage(self) -> PipelineStage:
        # Reuse the same defaults as the initial noise filter.
        default_nb_neighbors = 20
        default_std_ratio = 2.0
        return PipelineStage(
            name="Post-wall Noise Filter",
            func=lambda **kwargs: self._run_filter_noise_for_unclassified(**kwargs),
            params={
                "nb_neighbors": default_nb_neighbors,
                "std_ratio": default_std_ratio,
            },
        )

    # ---- Process pipeline helpers ----

    def build_process_pipeline_from_setup(self) -> Pipeline:
        """
        Clone the current setup pipeline into a process pipeline that operates
        on arbitrary chunks (pcd, points_attributes) instead of the preview chunk.
        """
        process_pipeline = Pipeline(kind="process")

        for setup_stage in self.setup_pipeline.stages:
            process_pipeline.stages.append(self._make_process_stage(setup_stage))

        self.process_pipeline = process_pipeline
        return process_pipeline

    def _make_process_stage(self, setup_stage: PipelineStage) -> PipelineStage:
        """
        Map a setup stage to its processing equivalent while freezing parameters.
        """
        name = setup_stage.name
        params = dict(setup_stage.params)

        if name == "Filter Noise":
            func = self._process_filter_noise_stage
        elif name == "Ground Segmentation":
            func = self._process_ground_segmentation_stage
        elif name == "Wall Segmentation":
            func = self._process_wall_segmentation_stage
        elif name == "Post-wall Noise Filter":
            func = self._process_post_wall_noise_filter_stage
        else:
            # Fallback: reuse the same callable, assuming it accepts
            # (pcd, points_attributes, **params) for processing as well.
            func = setup_stage.func

        return PipelineStage(name=name, func=func, params=params)

    def _process_filter_noise_stage(self, pcd, points_attributes, **kwargs):
        return pcd_tools.filter_noise(pcd, points_attributes, **kwargs)

    def _process_ground_segmentation_stage(self, pcd, points_attributes, **kwargs):
        return self.las_wrapper.segment_ground_with_params(
            pcd,
            points_attributes,
            **kwargs,
        )

    def _process_wall_segmentation_stage(self, pcd, points_attributes, **kwargs):
        return self.las_wrapper.segment_vertical_surfaces_with_params(
            pcd,
            points_attributes,
            **kwargs,
        )

    def _process_post_wall_noise_filter_stage(self, pcd, points_attributes, index=None, **kwargs):
        # Use the real chunk index when provided; otherwise default to 0.
        effective_index = 0 if index is None else index
        return self.las_wrapper.filter_noise_for_none_classified_points(
            effective_index,
            pcd,
            points_attributes,
            **kwargs,
        )

    def run_process_pipeline_on_all_chunks(self):
        """
        Execute the finalized process pipeline on every input chunk.
        """
        if self.process_pipeline is None:
            self.build_process_pipeline_from_setup()

        if self.process_pipeline is None or not self.process_pipeline.stages:
            return

        # Initialize output file once if writing results.
        if self.config.result.get("write_to_file"):
            self.las_wrapper.initialize_output_file()

        for index, pcd, points_attributes in self.las_wrapper.iter_input_chunks():
            for stage in self.process_pipeline.stages:
                stage_kwargs = dict(stage.params)
                # Provide chunk index for stages that care about it.
                stage_kwargs.setdefault("index", index)
                pcd, points_attributes = stage.func(
                    pcd=pcd,
                    points_attributes=points_attributes,
                    **stage_kwargs,
                )

            if self.config.result.get("display_result"):
                self.las_wrapper.display_pcb(pcd, points_attributes, colors="classification")

            if self.config.result.get("write_to_file"):
                self.las_wrapper.write_pcb_chunk_to_file(index, pcd, points_attributes)

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
        if current_stage is None:
            return
        # Merge stored params with any explicit overrides
        call_kwargs = {**current_stage.params, **kwargs}
        current_stage.func(**call_kwargs)


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