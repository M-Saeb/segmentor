import json
import os
from typing import Any, Mapping

import numpy as np

from modules.models import pcd_tools
from modules.models import pre_process_tools
from modules.utils.config import SegmentorConfig
from modules.utils.log_utils import get_dual_logger

from .utils import (
    _merge_chunks,
    _order_kdtree_index,
    _reorder_pcd_and_attributes,
    _voxel_downsample_preserve_attributes,
    _write_las,
)

log = get_dual_logger(__name__)


class PreProcessingController:
    def __init__(self, config: SegmentorConfig):
        self.config = config

    def run(
        self,
        *,
        input_chunk_size: int,
        output_chunk_size: int,
        filter_noise: Mapping[str, Any],
        downsample: Mapping[str, Any],
        input_path: str | None = None,
    ) -> str:
        path = input_path if input_path is not None else self.config.input_file

        pcd, merged_attrs = _merge_chunks(path, input_chunk_size)
        log.info("Initial point count: {:,}".format(len(pcd.points)))

        pcd, merged_attrs = pcd_tools.filter_noise(
            pcd,
            merged_attrs,
            nb_neighbors=filter_noise["nb_neighbors"],
            std_ratio=filter_noise["std_ratio"],
        )
        log.info("After statistical outlier removal: {:,}".format(len(pcd.points)))

        pcd, merged_attrs = _voxel_downsample_preserve_attributes(
            pcd,
            merged_attrs,
            voxel_size=downsample["voxel_size"],
        )
        log.info("After voxel downsampling: {:,}".format(len(pcd.points)))

        xyz = np.asarray(pcd.points)
        order, kd_chunks, kd_num_seeds = _order_kdtree_index(
            xyz,
            target_points=output_chunk_size,
            max_depth=64,
        )
        log.info("Reordering points and attributes")
        pcd, merged_attrs = _reorder_pcd_and_attributes(pcd, merged_attrs, order)

        tmp_folder_path = pre_process_tools.get_pre_processed_data_folder_path(path)
        log.info("Storing pre-processed data in: %s", tmp_folder_path)
        output_path = os.path.join(tmp_folder_path, "output.las")
        index_path = os.path.join(tmp_folder_path, "index.json")

        _write_las(
            input_path=path,
            output_path=output_path,
            chunk_size=output_chunk_size,
            pcd=pcd,
            merged_attrs=merged_attrs,
        )
        index_payload = {
            "target_points_per_tile": output_chunk_size,
            "num_seeds": kd_num_seeds,
            "chunks": kd_chunks,
        }
        with open(index_path, "w") as f:
            json.dump(index_payload, f, indent=2)
        log.info("Wrote spatial bin index to %s", index_path)

        return tmp_folder_path

    def run_from_user_input(
        self,
        user_input: Mapping[str, Any],
        *,
        input_path: str | None = None,
    ) -> str:
        return self.run(
            input_chunk_size=int(user_input["input_chunk_size"]),
            output_chunk_size=int(user_input["output_chunk_size"]),
            filter_noise=user_input["filter_noise"],
            downsample=user_input["downsample"],
            input_path=input_path,
        )
