#!/usr/bin/env python3
"""
Run the segmentor process pipeline from a YAML file (same shape as
``Pipeline.to_dict()`` / ``example_process_pipeline.yml``).

Application paths, chunk size, and result folder come from the main config
(``config.yml`` by default). The pipeline file defines stage order and
per-stage parameters.

Run from the project root so ``modules`` imports resolve, e.g.:

    python scripts/run_pipeline_from_file.py example_process_pipeline.yml
    python scripts/run_pipeline_from_file.py my_pipeline.yml -c config.yml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.controllers.pipeline.main import PipelineController
from modules.utils.config import SegmentorConfig
from modules.utils.log_utils import get_dual_logger

log = get_dual_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run process pipeline from a YAML definition using SegmentorConfig for I/O."
    )
    parser.add_argument(
        "pipeline_yml",
        type=str,
        help="Path to process pipeline YAML (kind: process, stages: ...)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yml",
        help="Segmentor config YAML (input LAS, chunk_size, result paths). Default: config.yml",
    )
    args = parser.parse_args()

    pipeline_path = os.path.abspath(args.pipeline_yml)
    if not os.path.isfile(pipeline_path):
        log.error("Pipeline file not found: %s", pipeline_path)
        return 1

    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        log.error("Config file not found: %s", config_path)
        return 1

    with open(pipeline_path, "r") as f:
        pipeline_data = yaml.safe_load(f)
    if not isinstance(pipeline_data, dict):
        log.error("Pipeline YAML must load to a mapping at the top level")
        return 1

    SegmentorConfig.initialize(config_path)
    controller = PipelineController(require_setup_pipeline=False)
    try:
        controller.load_process_pipeline_from_dict(pipeline_data)
    except ValueError as e:
        log.error("%s", e)
        return 1

    controller.run_process_pipeline_on_all_chunks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
