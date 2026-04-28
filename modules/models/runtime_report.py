"""
Singleton runtime analytics for post-processing (viewer path).

Accumulates total image-creation time and VLM response time across clusters;
writes ``3_runtime_report.yml`` to the result folder.
"""

from __future__ import annotations

import os

import yaml

RUNTIME_REPORT_FILENAME = "3_runtime_report.yml"


def _seconds_to_hms(total_seconds: float) -> dict[str, int]:
    s = int(round(max(0.0, float(total_seconds))))
    hours, rem = divmod(s, 3600)
    minute, seconds = divmod(rem, 60)
    return {
        "hours": int(hours),
        "minutes": int(minute),
        "seconds": int(seconds),
        "total_seconds": int(s),
    }


class RuntimeReport:
    """One instance per process; holds cumulative timing for post-process analytics."""

    _instance: RuntimeReport | None = None

    def __new__(cls) -> RuntimeReport:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._image_creation_seconds: float = 0.0
        self._vlm_response_seconds: float = 0.0
        self._initialized = True

    @classmethod
    def get_instance(cls) -> RuntimeReport:
        if cls._instance is None:
            cls()
        assert cls._instance is not None
        return cls._instance

    def reset(self) -> None:
        self._image_creation_seconds = 0.0
        self._vlm_response_seconds = 0.0

    def add_image_creation_seconds(self, seconds: float) -> None:
        self._image_creation_seconds += float(seconds)

    def add_vlm_response_seconds(self, seconds: float) -> None:
        self._vlm_response_seconds += float(seconds)

    def to_yaml_payload(self) -> dict:
        return {
            "ImageCreationDuration": _seconds_to_hms(self._image_creation_seconds),
            "VLMResponseDuration": _seconds_to_hms(self._vlm_response_seconds),
            "TotalDuration": _seconds_to_hms(self._image_creation_seconds + self._vlm_response_seconds),
        }

    def write_yaml(self, output_folder_path: str) -> None:
        path = os.path.join(output_folder_path, RUNTIME_REPORT_FILENAME)
        os.makedirs(output_folder_path, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.to_yaml_payload(),
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
