# Examples: LasWrapper2 regression tests

## Minimal LAS + config in `tmp_path`

```python
import numpy as np
import laspy
import pytest
from modules.models.las_wrapper_2 import LasWrapper2
from modules.utils.config import SegmentorConfig


def _write_tiny_las(path, n_points: int = 100, chunk_size: int = 30):
    las = laspy.create(point_format=3, file_version="1.4")
    las.x = np.random.randn(n_points)
    las.y = np.random.randn(n_points)
    las.z = np.random.randn(n_points)
    las.classification = np.zeros(n_points, dtype=np.uint8)
    las.write(path)


def _write_minimal_config(tmp_path, input_las, chunk_size: int, reset_classification: bool):
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        f"input_file: {input_las.as_posix()!r}\n"
        f"chunk_size: {chunk_size}\n"
        f"reset_classification: {reset_classification}\n",
        encoding="utf-8",
    )
    return cfg_path


@pytest.fixture
def wrapper(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    las_path = tmp_path / "in.las"
    _write_tiny_las(las_path, n_points=100, chunk_size=30)
    cfg = SegmentorConfig(str(_write_minimal_config(tmp_path, las_path, 30, True)))
    return LasWrapper2(cfg)
```

## Snapshot isolation

Always `monkeypatch.chdir(tmp_path)` before `save_snapshot` / `restore_snapshot` / `get_snapshot` / `delete_snapshot` so `.snapshot.las` stays inside the test temp dir.

## Example: `get_current_chunk` error

```python
def test_get_current_chunk_requires_navigation(wrapper):
    with pytest.raises(RuntimeError, match="not initialized"):
        wrapper.get_current_chunk()
    assert wrapper.iterate_chunk(0) is True
    pcd, attrs = wrapper.get_current_chunk()
    assert len(attrs["classification"]) == len(pcd.points)
```

Adjust YAML field names to match what `SegmentorConfig` actually loads from your real `config.yml` if the minimal file omits required keys.
