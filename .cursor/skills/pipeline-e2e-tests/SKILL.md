---
name: pipeline-e2e-tests
description: >-
  Runs and diagnoses PipelineController RANSAC end-to-end tests in this repo.
  Use when testing the pipeline process, validating PipelineController changes,
  debugging failing pipeline tests, or when the user asks for pipeline E2E or
  tests/test_pipeline_ransac_e2e.py.
---

# Pipeline E2E tests (segmentor)

## What to run

From the repository root, execute the pipeline RANSAC E2E tests:

```bash
python -m pytest tests/test_pipeline_ransac_e2e.py -v --tb=long
```

Use this workspace’s Segmentor conda interpreter when running commands (see `.cursor/rules/python_env_path.mdc` if present).

### Files involved

| Role | Path |
|------|------|
| Tests | [`tests/test_pipeline_ransac_e2e.py`](tests/test_pipeline_ransac_e2e.py) |
| Shared input fixtures | [`tests/fixtures/pipeline_ransac/1_pre_processing.las`](tests/fixtures/pipeline_ransac/1_pre_processing.las), [`tests/fixtures/pipeline_ransac/1_pre_processing_index.json`](tests/fixtures/pipeline_ransac/1_pre_processing_index.json) |
| Golden outputs | [`tests/fixtures/pipeline_ransac/expected_pipeline_ground.las`](tests/fixtures/pipeline_ransac/expected_pipeline_ground.las), [`expected_pipeline_ceiling.las`](tests/fixtures/pipeline_ransac/expected_pipeline_ceiling.las), [`expected_pipeline_wall.las`](tests/fixtures/pipeline_ransac/expected_pipeline_wall.las) |

The tests exercise **`PipelineController.build_process_pipeline_from_setup()`** and **`run_process_pipeline_on_all_chunks()`** (see [`modules/controllers/pipeline/main.py`](modules/controllers/pipeline/main.py)). Frozen YAML parameters live as constants in the test module (`GROUND_RANSAC_PARAMS`, `CEILING_PARAMS`, `WALL_PARAMS`) and must stay aligned with how the golden LAS files were generated.

After a successful full-chunk run, the controller also writes **CityJSON** to [`{result}/citygml.city.json`](modules/models/output_tools.py) (see [`modules/models/cityjson_export.py`](modules/models/cityjson_export.py)) when `cityjson_export.enable` is true (default in [`SegmentorConfig`](modules/utils/config.py)). The tests assert the file exists and has `type: CityJSON` / `version: "2.0"`. Contract-only tests live in [`tests/test_cityjson_pipeline_export.py`](tests/test_cityjson_pipeline_export.py).

**Full `tests/` suite (all E2E modules):**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v
```

## When tests fail: produce a failure report

Do **not** stop at “tests failed.” After collecting pytest output, deliver a short **structured report** that explains *why* the failure likely occurred.

### 1. Capture facts

- Which test names failed (`test_pipeline_ground_ransac`, `test_pipeline_ceiling_ransac`, `test_pipeline_wall_ransac`).
- Exact assertion traceback: file, line, and assertion message.
- Any warnings (e.g. `RuntimeWarning` from RANSAC / plane fitting in `pcd_tools`).
- Missing or invalid **`citygml.city.json`** after pipeline — check [`write_pipeline_cityjson`](modules/models/cityjson_export.py), `merge_chunks` on `2_pipeline.las`, or disable export via `cityjson_export.enable: false` for isolated LAS-only debugging.

### 2. Classify the failure using `_assert_las_numeric_core` behavior

The tests compare actual `2_pipeline.las` to the matching `expected_pipeline_*.las` using:

1. **XYZ**: `numpy.allclose` — mismatch means point coordinates differ (unexpected; usually writer/iterator/input path bug or wrong output file).
2. **Per-dimension arrays** (`classification`, `intensity`, etc., when present on both sides): `array_equal` — mismatch with matching XYZ usually means **classification (or another dim) drift**.

Map the failure:

| Symptom | Likely cause |
|---------|----------------|
| Shape mismatch | Chunking / index JSON / point count differs from golden. |
| XYZ `allclose` fails | Wrong LAS compared, corrupted run, or non-deterministic geometry change (rare here). |
| Same shape, XYZ pass, **classification** fails | RANSAC / Open3D **non-determinism**, or **golden file out of date** after code/threshold/env change. |
| Iterator / `FileNotFoundError` for index | Missing or mis-copied `1_pre_processing_index.json` under `result/`; `output_folder_path` vs `get_pre_processing_index_json_path()` mismatch. |
| `SegmentorConfig` / singleton errors | Test isolation: ensure autouse reset of `SegmentorConfig` or run single process. |

### 3. Report template (use this shape)

```markdown
## Pipeline E2E test report

### Summary
- **Failed:** <test names>
- **Root symptom:** <XYZ vs classification / import / runtime / missing file>

### Evidence (from pytest)
- <one-line quote of the failing assertion or error>

### Interpretation
- <why this maps to the table above; which stage: ground / ceiling / wall>

### Suggested next step
- <e.g. regenerate goldens once with fixed params; pin open3d; align `GROUND_RANSAC_PARAMS` with goldens; fix production bug in …>
```

### 4. If classification golden drift is suspected

Note (from test module docstring): Open3D `segment_plane` is randomized; goldens may need **refreshing on a known-good tree** after intentional algorithm or dependency updates. Regenerate by running the pipeline with the same fixtures and frozen params as in `tests/test_pipeline_ransac_e2e.py`, then replacing the corresponding `expected_pipeline_*.las`—only after confirming the new behavior is correct.

## Success criteria

- Exit code 0 from pytest for `tests/test_pipeline_ransac_e2e.py`.
- If not, the user receives the **Pipeline E2E test report** above, not only raw logs.
