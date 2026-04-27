---
name: post-process-e2e-tests
description: >-
  Runs and diagnoses PostProcessingController end-to-end tests in this repo.
  Use when testing post-processing, validating PostProcessingController or
  run_post_process_pipeline, debugging failing post-process tests, or when the
  user asks for post-process E2E, tests/test_post_process_e2e.py, or
  3_post_process.las.
---

# Post-process E2E tests (segmentor)

## What to run

From the repository root:

```bash
python -m pytest tests/test_post_process_e2e.py -v --tb=long
```

Use this workspace’s Segmentor conda interpreter when running commands (see `.cursor/rules/python_env_path.mdc` if present).

### Files involved

| Role | Path |
|------|------|
| Tests | [`tests/test_post_process_e2e.py`](tests/test_post_process_e2e.py) |
| Pipeline / original inputs | [`tests/fixtures/post_process/2_pipeline.las`](tests/fixtures/post_process/2_pipeline.las), [`original_input.las`](tests/fixtures/post_process/original_input.las) |
| Golden outputs | [`expected_post_process_no_vlm.las`](tests/fixtures/post_process/expected_post_process_no_vlm.las), [`expected_post_process_mock_vlm.las`](tests/fixtures/post_process/expected_post_process_mock_vlm.las) |

[`PostProcessingController.confirm_post_processing`](modules/controllers/post_process/main.py) merges **`2_pipeline.las`**, runs DBSCAN and optionally VLM, writes **`3_post_process.las`** and **`3_objects.yml`**. Post-process output LAS includes the Extra Bytes dimension **`Clusters`** (DBSCAN global cluster ids; 0 = none). Config **`input_file`** must reference the **`original_input.las`** path used for downstream restore/screenshot workflows.

Frozen constants in the test module (`EPS_NO_CLUSTER`, `MIN_POINTS_NO_CLUSTER`, `EPS_CLUSTER`, `MIN_POINTS_CLUSTER`) must stay aligned with how the golden LAS files were generated.

The **`identify_object_from_views`** integration test uses **`unittest.mock.patch`** plus a stub for **`_take_rich_media_screenshot_paths`** so Open3D **OffscreenRenderer** is never used (headless-safe). Tests assert **structural validity** of the mocked VLM dict (`classification`, `ai_label`, non-empty `raw_response`), **not** semantic label content.

## When tests fail: produce a failure report

Do **not** stop at “tests failed.” After collecting pytest output, deliver a short **structured report** that explains *why* the failure likely occurred.

### 1. Capture facts

- Which test failed (`test_post_process_confirm_no_clusters`, `test_post_process_confirm_mocked_vlm_response_valid`).
- Failure location: **LAS** comparison, **`3_objects.yml`**, config/setup, or **mock** assertion.
- Traceback line and message.

### 2. Classify using assertion helpers

**LAS (`_assert_las_numeric_core`):** XYZ `allclose`, then shared dims (`classification`, `intensity`, …) when present on both sides.

**Empty-registry test:** YAML totals zero for cars and other_objects.

**Mocked-VLM test:** Patches must remain applied in import path **`modules.models.post_processing_tools`**. If **`eglInitialize` / segfault** appears, the screenshot stub was bypassed—restore **`_take_rich_media_screenshot_paths`** patch.

| Symptom | Likely cause |
|---------|----------------|
| `FileNotFoundError` for pipeline LAS | Missing **`result/2_pipeline.las`** copy or wrong **`output_folder_path`**. |
| `ValueError` on `input_file` | **`original_input.las`** missing or config path wrong. |
| LAS shape / XYZ mismatch | Chunk size / merge changed; golden outdated after **`write_las_merged`** change. |
| Classification mismatch | DBSCAN params drift vs frozen constants; or golden not regenerated. |
| Mock test: `call_count == 0` | No INIT clusters (fixture/eps/min_points); or RGB missing so VLM loop skips clusters. |
| Registry totals wrong | **`_bucket_key_for_record`** / classification from mock vs expectation. |

### 3. Report template

```markdown
## Post-process E2E test report

### Summary
- **Failed:** <test names>
- **Where:** <LAS / YAML / mocks / runtime>

### Evidence (from pytest)
- <one-line quote>

### Interpretation
- <link symptom to table>

### Suggested next step
- <regenerate goldens; align constants; fix fixture RGB; fix patch target>
```

## Success criteria

- Exit code 0 for `tests/test_post_process_e2e.py`.
- On failure, user receives the **Post-process E2E test report** above, not only raw logs.
