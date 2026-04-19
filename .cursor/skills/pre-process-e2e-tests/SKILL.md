---
name: pre-process-e2e-tests
description: >-
  Runs and diagnoses PreProcessingController end-to-end tests in this repo.
  Use when testing pre-processing, validating PreProcessingController changes,
  debugging failing preprocess tests, or when the user asks for pre-process E2E,
  tests/test_pre_process_e2e.py, or 1_pre_processing.las fixtures.
---

# Pre-process E2E tests (segmentor)

## What to run

From the repository root:

```bash
python -m pytest tests/test_pre_process_e2e.py -v --tb=long
```

Use this workspace’s Segmentor conda interpreter when running commands (see `.cursor/rules/python_env_path.mdc` if present).

### Files involved

| Role | Path |
|------|------|
| Tests | [`tests/test_pre_process_e2e.py`](tests/test_pre_process_e2e.py) |
| Input LAS fixture | [`tests/fixtures/pre_process/input_minimal.las`](tests/fixtures/pre_process/input_minimal.las) |
| Golden LAS | [`tests/fixtures/pre_process/expected_preprocess_output.las`](tests/fixtures/pre_process/expected_preprocess_output.las) |
| Golden index JSON | [`tests/fixtures/pre_process/expected_preprocess_index.json`](tests/fixtures/pre_process/expected_preprocess_index.json) |

Both tests call **`PreProcessingController`** ([`modules/controllers/pre_process/main.py`](modules/controllers/pre_process/main.py)) and expect outputs **`1_pre_processing.las`** and **`1_pre_processing_index.json`** under the result folder.

### Frozen parameters (`RUN_PARAMS`)

The dict in `tests/test_pre_process_e2e.py` must stay aligned with how the golden files were produced:

- `input_chunk_size`, `output_chunk_size`
- `filter_noise` (`nb_neighbors`, `std_ratio`)
- `downsample` (`voxel_size`)

Changing `RUN_PARAMS` without regenerating **`expected_preprocess_output.las`** and **`expected_preprocess_index.json`** will break assertions.

### Test variants

| Test | Entry point |
|------|-------------|
| `test_preprocess_from_run_function` | `PreProcessingController().run(**RUN_PARAMS)` |
| `test_preprocess_run_from_user_input` | `PreProcessingController().run_from_user_input(user_input)` with stringified chunk sizes |

Both should produce identical outputs for the same params; if one passes and the other fails, compare code paths in `run` vs `run_from_user_input`.

## When tests fail: produce a failure report

Do **not** stop at “tests failed.” After collecting pytest output, deliver a short **structured report** that explains *why* the failure likely occurred.

### 1. Capture facts

- Which test names failed.
- Whether failure is in **LAS assertion**, **index JSON assertion**, or **before** (missing file, exception from controller).
- Exact traceback line and message.

### 2. Classify the failure

**LAS (`_assert_las_numeric_core`):**

1. **XYZ** `numpy.allclose` — coordinate mismatch after preprocess (downsample / noise filter / ordering changed).
2. **Preferred dimensions** (`classification`, `intensity`, `return_number`, `number_of_returns`) — bitwise mismatch with same XYZ (attribute pipeline or LAS header dims changed).

**Index (`_assert_index_payload`):**

- Requires keys exactly: `target_points_per_tile`, `num_seeds`, `chunks`.
- Equality is **deep** on `chunks` — tiling / chunking logic or seeding changed.

| Symptom | Likely cause |
|---------|----------------|
| LAS shape differs | Different point count after voxel / filter; input fixture path wrong. |
| LAS XYZ fails, shape matches | Floating drift beyond `atol=1e-8` (algorithm or dependency); or golden outdated. |
| LAS dims fail, XYZ passes | Classification or extra dims written differently. |
| Index top-level keys wrong | Output schema change in index writer. |
| `chunks` / `num_seeds` mismatch | Chunk size params or tiling algorithm changed vs golden. |
| Only `run_from_user_input` fails | Type coercion / defaults differ from `run` (string vs int for chunk sizes). |
| `SegmentorConfig` errors | Singleton not reset between tests (autouse fixture should handle in this file). |

### 3. Report template (use this shape)

```markdown
## Pre-process E2E test report

### Summary
- **Failed:** <test names>
- **Where:** <LAS assertion / index assertion / runtime before assert>

### Evidence (from pytest)
- <one-line quote of the failing assertion or error>

### Interpretation
- <LAS vs index; which dimension or JSON field if known>

### Suggested next step
- <e.g. regenerate goldens after intentional preprocess change; fix param typing in run_from_user_input; align RUN_PARAMS with goldens>
```

### 4. If golden refresh is needed

After **intentional** changes to preprocess, noise filtering, downsampling, or indexing:

1. Run a single successful pass (or script) with the **new** desired `RUN_PARAMS` and the same `input_minimal.las`.
2. Replace **`expected_preprocess_output.las`** and **`expected_preprocess_index.json`** only if the new outputs are correct by design.
3. Update **`RUN_PARAMS`** in the test file to document the frozen values used for regeneration.

## Success criteria

- Exit code 0 from pytest for `tests/test_pre_process_e2e.py`.
- If not, the user receives the **Pre-process E2E test report** above, not only raw logs.
