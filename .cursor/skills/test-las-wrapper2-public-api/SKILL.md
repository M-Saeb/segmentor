---
name: test-las-wrapper2-public-api
description: Defines the public API surface of LasWrapper2 and a regression test workflow before refactoring modules/models/las_wrapper_2.py. Use when refactoring LasWrapper2, adding tests for las_wrapper_2, or verifying backward compatibility of its initializer and public methods.
---

# LasWrapper2 public API regression tests

## When to apply

Run this workflow **before and after** refactors of `modules/models/las_wrapper_2.py` so behavior of the **public** surface stays pinned.

## Python environment

Run pytest and any scripts with the project interpreter:

`/home/mohammedalsalihi/miniconda3/envs/segmentor/bin/python`

From repo root, typical invocation:

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/mohammedalsalihi/miniconda3/envs/segmentor/bin/python -m pytest tests/test_las_wrapper2_public_api.py -v`

(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids broken ROS/Ament pytest entry points that may be on `PYTHONPATH`.)

## Public API inventory (current)

Treat **only** names without a leading underscore as the stable contract unless the refactor explicitly changes the API.

**Class:** `LasWrapper2`

| Symbol | Signature / role |
|--------|------------------|
| `__init__` | `(config: modules.utils.config.SegmentorConfig)` — stores `config`, sets `current_chunk = None`, `_chunk_index = 0`, `_total_chunks = None` |
| `genrate_unique_id` | `() -> str` — SHA-256 hex from serialized original LAS header fields (**name includes the typo** `genrate`). Key order follows `header.__dict__.keys()`, so the hash may vary between calls if that order is not stable. |
| `iterate_chunk` | `(delta: int) -> bool` — updates `_chunk_index` and `current_chunk`; returns `False` if out of range or no points |
| `get_current_chunk` | `() -> (pcd, points_attributes)` — **raises** `RuntimeError` if `current_chunk` was never set via `iterate_chunk` / `set_current_chunk` |
| `set_current_chunk` | `(pcd, points_attributes) -> None` |
| `save_snapshot` | `() -> None` — writes **`.snapshot.las`** in the **process current working directory** |
| `restore_snapshot` | `() -> None` — reads `.snapshot.las`; if missing, calls `save_snapshot()` first |
| `get_snapshot` | `() -> (pcd, points_attributes)` — reads `.snapshot.las` (**expects file to exist**; no create path) |
| `delete_snapshot` | `() -> None` — removes `.snapshot.las` if present |
| `iter_input_chunks` | `() -> iterator` yielding `(index, pcd, points_attributes)` for each input chunk |
| `initialize_output_file` | `(full_path: str) -> None` — creates/truncates output LAS with header from input |
| `write_pcb_chunk_to_file` | `(index, pcd, points_attributes, full_path) -> None` — appends chunk; skips `points_attributes` key `object_cluster_id` |

**Instance attributes callers may rely on:** `config`, `current_chunk` (and navigation state only if tests document intended behavior: `_chunk_index`, `_total_chunks`).

**Config fields used by this module:** `input_file`, `chunk_size`, `reset_classification` (with `modules.utils.constants.PCB_CLASSIFICATION_INIT` when reset is on).

## Fixture strategy

1. **Config:** `SegmentorConfig` requires a real YAML path (`FileNotFoundError` if missing). In tests, write a **minimal YAML** under `tmp_path` (only keys you need: e.g. `input_file`, `chunk_size`, `reset_classification`).
2. **LAS input:** Point `input_file` at a **small real `.las`** created under `tmp_path` (e.g. a few points via `laspy`) so `laspy.open` and chunk iteration succeed.
3. **Snapshots:** Because paths are **relative** (`.snapshot.las`), use `monkeypatch.chdir(tmp_path)` (or equivalent) so tests do not touch the repo root.
4. **Output files:** Use paths under `tmp_path` for `initialize_output_file` / `write_pcb_chunk_to_file`.

## Test checklist (minimum coverage)

Copy and track:

```text
LasWrapper2 regression checklist:
- [ ] __init__: config attached; current_chunk is None; chunk index state initialized
- [ ] genrate_unique_id: 64-char hex string; parses as hex (stability across calls depends on header `__dict__` key order)
- [ ] iterate_chunk(0): loads first chunk when points exist; False when empty / out of range
- [ ] iterate_chunk(+/-1): boundary behavior at first/last chunk
- [ ] get_current_chunk: RuntimeError before navigation/set; matches last successful load/set
- [ ] set_current_chunk: round-trip via get_current_chunk
- [ ] save_snapshot + restore_snapshot + get_snapshot: consistency of geometry and non-xyz dimensions (with chdir isolation)
- [ ] delete_snapshot: file removed; idempotent when missing
- [ ] iter_input_chunks: count matches `_compute_total_chunks` semantics (same chunk boundaries as `iterate_chunk`)
- [ ] initialize_output_file + write_pcb_chunk_to_file: readable LAS; `object_cluster_id` not written to LAS
```

## Assertions worth encoding

- **`reset_classification True`:** after loading a chunk, `points_attributes['classification']` exists and matches `constants.PCB_CLASSIFICATION_INIT` shape/dtype expectations for all points in that chunk.
- **`iterate_chunk`:** when `chunk_size <= 0`, wrapper treats as single chunk (see `_compute_total_chunks`).
- **`write_pcb_chunk_to_file`:** open written file with `laspy.open`; point count for appended chunks matches input chunk sizes.

## Examples

For a minimal LAS fixture, `SegmentorConfig` YAML, and pytest patterns, see [examples.md](examples.md).
