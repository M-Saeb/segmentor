# Segmentor

## Installation

### 1. Prerequisite

Please install conda beforehand and activate it. [link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

### 2. Setup Environment

Run the commands
```bash
conda env create -f environment.yml

conda activate segmentor
```

## Run the program

The main GUI entrypoint is `main.py`, which creates the controller/viewer pair
and drives the Open3D application. Before running, set up the `run.sh` helper
script as follows.

In `run.sh` set up the following variables:
1. `PYTHON_PATH`
2. `INPUT_FILE`
3. `OUTPUT_FILE`

Once you setup these variables, run the command:

```bash
bash run.sh
```

## Architecture overview

The project follows a loose MVC-style structure:

- **Models** (`modules/models/`): low-level LAS I/O (`las_wrapper.py`),
  point-cloud processing and classification (`pcd_tools.py`), and result
  generation (`output_tools.py`).
- **Controller** (`modules/controller/`): builds the interactive *setup*
  pipeline, clones it into a *process* pipeline, iterates over input chunks,
  and delegates preview generation to a dedicated service.
- **Views / UI** (`modules/viewer/`): Open3D GUI window, control panel, and
  per-stage parameter layouts under `modules/viewer/input_layout/`.


Configuration flows from `config.yml` into `SegmentorConfig` in
`modules/utils/config.py`, which provides defaults that are overridden by YAML
values. The controller reads this config to construct both pipelines and to
parameterize the processing functions in `pcd_tools.py`.