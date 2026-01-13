# ==== Change ====
# The path for your python executable/environment
PYTHON_PATH=/home/mohammed/miniconda3/envs/segmentor/bin/python
# The path for your input file
INPUT_FILE=/mnt/c/Users/Mohammed/Workspace/scans/office_cleaned.las
# The path for your output file
OUTPUT_FILE=/mnt/c/Users/Mohammed/Workspace/scans/office_cleaned_output.las

# ==== Do not change ====
PYTHON_FILE=/home/mohammed/workspace/segmentor/main.py

$PYTHON_PATH $PYTHON_FILE $INPUT_FILE $OUTPUT_FILE