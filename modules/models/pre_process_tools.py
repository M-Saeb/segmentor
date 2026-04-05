import os

import tempfile
from .las_tools import las_io

def get_pre_processed_data_folder_path(input_las_path: str) -> str:
    unique_id = las_io.get_las_file_uuid(input_las_path)
    tmp_root = tempfile.gettempdir()
    tmp_folder_path = os.path.join(tmp_root, f"segmentor_{unique_id}")
    os.makedirs(tmp_folder_path, exist_ok=True)
    return tmp_folder_path
