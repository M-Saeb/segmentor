import os
from modules.utils import config
from modules.models.las_wrapper import LasWrapper

class ResultGenerator:
    def __init__(self, config: config.SegmentorConfig):
        self.config = config
        self.las_wrapper = LasWrapper(config)
        self.output_folder_path = self._make_output_folder()
        self.full_las_path = os.path.join(self.output_folder_path, "output.las")

    def reset_las_file(self):
        self.las_wrapper.initialize_output_file(self.full_las_path)

    def _make_output_folder(self):
        output_folder_path = self.config.result["output_folder_path"]
        if not os.path.exists(output_folder_path):
            os.makedirs(
                os.path.dirname(output_folder_path),
                exist_ok=False
            )
        else:
            # create increamental folder name
            increment = 2
            while os.path.exists(output_folder_path + f"_{increment}"):
                increment += 1
            output_folder_path = output_folder_path + f"_{increment}"
            os.makedirs(output_folder_path, exist_ok=False)

        return output_folder_path

    def write_las_chunk_to_file(self, index, pcd, points_attributes):
        self.las_wrapper.write_pcb_chunk_to_file(index, pcd, points_attributes, self.full_las_path)