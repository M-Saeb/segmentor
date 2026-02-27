import yaml
import os


class SegmentorConfig:
    """Load configuration from a YAML file and expose as attributes."""

    # Pre-defined default parameters
    input_file: str = "./input.las"
    reset_classification: bool = True
    filter_noise: bool = False
    chunk_size: int = 5_000_000
    ground_segmentation: dict = {
        "enable": True,
        "method": "basic",
        "csf_params": {
            "threshold_factor": 0.1,
            "resolution": 1.0,
            "rigidness": 3,
            "max_iter": 500
        }
    }
    wall_segmentation: dict = {
        "enable": False,
        "filter_noise_after_segmenetation": True,
    }
    result = {
        "display_result": True,
        "write_to_file": "./output.las"
    }

    def __init__(self, config_file: str):
        """Initialize SegmentorConfig from a YAML file.

        Args:
            config_file (str): Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If config_file does not exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")

        # Load YAML file
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f) or {}

        self.validate_config_file(config_data)

        # Override with loaded values from YAML
        for key, value in config_data.items():
            setattr(self, key, value)

    def validate_config_file(self, config_data):
        input_file = config_data.get("input_file", self.input_file)
        if not input_file:
            raise FileNotFoundError(f"LAS file not found: {input_file}")
        