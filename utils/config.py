import yaml
import os


class SegmentorConfig:
    """Load configuration from a YAML file and expose as attributes."""

    # Pre-defined default parameters
    input_file: str = "./input.las"
    output_file_path: str = "./output.las"
    chunk_size: int = 5_000_000
    ground_filter_method: str = "basic"  # Options: "basic", "csf"

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

        # Override with loaded values from YAML
        for key, value in config_data.items():
            setattr(self, key, value)
        
        self._validate_input_file()

    def _validate_input_file(self):
        if not os.path.isfile(self.input_file):
            raise FileNotFoundError(f"LAS file not found: {self.input_file}")