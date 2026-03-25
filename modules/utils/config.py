import yaml
import os


class SegmentorConfig:
    """
    Load configuration from a YAML file and expose it as attributes.

    Class attributes in this type act as defaults and are overridden by values
    loaded from `config.yml` (or another provided config file). Downstream
    components should therefore always rely on this object rather than reading
    the YAML directly.
    """

    # Pre-defined default parameters
    input_file: str = "./input.las"
    reset_classification: bool = True
    filter_noise: bool = False
    chunk_size: int = 5_000_000
    previewer = {
        "maximum_points": 500_000
    }
    ground_segmentation: dict = {
        "enable": True,
        "method": "basic",
        "csf_params": {
            "threshold_factor": 0.1,
            "resolution": 1.0,
            "rigidness": 3,
            "max_iter": 500,
        },
        "ransac_params": {
            "distance_threshold": 0.05,
            "num_iterations": 1000,
        },
    }
    ceiling_segmentation: dict = {
        "enable": False,
        "epsilon": 0.01,
        "min_support_points": 2000,
        "max_tilt_from_horizontal_deg": 10.0,
        "max_iterations": 1000,
        "min_plane_size_ratio": 0.01,
        "min_height_above_ground": 2.0,
        "max_height_above_ground": 10.0,
    }
    wall_segmentation: dict = {
        "enable": False,
        "filter_noise_after_segmenetation": True,
    }
    objects_segmentation: dict = {
        "enable": False,
        "eps": 0.05,
        "min_points": 10,
        "ollama_model": "gemma3:4b",
        "vlm_traces_subdir": "vlm_object_traces",
    }
    result = {
        "display_result": True,
        "output_folder_path": "./result",
        "overwrite_output_folder": False
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

        ceiling_cfg = config_data.get("ceiling_segmentation", self.ceiling_segmentation)
        min_h = ceiling_cfg.get("min_height_above_ground", 2.0)
        max_h = ceiling_cfg.get("max_height_above_ground", 10.0)
        if min_h < 0 or max_h < 0 or min_h > max_h:
            raise ValueError(
                "Invalid ceiling_segmentation height configuration: "
                f"min_height_above_ground={min_h}, max_height_above_ground={max_h}"
            )

        ground_cfg = config_data.get("ground_segmentation", self.ground_segmentation)
        ground_method = ground_cfg.get("method", "basic")
        allowed_ground = ("basic", "csf", "ransac")
        if ground_method not in allowed_ground:
            raise ValueError(
                f"ground_segmentation.method must be one of {allowed_ground}, got {ground_method!r}"
            )
        ransac_cfg = ground_cfg.get("ransac_params", self.ground_segmentation["ransac_params"])
        dt = ransac_cfg.get("distance_threshold", 0.05)
        ni = ransac_cfg.get("num_iterations", 1000)
        if dt <= 0:
            raise ValueError(
                f"ground_segmentation.ransac_params.distance_threshold must be positive, got {dt}"
            )
        if ni < 1:
            raise ValueError(
                f"ground_segmentation.ransac_params.num_iterations must be >= 1, got {ni}"
            )

        obj_cfg = config_data.get("objects_segmentation", self.objects_segmentation)
        eps = obj_cfg.get("eps", 0.05)
        min_pts = obj_cfg.get("min_points", 10)
        if eps <= 0:
            raise ValueError(f"objects_segmentation.eps must be positive, got {eps}")
        if min_pts < 2:
            raise ValueError(
                f"objects_segmentation.min_points must be >= 2, got {min_pts}"
            )