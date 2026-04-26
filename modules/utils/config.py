import yaml
import os
import tempfile

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
    filter_noise: dict = {
        "nb_neighbors": 20,
        "std_ratio": 2.0
    }
    chunk_size: int = 5_000_000
    downsample: dict = {
        "voxel_size": 0.15
    }
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
        "facing_up_threshold": 0.95,
        "min_inlier_density": 15,
    }
    wall_segmentation: dict = {
        "enable": False,
        "rotation_threshold": 0.9,
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
    collect_runtime_report: bool = False
    cityjson_export: dict = {
        "enable": True,
        "max_faces": 1_000_000,
        "ball_radius": 5,
    }

    _instance = None
    _config_path: str | None = None

    def __new__(cls, config_file: str | None = None):
        """
        Enforce one SegmentorConfig instance per process.

        The first construction must provide ``config_file``. Any later
        construction in the same process must use the same config path.
        """
        normalized = cls._normalize_config_path(config_file)

        if cls._instance is None:
            if normalized is None:
                raise ValueError("SegmentorConfig is not initialized. Provide config_file.")
            cls._instance = super().__new__(cls)
            cls._config_path = normalized
            cls._instance._initialized = False
            return cls._instance

        if normalized is not None and normalized != cls._config_path:
            raise RuntimeError(
                "SegmentorConfig singleton already initialized with "
                f"{cls._config_path!r}; cannot reinitialize with {normalized!r}."
            )
        return cls._instance

    @classmethod
    def _normalize_config_path(cls, config_file: str | None) -> str | None:
        if config_file is None:
            return None
        return os.path.realpath(os.path.abspath(config_file))

    @classmethod
    def initialize(cls, config_file: str):
        """
        Canonical singleton initializer.
        """
        return cls(config_file)

    @classmethod
    def get_instance(cls):
        """
        Return the initialized singleton instance.
        """
        if cls._instance is None:
            raise RuntimeError("SegmentorConfig has not been initialized yet.")
        return cls._instance

    def __init__(self, config_file: str | None = None):
        """Initialize SegmentorConfig from a YAML file.

        Args:
            config_file (str): Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If config_file does not exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        if getattr(self, "_initialized", False):
            return

        normalized_config_path = self._normalize_config_path(config_file)
        if normalized_config_path is None:
            raise ValueError("config_file is required for first initialization.")

        # Keep an instance-level reference to the selected singleton config path.
        self.config_file = normalized_config_path

        if not os.path.exists(normalized_config_path):
            raise FileNotFoundError(f"Config file not found: {normalized_config_path}")

        # Load YAML file
        with open(normalized_config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}

        self.validate_config_file(config_data)

        # Override with loaded values from YAML
        for key, value in config_data.items():
            setattr(self, key, value)
        self._initialized = True

    def validate_config_file(self, config_data):
        input_file = config_data.get("input_file", self.input_file)
        if not input_file:
            raise FileNotFoundError(f"LAS file not found: {input_file}")

        ceiling_cfg = config_data.get("ceiling_segmentation", self.ceiling_segmentation)
        fu = float(ceiling_cfg.get("facing_up_threshold", 0.95))
        if fu <= 0 or fu > 1:
            raise ValueError(
                "ceiling_segmentation.facing_up_threshold must be in (0, 1], "
                f"got {fu}"
            )
        mid = float(ceiling_cfg.get("min_inlier_density", 15))
        if mid < 0:
            raise ValueError(
                "ceiling_segmentation.min_inlier_density must be >= 0, "
                f"got {mid}"
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