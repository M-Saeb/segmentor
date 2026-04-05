from modules.viewers import PipelineViewer


if __name__ == "__main__":
    import sys
    
    default_config_file = "config.yml"



    if len(sys.argv) > 2:
        mode_arg = sys.argv[1]
        config_file = sys.argv[2]
    elif len(sys.argv) == 2:
        mode_arg = sys.argv[1]
        config_file = "config.yml"
    else:
        mode_arg = "pipeline"
        config_file = "config.yml"

    if mode_arg == "pipeline":
        viewer = PipelineViewer(config_file)
        viewer.create_gui()
    else:
        print(f"Invalid mode: {mode_arg}. Valid modes are: pre, pipeline, post")
        sys.exit(1)

