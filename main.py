from modules.viewers import PipelineViewer, PostProcessingViewer, PreProcessingViewer


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
    elif mode_arg == "pre_process":
        viewer = PreProcessingViewer(config_file)
        viewer.create_gui()
    elif mode_arg == "post_process":
        viewer = PostProcessingViewer(config_file)
        viewer.create_gui()
    else:
        print(f"Invalid mode: {mode_arg}. Valid modes are: pre_process, pipeline, post_process")
        sys.exit(1)

