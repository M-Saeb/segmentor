from modules.viewers import PipelineViewer

if __name__ == "__main__":
    import sys
    
    default_config_file = "config.yml"

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = default_config_file

    viewer = PipelineViewer(config_file)
    viewer.create_gui()