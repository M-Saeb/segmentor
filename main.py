from las_wrapper import LasWrapper

if __name__ == "__main__":
    import sys
    
    default_config_file = "config.yml"

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = default_config_file

    las_wrapper = LasWrapper(config_file)
    las_wrapper.filter_noise()
    las_wrapper.segment_ground()
    las_wrapper.write_to_file()