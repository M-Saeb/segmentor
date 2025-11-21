from las_wrapper import LasWrapper

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python las_viewer.py <path_to_las_file> <output_las_file>")
        sys.exit(1)
    
    las_file = sys.argv[1]
    output_file = sys.argv[2]
    las_wrapper = LasWrapper(las_file)
    las_wrapper.write_to_file(output_file)