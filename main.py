from las_wrapper import LasWrapper

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python las_viewer.py <path_to_las_file> <output_las_file>")
        sys.exit(1)
    
    las_file = sys.argv[1]
    output_file = sys.argv[2]
    # o3d.visualization.draw_geometries([pcd],
    #                                 window_name="LAS Viewer",
    #                                 width=1024,
    #                                 height=768,
    #                                 point_show_normal=False)

    las_wrapper = LasWrapper(las_file)
    las_wrapper.filter_noise()
    las_wrapper.segment_ground()
    las_wrapper.write_to_file(output_file)