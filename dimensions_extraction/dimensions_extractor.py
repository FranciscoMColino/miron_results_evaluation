import os
import json
import yaml
import numpy as np
import open3d as o3d
import time
import argparse
import math

from common.o3d_visualizer import O3dVisualizer
from common.data_control.synthetic_bbox import SyntheticBbox
from common.data_control.detection_bbox import Detection3dBbox
from common.data_control.utils import *

from detection_evaluation.association import *

# TODO move to file
SYNTHETIC_POINT_COLOR = [0, 0.7, 0.7]
SYNTHETIC_BBOX_COLOR = [0, 0.3, 0.3]
SYNTHETIC_CENTER_COLOR = [0.3, 0.7, 0.7]
SYNTHETIC_MATCH_COLOR = [0, 0.4, 0]

def get_bbox_extremes(bbox):
    bbox = np.array(bbox)
    min_coords = np.min(bbox, axis=0)
    max_coords = np.max(bbox, axis=0)
    return [min_coords[0], min_coords[1], min_coords[2], max_coords[0], max_coords[1], max_coords[2]]

def draw_centroid(vis, centroid, color):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    sphere.translate(centroid)
    vis.add_geometry(sphere, reset_bounding_box=False)

def eucdist_vismatch_3ddet_data(config_data, o3d_visualizer, verbose=True):

    # Load from config data to variables
    synthetic_data_path = config_data['synthetic_path']
    camera_position = np.array(config_data['camera_position'])
    camera_rotation = np.array(config_data['camera_rotation'])
    synthetic_fps = config_data['synthetic_fps']
    playback_speed = config_data['playback_speed']

    # Load synthetic data 
    translation_vector, rotation_matrix = get_translation_and_rotation_from_camera_properties(camera_position, camera_rotation)
    synthetic_bbox = SyntheticBbox(synthetic_data_path,translation_vector=translation_vector, rotation_matrix=rotation_matrix)
    synthetic_bbox.setup()
    synthetic_bbox.load_labels()
    synthetic_bbox.load_data()
    synthetic_bbox.compute_bboxes()

    o3d_visualizer.setup()
    data_range = synthetic_bbox.data_range
    range_start = data_range[0]
    range_end = data_range[1]
    playback_fps = synthetic_fps * playback_speed

    paused = True
    run = True
    exit = False
    
    def pause_key_callback(vis):
        nonlocal paused
        paused = not paused
        return False
    
    def exit_key_callback(vis):
        nonlocal exit
        exit = True
        vis.close()
        return False

    o3d_visualizer.vis.register_key_callback(32, pause_key_callback)  # Space bar key
    o3d_visualizer.vis.register_key_callback(256, exit_key_callback)

    # evaluation results should hold the thresholds used, an array with the results for each frame, and the average results

    analysed_frames = 0

    dimensions_dict = {}

    while not exit:
        if run:
            run = False
        else:
            break

        for i in range(range_start, range_end + 1):
            start_time = time.time()
            if exit or run:
                break

            # calculate closest frame from synthetic data
            synthetic_frame = i - range_start

            if synthetic_frame < 0:
                continue
            elif synthetic_frame >= len(synthetic_bbox.complete_points):
                break
            
            synthetic_centroids = [get_centroid_from_points(bbox_points) for bbox_points in synthetic_bbox.complete_bboxes[synthetic_frame]]


            if verbose:
                print(f"\nFrame {analysed_frames}")

            for j, bbox in enumerate(synthetic_bbox.complete_bboxes[synthetic_frame]):
                
                if dimensions_dict.get(j) is None:
                    dimensions_dict[j] = []

                bbox_extremes = get_bbox_extremes(bbox)
                width = bbox_extremes[3] - bbox_extremes[0] # x
                depth = bbox_extremes[4] - bbox_extremes[1] # y
                height = bbox_extremes[5] - bbox_extremes[2] # z

                if width > 100 or depth > 100 or height > 100:
                    print(f"WARNING: Bbox {j} dimensions are too large: width {width}, depth {depth}, height {height}")
                    continue

                print(f"Synthetic bbox {j} dimensions: width {width}, depth {depth}, height {height}")

                dimensions_dict[j].append([width, depth, height])

            analysed_frames += 1

            o3d_visualizer.reset()

            for points in synthetic_bbox.complete_points[synthetic_frame]:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.paint_uniform_color(SYNTHETIC_POINT_COLOR)
                o3d_visualizer.vis.add_geometry(pcd, reset_bounding_box=False)

            for bbox_points in synthetic_bbox.complete_bboxes[synthetic_frame]:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(bbox_points)
                pcd.paint_uniform_color(SYNTHETIC_POINT_COLOR)
                o3d_visualizer.vis.add_geometry(pcd, reset_bounding_box=False)
                bbox = pcd.get_axis_aligned_bounding_box()
                bbox.color = SYNTHETIC_BBOX_COLOR
                o3d_visualizer.vis.add_geometry(bbox, reset_bounding_box=False)

            for centroid in synthetic_centroids:
                draw_centroid(o3d_visualizer.vis, centroid, SYNTHETIC_CENTER_COLOR)

            o3d_visualizer.render()

            end_time = time.time()

            while (end_time - start_time) < 1 / playback_fps:
                end_time = time.time()
                o3d_visualizer.render()
                if paused:
                    while paused and not exit and not run:
                        o3d_visualizer.render()
                        time.sleep(1/60)

        o3d_visualizer.render()
        while not exit and not run:
            o3d_visualizer.render()
            time.sleep(1/60)

    o3d_visualizer.vis.destroy_window()

    # for each bbox, calculate print the frame that has the smallest and largest dimensions, and the average dimensions
    for bbox_id, dimensions in dimensions_dict.items():
        print(f"\nBbox {bbox_id} dimensions")
        min_frame = None
        max_frame = None
        min_volume = None
        max_volume = None
        min_dimensions = None
        max_dimensions = None
        total_width = 0
        total_depth = 0
        total_height = 0

        for frame, frame_dimensions in enumerate(dimensions):
            width, depth, height = frame_dimensions
            volume = width * depth * height
            total_width += width
            total_depth += depth
            total_height += height

            if min_volume is None or volume < min_volume:
                min_volume = volume
                min_frame = frame
                min_dimensions = frame_dimensions

            if max_volume is None or volume > max_volume:
                max_volume = volume
                max_frame = frame
                max_dimensions = frame_dimensions

        average_width = total_width / len(dimensions)
        average_depth = total_depth / len(dimensions)
        average_height = total_height / len(dimensions)

        print(f"Min volume: {min_volume}, frame: {min_frame}, width: {min_dimensions[0]}, depth: {min_dimensions[1]}, height: {min_dimensions[2]}")
        print(f"Max volume: {max_volume}, frame: {max_frame}, width: {max_dimensions[0]}, depth: {max_dimensions[1]}, height: {max_dimensions[2]}")
        print(f"Average dimensions: width: {average_width}, depth: {average_depth}, height: {average_height}")

    return 0

def main():
    parser = argparse.ArgumentParser(description='Visualize matching detection against synthetic data')
    parser.add_argument('config_file', type=str, help='Path to the config file')
    args = parser.parse_args()
    
    # config file in yaml format
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"Config file {args.config_file} not found")
    
    with open(args.config_file, 'r') as file:
        config_data = yaml.safe_load(file)

    o3d_visualizer = O3dVisualizer()

    eucdist_vismatch_3ddet_data(config_data, o3d_visualizer)

if __name__ == "__main__":
    main()
