import os
import json
import yaml
import numpy as np
import open3d as o3d
import time
import argparse

from common.o3d_visualizer import O3dVisualizer
from common.data_control.synthetic_bbox import SyntheticBbox
from common.data_control.detection_bbox import DetectionBbox
from common.data_control.utils import *

from bbox_evaluation.association import *

# TODO move to file
DETECTION_POINT_COLOR = [0.7, 0, 0.7] 
SYNTHETIC_POINT_COLOR = [0, 0.7, 0.7]
DETECTION_BBOX_COLOR = [0.3, 0, 0.3]
SYNTHETIC_BBOX_COLOR = [0, 0.3, 0.3]
DETECTION_CENTER_COLOR = [0.7, 0.3, 0.7]
SYNTHETIC_CENTER_COLOR = [0.3, 0.7, 0.7]
DETECTION_MATCH_COLOR = [0, 0.8, 0]
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

def visualize_data(config_data, o3d_visualizer):
    ### Load detection data

    detection_data_path = config_data['detection_loader']['data_path']
    detection_int_precision = config_data['detection_loader']['int_precision']

    detection_bbox = DetectionBbox(detection_data_path, int_precision=detection_int_precision)
    detection_bbox.setup()
    detection_bbox.load_data()

    detection_data_range = detection_bbox.data_range

    complete_detection_pcds, complete_detection_bboxes, complete_detection_timestamps = extract_complete_data_from_detection_bboxes(detection_bbox)

    # Load synthetic data

    synthetic_data_path = config_data['synthetic_loader']['data_path']
    synthetic_int_precision = config_data['synthetic_loader']['int_precision']
    synthetic_multi_sem_classes = config_data['synthetic_loader']['multi_sem_classes']
    synthetic_camera_position = np.array(config_data['synthetic_loader']['camera_position'])
    synthetic_camera_rotation = np.array(config_data['synthetic_loader']['camera_rotation'])
    
    """
    Global Position: (-10.000000000540384, 5.0018717613276635, 0.21277073854091316)
    Rotation Matrix:
    ( (0.9999999999999993, 3.42285420007471e-8, 0), (-5.434188106286747e-22, 1.587618925213973e-14, 1), (3.42285420007471e-8, -0.9999999999999993, 1.587618925213974e-14) )
    """
        
    
    translation_vector, rotation_matrix = get_translation_and_rotation_from_camera_properties(synthetic_camera_position, synthetic_camera_rotation)
                
    synthetic_bbox = SyntheticBbox(synthetic_data_path, multi_sem_classes=synthetic_multi_sem_classes, int_precision=synthetic_int_precision,
                                    translation_vector=translation_vector, rotation_matrix=rotation_matrix)
    synthetic_bbox.setup()
    synthetic_bbox.load_labels()
    synthetic_bbox.load_data()

    complete_synthetic_pcds, complete_synthetic_bboxes = extract_complete_data_from_synthetic_bboxes(synthetic_bbox)
        
    ### Global parameters

    detection_frame_offset = config_data['alignment_config']['detection_frame_offset']
    synthetic_frame_offset = config_data['alignment_config']['synthetic_frame_offset']
    synthetic_fps = config_data['alignment_config']['synthetic_fps']
        
    if detection_frame_offset < 0 or synthetic_frame_offset < 0:
        raise ValueError("Frame offsets must be positive")
    
    # estimate fps from first and last timestamps and number of frames
    detection_fps = estimate_detection_frame_rate(complete_detection_timestamps)
    print(f"Estimated FPS: {detection_fps}")
        
    ### Visualization

    o3d_visualizer.setup()

    initial_detection_timestamp = complete_detection_timestamps[0]
    data_range = detection_data_range
    range_start = detection_data_range[0] + detection_frame_offset
    range_end = detection_data_range[1]
    playback_fps = detection_fps * config_data['alignment_config']['playback_speed']

    paused = False
    run = True
    exit = False
    
    def pause_key_callback(vis):
        nonlocal paused
        paused = not paused
        return False
    
    def rerun_key_callback(vis):
        nonlocal run
        run = True
        vis.close()
        return False
    
    def exit_key_callback(vis):
        nonlocal exit
        exit = True
        vis.close()
        return False

    o3d_visualizer.vis.register_key_callback(32, pause_key_callback)  # Space bar key
    o3d_visualizer.vis.register_key_callback(67, rerun_key_callback)  # C key
    o3d_visualizer.vis.register_key_callback(256, exit_key_callback)  # ESC key

    while not exit:
        if run:
            run = False
        else:
            break

        for i in range(range_start, range_end + 1):
            start_time = time.time()
            if exit or run:
                break

            current_detection_timestamp = complete_detection_timestamps[i - range_start]
            delta_seconds = current_detection_timestamp['seconds'] - initial_detection_timestamp['seconds']
            delta_nanoseconds = current_detection_timestamp['nanoseconds'] - initial_detection_timestamp['nanoseconds']

            # calculate closest frame from synthetic data
            detection_frame = i - data_range[0]
            synthetic_frame = int((delta_seconds + delta_nanoseconds / 1e9) * synthetic_fps) + synthetic_frame_offset

            if synthetic_frame < 0:
                continue
            elif synthetic_frame >= len(complete_synthetic_pcds):
                break
            
            detection_assoc_bounds = [get_bbox_extremes(np.asarray(bbox.get_box_points())) for bbox in complete_detection_bboxes[detection_frame]]
            synthetic_assoc_bounds = [get_bbox_extremes(np.asarray(bbox.get_box_points())) for bbox in complete_synthetic_bboxes[synthetic_frame]]
            iou_threshold = 0

            matches, unmatched_detections, unmatched_synthetic, iou_values = associate_3d(detection_assoc_bounds, synthetic_assoc_bounds, iou_threshold)

            # print shapes of the matched and unmatched
            print(f"Matched detections: {len(matches)}, Unmatched detections: {len(unmatched_detections)}, Unmatched synthetic: {len(unmatched_synthetic)}")
            print(f"IOU values: {iou_values}")

            detection_centroids = [np.asarray(bbox.get_center()) for bbox in complete_detection_bboxes[detection_frame]]
            synthetic_centroids = [np.asarray(bbox.get_center()) for bbox in complete_synthetic_bboxes[synthetic_frame]]

            o3d_visualizer.reset()

            for i, geometry in enumerate(complete_detection_pcds[detection_frame]):
                geometry.paint_uniform_color(DETECTION_POINT_COLOR)
                o3d_visualizer.vis.add_geometry(geometry, reset_bounding_box=False)

            for i, geometry in enumerate(complete_detection_bboxes[detection_frame]):
                if i in unmatched_detections:
                    geometry.color = DETECTION_BBOX_COLOR
                else:
                    geometry.color = DETECTION_MATCH_COLOR
                o3d_visualizer.vis.add_geometry(geometry, reset_bounding_box=False)

            for i, centroid in enumerate(detection_centroids):
                draw_centroid(o3d_visualizer.vis, centroid, DETECTION_CENTER_COLOR)

            for i, geometry in enumerate(complete_synthetic_pcds[synthetic_frame]):
                geometry.paint_uniform_color(SYNTHETIC_POINT_COLOR)
                o3d_visualizer.vis.add_geometry(geometry, reset_bounding_box=False)

            for i, geometry in enumerate(complete_synthetic_bboxes[synthetic_frame]):
                if i in unmatched_synthetic:
                    geometry.color = SYNTHETIC_BBOX_COLOR
                else:
                    geometry.color = SYNTHETIC_MATCH_COLOR
                o3d_visualizer.vis.add_geometry(geometry, reset_bounding_box=False)

            for i, centroid in enumerate(synthetic_centroids):
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



def main():
    parser = argparse.ArgumentParser(description='Visualize detection and synthetic data')
    parser.add_argument('config_file', type=str, help='Path to the config file')
    args = parser.parse_args()
    
    # config file in yaml format
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"Config file {args.config_file} not found")
    
    with open(args.config_file, 'r') as file:
        config_data = yaml.safe_load(file)
    
    o3d_visualizer = O3dVisualizer()

    visualize_data(config_data, o3d_visualizer)

if __name__ == "__main__":
    main()
