import os
import json
import yaml
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import time
import argparse

from common.o3d_visualizer import O3dVisualizer
from common.data_loaders.synthetic_bbox import SyntheticBbox
from common.data_loaders.detection_bbox import DetectionBbox


def visualize_data(config_data, o3d_visualizer):
    ### Load detection data

    detection_data_path = config_data['detection_loader']['data_path']
    detection_int_precision = config_data['detection_loader']['int_precision']

    detection_bbox = DetectionBbox(detection_data_path, int_precision=detection_int_precision)
    detection_bbox.setup()
    detection_bbox.load_data()

    detection_data_range = detection_bbox.data_range

    complete_detection_pcds = np.empty(detection_data_range[1] - detection_data_range[0] + 1, dtype=list)
    complete_detection_bboxes = np.empty(detection_data_range[1] - detection_data_range[0] + 1, dtype=list)
    complete_detection_timestamps = np.empty(detection_data_range[1] - detection_data_range[0] + 1, dtype=dict)

    for i in range(detection_data_range[0], detection_data_range[1] + 1):
        
        local_pcds = []
        local_bboxes = []
        
        for bbox_points in detection_bbox.complete_bbox_points[i - detection_data_range[0]]:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(bbox_points)
            pcd.paint_uniform_color([1, 0, 1])
            local_pcds.append(pcd)
            
            bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
            bbox.color = (1, 0, 0)
            local_bboxes.append(bbox)
            
        complete_detection_pcds[i - detection_data_range[0]] = local_pcds
        complete_detection_bboxes[i - detection_data_range[0]] = local_bboxes
        complete_detection_timestamps[i - detection_data_range[0]] = detection_bbox.complete_timestamps[i - detection_data_range[0]]

    # Load synthetic data

    synthetic_data_path = config_data['synthetic_loader']['data_path']
    synthetic_int_precision = config_data['synthetic_loader']['int_precision']
    synthetic_single_sem_classes = config_data['synthetic_loader']['single_sem_classes']
    synthetic_camera_position = np.array(config_data['synthetic_loader']['camera_position'])
    synthetic_camera_rotation = np.array(config_data['synthetic_loader']['camera_rotation'])
    
    """
    Global Position: (-10.000000000540384, 5.0018717613276635, 0.21277073854091316)
    Rotation Matrix:
    ( (0.9999999999999993, 3.42285420007471e-8, 0), (-5.434188106286747e-22, 1.587618925213973e-14, 1), (3.42285420007471e-8, -0.9999999999999993, 1.587618925213974e-14) )
    """
        
    ### Transformations according to the camera coordinate system
    rotation_x_minus_90 = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])

    # rotation over Z for angle
    rotation_angle = 90
    rotation_matrix = np.array([
        [np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle)), 0],
        [np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle)), 0],
        [0, 0, 1]
    ])

    # Define the translation vector (3x1)
    translation_vector = -synthetic_camera_position
    rotation_matrix = synthetic_camera_rotation @ rotation_x_minus_90.T @ rotation_matrix
                
    synthetic_bbox = SyntheticBbox(synthetic_data_path, singl_sem_classes=synthetic_single_sem_classes, int_precision=synthetic_int_precision,
                                    translation_vector=translation_vector, rotation_matrix=rotation_matrix)
    synthetic_bbox.setup()
    synthetic_bbox.load_labels()
    synthetic_bbox.load_data()

    synthetic_data_range = synthetic_bbox.data_range

    # initiate np array with size of data_range
    complete_synthetic_pcds = np.empty(synthetic_data_range[1] - synthetic_data_range[0] + 1, dtype=list)
    complete_synthetic_bboxes = np.empty(synthetic_data_range[1] - synthetic_data_range[0] + 1, dtype=list)
        
    for i in range(synthetic_data_range[0], synthetic_data_range[1] + 1):
        complete_synthetic_pcds[i - synthetic_data_range[0]] = []
        complete_synthetic_bboxes[i - synthetic_data_range[0]] = []

        for j, bbox_points in enumerate(synthetic_bbox.complete_bbox_points[i - synthetic_data_range[0]]):
            
            bbox_points = np.array(bbox_points).astype(np.float64)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(bbox_points).astype(np.float64))
            # colorize the synthetic point clouds
            pcd.paint_uniform_color([0, 1, 1])
            complete_synthetic_pcds[i - synthetic_data_range[0]].append(pcd)
            
            bbox = pcd.get_axis_aligned_bounding_box()
            bbox.color = (0, 1, 0)
            complete_synthetic_bboxes[i - synthetic_data_range[0]].append(bbox)
        
    ### Global parameters

    detection_frame_offset = config_data['alignment_config']['detection_frame_offset']
    synthetic_frame_offset = config_data['alignment_config']['synthetic_frame_offset']
    synthetic_fps = config_data['alignment_config']['synthetic_fps']
        
    if detection_frame_offset < 0 or synthetic_frame_offset < 0:
        raise ValueError("Frame offsets must be positive")
    
    # estimate fps from first and last timestamps and number of frames
    if complete_detection_timestamps.all():
        start_time = complete_detection_timestamps[0]
        end_time = complete_detection_timestamps[-1]
        start_seconds = start_time['seconds'] + start_time['nanoseconds'] / 1e9
        end_seconds = end_time['seconds'] + end_time['nanoseconds'] / 1e9
        num_frames = len(complete_detection_timestamps)
        detection_fps = num_frames / (end_seconds - start_seconds)
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
            synthetic_frame = int((delta_seconds + delta_nanoseconds / 1e9) * synthetic_fps) + synthetic_frame_offset

            if synthetic_frame < 0:
                continue
            elif synthetic_frame >= len(complete_synthetic_pcds):
                break

            o3d_visualizer.reset()

            for geometry in complete_detection_pcds[i - data_range[0]]:
                o3d_visualizer.vis.add_geometry(geometry, reset_bounding_box=False)

            for geometry in complete_detection_bboxes[i - data_range[0]]:
                geometry.color = (1, 0, 0)
                o3d_visualizer.vis.add_geometry(geometry, reset_bounding_box=False)

            for geometry in complete_synthetic_pcds[synthetic_frame]:
                o3d_visualizer.vis.add_geometry(geometry, reset_bounding_box=False)

            for geometry in complete_synthetic_bboxes[synthetic_frame]:
                geometry.color = (0, 1, 0)
                o3d_visualizer.vis.add_geometry(geometry, reset_bounding_box=False)

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
