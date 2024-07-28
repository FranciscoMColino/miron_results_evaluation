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

def evaluate_visualize_data(config_data, o3d_visualizer):

    ### Load detection data

    detection_data_path = config_data['detection_loader']['data_path']
    detection_int_precision = config_data['detection_loader']['int_precision']

    detection_bbox = Detection3dBbox(detection_data_path, int_precision=detection_int_precision)
    detection_bbox.setup()
    detection_bbox.load_data()
    detection_bbox.compute_bboxes()
    detection_data_range = detection_bbox.data_range

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
    synthetic_bbox.compute_bboxes()
    synthetic_data_range = synthetic_bbox.data_range
        
    ### Global parameters

    detection_frame_offset = config_data['alignment_config']['detection_frame_offset']
    synthetic_frame_offset = config_data['alignment_config']['synthetic_frame_offset']
    synthetic_fps = config_data['alignment_config']['synthetic_fps']
        
    if detection_frame_offset < 0 or synthetic_frame_offset < 0:
        raise ValueError("Frame offsets must be positive")
    
    # estimate fps from first and last timestamps and number of frames
    detection_fps = estimate_detection_frame_rate(detection_bbox.complete_timestamps)
    print(f"Estimated FPS: {detection_fps}")
   
    ### Visualization

    o3d_visualizer.setup()

    initial_detection_timestamp = detection_bbox.complete_timestamps[0]
    data_range = detection_data_range
    range_start = detection_data_range[0] + detection_frame_offset
    range_end = detection_data_range[1]
    playback_fps = detection_fps * config_data['alignment_config']['playback_speed']

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
    o3d_visualizer.vis.register_key_callback(256, exit_key_callback)  # ESC key


    iou_thresholds = config_data['IoU_thresholds']

    # evaluation results should hold the thresholds used, an array with the results for each frame, and the average results
    
    frame_results_dtype = [('precision', 'f4'), ('recall', 'f4'), ('iou', 'f4')]
    results_collection = np.zeros((len(detection_bbox.complete_timestamps), len(iou_thresholds)), dtype=frame_results_dtype)

    analysed_frames = 0
    match_ocurred_per_threshould = np.zeros((len(iou_thresholds),), dtype=int)

    avg_frame_iou_collection = np.zeros((len(detection_bbox.complete_timestamps),), dtype='f4')

    while not exit:
        if run:
            run = False
        else:
            break

        for i in range(range_start, range_end + 1):
            start_time = time.time()
            if exit or run:
                break

            current_detection_timestamp = detection_bbox.complete_timestamps[i - range_start]
            delta_seconds = current_detection_timestamp['seconds'] - initial_detection_timestamp['seconds']
            delta_nanoseconds = current_detection_timestamp['nanoseconds'] - initial_detection_timestamp['nanoseconds']

            # calculate closest frame from synthetic data
            detection_frame = i - data_range[0]
            synthetic_frame = int((delta_seconds + delta_nanoseconds / 1e9) * synthetic_fps) + synthetic_frame_offset

            if synthetic_frame < 0:
                continue
            elif synthetic_frame >= len(synthetic_bbox.complete_points):
                break
            
            detection_assoc_bounds = [get_bbox_extremes(bbox_points) for bbox_points in detection_bbox.complete_bboxes[detection_frame]]
            synthetic_assoc_bounds = [get_bbox_extremes(bbox_points) for bbox_points in synthetic_bbox.complete_bboxes[synthetic_frame]]

            """
                For every IoU threshold, associate the detections with the synthetic data
                the metrics to be calculated are:
                - Precision
                - IoU
                - Recall
            """

            
            frame_results = np.zeros((len(iou_thresholds),), dtype=frame_results_dtype)

            _, _, _, frame_iou = associate_3d(detection_assoc_bounds, synthetic_assoc_bounds, 0)
            avg_frame_iou = np.mean(frame_iou)
            avg_frame_iou_collection[analysed_frames] = avg_frame_iou

            print(f"\nFrame {analysed_frames}, Average IoU: {avg_frame_iou:.2f}")

            for j, iou_threshold in enumerate(iou_thresholds):
                matches, unmatched_detections, unmatched_synthetic, iou_values = associate_3d(detection_assoc_bounds, synthetic_assoc_bounds, iou_threshold)
                true_positives = len(matches)
                false_positives = len(unmatched_detections)
                ground_truth_count = len(synthetic_assoc_bounds)

                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / ground_truth_count if ground_truth_count > 0 else 0
                iou = np.mean(iou_values) if len(iou_values) > 0 else -1

                match_ocurred_per_threshould[j] += 1 if true_positives > 0 else 0

                frame_results[j] = (precision, recall, iou)
                # print rounded to 2 decimal places
                print(f"Threshold: {iou_threshold}, Precision: {precision:.2f}, Recall: {recall:.2f}, IoU: {iou:.2f}")

            
            results_collection[analysed_frames] = frame_results
            analysed_frames += 1

            detection_centroids = [get_centroid_from_points(bbox_points) for bbox_points in detection_bbox.complete_bboxes[detection_frame]]
            synthetic_centroids = [get_centroid_from_points(bbox_points) for bbox_points in synthetic_bbox.complete_bboxes[synthetic_frame]]

            o3d_visualizer.reset()


            for points in detection_bbox.complete_points[detection_frame]:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.paint_uniform_color(DETECTION_POINT_COLOR)
                o3d_visualizer.vis.add_geometry(pcd, reset_bounding_box=False)

            for points in synthetic_bbox.complete_points[synthetic_frame]:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.paint_uniform_color(SYNTHETIC_POINT_COLOR)
                o3d_visualizer.vis.add_geometry(pcd, reset_bounding_box=False)

            for bbox_points in detection_bbox.complete_bboxes[detection_frame]:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(bbox_points)
                pcd.paint_uniform_color(DETECTION_POINT_COLOR)
                o3d_visualizer.vis.add_geometry(pcd, reset_bounding_box=False)
                bbox = pcd.get_axis_aligned_bounding_box()
                bbox.color = DETECTION_BBOX_COLOR
                o3d_visualizer.vis.add_geometry(bbox, reset_bounding_box=False)

            for bbox_points in synthetic_bbox.complete_bboxes[synthetic_frame]:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(bbox_points)
                pcd.paint_uniform_color(SYNTHETIC_POINT_COLOR)
                o3d_visualizer.vis.add_geometry(pcd, reset_bounding_box=False)
                bbox = pcd.get_axis_aligned_bounding_box()
                bbox.color = SYNTHETIC_BBOX_COLOR
                o3d_visualizer.vis.add_geometry(bbox, reset_bounding_box=False)

            for centroid in detection_centroids:
                draw_centroid(o3d_visualizer.vis, centroid, DETECTION_CENTER_COLOR)

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

    # strip the zeros from the results collection
    results_collection = results_collection[:analysed_frames]

    average_precision = np.zeros(len(iou_thresholds))
    average_recall = np.zeros(len(iou_thresholds))
    average_iou = np.zeros(len(iou_thresholds))

    for frame_results in results_collection:
        for i, (precision, recall, iou) in enumerate(frame_results):
            average_precision[i] += precision
            average_recall[i] += recall
            if iou > 0:
                average_iou[i] += iou

    average_precision /= analysed_frames
    average_recall /= analysed_frames
    average_iou /= analysed_frames

    avg_frame_iou_collection = avg_frame_iou_collection[:analysed_frames]
    mean_iou = np.mean(avg_frame_iou_collection)

    evaluation_results_dtype = [
        ('thresholds' , 'f4', len(iou_thresholds)),
        ('precision', 'f4', len(iou_thresholds)),
        ('recall', 'f4', len(iou_thresholds)),
        ('iou', 'f4', len(iou_thresholds)),
    ]

    final_evaluation_results = np.array((
        iou_thresholds,
        average_precision,
        average_recall,
        average_iou
    ), dtype=evaluation_results_dtype)

    def pretty_print_evaluation_results(results):
        print("\nFinal Evaluation Results:")
        for name in results.dtype.names:
            values = results[name]
            print(f"{name.capitalize()}: {['{:.2f}'.format(val) for val in values]}")

    pretty_print_evaluation_results(final_evaluation_results)

    print(f"\nFinal evaluation results dtype: {final_evaluation_results.dtype}")



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

    evaluate_visualize_data(config_data, o3d_visualizer)

if __name__ == "__main__":
    main()
