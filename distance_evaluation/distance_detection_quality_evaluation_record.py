import os
import json
import yaml
import numpy as np
import open3d as o3d
import time
import argparse
import math
import pandas as pd
import datetime

from common.o3d_visualizer import O3dVisualizer
from common.data_control.synthetic_bbox import SyntheticBbox
from common.data_control.detection_bbox import Detection3dBbox
from common.data_control.utils import *

from detection_evaluation.association import *

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

def pretty_str_evaluation_results(results):
    result_str = "\nFinal Evaluation Results:\n"
    for name in results.dtype.names:
        values = results[name]
        result_str += f"{name.capitalize()}: {['{:.2f}'.format(val) for val in values]}\n"
    return result_str

def print_log(message, log_file=None):
    print(message)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

def distance_measure_for_eucdist_match(config_data, o3d_visualizer, verbose=True):

    results_recording_enabled = config_data['results_recording']['enabled']
    results_recording_output_dir = config_data['results_recording']['output_dir']

    logging_file = None

    if results_recording_enabled:
        if not os.path.exists(results_recording_output_dir):
            os.makedirs(results_recording_output_dir)
        elif not os.path.isdir(results_recording_output_dir):
            raise NotADirectoryError(f"Results recording output directory {results_recording_output_dir} is not a directory")
        else:
            raise FileExistsError(f"Results recording output directory {results_recording_output_dir} already exists")
        
        logging_file = os.path.join(results_recording_output_dir, 'log.txt')
        config_copy_file = os.path.join(results_recording_output_dir, 'config_copy.yaml')
        metadata_file = os.path.join(results_recording_output_dir, 'metadata.yaml')
        main_results_file = os.path.join(results_recording_output_dir, 'main_results.csv')

        with open(main_results_file, 'w') as file:
            file.write('sep=,\n')

        index_row = np.array(['frame_nr', 'precision', 'recall', 'translation_error', 'scale_error', 'synthetic_distance', 'detection_distance'])

        main_results_df = pd.DataFrame(columns=index_row)

        with open(config_copy_file, 'w') as file:
            yaml.dump(config_data, file)

        with open(metadata_file, 'w') as file:
            metadata = {
                'date': datetime.datetime.now().strftime("%Y-%m-%d"),
                'time': datetime.datetime.now().strftime("%H:%M:%S")
            }
            yaml.dump(metadata, file)
        
        print_log(f"Results recording enabled. Results will be saved to: {results_recording_output_dir}", logging_file)
    else:
        print_log("Results recording disabled", logging_file)

    # Load from config data to variables
    detection_data_path = config_data['evaluation_data']['detection_path']
    synthetic_data_path = config_data['evaluation_data']['synthetic_path']
    camera_position = np.array(config_data['camera_position'])
    camera_rotation = np.array(config_data['camera_rotation'])
    detection_frame_offset = config_data['evaluation_data']['detection_frame_offset']
    synthetic_frame_offset = config_data['evaluation_data']['synthetic_frame_offset']
    synthetic_fps = config_data['evaluation_data']['synthetic_fps']
    euclidean_distance_thresholds = config_data['euclidean_distance_thresholds']
    euclidean_distance_main_threshold = config_data['euclidean_distance_main_threshold']
    playback_speed = config_data['playback_speed']

    if euclidean_distance_main_threshold not in euclidean_distance_thresholds:
        print_log(f"Main threshold {euclidean_distance_main_threshold} not found in thresholds list, adding it", logging_file)
        euclidean_distance_thresholds.append(euclidean_distance_main_threshold)

    main_threshold_index = euclidean_distance_thresholds.index(euclidean_distance_main_threshold)

    # Load detection data    
    detection_bbox = Detection3dBbox(detection_data_path)
    detection_bbox.setup()
    detection_bbox.load_data()
    detection_bbox.compute_bboxes()
    detection_data_range = detection_bbox.data_range

    # Load synthetic data 
    translation_vector, rotation_matrix = get_translation_and_rotation_from_camera_properties(camera_position, camera_rotation)
    synthetic_bbox = SyntheticBbox(synthetic_data_path,translation_vector=translation_vector, rotation_matrix=rotation_matrix)
    synthetic_bbox.setup()
    synthetic_bbox.load_labels()
    synthetic_bbox.load_data()
    synthetic_bbox.compute_bboxes()
        
    ### Global parameters
    if detection_frame_offset < 0 or synthetic_frame_offset < 0:
        raise ValueError("Frame offsets must be positive")

    o3d_visualizer.setup()
    initial_detection_timestamp = detection_bbox.complete_timestamps[0]
    data_range = detection_data_range
    range_start = detection_data_range[0] + detection_frame_offset
    range_end = detection_data_range[1]
    detection_fps = estimate_detection_frame_rate(detection_bbox.complete_timestamps)
    playback_fps = detection_fps * playback_speed

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
    
    precision_values = [[] for _ in euclidean_distance_thresholds]
    recall_values = [[] for _ in euclidean_distance_thresholds]
    translation_error_values = [[] for _ in euclidean_distance_thresholds]
    scale_error_values = [[] for _ in euclidean_distance_thresholds]
    synthetic_distance_values = [[] for _ in euclidean_distance_thresholds]
    detection_distance_values = [[] for _ in euclidean_distance_thresholds]

    analysed_frames = 0

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
            
            detection_centroids = [get_centroid_from_points(bbox_points) for bbox_points in detection_bbox.complete_bboxes[detection_frame]]
            synthetic_centroids = [get_centroid_from_points(bbox_points) for bbox_points in synthetic_bbox.complete_bboxes[synthetic_frame]]

            detection_2d_centers = np.asarray([centroid[:2] for centroid in detection_centroids])
            synthetic_2d_centers = np.asarray([centroid[:2] for centroid in synthetic_centroids])

            if verbose:
                print_log(f"\nFrame {analysed_frames}", logging_file)

            for j, euclidean_dist_treshold in enumerate(euclidean_distance_thresholds):
                matches, unmatched_detections, unmatched_synthetic, dist_values = associate_euclidean(detection_2d_centers, synthetic_2d_centers, euclidean_dist_treshold)
                true_positives = len(matches)
                false_positives = len(unmatched_detections)
                ground_truth_count = len(synthetic_2d_centers)

                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / ground_truth_count if ground_truth_count > 0 else 0
                translation_error = np.mean(dist_values)

                scale_error_iou_values = []

                for match in matches:
                    detection_bbox_points = detection_bbox.complete_bboxes[detection_frame][match[0]]
                    synthetic_bbox_points = synthetic_bbox.complete_bboxes[synthetic_frame][match[1]]
                    detection_extremes = get_bbox_extremes(detection_bbox_points)
                    synthetic_extremes = get_bbox_extremes(synthetic_bbox_points)
                    detection_center = detection_centroids[match[0]]
                    synthetic_center = synthetic_centroids[match[1]]

                    # translate bboxes to origin
                    detection_extremes[:3] = np.array(detection_extremes[:3]) - detection_center
                    detection_extremes[3:] = np.array(detection_extremes[3:]) - detection_center
                    synthetic_extremes[:3] = np.array(synthetic_extremes[:3]) - synthetic_center
                    synthetic_extremes[3:] = np.array(synthetic_extremes[3:]) - synthetic_center

                    iou_value = iou_3d(detection_extremes, synthetic_extremes)
                    scale_error_iou_values.append(iou_value)

                scale_error = 1 - np.mean(scale_error_iou_values)

                synthetic_distance = np.linalg.norm(synthetic_centroids[0])
                detection_distance = -1

                if len(matches) > 1:
                    print_log(f"Error: More than one match found, this should not happen. Matches: {matches}", logging_file)
                    detection_distance = -np.linalg.norm(detection_centroids[matches[0][0]])
                elif len(matches) == 1:
                    detection_distance = np.linalg.norm(detection_centroids[matches[0][0]])

                precision_values[j].append(precision)
                recall_values[j].append(recall)
                translation_error_values[j].append(translation_error)
                scale_error_values[j].append(scale_error)
                synthetic_distance_values[j].append(synthetic_distance)
                detection_distance_values[j].append(detection_distance)

            if results_recording_enabled:
                main_results_df = pd.concat([main_results_df, pd.DataFrame([[analysed_frames, precision, recall, translation_error, scale_error, synthetic_distance, detection_distance]], columns=index_row)], ignore_index=True)

            if verbose:
                print_log(f"Thresholds: {euclidean_distance_thresholds}", logging_file)
                print_log(f"Precision: {[f'{precision[-1]:.2f}' for precision in precision_values]}", logging_file)
                print_log(f"Recall: {[f'{recall[-1]:.2f}' for recall in recall_values]}", logging_file)
                print_log(f"Translation error: {[f'{translation_error[-1]:.2f}' for translation_error in translation_error_values]}", logging_file)
                print_log(f"Scale error: {[f'{scale_error[-1]:.2f}' for scale_error in scale_error_values]}", logging_file)
                print_log(f"Synthetic distance: {[f'{synthetic_distance[-1]:.2f}' for synthetic_distance in synthetic_distance_values]}", logging_file)
                print_log(f"Detection distance: {[f'{detection_distance[-1]:.2f}' for detection_distance in detection_distance_values]}", logging_file)

            analysed_frames += 1

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

    def filter_array(sub_array):
        sub_array = sub_array[~np.isnan(sub_array)]
        sub_array = sub_array[sub_array >= 0]
        return sub_array
    
    translation_error_values = np.array(translation_error_values)
    translation_error_values = [filter_array(sub_array) for sub_array in translation_error_values]

    scale_error_values = np.array(scale_error_values)
    scale_error_values = [filter_array(sub_array) for sub_array in scale_error_values]

    average_precision = np.array([np.mean(values) for values in precision_values])
    average_recall = np.array([np.mean(values) for values in recall_values])
    average_translation_error = np.array([np.mean(values) for values in translation_error_values])
    average_scale_error = np.array([np.mean(values) for values in scale_error_values])

    avg_frame_iou_collection = avg_frame_iou_collection[:analysed_frames]

    evaluation_results_dtype = [('thresholds' , 'f4', len(euclidean_distance_thresholds)), 
                                ('average_precision', 'f4', len(euclidean_distance_thresholds)),
                                ('average_recall', 'f4', len(euclidean_distance_thresholds)),
                                ('average_translation_error', 'f4', len(euclidean_distance_thresholds)),
                                ('average_scale_error', 'f4', len(euclidean_distance_thresholds))]

    final_evaluation_results = np.array((
        euclidean_distance_thresholds,
        average_precision,
        average_recall,
        average_translation_error,
        average_scale_error
    ), dtype=evaluation_results_dtype)

    print_log(pretty_str_evaluation_results(final_evaluation_results), logging_file)

    if results_recording_enabled:
        main_results_df.to_csv(main_results_file, index=False, mode='a')

    return final_evaluation_results

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

    distance_measure_for_eucdist_match(config_data, o3d_visualizer)

if __name__ == "__main__":
    main()
