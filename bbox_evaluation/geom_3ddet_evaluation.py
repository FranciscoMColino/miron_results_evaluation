import os
import json
import yaml
import numpy as np
import open3d as o3d
import time
import argparse
import math

from common.data_control.synthetic_bbox import SyntheticBbox
from common.data_control.detection_bbox import Detection3dBbox
from common.data_control.utils import *

from bbox_evaluation.association import *

def get_bbox_extremes(bbox):
    bbox = np.array(bbox)
    min_coords = np.min(bbox, axis=0)
    max_coords = np.max(bbox, axis=0)
    return [min_coords[0], min_coords[1], min_coords[2], max_coords[0], max_coords[1], max_coords[2]]

def geom_evaluate_3ddet_data(config_data, verbose=False):

    # Load from config data to variables
    detection_data_path = config_data['detection_path']
    synthetic_data_path = config_data['synthetic_path']
    camera_position = np.array(config_data['camera_position'])
    camera_rotation = np.array(config_data['camera_rotation'])
    detection_frame_offset = config_data['detection_frame_offset']
    synthetic_frame_offset = config_data['synthetic_frame_offset']
    synthetic_fps = config_data['synthetic_fps']
    iou_thresholds = config_data['iou_thresholds']

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

    initial_detection_timestamp = detection_bbox.complete_timestamps[0]
    data_range = detection_data_range
    range_start = detection_data_range[0] + detection_frame_offset
    range_end = detection_data_range[1]

    # evaluation results should hold the thresholds used, an array with the results for each frame, and the average results
    
    frame_results_dtype = [('precision', 'f4'), ('recall', 'f4'), ('iou', 'f4')]
    results_collection = np.zeros((len(detection_bbox.complete_timestamps), len(iou_thresholds)), dtype=frame_results_dtype)

    analysed_frames = 0
    match_ocurred_per_threshould = np.zeros((len(iou_thresholds),), dtype=int)

    avg_frame_iou_collection = np.zeros((len(detection_bbox.complete_timestamps),), dtype='f4')

    for i in range(range_start, range_end + 1):
        start_time = time.time()

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
        
        if verbose:
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
            if verbose:
                print(f"Threshold: {iou_threshold}, Precision: {precision:.2f}, Recall: {recall:.2f}, IoU: {iou:.2f}")

        
        results_collection[analysed_frames] = frame_results
        analysed_frames += 1

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

    if verbose:
        def pretty_print_evaluation_results(results):
            print("\nFinal Evaluation Results:")
            for name in results.dtype.names:
                values = results[name]
                print(f"{name.capitalize()}: {['{:.2f}'.format(val) for val in values]}")

        pretty_print_evaluation_results(final_evaluation_results)

    if verbose:
        print(f"\nFinal evaluation results dtype: {final_evaluation_results.dtype}")

    return final_evaluation_results



def main():
    parser = argparse.ArgumentParser(description='Benchmark detection against synthetic data')
    parser.add_argument('config_file', type=str, help='Path to the config file')
    args = parser.parse_args()
    
    # config file in yaml format
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"Config file {args.config_file} not found")
    
    with open(args.config_file, 'r') as file:
        config_data = yaml.safe_load(file)

    geom_evaluate_3ddet_data(config_data, verbose=True)

if __name__ == "__main__":
    main()
