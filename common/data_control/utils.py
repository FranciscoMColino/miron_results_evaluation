import numpy as np
import open3d as o3d

def extract_complete_data_from_detection_bboxes(detector_bbox):
    detection_data_range = detector_bbox.data_range

    complete_detection_pcds = np.empty(detection_data_range[1] - detection_data_range[0] + 1, dtype=list)
    complete_detection_bboxes = np.empty(detection_data_range[1] - detection_data_range[0] + 1, dtype=list)
    complete_detection_timestamps = np.empty(detection_data_range[1] - detection_data_range[0] + 1, dtype=dict)

    for i in range(detection_data_range[0], detection_data_range[1] + 1):
        
        local_pcds = []
        local_bboxes = []
        
        for bbox_points in detector_bbox.complete_bbox_points[i - detection_data_range[0]]:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(bbox_points)
            pcd.paint_uniform_color([1, 0, 1])
            local_pcds.append(pcd)
            
            bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
            bbox.color = (1, 0, 0)
            local_bboxes.append(bbox)
            
        complete_detection_pcds[i - detection_data_range[0]] = local_pcds
        complete_detection_bboxes[i - detection_data_range[0]] = local_bboxes
        complete_detection_timestamps[i - detection_data_range[0]] = detector_bbox.complete_timestamps[i - detection_data_range[0]]

    return complete_detection_pcds, complete_detection_bboxes, complete_detection_timestamps


def get_translation_and_rotation_from_camera_properties(camera_position, camera_rotation):
    rotation_x_minus_90 = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])

    # rotation over Z for angle
    rotation_angle = -90
    rotation_matrix = np.array([
        [np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle)), 0],
        [np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle)), 0],
        [0, 0, 1]
    ])

    # Define the translation vector (3x1)
    translation_vector = -camera_position
    #rotation_matrix = camera_rotation @ rotation_x_minus_90 @ rotation_matrix # use for LeftCamera rotation
    rotation_matrix = camera_rotation @ rotation_x_minus_90.T @ rotation_matrix.T # use for ZED_X rotation


    return translation_vector, rotation_matrix

def extract_complete_data_from_synthetic_bboxes(synthetic_bbox):
    synthetic_data_range = synthetic_bbox.data_range

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

    return complete_synthetic_pcds, complete_synthetic_bboxes

def estimate_detection_frame_rate(complete_detection_timestamps):
    start_time = complete_detection_timestamps[0]
    end_time = complete_detection_timestamps[-1]
    start_seconds = start_time['seconds'] + start_time['nanoseconds'] / 1e9
    end_seconds = end_time['seconds'] + end_time['nanoseconds'] / 1e9
    num_frames = len(complete_detection_timestamps)
    detection_fps = num_frames / (end_seconds - start_seconds)
    return detection_fps