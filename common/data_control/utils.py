import numpy as np
import open3d as o3d

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

def get_centroid_from_points(points):
    centroid = np.mean(points, axis=0)
    return centroid

def estimate_detection_frame_rate(complete_detection_timestamps):
    start_time = complete_detection_timestamps[0]
    end_time = complete_detection_timestamps[-1]
    start_seconds = start_time['seconds'] + start_time['nanoseconds'] / 1e9
    end_seconds = end_time['seconds'] + end_time['nanoseconds'] / 1e9
    num_frames = len(complete_detection_timestamps)
    detection_fps = num_frames / (end_seconds - start_seconds)
    return detection_fps