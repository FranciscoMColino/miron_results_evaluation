import os
import json
import numpy as np
import open3d as o3d
import time

from common.o3d_visualizer import O3dVisualizer
from common.data_control.synthetic_bbox import SyntheticBbox

### Loading data

synthetic_data_path = '/home/digi2/colino_dir/gen_data_ground_truth/figure8_transporter_empty-30fps_800frames-rec1'
    
### Transformations according to the camera coordinate system
"""
Global Position: (-10.000000000540384, 5.0018717613276635, 0.21277073854091316)
Rotation Matrix:
( (0.9999999999999993, 3.42285420007471e-8, 0), (-5.434188106286747e-22, 1.587618925213973e-14, 1), (3.42285420007471e-8, -0.9999999999999993, 1.587618925213974e-14) )
"""

camera_position = np.array([-10.000000000540384, 5.0018717613276635, 0.21277073854091316])

# Define the rotation matrix (3x3)
camera_rotation = np.array([
    [0.9999999999999993, 3.42285420007471e-8, 0],
    [-5.434188106286747e-22, 1.587618925213973e-14, 1],
    [3.42285420007471e-8, -0.9999999999999993, 1.587618925213974e-14]
])

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
translation_vector = -camera_position
rotation_matrix = camera_rotation @ rotation_x_minus_90.T @ rotation_matrix
            
singl_sem_classes = ['transporter1_mesh']

synthetic_bbox = SyntheticBbox(synthetic_data_path, singl_sem_classes=singl_sem_classes,
                                translate_vector=translation_vector, rotation_matrix=rotation_matrix)
synthetic_bbox.setup()
synthetic_bbox.load_labels()
synthetic_bbox.load_data()

data_range = synthetic_bbox.data_range

# initiate np array with size of data_range
complete_pcds = np.empty(data_range[1] - data_range[0] + 1, dtype=list)
complete_bboxes = np.empty(data_range[1] - data_range[0] + 1, dtype=list)
    
for i in range(data_range[0], data_range[1] + 1):
    complete_pcds[i - data_range[0]] = []
    complete_bboxes[i - data_range[0]] = []

    for j, bbox_points in enumerate(synthetic_bbox.complete_bbox_points[i - data_range[0]]):
        
        bbox_points = np.array(bbox_points).astype(np.float64)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(bbox_points).astype(np.float64))
        complete_pcds[i - data_range[0]].append(pcd)
        
        bbox = pcd.get_axis_aligned_bounding_box()
        complete_bboxes[i - data_range[0]].append(bbox)
        
### visualization

o3d_visualizer = O3dVisualizer()
o3d_visualizer.setup()
        
FPS = 30

for i in range(data_range[0], data_range[1] + 1):
    o3d_visualizer.reset()
    
    for geometry in complete_pcds[i - data_range[0]]:
        o3d_visualizer.vis.add_geometry(geometry, reset_bounding_box=False)
        
    for geometry in complete_bboxes[i - data_range[0]]:
        geometry.color = (1, 0, 0)
        o3d_visualizer.vis.add_geometry(geometry, reset_bounding_box=False)
    
    o3d_visualizer.render()
    
    time.sleep(1/FPS)
        

o3d_visualizer.vis.destroy_window()