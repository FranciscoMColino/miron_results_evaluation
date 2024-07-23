import os
import json
import numpy as np
import open3d as o3d
import time

from common.o3d_visualizer import O3dVisualizer

### Loading data

synthetic_data_path = '/home/digi2/colino_dir/gen_data_ground_truth/figure8_transporter_empty-30fps_800frames-rec1'

# Initialize a list to store the numeric parts of the filenames
file_indices = []
int_precision = 0

# Iterate over the files in the directory
for filename in os.listdir(synthetic_data_path):
    if filename.startswith("bounding_box_3d_") and filename.endswith(".npy"):
        # Extract the numeric part of the filename and convert it to an integer
        suffix = filename.split("_")[-1].split(".")[0]
        int_precision = max(int_precision, len(suffix))
        file_indices.append(int(suffix))

# Calculate the range of the data points
if file_indices:
    data_range = (min(file_indices), max(file_indices))
else:
    data_range = (None, None)

print(f"Range of data points: {data_range}")

if data_range[0] is not None and data_range[1] is not None:
    for i in range(data_range[0], data_range[1] + 1):
        # Format the index with leading zeros based on the detected precision
        index_str = f"{i:0{int_precision}d}"
        
        # Construct the file paths using the formatted index
        labels_file_path = os.path.join(synthetic_data_path, f"bounding_box_3d_labels_{index_str}.json")
        
        if os.path.exists(labels_file_path):
            with open(labels_file_path, 'r') as labels_file:
                labels_data = json.load(labels_file)
else:
    print("No data points found")
    exit(1)
            

### Transformation matrix according to the camera coordinate system
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

# Create the 4x4 transformation matrix
transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = rotation_matrix
transformation_matrix[:3, 3] = translation_vector

            
multi_sem_classes = ['swetfloorsign']
singl_sem_classes = ['transporter1_mesh']

# initiate np array with size of data_range
complete_pcds = np.empty(data_range[1] - data_range[0] + 1, dtype=list)
complete_bboxes = np.empty(data_range[1] - data_range[0] + 1, dtype=list)
    

for i in range(data_range[0], data_range[1] + 1):
    # Format the index with leading zeros based on the detected precision
    index_str = f"{i:0{int_precision}d}"

    # Construct the file paths using the formatted index
    labels_file_path = os.path.join(synthetic_data_path, f"bounding_box_3d_labels_{index_str}.json")

    if os.path.exists(labels_file_path):
        with open(labels_file_path, 'r') as labels_file:
            labels_data = json.load(labels_file)
            
    bbox3d_data_file_path = os.path.join(synthetic_data_path, f"bounding_box_3d_{index_str}.npy")
    bbox3d_data = np.load(bbox3d_data_file_path)
    
    pcd_geometries = []
    
    singl_sem_classes_pcds = {}
    
    for class_name in singl_sem_classes:
        singl_sem_classes_pcds[class_name] = o3d.geometry.PointCloud()
    
    multi_sem_classes_pcds = {}
    
    for class_name in multi_sem_classes:
        multi_sem_classes_pcds[class_name] = []
    
    for row in bbox3d_data:
        
        class_id = str(row[0])
        class_name = labels_data[class_id]['class']
        
        x_min = row[1]
        y_min = row[2]
        x_max = row[3]
        y_max = row[4]
        z_min = row[5]
        z_max = row[6]
        
        points = [
            [x_min, y_min, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_min],
            [x_min, y_max, z_max],
            [x_max, y_min, z_min],
            [x_max, y_min, z_max],
            [x_max, y_max, z_min],
            [x_max, y_max, z_max]
        ]
        
        transform = row[7].T
        
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points).astype(np.float64))
        pcd.transform(np.array(transform).astype(np.float64))
        #
        pcd.translate(translation_vector)
        pcd.rotate(rotation_matrix, center=(0, 0, 0))
            
        if class_name in singl_sem_classes:
            singl_sem_classes_pcds[class_name].points.extend(pcd.points)
        elif class_name in multi_sem_classes:
            multi_sem_classes_pcds[class_name].append(pcd)
            
    for class_name in singl_sem_classes:
        pcd_geometries.append(singl_sem_classes_pcds[class_name])
        
    for class_name in multi_sem_classes:
        pcd_geometries.extend(multi_sem_classes_pcds[class_name])
        
    bboxes_geometries = []
    
    for pcd in pcd_geometries:
        bboxes_geometries.append(pcd.get_axis_aligned_bounding_box())
        
        
    complete_pcds[i - data_range[0]] = pcd_geometries
    complete_bboxes[i - data_range[0]] = bboxes_geometries
            
        
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