import os
import json
import numpy as np
import open3d as o3d

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
            
            
            
multi_sem_classes = ['swetfloorsign']
singl_sem_classes = ['transporter1_mesh']

# initiate np array with size of data_range
complete_geometries = np.empty(data_range[1] - data_range[0] + 1, dtype=list)


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
        
        if class_name in singl_sem_classes:
            singl_sem_classes_pcds[class_name].points.extend(pcd.points)
        elif class_name in multi_sem_classes:
            multi_sem_classes_pcds[class_name].append(pcd)
            
    geometries = []
    
    for class_name in singl_sem_classes:
        geometries.append(singl_sem_classes_pcds[class_name])
        
    for class_name in multi_sem_classes:
        geometries.extend(multi_sem_classes_pcds[class_name])
        
    complete_geometries[i - data_range[0]] = geometries
            
        
        
        
        
    
    
