import os
import numpy as np
import json
import open3d as o3d

class SyntheticBbox():
    def __init__(self, data_path, int_precision=4, 
                 multi_sem_classes=None, translation_vector=None, rotation_matrix=None):
        self.data_path = data_path
        self.int_precision = int_precision
        self.file_indices = []
        self.data_range = (None, None)
        self.complete_bbox_points = None
        self.complete_labels = None
        self.single_sem_classes = None
        self.multi_sem_classes = multi_sem_classes
        self.translation_vector = translation_vector
        self.rotation_matrix = rotation_matrix
        
        
    def setup(self):
        for filename in os.listdir(self.data_path):
            if filename.startswith("bounding_box_3d_") and filename.endswith(".npy"):
                suffix = filename.split("_")[-1].split(".")[0]
                self.int_precision = max(self.int_precision, len(suffix))
                self.file_indices.append(int(suffix))
        
        if self.file_indices:
            self.data_range = (min(self.file_indices), max(self.file_indices))
        
        self.complete_bbox_points = np.empty(self.data_range[1] - self.data_range[0] + 1, dtype=list)
        self.complete_labels = np.empty(self.data_range[1] - self.data_range[0] + 1, dtype=dict)
        
    def load_labels(self):
        for i in range(self.data_range[0], self.data_range[1] + 1):
            index_str = f"{i:0{self.int_precision}d}"
            labels_file_path = os.path.join(self.data_path, f"bounding_box_3d_labels_{index_str}.json")
            if os.path.exists(labels_file_path):
                with open(labels_file_path, 'r') as labels_file:
                    labels_data = json.load(labels_file)
                    self.complete_labels[i - self.data_range[0]] = labels_data
            else:
                self.complete_labels[i - self.data_range[0]] = None
                
        # find all unique labels
        all_labels = set()
        for labels in self.complete_labels:
            if labels:
                for label in labels.values():
                    all_labels.add(label['class'])

        self.single_sem_classes = list(all_labels - set(self.multi_sem_classes))
                
    def load_data(self):
        for i in range(self.data_range[0], self.data_range[1] + 1):
            index_str = f"{i:0{self.int_precision}d}"
            bbox3d_data_file_path = os.path.join(self.data_path, f"bounding_box_3d_{index_str}.npy")
            bbox3d_data = np.load(bbox3d_data_file_path)
            local_bbox_points = []
            
            single_sem_points = {}
            for class_name in self.single_sem_classes:
                single_sem_points[class_name] = [] # list of points
                
            multi_sem_points = {}
            for class_name in self.multi_sem_classes:
                multi_sem_points[class_name] = [] # list of list of points
            
            for row in bbox3d_data:
                class_id = str(row[0])
                class_name = self.complete_labels[i - self.data_range[0]][class_id]['class']
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
                pcd.translate(self.translation_vector)
                pcd.rotate(self.rotation_matrix, center=(0, 0, 0))
                points = np.asarray(pcd.points)
                
                if class_name in self.single_sem_classes:
                    single_sem_points[class_name].extend(points)
                elif class_name in self.multi_sem_classes:
                    multi_sem_points[class_name].append(points)
                    
            for class_name, points in single_sem_points.items():
                local_bbox_points.append(points)
            
            for class_name, points_list in multi_sem_points.items():
                local_bbox_points.extend(points_list)
            
            self.complete_bbox_points[i - self.data_range[0]] = local_bbox_points
            