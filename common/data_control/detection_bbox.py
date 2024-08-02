import os
import numpy as np
import open3d as o3d

class Detection3dBbox():
    def __init__(self, data_path, int_precision=5):
        self.data_path = data_path
        self.int_precision = int_precision
        self.file_indices = []
        self.data_range = (None, None)
        self.complete_points = None
        self.complete_bboxes = None
        self.complete_timestamps = None
        
    def setup(self):
        for filename in os.listdir(self.data_path):
            if filename.startswith("detected_bbox3d_") and filename.endswith(".npy"):
                suffix = filename.split("_")[-1].split(".")[0]
                self.int_precision = max(self.int_precision, len(suffix))
                self.file_indices.append(int(suffix))
        
        if self.file_indices:
            self.data_range = (min(self.file_indices), max(self.file_indices))

        if self.data_range[0] is None:
            raise ValueError("No data found in the specified path or not correct geometry type")
        
        self.complete_points = np.empty(self.data_range[1] - self.data_range[0] + 1, dtype=list)
        self.complete_bboxes = np.empty(self.data_range[1] - self.data_range[0] + 1, dtype=list)
        self.complete_timestamps = np.empty(self.data_range[1] - self.data_range[0] + 1, dtype=dict)
        
    def load_data(self):
        for i in range(self.data_range[0], self.data_range[1] + 1):
            index_str = f"{i:0{self.int_precision}d}"
            bboxes_file_path = os.path.join(self.data_path, f"detected_bbox3d_{index_str}.npy")
            bboxes_data_np = np.load(bboxes_file_path, allow_pickle=True)
            detections_data = bboxes_data_np['detections']
            local_bbox_points = []
            for detections in detections_data:
                id = detections['id'] # 0, not used
                label = detections['label'] # 1, not used
                bbox_points = detections['points'].astype(np.float64) # 2
                bbox_centroid = detections['centroid'].astype(np.float64) # 3, not used
                bbox_transform = detections['transform'].astype(np.float64) # 4, not used
                local_bbox_points.append(np.array(bbox_points).astype(np.float64))
                
            self.complete_points[i - self.data_range[0]] = local_bbox_points
            self.complete_timestamps[i - self.data_range[0]] = {
                'seconds': bboxes_data_np['seconds'],
                'nanoseconds': bboxes_data_np['nanoseconds']
            }

    def compute_bboxes(self):
        detection_data_range = self.data_range

        self.complete_bboxes = np.empty(detection_data_range[1] - detection_data_range[0] + 1, dtype=list)

        for i in range(detection_data_range[0], detection_data_range[1] + 1):
            
            self.complete_bboxes[i - detection_data_range[0]] = []
            
            for points in self.complete_points[i - detection_data_range[0]]:
                points = np.array(points).astype(np.float64)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                bbox = pcd.get_axis_aligned_bounding_box()
                bbox_points = np.asarray(bbox.get_box_points()).astype(np.float64)
                self.complete_bboxes[i - detection_data_range[0]].append(bbox_points)

    def convert_to_2d_boxes(self):
        detection_data_range = self.data_range
        self.complete_2d_bboxes = np.empty(detection_data_range[1] - detection_data_range[0] + 1, dtype=list)

        for i in range(detection_data_range[0], detection_data_range[1] + 1):
            self.complete_2d_bboxes[i - detection_data_range[0]] = []
            for bbox_points in self.complete_bboxes[i - detection_data_range[0]]:
                min_x, max_x = np.min(bbox_points[:, 0]), np.max(bbox_points[:, 0])
                min_y, max_y = np.min(bbox_points[:, 1]), np.max(bbox_points[:, 1])
                bbox_2d = np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
                self.complete_2d_bboxes[i - detection_data_range[0]].append(bbox_2d)

class Detection2dBbox():
    def __init__(self, data_path, int_precision=5):
        self.data_path = data_path
        self.int_precision = int_precision
        self.file_indices = []
        self.data_range = (None, None)
        self.complete_points = None
        self.complete_bboxes = None
        self.complete_timestamps = None
        
    def setup(self):
        for filename in os.listdir(self.data_path):
            if filename.startswith("detected_bbox2d_") and filename.endswith(".npy"):
                suffix = filename.split("_")[-1].split(".")[0]
                self.int_precision = max(self.int_precision, len(suffix))
                self.file_indices.append(int(suffix))
        
        if self.file_indices:
            self.data_range = (min(self.file_indices), max(self.file_indices))

        if self.data_range[0] is None:
            raise ValueError("No data found in the specified path or not correct geometry type")
        
        self.complete_points = np.empty(self.data_range[1] - self.data_range[0] + 1, dtype=list)
        self.complete_bboxes = np.empty(self.data_range[1] - self.data_range[0] + 1, dtype=list)
        self.complete_timestamps = np.empty(self.data_range[1] - self.data_range[0] + 1, dtype=dict)
        
    def load_data(self):
        for i in range(self.data_range[0], self.data_range[1] + 1):
            index_str = f"{i:0{self.int_precision}d}"
            bboxes_file_path = os.path.join(self.data_path, f"detected_bbox2d_{index_str}.npy")
            bboxes_data_np = np.load(bboxes_file_path, allow_pickle=True)
            detections_data = bboxes_data_np['detections']
            local_bbox_points = []
            for detections in detections_data:
                id = detections['id'] # 0, not used
                label = detections['label'] # 1, not used
                bbox_points = detections['points'].astype(np.float64) # 2
                bbox_centroid = detections['centroid'].astype(np.float64) # 3, not used
                bbox_transform = detections['transform'].astype(np.float64) # 4, not used
                local_bbox_points.append(np.array(bbox_points).astype(np.float64))
                
            self.complete_points[i - self.data_range[0]] = local_bbox_points
            self.complete_timestamps[i - self.data_range[0]] = {
                'seconds': bboxes_data_np['seconds'],
                'nanoseconds': bboxes_data_np['nanoseconds']
            }
    
    def compute_bboxes(self): # compute 3d bounding boxes with the assumption that the object is a cuboid
        

        detection_data_range = self.data_range

        self.complete_bboxes = np.empty(detection_data_range[1] - detection_data_range[0] + 1, dtype=list)

        for i in range(detection_data_range[0], detection_data_range[1] + 1):
            
            self.complete_bboxes[i - detection_data_range[0]] = []
            
            for points in self.complete_points[i - detection_data_range[0]]:
                points = np.array(points).astype(np.float64)
                min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
                min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

                dummy_min_z = -0.5 # TODO inverse height of the camera
                dummy_max_z = ((max_x - min_x) + (max_y - min_y)) / 2 + dummy_min_z
                
                points = np.array([[min_x, min_y, dummy_min_z], [min_x, max_y, dummy_min_z], [max_x, max_y, dummy_min_z], [max_x, min_y, dummy_min_z],
                                   [min_x, min_y, dummy_max_z], [min_x, max_y, dummy_max_z], [max_x, max_y, dummy_max_z], [max_x, min_y, dummy_max_z]])
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                bbox = pcd.get_axis_aligned_bounding_box()
                bbox_points = np.asarray(bbox.get_box_points()).astype(np.float64)
                self.complete_bboxes[i - detection_data_range[0]].append(bbox_points) # to avoid the error

        # compute 2d bounding boxes

    def convert_to_2d_boxes(self):
        detection_data_range = self.data_range
        self.complete_2d_bboxes = np.empty(detection_data_range[1] - detection_data_range[0] + 1, dtype=list)
        for i in range(detection_data_range[0], detection_data_range[1] + 1):
            self.complete_2d_bboxes[i - detection_data_range[0]] = []
            
            for bbox_points in self.complete_bboxes[i - detection_data_range[0]]:
                min_x, max_x = np.min(bbox_points[:, 0]), np.max(bbox_points[:, 0])
                min_y, max_y = np.min(bbox_points[:, 1]), np.max(bbox_points[:, 1])
                bbox_2d = np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
                self.complete_2d_bboxes[i - detection_data_range[0]].append(bbox_2d)
        