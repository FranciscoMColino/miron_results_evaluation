import os
import json
import numpy as np
import open3d as o3d
import time

from common.o3d_visualizer import O3dVisualizer
from common.data_control.detection_bbox import DetectionBbox

detection_data_path = '/home/digi2/colino_dir/detection_data/ermis_detection/detection_res_2'

detection_bbox = DetectionBbox(detection_data_path, int_precision=5)
detection_bbox.setup()
detection_bbox.load_data()

data_range = detection_bbox.data_range

complete_pcds = np.empty(data_range[1] - data_range[0] + 1, dtype=list)
complete_bboxes = np.empty(data_range[1] - data_range[0] + 1, dtype=list)
complete_timestamps = np.empty(data_range[1] - data_range[0] + 1, dtype=dict)

for i in range(data_range[0], data_range[1] + 1):
    
    local_pcds = []
    local_bboxes = []
    
    for bbox_points in detection_bbox.complete_bbox_points[i - data_range[0]]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(bbox_points)
        local_pcds.append(pcd)
        
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
        local_bboxes.append(bbox)
        
    complete_pcds[i - data_range[0]] = local_pcds
    complete_bboxes[i - data_range[0]] = local_bboxes
    complete_timestamps[i - data_range[0]] = detection_bbox.complete_timestamps[i - data_range[0]]
    
# estimate fps from first and last timestamps and number of frames
if complete_timestamps.all():
    start_time = complete_timestamps[0]
    end_time = complete_timestamps[-1]
    start_seconds = start_time['seconds'] + start_time['nanoseconds'] / 1e9
    end_seconds = end_time['seconds'] + end_time['nanoseconds'] / 1e9
    num_frames = len(complete_timestamps)
    fps = num_frames / (end_seconds - start_seconds)
print(f"Estimated FPS: {fps}")
    

### visualization

o3d_visualizer = O3dVisualizer()
o3d_visualizer.setup()

for i in range(data_range[0], data_range[1] + 1):
    o3d_visualizer.reset()
    
    for geometry in complete_pcds[i - data_range[0]]:
        o3d_visualizer.vis.add_geometry(geometry, reset_bounding_box=False)
        
    for geometry in complete_bboxes[i - data_range[0]]:
        geometry.color = (1, 0, 0)
        o3d_visualizer.vis.add_geometry(geometry, reset_bounding_box=False)
    
    o3d_visualizer.render()
    
    time.sleep(1/fps)
        

o3d_visualizer.vis.destroy_window()
    
    