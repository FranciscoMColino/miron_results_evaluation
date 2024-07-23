import open3d as o3d
import numpy as np

class O3dVisualizer:
    
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window('Open3D', width=640, height=480)
    
    def setup(self):
        points = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [10, 0, 0],
            [10, 0, 1],
            [10, 1, 0],
            [10, 1, 1]
        ])

        points *= 4

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        self.vis.add_geometry(pcd, reset_bounding_box=True)
        
        view_control = self.vis.get_view_control()
        view_control.rotate(0, -525)
        view_control.rotate(500, 0)
        
        self.vis.get_render_option().point_size = 2.0
        self.vis.get_render_option().line_width = 10.0
        
    def reset(self):
        self.vis.clear_geometries()
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0), reset_bounding_box=False)
        
    def render(self):
        self.vis.poll_events()
        self.vis.update_renderer()
        