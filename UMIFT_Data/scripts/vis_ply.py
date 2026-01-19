import open3d as o3d

# Load the .ply file
ply_file = "/Users/chuerpan/Documents/repo/umiFT/umi_day_data/20241205-cup/2024-12-05/2024-12-05T05-29-02.992Z_18612_chuer-1204-cup_demonstration/colmap_pcd/sparse_aligned_3dgs/points3D.ply"
mesh = o3d.io.read_triangle_mesh(ply_file)

# Visualize the mesh
o3d.visualization.draw_geometries([mesh])
