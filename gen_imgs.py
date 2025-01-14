import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

class OBJVisualizer:
    def __init__(self, obj_file_path):
        self.obj_file_path = obj_file_path
        self.points, self.faces = self._load_points_and_faces()

    def _load_points_and_faces(self):
        points = []
        faces = []
        with open(self.obj_file_path, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    _, x, y, z = line.split()
                    points.append([float(x), float(y), float(z)])
                elif line.startswith('f '):
                    face = [int(idx.split('/')[0]) - 1 for idx in line.split()[1:]]
                    faces.append(face)
        return np.array(points), faces

    def _normalize_points(self):
        max_vals = np.max(self.points, axis=0)
        min_vals = np.min(self.points, axis=0)
        center = (max_vals + min_vals) / 2
        self.points -= center
        scale = np.max(max_vals - min_vals)
        self.points /= scale

        # Rotate the points to adjust orientation (flip to face the viewer)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        self.points = self.points.dot(rotation_matrix)

    def _create_frame(self, points, angle):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        ax.view_init(elev=10, azim=angle)
        plt.axis('off')
        temp_file_path = f'./temp_frame_{angle}.png'
        plt.savefig(temp_file_path)
        plt.close()
        return temp_file_path

    def _create_frame_with_surface(self, points, faces, angle):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface using trisurf
        ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=faces, color='lightblue', edgecolor='gray', alpha=0.8)

        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        ax.view_init(elev=10, azim=angle)

        plt.axis('off')
        temp_file_path = f'./temp_frame_{angle}_textured.png'
        plt.savefig(temp_file_path)
        plt.close()
        return temp_file_path

    def create_gif(self):
        self._normalize_points()
        obj_name = os.path.splitext(os.path.basename(self.obj_file_path))[0]
        output_dir = f'./final_outputs/{obj_name}'
        os.makedirs(output_dir, exist_ok=True)

        frames = []
        for angle in range(0, 360, 5):
            frame_path = self._create_frame(self.points, angle)
            frames.append(Image.open(frame_path))

        gif_path = os.path.join(output_dir, f'{obj_name}.gif')
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

        # Clean up temporary frames
        for frame in frames:
            os.remove(frame.filename)

        print(f"GIF saved at {gif_path}")

    def create_gif_with_surface(self):
        self._normalize_points()
        obj_name = os.path.splitext(os.path.basename(self.obj_file_path))[0]
        output_dir = f'./final_outputs/{obj_name}'
        os.makedirs(output_dir, exist_ok=True)

        frames = []
        for angle in range(0, 360, 5):
            frame_path = self._create_frame_with_surface(self.points, self.faces, angle)
            frames.append(Image.open(frame_path))

        gif_path = os.path.join(output_dir, f'{obj_name}_textured.gif')
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

        # Clean up temporary frames
        for frame in frames:
            os.remove(frame.filename)

        print(f"Textured GIF saved at {gif_path}")





