import os
import sys
import h5py
import cv2
import numpy as np
import click

ROOT_DIR = os.getcwd()     # /home/mkdusgml/carla_ws/opal

@click.command()
@click.option("-d", '--dataset_name', type=str, required=True, default='0007')
@click.option("-v", '--view', type=str, required=True, default='front')
def visualize_dataset(dataset_name, view):
    dataset_path = os.path.join(ROOT_DIR, f'data/{dataset_name}.hdf5')
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist")
        sys.exit(1)
    
    frames = []
    with h5py.File(dataset_path, 'r') as f:
        for i in range(len(list(f.keys()))):
            if view == 'front':
                frames.append(f[f'step_{i:04d}']['obs']['central_rgb']['data'][:])
            elif view == 'birdview':
                frames.append(f[f'step_{i:04d}']['obs']['birdview']['rendered'][:])
            else:
                print(f"Error: Invalid view {view}")
                sys.exit(1)
    
    video_name = os.path.join(ROOT_DIR, f'data/video/{dataset_name}_{view}.mp4')
    h, w, c = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_name, fourcc, 10, (w, h))
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    video_writer.release()
    
    print(f"Video saved to {video_name}")
    
if __name__ == '__main__':
    visualize_dataset()