### Import libraries
import torch
import os
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from utils import overlay_heatmap_single
import argparse

class HeatmapOverlay:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = torch.cuda.is_available()

        # Directories
        self.root_dir = Path('/dhc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D')
        self.svd_split = self.root_dir / 'Explainability' / 'Reproducibility' / 'data'
        self.svd_dir_heatmap = self.root_dir / 'Explainability' / 'Codes' / 'Two_Test' / 'heatmaps'
        self.svd_dir_image = self.root_dir / 'Explainability' / 'Codes' / 'Two_Test' / 'images'
        self.svd_dir_overlay = self.root_dir / 'Explainability' / 'Codes' / 'Two_Test' / 'overlays'

        # Initialize data
        self.group1_images_np, self.group2_images_np = self.load_images()
        self.heatmaps = self.load_heatmaps()
        self.overlay_dir = self.prepare_overlay_directory()

    def load_images(self):
        """Loads the images for group X and group Y using memory mapping for efficiency."""
        base_path_img = os.path.join(self.svd_dir_image, f'{self.args.seed}_{self.args.exp}_{self.args.latentvar}')
        file_name_imgX = f'{self.args.latentcls}_X.npy'
        file_name_imgY = f'{self.args.latentcls}_Y.npy'
        full_path_imgX = os.path.join(base_path_img, file_name_imgX)
        full_path_imgY = os.path.join(base_path_img, file_name_imgY)
        
        # Memory-mapping for large datasets
        group1_images_np = np.load(full_path_imgX, mmap_mode='r')
        group2_images_np = np.load(full_path_imgY, mmap_mode='r')

        print(f'imgX shape: {group1_images_np.shape}')
        print(f'imgY shape: {group2_images_np.shape}')
        return group1_images_np, group2_images_np

    def load_heatmaps(self):
        """Loads the heatmaps using memory mapping."""
        base_path = os.path.join(self.svd_dir_heatmap, f'{self.args.seed}_{self.args.exp}_{self.args.latentvar}')
        file_name = f'{self.args.latentcls}_{self.args.group}_{self.args.relu}'
        if self.args.attention:
            file_name += f'_{self.args.attention}'
        file_name += '_heatmap.npy'    
        full_path = os.path.join(base_path, file_name)
        
        # Memory-mapping for large heatmaps
        heatmaps = np.load(full_path, mmap_mode='r')
        return heatmaps

    def prepare_overlay_directory(self):
        """Creates the overlay directory and returns its path."""
        base_path_ov = os.path.join(self.svd_dir_overlay, f'{self.args.seed}_{self.args.exp}_{self.args.latentvar}')
        os.makedirs(base_path_ov, exist_ok=True)

        # Create directories based on ReLU application
        relu_path = f'relu_{self.args.relu}'
        if self.args.attention:
            relu_path += f'_{self.args.attention}'
        base_path_ov_relu = os.path.join(base_path_ov, relu_path)
        print(base_path_ov_relu)
        os.makedirs(base_path_ov_relu, exist_ok=True)
        return base_path_ov_relu

    def overlay_heatmaps_all(self, images, heatmaps):
        """Overlays heatmaps on all the images concurrently using ThreadPoolExecutor."""
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(lambda idx: overlay_heatmap_single(images, heatmaps, self.overlay_dir, idx), 
                                   range(images.shape[0])), total=images.shape[0]))
            print("All heatmaps have been overlayed.")

    def process_heatmaps(self):
        """Handles the main logic of overlaying heatmaps based on user input."""
        if self.args.group == 'XY':
            # We only stack if necessary, but better to avoid large array stacking
            images_np = np.vstack((self.group1_images_np, self.group2_images_np))
            print(f'Images combined shape: {images_np.shape}')

            if self.args.all:
                # Overlay heatmaps for all images (concurrently)
                self.overlay_heatmaps_all(images_np, self.heatmaps)
            elif self.args.subset:
                # Overlay heatmaps for a subset of images
                for img_idx in tqdm(self.args.subset, desc="Overlaying Subset"):
                    print(f'overlay_dir:{self.overlay_dir}')
                    overlay_heatmap_single(images_np, self.heatmaps, self.overlay_dir, img_idx)
                    print(f'Overlay heatmap {img_idx} was done!')
            else:
                # Overlay heatmap for a single image
                overlay_heatmap_single(images_np, self.heatmaps, self.overlay_dir, self.args.img_idx)
                print(f'Overlay heatmap {self.args.img_idx} was done!')

parser = argparse.ArgumentParser(description='Visualizing Two-Sample Test VAE')

# Training parameters
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
parser.add_argument('--exp', type=int, default=4, help='ID of the current experiment!')
parser.add_argument('--bs', type=int, default=128, help='input batch size for training (default: 128)')

# Dataset parameters
parser.add_argument('--latentvar', type=str, default='shape', choices=('shape', 'scale', 'orientation', 'position'),
                        help='the latent variable for defining groups!')
parser.add_argument('--latentcls', type=float, nargs="+", default=[1, 2], help='the latent class for defining groups!')
parser.add_argument('--norm', type=bool, default=False, help='If we want to normalize data or not.')

# Heatmap visualizations
parser.add_argument('--group', type=str, default='XY', choices=['X', 'Y', 'XY'],
                        help='Which group we considered for backpropagating test statistic!')
parser.add_argument('--relu', type=bool, default=True, help='If ReLU is applied on heatmaps in Grad-CAM or not!')
parser.add_argument('--attention', type=str, default='spatial', choices = ('spatial','channel'),
                    help='If we want to apply attention or not!')
parser.add_argument('--all', type=bool, default=False, help='If True, overlay heatmaps of all images over them!')
parser.add_argument('--subset', type=int, nargs='*', default=[0,1,2,3,4,5],
                        help='Showing a subset of overlaid heatmaps for images!')
parser.add_argument('--img_idx', type=int, default=62000, help='The index of image that we want to overlay heatmap on!')

args = parser.parse_args()

if __name__ == '__main__':

    # Create instance of HeatmapOverlay class and process the heatmaps
    heatmap_overlay = HeatmapOverlay(args)
    heatmap_overlay.process_heatmaps()
