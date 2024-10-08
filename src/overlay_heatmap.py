### import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import numpy as np
import pandas as pd
from pathlib import Path
import random
import argparse
try:
    import cPickle as pickle
except:
    import pickle

from dataloader import dSprites
from utils import overlay_heatmap_single


def overlay_heatmaps_all(images, heatmaps, overlay_dir):    
    for img_idx in range(images.shape[0]):
        # Save each overlaid heatmap
        full_path = os.path.join(overlay_dir, f'overlay_{img_idx}.png')
        overlay_heatmap_single(images, heatmaps, overlay_dir, img_idx)
        print(f'Overlay heatmap {img_idx} was done!')


parser = argparse.ArgumentParser(description='Visualaizing Two-Sample Test VAE')

# Training parameters
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--exp',type=int,default=4,
                    help='ID of the current expriment!')
parser.add_argument('--bs', type=int, default=128,
                    help='input batch size for training (default: 128)')
# Dataset parameters
parser.add_argument('--latentvar',type=str,default='shape',
                    choices=('shape','scale','orientation','position'),
                    help='the latent varibale for defining groups!')
parser.add_argument('--latentcls',type=float,nargs="+",default=[1,2],
                    help='the latent class for defining groups!')
parser.add_argument('--norm', type=bool, default=False,
                    help='If we want to normalize data or not.')
# Heatmap visualizations
parser.add_argument('--group', type=str, default='XY',choices=['X','Y','XY'],
                    help='Which group we considered for backprobagating test statistic!')
parser.add_argument('--relu', type=bool, default=False,
                   help='If relu applied on heatmaps in Gradcam or not!')
parser.add_argument('--all', type=bool,default=False,
                    help='If True, we overlay heatmaps of all images over them!')
parser.add_argument('--subset', type=int,nargs='*',default=[150000,150100,150200,150300,150400],
                    help='Showing a subset of overlaid heatmaps for images!')
parser.add_argument('--img_idx', type=int,default=20,
                    help='The index of image that we want to overlay heatmap on it!')

args = parser.parse_args()

def main(args):
    
    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    
    
    root_dir = Path('/dhc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D')
    svd_split = root_dir/'Explainability'/'Reproducibility'/'data'
    svd_dir_heatmap = root_dir / 'Explainability' / 'Codes' / 'Two_Test' / 'heatmaps'
    svd_dir_image = root_dir / 'Explainability' / 'Codes' / 'Two_Test' / 'images'
    svd_dir_overlay = root_dir / 'Explainability' / 'Codes' / 'Two_Test' / 'overlays'
    
        
    # loading images 
    base_path_img = os.path.join(svd_dir_image, f'{args.seed}_{args.exp}_{args.latentvar}')
    file_name = f'{args.latentcls}_{args.group}_{args.relu}_heatmap.npy'
    file_name_imgX = f'{args.latentcls}_X.npy'
    file_name_imgY = f'{args.latentcls}_Y.npy'
    full_path_imgX = os.path.join(base_path_img, file_name_imgX)
    full_path_imgY = os.path.join(base_path_img, file_name_imgY)
    group1_images_np = np.load(full_path_imgX)
    group2_images_np = np.load(full_path_imgY)
    
    print(f'imgX shape:{group1_images_np.shape}')
    print(f'imgY shape:{group2_images_np.shape}')
    
    
    # loading heatmaps
    base_path = os.path.join(svd_dir_heatmap, f'{args.seed}_{args.exp}_{args.latentvar}')
    file_name = f'{args.latentcls}_{args.group}_{args.relu}_heatmap.npy'
    full_path = os.path.join(base_path, file_name)
    heatmaps = np.load(full_path)
    
    # overlay_dir 
    base_path_ov = os.path.join(svd_dir_overlay, f'{args.seed}_{args.exp}_{args.latentvar}')
    os.makedirs(base_path_ov,exist_ok=True)
    # make two directories based on applying Relu
    base_path_ov_relu = os.path.join(base_path_ov, f'relu_{args.relu}')
    os.makedirs(base_path_ov_relu,exist_ok=True)
        
    if args.group == 'XY':
        images_np = np.vstack((group1_images_np,group2_images_np))
        print(f'images:{images_np.shape}')
        if args.all:
            # Overlay heatmaps for all images
            overlay_heatmaps_all(images_np, heatmaps, base_path_ov_relu)
        elif args.subset:
            # Overlay heatmap for subset of images
            for img_idx in args.subset:
                overlay_heatmap_single(images_np, heatmaps, base_path_ov_relu, img_idx)
                print(f'Overlay heatmap {img_idx} was done!')
        else:
            # Overlay heatmap for single image
            overlay_heatmap_single(images_np, heatmaps, base_path_ov_relu, args.img_idx)
            print(f'Overlay heatmap {args.img_idx} was done!')

            
                 
    
if __name__=='__main__':
    main(args)
    
    
    
    
    