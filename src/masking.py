import torch
import os
import numpy as np
import random
import argparse
from pathlib import Path


# Define the class for processing the dataset
class FeatureMasking:
    def __init__(self, args):
        self.seed = args.seed
        self.exp = args.exp
        self.latentvar = args.latentvar
        self.latentcls = args.latentcls
        self.group = args.group
        self.relu = args.relu
        self.perc = args.perc
        
        # Set directories for loading and saving data
        self.root_dir = Path('/dhc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D')
        self.svd_dir_heatmap = self.root_dir / 'Explainability' / 'Codes' / 'Two_Test' / 'heatmaps'
        self.svd_dir_image = self.root_dir / 'Explainability' / 'Codes' / 'Two_Test' / 'images'
        self.svd_dir_masked = self.root_dir / 'Explainability' / 'Codes' / 'Two_Test' / 'masked'
        self.svd_dir_masked_least = self.root_dir / 'Explainability' / 'Codes' / 'Two_Test' / 'masked_least'
        self.svd_dir_masked_non_imp = self.root_dir / 'Explainability' / 'Codes' / 'Two_Test' / 'masked_non_imp'
        
        # Create directories if they don't exist
        os.makedirs(self.svd_dir_masked, exist_ok=True)
        os.makedirs(self.svd_dir_masked_least, exist_ok=True)
        os.makedirs(self.svd_dir_masked_non_imp, exist_ok=True)
        
        # Initialize random seed for reproducibility
        self._set_random_seed()

    def _set_random_seed(self):
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(self.seed)
        np.random.seed(self.seed)

    def load_data(self):
        # Load images
        base_path_img = os.path.join(self.svd_dir_image, f'{self.seed}_{self.exp}_{self.latentvar}')
        full_path_imgX = os.path.join(base_path_img, f'{self.latentcls}_X.npy')
        full_path_imgY = os.path.join(base_path_img, f'{self.latentcls}_Y.npy')
        imgs_X = np.load(full_path_imgX)
        imgs_Y = np.load(full_path_imgY)
        print(f'Loaded Images:\nimgs_X:{imgs_X.shape}\nimgs_Y:{imgs_Y.shape}')
        self.imgs = np.vstack((imgs_X, imgs_Y)).squeeze()
        print(f'Merged Images Shape: {self.imgs.shape}')
        
        # Load heatmaps
        base_path = os.path.join(self.svd_dir_heatmap, f'{self.seed}_{self.exp}_{self.latentvar}')
        file_name = f'{self.latentcls}_{self.group}_{self.relu}_heatmap.npy'
        full_path = os.path.join(base_path, file_name)
        self.heatmaps = np.load(full_path)
        print(f'Loaded Heatmaps Shape: {self.heatmaps.shape}')
        
        assert self.imgs.shape == self.heatmaps.shape, "Images and heatmaps have different shapes."
        print("Images and heatmaps have matching shapes!")

    def mask_most_important_features(self, percentage):
        num_images, height, width = self.imgs.shape
        masked_images_array = np.zeros((num_images, height, width), dtype=self.imgs[0].dtype)
        
        for i in range(len(self.heatmaps)):
            heatmap = self.heatmaps[i]
            image = self.imgs[i]
            important_features_mask = heatmap > 0
            positive_scores = heatmap[important_features_mask]
            positive_indices = np.nonzero(important_features_mask)
            sorted_positive_indices = np.argsort(positive_scores)
            num_important_features = len(positive_scores)
            num_elements_to_mask = int(percentage / 100 * num_important_features)
            mask = np.zeros_like(image, dtype=bool)
            
            if num_elements_to_mask > 0:
                indices_to_mask = sorted_positive_indices[-num_elements_to_mask:]
                for idx in indices_to_mask:
                    row, col = positive_indices[0][idx], positive_indices[1][idx]
                    mask[row, col] = True
                    
            masked_image = np.where(mask, 0, image)
            masked_images_array[i] = masked_image
        
        self._save_masked_images(masked_images_array, percentage, self.svd_dir_masked, "most")

    def mask_least_important_features(self, percentage):
        num_images, height, width = self.imgs.shape
        masked_images_least_array = np.zeros((num_images, height, width), dtype=self.imgs[0].dtype)
        
        for i in range(len(self.heatmaps)):
            heatmap = self.heatmaps[i]
            image = self.imgs[i]
            important_features_mask = heatmap > 0
            positive_scores = heatmap[important_features_mask]
            positive_indices = np.nonzero(important_features_mask)
            sorted_positive_indices = np.argsort(positive_scores)
            num_important_features = len(positive_scores)
            num_elements_to_mask = int(percentage / 100 * num_important_features)
            mask_least = np.zeros_like(image, dtype=bool)
            
            if num_elements_to_mask > 0:
                indices_to_mask_least = sorted_positive_indices[:num_elements_to_mask]
                for idx in indices_to_mask_least:
                    row, col = positive_indices[0][idx], positive_indices[1][idx]
                    mask_least[row, col] = True
                    
            masked_image_least = np.where(mask_least, 0, image)
            masked_images_least_array[i] = masked_image_least
            
        self._save_masked_images(masked_images_least_array, percentage, self.svd_dir_masked_least, "least")
        
    def mask_non_important_features(self, percentage):
        num_images, height, width = self.imgs.shape
        masked_images_non_imp_array = np.zeros((num_images, height, width), dtype=self.imgs[0].dtype)
        
        for i in range(len(self.heatmaps)):
            heatmap = self.heatmaps[i]
            image = self.imgs[i]
            non_important_features_mask = heatmap <= 0
            non_important_indices = np.nonzero(non_important_features_mask)
            # Step 2: Calculate how many non-important features to mask
            num_non_important_features = len(non_important_indices[0])
            num_elements_to_mask = int(percentage / 100 * num_non_important_features)
            
            # Step 3: Randomly select non-important features to mask
            if num_elements_to_mask > 0:
                random_indices = np.random.choice(num_non_important_features, size=num_elements_to_mask, replace=False)
                selected_indices = (non_important_indices[0][random_indices], non_important_indices[1][random_indices])
                mask = np.zeros_like(image, dtype=bool)
                for row, col in zip(selected_indices[0], selected_indices[1]):
                    mask[row, col] = True

            # Step 5: Apply the mask to the image
            masked_image = np.where(mask, 0, image)
            masked_images_non_imp_array[i] = masked_image
            
        self._save_masked_images(masked_images_non_imp_array, percentage, self.svd_dir_masked_non_imp, "non_imp")
         
    def _save_masked_images(self, masked_images, percentage, save_dir, mask_type):
        file_base_dir = os.path.join(save_dir, f'{self.seed}_{self.exp}_{self.latentvar}')
        os.makedirs(file_base_dir, exist_ok=True)
        file_name = f'{self.latentcls}_{self.group}_{self.relu}_{percentage}%_mask_{mask_type}_img.npy'
        full_path = os.path.join(file_base_dir, file_name)
        np.save(full_path, masked_images)
        print(f'Saved {percentage}% {mask_type} important features masked images to {full_path}')

parser = argparse.ArgumentParser(description='Masking most/least/non_imp important features of VAE!')
    
# Training parameters
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
parser.add_argument('--exp', type=int, default=4, help='ID of the current experiment!')
    
# Dataset parameters
parser.add_argument('--latentvar', type=str, default='shape', choices=('shape', 'scale', 'orientation', 'position'),
                        help='the latent variable for defining groups!')
parser.add_argument('--latentcls', type=float, nargs="+", default=[1, 2], help='the latent class for defining groups!')
parser.add_argument('--norm', type=bool, default=False, help='If data was normalized during training.')
    
# Heatmap visualizations
parser.add_argument('--group', type=str, default='XY', choices=['X', 'Y', 'XY'],
                        help='Which group to consider for backpropagating the test statistic!')
parser.add_argument('--relu', type=bool, default=True, help='If ReLU applied on heatmaps in Grad-CAM or not!')
parser.add_argument('--start', type=int, default=2, help='The min percentage of features to mask (default: 1%)')
parser.add_argument('--perc', type=int, default=60, help='The max percentage of features to mask (default: 50%)')
parser.add_argument('--mask_type', type=str, default='non_imp', choices=['most', 'least','non_imp'],
                        help='The type of masking: mask most/least/non_imp importnat features!')

args = parser.parse_args()

# Main function to run the class
def main(args):
    feature_masker = FeatureMasking(args)
    
    # Load data (images and heatmaps)
    feature_masker.load_data()
    
    # Mask most important features for varying percentages: 1:args.per% 
    if args.mask_type=='most':
        for percentage in range(args.start,args.perc+1):
            feature_masker.mask_most_important_features(percentage)
    
    # Mask least important features (can change the percentage as needed)
    if args.mask_type=='least':
        feature_masker.mask_least_important_features(args.perc)
    
    # Mask non important features (can change the percentage as needed)    
    if args.mask_type=='non_imp':
        feature_masker.mask_non_important_features(args.perc)


if __name__ == "__main__":

    main(args)
