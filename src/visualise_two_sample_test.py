### import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import random
import os
import argparse
try:
    import cPickle as pickle
except ImportError:
    import pickle

from dataloader import dSprites
from model import VAE
from gradcam import GradCAM

class TestStatisticBackprop:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.latent_variable_mapping = {
            'color': 0,
            'shape': 1,
            'scale': 2,
            'orientation': 3,
            'position X': 4,
            'position Y': 5
        }
        self.latent_idx = self.latent_variable_mapping[args.latentvar]
        self.setup_experiment()

    def setup_experiment(self):
        """Set random seeds, directories, and test loader."""
        seed = self.args.seed
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        
        # Directory for loading checkpoints and saving outputs
        self.root_dir = Path('/dhc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D')
        self.svd_split = self.root_dir / 'Explainability' / 'Reproducibility' / 'data'
        self.save_dir_ckpts = self.root_dir / 'Explainability' / 'Codes' / 'ckpts'
        self.svd_dir_heatmap = self.root_dir / 'Explainability' / 'Codes' / 'Two_Test' / 'heatmaps'
        self.svd_dir_image = self.root_dir / 'Explainability' / 'Codes' / 'Two_Test' / 'images'
        
        self.load_test_set()

    def load_test_set(self):
        """Load test dataset."""
        test_set_path = self.svd_split / f'test_set_dSprites_{self.args.seed}_{self.args.latentvar}_{self.args.latentcls}_{self.args.norm}.pickle'
        with open(test_set_path, 'rb') as f:
            testset = pickle.load(f)
        self.test_loader = DataLoader(testset, batch_size=self.args.bs, shuffle=False)
        print(f'len testset: {len(self.test_loader.dataset)}')
        print(f'number of batches: {len(self.test_loader)}')

    def load_vae_model(self):
        """Load pre-trained VAE model."""
        vae = VAE(self.args.latent_dim, self.args.single_output).to(self.device)
        if self.args.best_model:
            model_path = os.path.join(self.save_dir_ckpts, f'{self.args.exp}_{self.args.seed}_{self.args.latentvar}_{self.args.latentcls}_best_model.pt')
            print('Best model loaded!')
        else:
            model_path = os.path.join(self.save_dir_ckpts, f'{self.args.exp}_{self.args.seed}_{self.args.latentvar}_{self.args.latentcls}_last_model.pt')
            print('Last model loaded!')
        
        ckp_dict = torch.load(model_path, map_location=self.device)
        vae.load_state_dict(ckp_dict['state_dict'])
        vae.eval()
        return vae

    def calculate_test_statistics(self, vae):
        """Calculate the test statistic for the difference between two groups."""
        sum_fX = torch.zeros(self.args.latent_dim).to(self.device)
        sum_fY = torch.zeros(self.args.latent_dim).to(self.device)
        count_fX, count_fY = 0, 0
        group1_images, group2_images = [], []

        # Instantiate GradCAM for feature attribution
        gcam = GradCAM(vae, target_layer=self.args.target_layer, relu=self.args.relu, device=self.device)

        for batch_idx, (x_batch, target_batch) in enumerate(self.test_loader):
            x_batch = x_batch.to(self.device)
            target_batch = target_batch.to(self.device)
            
            # Forward pass
            _, mu, _ = gcam.forward(x_batch)
            
            # Accumulate stats for group X
            if all(target_batch[:, self.latent_idx] == self.args.latentcls[0]):
                if 0 <= batch_idx <= 610:
                    sum_fX += mu.sum(dim=0)
                    count_fX += mu.size(0)
                    group1_images.append(x_batch.cpu())
            
            # Accumulate stats for group Y
            elif all(target_batch[:, self.latent_idx] == self.args.latentcls[1]):
                if 1920 <= batch_idx <= 2530:
                    sum_fY += mu.sum(dim=0)
                    count_fY += mu.size(0)
                    group2_images.append(x_batch.cpu())

        print(f'count_fX: {count_fX}')
        print(f'count_fY: {count_fY}')

        # Calculate mean embeddings
        mean_fX = sum_fX / count_fX if count_fX > 0 else torch.zeros(self.args.latent_dim).to(self.device)
        mean_fY = sum_fY / count_fY if count_fY > 0 else torch.zeros(self.args.latent_dim).to(self.device)

        # Compute the test statistic D (dimension of latent vector)
        D = mean_fX - mean_fY
        test_statistic = torch.sum(torch.pow(D, 2))

        group1_images_tensor = torch.cat(group1_images)
        group2_images_tensor = torch.cat(group2_images)

        return test_statistic, D, group1_images_tensor, group2_images_tensor

    def process_attributions(self, gcam, original_images, backprop_value):
        """Process and return GradCAM attributions in batches."""
        attributions_list = []
        num_batches = (original_images.size(0) + self.args.bs - 1) // self.args.bs
        print(f'num_batches: {num_batches}')

        for i in range(num_batches):
            print(f'batch: {i}')
            batch_data = original_images[i * self.args.bs: (i + 1) * self.args.bs].to(self.device)
            _, mu, _ = gcam.forward(batch_data)
            gcam.backward(backprop_value)
            attributions = gcam.generate()
            attributions = attributions.squeeze().cpu().data.numpy()
            attributions_list.append(attributions)

        return np.vstack(attributions_list)

    def save_attributions(self, attributions, group_label, latent_dim_idx=None):
        """Save the generated attributions to file."""
        base_path = os.path.join(self.svd_dir_heatmap, f'{self.args.seed}_{self.args.exp}_{self.args.latentvar}')
        os.makedirs(base_path, exist_ok=True)
        if latent_dim_idx:
            file_name = f'{self.args.latentcls}_{group_label}_{self.args.relu}_{latent_dim_idx}_heatmap.npy'
        full_path = os.path.join(base_path, file_name)
        np.save(full_path, attributions)

    def run(self, backprop_type='test_statistic', latent_dim_idx=None):
        """Main experiment function."""
        vae = self.load_vae_model()
        test_statistic, D, group1_images_tensor, group2_images_tensor = self.calculate_test_statistics(vae)
        print(f'test_statistic: {test_statistic:0.4f}, D:{D}\n')

        # Determine which value to backpropagate (test-statistic or specific latent dimension)
        if backprop_type == 'test_statistic':
            backprop_value = test_statistic
            print(f'Backpropagating test-statistic with value: {test_statistic}\n')
        elif backprop_type == 'latent_dim':
            if latent_dim_idx is None or latent_dim_idx >= self.args.latent_dim:
                raise ValueError(f"Invalid latent dimension index: {latent_dim_idx}")
            backprop_value = D[latent_dim_idx]
            print(f'Backpropagating latent dimension {latent_dim_idx} with value: {backprop_value}\n')
        else:
            raise ValueError("Invalid backpropagation type. Choose 'test_statistic' or 'latent_dim'.")

        # Instantiate GradCAM again for attribution processing
        gcam = GradCAM(vae, target_layer=self.args.target_layer, relu=self.args.relu, device=self.device)

        # Determine which group to use for backpropagation
        if self.args.group == 'X':
            print('Group X selected')
            attributions = self.process_attributions(gcam, group1_images_tensor, backprop_value)
            self.save_attributions(attributions, 'X',latent_dim_idx)
        elif self.args.group == 'Y':
            print('Group Y selected')
            attributions = self.process_attributions(gcam, group2_images_tensor, backprop_value)
            self.save_attributions(attributions, 'Y',latent_dim_idx)
        else:
            print('Both groups (XY) selected')
            original_images = torch.cat((group1_images_tensor, group2_images_tensor), dim=0)
            attributions = self.process_attributions(gcam, original_images, backprop_value)
            self.save_attributions(attributions, 'XY',latent_dim_idx)


parser = argparse.ArgumentParser(description='Visualizing Two-Sample Test VAE')

# Training parameters
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--exp', type=int, default=4)
parser.add_argument('--bs', type=int, default=128)
    
# Model parameters
parser.add_argument('--latent_dim', type=int, default=10)
parser.add_argument('--best_model', type=bool, default=True)
parser.add_argument('--target_layer', type=str, default='encoder.conv2', choices=('conv1', 'conv2', 'conv3', 'conv4'))
parser.add_argument('--single_output', type=bool, default=False)
    
# Dataset parameters
parser.add_argument('--latentvar', type=str, default='shape', choices=('shape', 'scale', 'orientation', 'position'))
parser.add_argument('--latentcls',type=float,nargs="+",default=[1,2], help='the latent class for defining groups!')
parser.add_argument('--norm', type=bool, default=False, help='If we want to normalize data or not.')
# Heatmap visualizations
parser.add_argument('--group', type=str, default='XY',choices=['X','Y','XY'],help='Which group we want to consider for backprobagating test statistic!')
parser.add_argument('--backprop_type', type=str, default='latent_dim', choices= ('test_statistic','latent_dim'))
parser.add_argument('--latent_dim_idx', type=int, default=3)
parser.add_argument('--relu', type=bool, default=True, help='If we apply relu on heatmaps in GradCAM!')               

args = parser.parse_args()


def main(args):
    
    backprop_test = TestStatisticBackprop(args)
    
    backprop_test.run(backprop_type=args.backprop_type, latent_dim_idx=args.latent_dim_idx)
            
if __name__ == '__main__':
    
    main(args)

                    