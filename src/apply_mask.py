import torch
import numpy as np
import os
import random
from collections import defaultdict
from pathlib import Path
import argparse

try:
    import cPickle as pickle
except:
    import pickle

from model import VAE


class ImageMasker:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.vae = VAE(args.latent_dim, args.single_output).to(self.device)
        self.vae.eval()

    def load_images(self):
        root_dir = Path('/dhc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D')
        svd_dir_image = root_dir / 'Explainability' / 'Codes' / 'Two_Test' / 'images'

        base_path_img = os.path.join(svd_dir_image, f'{self.args.seed}_{self.args.exp}_{self.args.latentvar}')
        full_path_imgX = os.path.join(base_path_img, f'{self.args.latentcls}_X.npy')
        full_path_imgY = os.path.join(base_path_img, f'{self.args.latentcls}_Y.npy')

        imgs_X = np.load(full_path_imgX)
        imgs_Y = np.load(full_path_imgY)
        imgs = np.vstack((imgs_X, imgs_Y)).squeeze()
        print(f'\nimgs_X: {imgs_X.shape}\nimgs_Y: {imgs_Y.shape}\n')
        return imgs

    def load_masked_images(self, t):
        root_dir = Path('/dhc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D')
        svd_dir_masked = root_dir / 'Explainability' / 'Codes' / 'Two_Test' / 'masked'
        if self.args.mask_type=='least':
            svd_dir_masked = svd_dir_masked.with_name('masked_least')
        file_base_mask_dir = os.path.join(svd_dir_masked, f'{self.args.seed}_{self.args.exp}_{self.args.latentvar}')
        file_name_mask = f'{self.args.latentcls}_{self.args.group}_{self.args.relu}_{t}%_mask_{self.args.mask_type}_img.npy'
        full_path_mask = os.path.join(file_base_mask_dir, file_name_mask)

        mask_img = np.load(full_path_mask)
        print(f'masked img shape: {mask_img.shape}')
        return mask_img

    @staticmethod
    def compute_test_statistic(embed):
        idx = len(embed) // 2
        fX = embed[:idx]
        fY = embed[idx:]

        mean_fX = fX.mean(0)
        mean_fY = fY.mean(0)

        D = mean_fX - mean_fY
        statistic = np.linalg.norm(D) ** 2
        return statistic


class Experiment:
    def __init__(self, args):
        self.args = args
        self.image_masker = ImageMasker(args)
        self.imgs = self.image_masker.load_images()
        self.test_statistic = defaultdict(int)

    def run(self):
        num_images_to_replace = int(len(self.imgs) * (self.args.sub_test / 100))
        print(f'Percentage of masked samples {self.args.sub_test}%: {num_images_to_replace} images of test set!')

        assert num_images_to_replace <= self.imgs.shape[0], "num_images_to_replace exceeds the number of available images"

        for t in range(self.args.start, self.args.perc + 1):
            print(f'Percentage of masked features: {t}%')
            mask_img = self.image_masker.load_masked_images(t)

            modified_test_set = self.imgs.copy()
            modified_test_set[:num_images_to_replace] = mask_img[:num_images_to_replace]

            num_batches = (self.imgs.shape[0] + self.args.bs - 1) // self.args.bs
            mu_embed = []

            for i in range(num_batches):
                batch_data = modified_test_set[i * self.args.bs:(i + 1) * self.args.bs]
                batch_data_tensor = torch.tensor(batch_data).unsqueeze(1).to(self.image_masker.device)
                _, mu, _ = self.image_masker.vae(batch_data_tensor)
                mu_embed.append(mu.detach().cpu().numpy())

            mu_embed = np.vstack(mu_embed)
            statistic = self.image_masker.compute_test_statistic(mu_embed)
            print(f'perc_masked: {t}, test_statistic: {statistic:.6e}\n')
            self.test_statistic[t] = statistic

        print(self.test_statistic)

parser = argparse.ArgumentParser(description='Replacing subsets of testset with masked images!')

# Training parameters
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
parser.add_argument('--exp', type=int, default=4, help='ID of the current experiment!')
parser.add_argument('--bs', type=int, default=128, help='batch size!')

# Model parameters
parser.add_argument('--latent_dim', type=int, default=10, help='latent vector size of encoder')
parser.add_argument('--single_output', type=bool, default=False, help='if true, only returns mean embedding in VAE.')

# Dataset parameters
parser.add_argument('--latentvar', type=str, default='shape', choices=('shape', 'scale', 'orientation', 'position'),
                    help='the latent variable for defining groups!')
parser.add_argument('--latentcls', type=float, nargs="+", default=[1, 2], help='the latent class for defining groups!')
parser.add_argument('--norm', type=bool, default=False, help='If we normalized data during training or not.')

# Heatmap visualizations
parser.add_argument('--group', type=str, default='XY', choices=['X', 'Y', 'XY'],
                    help='Which group we want to consider for backpropagating test statistic!')
parser.add_argument('--relu', type=bool, default=True, help='If relu applied on heatmaps in Gradcam or not!')
parser.add_argument('--start', type=int, default=2,
                    help='The min percentage of (most/least) important features that we masked in each individual image!')
parser.add_argument('--perc', type=int, default=2,
                    help='The max percentage of (most/least) important features that we masked in each individual image!')
parser.add_argument('--mask_type', type=str, default='most', choices=['most', 'least'],
                        help='If we want to mask most or lest importnat features!')
parser.add_argument('--sub_test', type=int, default=1, choices=(1, 5, 10, 20, 40, 50, 60, 70, 80, 90, 100),
                    help='Percent of test set images that we want to mask their (most/least) important features!')

args = parser.parse_args()

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

    experiment = Experiment(args)
    experiment.run()


if __name__ == "__main__":
    main(args)
