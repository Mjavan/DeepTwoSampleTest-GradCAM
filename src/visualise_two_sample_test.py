## main code for my experiments
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
from model import VAE
from gradcam import GradCAM


parser = argparse.ArgumentParser(description='Visualaizing Two-Sample Test VAE')

# Training parameters
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--exp',type=int,default=4,
                    help='ID of the current expriment!')
parser.add_argument('--bs', type=int, default=128,
                    help='input batch size for training (default: 128)')
# Model parameters    
parser.add_argument('--latent_dim', type=int, default=10,
                    help='latent vector size of encoder')
parser.add_argument('--best_model',type=bool, default=True,
                    help='If True load the best checkpoint otherwise last one.')
parser.add_argument('--target_layer', type=str, default='encoder.conv3',
                    choices=('conv1','conv2','conv3','conv4'),
                    help='select a target layer for generating attention map.')
parser.add_argument('--backprop_type', type=str, default='latent_dim', choices= ('test_statistic','latent_dim'))
parser.add_argument('--latent_dim_idx', type=int, default=9,
                   help='Dimension of the latent vecor that we want to backprobagte')
parser.add_argument('--single_output', type=bool, default=False,
                    help='if true, only returns mean in VAE.')

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
                    help='Which group we want to consider for backprobagating test statistic!')
parser.add_argument('--relu', type=bool, default=True,
                   help='If we apply relu on heatmaps in GradCAM!')
parser.add_argument('--save_gcam_image', type=bool, default=True,
                    help='If we want to visualize heatmaps or not!')
parser.add_argument('--attention', type=str, default=None, choices = ('spatial','channel'),
                    help='If we want to apply attention or not!')
parser.add_argument('--save_image', type=bool, default=False,
                    help='If we want to save images or not!')

args = parser.parse_args()

def main(args):
    seed =args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    
    # directory for loading checkpoints
    root_dir = Path('/dhc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D')
    svd_split = root_dir/'Explainability'/'Reproducibility'/'data'
    save_dir_ckpts = root_dir / 'Explainability' / 'Codes' / 'ckpts'
    svd_dir_heatmap = root_dir / 'Explainability' / 'Codes' / 'Two_Test' / 'heatmaps'
    svd_dir_image = root_dir / 'Explainability' / 'Codes' / 'Two_Test' / 'images'
    
    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    
    latent_variable_mapping = {
    'color': 0,
    'shape': 1,
    'scale': 2,
    'orientation': 3,
    'position X': 4,
    'position Y': 5}
    latent_idx = latent_variable_mapping[args.latentvar]
    
    # making test set & test loader
    with open(
        svd_split/f'test_set_dSprites_{args.seed}_{args.latentvar}_{args.latentcls}_{args.norm}.pickle',
        'rb') as f:
        testset = pickle.load(f)
    test_loader = DataLoader(testset, batch_size=args.bs, shuffle=False)
    print(f'len testset:{len(test_loader.dataset)}\n')
    print(f'number of batches:{len(test_loader)}\n')
    
    vae = VAE(args.latent_dim, args.single_output).to(device)
    
    # load checkpoints 
    if args.best_model:
        model_path = os.path.join(save_dir_ckpts,f'{args.exp}_{args.seed}_{args.latentvar}_{args.latentcls}_best_model.pt')
        print(f'Best model is loaded!')    
    else:
        model_path = os.path.join(save_dir_ckpts,f'{args.exp}_{args.seed}_{args.latentvar}_{args.latentcls}_last_model.pt')
        print(f'\nLast model is loaded!')
    ckp_dict = torch.load(model_path, map_location=device)
    vae.load_state_dict(ckp_dict['state_dict'])
    
    vae.eval()
    
    # Instantiate GradCAM
    gcam = GradCAM(vae, target_layer=args.target_layer, relu=args.relu, device=device, attention=args.attention)
    print(f'{args.target_layer} was chosen for backprobagation.')
    
    # Initialize variables to store sums and counts
    sum_fX = torch.zeros(args.latent_dim).to(device)
    sum_fY = torch.zeros(args.latent_dim).to(device)
    count_fX = 0
    count_fY = 0
    group1_images = []
    group2_images = []
    
    for batch_idx,(x_batch, target_batch) in enumerate(test_loader):
        x_batch = x_batch.to(device)
        target_batch = target_batch.to(device)
        
        # performs forward pass
        _,mu,_ = gcam.forward(x_batch)
        
        # Calculate running sum and count for class X
        if all(target_batch[:, latent_idx] == args.latentcls[0]):
            if 0 <= batch_idx <= 610:
                sum_fX += mu.sum(dim=0)
                count_fX += mu.size(0)
                group1_images.append(x_batch.cpu())
        # Calculate running sum and count for class Y
        elif all(target_batch[:, latent_idx] == args.latentcls[1]):
            if 1920<=batch_idx<=2530:
                sum_fY += mu.sum(dim=0)
                count_fY += mu.size(0)
                group2_images.append(x_batch.cpu())
    
    print(f'count_fX:{count_fX}')
    print(f'count_fY:{count_fY}\n')
    
    # Convert lists of images to a single tensor
    group1_images_tensor = torch.cat(group1_images)
    group2_images_tensor = torch.cat(group2_images)
    
    print(f'group1_size:{group1_images_tensor.size()}')
    print(f'group2_size:{group2_images_tensor.size()}\n')
    
    if args.save_image:
        group1_images_np = group1_images_tensor.cpu().numpy()
        group2_images_np = group2_images_tensor.cpu().numpy()
        base_path_img = os.path.join(svd_dir_image, f'{args.seed}_{args.exp}_{args.latentvar}')
        os.makedirs(base_path_img,exist_ok=True)
        file_name_imgX = f'{args.latentcls}_X.npy'
        file_name_imgY = f'{args.latentcls}_Y.npy'
        full_path_imgX = os.path.join(base_path_img, file_name_imgX)
        full_path_imgY = os.path.join(base_path_img, file_name_imgY)
        
        # Save the numpy images
        np.save(full_path_imgX,group1_images_np)
        np.save(full_path_imgY,group2_images_np)
                
    # Compute the means
    mean_fX = sum_fX / count_fX if count_fX > 0 else torch.zeros(latent_dim).to(device)
    mean_fY = sum_fY / count_fY if count_fY > 0 else torch.zeros(latent_dim).to(device)
    
    # Compute the test statistic
    D = mean_fX - mean_fY    
    
    del sum_fX, sum_fY
    torch.cuda.empty_cache()
    
    print(f'Dimension of D:{D.size()}\n')
    
    print(f'D:{D}')
                
    statistic = torch.sum(torch.pow(D, 2))
    
    print(f'test_statistic:{statistic}')
    
    if args.backprop_type=='test_statistic':
        backprob_value = statistic
        print(f'test_statistic is backprobagated')
    else:
        backprob_value = D[args.latent_dim_idx]
        print(f'latent_dimension {args.latent_dim_idx} is backprobagated')
    
    def process_in_batches(tensor_data, batch_size):
        attributions_list = []
        num_batches = (tensor_data.size(0) + batch_size - 1) // batch_size
        print(f'num_batches:{num_batches}\n')
        for i in range(num_batches):
            print(f'batch:{i}\n')
            batch_data = tensor_data[i * batch_size:(i + 1) * batch_size].to(device)
            # Perform forward pass again
            print(f'forwarding batch_data')
            _, mu, _ = gcam.forward(batch_data)
            gcam.backward(backprob_value)
            print(f'backward statistic')
            attributions = gcam.generate()
            attributions = attributions.squeeze().cpu().data.numpy()
            attributions_list.append(attributions)
        attributions_np = np.vstack(attributions_list)
        print(f'attributions:{attributions_np.shape}')
        return attributions_np
    
    if args.group == 'X':
        print(f'group_X was chosen\n')
        original_images = group1_images_tensor
        attributions = process_in_batches(original_images, args.bs)
    
    elif args.group == 'Y':
        print(f'group_Y was chosen')
        original_images = group2_images_tensor
        attributions = process_in_batches(original_images, args.bs)
        
    elif args.group == 'XY':
        print(f'group_XY was chosen')
        original_images = torch.cat((group1_images_tensor,group2_images_tensor),dim=0)
        print(f'original_images:{original_images.size()}')
        attributions = process_in_batches(original_images, args.bs)
        
    # Let's look at attribution map shape and save them
    base_path = os.path.join(svd_dir_heatmap, f'{args.seed}_{args.exp}_{args.latentvar}_{args.target_layer[8:]}')
    os.makedirs(base_path,exist_ok=True)
    if args.attention:
        file_name = f'{args.latentcls}_{args.group}_{args.relu}_{args.attention}_heatmap.npy'
    elif args.backprop_type =='latent_dim':
        file_name = f'{args.latentcls}_{args.group}_{args.relu}_{args.latent_dim_idx}_heatmap.npy'
    else:
        file_name = f'{args.latentcls}_{args.group}_{args.relu}_heatmap.npy'
    full_path = os.path.join(base_path, file_name)
    
    # Save the numpy array
    np.save(full_path, attributions)
         
    
if __name__=='__main__':
    main(args)
        
    
        
    
    
    
    
    