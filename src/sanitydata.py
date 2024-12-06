import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import datasets,transforms

## importing libraries 
import numpy as np
import pandas as pd
from disentanglement_datasets import DSprites
from sklearn.model_selection import ShuffleSplit
from pathlib import Path
import argparse
import json

import warnings
warnings.filterwarnings("ignore")

try:
    import cPickle as pickle
except:
    import pickle


from dataloader import dSprites


### making test sets with different portions of two classes 
parser = argparse.ArgumentParser(description='Making test set from two classes with different portion')

parser.add_argument('--seed',type=int,default=42,
                    help='the seed for experiments!')
parser.add_argument('--latentvar',type=str,default='shape',choices=('shape','scale','orientation','position'),
                    help='the latent varibale for defining groups!')
parser.add_argument('--latentcls',type=float,nargs="+",default=[1,2],
                    help='the latent class for defining groups!, 1,2 correspond to square and ellipse')
parser.add_argument('--n_samples',type=int,default=200,
                    help='Number of samples in each group')
parser.add_argument('--pr',type=int,default=10,
                    help='Portion of second group to be added to first group.')
parser.add_argument('--norm',type=bool,default=False,
                    help='the latent class for defining groups!')

args = parser.parse_args()

print(args.pr) 

def get_groups(args):
    
    root_dir = Path('/dhc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D')
    svd_split = root_dir/'Explainability/Reproducibility/data'
    
    config_dir = root_dir / 'Explainability' / 'Codes' / 'config'
    config_path = os.path.join(config_dir,f'{args.seed}_{args.n_samples}_{args.pr}_configdata.json')
    args_dict = vars(args)
    with open(config_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    dataset = dSprites('data',norm=args.norm)
    
    indices = np.arange(len(dataset))
    
    print(f'len dataset:{len(indices)}\n')
    
    latents_values = np.load(os.path.join(root_dir, 'Explainability', 'Codes', 'lval.npy'))
    latents_classes = np.load(os.path.join(root_dir, 'Explainability', 'Codes', 'lcls.npy'))
    
    images = torch.load(os.path.join(root_dir, 'Explainability', 'Codes', 'img.pt')).numpy()
    latents = torch.load(os.path.join(root_dir, 'Explainability', 'Codes', 'latents.pt')).numpy()
    
    if args.latentvar=='shape':        
        latent_idx=1
            
    g1_indices = np.where(latents[:,latent_idx] == args.latentcls[0])[0]
    g2_indices = np.where(latents[:,latent_idx] == args.latentcls[1])[0]
    
    print(f'len first group:{len(g1_indices)}')
    print(f'len second group:{len(g2_indices)}')
    
    g1_subset = np.random.choice(g1_indices, size=args.n_samples, replace=False)
    print(f'{len(g1_subset)} of samples in first group was selected!')
    
    g2_num = int((args.pr/100) * len(g1_subset))
    
    # number of samples from group1 that we want to include in group2
    g1_part = len(g1_subset) - g2_num
    print(f'{g1_part} samples from group1 will be added to group2.')
    # subset of g1 that we use for making group2
    g1_subset_for_g2 = np.random.choice(g1_subset, size=g1_part, replace=False)
    
    print(f'{g2_num} samples from group2 will be added to group1.')
    g2_part = np.random.choice(g2_indices, size=g2_num, replace=False)
    
    g2_subset = np.concatenate([g1_subset_for_g2,g2_part])
    
    print(f'g1 size:{len(g1_subset)}, g2 size:{len(g2_subset)}')
    
    g1 = Subset(dataset, g1_subset)
    g2 = Subset(dataset, g2_subset)
    
    print(f'g1_type:{type(g1)}')
    print(f'g2_type:{type(g2)}')
    
    # save as a pickle file
    save_path = root_dir /'Explainability'/ 'Codes' / 'Syn_Data'    
    save_test_g1 = os.path.join(save_path, f'test_g1_{args.seed}_{args.n_samples}_{args.pr}.pickle')
    save_test_g2 = os.path.join(save_path, f'test_g2_{args.seed}_{args.n_samples}_{args.pr}.pickle') 
    
    # check if the files have not been saved  
    if not os.path.exists(save_test_g1) or not os.path.exists(save_test_g2):
        with open(save_test_g1, 'wb') as f:
            pickle.dump(g1,f,protocol=pickle.HIGHEST_PROTOCOL)
        with open(save_test_g2, 'wb') as f:
            pickle.dump(g2,f,protocol=pickle.HIGHEST_PROTOCOL)
        

if __name__=="__main__":
    get_groups(args)
    

