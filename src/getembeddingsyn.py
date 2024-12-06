# required packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

import os
import numpy as np
import pandas as pd
from pathlib import Path
import random
import argparse
    
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import cPickle as pickle
except:
    import pickle

from model import VAE
from sanitydata import get_groups
from embeddingtest import MMDTest 

class Embeddings:
    def __init__(self,args):        
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root_dir = Path('/dhc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability/Codes')        
        self.config = self._read_config(self.args.seed_data, self.args.n_samples, self.args.pr)
            
    def _read_config(self,seed_data,n_samples,pr):
        config_path = os.path.join(self.root_dir, 'config', f'{seed_data}_{n_samples}_{pr}_configdata.json')
        with open(config_path,'r') as f:
            args_dict = json.load(f) 
        return args_dict
            
    def _get_data(self):
        """
        This function loads different groups
        """
        svd_data = self.root_dir / 'Syn_Data'
        seed, n_samples, pr = self.args.seed_data, self.args.n_samples, self.args.pr 
        g1_svd = os.path.join(svd_data,f'test_g1_{seed}_{n_samples}_{pr}.pickle')
        g2_svd = os.path.join(svd_data,f'test_g2_{seed}_{n_samples}_{pr}.pickle')
        with open(g1_svd,'rb') as f:
            g1 = pickle.load(f)
        with open(g2_svd,'rb') as f:
            g2 = pickle.load(f)
        batch_size = 32
        self.g1_loader = DataLoader(g1, batch_size=batch_size, shuffle=False) 
        self.g2_loader = DataLoader(g2, batch_size=batch_size, shuffle=False)
        
    def _load_checkpoint(self):
        """
        This function loads checkpoints of a pretrained model on dSpreites 
        """
        self.vae = VAE(latent_dim=10, return_single_output=False).to(self.device)
        save_dir_ckpts = self.root_dir/'ckpts'
        seed, exp = self.args.seed_model, self.args.exp
        # load checkpoints 
        if self.args.best_model:
            model_path = os.path.join(save_dir_ckpts,f'{exp}_{seed}_shape_[1, 2]_best_model.pt')
            print(f'Best model is loaded!')    
        else:
            model_path = os.path.join(save_dir_ckpts,f'{exp}_{seed}_shape_[1, 2]_last_model.pt')
            print(f'\nLast model is loaded!')
        ckp_dict = torch.load(model_path, map_location=self.device)
        self.vae.load_state_dict(ckp_dict['state_dict'])
        self.vae.eval()
        
    def _make_embeddings(self,dataloader):
        """
        This function returns embeddings from a dataloder
        """
        g_embed = []
        with torch.no_grad():
            for data in dataloader:
                images, _ = data
                images = images.to(self.device)
                rec, embed, _ = self.vae(images)
                g_embed.append(embed.cpu().numpy())
        return np.vstack(g_embed)
        
    def _process_embed(self):
        """
        This function loads embeddings if they are saved, if they are not saved it makes, saves and returns them
        """
        embed_dir = self.root_dir / 'syn_embed'
        arg_dic = self.config
        print(f'arg_dic:{arg_dic}')
        seed, n_samples, pr = arg_dic['seed'], arg_dic['n_samples'], arg_dic['pr']
        file_path = os.path.join(embed_dir, f'{seed}_{n_samples}_{pr}')
        os.makedirs(file_path, exist_ok=True)
        
        path_g1 = os.path.join(file_path, 'g1_embed.npy')
        path_g2 = os.path.join(file_path, 'g2_embed.npy')
        
        if not os.path.exists(path_g1) or not os.path.exists(path_g2):
            # make embeddings and save them
            g1_embed = self._make_embeddings(self.g1_loader)
            g2_embed = self._make_embeddings(self.g2_loader)
            print(f'g1_embed:{g1_embed.shape}, g2_embed:{g2_embed.shape}')
            if self.args.save_embed:
                np.save(path_g1, g1_embed)
                np.save(path_g2, g2_embed)
                print('embeddings were saved')
        else:
            # load embeddings
            g1_embed = np.load(path_g1)
            g2_embed = np.load(path_g2)
        
        return g1_embed, g2_embed
        
    def compute_p_value(self, g1, g2):
        """
        This function returns p-value for two groups of embeddings
        """
        mmdtest = MMDTest(g1,g2)
        p_value = mmdtest.test()
        print(f'p_value:{p_value:0.8f}')
        
    def run(self):
        """
        This function run experiment, load data, model, embedding and return p-value
        """
        self._get_data()
        self._load_checkpoint()
        g1_embed, g2_embed = self._process_embed()
        self.compute_p_value(g1_embed, g2_embed)
        
               

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Making test set from two classes with different portion')
    # parameters to make synthetic data
    parser.add_argument('--seed_data',type=int,default=42,
                    help='the seed for experiments!')
    parser.add_argument('--n_samples',type=int,default=200,
                    help='Number of samples in each group')
    parser.add_argument('--pr',type=int,default=80,
                    help='Portion of second group to be added to first group.')
    parser.add_argument('--save_embed',type=bool,default=True,
                    help='If we want to save embeddings or not!')
    # parameters of the pretrained model
    parser.add_argument('--seed_model',type=int,default=42,
                    help='the seed for experiments!')
    parser.add_argument('--exp',type=int,default=4,
                    help='the exp that we want to load its chekpoints!')
    parser.add_argument('--best_model',type=bool,default=False,
                    help='if we want to load checkpoints from best or last checkpoint!')
    args = parser.parse_args()
    
    embedding = Embeddings(args)
    
    embedding.run()
    


