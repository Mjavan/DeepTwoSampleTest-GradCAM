# required packages
import torch
import os
import numpy as np
import pandas as pd
from pathlib import Path
import random
import argparse
    
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class SampleImportance:
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root_dir = Path('/dhc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D/Explainability/Codes')
        self.inf_dir = self.root_dir / 'infscores'
        os.makedirs(self.inf_dir,exist_ok=True)
        self._set_random_seed()    
    
    def _set_random_seed(self):
        """Set random seed for reproducibility."""
        self.seed = self.args.seed_data
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    def _load_embeddings(self):
        """
        This function loads and returns embeddings
        """
        embed_dir = self.root_dir / 'syn_embed'
        seed, n_samples, pr = self.args.seed_data, self.args.n_samples, self.args.pr
        file_path = os.path.join(embed_dir, f'{seed}_{n_samples}_{pr}')
        
        path_g1 = os.path.join(file_path, 'g1_embed.npy')
        path_g2 = os.path.join(file_path, 'g2_embed.npy')
        try:
            g1_embed = np.load(path_g1)
            g2_embed = np.load(path_g2)
            return g1_embed, g2_embed
        except Exception as e:
            raise RuntimeError(f"Error loading .npy files: {e}")
                
    def _exclude_sample(self, i, gr_embed):
        """
        Exclude i'th embedding from an array of embeddings
        Input: gr_mebd, a numpy array of embedding vectors
        i:     an integer, indicating the index of sample to be excluded
        Return: a new numpy array of embeddings where i'th embedding is excluded 
        """
        assert gr_embed.ndim == 2, "Expecting a 2D numpy array"
        exclude_idx = i
        new_gr_embed = np.delete(gr_embed, i, axis=0)
        print(f'{len(new_gr_embed)}')
        return new_gr_embed
    
    def _calculate_test_statistic(self, X_embed, Y_embed):
        """
        This function computes test statistic for two groups of embeddings
        """
        mean_X = X_embed.mean(0)
        mean_Y = Y_embed.mean(0)
        D = mean_X - mean_Y
        statistic = np.linalg.norm(D)**2
        return statistic
    
    def _get_inf_score(self, gr1_embed, D, gr2_embed):
        inf_list = []
        for i in range(len(gr1_embed)):
            new_gr1_embed = self._exclude_sample(i, gr1_embed)
            d_new = self._calculate_test_statistic(new_gr1_embed,gr2_embed)
            inf_score = D - d_new
            inf_list.append(inf_score)            
        return inf_list
         
    def compute_all_inf_score(self):
        """
        This function computes influence scores for all samples
        """
        g1_embed, g2_embed = self._load_embeddings()
        D = self._calculate_test_statistic(g1_embed, g2_embed)        
        inf_list_X = self._get_inf_score(g1_embed, D, g2_embed)
        inf_list_Y = self._get_inf_score(g2_embed, D, g1_embed)        
        # save influnece_score
        print(f'inf_list_X:{inf_list_X}')
        print(f'inf_list_Y:{inf_list_Y}')
        
        # Build file paths
        n = len(inf_list_X)
        m = len(inf_list_Y)
        file_path = os.path.join(self.inf_dir, f'{self.args.seed_data}_{n+m}_{self.args.pr}')
        os.makedirs(file_path, exist_ok=True)
        
        path_g1 = os.path.join(file_path,'g1_inf.npy')
        path_g2 = os.path.join(file_path,'g2_inf.npy')
        
        np.save(path_g1,np.array(inf_list_X))
        np.save(path_g2,np.array(inf_list_Y))
        
        return inf_list_X,inf_list_Y 
    
    
        
                             
parser = argparse.ArgumentParser(description='Computing Sample_Importance') 
# parameters of group mixtures
parser.add_argument('--seed_data',type=int,default=42,
                    help='the seed for experiments!')
parser.add_argument('--n_samples',type=int,default=200,
                    help='Number of samples in each group')
parser.add_argument('--pr',type=int,default=80,
                    help='Portion of second group to be added to first group.')

args = parser.parse_args()
            
if __name__=="__main__":
    
    test = SampleImportance(args)
    inf_X, inf_Y = test.compute_all_inf_score()



        
        
        
        
        
