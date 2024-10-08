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

try:
    import cPickle as pickle
except:
    import pickle

import warnings
warnings.filterwarnings("ignore")

# using mean and std with 5 digits precesion for standadization
mean5 = 0.04249
std5 = 0.20171

# if we want to go for mean and std with 3 digits precesion
mean3 = 0.042
std3 = 0.202


### download dSprites dataset
class dSprites(Dataset):
    
    def __init__(self,root_dir,norm=None,mean_ds=0.04249,std_ds=0.20171):
        self.root = root_dir
        self.dataset = DSprites(root=self.root, download=True)
        self.len = len(self.dataset)
        self.norm = norm
        self.mean_ds = mean_ds
        self.std_ds = std_ds    
        
    def __len__(self):
        return(self.len)

    def __getitem__(self,index):
        # returning images of dim:[C,H,W]= [1,64,64]
        self.img = self.dataset[index]['input'].unsqueeze(0).float()
        self.latent = self.dataset[index]['latent']
        if self.norm:
            # this returns image as [C,H,W]
            transform = transforms.Compose([transforms.Normalize(mean_ds,std_ds),])
            self.img = transform(self.img)                          
        return(self.img, self.latent)
    

### making train, test, validation sets 
parser = argparse.ArgumentParser(description='Making datasets and dataloaders')

parser.add_argument('--seed',type=int,default=42,
                    help='the seed for experiments!')

parser.add_argument('--valsize',type=float,default=0.1,
                    help='the size of validation set!')

parser.add_argument('--testsize',type=float,default=0.3,
                    help='the size of test set!')

parser.add_argument('--latentvar',type=str,default='shape',choices=('shape','scale','orientation','position'),
                    help='the latent varibale for defining groups!')

parser.add_argument('--latentcls',type=float,nargs="+",default=[1,2],
                    help='the latent class for defining groups!, 1,2 correspond to square and ellipse')

parser.add_argument('--norm',type=bool,default=False,
                    help='the latent class for defining groups!')

parser.add_argument('--save',type=bool,default=True,
                    help='if we want to save train, val, test sets or not!')

args = parser.parse_args() 

def get_train_test_set(args):
    
    root_dir = Path('/dhc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D')
    # directory that we save train, val, test sets
    svd_split = root_dir/'Explainability/Reproducibility/data'
    
    dataset = dSprites('data',norm=args.norm)
    
    indices = np.arange(len(dataset))
    
    print(f'len dataset:{len(indices)}\n')
    
    latents_values = np.load(os.path.join(root_dir, 'Explainability', 'Codes', 'lval.npy'))
    latents_classes = np.load(os.path.join(root_dir, 'Explainability', 'Codes', 'lcls.npy'))
    
    images = torch.load(os.path.join(root_dir, 'Explainability', 'Codes', 'img.pt')).numpy()
    latents = torch.load(os.path.join(root_dir, 'Explainability', 'Codes', 'latents.pt')).numpy()
    
    # if we have latent_var we need to exclude it from training and define groups
    if args.latentvar:
        if args.latentvar=='shape':
            latent_idx=1
            
        #g1_indices = np.where(latents_classes[:, latent_idx] == args.latentcls[0])[0]
        #g2_indices = np.where(latents_classes[:, latent_idx] == args.latentcls[1])[0]
        
        g1_indices = np.where(latents[:,latent_idx] == args.latentcls[0])[0]
        g2_indices = np.where(latents[:,latent_idx] == args.latentcls[1])[0]
        
        print(f'len first group:{len(g1_indices)}')
        print(f'len second group:{len(g2_indices)}')
        
        # exclude groups 1 and 2 from train set and trainset containes only one class
        #train_indices = np.setdiff1d(indices, np.concatenate((g1_indices, g2_indices)))
        #test_indices = np.concatenate((g1_indices, g2_indices))
        
        # Combine the indices of the two classes for the test set
        test_indices = np.concatenate((g1_indices, g2_indices))
        # Indices for the training set are those not in the test set
        train_indices = np.setdiff1d(indices, test_indices)
        
        print(f'\nlen train_index:{len(train_indices)}')
        print(f'len test_index:{len(test_indices)}\n')
        
        if args.valsize:
            split = ShuffleSplit(n_splits=1,test_size=args.valsize,random_state=args.seed)
            for train_idx,val_idx in split.split(train_indices):
                trainset = Subset(dataset, train_indices[train_idx])
                valset = Subset(dataset, train_indices[val_idx])
        else:
            trainset = Subset(dataset, train_indices)
            valset = None  # or some other placeholder if you prefer             
        testset = Subset(dataset,test_indices) 
        
        if args.save:
            save_train = svd_split / f'train_set_dSprites_{args.seed}_{args.latentvar}_{args.latentcls}_{args.norm}.pickle'
            save_test = svd_split / f'test_set_dSprites_{args.seed}_{args.latentvar}_{args.latentcls}_{args.norm}.pickle'
            with open(save_train, 'wb') as f: 
                pickle.dump(trainset,f,protocol=pickle.HIGHEST_PROTOCOL)    
            with open(save_test, 'wb') as f:
                pickle.dump(testset,f,protocol=pickle.HIGHEST_PROTOCOL)    
            if args.valsize:
                save_val = svd_split / f'val_set_dSprites_{args.seed}_{args.latentvar}_{args.latentcls}_{args.norm}.pickle'
                with open(save_val, 'wb') as f: 
                    pickle.dump(valset,f,protocol=pickle.HIGHEST_PROTOCOL)            
        return(trainset,valset,testset)
                                     
    else:
        ## calculate splits 
        split = ShuffleSplit(n_splits=1,test_size=args.testsize,random_state=args.seed)
        for train_index, test_index in split.split(indices):
            trainset = Subset(dataset,indices[train_index])
            testset = Subset(dataset, indices[test_index])      
        ## we hold out a prt of trianing set as valset
        if args.valsize:
            split = ShuffleSplit(n_splits=1,test_size=args.valsize,random_state=args.seed)
            for train_index,val_index in split.split(train_index):
                train_ind = train_index
                val_ind = val_index
            trainset = Subset(trainset,train_ind)
            valset = Subset(trainset,val_ind)     
        
        if args.save:
        
            # save train, test & val sets for reproducibility
            if args.valsize:
                trainsize = 1-args.testsize-args.valsize
            else:
                trainsize = 1-args.testsize
            
            save_train = svd_split / f'train_set_dSprites_{args.seed}_{trainsize}.pickle'
            save_test = svd_split / f'test_set_dSprites_{args.seed}_{testsize}.pickle'
    
            with open(save_train, 'wb') as f: 
                pickle.dump(trainset,f,protocol=pickle.HIGHEST_PROTOCOL)
        
            with open(save_test, 'wb') as f:
                pickle.dump(testset,f,protocol=pickle.HIGHEST_PROTOCOL)
            
            if args.valsize:
                save_val = svd_split / f'val_set_dSprites_{args.seed}_{args.valsize}.pickle'
            
                with open(save_val, 'wb') as f: 
                    pickle.dump(valset,f,protocol=pickle.HIGHEST_PROTOCOL)
        if args.valsize:
            return(trainset,testset,valset)
    
        return(trainset,testset)


if __name__=="__main__":
    
    get_train_test_set(args)
    
    
