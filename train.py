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
    
from utils import visualize_reconstruction, plot_losses
from dataloader import dSprites
from model import VAE

# Loss function
def vae_loss(x, x_recon, mu, logvar):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss


# Training VAE
def train_vae(model, optimizer, epoch, dataloader, device):
    model.train()
    train_loss = 0
    for batch in dataloader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x)
        loss = vae_loss(x, x_recon, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()    
    train_loss /= len(dataloader.dataset)
    return(train_loss)
        
        
        
# Evaluating VAE
def evaluate_vae(model, epoch, dataloader, device):
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(device)
                x_recon, mu, logvar = model(x)
                loss = vae_loss(x, x_recon, mu, logvar)
                val_loss += loss.item()
            val_loss /= len(dataloader.dataset)
        return(val_loss)
            
                        
parser = argparse.ArgumentParser(description='Training ConvVAE')

# Training parameters
parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
parser.add_argument('--exp',type=int,default=4,
                        help='ID of the current expriment!')
parser.add_argument('--bs', type=int, default=128,
                        help='input batch size for training (default: 128)')
parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 10)')
parser.add_argument('--n_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders.')
parser.add_argument('--vis_rec', type=bool, default=True,
                        help='If we want to visulaize reconstructions or not.')
parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs for early stopping.')
# Model parameters    
parser.add_argument('--latent_dim', type=int, default=10,
                        help='latent vector size of encoder')            
# Dataset parameters
parser.add_argument('--latentvar',type=str,default='shape',choices=('shape','scale','orientation','position'),
                    help='the latent varibale for defining groups!')
parser.add_argument('--latentcls',type=float,nargs="+",default=[1,2],
                    help='the latent class for defining groups!')
parser.add_argument('--norm', type=bool, default=False,
                        help='If we want to normalize data or not.')
parser.add_argument('--plot', type=bool, default=True,
                        help='If we want to plot losses.')


args = parser.parse_args()


def main(args):
    
    seed =args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    print(f'use_cuda:{use_cuda}')
    
    root_dir = Path('/dhc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D')
    svd_split = root_dir/'Explainability'/'Reproducibility'/'data'
    save_dir_ckpts = root_dir / 'Explainability' / 'Codes' / 'ckpts'
    os.makedirs(save_dir_ckpts, exist_ok=True)
    result_dir = root_dir / 'Explainability' / 'Codes' /'results' / f'exp_{args.exp}'
    os.makedirs(result_dir, exist_ok=True)
    save_dir_loss = root_dir / 'Explainability' / 'Codes' / 'loss'
    os.makedirs(save_dir_loss, exist_ok=True)
    
    print(f'training was started!')
    
    # These datasets are normalized
    with open(svd_split / f'train_set_dSprites_{args.seed}_{args.latentvar}_{args.latentcls}_{args.norm}.pickle','rb') as f:
        trainset = pickle.load(f)    
    with open(svd_split / f'val_set_dSprites_{args.seed}_{args.latentvar}_{args.latentcls}_{args.norm}.pickle','rb') as f:
        valset = pickle.load(f)
                 
    # Load training & validation sets 
    train_loader = DataLoader(trainset, batch_size=args.bs, shuffle=True) 
    val_loader = DataLoader(valset, batch_size=args.bs, shuffle=True) 
    
    print(f'Datasets and Daraloaders are built')
    

    # Initialize and train the VAE
    vae = VAE(args.latent_dim).to(device)
    # Defining optimzier
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    best_val_loss = np.finfo('f').max
    epochs_no_improve = 0
    train_hist =[]
    val_hist = []
    
    for epoch in range(args.epochs):
        train_loss = train_vae(vae, optimizer, epoch, train_loader, device)
        val_loss = evaluate_vae(vae, epoch, val_loader, device)
        print('Epoch [%d/%d] Train Loss: %.3f Val Loss: %.3f'  % \
              (epoch + 1, args.epochs, train_loss, val_loss))
        
        train_hist.append(train_loss)
        val_hist.append(val_loss)
        
        # Check if model is good enough for checkpoint to be saved
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        if is_best:
            torch.save({'epoch':epoch,
                        'best_val_loss':best_val_loss,
                        'state_dict':vae.state_dict(),
                        'optimizer': optimizer.state_dict()},\
                        os.path.join(save_dir_ckpts,f'{args.exp}_{args.seed}_{args.latentvar}_{args.latentcls}_best_model.pt'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= args.patience:
            print('Early stopping triggered')
            break
            
        # Visualize reconstruction every 5 epochs (or any frequency you prefer)
        if (epoch + 1) % 5 == 0:
            result_dir_epoch = result_dir/ f'epoch_{epoch+1}'
            visualize_reconstruction(vae, val_loader, result_dir_epoch, device)
            
    # saving last model
    torch.save({'epoch':epoch,
                'last_val_loss':val_loss,
                'state_dict':vae.state_dict(),
                'optimizer': optimizer.state_dict()},\
                os.path.join(save_dir_ckpts,f'{args.exp}_{args.seed}_{args.latentvar}_{args.latentcls}_last_model.pt'))
    # save loss
    np.save(os.path.join(save_dir_loss,f'{args.exp}_{args.latentvar}_{args.latentcls}_{args.norm}_train_loss.npy'),\
            np.array(train_hist))
    np.save(os.path.join(save_dir_loss,f'{args.exp}_{args.latentvar}_{args.latentcls}_{args.norm}_val_loss.npy'),\
            np.array(val_hist))
    # plot loss
    if args.plot:
        save_dir_plot = os.path.join(save_dir_loss / 'Plots',\
                                     f'{args.exp}_{args.latentvar}_{args.latentcls}_{args.norm}')
        os.makedirs(save_dir_plot, exist_ok=True)
        plot_losses(train_hist, val_hist, save_dir_plot)
    
    print("Training was finished")
    
if __name__=='__main__':
    main(args)
    

        
        
   
            

        
        
        
        