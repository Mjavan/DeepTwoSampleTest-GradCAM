import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet18_Weights
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pathlib import Path
import os
import argparse


def plot_ellipses(gmm, ax, data):
    for n, color in enumerate(['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']):
        if n >= gmm.n_components: break
        if data.shape[1] == 2:
            mean = gmm.means_[n]
            cov = gmm.covariances_[n] if gmm.covariance_type == 'full' else np.diag(gmm.covariances_[n])
            v, w = np.linalg.eigh(cov)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            #ell = Ellipse(mean, v[0], v[1], 180. + angle, color=color, alpha=0.5)
            ell = Ellipse(mean, v[0], v[1], angle, edgecolor='k', facecolor='none', alpha=0.5)
            ax.add_patch(ell)
        elif data.shape[1] == 3:
            # 3D ellipses visualization is more complex; skipping it for simplicity
            pass
        

parser = argparse.ArgumentParser(description='Clustering Embeded Heatmaps')

# Training parameters
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--exp',type=int,default=4,
                    help='ID of the current expriment!')
# Dataset parameters
parser.add_argument('--latentvar',type=str,default='shape',
                    choices=('shape','scale','orientation','position'),
                    help='the latent varibale for defining groups!')
parser.add_argument('--latentcls',type=float,nargs="+",default=[1,2],
                    help='the latent class for defining groups!')
# Embedding parameters
parser.add_argument('--embed_method',type=str,default='sup-res18',
                    help='method to embed the heatmaps!')
parser.add_argument('--embed_dim',type=int,default=50,
                    help='the dimension of embedding that we want to embed heatmaps!')

# Parameters of clustering
parser.add_argument('--max_clusters',type=int,default=2,
                    help='Number of clusters in GMM!')
parser.add_argument('--cov_types',type=str,nargs="+", default= ['full'],choices = ('full', 'tied', 'diag', 'spherical'),
                    help='The covariance type!')

# Visualising clusters
parser.add_argument('--visualize',type=bool,default=True,
                    help='If we want to visualaise the clusters or not!')
parser.add_argument('--dim_reduction',type=str,default='pca', choices=('pca','tsne'),
                    help='Dim reduction method for visualisation!')


args = parser.parse_args()

def main(args):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    
    root_dir = Path('/dhc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D')
    heatmap_dir = root_dir/'Explainability'/'Codes'/'heatmaps'
    heatmap_embed_dir = root_dir/'Explainability'/'Codes'/'heatmap_embeddings'
    cluster_dir = root_dir/'Explainability'/'Codes'/'clusters'/f'{args.seed}_{args.exp}_{args.latentvar}_{args.latentcls}'
    os.makedirs(cluster_dir, exist_ok=True)
    
    # Load embeded heatmaps
    heatmaps =np.load(
        os.path.join(heatmap_embed_dir,f'{args.seed}_{args.exp}_{args.latentvar}_{args.latentcls}_{args.embed_method}_{args.embed_dim}_heatmap_embed.npy'))
    print('Embeded heatmaps are loaded!')
    
    scaler = StandardScaler()
    
    results = {}
    data = heatmaps
    data_scaled = scaler.fit_transform(data)
    
    for cov_type in args.cov_types:
        results[cov_type] = []
        for n_clusters in range(2, args.max_clusters + 1):
            print(f'gmm for {n_clusters}!')
            gmm = GaussianMixture(n_components=n_clusters, covariance_type=cov_type, random_state=42, n_init=10)
            gmm.fit(data)
            
            # Hard predictions (cluster labels)
            hard_predictions = gmm.predict(data)
            
            # Soft predictions (probabilities of belonging to each cluster)
            soft_predictions = gmm.predict_proba(data)
            
            # Calculate metrics
            aic = gmm.aic(data)
            bic = gmm.bic(data)
            ch_score = calinski_harabasz_score(data, hard_predictions)
            db_score = davies_bouldin_score(data, hard_predictions)
            
            # Save results
            results[cov_type].append({
                'n_clusters': n_clusters,
                'aic': aic,
                'bic': bic,
                'calinski_harabasz_score': ch_score,
                'davies_bouldin_score': db_score,
                'hard_predictions': hard_predictions,
                'soft_predictions': soft_predictions
            })
            
            
            # Visualize clusters after dimensionality reduction (if required)
            if args.visualize:
                if data.shape[1] == 2:  # 2D data
                    fig, ax = plt.subplots()
                    ax.scatter(data[:, 0], data[:, 1], c=hard_predictions, cmap='viridis', s=40, edgecolors='k')
                    plot_ellipses(gmm, ax, data)
                    ax.set_title(f"GMM Clustering with {n_clusters} Clusters (Covariance: {cov_type})")
                    plt.show()
                    filename = f'{n_clusters}_{cov_type}_gmm_plot_2d.png'
                    plt.savefig(os.path.join(cluster_dir,filename))
                    
                elif data.shape[1] == 3:  # 3D data
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=hard_predictions, cmap='viridis', s=40, edgecolors='k')
                    ax.set_title(f"GMM Clustering with {n_clusters} Clusters (Covariance: {cov_type})")
                    plt.show()
                    filename = f'{n_clusters}_{cov_type}_gmm_plot_3d.png'
                    plt.savefig(os.path.join(cluster_dir,filename))
                    
                else:  # Data with more than 3 dimensions
                    if args.dim_reduction == 'pca':
                        # Reduce data to 2D using PCA
                        pca = PCA(n_components=2)
                        reduced_data = pca.fit_transform(data_scaled)
                    elif args.dim_reduction == 'tsne':
                        # Reduce data to 2D using t-SNE
                        tsne = TSNE(n_components=2, random_state=42)
                        reduced_data = tsne.fit_transform(data_scaled)
                    else:
                        raise ValueError("dim_reduction should be either 'pca' or 'tsne'")
                    
                    # Visualize the reduced data
                    fig, ax = plt.subplots()
                    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=hard_predictions, cmap='viridis', s=40, edgecolors='k')
                    
                    plot_ellipses(gmm, ax, reduced_data)
                    ax.set_title(f"GMM Clustering with {n_clusters} Clusters (Covariance: {cov_type}, Reduced to 2D)")
                    plt.show()
                    filename = f'{n_clusters}_{cov_type}_{args.dim_reduction}_gmm_plot_2d.png'
                    plt.savefig(os.path.join(cluster_dir,filename))
    print(results)
    
if __name__ == '__main__':
    main(args)
    
    
    
    
    
    
    
    
    
    
    