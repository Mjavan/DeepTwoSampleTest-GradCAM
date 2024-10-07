import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import seaborn as sns
from sklearn.cluster import KMeans
import time
import argparse
from sklearn.preprocessing import StandardScaler
import umap


from embeddingtest import MMDTest

def pca_heatmap(heatmap_pca, pca_dir):
    plt.figure(figsize=(10, 7))
    # Assuming `labels` is a list of class labels corresponding to the heatmaps
    # Total number of points
    total_points = 491520
    # Half points belong to class 1 (square), the other half to class 2 (ellipse)
    labels = np.array([1] * (total_points // 2) + [2] * (total_points // 2))
    plt.scatter(heatmap_pca[:, 0], heatmap_pca[:, 1],c=labels, cmap='viridis', marker='o')
    plt.title('PCA of Heatmaps')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Class Label')
    plt.savefig(os.path.join(pca_dir, 'pca.png'))
    plt.show()

def tsne_heatmap(heatmap_tsne, tsne_dir):
    plt.figure(figsize=(10, 7))
    total_points = 491520
    # Half points belong to class 1 (square), the other half to class 2 (ellipse)
    labels = np.array([1] * (total_points // 2) + [2] * (total_points // 2))
    plt.scatter(heatmap_tsne[:, 0], heatmap_tsne[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title('t-SNE of Heatmaps')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.colorbar(label='Class Label')
    plt.savefig(os.path.join(tsne_dir, 'tsne.png'))
    plt.show()
    

def create_heatmap(significant_features, pca_dir):
    # Create a heatmap of the significant features
    plt.figure(figsize=(10, 8))
    #sns.heatmap(significant_features_reshaped, cmap="Reds", cbar=False)
    plt.imshow(significant_features, cmap='hot', interpolation='nearest')
    plt.title('Significant Features between Groups')
    plt.show()
    plt.savefig(os.path.join(pca_dir, 'significant_features.png'))


parser = argparse.ArgumentParser(description='Investigating the statistical difference!')    
parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
parser.add_argument('--exp',type=int,default=4,
                        help='ID of the current expriment!')
parser.add_argument('--latentvar',type=str,default='shape',choices=('shape','scale','orientation','position'),
                    help='the latent variable for defining groups!')
parser.add_argument('--latentcls',type=float,nargs="+",default=[1,2],
                    help='the latent class for defining groups!')
parser.add_argument('--sttest',type=str,default='D2T',choices=('D2T','2TT','WTT'),
                    help='the statistical test that we want to do')

parser.add_argument('--embed_net',type=str,default='vae',choices=('sup-res18','vae'),
                    help='the network that we used for heatmap embedding or image embedding!')
parser.add_argument('--dimension',type=int,default=50,
                    help='The dimension that we used for heatmap embedding!')

parser.add_argument('--pca_vis',type=bool,default=True,
                    help='visualizing clusters using pca')
parser.add_argument('--tsne_vis',type=bool,default=False,
                    help='visualizing clusters using tsne')
parser.add_argument('--umap_vis',type=bool,default=False,
                    help='visualizing clusters using umap')


args = parser.parse_args()

def main(args):
    
    root_dir = Path('/dhc/home/masoumeh.javanbakhat/netstore-old/Baysian/3D')
    heatmap_dir = root_dir/'Explainability'/'Codes'/'heatmaps'
    heatmap_embed_dir = root_dir/'Explainability'/'Codes'/'heatmap_embeddings'
    embed_dir = root_dir / 'Explainability' / 'Codes'/ 'embeddings'
    
    cluster_dir = root_dir/'Explainability'/'Codes'/'clusters'
    os.makedirs(cluster_dir, exist_ok=True)
    
    pca_dir = os.path.join(cluster_dir,f'pca_{args.exp}_{args.seed}_{args.latentvar}_{args.latentcls}')
    os.makedirs(pca_dir, exist_ok=True)
    
    if args.embed_net!='vae':
        
        # loading raw heatmaps
        heatmaps = np.load(os.path.join(heatmap_dir,f'{args.exp}_{args.seed}_{args.latentvar}_{args.latentcls}_htm.npy'))
        heatmap_vectors = [heatmap.flatten() for heatmap in heatmaps]  # Flattening the heatmaps
        print(f'len heatmaps:{len(heatmap_vectors)}')
        mid_point = len(heatmap_vectors) // 2
        group1 = heatmap_vectors[:mid_point]
        group2 = heatmap_vectors[mid_point:]
        # Convert to numpy arrays
        # features_group1: [n_samples, n_features]
        features_group1 = np.array(group1)
        features_group2 = np.array(group2)
    
        print(f'group1_np:{features_group1.shape}')
        print(f'group2_np:{features_group1.shape}')
        
        # loading embeded heatmaps
        heatmap_embed = np.load(
        os.path.join(         heatmap_embed_dir,f'{args.seed}_{args.exp}_{args.latentvar}_{args.latentcls}_{args.embed_net}_{args.dimension}_heatmap_embed.npy'))
        heatmap_embed1= heatmap_embed[:mid_point]
        heatmap_embed2= heatmap_embed[mid_point:]
        print(f'embed_heat1:{heatmap_embed1.shape}')
        print(f'embed_heat2:{heatmap_embed2.shape}')
        
    elif args.embed_net=='vae':
        # loading raw image features
        embeddings = np.load(os.path.join(embed_dir,f'{args.exp}_{args.seed}_{args.latentvar}_{args.latentcls}_emb.npy'))
        print(f'shape embeddings:{embeddings.shape}')
        mid_point = embeddings.shape[0]//2 
        
        embed_group1 = embeddings[:mid_point]
        embed_group2 = embeddings[mid_point:]    
        print(f'embed_group1:{embed_group1.shape}')
        print(f'embed_group2:{embed_group2.shape}')
    
     
    if args.sttest=='D2T':
        start_time = time.time()
        if args.embed_net=='sup-res18':
            # apply embedding's normalization 
            print(f'D2T is done on normalaised embedded heatmaps!')
            f_X = (heatmap_embed1 - np.mean(heatmap_embed1, axis=0)) / np.std(heatmap_embed1, axis=0)
            f_Y = (heatmap_embed2 - np.mean(heatmap_embed2, axis=0)) / np.std(heatmap_embed2, axis=0)
        elif args.embed_net=='vae':
            # here, we apply D2T on features from images
            print(f'D2T is done on raw feature vectors from images!')
            f_X = embed_group1
            f_Y = embed_group2
        else:
            # here, we apply D2T on raw heatmaps without being embedded
            print(f'D2T is done on raw heatmaps!')
            f_X = features_group1
            f_Y = features_group2
        # perfrom deep two sample test
        mmd_test = MMDTest(f_X, f_Y)
        pvalue = mmd_test.test()
        end_time = time.time()
        print(f'\nMMDTest was done!\n')
        print(f'pvalue for deep two sample test is:{pvalue}')
        print(f'test was done in {end_time-start_time:.4f} seconds\n!')
    
    if args.sttest=='2TT':
    
        # Perform t-test
        # t_stat:[n_features], 
        # p_values:[n_features]
        # ttest_ind: assumes that the samples are independent and that
        # the data in each group is normally distributed with equal variances
        t_stat, p_values = ttest_ind(features_group1, features_group2, axis=0)    
        # Multiple comparisons correction= Benjamini-Hochberg (BH) Correction
        # Adjust p-values using False Discovery Rate (FDR)
        # tells you whether each feature is significantly different after applying FDR correction.
        significant_features_idx = multipletests(p_values, alpha=0.05, method='fdr_bh')[0]    
        print(f'significant_features_idx:{significant_features_idx.shape}')    
        # reshaping the significant features
        significant_features_reshaped = significant_features_idx.reshape(64, 64)
        create_heatmap(significant_features_reshaped, pca_dir)
        
    if args.sttest=='WTT':
        
        # Perform Welch's t-test
        t_stat, p_values = ttest_ind(features_group1, features_group2, axis=0, equal_var=False)        
        # Apply False Discovery Rate (FDR) correction to the p-values
        significant_features_idx = multipletests(p_values, alpha=0.05, method='fdr_bh')[0]        
        # reshaping the significant features
        significant_features_reshaped = significant_features_idx.reshape(64, 64)        
        create_heatmap(significant_features_reshaped, pca_dir)
    
    if args.pca_vis:
        # Looking at clusters of heatmaps using PCA
        pca = PCA(n_components=2) 
        if args.embed_net == 'vae':
            print('\nDimensionality reduction on raw feature embeddings using pca!')
            heatmap_pca = pca.fit_transform(embeddings)
            pca_heatmap(heatmap_pca,pca_dir)
        else:
            heatmap_pca = pca.fit_transform(heatmap_vectors)
            pca_heatmap(heatmap_pca,pca_dir)
        
    if args.tsne_vis:
        tsne = TSNE(n_components=2, random_state=42)
        if args.embed_net =='vae':
            print('\nDimensionality reduction on raw feature emebddinsg using tsne!')
            scaled_embeddings = StandardScaler().fit_transform(embeddings)
            embeddings_tsne = tsne.fit_transform(scaled_embeddings)
            tsne_dir = os.path.join(cluster_dir,f'tsne_{args.exp}_{args.seed}_{args.latentvar}_{args.latentcls}') 
            os.makedirs(tsne_dir, exist_ok=True)
            tsne_heatmap(embeddings_tsne,tsne_dir)
            
    if args.umap_vis:
        reducer = umap.UMAP(n_components=2, random_state=42)
        if args.embed_net =='vae':
            print('\nDimensionality reduction on raw feature emebddinsg using u-map!')
            embeddings_umap = reducer.fit_transform(embeddings)
            umap_dir = os.path.join(cluster_dir,f'umap_{args.exp}_{args.seed}_{args.latentvar}_{args.latentcls}') 
            os.makedirs(umap_dir, exist_ok=True)
            tsne_heatmap(embeddings_umap,umap_dir)
        

if __name__=='__main__':
    main(args)





