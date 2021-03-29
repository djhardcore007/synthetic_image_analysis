'''
Usage: !python3 makeUmap.py --n_samples 100 --output_dir ... --data_dir ...
'''


import glob
from collections import defaultdict
import numpy as np
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import seaborn as sns
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os


def load_data(DATA_DIR):
    data_dic = defaultdict()
    for level in range(2,7):
        file_lst = glob.glob(DATA_DIR+"*p"+str(level)+"*.npy")
        data_dic[level] = np.vstack([np.load(f) for f in file_lst])
        print('Loading data...')
        print('p'+str(level),data_dic[level].shape)
    return data_dic


def umap_embedding(real_data, syn_data, num_components=3, isReal2Syn=True, randn_state=42):
    if isReal2Syn:
        base, map_ = real_data, syn_data
    else:
        base, map_ = syn_data, real_data
    umap_model = umap.UMAP(random_state=randn_state, n_components=num_components).fit(base)
    base_emb = umap_model.transform(base)
    map_emb = umap_model.transform(map_)
    if isReal2Syn:
        real_emb, syn_emb = base_emb, map_emb
    else:
        real_emb, syn_emb = map_emb, base_emb
    return real_emb, syn_emb


def plot_2d(real_emb, syn_emb, img_out_dir, isReal2Syn, level):
    plt.scatter(real_emb[:,0], real_emb[:,1], color='r', label='real')
    plt.scatter(syn_emb[:,0], syn_emb[:,1], color='g', label='synthetic')
    plt.legend()
    LABEL = ' (Synthetic mapped to Real latent space)' if isReal2Syn else ' (Real mapped to Synthetic latent space)'
    plt.title('Xview UMAP visualization of avg feature map: p'+str(level)+LABEL)
    plt.xlabel('latent_dim1')
    plt.ylabel('latent_dim2')
    # save visual plot
    f_name = '2d_UMAP_p'+str(level)+LABEL+'.png'
    plt.savefig(img_out_dir+f_name)
    print('UMAP vis saved at: ', img_out_dir+f_name)


def plot_3d(real_emb, syn_emb, img_out_dir, isReal2Syn, level):
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter3D(real_emb[:,0], real_emb[:,1], real_emb[:,2], marker='x', c='red', label='Real');
    ax.scatter3D(syn_emb[:,0], syn_emb[:,1], syn_emb[:,2], marker='o', c='green', label='Syn');
    plt.legend()
    LABEL = ' (Synthetic mapped to Real latent space)' if isReal2Syn else ' (Real mapped to Synthetic latent space)'
    print(LABEL)
    plt.title('Xview 3D UMAP of avg feature map: P'+str(level)+LABEL)
    plt.xlabel('latent_dim1')
    plt.ylabel('latent_dim2')
    plt.ylabel('latent_dim3')
    out_dir = img_out_dir+'3d_UMAP_p'+str(level)+LABEL+'.png'
    plt.savefig(out_dir)
    print('UMAP vis saved at: ', out_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create UMAP visualization maps')
    parser.add_argument('--output_dir', help='Output images directory', default='/content/drive/MyDrive/111 Rendered.ai/xview/clustering/training3000iters/')
    parser.add_argument('--n_samples', help='Number of samples', default=100, type=int)
    parser.add_argument('--data_dir', help='Data directory', default='/content/drive/MyDrive/111 Rendered.ai/xview/clustering/training3000iters/')
    args = parser.parse_args()
    
    OUT_DIR = args.output_dir
    N_SAMPLES = args.n_samples
    DATA_DIR = args.data_dir

    sns.set(style='white', rc={'figure.figsize':(10,8)})

    data_dic = load_data(DATA_DIR)

    for LEVEL in range(2, 7):
        real_data = data_dic[LEVEL].copy()[:N_SAMPLES]
        syn_data = data_dic[LEVEL].copy()[N_SAMPLES:]
        for isReal2Syn in [True, False]:
            real_emb, syn_emb = umap_embedding(real_data, syn_data, num_components=3, isReal2Syn=isReal2Syn)
            img_out_dir = OUT_DIR+'visualization/'
            os.makedirs(img_out_dir, exist_ok=True)
            plot_3d(real_emb, syn_emb, img_out_dir, isReal2Syn=isReal2Syn, level=LEVEL)