'''
Usage: !python3 predict_backbone.py --model_dir ... --output_dir ... --data_dir ...
'''


# Make sure you installed all necessary libs before running this script.
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# Import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
# Import some common libraries
import json
import pandas as pd
import math
from tqdm import tqdm 
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import os
import torch 
import torchvision
import torch.nn as nn 
from IPython.display import Image 
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import PIL
import requests
from sklearn import decomposition    
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from matplotlib.pyplot import imshow
from PIL import Image
import IPython
import argparse


def cv2_imshow(img):
    img = img[:,:,[2,1,0]]
    img = Image.fromarray(img)
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def normalize(img_collection):
    data = torch.Tensor(img_collection)
    # norm_data = (data - data.numpy().mean())/data.numpy().std()
    return data


def inference(data_syn, data_type, batch_size=1, img_size=512, output_dir='/content/drive/MyDrive/111 Rendered.ai/xview/clustering/training3000iters/'):
    dataloader = DataLoader(data_syn, batch_size=1,
                            shuffle=False, num_workers=0)
    # Create feature maps for each level
    for level in range(2,7):
        real_ys = []
        torch.cuda.empty_cache()
        print('This is level:', level)
        for batch_id, images in tqdm_notebook(enumerate(dataloader)):
            torch.cuda.empty_cache()
            images = images.cuda()
            x_real = images.reshape((batch_size, 3, img_size, img_size))
            y_real = model.backbone(x_real)['p'+str(level)].sum(dim=1).reshape(batch_size, -1).cpu().detach()
            real_ys.append(y_real)
        y_real = torch.cat(real_ys, dim=0)
        print('p'+str(level), y_real.shape)
        
        # Save output
        data = y_real.numpy()
        print(data.shape)
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_dir+'pretrained_detectron2_p'+str(level)+'_mean_'+data_type, data)
        print('Done! level:',level)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create UMAP visualization maps')
    parser.add_argument('--output_dir', help='Output directory', default='/content/drive/MyDrive/111 Rendered.ai/xview/clustering/')
    parser.add_argument('--data_dir', help='Data directory', default='/content/drive/MyDrive/111 Rendered.ai/xview/clustering/')
    parser.add_argument('--model_dir', help='Data directory', default='/content/drive/My Drive/111 Rendered.ai/xview/detectron2_models/')
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir 
    DATA_DIR = args.data_dir
    MODEL_DIR = args.model_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 12345
    random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    real_images = np.load(DATA_DIR+'real_images.npy')
    syn_images = np.load(DATA_DIR+'syn_images.npy')

    # Normalize data
    data_real = normalize(real_images)
    data_syn = normalize(syn_images)
    print(data_real.dtype, data_real.shape, data_syn.dtype, data_syn.shape)

    # Check images
    # plt.imshow(data_real[0].numpy().astype(np.int8));plt.show();
    # plt.imshow(data_syn[0].numpy().astype(np.int8));plt.show();

    # Load pretrained model
    model_weight_dir = 'model_final.pth'    # change your dir here
    yaml_dir = 'RCNN_Baseline.yaml'         # change your dir here
    cfg = get_cfg()
    cfg.merge_from_file(MODEL_DIR+yaml_dir)
    cfg.MODEL.WEIGHTS = MODEL_DIR+model_weight_dir
    cfg.OUTPUT_DIR = MODEL_DIR
    model = build_model(cfg)  # returns a torch.nn.Module
    checkpointer = DetectionCheckpointer(model).load(MODEL_DIR+model_weight_dir)  # load a file, usually from cfg.MODEL.WEIGHTS

    # Inference 
    model.eval()
    print('Inference on real data...')
    inference(data_real, 'real')
    print('Inference on synthetic data...')
    inference(data_syn, 'syn')