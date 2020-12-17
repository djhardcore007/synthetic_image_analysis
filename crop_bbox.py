#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Usage: python3 crop_bbox.py real_img_dir syn_img_dir real_coco_dir syn_coco_dir select_label 500 output_dir
    where select_label could only be the role of the civil planes: large, medium or small
    and 500 means selecting only 500 instances from both real and synthetic image collections
    Goal: Crop out bbox from original images.
"""


import sys
import os
import json
import pandas as pd
import math
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm


def load_ann(coco_dir):
    with open(coco_dir) as json_file:
        coco = json.load(json_file)
    print(coco.keys())
    print("Num images:{}".format(len(coco['images'])))
    print("Num classes:{}".format(len(coco['categories'])))
    print("Num bbox/instances:{}".format(len(coco['annotations'])))
    img_list = pd.DataFrame(coco['images'])
    annotations = pd.DataFrame(coco['annotations'])
    ann = annotations.merge(img_list, how='left', left_on="image_id", right_on="id")
    print(ann.shape)
    return ann


def crop_image(original_img_folder, file_name, bbox):
    ori_img_dir = original_img_folder + file_name
    my_image = cv2.imread(ori_img_dir)
    buffer = 0
    cropped_im = my_image[math.floor(bbox[1]) - buffer: math.floor(bbox[1]) + math.ceil(bbox[3]) + buffer,
                 math.floor(bbox[0]) - buffer: math.floor(bbox[0]) + math.ceil(bbox[2]) + buffer, :]
    return cropped_im


def get_bbox(image_dir, ann, n_instances=500, save_bbox=False, output_dir=None):
    crop_ims = []
    for i in tqdm(range(n_instances)):
        instance = ann.iloc[i]
        file_name = instance['file_name']
        bbox = instance['bbox']
        cropped_im = crop_image(image_dir, file_name, bbox)
        crop_ims.append(cropped_im)
        # save cropped images
        if save_bbox:
            np.save(output_dir + str(i) + "_" + str(file_name)[:-4], cropped_im)
    return crop_ims


def get_max_shape(crop_ims):
    """
    input: crop_ims: list of cropped images
    """
    crop_ims_shape = [im.shape for im in crop_ims]
    max_shape, min_shape = max(crop_ims_shape), min(crop_ims_shape)
    # print(min_shape, max_shape)
    return max(max_shape)


def upsampling(crop_im, max_shape):
    crop_im = crop_im.view(1, crop_im.shape[0], crop_im.shape[1], crop_im.shape[2]).permute(0, 3, 1, 2)
    # print(crop_im.shape)
    upsampler = nn.Upsample(size=(max_shape[0], max_shape[1]), mode="nearest")
    upsampled_crop_im = upsampler(crop_im)
    # print(upsampled_crop_im.shape)
    return upsampled_crop_im


def get_resampled_img_collections(crop_ims, max_size):
    img_collection = []
    for i in tqdm(range(len(crop_ims))):
        output_size = [max_size, max_size, 3]
        upsampled_crop_im = upsampling(torch.tensor(crop_ims[i]), output_size)
        upsampled_crop_im = upsampled_crop_im.permute(0,2,3,1)
        img_collection.append(upsampled_crop_im)
    return img_collection


def get_resampled_img_collections(crop_ims, max_size, output_dir):
    """
    output crop_ims shape (1, height, width, n_channel=3)
    """
    for i in tqdm(range(len(crop_ims))):
        output_size = [max_size, max_size, 3]
        # make sure weight and height of the cropped images > 0
        if crop_ims[i].shape[0]>0 and crop_ims[i].shape[1]>0:
            upsampled_crop_im = upsampling(torch.tensor(crop_ims[i]), output_size)
            upsampled_crop_im = upsampled_crop_im.permute(0,2,3,1)
            np.save(output_dir+"/cropped_instance_"+str(i), upsampled_crop_im.numpy()[0])


if __name__ == '__main__':
    real_img_dir = sys.argv[1]
    syn_img_dir = sys.argv[2]
    real_coco_dir = sys.argv[3]
    syn_coco_dir = sys.argv[4]
    select_label = sys.argv[5]  # could only be large, medium and small
    n_instance = sys.argv[6]
    output_dir = sys.argv[7]

    # load synthetic and real data coco file
    real_ann = load_ann(real_coco_dir)
    syn_ann = load_ann(syn_coco_dir)

    # select instances whose label == "large"
    if select_label:
        label2id_dic = {"large": 3, "medium": 2, "small": 1}
        large_syn_ann = syn_ann[["file_name", "bbox", "category_id"]][syn_ann["category_id"] == label2id_dic[select_label]]
        large_real_ann = real_ann[["file_name", "bbox", "category_id"]][real_ann["category_id"] == label2id_dic[select_label] - 1]
        syn_ann, real_ann = large_syn_ann, large_real_ann

    print("num syn instances: {}, num real instances".format(syn_ann.shape[0], real_ann.shape[0]))

    # crop bbox from original images
    crops_ims_real = get_bbox(real_img_dir, real_ann, n_instances=n_instance)
    crops_ims_syn = get_bbox(syn_img_dir, syn_ann, n_instances=n_instance)

    # up-sampling
    real_max, syn_max = get_max_shape(crops_ims_real), get_max_shape(crops_ims_syn)
    max_size = max(real_max, syn_max)

    # create syn/real paths
    real_path = os.path.join(output_dir, "real")
    if not os.path.exists(real_path):
        os.mkdir(real_path)
    syn_path = os.path.join(output_dir, "synthetic")
    if not os.path.exists(syn_path):
        os.mkdir(syn_path)

    # crop and save results
    print("Cropping bbox and saving results...")
    get_resampled_img_collections(crops_ims_real, max_size, real_path)
    get_resampled_img_collections(crops_ims_syn, max_size, syn_path)
    print("Done!")
    print("Cropped real data saved to: " + real_path)
    print("Cropped synthetic data saved to: " + syn_path)