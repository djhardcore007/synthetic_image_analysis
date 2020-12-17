# About the Project
The project is an experiment to apply existing state-of-the-art machine learning frameworks to explore approaches 
to use synthetically generated data for object detection. 

The ability to generate synthetic data creates new opportunities to include training data generation as 
part of a closed-loop training process.

The overarching goal of the project is to develop algorithms and methodologies to help improve synthetic data generation in order to improve model performance.

# Underlying Task

## Data
We used synthetic and real data as well as pretrained models from [RarePlane](https://github.com/aireveries/RarePlanes)
## Task
The underlying task is to detect planes in satellite images. There are three level tasks, including:

1. detect whether there is an aircraft
2. detect the role and the size of an aircraft: civil/military, small/medium/large 
3. detect an aircraft type: Boeing747 etc.

## Model
We used faster-RCNN and mask-RCNN from facebook research [detectron2](https://github.com/facebookresearch/detectron2) for object detection.
We start with pre-trained Faster-RCNN on real data from [RarePlane](https://github.com/aireveries/RarePlanes).

# Goal
The goal of this project is to explore ways to close the feedback loop from synthetic data generation to model training to iteratively generate synthetic data 
that can be used to improve model performance. 

# Methods
There are several potential paths to completing this feedback loop, 
such as leveraging principles from active learning and research work in model explainability 
such as deep taylor decomposition.

We experimented with the following approaches:

1. Latent Variable Models: 
    - PCA
    - NMF
    - VAE 
2. Localization Algorithms:
    - Spatial Attention Map
    - CAM
    - Deep Taylor Decomposition
    
# Result
Please refer to the report.

# Usage

For latent variable models, you need first to get reshaped bbox from real and synthetic data:
```
!python3 crop_bbox.py real_img_dir syn_img_dir real_coco_dir syn_coco_dir select_label n_instances output_dir
```
Then you could explore cropped image collections with PCA/NMF jupyter notebooks.

As for localization algorithms, this repo built and trained models using [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb). 
 
    
# Contributions
This repo exists thanks to our mentor [Lucy Wang](linkedin.com/in/lucy-wang-5a560525) (lucywang1189@gmail.com) from [Rendered.AI](https://rendered.ai/) and the team:
Diwen Lu (dl3209@nyu.edu), Yuwei Wang (yw1854@nyu.edu)