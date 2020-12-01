# Overview
The project is an experiment to apply existing state-of-the-art machine learning frameworks to explore approaches 
to use synthetically generated data for object detection. 

The ability to generate synthetic data creates new opportunities to include training data generation as 
part of a closed-loop training process.

The overarching goal of the project is to develop algorithms and methodologies to help improve synthetic data generation in order to improve model performance.

# Task
The task is to detect planes in satellite images. There are three level tasks, including:

1. detect whether there is a aircraft
2. detect the role and the size of the aircraft: civil/military, small/medium/large 
3. detect the plane type: airbus330, boeing7474 etc.

We used faster-RCNN and mask-RCNN from facebook research [detectron2](https://github.com/facebookresearch/detectron2) for object detection.

# Data
We used synthetic and real data as well as pretrained models from [RarePlane](https://github.com/aireveries/RarePlanes)

# Goal
The goal of this project is to explore ways to close the feedback loop from synthetic data generation to model training to iteratively generate synthetic data 
that can be used to improve model performance. 

# Methods
There are several potential paths to completing this feedback loop, 
such as leveraging principles from active learning and research work in model explainability 
such as deep taylor decomposition.

We experimented with the following approaches:

1. Matrix Decomposition Methods like PCA/NMF 
2. Spatial Attention Map
3. Deep Taylor Decomposition