# Description

We developed two different methodologies to improve synthetic data generation in order 
to improve FasterRCNN/MaskRCNN performance in object detection tasks.

- latent variable models
- use saliency maps to explain Faster-RCNN

## Latent Variable Models
1. Use "crop_bbox.py" to get collections of cropped bounding boxes 
for both real and synthetic images;
2. Explore the data in "latent_models.ipynb" notebook.

## Explain Faster-RCNN
1. Separate civil role images from role data, and make a civil role coco file (see "make_civil_role_data.ipynb")
2. Get false predictions 
from a pre-trained FasterRCNN (trained by real data) (see "FasterRCNN_inference.ipynb")
3. Produce Feature Map, [Spatial Attention Map](https://arxiv.org/abs/1612.03928), Objectiveness Logits Map (see "localization.ipynb")
4. [Deep Taylor Decomposition](https://github.com/1202kbs/Understanding-NN/blob/master/2.4%20Deep%20Taylor%20Decomposition%20(1).ipynb)
    * Replace backbone params of Resnet50 using FasterRCNN's params, 
    and train the classification layer of Resnet50 while freezing its backbone (see "Resnet50_training.ipynb")
    * DTD inference using Resnet50 (see "DTD.ipynb" for demo, and "DTD_inference.ipynb" for complete procedures)
    
# Reference
1. DTD implementation on Resnet50 borrowed from [here](https://github.com/myc159/Deep-Taylor-Decomposition)
2. FasterRCNN inference adapted from work of [Rendered.AI](https://rendered.ai/)
3. We use FAIR's [detectron2](https://github.com/facebookresearch/detectron2) library for object detection 
4. Data and pre-trained models are from [RarePlanes](https://github.com/aireveries/RarePlanes)