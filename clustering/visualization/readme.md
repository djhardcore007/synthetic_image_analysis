# UMAP Visualization result

In this folder, I listed UMAP visualization results for both Rareplan and Xview data.

### What is the UMAP Pipeline? ###

1. Sample 100 synthetic and real images each from full datasets.

2. Crop real images into (512, 512, 3) such that synthetic images and real images are of the same size. Now syn or real data shape is (100, 512, 512, 3)

3. Use the backbone of a pretrained detectron2 model (we get this pretrained model from our customers) to get feature maps for 5 different Feauture Pyramid Networks.

   - p2 (100, 16384) for only real or synthetic data
   - p3 (100, 4096)
   - p4 (100, 1024)
   - p5 (100, 256)
   - p6 (100, 64)

   The output is saved in '.npy files'

4. For each pyramid level, we vertically stack the real and synthetic data to get the following shape:

   - p2 (200, 16384) for both real and synthetic data
   - p3 (200, 4096)
   - p4 (200, 1024)
   - p5 (200, 256)
   - p6 (200, 64)

5. Use UMAP to reduce feautres. We fit a UMAP model using only real data (synthetic data), and then transform the real data (synthetic data) using that model. Intuitively, we have mapped the synehtic data (real data) onto a latent space of real data (synthetic data).

   - p2 (200, 16384) 	--> (200,3)
   - p3 (200, 4096)	--> (200,3)
   - p4 (200, 1024)	--> (200,3)
   - p5 (200, 256)		--> (200,3)
   - p6 (200, 64)		--> (200,3)

6. Visualization using 3D scatter plot.



### Xview

Xview model is from our customer.

I trained the model on synthetic data for **another 3000 epochs**



Distribution of test instances from 2000 images among all 4 categories:

| category     | #instances |
| ------------ | ---------- |
| Crane truck  | 1038       |
| Tower crane  | 757        |
| Mobile crane | 3798       |
| Distractors  | 0          |
| Total        | 5593       |



**Model performance:**

On **synthetic** data:

```
Evaluation results for bbox: 
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl  |
|:------:|:------:|:------:|:------:|:------:|:-----:|
| 29.781 | 55.400 | 28.533 | 37.574 | 14.537 |  nan  |
[03/29 20:00:46 d2.evaluation.coco_evaluation]: Note that some metrics cannot be computed.
[03/29 20:00:46 d2.evaluation.coco_evaluation]: Per-category bbox AP: 
| category    | AP     | category    | AP     | category     | AP     |
|:------------|:-------|:------------|:-------|:-------------|:-------|
| Crane Truck | 16.636 | Tower crane | 33.563 | Mobile Crane | 39.144 |
| Distractor  | nan    |             |        |              |        |
OrderedDict([('bbox',
              {'AP': 29.781003326661782,
               'AP-Crane Truck': 16.63583946340633,
               'AP-Distractor': nan,
               'AP-Mobile Crane': 39.14445790466138,
               'AP-Tower crane': 33.56271261191765,
               'AP50': 55.399535401604396,
               'AP75': 28.532690534042953,
               'APl': nan,
               'APm': 14.536574413689676,
               'APs': 37.57410254295049})])
```