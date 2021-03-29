# UMAP Visualization result

In this folder, I listed UMAP visualization results for both Rareplan and Xview data.



### Rareplane

Rareplane model is from [AIreveries Rareplane pretained FasterRCNN on real data](https://github.com/aireveries/RarePlanes/tree/master/models) 

*Note that this model is a civil role model.*



**Model performance:** 

Boxed AP on **real** data = 68.21%

Boxed AP on **synthetic** data = 35.88%



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