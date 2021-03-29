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

I trained the model on synthetic data for another 3000 epochs.



Distribution of test instances among all 4 categories:

| category    | #instances | category    | #instances | category     | #instances |
| ----------- | ---------- | ----------- | ---------- | ------------ | ---------- |
| Crane truck | 1038       | Tower crane | 757        | Mobile crane | 3798       |

Total number of instances: 5593  



**Model performance:**

