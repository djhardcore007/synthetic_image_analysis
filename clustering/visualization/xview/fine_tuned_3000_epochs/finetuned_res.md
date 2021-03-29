

# Fine-tuned model results

This is the xview visualization from the fine-tuned model.

## Model Info 

**Model type**: Detectron2 FasterRCNN

**Model Labels**: Crane Truck, Mobile Crane, Tower Crane, Distractor 

**Pretrained**: Yes. using Real Rareplane data

**Pretrained Model Source**: Customer

**Fine-tuned**: Yes. The pretrained model is **further finetuned for another 3000 epochs** using synthetic data (crane_look_angle_plus.003)

**Fine-tuned Model Performance**:

See the Per-category bbox AP below.
| category     | AP     |
| :----------- | :----- |
| Crane Truck  | 16.636 |
| Mobile Crane | 39.144 |
| Tower crane  | 33.563 |
| Distractor   | nan    |

| Evaluation results for bbox | AP     |
| --------------------------- | ------ |
| AP                          | 29.781 |
| AP50                        | 55.400 |
| AP75                        | 28.533 |
| APs                         | 37.574 |
| APm                         | 14.537 |
| Ap1                         | nan    |

Distribution of test instances from 2000 images among all 4 categories:

| category     | #instances |
| ------------ | ---------- |
| Crane truck  | 1038       |
| Tower crane  | 757        |
| Mobile crane | 3798       |
| Distractors  | 0          |
| Total        | 5593       |



## Data Info

| Data type | Description                             | Shape                                |
| --------- | --------------------------------------- | ------------------------------------ |
| Real      | cropped xview real data from xview repo | (3, w, h) --> crop --> (3, 512, 512) |
| Synthetic | crane_look_angle_plus.003               | (3, 512, 512)                        |

*Cropping: selecting top-left part of the images.*

