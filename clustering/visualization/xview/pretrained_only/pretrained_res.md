# Pretrained model results

This is the xview visualization from pretrained model.

## Model Info 

**Model type**: Detectron2 FasterRCNN

**Model Labels**: Crane Truck, Mobile Crane, Tower Crane, Distractor 

**Pretrained**: Yes.

**Fine-tuned**: No

**Pretrained Model Source**: Customer

**Pretrained Model Performance**:

See the Per-category bbox AP  **on crane_look_angle_plus.003** below.

| category     | AP     |
| :----------- | :----- |
| Crane Truck  | 0.785  |
| Mobile Crane | 11.091 |
| Tower crane  | 3.589  |
| Distractor   | nan    |

| Evaluation results for bbox | AP    |
| --------------------------- | ----- |
| AP                          | 5.15  |
| AP50                        | 11.90 |
| AP75                        | 3.554 |
| APs                         | 10.84 |
| APm                         | 2.38  |
| Ap1                         | nan   |

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

