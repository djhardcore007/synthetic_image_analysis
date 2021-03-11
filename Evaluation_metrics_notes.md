# Object Detection Evaluation Metrics

This is the summary of some common object detection eval metrics.



There are various types of AP.

1. IoU: intersection of union

   IOU = area of overlap / area of union

2. AP

   precision $= \frac{tp}{tp+fp}$

   <img src="/Users/jiangdenglin/Library/Application Support/typora-user-images/image-20210310154907989.png" alt="image-20210310154907989" style="zoom:25%;" />

   How to define true positive? use IOU

3. AP50: the IoU threshold is fixed at 50%.

4. AP75: the IoU threshold is fixed at 75%.

5. AP1: the IoU threshold is fixed at 100%.

6. APm: also stands for mean average precision (mAP) 

   mAP stands for the area under PR curve (What's the difference between PR curve and ROC curve? 

7. APs



The same for AR, mAR

recall $=\frac{tp}{tp+fn}$



#### What does a high AP50 (0.8) and low APm (0.05) mean? 

The model is able to produce lots of bboxes, for 80% of the time, the model is able to capture at least 50% of the bbox. However, the recall is low, which means lot of true bbox is ignored or wrongly predicted.



### Reference:

1. [medium post by Yanfeng](https://medium.com/@yanfengliux/the-confusing-metrics-of-ap-and-map-for-object-detection-3113ba0386ef)

2. 