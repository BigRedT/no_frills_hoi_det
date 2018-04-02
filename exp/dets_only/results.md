# How far can Faster-RCNN detections take you in HICO?

## Method
**Box Selection.** For each HICO class get candidate human and *corresponding* subject boxes from Faster-RCNN (eg. all human and bike detections for human-ride-bike). The detections for an object class are selected by performing nms on RPN boxes and then selecting upto 10 highest scoring boxes for that class above the threshold of 0.1. 

**Ranking.** All candidate box pairs for a given HICO class across all images are ranked by a score. The score in this experiment is the product of human and object class scores for the box pair. Note that for a given object class all relations use the same ranking of the same set of candidate boxes since the scoring function is independent of relation category (i.e. same ranking of candidate boxes for human-relation-dog for all possible relations). 

## Evaluation
For each HICO class AP (@0.5 IOU) is computed using the interpolated area under the Precision-Recall curve. The mAP is 8.33. Even this compares favorably to the full model mAP of 7.81 reported in [1]. [1] also reports the mAP of 2.85 for an equivalent approach where a learned linear combination of the human and object detection scores is used to rank candidate box pairs. I believe the difference is because [1] uses 10 proposals per object class whereas we use variable number of proposals (often 1-2) using a hard threshold on the object scores. This helps improve precision.  Also we use Faster-RCNN instead of Fast-RCNN. Faster-RCNN is 3.4-6.6 points (mAP@0.5) better than Fast-RCNN on COCO test-dev depending on the implementation used [2]. 

## Observations
Using the Faster-RCNN box candidates, we achieve a high recall of ground truth HICO dets per image as shown in the table below, however mAP is quite low. PR curves show the performance is mainly limited by precision. See the example PR curve for HICO class human-load-airplane (class id 006).

|   |  Box Recall @ 0.5 | Box and Label Recall @ 0.5 |
| --- | :---: | :---: |
| Human | 87.19 | 84.69 |
| Object | 86.02 | 69.77 |
| Connection | 75.91 | 58.08 |

![PR Curve](/imgs/dets_only_006_pr.png)

## References
[1] Chao, Yu-Wei, et al. "Learning to Detect Human-Object Interactions." arXiv preprint arXiv:1702.05448 (2017).

[2] Ren, Shaoqing, et al. "Faster r-cnn: Towards real-time object detection with region proposal networks." Advances in neural information processing systems. 2015.
