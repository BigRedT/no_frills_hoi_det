# Description of selected_coco_cls_dets.hdf5 file

## HDF5 Directory Structure
```
.
+-- global_id1
|   +-- boxes_scores_rpn_ids
|   +-- start_end_ids
+-- global_id2
|   +-- boxes_scores_rpn_ids
|   +-- start_end_ids
...
```
Each `global_id` is an *hdf5 group* with `boxes_scores_rpn_ids` and `start_end_ids` as *hdf5 datasets*.

## HDF5 datasets description
- `boxes_scores_rpn_ids` is a Nx6 (=4+1+1) dimensional numpy array with each row containing the box coordinates (`[x1,y1,x2,y2]` where `(x1,y1)` and `(x2,y2)` are the top-left and bottom-right coordinates respectively), score for the selected class, and index of the box in the list of 300 boxes proposed by RPN in the Faster-RCNN framework. 

- `start_end_ids` is a 81x2 dimensional numpy array with i^th row containing the start and end row numbers in `box_scores_rpn_ids` for i^th class in the list of `COCO_CLASSES` (see `exp/detect_coco_objects/coco_classes.py`). So detections for i^th category in `COCO_CLASSES` for a given `global_id` are obtained by 

```python
import h5py
from data.coco_classes import COCO_CLASSES

f = h5py.File(selected_coco_cls_dets_hdf5_path,'r')
cls_name = COCO_CLASSES[i]
start_id, end_id = f[global_id]['start_end_ids'][i]
dets = f[global_id]['boxes_scores_rpn_ids'][start_id:end_id]
boxes = dets[:,:4] # Box coordinates
scores = dets[:,4] # Scores for object category cls_name
rpn_ids = dets[:,5] # ID of the box in the list of predictions made by faster-rcnn (an integer in [0,300))
```