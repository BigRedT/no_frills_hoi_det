# Description of hoi_candidates_{subset}.hdf5

Similar to `pred_hoi_dets.hdf5` but with scores for human and object separated and rpn ids provided for them as well. This can be used to get corresponding faster-rcnn features. 

## HDF5 Directory Structure
```
.
+-- global_id1
|   +-- human_obj_boxes_scores
|   +-- start_end_ids
+-- global_id2
|   +-- human_obj_boxes_scores
|   +-- start_end_ids
...
```

**human_obj_boxes_scores** is a Nx12 matrix with each row containing the following in formation in order
- human box coordinates (x,y,w,h)   [4]
- object box coordinates (x,y,w,h)  [4]
- human score   [1]
- object score  [1]
- human rpn id  [1]
- object rpn id [1]

