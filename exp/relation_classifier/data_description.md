# Description of hoi_candidates_{subset}.hdf5

Similar to `pred_hoi_dets.hdf5` but with scores for human and object separated and rpn ids provided for them as well. This can be used to get corresponding faster-rcnn features. 

## HDF5 Directory Structure
```
.
+-- global_id1
|   +-- boxes_scores_rpn_ids_hoi_idx
|   +-- start_end_ids
+-- global_id2
|   +-- boxes_scores_rpn_ids_hoi_idx
|   +-- start_end_ids
...
```

## HDF5 datasets description
**`boxes_scores_rpn_ids_hoi_idx`** is a Nx13 matrix with each row containing the following in formation in order
- human box coordinates (x,y,w,h)   [4]
- object box coordinates (x,y,w,h)  [4]
- human score   [1]
- object score  [1]
- human rpn id  [1]
- object rpn id [1]
- hoi_idx       [1]

`hoi_id` and `hoi_idx` are related as follows:
```python
hoi_idx = int(hoi_id)-1
hoi_id = str(hoi_idx+1).zfill(3)
```

**`start_end_ids`** maps hoi ids to start and end indices of rows in the `human_obj_boxes_scores`