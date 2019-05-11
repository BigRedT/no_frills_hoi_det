# Description of pred_hoi_dets.hdf5 file
This is the format in which the hoi detections from any model need to be saved in order to use `exp/hico_eval/compute_map.sh/py` files for mAP evaluation. 

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

## HDF5 datasets description
**`human_obj_boxes_scores`** is a Nx9 (=4+4+1) dimensional numpy array with each row containing the box coordinates (`x1,y1,x2,y2`) for the human and object boxes, and score for the predicted hoi class. 

**`start_end_ids`** is a 600x2 dimensional numpy array with i^th row containing the start and end row numbers in `box_scores_rpn_ids` for i^th class in the [list of hico classes](http://napoli18.eecs.umich.edu/public_html/data/hico_list_hoi.txt). Since the hoi class ids begin with '001', for an hoi class with id '006', the set of detections for a given `global_id` are obtained by 

```python
f = h5py.File(pred_dets_hdf5_path,'r')
hoi_id = '006'
start_id, end_id = f[global_id]['start_end_ids'][int(hoi_id)-1]
hoi_dets = f[global_id]['human_obj_boxes_scores'][start_id:end_id]
human_boxes = hoi_dets[:,:4]
object_boxes = hoi_dets[:,4:8]
scores = hoi_dets[:,8]
```