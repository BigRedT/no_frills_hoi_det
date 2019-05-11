# Description of hoi_candidates_{subset}.hdf5

Stores information about HOI candidates such as bounding boxes, human and object detection scores, rpn ids (used to get corresponding faster-rcnn features), index of HOI category for which each pair is a candidate. Note that this index is not the ground truth HOI category of the pair but the category for which our model is going to consider the pair as a candidate.

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
- human box coordinates (x1,y1,x2,y2)   [4]
- object box coordinates (x1,y1,x2,y2)  [4]
- human score   [1]
- object score  [1]
- human rpn id  [1]
- object rpn id [1]
- hoi_idx       [1]

`hoi_id` and `hoi_idx` are related as follows:
```python
# hoi_id: string between "1" to "600" as occurring in hico_processed/hoi_list.json
# hoi_idx: conversion of hoi_id to 0 indexed integers
hoi_idx = int(hoi_id)-1
hoi_id = str(hoi_idx+1).zfill(3)
```

**`start_end_ids`** maps hoi ids to start and end indices of rows in the `boxes_scores_rpn_ids_hoi_idx`

Information about candidates for HOI category say "pet_zerba" can be obtained as follows:
```python
import h5py
import utils.io as io

def get_id(object,verb,hoi_list):
    for hoi in hoi_list:
        if hoi['object']==object and hoi['verb']==verb:
            return hoi['id']

    assert(False), 'object and verb not found in hoi_list'

# Get hoi_idx corresponding to "pet_zebra"
hoi_list = io.load_json_object(<path to hoi_list.json>)
hoi_id = get_id('zebra','pet',hoi_list)
hoi_idx = int(hoi_id)-1

# Get candidates for "pet_zebra" from an image with specified global_id
f = h5py.File(<path to hoi_candidates_{subset}.hdf5>,'r')
start_id, end_id = f[global_id]['start_end_ids'][hoi_idx]
candidates = f[global_id]['boxes_scores_rpn_ids_hoi_idx'][start_id:end_id]
human_boxes = candidates[:,:4]
object_boxes = candidates[:,4:8]
human_scores = candidates[:,8]
object_scores = candidates[:,9]
...
```

