import os
import h5py
import numpy as np
from tqdm import tqdm

import utils.io as io
from utils.bbox_utils import compute_area
from data.coco_classes import COCO_CLASSES


def select_det_ids(boxes,scores,nms_keep_ids,score_thresh,max_dets):
    if nms_keep_ids is None:
        nms_keep_ids = np.arange(0,scores.shape[0])
    
    # Select non max suppressed dets
    nms_scores = scores[nms_keep_ids]
    nms_boxes = boxes[nms_keep_ids]

    # Select dets above a score_thresh and which have area > 1
    nms_ids_above_thresh = np.nonzero(nms_scores > score_thresh)[0]
    nms_ids = []
    for i in range(min(nms_ids_above_thresh.shape[0],max_dets)):
        area = compute_area(nms_boxes[i],invalid=-1)
        if area > 1:
            nms_ids.append(i)
        
    # If no dets satisfy previous criterion select the highest ranking one with area > 1
    if len(nms_ids)==0:
        for i in range(nms_keep_ids.shape[0]):
            area = compute_area(nms_boxes[i],invalid=-1)
            if area > 1:
                nms_ids = [i]
                break

    # Convert nms ids to box ids
    nms_ids = np.array(nms_ids,dtype=np.int32)
    try:
        ids = nms_keep_ids[nms_ids]
    except:
        import pdb; pdb.set_trace()

    return ids
        

def select_dets(
        boxes,
        scores,
        nms_keep_indices,
        exp_const):
    selected_dets = []
    
    start_end_ids = np.zeros([len(COCO_CLASSES),2],dtype=np.int32)
    start_id = 0
    for cls_ind, cls_name in enumerate(COCO_CLASSES):
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        cls_nms_keep_ids = np.array(nms_keep_indices[cls_ind])

        if cls_name=='person':
            select_ids = select_det_ids(
                cls_boxes,
                cls_scores,
                cls_nms_keep_ids,
                exp_const.human_score_thresh,
                exp_const.max_humans)
        elif cls_name=='background':
            select_ids = select_det_ids(
                cls_boxes,
                cls_scores,
                cls_nms_keep_ids,
                exp_const.background_score_thresh,
                exp_const.max_background)
        else:
            select_ids = select_det_ids(
                cls_boxes,
                cls_scores,
                cls_nms_keep_ids,
                exp_const.object_score_thresh,
                exp_const.max_objects_per_class)
                
        boxes_scores_rpn_id = np.concatenate((
            cls_boxes[select_ids],
            np.expand_dims(cls_scores[select_ids],1),
            np.expand_dims(select_ids,1)),1)
        selected_dets.append(boxes_scores_rpn_id)
        num_boxes = boxes_scores_rpn_id.shape[0]
        start_end_ids[cls_ind,:] = [start_id,start_id+num_boxes]
        start_id += num_boxes

    selected_dets = np.concatenate(selected_dets)
    return selected_dets, start_end_ids


def select(exp_const,data_const):
    io.mkdir_if_not_exists(exp_const.exp_dir)
    
    select_boxes_dir = os.path.join(
        exp_const.exp_dir,
        f'select_boxes_' + \
        f'human_thresh_{exp_const.human_score_thresh}_' + \
        f'max_{exp_const.max_humans}_' + \
        f'object_thresh_{exp_const.object_score_thresh}_' + \
        f'max_{exp_const.max_objects_per_class}')
    io.mkdir_if_not_exists(select_boxes_dir)

    # Print where the boxes are coming from and where the output is written
    print(f'Boxes will be read from: {data_const.faster_rcnn_boxes}')
    print(f'Boxes will be written to: {select_boxes_dir}')
    
    print('Writing constants to exp dir ...')
    data_const_json = os.path.join(exp_const.exp_dir,'data_const.json')
    data_const.to_json(data_const_json)

    exp_const_json = os.path.join(exp_const.exp_dir,'exp_const.json')
    exp_const.to_json(exp_const_json)

    print('Loading anno_list.json ...')
    anno_list = io.load_json_object(data_const.anno_list_json)

    print('Creating selected_coco_cls_dets.hdf5 file ...')
    hdf5_file = os.path.join(select_boxes_dir,'selected_coco_cls_dets.hdf5')
    f = h5py.File(hdf5_file,'w')

    print('Selecting boxes ...')
    for anno in tqdm(anno_list):
        global_id = anno['global_id']

        boxes_npy = os.path.join(
            data_const.faster_rcnn_boxes,
            f'{global_id}_boxes.npy')
        boxes = np.load(boxes_npy)
        
        scores_npy = os.path.join(
            data_const.faster_rcnn_boxes,
            f'{global_id}_scores.npy')
        scores = np.load(scores_npy)
        
        nms_keep_indices_json = os.path.join(
            data_const.faster_rcnn_boxes,
            f'{global_id}_nms_keep_indices.json')
        nms_keep_indices = io.load_json_object(nms_keep_indices_json)

        selected_dets, start_end_ids = select_dets(boxes,scores,nms_keep_indices,exp_const)
        f.create_group(global_id)
        f[global_id].create_dataset('boxes_scores_rpn_ids',data=selected_dets)
        f[global_id].create_dataset('start_end_ids',data=start_end_ids)
        
    f.close()