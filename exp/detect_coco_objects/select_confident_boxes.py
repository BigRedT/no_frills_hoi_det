import os
import numpy as np
from tqdm import tqdm

import utils.io as io
from utils.bbox_utils import compute_area
from exp.detect_coco_objects.coco_classes import COCO_CLASSES


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
    selected_dets = {
        'boxes': {},
        'scores': {},
        'rpn_ids': {}
    }
    
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
                
        selected_dets['boxes'][cls_name] = cls_boxes[select_ids]
        selected_dets['scores'][cls_name] = cls_scores[select_ids]
        selected_dets['rpn_ids'][cls_name] = select_ids
    
    return selected_dets


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

        selected_dets = select_dets(boxes,scores,nms_keep_indices,exp_const)
        selected_dets_npy = os.path.join(
            select_boxes_dir,
            f'{global_id}_selected_dets.npy')
        np.save(selected_dets_npy,selected_dets)