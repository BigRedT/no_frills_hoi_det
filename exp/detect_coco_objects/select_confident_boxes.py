import os
import numpy as np
from tqdm import tqdm

import utils.io as io
from utils.bbox_utils import compute_area
from exp.detect_coco_objects.coco_classes import COCO_CLASSES


def select_boxes_inner(dets,score_thresh,max_dets):
    ids_ = np.nonzero(dets[:,-1]>score_thresh)[0]
    ids = []
    for i, idx in enumerate(ids_.tolist()):
        if i==max_dets:
            break

        area = compute_area(dets[idx,:4],invalid=-1)
        if area > 1:
            ids.append(idx)
        
    if len(ids)==0:
        for i in range(dets.shape[0]):
            area = compute_area(dets[i,:4],invalid=-1)
            if area > 1:
                ids = [i]
                break

    ids = np.array(ids,dtype=np.int32)

    if ids.shape[0]==0:
        select_dets = None
    elif ids.shape[0] > max_dets:
        select_dets = dets[ids[:max_dets],:]
    else:
        select_dets = dets[ids,:]

    return select_dets
        

def select_boxes(
        boxes,
        scores,
        nms_keep_indices,
        exp_const):

    select_dets = {
        'human': None,
        'object': [],
        'background': None,
    }

    for cls_ind, cls_name in enumerate(COCO_CLASSES):
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        dets = dets[nms_keep_indices[cls_ind], :]
        if cls_name=='person':
            select_dets['human'] = select_boxes_inner(
                dets,
                exp_const.human_score_thresh,
                exp_const.max_humans)
        elif cls_name=='background':
            select_dets['background'] = select_boxes_inner(
                dets,
                exp_const.background_score_thresh,
                exp_const.max_background)
        else:
            select_dets_ = select_boxes_inner(
                dets,
                exp_const.object_score_thresh,
                exp_const.max_objects_per_class)
                
            if select_dets_ is not None:
                select_dets['object'].append(select_dets_)

    if len(select_dets['object'])==0:
        select_dets['object'].append(select_dets['background'])

    select_dets['object'] = np.concatenate(select_dets['object'])
    
    return select_dets


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

        select_dets = select_boxes(boxes,scores,nms_keep_indices,exp_const)
        for box_type in ['human','object']:
            boxes_npy = os.path.join(
                select_boxes_dir,
                f'{global_id}_{box_type}.npy')
            np.save(boxes_npy,select_dets[box_type])