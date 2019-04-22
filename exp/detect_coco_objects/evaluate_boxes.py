"""
Depricated: This file does not support selected boxes stored in hdf5 format. To use
these functions please replace the code where selected_dets_npy is loaded with
code to read the same information from hdf5 file. Format in which the data is 
stored in the hdf5 is described in exp/detect_coco_objects/data_description.md
"""
import os
import h5py
import numpy as np
from tqdm import tqdm

import utils.io as io
from utils.bbox_utils import compute_iou, compute_area
from data.coco_classes import COCO_CLASSES


def box_recall(gt_hois,human_boxes,object_boxes,iou_thresh):
    num_pred_human_boxes = len(human_boxes)
    num_pred_object_boxes = len(object_boxes)
    num_pred_connections = num_pred_human_boxes*num_pred_object_boxes
    
    num_gt_connections_recalled = 0
    num_gt_connections = 0
    num_gt_human_boxes_recalled = 0
    num_gt_human_boxes = 0
    num_gt_object_boxes_recalled = 0
    num_gt_object_boxes = 0

    for hois_per_type in gt_hois:
        gt_connections = hois_per_type['connections']
        gt_human_boxes = hois_per_type['human_bboxes']
        gt_object_boxes = hois_per_type['object_bboxes']
        invis = hois_per_type['invis']
        
        gt_human_boxes_recalled = [False]*len(gt_human_boxes)
        for i, gt_box in enumerate(gt_human_boxes):
            for box in human_boxes:
                try:
                    iou = compute_iou(box,gt_box)
                except:
                    import pdb; pdb.set_trace()
                if iou >= iou_thresh:
                    gt_human_boxes_recalled[i] = True
                    break

        gt_object_boxes_recalled = [False]*len(gt_object_boxes)
        for i, gt_box in enumerate(gt_object_boxes):
            for box in object_boxes:
                try:
                    iou = compute_iou(box,gt_box)
                except:
                    import pdb; pdb.set_trace()
                if iou >= iou_thresh:
                    gt_object_boxes_recalled[i] = True
                    break
                
        gt_connections_recalled = [False]*len(gt_connections)
        for k,(i,j) in enumerate(gt_connections):
            if gt_human_boxes_recalled[i] and gt_object_boxes_recalled[j]:
                gt_connections_recalled[k] = True

        num_gt_connections += len(gt_connections)
        num_gt_connections_recalled += gt_connections_recalled.count(True)

        num_gt_human_boxes += len(gt_human_boxes)
        num_gt_human_boxes_recalled += gt_human_boxes_recalled.count(True)

        num_gt_object_boxes += len(gt_object_boxes)
        num_gt_object_boxes_recalled += gt_object_boxes_recalled.count(True)

    try:
        connection_recall = num_gt_connections_recalled / num_gt_connections
    except ZeroDivisionError:
        connection_recall = None

    try:
        human_recall = num_gt_human_boxes_recalled / num_gt_human_boxes
    except ZeroDivisionError:
        human_recall = None

    try:
        object_recall = num_gt_object_boxes_recalled / num_gt_object_boxes
    except ZeroDivisionError:
        object_recall = None

    stats = {
        'connection_recall': connection_recall,
        'human_recall': human_recall,
        'object_recall': object_recall,
        'num_gt_connections_recalled': num_gt_connections_recalled,
        'num_gt_connections': num_gt_connections,
        'num_gt_human_boxes_recalled': num_gt_human_boxes_recalled,
        'num_gt_human_boxes': num_gt_human_boxes,
        'num_gt_object_boxes_recalled': num_gt_object_boxes_recalled,
        'num_gt_object_boxes': num_gt_object_boxes,
        'num_connection_proposals': num_pred_connections,
        'num_human_proposals': num_pred_human_boxes,
        'num_object_proposals': num_pred_object_boxes,
    }

    return stats


def box_label_recall(gt_hois,human_boxes,object_boxes,object_labels,iou_thresh,hoi_list):
    num_pred_human_boxes = len(human_boxes)
    num_pred_object_boxes = len(object_boxes)
    num_pred_connections = num_pred_human_boxes*num_pred_object_boxes

    hoi_dict = {hoi['id']:hoi for hoi in hoi_list}
    
    num_gt_connections_recalled = 0
    num_gt_connections = 0
    num_gt_human_boxes_recalled = 0
    num_gt_human_boxes = 0
    num_gt_object_boxes_recalled = 0
    num_gt_object_boxes = 0

    for hois_per_type in gt_hois:
        gt_id = hois_per_type['id']
        gt_hoi = hoi_dict[gt_id]

        gt_connections = hois_per_type['connections']
        gt_human_boxes = hois_per_type['human_bboxes']
        gt_object_boxes = hois_per_type['object_bboxes']
        invis = hois_per_type['invis']
        
        gt_human_boxes_recalled = [False]*len(gt_human_boxes)
        for i, gt_box in enumerate(gt_human_boxes):
            for box in human_boxes:
                try:
                    iou = compute_iou(box,gt_box)
                except:
                    import pdb; pdb.set_trace()
                if iou >= iou_thresh:
                    gt_human_boxes_recalled[i] = True
                    break

        gt_object_boxes_recalled = [False]*len(gt_object_boxes)
        for i, gt_box in enumerate(gt_object_boxes):
            for box,label in zip(object_boxes,object_labels):
                try:
                    iou = compute_iou(box,gt_box)
                except:
                    import pdb; pdb.set_trace()
                if iou >= iou_thresh and label == gt_hoi['object']:
                    gt_object_boxes_recalled[i] = True
                    break
                
        gt_connections_recalled = [False]*len(gt_connections)
        for k,(i,j) in enumerate(gt_connections):
            if gt_human_boxes_recalled[i] and gt_object_boxes_recalled[j]:
                gt_connections_recalled[k] = True

        num_gt_connections += len(gt_connections)
        num_gt_connections_recalled += gt_connections_recalled.count(True)

        num_gt_human_boxes += len(gt_human_boxes)
        num_gt_human_boxes_recalled += gt_human_boxes_recalled.count(True)

        num_gt_object_boxes += len(gt_object_boxes)
        num_gt_object_boxes_recalled += gt_object_boxes_recalled.count(True)

    try:
        connection_recall = num_gt_connections_recalled / num_gt_connections
    except ZeroDivisionError:
        connection_recall = None

    try:
        human_recall = num_gt_human_boxes_recalled / num_gt_human_boxes
    except ZeroDivisionError:
        human_recall = None

    try:
        object_recall = num_gt_object_boxes_recalled / num_gt_object_boxes
    except ZeroDivisionError:
        object_recall = None

    stats = {
        'connection_recall': connection_recall,
        'human_recall': human_recall,
        'object_recall': object_recall,
        'num_gt_connections_recalled': num_gt_connections_recalled,
        'num_gt_connections': num_gt_connections,
        'num_gt_human_boxes_recalled': num_gt_human_boxes_recalled,
        'num_gt_human_boxes': num_gt_human_boxes,
        'num_gt_object_boxes_recalled': num_gt_object_boxes_recalled,
        'num_gt_object_boxes': num_gt_object_boxes,
        'num_connection_proposals': num_pred_connections,
        'num_human_proposals': num_pred_human_boxes,
        'num_object_proposals': num_pred_object_boxes,
    }

    return stats

    
def evaluate_boxes(exp_const,data_const):
    select_boxes_dir = exp_const.exp_dir

    select_boxes_h5py = os.path.join(
        select_boxes_dir,
        'selected_coco_cls_dets.hdf5')
    select_boxes = h5py.File(select_boxes_h5py)

    print('Loading anno_list.json ...')
    anno_list = io.load_json_object(data_const.anno_list_json)

    print('Evaluating box proposals ...')
    evaluation_stats = {
        'num_gt_connections_recalled': 0,
        'num_gt_connections': 0,
        'num_gt_human_boxes_recalled': 0,
        'num_gt_human_boxes': 0,
        'num_gt_object_boxes_recalled': 0,
        'num_gt_object_boxes': 0,
        'num_connection_proposals': 0,
        'num_human_proposals': 0,
        'num_object_proposals': 0,
    }

    index_error_misses = 0
    num_images = 0
    for anno in tqdm(anno_list):
        global_id = anno['global_id']
        if 'test' in global_id:
            num_images += 1
        else:
            continue

        # selected_dets_npy = os.path.join(
        #     select_boxes_dir,
        #     f'{global_id}_selected_dets.npy')
        # selected_dets = np.load(selected_dets_npy)[()]

        boxes_scores_rpn_ids = select_boxes[global_id]['boxes_scores_rpn_ids'][()]
        start_end_ids = select_boxes[global_id]['start_end_ids'][()]
        selected_dets = {'boxes':{}}
        for cls_ind,cls_name in enumerate(COCO_CLASSES):
            start_id, end_id = start_end_ids[cls_ind]
            selected_dets['boxes'][cls_name] = boxes_scores_rpn_ids[start_id:end_id,:4]

        human_boxes = selected_dets['boxes']['person']

        object_boxes = []
        object_labels = []
        for cls_name in selected_dets['boxes']:
            if cls_name in ['person','background']:
                continue
            cls_object_boxes = selected_dets['boxes'][cls_name]
            object_boxes.append(cls_object_boxes)
            object_labels += [cls_name]*cls_object_boxes.shape[0]
        object_boxes = np.concatenate(object_boxes)

        all_boxes = np.concatenate((human_boxes,object_boxes))

        try:
            recall_stats = box_recall(
                anno['hois'],
                all_boxes.tolist(),
                all_boxes.tolist(),
                exp_const.iou_thresh)
        except IndexError:
            index_error_misses += 1
            num_images -= index_error_misses
        
        for k in evaluation_stats.keys():
            evaluation_stats[k]+= recall_stats[k]

    evaluation_stats['human_recall'] = \
        evaluation_stats['num_gt_human_boxes_recalled'] / \
        evaluation_stats['num_gt_human_boxes']
    evaluation_stats['object_recall'] = \
        evaluation_stats['num_gt_object_boxes_recalled'] / \
        evaluation_stats['num_gt_object_boxes']
    evaluation_stats['connection_recall'] = \
        evaluation_stats['num_gt_connections_recalled'] / \
        evaluation_stats['num_gt_connections']
    evaluation_stats['average_human_proposals_per_image'] = \
        evaluation_stats['num_human_proposals'] / num_images
    evaluation_stats['average_object_proposals_per_image'] = \
        evaluation_stats['num_object_proposals'] / num_images
    evaluation_stats['average_connection_proposals_per_image'] = \
        evaluation_stats['average_human_proposals_per_image'] * \
        evaluation_stats['average_object_proposals_per_image']
    evaluation_stats['index_error_misses'] = index_error_misses

    evaluation_stats_json = os.path.join(
        exp_const.exp_dir,
        f'eval_stats_boxes.json')

    io.dump_json_object(evaluation_stats,evaluation_stats_json)
    

def evaluate_boxes_and_labels(exp_const,data_const):
    select_boxes_dir = exp_const.exp_dir

    select_boxes_h5py = os.path.join(
        select_boxes_dir,
        'selected_coco_cls_dets.hdf5')
    select_boxes = h5py.File(select_boxes_h5py)

    print('Loading anno_list.json ...')
    anno_list = io.load_json_object(data_const.anno_list_json)

    print('Loading hoi_list.json ...')
    hoi_list = io.load_json_object(data_const.hoi_list_json)
    
    print('Evaluating box proposals ...')
    evaluation_stats = {
        'num_gt_connections_recalled': 0,
        'num_gt_connections': 0,
        'num_gt_human_boxes_recalled': 0,
        'num_gt_human_boxes': 0,
        'num_gt_object_boxes_recalled': 0,
        'num_gt_object_boxes': 0,
        'num_connection_proposals': 0,
        'num_human_proposals': 0,
        'num_object_proposals': 0,
    }

    index_error_misses = 0
    num_images = 0
    for anno in tqdm(anno_list):
        global_id = anno['global_id']
        if 'test' in global_id:
            num_images += 1
        else:
            continue

        boxes_scores_rpn_ids = select_boxes[global_id]['boxes_scores_rpn_ids'][()]
        start_end_ids = select_boxes[global_id]['start_end_ids'][()]
        selected_dets = {'boxes':{}}
        for cls_ind,cls_name in enumerate(COCO_CLASSES):
            start_id, end_id = start_end_ids[cls_ind]
            selected_dets['boxes'][cls_name] = boxes_scores_rpn_ids[start_id:end_id,:4]
        
        human_boxes = selected_dets['boxes']['person']

        object_boxes = []
        object_labels = []
        for cls_name in selected_dets['boxes']:
            if cls_name in ['person','background']:
                continue
            cls_object_boxes = selected_dets['boxes'][cls_name]
            object_boxes.append(cls_object_boxes)
            object_labels += [cls_name]*cls_object_boxes.shape[0]
        object_boxes = np.concatenate(object_boxes)

        all_boxes = np.concatenate((human_boxes,object_boxes))

        try:
            recall_stats = box_label_recall(
                anno['hois'],
                human_boxes.tolist(),
                object_boxes.tolist(),
                object_labels,
                exp_const.iou_thresh,
                hoi_list)
        except IndexError:
            index_error_misses += 1
            num_images -= index_error_misses
        
        for k in evaluation_stats.keys():
            evaluation_stats[k]+= recall_stats[k]

    evaluation_stats['human_recall'] = \
        evaluation_stats['num_gt_human_boxes_recalled'] / \
        evaluation_stats['num_gt_human_boxes']
    evaluation_stats['object_recall'] = \
        evaluation_stats['num_gt_object_boxes_recalled'] / \
        evaluation_stats['num_gt_object_boxes']
    evaluation_stats['connection_recall'] = \
        evaluation_stats['num_gt_connections_recalled'] / \
        evaluation_stats['num_gt_connections']
    evaluation_stats['average_human_proposals_per_image'] = \
        evaluation_stats['num_human_proposals'] / num_images
    evaluation_stats['average_object_proposals_per_image'] = \
        evaluation_stats['num_object_proposals'] / num_images
    evaluation_stats['average_connection_proposals_per_image'] = \
        evaluation_stats['average_human_proposals_per_image'] * \
        evaluation_stats['average_object_proposals_per_image']
    evaluation_stats['index_error_misses'] = index_error_misses


    evaluation_stats_json = os.path.join(
        exp_const.exp_dir,
        f'eval_stats_boxes_labels.json')

    io.dump_json_object(evaluation_stats,evaluation_stats_json)