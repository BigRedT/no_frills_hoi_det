import os
import numpy as np
from tqdm import tqdm

import utils.io as io
from utils.bbox_utils import compute_iou, compute_area


def recall(gt_hois,human_boxes,object_boxes,iou_thresh):
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

    
def evaluate(exp_const,data_const):
    select_boxes_dir = os.path.join(
        exp_const.exp_dir,
        f'select_boxes_' + \
        f'human_thresh_{exp_const.human_score_thresh}_' + \
        f'max_{exp_const.max_humans}_' + \
        f'object_thresh_{exp_const.object_score_thresh}_' + \
        f'max_{exp_const.max_objects_per_class}')

    print('Loading anno_list.json ...')
    anno_list = io.load_json_object(data_const.anno_list_json)
    
    print('Selecting boxes ...')
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
            continue
        else:
            num_images += 1

        boxes_npy = os.path.join(
            select_boxes_dir,
            f'{global_id}_human.npy')
        human_boxes = np.load(boxes_npy)

        boxes_npy = os.path.join(
            select_boxes_dir,
            f'{global_id}_object.npy')
        object_boxes = np.load(boxes_npy)

        try:
            recall_stats = recall(
                anno['hois'],
                human_boxes[:,:4].tolist(),
                object_boxes[:,:4].tolist(),
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
        f'eval_stats_' + \
        f'human_thresh_{exp_const.human_score_thresh}_' + \
        f'max_{exp_const.max_humans}_' + \
        f'object_thresh_{exp_const.object_score_thresh}_' + \
        f'max_{exp_const.max_objects_per_class}.json')
    io.dump_json_object(evaluation_stats,evaluation_stats_json)
    