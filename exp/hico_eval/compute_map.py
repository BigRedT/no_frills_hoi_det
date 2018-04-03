import os
import argparse
import time
import h5py
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.metrics import average_precision_score, precision_recall_curve

import utils.io as io
from utils.bbox_utils import compute_iou

parser = argparse.ArgumentParser()
parser.add_argument(
    '--hico_dets_dir', 
    type=str, 
    default=None,
    required=True,
    help='Path to predicted hico detections directory')
parser.add_argument(
    '--out_dir', 
    type=str, 
    default=None,
    required=True,
    help='Output directory')
parser.add_argument(
    '--proc_dir',
    type=str,
    default=None,
    required=True,
    help='Path to HICO processed data directory')
parser.add_argument(
    '--subset',
    type=str,
    default='test',
    choices=['train','test','val','train_val'],
    help='Subset of data to run the evaluation on')
parser.add_argument(
    '--num_processes',
    type=int,
    default=12,
    help='Number of processes to parallelize across')   


def match_hoi(pred_det,gt_dets):
    is_match = False
    for gt_det in gt_dets:
        human_iou = compute_iou(pred_det['human_box'],gt_det['human_box'])
        if human_iou > 0.5:
            object_iou = compute_iou(pred_det['object_box'],gt_det['object_box'])
            if object_iou > 0.5:
                is_match = True
                break

    return is_match


def compute_ap(precision,recall):
    if np.any(np.isnan(recall)):
        return np.nan

    ap = 0
    for t in np.arange(0,1.1,0.1): # 0, 0.1, 0.2, ..., 1.0
        selected_p = precision[recall>=t]
        if selected_p.size==0:
            p = 0
        else:
            p = np.max(selected_p)   
        ap += p/11.
    
    return ap


def compute_pr(y_true,y_score,npos):
    sorted_y_true = [y for y,_ in 
        sorted(zip(y_true,y_score),key=lambda x: x[1],reverse=True)]
    tp = np.array(sorted_y_true)
    fp = ~tp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    if npos==0:
        recall = np.nan*tp
    else:
        recall = tp / npos
    precision = tp / (tp + fp)
    return precision, recall


def eval_hoi(hoi_id,global_ids,gt_dets,pred_dets_hdf5,out_dir):
    print(f'Evaluating hoi_id: {hoi_id} ...')
    pred_dets = h5py.File(pred_dets_hdf5,'r')
    y_true = []
    y_score = []
    npos = 0
    for global_id in global_ids:
        if hoi_id in gt_dets[global_id]:
            candidate_gt_dets = gt_dets[global_id][hoi_id]
        else:
            candidate_gt_dets = []
        npos += len(candidate_gt_dets)

        hoi_dets = pred_dets[global_id][hoi_id].value
        for i in range(hoi_dets.shape[0]):     
            pred_det = {
                'human_box': hoi_dets[i,:4],
                'object_box': hoi_dets[i,4:8],
                'score': hoi_dets[i,8]
            }
            y_true.append(match_hoi(pred_det,candidate_gt_dets))
            y_score.append(pred_det['score'])

    # Compute PR
    # precision,recall,_ = precision_recall_curve(y_true,y_score)
    precision,recall = compute_pr(y_true,y_score,npos)

    # Compute AP
    # ap = average_precision_score(y_true,y_score)
    ap = compute_ap(precision,recall)
    print(f'AP:{ap}')

    # Plot PR curve
    plt.figure()
    plt.step(recall,precision,color='b',alpha=0.2,where='post')
    plt.fill_between(recall,precision,step='post',alpha=0.2,color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.4f}'.format(ap))
    plt.savefig(
        os.path.join(out_dir,f'{hoi_id}_pr.png'),
        bbox_inches='tight')
    plt.close()

    # Save AP data
    ap_data = {
        'y_true': y_true,
        'y_score': y_score,
        'precision': precision,
        'recall': recall,
        'ap': ap,
    }
    np.save(
        os.path.join(out_dir,f'{hoi_id}_ap_data.npy'),
        ap_data)

    return (ap, hoi_id)


def load_gt_dets(proc_dir,global_ids_set):
    # Load anno_list
    print('Loading anno_list.json ...')
    anno_list_json = os.path.join(proc_dir,'anno_list.json')
    anno_list = io.load_json_object(anno_list_json)

    gt_dets = {}
    for anno in anno_list:
        if anno['global_id'] not in global_ids_set:
            continue

        global_id = anno['global_id']
        gt_dets[global_id] = {}
        for hoi in anno['hois']:
            hoi_id = hoi['id']
            gt_dets[global_id][hoi_id] = []
            for human_box_num, object_box_num in hoi['connections']:
                human_box = hoi['human_bboxes'][human_box_num]
                object_box = hoi['object_bboxes'][object_box_num]
                det = {
                    'human_box': human_box,
                    'object_box': object_box,
                }
                gt_dets[global_id][hoi_id].append(det)

    return gt_dets


def read_pred_dets_npy(global_id,pred_dets_dir):
    print(f'Loading {global_id} ...')
    pred_dets_npy = os.path.join(
        pred_dets_dir,
        f'{global_id}_pred_hoi_dets.npy')
    return global_id, np.load(pred_dets_npy)[()]


def load_pred_dets(global_ids,pred_dets_dir,num_processes):
    starmap_inputs = []
    for global_id in global_ids:
        starmap_inputs.append((global_id,pred_dets_dir))

    p = Pool(num_processes)
    output = p.starmap(read_pred_dets_npy,starmap_inputs)
    p.close()
    p.join()
    
    pred_dets = {}
    for global_id, dets in output:
        pred_dets[global_id] = dets
    
    return pred_dets


def main():
    args = parser.parse_args()

    print('Creating output dir ...')
    io.mkdir_if_not_exists(args.out_dir)

    # Load hoi_list
    hoi_list_json = os.path.join(args.proc_dir,'hoi_list.json')
    hoi_list = io.load_json_object(hoi_list_json)

    # Load subset ids to eval on
    split_ids_json = os.path.join(args.proc_dir,'split_ids.json')
    split_ids = io.load_json_object(split_ids_json)
    global_ids = split_ids[args.subset]
    global_ids_set = set(global_ids)

    # Create gt_dets
    print('Creating GT dets ...')
    gt_dets = load_gt_dets(args.proc_dir,global_ids_set)

    # Load predictions
    print('Loading predicted dets ...')
    pred_dets_hdf5 = os.path.join(args.hico_dets_dir,'pred_hoi_dets.hdf5')
    #pred_dets = h5py.File(pred_dets_hdf5,'r')
    # if os.path.exists(pred_dets_npy):
    #     print(f'    Reading from {pred_dets_npy}...')
    #     pred_dets = np.load(pred_dets_npy)[()]
    # else:
    #     print(f'    File not found {pred_dets_npy}; creating one')
    #     pred_dets = load_pred_dets(
    #         global_ids,
    #         args.hico_dets_dir,
    #         args.num_processes)
        
    #     print('Saving all pred dets in a single file ...')
    #     pred_dets_npy = os.path.join(args.hico_dets_dir,'pred_hoi_dets.npy')
    #     np.save(pred_dets_npy,pred_dets)

    eval_inputs = []
    for hoi in hoi_list:
        eval_inputs.append(
            (hoi['id'],global_ids,gt_dets,pred_dets_hdf5,args.out_dir))


    print(f'Starting a pool of {args.num_processes} workers ...')
    p = Pool(args.num_processes)

    print(f'Begin mAP computation ...')
    output = p.starmap(eval_hoi,eval_inputs)
    #output = eval_hoi('003',global_ids,gt_dets,pred_dets,args.out_dir)

    p.close()
    p.join()

    mAP = {
        'AP': {},
        'mAP': 0,
        'invalid': 0,
    }
    map_ = 0
    count = 0
    for ap,hoi_id in output:
        mAP['AP'][hoi_id] = ap
        if not np.isnan(ap):
            count += 1
            map_ += ap

    mAP['mAP'] = map_ / count
    mAP['invalid'] = len(output) - count

    mAP_json = os.path.join(
        args.out_dir,
        'mAP.json') 
    io.dump_json_object(mAP,mAP_json)
    import pdb; pdb.set_trace()

if __name__=='__main__':
    main()

    