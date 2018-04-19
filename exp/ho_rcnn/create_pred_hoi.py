import os
import h5py
import scipy.io as scio
from tqdm import tqdm

import utils.io as io
import numpy as np
from data.hico.hico_constants import HicoConstants
from data.coco_classes import COCO_CLASSES


def load_all_detections(dets_mat_dir):
    dets = {}
    for i, cls_name in enumerate(tqdm(COCO_CLASSES)):
        cls_name = '_'.join(cls_name.split(' '))
        if i==0:
            continue    
        mat_name = os.path.join(
            dets_mat_dir,
            f'detections_{str(i).zfill(2)}.mat')
        dets[cls_name] = scio.loadmat(mat_name)['all_boxes']
    return dets


def group_hoi_classes_by_obj(hoi_list):
    obj_to_hoi_ids = {}
    for hoi in hoi_list:
        obj = hoi['object']
        if obj not in obj_to_hoi_ids:
            obj_to_hoi_ids[obj] = []
        obj_to_hoi_ids[obj].append(hoi['id'])
    return obj_to_hoi_ids
        

def main():
    dets_mat_dir = '/home/ssd/ho-rcnn/precomputed_hoi_detection/ho_1/' + \
        'hico_det_test2015/rcnn_caffenet_pconv_ip_iter_150000'
    outdir = '/home/tanmay/Data/weakly_supervised_hoi_exp/ho_rcnn/ho_1'
    io.mkdir_if_not_exists(outdir,recursive=True)
    pred_hoi_dets_hdf5 = os.path.join(outdir,'pred_hoi_dets.hdf5')
    pred_hoi_dets = h5py.File(pred_hoi_dets_hdf5,'w')
    data_const = HicoConstants()
    global_ids = io.load_json_object(data_const.split_ids_json)['test']
    hoi_list = io.load_json_object(data_const.hoi_list_json)
    obj_to_hoi_ids = group_hoi_classes_by_obj(hoi_list)
    print('Loading ho_rcnn dets ...')
    dets = load_all_detections(dets_mat_dir)
    for k, global_id in enumerate(tqdm(global_ids)):
        human_obj_boxes_scores = []
        start_end_ids = np.zeros([len(hoi_list),2],dtype=np.int32)
        start_id = 0
        for i, cls_name in enumerate(COCO_CLASSES):
            cls_name = '_'.join(cls_name.split(' '))
            if i==0:
                continue
            for j, hoi_id in enumerate(obj_to_hoi_ids[cls_name]):
                hoi_idx = int(hoi_id)-1
                dets_ = dets[cls_name][j][k]
                if dets_.size > 0:
                    human_obj_boxes_scores.append(dets_)
                num_dets = dets_.shape[0]
                start_end_ids[hoi_idx,:] = [start_id, start_id+num_dets]
                start_id += num_dets
        human_obj_boxes_scores = np.concatenate(human_obj_boxes_scores,0)
        pred_hoi_dets.create_group(global_id)
        pred_hoi_dets[global_id].create_dataset(
            'human_obj_boxes_scores',
            data=human_obj_boxes_scores)
        pred_hoi_dets[global_id].create_dataset(
            'start_end_ids',
            data=start_end_ids)
    pred_hoi_dets.close()


if __name__=='__main__':
    main()
