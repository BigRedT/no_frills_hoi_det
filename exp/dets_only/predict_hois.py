import os
import numpy as np
from tqdm import tqdm
import threading
import h5py

import utils.io as io
from utils.constants import save_constants
from exp.detect_coco_objects.coco_classes import COCO_CLASSES


class HoiPredictor():
    def __init__(self,data_const):
        self.data_const = data_const
        self.hoi_classes = self.get_hoi_classes()
        
    def get_hoi_classes(self):
        hoi_list = io.load_json_object(self.data_const.hoi_list_json)
        hoi_classes = {hoi['id']:hoi for hoi in hoi_list}
        return hoi_classes

    def predict(self,selected_dets):
        pred_hoi_dets = []
        start_end_ids = np.zeros([len(self.hoi_classes),2],dtype=np.int32)
        start_id = 0
        for hoi_id, hoi_info in self.hoi_classes.items():
            dets = self.predict_hoi(selected_dets,hoi_info)
            pred_hoi_dets.append(dets)
            start_end_ids[int(hoi_id)-1,:] = [start_id,start_id+dets.shape[0]]
            start_id += dets.shape[0]

        pred_hoi_dets = np.concatenate(pred_hoi_dets)
        return pred_hoi_dets, start_end_ids

    def predict_hoi(self,selected_dets,hoi_info):
        hoi_object = ' '.join(hoi_info['object'].split('_'))
        human_boxes = selected_dets['boxes']['person']
        human_scores = selected_dets['scores']['person']
        object_boxes = selected_dets['boxes'][hoi_object]
        object_scores = selected_dets['scores'][hoi_object]
        hoi_dets = []
        for i in range(human_boxes.shape[0]):
            for j in range(object_boxes.shape[0]):
                hoi_det = np.concatenate((
                    human_boxes[i],
                    object_boxes[j],
                    [human_scores[i]*object_scores[j]]))
                hoi_dets.append(hoi_det)

        hoi_dets = np.stack(hoi_dets,0)

        return hoi_dets


def main(exp_const,data_const):
    print(f'Creating exp_dir: {exp_const.exp_dir}')
    io.mkdir_if_not_exists(exp_const.exp_dir)

    save_constants({'exp': exp_const,'data': data_const},exp_const.exp_dir)

    print(f'Creating pred_hoi_dets dir ...')
    pred_hoi_dets_dir = os.path.join(exp_const.exp_dir,'pred_hoi_dets')
    io.mkdir_if_not_exists(pred_hoi_dets_dir)

    print(f'Reading split_ids.json ...')
    split_ids = io.load_json_object(data_const.split_ids_json)

    print('Creating an object-detector-only HOI detector ...')
    hoi_predictor = HoiPredictor(data_const)    

    print('Creating a pred_hoi_dets.hdf5 dataset ...')
    pred_dets_hdf5 = os.path.join(pred_hoi_dets_dir,'pred_hoi_dets.hdf5')
    f = h5py.File(pred_dets_hdf5,'w')

    print('Reading selected dets from hdf5 file ...')
    all_selected_dets = h5py.File(data_const.selected_dets_hdf5,'r')

    for global_id in tqdm(split_ids['test']):
        selected_dets = {
            'boxes': {},
            'scores': {}
        }
        start_end_ids = all_selected_dets[global_id]['start_end_ids'].value
        boxes_scores_rpn_ids = \
            all_selected_dets[global_id]['boxes_scores_rpn_ids'].value

        for cls_ind, cls_name in enumerate(COCO_CLASSES):
            start_id,end_id = start_end_ids[cls_ind]
            boxes = boxes_scores_rpn_ids[start_id:end_id,:4]
            scores = boxes_scores_rpn_ids[start_id:end_id,4]
            selected_dets['boxes'][cls_name] = boxes
            selected_dets['scores'][cls_name] = scores

        pred_dets, start_end_ids = hoi_predictor.predict(selected_dets)
        f.create_group(global_id)
        f[global_id].create_dataset('human_obj_boxes_scores',data=pred_dets)
        f[global_id].create_dataset('start_end_ids',data=start_end_ids)

    f.close()