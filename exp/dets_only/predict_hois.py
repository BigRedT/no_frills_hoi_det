import os
import numpy as np
from tqdm import tqdm
import threading
import h5py

import utils.io as io
from utils.constants import save_constants


class HoiPredictor():
    def __init__(self,data_const):
        self.data_const = data_const
        self.hoi_classes = self.get_hoi_classes()

    def get_hoi_classes(self):
        hoi_list = io.load_json_object(self.data_const.hoi_list_json)
        hoi_classes = {hoi['id']:hoi for hoi in hoi_list}
        return hoi_classes

    def predict(self,selected_dets):
        threads = []
        pred_hoi_dets = {}
        for hoi_id in self.hoi_classes.keys():
            pred_hoi_dets[hoi_id] = None
            
        for hoi_id, hoi_info in self.hoi_classes.items():
            pred_hoi_dets[hoi_id] = self.predict_hoi(selected_dets,hoi_info)

        return pred_hoi_dets

    def predict_hoi(self,selected_dets,hoi_info):
        hoi_object = ' '.join(hoi_info['object'].split('_'))
        human_boxes = selected_dets['boxes']['person']
        human_scores = selected_dets['scores']['person']
        object_boxes = selected_dets['boxes'][hoi_object]
        object_scores = selected_dets['scores'][hoi_object]
        hoi_dets = []
        for i in range(human_boxes.shape[0]):
            for j in range(object_boxes.shape[0]):
                hoi_det = {
                    'human_box': human_boxes[i],
                    'object_box': object_boxes[j],
                    'score': human_scores[i]*object_scores[j],
                }
                hoi_det = np.concatenate((
                    hoi_det['human_box'],
                    hoi_det['object_box'],
                    [hoi_det['score']]
                ))
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

    hoi_predictor = HoiPredictor(data_const)    

    print('Creating a pred_dets.hdf5 dataset ...')
    pred_dets_hdf5 = os.path.join(pred_hoi_dets_dir,'pred_hoi_dets.hdf5')
    f = h5py.File(pred_dets_hdf5,'w')

    for global_id in tqdm(split_ids['test']):
        # Read selected_dets
        selected_dets_npy = os.path.join(
            data_const.selected_dets_dir,
            f'{global_id}_selected_dets.npy')
        selected_dets = np.load(selected_dets_npy)[()]

        f.create_group(global_id)
        pred_dets = hoi_predictor.predict(selected_dets)
        for hoi_id, pred_hoi_dets in pred_dets.items():
            f[global_id].create_dataset(hoi_id,data=pred_hoi_dets)

        #import pdb; pdb.set_trace()

    f.close()