import h5py
import copy
import itertools
import numpy as np
from torch.utils.data import Dataset

import utils.io as io
from utils.constants import Constants
from data.hico.hico_constants import HicoConstants


class FeatureConstants(HicoConstants,io.JsonSerializableClass):
    def __init__(self):
        super(FeatureConstants,self).__init__()
        self.hoi_cands_hdf5 = None
        self.hoi_cand_labels_hdf5 = None
        self.faster_rcnn_feats_hdf5 = None
        self.subset = 'train'


class Features(Dataset):
    def __init__(self,const):
        self.const = copy.deepcopy(const)
        self.hoi_cands = self.load_hdf5_file(self.const.hoi_cands_hdf5)
        self.hoi_cand_labels = self.load_hdf5_file(
            self.const.hoi_cand_labels_hdf5)
        self.faster_rcnn_feats = self.load_hdf5_file(
            self.const.faster_rcnn_feats_hdf5)
        self.global_ids = self.load_subset_ids(self.const.subset)
        self.sample_ids, self.global_id_to_num_cands = \
            self.create_list_of_sample_ids(self.global_ids)
        self.hoi_dict = self.get_hoi_dict(self.const.hoi_list_json)
        self.obj_to_hoi_ids = self.get_obj_to_hoi_ids(self.hoi_dict)

    def load_hdf5_file(self,hdf5_filename,mode='r'):
        return h5py.File(hdf5_filename,mode)

    def get_hoi_dict(self,hoi_list_json):
        hoi_list = io.load_json_object(hoi_list_json)
        hoi_dict = {hoi['id']: hoi for hoi in hoi_list}
        return hoi_dict

    def get_obj_to_hoi_ids(self,hoi_dict):
        obj_to_hoi_ids = {}
        for hoi_id, hoi in hoi_dict.items()
            obj = hoi['object']
            if obj in obj_to_hoi_ids:
                obj_to_hoi_ids[obj].append(hoi_id)
            else:
                obj_to_hoi_ids[obj] = [hoi_id]
        return obj_to_hoi_ids

    def load_subset_ids(self,subset):
        split_ids = io.load_json_object(self.const.split_ids_json)
        return sorted(split_ids[subset])

    def create_list_of_sample_ids(self,global_ids):
        sample_ids = []
        global_id_to_num_cands = {}
        for global_id in global_ids:
            num_cands = self.hoi_cand_labels[global_id].shape[0]
            sample_ids += list(itertools.product([global_id],range(num_cands)))
            global_id_to_num_cands[global_id] = num_cands
        return sample_ids, global_id_to_num_cands

    def __len__(self):
        return len(self.sample_ids)

    def get_label(self,global_id,cand_id):
        hoi_label_idx = int(self.hoi_cand_labels[global_id][cand_id])
        hoi_label_vec = np.zeros([len(self.idx_to_hoi_id)])
        if hoi_label_idx==-1:
            hoi_id = 'background'
        else:
            hoi_id = str(hoi_label_idx+1).zfill(3)
            hoi_label_vec[hoi_label_idx] = 1.0
        return hoi_id, hoi_label_vec

    def get_faster_rcnn_prob_vecs(self,hoi_id,human_prob,object_prob):
        num_hois = len(self.hoi_dict)
        human_prob_vec = human_prob*np.ones([num_hois])
        object_prob_vec = np.zeros([num_hois])
        obj = self.hoi_dict[hoi_id]
        for other_hoi_id in self.hoi_dict[hoi_id]:
            object_prob_vec[int(other_hoi_id)-1] = object_prob
        return human_prob_vec, object_prob_vec

    def __getitem__(self,i):
        global_id, cand_id = self.sample_ids[i]
        hoi_cand = self.hoi_cands[global_id]['boxes_scores_rpn_ids'][cand_id]
        hoi_id, hoi_label_vec = self.get_label(global_id,cand_id)
        to_return = {
            'global_id': global_id,
            'human_box': hoi_cand[:4],
            'object_box': hoi_cand[4:8],
            'human_prob': hoi_cand[8],
            'object_prob': hoi_cand[9],
            'human_rpn_id': hoi_cand[10],
            'object_rpn_id': hoi_cand[11],
            'hoi_id': hoi_id,
            'hoi_label_vec': hoi_label_vec,
        }
        to_return['human_feat'] = \
            self.faster_rcnn_feats[global_id][to_return['human_rpn_id']]
        to_return['object_feat'] = \
            self.faster_rcnn_feats[global_id][to_return['object_rpn_id']]
        human_prob_vec, object_prob_vec = self.get_faster_rcnn_prob_vecs(
            to_return['hoi_id'], 
            to_return['human_prob'],
            to_return['object_prob'])
        to_return['human_prob_vec'] = human_prob_vec
        to_return['object_prob_vec'] = object_prob_vec
        return to_return
        



    

