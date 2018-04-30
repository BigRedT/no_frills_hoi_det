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
        self.box_feats_hdf5 = None
        self.balanced_sampling = True
        self.fp_to_tp_ratio = 1000
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
        self.hoi_dict = self.get_hoi_dict(self.const.hoi_list_json)
        self.obj_to_hoi_ids = self.get_obj_to_hoi_ids(self.hoi_dict)
        self.obj_to_id = self.get_obj_to_id(self.const.object_list_json)
        self.verb_to_id = self.get_verb_to_id(self.const.verb_list_json)
        if self.const.box_feats_hdf5:
            self.box_feats = self.load_hdf5_file(self.const.box_feats_hdf5)

    def load_hdf5_file(self,hdf5_filename,mode='r'):
        return h5py.File(hdf5_filename,mode)

    def get_hoi_dict(self,hoi_list_json):
        hoi_list = io.load_json_object(hoi_list_json)
        hoi_dict = {hoi['id']: hoi for hoi in hoi_list}
        return hoi_dict

    def get_obj_to_id(self,object_list_json):
        object_list = io.load_json_object(object_list_json)
        obj_to_id = {obj['name']:obj['id'] for obj in object_list}
        return obj_to_id

    def get_verb_to_id(self,verb_list_json):
        verb_list = io.load_json_object(verb_list_json)
        verb_to_id = {verb['name']:verb['id'] for verb in verb_list}
        return verb_to_id

    def get_obj_to_hoi_ids(self,hoi_dict):
        obj_to_hoi_ids = {}
        for hoi_id, hoi in hoi_dict.items():
            obj = hoi['object']
            if obj in obj_to_hoi_ids:
                obj_to_hoi_ids[obj].append(hoi_id)
            else:
                obj_to_hoi_ids[obj] = [hoi_id]
        return obj_to_hoi_ids

    def load_subset_ids(self,subset):
        split_ids = io.load_json_object(self.const.split_ids_json)
        return sorted(split_ids[subset])

    def __len__(self):
        return len(self.global_ids)

    def get_labels(self,global_id):
        # hoi_idx: number in [0,599]
        hoi_cands = self.hoi_cands[global_id]
        hoi_idxs = hoi_cands['boxes_scores_rpn_ids_hoi_idx'][:,-1]
        hoi_idxs = hoi_idxs.astype(np.int)
        # label: 0/1 indicating if there was a match with any gt for that hoi
        labels = self.hoi_cand_labels[global_id][()]
        num_cand = labels.shape[0]
        hoi_label_vecs = np.zeros([num_cand,len(self.hoi_dict)])
        hoi_label_vecs[np.arange(num_cand),hoi_idxs] = labels
        hoi_ids = [None]*num_cand
        for i in range(num_cand):
            hoi_ids[i] = str(hoi_idxs[i]+1).zfill(3)
        return hoi_ids, labels, hoi_label_vecs

    def get_faster_rcnn_prob_vecs(self,hoi_ids,human_probs,object_probs):
        num_hois = len(self.hoi_dict)
        num_cand = len(hoi_ids)
        human_prob_vecs = np.tile(np.expand_dims(human_probs,1),[1,num_hois])
        object_prob_vecs = np.zeros([num_cand,num_hois])
        for i,hoi_id in enumerate(hoi_ids):
            obj = self.hoi_dict[hoi_id]['object']
            obj_hoi_ids = self.obj_to_hoi_ids[obj]
            for obj_hoi_id in obj_hoi_ids:
                object_prob_vecs[i,int(obj_hoi_id)-1] = object_probs[i]
        return human_prob_vecs, object_prob_vecs

    def sample_cands(self,hoi_labels):
        num_cands = hoi_labels.shape[0]
        indices = np.arange(num_cands)
        tp_ids = indices[hoi_labels==1.0]
        fp_ids = indices[hoi_labels==0]
        num_tp = tp_ids.shape[0]
        num_fp = fp_ids.shape[0]
        if num_tp==0:
            num_fp_to_sample = self.const.fp_to_tp_ratio
        else:
            num_fp_to_sample = min(num_fp,self.const.fp_to_tp_ratio*num_tp)
        sampled_fp_ids = np.random.permutation(fp_ids)[:num_fp_to_sample]
        sampled_ids = np.concatenate((tp_ids,sampled_fp_ids),0)
        return sampled_ids

    def get_obj_one_hot(self,hoi_ids):
        num_cand = len(hoi_ids)
        obj_one_hot = np.zeros([num_cand,len(self.obj_to_id)])
        for i, hoi_id in enumerate(hoi_ids):
            obj_id = self.obj_to_id[self.hoi_dict[hoi_id]['object']]
            obj_idx = int(obj_id)-1
            obj_one_hot[i,obj_idx] = 1.0
        return obj_one_hot

    def get_verb_one_hot(self,hoi_ids):
        num_cand = len(hoi_ids)
        verb_one_hot = np.zeros([num_cand,len(self.verb_to_id)])
        for i, hoi_id in enumerate(hoi_ids):
            verb_id = self.verb_to_id[self.hoi_dict[hoi_id]['verb']]
            verb_idx = int(verb_id)-1
            verb_one_hot[i,verb_idx] = 1.0
        return verb_one_hot

    def get_prob_mask(self,hoi_idx):
        num_cand = len(hoi_idx)
        prob_mask = np.zeros([num_cand,len(self.hoi_dict)])
        prob_mask[np.arange(num_cand),hoi_idx] = 1.0
        return prob_mask

    def __getitem__(self,i):
        global_id = self.global_ids[i]

        start_end_ids = self.hoi_cands[global_id]['start_end_ids'][()]
        hoi_cands_ = self.hoi_cands[global_id]['boxes_scores_rpn_ids_hoi_idx'][()]
        hoi_ids_, hoi_labels_, hoi_label_vecs_ = self.get_labels(global_id)
        if self.const.box_feats_hdf5:
            box_feats_ =  self.box_feats[global_id][()]
        else:
            box_feats_ = None

        if self.const.balanced_sampling:
            cand_ids = self.sample_cands(hoi_labels_)
            hoi_cands = hoi_cands_[cand_ids]
            hoi_ids = np.array(hoi_ids_)[cand_ids].tolist()
            hoi_labels = hoi_labels_[cand_ids]
            hoi_label_vecs = hoi_label_vecs_[cand_ids]
            if box_feats_ is not None:
                box_feats = box_feats_[cand_ids]
            else:
                box_feats = None
        else:
            hoi_cands = hoi_cands_
            hoi_ids = hoi_ids_
            hoi_labels = hoi_labels_
            hoi_label_vecs = hoi_label_vecs_
            box_feats = box_feats_
        
        to_return = {
            'global_id': global_id,
            'human_box': hoi_cands[:,:4],
            'object_box': hoi_cands[:,4:8],
            'human_prob': hoi_cands[:,8],
            'object_prob': hoi_cands[:,9],
            'human_rpn_id': hoi_cands[:,10].astype(np.int),
            'object_rpn_id': hoi_cands[:,11].astype(np.int),
            'hoi_idx': hoi_cands[:,-1].astype(np.int),
            'hoi_id': hoi_ids,
            'hoi_label': hoi_labels,
            'hoi_label_vec': hoi_label_vecs,
            'box_feat': box_feats,
            'hoi_cands_': hoi_cands_,
            'start_end_ids_': start_end_ids.astype(np.int), # Corresponds to non sampled hoi_cands_ which is the same as hoi_cands when balanced sampling is not used
        }
        
        to_return['human_feat'] = np.take(
            self.faster_rcnn_feats[global_id],
            to_return['human_rpn_id'],
            axis=0)
        to_return['object_feat'] = np.take(
            self.faster_rcnn_feats[global_id],
            to_return['object_rpn_id'],
            axis=0)
        human_prob_vecs, object_prob_vecs = self.get_faster_rcnn_prob_vecs(
            to_return['hoi_id'], 
            to_return['human_prob'],
            to_return['object_prob'])
        to_return['human_prob_vec'] = human_prob_vecs
        to_return['object_prob_vec'] = object_prob_vecs
        to_return['object_one_hot'] = self.get_obj_one_hot(to_return['hoi_id'])
        to_return['verb_one_hot'] = self.get_verb_one_hot(to_return['hoi_id'])
        to_return['prob_mask'] = self.get_prob_mask(to_return['hoi_idx'])
        return to_return
        



    

