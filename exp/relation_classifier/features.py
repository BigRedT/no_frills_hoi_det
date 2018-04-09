import h5py
import copy
import itertools
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
        self.sample_ids = self.create_list_of_sample_ids()

    def load_hdf5_file(self,hdf5_filename,mode='r'):
        return h5py.File(hdf5_filename,mode)

    def load_subset_ids(self):
        split_ids = io.load_json_object(self.const.split_ids_json)
        return sorted(split_ids[subset])

    def create_list_of_sample_ids(self):
        sample_ids = []
        for global_id in self.global_ids:
            num_cands = self.hoi_cand_labels[global_id].shape[0]
            sampled_ids += list(itertools.product([global_id],range(num_cands)))
        return sample_ids

    def __len__():
        return len(self.sample_ids)

    def __getitem__(self,i):
        global_id, cand_id = self.sample_id[i]
        hoi_cand = self.hoi_cands[global_id][cand_id]
        



    

