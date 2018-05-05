import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from tests.tester import *
from exp.hoi_classifier.data.features_dataset import \
    FeatureConstants, Features


def test_features():
    data_const = FeatureConstants()
    hoi_cand_dir = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_candidates')
    data_const.hoi_cands_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidates_test.hdf5')
    data_const.box_feats_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidates_box_feats_test.hdf5')
    data_const.hoi_cand_labels_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidate_labels_test.hdf5')
    # data_const.human_cand_pose_hdf5 = os.path.join(
    #     hoi_cand_dir,
    #     'human_candidates_pose_test.hdf5')
    data_const.human_cand_pose_hdf5 = None
    data_const.human_pose_feats_hdf5 = os.path.join(
        hoi_cand_dir,
        'human_pose_feats_test.hdf5')
    data_const.faster_rcnn_feats_hdf5 = os.path.join(
        data_const.proc_dir,
        'faster_rcnn_fc7.hdf5')
    data_const.fp_to_tp_ratio = 1000
    data_const.subset = 'test' # to be set in the train_balanced.py script

    dataset = Features(data_const)
    sampler = RandomSampler(dataset)
    for i, sample_id in enumerate(tqdm(sampler)):
        data = dataset[sample_id]
        #import pdb; pdb.set_trace()

    

if __name__=='__main__':
    list_tests(globals())