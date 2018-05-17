import os
import h5py
import numpy as np
from tests.tester import *

def test_oracle():
    oracle_labels_h5py = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_candidates/' + \
        'hoi_candidate_oracle_labels_test.hdf5')
    oracle_labels = h5py.File(oracle_labels_h5py,'r')

    gt_labels_h5py = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_candidates/' + \
        'hoi_candidate_labels_test.hdf5')
    gt_labels = h5py.File(gt_labels_h5py,'r')

    oracle_pred_h5py = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_classifier/' + \
        'factors_rcnn_det_prob_appearance_boxes_and_object_label_human_pose/' + \
        'pred_hoi_dets_oracle_human_True_object_True_verb_True_test_25000.hdf5')
    oracle_preds = h5py.File(oracle_pred_h5py,'r')
    global_ids = [gid for gid in oracle_preds.keys()]
    for global_id in global_ids:
        scores = oracle_preds[global_id]['human_obj_boxes_scores'][:,-1]
        labels = oracle_labels[global_id][()]
        gt = gt_labels[global_id][()]
        if np.sum(scores) > np.sum(gt):
            print(
                np.sum(gt*scores),
                np.sum(gt),
                np.sum(scores))
            print(np.sum(labels,0))
            import pdb; pdb.set_trace()

if __name__=='__main__':
    list_tests(globals())