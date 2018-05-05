import os
import h5py
import time
import argparse
import numpy as np
from tqdm import tqdm

import utils.io as io
from exp.hoi_classifier.data.pose_features import PoseFeatures


def main(exp_const,data_const):
    hoi_cands = h5py.File(data_const.hoi_cand_hdf5,'r')
    human_cands_pose = h5py.File(data_const.human_cands_pose_hdf5,'r')

    human_pose_feats_hdf5 = os.path.join(
        exp_const.exp_dir,
        f'human_pose_feats_{exp_const.subset}.hdf5')
    human_pose_feats = h5py.File(human_pose_feats_hdf5,'w')

    anno_list = io.load_json_object(data_const.anno_list_json)
    anno_dict = {anno['global_id']:anno for anno in anno_list}

    pose_feat_computer = PoseFeatures(num_keypts=data_const.num_keypoints)
    for global_id in tqdm(hoi_cands.keys()):
        img_hoi_cands = hoi_cands[global_id]
        human_boxes = img_hoi_cands['boxes_scores_rpn_ids_hoi_idx'][:,:4] 
        object_boxes = img_hoi_cands['boxes_scores_rpn_ids_hoi_idx'][:,4:8]
        human_rpn_ids = img_hoi_cands['boxes_scores_rpn_ids_hoi_idx'][:,10]
        rpn_id_to_pose = pose_feat_computer.rpn_id_to_pose_h5py_to_npy(
            human_cands_pose[global_id])
        img_size = anno_dict[global_id]['image_size'][:2]
        imh, imw = [float(v) for v in img_size[:2]]
        im_wh = np.array([[imw,imh]],dtype=np.float32)
        num_cand = human_boxes.shape[0]
        im_wh = np.tile(im_wh,(num_cand,1))
        feats = pose_feat_computer.compute_pose_feats(
            human_boxes,
            object_boxes,
            human_rpn_ids,
            rpn_id_to_pose,
            im_wh)
        human_pose_feats.create_group(global_id)
        human_pose_feats[global_id].create_dataset(
            'absolute_pose',
            data=feats['absolute_pose'])
        human_pose_feats[global_id].create_dataset(
            'relative_pose',
            data=feats['relative_pose'])

    human_pose_feats.close()