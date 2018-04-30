import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm

import utils.io as io
from exp.hoi_classifier.data.box_features import BoxFeatures


def compute_box_feats(human_boxes,object_boxes,img_size):
    feat_extractor = BoxFeatures()
    num_cand = human_boxes.shape[0]
    imh, imw = [float(v) for v in img_size[:2]]
    im_wh = np.array([[imw,imh]],dtype=np.float32)
    im_wh = np.tile(im_wh,(num_cand,1))
    feats = feat_extractor.compute_features(
        human_boxes,
        object_boxes,
        im_wh)
    return feats


def main(exp_const,data_const):
    hoi_cands = h5py.File(data_const.hoi_cand_hdf5,'r')

    box_feats_hdf5 = os.path.join(
        exp_const.exp_dir,
        f'hoi_candidates_box_feats_{exp_const.subset}.hdf5')
    box_feats = h5py.File(box_feats_hdf5,'w')

    anno_list = io.load_json_object(data_const.anno_list_json)
    anno_dict = {anno['global_id']:anno for anno in anno_list}

    for global_id in tqdm(hoi_cands.keys()):
        img_hoi_cands = hoi_cands[global_id]
        human_boxes = img_hoi_cands['boxes_scores_rpn_ids_hoi_idx'][:,:4] 
        object_boxes = img_hoi_cands['boxes_scores_rpn_ids_hoi_idx'][:,4:8]
        img_size = anno_dict[global_id]['image_size'][:2]
        feats = compute_box_feats(human_boxes,object_boxes,img_size)
        box_feats.create_dataset(global_id,data=feats)

    box_feats.close()