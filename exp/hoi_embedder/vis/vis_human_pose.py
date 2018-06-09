import os
import h5py
import numpy as np
import skimage.io as skio
from tqdm import tqdm

import utils.io as io
import utils.bbox_utils as bbox_utils
from utils.constants import ExpConstants
from data.hico.hico_constants import HicoConstants


def main(exp_const,data_const):
    io.mkdir_if_not_exists(exp_const.exp_dir)
    
    print('Reading anno_list.json ...')
    anno_list  = io.load_json_object(data_const.anno_list_json)
    anno_dict = {anno['global_id']:anno for anno in anno_list}

    print('Reading box and pose features ...')
    human_pose_feats = h5py.File(data_const.human_pose_feats_h5py,'r')
    hoi_cand = h5py.File(data_const.hoi_cand_h5py,'r')

    for count,global_id in enumerate(tqdm(human_pose_feats.keys())):
        if count>=exp_const.max_count:
            break
        human_boxes = hoi_cand[global_id]['boxes_scores_rpn_ids_hoi_idx'][:,:4]
        human_rpn_ids = hoi_cand[global_id]['boxes_scores_rpn_ids_hoi_idx'][:,10]
        B = human_boxes.shape[0]
        absolute_pose = human_pose_feats[global_id]['absolute_pose'][()]
        absolute_pose = np.reshape(absolute_pose,(B,data_const.num_keypts,3))
        x1y1 = human_boxes[:,:2]    # Bx2
        wh = 0*x1y1 # Bx2
        wh[:,0] = (human_boxes[:,2] - human_boxes[:,0])
        wh[:,1] = (human_boxes[:,3] - human_boxes[:,1])
        x1y1 = np.tile(x1y1[:,np.newaxis,:],(1,data_const.num_keypts,1)) # Bx18x2
        wh = np.tile(wh[:,np.newaxis,:],(1,data_const.num_keypts,1))    # Bx18x2
        keypts = 0*absolute_pose
        keypts[:,:,:2] = absolute_pose[:,:,:2]*wh + x1y1
        keypts[:,:,2] = absolute_pose[:,:,2]
        img_path = os.path.join(
            data_const.images_dir,
            anno_dict[global_id]['image_path_postfix'])
        img = skio.imread(img_path)
        if len(img.shape)==2:
            img = np.tile(img[:,:,np.newaxis],(1,1,3))

        seen_rpn_ids = set()
        for i in range(B):
            rpn_id = human_rpn_ids[i]
            if rpn_id in seen_rpn_ids:
                continue
            else:
                seen_rpn_ids.add(rpn_id)
        
            img = bbox_utils.vis_human_keypts(img,keypts[i],modify=True)

            img_out_path = os.path.join(
                exp_const.exp_dir,
                f'{global_id}.png')
            skio.imsave(img_out_path,img)


if __name__=='__main__':
    exp_const = ExpConstants(exp_name='vis_human_pose')
    exp_const.max_count = 100

    data_const = HicoConstants()
    hoi_cand_dir = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_candidates')
    data_const.human_pose_feats_h5py = os.path.join(
        hoi_cand_dir,
        'human_pose_feats_test.hdf5')
    data_const.hoi_cand_h5py = os.path.join(
        hoi_cand_dir,
        'hoi_candidates_test.hdf5')
    data_const.num_keypts = 18

    main(exp_const,data_const)