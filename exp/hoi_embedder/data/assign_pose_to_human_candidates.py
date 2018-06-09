import os
import numpy as np
from tqdm import tqdm
import h5py

import utils.io as io
from utils.bbox_utils import compute_iou, compute_area


def get_pose_box(pose):
    valid_mask = pose[:,2] > 0  # consider points with non-zero confidence
    if not np.any(valid_mask):
        return np.zeros([4])
    keypoints = pose[valid_mask,:2]
    x1,y1 = np.amin(keypoints,0)
    x2,y2 = np.amax(keypoints,0)
    box = np.array([x1,y1,x2,y2])
    return box


def count_keypoints_in_box(keypoints,box):
    x1,y1,x2,y2 = box
    

def assign_pose(human_box,pose_boxes,pose_keypoints,num_keypoints):
    max_idx = -1
    max_frac_inside = 0.7
    found_match = False
    for i, pose_box in enumerate(pose_boxes):
        iou,intersection,union = compute_iou(human_box,pose_box,True)
        pose_area = compute_area(pose_box)
        frac_inside = intersection / pose_area
        if frac_inside > max_frac_inside:
            max_frac_inside = frac_inside
            max_idx = i
            found_match = True

    if max_idx==-1:
        keypoints = np.zeros([num_keypoints,3])
    else:
        keypoints = pose_keypoints[max_idx]

    return keypoints, found_match


def main(exp_const,data_const):
    print(f'Reading split_ids.json ...')
    split_ids = io.load_json_object(data_const.split_ids_json)

    print(f'Creating a human_candidates_pose_{exp_const.subset}.hdf5 file ...')
    human_cand_pose_hdf5 = os.path.join(
        exp_const.exp_dir,f'human_candidates_pose_{exp_const.subset}.hdf5')
    human_cand_pose = h5py.File(human_cand_pose_hdf5,'w')

    print(f'Reading hoi_candidates_{exp_const.subset}.hdf5 file ...')
    hoi_cand = h5py.File(data_const.hoi_cand_hdf5,'r')
    count_assignments = 0
    for global_id in tqdm(split_ids[exp_const.subset]):
        boxes_scores_rpn_ids_hoi_idx = \
            hoi_cand[global_id]['boxes_scores_rpn_ids_hoi_idx']
        human_boxes = boxes_scores_rpn_ids_hoi_idx[:,:4]
        human_rpn_ids = boxes_scores_rpn_ids_hoi_idx[:,10]
        num_cand = human_boxes.shape[0]

        if 'test' in global_id:
            pose_prefix = 'test2015/'
        else:
            pose_prefix = 'train2015/'
        pose_json = os.path.join(
            data_const.human_pose_dir,
            f'{pose_prefix}{global_id}_keypoints.json')
        
        human_poses = [
            np.reshape(np.array(pose['pose_keypoints_2d']),(-1,3)) 
            for pose in io.load_json_object(pose_json)['people']]
        pose_boxes = [get_pose_box(pose) for pose in human_poses]

        rpn_id_to_pose = {}
        for i in range(num_cand):
            rpn_id = str(int(human_rpn_ids[i]))
            if rpn_id in rpn_id_to_pose:
                continue
            else:
                rpn_id_to_pose[rpn_id], match_status = assign_pose(
                    human_boxes[i],
                    pose_boxes,
                    human_poses,
                    data_const.num_keypoints)
                if match_status:
                    count_assignments += 1

        human_cand_pose.create_group(global_id)
        for rpn_id, pose in rpn_id_to_pose.items():
            human_cand_pose[global_id].create_dataset(rpn_id,data=pose)

    print(f'Number of assignments: {count_assignments}')

    human_cand_pose.close()
    hoi_cand.close()