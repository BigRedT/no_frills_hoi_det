import h5py
import copy
import time
import numpy as np

import utils.bbox_utils as bbox_utils


class PoseFeatures():
    def __init__(self,num_keypts=18):
        self.num_keypts = num_keypts

    def rpn_id_to_pose_h5py_to_npy(self,rpn_id_to_pose_h5py):
        rpn_id_to_pose_npy = {}
        for rpn_id in rpn_id_to_pose_h5py.keys():
            rpn_id_to_pose_npy[rpn_id] = rpn_id_to_pose_h5py[rpn_id][()]
        return rpn_id_to_pose_npy

    def get_keypoints(self,rpn_ids,rpn_id_to_pose):
        num_cand = rpn_ids.shape[0]
        keypts = np.zeros([num_cand,self.num_keypts,3])
        for i in range(num_cand):
            rpn_id = str(int(rpn_ids[i]))
            keypts_ = rpn_id_to_pose[rpn_id]
            keypts[i] = keypts_
        return keypts

    def compute_bbox_wh(self,bbox):
        num_boxes = bbox.shape[0]
        wh = np.zeros([num_boxes,2])
        wh[:,0] = 0.5*(bbox[:,2]-bbox[:,0])
        wh[:,1] = 0.5*(bbox[:,3]-bbox[:,1])
        return wh
    
    def encode_pose(self,keypts,human_box):
        wh = self.compute_bbox_wh(human_box) # Bx2
        wh = np.tile(wh[:,np.newaxis,:],(1,self.num_keypts,1)) # Bx18x2
        xy = np.tile(human_box[:,np.newaxis,:2],(1,self.num_keypts,1))  # Bx18x2
        pose = keypts[:,:,:]    # Bx18x3
        pose[:,:,:2] = (pose[:,:,:2] - xy)/(wh+1e-6)
        return pose

    def encode_relative_pose(self,keypts,object_box,im_wh):
        keypts[:,:,:2] = keypts[:,:,:2] / im_wh
        x1y1 = object_box[:,:2]
        x1y1 = np.tile(x1y1[:,np.newaxis,:],(1,self.num_keypts,1))
        x1y1 = x1y1 / im_wh
        x2y2 = object_box[:,2:4] 
        x2y2 = np.tile(x2y2[:,np.newaxis,:],(1,self.num_keypts,1))
        x2y2 = x2y2 / im_wh
        x1y1_wrt_keypts = x1y1 - keypts[:,:,:2] # Bx18x2
        x2y2_wrt_keypts = x2y2 - keypts[:,:,:2] # Bx18x2
        return x1y1_wrt_keypts, x2y2_wrt_keypts

    def compute_pose_feats(
            self,
            human_bbox,
            object_bbox,
            rpn_ids,
            rpn_id_to_pose,
            im_wh):
        B = human_bbox.shape[0]
        im_wh = np.tile(im_wh[:,np.newaxis,:],(1,self.num_keypts,1))
        keypts = self.get_keypoints(rpn_ids,rpn_id_to_pose)
        absolute_pose = self.encode_pose(keypts,human_bbox) # Bx18x3
        keypts_conf = absolute_pose[:,:,2][:,:,np.newaxis] # Bx18x1
        start = time.time()
        absolute_pose = np.reshape(absolute_pose,(B,-1)) # Bx54
        x1y1_wrt_keypts, x2y2_wrt_keypts = self.encode_relative_pose(
            keypts,
            object_bbox,
            im_wh)
        relative_pose = np.concatenate((
            x1y1_wrt_keypts,
            x2y2_wrt_keypts,
            keypts_conf),2)
        relative_pose = np.reshape(relative_pose,(B,-1))
        feats = {
            'absolute_pose': absolute_pose,   # Bx54
            'relative_pose': relative_pose, # Bx90 (18*2 + 18*2 + 18)
        }
        
        return feats

    # def compute_pose_feats_gpu(
    #         self,
    #         human_bbox,
    #         object_bbox,
    #         rpn_ids,
    #         rpn_id_to_pose,
    #         im_wh):
    #     B = human_bbox.shape[0]


        