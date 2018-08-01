import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class VerbGivenHumanPoseConstants(io.JsonSerializableClass):
    def __init__(self):
        super(VerbGivenHumanPoseConstants,self).__init__()
        self.pose_feat_size = 54+90
        self.use_absolute_pose = True
        self.use_relative_pose = True
        self.num_objects = 80
        self.num_verbs = 117
        self.use_object_label = True

    @property
    def mlp_const(self):
        in_dim = 2*self.pose_feat_size + self.num_objects
        layer_units = [in_dim]*2
        factor_const = {
            'in_dim': in_dim,
            'out_dim': self.num_verbs,
            'out_activation': 'Identity',
            'layer_units': layer_units,
            'activation': 'ReLU',
            'use_out_bn': False,
            'use_bn': True
        }
        return factor_const
    
    
class VerbGivenHumanPose(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(VerbGivenHumanPose,self).__init__()
        self.const = copy.deepcopy(const)
        self.mlp = pytorch_layers.create_mlp(self.const.mlp_const)

    def transform_feat(self,feat):
        log_feat = torch.log(torch.abs(feat)+1e-6)
        transformed_feat = torch.cat((feat,log_feat),1) 
        return transformed_feat

    def forward(self,feats):
        if not self.const.use_absolute_pose:
            absolute_pose = 0*feats['absolute_pose']
        else:
            absolute_pose = feats['absolute_pose']
        if not self.const.use_relative_pose:
            relative_pose = 0*feats['relative_pose']
        else:
            relative_pose = feats['relative_pose']
        pose_feats = torch.cat((absolute_pose,relative_pose),1)
        transformed_box_feats = self.transform_feat(pose_feats)
        if self.const.use_object_label is True:
            object_label = feats['object_one_hot']
        else:
            object_label = 0*feats['object_one_hot']
        in_feat = torch.cat((transformed_box_feats,object_label),1)
        factor_scores = self.mlp(in_feat)
        return factor_scores