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
        self.verb_vec_dim = 300

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
        self.verb_vec_xform = nn.Sequential(
            nn.Linear(self.const.verb_vec_dim,self.const.verb_vec_dim),
            nn.ReLU(),
            nn.Linear(
                self.const.verb_vec_dim,
                self.mlp_penultimate_feat_dim(self.mlp)))

    def transform_feat(self,feat):
        log_feat = torch.log(torch.abs(feat)+1e-6)
        transformed_feat = torch.cat((feat,log_feat),1) 
        return transformed_feat

    def mlp_penultimate_feat_dim(self,mlp):
        last_linear_layer = [layer for layer in mlp.layers][-1][0]
        assert_str = \
            'Not a linear layer. ' + \
            'Please check mlp_penultimate_feat_dim implementation'
        assert(isinstance(last_linear_layer,nn.Linear)), assert_str
        return last_linear_layer.weight.size(1)

    def forward_mlp_all_but_last(self,feats,mlp):
        all_but_last_layer = [layer for layer in mlp.layers][:-1]
        x = feats
        for layer in all_but_last_layer:
            x = layer(x)
        return x

    def forward(self,feats,verb_vecs):
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
        in_feat = torch.cat((transformed_box_feats,feats['object_one_hot']),1)
        factor_feats = self.forward_mlp_all_but_last(in_feat,self.mlp)
        xformed_verb_vec = self.verb_vec_xform(verb_vecs)
        factor_scores = torch.mm(factor_feats,torch.transpose(xformed_verb_vec,0,1))
        return factor_scores