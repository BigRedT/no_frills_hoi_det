import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class VerbGivenObjectAppearanceConstants(io.JsonSerializableClass):
    def __init__(self):
        super(VerbGivenObjectAppearanceConstants,self).__init__()
        self.appearance_feat_size = 2048
        self.num_verbs = 117
        self.verb_vec_dim = 300

    @property
    def mlp_const(self):
        factor_const = {
            'in_dim': self.appearance_feat_size,
            'out_dim': self.num_verbs,
            'out_activation': 'Identity',
            'layer_units': [self.appearance_feat_size],
            'activation': 'ReLU',
            'use_out_bn': False,
            'use_bn': True
        }
        return factor_const


class VerbGivenObjectAppearance(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(VerbGivenObjectAppearance,self).__init__()
        self.const = copy.deepcopy(const)
        self.mlp = pytorch_layers.create_mlp(self.const.mlp_const)
        self.verb_vec_xform = nn.Sequential(
            nn.Linear(self.const.verb_vec_dim,self.const.verb_vec_dim),
            nn.ReLU(),
            nn.Linear(
                self.const.verb_vec_dim,
                self.mlp_penultimate_feat_dim(self.mlp)))

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
        factor_feats = self.forward_mlp_all_but_last(
            feats['object_rcnn'],
            self.mlp) 
        xformed_verb_vec = self.verb_vec_xform(verb_vecs)
        factor_scores = torch.mm(factor_feats,torch.transpose(xformed_verb_vec,0,1))
        return factor_scores

    
