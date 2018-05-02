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
        
    def forward(self,feats):
        factor_scores = self.mlp(feats['object_rcnn']) 
        return factor_scores

    
