import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class GeometricFactorConstants(io.JsonSerializableClass):
    def __init__(self):
        super(GeometricFactorConstants,self).__init__()
        self.box_feat_size = 24
        self.out_dim = 117

    @property
    def box_feature_factor_const(self):
        factor_const = {
            'in_dim': self.box_feat_size,
            'out_dim': self.out_dim,
            'out_activation': 'Identity',
            'layer_units': [],
            'activation': 'ReLU',
            'use_out_bn': False,
            'use_bn': True
        }
        return factor_const
    
    
class GeometricFactor(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(GeometricFactor,self).__init__()
        self.const = copy.deepcopy(const)
        self.box_feature_factor = pytorch_layers.create_mlp(
            self.const.box_feature_factor_const)

    def forward(self,feats):
        box_feature_factor_scores = self.box_feature_factor(feats['box'])
        return box_feature_factor_scores