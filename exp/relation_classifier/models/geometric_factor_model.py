import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class GeometricFactorConstants(io.JsonSerializableClass):
    def __init__(self):
        super(GeometricFactorConstants,self).__init__()
        self.box_feat_size = 40 + 40*40 + 80
        self.out_dim = 117
        self.non_linear = False

    @property
    def box_feature_factor_const(self):
        if self.non_linear:
            layer_units = [int(self.box_feat_size/2),int(self.box_feat_size/4),int(self.box_feat_size/8)]
        else:
            layer_units = []

        factor_const = {
            'in_dim': self.box_feat_size,
            'out_dim': self.out_dim,
            'out_activation': 'Identity',
            'layer_units': layer_units,
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

    def outer_product(self,feat):
        B,D = feat.size()
        outer_prod_feat = feat.view(B,1,D) * feat.view(B,D,1)
        outer_prod_feat = outer_prod_feat.view(B,-1)
        return outer_prod_feat

    def transform_feat(self,feat):
        log_feat = torch.log(feat+1e-6)
        linear_log_feat = torch.cat((feat,log_feat),1)
        outer_prod_feat = self.outer_product(linear_log_feat)
        transformed_feat = torch.cat((
            linear_log_feat,
            outer_prod_feat),1)
        return transformed_feat

    def forward(self,feats):
        in_feat = torch.cat((
            self.transform_feat(feats['box']),
            feats['object_one_hot']),1)
        #in_feat = feats['box']
        box_feature_factor_scores = self.box_feature_factor(in_feat)
        return box_feature_factor_scores


class GeometricFactorPairwiseConstants(GeometricFactorConstants):
    def __init__(self):
        super(GeometricFactorPairwiseConstants,self).__init__()

    @property
    def pairwise_linear_const(self):
        const = {
            'in_dim': self.box_feat_size**2,
            'out_dim': self.box_feat_size*2,
        }
        return const

    @property
    def agg_linear_const(self):
        const = {
            'in_dim': self.pairwise_linear_const['out_dim']+self.box_feat_size,
            'out_dim': self.out_dim,
        }
        return const
    

class GeometricFactorPairwise(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(GeometricFactorPairwise,self).__init__()
        self.const = copy.deepcopy(const)
        
        #self.box_feat_bn = nn.BatchNorm2d(self.const.box_feat_size)

        pairwise_linear_const = self.const.pairwise_linear_const
        self.pairwise_linear = nn.Linear(
            pairwise_linear_const['in_dim'],
            pairwise_linear_const['out_dim'])
        
        #self.pairwise_bn = nn.BatchNorm2d(pairwise_linear_const['out_dim'])
        
        agg_linear_const = self.const.agg_linear_const
        self.agg_linear = nn.Linear(
            agg_linear_const['in_dim'],
            agg_linear_const['out_dim'])

    def get_pairwise_feat(self,x):
        """
        x is B x F
        """
        B,F = x.size()
        x1 = x.view(B,F,1).repeat(1,1,F)
        x2 = x.view(B,1,F).repeat(1,F,1)
        y = x2-x1
        y = y.view(B,-1) # B x F*F
        return y
    
    def forward(self,feats):
        pairwise_feats = self.get_pairwise_feat(feats['box'])
        pairwise_feats = self.pairwise_linear(pairwise_feats)
        #pairwise_feats = self.pairwise_bn(pairwise_feats)
        box_feat = feats['box'] #self.box_feat_bn(feats['box'])
        agg_feats = torch.cat((box_feat,pairwise_feats*pairwise_feats),1)
        box_feature_factor_scores = self.agg_linear(agg_feats)
        return box_feature_factor_scores