import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class RelationClassifierConstants(io.JsonSerializableClass):
    def __init__(self):
        super(RelationClassifierConstants,self).__init__()
        self.faster_rcnn_feat_size = 2048
        self.num_relation_classes = 117

    @property
    def faster_rcnn_human_feature_factor_const(self):
        factor_const = {
            'in_dim': self.faster_rcnn_feat_size,
            'out_dim': self.num_relation_classes,
            'out_activation': 'Identity',
            'layer_units': [self.faster_rcnn_feat_size],
            'activation': 'ReLU',
            'use_out_bn': False,
            'use_bn': True
        }
        return factor_const

    @property
    def faster_rcnn_object_feature_factor_const(self):
        factor_const = {
            'in_dim': self.faster_rcnn_feat_size,
            'out_dim': self.num_relation_classes,
            'out_activation': 'Identity',
            'layer_units': [self.faster_rcnn_feat_size],
            'activation': 'ReLU',
            'use_out_bn': False,
            'use_bn': True
        }
        return factor_const


class RelationClassifier(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(RelationClassifier,self).__init__()
        self.const = copy.deepcopy(const)
        self.faster_rcnn_human_feature_factor = pytorch_layers.create_mlp(
            self.const.faster_rcnn_human_feature_factor_const)
        self.faster_rcnn_object_feature_factor = pytorch_layers.create_mlp(
            self.const.faster_rcnn_object_feature_factor_const)
        self.sigmoid = pytorch_layers.get_activation('Sigmoid')
        
    def forward(self,feats):
        B = feats['human_rcnn'].size(0)
        faster_rcnn_feature_factor_scores = \
            self.faster_rcnn_human_feature_factor(feats['human_rcnn']) + \
            self.faster_rcnn_object_feature_factor(feats['object_rcnn'])
        relation_prob = self.sigmoid(faster_rcnn_feature_factor_scores)
        return relation_prob

    
class BoxAwareRelationClassifierConstants(RelationClassifierConstants):
    def __init__(self):
        super(BoxAwareRelationClassifierConstants,self).__init__()
        self.box_feat_size = 12

    @property
    def box_feature_factor_const(self):
        factor_const = {
            'in_dim': self.box_feat_size,
            'out_dim': self.num_relation_classes,
            'out_activation': 'Identity',
            'layer_units': [self.box_feat_size],
            'activation': 'ReLU',
            'use_out_bn': False,
            'use_bn': True
        }
        return factor_const
    
class BoxAwareRelationClassifier(RelationClassifier):
    def __init__(self,const):
        super(BoxAwareRelationClassifier,self).__init__(const)
        self.box_feature_factor = pytorch_layers.create_mlp(
            self.const.box_feature_factor_const)

    def forward(self,feats):
        B = feats['human_rcnn'].size(0)
        faster_rcnn_feature_factor_scores = \
            self.faster_rcnn_human_feature_factor(feats['human_rcnn']) + \
            self.faster_rcnn_object_feature_factor(feats['object_rcnn']) #+ \
        box_feature_factor_scores = self.box_feature_factor(feats['box'])
        relation_prob = self.sigmoid(faster_rcnn_feature_factor_scores) * \
            self.sigmoid(box_feature_factor_scores)
        return relation_prob