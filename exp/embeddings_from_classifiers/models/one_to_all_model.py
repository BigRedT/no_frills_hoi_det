import torch
import copy
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import utils.io as io
import utils.pytorch_layers as pytorch_layers

class OneToAllConstants(io.JsonSerializableClass):
    def __init__(self):
        self.feat_dims = {
            'verb_given_human_app': 2048,
            'verb_given_object_app': 2048,
            'verb_given_boxes_and_object_label': 122,
            'verb_given_human_pose': 368,
            'word_vector': 300,
        }
        self.verb_vec_dim = 300
        self.num_verbs = 117
        self.use_coupling_variable = True

    def one_to_all_mlp_const(self,feat_dim):
        in_dim = self.verb_vec_dim
        layer_units = [in_dim]*0
        mlp_const = {
            'in_dim': in_dim,
            'out_dim': feat_dim,
            'out_activation': 'Identity',
            'layer_units': layer_units,
            'activation': 'ReLU',
            'use_out_bn': False,
            'use_bn': True
        }
        return mlp_const

    def all_to_one_mlp_const(self,feat_dim):
        in_dim = feat_dim
        layer_units = [in_dim]*0
        mlp_const = {
            'in_dim': in_dim,
            'out_dim': self.verb_vec_dim,
            'out_activation': 'Identity',
            'layer_units': layer_units,
            'activation': 'ReLU',
            'use_out_bn': False,
            'use_bn': True
        }
        return mlp_const


class OneToAll(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(OneToAll,self).__init__()
        self.const = copy.deepcopy(const)
        self.one_to_all_mlps = {}
        for factor, feat_dim in self.const.feat_dims.items():
            mlp_const = self.const.one_to_all_mlp_const(feat_dim)
            self.one_to_all_mlps[factor] = pytorch_layers.create_mlp(mlp_const)
            self.add_module(f'one_to_all_{factor}',self.one_to_all_mlps[factor])
        
        self.all_to_one_mlps = {}
        for factor, feat_dim in self.const.feat_dims.items():
            if factor!='word_vector':
                continue
            mlp_const = self.const.all_to_one_mlp_const(feat_dim)
            self.all_to_one_mlps[factor] = pytorch_layers.create_mlp(mlp_const)
            self.add_module(f'all_to_one_{factor}',self.all_to_one_mlps[factor])

        self.verb_vecs = self.create_verb_vecs(
            self.const.num_verbs,
            self.const.verb_vec_dim)

    def create_verb_vecs(
            self,
            num_verbs,
            verb_vec_dim,
            init='random',
            fine_tune=True):
        if init=='random':
            std = np.sqrt(1/verb_vec_dim)
            verb_vecs = np.random.normal(scale=std,size=(num_verbs,verb_vec_dim))
        else:
            assert(False),'only random initialization supported'
        verb_vecs = torch.FloatTensor(verb_vecs)
        return nn.Parameter(data=verb_vecs,requires_grad=fine_tune)

    def forward(self,feats):
        pred_verb_vecs = {}
        for factor, mlp in self.all_to_one_mlps.items():
            pred_verb_vecs[factor] = mlp(feats[factor])

        if self.const.use_coupling_variable is True: 
            coupling_variable = self.verb_vecs
        else:
            coupling_variable = pred_verb_vecs['word_vector']
    
        pred_feats = {}
        for factor, mlp in self.one_to_all_mlps.items():
            pred_feats[factor] = mlp(coupling_variable)

        return pred_feats, pred_verb_vecs 

    def l1_loss(self,x,y):
        return torch.mean(torch.abs(x-y))
        
    def l2_loss(self,x,y):
        return torch.mean((x-y)*(x-y))

    def compute_one_to_all_losses(self,pred_feats,feats,verb_ids=None):
        if verb_ids is not None:
            feats = {factor: feats[factor][verb_ids] for factor in feats.keys()}
            pred_feats = {factor: pred_feats[factor][verb_ids] \
                for factor in pred_feats.keys()}

        losses = {}
        total_loss = 0
        for factor, pred_feat in pred_feats.items():
            feat = feats[factor]
            losses[factor] = self.l2_loss(pred_feat,feat)
            total_loss += losses[factor]
        losses['total'] = total_loss
        return losses

    def compute_all_to_one_losses(self,pred_verb_vecs,verb_vecs,verb_ids=None):
        if verb_ids is not None:
            pred_verb_vecs = {factor: pred_verb_vecs[factor][verb_ids] \
                for factor in pred_verb_vecs.keys()}
            verb_vecs = verb_vecs[verb_ids]

        losses = {}
        total_loss = 0
        for factor, pred_verb_vec in pred_verb_vecs.items():
            losses[factor] = self.l2_loss(pred_verb_vec,verb_vecs)
            total_loss += losses[factor]
        losses['total'] = total_loss
        return losses

    def compute_loss(
            self,
            pred_feats,
            feats,
            pred_verb_vecs,
            verb_vecs,
            verb_ids=None):
        one_to_all_losses = \
            self.compute_one_to_all_losses(pred_feats,feats,verb_ids)
        all_to_one_losses = \
            self.compute_all_to_one_losses(pred_verb_vecs,verb_vecs,verb_ids)
        
        return one_to_all_losses, all_to_one_losses