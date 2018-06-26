import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import copy

from utils.io import WritableToFile, JsonSerializableClass


class Sim2GloveConst(JsonSerializableClass):
    def __init__(self):
        super(Sim2GloveConst,self).__init__()
        self.glove_verb_vecs_npy = None
        self.glove_verb_vec_dim = 300
        self.verb_vec_dim = 300
        self.projection = 'mlp' # mlp 
        self.loss = 'L2' # L1


class Sim2Glove(nn.Module,WritableToFile):
    def __init__(self,const):
        super(Sim2Glove,self).__init__()
        self.const = copy.deepcopy(const)
        self.glove_verb_vecs = self.load_glove_verb_vecs()
        self.verb_vec_xform = self.get_verb_vec_xform()
        self.loss_fn = self.get_loss_fn()
    
    def get_verb_vec_xform(self):
        if self.const.projection=='affine':
            verb_vec_xform = nn.Sequential(
                nn.Linear(
                    self.const.verb_vec_dim,
                    self.const.glove_verb_vec_dim))
        elif self.const.projection=='mlp':
            verb_vec_xform = nn.Sequential(
                nn.Linear(
                    self.const.verb_vec_dim,
                    self.const.verb_vec_dim),
                nn.ReLU(),
                nn.Linear(
                    self.const.verb_vec_dim,
                    self.const.glove_verb_vec_dim))
        else:
            assert(False),'Projection not supported'
        
        return verb_vec_xform

    def get_loss_fn(self):
        if self.const.loss=='L1':
            loss_fn = nn.L1Loss()
        elif self.const.loss=='L2':
            loss_fn = nn.MSELoss()
        else:
            assert(False), 'Loss not supported'

        return loss_fn

    def load_glove_verb_vecs(self):
        glove_verb_vecs = np.load(self.const.glove_verb_vecs_npy)
        glove_verb_vecs = torch.FloatTensor(glove_verb_vecs).cuda()
        return Variable(glove_verb_vecs,requires_grad=False)

    def forward(self,verb_vec):
        xformed_verb_vec = self.verb_vec_xform(verb_vec)
        loss = self.loss_fn(xformed_verb_vec,self.glove_verb_vecs)
        return loss