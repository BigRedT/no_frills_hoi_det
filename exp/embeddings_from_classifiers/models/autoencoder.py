import torch
import copy
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class AutoencoderConstants(io.JsonSerializableClass):
    def __init__(self):
        self.input_dim = 300
        self.code_dim = 5
        self.num_hidden_layers = 2
        self.drop_prob = 0.2
    
    @property
    def encoder_layer_units(self):
        units = []
        for i in range(self.num_hidden_layers):
            units.append(max(
                self.input_dim//(2**(i+1)),
                self.code_dim))
        return units

    @property
    def decoder_layer_units(self):
        units = self.encoder_layer_units[::-1]
        return units

    @property
    def encoder_mlp_const(self):
        mlp_const = {
            'in_dim': self.input_dim,
            'out_dim': self.code_dim,
            'out_activation': 'Identity',
            'layer_units': self.encoder_layer_units,
            'activation': 'Tanh',
            'use_out_bn': False,
            'use_bn': False,
            'drop_prob': self.drop_prob,
        }
        return mlp_const

    @property
    def decoder_mlp_const(self):
        mlp_const = {
            'in_dim': self.code_dim,
            'out_dim': self.input_dim,
            'out_activation': 'Identity',
            'layer_units': self.decoder_layer_units,
            'activation': 'Tanh',
            'use_out_bn': False,
            'use_bn': False,
            'drop_prob': self.drop_prob,
        }
        return mlp_const


class Autoencoder(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(Autoencoder,self).__init__()
        self.const = copy.deepcopy(const)
        self.encoder = pytorch_layers.create_mlp(self.const.encoder_mlp_const)
        self.decoder = pytorch_layers.create_mlp(self.const.decoder_mlp_const) 

    def forward(self,x):
        code = self.encoder(x)
        output = self.decoder(code)
        return code, output

    def loss_fn(self,x,y,loss_type='L1'):
        x = x.detach()
        if loss_type=='L2':
            loss = torch.mean((x-y)*(x-y))
        elif loss_type=='L1':
            loss = torch.mean(torch.abs(x-y))
        else:
            assert(False),'Only L1 and L2 losses supported'
        return loss
        

    
