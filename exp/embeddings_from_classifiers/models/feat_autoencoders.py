import torch
import copy
import torch.nn as nn

import utils.io as io
from exp.embeddings_from_classifiers.models.autoencoder import \
    AutoencoderConstants, Autoencoder

class FeatAutoencodersConstants(io.JsonSerializableClass):
    def __init__(self):
        self.input_dims = {
            'verb_given_human_app': 2048,
            'verb_given_object_app': 2048,
            'verb_given_boxes_and_object_label': 122,
            'verb_given_human_pose': 368,
            'word_vector': 300,
        }
        
        self.code_dims = {
            'verb_given_human_app': 2,
            'verb_given_object_app': 2,
            'verb_given_boxes_and_object_label': 2,
            'verb_given_human_pose': 2,
            'word_vector': 2,
        }

        self.num_hidden_layers = {
            'verb_given_human_app': 2,
            'verb_given_object_app': 2,
            'verb_given_boxes_and_object_label': 2,
            'verb_given_human_pose': 2,
            'word_vector': 2,
        }

        self.drop_prob = 0.2


class FeatAutoencoders(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(FeatAutoencoders,self).__init__()
        self.const = copy.deepcopy(const)
        self.aes = {}
        for factor_name in self.const.input_dims.keys():
            self.aes[factor_name] = self.create_autoencoder(
                self.const.input_dims[factor_name],
                self.const.code_dims[factor_name],
                self.const.num_hidden_layers[factor_name],
                self.const.drop_prob)
            self.add_module(factor_name,self.aes[factor_name])
        
    def create_autoencoder(
            self,
            input_dim,
            code_dim,
            num_hidden_layers,
            drop_prob):
        ae_const = AutoencoderConstants()
        ae_const.input_dim = input_dim
        ae_const.code_dim = code_dim
        ae_const.num_hidden_layers = num_hidden_layers
        ae_const.drop_prob = drop_prob
        ae = Autoencoder(ae_const)
        return ae

    def forward(self,feats):
        codes = {}
        outputs = {}
        for factor_name in feats.keys():
            ae = self.aes[factor_name]
            x = feats[factor_name]
            code, output = ae(x)
            codes[factor_name] = code
            outputs[factor_name] = output
        return codes, outputs

    def compute_recon_loss(self,outputs,feats,verb_ids=None):
        if verb_ids is not None:
            feats = {factor: feats[factor][verb_ids] for factor in feats.keys()}
            outputs = {factor: outputs[factor][verb_ids] \
                for factor in outputs.keys()}

        recon_losses = {'total': 0}
        for factor_name in feats.keys():
            ae = self.aes[factor_name]
            x = feats[factor_name]
            output = outputs[factor_name]
            recon_loss = ae.loss_fn(x,output)
            recon_losses[factor_name] = recon_loss
            recon_losses['total'] += recon_loss
        return recon_losses

    def reconstruct(self,codes):
        outputs = {}
        for factor_name in codes.keys():
            ae = self.aes[factor_name]
            code = codes[factor_name]
            output = ae.decoder(code)
            outputs[factor_name] = output
        return outputs
