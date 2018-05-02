import os
import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class ScatterVerbsToHoisConstants(io.JsonSerializableClass):
    def __init__(self):
        super(ScatterVerbsToHoisConstants,self).__init__()
        self.hoi_list_json = os.path.join(
            os.getcwd(),
            'data_symlinks/hico_processed/hoi_list.json')
        self.verb_list_json = os.path.join(
            os.getcwd(),
            'data_symlinks/hico_processed/verb_list.json')
        

class ScatterVerbsToHois(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(ScatterVerbsToHois,self).__init__()
        self.const = copy.deepcopy(const)
        self.hoi_dict = self.get_hoi_dict(self.const.hoi_list_json)
        self.verb_to_id = self.get_verb_to_id(self.const.verb_list_json)

    def get_hoi_dict(self,hoi_list_json):
        hoi_list = io.load_json_object(hoi_list_json)
        hoi_dict = {hoi['id']:hoi for hoi in hoi_list}
        return hoi_dict

    def get_verb_to_id(self,verb_list_json):
        verb_list = io.load_json_object(verb_list_json)
        verb_to_id = {verb['name']: verb['id'] for verb in verb_list}
        return verb_to_id

    def forward(self,verb_scores):
        batch_size, num_verbs = verb_scores.size()
        num_hois = len(self.hoi_dict)
        hoi_scores = Variable(torch.zeros(batch_size,num_hois)).cuda()
        for hoi_id, hoi in self.hoi_dict.items():
            verb = hoi['verb']
            verb_idx = int(self.verb_to_id[verb])-1
            hoi_idx = int(hoi_id)-1
            hoi_scores[:,hoi_idx] = verb_scores[:,verb_idx]
        return hoi_scores