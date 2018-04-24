import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers


class GatherRelationConstants(io.JsonSerializableClass):
    def __init__(self):
        super(GatherRelationConstants,self).__init__()
        self.hoi_list_json = None
        self.verb_list_json = None
        

class GatherRelation(nn.Module,io.WritableToFile):
    def __init__(self,const):
        super(GatherRelation,self).__init__()
        self.const = copy.deepcopy(const)
        self.hoi_dict = self.get_hoi_dict(self.const.hoi_list_json)
        self.relation_to_id = \
            self.get_relation_to_id(self.const.verb_list_json)

    def get_hoi_dict(self,hoi_list_json):
        hoi_list = io.load_json_object(hoi_list_json)
        hoi_dict = {hoi['id']:hoi for hoi in hoi_list}
        return hoi_dict

    def get_relation_to_id(self,verb_list_json):
        verb_list = io.load_json_object(verb_list_json)
        relation_to_id = {verb['name']: verb['id'] for verb in verb_list}
        return relation_to_id

    def forward(self,relation_prob):
        batch_size, num_relations = relation_prob.size()
        num_hois = len(self.hoi_dict)
        output = Variable(torch.zeros(batch_size,num_hois)).cuda()
        for hoi_id, hoi in self.hoi_dict.items():
            relation = hoi['verb']
            relation_idx = int(self.relation_to_id[relation])-1
            hoi_idx = int(hoi_id)-1
            output[:,hoi_idx] = relation_prob[:,relation_idx]
        return output
            
        
