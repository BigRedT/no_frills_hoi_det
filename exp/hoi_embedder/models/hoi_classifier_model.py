import torch
import copy
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import utils.io as io
import utils.pytorch_layers as pytorch_layers
from exp.hoi_embedder.models.verb_given_object_appearance import \
    VerbGivenObjectAppearanceConstants, VerbGivenObjectAppearance
from exp.hoi_embedder.models.verb_given_human_appearance import \
    VerbGivenHumanAppearanceConstants, VerbGivenHumanAppearance    
from exp.hoi_embedder.models.verb_given_boxes_and_object_label import \
    VerbGivenBoxesAndObjectLabelConstants, VerbGivenBoxesAndObjectLabel
from exp.hoi_embedder.models.verb_given_human_pose import \
    VerbGivenHumanPoseConstants, VerbGivenHumanPose
from exp.hoi_embedder.models.scatter_verbs_to_hois import \
    ScatterVerbsToHoisConstants, ScatterVerbsToHois


class HoiClassifierConstants(io.JsonSerializableClass):
    FACTOR_NAME_TO_MODULE_CONSTANTS = {
        'verb_given_object_app': VerbGivenObjectAppearanceConstants(),
        'verb_given_human_app': VerbGivenHumanAppearanceConstants(),
        'verb_given_boxes_and_object_label': VerbGivenBoxesAndObjectLabelConstants(),
        'verb_given_human_pose': VerbGivenHumanPoseConstants()
    }

    def __init__(self):
        super(HoiClassifierConstants,self).__init__()
        self.verb_given_appearance = True
        self.verb_given_human_appearance = True
        self.verb_given_object_appearance = True
        self.verb_given_boxes_and_object_label = True
        self.verb_given_human_pose = True
        self.rcnn_det_prob = True
        self.scatter_verbs_to_hois = ScatterVerbsToHoisConstants()
        self.verb_vec_dim = 300 
        self.num_verbs = 117
        self.verb_vec_init = 'random'
        self.verb_vec_fine_tune = True
        self.glove_verb_vecs_npy = None

    @property
    def selected_factor_constants(self):
        factor_constants = {}
        for factor_name in self.selected_factor_names:
            const = self.FACTOR_NAME_TO_MODULE_CONSTANTS[factor_name]
            if 'verb_given' in factor_name:
                const.verb_vec_dim = self.verb_vec_dim
                assert_str = 'num_verbs does not match between ' + \
                    f'HoiClassifier and factor {factor_name}'
                assert(self.num_verbs==const.num_verbs), assert_str
            factor_constants[factor_name] = const
        return factor_constants

    @property
    def selected_factor_names(self): 
        factor_names = []
        if self.verb_given_appearance:
            factor_names.append('verb_given_object_app')
            factor_names.append('verb_given_human_app')
        elif self.verb_given_human_appearance:
            factor_names.append('verb_given_human_app')
        elif self.verb_given_object_appearance:
            factor_names.append('verb_given_object_app')

        if self.verb_given_boxes_and_object_label:
            factor_names.append('verb_given_boxes_and_object_label')
        
        if self.verb_given_human_pose:
            factor_names.append('verb_given_human_pose')
        
        return factor_names


class HoiClassifier(nn.Module,io.WritableToFile):
    FACTOR_NAME_TO_MODULE = {
        'verb_given_object_app': VerbGivenObjectAppearance,
        'verb_given_human_app': VerbGivenHumanAppearance,
        'verb_given_boxes_and_object_label': VerbGivenBoxesAndObjectLabel,
        'verb_given_human_pose': VerbGivenHumanPose
    }

    def __init__(self,const):
        super(HoiClassifier,self).__init__()
        self.const = copy.deepcopy(const)
        self.sigmoid = pytorch_layers.get_activation('Sigmoid')
        self.scatter_verbs_to_hois = ScatterVerbsToHois(
            self.const.scatter_verbs_to_hois)
        self.verb_vecs = self.create_verb_vecs(
            self.const.num_verbs,
            self.const.verb_vec_dim,
            self.const.verb_vec_init,
            self.const.verb_vec_fine_tune)
        for name, const in self.const.selected_factor_constants.items():
            self.create_factor(name,const)

    def create_verb_vecs(
            self,
            num_verbs,
            verb_vec_dim,
            init='random',
            fine_tune=True):
        if init=='random':
            std = np.sqrt(1/verb_vec_dim)
            verb_vecs = np.random.normal(scale=std,size=(num_verbs,verb_vec_dim))
        elif init=='glove':
            verb_vecs = np.load(self.const.glove_verb_vecs_npy)
        else:
            assert(False),'only random and glove initialization supported'
        verb_vecs = torch.FloatTensor(verb_vecs)
        return nn.Parameter(data=verb_vecs,requires_grad=fine_tune)

    def create_factor(self,factor_name,factor_const):
        factor = self.FACTOR_NAME_TO_MODULE[factor_name](factor_const)
        setattr(self,factor_name,factor)

    def forward(self,feats):
        factor_scores = {}
        any_verb_factor = False
        verb_factor_scores = 0
        for factor_name in self.const.selected_factor_names:
            module = getattr(self,factor_name)
            if 'verb_given' not in factor_name:
                factor_scores[factor_name] = module(feats)
            else:
                try:
                    factor_scores[factor_name] = module(feats,self.verb_vecs)
                except:
                    import pdb; pdb.set_trace()
                any_verb_factor = True
                verb_factor_scores += factor_scores[factor_name]

        if any_verb_factor:
            verb_prob = self.sigmoid(verb_factor_scores)
            verb_prob_vec = self.scatter_verbs_to_hois(verb_prob)
        else:
            verb_prob_vec = 0*feats['human_prob_vec'] + 1

        if self.const.rcnn_det_prob:
            human_prob_vec = feats['human_prob_vec']
            object_prob_vec = feats['object_prob_vec']
        else:
            human_prob_vec = 0*feats['human_prob_vec'] + 1
            object_prob_vec = 0*feats['object_prob_vec'] + 1

        prob_vec = {
            'human': human_prob_vec,
            'object': object_prob_vec,
            'verb': verb_prob_vec,
        }

        prob_vec['hoi'] = \
            feats['prob_mask'] * \
            prob_vec['human'] * \
            prob_vec['object'] * \
            prob_vec['verb']

        return prob_vec, factor_scores