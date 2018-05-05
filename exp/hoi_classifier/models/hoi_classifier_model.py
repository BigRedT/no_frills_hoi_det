import torch
import copy
import torch.nn as nn
from torch.autograd import Variable

import utils.io as io
import utils.pytorch_layers as pytorch_layers
from exp.hoi_classifier.models.verb_given_object_appearance import \
    VerbGivenObjectAppearanceConstants, VerbGivenObjectAppearance
from exp.hoi_classifier.models.verb_given_human_appearance import \
    VerbGivenHumanAppearanceConstants, VerbGivenHumanAppearance    
from exp.hoi_classifier.models.verb_given_boxes_and_object_label import \
    VerbGivenBoxesAndObjectLabelConstants, VerbGivenBoxesAndObjectLabel
from exp.hoi_classifier.models.verb_given_human_pose import \
    VerbGivenHumanPoseConstants, VerbGivenHumanPose
from exp.hoi_classifier.models.scatter_verbs_to_hois import \
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
        self.verb_given_boxes_and_object_label = True
        self.verb_given_human_pose = True
        self.rcnn_det_prob = True
        self.scatter_verbs_to_hois = ScatterVerbsToHoisConstants()

    @property
    def selected_factor_constants(self):
        factor_constants = {}
        for factor_name in self.selected_factor_names:
            const = self.FACTOR_NAME_TO_MODULE_CONSTANTS[factor_name]
            factor_constants[factor_name] = const
        return factor_constants

    @property
    def selected_factor_names(self): 
        factor_names = []
        if self.verb_given_appearance:
            factor_names.append('verb_given_object_app')
            factor_names.append('verb_given_human_app')
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
        for name, const in self.const.selected_factor_constants.items():
            self.create_factor(name,const)

    def create_factor(self,factor_name,factor_const):
        factor = self.FACTOR_NAME_TO_MODULE[factor_name](factor_const)
        setattr(self,factor_name,factor)

    def forward(self,feats):
        factor_scores = {}
        any_verb_factor = False
        verb_factor_scores = 0
        for factor_name in self.const.selected_factor_names:
            module = getattr(self,factor_name)
            factor_scores[factor_name] = module(feats)
            if 'verb_given' in factor_name:
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