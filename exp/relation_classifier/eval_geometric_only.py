import os
import h5py
import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
tqdm.monitor_interval = 0
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tensorboard_logger import configure, log_value

import utils.io as io
from utils.model import Model
from utils.pytorch_layers import get_activation
from utils.constants import save_constants
from exp.relation_classifier.models.relation_classifier_model import \
    RelationClassifier, BoxAwareRelationClassifier
from exp.relation_classifier.models.geometric_factor_model import \
    GeometricFactor, GeometricFactorPairwise
from exp.relation_classifier.models.gather_relation_model import GatherRelation
from exp.relation_classifier.data.features_balanced import FeaturesBalanced


def eval_model(model,dataset,exp_const):
    print('Creating hdf5 file for predicted hoi dets ...')
    pred_hoi_dets_hdf5 = os.path.join(
        exp_const.exp_dir,
        f'pred_hoi_dets_{dataset.const.subset}_{model.const.model_num}.hdf5')
    pred_hois = h5py.File(pred_hoi_dets_hdf5,'w')
    model.geometric_factor.eval()
    model.gather_relation.eval()
    sigmoid = get_activation('Sigmoid')
    sampler = SequentialSampler(dataset)
    for sample_id in tqdm(sampler):
        data = dataset[sample_id]
        
        feats = {}
        feats['box'] = Variable(torch.cuda.FloatTensor(data['box_feat']))

        hoi_labels = Variable(torch.cuda.FloatTensor(data['hoi_label_vec']))

        if model.const.geometric_per_hoi:
            geometric_logits = model.geometric_factor(feats)
        else:
            geometric_factor = model.geometric_factor(feats)
            geometric_logits = model.gather_relation(geometric_factor)
        relation_prob_vec = sigmoid(geometric_logits)

        hoi_prob = relation_prob_vec
        hoi_prob = hoi_prob.data.cpu().numpy()
        
        num_cand = hoi_prob.shape[0]
        scores = hoi_prob[np.arange(num_cand),np.array(data['hoi_idx'])]
        human_obj_boxes_scores = np.concatenate((
            data['human_box'],
            data['object_box'],
            np.expand_dims(scores,1)),1)

        global_id = data['global_id']
        pred_hois.create_group(global_id)
        pred_hois[global_id].create_dataset(
            'human_obj_boxes_scores',
            data=human_obj_boxes_scores)
        pred_hois[global_id].create_dataset(
            'start_end_ids',
            data=data['start_end_ids_'])

    pred_hois.close()


def main(exp_const,data_const,model_const):
    print('Loading model ...')
    model = Model()
    model.const = model_const
    if model.const.geometric_pairwise:
        model.geometric_factor = \
            GeometricFactorPairwise(model.const.geometric_factor).cuda()
    else:
        model.geometric_factor = \
            GeometricFactor(model.const.geometric_factor).cuda()
    model.gather_relation = GatherRelation(model.const.gather_relation).cuda()
    model.geometric_factor.load_state_dict(torch.load(
        model.const.geometric_factor.model_pth))

    print('Creating data loader ...')
    dataset = FeaturesBalanced(data_const)

    eval_model(model,dataset,exp_const)


    
    

    