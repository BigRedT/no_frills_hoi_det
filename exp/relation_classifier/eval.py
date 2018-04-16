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
from utils.constants import save_constants
from exp.relation_classifier.relation_classifier_model import RelationClassifier
from exp.relation_classifier.gather_relation_model import GatherRelation
from exp.relation_classifier.features_balanced import FeaturesBalanced


def eval_model(model,dataset,exp_const):
    print('Creating hdf5 file for predicted hoi dets ...')
    pred_hoi_dets_hdf5 = os.path.join(
        exp_const.exp_dir,
        f'pred_hoi_dets_{dataset.const.subset}.hdf5')
    pred_hois = h5py.File(pred_hoi_dets_hdf5,'w')
    model.relation_classifier.eval()
    model.gather_relation.eval()
    sampler = SequentialSampler(dataset)
    for sample_id in tqdm(sampler):
        data = dataset[sample_id]
        feats = {
            'human_rcnn': Variable(torch.cuda.FloatTensor(data['human_feat'])),
            'object_rcnn': Variable(torch.cuda.FloatTensor(data['object_feat']))
        }
        human_prob_vec = Variable(torch.cuda.FloatTensor(data['human_prob_vec']))
        object_prob_vec = Variable(torch.cuda.FloatTensor(data['object_prob_vec']))
        hoi_labels = Variable(torch.cuda.FloatTensor(data['hoi_label_vec']))

        relation_prob = model.relation_classifier(feats)
        relation_prob_vec = model.gather_relation(relation_prob)

        hoi_prob = human_prob_vec * object_prob_vec * relation_prob_vec
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
    model.relation_classifier = \
        RelationClassifier(model_const.relation_classifier).cuda()
    model.gather_relation = GatherRelation(model_const.gather_relation).cuda()
    model.relation_classifier.load_state_dict(torch.load(
        model_const.relation_classifier.model_pth))

    print('Creating data loader ...')
    dataset = FeaturesBalanced(data_const)

    eval_model(model,dataset,exp_const)


    
    

    