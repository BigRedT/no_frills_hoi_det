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
from exp.hoi_classifier.models.hoi_classifier_model import HoiClassifier
from exp.hoi_classifier.data.features_dataset import Features


def oracle_prob(
        prob_vec,
        prob_mask,
        hoi_idx,
        oracle_labels,
        oracle_human,
        oracle_object,
        oracle_verb):
    B = prob_vec['human'].shape[0]
    
    if oracle_human:
        prob_vec_human = 0*prob_vec['human']
        prob_vec_human[np.arange(B),hoi_idx] = oracle_labels[:,0]
    else:
        prob_vec_human = prob_vec['human']

    if oracle_object:
        prob_vec_object = 0*prob_vec['object']
        prob_vec_object[np.arange(B),hoi_idx] = oracle_labels[:,1]
    else:
        prob_vec_object = prob_vec['object']

    if oracle_verb:
        prob_vec_verb = 0*prob_vec['verb']
        prob_vec_verb[np.arange(B),hoi_idx] = oracle_labels[:,2]
    else:
        prob_vec_verb = prob_vec['verb']

    prob_vec = \
        prob_mask * \
        prob_vec_human * \
        prob_vec_object * \
        prob_vec_verb

    return prob_vec

def eval_model(model,dataset,exp_const):
    print('Read oracle labels ...')
    oracle_labels_hdf5 = h5py.File(dataset.const.hoi_cand_oracle_labels_hdf5,'r')

    print('Creating hdf5 file for predicted hoi dets ...')
    pred_hoi_dets_hdf5 = os.path.join(
        exp_const.exp_dir,
        f'pred_hoi_dets_' + \
        f'oracle_human_{exp_const.oracle_human}_' + \
        f'object_{exp_const.oracle_object}_' + \
        f'verb_{exp_const.oracle_verb}_' + \
        f'{dataset.const.subset}_{model.const.model_num}.hdf5')
    pred_hois = h5py.File(pred_hoi_dets_hdf5,'w')
    model.hoi_classifier.eval()
    sampler = SequentialSampler(dataset)
    for sample_id in tqdm(sampler):
        data = dataset[sample_id]
        global_id = data['global_id']

        feats = {
            'human_rcnn': Variable(torch.cuda.FloatTensor(data['human_feat'])),
            'object_rcnn': Variable(torch.cuda.FloatTensor(data['object_feat'])),
            'box': Variable(torch.cuda.FloatTensor(data['box_feat'])),
            'absolute_pose': Variable(torch.cuda.FloatTensor(data['absolute_pose'])),
            'relative_pose': Variable(torch.cuda.FloatTensor(data['relative_pose'])),
            'human_prob_vec': Variable(torch.cuda.FloatTensor(data['human_prob_vec'])),
            'object_prob_vec': Variable(torch.cuda.FloatTensor(data['object_prob_vec'])),
            'object_one_hot': Variable(torch.cuda.FloatTensor(data['object_one_hot'])),
            'prob_mask': Variable(torch.cuda.FloatTensor(data['prob_mask']))
        }        

        prob_vec, factor_scores = model.hoi_classifier(feats)
        
        #hoi_prob = prob_vec['hoi']
        oracle_labels = Variable(torch.cuda.FloatTensor(oracle_labels_hdf5[global_id]))
        hoi_prob = oracle_prob(
            prob_vec,
            feats['prob_mask'],
            data['hoi_idx'],
            oracle_labels,
            oracle_human=exp_const.oracle_human,
            oracle_object=exp_const.oracle_object,
            oracle_verb=exp_const.oracle_verb)
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
    model.hoi_classifier = HoiClassifier(model.const.hoi_classifier).cuda()
    if model.const.model_num == -1:
        print('No pretrained model will be loaded since model_num is set to -1')
    else:
        model.hoi_classifier.load_state_dict(
            torch.load(model.const.hoi_classifier.model_pth))

    print('Creating data loader ...')
    dataset = Features(data_const)

    eval_model(model,dataset,exp_const)


    
    

    