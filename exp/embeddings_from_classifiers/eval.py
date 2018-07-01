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
from exp.hoi_classifier.data.features_dataset import Features
from exp.embeddings_from_classifiers.load_classifiers import \
    load_pretrained_hoi_classifier


def eval_model(model,dataset,exp_const):
    print('Creating hdf5 file for predicted hoi dets ...')
    pred_hoi_dets_hdf5 = os.path.join(
        exp_const.exp_dir,
        f'pred_hoi_dets_{dataset.const.subset}_{model.const.model_num}.hdf5')
    pred_hois = h5py.File(pred_hoi_dets_hdf5,'w')
    model.hoi_classifier.eval()
    sampler = SequentialSampler(dataset)
    for sample_id in tqdm(sampler):
        data = dataset[sample_id]
        
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
        
        hoi_prob = prob_vec['hoi']
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
    _, model.hoi_classifier = load_pretrained_hoi_classifier()
    model.hoi_classifier.load_state_dict(
        torch.load(model.const.hoi_classifier_path))
    model.hoi_classifier.cuda()

    print('Creating data loader ...')
    dataset = Features(data_const)

    eval_model(model,dataset,exp_const)


    
    

    