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


def update_confusion_matrix(conf_mat,count,hoi_idx,factor_scores,dataset):
    sigmoid = nn.Sigmoid()
    verb_scores = 0
    for factor_name, scores in factor_scores.items():
        if 'verb_given' in factor_name:
            verb_scores += scores
    verb_scores = sigmoid(verb_scores)
    verb_scores = verb_scores.data.cpu().numpy()
    B = verb_scores.shape[0]
    for i in range(B):
        hoi_id = str(hoi_idx[i]+1).zfill(3)
        verb = dataset.hoi_dict[hoi_id]['verb']
        j = int(dataset.verb_to_id[verb])-1
        conf_mat[j] = conf_mat[j] + verb_scores[i]
        count[j] = count[j] + 1


def eval_model(model,dataset,exp_const):
    conf_mat = np.zeros([117,117])
    count = np.zeros([117,117])
    model.hoi_classifier.eval()
    sampler = SequentialSampler(dataset)
    for sample_id in tqdm(sampler):
        data = dataset[sample_id]
        if data['human_feat'].shape[0]==0:
            continue

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
        
        update_confusion_matrix(conf_mat,count,data['hoi_idx'],factor_scores,dataset)
    
    conf_mat = conf_mat / (count+1e-6)
    conf_mat_npy = os.path.join(exp_const.exp_dir,'verb_conf_mat.npy')
    np.save(conf_mat_npy,conf_mat)        

    import pdb; pdb.set_trace()


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
    data_const.fp_to_tp_ratio = 0
    data_const.balanced_sampling = True
    dataset = Features(data_const)

    eval_model(model,dataset,exp_const)


    
    

    