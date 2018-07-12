import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
tqdm.monitor_interval = 0

import utils.io as io
from utils.model import Model
from exp.embeddings_from_classifiers.models.one_to_all_model import \
    OneToAll, OneToAllConstants
from exp.embeddings_from_classifiers.load_classifiers import \
    load_pretrained_hoi_classifier


def main(model_path,num_train_verbs=100):
    print('Load gt model ...')
    gt_classifiers, gt_model = load_pretrained_hoi_classifier()

    print('Load predicted model ...')
    pred_model = copy.deepcopy(gt_model)
    pred_model.load_state_dict(torch.load(model_path))

    const = OneToAllConstants()
    one_to_all_model = OneToAll(const)

    pred_classifiers = {}
    for factor in gt_classifiers.keys():
        #import pdb; pdb.set_trace()
        pred_classifiers[factor] = \
            pred_model.__getattr__(factor).mlp.layers[-1][0].weight

    losses = one_to_all_model.compute_one_to_all_losses(
        pred_classifiers,
        gt_classifiers,
        range(num_train_verbs,117))
    for k,v in losses.items():
        print(f'{k}: {v.data[0]}')

if __name__=='__main__':
    model_path = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/embeddings_from_classifier/' + \
        'ablation_identity_vs_mlp/make_identity_True/' + \ #_2_hidden_layers/' + \
        'models/hoi_classifier_25000')
    main(model_path)