import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
tqdm.monitor_interval = 0

import utils.io as io
from utils.model import Model
from exp.embeddings_from_classifiers.models.one_to_all_model import OneToAll
from exp.embeddings_from_classifiers.load_classifiers import \
    load_pretrained_hoi_classifier

def main(exp_const,data_const,model_const):
    print('Creating models ...')
    print('    Creating one_to_all model ...')
    model = Model()
    model.const = model_const
    model.one_to_all = OneToAll(model_const.one_to_all)
    model.one_to_all.load_state_dict(
        torch.load(model.const.one_to_all.model_path))
    print('    Creating hoi_classifier model ...')
    _, model.hoi_classifier = load_pretrained_hoi_classifier()

    print('Load glove vectors ...')
    word_vecs = 0.1*np.load(data_const.glove_verb_vecs_npy)
    word_vecs = Variable(torch.FloatTensor(word_vecs))

    print('Predict classifiers from glove vectors ...')
    pred_verb_vecs = model.one_to_all.all_to_one_mlps['word_vector'](word_vecs)
    pred_feats = {}
    for factor, mlp in model.one_to_all.one_to_all_mlps.items():
        pred_feats[factor] = mlp(pred_verb_vecs)

    print('Replace pretrained hoi_classifier weights with predicted weights ...')
    for factor, weight in pred_feats.items():
        if factor=='word_vector':
            continue
        model.hoi_classifier.__getattr__(factor).mlp.layers[-1][0].weight = \
            nn.Parameter(weight.data)
    
    print('Saving model ...')
    hoi_classifier_path = os.path.join(
        exp_const.model_dir,
        f'hoi_classifier_{model.const.one_to_all.model_num}')
    torch.save(
        model.hoi_classifier.state_dict(),
        hoi_classifier_path)
