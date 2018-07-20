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
from exp.embeddings_from_classifiers.models.feat_autoencoders import \
    FeatAutoencoders
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
    model.feat_ae = FeatAutoencoders(model_const.feat_ae)
    model.feat_ae.load_state_dict(
        torch.load(model.const.feat_ae.model_path))

    print('Load glove vectors ...')
    word_vecs = 0.1*np.load(data_const.glove_verb_vecs_npy)

    print('    Creating hoi_classifier model ...')
    feats, model.hoi_classifier = load_pretrained_hoi_classifier()
    feats['word_vector'] = Variable(torch.FloatTensor(word_vecs))

    print('Predict classifiers from glove vectors ...')
    model.feat_ae.eval()
    model.one_to_all.eval()
    codes, recon_feats = model.feat_ae(feats)

    pred_codes = {}
    for factor, mlp in model.one_to_all.one_to_all_mlps.items():
        pred_codes[factor] = mlp(codes['word_vector'])

    pred_feats = model.feat_ae.reconstruct(pred_codes)
    np.set_printoptions(suppress=True)
    print('-'*80)
    print('Seen')
    print('-'*80)
    print('GT feat')
    print(feats['verb_given_object_app'].data.numpy()[:3,:5])
    print('Recon feat')
    print(recon_feats['verb_given_object_app'].data.numpy()[:3,:5])
    print('Pred feat')
    print(pred_feats['verb_given_object_app'].data.numpy()[:3,:5])
    print('-'*80)
    print('Unseen')
    print('-'*80)
    print('GT feat')
    print(feats['verb_given_object_app'].data.numpy()[-3:,:5])
    print('Recon feat')
    print(recon_feats['verb_given_object_app'].data.numpy()[-3:,:5])
    print('Pred feat')
    print(pred_feats['verb_given_object_app'].data.numpy()[-3:,:5])

    print('Replace pretrained hoi_classifier weights with pred_feats ...')
    for factor, weight in pred_feats.items():
        if factor=='word_vector':
            continue
        model.hoi_classifier.__getattr__(factor).mlp.layers[-1][0].weight = \
            nn.Parameter(weight.data)

    print('Saving model ...')
    hoi_classifier_path = os.path.join(
        exp_const.model_dir,
        f'hoi_classifier_pred_feats_{model.const.model_num}')
    torch.save(
        model.hoi_classifier.state_dict(),
        hoi_classifier_path)

    print('Replace pretrained hoi_classifier weights with recon_feats ...')
    for factor, weight in recon_feats.items():
        if factor=='word_vector':
            continue
        model.hoi_classifier.__getattr__(factor).mlp.layers[-1][0].weight = \
            nn.Parameter(weight.data)

    print('Saving model ...')
    hoi_classifier_path = os.path.join(
        exp_const.model_dir,
        f'hoi_classifier_recon_feat_{model.const.model_num}')
    torch.save(
        model.hoi_classifier.state_dict(),
        hoi_classifier_path)