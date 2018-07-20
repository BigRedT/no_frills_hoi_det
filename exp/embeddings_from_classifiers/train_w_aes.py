import os
import csv
import time
import copy
import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
tqdm.monitor_interval = 0
import torch.optim as optim
from torch.autograd import Variable
from tensorboard_logger import configure, log_value

import utils.io as io
from utils.model import Model
from utils.constants import save_constants
from utils.pytorch_layers import Identity, adjust_learning_rate
from exp.embeddings_from_classifiers.models.one_to_all_model import \
    OneToAll
from exp.embeddings_from_classifiers.models.feat_autoencoders import \
    FeatAutoencoders
from exp.embeddings_from_classifiers.load_classifiers import \
    load_pretrained_hoi_classifier


def train(model,data_const,exp_const):
    params = itertools.chain(
        model.one_to_all.parameters(),
        model.feat_ae.parameters())
    
    optimizer = optim.Adam(
        params,
        lr=exp_const.lr,
        weight_decay=exp_const.weight_decay)
    
    for step in range(exp_const.num_steps):
        adjust_learning_rate(
            optimizer, 
            exp_const.lr, 
            step, 
            decay_by=0.5, 
            decay_every=5000)

        # Train
        model.feat_ae.train()
        model.one_to_all.train()

        codes, recon_feats = model.feat_ae(data_const.feats)
        pred_codes, pred_verb_vecs = model.one_to_all(codes)
        
        if model.one_to_all.const.use_coupling_variable is True:
            coupling_variable = model.one_to_all.verb_vecs
        else:
            coupling_variable = pred_verb_vecs['word_vector']

        one_to_all_train_losses, all_to_one_train_losses = \
            model.one_to_all.compute_loss(
                pred_codes,
                codes,
                pred_verb_vecs,
                coupling_variable,
                range(0,exp_const.num_train_verbs-1))
        
        train_recon_losses = model.feat_ae.compute_recon_loss(
            recon_feats,
            data_const.feats,
            range(0,exp_const.num_train_verbs-1))

        # recon_feats_from_pred_codes = model.feat_ae.reconstruct(pred_codes)
        # recon_from_pred_code_losses = \
        #     model.one_to_all.compute_one_to_all_losses(
        #         recon_feats_from_pred_codes,
        #         data_const.feats,
        #         range(0,exp_const.num_train_verbs-1))

        total_train_loss = \
            exp_const.concept_loss_weight*(
                one_to_all_train_losses['total'] + \
                all_to_one_train_losses['total']) + \
            train_recon_losses['total']

        optimizer.zero_grad()
        total_train_loss.backward()
        optimizer.step()

        # Evaluate
        model.one_to_all.eval()
        model.feat_ae.eval()

        pred_verb_vecs = {}
        pred_verb_vecs['word_vector'] = \
            model.one_to_all.all_to_one_mlps['word_vector'](
                codes['word_vector'])
        
        pred_codes = {}
        for factor, mlp in model.one_to_all.one_to_all_mlps.items():
            pred_codes[factor] = mlp(pred_verb_vecs['word_vector'])

        one_to_all_test_losses = model.one_to_all.compute_one_to_all_losses(
            pred_codes,
            codes,
            range(117-exp_const.num_test_verbs,117))

        pred_feats = model.feat_ae.reconstruct(pred_codes)
        recon_feats = model.feat_ae(data_const.feats)[1]

        test_recon_losses = model.feat_ae.compute_recon_loss(
            recon_feats,
            data_const.feats,
            range(117-exp_const.num_test_verbs,117))
        
        test_pred_losses = model.feat_ae.compute_recon_loss(
            pred_feats,
            data_const.feats,
            range(117-exp_const.num_test_verbs,117))
        train_pred_losses = model.feat_ae.compute_recon_loss(
            pred_feats,
            data_const.feats,
            range(0,exp_const.num_train_verbs-1))
        
        if step%10==0:
            print_str = \
                'Step: {} | ' + \
                'Total Train loss: {} | ' + \
                'One-To-All Loss: {} (Train) / {} (Test) | ' + \
                'Recon Loss: {} (Train) / {} (Test) | ' + \
                'Pred Loss: {} (Train) / {} (Test)'
            print_str = print_str.format(
                step,
                round(total_train_loss.data[0],4),
                round(one_to_all_train_losses['total'].data[0],4),
                round(one_to_all_test_losses['total'].data[0],4),
                round(train_recon_losses['total'].data[0],4),
                round(test_recon_losses['total'].data[0],4),
                round(train_pred_losses['total'].data[0],4),
                round(test_recon_losses['total'].data[0],4))
            print(print_str)

            # Log train losses
            prefix = 'one_to_all_train'
            for loss_type, loss_val in one_to_all_train_losses.items():
                log_name = f'{prefix}_{loss_type}'
                log_value(log_name,loss_val.data[0],step)

            prefix = 'all_to_one_train'
            for loss_type, loss_val in all_to_one_train_losses.items():
                log_name = f'{prefix}_{loss_type}'
                log_value(log_name,loss_val.data[0],step)

            prefix = 'ae_recon_train'
            for loss_type, loss_val in train_recon_losses.items():
                log_name = f'{prefix}_{loss_type}'
                log_value(log_name,loss_val.data[0],step)

            prefix = 'pred_train'
            for loss_type, loss_val in train_pred_losses.items():
                log_name = f'{prefix}_{loss_type}'
                log_value(log_name,loss_val.data[0],step)

            # Log test losses
            prefix = 'one_to_all_test'
            for loss_type, loss_val in one_to_all_test_losses.items():
                log_name = f'{prefix}_{loss_type}'
                log_value(log_name,loss_val.data[0],step)

            prefix = 'ae_recon_test'
            for loss_type, loss_val in test_recon_losses.items():
                log_name = f'{prefix}_{loss_type}'
                log_value(log_name,loss_val.data[0],step)

            prefix = 'pred_test'
            for loss_type, loss_val in test_pred_losses.items():
                log_name = f'{prefix}_{loss_type}'
                log_value(log_name,loss_val.data[0],step)

        if step%2500==0:
            one_to_all_pth = os.path.join(
                exp_const.model_dir,
                f'one_to_all_{step}')
            torch.save(
                model.one_to_all.state_dict(),
                one_to_all_pth)

            feat_ae_pth = os.path.join(
                exp_const.model_dir,
                f'feat_ae_{step}')
            torch.save(
                model.feat_ae.state_dict(),
                feat_ae_pth)
        

def main(exp_const,data_const,model_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    io.mkdir_if_not_exists(exp_const.log_dir)
    io.mkdir_if_not_exists(exp_const.model_dir)
    configure(exp_const.log_dir)
    save_constants(
        {'exp':exp_const,'data':data_const,'model':model_const},
        exp_const.exp_dir)

    print('Creating model ...')
    model = Model()
    model.const = model_const
    model.feat_ae = FeatAutoencoders(model.const.feat_ae).cuda()
    model.one_to_all = OneToAll(model_const.one_to_all).cuda()
    model.to_txt(exp_const.exp_dir,single_file=True)

    print('Load pretrained Hoi Classifier model ...')
    feats,_ = load_pretrained_hoi_classifier()

    print('Load pretrained glove verb vectors')
    if exp_const.word_vec=='random':
        std = np.sqrt(1/model.const.one_to_all.verb_vec_dim)
        word_vecs = np.random.normal(
            scale=std,
            size=(
                model.const.one_to_all.num_verbs,
                model.const.one_to_all.verb_vec_dim))
    elif exp_const.word_vec=='glove':
        word_vecs = 0.1*np.load(data_const.glove_verb_vecs_npy)
    else:
        assert_str = 'Only glove and random word vecs are supported'
        assert(False), assert_str

    feats['word_vector'] = Variable(torch.FloatTensor(word_vecs))

    for factor, feat in feats.items():
        feats[factor] = feat.cuda()

    data_const.feats = feats

    train(model,data_const,exp_const)
