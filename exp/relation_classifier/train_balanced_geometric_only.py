import os
import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
tqdm.monitor_interval = 0
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboard_logger import configure, log_value
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import utils.io as io
import utils.losses as losses
from utils.pytorch_layers import get_activation
from utils.model import Model
from utils.constants import save_constants
from exp.relation_classifier.models.relation_classifier_model import \
    RelationClassifier, BoxAwareRelationClassifier
from exp.relation_classifier.models.geometric_factor_model import \
    GeometricFactor, GeometricFactorPairwise
from exp.relation_classifier.models.gather_relation_model import GatherRelation
from exp.relation_classifier.data.features_balanced import FeaturesBalanced


def train_model(model,dataset_train,dataset_val,exp_const):
    params = itertools.chain(
        model.geometric_factor.parameters(),
        model.gather_relation.parameters())
    optimizer = optim.Adam(params,lr=exp_const.lr)
    
    if exp_const.focal_loss:
        criterion = losses.FocalLoss()
    else:
        criterion = nn.BCELoss()
    
    sigmoid = get_activation('Sigmoid')

    step = 0
    optimizer.zero_grad()
    for epoch in range(exp_const.num_epochs):
        sampler = RandomSampler(dataset_train)
        for i, sample_id in enumerate(sampler):
            data = dataset_train[sample_id]
            
            feats = {}
            feats['box'] = Variable(torch.cuda.FloatTensor(data['box_feat']))
            
            hoi_labels = Variable(torch.cuda.FloatTensor(data['hoi_label_vec']))

            model.geometric_factor.train()
            model.gather_relation.train()
            if model.const.geometric_per_hoi:
                geometric_logits = model.geometric_factor(feats)
            else:
                geometric_factor = model.geometric_factor(feats)
                geometric_logits = model.gather_relation(geometric_factor)
            relation_prob_vec = sigmoid(geometric_logits)

            hoi_prob = relation_prob_vec

            loss = criterion(hoi_prob,hoi_labels)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            loss.backward()
            if step%exp_const.imgs_per_batch==0:
                optimizer.step()
                optimizer.zero_grad()

            
            if step%20==0:
                num_tp = np.sum(data['hoi_label'])
                num_fp = data['hoi_label'].shape[0]-num_tp
                max_prob = hoi_prob.max().data[0]
                max_prob_tp = torch.max(hoi_prob*hoi_labels).data[0]
                log_str = \
                    'Epoch: {} | Iter: {} | Step: {} | ' + \
                    'Train Loss: {:.8f} | TPs: {} | FPs: {} | ' + \
                    'Max TP Prob: {:.8f} | Max Prob: {:.8f}'
                log_str = log_str.format(
                    epoch,
                    i,
                    step,
                    loss.data[0],
                    num_tp,
                    num_fp,
                    max_prob_tp,
                    max_prob)
                print(log_str)
                log_value('train_loss',loss.data[0],step)
                log_value('max_prob',max_prob,step)
                log_value('max_prob_tp',max_prob_tp,step)

            if step%400==0:
                print(exp_const.exp_name)

            if step%2000==0:
                val_loss = eval_model(model,dataset_val,exp_const)
                log_value('val_loss',val_loss,step)
                log_str = \
                    'Epoch: {} | Iter: {} | Step: {} | Val Loss: {:.8f}'
                log_str = log_str.format(
                    epoch,
                    i,
                    step,
                    loss.data[0],
                    val_loss)
                print(log_str)

            if step%5000==0:
                geometric_factor_pth = os.path.join(
                    exp_const.model_dir,
                    f'geometric_factor_{step}')
                torch.save(
                    model.geometric_factor.state_dict(),
                    geometric_factor_pth)

            step += 1


def eval_model(model,dataset,exp_const):
    sigmoid = get_activation('Sigmoid')
    model.geometric_factor.eval()
    model.gather_relation.eval()
    criterion = nn.BCELoss()
    step = 0
    val_loss = 0
    count = 0
    sampler = SequentialSampler(dataset)
    for sample_id in tqdm(sampler):
        if step==500:
            break

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

        loss = criterion(hoi_prob,hoi_labels)

        batch_size = hoi_prob.size(0)
        val_loss += (batch_size*loss.data[0])
        count += batch_size
        step += 1

    val_loss = val_loss / float(count)
    return val_loss


def main(exp_const,data_const,model_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    io.mkdir_if_not_exists(exp_const.log_dir)
    io.mkdir_if_not_exists(exp_const.model_dir)
    configure(exp_const.log_dir)
    save_constants({'exp':exp_const,'data':data_const},exp_const.exp_dir)

    print('Creating model ...')
    model = Model()
    model.const = model_const
    if model.const.geometric_pairwise:
        model.geometric_factor = \
            GeometricFactorPairwise(model.const.geometric_factor).cuda()
    else:
        model.geometric_factor = \
            GeometricFactor(model.const.geometric_factor).cuda()
    model.gather_relation = GatherRelation(model.const.gather_relation).cuda()
    model.to_txt(exp_const.exp_dir,single_file=True)

    print('Creating data loaders ...')
    data_const.subset = 'train'
    data_const.balanced_sampling = True
    dataset_train = FeaturesBalanced(data_const)

    data_const.subset = 'val'
    data_const.balanced_sampling = False
    dataset_val = FeaturesBalanced(data_const)

    train_model(model,dataset_train,dataset_val,exp_const)


    
    

    