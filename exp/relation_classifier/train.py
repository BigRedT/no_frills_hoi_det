import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils.io as io
from utils.model import Model
from utils.constants import save_constants
from exp.relation_classifier.relation_classifier_model import RelationClassifier
from exp.relation_classifier.gather_relation_model import GatherRelation
from exp.relation_classifier.features import Features


def train_model(model,data_loader_train,data_loader_val,exp_const):
    

def main(exp_const,data_const,model_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    save_constants({'exp':exp_const,'data':data_const},exp_const.exp_dir)

    print('Creating model ...')
    model = Model()
    model.relation_classifier = \
        RelationClassifier(model_const.relation_classifier)
    model.gather_relation = GatherRelation(model_const.gather_relation)
    model.to_txt(exp_const.exp_dir,single_file=True)

    print('Creating data loaders ...')
    data_const.subset = 'train'
    dataset_train = Features(data_const)
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=exp_const.batch_size,
        shuffle=True,
        drop_last=False)

    data_const.subset = 'val'
    dataset_val = Features(data_const)
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=exp_const.batch_size,
        shuffle=True,
        drop_last=False)

    train_model(model,data_loader_train,data_loader_val,exp_const)


    
    

    