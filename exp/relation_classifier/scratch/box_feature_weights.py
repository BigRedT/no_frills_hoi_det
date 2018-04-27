import os
import numpy as np
import torch
import torch.nn as nn
import plotly
import plotly.graph_objs as go

import utils.io as io
from utils.model import Model
from utils.pytorch_layers import get_activation
from utils.constants import Constants, ExpConstants
from data.hico.hico_constants import HicoConstants
from exp.relation_classifier.models.geometric_factor_model import \
    GeometricFactor, GeometricFactorPairwise
from exp.relation_classifier.models.gather_relation_model import GatherRelation
from exp.relation_classifier.models.geometric_factor_model import \
    GeometricFactorConstants, GeometricFactorPairwiseConstants
from exp.relation_classifier.models.gather_relation_model import \
    GatherRelationConstants


def main():
    exp_name = 'factors_geometric_indicator_imgs_per_batch_1_focal_loss_False_fp_to_tp_ratio_1000'
    out_base_dir = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/relation_classifier')
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')
    io.mkdir_if_not_exists(exp_const.vis_dir)

    # Model Constants
    model_const = Constants()
    model_const.model_num = 60000
    
    model_const.geometric_pairwise = False
    if model_const.geometric_pairwise:
        model_const.geometric_factor = GeometricFactorPairwiseConstants()
    else:
        model_const.geometric_factor = GeometricFactorConstants() 
    
    model_const.geometric_per_hoi = False
    if model_const.geometric_per_hoi:
        model_const.geometric_factor.out_dim = 600

    model_const.geometric_factor.model_pth = os.path.join(
        exp_const.model_dir,
        f'geometric_factor_{model_const.model_num}')

    # Create Model
    model = Model()
    model.const = model_const
    if model.const.geometric_pairwise:
        model.geometric_factor = \
            GeometricFactorPairwise(model.const.geometric_factor).cuda()
    else:
        model.geometric_factor = \
            GeometricFactor(model.const.geometric_factor).cuda()
    model.geometric_factor.load_state_dict(torch.load(
        model.const.geometric_factor.model_pth))

    weight = model.geometric_factor.box_feature_factor.layers[0][0].weight # 117 x 24

    feature_names = [
        'offset_x',
        'offset_y',
        'offset_x_norm',
        'offset_y_norm',
        'log_area_ratio',
        'iou',
        'log_aspect_ratio_human',
        'log_aspect_ratio_object',
        'log_area_ratio_human',
        'log_area_ratio_object',
        'human_center_x',
        'human_center_y',
        'object_center_x',
        'object_center_y',
        'human_center_x_norm',
        'human_center_y_norm',
        'object_center_x_norm',
        'object_center_y_norm',
        'log_w_human',
        'log_h_human',
        'log_w_object',
        'log_h_object',
        'log_w_im',
        'log_h_im',
    ]

    relation_names = []
    data_const = HicoConstants()
    verb_list = io.load_json_object(data_const.verb_list_json)
    verb_dict = {verb['id']: verb['name'] for verb in verb_list}
    for i in range(len(verb_list)):
        verb_id = str(i+1).zfill(3)
        relation_names.append(verb_dict[verb_id])

    trace = go.Heatmap(
        z=weight.data.cpu().numpy(),
        x=feature_names,
        y=relation_names)
    data=[trace]
    filename = os.path.join(exp_const.vis_dir,'weights.html')
    plotly.offline.plot(
        {'data': data},
        filename=filename,
        auto_open=False)


if __name__=='__main__':
    main()