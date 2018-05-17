import os
import plotly
import plotly.graph_objs as go
import numpy as np
import argparse

import utils.io as io
from data.hico.hico_constants import HicoConstants
from utils.argparse_utils import manage_required_args


def select_verbs():
    exp_name = 'factors_rcnn_det_prob_appearance_boxes_and_object_label_human_pose'
    exp_dir = os.path.join(
        os.getcwd(),
        f'data_symlinks/hico_exp/hoi_classifier/{exp_name}')
    map_json = os.path.join(exp_dir,'mAP_eval/test_25000/mAP.json')
    AP = io.load_json_object(map_json)['AP']

def main():
    #exp_name = 'factors_rcnn_det_prob_appearance_boxes_and_object_label_human_pose'
    exp_name = 'factors_rcnn_det_prob_boxes_and_object_label_human_pose'
    exp_dir = os.path.join(
        os.getcwd(),
        f'data_symlinks/hico_exp/hoi_classifier/{exp_name}')
    vis_dir = os.path.join(exp_dir,'vis')
    
    map_exp_name = 'factors_rcnn_det_prob_appearance_boxes_and_object_label_human_pose'
    map_exp_dir = os.path.join(
        os.getcwd(),
        f'data_symlinks/hico_exp/hoi_classifier/{map_exp_name}')
    map_json = os.path.join(map_exp_dir,'mAP_eval/test_25000/mAP.json')
    hoi_aps = io.load_json_object(map_json)['AP']
    
    data_const = HicoConstants()
    hoi_list = io.load_json_object(data_const.hoi_list_json)
    
    verb_to_hoi_id = {}
    for hoi in hoi_list:
        hoi_id = hoi['id']
        verb = hoi['verb']
        if verb not in verb_to_hoi_id:
            verb_to_hoi_id[verb] = []   
        verb_to_hoi_id[verb].append(hoi_id)

    per_verb_hoi_aps = []
    for verb, hoi_ids in verb_to_hoi_id.items():
        verb_obj_aps = []
        for hoi_id in hoi_ids:
            verb_obj_aps.append(hoi_aps[hoi_id]*100)

        per_verb_hoi_aps.append((verb,verb_obj_aps))

    per_verb_hoi_aps = sorted(per_verb_hoi_aps,key=lambda x: np.median(x[1]),reverse=True)

    verb_list = io.load_json_object(data_const.verb_list_json)
    verb_name_to_idx = {verb['name']: int(verb['id'])-1 for verb in verb_list}
    idx = [verb_name_to_idx[per_verb_hoi_aps[i][0]] for i in range(30)]
    
    verb_labels = [None]*len(verb_list)
    for verb in verb_list:
        i = int(verb['id'])-1
        verb_labels[i] = verb['name']
    selected_verb_labels = [verb_labels[i] for i in idx]
    
    conf_mat_npy = os.path.join(
        exp_dir,
        f'verb_conf_mat.npy')
    conf_mat = np.load(conf_mat_npy)[idx,:][:,idx]
    # conf_mat_mean = np.mean(conf_mat,0,keepdims=True)
    # conf_mat_std = np.std(conf_mat,0,keepdims=True)
    # conf_mat = (conf_mat - conf_mat_mean) / (conf_mat_std + 1e-6)
    # trace = go.Heatmap(
    #     z=np.exp(0.3*conf_mat)[::-1],
    #     x = selected_verb_labels,
    #     y = selected_verb_labels[::-1])
    conf_mat_min = np.min(conf_mat,0,keepdims=True)
    conf_mat_max = np.max(conf_mat,0,keepdims=True)
    conf_mat_std = conf_mat_max - conf_mat_min
    conf_mat = (conf_mat - conf_mat_max) / (conf_mat_std)
    trace = go.Heatmap(
        z = np.transpose(np.exp(5*conf_mat))[::-1],
        x = selected_verb_labels,
        y = selected_verb_labels[::-1],
        showscale=True,)

    layout = go.Layout(
        yaxis=dict(
            #title='HOI Class',
            tickfont=dict(
                size=12,
            ),
        ),
        xaxis=dict(
            #title='HOI Class',
            tickangle=-45,
            side='top',
            tickfont=dict(
                size=12,
            ),
        ),
        height=800,
        width=800,
        autosize=False,
        showlegend=False,
        margin=go.Margin(
            l=100,
            r=100,
            b=150,
            t=150,
        ),
    )
    filename = os.path.join(vis_dir,'verb_conf_mat.html')
    plotly.offline.plot(
        {'data': [trace],'layout': layout},
        filename=filename,
        auto_open=False)


if __name__=='__main__':
    main()