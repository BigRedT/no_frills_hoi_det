import os
import plotly
import plotly.graph_objs as go
import numpy as np

from exp.hoi_classifier.vis.faster_rcnn_aps import COCO_CLS_TO_FASTER_RCNN_AP
import utils.io as io
from data.hico.hico_constants import HicoConstants

def main():
    exp_name = 'factors_rcnn_det_prob_appearance_boxes_and_object_label_human_pose'
    exp_dir = os.path.join(
        os.getcwd(),
        f'data_symlinks/hico_exp/hoi_classifier/{exp_name}')
    
    map_json = os.path.join(
        exp_dir,
        'mAP_eval/test_30000/mAP.json')
    
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

    per_verb_hoi_aps = sorted(per_verb_hoi_aps,key=lambda x: np.median(x[1]))

    N = len(per_verb_hoi_aps)
    c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]
    data = []
    for i, (verb,aps) in enumerate(per_verb_hoi_aps): 
        trace = go.Box(
            y=aps, 
            name=" ".join(verb.split("_")),
            boxpoints=False, #"outliers"
            marker={'color': c[i]},
            line={'width':1}
        )
        data.append(trace)

    layout = go.Layout(
        yaxis=dict(
            title='AP of HOI Categories',
            range=[0,100],
        ),
        xaxis=dict(
            title='Interactions',
            tickangle=45,
            tickfont=dict(
                size=8,
            ),
        ),
        height=500,
        margin=go.Margin(
            l=100,
            r=100,
            b=150,
            t=50,
        ),
    )

    filename = os.path.join(exp_dir,'vis/obj_aps_per_interaction.html')
    plotly.offline.plot(
        {'data': data, 'layout': layout},
        filename=filename,
        auto_open=False)


if __name__=='__main__':
    main()