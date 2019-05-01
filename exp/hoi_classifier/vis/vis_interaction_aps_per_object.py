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
    
    obj_to_hoi_id = {}
    for hoi in hoi_list:
        hoi_id = hoi['id']
        obj = hoi['object']
        if obj not in obj_to_hoi_id:
            obj_to_hoi_id[obj] = []   
        obj_to_hoi_id[obj].append(hoi_id)

    obj_aps = []
    for obj in obj_to_hoi_id.keys():
        obj_aps.append((obj,COCO_CLS_TO_FASTER_RCNN_AP[obj]))

    obj_aps = sorted(obj_aps,key=lambda x:x[1])

    per_obj_hoi_aps = []
    for obj, obj_ap in obj_aps:
        obj_interaction_aps = []
        for hoi_id in obj_to_hoi_id[obj]:
            obj_interaction_aps.append(hoi_aps[hoi_id]*100)

        per_obj_hoi_aps.append((obj,obj_interaction_aps))

    per_obj_hoi_aps = sorted(per_obj_hoi_aps,key=lambda x: np.median(x[1]))

    N = len(per_obj_hoi_aps)
    c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]
    data = []
    obj_aps_x = []
    obj_aps_y = []
    for i, (obj,aps) in enumerate(per_obj_hoi_aps): 
        trace = go.Box(
            y=aps, 
            name=" ".join(obj.split("_")),
            boxpoints=False, #"outliers"
            marker={'color': c[i]},
            line={'width':1}
        )
        data.append(trace)
        obj_aps_x.append(" ".join(obj.split("_")))
        obj_aps_y.append(COCO_CLS_TO_FASTER_RCNN_AP[obj]*100)

    line_char_trace = go.Scatter(
        x = obj_aps_x,
        y = obj_aps_y,
        mode = 'lines+markers',
        line = dict(
            color = ('rgba(150, 150, 200, 1)'),
            width = 1,),
            #dash = 'dash'),
        marker=dict(size=4))
    #data.append(line_char_trace)

    layout = go.Layout(
        yaxis=dict(
            title='AP of HOI Categories',
            range=[0,100],
        ),
        xaxis=dict(
            title='Objects',
            tickangle=45,
            tickfont=dict(
                size=12,
            ),
        ),
        height=500,
        margin=go.Margin(
            l=100,
            r=100,
            b=150,
            t=50,
        )
    )

    filename = os.path.join(exp_dir,'vis/interaction_aps_per_object.html')
    plotly.offline.plot(
        {'data': data, 'layout': layout},
        filename=filename,
        auto_open=False)

    corr_x = []
    corr_y = []
    for i, (obj,aps) in enumerate(per_obj_hoi_aps):
        obj_ap = COCO_CLS_TO_FASTER_RCNN_AP[obj]*100
        for hoi_ap in aps:
            corr_x.append(obj_ap)
            corr_y.append(hoi_ap)
    
    corr_trace = go.Scatter(
        x = corr_x,
        y = corr_y,
        mode = 'markers',
        marker = dict(
            size = 8,
            color = 'rgba(255, 182, 193, .8)',
            line = dict(
                width = 2,
                color = 'rgba(100, 0, 0, 1)'
            )
        )
    )

    corr_layout = go.Layout(
        yaxis=dict(
            title='AP of HOI Categories',
            range=[0,100],
        ),
        xaxis=dict(
            title='AP of Object Categories',
            range=[0,100],
        ),
        height=800,
        width=800,
        margin=go.Margin(
            l=100,
            r=100,
            b=150,
            t=50,
        )
    )

    filename = os.path.join(exp_dir,'vis/hoi_ap_vs_obj_ap.html')
    plotly.offline.plot(
        {'data': [corr_trace], 'layout': corr_layout},
        filename=filename,
        auto_open=False)


if __name__=='__main__':
    main()