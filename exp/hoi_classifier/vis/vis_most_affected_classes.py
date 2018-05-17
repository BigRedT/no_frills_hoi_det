import os
import plotly
import plotly.graph_objs as go
import numpy as np
import argparse

import utils.io as io
from data.hico.hico_constants import HicoConstants
from utils.argparse_utils import manage_required_args

parser = argparse.ArgumentParser()
parser.add_argument(
    '--ap_json',
    type=str,
    help='Path to ap json file of model')
parser.add_argument(
    '--ap_baseline_json',
    type=str,
    help='Path to ap json file of baseline model')
parser.add_argument(
    '--outdir',
    type=str,
    help='Output directory')


def compute_change(APs,APs_baseline):
    deltas = []
    for hoi_id in APs.keys():
        ap = APs[hoi_id]
        ap_baseline = APs_baseline[hoi_id]
        delta = (ap - ap_baseline)*100 #/ap_baseline
        deltas.append([hoi_id,delta,ap,ap_baseline])

    sorted_deltas = sorted(deltas,key=lambda x:x[1],reverse=True)
    return sorted_deltas

def main():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['ap_json','ap_baseline_json'],
        optional_args=[],
        exit_if_unspecified=True)

    data_const = HicoConstants()
    APs = io.load_json_object(args.ap_json)['AP']
    APs_baseline = io.load_json_object(args.ap_baseline_json)['AP']

    hoi_cls_count = io.load_json_object(data_const.hoi_cls_count_json)
    hoi_list = io.load_json_object(data_const.hoi_list_json)
    hoi_dict = {hoi['id']: hoi for hoi in hoi_list}
    
    deltas = compute_change(APs,APs_baseline)
    for delta in deltas:
        hoi_id = delta[0]
        delta.append(hoi_cls_count[hoi_id])
    
    plot_dict_pos = {
        'classes': [],
        'deltas': [],
        'color': [],
    }

    count = 0
    for delta in deltas:
        if count >= 20:
            break

        if delta[-1] > 50:
            count += 1
            print(delta)
            hoi_id = delta[0]
            hoi_name = hoi_dict[hoi_id]['verb'] + ' ' + hoi_dict[hoi_id]['object']
            hoi_name = ' '.join(hoi_name.split('_'))
            plot_dict_pos['classes'].append(hoi_name)
            plot_dict_pos['deltas'].append(delta[1])
            if delta[1] > 0:
                plot_dict_pos['color'].append('rgba(200,200,200,1)')
            else:
                plot_dict_pos['color'].append('rgba(222,45,38,0.8)')

    plot_dict_neg = {
        'classes': [],
        'deltas': [],
        'color': [],
    }

    count = 0
    for delta in deltas[::-1]:
        if count >= 20:
            break
            
        if delta[-1] > 50:
            count += 1
            print(delta)
            hoi_id = delta[0]
            hoi_name = hoi_dict[hoi_id]['verb'] + ' ' + hoi_dict[hoi_id]['object']
            hoi_name = ' '.join(hoi_name.split('_'))
            plot_dict_neg['classes'].append(hoi_name)
            plot_dict_neg['deltas'].append(delta[1])
            if delta[1] > 0:
                plot_dict_neg['color'].append('rgba(200,200,200,1)')
            else:
                plot_dict_neg['color'].append('rgba(222,45,38,0.8)')

    #print(plot_dict)
    data = [
        go.Bar(
            x = plot_dict_pos['classes'] + plot_dict_neg['classes'][::-1],
            y = plot_dict_pos['deltas'] + plot_dict_neg['deltas'][::-1],
            # text = plot_dict_pos['classes'] + plot_dict_neg['classes'][::-1],
            # textposition = 'top',
            marker = dict(
                color=plot_dict_pos['color']+plot_dict_neg['color'][::-1])
        )
    ]
    layout = go.Layout(
        yaxis=dict(
            title='Change in AP',
            range=[-25,60],
        ),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(
                size=12,
            ),
        ),
        margin=go.Margin(
            l=100,
            r=100,
            b=150,
            t=50,
        ),
        height=400,
        width=800,
    )

    io.mkdir_if_not_exists(args.outdir)
    filename = os.path.join(args.outdir,'most_affected_classes.html')
    plotly.offline.plot(
        {'data': data,'layout':layout},
        filename=filename,
        auto_open=False)


if __name__=='__main__':
    main()
    