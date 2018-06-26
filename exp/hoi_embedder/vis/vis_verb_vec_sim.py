import numpy as np
import torch
import os
import plotly
import plotly.graph_objs as go
from scipy.spatial.distance import pdist, squareform

from utils.html_writer import HtmlWriter
import utils.io as io
from utils.model import Model
from exp.hoi_embedder.models.hoi_classifier_model import HoiClassifier


def compute_verb_vec_sim(verb_vecs,sim_type='cosine'):
    if sim_type=='cosine':
        verb_vec_sim = 1-squareform(pdist(verb_vecs,'cosine'))
    else:
        assert(False), 'sim_type not supported'
    return verb_vec_sim

def find_most_sim_verbs(verb_vec_sim,verbs):
    most_sim_verbs = {}
    for i,verb in enumerate(verbs):
        idxs = np.argsort(verb_vec_sim[i])[::-1].tolist()
        most_sim_verbs[verb] = [verbs[j] for j in idxs]
    return most_sim_verbs
    
def main(exp_const,data_const,model_const):
    print('Loading model ...')
    model = Model()
    model.const = model_const
    model.hoi_classifier = HoiClassifier(model.const.hoi_classifier).cuda()
    if model.const.model_num == -1:
        print('No pretrained model will be loaded since model_num is set to -1')
    else:
        model.hoi_classifier.load_state_dict(
            torch.load(model.const.hoi_classifier.model_pth))

    verb_vecs = model.hoi_classifier.verb_vecs.cpu().data.numpy()

    verb_list = io.load_json_object(data_const.verb_list_json)
    verbs = [None]*len(verb_list)
    for i,verb in enumerate(verb_list):
        idx = int(verb['id'])-1
        verbs[idx] = verb['name']
    
    verb_vec_sim = compute_verb_vec_sim(verb_vecs)
    most_sim_verbs = find_most_sim_verbs(verb_vec_sim,verbs)

    trace = go.Heatmap(
        z = verb_vec_sim[::-1],
        x = verbs,
        y = verbs[::-1],
        showscale=True)

    io.mkdir_if_not_exists(exp_const.vis_dir)

    filename = os.path.join(exp_const.vis_dir,'verb_vec_sim.html')
    plotly.offline.plot(
        {'data': [trace]},
        filename=filename,
        auto_open=False)

    html_filename = os.path.join(exp_const.vis_dir,'most_sim_verbs.html')
    html_writer = HtmlWriter(html_filename)
    col_dict = {
        0: 'Query Verb',
        1: 'Nearest Neighbors'
    }
    html_writer.add_element(col_dict)
    for verb,sim_verbs in most_sim_verbs.items():
        col_dict = {
            0: verb,
            1: sim_verbs[1:11]
        }
        html_writer.add_element(col_dict)
    