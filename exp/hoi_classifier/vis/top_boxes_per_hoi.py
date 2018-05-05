import os
import h5py
import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
tqdm.monitor_interval = 0
from torch.autograd import Variable
import skimage.io as skio
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import utils.io as io
from utils.pytorch_layers import get_activation
from utils.html_writer import HtmlWriter
from utils.bbox_utils import vis_sub_obj_bboxes
from utils.model import Model
from exp.hoi_classifier.models.hoi_classifier_model import HoiClassifier
from exp.hoi_classifier.data.features_dataset import Features


def select_best_boxes_per_hoi_per_image(
        human_box, 
        object_box, 
        hoi_prob):
    num_hois = hoi_prob.shape[1]  # hoi_prob is num_boxes x num_hois
    box_ids = np.argmax(hoi_prob,0) # (600,) if num_hois=600
    box_hoi_prob = hoi_prob[box_ids,np.arange(num_hois)]
    per_hoi_boxes = np.concatenate((
        human_box[box_ids,:],   # (600,4)
        object_box[box_ids,:],  # (600,4)
        box_hoi_prob[:,np.newaxis], # (600,1)
    ),1)
    return per_hoi_boxes   # 600 x 9


def select_best_boxes_across_dataset_using_box_score_only(
        model,
        dataset,
        exp_const):
    model.hoi_classifier.eval()
    sampler = SequentialSampler(dataset)
    num_samples = len(sampler)
    num_hois = len(model.hoi_classifier.scatter_verbs_to_hois.hoi_dict)
    best_boxes_scores = np.zeros([num_samples,num_hois,9])
    global_ids = [None]*num_samples
    for i, sample_id in enumerate(tqdm(sampler)):
        if i==num_samples:
            break
        data = dataset[sample_id]
        feats = {
            'human_rcnn': Variable(torch.cuda.FloatTensor(data['human_feat'])),
            'object_rcnn': Variable(torch.cuda.FloatTensor(data['object_feat'])),
            'box': Variable(torch.cuda.FloatTensor(data['box_feat'])),
            'human_prob_vec': Variable(torch.cuda.FloatTensor(data['human_prob_vec'])),
            'object_prob_vec': Variable(torch.cuda.FloatTensor(data['object_prob_vec'])),
            'object_one_hot': Variable(torch.cuda.FloatTensor(data['object_one_hot'])),
            'prob_mask': Variable(torch.cuda.FloatTensor(data['prob_mask']))
        }       
        
        prob_vec, factor_scores = model.hoi_classifier(feats)
        
        hoi_prob = prob_vec['hoi']
        hoi_prob = hoi_prob.data.cpu().numpy()

        per_hoi_boxes_scores = select_best_boxes_per_hoi_per_image(
            data['human_box'],
            data['object_box'],
            hoi_prob)
        best_boxes_scores[i] = per_hoi_boxes_scores
        global_ids[i] = data['global_id']

    global_ids = np.array(global_ids)
    top_boxes = {}
    for hoi_id, hoi in model.hoi_classifier.scatter_verbs_to_hois.hoi_dict.items():
        hoi_idx = int(hoi_id)-1
        hoi_name = hoi_id + '_' + hoi['object'] + '_' + hoi['verb']
        hoi_scores = best_boxes_scores[:,hoi_idx,-1]
        ids = np.argsort(hoi_scores)[::-1][:20]
        top_boxes[hoi_name] = {
            'boxes_scores': best_boxes_scores[ids,hoi_idx],
            'global_ids': global_ids[ids].tolist()
        }

    return top_boxes


def create_html(top_boxes,anno_dict,img_dir,vis_dir):
    for relation in top_boxes.keys():
        relation_vis_dir = os.path.join(vis_dir,relation)
        io.mkdir_if_not_exists(relation_vis_dir,recursive=True)
        html_filename = os.path.join(relation_vis_dir,'index.html')
        html_writer = HtmlWriter(html_filename)
        boxes_scores = top_boxes[relation]['boxes_scores']
        for i, global_id in enumerate(top_boxes[relation]['global_ids']):
            anno = anno_dict[global_id]
            img_path = os.path.join(img_dir,anno['image_path_postfix'])
            img = skio.imread(img_path)
            if len(img.shape)==2:
                img = img[:,:,np.newaxis]
                img = np.tile(img,(1,1,3))

            out_img = vis_sub_obj_bboxes(
                [boxes_scores[i,:4]],
                [boxes_scores[i,4:8]],
                img,
                modify=True)
            out_img_path = os.path.join(
                relation_vis_dir,
                str(i).zfill(3)+'.png')
            skio.imsave(out_img_path,out_img)
            col_dict = {
                0: global_id,
                1: html_writer.image_tag(str(i).zfill(3)+'.png'),
                2: round(boxes_scores[i,8],4)
            }
            html_writer.add_element(col_dict)
        
        html_writer.close()


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

    print('Creating data loader ...')
    dataset = Features(data_const)

    print('Selecting top box configurations for each relation ...')
    top_boxes = select_best_boxes_across_dataset_using_box_score_only(
        model,
        dataset,
        exp_const)
    
    print('Reading anno_list.json ...')
    anno_list = io.load_json_object(data_const.anno_list_json)
    anno_dict = {anno['global_id']:anno for anno in anno_list}

    print('Creating visualization images ...')
    vis_dir = os.path.join(exp_const.exp_dir,'vis/top_boxes_per_hoi')
    create_html(
        top_boxes,
        anno_dict,
        data_const.images_dir,
        vis_dir)


    
