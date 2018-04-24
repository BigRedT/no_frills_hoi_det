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
from utils.html_writer import HtmlWriter
from utils.bbox_utils import vis_sub_obj_bboxes
from utils.model import Model
from exp.relation_classifier.models.relation_classifier_model import \
    BoxAwareRelationClassifier
from exp.relation_classifier.models.gather_relation_model import \
    GatherRelation
from exp.relation_classifier.data.features_balanced import FeaturesBalanced


def select_best_boxes_per_relation_per_image(
        human_box, 
        object_box, 
        relation_prob):
    num_relations = relation_prob.shape[1]  # relation_prob is num_boxes x num_relations
    box_ids = np.argmax(relation_prob,0) # (117,) if num_relations=117
    box_relation_prob = relation_prob[box_ids,np.arange(num_relations)]
    per_relation_boxes = np.concatenate((
        human_box[box_ids,:],   # (117,4)
        object_box[box_ids,:],  # (117,4)
        box_relation_prob[:,np.newaxis], # (117,1)
    ),1)
    return per_relation_boxes   # 117 x 9


def select_best_boxes_across_dataset_using_box_score_only(
        model,
        dataset,
        exp_const):
    model.relation_classifier.eval()
    model.gather_relation.eval()
    sampler = SequentialSampler(dataset)
    num_samples = len(sampler)
    num_relations = len(model.gather_relation.relation_to_id)
    best_boxes_scores = np.zeros([num_samples,num_relations,9])
    global_ids = [None]*num_samples
    for i, sample_id in enumerate(tqdm(sampler)):
        if i==num_samples:
            break
        data = dataset[sample_id]
        feats = {}
        feats['box'] = Variable(torch.cuda.FloatTensor(data['box_feat']))
        relation_prob,_ = \
            model.relation_classifier.forward_box_feature_factor(feats)
        per_relation_boxes_scores = select_best_boxes_per_relation_per_image(
            data['human_box'],
            data['object_box'],
            relation_prob.data.cpu().numpy())
        best_boxes_scores[i] = per_relation_boxes_scores
        global_ids[i] = data['global_id']

    global_ids = np.array(global_ids)
    top_boxes = {}
    for relation, relation_id in model.gather_relation.relation_to_id.items():
        relation_idx = int(relation_id)-1
        relation_scores = best_boxes_scores[:,relation_idx,-1]
        ids = np.argsort(relation_scores)[::-1][:10]
        top_boxes[relation] = {
            'boxes_scores': best_boxes_scores[ids,relation_idx],
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
    model.relation_classifier = \
        BoxAwareRelationClassifier(model_const.relation_classifier).cuda()
    model.relation_classifier.load_state_dict(torch.load(
        model_const.relation_classifier.model_pth))
    model.gather_relation = GatherRelation(model_const.gather_relation).cuda()
    
    print('Creating data loader ...')
    dataset = FeaturesBalanced(data_const)

    print('Selecting top box configurations for each relation ...')
    top_boxes = select_best_boxes_across_dataset_using_box_score_only(
        model,
        dataset,
        exp_const)
    
    print('Reading anno_list.json ...')
    anno_list = io.load_json_object(data_const.anno_list_json)
    anno_dict = {anno['global_id']:anno for anno in anno_list}

    print('Creating visualization images ...')
    vis_dir = os.path.join(exp_const.exp_dir,'vis/top_boxes_per_relation')
    create_html(
        top_boxes,
        anno_dict,
        data_const.images_dir,
        vis_dir)


    
