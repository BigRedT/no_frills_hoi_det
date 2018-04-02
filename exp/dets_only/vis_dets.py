import os
import numpy as np
import skimage.io as skio
from tqdm import tqdm

import utils.bbox_utils as bbox_utils
import utils.io as io
from data.hico.hico_constants import HicoConstants
from utils.constants import ExpConstants
from utils.html_writer import HtmlWriter


def load_gt_dets(proc_dir,global_ids_set):
    # Load anno_list
    print('Loading anno_list.json ...')
    anno_list_json = os.path.join(proc_dir,'anno_list.json')
    anno_list = io.load_json_object(anno_list_json)

    gt_dets = {}
    for anno in anno_list:
        if anno['global_id'] not in global_ids_set:
            continue

        global_id = anno['global_id']
        gt_dets[global_id] = {}
        for hoi in anno['hois']:
            hoi_id = hoi['id']
            gt_dets[global_id][hoi_id] = []
            for human_box_num, object_box_num in hoi['connections']:
                human_box = hoi['human_bboxes'][human_box_num]
                object_box = hoi['object_bboxes'][object_box_num]
                det = {
                    'human_box': human_box,
                    'object_box': object_box,
                }
                gt_dets[global_id][hoi_id].append(det)

    return gt_dets


def main():
    exp_const = ExpConstants(
        exp_name='dets_only',
        out_base_dir='/home/tanmay/Data/weakly_supervised_hoi_exp/')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis/pred_dets')
    io.mkdir_if_not_exists(exp_const.vis_dir,recursive=True)

    data_const = HicoConstants()
    split_ids = io.load_json_object(data_const.split_ids_json)
    annos = io.load_json_object(data_const.anno_list_json)
    annos = {anno['global_id']:anno for anno in annos}

    html_filename = os.path.join(exp_const.vis_dir,'pred_dets.html')
    html_writer = HtmlWriter(html_filename)

    gt_dets = load_gt_dets(data_const.proc_dir,set(split_ids['test'][0:10]))
    pred_hoi_dets_dir = os.path.join(exp_const.exp_dir,'pred_hoi_dets')
    col_dict = {
        0: 'GT Boxes',
        1: 'Human Boxes',
        2: 'Object Boxes',
        3: 'Human + Object Boxes'
    }
    html_writer.add_element(col_dict)
    for global_id in tqdm(split_ids['test'][0:10]):
        src_img_path = os.path.join(
            data_const.images_dir,
            annos[global_id]['image_path_postfix'])
        img = skio.imread(src_img_path)
        if len(img.shape)==2:
            img = img[...,np.newaxis]
            img = np.tile(img,[1,1,3])
        
        tgt_img_path = os.path.join(exp_const.vis_dir,f'{global_id}.jpg')
        #skio.imsave(tgt_img_path,img)
        try:
            os.symlink(src_img_path,tgt_img_path)
        except FileExistsError:
            pass

        # Read dets
        pred_dets_npy = os.path.join(pred_hoi_dets_dir,f'{global_id}_pred_hoi_dets.npy')
        pred_dets = np.load(pred_dets_npy)[()]
        sub_boxes = []
        obj_boxes = []
        for hoi_id, dets in pred_dets.items():
            for det in dets:
                if det['score'] > 0.8:
                    sub_boxes.append(det['human_box'])
                    obj_boxes.append(det['object_box'])

        img_box = bbox_utils.vis_sub_obj_bboxes(sub_boxes,obj_boxes,img)
        img_box_jpg = os.path.join(exp_const.vis_dir,f'{global_id}_boxes.jpg')
        skio.imsave(img_box_jpg,img_box)

        img_box = bbox_utils.vis_bboxes(sub_boxes,img,(0,0,255))
        img_box_jpg = os.path.join(exp_const.vis_dir,f'{global_id}_human_boxes.jpg')
        skio.imsave(img_box_jpg,img_box)

        img_box = bbox_utils.vis_bboxes(obj_boxes,img,(0,255,0))
        img_box_jpg = os.path.join(exp_const.vis_dir,f'{global_id}_object_boxes.jpg')
        skio.imsave(img_box_jpg,img_box)

        # GT dets
        sub_boxes = []
        obj_boxes = []
        for hoi_id, dets in gt_dets[global_id].items():
            for det in dets:
                sub_boxes.append(det['human_box'])
                obj_boxes.append(det['object_box'])

        img_box = bbox_utils.vis_sub_obj_bboxes(sub_boxes,obj_boxes,img)
        img_box_jpg = os.path.join(exp_const.vis_dir,f'{global_id}_gt_boxes.jpg')
        skio.imsave(img_box_jpg,img_box)

        col_dict = {
            0: html_writer.image_tag(f'{global_id}_gt_boxes.jpg'),
            1: html_writer.image_tag(f'{global_id}_human_boxes.jpg'),
            2: html_writer.image_tag(f'{global_id}_object_boxes.jpg'),
            3: html_writer.image_tag(f'{global_id}_boxes.jpg'),
        }
        html_writer.add_element(col_dict)

    html_writer.close()

if __name__=='__main__':
    main()
