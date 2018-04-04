import os
import h5py
import heapq
import argparse
from tqdm import tqdm
from copy import deepcopy
import skimage.io as skio
from multiprocessing import Pool

import utils.io as io
from utils.bbox_utils import vis_sub_obj_bboxes
from utils.html_writer import HtmlWriter
from data.hico.hico_constants import HicoConstants


parser = argparse.ArgumentParser()
parser.add_argument(
    '--pred_hoi_dets_hdf5', 
    type=str, 
    default=None,
    required=True,
    help='Path to predicted hoi detections hdf5 file')
parser.add_argument(
    '--out_dir', 
    type=str, 
    default=None,
    required=True,
    help='Output directory')
parser.add_argument(
    '--subset',
    type=str,
    default='test',
    choices=['train','test','val','train_val'],
    help='Subset of data to run the evaluation on')
parser.add_argument(
    '--num_processes',
    type=int,
    default=10,
    help='Number of processes to parallelize across')


class TopKDetSelector():
    def __init__(self,k):
        self.k = k
        self.min_heap = []
    
    def update(self,det):
        if len(self.min_heap) <= self.k:
            update_fn = heapq.heappush
        else:
            update_fn = heapq.heappushpop

        update_fn(self.min_heap,(det['score'],det))

    def topk(self):
        top_score_dets = sorted(self.min_heap,key=lambda x:x[0],reverse=True)
        return [det for score,det in top_score_dets]


def write_to_html(hoi_id,dets,html_filename,global_id_to_img_path):
    vis_dir = os.path.dirname(html_filename)
    html_writer = HtmlWriter(html_filename)
    col_dict = {
        0: 'global_id',
        1: 'score',
        2: 'hoi'
    }
    for i, det in enumerate(dets):
        src_img_path = global_id_to_img_path[det['global_id']]
        img = skio.imread(src_img_path)
        img = vis_sub_obj_bboxes(
            [det['human_box']],
            [det['object_box']],
            img,
            modify=True)
        tgt_img_name = str(i).zfill(5)+'.png'
        tgt_img_path = os.path.join(vis_dir,tgt_img_name)
        skio.imsave(tgt_img_path,img)
        col_dict = {
            0: det['global_id'],
            1: det['score'],
            2: html_writer.image_tag(tgt_img_name)}
        html_writer.add_element(col_dict)
        
    html_writer.close()


def vis_topk_dets(
        k,
        hoi_id,
        pred_dets_hdf5,
        global_ids,
        global_id_to_img_path,
        hoi_dict,
        out_dir):
    print(f'Visualizing hoi_id {hoi_id} ...')
    hoi_name = hoi_id+'_'+hoi_dict[hoi_id]['verb']+'_'+hoi_dict[hoi_id]['object']
    vis_dir = os.path.join(out_dir,hoi_name)
    io.mkdir_if_not_exists(vis_dir,recursive=True)

    pred_dets = h5py.File(pred_dets_hdf5,'r')

    topk_det_selector = TopKDetSelector(k)
    for global_id in global_ids:
        start_id,end_id = pred_dets[global_id]['start_end_ids'][int(hoi_id)-1]
        dets = pred_dets[global_id]['human_obj_boxes_scores'][start_id:end_id]
        for i in range(dets.shape[0]):
            det = {
                'global_id': global_id,
                'human_box': dets[i,:4],
                'object_box': dets[i,4:8],
                'score': dets[i,8]
            }
            topk_det_selector.update(det)

    pred_dets.close()

    topk_dets = topk_det_selector.topk()
    html_filename = os.path.join(vis_dir,'pred_hoi.html')
    write_to_html(hoi_id,topk_dets,html_filename,global_id_to_img_path)


def main():
    args = parser.parse_args()

    data_const = HicoConstants()

     # Load subset ids to eval on
    split_ids = io.load_json_object(data_const.split_ids_json)
    global_ids = split_ids[args.subset]

    # Load anno_list.json
    anno_list = io.load_json_object(data_const.anno_list_json)
    global_id_to_img_path = {}
    for anno in anno_list:
        global_id_to_img_path[anno['global_id']] = os.path.join(
            data_const.images_dir,
            anno['image_path_postfix'])

    # Load hoi_list.json
    hoi_list = io.load_json_object(data_const.hoi_list_json)
    hoi_dict = {hoi['id']:hoi for hoi in hoi_list}

    print(f'Starting a pool of {args.num_processes} workers ...')
    p = Pool(args.num_processes)

    inputs = []
    for hoi_id in hoi_dict.keys():
        inputs.append(
            (
                50,
                hoi_id,
                args.pred_hoi_dets_hdf5,
                deepcopy(global_ids),
                deepcopy(global_id_to_img_path),
                deepcopy(hoi_dict),
                args.out_dir,
            ))

    p.starmap(vis_topk_dets,inputs)

    # vis_topk_dets(
    #     100,
    #     '001',
    #     args.pred_hoi_dets_hdf5,
    #     global_ids,
    #     global_id_to_img_path,
    #     hoi_dict,
    #     args.out_dir)


if __name__=='__main__':
    main()