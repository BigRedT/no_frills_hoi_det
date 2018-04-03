import os
import h5py
import heapq
import argparse
from tqdm import tqdm

import utils.io as io

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
    '--proc_dir',
    type=str,
    default=None,
    required=True,
    help='Path to HICO processed data directory')
parser.add_argument(
    '--subset',
    type=str,
    default='test',
    choices=['train','test','val','train_val'],
    help='Subset of data to run the evaluation on')
parser.add_argument(
    '--num_processes',
    type=int,
    default=12,
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


def write_to_html()
def vis_topk_dets(k,hoi_id,pred_dets_hdf5,global_ids):
    pred_dets = h5py.File(pred_dets_hdf5,'r')
    topk_det_selector = TopKDetSelector(k)
    for global_id in tqdm(global_ids):
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
    


def main():
    args = parser.parse_args()

     # Load subset ids to eval on
    split_ids_json = os.path.join(args.proc_dir,'split_ids.json')
    split_ids = io.load_json_object(split_ids_json)
    global_ids = split_ids[args.subset]

    vis_topk_dets(100,'001',args.pred_hoi_dets_hdf5,global_ids)


if __name__=='__main__':
    main()