import os
import h5py
import numpy as np
from tqdm import tqdm

import utils.io as io
from data.hico.hico_constants import HicoConstants


def main():
    data_const = HicoConstants()
    anno_list = io.load_json_object(data_const.anno_list_json)
    global_ids = [anno['global_id'] for anno in anno_list]
    feats_hdf5 = os.path.join(data_const.proc_dir,'faster_rcnn_fc7.hdf5')
    feats = h5py.File(feats_hdf5,'w') 
    for global_id in tqdm(global_ids):
        fc7_npy = os.path.join(
            data_const.faster_rcnn_boxes,
            f'{global_id}_fc7.npy')
        fc7 = np.load(fc7_npy)
        feats.create_dataset(global_id,data=fc7)
        
    feats.close()

if __name__=='__main__':
    main()