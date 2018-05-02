import os
from tqdm import tqdm

import utils.io as io
from data.hico.hico_constants import HicoConstants


def bin_hoi_ids(hoi_cls_count,upper_limits):
    bins = {str(ul): [] for ul in upper_limits}
    for hoi_id,count in hoi_cls_count.items():
        ll = 0
        for ul in upper_limits:
            if count >= ll and count < ul:
                bins[str(ul)].append(hoi_id)
                break
            else:
                ll = ul
    return bins


def main():
    data_const = HicoConstants()
    anno_list = io.load_json_object(data_const.anno_list_json)
    hoi_cls_count = {}
    for anno in tqdm(anno_list):
        if 'test' in anno['global_id']:
            continue

        for hoi in anno['hois']:
            hoi_id = hoi['id']
            if hoi_id not in hoi_cls_count:
                hoi_cls_count[hoi_id] = 0
            hoi_cls_count[hoi_id] += len(hoi['connections'])

    upper_limits = [10,50,100,500,1000,10000]
    bin_to_hoi_ids = bin_hoi_ids(hoi_cls_count,upper_limits)

    hoi_cls_count_json = os.path.join(data_const.proc_dir,'hoi_cls_count.json')
    io.dump_json_object(hoi_cls_count,hoi_cls_count_json)

    bin_to_hoi_ids_json = os.path.join(data_const.proc_dir,'bin_to_hoi_ids.json')
    io.dump_json_object(bin_to_hoi_ids,bin_to_hoi_ids_json)


if __name__=='__main__':
    main()