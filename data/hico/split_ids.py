import os
import random

import utils.io as io
from data.hico.hico_constants import HicoConstants


def split(global_ids,val_frac):
    # val_frac is num_val / num_train_val

    split_ids = {
        'train': [],
        'val': [],
        'train_val': [],
        'test': []
    }

    for global_id in global_ids:
        if 'test' in global_id:
            split_ids['test'].append(global_id)
        else:
            split_ids['train_val'].append(global_id)
    
    num_val = int(len(split_ids['train_val'])*val_frac)
    split_ids['val'] = random.sample(split_ids['train_val'],num_val)

    val_set = set(split_ids['val'])
    for global_id in split_ids['train_val']:
        if global_id not in val_set:
            split_ids['train'].append(global_id)

    return split_ids


def main():
    data_const = HicoConstants()

    hico_list = io.load_json_object(data_const.anno_list_json)
    global_ids = [anno['global_id'] for anno in hico_list]
    
    # Create and save splits
    split_ids = split(global_ids,0.2)

    split_ids_json = os.path.join(
        data_const.proc_dir,
        'split_ids.json')
    io.dump_json_object(split_ids,split_ids_json)

    # Create and save split stats
    split_stats = {}
    for subset_name,subset_ids in split_ids.items():
        split_stats[subset_name] = len(subset_ids)
        print(f'{subset_name}: {len(subset_ids)}')

    split_stats_json = os.path.join(
        data_const.proc_dir,
        'split_ids_stats.json')
    io.dump_json_object(split_stats,split_stats_json)


if __name__=='__main__':
    main()
