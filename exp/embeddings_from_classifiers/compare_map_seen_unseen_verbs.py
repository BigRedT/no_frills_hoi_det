import os
import argparse
import numpy as np

import utils.io as io
from utils.argparse_utils import manage_required_args, str_to_bool
from data.hico.hico_constants import HicoConstants
from utils.constants import Constants, ExpConstants

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_train_verbs',
    type=int,
    default=100,
    help='Number of training verbs')
parser.add_argument(
    '--map_json',
    type=str,
    help='Path to mAP.json')

def mark_hois_seen_or_unseen(hoi_list,verb_list,num_train_verbs):
    # hoi_list is itself modified
    verb_to_id = {v['name']: v['id'] for v in verb_list}
    for hoi in hoi_list:
        verb = hoi['verb']
        verb_id = verb_to_id[verb]
        verb_idx = int(verb_id)-1
        if verb_idx < num_train_verbs:
            hoi['is_seen'] = True
        else:
            hoi['is_seen'] = False

def main():
    args = parser.parse_args()
    hico_const = HicoConstants()
    verb_list = io.load_json_object(hico_const.verb_list_json)
    hoi_list = io.load_json_object(hico_const.hoi_list_json)
    mark_hois_seen_or_unseen(hoi_list,verb_list,args.num_train_verbs)
    hoi_dict = {hoi['id']:hoi for hoi in hoi_list}
    APs = io.load_json_object(args.map_json)['AP']
    APs_categorized = {
        'seen': [],
        'unseen': [],
    }
    AP_ = []
    for hoi_id, hoi in hoi_dict.items():
        AP = APs[hoi_id]
        AP_.append(AP)
        if hoi['is_seen']==True:
            APs_categorized['seen'].append(AP)
        else:
            APs_categorized['unseen'].append(AP)

    mAPs = {}
    for categories, list_of_APs in APs_categorized.items():
        mAPs[categories] = np.mean(list_of_APs)

    print(mAPs)
    print(len(APs_categorized['seen']),len(APs_categorized['unseen']))
    print(np.mean(AP_))

if __name__=='__main__':
    main()