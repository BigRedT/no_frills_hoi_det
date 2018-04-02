import os

from data.hico.hico_constants import HicoConstants
import exp.dets_only.predict_hois as predict_hois
from exp.experimenter import *
from utils.constants import ExpConstants


def exp_detect_hoi():
    exp_name = 'dets_only'
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir='/home/tanmay/Data/weakly_supervised_hoi_exp')

    data_const = HicoConstants(
        clean_dir='/home/ssd/hico_det_clean_20160224',
        proc_dir='/home/ssd/hico_det_processed_20160224')
    data_const.selected_dets_dir = '/home/tanmay/Data/' + \
        'weakly_supervised_hoi_exp/select_confident_boxes_in_hico/' + \
        'select_boxes_human_thresh_0.01_max_10_object_thresh_0.01_max_10'
    
    predict_hois.main(exp_const,data_const)

if __name__=='__main__':
    list_exps(globals())
