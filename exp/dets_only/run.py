import os

from data.hico.hico_constants import HicoConstants
import exp.dets_only.predict_hois as predict_hois
from exp.experimenter import *
from utils.constants import ExpConstants


def exp_detect_hoi():
    exp_name = 'dets_only'
    exp_const = ExpConstants(exp_name=exp_name)
    exp_const.subset = 'test'

    data_const = HicoConstants()
    data_const.selected_dets_hdf5 = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/' + \
        'select_confident_boxes_in_hico/selected_coco_cls_dets.hdf5')
    
    predict_hois.main(exp_const,data_const)

if __name__=='__main__':
    list_exps(globals())
