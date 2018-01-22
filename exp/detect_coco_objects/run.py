import exp.detect_coco_objects.prepare_data_for_faster_rcnn as \
    prepare_data_for_faster_rcnn
from data.hico.hico_constants import HicoBoxesConstants
from utils.constants import ExpConstants
from exp.experimenter import *

def exp_detect_coco_objects():
    exp_name = 'detect_coco_objects'
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir='/home/tanmay/Data/weakly_supervised_hoi_exp')
    
    data_const = HicoBoxesConstants(
        clean_dir='/home/ssd/hico_det_clean_20160224',
        proc_dir='/home/ssd/hico_det_processed_20160224')

    prepare_data_for_faster_rcnn.prepare(exp_const,data_const)


if __name__=='__main__':
    list_exps(globals())