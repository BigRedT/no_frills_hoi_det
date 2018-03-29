import exp.detect_coco_objects.evaluate_boxes as evaluate_boxes
import exp.detect_coco_objects.prepare_data_for_faster_rcnn as prepare_data_for_faster_rcnn
import exp.detect_coco_objects.select_confident_boxes as select_confident_boxes
from data.hico.hico_constants import HicoBoxesConstants
from exp.experimenter import *
from utils.constants import ExpConstants


def exp_detect_coco_objects_in_hico():
    exp_name = 'detect_coco_objects_in_hico'
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir='/home/tanmay/Data/weakly_supervised_hoi_exp')
    
    data_const = HicoBoxesConstants(
        clean_dir='/home/ssd/hico_det_clean_20160224',
        proc_dir='/home/ssd/hico_det_processed_20160224')

    prepare_data_for_faster_rcnn.prepare_hico(exp_const,data_const)


def exp_select_and_evaluate_confident_boxes_in_hico():
    exp_name = 'select_confident_boxes_in_hico'
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir='/home/tanmay/Data/weakly_supervised_hoi_exp')
    exp_const.background_score_thresh = 0.01
    exp_const.max_humans = 10
    exp_const.max_objects_per_class = 10
    exp_const.max_background = 10
    exp_const.iou_thresh = 0.5

    data_const = HicoBoxesConstants(
        clean_dir='/home/ssd/hico_det_clean_20160224',
        proc_dir='/home/ssd/hico_det_processed_20160224')

    human_score_thresholds = [0.01] # [0.01,0.05,0.1,0.5]
    object_score_thresholds = [0.01] # [0.01,0.05,0.1,0.5]
    
    for human_score_thresh in human_score_thresholds:
        for object_score_thresh in object_score_thresholds:
            exp_const.human_score_thresh = human_score_thresh
            exp_const.object_score_thresh = object_score_thresh
            
            select_confident_boxes.select(exp_const,data_const)
            evaluate_boxes.evaluate_boxes(exp_const,data_const)
            evaluate_boxes.evaluate_boxes_and_labels(exp_const,data_const)

if __name__=='__main__':
    list_exps(globals())
