import os

from exp.experimenter import *
from data.hico.hico_constants import HicoConstants
from utils.constants import Constants, ExpConstants
import exp.relation_classifier.hoi_candidates as hoi_candidates
import exp.relation_classifier.label_hoi_candidates as label_hoi_candidates
from exp.relation_classifier.relation_classifier_model import \
    RelationClassifierConstants
from exp.relation_classifier.gather_relation_model import \
    GatherRelationConstants
import exp.relation_classifier.train as train
from exp.relation_classifier.features import FeatureConstants

parser.add_argument(
    '--gen_hoi_cand',
    default=False,
    action='store_true',
    help='Apply this flag to generate hoi candidates')
parser.add_argument(
    '--label_hoi_cand',
    default=False,
    action='store_true',
    help='Apply this flag to label hoi candidates')


def exp_gen_and_label_hoi_cand():
    args = parser.parse_args()
    exp_name = 'hoi_candidates'
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir='/home/tanmay/Data/weakly_supervised_hoi_exp')
    exp_const.subset = 'test'

    data_const = HicoConstants(
        clean_dir='/home/ssd/hico_det_clean_20160224',
        proc_dir='/home/ssd/hico_det_processed_20160224')
    data_const.selected_dets_hdf5 = '/home/tanmay/Data/' + \
        'weakly_supervised_hoi_exp/select_confident_boxes_in_hico/' + \
        'select_boxes_human_thresh_0.01_max_10_object_thresh_0.01_max_10/' + \
        'selected_coco_cls_dets.hdf5' 

    if args.gen_hoi_cand:
        print('Generating HOI candidates from Faster-RCNN dets...')
        hoi_candidates.generate(exp_const,data_const)
    
    if args.label_hoi_cand:
        print('Labelling HOI candidates from Faster-RCNN dets...')
        data_const.hoi_cand_hdf5 = os.path.join(
            exp_const.exp_dir,
            f'hoi_candidates_{exp_const.subset}.hdf5')
        label_hoi_candidates.assign(exp_const,data_const)

    
def exp_train():
    exp_name = 'factors_rcnn_feats_scores'
    out_base_dir = \
        '/home/tanmay/Data/weakly_supervised_hoi_exp' + \
        '/relation_classifier'
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)

    data_const = FeatureConstants()
    data_const.hoi_cands_hdf5 = '/home/tanmay/Data' + \
        '/weakly_supervised_hoi_exp/hoi_candidates' + \
        'hoi_candidates_train_val.hdf5'
    data_const.hoi_cand_labels_hdf5 = '/home/tanmay/Data' + \
        '/weakly_supervised_hoi_exp/hoi_candidates' + \
        'hoi_candidate_labels_train_val.hdf5'
    data_const.faster_rcnn_feats_hdf5 = os.path.join(
        data_const.proc_dir,
        'faster_rcnn_fc7.hdf5')
    data_const.subset = None # to be set in the train.py script
    
    model_const = Constants()
    model_const.relation_classifier = RelationClassifierConstants()
    model_const.gather_relation = GatherRelationConstants()
    model_const.gather_relation.hoi_list_json = data_const.hoi_list_json
    model_const.gather_relation.verb_list_json = data_const.verb_list_json

    train.main(exp_const,data_const,model_const)


if __name__=='__main__':
    list_exps(globals())