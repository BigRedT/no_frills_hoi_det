import os

from utils.argparse_utils import manage_required_args
from exp.experimenter import *
from data.hico.hico_constants import HicoConstants
from utils.constants import Constants, ExpConstants
import exp.relation_classifier.hoi_candidates as hoi_candidates
import exp.relation_classifier.label_hoi_candidates as label_hoi_candidates
from exp.relation_classifier.relation_classifier_model import \
    RelationClassifierConstants, BoxAwareRelationClassifierConstants
from exp.relation_classifier.gather_relation_model import \
    GatherRelationConstants
import exp.relation_classifier.train as train
import exp.relation_classifier.train_balanced as train_balanced
import exp.relation_classifier.eval as evaluate
from exp.relation_classifier.features import FeatureConstants
from exp.relation_classifier.features_balanced import \
    FeatureBalancedConstants
import exp.relation_classifier.write_geometric_feats_to_hdf5 as \
    write_geometric_feats_to_hdf5

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
parser.add_argument(
    '--box_aware_model',
    default=False,
    action='store_true',
    help='Apply this flag to use model that utilizes box features')
parser.add_argument(
    '--subset',
    type=str,
    choices=['train','train_val','val','test'],
    help='Apply this flag to specify subset of data')
parser.add_argument(
    '--imgs_per_batch',
    type=int,
    default=1,
    help='Number of images per batch')
parser.add_argument(
    '--focal_loss',
    default=False,
    action='store_true',
    help='Apply this flag to use focal loss instead of BCE')
parser.add_argument(
    '--fp_to_tp_ratio',
    type=int,
    default=1000,
    help='Number of images per batch')


def exp_gen_and_label_hoi_cand():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['subset'],
        optional_args=['gen_hoi_cand','label_hoi_cand'])
    if len(not_specified_args) > 0:
        return
    
    exp_name = 'hoi_candidates'
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir='/home/tanmay/Data/weakly_supervised_hoi_exp')
    exp_const.subset = args.subset

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


def exp_compute_geometric_feats():
    args = parser.parse_args()
    
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['subset'])
    
    exp_name = 'hoi_candidates'
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir='/home/tanmay/Data/weakly_supervised_hoi_exp')
    exp_const.subset = args.subset

    data_const = HicoConstants()
    data_const.hoi_cand_hdf5 = os.path.join(
        exp_const.exp_dir,
        f'hoi_candidates_{exp_const.subset}.hdf5')

    write_geometric_feats_to_hdf5.main(exp_const,data_const)


def exp_train():
    exp_name = 'factors_rcnn_feats_scores'
    out_base_dir = \
        '/home/tanmay/Data/weakly_supervised_hoi_exp' + \
        '/relation_classifier'
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.num_epochs = 10
    exp_const.batch_size = 1024
    exp_const.lr = 1e-3

    data_const = FeatureConstants()
    hoi_cand_dir = '/home/tanmay/Data' + \
        '/weakly_supervised_hoi_exp/hoi_candidates'
    data_const.hoi_cands_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidates_train_val.hdf5')
    data_const.hoi_cand_labels_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidate_labels_train_val.hdf5')
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


def exp_train_balanced():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['imgs_per_batch','fp_to_tp_ratio'],
        optional_args=['focal_loss','box_aware_model'])

    exp_name = 'factors_rcnn_feats_scores_' + \
        f'imgs_per_batch_{args.imgs_per_batch}_' + \
        f'focal_loss_{args.focal_loss}_' + \
        f'fp_to_tp_ratio_{args.fp_to_tp_ratio}_' + \
        f'box_aware_model_{args.box_aware_model}'
    out_base_dir = \
        '/home/tanmay/Data/weakly_supervised_hoi_exp' + \
        '/relation_classifier'
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.num_epochs = 10
    exp_const.imgs_per_batch = args.imgs_per_batch
    exp_const.lr = 1e-3
    exp_const.focal_loss = args.focal_loss

    data_const = FeatureBalancedConstants()
    hoi_cand_dir = '/home/tanmay/Data' + \
        '/weakly_supervised_hoi_exp/hoi_candidates'
    data_const.hoi_cands_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidates_train_val.hdf5')
    data_const.box_feats_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidates_geometric_feats_train_val.hdf5')
    data_const.hoi_cand_labels_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidate_labels_train_val.hdf5')
    data_const.faster_rcnn_feats_hdf5 = os.path.join(
        data_const.proc_dir,
        'faster_rcnn_fc7.hdf5')
    data_const.fp_to_tp_ratio = args.fp_to_tp_ratio
    data_const.subset = None # to be set in the train_balanced.py script
    
    model_const = Constants()
    model_const.box_aware_model = args.box_aware_model
    if model_const.box_aware_model:
        model_const.relation_classifier = BoxAwareRelationClassifierConstants()
    else:
        model_const.relation_classifier = RelationClassifierConstants()
    model_const.gather_relation = GatherRelationConstants()
    model_const.gather_relation.hoi_list_json = data_const.hoi_list_json
    model_const.gather_relation.verb_list_json = data_const.verb_list_json

    train_balanced.main(exp_const,data_const,model_const)


def exp_eval():
    exp_name = 'factors_rcnn_feats_scores_imgs_per_batch_1_focal_loss_False_fp_to_tp_ratio_1000_box_aware_model_True'
    out_base_dir = \
        '/home/tanmay/Data/weakly_supervised_hoi_exp' + \
        '/relation_classifier'
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')

    data_const = FeatureBalancedConstants()
    hoi_cand_dir = '/home/tanmay/Data' + \
        '/weakly_supervised_hoi_exp/hoi_candidates'
    data_const.hoi_cands_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidates_test.hdf5')
    data_const.box_feats_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidates_geometric_feats_test.hdf5')
    data_const.hoi_cand_labels_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidate_labels_test.hdf5')
    data_const.faster_rcnn_feats_hdf5 = os.path.join(
        data_const.proc_dir,
        'faster_rcnn_fc7.hdf5')
    data_const.balanced_sampling = False
    data_const.subset = 'test' 
    
    model_const = Constants()
    model_const.model_num = 80000
    model_const.box_aware_model = True
    if model_const.box_aware_model:
        model_const.relation_classifier = BoxAwareRelationClassifierConstants()
    else:
        model_const.relation_classifier = RelationClassifierConstants()
    model_const.gather_relation = GatherRelationConstants()
    model_const.gather_relation.hoi_list_json = data_const.hoi_list_json
    model_const.gather_relation.verb_list_json = data_const.verb_list_json
    model_const.relation_classifier.model_pth = os.path.join(
        exp_const.model_dir,
        f'relation_classifier_{model_const.model_num}')
    evaluate.main(exp_const,data_const,model_const)


if __name__=='__main__':
    list_exps(globals())