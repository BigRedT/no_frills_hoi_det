import os

from utils.argparse_utils import manage_required_args, str_to_bool
from exp.experimenter import *
from data.hico.hico_constants import HicoConstants
from utils.constants import Constants, ExpConstants
import exp.hoi_embedder.data.hoi_candidates as hoi_candidates
import exp.hoi_embedder.data.label_hoi_candidates as label_hoi_candidates
from exp.hoi_embedder.models.hoi_classifier_model import HoiClassifierConstants
import exp.hoi_embedder.train as train
import exp.hoi_embedder.eval as evaluate
import exp.hoi_embedder.eval_verb_confusion as evaluate_verb_confusion
from exp.hoi_embedder.data.features_dataset import FeatureConstants
import exp.hoi_embedder.data.cache_box_features as cache_box_features
import exp.hoi_embedder.data.cache_pose_features as cache_pose_features
import exp.hoi_embedder.data.assign_pose_to_human_candidates as \
    assign_pose_to_human_candidates
import exp.hoi_embedder.vis.top_boxes_per_hoi as \
    vis_top_boxes_per_hoi
import exp.hoi_embedder.vis.top_boxes_per_hoi_wo_inference as \
    vis_top_boxes_per_hoi_wo_inference
import exp.hoi_embedder.vis.vis_verb_vec_sim as vis_verb_vec_sim

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
    '--label_hoi_cand_oracle',
    default=False,
    action='store_true',
    help='Apply this flag to label hoi candidates with human, obj, verb oracles')
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
    '--fp_to_tp_ratio',
    type=int,
    default=1000,
    help='Number of images per batch')
parser.add_argument(
    '--model_num',
    type=int,
    help='Specify model number to evaluate')
parser.add_argument(
    '--verb_given_appearance',
    default=True,
    action='store_true',
    help='Use verb_given_human/object_appearance factor')
parser.add_argument(
    '--verb_given_human_appearance',
    default=False,
    type=str_to_bool,
    help='Set verb_given_human_appearance factor')
parser.add_argument(
    '--verb_given_object_appearance',
    default=False,
    type=str_to_bool,
    help='Set verb_given_object_appearance factor')
parser.add_argument(
    '--verb_given_boxes_and_object_label',
    default=True,
    type=str_to_bool,
    help='Use verb_given_boxes_and_object_label factor')
parser.add_argument(
    '--verb_given_human_pose',
    default=False,
    type=str_to_bool,
    help='Use verb_given_human_pose factor')
parser.add_argument(
    '--rcnn_det_prob',
    default=True,
    type=str_to_bool,
    help='Use detection prob from Faster-RCNN')
parser.add_argument(
    '--verb_vec_init',
    type=str,
    default='random',
    choices=['random','glove'],
    help='verb vector initialization')
parser.add_argument(
    '--verb_vec_fine_tune',
    default=False,
    type=str_to_bool,
    help='Whether to fine-tune verb vectors')
parser.add_argument(
    '--verb_vec_dim',
    type=int,
    default=300,
    help='Verb vector dimension')
parser.add_argument(
    '--sim2glove',
    default=False,
    type=str_to_bool,
    help='Enforce similarity to glove')


def exp_gen_and_label_hoi_cand():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['subset'],
        optional_args=['gen_hoi_cand','label_hoi_cand','label_hoi_cand_oracle'])
    if len(not_specified_args) > 0:
        return
    
    exp_name = 'hoi_candidates'
    exp_const = ExpConstants(exp_name=exp_name)
    exp_const.subset = args.subset

    data_const = HicoConstants()
    selected_dets_hdf5_relative_path = 'data_symlinks/' + \
        'hico_exp/select_confident_boxes_in_hico/' + \
        'select_boxes_human_thresh_0.01_max_10_object_thresh_0.01_max_10/' + \
        'selected_coco_cls_dets.hdf5'
    data_const.selected_dets_hdf5 = os.path.join(
        os.getcwd(),
        selected_dets_hdf5_relative_path)

    if args.gen_hoi_cand:
        print('Generating HOI candidates from Faster-RCNN dets...')
        hoi_candidates.generate(exp_const,data_const)
    
    if args.label_hoi_cand:
        print('Labelling HOI candidates from Faster-RCNN dets...')
        data_const.hoi_cand_hdf5 = os.path.join(
            exp_const.exp_dir,
            f'hoi_candidates_{exp_const.subset}.hdf5')
        label_hoi_candidates.assign(exp_const,data_const)

    if args.label_hoi_cand_oracle:
        print('Labelling HOI candidates from Faster-RCNN dets using oracle...')
        data_const.hoi_cand_hdf5 = os.path.join(
            exp_const.exp_dir,
            f'hoi_candidates_{exp_const.subset}.hdf5')
        label_hoi_candidates_oracle.assign(exp_const,data_const)


def exp_cache_box_feats():
    args = parser.parse_args()
    
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['subset'])
    
    exp_name = 'hoi_candidates'
    exp_const = ExpConstants(exp_name=exp_name)
    exp_const.subset = args.subset

    data_const = HicoConstants()
    data_const.hoi_cand_hdf5 = os.path.join(
        exp_const.exp_dir,
        f'hoi_candidates_{exp_const.subset}.hdf5')

    cache_box_features.main(exp_const,data_const)


def exp_assign_pose_to_human_cand():
    args = parser.parse_args()
    
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['subset'])
    
    exp_name = 'hoi_candidates'
    exp_const = ExpConstants(exp_name=exp_name)
    exp_const.subset = args.subset

    data_const = HicoConstants()
    data_const.hoi_cand_hdf5 = os.path.join(
        exp_const.exp_dir,
        f'hoi_candidates_{exp_const.subset}.hdf5')
    data_const.human_pose_dir = os.path.join(
        data_const.proc_dir,
        'human_pose')
    data_const.num_keypoints = 18

    assign_pose_to_human_candidates.main(exp_const,data_const)


def exp_cache_pose_feats():
    args = parser.parse_args()
    
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['subset'])
    
    exp_name = 'hoi_candidates'
    exp_const = ExpConstants(exp_name=exp_name)
    exp_const.subset = args.subset

    data_const = HicoConstants()
    data_const.hoi_cand_hdf5 = os.path.join(
        exp_const.exp_dir,
        f'hoi_candidates_{exp_const.subset}.hdf5')
    data_const.human_cands_pose_hdf5 = os.path.join(
        exp_const.exp_dir,
        f'human_candidates_pose_{exp_const.subset}.hdf5')
    data_const.num_keypoints = 18

    cache_pose_features.main(exp_const,data_const)


def exp_train():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=[
            'imgs_per_batch',
            'fp_to_tp_ratio',
            'verb_vec_init',
            'verb_vec_fine_tune',
            'verb_vec_dim',
            'sim2glove',
            'verb_given_appearance',
            'verb_given_human_appearance',
            'verb_given_object_appearance',
            'verb_given_boxes_and_object_label',
            'verb_given_human_pose',
            'rcnn_det_prob'])

    exp_name = 'verb_vec'
    exp_name += f'_init_{args.verb_vec_init}'
    exp_name += f'_finetune_{args.verb_vec_fine_tune}'
    exp_name += f'_dim_{args.verb_vec_dim}'
    if args.sim2glove:
        exp_name += f'_sim2glove_mlp_l2'
    
    out_base_dir = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_embedder')
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.num_epochs = 10
    exp_const.imgs_per_batch = args.imgs_per_batch
    exp_const.lr = 1e-3
    exp_const.sim2glove = args.sim2glove

    data_const = FeatureConstants()
    hoi_cand_dir = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_candidates')
    data_const.hoi_cands_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidates_train_val.hdf5')
    data_const.box_feats_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidates_box_feats_train_val.hdf5')
    data_const.hoi_cand_labels_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidate_labels_train_val.hdf5')
    data_const.human_pose_feats_hdf5 = os.path.join(
        hoi_cand_dir,
        'human_pose_feats_train_val.hdf5')
    data_const.faster_rcnn_feats_hdf5 = os.path.join(
        data_const.proc_dir,
        'faster_rcnn_fc7.hdf5')
    data_const.fp_to_tp_ratio = args.fp_to_tp_ratio
    data_const.subset = None # to be set in the train_balanced.py script
    
    model_const = Constants()
    model_const.hoi_classifier = HoiClassifierConstants()
    model_const.hoi_classifier.verb_given_appearance = args.verb_given_appearance
    model_const.hoi_classifier.verb_given_human_appearance = args.verb_given_human_appearance
    model_const.hoi_classifier.verb_given_object_appearance = args.verb_given_object_appearance
    model_const.hoi_classifier.verb_given_boxes_and_object_label = args.verb_given_boxes_and_object_label
    model_const.hoi_classifier.verb_given_human_pose = args.verb_given_human_pose
    model_const.hoi_classifier.rcnn_det_prob = args.rcnn_det_prob
    model_const.hoi_classifier.verb_vec_init = args.verb_vec_init
    model_const.hoi_classifier.verb_vec_fine_tune = args.verb_vec_fine_tune
    model_const.hoi_classifier.verb_vec_dim = args.verb_vec_dim
    model_const.hoi_classifier.glove_verb_vecs_npy = os.path.join(
        data_const.proc_dir,
        'glove_verb_vecs.npy')
    train.main(exp_const,data_const,model_const)


def exp_eval():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=[
            'model_num',
            'verb_vec_init',
            'verb_vec_fine_tune',
            'verb_vec_dim',
            'verb_given_appearance',
            'verb_given_human_appearance',
            'verb_given_object_appearance',
            'verb_given_boxes_and_object_label',
            'verb_given_human_pose',
            'rcnn_det_prob'])

    exp_name = 'verb_vec'
    exp_name += f'_init_{args.verb_vec_init}'
    exp_name += f'_finetune_{args.verb_vec_fine_tune}'
    exp_name += f'_dim_{args.verb_vec_dim}'
    
    out_base_dir = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_embedder')
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')

    data_const = FeatureConstants()
    hoi_cand_dir = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_candidates')
    data_const.hoi_cands_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidates_test.hdf5')
    data_const.box_feats_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidates_box_feats_test.hdf5')
    data_const.hoi_cand_labels_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidate_labels_test.hdf5')
    data_const.human_pose_feats_hdf5 = os.path.join(
        hoi_cand_dir,
        'human_pose_feats_test.hdf5')
    data_const.faster_rcnn_feats_hdf5 = os.path.join(
        data_const.proc_dir,
        'faster_rcnn_fc7.hdf5')
    data_const.balanced_sampling = False
    data_const.subset = 'test' 
    
    model_const = Constants()
    model_const.model_num = args.model_num
    model_const.hoi_classifier = HoiClassifierConstants()
    model_const.hoi_classifier.verb_given_appearance = args.verb_given_appearance
    model_const.hoi_classifier.verb_given_human_appearance = args.verb_given_human_appearance
    model_const.hoi_classifier.verb_given_object_appearance = args.verb_given_object_appearance
    model_const.hoi_classifier.verb_given_boxes_and_object_label = args.verb_given_boxes_and_object_label
    model_const.hoi_classifier.verb_given_human_pose = args.verb_given_human_pose
    model_const.hoi_classifier.rcnn_det_prob = args.rcnn_det_prob
    model_const.hoi_classifier.verb_vec_init = args.verb_vec_init
    model_const.hoi_classifier.verb_vec_fine_tune = args.verb_vec_fine_tune
    model_const.hoi_classifier.verb_vec_dim = args.verb_vec_dim
    model_const.hoi_classifier.glove_verb_vecs_npy = os.path.join(
        data_const.proc_dir,
        'glove_verb_vecs.npy')
    model_const.hoi_classifier.model_pth = os.path.join(
        exp_const.model_dir,
        f'hoi_classifier_{model_const.model_num}')
    evaluate.main(exp_const,data_const,model_const)


def exp_eval_verb_conf():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['model_num'],
        optional_args=[
            'verb_given_appearance',
            'verb_given_human_appearance',
            'verb_given_object_appearance',
            'verb_given_boxes_and_object_label',
            'verb_given_human_pose',
            'rcnn_det_prob'])

    exp_name = 'factors'
    if args.rcnn_det_prob:
        exp_name += '_rcnn_det_prob'
    if args.verb_given_appearance:
        exp_name += '_appearance'
    if args.verb_given_human_appearance:
        exp_name += '_human_appearance'
    if args.verb_given_object_appearance:
        exp_name += '_object_appearance'
    if args.verb_given_boxes_and_object_label:
        exp_name += '_boxes_and_object_label'
    if args.verb_given_human_pose:
        exp_name += '_human_pose'

    out_base_dir = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_classifier')
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')

    data_const = FeatureConstants()
    hoi_cand_dir = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_candidates')
    data_const.hoi_cands_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidates_test.hdf5')
    data_const.box_feats_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidates_box_feats_test.hdf5')
    data_const.hoi_cand_labels_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidate_labels_test.hdf5')
    data_const.human_pose_feats_hdf5 = os.path.join(
        hoi_cand_dir,
        'human_pose_feats_test.hdf5')
    data_const.faster_rcnn_feats_hdf5 = os.path.join(
        data_const.proc_dir,
        'faster_rcnn_fc7.hdf5')
    data_const.balanced_sampling = False
    data_const.subset = 'test' 
    
    model_const = Constants()
    model_const.model_num = args.model_num
    model_const.hoi_classifier = HoiClassifierConstants()
    model_const.hoi_classifier.verb_given_appearance = args.verb_given_appearance
    model_const.hoi_classifier.verb_given_human_appearance = args.verb_given_human_appearance
    model_const.hoi_classifier.verb_given_object_appearance = args.verb_given_object_appearance
    model_const.hoi_classifier.verb_given_boxes_and_object_label = args.verb_given_boxes_and_object_label
    model_const.hoi_classifier.verb_given_human_pose = args.verb_given_human_pose
    model_const.hoi_classifier.rcnn_det_prob = args.rcnn_det_prob
    model_const.hoi_classifier.model_pth = os.path.join(
        exp_const.model_dir,
        f'hoi_classifier_{model_const.model_num}')
    evaluate_verb_confusion.main(exp_const,data_const,model_const)


def exp_top_boxes_per_hoi():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['model_num'],
        optional_args=[
            'verb_given_appearance',
            'verb_given_boxes_and_object_label',
            'rcnn_det_prob'])

    exp_name = 'factors'
    if args.rcnn_det_prob:
        exp_name += '_rcnn_det_prob'
    if args.verb_given_appearance:
        exp_name += '_appearance'
    if args.verb_given_boxes_and_object_label:
        exp_name += '_boxes_and_object_label'
     
    out_base_dir = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_classifier')
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')

    data_const = FeatureConstants()
    hoi_cand_dir = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_candidates')
    data_const.hoi_cands_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidates_train_val.hdf5')
    data_const.box_feats_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidates_box_feats_train_val.hdf5')
    data_const.hoi_cand_labels_hdf5 = os.path.join(
        hoi_cand_dir,
        'hoi_candidate_labels_train_val.hdf5')
    data_const.faster_rcnn_feats_hdf5 = os.path.join(
        data_const.proc_dir,
        'faster_rcnn_fc7.hdf5')
    data_const.balanced_sampling = False
    data_const.subset = 'val' 
    
    model_const = Constants()
    model_const.model_num = args.model_num
    model_const.hoi_classifier = HoiClassifierConstants()
    model_const.hoi_classifier.verb_given_appearance = args.verb_given_appearance
    model_const.hoi_classifier.verb_given_boxes_and_object_label = args.verb_given_boxes_and_object_label
    model_const.hoi_classifier.rcnn_det_prob = args.rcnn_det_prob
    model_const.hoi_classifier.model_pth = os.path.join(
        exp_const.model_dir,
        f'hoi_classifier_{model_const.model_num}')

    vis_top_boxes_per_hoi.main(exp_const,data_const,model_const)


def exp_top_boxes_per_hoi_wo_inference():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['model_num'],
        optional_args=[
            'verb_given_appearance',
            'verb_given_boxes_and_object_label',
            'verb_given_human_pose',
            'rcnn_det_prob'])

    exp_name = 'factors'
    if args.rcnn_det_prob:
        exp_name += '_rcnn_det_prob'
    if args.verb_given_appearance:
        exp_name += '_appearance'
    if args.verb_given_boxes_and_object_label:
        exp_name += '_boxes_and_object_label'
    if args.verb_given_human_pose:
        exp_name += '_human_pose'
     
    out_base_dir = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_classifier')
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.num_to_vis = 10

    data_const = FeatureConstants()
    data_const.pred_hoi_dets_h5py = os.path.join(
        exp_const.exp_dir,
        f'pred_hoi_dets_test_{args.model_num}.hdf5')
    hoi_cand_dir = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_candidates')
    data_const.human_pose_feats_hdf5 = os.path.join(
        hoi_cand_dir,
        'human_pose_feats_test.hdf5')
    data_const.num_pose_keypoints = 18
    
    model_const = Constants()
    model_const.model_num = args.model_num
    model_const.hoi_classifier = HoiClassifierConstants()
    model_const.hoi_classifier.verb_given_appearance = args.verb_given_appearance
    model_const.hoi_classifier.verb_given_boxes_and_object_label = args.verb_given_boxes_and_object_label
    model_const.hoi_classifier.verb_given_human_pose = args.verb_given_human_pose
    model_const.hoi_classifier.rcnn_det_prob = args.rcnn_det_prob
    model_const.hoi_classifier.model_pth = os.path.join(
        exp_const.model_dir,
        f'hoi_classifier_{model_const.model_num}')

    vis_top_boxes_per_hoi_wo_inference.main(exp_const,data_const,model_const)


def exp_vis_verb_vec_sim():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=[
            'model_num',
            'verb_vec_init',
            'verb_vec_fine_tune',
            'verb_vec_dim',
            'verb_given_appearance',
            'verb_given_human_appearance',
            'verb_given_object_appearance',
            'verb_given_boxes_and_object_label',
            'verb_given_human_pose',
            'rcnn_det_prob'])

    exp_name = 'verb_vec'
    exp_name += f'_init_{args.verb_vec_init}'
    exp_name += f'_finetune_{args.verb_vec_fine_tune}'
    exp_name += f'_dim_{args.verb_vec_dim}'
    
    out_base_dir = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_embedder')
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.vis_dir = os.path.join(exp_const.exp_dir,'vis')

    data_const = HicoConstants()
    
    model_const = Constants()
    model_const.model_num = args.model_num
    model_const.hoi_classifier = HoiClassifierConstants()
    model_const.hoi_classifier.verb_given_appearance = args.verb_given_appearance
    model_const.hoi_classifier.verb_given_human_appearance = args.verb_given_human_appearance
    model_const.hoi_classifier.verb_given_object_appearance = args.verb_given_object_appearance
    model_const.hoi_classifier.verb_given_boxes_and_object_label = args.verb_given_boxes_and_object_label
    model_const.hoi_classifier.verb_given_human_pose = args.verb_given_human_pose
    model_const.hoi_classifier.rcnn_det_prob = args.rcnn_det_prob
    model_const.hoi_classifier.verb_vec_init = args.verb_vec_init
    model_const.hoi_classifier.verb_vec_fine_tune = args.verb_vec_fine_tune
    model_const.hoi_classifier.verb_vec_dim = args.verb_vec_dim
    model_const.hoi_classifier.glove_verb_vecs_npy = os.path.join(
        data_const.proc_dir,
        'glove_verb_vecs.npy')
    model_const.hoi_classifier.model_pth = os.path.join(
        exp_const.model_dir,
        f'hoi_classifier_{model_const.model_num}')

    vis_verb_vec_sim.main(exp_const,data_const,model_const)

if __name__=='__main__':
    list_exps(globals())