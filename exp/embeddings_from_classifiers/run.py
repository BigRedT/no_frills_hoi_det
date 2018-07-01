import os

from utils.argparse_utils import manage_required_args, str_to_bool
from data.hico.hico_constants import HicoConstants
from utils.constants import Constants, ExpConstants
from exp.experimenter import *
from exp.hoi_classifier.data.features_dataset import FeatureConstants
import exp.embeddings_from_classifiers.train as train
from exp.embeddings_from_classifiers.models.one_to_all_model import \
    OneToAllConstants
import exp.embeddings_from_classifiers.update_hoi_classifier as \
    update_hoi_classifier
import exp.embeddings_from_classifiers.eval as evaluate


parser.add_argument(
    '--num_train_verbs',
    type=int,
    default=100,
    help='Number of training verbs')
parser.add_argument(
    '--word_vec',
    type=str,
    choices=['random','glove'],
    help='Use glove or random as pretrained word vectors')
parser.add_argument(
    '--make_identity',
    type=str_to_bool,
    default=False,
    help='Whether to use glove as global space')
parser.add_argument(
    '--coupling',
    type=str_to_bool,
    default=True,
    help='Whether to use coupling variable')

def exp_train():
    exp_name = 'first_try'
    out_base_dir=os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/embeddings_from_classifier')
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.num_steps = 40000
    exp_const.lr = 1e-2
    exp_const.num_train_verbs = 100
    exp_const.num_test_verbs = 17

    data_const = HicoConstants()
    data_const.glove_verb_vecs_npy = os.path.join(
        data_const.proc_dir,
        'glove_verb_vecs.npy')

    model_const = Constants()
    model_const.one_to_all = OneToAllConstants()
    train.main(exp_const,data_const,model_const)


def exp_ablation_num_train_verbs():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['num_train_verbs'])
    if len(not_specified_args) > 0:
        return

    exp_name = f'num_train_verbs_{args.num_train_verbs}'
    out_base_dir=os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/embeddings_from_classifier/ablation_num_train_verbs')
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.num_steps = 40000
    exp_const.lr = 1e-2
    exp_const.num_train_verbs = args.num_train_verbs
    exp_const.num_test_verbs = 17
    exp_const.word_vec = 'glove'
    exp_const.make_identity = False

    data_const = HicoConstants()
    data_const.glove_verb_vecs_npy = os.path.join(
        data_const.proc_dir,
        'glove_verb_vecs.npy')

    model_const = Constants()
    model_const.one_to_all = OneToAllConstants()
    train.main(exp_const,data_const,model_const)


def exp_ablation_glove_vs_random():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['word_vec'])
    if len(not_specified_args) > 0:
        return

    exp_name = f'word_vec_{args.word_vec}'
    out_base_dir=os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/embeddings_from_classifier/ablation_word_vec')
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.num_steps = 40000
    exp_const.lr = 1e-2
    exp_const.num_train_verbs = 100
    exp_const.num_test_verbs = 17
    exp_const.word_vec = args.word_vec

    data_const = HicoConstants()
    data_const.glove_verb_vecs_npy = os.path.join(
        data_const.proc_dir,
        'glove_verb_vecs.npy')

    model_const = Constants()
    model_const.one_to_all = OneToAllConstants()
    train.main(exp_const,data_const,model_const)


def exp_ablation_identity_vs_mlp():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['make_identity'])
    if len(not_specified_args) > 0:
        return

    exp_name = f'make_identity_{args.make_identity}'
    out_base_dir=os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/embeddings_from_classifier/ablation_identity_vs_mlp')
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.num_steps = 40000
    exp_const.lr = 1e-2
    exp_const.num_train_verbs = 100
    exp_const.num_test_verbs = 17
    exp_const.word_vec = 'glove'
    exp_const.make_identity = args.make_identity

    data_const = HicoConstants()
    data_const.glove_verb_vecs_npy = os.path.join(
        data_const.proc_dir,
        'glove_verb_vecs.npy')

    model_const = Constants()
    model_const.one_to_all = OneToAllConstants()
    train.main(exp_const,data_const,model_const)


def exp_ablation_coupling():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['coupling'])
    if len(not_specified_args) > 0:
        return

    exp_name = f'coupling_{args.coupling}'
    out_base_dir=os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/embeddings_from_classifier/ablation_coupling')
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.num_steps = 40000
    exp_const.lr = 1e-2
    exp_const.num_train_verbs = 100
    exp_const.num_test_verbs = 17
    exp_const.word_vec = 'glove'
    exp_const.make_identity = False

    data_const = HicoConstants()
    data_const.glove_verb_vecs_npy = os.path.join(
        data_const.proc_dir,
        'glove_verb_vecs.npy')

    model_const = Constants()
    model_const.one_to_all = OneToAllConstants()
    model_const.one_to_all.use_coupling_variable = args.coupling
    train.main(exp_const,data_const,model_const)


def exp_update_hoi_classifier():
    exp_name = f'coupling_False'
    out_base_dir=os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/embeddings_from_classifier/ablation_coupling')
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')

    data_const = HicoConstants()
    data_const.glove_verb_vecs_npy = os.path.join(
        data_const.proc_dir,
        'glove_verb_vecs.npy')

    model_const = Constants()
    # one_to_all model constants
    model_const.one_to_all = OneToAllConstants()
    model_const.one_to_all.use_coupling_variable = False
    model_const.one_to_all.model_num = 25000
    model_const.one_to_all.model_path = os.path.join(
        exp_const.model_dir,
        f'one_to_all_{model_const.one_to_all.model_num}')
    # hoi_classifier constants have been prespecified in load_classifiers.py

    update_hoi_classifier.main(exp_const,data_const,model_const)


def exp_eval():
    exp_name = f'coupling_False'
    out_base_dir=os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/embeddings_from_classifier/ablation_coupling')
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
    model_const.model_num = 25000
    model_const.hoi_classifier_path = os.path.join(
        exp_const.model_dir,
        f'hoi_classifier_{model_const.model_num}')
    evaluate.main(exp_const,data_const,model_const)


if __name__=='__main__':
    list_exps(globals())