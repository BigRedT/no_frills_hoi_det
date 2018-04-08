from exp.experimenter import *
from data.hico.hico_constants import HicoConstants
from utils.constants import ExpConstants
import exp.relation_classifier.hoi_candidates as hoi_candidates
import exp.relation_classifier.label_hoi_candidates as label_hoi_candidates

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


def exp_gen_and_label_hoi_candidates():
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


if __name__=='__main__':
    list_exps(globals())