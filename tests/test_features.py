from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from tests.tester import *
from exp.relation_classifier.features_balanced import \
    FeaturesBalanced, FeatureBalancedConstants


def test_features():
    const = FeatureBalancedConstants()
    const.hoi_cands_hdf5 = \
        '/home/tanmay/Data/weakly_supervised_hoi_exp/hoi_candidates/' + \
        'hoi_candidates_train_val.hdf5'
    const.hoi_cand_labels_hdf5 = \
        '/home/tanmay/Data/weakly_supervised_hoi_exp/hoi_candidates/' + \
        'hoi_candidate_labels_train_val.hdf5'
    const.faster_rcnn_feats_hdf5 = \
        '/home/ssd/hico_det_processed_20160224/faster_rcnn_fc7.hdf5'
    const.subset = 'train_val'
    const.balanced_sampling = True

    dataset = FeaturesBalanced(const)
    # data_loader = DataLoader(dataset,batch_size=1,shuffle=True)
    sampler = RandomSampler(dataset)
    for i, sample_id in enumerate(sampler):
        data = dataset[sample_id]
        import pdb; pdb.set_trace()

    

if __name__=='__main__':
    list_tests(globals())