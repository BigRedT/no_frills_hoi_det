from tqdm import tqdm

from tests.tester import *
from exp.relation_classifier.features import Features, FeatureConstants


def test_features():
    const = FeatureConstants()
    const.hoi_cands_hdf5 = \
        '/home/tanmay/Data/weakly_supervised_hoi_exp/hoi_candidates/' + \
        'hoi_candidates_test.hdf5'
    const.hoi_cand_labels_hdf5 = \
        '/home/tanmay/Data/weakly_supervised_hoi_exp/hoi_candidates/' + \
        'hoi_candidate_labels_test.hdf5'
    const.faster_rcnn_feats_hdf5 = \
        '/home/ssd/hico_det_processed_20160224/faster_rcnn_fc7.hdf5'
    const.subset = 'test'

    dataset = Features(const)
    #import pdb; pdb.set_trace()
    print(len(dataset))
    for i in tqdm(range(40000)):
        data = dataset[i]
        if data['hoi_id']=='background':
            continue

        print(data['hoi_id'])
        print(dataset.global_id_to_num_cands[data['global_id']])

    import pdb; pdb.set_trace()

    

if __name__=='__main__':
    list_tests(globals())