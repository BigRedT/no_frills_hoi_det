import torch
from torch.autograd import Variable

from tests.tester import *
from exp.hoi_embedder.models.verb_given_boxes_and_object_label import \
    VerbGivenBoxesAndObjectLabelConstants, VerbGivenBoxesAndObjectLabel

def test_VerbGivenBoxesAndObjectLabel():
    model_const = VerbGivenBoxesAndObjectLabelConstants()
    model = VerbGivenBoxesAndObjectLabel(model_const).cuda()
    factor_feat_dim = model.mlp_penultimate_feat_dim(model.mlp)
    feats = {
        'box': Variable(torch.zeros([10,21])).cuda(),
        'object_one_hot': Variable(torch.zeros([10,80]).cuda())
    }
    verb_vecs = Variable(torch.zeros([117,300])).cuda()
    factor_scores = model(feats,verb_vecs)
    import pdb; pdb.set_trace()

if __name__=='__main__':
    list_tests(globals())