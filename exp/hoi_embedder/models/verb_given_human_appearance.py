import torch

from exp.hoi_embedder.models.verb_given_object_appearance import \
    VerbGivenObjectAppearanceConstants, VerbGivenObjectAppearance


class VerbGivenHumanAppearanceConstants(VerbGivenObjectAppearanceConstants):
    def __init__(self):
        super(VerbGivenHumanAppearanceConstants,self).__init__()


class VerbGivenHumanAppearance(VerbGivenObjectAppearance):
    def __init__(self,const):
        super(VerbGivenHumanAppearance,self).__init__(const)

    def forward(self,feats,verb_vecs):
        factor_feats = self.forward_mlp_all_but_last(
            feats['human_rcnn'],
            self.mlp)
        xformed_verb_vec = self.verb_vec_xform(verb_vecs)
        factor_scores = torch.mm(factor_feats,torch.transpose(xformed_verb_vec,0,1))
        return factor_scores

