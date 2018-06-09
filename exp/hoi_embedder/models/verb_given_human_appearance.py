from exp.hoi_classifier.models.verb_given_object_appearance import \
    VerbGivenObjectAppearanceConstants, VerbGivenObjectAppearance


class VerbGivenHumanAppearanceConstants(VerbGivenObjectAppearanceConstants):
    def __init__(self):
        super(VerbGivenHumanAppearanceConstants,self).__init__()


class VerbGivenHumanAppearance(VerbGivenObjectAppearance):
    def __init__(self,const):
        super(VerbGivenHumanAppearance,self).__init__(const)

    def forward(self,feats):
        factor_scores = self.mlp(feats['human_rcnn']) 
        return factor_scores

