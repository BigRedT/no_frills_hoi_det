class GeometricFactorConstants(RelationClassifierConstants):
    def __init__(self):
        super(GeometricFactorConstants,self).__init__()
        self.box_feat_size = 12

    @property
    def box_feature_factor_const(self):
        factor_const = {
            'in_dim': self.box_feat_size,
            'out_dim': self.num_relation_classes,
            'out_activation': 'Identity',
            'layer_units': [],
            'activation': 'ReLU',
            'use_out_bn': False,
            'use_bn': True
        }
        return factor_const
    
    
class GeometricFactor(RelationClassifier):
    def __init__(self,const):
        super(GeometricFactor,self).__init__(const)
        self.box_feature_factor = pytorch_layers.create_mlp(
            self.const.box_feature_factor_const)

    def forward(self,feats):
        B = feats['human_rcnn'].size(0)
        faster_rcnn_feature_factor_scores = \
            self.faster_rcnn_human_feature_factor(feats['human_rcnn']) + \
            self.faster_rcnn_object_feature_factor(feats['object_rcnn']) #+ \
        box_feature_factor_scores = self.box_feature_factor(feats['box'])
        relation_prob = self.sigmoid(faster_rcnn_feature_factor_scores) * \
            self.sigmoid(box_feature_factor_scores)
        return relation_prob

    def forward_box_feature_factor(self,feats):
        box_feature_factor_scores = self.box_feature_factor(feats['box'])
        box_feature_factor_prob = self.sigmoid(box_feature_factor_scores)
        return box_feature_factor_prob, box_feature_factor_scores