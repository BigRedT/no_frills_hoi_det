import os
import torch

from exp.hoi_classifier.models.hoi_classifier_model import \
    HoiClassifier, HoiClassifierConstants


def load_pretrained_hoi_classifier(model_path=None):
    # Try to find model at the following location if model_path is unspecified
    if model_path is None:
        exp_name = \
        'factors_' + \
        'rcnn_det_prob_' + \
        'appearance_' + \
        'boxes_and_object_label_' + \
        'human_pose'
        model_num = 25000
        model_path = os.path.join(
            os.getcwd(),
            f'data_symlinks/hico_exp/hoi_classifier/{exp_name}/' + \
            f'models/hoi_classifier_{model_num}')
    
    hoi_classifier_const = HoiClassifierConstants()
    hoi_classifier = HoiClassifier(hoi_classifier_const)
    hoi_classifier.load_state_dict(torch.load(model_path))
    factors = [
        'verb_given_human_app',
        'verb_given_object_app',
        'verb_given_boxes_and_object_label',
        'verb_given_human_pose'
    ]
    classifiers = {}
    for factor in factors:
        classifiers[factor] = \
            hoi_classifier.__getattr__(factor).mlp.layers[-1][0].weight

    return classifiers, hoi_classifier


def test():
    classifiers = load_pretrained_hoi_classifier()
    for factor,classifier in classifiers.items():
        print(f'{factor}: {classifier.size()}')


if __name__=='__main__':
    test()