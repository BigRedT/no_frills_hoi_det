from tests.tester import *
from exp.hoi_classifier.models.hoi_classifier_model import \
    HoiClassifier, HoiClassifierConstants

def test_hoi_classifier():
    const = HoiClassifierConstants()
    classifier = HoiClassifier(const)
    import pdb; pdb.set_trace()

if __name__=='__main__':
    list_tests(globals())