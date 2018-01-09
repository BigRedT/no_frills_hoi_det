import os

from tests.tester import *
from data.hico.hico_constants import HicoConstants

def test_HicoConstants():
    hico_const = HicoConstants()
    hico_const_json = os.path.join(
        '/home/tanmay/Data/weakly_supervised_hoi_exp/scratch',
        'hico_constants.json')
    hico_const.to_json(hico_const_json)

if __name__=='__main__':
    list_tests(globals())