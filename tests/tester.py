import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--test', 
    type=str, 
    default=None, 
    help='Name of the experiment to run')


def list_tests(module_globals):
    args = parser.parse_args()
    if args.test:
        module_globals[args.test]()
    else:
        list_of_tests = [name for name in module_globals.keys() if 'test_' in name]
        print('-'*80)
        print('Select one of the following test to run using flag --test:')
        print('-'*80)
        for test_name in list_of_tests:
            print('  ' + test_name)