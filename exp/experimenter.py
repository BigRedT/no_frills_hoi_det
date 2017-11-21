import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--exp', 
    type=str, 
    default=None, 
    help='Name of the experiment to run')


def list_exps(module_globals):
    args = parser.parse_args()
    if args.exp:
        module_globals[args.exp]()
    else:
        list_of_exps = [name for name in module_globals.keys() if 'exp_' in name]
        print('-'*80)
        print('Select one of the following exp to run using flag --exp:')
        print('-'*80)
        for exp_name in list_of_exps:
            print('  ' + exp_name)