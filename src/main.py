import sys
import argparse
from dataset import prepare_nft_data
from experiments import *
from utils import set_seed, set_verbose

def main():

    print('cdd <=> cdp')
    print('\n === \n work on: main experiments => \n === \n')
    print('''
        ~~1. NFT data in dataset.py~~
        2. solver/base.py: => make random work!
        - solver/project.py
        - solver/base.py => opt_uniform_holding

        3. finish main experiments, check result
        ===
        plan ablation and further experiments
        ''')

    choices = ['main']
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', choices=choices+['all'], default='main')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', choices=[0,1,2], default=2) # warning; info; debug
    args = parser.parse_args()

    set_seed(args.seed)
    set_verbose(args.verbose)

    prepare_nft_data()

    if args.c != all:
        getattr(sys.modules[__name__], f'run_{args.c}_exp')()
    else:
        for choice in choices:
            getattr(sys.modules[__name__], f'run_{choice}_exp')()

if __name__ == '__main__':
    # set_verbose(2)
    main()