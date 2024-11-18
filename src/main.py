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
        2. solver/base.py: greedy init pricing + topk + optimize quantity
        3. finish main experiments, check result
        ===
        plan ablation and further experiments
        ''')

    prepare_nft_data()

    print('>stop<')
    return

    choices = ['main']
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', choices=choices+['all'], default='main')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', choices=[0,1,2], default=1) # warning; info; debug
    args = parser.parse_args()

    set_seed(args.seed)
    set_verbose(args.seed)

    if args.c != all:
        getattr(sys.modules[__name__], f'run_{args.c}_exp')()
    else:
        for choice in choices:
            getattr(sys.modules[__name__], f'run_{choice}_exp')()



if __name__ == '__main__':
    main()