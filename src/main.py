import sys
import argparse
from dataset import prepare_nft_data
from experiments import *
from utils import set_seed, set_verbose

def main():

    choices = ['main', 'ablation', 'scale', 'qualitative']
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', choices=choices+['all'], default='scale')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', choices=[0,1,2], default=2) # warning; info; debug
    args = parser.parse_args()

    set_seed(args.seed)
    set_verbose(args.verbose)

    prepare_nft_data()

    if args.c != 'all':
        getattr(sys.modules[__name__], f'run_{args.c}_exp')()
    else:
        for choice in choices:
            getattr(sys.modules[__name__], f'run_{choice}_exp')()

if __name__ == '__main__':
    # set_verbose(2)
    main()