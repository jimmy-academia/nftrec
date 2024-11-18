import sys
import argparse
from dataset import prepare_nft_data
from experiments import *

def main():

    print('cdn <=> cdp')
    print('\n === \n work on: main experiments => \n === \n')
    print('''
        1. NFT data in dataset.py
        2. solver/base.py: greedy init pricing + topk + optimize quantity
        3. finish main experiments, check result
        ===
        plan ablation and further experiments
        ''')
    print('>stop<')
    return

    prepare_nft_data()

    choices = ['main']
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', choices=choices+['all'], default='main')
    args = parser.parse_args()

    if args.c != all:
        getattr(sys.modules[__name__], f'run_{args.c}_exp')()
    else:
        for choice in choices:
            getattr(sys.modules[__name__], f'run_{choice}_exp')()



if __name__ == '__main__':
    main()