import sys
import argparse
from experiments import *

def main():

    choices = ['main']
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', choices=choices+['all'], default='main')
    args = parser.parse_args()

    if args.c != all:
        getattr(sys.modules[__name__], f'run_{args.c}_exp')()
    else:
        for choice in choices:
            getattr(sys.modules[__name__], f'run_{choice}_exp')()

    # print('same structure: main => experiment => module files')
    # print('but clean up the module files')
    # print('start with main experiment')
    pass


if __name__ == '__main__':
    main()