import sys
import argparse
from renders import *
from utils import set_verbose

def main():
    """
    Run all visualizations
    """
    choices = ['stats', 'main']
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', choices=choices+['all'], default='main')
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()

    set_verbose(1)

    if args.c != 'all':
        getattr(sys.modules[__name__], f'print_{args.c}')(args.preview)
    else:
        for choice in choices:
            getattr(sys.modules[__name__], f'print_{choice}')(args.preview)


if __name__ == "__main__":
    main()