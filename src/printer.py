import sys
import argparse
from renders import *

def main():
    """
    Run all visualizations
    """
    choices = ['stats', 'main']
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', choices=choices+['all'], default='main')
    args = parser.parse_args()

    if args.c != 'all':
        getattr(sys.modules[__name__], f'print_{args.c}')()
    else:
        for choice in choices:
            getattr(sys.modules[__name__], f'print_{choice}')()


if __name__ == "__main__":
    main()