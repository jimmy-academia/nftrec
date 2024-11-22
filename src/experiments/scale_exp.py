import time
from collections import defaultdict
from pathlib import Path
import logging

from solver import get_solver
from arguments import default_args, Baseline_Methods, Breeding_Types
from solver import BANTERSolver
from utils import deep_to_pylist, dumpj


def run_scale_exp():
    
    msg = f'''
    >>> (scale_exp.py) Sensitivity experiments:
        1. yelp dataset
        2. synthetic large dataset
        3. large candidate set (Fat Ape Club)

        '''

    logging.info("1. yelp dataset compare all breeding; all method;")
    run_yelp()
    logging.info("2. Scalability test on large number of buyers.")
    run_large()
    logging.info("3. scale candidate set length")
    run_candidate()

def run_yelp():
    args = default_args()
    args.checkpoint_dir = args.ckpt_dir / 'scale/yelp'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not Path(f'../NFT_data/clean/yelp.json').exists():
        logging.error('Prepare yelp data in NFT_data with `python process_yelp.py`')

    nft_project_name = 'yelp'
    args.nft_project_name = nft_project_name
    for _method in Baseline_Methods:
        for _breed in Breeding_Types[:-1]:
            result_file = args.checkpoint_dir / f'{nft_project_name}_{_method}_{_breed}.json'
            if result_file.exists() and not args.overwrite:
                logging.info(f'|> result file:{result_file} exists <|')
            else:
                logging.info(f'...running [yelp, {_method}, {_breed}] experiment...')
                args.breeding_type = _breed
                Solver = get_solver(args, _method)

                start_time = time.time()
                add_time = Solver.solve() 
                add_time = 0 if add_time is None else add_time
                Solver.count_results() 
                runtime = time.time() - start_time + add_time
                Result = {
                    'runtime': runtime,
                    'seller_revenue': Solver.seller_revenue,
                    'avg_buyer_utility': Solver.buyer_utilities.mean().item(),
                    'utility_component': Solver.utility_component
                }
                Result = {k:deep_to_pylist(v) for k, v in Result.items()}

                dumpj(Result, result_file)
                torch.save(
                    {'buyer_utilities': Solver.buyer_utilities, 
                    'pricing': Solver.pricing}, 
                    result_file.with_suffix('.pth')
                    )
                print('______________________________________experiment done.')



def run_large():
    args = default_args()
    args.checkpoint_dir = args.ckpt_dir / 'scale/large'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.cand_lim = 4
    args.large = True
    args.nft_project_name = 'fatapeclub'

    for _breeding in ['Heterogeneous', 'ChildProject', 'Homogeneous']:
        for _method in Baseline_Methods:
            for scale in range(1, 11):
                try:
                    result_file = args.checkpoint_dir / f'{args.nft_project_name}_{_breeding}_{_method}_scale{scale}.pth'

                    if result_file.exists():
                        print(f'{result_file}_____________ is finished.')
                        continue
                    print(f'running [{args.nft_project_name}_{_breeding}_{_method}_scale{scale}] experiment...')
                    args.breeding_type = _breeding
                    Solver = get_solver(args, _method)

                    ## just load (None, None) project file and adjust tensors
                    thenumber = scale*10000
                    M = 1000
                    etk = 2
                    Solver.nftP.N = thenumber
                    Solver.nftP.M = M
                    Solver.nft_counts = torch.ones(M).to(args.device)
                    _attr = torch.rand(M, etk).to(args.device)
                    Solver.nft_attributes = torch.where(_attr>0.5, torch.ones_like(_attr), torch.zeros_like(_attr)).long()
                    Solver.nft_trait_counts = (Solver.nft_attributes * Solver.nft_counts.unsqueeze(1)).sum(0)
                    Solver.buyer_preferences = Solver.buyer_preferences[:, :etk].repeat(thenumber// Solver.buyer_preferences.size(0)+1, 1) [:thenumber]
                    Solver.buyer_budgets = Solver.buyer_budgets.repeat(thenumber// Solver.buyer_budgets.size(0)+1) [:thenumber]
                    Solver.ranked_parent_nfts =  Solver.ranked_parent_nfts[:, :etk, :].repeat(thenumber//Solver.ranked_parent_nfts.size(0) +1, 1, 1)[:thenumber]
                    Solver.ranked_parent_expectations = Solver.ranked_parent_expectations[:, :etk].repeat(thenumber//Solver.ranked_parent_expectations.size(0) +1, 1)[:thenumber]
                    Solver.Vj = Solver.Vj.repeat(thenumber// Solver.Vj.size(0)+1) [:M]
                    Solver.Uij = torch.rand(thenumber, M).to(args.device)/10

                    if _method == 'HetRecSys':
                        Solver.do_preparations()
                        # Solver.embed_dim = 2

                    start = time.time()
                    Solver.solve()  
                    runtime = time.time() - start
                    Solver.evaluate() # evaluate buyer utility, seller revenue

                    dd = 3 if _method == 'BANTER' else 2
                    Result = {
                        'runtime': runtime,
                        'seller_revenue': Solver.seller_revenue.item(),
                        'buyer_utilities': Solver.buyer_utilities[:, :dd].sum(1).mean().item()
                        }
                    
                    torch.save(Result, result_file)
                    print('____________________________________________experiment done.')
                except:
                    print(f'[{args.nft_project_name}_{_breeding}_{_method}_scale{scale}] experiment cannot run XXXXXXX')


def run_candidate():
    pass