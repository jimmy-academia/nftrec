import time
from solver import get_solver
from arguments import default_args, nft_project_names, Breeding_Types, Baseline_Methods
from utils import dumpj

def run_main_exp():
    args = default_args()
    args.checkpoint_dir = args.ckpt_dir / 'main_exp'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    msg = f'''
    >>> (main_exp.py) Main experiments:
        {nft_project_names}
        x {Breeding_Types}
        x {Baseline_Methods}'''
    print(msg)

    for nft_project_name in nft_project_names:
        args.nft_project_name = nft_project_name
        for _method in Baseline_Methods:
            for _breed in Breeding_Types:
                result_file = args.checkpoint_dir / f'{nft_project_name}_{_method}_{_breed}.pth'
            
                if result_file.exists() and not args.overwrite:
                    print(f'|> result file:{result_file} exists <|')
                else:
                    print(f'...running [{nft_project_name}, {_method}, {_breed}] experiment...')
                    args.breeding_type = _breed
                    Solver = get_solver(args, _method)

                    start_time = time.time()
                    Solver.solve() 
                    runtime = time.time() - start_time
                    Solver.evaluate() 
                    Result = {
                        'runtime': runtime,
                        'seller_revenue': Solver.seller_revenue,
                        'buyer_utilities': Solver.buyer_utilities, 
                        'pricing': Solver.pricing, 
                        'holdings': Solver.holdings, 
                        'buyer_budgets': Solver.buyer_budgets,
                        'nft_counts': Solver.nft_counts,
                    }
                    dumpj(Result, result_file)
                    print('______________________________________experiment done.')
