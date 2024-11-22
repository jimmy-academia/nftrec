import shutil
import logging
from arguments import default_args
from solver import BANTERSolver
from collections import defaultdict

nft_project = 'Fat Ape Club'
breeding = 'ChildProject'
method = 'BANTER'

N = 1000

def run_sensitivity_exp():
    
    msg = f'''
    >>> (sensitivity_exp.py) Sensitivity experiments:
        {nft_project} x {breeding} x {method}\n'''
    print(msg)

    task1msg = '''1. Sensitivity test on the number of buyers.''' 
    print(task1msg)
    run_task1_sensitivity()

    task2msg = '''2. Scalability test on large number of buyers.''' 
    print(task2msg)
    run_task2_scalability()

def run_task1_sensitivity():
    args = default_args()
    args.checkpoint_dir = args.ckpt_dir / 'sensitivity'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for method in Baseline_Methods:
        Result = defaultdict(list)
        result_file = args.checkpoint_dir / f'{method}_{nft_project}_{breeding}.pth'
        if result_file.exists(): 
            logging.info(f'|> result file:{result_file} exists <|')
            continue

        logging.info(f'...running sensitivity test for {method} method...')
        for scale in range(1, 11):
        
            args.setN = int(N/10 * scale)
            Solver = get_solver(args, method)
            start = time.time()
            Solver.solve()  
            Solver.count_results() 
            runtime = time.time() - start

            Result['runtime'].append(runtime)
            Result['revenue'].append(Solver.seller_revenue)
            Result['utility'].append(Solver.buyer_utilities.sum(1).mean().item())
            
        Result = {k:deep_to_pylist(v) for k, v in Result.items()}
        dumpj(Result, result_file.with_suffix('.json'))
    print('______________________________ experiment done.')


def setup_scaled_solver(args, scale, method, breeding):
    Solver = get_solver(args, method)
    setN = scale*10000
    M = 1000; etk = 2
    Solver.nftP.N = setN; Solver.nftP.M = M
    Solver.nft_counts = torch.ones(M).to(args.device)
    _attr = torch.rand(M, etk).to(args.device)
    Solver.nft_attributes = torch.where(_attr>0.5, torch.ones_like(_attr), torch.zeros_like(_attr)).long()
    Solver.nft_trait_counts = (Solver.nft_attributes * Solver.nft_counts.unsqueeze(1)).sum(0)
    Solver.buyer_preferences = Solver.buyer_preferences[:, :etk].repeat(setN// Solver.buyer_preferences.size(0)+1, 1) [:setN]
    Solver.buyer_budgets = Solver.buyer_budgets.repeat(setN// Solver.buyer_budgets.size(0)+1) [:setN]
    if breeding != 'None':
        Solver.ranked_parent_nfts =  Solver.ranked_parent_nfts[:, :etk, :].repeat(setN//Solver.ranked_parent_nfts.size(0) +1, 1, 1)[:setN]
        Solver.ranked_parent_expectations = Solver.ranked_parent_expectations[:, :etk].repeat(setN//Solver.ranked_parent_expectations.size(0) +1, 1)[:setN]

    Solver.Vj = Solver.Vj.repeat(setN// Solver.Vj.size(0)+1) [:M]
    Solver.Uij = torch.rand(setN, M).to(args.device)/10

    if method == 'HetRecSys':
        Solver.do_preparations()
    return Solver

def run_task2_scalability():
    args = default_args()
    args.checkpoint_dir = args.ckpt_dir / 'scalability'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.cand_lim = 4
    args.large = True

    for method in Baseline_Methods:
        Result = defaultdict(list)
        result_file = args.checkpoint_dir / f'{method}_{nft_project}_{breeding}.pth'
        if result_file.exists(): 
            logging.info(f'|> result file:{result_file} exists <|')
            continue

        logging.info(f'...running scalability test for {method} method...')
        for scale in range(1, 11):
            
            Solver = setup_scaled_solver(args, scale, method, breeding)
            start = time.time()
            Solver.solve()  
            Solver.count_results()
            runtime = time.time() - start
            Result['runtime'].append(runtime)
            Result['revenue'].append(Solver.seller_revenue)
            Result['utility'].append(Solver.buyer_utilities.sum(1).mean().item())
            
        Result = {k:deep_to_pylist(v) for k, v in Result.items()}
        dumpj(Result, result_file.with_suffix('.json'))
    print('______________________________ experiment done.')

