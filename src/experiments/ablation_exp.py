import shutil
import logging
from arguments import default_args
from solver import BANTERSolver

nft_project = 'Fat Ape Club'
breeding = 'ChildProject'
method = 'BANTER'

def run_ablation_exp():
    
    msg = f'''
    >>> (ablation_exp.py) Ablation experiments:
        {nft_project} x {breeding} x {method}\n'''
    print(msg)

    task1msg = '''1. pricing optimization: BANTER vs BANTER (no INIT) vs INIT ''' 
    print(task1msg)
    run_task1_optimization()

    task2msg = '''2. step size scheduling: BANTER vs BANTER (fixed) vs BANTER (none)''' 
    print(task2msg)
    run_task2_scheduling()

    task3msg = '''3. breeding candidate sampling: BANTER vs BANTER (objective) vs BANTER (random)'''
    print(task3msg)
    run_task3_sampling()

def run_task1_optimization():
    args = default_args()
    args.checkpoint_dir = args.ckpt_dir / 'ablation/optimization'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.read_initial_steps = True

    for ablation_id in range(3):
        result_file = args.checkpoint_dir / f'{nft_project}_{breeding}_optimization{ablation_id}'
        if result_file.exists(): 
            logging.info(f'|> result file:{result_file} exists <|')
        else:
            logging.info(f'...running optimization ablation {ablation_id}...')
            args.ablation_id = ablation_id
            start = time.time()
            Solver = BANTERSolver(args)
            Solver.solve()    
            runtime = time.time() - start
            Result = {'revenue_list': Solver.seller_revenue_list}
            Result = {k:deep_to_pylist(v) for k, v in Result.items()}
            dumpj(Result, result_file.with_suffix('.json'))
    print('______________________________ experiment done.')




def run_task2_scheduling():
    args = default_args()
    args.checkpoint_dir = args.ckpt_dir / 'ablation/scheduling'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.read_initial_steps = True

    for schedule_id in range(3):
        result_file = args.checkpoint_dir / f'{nft_project}_{breeding}_schedule{schedule_id}'
        if result_file.exists(): 
            logging.info(f'|> result file:{result_file} exists <|')
        else:
            logging.info(f'...running scheduling ablation {schedule_id}...')
            args.schedule_id = schedule_id
            start = time.time()
            Solver = BANTERSolver(args)
            Solver.solve()    
            Solver.count_results()
            runtime = time.time() - start
            Result = {'revenue_list': Solver.seller_revenue_list}
            Result = {k:deep_to_pylist(v) for k, v in Result.items()}
            dumpj(Result, result_file.with_suffix('.json'))
    print('______________________________ experiment done.')


def run_task3_sampling():
    args = default_args()
    args.checkpoint_dir = args.ckpt_dir / 'ablation/sampling'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for module_id in range(3):
        result_file = args.checkpoint_dir / f'{nft_project}_{breeding}_module{module_id}'
        if result_file.exists(): 
            logging.info(f'|> result file:{result_file} exists <|')
        else:
            if module_id == 0:
                # copy results from original version
                shutil.copy(args.ckpt_dir/'main_exp'/f'{nft_project}_BANTER_{breeding}.pth', result_file)
            else:
                logging.info(f'...running sampling ablation {module_id}...')
                args.module_id = module_id
                start = time.time()
                Solver = BANTERSolver(args)
                Solver.solve()    
                Solver.count_results()
                runtime = time.time() - start
                Result = {
                        'run_time': runtime,
                        'seller_revenue': Solver.seller_revenue,
                        'avg_buyer_utility': Solver.buyer_utilities.sum(1).mean().item(),
                    }
                Result = {k:deep_to_pylist(v) for k, v in Result.items()}
                dumpj(Result, result_file.with_suffix('.json'))
    print('______________________________ experiment done.')


