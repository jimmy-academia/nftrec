import torch
from pathlib import Path
from types import SimpleNamespace

NFT_Projects = ['Axies Infinity', 'Bored Ape Yacht Club', 'Crypto Kitties', 'Fat Ape Club', 'Roaring Leader'] 
nft_project_names = [''.join(Project_Name.split()).lower() for Project_Name in NFT_Projects]
# Baseline_Methods = ['Greedy', 'Auction', 'Group', 'NCF', 'LightGCN', 'Reciprocal', 'HetRecSys', 'BANTER']
Baseline_Methods = ['Random']
Breeding_Types = ['Heterogeneous', 'Homogeneous', 'ChildProject']


def default_args():
    args = SimpleNamespace()
    args.ckpt_dir = Path('ckpt')
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.device = torch.device("cuda:0")
    args.breeding_topk = 10
    args.cand_lim = 50
    args.num_child_sample = 100
    args.mutation_rate = 0.1
    args.num_trait_div = 2
    args.num_attr_class = 2
    args.decay = 0.9
    
    args.large = False
    args.ablation_id = 0
    args.schedule_id = 0 # 0: dynamic, 1: fix weight 2: none
    args.module_id = 0 # 0: f_pop *TildeV (homo)/TildeV+attrclass (heter) 1:TildeV 2:rand
    args.setN = None
    args.setM = None
    
    args.overwrite = False
    # args.gamma1 = 0.1
    # args.gamma2 = 0.001
    return args

