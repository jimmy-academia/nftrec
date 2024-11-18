import torch
from tqdm import tqdm

from .base import BaseSolver
from utils import *

class BANTERSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)

    def solve(self):
        '''
        proposed BANTER method for NFT and pricing recommendation to obtain Market Equilibrium
        '''
        # self.pricing, self.holdings.
        ## pricing initialization
        # ablation_id 0: full, 1: no init, 2: only init
        self.pricing_list = []

        if self.args.ablation_id in [0, 2]:
            self.pricing = torch.rand(self.nftP.M, device=self.args.device) * self.buyer_budgets.mean()
            for __ in range(16):
                spending = self.Uij/self.pricing 
                spending = spending/spending.sum(1).unsqueeze(1)
                self.pricing = (spending * self.buyer_budgets.unsqueeze(1)).sum(0) / self.nft_counts
            
        else:
            self.pricing = torch.rand(self.nftP.M, device=self.args.device)

        ## demand-based optimization
        eps = 1001
        pbar = tqdm(range(10), ncols=88, desc='BANTER Solver!') #64
        self.pricing_list.append(self.pricing)
        if self.args.ablation_id == 2:
            return

        for __ in pbar:

            demand = self.solve_user_demand()
            demand = demand.sum(0)
            excess = demand - self.nft_counts

            if self.args.schedule_id == 0:
                eps = eps*torch.exp(-self.args.gamma1*excess.norm().sum()/self.nft_counts.sum() \
                + self.args.gamma2 * torch.tanh(self.ratio - 1))
            elif self.args.schedule_id == 1:
                eps = eps * 0.99

            self.pricing *= ( 1 +  eps * excess/(excess.abs().sum()))
            self.pricing = torch.where(self.pricing < 1e-10, 1e-10, self.pricing) 
            pbar.set_postfix(excess=float(excess.sum()))
            self.pricing_list.append(self.pricing)


            # eps *= self.args.decay
        
        self.holdings = self.solve_user_demand()

