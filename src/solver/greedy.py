import random
import torch
from .base import BaseSolver
from tqdm import tqdm

class OptimizationSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)

    def optimize_pricing(self):
        raise NotImplementedError
    
    def optimize_spending(self):
        raise NotImplementedError

    def solve(self, set_pricing=None):
        if set_pricing is None:
            self.optimize_pricing()
            spending = self.Uij/self.pricing
            spending = spending/spending.sum(1).unsqueeze(1) * self.buyer_budgets.unsqueeze(1)
            self.holdings = spending/spending.sum(0) * self.nft_counts
        else:
            self.pricing = set_pricing
            self.optimize_spending()


class GreedySolver(OptimizationSolver):
    def __init__(self, args):
        super().__init__(args)

    def optimize_pricing(self):
        ## greedy recommend NFT with highest value/price ratio
        self.pricing = torch.rand(self.nftP.M, device=self.args.device)
        for __ in range(16):
            spending = (self.Uij/self.pricing)
            # num_sel = self.nftP.M//10
            # un_sel = spending.topk(num_sel, largest=False)[1]
            # batch_size = 128

            # mask = torch.ones_like(spending)
            # mask.scatter_(1, spending.topk(self.nftP.M//10, largest=False)[1], 0)
            # spending.mul_(mask)
            spending = spending/spending.sum(1).unsqueeze(1)
            self.pricing = (spending * self.buyer_budgets.unsqueeze(1)).sum(0) / self.nft_counts
    
    def optimize_spending(self):
        spending = self.Uij/self.pricing 
        spending = spending/spending.sum(1).unsqueeze(1) * self.buyer_budgets.unsqueeze(1)
        self.holdings = spending/self.pricing
        fulfillment = self.holdings.sum(0) /self.nft_counts
        self.holdings = self.holdings * fulfillment

        