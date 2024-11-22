import torch
from tqdm import tqdm

from utils import *
from debug import *

from .base import BaseSolver
from .constant import make_batch_indexes

class BaselineSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)
        self.k = 20
        self.add_time = 0

    def initial_assignment(self):
        raise NotImplementedError
    
    def objective_pricing(self):
        pricing = self.Vj/self.Vj.mean() * (self.buyer_budgets.sum()/self.nft_counts.sum())
        return pricing

    def solve(self):
        self.pricing = self.objective_pricing()
        _assignments = self.initial_assignment()
        assert _assignments.shape == torch.Size([self.nftP.N, self.k])
        self.holdings = self.opt_uniform_holding(_assignments)

        return self.add_time


    def opt_uniform_holding(self, _assignments):
        # iterate over batch buyer to adjust holding recommendation for top k assignemnts
        N, M = self.nftP.N, self.nftP.M
        batch_size = N // 500 if N >= 9000 else N // 20

        spending = torch.zeros(N, M + 1, device=self.args.device)
        row_indices = torch.arange(spending.size(0), device='cuda:0').unsqueeze(1)
        spending[row_indices, _assignments] = 1.0

        spending_scales = torch.rand(N, device=self.args.device)
        spending_scales.clamp_(0, 1)

        user_iterator = self.buyer_budgets.argsort(descending=True).tolist()
        user_iterator = make_batch_indexes(user_iterator, batch_size)

        pbar = tqdm(range(16), ncols=88, desc='Optimizing holdings')

        prev_scale = float('inf')
        for __ in pbar:
            for user_index in user_iterator:

                spending_scale = spending_scales[user_index].clone().detach().requires_grad_(True)
                batch_budgets = self.buyer_budgets[user_index]
                batch_spending = spending[user_index] * spending_scale.unsqueeze(1) * batch_budgets.unsqueeze(1) /self.k
                holdings = batch_spending[:, :-1] / self.pricing.unsqueeze(0)

                utility = self.calculate_buyer_utilities(
                    user_index, holdings, batch_budgets, self.pricing
                )
                    
                # Update spending scales using gradients
                utility.sum().backward()
                _grad = spending_scale.grad

                spending_scales[user_index] += 1e-2 * _grad/_grad.max()
                spending_scales.clamp_(min=0, max=1)

                # Normalize spending
                # spending = spending / spending.sum(dim=1, keepdim=True)
                
                # Track convergence
                delta = (spending_scales - prev_scale).abs().sum()
                prev_scale = spending_scales
                pbar.set_postfix(delta=float(delta))
                
                if delta < 1e-6:
                    break
        
        # Calculate final demand
        final_spending = spending * spending_scales.unsqueeze(1) * self.buyer_budgets.unsqueeze(1) / self.k
        holdings = final_spending[:, :-1] / self.pricing.unsqueeze(0)
        return holdings


class RandomSolver(BaselineSolver):
    def __init__(self, args):
        super().__init__(args)
    def initial_assignment(self):
        random_assignments = torch.stack([torch.randperm(self.nftP.M)[:self.k] for _ in range(self.nftP.N)]).to(self.args.device)
        return random_assignments

class GreedySolver(BaselineSolver):
    def __init__(self, args):
        super().__init__(args)

    def initial_assignment(self):
        favorite_assignments = ((self.Uij+self.Vj)/self.pricing).topk(self.k)[1]
        # .expand(self.nftP.N, self.k)
        return favorite_assignments
    