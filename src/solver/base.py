import copy
import torch
import math
from tqdm import tqdm

from utils import *
from .project import NFTProject
from debug import *

from .constant import Constfuncs, make_batch_indexes, hatrelu
import logging

class BaseSolver(Constfuncs):
    def __init__(self, args):
        super()
        self.args = args
        self.breeding_type = args.breeding_type

        # cache system: same setN, setM, nft_project_name yeilds same results.
        cache_dir = args.ckpt_dir/'cache'/f'{args.nft_project_name}_N_{args.setN}_M_{args.setM}'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_nft_project_file = cache_dir / f'project.pth'
        if cache_nft_project_file.exists():
            self.nftP = torch.load(cache_nft_project_file, weights_only=False)
        else:
            nft_project_data = loadj(f'../NFT_data/clean/{args.nft_project_name}.json')
            self.nftP = NFTProject(nft_project_data, args.setN, args.setM, args.nft_project_name)
            torch.save(self.nftP, cache_nft_project_file)
        
        logging.info(f'solving for {self.nftP.N} buyers and {self.nftP.M} NFTs')
        self.population_factor = 1
        self.num_traits = len(self.nftP.trait_dict)
        self.num_selections_list = [len(options) for (_, options) in self.nftP.trait_dict.items()]
        self.max_selections = max(self.num_selections_list)
        
        self.prepare_tensors()
        self.alpha = None
        self.Vj = self.calc_objective_valuations(self.nft_attributes)
        self.Uij = torch.matmul(self.buyer_preferences, self.nft_attributes.T.float())

        if self.breeding_type == 'Heterogeneous':
            cache_heter_labels_path = cache_dir / f'heter_files_{args.num_trait_div}_{args.num_attr_class}_{self.nftP.N}_{self.nftP.M}.pth'
            if cache_heter_labels_path.exists():
                self.nft_trait_divisions, self.nft_attribute_classes, self.buyer_types = torch_cleanload(cache_heter_labels_path, self.args.device)
            else:
                self.nft_trait_divisions =  torch.randint(args.num_trait_div, (self.nftP.M,)).to(self.args.device)
                self.nft_attribute_classes =  torch.randint(args.num_attr_class, (self.nftP.M,)).to(self.args.device)
                self.buyer_types = torch.randint(2, (self.nftP.N,)).to(self.args.device)
                torch_cleansave((self.nft_trait_divisions, self.nft_attribute_classes, self.buyer_types), cache_heter_labels_path)
        if self.breeding_type != 'None':
            cache_parents_path = cache_dir / f'parents_{args.breeding_type}_{args.num_child_sample}_{args.mutation_rate}_mod{args.module_id}.pth'
            if cache_parents_path.exists():
                # print('loading from...', cache_parents_path)
                self.ranked_parent_nfts, self.ranked_parent_expectations = torch_cleanload(cache_parents_path, self.args.device)
            else:
                print('saving to...', cache_parents_path)
                self.ranked_parent_nfts, self.ranked_parent_expectations = self.prepare_parent_nfts()
                torch_cleansave((self.ranked_parent_nfts, self.ranked_parent_expectations), cache_parents_path)

    def solve(self):
        '''yields:
        self.pricing
        self.holdings (recommendation of purchase amount to each buyer)
        '''
        raise NotImplementedError

    def count_results(self):
        # iterate over buyers, modifying if budget or supply limit is reached.
        self.seller_revenue = 0
        self.buyer_utilities = []
        self.assigned = torch.zeros_like(self.pricing)
        self.item_mask = torch.ones_like(self.pricing) # close item when sold out
        self.pricing.clamp_(min=0)

        holdings = copy.deepcopy(self.holdings)

        for batch_users in make_batch_indexes(self.nftP.N, 100, True):

            # Apply current item availability mask
            holdings *= self.item_mask

            # Calculate current batch's allocation
            current_assigned = holdings[batch_users].sum(0)
            subtotal = self.assigned + current_assigned

            # Handle supply constraints
            if not all(subtotal <= self.nft_counts):

                # Find items exceeding supply
                excess = subtotal - self.nft_counts
                excess_mask = excess > 0
                
                # Update item mask for future iterations
                self.item_mask = torch.where(excess_mask, 0.0, self.item_mask)


                # Calculate scaling factors for exceeding items
                item_scaling = torch.ones_like(current_assigned)
                valid_items = current_assigned > 0
                item_scaling[valid_items & excess_mask] = (
                    1 - excess[valid_items & excess_mask] / current_assigned[valid_items & excess_mask]
                )

                holdings[batch_users] *= item_scaling.unsqueeze(0)  # Broadcasting for users dimension


            # Handle budget constraints
            buyer_spending = (self.pricing * holdings[batch_users]).sum(1)  # Sum along items dimension
            budget_exceeded = buyer_spending > self.buyer_budgets[batch_users]
        
            if torch.any(budget_exceeded):
                # Calculate scaling factors for users exceeding budget
                scaling_factor = torch.ones_like(buyer_spending)
                scaling_factor[budget_exceeded] = (
                    self.buyer_budgets[batch_users][budget_exceeded] / 
                    buyer_spending[budget_exceeded]
                )
            
                # Apply scaling to holdings
                holdings[batch_users] *= scaling_factor.unsqueeze(1)  # Broadcasting for items dimension
        
            # Update assigned totals for next iteration
            self.assigned += holdings[batch_users].sum(0)

            self.buyer_utilities.append(self.calculate_buyer_utilities(
                batch_users,
                holdings[batch_users],
                self.buyer_budgets[batch_users],
                self.pricing,   
                True,
            ))

            self.seller_revenue += (self.pricing * holdings[batch_users]).sum()
        self.buyer_utilities = torch.cat(self.buyer_utilities, dim=0)
        
    def greedy_init_pricing(self):
        pricing = torch.rand(self.nftP.M, device=self.args.device) * self.buyer_budgets.mean()
        for __ in range(16):
            spending = self.Uij/pricing 
            spending = spending/spending.sum(1).unsqueeze(1)
            pricing = (spending * self.buyer_budgets.unsqueeze(1)).sum(0) / self.nft_counts
        pricing.clamp_(0.1)
        pricing = pricing.to(self.args.device)
        return pricing



class BaselineSolver(BaseSolver):
    def __init__(self, args):
        super().__init__(args)
        self.k = 20

    def initial_assignment(self):
        raise NotImplementedError
    
    def solve(self):
        _assignments = self.initial_assignment()
        self.pricing = self.greedy_init_pricing()
        self.holdings = self.opt_uniform_holding(_assignments)

    def opt_uniform_holding(self, _assignments):
        # iterate over batch buyer to adjust holding recommendation for top k assignemnts
        N, M = self.nftP.N, self.nftP.M
        batch_size = N // 500 if N >= 9000 else N // 20

        spending = torch.zeros(N, M + 1, device=self.args.device)
        row_indices = torch.arange(spending.size(0), device='cuda:0').unsqueeze(1)
        spending[row_indices, _assignments] = 1.0

        spending_scales = torch.rand(N, device=self.args.device) * 1/self.k

        user_iterator = self.buyer_budgets.argsort(descending=True).tolist()
        user_iterator = make_batch_indexes(user_iterator, batch_size)

        pbar = tqdm(range(16), ncols=88, desc='Optimizing holdings')

        prev_remaining = float('inf')
        for __ in pbar:
            for user_index in user_iterator:

                spending_scale = spending_scales[user_index].clone().detach().requires_grad_(True)
                batch_spending = spending[user_index] * spending_scale.unsqueeze(1)
                # assert all(batch_spending.sum(1) < 1)
                batch_budgets = self.buyer_budgets[user_index]
                holdings = batch_spending[:, :-1] * batch_budgets.unsqueeze(1) / self.pricing.unsqueeze(0)
                utility = self.calculate_buyer_utilities(
                    user_index, holdings, batch_budgets, self.pricing
                )
                    
                # Update spending scales using gradients
                utility.sum().backward()
                _grad = spending_scale.grad

                spending_scales[user_index] += 1e-2 * _grad/_grad.max()
                spending_scales.clamp_(min=0, max=1/self.k)

                # Normalize spending
                # spending = spending / spending.sum(dim=1, keepdim=True)
                
                # Track convergence
                remaining = spending[:, -1].sum()
                delta = abs(remaining - prev_remaining)
                prev_remaining = remaining
                pbar.set_postfix(delta=float(delta))
                
                if delta < 1e-6:
                    break
        
        # Calculate final demand
        final_spending = spending * spending_scales.unsqueeze(1)
        holdings = final_spending[:, :-1] * self.buyer_budgets.unsqueeze(1) / self.pricing.unsqueeze(0)
        return holdings


class RandomSolver(BaselineSolver):
    def __init__(self, args):
        super().__init__(args)
    def initial_assignment(self):
        random_assignments = torch.stack([torch.randperm(self.nftP.M)[:self.k] for _ in range(self.nftP.N)]).to(self.args.device)
        return random_assignments
