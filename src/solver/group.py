import torch
from fast_pytorch_kmeans import KMeans
from .base import BaselineSolver
from tqdm import tqdm

# class HeuristicsSolver(BaseSolver):
#     def __init__(self, args):
#         super().__init__(args)
#         # self.k = 128
#         self.k = 2

#     def initial_assignment(self):
#         raise NotImplementedError
    
#     def solve(self):
#         # random or popular recommendation of NFT to buyers
#         _assignments = self.initial_assignment()

#         # self.holdings = torch.zeros(self.nftP.N, self.nftP.M)
#         # self.holdings[torch.arange(self.nftP.N)[:, None], _assignments] = 1
        
#         batch_size = 1000
#         budget_per_item = self.buyer_budgets.cpu() / self.k

#         self.pricing = torch.zeros(self.nftP.M)

#         for batch_users in make_batch_indexes(self.nftP.N, batch_size):
#             batch_size = len(batch_users)
#             holdings = torch.zeros(batch_size, self.nftP.M)
#             holdings[torch.arange(batch_size)[:, None], _assignments[batch_users]] = 1
#             buyer_spendings = holdings * budget_per_item[batch_users].unsqueeze(1)
#             self.pricing += buyer_spendings.sum(0)/self.nft_counts.cpu()

#         self.pricing.clamp_(0.1)
#         self.pricing = self.pricing.to(self.args.device)


class GroupSolver(BaselineSolver):
    def __init__(self, args):
        super().__init__(args)

    def initial_assignment(self):
        # group buyers based on preferences
        # not using kmeans-pytorch https://github.com/subhadarship/kmeans_pytorch/tree/master => nan issue
        # using Fast Pytorch Kmeans https://github.com/DeMoriarty/fast_pytorch_kmeans

        # num_clusters = 256
        num_clusters = 20
        kmeans = KMeans(n_clusters=num_clusters, mode='euclidean', verbose=0)
        labels = kmeans.fit_predict(self.buyer_preferences)
        labels = labels.cpu()
        buyer_ids_list = [torch.arange(self.nftP.N)[labels==i] for i in range(num_clusters)]
        _len = max([len(x) for x in buyer_ids_list])
        _assignment = torch.ones((self.nftP.N, _len), device=self.args.device).long()

        for batch_buyer_ids in tqdm(buyer_ids_list, leave=False, ncols=88, desc='iterate group equilibrium'):
            ## recommend items to group, each member has a vote on which item to choose
            selections = torch.randint(0, self.nftP.M, (_len, ), device=self.args.device)

            for __ in range(20): #256
                for buyer_id in batch_buyer_ids:
                    # choose a better item for buyer_id
                    # follow Lucas et al. Users’ satisfaction in recommendation systems for groups: an approach based on noncooperative games. WWW 2013
                    mask = torch.ones_like(self.Uij[buyer_id], dtype=torch.bool)
                    mask[selections] = False
                    p = torch.argmax(self.Uij[buyer_id]* mask) # best choice not selected (out of M)
                    q = torch.argmin(self.Uij[buyer_id].masked_select(~mask)) # worst selected (out of _len)
                    selections[q] = p

            _assignment[batch_buyer_ids] = selections

        return _assignment

