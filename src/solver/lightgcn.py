import random
import torch
from torch_geometric.nn import LightGCN

from tqdm import tqdm

from .base import BaselineSolver


from utils import *

# next: NCF

# N buyer M instance

class LightGCNSolver(BaselineSolver):
    def __init__(self, args):
        super().__init__(args)
        self.model = LightGCN(self.nftP.N+self.nftP.M, 64, 5)
        self.model.to(self.args.device)

    def initial_assignment(self):
        self.edge_index, self.neg_edge_index = self.prepare_data()
        self.train_model()
        _len = 5
        # _len = 32
        dst_index = torch.arange(self.nftP.N, self.nftP.N+self.nftP.M).to(self.args.device)
        _assignment = self.model.recommend(self.edge_index, src_index=torch.arange(self.nftP.N), dst_index=dst_index, k=_len)
        _assignment = _assignment - self.nftP.N

        return _assignment

    def prepare_data(self):
        # use self.Uij = preference dot attribute to derive edge_index
        k = self.num_negatives = 128

        __, topk_indices = torch.topk(self.Uij, k, dim=1)

        edge_index = []
        for i in range(self.nftP.N):
            for j in topk_indices[i]:
                edge_index.append([i, j.item()+ self.nftP.N])

        edge_index = torch.tensor(edge_index).T 
        return edge_index, self.gen_neg_edge(edge_index)

    def gen_neg_edge(self, edge_index):
        neg_edge_index = []
        for user in tqdm(edge_index[0].unique(), ncols=80, desc='prepare neg edge'):
            for _ in range(self.num_negatives):
                negative_item = torch.randint(self.nftP.N, self.nftP.N+self.nftP.M, (1,))
                while torch.any(torch.logical_and(edge_index[0] == user, edge_index[1] == negative_item)):
                    negative_item = torch.randint(self.nftP.N, self.nftP.N+self.nftP.M, (1,))
                neg_edge_index.append([user.item(), negative_item.item()])
        
        neg_edge_index = torch.tensor(neg_edge_index).T
        return neg_edge_index


    def train_model(self):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.edge_index = self.edge_index.to(self.args.device)
        self.neg_edge_index = self.neg_edge_index.to(self.args.device)

        # Training loop
        pbar = tqdm(range(1280), ncols=90, desc='LightGCN training')
        for epoch in pbar:
            self.model.train()
            optimizer.zero_grad()
            
            # Forward pass to compute rankings
            nd_embeddings = self.model.get_embedding(self.edge_index)

            pos_src, pos_dst = self.edge_index
            neg_src, neg_dst = self.neg_edge_index

            # Calculate positive and negative edge rankings (dot product of embeddings)
            pos_edge_rank = (nd_embeddings[pos_src] * nd_embeddings[pos_dst]).sum(dim=-1)
            neg_edge_rank = (nd_embeddings[neg_src] * nd_embeddings[neg_dst]).sum(dim=-1)

            loss = self.model.recommendation_loss(pos_edge_rank, neg_edge_rank)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Epoch':{epoch+1}, 'Loss': f'{loss.cpu().item():.4f}'})



