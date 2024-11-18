import copy
import torch
import random
from tqdm import tqdm

from torch_geometric.nn import LightGCN

from .group import HeuristicsSolver

from utils import *

class ReciprocalSolver(HeuristicsSolver):
    def __init__(self, args):
        super().__init__(args)
        self.model = LightGCN(self.nftP.N+self.nftP.M, 64, 5)
        self.model.to(self.args.device)

    def initial_assignment(self):
        self.edge_index, self.neg_edge_index = self.prepare_data()
        self.edge_index = self.edge_index.to(self.args.device)

        ## the full thing!
        self.train(self.model, self.edge_index, self.neg_edge_index)
        _len = 32
        dst_index = torch.arange(self.nftP.N, self.nftP.N+self.nftP.M).to(self.args.device)
        _assignment = self.model.recommend(self.edge_index, src_index=torch.arange(self.nftP.N), dst_index=dst_index, k=_len)

        ## reciprocal train 
        '''
        A = buyer B = NFT, T_A ~ recommend to A, T_B ~ recommend to B;
        T_A = {above _len assigned}; T_B = {all user}
        D_10 = empty
        split all pair to D_01, D_11
        finetune 2 more model with bpr loss
        '''
        model_01 = copy.deepcopy(self.model)
        model_11 = copy.deepcopy(self.model)

        r_edge_index = []
        for i in range(self.nftP.N):
            for j in _assignment[i]:
                r_edge_index.append([i, j.item()])


        r_edge_index = torch.tensor(r_edge_index).T 
        neg_r_edge_index = self.gen_neg_edge(r_edge_index, _len)
        self.train(model_11, r_edge_index, neg_r_edge_index, num_epochs=128)
        self.train(model_01, neg_r_edge_index, r_edge_index, num_epochs=32)

        ## rerank, final assignment
        ui_edge = [torch.arange(self.nftP.N).repeat_interleave(self.nftP.M), dst_index.tile(self.nftP.N)]
        pred11 = model_11.predict_link(self.edge_index, ui_edge, prob=True)
        pred01 = model_01.predict_link(self.edge_index, ui_edge, prob=True)
        prediction = 0.5 * (pred11 + pred01)
        prediction = prediction.view(self.nftP.N, self.nftP.M)
        __, topk_indices = torch.topk(prediction, _len, dim=1)

        return topk_indices

    def prepare_data(self):
        # use self.Uij = preference dot attribute to derive edge_index
        k = num_negatives = 256

        __, topk_indices = torch.topk(self.Uij, k, dim=1)

        edge_index = []
        for i in range(self.nftP.N):
            for j in topk_indices[i]:
                edge_index.append([i, j.item()+ self.nftP.N])

        edge_index = torch.tensor(edge_index).T 
        return edge_index, self.gen_neg_edge(edge_index, k)

    def gen_neg_edge(self, edge_index, k=128):
        neg_edge_index = []
        for user in tqdm(edge_index[0].unique(), ncols=80, desc='prepare neg edge'):
            for _ in range(k):
                negative_item = torch.randint(self.nftP.N, self.nftP.N+self.nftP.M, (1,))
                while torch.any(torch.logical_and(edge_index[0] == user, edge_index[1] == negative_item)):
                    negative_item = torch.randint(self.nftP.N, self.nftP.N+self.nftP.M, (1,))
                neg_edge_index.append([user.item(), negative_item.item()])
        
        neg_edge_index = torch.tensor(neg_edge_index).T
        return neg_edge_index

    def train(self, model, pos_edge, neg_edge, num_epochs=1280, desc='LightGCN training'):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        pos_edge, neg_edge = pos_edge.to(self.args.device), neg_edge.to(self.args.device)
        # Training loop
        pbar = tqdm(range(num_epochs), ncols=90, desc=desc, leave=False)
        for epoch in pbar:
            model.train()
            optimizer.zero_grad()
            
            # Forward pass to compute rankings
            nd_embeddings = model.get_embedding(self.edge_index)

            pos_src, pos_dst = pos_edge
            neg_src, neg_dst = neg_edge

            # Calculate positive and negative edge rankings (dot product of embeddings)
            pos_edge_rank = (nd_embeddings[pos_src] * nd_embeddings[pos_dst]).sum(dim=-1)
            neg_edge_rank = (nd_embeddings[neg_src] * nd_embeddings[neg_dst]).sum(dim=-1)

            loss = model.recommendation_loss(pos_edge_rank, neg_edge_rank)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Epoch':{epoch+1}, 'Loss': f'{loss.cpu().item():.4f}'})
        
        

