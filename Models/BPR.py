import torch.nn.functional as F
import torch.nn as nn
import torch
from pdb import set_trace as bp

class BPR(nn.Module):

    def __init__(self, user_count, item_count, dim, gpu):
        super(BPR, self).__init__()
        self.user_count = user_count
        self.item_count = item_count

        if gpu != None:
            self.user_list = torch.LongTensor([i for i in range(user_count)]).to(gpu)
            self.item_list = torch.LongTensor([i for i in range(item_count)]).to(gpu)
        else:
            self.user_list = torch.LongTensor([i for i in range(user_count)])
            self.item_list = torch.LongTensor([i for i in range(item_count)])

        self.user_emb = nn.Embedding(self.user_count, dim)
        self.item_emb = nn.Embedding(self.item_count, dim)

        nn.init.normal_(self.user_emb.weight, mean=0., std= 0.01)
        nn.init.normal_(self.item_emb.weight, mean=0., std= 0.01)

        self.sim_type = 'inner product'

    def forward(self, mini_batch):

        user = mini_batch['user']
        pos_item = mini_batch['pos_item']
        neg_item = mini_batch['neg_item']

        u = self.user_emb(user)
        i = self.item_emb(pos_item)
        j = self.item_emb(neg_item)

        return u, i, j
        
    def get_loss(self, output):

        h_u, h_i, h_j = output[0], output[1], output[2]

        bpr_pos_score = (h_u * h_i).sum(dim=1, keepdim=True)
        bpr_neg_score = (h_u * h_j).sum(dim=1, keepdim=True)

        bpr_loss = -(bpr_pos_score - bpr_neg_score).sigmoid().log().sum()

        return bpr_loss
    

    def get_zerosum_loss(self, output):

        h_u, h_i, h_j = output[0], output[1], output[2]

        bpr_pos_score = (h_u * h_i).sum(dim=1, keepdim=True)
        bpr_neg_score = (h_u * h_j).sum(dim=1, keepdim=True)

        bpr_loss = -(bpr_pos_score - bpr_neg_score).sigmoid().log().sum()
        zerosum_loss = - torch.log((1 - torch.tanh(torch.abs(bpr_pos_score + bpr_neg_score)))+1e-5).sum()

        return zerosum_loss * 0.5


    def get_embedding(self):
        user = self.user_emb.weight
        item = self.item_emb.weight

        return user, item
    
    def get_score_mat(self):
        u_emb, i_emb = self.get_embedding()
        return u_emb @ i_emb.T

    # def forward_multiple_items(self, user, item, span_required=True):
    #     # user : bs x 1
    #     # item : bs x k
        
    #     if span_required:
    #         user = user.unsqueeze(-1)
    #         user = torch.cat(item.size(1) * [user], 1)

    #     user = self.user_emb(user)		# bs x k x dim
    #     item = self.item_emb(item)		# bs x k x dim


    #     score = (user * item).sum(dim=-1, keepdim=True)	
    #     return score
    
    # def forward_multi_items(self, batch_user, batch_items):

    #     u = self.user_emb(batch_user) # batch_size x dim
    #     u = u.unsqueeze(-1) # batch_size x dim x 1
    #     i = self.item_emb(batch_items) # batch_size x items x dim

    #     score = torch.bmm(i, u).squeeze() # batch_size x items

    #     return score
    
    # def forward_multi_users(self, batch_users, batch_item):
        
    #     i = self.item_emb(batch_item).unsqueeze(-1) # batch_size x dim x 1
    #     u = self.user_emb(batch_users) # batch_size x users x dim

    #     score = torch.bmm(u, i).squeeze() # batch_size x users

    #     return score

    # def forward_full_items(self, batch_user):
    #     # user : bs x 1	
    #     user = self.user_emb(batch_user)	# bs x d
    #     item = self.item_emb(self.item_list) # item x d

    #     return torch.matmul(user, item.T)