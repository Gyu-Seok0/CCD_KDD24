import torch.nn.functional as F
import torch.nn as nn
import torch
from pdb import set_trace as bp

class LightGCN(nn.Module):
    def __init__(self, user_count, item_count, dim, gpu, SNM, num_layer, using_layer_index, keep_prob = 0.8):
        super(LightGCN, self).__init__()
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

        self.sim_type = 'inner product'
        self.SNM = SNM.detach() # (user + item) x (user + item)
        self.SNM.requires_grad = False
        
        self.num_layer = num_layer
        self.layer_index = None
        self.set_layer_index(using_layer_index)
        self.gpu = gpu
        self.keep_prob = keep_prob
        
        nn.init.normal_(self.user_emb.weight, mean=0., std= 0.01)
        nn.init.normal_(self.item_emb.weight, mean=0., std= 0.01)
    
    def set_layer_index(self, using_layer_index):
        if using_layer_index == "first":
            self.layer_index = 0 
        elif using_layer_index == "last":
            self.layer_index = self.num_layer - 1
        elif using_layer_index == "avg":
            self.layer_index = -1

    def _dropout_x(self, x):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + self.keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/self.keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
        
    def forward(self, mini_batch):
        
        user_emb, item_emb = self.get_embedding()  
        user = mini_batch['user']
        pos_item = mini_batch['pos_item']
        neg_item = mini_batch['neg_item']
  
        u = user_emb[user]
        i = item_emb[pos_item]
        j = item_emb[neg_item]

        pos_score = (u * i).sum(dim=1, keepdim=True)
        neg_score = (u * j).sum(dim=1, keepdim=True)

        return (pos_score, neg_score)

    def get_embedding(self, return_all = False):
        
        u_0 = self.user_emb.weight
        i_0 = self.item_emb.weight
        if self.layer_index == 0:
            return u_0, i_0
        
        # propagation
        total_ui_emb = [torch.cat([u_0, i_0])]
        for _ in range(self.num_layer):       
            total_ui_emb.append(torch.spmm(self.SNM, total_ui_emb[-1]))
        
        if return_all:
            user_embs = []
            item_embs = []
            for ui_emb in total_ui_emb:
                user_emb = ui_emb[:self.user_count]
                item_emb = ui_emb[self.user_count:]
                user_embs.append(user_emb)
                item_embs.append(item_emb)
            return user_embs, item_embs
            
        # aggregation
        if self.layer_index == -1:
            ui_emb = torch.mean(torch.stack(total_ui_emb), dim = 0)
        else:
            ui_emb = total_ui_emb[self.layer_index]
        
        user_emb = ui_emb[:self.user_count]
        item_emb = ui_emb[self.user_count:]

        return user_emb, item_emb


    def get_loss(self, output):
        pos_score, neg_score = output[0], output[1]
        loss = -(pos_score - neg_score).sigmoid().log().sum()
        
        return loss

    def forward_multi_items(self, batch_user, batch_items):
        
        user_emb, item_emb = self.get_embedding()

        u = user_emb[batch_user]# batch_size x dim
        u = u.unsqueeze(-1) # batch_size x dim x 1
        i = item_emb[batch_items] # batch_size x items x dim

        score = torch.bmm(i, u).squeeze() # batch_size x items

        return score
    
    def forward_multi_users(self, batch_users, batch_item):
        
        user_emb, item_emb = self.get_embedding()

        i = item_emb[batch_item].unsqueeze(-1) # batch_size x dim x 1
        u = user_emb[batch_users] # batch_size x users x dim

        score = torch.bmm(u, i).squeeze() # batch_size x users

        return score

    def forward_multiple_items(self, user, item, span_required=True):
        # user : bs x 1
        # item : bs x k
        
        if span_required:
            user = user.unsqueeze(-1)
            user = torch.cat(item.size(1) * [user], 1)
            
        u = self.user_emb(user)		# bs x k x dim
        i = self.item_emb(item)		# bs x k x dim
        
        score = (u * i).sum(dim=-1, keepdim=True)	
        return score


    def forward_full_items(self, batch_user):
        
        # user : bs x 1	
        user_emb, item_emb = self.get_embedding()
        user = user_emb[batch_user]	# bs x d
        item = item_emb # item x d

        return torch.matmul(user, item.T)