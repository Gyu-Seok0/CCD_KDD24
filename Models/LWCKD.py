import torch
import torch.nn as nn 
import torch.nn.functional as F
import pandas as pd
import random
from copy import deepcopy
from collections import deque
#from utils import get_common_ids

def get_two_hop_emb_total(R, start_new_id, base_emb, dim, topk = 5):
    
    target_size = R.size(0)
    two_hop_mat = torch.sparse.mm(R, R.T).to_dense()[:, :start_new_id] # U x before U
    two_hop_mat = torch.where(two_hop_mat > 0, 1, 0)    
    num_two_hop_mat_interaction = two_hop_mat.sum(1)
    
    Frequency = torch.sparse.sum(R, dim = 1).to_dense()[:start_new_id] # |before User(Item)|
    Frequency_sort = torch.argsort(Frequency, descending = False)
    Frequency_rank = torch.zeros_like(Frequency_sort)
    for rank, id in enumerate(Frequency_sort.tolist()):
        Frequency_rank[id] = rank
    
    embs = []
    
    for id in range(target_size):
        num_id_interaction = num_two_hop_mat_interaction[id]
        torch.where(two_hop_mat[id] > 0)
        
        id_emb = torch.zeros(dim)

        if num_id_interaction == 0:
            id_emb = base_emb[Frequency_sort[:topk]].mean(0)
        
        elif num_id_interaction < topk:

            ids = torch.where(two_hop_mat[id] > 0)[0]
            candidate_list = Frequency_sort[~torch.isin(Frequency_sort, ids)]
            
            id_emb += base_emb[ids].sum(0)
            id_emb += base_emb[candidate_list[: topk - num_id_interaction]].sum(0)
            id_emb /= topk
        else:
            Frequency_rank_for_id = two_hop_mat[id] * Frequency_rank
            ids = torch.topk(Frequency_rank_for_id, k = topk, largest = True).indices
            id_emb = base_emb[ids].mean(0)
                
        embs.append(id_emb)
        
    return torch.stack(embs)

def get_two_hop_emb(R, start_id, end_id, base_emb, dim, topk = 5):
    
    two_hop_mat = torch.sparse.mm(R, R.T).to_dense() # U x U
    two_hop_mat = two_hop_mat[:, :start_id] # U x (Before_U)
    two_hop_mat = torch.where(two_hop_mat > 0, 1, 0)
    num_interaction = two_hop_mat.sum(1)
    
    Frequency = torch.sparse.sum(R, dim = 1).to_dense() # 1-D array
    Frequency = Frequency[:start_id] # Before_U
    Frequency_argsort = torch.argsort(Frequency, descending = False)
    Frequency_Rank = torch.zeros_like(Frequency_argsort, device = Frequency_argsort.device)
    for rank, u_id in enumerate(Frequency_argsort):
        Frequency_Rank[u_id] = rank
    
    two_hop_rank_mat = two_hop_mat * Frequency_Rank
    new_user_embs = torch.zeros((0, dim), device = R.device)
    
    for u_id in range(start_id, end_id + 1):
        
        num_u_interaction = num_interaction[u_id]
        
        if num_u_interaction == 0:
            idxs = Frequency_argsort[-topk:]
            
        elif num_u_interaction <= topk:
            idxs1 = torch.where(two_hop_mat[u_id] > 0)[0]
            idxs2 = Frequency_argsort[~torch.isin(Frequency_argsort, idxs1)][-(topk - num_u_interaction): ]
            idxs = torch.cat([idxs1, idxs2])
            
        else:
            idxs = torch.topk(two_hop_rank_mat[u_id], k = topk, largest = True).indices
        
        new_user_emb = torch.mean(base_emb[idxs], dim = 0, keepdim = True)
        new_user_embs = torch.cat([new_user_embs, new_user_emb])
 
    return new_user_embs
        
def get_common_ids(before_ids, present_ids):
    common_ids_mask = torch.eq(before_ids.unsqueeze(0), present_ids.unsqueeze(1)).any(dim = 0)
    common_ids = before_ids[common_ids_mask]
    return common_ids

class PIW_LWCKD(nn.Module):
    def __init__(self, base_model, LWCKD_flag, PIW_flag, 
                 temperature = 5., num_cluster = 10, dim = 64,  gpu = None, model_type = "BPR", num_layer = 2):
        super().__init__()
        
        # basis
        self.gpu = gpu
        self.dim = dim
        self.LWCKD_flag = LWCKD_flag
        self.PIW_flag = PIW_flag

        # Model1: base_model
        self.base_model = base_model
        
        # Model2: PIW Module
        self.CSTM = nn.Parameter(torch.randn(num_cluster, dim, dim)) # cluster-specific transformation matrix
        self.PSV = PIW_State_Vector(num_cluster, dim) # MLP for PIW State Vector
        self.kl_loss = nn.KLDivLoss(reduction = "batchmean")
        
        # weight init for CSTM & PSV
        nn.init.normal_(self.CSTM, mean = 0., std = 0.01)
        for layer in self.PSV.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean= 0., std = 0.01)
        
        # LWCKD
        self.before_user_ids = None
        self.before_item_ids = None
        self.before_user_mapping = None
        self.before_item_mapping = None
        self.before_rating_mat = None
        self.before_user_emb = None
        self.before_item_emb = None
        self.UU = None # User-User graph
        self.II = None # Item-Item graph
        self.temperature = temperature

        # PIW
        self.present_user_ids = None
        self.present_item_ids = None
        self.present_cluster_center = None
        self.num_cluster = num_cluster
        self.v = None # the degree of freedom for T-dist
        self.user_PIW = dict()
        self.cluster = None
        
        # model_type
        self.model_type = model_type # BPR or L
        self.num_layer = num_layer

    def forward(self, mini_batch):
               
        # CF
        mini_batch = {key : values.to(self.gpu) for key, values in mini_batch.items()}
        output = self.base_model.forward(mini_batch)
        base_loss = self.base_model.get_loss(output)
        
        # LWCKD + PIW
        batch_user = mini_batch['user'].cpu()
        batch_pos_item = mini_batch['pos_item'].cpu()
        
        UI_loss = torch.tensor(0.0).to(self.gpu)
        IU_loss = torch.tensor(0.0).to(self.gpu)
        UU_loss = torch.tensor(0.0).to(self.gpu)
        II_loss = torch.tensor(0.0).to(self.gpu)
        cluster_loss = torch.tensor(0.0).to(self.gpu)
        
        # LWCKD + PIW
        if self.LWCKD_flag:
            if self.model_type == "BPR":
                user_emb, item_emb = self.base_model.get_embedding()
                UI_loss, IU_loss, UU_loss, II_loss, cluster_loss = self.LWCKD_PIW(batch_user, batch_pos_item, self.before_user_emb, user_emb, self.before_item_emb, item_emb, self.cluster)
                
            elif self.model_type == "LightGCN":
                user_embs, item_embs = self.base_model.get_embedding(return_all = True)
                                
                for i in range(self.num_layer):
                    layer_losses = self.LWCKD_PIW(batch_user, batch_pos_item, self.before_user_emb[i], user_embs[i], self.before_item_emb[i], item_embs[i], self.cluster[i])
                    
                    UI_loss += layer_losses[0] * (1 / self.num_layer + 1e-8)
                    IU_loss += layer_losses[1] * (1 / self.num_layer + 1e-8)
                    UU_loss += layer_losses[2] * (1 / self.num_layer + 1e-8)
                    II_loss += layer_losses[3] * (1 / self.num_layer + 1e-8)
                    cluster_loss += layer_losses[4] * (1 / self.num_layer + 1e-8)
                 
        return base_loss, UI_loss, IU_loss, UU_loss, II_loss, cluster_loss

                
    
    def LWCKD_PIW(self, batch_user, batch_pos_item, before_user_emb, user_emb, before_item_emb, item_emb, cluster):
        
        # LWCKD + PIW
        common_batch_user = get_common_ids(batch_user, self.before_user_ids)
        map_common_batch_user = torch.tensor(list(map(lambda x : self.before_user_mapping[x.item()], common_batch_user)))
        
        common_batch_item = get_common_ids(batch_pos_item, self.before_item_ids)
        map_common_batch_item = torch.tensor(list(map(lambda x : self.before_item_mapping[x.item()], common_batch_item)))
        
        batch_user_rating_mat = self.before_rating_mat.to_dense()[map_common_batch_user].to(self.gpu)
        batch_item_rating_mat = self.before_rating_mat.to_dense().t()[map_common_batch_item].to(self.gpu)
        
        batch_UU = self.UU.to_dense()[map_common_batch_user].to(self.gpu)
        batch_II = self.II.to_dense()[map_common_batch_item].to(self.gpu)
        
        batch_before_user_emb = before_user_emb[map_common_batch_user]
        batch_before_item_emb = before_item_emb[map_common_batch_item]
        
        present_common_user_emb = user_emb[self.before_user_ids]
        present_common_item_emb = item_emb[self.before_item_ids]

        UI_loss = self.LWCKD(batch_before_user_emb, present_common_item_emb, batch_user_rating_mat)
        IU_loss = self.LWCKD(batch_before_item_emb, present_common_user_emb, batch_item_rating_mat)
        UU_loss = self.LWCKD(batch_before_user_emb, present_common_user_emb, batch_UU)
        II_loss = self.LWCKD(batch_before_item_emb, present_common_item_emb, batch_II)

        if self.PIW_flag:
            batch_present_user_emb = user_emb[common_batch_user]
            before_G = self.get_G(batch_before_user_emb, cluster)
            present_G = self.get_G(batch_present_user_emb, cluster)
            State_Vector = (before_G - present_G)**2
            batch_PIW = self.PSV(State_Vector)
            batch_PIW = (batch_PIW / (batch_PIW.sum() + 1e-8)) * batch_PIW.size(0) # Scaling
            
            UI_loss = UI_loss * batch_PIW
            UU_loss = UU_loss * batch_PIW
            
        UI_loss = -torch.mean(UI_loss)
        IU_loss = -torch.mean(IU_loss)
        UU_loss = -torch.mean(UU_loss)
        II_loss = -torch.mean(II_loss)
        cluster_loss = self.get_cluster_loss(item_emb, cluster)
                    
        return UI_loss, IU_loss, UU_loss, II_loss, cluster_loss
        
    def get_mask_for_before_ids(self, batch, before_ids):
        mask = torch.eq(batch.unsqueeze(0), before_ids.unsqueeze(1)).any(dim = 0)
        return mask
    
    def LWCKD(self, target_emb, neighbor_emb, rating_mat):
        
        exp = torch.exp(target_emb @ neighbor_emb.t() / (self.temperature)) # We recommend you to increase the temperature > 1 if "nan" occurs. 
        log = torch.log(exp / exp.sum(dim = 1, keepdim = True))
        loss = torch.divide(torch.sum(log * rating_mat, dim = 1), torch.sum(rating_mat, dim = 1) + 1e-8) # interaction이 있는 애들만 고려.

        # DEBUG
        nan_flag = False
        if torch.isnan(target_emb).sum() > 0:
            print("[NAN] target_emb")
            nan_flag = True
        
        if torch.isnan(neighbor_emb).sum() > 0:
            print("[NAN] neighbor_emb")
            nan_flag = True

        if torch.isnan(exp).sum() > 0:
            print("[NAN] exp")
            nan_flag = True

        if torch.isnan(log).sum() > 0:
            print("[NAN] log")
            nan_flag = True

        if nan_flag:
            print("[TYPE]", type)
            raise ValueError("NAN")
                
        return loss

    def get_PIW(self, masked_batch_user, origin_masked_batch_user):
       
        # before / current user embedding
        before_batch_user_emb = self.before_user_emb[masked_batch_user]
        present_batch_user_emb = self.model.user_emb.weight[origin_masked_batch_user]
        
        # modeling for PIW
        G_before = self.get_G(before_batch_user_emb, self.present_cluster_center)
        G_present = self.get_G(present_batch_user_emb, self.present_cluster_center)
        State_Vector = (G_before - G_present)*(G_before - G_present)
        
        batch_PIW = self.PSV(State_Vector)
        return batch_PIW

    def get_G(self, user_embs, center):
        
        G_ = user_embs @ (center.unsqueeze(1) @ self.CSTM).squeeze().T
        G = F.softmax(G_, dim = 1)
        
        return G

    def update(self, 
               before_user_ids, before_item_ids, 
               before_user_mapping, before_item_mapping,
               before_rating_mat, num_new_user, num_new_item, 
               UU = None, II = None,
               present_user_ids = None, present_item_ids = None, R = None, random_init = True, SNM = None, topk = 5, only_one_hop = False):
        

        user_emb = self.base_model.user_emb.weight.detach().cpu()
        item_emb = self.base_model.item_emb.weight.detach().cpu()
        
        if random_init:
            print("\nRandom Initatlized Embedding")
            new_user_embedding = nn.Parameter(torch.randn(num_new_user, self.dim))#.to(self.gpu)
            new_item_embedding = nn.Parameter(torch.randn(num_new_item, self.dim))#.to(self.gpu)
            
            nn.init.normal_(new_user_embedding, mean = 0, std = 0.01)
            nn.init.normal_(new_item_embedding, mean = 0, std = 0.01)
        
        else:
            print(f"\n[No_Random Initatlized Embedding(Our Method)] topk = {topk}")

            end_newuser_id, end_newitem_id = R.shape
            end_newuser_id -= 1
            end_newitem_id -= 1
            start_newuser_id = end_newuser_id - num_new_user + 1
            start_newitem_id = end_newitem_id - num_new_item + 1
            
            # 1 hop (interacted)
            new_user_interactions = R[start_newuser_id:, :start_newitem_id]
            Norm_Mat_for_new_user = new_user_interactions / (new_user_interactions.sum(dim = 1, keepdims = True) + 1e-8)
            one_hop_emb_for_user = Norm_Mat_for_new_user @ item_emb
            
            new_item_interactions = R.T[start_newitem_id:, :start_newuser_id]
            Norm_Mat_for_new_item = new_item_interactions / (new_item_interactions.sum(dim = 1, keepdims = True) + 1e-8)
            one_hop_emb_for_item = Norm_Mat_for_new_item @ user_emb
            
            if only_one_hop:
                print("[Only one hop]")
                new_user_embedding = nn.Parameter(one_hop_emb_for_user)
                new_item_embedding = nn.Parameter(one_hop_emb_for_item)
            
            else:
                print("[One hop and two hop]")
                # 2 hop (represenative)
                R = R.to_sparse()
                two_hop_emb_for_user = get_two_hop_emb(R, start_newuser_id, end_newuser_id, user_emb, self.dim, topk = topk)
                two_hop_emb_for_item = get_two_hop_emb(R.T, start_newitem_id, end_newitem_id, item_emb, self.dim, topk = topk)
                
                # Result
                new_user_embedding = nn.Parameter((one_hop_emb_for_user + two_hop_emb_for_user) / 2)#.to(self.gpu) # num_new_user x dim
                new_item_embedding = nn.Parameter((one_hop_emb_for_item + two_hop_emb_for_item) / 2)#.to(self.gpu) # num_new_item x dim
                
################################################################################################################################################
        
        self.user_PIW.clear()
        
        # assgin
        self.before_user_ids = before_user_ids
        self.before_item_ids = before_item_ids
        self.before_user_mapping = before_user_mapping
        self.before_item_mapping = before_item_mapping
        self.before_rating_mat = before_rating_mat
        self.UU = UU
        self.II = II
        
        # saving the before embedding
        if self.model_type == "BPR":
            self.before_user_emb = self.base_model.user_emb.weight[before_user_ids].detach()
            self.before_item_emb = self.base_model.item_emb.weight[before_item_ids].detach()
            self.before_user_emb.requires_grad = False
            self.before_item_emb.requires_grad = False
            
        elif self.model_type == "LightGCN":
            self.num_layer = self.base_model.num_layer
            self.before_user_emb, self.before_item_emb = self.base_model.get_embedding(return_all = True)
            for i in range(self.num_layer):
                
                self.before_user_emb[i] = self.before_user_emb[i].detach()
                self.before_user_emb[i].requires_grad = False
                
                self.before_item_emb[i] = self.before_item_emb[i].detach()
                self.before_item_emb[i].requires_grad = False
            self.base_model.SNM = SNM
        
        self.base_model.user_emb = nn.Embedding.from_pretrained(torch.cat([user_emb, new_user_embedding])).to(self.gpu)
        self.base_model.item_emb = nn.Embedding.from_pretrained(torch.cat([item_emb, new_item_embedding])).to(self.gpu)
        self.base_model.user_emb.weight.requires_grad = True
        self.base_model.item_emb.weight.requires_grad = True
        
        # update user/item count
        self.base_model.user_count += num_new_user
        self.base_model.item_count += num_new_item
        
        # Clustering(KMeans or DeepCluster) based on before items with before_item_ids
        if self.PIW_flag:
            self.present_user_ids = present_user_ids.to(self.gpu)
            self.present_item_ids = present_item_ids.to(self.gpu)
            self.v = len(present_item_ids) - 1
            
            # Random initalization
            if self.model_type == "BPR":
                cluster_id = present_item_ids[random.sample(range(len(present_item_ids)), self.num_cluster)]
                self.cluster = deepcopy(self.base_model.item_emb.weight[cluster_id].detach())
                self.cluster.requires_grad_(True)
                
            elif self.model_type == "LightGCN":
                self.cluster = []
                for i in range(self.num_layer):
                    cluster_id = before_item_ids[random.sample(range(len(before_item_ids)), self.num_cluster)]
                    cluster = deepcopy(self.before_item_emb[i][cluster_id].detach())
                    cluster.requires_grad = True
                    self.cluster.append(cluster)
                    
    def get_cluster_loss(self, item_emb, cluster):
        present_item_emb = item_emb[self.present_item_ids]
        present_item_emb = present_item_emb.detach()
        present_item_emb.requires_grad = False
        
        Q = torch.pow(1 + (torch.sum((present_item_emb.unsqueeze(1) - cluster)**2, dim = -1) / self.v), -((self.v + 1)/2))
        Q =  Q / (torch.sum(Q, dim = -1, keepdim = True) + 1e-8)# num_item x num_cluster

        P = Q**2 / (torch.sum(Q, dim = 0, keepdim = True) + 1e-8)
        P = P / (torch.sum(P, dim = -1, keepdim = True) + 1e-8)

        P = P.detach()
        P.requires_grad = False # target 
        
        soft_kl_loss = self.kl_loss(torch.log(Q + 1e-8), P)
        return soft_kl_loss
        

class PIW_State_Vector(nn.Module):
    def __init__(self, num_cluser, dim = 16):
        super().__init__()
        self.layer = nn.Sequential(
                                    nn.Linear(num_cluser, dim),
                                    nn.ReLU(),

                                    nn.Linear(dim, 1),
                                    nn.Softplus(),
                                    )
    def forward(self, x):
        return self.layer(x)
    

class CL_VAE(nn.Module):
    def __init__(self, base_model, dim, gpu, CL_flag):
        super().__init__()
        self.base_model = base_model
        self.dim = dim
        self.gpu = gpu
        
        self.before_score_mat = None # tensor (2D)
        self.common_interaction = None # dict
        self.common_user_ids = None # tensor (1D)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean', log_target = True)        
        self.CL_flag = CL_flag
        
    def forward(self, mini_batch):
        
        # CF
        mini_batch = {key: values.to(self.gpu) for key, values in mini_batch.items()} # key: "user" or "rating_vec"
        output = self.base_model.forward(mini_batch)
        base_loss = self.base_model.get_loss(output)
        
        # CL
        total_kl_loss = torch.tensor(0.).to(self.gpu)
        
        if self.CL_flag:
            user_output, _, _, _ = output
            common_batch_user  = get_common_ids(self.common_pos_user_ids, mini_batch['user']) # common_pos_user_ids에서 user는 (common_user + common_item에 의해 선택된 user)의 집합 -> PIW값이 존재하지 않을 수 있다.

            for u in common_batch_user:
                
                i = torch.tensor(self.common_interaction[u.item()]).to(self.gpu)
                indices = torch.where(mini_batch['user'] == u)[0].to(self.gpu)
                
                before_dist = self.before_score_mat[u.detach().cpu()][i.detach().cpu()].to(self.gpu)
                present_dist = user_output[indices].squeeze()[i]
                
                kl_loss = self.kl_loss(present_dist.log() + 1e-8, before_dist.log() + 1e-8)
                if self.task_user_piw_mean and u.item() in self.task_user_piw_mean.keys():
                    piw_value = torch.tensor(self.task_user_piw_mean[u.item()]).to(self.gpu)
                else:
                    piw_value = torch.tensor(1.0).to(self.gpu)
                    
                kl_loss = kl_loss * piw_value
                total_kl_loss += kl_loss
            
            total_kl_loss /= len(common_batch_user)
        
        return base_loss, total_kl_loss
        
    def update(self, num_new_user, num_new_item, before_R, common_interaction, task_user_piw_mean):
        
        if self.CL_flag:
            # save historical info
            self.task_user_piw_mean = task_user_piw_mean
            self.common_interaction = common_interaction
            self.common_pos_user_ids = torch.tensor(list(self.common_interaction.keys())).to(self.gpu)
            
            before_user_count, before_item_count = before_R.shape
            dataset = {"user" : torch.arange(before_user_count).to(self.gpu),
                    "rating_vec" : before_R.to(self.gpu)}
            
            with torch.no_grad():
                self.before_score_mat = self.base_model.forward_eval(dataset)
                self.before_score_mat = self.before_score_mat.detach().cpu()
                self.before_score_mat.requires_grad_(False)
        
        # update new info
        self.base_model.user_count += num_new_user
        self.base_model.item_count += num_new_item
        
        e_extra_layer = nn.Linear(num_new_item, self.dim).to(self.gpu) # Dim x items
        d_extra_layer = nn.Linear(self.dim, num_new_item).to(self.gpu)
        nn.init.xavier_normal_(e_extra_layer.weight.data)
        nn.init.xavier_normal_(d_extra_layer.weight.data)
        nn.init.normal_(d_extra_layer.bias.data)
        
        e_weight = nn.Parameter(torch.cat([self.base_model.encoder.weight.data, e_extra_layer.weight.data], dim = 1)) # Dim x (items + @) (wx + b)
        e_bias = nn.Parameter(self.base_model.encoder.bias.data)
        d_weight = nn.Parameter(torch.cat([self.base_model.decoder.weight.data, d_extra_layer.weight.data], dim = 0)) # (items + @) x Dim (wx + b)
        d_bias = nn.Parameter(torch.cat([self.base_model.decoder.bias.data, d_extra_layer.bias.data]))
        
        self.base_model.encoder = nn.Linear(self.base_model.item_count, self.dim)
        self.base_model.encoder.weight = e_weight
        self.base_model.encoder.bias = e_bias
        
        self.base_model.decoder = nn.Linear(self.dim, self.base_model.item_count).to(self.gpu)
        self.base_model.decoder.weight = d_weight
        self.base_model.decoder.bias = d_bias
        
        del e_extra_layer, d_extra_layer, e_weight, e_bias, d_weight, d_bias
        
class CL_VAE_expand(nn.Module):
    def __init__(self, base_model, dim, gpu, CL_flag = True):
        super().__init__()
        
        self.base_model = base_model
        self.dim = dim
        self.gpu = gpu
        
        self.before_score_mat = None # tensor (2D)
        self.common_interaction = None # dict
        self.common_user_ids = None # tensor (1D)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean', log_target = True)
        
        self.p_total_user = 0
        self.p_total_item = 0
        self.b_total_user = 0
        self.b_total_item = 0
        
        self.CL_flag = CL_flag

    def forward(self, mini_batch):
        
        # CF
        mini_batch = {key: values.to(self.gpu) for key, values in mini_batch.items()} # key: "user" or "rating_vec"
        output = self.base_model.forward(mini_batch)
        base_loss = self.base_model.get_loss(output)
        
        # CL
        total_kl_loss = torch.tensor(0.).to(self.gpu)
        
        if self.CL_flag:
            user_output, _, _, _ = output
            common_batch_user  = get_common_ids(self.common_pos_user_ids, mini_batch['user']) # common_pos_user_ids에서 user는 (common_user + common_item에 의해 선택된 user)의 집합 -> PIW값이 존재하지 않을 수 있다.
            
            for user_id in common_batch_user:
                items_id = torch.tensor(self.common_interaction[user_id.item()]).to(self.gpu)
                user_id_in_batch = torch.where(mini_batch['user'] == user_id)[0].to(self.gpu)
                
                before_dist = self.before_score_mat[user_id][items_id]
                present_dist = user_output[user_id_in_batch].squeeze()[items_id]
                kl_loss = self.kl_loss(present_dist.log() + 1e-8, before_dist.log() + 1e-8)
                total_kl_loss += kl_loss
            total_kl_loss /= len(common_batch_user)
            
        return base_loss, total_kl_loss
        
    def update(self, p_total_user, p_total_item, b_total_user, b_total_item, common_interaction, before_score_mat):
        
        # update new info
        self.p_total_user = p_total_user
        self.p_total_item = p_total_item
        self.b_total_user = b_total_user
        self.b_total_item = b_total_item
        
        if self.CL_flag:
            self.common_interaction = common_interaction # {u1:{i1, i2..}, }
            self.common_pos_user_ids = torch.tensor(list(self.common_interaction.keys())).to(self.gpu) # [u1, u2, ...]
        
        self.before_score_mat = before_score_mat.to(self.gpu)
        self.before_score_mat.requires_grad = False
        
        # update new info
        self.base_model.user_count = p_total_user