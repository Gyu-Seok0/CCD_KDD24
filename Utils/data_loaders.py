import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F 
from Utils.data_utils import *
import numpy as np
from pdb import set_trace as bp
from Utils.utils import *

class implicit_CF_dataset(data.Dataset):
    def __init__(self, user_count, item_count, rating_mat, num_ns, interactions, RRD_interesting_items = None):
        super(implicit_CF_dataset, self).__init__()
        
        self.user_count = user_count
        self.item_count = item_count
        self.rating_mat = rating_mat
        self.interactions = interactions
        self.num_ns = num_ns
        self.RRD_interesting_items = RRD_interesting_items
        self.train_arr = None
        
        if RRD_interesting_items is not None:
            self.num_b_user = RRD_interesting_items.size(0)
        else:
            self.num_b_user = -1
        
    def __len__(self):
        return len(self.interactions) * self.num_ns

    def __getitem__(self, idx):
        
        assert self.train_arr
               
        return {'user': self.train_arr[idx][0], 
                'pos_item': self.train_arr[idx][1], 
                'neg_item': self.train_arr[idx][2]}
    
    def negative_sampling(self):
        
        self.train_arr = []
        sample_list = np.random.choice(list(range(self.item_count)), size = 10 * len(self.interactions) * self.num_ns)
                
        sample_idx = 0
        for user, u_dict in self.rating_mat.items():
            pos_items = list(u_dict.keys()) # u_dict = {item1: 1, item2: 1, ...,}
            
            if self.RRD_interesting_items is not None and user < self.num_b_user:
                filtering_items = list(set(pos_items + self.RRD_interesting_items[user].tolist()))
            else:
                filtering_items = pos_items
        
            for pos_item in pos_items:
                ns_count = 0
                
                while True:
                    neg_item = sample_list[sample_idx]
                    sample_idx += 1
                    
                    if not neg_item in filtering_items:
                        
                        self.train_arr.append((user, pos_item, neg_item))
                        ns_count += 1
                        
                        if ns_count == self.num_ns:
                            break
                        

class implicit_CF_dataset_E_Rec(implicit_CF_dataset):
    def __init__(self, user_count, item_count, rating_mat, num_ns, interactions):
        implicit_CF_dataset.__init__(self, user_count, item_count, rating_mat, num_ns, interactions)

        self.R = torch.zeros((user_count, item_count))
        for user in rating_mat:
            items = list(rating_mat[user].keys())
            self.R[user][items] = 1.		

    def __getitem__(self, idx):

        return {'user': self.train_arr[idx][0], 
                'pos_item': self.train_arr[idx][1], 
                'neg_item': self.train_arr[idx][2],
                }

    def for_AE_batch(self, mini_batch, gpu):

        batch_user = mini_batch['user']
        batch_pos = mini_batch['pos_item']
        batch_neg = mini_batch['neg_item']

        batch_user = to_np(batch_user)
        batch_item = to_np(torch.cat([batch_pos, batch_neg], 0))

        u_value, u_indices = np.unique(batch_user, return_index=True)
        i_value, i_indices = np.unique(batch_item, return_index=True)

        mini_batch['user_AE'] = torch.LongTensor(u_indices).to(gpu)
        mini_batch['item_rating_vec'] = self.R[torch.LongTensor(u_value)].to(gpu)

        mini_batch['item_AE'] = torch.LongTensor(i_indices).to(gpu)
        mini_batch['user_rating_vec'] = self.R.T[torch.LongTensor(i_value)].to(gpu)

        return mini_batch


class implicit_CF_dataset_test(data.Dataset):
    def __init__(self, user_count, item_count, valid_mat, test_mat, batch_size=64):
        super(implicit_CF_dataset_test, self).__init__()

        self.user_count = user_count
        self.item_count = item_count
        self.user_list = torch.LongTensor([i for i in range(user_count)])

        self.valid_mat = valid_mat
        self.test_mat = test_mat
        self.batch_size = batch_size

        self.batch_start = 0

    def get_next_batch_users(self):
        batch_start = self.batch_start
        batch_end = self.batch_start + self.batch_size

        if batch_end >= self.user_count:
            batch_end = self.user_count
            self.batch_start = 0
            return self.user_list[batch_start: batch_end], True
        else:
            self.batch_start += self.batch_size
            return self.user_list[batch_start: batch_end], False

        
class implicit_CF_dataset_AE(data.Dataset):
    def __init__(self, user_count, item_count, rating_mat = None, is_user_side=True, R = None):
        super(implicit_CF_dataset_AE, self).__init__()
        
        self.user_count = user_count
        self.item_count = item_count
        self.rating_mat = rating_mat
        
        if R is not None:
            self.R = R
        else:
            self.R = torch.zeros((user_count, item_count))
            for user in rating_mat:
                items = list(rating_mat[user].keys())
                self.R[user][items] = 1.
                
        assert rating_mat is not None or R is not None
            
        self.is_user_side = is_user_side
        if not is_user_side:
            self.R = self.R.T

    def __len__(self):
        if self.is_user_side:
            return self.user_count
        else: return self.item_count
        

    def __getitem__(self, idx):
        return {'user': idx, 'rating_vec': self.R[idx]}

    def negative_sampling(self):
        pass



class implicit_CF_dataset_mask(data.Dataset):
    def __init__(self, user_count, item_count, rating_mat, num_ns, interactions, exception_interactions=[]):
        super(implicit_CF_dataset_mask, self).__init__()
        
        self.user_count = user_count
        self.item_count = item_count
        self.rating_mat = rating_mat
        self.num_ns = num_ns
        self.interactions = interactions
        self.exception_interactions = exception_interactions

        self.R = torch.zeros((user_count, item_count))
        for user in rating_mat:
            items = list(rating_mat[user].keys())
            self.R[user][items] = 1.

        if len(exception_interactions) > 0:
            self.exception_mat = {}
            for u, i, _ in exception_interactions:
                dict_set(self.exception_mat, u, i, 1)
        
    def negative_sampling(self):
        
        self.train_arr = []
        sample_list = np.random.choice(list(range(self.item_count)), size = 10 * len(self.interactions) * self.num_ns)
        
        sample_idx = 0
        for user, pos_item, _ in self.interactions:
            ns_count = 0
            
            while True:
                neg_item = sample_list[sample_idx]
                if len(self.exception_interactions) > 0:
                    if not is_visited(self.rating_mat, user, neg_item) and not is_visited(self.exception_mat, user, neg_item) :
                        self.train_arr.append((user, pos_item, neg_item))
                        sample_idx += 1
                        ns_count += 1
                        if ns_count == self.num_ns:
                            break
                else:
                    if not is_visited(self.rating_mat, user, neg_item):
                        self.train_arr.append((user, pos_item, neg_item))
                        sample_idx += 1
                        ns_count += 1
                        if ns_count == self.num_ns:
                            break					
                        
                sample_idx += 1
    
    def __len__(self):
        return len(self.interactions) * self.num_ns
        
    def __getitem__(self, idx):

        return {'user': self.train_arr[idx][0], 
                'pos_item': self.train_arr[idx][1], 
                'neg_item': self.train_arr[idx][2]}

    def get_user_side_mask(self, batch_user):
        return torch.index_select(self.R, 0 , batch_user.cpu())


class implicit_CF_dataset_mask_RD(implicit_CF_dataset_mask):
    def __init__(self, user_count, item_count, rating_mat, num_ns, interactions, topk_items, gpu):
        implicit_CF_dataset_mask.__init__(self, user_count, item_count, rating_mat, num_ns, interactions)
    
        self.RD_k = 50
        self.gpu = gpu

        self.dynamic_mask = torch.ones((self.user_count, self.item_count))
        for user, item, _ in self.interactions:  
            self.dynamic_mask[user][item] = 0
        self.dynamic_mask = self.dynamic_mask.to(gpu)
        self.dynamic_mask.requires_grad = False

        self.topk_items = topk_items

    def dynamic_candidates_sampling(self, k=50):
        self.dynamic_candidates = torch.multinomial(self.dynamic_mask, k, replacement=False)
   
    def get_dynamic_samples(self, users):
        return torch.index_select(self.dynamic_candidates, 0, users)

    def get_topk_items(self, users):
        return torch.index_select(self.topk_items, 0, users)  		
  

class implicit_CF_dataset_URRD(data.Dataset):
    def __init__(self, train_dict, num_items, num_neg_sample, 
                 score_mat, interesting_items, 
                 uninteresting_items = None, num_uninteresting_items = 30000, hard_negative = False):
        super().__init__()
        
        self.train_dict = train_dict
        self.train_arr = []

        self.score_mat = score_mat # already masking (observed dataset + interesting_items)
        self.interesting_items = interesting_items
        self.uninteresting_items = uninteresting_items
        self.num_uninteresting_items = num_uninteresting_items

        self.num_neg_sample = num_neg_sample
        self.num_items = num_items
        self.length = 0
        self.hard_negative = hard_negative
        
        for u in list(train_dict.keys()):
            self.length += len(train_dict[u])

    def __len__(self):
        return self.length * self.num_neg_sample
    
    def __getitem__(self, idx):
        assert self.train_arr

        # user, pos_item, neg_item = self.train_arr[idx]
        # return user, pos_item, neg_item
        
        return {'user': self.train_arr[idx][0], 
                'pos_item': self.train_arr[idx][1], 
                'neg_item': self.train_arr[idx][2]}
    
    # per each epoch
    def negative_sampling(self):
        
        print("Negative sampling...")
        print(f"Hard_negative = {self.hard_negative}")
        
        self.train_arr = []

        for u, pos_items in self.train_dict.items():
            
            if self.hard_negative and u < self.interesting_items.size(0) :
                filtering_items = list(set(pos_items + self.interesting_items[u].tolist()))
            else:
                filtering_items = pos_items
            
            for i in pos_items:
                for _ in range(self.num_neg_sample):
                    j = np.random.randint(self.num_items)
                    while j in filtering_items:
                        j = np.random.randint(self.num_items)

                    self.train_arr.append((u,i,j))
                    
        print("[Done] Negative sampling...")

    # per each epoch
    def sampling_for_uninteresting_items(self):
        print(f"Sampling_for_uninteresting_items({self.num_uninteresting_items})...")
  
        self.uninteresting_items = torch.multinomial(self.score_mat, 
                                                     num_samples = self.num_uninteresting_items) # (users, num_uninteresting_items)
        # print("[Done] Sampling_for_uninteresting_items")
        
    def get_samples(self, batch_user):

        interesting_samples = torch.index_select(self.interesting_items, 0, batch_user)
        uninteresting_samples = torch.index_select(self.uninteresting_items, 0, batch_user)

        return interesting_samples, uninteresting_samples

class RRD_dataset(data.Dataset):
    def __init__(self, filtered_dict, num_interesting_items, num_uninteresting_items, score_mat,
                 interesting_items = None):

        super().__init__()
        self.num_uninteresting_items = num_uninteresting_items
        self.score_mat = score_mat # already masking (observed dataset + interesting_items)
        
        if interesting_items is not None:
            self.interesting_items = interesting_items
            
        else:
            for user, items in filtered_dict.items(): # train_dict.items()
                score_mat[user][items] = -1e9          
            self.interesting_items = torch.topk(score_mat, k = num_interesting_items, dim = 1).indices
    
        # masking(interesting_items)
        for user, items in enumerate(self.interesting_items):
            score_mat[user][items] = -1e9
        self.score_mat = torch.where(score_mat <= -1e9, 0.0, 1.0)
        
        #    
        batch_users = []
        for u, pos_items in filtered_dict.items():
            batch_users += [u] * len(pos_items)
        self.batch_users = torch.tensor(batch_users)

    def __len__(self):
        #return self.length
        return len(self.batch_users)
    
    def __getitem__(self, idx):
        batch_user = self.batch_users[idx]
        return batch_user #, interesting_samples, uninteresting_samples
    
    def get_samples(self, batch_user):

        interesting_samples = torch.index_select(self.interesting_items, 0, batch_user)
        uninteresting_samples = torch.index_select(self.uninteresting_items, 0, batch_user)

        return interesting_samples, uninteresting_samples
    

    def sampling_for_uninteresting_items(self):
        print(f"Sampling_for_uninteresting_items({self.num_uninteresting_items})...")
        self.uninteresting_items = torch.multinomial(self.score_mat, 
                                                     num_samples = self.num_uninteresting_items) # (users, num_uninteresting_items)
        # print("[Done] Sampling_for_uninteresting_items")

class implicit_CF_dataset_IR_reg(data.Dataset):
    def __init__(self, filtered_dict, num_interesting_users, num_uninteresting_users, score_mat):
        super().__init__()
        
        self.score_mat = score_mat.t()
        self.num_interesting_users = num_interesting_users
        self.num_uninteresting_users = num_uninteresting_users
        
        for user, items in filtered_dict.items():
            for item in items:
                self.score_mat[item][user] = -1e9
        self.interesting_users = torch.topk(self.score_mat, k = num_interesting_users, dim = 1).indices
                
        for item, users in enumerate(self.interesting_users):
            self.score_mat[item][users] = -1e9
        self.score_mat = torch.where(self.score_mat <= -1e9, 0.0, 1.0)
        
        # IR-RRD frequency testing
        batch_items = []
        for u, pos_items in filtered_dict.items():
            batch_items += pos_items
        self.batch_items = torch.tensor(batch_items)

    def __len__(self):
        return len(self.batch_items)
    
    def __getitem__(self, idx):
        return self.batch_items[idx]
    
    def get_samples(self, batch_item):
        
        interesting_samples = torch.index_select(self.interesting_users, 0, batch_item)
        uninteresting_samples = torch.index_select(self.uninteresting_users, 0, batch_item)

        return interesting_samples, uninteresting_samples
    
    def sampling_for_uninteresting_users(self):
        print(f"Sampling_for_uninteresting_users({self.num_uninteresting_users})...")
        self.uninteresting_users = torch.multinomial(self.score_mat, 
                                                     num_samples = self.num_uninteresting_users) # (users, num_uninteresting_items)

class RRD_dataset_with_nn(data.Dataset): # nn means not negative
    def __init__(self, topk_items, nn_items, b_train_dict, b_total_user, b_total_item, 
                 num_sample_topk = 5, num_sample_nn = 5, num_sample_uninter = 10000): # 이전

        super().__init__()
        score_mat = torch.full((b_total_user, b_total_item), fill_value = 1.0)
        
        # uninteresting_items
        for u in range(b_total_user):
            score_mat[u][topk_items[u]] = 0.0
            score_mat[u][nn_items[u]] = 0.0
            
            if u in b_train_dict.keys():
                score_mat[u][b_train_dict[u]] = 0.0
        
        self.topk_items = topk_items
        self.nn_items = nn_items
        
        self.score_mat = score_mat
        self.b_total_user = b_total_user
        
        self.num_sample_topk = num_sample_topk
        self.num_sample_nn = num_sample_nn
        self.num_sample_uninter = num_sample_uninter

        self.num_topk = topk_items.size(1)
        self.num_nn = nn_items.size(1)

    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
    
    def get_samples(self, batch_user):

        interesting_samples = torch.index_select(self.interesting_items, 0, batch_user)
        uninteresting_samples = torch.index_select(self.uninteresting_items, 0, batch_user)

        return interesting_samples, uninteresting_samples
    
    def sampling(self):
        print(f"[RRD Sampling] num_sample_topk = {self.num_sample_topk}, num_sample_nn = {self.num_sample_nn}, num_sample_uninter({self.num_sample_uninter})...")
        
        interesting_items = []
        
        for u in range(self.b_total_user):
            u_interesting_items = []
            
            sampled_nn_idx = torch.randperm(self.num_nn)[:self.num_sample_nn]
            u_interesting_items += self.nn_items[u][sampled_nn_idx]
            
            sampled_topk_idx = torch.randperm(self.num_topk)[:self.num_sample_topk]
            u_interesting_items += self.topk_items[u][sampled_topk_idx]
            
            interesting_items.append(u_interesting_items)
        
        self.interesting_items = torch.tensor(interesting_items)
        self.uninteresting_items = torch.multinomial(self.score_mat, 
                                                     num_samples = self.num_sample_uninter)
        
        print(f"self.interesting_items = {self.interesting_items.shape}")        
        print(f"self.uninteresting_items = {self.uninteresting_items.shape}")
        
        
        
class BD_dataset(data.Dataset):
    def __init__(self, items, labels):
        super(BD_dataset, self).__init__()
        
        self.items = items
        self.labels = labels
        
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
        
    def get_samples(self, batch_user):

        BD_items = torch.index_select(self.items, 0, batch_user)
        BD_labels = torch.index_select(self.labels, 0, batch_user)

        return BD_items, BD_labels
    
    
    
class RRD_dataset_simple(data.Dataset):
    def __init__(self, interesting_items, score_mat, num_uninteresting_items):
        super().__init__()
        
        self.interesting_items = interesting_items
        self.uninteresting_items = None
        self.score_mat = score_mat # already masking (observed dataset + interesting_items)
        self.num_uninteresting_items = num_uninteresting_items

    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
    
    def get_samples(self, batch_user):

        interesting_samples = torch.index_select(self.interesting_items, 0, batch_user)
        uninteresting_samples = torch.index_select(self.uninteresting_items, 0, batch_user)

        return interesting_samples, uninteresting_samples
    

    def sampling_for_uninteresting_items(self):
        print(f"Sampling_for_uninteresting_items({self.num_uninteresting_items})...")
        self.uninteresting_items = torch.multinomial(self.score_mat, 
                                                     num_samples = self.num_uninteresting_items)

class IR_RRD_dataset_simple(data.Dataset):
    def __init__(self, interesting_users, score_mat, num_uninteresting_users):
        super().__init__()
        
        self.interesting_users = interesting_users
        self.uninteresting_users = None
        self.score_mat = score_mat
        self.num_uninteresting_users = num_uninteresting_users
        
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
    
    def get_samples(self, batch_item):
        
        interesting_samples = torch.index_select(self.interesting_users, 0, batch_item)
        uninteresting_samples = torch.index_select(self.uninteresting_users, 0, batch_item)

        return interesting_samples, uninteresting_samples
    
    def sampling_for_uninteresting_users(self):
        print(f"Sampling_for_uninteresting_users({self.num_uninteresting_users})...")
        self.uninteresting_users = torch.multinomial(self.score_mat, 
                                                     num_samples = self.num_uninteresting_users)