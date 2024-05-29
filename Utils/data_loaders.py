import numpy as np
import torch
import torch.utils.data as data

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