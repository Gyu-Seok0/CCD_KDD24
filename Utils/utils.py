import os
import sys
import gc
import random
import copy
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import chain
from collections import defaultdict
from copy import deepcopy
from requests import get
from torch.utils.data import DataLoader

# Custom Models
from Models.BPR import BPR
from Models.LightGCN import LightGCN
from Models.VAE import VAE
from Models.LWCKD import CL_VAE_expand, PIW_LWCKD

# Custom Utils
from Utils.data_loaders import *

def get_rank_score_mat_list(load_path, model_list, task_idx, gpu, p_total_user, p_total_item, 
                            total_train_dataset, total_valid_dataset, total_test_dataset, rank_importance,
                            new_user_train_mat, new_user_valid_mat, new_user_test_mat, k_list, eval_task_idx):
    """ Get the rank score matrix for an ensemble teacher """

    rank_score_mat_list = []
    for m_name in model_list:
        
        model_path = os.path.join(load_path, m_name, f"TASK_{task_idx}.pth")
        print(f"model_name = {m_name} and model_path = {model_path}")
        
        # Score and Sorted mat
        score_mat = torch.load(model_path, map_location = torch.device(gpu))["score_mat"]#.detach().cpu()
        score_mat = score_mat[:p_total_user, :p_total_item]
        sorted_mat = to_np(torch.topk(score_mat, k = 1000).indices.detach().cpu())
        get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, sorted_mat, k_list, eval_task_idx)

        # Evaluation for New users
        print(f"\t[The Result of new users in {task_idx}-th Block]")
        new_user_results = get_eval_with_mat(new_user_train_mat, new_user_valid_mat, new_user_test_mat, sorted_mat, k_list)
        print(f"\tvalid_R20 = {new_user_results['valid']['R20']}, test_R20 = {new_user_results['test']['R20']}")
        
        # Save as rank_score_mat
        rank_score_mat = convert_to_rank_score_mat(score_mat, rank_importance)
        rank_score_mat_list.append(rank_score_mat.detach().cpu())
        
    return rank_score_mat_list

def get_VAE_replay_learning_loader_integrate_with_R(replay_learning_dataset, R, total_user, total_item, args):
    """ If the model is VAE, we use pseudo-labeling by imputing the replay_learning_dataset with args.VAE_replay_learning_value."""
    Integrate_R = torch.zeros(total_user, total_item)
    user_size, item_size = R.shape
    Integrate_R[:user_size, :item_size] = R

    for u, i, _ in replay_learning_dataset:
        Integrate_R[u][i] = max(Integrate_R[u][i], args.VAE_replay_learning_value)

    VAE_train_dataset = implicit_CF_dataset_AE(total_user, total_item, rating_mat = None, is_user_side = True, R = Integrate_R)
    VAE_train_loader = DataLoader(VAE_train_dataset, batch_size = args.bs, shuffle = True, drop_last = False)

    return VAE_train_loader

def get_score_mat_for_VAE(model, train_loader, gpu):
    """ Get the rating score matrix of VAE """
    model = model.to(gpu)
    score_mat = torch.zeros(model.user_count, model.item_count)
    with torch.no_grad():
        for mini_batch in train_loader:
            mini_batch = {key: value.to(gpu) for key, value in mini_batch.items()}
            output = model.forward_eval(mini_batch)
            score_mat[mini_batch['user'], :] = output.cpu()
    return score_mat

def get_info_for_VAE_Teacher(before_user_ids, before_item_ids, present_user_ids, present_item_ids, before_R):
    """ Get the basic information of Teacher(VAE) """
    common_user_ids = get_common_ids(before_user_ids, present_user_ids) # 순서 바뀌어도 상관 없음.
    common_item_ids = get_common_ids(before_item_ids, present_item_ids)
    
    common_R = torch.zeros(before_R.shape)
    common_R[common_user_ids] = before_R[common_user_ids]
    common_R[:, common_item_ids] = before_R[:, common_item_ids]
    
    common_interaction = defaultdict(list) # {u1:{i1, i2..},}
    u_ids, i_ids = torch.where(torch.eq(common_R, 1.))
    for u_id, i_id in zip(u_ids.tolist(), i_ids.tolist()):
        common_interaction[u_id].append(i_id)
        
    return common_interaction

def get_info_for_BPR_Teacher(before_user_ids, before_item_ids, before_train_dict):
    """ Get the basic information of Teacher(BPR) """
    before_user_mapping = {id.item() : idx for idx, id in enumerate(before_user_ids)}
    before_item_mapping = {id.item() : idx for idx, id in enumerate(before_item_ids)}    
    before_rating_mat = make_before_rating_mat(before_train_dict, before_user_mapping, before_item_mapping,
                                                len(before_user_ids), len(before_item_ids)) # applying before user/item mapping for squeezing
    UU, II = get_UU_II_graph(before_rating_mat)
    
    return before_user_mapping, before_item_mapping, before_rating_mat, UU, II

def Teacher_update(model_type, Teacher,
                   b_total_user, b_total_item, b_user_ids, b_item_ids, b_train_dict, 
                   p_total_user, p_total_item, p_R, p_user_ids, p_item_ids,
                   num_new_user, num_new_item, gpu, train_loader, args):
    """ Update Teacher model from before model to present model including new users/items """
    if model_type in ["BPR", "LightGCN"]:
        b_user_mapping, b_item_mapping, b_rating_mat, UU, II = get_info_for_BPR_Teacher(b_user_ids, b_item_ids, b_train_dict)    
        p_SNM = get_SNM(p_total_user, p_total_item, p_R, gpu)

        Teacher = Teacher.to(gpu)
        Teacher.update(b_user_ids, b_item_ids, 
                        b_user_mapping, b_item_mapping,
                        b_rating_mat, num_new_user, num_new_item,
                        UU, II, p_user_ids, p_item_ids, p_R, args.random_init, SNM = p_SNM, topk = args.init_topk,
                        only_one_hop = args.only_one_hop)

        del b_user_mapping, b_item_mapping, b_rating_mat, UU, II, p_SNM
        
        with torch.no_grad():
            T_score_mat, T_sorted_mat = get_sorted_score_mat(Teacher, topk = 1000, return_sorted_mat = True)

    elif model_type == "VAE":
        common_interaction = get_info_for_VAE_Teacher(b_user_ids, b_item_ids, p_user_ids, p_item_ids, b_R)
        Teacher.update(p_total_user, p_total_item, b_total_user, b_total_item, common_interaction, T_score_mat)
        Teacher = Teacher.to(gpu)

        del common_interaction, T_score_mat
        
        with torch.no_grad():
            T_score_mat = get_score_mat_for_VAE(Teacher.base_model, train_loader, gpu).detach().cpu()    
            T_sorted_mat = to_np(torch.topk(T_score_mat, k = 1000).indices)
    
    return Teacher, T_score_mat, T_sorted_mat

def get_teacher_model(model_type, b_total_user, b_total_item, b_R, task_idx, max_item, gpu, args):
    """ Load teacher model"""
    if model_type == "VAE":
        model_args = [b_total_user, max_item, args.td, gpu]
        
    elif model_type == "LightGCN":
        b_SNM = get_SNM(b_total_user, b_total_item, b_R, gpu)
        model_args = [b_total_user, b_total_item, args.td, gpu, b_SNM, args.num_layer, args.using_layer_index] # user_count, item_count, dim, gpu, SNM, num_layer, CF_return_average, RRD_return_average
        
    elif model_type == "BPR":
        model_args = [b_total_user, b_total_item, args.td, gpu]
    
    if task_idx == 1:
        T_weight = None
        if args.dataset == "Yelp":
            T_model_path = f"../ckpt/Yelp/Teacher/base_teacher/{args.teacher}/Base_Model.pth" # m = LightGCN_0, ..., LightGCN_4 (5)
            pth = torch.load(T_model_path, map_location = gpu)
            T_score_mat = pth["score_mat"].detach().cpu()
            T_sorted_mat = to_np(torch.topk(T_score_mat, k = 1000).indices)
            T_base_weight = pth["best_model"] # LightGCN_0
            
            T_base_model = eval(model_type)(*model_args)
            T_base_model.load_state_dict(T_base_weight)
    else:    
        T_model_path = os.path.join(args.T_load_path, args.teacher, f"TASK_{task_idx - 1}.pth")
        
        pth = torch.load(T_model_path, map_location = gpu)
        T_score_mat = pth["score_mat"].detach().cpu()
        T_sorted_mat = to_np(torch.topk(T_score_mat, k = 1000).indices)
        T_weight = pth["best_model"] # LightGCN_0
    
    T_base_model = eval(model_type)(*model_args)
    
    if model_type in ["BPR", "LightGCN"]:
        Teacher = PIW_LWCKD(T_base_model, 
                            LWCKD_flag = True, PIW_flag = True,
                            temperature = args.T, 
                            num_cluster = args.nc,
                            dim = args.td, gpu = gpu, model_type = model_type)
            
    elif model_type == "VAE":
        Teacher = CL_VAE_expand(T_base_model, args.td, gpu)
    
    if T_weight is not None:
        try:
            if type(T_weight) != dict:
                T_weight = T_weight.state_dict()
            Teacher.load_state_dict(T_weight)
            print("Teacher's weight loading Success!")
        except:
            print("Teacher's weight loading Fail!")
            pass
    
    return Teacher, T_sorted_mat

def forward_multi_users(user_emb, item_emb, batch_users, batch_item):
    """ Calculates rating scores given a item and multi users"""

    i = item_emb[batch_item].unsqueeze(-1) # batch_size x dim x 1
    u = user_emb[batch_users] # batch_size x users x dim
    score = torch.bmm(u, i).squeeze() # batch_size x users

    return score

def forward_multi_items(user_emb, item_emb, batch_user, batch_items):
    """ Calculates rating scores given a user and multi items"""

    u = user_emb[batch_user].unsqueeze(-1) # batch_size x dim x 1
    i = item_emb[batch_items] # batch_size x items x dim
    score = torch.bmm(i, u).squeeze() # batch_size x items

    return score

def get_RRD_IR_BPR_dataset(T_score_mat, negatvie_exclude, p_train_dict, p_train_mat, p_train_interaction, p_total_user, p_total_item, nns, nui, nuu):
    """
    Retrieves RRD, IR, and BPR datasets. 
        * The RRD and IR dataset are the list-wise distillation datasets for users and items, respectively. 
        * The BPR dataset composes pairwise samples in (user, positive item of user, negative item of user).

    Args:
        T_score_mat (torch.Tensor): Rating matrix of the teacher system. shape: (|U|, |I|)
        negatvie_exclude (torch.Tensor): Negatvie exclude data. shape: (p_total_user, n).
        p_train_dict (dict[list]): Dictionary of training data for the current data block.
            Example: {
                        user_1: [item_1, item_2, ..., item_k],
                        ...,
                        user_U: [item_1, item_2, ..., item_k]
                     }
        p_train_mat (dict[dict]): Training matrix in the current data block.
            Example: {user_1: {item_1: 1, ..., item_k: 1}, ..., user_U: {item_1: 1, ..., item_k: 1}}
        p_train_interaction (list): Pairwise training data.
            Example: [(u, pos_i, neg_i), ..., ]
        p_total_user (int): The last user_id in the current data block.
        p_total_item (int): The last item_id in the current data block.
        nns (int): The number of negative samples.
        nui (int): The number of uninterested items.
        nuu (int): The number of uninterested users.
    
    Returns:
        Tuple: RRD_train_dataset, IR_reg_train_dataset, BPR_train_dataset
    """

    # Filtering
    T_score_mat_for_RRD = deepcopy(T_score_mat)

    for user, items in p_train_dict.items(): # train data
        for item in items:
            T_score_mat_for_RRD[user][item] = torch.nan
    
    T_RRD_interesting_items = torch.topk(T_score_mat_for_RRD, k = 40, dim = 1).indices
    T_RRD_interesting_users = torch.topk(T_score_mat_for_RRD.T, k = 40, dim = 1).indices
    
    for user, items in enumerate(negatvie_exclude): # negative
        T_score_mat_for_RRD[user][items.long()] = torch.nan
        
    for user, items in enumerate(T_RRD_interesting_items): # RRD
        T_score_mat_for_RRD[user][items] = torch.nan
        
    for item, users in enumerate(T_RRD_interesting_users): # IR_RRD
        T_score_mat_for_RRD[users, item] = torch.nan
        
    T_score_mat_for_RRD = torch.where(torch.isnan(T_score_mat_for_RRD), 0.0, 1.0)
    
    # RRD dataset
    RRD_train_dataset = RRD_dataset_simple(T_RRD_interesting_items, T_score_mat_for_RRD, num_uninteresting_items = nui)
    IR_reg_train_dataset = IR_RRD_dataset_simple(T_RRD_interesting_users, T_score_mat_for_RRD.t(), num_uninteresting_users = nuu)    
    
    negatvie_exclude = torch.cat([negatvie_exclude, T_RRD_interesting_items], dim = 1)

    # CF dataset
    BPR_train_dataset = implicit_CF_dataset(p_total_user, p_total_item, p_train_mat, nns, p_train_interaction, negatvie_exclude)
    
    return RRD_train_dataset, IR_reg_train_dataset, BPR_train_dataset

def get_interesting_items_after_filtering(score_mat, filter_dict, item_side = False):
    score_mat = filtering_simple(score_mat, filter_dict)
    if item_side:
        interesting_things = torch.topk(score_mat.T, k = 40, dim = 1).indices
    else:
        interesting_things = torch.topk(score_mat, k = 40, dim = 1).indices
    return interesting_things

def expand_mat(mat, shape, value = torch.nan):
    """
    The function for expanding matrix size into shape
    """
    expand_mat = torch.full((shape), value)
    b_u_size, b_i_size = mat.size()
    expand_mat[:b_u_size, :b_i_size] = mat
    return expand_mat

def get_negative_exclude(distillation_idx, p_train_dict, p_total_user, gpu,
                        load_S_proxy_dir_path, load_P_proxy_dir_path, load_CL_model_dir_path):
    """
    Composes the negative exclude data, which is the most likely in each model (e.g., S/P proxies, and student model).

    Args:
        distillation_idx (int): The index for the current data block.
        p_train_dict (dict[list]): Dictionary of training data for the current data block.
            Example: {
                        user_1: [item_1, item_2, ..., item_k],
                        ...,
                        user_U: [item_1, item_2, ..., item_k]
                     }
        p_total_user (int): The last user_id in the current data block.
        gpu (str): 'cuda' or 'cpu'
        load_S_proxy_dir_path (str): The path for S_proxy.
        load_P_proxy_dir_path (str): The path for P_proxy.
        load_CL_model_dir_path (str): The path for the student model.

    Returns:
        torch.Tensor: The negative exclude data.
    """
    
    negatvie_exclude = torch.empty((p_total_user, 0))
    
    # negative exclude
    if distillation_idx > 0:
        
        # S_proxy
        S_proxy_task_path = os.path.join(load_S_proxy_dir_path, f"TASK_{distillation_idx-1}.pth")
        S_score_mat = torch.load(S_proxy_task_path, map_location = gpu)["score_mat"].detach().cpu()
        S_interesting_items = get_interesting_items_after_filtering(S_score_mat, p_train_dict)
        S_interesting_items = expand_mat(S_interesting_items, (p_total_user, 40), -1)
        negatvie_exclude = torch.cat([negatvie_exclude, S_interesting_items], dim = 1)
        del S_score_mat, S_interesting_items
        
        # P_proxy
        if distillation_idx >= 2:
            P_proxy_task_path = os.path.join(load_P_proxy_dir_path, f"TASK_{distillation_idx-1}.pth")
            P_score_mat = torch.load(P_proxy_task_path, map_location = gpu)["score_mat"].detach().cpu()
            P_interesting_items = get_interesting_items_after_filtering(P_score_mat, p_train_dict)
            P_interesting_items = expand_mat(P_interesting_items, (p_total_user, 40), -1)
            negatvie_exclude = torch.cat([negatvie_exclude, P_interesting_items], dim = 1)
            del P_score_mat, P_interesting_items
        
        # Student
        CL_model_task_path = os.path.join(load_CL_model_dir_path, f"TASK_{distillation_idx}.pth")
        CL_score_mat = torch.load(CL_model_task_path, map_location = gpu)["score_mat"].detach().cpu()
        CL_interesting_items = get_interesting_items_after_filtering(CL_score_mat, p_train_dict)
        negatvie_exclude = torch.cat([negatvie_exclude, CL_interesting_items], dim = 1)
        del CL_score_mat, CL_interesting_items
        
    return negatvie_exclude

def get_train_valid_test_mat_for_new_users(b_total_user, p_total_user, p_train_mat, p_valid_mat, p_test_mat):
    """
    Retrieves the train, valid, and test matrices for new users.

    Args:
        b_total_user (int): The last user_id in the previous data block.
        p_total_user (int): The last user_id in the current data block. (e.g., )
        p_train_mat (dict[dict]): Training matrix in the current data block.
            Example: {user_1: {item_1: 1, ..., item_k: 1}, ..., user_U: {item_1: 1, ..., item_k: 1}}
        p_valid_mat (dict[dict]): Validation matrix in the current data block.
            Example: {user_1: {item_1: 1, ..., item_k: 1}, ..., user_U: {item_1: 1, ..., item_k: 1}}
        p_test_mat (dict[dict]): Test matrix in the current data block.
            Example: {user_1: {item_1: 1, ..., item_k: 1}, ..., user_U: {item_1: 1, ..., item_k: 1}}

    Returns:
        tuple: Contains three dictionaries for new users' train, valid, and test matrices.
    """
            
    new_user_train_mat = {}
    new_user_valid_mat = {}
    new_user_test_mat = {}
    
    for new_user_id in range(b_total_user, p_total_user):
        if new_user_id in p_train_mat.keys():
            new_user_train_mat[new_user_id] = p_train_mat[new_user_id]
        if new_user_id in p_valid_mat.keys():
            new_user_valid_mat[new_user_id] = p_valid_mat[new_user_id]
        if new_user_id in p_test_mat.keys():
            new_user_test_mat[new_user_id] = p_test_mat[new_user_id]

    return new_user_train_mat, new_user_valid_mat, new_user_test_mat

def get_train_valid_test_mat(task_idx, total_train_dataset, total_valid_dataset, total_test_dataset):
    """
    Retrieves the train, valid, and test matrices for new users.

    Args:
        task_idx (int): The index for the current data block.
        total_train_dataset (dict[dict[list]]): Training dataset in data blocks.
            Example: {TASK_0: {
                                user_1: [item_1, item_2, ..., item_k], ..., 
                                user_U: [item_1, item_2, ..., item_k]
                                },
                      ...,
                      TASK_5: {
                                user_1: [item_1, item_2, ..., item_k], ..., 
                                user_U: [item_1, item_2, ..., item_k]}
                                }
                    }
        total_valid_dataset (dict[dict[list]]): Validation dataset in data blocks.
        total_test_dataset (dict[dict[list]]): Test dataset in data blocks.
    Returns:
        tuple: Contains one list and three dictionaries for train, valid, and test matrices.
    """
    
    p_train_dict = total_train_dataset[f"TASK_{task_idx}"]
    p_valid_dict = total_valid_dataset[f"TASK_{task_idx}"]
    p_test_dict = total_test_dataset[f"TASK_{task_idx}"]
    
    train_interaction = make_interaction(p_train_dict) # [(u, i, 1), ..., ]
    train_mat = make_rating_mat(p_train_dict) # {u: {i : 1 for i in the interacted items of user u}, ...}
    valid_mat = make_rating_mat(p_valid_dict)
    test_mat = make_rating_mat(p_test_dict)
    
    return train_interaction, train_mat, valid_mat, test_mat

def get_model(before_total_user, before_total_item, b_SNM, gpu, args, model_type, model_weight = None):
        
    if model_type == "LightGCN":
        model_args = [before_total_user, before_total_item, args.sd, gpu, b_SNM, args.num_layer, args.using_layer_index] # user_count, item_count, dim, gpu, SNM, num_layer, CF_return_average, RRD_return_average
        
    elif model_type == "BPR":
        model_args = [before_total_user, before_total_item, args.sd, gpu]
    
    base_model = eval(model_type)(*model_args)
    base_model = base_model.to(gpu)

    model = PIW_LWCKD(base_model,
                        LWCKD_flag = False, PIW_flag = False,
                        temperature = args.T, num_cluster = args.nc,
                        dim = args.sd, gpu = gpu, model_type = model_type)
    
    if model_weight is not None:
        model.load_state_dict(model_weight)
    
    return model
    
def get_replay_learning_rank_dataset(T_rank_mat, rank_mat, sig_mat, args, num_replay_learning_sample):
    
    # Sampling
    user_size, item_size = rank_mat.shape
    prob_mat_for_replay_learning = torch.exp((T_rank_mat[:user_size, :item_size] - rank_mat) * args.eps)

    prob_mat_for_replay_learning = prob_mat_for_replay_learning * (rank_mat < args.absolute)
    items_for_replay_learning = torch.multinomial(prob_mat_for_replay_learning, num_replay_learning_sample)
    
    # Saving
    Dataset_for_replay_learning = []

    for u, items in enumerate(items_for_replay_learning):
        rating = sig_mat[u][items]
        Dataset_for_replay_learning += list(zip([u] * len(rating) , items.tolist(), rating.tolist()))
        
    return Dataset_for_replay_learning


def get_total_replay_learning_dataset_Teacher(T_score_mat, S_score_mat, S_rank_mat, P_score_mat, P_rank_mat, CL_score_mat, CL_rank_mat, args):
    
    print("\n[Get_total_replay_learning_dataset for Teacher]")
    
    T_rank_mat = convert_to_rank_mat(T_score_mat)
    
    S_replay_learning_dataset, P_replay_learning_dataset, CL_replay_learning_dataset = [], [], []
    
    if S_score_mat is not None:
        S_replay_learning_dataset = get_replay_learning_rank_dataset(T_rank_mat, S_score_mat, S_rank_mat, args, args.S_sample)
        
    if P_score_mat is not None:
        P_replay_learning_dataset = get_replay_learning_rank_dataset(T_rank_mat, P_score_mat, P_rank_mat, args, args.P_sample)
        
    if CL_score_mat is not None:
        CL_replay_learning_dataset = get_replay_learning_rank_dataset(T_rank_mat, CL_score_mat, CL_rank_mat, args, args.CL_sample)
        
    print("\tS_replay_learning_dataset", len(S_replay_learning_dataset))
    print("\tP_replay_learning_dataset", len(P_replay_learning_dataset))
    print("\tCL_replay_learning_dataset", len(CL_replay_learning_dataset))
    
    replay_learning_dataset = S_replay_learning_dataset + P_replay_learning_dataset + CL_replay_learning_dataset
    print(f"\tTotal_replay_learning_dataset Before Filtering = {len(replay_learning_dataset)}")
    
    max_dict = defaultdict(int)
    for u, i, r in replay_learning_dataset:
        max_dict[(u, i)] = max(max_dict[(u, i)], r)
    
    replay_learning_dataset = [(ui[0], ui[1], r) for ui, r in max_dict.items()]
    print(f"\tTotal_replay_learning_dataset After Filtering = {len(replay_learning_dataset)}")
    
    return replay_learning_dataset

def get_total_replay_learning_dataset(W_score_mat, S_rank_mat, S_sig_mat, P_rank_mat, P_sig_mat, args):
    
    print("\n[Get_total_replay_learning_dataset for Student]")
    
    W_rank_mat = convert_to_rank_mat(W_score_mat)
    
    S_replay_learning_dataset, P_replay_learning_dataset = [], []
    
    if S_rank_mat is not None:
        S_replay_learning_dataset = get_replay_learning_rank_dataset(W_rank_mat, S_rank_mat, S_sig_mat, args, args.S_sample)

    if P_rank_mat is not None:
        P_replay_learning_dataset = get_replay_learning_rank_dataset(W_rank_mat, P_rank_mat, P_sig_mat, args, args.P_sample)

    print(f"\tS_replay_learning_dataset = {len(S_replay_learning_dataset)}")
    print(f"\tP_replay_learning_dataset = {len(P_replay_learning_dataset)}")

    replay_learning_dataset = S_replay_learning_dataset + P_replay_learning_dataset
    print(f"\tTotal_replay_learning_dataset Before Filtering = {len(replay_learning_dataset)}")
    
    max_dict = defaultdict(int)
    for u, i, r in replay_learning_dataset:
        max_dict[(u, i)] = max(max_dict[(u, i)], r)
    
    replay_learning_dataset = [(ui[0], ui[1], r) for ui, r in max_dict.items()]
    print(f"\tTotal_replay_learning_dataset After Filtering = {len(replay_learning_dataset)}")

    return replay_learning_dataset

def get_SNM(total_user, total_item, R, gpu):
    """Get the symmetrically normalized matrix for LightGCN"""
    Zero_top = torch.zeros(total_user, total_user)
    Zero_under = torch.zeros(total_item, total_item)
    upper = torch.cat([Zero_top, R], dim = 1)
    lower = torch.cat([R.T, Zero_under], dim = 1)
    Adj_mat = torch.cat([upper,lower])
    Adj_mat = Adj_mat.to_sparse().to(gpu)

    interactions = torch.cat([torch.sum(R,dim = 1), torch.sum(R,dim = 0)])
    D = torch.diag(interactions)
    half_D = torch.sqrt(1/D)
    half_D[half_D == float("inf")] = 0
    half_D = half_D.to_sparse().to(gpu)
    
    SNM = torch.spmm(torch.spmm(half_D, Adj_mat), half_D).detach()
    SNM.requires_grad = False

    del Adj_mat, D, half_D
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return SNM

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
def load_saved_model(path, gpu):
    pth = torch.load(path, map_location = gpu)
    model = pth["best_model"]
    score_mat = pth["score_mat"].detach().cpu()
    sorted_mat = to_np(torch.topk(score_mat, k = 1000).indices)
    return model, score_mat, sorted_mat

def load_data_as_dict(data_dict_path, num_task = 6):
    total_train_dataset = dict()
    total_valid_dataset = dict()
    total_test_dataset = dict()
    total_item_list = dict()
    
    for task_idx in range(num_task):
        task_data_dict_path = os.path.join(data_dict_path, f"TASK_{task_idx}.pickle")
        task_data = load_pickle(task_data_dict_path)
        total_train_dataset[f"TASK_{task_idx}"] = task_data["train_dict"]
        total_valid_dataset[f"TASK_{task_idx}"] = task_data["valid_dict"]
        total_test_dataset[f"TASK_{task_idx}"] = task_data["test_dict"]
        total_item_list[f"TASK_{task_idx}"] = task_data["item_list"]
    
    return total_train_dataset, total_valid_dataset, total_test_dataset, total_item_list
    
def get_average_score(score_list, target):
    score = 0
    for b_score_list in score_list:
        score += b_score_list[target]
    score = score / len(score_list)
    score = round(score, 4)
    return score

def score2sorted(score_mat, topk = 1000):
    score_mat = score_mat.detach().cpu()
    sorted_mat = torch.topk(score_mat, k = topk, largest = True).indices
    sorted_mat = sorted_mat.numpy()
    
    return sorted_mat

def get_random_seed():
    rng_state = np.random.get_state()
    current_seed = rng_state[1][0]
    print("current_seed:", current_seed)

def save_pickle(file_path, file):
    with open(file_path,"wb") as f:
        pickle.dump(file, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(file_path):
    with open(file_path,"rb") as f:
        return pickle.load(f)
    
def get_user_item_interact(df:pd.DataFrame) -> defaultdict(list):
    user_item_interact = defaultdict(list)
    for row in df.itertuples():
        user_item_interact[row.user].append(row.item)
    return user_item_interact

def train_valid_test_split(block):

    user_item_interact = get_user_item_interact(block)
    
    test_size = 0.1
    train_dict = defaultdict(list)
    valid_dict = defaultdict(list)
    test_dict = defaultdict(list)

    # item split
    for user in user_item_interact:
        
        items = user_item_interact[user]
        np.random.shuffle(items)
        num_test_items = max(int(len(items) * test_size), 1)
        
        test_items = items[:num_test_items]
        valid_items = items[num_test_items:num_test_items*2]
        train_items = items[num_test_items*2:]
        
        # assign
        test_dict[user] = test_items
        valid_dict[user] = valid_items
        train_dict[user] = train_items
    
    # filtering
    train_mat_R = defaultdict(list)
    for user in train_dict:
        for item in train_dict[user]:
            train_mat_R[item].append(user)
            
    for u in list(valid_dict.keys()):
        for i in list(valid_dict[u]):
            if i not in train_mat_R:
                valid_dict[u].remove(i)
        
        if len(valid_dict[u]) == 0:
            del valid_dict[u]
            del test_dict[u]
    
            
    for u in list(test_dict.keys()):
        for i in list(test_dict[u]):
            if i not in train_mat_R:
                test_dict[u].remove(i)

        if len(test_dict[u]) == 0:
            del valid_dict[u]
            del test_dict[u]

    item_list = list(train_mat_R.keys())

    return train_dict, valid_dict, test_dict, item_list

def sampling_for_graph(graph):
    """
    Sample the neighborhood based on similarity in the graph.

    Args:
        graph (torch.Tensor): The input graph in sparse format.

    Returns:
        torch.Tensor: The sampled graph in sparse format.
    """
    THRESHOLD_MIN = 0.1 + 1e-4
    THRESHOLD_MAX = 0.9 - 1e-4
    TARGET_AVERAGE_DEGREE = 10
    TOLERANCE = 0.6
    MAX_ITERATIONS = 100

    threshold_min = THRESHOLD_MIN
    threshold_max = THRESHOLD_MAX
    
    # Binary Search
    for iteration in range(MAX_ITERATIONS):
        threshold_candidate = (threshold_min + threshold_max) / 2.0
        temp_graph = graph.to_dense() > threshold_candidate
        temp_graph.diagonal().fill_(0)
        temp_graph = temp_graph.to_sparse().float()
        average_degree = torch.sparse.sum(temp_graph) / temp_graph.size(0)

        if abs(average_degree - TARGET_AVERAGE_DEGREE) < TOLERANCE:
            break
        elif average_degree > TARGET_AVERAGE_DEGREE:
            threshold_min = threshold_candidate
        else:
            threshold_max = threshold_candidate
    
    return temp_graph
    
def get_cos_similarity_pair(M):
    norms = torch.norm(M, dim=1, keepdim = True)
    norms = torch.clamp(norms, min = 1e-8)
    norm_M = (M / norms).to_sparse()
    similarity = torch.spmm(norm_M, norm_M.t())
    return similarity

def get_UU_II_graph(R:torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
    
    RF = R.to_dense().to(float)

    print("\nGetting UU and II...(cosine similarity)")
    UU = get_cos_similarity_pair(RF)
    II = get_cos_similarity_pair(RF.T)
    
    print("\nGetting UU and II...(sampling for graph)")
    UU = sampling_for_graph(UU)
    II = sampling_for_graph(II)
        
    return UU, II

def make_before_rating_mat(before_train_dict, before_user_mapping, before_item_mapping, 
                           before_user_size, before_item_size) -> torch.sparse_coo_tensor:
    user_idx = []
    item_idx = []
    
    for u, items in before_train_dict.items():
        if len(items) > 0:
            u = before_user_mapping[u]
            items = list(map(lambda x: before_item_mapping[x], items))
            user_idx += [u] * len(items)
            item_idx += items
            
    value = [1] * len(user_idx)
    mat = torch.sparse_coo_tensor((user_idx, item_idx), value,
                                  (before_user_size, before_item_size))
    
    return mat

def save_before_model(model, m_idx, before_models):
    before_model = deepcopy(model)
    before_model.requires_grad = False
    before_models[m_idx] = before_model
    
def get_common_ids(before_ids, present_ids):
    common_ids_mask = torch.eq(before_ids.unsqueeze(0), present_ids.unsqueeze(1)).any(dim = 0)
    common_ids = before_ids[common_ids_mask]
    return common_ids

def make_interaction(dict):
    interactions = []
    for u, items in dict.items():
        for i in items:
            interactions.append((u, i, 1))
    interactions = list(set(interactions))
    return interactions

def make_rating_mat(dict):
    rating_mat = {}
    for u,items in dict.items():
        u_dict = {i : 1 for i in items}
        rating_mat.update({u : u_dict})
    return rating_mat

def make_R(user_count, item_count, rating_mat):
    R = torch.zeros((user_count, item_count))
    for user in rating_mat:
        items = list(rating_mat[user].keys())
        R[user][items] = 1.
    return R

def print_command_args(args):
    # Get the current datetime
    print("[Time]", str(datetime.now()))
    # ip = get("https://api.ipify.org").text
    # print()
    # print(f"IP = {ip}")
    cmd = "python -u " + " ".join(sys.argv)
    print(f"command = {cmd}")
    print(f"args = {vars(args)}")
    print()

def to_np(x):
    return x.detach().cpu().numpy()

def create_metrics(k_list):
    metrics = {}
    for k in k_list:
        for metric in ["P","R","N"]:
            metrics[f'{metric}{k}'] = []

    eval_results = {'valid' : deepcopy(metrics), 'test': deepcopy(metrics)}
    return eval_results

def get_eval_Ensemble(E_score_mat, total_train_dataset, total_valid_dataset, total_test_dataset, task_idx, k_list, 
             new_user_train_mat, new_user_valid_mat, new_user_test_mat, print_out = "Teacher"):
    
    E_sorted_mat = score2sorted(E_score_mat)
    print(f"\n[Ensemble for {print_out}]")
    get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, E_sorted_mat, k_list, task_idx)
    
    print(f"\t[The Result of new users in {task_idx}-th Block]")
    new_user_results = get_eval_with_mat(new_user_train_mat, new_user_valid_mat, new_user_test_mat, E_sorted_mat, k_list)
    print(f"\tvalid_R20 = {new_user_results['valid']['R20']}, test_R20 = {new_user_results['test']['R20']}")

def get_eval(model, gpu, train_loader, test_dataset, k_list):

    train_mat = train_loader.dataset.rating_mat
    valid_mat = test_dataset.valid_mat
    test_mat = test_dataset.test_mat
    
    max_k = max(k_list)
    eval_results = create_metrics(k_list)
    
    # score_mat
    if model.sim_type == "inner product":
        user_emb, item_emb = model.get_embedding()
        score_mat = torch.matmul(user_emb, item_emb.T)
    
    elif model.sim_type == "UAE":
        score_mat = torch.zeros(model.user_count, model.item_count)
        for mini_batch in train_loader:
            mini_batch = {key: value.to(gpu) for key, value in mini_batch.items()}
            output = model.forward_eval(mini_batch)
            score_mat[mini_batch['user'], :] = output.cpu()
            
    # sorted_mat
    score_mat = score_mat.detach().cpu()
    sorted_mat = torch.topk(score_mat, k = 1000, dim = -1, largest = True).indices
    sorted_mat = to_np(sorted_mat)

    for test_user in test_mat:
        sorted_list = list(sorted_mat[test_user])

        for mode in ["valid",'test']:
            sorted_list_tmp = []
            
            if mode == "valid":
                gt_mat = valid_mat
                already_seen_items = set(train_mat[test_user]) | set(test_mat[test_user].keys())

            elif mode == "test":
                gt_mat = test_mat
                already_seen_items = set(train_mat[test_user]) | set(valid_mat[test_user].keys())

            for item in sorted_list:
                if item not in already_seen_items:
                    sorted_list_tmp.append(item)

                if len(sorted_list_tmp) > max_k: break
            
            for k in k_list:
                hit_k = len(set(sorted_list_tmp[:k]) & set(gt_mat[test_user].keys()))

                # Hit & Recall
                eval_results[mode][f"P{k}"].append(hit_k / min(k, len(gt_mat[test_user].keys())))
                eval_results[mode][f"R{k}"].append(hit_k / len(gt_mat[test_user].keys()))

                # NDCG
                denom = np.log2(np.arange(2, k+2))
                dcg_k = np.sum(np.in1d(sorted_list_tmp[:k], list(gt_mat[test_user].keys())) / denom)
                idcg_k = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), k)])
                NDCG_k = dcg_k / idcg_k
                
                eval_results[mode][f"N{k}"].append(NDCG_k)

    # average
    for mode in ["valid", "test"]:
        for k in k_list:
            eval_results[mode][f"P{k}"] = round(np.mean(eval_results[mode][f"P{k}"]), 4)
            eval_results[mode][f"R{k}"] = round(np.mean(eval_results[mode][f"R{k}"]), 4)
            eval_results[mode][f"N{k}"] = round(np.mean(eval_results[mode][f"N{k}"]), 4)
    
    return eval_results, score_mat, sorted_mat


def get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, sorted_mat, k_list, current_task_idx, 
                  FB_flag = False, return_value = False, max_task = 5):
        
    if return_value:
        valid_list = []
        test_list = []
        
    for before_task_id in range(min(current_task_idx + 1, max_task + 1)):
        if before_task_id > 0 and FB_flag:
            train_dict = merge_train_dict(train_dict, total_train_dataset[f"TASK_{before_task_id}"])
        else:
            train_dict = total_train_dataset[f"TASK_{before_task_id}"]
            
        train_mat = make_rating_mat(train_dict)
        valid_mat = make_rating_mat(total_valid_dataset[f"TASK_{before_task_id}"])
        test_mat = make_rating_mat(total_test_dataset[f"TASK_{before_task_id}"])
        
        results = {}
        results = get_eval_with_mat(train_mat, valid_mat, test_mat, sorted_mat, k_list)
        
        print(f"\n[TASK_ID:{before_task_id}]")
        valid_dict = {f"valid_{key}" : value for key, value in results["valid"].items()}
        test_dict = {f"test_{key}" : value for key, value in results["test"].items()}
        print(valid_dict)
        print(test_dict)
        
        if return_value:
            valid_list.append(valid_dict)
            test_list.append(test_dict)
    
    if return_value:
        return valid_list, test_list

def get_eval_with_mat(train_mat, valid_mat, test_mat, sorted_mat, k_list, target_users = None):
    
    max_k = max(k_list)
    eval_results = create_metrics(k_list)

    if target_users is not None:
        test_users = target_users
    else:
        test_users = list(test_mat.keys())

    for test_user in test_users:
        
        try:
            sorted_list = list(sorted_mat[test_user])
        except:
            continue

        for mode in ["valid",'test']:
            sorted_list_tmp = []
            
            try:
                if mode == "valid":
                    gt_mat = valid_mat
                    already_seen_items = set(train_mat[test_user].keys()) | set(test_mat[test_user].keys())

                elif mode == "test":
                    gt_mat = test_mat
                    already_seen_items = set(train_mat[test_user].keys()) | set(valid_mat[test_user].keys())
            except:
                continue
            
            for item in sorted_list:
                if item not in already_seen_items:
                    sorted_list_tmp.append(item)

                if len(sorted_list_tmp) > max_k: break

            for k in k_list:
                hit_k = len(set(sorted_list_tmp[:k]) & set(gt_mat[test_user].keys()))

                # Hit & Recall
                eval_results[mode][f"P{k}"].append(hit_k / min(k, len(gt_mat[test_user].keys())))
                eval_results[mode][f"R{k}"].append(hit_k / len(gt_mat[test_user].keys()))

                # NDCG
                denom = np.log2(np.arange(2, k+2))
                dcg_k = np.sum(np.in1d(sorted_list_tmp[:k], list(gt_mat[test_user].keys())) / denom)
                idcg_k = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), k)])
                NDCG_k = dcg_k / idcg_k

                eval_results[mode][f"N{k}"].append(NDCG_k)

    # average
    for mode in ["valid", "test"]:
        for k in k_list:
            eval_results[mode][f"P{k}"] = round(np.asarray(eval_results[mode][f"P{k}"]).mean(), 4)
            eval_results[mode][f"R{k}"] = round(np.asarray(eval_results[mode][f"R{k}"]).mean(), 4)
            eval_results[mode][f"N{k}"] = round(np.asarray(eval_results[mode][f"N{k}"]).mean(), 4)
    return eval_results

def train_epoch(train_loader, loss_type, model, model_type, optimizer, scaler, args, gpu, report): #, RRD_flag = False, URRD_lambda = 0.0, before_num_user = None):
    
    train_loader.dataset.negative_sampling()
    epoch_loss = {f"epoch_{l}_loss": 0.0 for l in loss_type}

    for mini_batch in train_loader:
        # forward            
        batch_loss = torch.tensor(0.0).to(gpu)
        
        if model_type in ["BPR", "LightGCN"]:
            base_loss, UI_loss, IU_loss, UU_loss, II_loss, cluster_loss = model(mini_batch)
            batch_loss = base_loss + args.LWCKD_lambda * (UI_loss + IU_loss + UU_loss + II_loss) + (args.cluster_lambda * cluster_loss)
            
        elif model_type == "VAE":
            base_loss, kl_loss = model(mini_batch)
            batch_loss = base_loss + args.kl_lambda * kl_loss
        
        # backward
        optimizer.zero_grad()
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        for l in loss_type:
            epoch_loss[f"epoch_{l}_loss"] += eval(f"{l}_loss").item()
    
    for l in loss_type:
        loss_name = f"epoch_{l}_loss"
        epoch_loss[loss_name] = round(epoch_loss[loss_name] / len(train_loader), 4)
    
    # report
    report.update(epoch_loss)

def train_epoch_base_model(train_loader, loss_type, model, optimizer, scaler, gpu, report):
    
    train_loader.dataset.negative_sampling()
    epoch_loss = {f"epoch_{l}_loss": 0.0 for l in loss_type}

    for mini_batch in train_loader:
        # forward
        base_loss = torch.tensor(0.0).to(gpu)
        mini_batch = {key : values.to(gpu) for key, values in mini_batch.items()}
        output = model.forward(mini_batch)
        base_loss = model.get_loss(output)
            
        # backward
        optimizer.zero_grad()
        scaler.scale(base_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # result save (batch)
        for l in loss_type:
            epoch_loss[f"epoch_{l}_loss"] += eval(f"{l}_loss").item()
            
    # result save (epoch)     
    for l in loss_type:
        loss_name = f"epoch_{l}_loss"
        epoch_loss[loss_name] = round(epoch_loss[loss_name] / len(train_loader), 4)
    
    # report
    report.update(epoch_loss)

def eval_epoch(model, gpu, train_loader, test_dataset, k_list, report, epoch, eval_args, CL_flag = False):

    with torch.no_grad():
        if CL_flag:
            results, score_mat, sorted_mat  = get_eval(model.base_model, gpu, train_loader, test_dataset, k_list)
        else:
            results, score_mat, sorted_mat  = get_eval(model, gpu, train_loader, test_dataset, k_list)
        
        report.update({f"valid_{key}" : value for key, value in results["valid"].items()})
        report.update({f"test_{key}" : value for key, value in results["test"].items()})
        
        if report['valid_R20'] > eval_args["best_score"]:
            eval_args["best_score"] = deepcopy(report['valid_R20'])
            eval_args["test_score"] = deepcopy(report['test_R20'])
            eval_args["best_epoch"] = deepcopy(epoch)
            eval_args["best_model"] = deepcopy(model.state_dict()) #.state_dict()) # detach(), cpu() 필요할듯? 근데 detach()가 에러가나는데 파라미터가 없어서 그런건가? 
            eval_args["score_mat"] = deepcopy(score_mat) #.detach().cpu())
            eval_args["sorted_mat"] = deepcopy(sorted_mat)
            eval_args["patience"] = 0
            
            print(f"[Best Model is changed] best_valid_score = {eval_args['best_score']}, test_score = {eval_args['test_score']}")

            if eval_args["save_flag"] == 1:
                print(f"[Model Saved]")
                torch.save(eval_args, eval_args["save_path"])
        else:
            eval_args["patience"] += 1
            
def get_piw():
    models = ["BPR_0", "BPR_1", "BPR_2", "BPR_3", "BPR_4"]
    root_path = "/home/gslee22/WWW_117/ckpt/New_CL_teacher"
    user_piw_list = {task_idx : defaultdict(list) for task_idx in range(1,6)}
    for m in models:
        path = os.path.join(root_path, m, "User_PIW.pickle")
        with open(path, "rb") as f:
            PIW_dict = pickle.load(f)
        
        for task_idx in range(1,6):
            if task_idx in PIW_dict.keys():
                for u, piw in PIW_dict[task_idx][m].items():
                    user_piw_list[task_idx][u].append(piw)
                    
    user_piw_mean = {task_idx : dict() for task_idx in range(1,6)}
    for task_idx in range(1,6):
        for u, piw_list in user_piw_list[task_idx].items():
            if len(piw_list) > 0:
                user_piw_mean[task_idx][u] = sum(piw_list) / len(piw_list)
            else:
                user_piw_mean[task_idx][u] = 1.0
    
    return user_piw_mean

def convert_to_rank_score_mat(score_mat, rank_importance):
    
    rank = torch.argsort(score_mat, descending = True)
    u_size, i_size = rank.shape
    
    empty = torch.zeros((u_size, i_size))
    rank_value = torch.arange(i_size).float()
    
    for u in range(u_size):
        empty[u][rank[u]] = rank_value

    rank_score_mat  = rank_importance[empty.long()]
    return rank_score_mat

def convert_to_rank_mat(score_mat):
    rank = torch.argsort(score_mat, descending = True)
    u_size, i_size = rank.shape
    
    empty = torch.zeros((u_size, i_size))
    rank_value = torch.arange(i_size).float() # [0, 1, 2, 3]
    
    for u in range(u_size):
        empty[u][rank[u]] = rank_value
    
    return empty.long()

def merge_train_dict(dict1, dict2):
    merget_dict = defaultdict(list)
    
    for k, v in chain(dict1.items(), dict2.items()):
        merget_dict[k].append(v)
        
    for k, v in merget_dict.items():
        merget_dict[k] = sum(v, [])
    
    return merget_dict

def relaxed_ranking_loss(S1, S2):
    """
        The function for the list-wise distillation loss used in DE-RRD and IR-RRD.
        We provide several debug codes.
    """
    
    S1 = torch.clamp_max(S1, max=10)
    S2 = torch.clamp_max(S2, max=10)

    above = S1.sum(1, keepdims=True) # for each user
    below1 = S1.flip(-1).exp().cumsum(1)
    below2 = S2.exp().sum(1, keepdims=True)
    below = (below1 + below2 + 1e-8).log().sum(1, keepdims=True)

    # Debug
    nan_toggle = False
    inf_toggle = False
    
    if torch.isinf(above).sum() >= 1:
        print("[inf]above", above)
        inf_toggle = True
    
    if torch.isinf(below1).sum() >= 1:
        below1[torch.isinf(below1)] = below1[torch.isinf(below1) == False].max()
        print("[inf]below1")

    if torch.isinf(below2).sum() >= 1:
        below2[torch.isinf(below2)] = below2[torch.isinf(below2) == False].max()
        print("[inf]below2")

    if torch.isinf(below).sum() >= 1:
        print("[inf]below", below)
        inf_toggle = True
    
    if torch.isnan(above).sum() >= 1:
        print("[NAN]above", above)
        nan_toggle = True
    
    if torch.isnan(below).sum() >= 1:
        print("[NAN]below", below)
        nan_toggle = True
    
    if nan_toggle or inf_toggle:
        print("EXIT")
        raise AssertionError

    return -(above - below).sum()

def RRD_epoch_only_once(RRD_dataset, before_user_ids, batch_size, gpu, model, URRD_lambda, optimizer, scaler, report):
    """ IR-RRD based on user ids, which uses only once"""

    RRD_dataset.sampling_for_uninteresting_items()    
    epoch_loss = dict()
    epoch_loss["epoch_URRD_loss"] = 0.0
    
    iteration = (len(before_user_ids) // batch_size) + 1
    for idx in range(iteration):
        if idx + 1 == iteration:
            batch_user = before_user_ids[idx * batch_size : ]
        else:
            batch_user = before_user_ids[idx * batch_size : (idx + 1) * batch_size]
            
        with torch.cuda.amp.autocast():

            interesting_items, uninteresting_items = RRD_dataset.get_samples(batch_user)
            
            batch_user = batch_user.to(gpu)
            interesting_items = interesting_items.to(gpu)#.type(torch.cuda.LongTensor)
            uninteresting_items = uninteresting_items.to(gpu)#.type(torch.cuda.LongTensor)
            
            interesting_prediction = model.base_model.forward_multi_items(batch_user, interesting_items)
            uninteresting_prediction = model.base_model.forward_multi_items(batch_user, uninteresting_items)
            URRD_loss = relaxed_ranking_loss(interesting_prediction, uninteresting_prediction)
            
            epoch_loss["epoch_URRD_loss"] += URRD_lambda * URRD_loss.item()
            batch_loss =  URRD_lambda * URRD_loss 
        
        # backward
        optimizer.zero_grad()
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # report
    epoch_loss["epoch_URRD_loss"] = round(epoch_loss["epoch_URRD_loss"] / iteration, 4)
    report.update(epoch_loss)

def RRD_epoch_frequency(RRD_loader, gpu, model, URRD_lambda, optimizer, scaler, report):
    """ RRD based on RRD_loader, reflection on user's frequency without negative samples """

    RRD_loader.dataset.sampling_for_uninteresting_items()    
    epoch_loss = dict()
    epoch_loss["epoch_URRD_loss"] = 0.0
    
    for batch_user in RRD_loader:            
        batch_user = batch_user.unique()
        interesting_items, uninteresting_items = RRD_loader.dataset.get_samples(batch_user)
        
        batch_user = batch_user.to(gpu)
        interesting_items = interesting_items.to(gpu)#.type(torch.cuda.LongTensor)
        uninteresting_items = uninteresting_items.to(gpu)#.type(torch.cuda.LongTensor)
        
        user_emb = model.base_model.user_emb.weight
        item_emb = model.base_model.item_emb.weight
                            
        interesting_prediction = forward_multi_items(user_emb, item_emb, batch_user, interesting_items)
        uninteresting_prediction = forward_multi_items(user_emb, item_emb, batch_user, uninteresting_items)
        
        URRD_loss = relaxed_ranking_loss(interesting_prediction, uninteresting_prediction)
        epoch_loss["epoch_URRD_loss"] += URRD_lambda * URRD_loss.item()
        
        batch_loss =  URRD_lambda * URRD_loss
        
        # backward
        optimizer.zero_grad()
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # report
    epoch_loss["epoch_URRD_loss"] = round(epoch_loss["epoch_URRD_loss"] / len(RRD_loader), 4)
    report.update(epoch_loss)
    

def RRD_epoch_train_loader(train_loader, RRD_loader, gpu, model, URRD_lambda, optimizer, scaler, report):
    """ RRD based on train_loader, reflection on user's frequency including negative samples """
    train_loader.dataset.negative_sampling()
    RRD_loader.dataset.sampling_for_uninteresting_items()
    epoch_loss = dict()
    epoch_loss["epoch_URRD_loss"] = 0.0
    
    for mini_batch in train_loader:

        with torch.cuda.amp.autocast():
            
            batch_user = mini_batch["user"].unique()
            interesting_items, uninteresting_items = RRD_loader.dataset.get_samples(batch_user)
            
            batch_user = batch_user.to(gpu)
            interesting_items = interesting_items.to(gpu)#.type(torch.cuda.LongTensor)
            uninteresting_items = uninteresting_items.to(gpu)#.type(torch.cuda.LongTensor)

            interesting_prediction = model.base_model.forward_multi_items(batch_user, interesting_items)
            uninteresting_prediction = model.base_model.forward_multi_items(batch_user, uninteresting_items)
            
            URRD_loss = relaxed_ranking_loss(interesting_prediction, uninteresting_prediction)
            epoch_loss["epoch_URRD_loss"] += URRD_lambda * URRD_loss.item()
            
            batch_loss =  URRD_lambda * URRD_loss
        
        # backward
        optimizer.zero_grad()
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # report
    epoch_loss["epoch_URRD_loss"] = round(epoch_loss["epoch_URRD_loss"] / len(RRD_loader), 4)
    report.update(epoch_loss)

def IR_RRD(IR_reg_train_dataset, RRD_item_ids, batch_size, gpu, model, IR_reg_lmbda, optimizer, scaler, report):
    """ IR-RRD based on item ids, which uses only once"""
    IR_reg_train_dataset.sampling_for_uninteresting_users()
    epoch_loss = dict()
    epoch_loss["epoch_IR_RRD_loss"] = 0.0
    iteration = (len(RRD_item_ids) // batch_size) + 1
    
    # shuffle
    shuffle_item_ids = RRD_item_ids[torch.randperm(RRD_item_ids.size(0))]
    
    for idx in range(iteration):
        if idx + 1 == iteration:
            batch_item = shuffle_item_ids[idx * batch_size : ]
        else:
            batch_item = shuffle_item_ids[idx * batch_size : (idx + 1) * batch_size]
        
        interesting_users, uninteresting_users = IR_reg_train_dataset.get_samples(batch_item)#.detach().cpu())

        batch_item = batch_item.to(gpu)
        interesting_users = interesting_users.to(gpu)
        uninteresting_users = uninteresting_users.to(gpu)
        
        user_emb = model.base_model.user_emb.weight
        item_emb = model.base_model.item_emb.weight
            
        interesting_user_prediction = forward_multi_users(user_emb, item_emb, interesting_users, batch_item)
        uninteresting_user_prediction = forward_multi_users(user_emb, item_emb, uninteresting_users, batch_item)
        
        IR_reg = relaxed_ranking_loss(interesting_user_prediction, uninteresting_user_prediction)
        
        batch_loss = IR_reg_lmbda * IR_reg
        epoch_loss["epoch_IR_RRD_loss"] += IR_reg_lmbda * IR_reg.item()
            
        # backward
        optimizer.zero_grad()
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss["epoch_IR_RRD_loss"] = round(epoch_loss["epoch_IR_RRD_loss"] / iteration, 4)
        report.update(epoch_loss)
        

def IR_RRD_item_frequency(IR_RRD_loader, gpu, model, IR_reg_lmbda, optimizer, scaler, report):
    """ IR-RRD based on IR_RRD_loader, reflection on item's frequency """
    IR_RRD_loader.dataset.sampling_for_uninteresting_users()
    epoch_loss = dict()
    epoch_loss["epoch_IR_RRD_loss"] = 0.0
    
    for batch_item in IR_RRD_loader:
      
        
        with torch.cuda.amp.autocast():

            interesting_users, uninteresting_users = IR_RRD_loader.dataset.get_samples(batch_item)#.detach().cpu())

            batch_item = batch_item.to(gpu)
            interesting_users = interesting_users.to(gpu)
            uninteresting_users = uninteresting_users.to(gpu)
            
            interesting_user_prediction = model.base_model.forward_multi_users(interesting_users, batch_item)
            uninteresting_user_prediction = model.base_model.forward_multi_users(uninteresting_users, batch_item)

            IR_reg = relaxed_ranking_loss(interesting_user_prediction, uninteresting_user_prediction)
            
            batch_loss = IR_reg_lmbda * IR_reg
            epoch_loss["epoch_IR_RRD_loss"] += IR_reg_lmbda * IR_reg.item()
            
        # backward
        optimizer.zero_grad()
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss["epoch_IR_RRD_loss"] = round(epoch_loss["epoch_IR_RRD_loss"] / len(IR_RRD_loader), 4)
        report.update(epoch_loss)

def merge_model(before_model, present_model, task_idx, wme):
    
    # init
    interpolated_model = deepcopy(present_model)
    
    # user_emb
    b_user_emb = before_model.base_model.user_emb.weight.detach().cpu()
    p_user_emb = present_model.base_model.user_emb.weight.detach().cpu()
    
    if wme:
        bp_user_emb = weighted_merge_emb(b_user_emb, p_user_emb, task_idx)
    else:
        bp_user_emb = merge_emb(b_user_emb, p_user_emb)
    
    # item_emb
    b_item_emb = before_model.base_model.item_emb.weight.detach().cpu()
    p_item_emb = present_model.base_model.item_emb.weight.detach().cpu()
    
    if wme:
        bp_item_emb = weighted_merge_emb(b_item_emb, p_item_emb, task_idx)
    else:
        bp_item_emb = merge_emb(b_item_emb, p_item_emb)
    
    # assign
    interpolated_model.base_model.user_emb.weight = nn.Parameter(bp_user_emb)
    interpolated_model.base_model.item_emb.weight = nn.Parameter(bp_item_emb)
    
    return interpolated_model
    
def merge_emb(b_user_emb, p_user_emb):
    bp_user_emb = torch.full((p_user_emb.shape), torch.nan)
    u_size, emb_size = b_user_emb.size()
    bp_user_emb[:u_size, :emb_size] = b_user_emb
    stacked_tensor = torch.stack([bp_user_emb, p_user_emb])
    mean_matrix = torch.nanmean(stacked_tensor, dim = 0)
    return mean_matrix

def weighted_merge_emb(b_user_emb, p_user_emb, task_idx):
    """ Exponential Moving Average for updating S/P proxies"""
    m = task_idx + 1
    u_size = b_user_emb.size(0)
    
    bp_user_emb = deepcopy(p_user_emb)
    bp_user_emb[:u_size] = ((m-1) / m) * b_user_emb + (1 / m) * bp_user_emb[:u_size]
    return bp_user_emb

def merge_model(before_model, present_model, wme = True, b_weight = None, p_weight = None):
    
    # init
    interpolated_model = deepcopy(present_model)
    
    # user_emb
    b_user_emb = before_model.base_model.user_emb.weight.detach().cpu()
    p_user_emb = present_model.base_model.user_emb.weight.detach().cpu()
    
    if wme:
        bp_user_emb = weighted_merge_emb(b_user_emb, p_user_emb, b_weight, p_weight)
    else:
        bp_user_emb = merge_emb(b_user_emb, p_user_emb)
    
    # item_emb
    b_item_emb = before_model.base_model.item_emb.weight.detach().cpu()
    p_item_emb = present_model.base_model.item_emb.weight.detach().cpu()
    
    if wme:
        bp_item_emb = weighted_merge_emb(b_item_emb, p_item_emb, b_weight, p_weight)
    else:
        bp_item_emb = merge_emb(b_item_emb, p_item_emb)
    
    # assign
    interpolated_model.base_model.user_emb.weight = nn.Parameter(bp_user_emb)
    interpolated_model.base_model.item_emb.weight = nn.Parameter(bp_item_emb)
    
    return interpolated_model
    
def merge_emb(b_user_emb, p_user_emb):
    bp_user_emb = torch.full((p_user_emb.shape), torch.nan)
    u_size, emb_size = b_user_emb.size()
    bp_user_emb[:u_size, :emb_size] = b_user_emb
    stacked_tensor = torch.stack([bp_user_emb, p_user_emb])
    mean_matrix = torch.nanmean(stacked_tensor, dim = 0)
    return mean_matrix

def weighted_merge_emb(b_user_emb, p_user_emb, b_weight, p_weight):
    u_size = b_user_emb.size(0)
    bp_user_emb = deepcopy(p_user_emb)
    bp_user_emb[:u_size] = p_weight * bp_user_emb[:u_size] + b_weight * b_user_emb
    return bp_user_emb

def get_score_mat_for_VAE(model, train_loader, gpu):
    model = model.to(gpu)
    score_mat = torch.zeros(model.user_count, model.item_count)
    with torch.no_grad():
        for mini_batch in train_loader:
            mini_batch = {key: value.to(gpu) for key, value in mini_batch.items()}
            output = model.forward_eval(mini_batch)
            #output = model.forward(mini_batch, return_score = True)
            score_mat[mini_batch['user'], :] = output.cpu()
    return score_mat

def get_sorted_score_mat(model, topk = 1000, return_sorted_mat = False):
    user_emb, item_emb = model.base_model.get_embedding()
    score_mat = user_emb @ item_emb.T
    score_mat = score_mat.detach().cpu()
    
    if return_sorted_mat:
        sorted_mat = torch.topk(score_mat, k = topk, dim = -1, largest = True).indices
        sorted_mat = to_np(sorted_mat)
        return score_mat, sorted_mat
    
    return score_mat

def filtering_simple(mat, filtered_data):
    """ Filter the mat given filtered data (the size of mat > filtered data)"""
    u_size, i_size = mat.shape
    for u, items in filtered_data.items():
        items = torch.tensor(items).long()
        if u < u_size:
            mat[u][items[items < i_size].long()] = -1e8
    return mat

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def save_model(dir_path, task_idx, dict):
    save_path = os.path.join(dir_path, f"TASK_{task_idx}.pth")
    torch.save(dict, save_path)
    print(f"[Model Save Success] save_path = {save_path}")