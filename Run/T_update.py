import argparse
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from copy import deepcopy
import random
import gc

from Utils.data_loaders import *
from Utils.utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from Models.LWCKD import CL_VAE_expand, PIW_LWCKD
from Models.BPR import BPR
from Models.VAE import VAE
from Models.LightGCN_V2 import LightGCN

from time import time

line = "##############################"

def get_score_mat_for_BPR(model):
    user_emb, item_emb = model.get_embedding()
    score_mat = torch.matmul(user_emb, item_emb.T)
    return score_mat

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

def get_info_for_VAE_Teacher(before_user_ids, before_item_ids, present_user_ids, present_item_ids, before_R):
    
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

    before_user_mapping = {id.item() : idx for idx, id in enumerate(before_user_ids)}
    before_item_mapping = {id.item() : idx for idx, id in enumerate(before_item_ids)}    
    before_rating_mat = make_before_rating_mat(before_train_dict, before_user_mapping, before_item_mapping,
                                                len(before_user_ids), len(before_item_ids)) # applying before user/item mapping for squeezing
    UU, II = get_UU_II_graph(before_rating_mat)
    
    return before_user_mapping, before_item_mapping, before_rating_mat, UU, II

def get_train_valid_test_mat(task_idx, total_train_dataset, total_valid_dataset, total_test_dataset):
    p_train_dict = total_train_dataset[f"TASK_{task_idx}"] # {u_1 : {i_1, i_2, i_3}, ..., }
    p_valid_dict = total_valid_dataset[f"TASK_{task_idx}"]
    p_test_dict = total_test_dataset[f"TASK_{task_idx}"]
    
    train_interaction = make_interaction(p_train_dict)
    train_mat = make_rating_mat(p_train_dict)
    valid_mat = make_rating_mat(p_valid_dict)
    test_mat = make_rating_mat(p_test_dict)
    
    return train_interaction, train_mat, valid_mat, test_mat

def get_train_test_dataset(total_user, total_item, train_interaction, train_mat, valid_mat, test_mat, num_negative_sampling):
    
    BPR_train_dataset = implicit_CF_dataset(total_user, total_item, train_mat, num_negative_sampling, train_interaction)
    VAE_train_dataset = implicit_CF_dataset_AE(total_user, total_item, train_mat, is_user_side = True)
    test_dataset = implicit_CF_dataset_test(total_user, total_item, valid_mat, test_mat)
    
    return BPR_train_dataset, VAE_train_dataset, test_dataset

def get_uir_list(data):
    
    u_list, i_list, r_list  = [], [], []
    
    for u, i, r in data:
        u_list.append(u)
        i_list.append(i)
        r_list.append(r)
    
    return u_list, i_list, r_list

def get_uir_list_for_VAE(data):
    
    u_list, i_list, r_list  = [], [], []
    
    for u, i, r in data:
        u_list.append(u)
        i_list.append(i)
        r_list.append(r)
    
    return u_list, i_list, r_list

def get_BD_dataset(T_rank_mat, score_mat, rank_mat, args, num_BD_sample):
                
    # Sampling
    user_size, item_size = rank_mat.shape
    prob_mat_for_BD = torch.exp((T_rank_mat[:user_size, :item_size] - rank_mat) * args.eps)
    
    # rank가 높아야지
    prob_mat_for_BD = prob_mat_for_BD * (rank_mat < args.absolute)
    items_for_BD = torch.multinomial(prob_mat_for_BD, num_BD_sample)
    
    # Saving
    Dataset_for_BD = []

    for u, items in enumerate(items_for_BD):
        rating = score_mat[u][items]
        Dataset_for_BD += list(zip([u] * len(rating) , items.tolist(), rating.tolist()))
        
    return Dataset_for_BD


def get_total_BD_dataset(T_score_mat, S_score_mat, S_rank_mat, P_score_mat, P_rank_mat, CL_score_mat, CL_rank_mat):
    
    print("\n[Get_total_BD_dataset]")
    
    T_rank_mat = convert_to_rank_mat(T_score_mat)
    
    S_BD_dataset, P_BD_dataset, CL_BD_dataset = [], [], []
    
    if S_score_mat is not None:
        S_BD_dataset = get_BD_dataset(T_rank_mat, S_score_mat, S_rank_mat, args, args.S_sample)
        
    if P_score_mat is not None:
        P_BD_dataset = get_BD_dataset(T_rank_mat, P_score_mat, P_rank_mat, args, args.P_sample)
        
    if CL_score_mat is not None:
        CL_BD_dataset = get_BD_dataset(T_rank_mat, CL_score_mat, CL_rank_mat, args, args.CL_sample)
        
    print("\tS_BD_dataset", len(S_BD_dataset))
    print("\tP_BD_dataset", len(P_BD_dataset))
    print("\tCL_BD_dataset", len(CL_BD_dataset))
    
    BD_dataset = S_BD_dataset + P_BD_dataset + CL_BD_dataset
    print(f"\tTotal_BD_dataset Before Filtering = {len(BD_dataset)}")
    
    max_dict = defaultdict(int)
    for u, i, r in BD_dataset:
        max_dict[(u, i)] = max(max_dict[(u, i)], r)
    
    BD_dataset = [(ui[0], ui[1], r) for ui, r in max_dict.items()]
    print(f"\tTotal_BD_dataset After Filtering = {len(BD_dataset)}")
    
    return BD_dataset

def save_model(dir_path, task_idx, dict):
    save_path = os.path.join(dir_path, f"TASK_{task_idx}.pth")
    #model_state = {'best_model' : deepcopy(model.cpu())}
    torch.save(dict, save_path)
    print(f"save_path = {save_path}")
    
def load_saved_model(path, gpu):
    pth = torch.load(path, map_location = gpu)
    model = pth["best_model"]
    score_mat = pth["score_mat"].detach().cpu()
    sorted_mat = to_np(torch.topk(score_mat, k = 1000).indices)
    
    return model, score_mat, sorted_mat

def get_VAE_BD_loader(BD_dataset, total_user, total_item):
    
    VAE_BD_mat = defaultdict(dict)
    for u, i, _ in BD_dataset:
        VAE_BD_mat[u][i] = 0.5
    VAE_train_dataset = implicit_CF_dataset_AE(total_user, total_item, VAE_BD_mat, is_user_side = True)
    VAE_train_loader = DataLoader(VAE_train_dataset, batch_size = args.bs, shuffle = True, drop_last = False)
    
    return VAE_train_loader

def filtering_simple(mat, filtered_data):
    u_size, i_size = mat.shape
    
    for u, items in filtered_data.items():
        items = torch.tensor(items).long()
        if u < u_size:
            mat[u][items[items < i_size].long()] = -1e8
    return mat

def get_VAE_BD_loader_integrate_with_R(BD_dataset, R, total_user, total_item, args):

    Integrate_R = torch.zeros(total_user, total_item)
    user_size, item_size = R.shape
    Integrate_R[:user_size, :item_size] = R

    for u, i, _ in BD_dataset:
        Integrate_R[u][i] = max(Integrate_R[u][i], args.VAE_BD_value)

    VAE_train_dataset = implicit_CF_dataset_AE(total_user, total_item, rating_mat = None, is_user_side = True, R = Integrate_R)
    VAE_train_loader = DataLoader(VAE_train_dataset, batch_size = args.bs, shuffle = True, drop_last = False)

    return VAE_train_loader

def main(args):
    
    gpu = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    scaler = torch.cuda.amp.GradScaler()
    
    assert args.dataset in ["Gowalla", "Yelp"]
    
    print(f"[Student] {args.student}")
    print(f"[Teacher] {args.teacher}")

    data_path = f"../dataset/{args.dataset}/total_blocks_timestamp.pickle"
    data_dict_path = f"../dataset/{args.dataset}"
    
    if args.S_load_path is None:
       args.S_load_path = f"../ckpt/{args.dataset}/Student/{args.student}/Method"
    
    if args.T_load_path is None:
        args.T_load_path = f"../ckpt/{args.dataset}/Teacher/using_{args.student}/Method"
    
    print(f"args.S_load_path = {args.S_load_path}")
    print(f"args.T_load_path = {args.T_load_path}")

    load_D_model_dir_path = f"{args.S_load_path}/Distilled"
    load_S_model_dir_path = f"{args.S_load_path}/Stability"
    load_P_model_dir_path = f"{args.S_load_path}/Plasticity"
    load_CL_model_dir_path = f"{args.S_load_path}/CL"
    RRD_SM_dir_path = f"{args.T_load_path}/Ensemble"
    
    if args.dataset == "Gowalla":
        args.td = 64
        
    elif args.dataset == "Yelp":
        args.td = 128

    # dataset
    total_blocks = load_pickle(data_path)
    max_item = load_pickle(data_path)[-1].item.max() + 1
    total_train_dataset, total_valid_dataset, total_test_dataset, total_item_list = load_data_as_dict(data_dict_path, num_task = args.num_task)
    
    # idx
    task_idx = args.target_task
    distilled_idx = task_idx - 1
    
    p_block = total_blocks[task_idx]
    b_block = total_blocks[task_idx - 1]

    p_total_user = p_block.user.max() + 1
    p_total_item = p_block.item.max() + 1
    b_total_user = b_block.user.max() + 1
    b_total_item = b_block.item.max() + 1
    
    num_new_user = p_total_user - b_total_user
    num_new_item = p_total_item - b_total_item
    
    b_train_dict = total_train_dataset[f"TASK_{task_idx - 1}"] # {u_1 : {i_1, i_2, i_3}, ..., }
    _, b_train_mat, _, _ = get_train_valid_test_mat(task_idx-1, total_train_dataset, total_valid_dataset, total_test_dataset)
    p_train_interaction, p_train_mat, p_valid_mat, p_test_mat = get_train_valid_test_mat(task_idx, total_train_dataset, total_valid_dataset, total_test_dataset)
    
    # 새로운 유저에 대응되는 train/test/valid
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
        
    b_R = make_R(b_total_user, b_total_item, b_train_mat)
    p_R = make_R(p_total_user, p_total_item, p_train_mat)
    
    # LWCKD + PIW / VAE
    p_user_ids = torch.tensor(sorted(p_block.user.unique()))
    p_item_ids = torch.tensor(sorted(p_block.item.unique()))
    b_user_ids = torch.tensor(sorted(list(b_train_dict.keys())))#.to(gpu)
    b_item_ids = torch.tensor(sorted(total_item_list[f"TASK_{task_idx - 1}"]))#.to(gpu)
    
    _, b_user_mapping, b_item_mapping, b_rating_mat, UU, II = None, None, None, None, None, None
    
    print(f"\n{line} Target_Task = {task_idx} (T,S -> T) {line}")
    
    
    #model_list = args.model_list #['BPR_0'] #['BPR_0', 'BPR_1', 'BPR_2', 'BPR_3', 'BPR_4', 'VAE_0', 'VAE_2']
    
    RRD_SM_path = f"{RRD_SM_dir_path}/TASK_{distilled_idx}_score_mat.pth"
    if distilled_idx == 0:
        RRD_SM_path = f"../ckpt/{args.dataset}/Teacher/base_teacher/Ensemble/TASK_0_score_mat.pth"
    T_score_mat = torch.load(RRD_SM_path, map_location = gpu)["score_mat"].detach().cpu() #RRD_SM[f"TASK_{distilled_idx}"]
    FT_score_mat = filtering_simple(T_score_mat, b_train_dict).detach().cpu()
    T_RRD_interesting_items = torch.topk(FT_score_mat, k = 40, dim = 1).indices
    Negatvie_exclude = T_RRD_interesting_items.clone().detach()
    del RRD_SM_path, T_score_mat, FT_score_mat, T_RRD_interesting_items
    
    # Students models
    S_score_mat, P_score_mat, CL_score_mat = None, None, None
    
    #if args.S_model_path != "None":
    if args.Using_S:
        S_model_task_path = os.path.join(load_S_model_dir_path, f"TASK_{distilled_idx}.pth")
        _, S_score_mat, S_sorted_mat = load_saved_model(S_model_task_path, gpu)
        
        print("\n[Evaluation for S_proxy]")
        get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, 
                        S_sorted_mat, args.k_list, current_task_idx = args.num_task, FB_flag = False, return_value = False)
        
        Negatvie_exclude = torch.cat([Negatvie_exclude, torch.tensor(S_sorted_mat[:, :40])], dim = 1)
        del S_sorted_mat

    if (distilled_idx > 0 and args.Using_P) or (distilled_idx == 0 and args.Using_P and args.Using_S != True): #and args.P_model_path != "None":
        P_model_task_path = os.path.join(load_P_model_dir_path, f"TASK_{distilled_idx}.pth")
        _, P_score_mat, P_sorted_mat = load_saved_model(P_model_task_path, gpu)
        
        print("\n[Evaluation for P_proxy]")
        get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, 
                    P_sorted_mat, args.k_list, current_task_idx = args.num_task, FB_flag = False, return_value = False)
        
        Negatvie_exclude = torch.cat([Negatvie_exclude, torch.tensor(P_sorted_mat[:, :40])], dim = 1)
        del P_sorted_mat
        
        if (distilled_idx == 0 and args.Using_P and args.Using_S != True):
            args.P_sample = args.S_sample
        
    #if args.CL_model_path != "None":
    if args.Using_CL:    
        CL_model_task_path = os.path.join(load_CL_model_dir_path, f"TASK_{task_idx}.pth")
        _, CL_score_mat, CL_sorted_mat = load_saved_model(CL_model_task_path, gpu)
        
        print("\n[Evaluation for CL_Student]")
        get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, 
                        CL_sorted_mat, args.k_list, current_task_idx = args.num_task, FB_flag = False, return_value = False)
        
        u_size, i_size = Negatvie_exclude.shape
        Negatvie_exclude_expand = torch.full((p_total_user, i_size), -1.0)
        Negatvie_exclude_expand[:u_size, :i_size] = Negatvie_exclude
        Negatvie_exclude = torch.cat([Negatvie_exclude_expand, torch.tensor(CL_sorted_mat[:, :40])], dim = 1)
        del CL_sorted_mat
            
    model_name, model_seed = args.teacher.split("_")
    set_random_seed(int(model_seed))
    
    if model_name == "VAE":
        model_args = [b_total_user, max_item, args.td, gpu]
        
    elif model_name == "LightGCN":
        b_SNM = get_SNM(b_total_user, b_total_item, b_R, gpu)
        model_args = [b_total_user, b_total_item, args.td, gpu, b_SNM, args.num_layer, args.using_layer_index] # user_count, item_count, dim, gpu, SNM, num_layer, CF_return_average, RRD_return_average
        
    elif model_name == "BPR":
        model_args = [b_total_user, b_total_item, args.td, gpu]
    
    if task_idx == 1:
        T_weight = None
        if args.dataset == "Yelp":
            T_model_path = f"../ckpt/Yelp/Teacher/base_teacher/{args.teacher}/Base_Model.pth" # m = LightGCN_0, ..., LightGCN_4 (5)
            pth = torch.load(T_model_path, map_location = gpu)
            T_score_mat = pth["score_mat"].detach().cpu()
            T_sorted_mat = to_np(torch.topk(T_score_mat, k = 1000).indices)
            T_base_weight = pth["best_model"] # LightGCN_0
            
            T_base_model = eval(model_name)(*model_args)
            T_base_model.load_state_dict(T_base_weight)
            
        # elif args.dataset == "Gowalla":
        #     T_model_path = f"../ckpt/New_CL_teacher/{args.teacher}/TASK_0.pth"
        #     T_weight, T_score_mat, T_sorted_mat = load_saved_model(T_model_path, gpu)
        
    else:    
        T_model_path = os.path.join(args.T_load_path, args.teacher, f"TASK_{task_idx - 1}.pth")
        
        pth = torch.load(T_model_path, map_location = gpu)
        T_score_mat = pth["score_mat"].detach().cpu()
        T_sorted_mat = to_np(torch.topk(T_score_mat, k = 1000).indices)
        T_weight = pth["best_model"] # LightGCN_0
    
    T_base_model = eval(model_name)(*model_args)
    
    if model_name in ["BPR", "LightGCN"]:
        Teacher = PIW_LWCKD(T_base_model, 
                            LWCKD_flag = True, PIW_flag = True,
                            temperature = args.T, 
                            num_cluster = args.nc,
                            dim = args.td, gpu = gpu, model_type = model_name)
        
        train_dataset = implicit_CF_dataset(p_total_user, p_total_item, p_train_mat, args.nns, p_train_interaction, Negatvie_exclude)
    
    elif model_name == "VAE":
        Teacher = CL_VAE_expand(T_base_model, args.td, gpu)
        train_dataset = implicit_CF_dataset_AE(p_total_user, max_item, p_train_mat, is_user_side = True)

    train_loader = DataLoader(train_dataset, batch_size = args.bs, shuffle = True, drop_last = False)
        
    if T_weight is not None:
        try:
            if type(T_weight) != dict:
                T_weight = T_weight.state_dict()
            Teacher.load_state_dict(T_weight) # VAE shape이 안맞네..
            print("Teacher's weight loading Success!")
        except:
            print("Teacher's weight loading Fail!")
            pass
    
    print(f"\n[Teacher] model = {Teacher}")
    print("\n[[Before Update] Evalutation for Teacher]")
    get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, 
                    T_sorted_mat, args.k_list, current_task_idx = args.num_task, FB_flag = False, return_value = False)
    
    # Teacher.update
    if model_name in ["BPR", "LightGCN"]:
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

    elif model_name == "VAE":
        common_interaction = get_info_for_VAE_Teacher(b_user_ids, b_item_ids, p_user_ids, p_item_ids, b_R)
        Teacher.update(p_total_user, p_total_item, b_total_user, b_total_item, common_interaction, T_score_mat)
        Teacher = Teacher.to(gpu)

        del common_interaction, T_score_mat
        
        with torch.no_grad():
            T_score_mat = get_score_mat_for_VAE(Teacher.base_model, train_loader, gpu).detach().cpu()    
            T_sorted_mat = to_np(torch.topk(T_score_mat, k = 1000).indices)

    
    # Teacher Test (Init)
    print("\n[[After Update] Evalutation for Teacher]")
    get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, 
                    T_sorted_mat, args.k_list, current_task_idx = args.num_task, FB_flag = False, return_value = False)
    if args.BD:
        if model_name in ["BPR", "LightGCN"]:
            S_score_mat = torch.sigmoid(S_score_mat) if S_score_mat is not None else None
            P_score_mat = torch.sigmoid(P_score_mat) if P_score_mat is not None else None
            CL_score_mat = torch.sigmoid(CL_score_mat) if CL_score_mat is not None else None
            
        elif model_name == "VAE":
            S_score_mat = F.softmax(S_score_mat, dim=-1) if S_score_mat is not None else None
            P_score_mat = F.softmax(P_score_mat, dim=-1) if P_score_mat is not None else None
            CL_score_mat = F.softmax(CL_score_mat, dim=-1) if CL_score_mat is not None else None
        
        S_rank_mat = convert_to_rank_mat(S_score_mat) if S_score_mat is not None else None
        P_rank_mat = convert_to_rank_mat(P_score_mat) if P_score_mat is not None else None
        CL_rank_mat = convert_to_rank_mat(CL_score_mat) if CL_score_mat is not None else None
        
        BD_dataset = get_total_BD_dataset(T_score_mat, S_score_mat, S_rank_mat, P_score_mat, P_rank_mat, CL_score_mat, CL_rank_mat)
        if model_name == "VAE":
            train_loader =  get_VAE_BD_loader_integrate_with_R(BD_dataset, p_R, p_total_user, max_item, args)                

    if model_name in ["BPR", "LightGCN"]:
        param = [{"params" : Teacher.parameters()}, {"params" : Teacher.cluster}]
        loss_type = ["base", "UI", "IU", "UU", "II", "cluster"]
    elif model_name == "VAE":
        param = Teacher.parameters()
        loss_type = ["base", "kl"]
        
    optimizer = optim.Adam(param, lr = args.lr, weight_decay = args.reg)
    relu = nn.ReLU()
    criterion = nn.BCEWithLogitsLoss(reduction = 'sum')

    eval_args = {"best_score" : 0, "test_score" : 0, "best_epoch" : 0, "best_model" : None,
                    "score_mat" : None,  "sorted_mat" : None, "patience" : 0,  "avg_valid_score" : 0, "avg_test_score": 0}
    total_time = 0
    
    # get gpu memory
    gc.collect()
    torch.cuda.empty_cache()
    
    for epoch in range(args.max_epoch):
        print(f"\n[Epoch:{epoch + 1}/{args.max_epoch}]")
        epoch_loss = {f"epoch_{l}_loss": 0.0 for l in loss_type}
        epoch_BD_loss = 0.0
        start_time = time()
        train_loader.dataset.negative_sampling()

        Teacher.train()
        for mini_batch in train_loader:
            
            # with torch.cuda.amp.autocast():
            if model_name in ["BPR", "LightGCN"]:
                base_loss, UI_loss, IU_loss, UU_loss, II_loss, cluster_loss = Teacher(mini_batch)
                batch_loss = base_loss + args.LWCKD_lambda * (UI_loss + IU_loss + UU_loss + II_loss) + (args.cluster_lambda * cluster_loss)
            
            elif model_name == "VAE":
                base_loss, kl_loss = Teacher(mini_batch)
                batch_loss = base_loss + args.kl_lambda * kl_loss
                
            # Backward
            optimizer.zero_grad()
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            for l in loss_type:
                epoch_loss[f"epoch_{l}_loss"] += eval(f"{l}_loss").item()
        
        for l in loss_type:
            loss_name = f"epoch_{l}_loss"
            epoch_loss[loss_name] = round(epoch_loss[loss_name] / len(train_loader), 4)
        
        CF_time = time()
        print(f"{epoch_loss}, CF_time = {CF_time - start_time:.4f} seconds", end = " ")
                    
############################################ BD ############################################

        if args.BD and model_name in ["BPR", "LightGCN"]:
            if args.annealing:
                BD_lambda = args.BD_lambda * torch.exp(torch.tensor(-epoch)/args.T)
            else:
                BD_lambda = args.BD_lambda
                    
            # Data shuffle
            shuffled_list = random.sample(BD_dataset, len(BD_dataset))
            u_list, i_list, r_list = list(zip(*shuffled_list))
            iteration = (len(shuffled_list) // args.bs) + 1
            epoch_BD_loss = 0.0
                            
            for idx in range(iteration):
                if idx + 1 == iteration:
                    start, end = idx * args.bs , -1
                else:
                    start, end = idx * args.bs , (idx + 1) * args.bs
                
                # with torch.cuda.amp.autocast():
                    
                batch_user = torch.tensor(u_list[start : end], dtype=torch.long).to(gpu)
                batch_item = torch.tensor(i_list[start : end], dtype=torch.long).to(gpu)
                batch_label = torch.tensor(r_list[start : end], dtype=torch.float16).to(gpu)
                
                user_emb, item_emb = Teacher.base_model.get_embedding()
                batch_user_emb = user_emb[batch_user]
                batch_item_emb = item_emb[batch_item]
                
                output = (batch_user_emb * batch_item_emb).sum(1)
                
                if args.correction:
                    batch_loss = relu(output - batch_label).sum()
                else:
                    batch_loss = criterion(output, batch_label)
                batch_loss *= BD_lambda

                # Backward
                optimizer.zero_grad()
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_BD_loss += batch_loss.item()

            BD_time = time()
            print(f"epoch_BD_loss = {round(epoch_BD_loss/iteration, 4)}, BD_lambda = {BD_lambda:.5f}, BD_time = {BD_time - CF_time:.4f} seconds", end = " ")

############################################ EVAL ############################################

        # time
        end_time = time()
        epoch_time = end_time - start_time
        total_time += epoch_time
        print(f"epoch_time = {epoch_time:.4f} seconds, total_time = {total_time:.4f} seconds")

        if epoch % 5 == 0:
            print("\n[Evaluation]")
            Teacher.eval()
            with torch.no_grad():
                if model_name in ["BPR", "LightGCN"]:
                    T_score_mat, T_sorted_mat = get_sorted_score_mat(Teacher, topk = 1000, return_sorted_mat = True)
                
                elif model_name == "VAE":
                    T_score_mat = get_score_mat_for_VAE(Teacher.base_model, train_loader, gpu).detach().cpu()
                    T_sorted_mat = to_np(torch.topk(T_score_mat, k = 1000).indices)
    
            valid_list, test_list = get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, 
                                                    T_sorted_mat, args.k_list, current_task_idx = task_idx, FB_flag = False, return_value = True)
            
            avg_valid_score = get_average_score(valid_list[:task_idx+1], "valid_R20")
            avg_test_score = get_average_score(test_list[:task_idx+1], "test_R20")
            
            if args.eval_average_acc:
                valid_score = avg_valid_score
                test_score =  avg_test_score
            else:
                valid_score = valid_list[task_idx]["valid_R20"]
                test_score = test_list[task_idx]["test_R20"]
                
            print("\n[New user result]")
            new_user_results = get_eval_with_mat(new_user_train_mat, new_user_valid_mat, new_user_test_mat, T_sorted_mat, args.k_list)
            new_user_results_txt = f"valid_R20 = {new_user_results['valid']['R20']}, test_R20 = {new_user_results['test']['R20']}"
            print(f"\t{new_user_results_txt}\n")

            if valid_score > eval_args["best_score"]:
                print(f"[Best Model Changed]\n\tvalid_score = {valid_score:.4f}, test_score = {test_score:.4f}")
                eval_args["best_score"] = valid_score
                eval_args["test_score"] = test_score
                eval_args["avg_valid_score"] = avg_valid_score
                eval_args["avg_test_score"] = avg_test_score
                
                eval_args["best_epoch"] = epoch
                eval_args["best_model"] = deepcopy({k: v.cpu() for k, v in Teacher.state_dict().items()}) #deepcopy(Teacher)
                eval_args["score_mat"] = deepcopy(T_score_mat)
                eval_args["sorted_mat"] = deepcopy(T_sorted_mat)
                eval_args["patience"] = 0
                
                # best_valid_list = deepcopy(valid_list)
                # best_test_list = deepcopy(test_list)
                best_new_user_results = deepcopy(new_user_results_txt)
            else:
                eval_args["patience"] += 1
                if eval_args["patience"] >= args.early_stop:
                    print("[Early Stopping]")
                    break
            
            if args.BD and epoch > 0:
                BD_dataset = get_total_BD_dataset(T_score_mat, S_score_mat, S_rank_mat, P_score_mat, P_rank_mat, CL_score_mat, CL_rank_mat)
                if model_name == "VAE":
                    train_loader =  get_VAE_BD_loader_integrate_with_R(BD_dataset, p_R, p_total_user, max_item, args)  
        
        # get gpu memory
        gc.collect()
        torch.cuda.empty_cache()
    
    # Result
    print(f"\n[Teacher Best_Result for Task_{task_idx}]")
    print(f"best_epoch = {eval_args['best_epoch']}, valid_score = {eval_args['best_score']}, test_score = {eval_args['test_score']}")
    print(f"avg_valid_score = {eval_args['avg_valid_score']}, avg_test_score = {eval_args['avg_test_score']}")
    
    get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, eval_args["sorted_mat"], args.k_list, 
                    current_task_idx = args.num_task, FB_flag = False, return_value = False)
        
    print("\n[New user result]")
    print(f"\t{best_new_user_results}\n")
    
    if args.save:
        print("[Model Save]")
        
        if args.save_path is not None:
            save_path = args.save_path
        else:
            save_path = args.T_load_path
        
        Teacher_dir_path = os.path.join(save_path, args.teacher) # model name
        
        if not os.path.exists(Teacher_dir_path):
            os.makedirs(Teacher_dir_path)
        save_model(Teacher_dir_path, args.target_task, {"best_model" : eval_args["best_model"], "score_mat" : eval_args["score_mat"]})
        
        print("Teacher_dir_path", Teacher_dir_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # data path
    parser.add_argument('--dataset', "--d", type = str, default = None, help = "Gowalla or Yelp")
    
    # model path
    parser.add_argument("--save", "--s", action = argparse.BooleanOptionalAction, help = "whether saving param or not (--s or --no-s)")
    parser.add_argument('--save_path', "--sp", type = str, default = "../ckpt/Yelp/Teacher/using_LightGCN_0/Method_Test")
    parser.add_argument("--S_model_path", type = str, default = "../ckpt/Method/New_Student/Stability")
    parser.add_argument("--P_model_path", type = str, default = "../ckpt/Method/New_Student/Plasticity") 
    parser.add_argument("--CL_model_path", type = str, default = "../ckpt/Method/New_Student/CL")
    parser.add_argument("--RRD_SM_path",  type = str, default = "../ckpt/Method/New_Teacher/Ensemble")
    parser.add_argument("--tcp", type = str, default = "../ckpt/Method/New_Teacher", help = "Teacher Ckpt Path")
    
    # etc
    parser.add_argument("--gpu", type = int, default = 0)
    parser.add_argument('--k_list', type = list, default = [20, 50, 100])
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument("--reg", type = float, default = 0.0001)
    parser.add_argument("--bs", help = "batch_size", type = int, default = 2048)
    parser.add_argument('--max_epoch', type = int, default = 100)
    parser.add_argument('--rs', type = int, default = 1, help = "random seed")
    parser.add_argument('--early_stop', type = int, default = 2)
    parser.add_argument("--nns", help = "the number of negative sample", type = int, default = 1)
    parser.add_argument("--td", help = "teacher embedding dims", type = int, default = 64)

    # LWCKD + PIW
    parser.add_argument("--nc", help = "num_cluster", type = int, default = 10)
    parser.add_argument("--T", help = "temperature", type = int, default = 5.)
    parser.add_argument("--LWCKD_lambda", type = float, default = 0.01)
    parser.add_argument("--cluster_lambda", type = float, default = 1)
    
    # Method
    parser.add_argument("--eps", type = float, default = 0.0001)
    parser.add_argument("--correction", "--c", action = "store_true", default = False, help = "using correction loss")
    parser.add_argument("--annealing", "--a", action = "store_true", default = True, help = "using annealing")
    parser.add_argument('--random_init', "--r", action = "store_true", default = False, help = 'random_initalization for new user/items')
    parser.add_argument("--absolute", type = float, default = 100)
    parser.add_argument("--BD_lambda", type = float, default = 0.5)

    # initalization for new users/items
    parser.add_argument("--init_topk", type = int, default = 20)
    parser.add_argument("--only_one_hop", "--ooh", action = argparse.BooleanOptionalAction, help = "whether saving param or not (--ooh or --no-ooh)")

    # eval
    parser.add_argument("--num_task", type = int, default = 6)
    parser.add_argument("--eval_average_acc", "--eaa", action = argparse.BooleanOptionalAction, help = "whether saving param or not (--eaa or --no-eaa)")
    parser.add_argument("--hyper_param", "--h", action = argparse.BooleanOptionalAction, help = "whether saving param or not (--h or --no-h)")

    # LightGCN
    parser.add_argument('--num_layer', type = int, default = 2)
    parser.add_argument('--using_layer_index', type = str, default = "avg", help = "first, last, avg")
    parser.add_argument("--VAE_BD_value", type = float, default = 0.1)
    parser.add_argument("--kl_lambda", '-kl', type = float, default = 1.0)
    
    # S & P proxies
    parser.add_argument("--target_task", "--tt", help = "target_task", type = int, default = -1)
    parser.add_argument("--BD", action = argparse.BooleanOptionalAction, help = "whether using param or not (--BD or --no-BD)")
    parser.add_argument("--Using_S", "--US", action = argparse.BooleanOptionalAction, help = "whether using stablity proxy or not (--US or --no-US)")
    parser.add_argument("--Using_P", "--UP", action = argparse.BooleanOptionalAction, help = "whether using plasticity proxy or not (--UP or --no-UP)")
    parser.add_argument("--Using_CL", "--UCL", action = argparse.BooleanOptionalAction, help = "whether using student or not (--UCL or --no-UCL)")
    parser.add_argument("--S_sample", "--ss",type = int, default = 5, help = "# for stability proxy")
    parser.add_argument("--P_sample", "--ps", type = int, default = 5, help = "# for plasticity proxy")
    parser.add_argument("--CL_sample", "--cs", type = int, default = 5, help = "# for student")
    parser.add_argument('--S_load_path', "--Slp", type = str, default = None)
    parser.add_argument('--T_load_path', "--Tlp", type = str, default = None)
    parser.add_argument("--student", type = str, default = None, help = "LightGCN_0, BPR_0")
    parser.add_argument("--teacher", type = str, default = None, help = "LightGCN_0, ..., LightGCN_4, BPR_0, ..., BPR_4, VAE_0, VAE_2")
    
    args = parser.parse_args()
    print_command_args(args)
    main(args)
    print_command_args(args)
    
    #python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_1 --tt 1 --BD --UCL --US --UP --ss 1 --ps 3 --cs 5 --s --sp ../ckpt/Yelp/Teacher/using_LightGCN_1/Method_Test --max_epoch 10