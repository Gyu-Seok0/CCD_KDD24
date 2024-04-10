import pickle
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from requests import get
import sys
import copy
import os
from itertools import chain

# SeongKu's custom
from Models.BPR import BPR
#from Models.LightGCN import LightGCN
from Models.LightGCN_V2 import LightGCN
from Models.VAE import VAE
from Models.LWCKD import PIW_LWCKD, CL_VAE
from datetime import datetime
import random
import gc
from sklearn.metrics.pairwise import cosine_similarity

def get_model(before_total_user, before_total_item, b_SNM, gpu, args, model_type, model_weight):
        
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
        
        model.load_state_dict(model_weight)
        
        return model
    
def get_BD_rank_dataset(T_rank_mat, rank_mat, sig_mat, args, num_BD_sample):
                
    # Sampling
    user_size, item_size = rank_mat.shape
    prob_mat_for_BD = torch.exp((T_rank_mat[:user_size, :item_size] - rank_mat) * args.eps)

    prob_mat_for_BD = prob_mat_for_BD * (rank_mat < args.absolute)
    items_for_BD = torch.multinomial(prob_mat_for_BD, num_BD_sample)
    
    # Saving
    Dataset_for_BD = []

    for u, items in enumerate(items_for_BD):
        rating = sig_mat[u][items]
        Dataset_for_BD += list(zip([u] * len(rating) , items.tolist(), rating.tolist()))
        
    return Dataset_for_BD


def get_total_BD_dataset(W_score_mat, S_rank_mat, S_sig_mat, P_rank_mat, P_sig_mat, args):
    
    print("\n[Get_total_BD_dataset]")
    
    W_rank_mat = convert_to_rank_mat(W_score_mat)
    
    S_BD_dataset, P_BD_dataset = [], []
    
    if S_rank_mat is not None:
        S_BD_dataset = get_BD_rank_dataset(W_rank_mat, S_rank_mat, S_sig_mat, args, args.S_sample)

    if P_rank_mat is not None:
        P_BD_dataset = get_BD_rank_dataset(W_rank_mat, P_rank_mat, P_sig_mat, args, args.P_sample)

    print(f"\tS_BD_dataset = {len(S_BD_dataset)}")
    print(f"\tP_BD_dataset = {len(P_BD_dataset)}")

    BD_dataset = S_BD_dataset + P_BD_dataset
    print(f"\tTotal_BD_dataset Before Filtering = {len(BD_dataset)}")
    
    max_dict = defaultdict(int)
    for u, i, r in BD_dataset:
        max_dict[(u, i)] = max(max_dict[(u, i)], r)
    
    BD_dataset = [(ui[0], ui[1], r) for ui, r in max_dict.items()]
    print(f"\tTotal_BD_dataset After Filtering = {len(BD_dataset)}")

    return BD_dataset

def get_SNM(total_user, total_item, R, gpu):
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
    
    # Random Seed
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


def Sampling_for_graph(Graph):
    threshold_min = 0.1 + 1e-4
    threshold_max = 0.9 - 1e-4
    target_average_degree = 10
    tolerance = 0.6
    
    # Binary Search
    iteration = 0
    while iteration < 100:
        threshold_candidate = (threshold_min + threshold_max) / 2.0
        Temp_Graph = Graph.to_dense() > threshold_candidate
        Temp_Graph.diagonal().fill_(0)
        Temp_Graph = Temp_Graph.to_sparse().float()
        average_degree = torch.sparse.sum(Temp_Graph) / Temp_Graph.size(0)

        if abs(average_degree - target_average_degree) < tolerance:
            break
        elif average_degree > target_average_degree: # degree가 크다 이웃이 많다 threshold를 높여야지
            threshold_min = threshold_candidate
        else:
            threshold_max = threshold_candidate
        iteration += 1
    
    return Temp_Graph
    
def get_cos_similarity_pair(M):
    
    norms = torch.norm(M, dim=1, keepdim = True)
    norms = torch.clamp(norms, min = 1e-8)
    norm_M = (M / norms).to_sparse()
    similarity = torch.spmm(norm_M, norm_M.t())
    
    return similarity

def get_cos_similarity_pair2(M):
    cosine_sim = cosine_similarity(M)
    np.fill_diagonal(cosine_sim, 0.0)
    cosine_sim = torch.tensor(cosine_sim)
    return cosine_sim

def get_UU_II_graph(R:torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
    
    print("\nGetting UU and II...(cosine similarity)")

    RF = R.to_dense().to(float)
    UU = get_cos_similarity_pair(RF)
    II = get_cos_similarity_pair(RF.T)
    
    # RF = R.to_dense().numpy()
    # UU = get_cos_similarity_pair2(RF)
    # II = get_cos_similarity_pair2(RF.T)
    
    print("\nGetting UU and II...(sampling for graph)")

    UU = Sampling_for_graph(UU)
    II = Sampling_for_graph(II)
        
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
    before_model = deepcopy(model) # detach는 그냥 tensor type에 하는거임.
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
        #results = get_eval_result(train_mat, valid_mat, test_mat, sorted_mat)
        
        print(f"\nbefore_task_id = {before_task_id}")
        print(f"intetactions = {len(sum(train_dict.values(), []))}")

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
    #sorted_mat = to_np(sorted_mat)

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
            # eval_results[mode][f"P{k}"] = round(np.mean(eval_results[mode][f"P{k}"]), 4)
            # eval_results[mode][f"R{k}"] = round(np.mean(eval_results[mode][f"R{k}"]), 4)
            # eval_results[mode][f"N{k}"] = round(np.mean(eval_results[mode][f"N{k}"]), 4)
            
            eval_results[mode][f"P{k}"] = round(np.asarray(eval_results[mode][f"P{k}"]).mean(), 4)
            eval_results[mode][f"R{k}"] = round(np.asarray(eval_results[mode][f"R{k}"]).mean(), 4)
            eval_results[mode][f"N{k}"] = round(np.asarray(eval_results[mode][f"N{k}"]).mean(), 4)
    return eval_results

def train_epoch(train_loader, loss_type, model, model_name, optimizer, scaler, args, gpu, report): #, RRD_flag = False, URRD_lambda = 0.0, before_num_user = None):
    
    train_loader.dataset.negative_sampling()
    epoch_loss = {f"epoch_{l}_loss": 0.0 for l in loss_type}
    
    # if RRD_flag:
    #     train_loader.dataset.sampling_for_uninteresting_items()
    #     epoch_loss["epoch_URRD_loss"] = 0.0
    # print("args.LWCKD_lambda", args.LWCKD_lambda)
    for mini_batch in train_loader:
        
        # forward
        #with torch.cuda.amp.autocast():
            
        batch_loss = torch.tensor(0.0).to(gpu)
        
        if model_name in ["BPR", "LightGCN"]:
            base_loss, UI_loss, IU_loss, UU_loss, II_loss, cluster_loss = model(mini_batch)
            batch_loss = base_loss + args.LWCKD_lambda * (UI_loss + IU_loss + UU_loss + II_loss) + (args.cluster_lambda * cluster_loss)
            
        elif model_name == "VAE":
            base_loss, kl_loss = model(mini_batch)
            batch_loss = base_loss + args.kl_lambda * kl_loss
        
        # backward
        optimizer.zero_grad()
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # result save (batch)
        for l in loss_type:
            epoch_loss[f"epoch_{l}_loss"] += eval(f"{l}_loss").item()
    
    # result save (epoch)     
    for l in loss_type:
        loss_name = f"epoch_{l}_loss"
        epoch_loss[loss_name] = round(epoch_loss[loss_name] / len(train_loader), 4)
    
    # if RRD_flag:
    #     epoch_loss["epoch_URRD_loss"] = round(epoch_loss["epoch_URRD_loss"] / len(train_loader), 4)
    
    # report
    report.update(epoch_loss)

def train_epoch_base_model(train_loader, loss_type, model, optimizer, scaler, gpu, report):
    
    train_loader.dataset.negative_sampling()
    epoch_loss = {f"epoch_{l}_loss": 0.0 for l in loss_type}

    for mini_batch in train_loader:
        
        # forward
        #with torch.cuda.amp.autocast():
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


nan_toggle = False
inf_toggle = False

def relaxed_ranking_loss(S1, S2):
    global nan_toggle, inf_toggle
    # 추가
    S1 = torch.clamp_max(S1, max=10)
    S2 = torch.clamp_max(S2, max=10)

    above = S1.sum(1, keepdims=True) # for each user
    if torch.isinf(above).sum() >= 1:
        print("[inf]above", above)
        inf_toggle = True
    
    below1 = S1.flip(-1).exp().cumsum(1)
    
    if torch.isinf(below1).sum() >= 1:
        below1[torch.isinf(below1)] = below1[torch.isinf(below1) == False].max()
        print("[inf]below1")

        #print("[inf]below1", below1)
        #inf_toggle = True
    
    below2 = S2.exp().sum(1, keepdims=True)
    if torch.isinf(below2).sum() >= 1:
        below2[torch.isinf(below2)] = below2[torch.isinf(below2) == False].max()
        print("[inf]below2")

        # print("[inf]below2", below2)
        # inf_toggle = True

    below = (below1 + below2 + 1e-8).log().sum(1, keepdims=True)
    if torch.isinf(below).sum() >= 1:
        print("[inf]below", below)
        inf_toggle = True
    
    #below = (below1 + 1e-8).log().sum(1, keepdims=True)

    if torch.isnan(above).sum() >= 1:
        print("[NAN]above", above)
        nan_toggle = True
    
    if torch.isnan(below).sum() >= 1:
        print("[NAN]below", below)
        nan_toggle = True
    
    if nan_toggle or inf_toggle:
        print("EXIT")
        raise AssertionError

    return -(above - below).sum() # users

# def RRD_epoch(train_loader, gpu, before_num_user, model, URRD_lambda, optimizer, scaler, report):
    
#     train_loader.dataset.negative_sampling()
#     train_loader.dataset.sampling_for_uninteresting_items()    
#     epoch_loss = dict()
#     epoch_loss["epoch_URRD_loss"] = 0.0
    
#     for mini_batch in train_loader:
#         with torch.cuda.amp.autocast():
#             batch_loss = torch.tensor(0.0).to(gpu)

#             batch_user = mini_batch["user"].unique()
#             batch_user = batch_user[batch_user < before_num_user]
            
#             interesting_items, uninteresting_items = train_loader.dataset.get_samples(batch_user) #.detach().cpu())
            
#             batch_user = batch_user.to(gpu)
#             interesting_items = interesting_items.to(gpu).type(torch.cuda.LongTensor)
#             uninteresting_items = uninteresting_items.to(gpu).type(torch.cuda.LongTensor)
            
#             interesting_prediction = model.base_model.forward_multi_items(batch_user, interesting_items)
#             uninteresting_prediction = model.base_model.forward_multi_items(batch_user, uninteresting_items)
            
#             URRD_loss = relaxed_ranking_loss(interesting_prediction, uninteresting_prediction)
#             epoch_loss["epoch_URRD_loss"] += URRD_lambda * URRD_loss.item()
            
#             batch_loss =  URRD_lambda * URRD_loss
        
#         # backward
#         optimizer.zero_grad()
#         scaler.scale(batch_loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
    
#     # report
#     epoch_loss["epoch_URRD_loss"] = round(epoch_loss["epoch_URRD_loss"] / len(train_loader), 4)    
#     report.update(epoch_loss)


# def RRD_epoch_only_once(RRD_dataset, before_user_ids, batch_size, gpu, model, URRD_lambda, optimizer, scaler, report, IR_RRD_flag = None, IR_reg_train_dataset = None, IR_reg_lmbda = None, RRD_item_ids = None):
    
#     RRD_dataset.sampling_for_uninteresting_items()    
#     epoch_loss = dict()
#     epoch_loss["epoch_URRD_loss"] = 0.0
    
#     if IR_RRD_flag:
#         IR_reg_train_dataset.sampling_for_uninteresting_users()
#         epoch_loss["epoch_IR_RRD_loss"] = 0.0

#     iteration = (len(before_user_ids) // batch_size) + 1
#     for idx in range(iteration):
#         if idx + 1 == iteration:
#             batch_user = before_user_ids[idx * batch_size : ]
#         else:
#             batch_user = before_user_ids[idx * batch_size : (idx + 1) * batch_size]
            
#         with torch.cuda.amp.autocast():

#             interesting_items, uninteresting_items = RRD_dataset.get_samples(batch_user)
            
#             batch_user = batch_user.to(gpu)
#             interesting_items = interesting_items.to(gpu)#.type(torch.cuda.LongTensor)
#             uninteresting_items = uninteresting_items.to(gpu)#.type(torch.cuda.LongTensor)
            
#             interesting_prediction = model.base_model.forward_multi_items(batch_user, interesting_items)
#             uninteresting_prediction = model.base_model.forward_multi_items(batch_user, uninteresting_items)
#             URRD_loss = relaxed_ranking_loss(interesting_prediction, uninteresting_prediction)
            
#             epoch_loss["epoch_URRD_loss"] += URRD_lambda * URRD_loss.item()
            
#             if IR_RRD_flag:
                
#                 iteration = (len(RRD_item_ids) // batch_size) + 1
#                 for idx in range(iteration):
#                     if idx + 1 == iteration:
#                         batch_item = RRD_item_ids[idx * batch_size : ]
#                     else:
#                         batch_item = RRD_item_ids[idx * batch_size : (idx + 1) * batch_size]
                    
#                     #batch_item = torch.cat([interesting_items.view((-1,)), uninteresting_items.view((-1,))]).unique()
#                     print("batch_item", batch_item.shape)
                    
#                     interesting_users, uninteresting_users = IR_reg_train_dataset.get_samples(batch_item)#.detach().cpu())
#                     print("interesting_users", interesting_users.shape)
#                     print("uninteresting_users", uninteresting_users.shape)
                    
#                     batch_item = batch_item.to(gpu)
#                     interesting_users = interesting_users.to(gpu)
#                     uninteresting_users = uninteresting_users.to(gpu)
                    
#                     interesting_user_prediction = model.base_model.forward_multi_users(interesting_users, batch_item)
#                     print("interesting_user_prediction", interesting_user_prediction.shape)
                    
#                     uninteresting_user_prediction = model.base_model.forward_multi_users(uninteresting_users, batch_item)
#                     print("uninteresting_user_prediction", uninteresting_user_prediction.shape)

#                     IR_reg = relaxed_ranking_loss(interesting_user_prediction, uninteresting_user_prediction)
                    
#                     epoch_loss["epoch_IR_RRD_loss"] += IR_reg_lmbda * IR_reg.item()

#             batch_loss =  URRD_lambda * URRD_loss + IR_reg_lmbda * IR_reg
        
#         # backward
#         optimizer.zero_grad()
#         scaler.scale(batch_loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
    
#     # report
#     epoch_loss["epoch_URRD_loss"] = round(epoch_loss["epoch_URRD_loss"] / iteration, 4)
#     if IR_RRD_flag:
#         epoch_loss["epoch_IR_RRD_loss"] = round(epoch_loss["epoch_IR_RRD_loss"] / iteration, 4)
#         target_loss = epoch_loss["epoch_URRD_loss"] + epoch_loss["epoch_IR_RRD_loss"]
#     else:
#         target_loss = epoch_loss["epoch_URRD_loss"]
        
#     if report["Best_loss"] <= target_loss:
#         report["RRD_patience"] += 1
#     else:
#         report["Best_loss"] = target_loss
#         report["RRD_patience"] = 0
        
#     report.update(epoch_loss)


def RRD_epoch_only_once(RRD_dataset, before_user_ids, batch_size, gpu, model, URRD_lambda, optimizer, scaler, report):
    
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
    # if IR_RRD_flag:
    #     epoch_loss["epoch_IR_RRD_loss"] = round(epoch_loss["epoch_IR_RRD_loss"] / iteration, 4)
    #     target_loss = epoch_loss["epoch_URRD_loss"] + epoch_loss["epoch_IR_RRD_loss"]
    # else:
    #     target_loss = epoch_loss["epoch_URRD_loss"]
        
    # if report["Best_loss"] <= target_loss:
    #     report["RRD_patience"] += 1
    # else:
    #     report["Best_loss"] = target_loss
    #     report["RRD_patience"] = 0
        
    report.update(epoch_loss)



def RRD_epoch_frequency(RRD_loader, gpu, model, URRD_lambda, optimizer, scaler, report):
    
    RRD_loader.dataset.sampling_for_uninteresting_items()    
    epoch_loss = dict()
    epoch_loss["epoch_URRD_loss"] = 0.0
    
    for batch_user in RRD_loader:

        #with torch.cuda.amp.autocast():
            
        batch_user = batch_user.unique()
        interesting_items, uninteresting_items = RRD_loader.dataset.get_samples(batch_user)
        
        batch_user = batch_user.to(gpu)
        interesting_items = interesting_items.to(gpu)#.type(torch.cuda.LongTensor)
        uninteresting_items = uninteresting_items.to(gpu)#.type(torch.cuda.LongTensor)
        
        user_emb = model.base_model.user_emb.weight
        item_emb = model.base_model.item_emb.weight
                            
        interesting_prediction = forward_multi_items(user_emb, item_emb, batch_user, interesting_items)
        uninteresting_prediction = forward_multi_items(user_emb, item_emb, batch_user, uninteresting_items)

        #interesting_prediction = model.base_model.forward_multi_items(batch_user, interesting_items)
        #uninteresting_prediction = model.base_model.forward_multi_items(batch_user, uninteresting_items)
        
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
    
    # if report["Best_loss"] <= epoch_loss["epoch_URRD_loss"]:
    #     report["RRD_patience"] += 1
    # else:
    #     report["Best_loss"] = epoch_loss["epoch_URRD_loss"]
    #     report["RRD_patience"] = 0
        
    report.update(epoch_loss)
    

def RRD_epoch_train_loader(train_loader, RRD_loader, gpu, model, URRD_lambda, optimizer, scaler, report):
    
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
    
    # if report["Best_loss"] <= epoch_loss["epoch_URRD_loss"]:
    #     report["RRD_patience"] += 1
    # else:
    #     report["Best_loss"] = epoch_loss["epoch_URRD_loss"]
    #     report["RRD_patience"] = 0
        
    report.update(epoch_loss)

def IR_RRD(IR_reg_train_dataset, RRD_item_ids, batch_size, gpu, model, IR_reg_lmbda, optimizer, scaler, report):
    
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
        
        #with torch.cuda.amp.autocast():

        interesting_users, uninteresting_users = IR_reg_train_dataset.get_samples(batch_item)#.detach().cpu())

        batch_item = batch_item.to(gpu)
        interesting_users = interesting_users.to(gpu)
        uninteresting_users = uninteresting_users.to(gpu)
        
        user_emb = model.base_model.user_emb.weight
        item_emb = model.base_model.item_emb.weight
            
        interesting_user_prediction = forward_multi_users(user_emb, item_emb, interesting_users, batch_item)
        uninteresting_user_prediction = forward_multi_users(user_emb, item_emb, uninteresting_users, batch_item)
        
        #interesting_user_prediction = model.base_model.forward_multi_users(interesting_users, batch_item)
        #uninteresting_user_prediction = model.base_model.forward_multi_users(uninteresting_users, batch_item)

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
        

def load_model(total_user, total_item, gpu, model_name, args):
    model_args = [total_user, total_item, args.sd, gpu]
    base_model = eval(model_name)(*model_args)
    base_model = base_model.to(gpu)

    # Wrap up as PIW_LWCKD or CL_VAE
    if model_name == "BPR":
        model = PIW_LWCKD(base_model, 
                            LWCKD_flag = False, PIW_flag = args.PIW_flag, # task0때는 LWCKD_flag는 False (이전 모델로부터 받을 수가 없음.)
                            temperature = args.T, 
                            num_cluster = args.nc,
                            dim = args.sd, gpu = gpu)
        
    elif model_name == "VAE":
        model = CL_VAE(base_model, args.sd, gpu, args.CL_flag)
    
    return model

# def load_saved_model(path):
#     pth_file = torch.load(path)
    
#     model = pth_file["best_model"].cpu()
#     score_mat = pth_file["score_mat"].cpu()
#     sorted_mat = pth_file["sorted_mat"]
    
#     del pth_file
    
#     return model, score_mat, sorted_mat


def Model_Generate(total_user, total_item, gpu, model_name, args, dim):
    
    model_args = [total_user, total_item, dim, gpu]
    base_model = eval(model_name)(*model_args)
    base_model = base_model.to(gpu)

    # Wrap up as PIW_LWCKD or CL_VAE
    if model_name == "BPR":
        model = PIW_LWCKD(base_model, 
                        LWCKD_flag = True, 
                        PIW_flag = True, # task0때는 LWCKD_flag는 False (이전 모델로부터 받을 수가 없음.)
                        temperature = args.T, 
                        num_cluster = args.nc,
                        dim = dim, gpu = gpu)
        
    elif model_name == "VAE":
        model = CL_VAE(base_model, dim, gpu, args.CL_flag)
    
    return model

def simple_X(model, total_train_dataset, total_valid_dataset, total_test_dataset, R, args, task_idx):
    # Freeze
    model.base_model.user_emb.weight.requires_grad_(False)
    model.base_model.item_emb.weight.requires_grad_(False)
    user_emb = model.base_model.user_emb.weight.detach().cpu()
    item_emb = model.base_model.item_emb.weight.detach().cpu()
    
    # Nothing
    score_mat = torch.matmul(user_emb, item_emb.T)
    sorted_mat = torch.topk(score_mat, k = 1000, dim = -1, largest = True).indices
    sorted_mat = to_np(sorted_mat)
    
    print("[Nothing]")
    get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, sorted_mat, args.k_list, task_idx, FB_flag = False)
    del score_mat, sorted_mat

    # Simple_X
    num_interacted_item = R.sum(dim = 1, keepdims = True)
    denominator_for_user = torch.where(num_interacted_item == 0, 1.0, 2.0)
    Norm_Mat_for_user = R / (num_interacted_item + 1e-8) # U x I, 1e-8로 nan값 해결.
    
    num_interacted_user = R.sum(dim = 0, keepdims = True)
    denominator_for_item = torch.where(num_interacted_user == 0, 1.0, 2.0)
    Norm_Mat_for_item = R / (num_interacted_user + 1e-8) # U x I
    
    new_user_emb = (user_emb + Norm_Mat_for_user @ item_emb) / denominator_for_user
    # print("item_emb", item_emb, item_emb.shape)
    # print("Norm_Mat_for_item.T", Norm_Mat_for_item.T, Norm_Mat_for_item.T.shape)
    # print("user_emb", user_emb, user_emb.shape)
    # print("denominator_for_item", denominator_for_item, denominator_for_item.shape)
    new_item_emb = (item_emb + Norm_Mat_for_item.T @ user_emb) / denominator_for_item.T
    
    score_mat = torch.matmul(new_user_emb, new_item_emb.T)
    sorted_mat = torch.topk(score_mat, k = 1000, dim = -1, largest = True).indices
    sorted_mat = to_np(sorted_mat)
    
    print("\n[SIMPLE_X]")
    get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, sorted_mat, args.k_list, task_idx, FB_flag = False)
    
    model.base_model.user_emb.weight.requires_grad_(True)
    model.base_model.item_emb.weight.requires_grad_(True)
    

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
    
    #score_mat = model.base_model.user_emb.weight @ model.base_model.item_emb.weight.T
    score_mat = score_mat.detach().cpu()
    
    if return_sorted_mat:
        sorted_mat = torch.topk(score_mat, k = topk, dim = -1, largest = True).indices
        sorted_mat = to_np(sorted_mat)
        return score_mat, sorted_mat
    
    return score_mat

def filtering_simple(mat, filtered_data):
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
    
def forward_multi_items(user_emb, item_emb, batch_user, batch_items):

    u = user_emb[batch_user] # batch_size x dim
    u = u.unsqueeze(-1) # batch_size x dim x 1
    i = item_emb[batch_items] # batch_size x items x dim

    score = torch.bmm(i, u).squeeze() # batch_size x items

    return score

def forward_multi_users(user_emb, item_emb, batch_users, batch_item):
    
    i = item_emb[batch_item].unsqueeze(-1) # batch_size x dim x 1
    u = user_emb[batch_users] # batch_size x users x dim

    score = torch.bmm(u, i).squeeze() # batch_size x users

    return score