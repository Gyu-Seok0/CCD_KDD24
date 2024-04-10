import argparse
from copy import deepcopy
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import torch.nn as nn
import torch.optim as optim

from Utils.data_loaders import *
from Utils.utils import *

def load_model(path):
    pth_file = torch.load(path)
    
    model = pth_file["best_model"].cpu()
    score_mat = pth_file["score_mat"].cpu()
    sorted_mat = pth_file["sorted_mat"]
    
    del pth_file
    
    return model, score_mat, sorted_mat

def filtering(b_score_mat, p_score_mat, bp_score_mat, filtered_data):
    
    b_u_size, b_i_size = b_score_mat.shape
    
    for u, items in filtered_data.items():
        items = torch.tensor(items).long()
        
        if u < b_u_size:
            b_score_mat[u][items[items < b_i_size].long()] = -1e8
        
        p_score_mat[u][items] = -1e8
        bp_score_mat[u][items] = -1e8
    
    return b_score_mat, p_score_mat, bp_score_mat


def mat_mean(mat_list : list):
    stacked_tensor = torch.stack(mat_list)
    mean_matrix = torch.nanmean(stacked_tensor, dim = 0)
    return mean_matrix

def score2sorted(score_mat):
    
    score_mat = score_mat.detach().cpu()
    sorted_mat = torch.topk(score_mat, k = 1000, largest = True).indices
    sorted_mat = sorted_mat.numpy()
    
    return sorted_mat

def expand_mat(mat, shape):
    expand_mat = torch.full((shape), torch.nan)
    b_u_size, b_i_size = mat.size()
    expand_mat[:b_u_size, :b_i_size] = mat
    return expand_mat

def get_score_mat_for_BPR(model):
    user_emb, item_emb = model.get_embedding()
    score_mat = torch.matmul(user_emb, item_emb.T)
    return score_mat


def get_sum_score_mat(load_path, model_list, task_idx, gpu, p_total_user, p_total_item, 
                      total_train_dataset, total_valid_dataset, total_test_dataset, rank_importance,
                      new_user_train_mat, new_user_valid_mat, new_user_test_mat, k_list, eval_task_idx):
    
    rank_score_mat_list = []
    
    for m_name in model_list:
        print(f"\nmodel_name = {m_name}")
        
        if m_name in ["Stability", "Plasticity"]:
            train_finish_task = task_idx - 1
        else:
            train_finish_task = task_idx
        
        model_path = os.path.join(load_path, m_name, f"TASK_{train_finish_task}.pth")
        print("model_path", model_path)
        
        score_mat = torch.load(model_path, map_location = torch.device(gpu))["score_mat"]#.detach().cpu()
        score_mat = score_mat[:p_total_user, :p_total_item]
        
        sorted_mat = to_np(torch.topk(score_mat, k = 1000).indices.detach().cpu())
        get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, sorted_mat, k_list, eval_task_idx)
        
        rank_score_mat = convert_to_rank_score_mat(score_mat, rank_importance)
        if m_name in ["Stability", "Plasticity"]:
            rank_score_mat = expand_mat(rank_score_mat, (p_total_user, p_total_item))
        else:
            print("\n[New user result]\n")
            new_user_results = get_eval_with_mat(new_user_train_mat, new_user_valid_mat, new_user_test_mat, sorted_mat, k_list)
            print(f"valid_R20 = {new_user_results['valid']['R20']}, test_R20 = {new_user_results['test']['R20']}")
        
        rank_score_mat_list.append(rank_score_mat.detach().cpu())
    return rank_score_mat_list

def get_eval(E_score_mat, total_train_dataset, total_valid_dataset, total_test_dataset, task_idx, k_list, 
             new_user_train_mat, new_user_valid_mat, new_user_test_mat, print_out = "Teacher"):
    
    E_sorted_mat = score2sorted(E_score_mat)
    print(f"\n[Ensemble for {print_out}]")
    get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, E_sorted_mat, k_list, task_idx)
    
    print("\n[New user result]\n")
    new_user_results = get_eval_with_mat(new_user_train_mat, new_user_valid_mat, new_user_test_mat, E_sorted_mat, k_list)
    print(f"valid_R20 = {new_user_results['valid']['R20']}, test_R20 = {new_user_results['test']['R20']}")
    
def get_train_valid_test_mat(task_idx, total_train_dataset, total_valid_dataset, total_test_dataset):
    p_train_dict = total_train_dataset[f"TASK_{task_idx}"] # {u_1 : {i_1, i_2, i_3}, ..., }
    p_valid_dict = total_valid_dataset[f"TASK_{task_idx}"]
    p_test_dict = total_test_dataset[f"TASK_{task_idx}"]
    
    train_interaction = make_interaction(p_train_dict)
    train_mat = make_rating_mat(p_train_dict) # {u : {i_1:1, i_2:1}}
    valid_mat = make_rating_mat(p_valid_dict)
    test_mat = make_rating_mat(p_test_dict)
    
    return train_interaction, train_mat, valid_mat, test_mat


def main(args):
    
    assert args.dataset in ["Gowalla", "Yelp"]
    assert args.T_load_path

    gpu = "cpu"
    print(f"gpu = {gpu}")
    
    # ranking score ensemble
    length = 45000
    raw = 0.05
    rank_importance = ((1 / torch.arange(1, length+1)) ** raw) # tensor([1.0000, 0.9659, 0.9466,  ..., 0.6567, 0.6567, 0.6567])

    # Dataset
    data_path = f"../dataset/{args.dataset}/total_blocks_timestamp.pickle"
    data_dict_path = f"../dataset/{args.dataset}"
    print("Dataset = ", args.dataset)

    # Model
    if args.dataset == "Gowalla":
        T_model_list = ["VAE_0", "VAE_2", "BPR_0", "BPR_1", "BPR_2", "BPR_3", "BPR_4"]
    
    elif args.dataset == "Yelp":
        T_model_list = ["LightGCN_0", "LightGCN_1", "LightGCN_2", "LightGCN_3", "LightGCN_4"]

    load_path = args.T_load_path

    print("T_model_list = ", T_model_list)
    print("T_load_path = ", load_path)

    total_train_dataset, total_valid_dataset, total_test_dataset, total_item_list = load_data_as_dict(data_dict_path, num_task = args.num_task)

    # SAVE
    save_dir_path = None
    if args.save:
        
        if args.save_path is None:
            save_path = load_path
        else:
            save_path = args.save_path
        
        save_dir_path = os.path.join(save_path, "Ensemble")
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
    
    # Rank_score
    E_score_mat = None
    task_idx, distilled_idx = args.target_task, args.target_task-1
    
    total_blocks = load_pickle(data_path)
    p_total_user = total_blocks[task_idx].user.max() + 1
    b_total_user = total_blocks[task_idx-1].user.max() + 1

    p_total_item = total_blocks[task_idx].item.max() + 1
    _, p_train_mat, p_valid_mat, p_test_mat = get_train_valid_test_mat(task_idx, total_train_dataset, total_valid_dataset, total_test_dataset)
    
    # train/test/valid for new users
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
    
    # integreate
    
    E_teacher_rank_list = get_sum_score_mat(load_path, T_model_list, task_idx, gpu, p_total_user, p_total_item, 
                                            total_train_dataset, total_valid_dataset, total_test_dataset, rank_importance,
                                            new_user_train_mat, new_user_valid_mat, new_user_test_mat, args.k_list, 
                                            eval_task_idx = 5)
    
    E_teacher_score_mat = torch.nanmean(torch.stack(E_teacher_rank_list), dim = 0)
    
    get_eval(E_teacher_score_mat, total_train_dataset, total_valid_dataset, total_test_dataset, args.num_task, args.k_list,
             new_user_train_mat, new_user_valid_mat, new_user_test_mat, print_out = "Teacher")
    
    if args.save:
        save_path = os.path.join(save_dir_path, f"TASK_{task_idx}_score_mat.pth")
        model_state = {'score_mat' : deepcopy(E_teacher_score_mat)}
        torch.save(model_state, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_task', type = int, default = 6, help = "task_idx")
    parser.add_argument("--gpu", type = int, default = 0)
    parser.add_argument('--k_list', type = list, default = [20, 50, 100])
    
    parser.add_argument('--target_task', "--tt", type = int, default = -1, help = "task_idx")
    parser.add_argument('--dataset', "--d", type = str, default = None, help = "Gowalla or Yelp")
    parser.add_argument('--T_load_path', "--Tlp", type = str, default = "../ckpt/Yelp/Teacher/using_LightGCN_1/Method")
    
    parser.add_argument("--save", "--s", action = argparse.BooleanOptionalAction, help = "whether saving param or not (--s or --no-s)")
    parser.add_argument('--save_path', "--sp", type = str, default = "../ckpt/Yelp/Teacher/using_LightGCN_1/Method_Test")

    args = parser.parse_args()
    print_command_args(args)
    main(args)
    print_command_args(args)
    
#python -u Ensemble.py --d Yelp --tt 1 --s