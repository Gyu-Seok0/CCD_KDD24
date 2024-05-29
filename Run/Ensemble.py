import argparse
import sys
import os
from copy import deepcopy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch

from Utils.data_loaders import *
from Utils.utils import *

def main(args):
    """ Main function for creating an ensemble teacher system """

    assert args.dataset in ["Gowalla", "Yelp"], "Dataset must be either 'Gowalla' or 'Yelp'."
    assert args.T_load_path, "the path for the parameters of each teacher should be specified."

    # Set cpu
    gpu = "cpu"
    
    # Ranking score-based ensemble
    length = 45000
    raw = 0.05
    rank_importance = ((1 / torch.arange(1, length+1)) ** raw) # tensor([1.0000, 0.9659, 0.9466,  ..., 0.6567, 0.6567, 0.6567])

    # Model
    if args.dataset == "Gowalla":
        T_model_list = ["VAE_0", "VAE_2", "BPR_0", "BPR_1", "BPR_2", "BPR_3", "BPR_4"]
    
    elif args.dataset == "Yelp":
        T_model_list = ["LightGCN_0", "LightGCN_1", "LightGCN_2", "LightGCN_3", "LightGCN_4"]

    load_path = args.T_load_path
    print("T_model_list = ", T_model_list)
    print("T_load_path = ", load_path)
    
    # Dataset
    data_path = f"../dataset/{args.dataset}/total_blocks_timestamp.pickle"
    data_dict_path = f"../dataset/{args.dataset}"
    total_train_dataset, total_valid_dataset, total_test_dataset, total_item_list = load_data_as_dict(data_dict_path, num_task = args.num_task)

    task_idx = args.target_task
    
    total_blocks = load_pickle(data_path)
    p_total_user = total_blocks[task_idx].user.max() + 1
    b_total_user = total_blocks[task_idx-1].user.max() + 1

    p_total_item = total_blocks[task_idx].item.max() + 1
    _, p_train_mat, p_valid_mat, p_test_mat = get_train_valid_test_mat(task_idx, total_train_dataset, total_valid_dataset, total_test_dataset)
    
    # Train/test/valid data split for new users/items
    new_user_train_mat, new_user_valid_mat, new_user_test_mat = get_train_valid_test_mat_for_new_users(b_total_user, p_total_user, p_train_mat, p_valid_mat, p_test_mat)

################################### Ensemble ##########################################################################################################################################################
    E_teacher_rank_list = get_rank_score_mat_list(load_path, T_model_list, task_idx, gpu, p_total_user, p_total_item, 
                                                  total_train_dataset, total_valid_dataset, total_test_dataset, rank_importance,
                                                  new_user_train_mat, new_user_valid_mat, new_user_test_mat, args.k_list, 
                                                  eval_task_idx = task_idx)
            
    E_teacher_score_mat = torch.nanmean(torch.stack(E_teacher_rank_list), dim = 0)
    
    get_eval_Ensemble(E_teacher_score_mat, total_train_dataset, total_valid_dataset, total_test_dataset, args.num_task, args.k_list,
                      new_user_train_mat, new_user_valid_mat, new_user_test_mat, print_out = "Teacher")
    
    # Save
    if args.save:
        if args.save_path is None:
            save_dir_path = load_path
        else:
            save_dir_path = args.save_path
        
        save_dir_path = os.path.join(save_dir_path, "Ensemble")
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        
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
    parser.add_argument('--T_load_path', "--Tlp", type = str, default = "../ckpt/Yelp/Teacher/using_student_LightGCN_1/Method")
    
    parser.add_argument("--save", "--s", action = argparse.BooleanOptionalAction, help = "whether saving param or not (--s or --no-s)")
    parser.add_argument('--save_path', "--sp", type = str, default = "../ckpt/Yelp/Teacher/using_studentLightGCN_1/Method_Test")

    args = parser.parse_args()
    print_command_args(args)
    main(args)
    print_command_args(args)
    
#python -u Ensemble.py --d Yelp --tt 1 --s