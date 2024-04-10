import argparse
import random
import time
import gc
from copy import deepcopy
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Utils.data_loaders import *
from Utils.utils import *

from Models.BPR import BPR
from Models.LightGCN_V2 import LightGCN
from Models.VAE import VAE

def main(args):
    
    gpu = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    scaler = torch.cuda.amp.GradScaler()
    
    assert args.dataset in ["Gowalla", "Yelp"]
    
    # student's embedding dimension
    if args.sd == None:
        if args.dataset == "Gowalla":
            args.sd = 16
            
        elif args.dataset == "Yelp":
            args.sd = 8

    # data path
    data_block_path = f"../dataset/{args.dataset}/total_blocks_timestamp.pickle"
    data_dict_path = f"../dataset/{args.dataset}"
    
    # dataset
    total_blocks = load_pickle(data_block_path)
    total_train_dataset, total_valid_dataset, total_test_dataset, total_item_list = load_data_as_dict(data_dict_path, num_task = args.num_task)
    
    task_idx = args.target_task
    distilled_idx = task_idx - 1
    
    print(f"\n## Distlled_IDX = {distilled_idx} ##")
        
    b_train_dict = total_train_dataset[f"TASK_{distilled_idx}"]
    p_train_dict = total_train_dataset[f"TASK_{task_idx}"] # {u_1 : {i_1, i_2, i_3}, ..., }
    p_valid_dict = total_valid_dataset[f"TASK_{task_idx}"]
    p_test_dict = total_test_dataset[f"TASK_{task_idx}"]
    
    train_interaction = make_interaction(p_train_dict) # [(u, i, 1), ..., ]
    train_mat = make_rating_mat(p_train_dict) # {u_1 : {i_1: 1, i_2: 1, i_3: 1}, ..., }
    valid_mat = make_rating_mat(p_valid_dict)
    test_mat = make_rating_mat(p_test_dict)
    b_train_mat = make_rating_mat(b_train_dict)
    
    before_user_mapping, before_item_mapping, before_rating_mat, UU, II = None, None, None, None, None
    before_user_ids, before_item_ids, present_user_ids, present_item_ids = None, None, None, None
    
    # model path
    Teacher_Ensemble_RatingMat_path = f"../ckpt/{args.dataset}/Teacher/using_{args.model}/Method/Ensemble/TASK_{distilled_idx}_score_mat.pth"
    if distilled_idx == 0:
        Teacher_Ensemble_RatingMat_path = f"../ckpt/{args.dataset}/Teacher/base_teacher/Ensemble/TASK_0_score_mat.pth"
        
    Student_model_path = f"../ckpt/{args.dataset}/Student/{args.model}/Method"
    load_D_model_dir_path = f"{Student_model_path}/Distilled"
    load_S_proxy_dir_path = f"{Student_model_path}/Stability"
    load_P_proxy_dir_path = f"{Student_model_path}/Plasticity"
    load_CL_model_dir_path = f"{Student_model_path}/CL"
    
    T_score_mat = torch.load(Teacher_Ensemble_RatingMat_path, map_location = gpu)["score_mat"].detach().cpu() #RRD_SM[f"TASK_{distilled_idx}"]
    FT_score_mat = filtering_simple(T_score_mat, b_train_dict).detach().cpu()
    T_RRD_interesting_items = torch.topk(FT_score_mat, k = 40, dim = 1).indices
    Negatvie_exclude = T_RRD_interesting_items.clone().detach()
    del Teacher_Ensemble_RatingMat_path, T_score_mat, FT_score_mat, T_RRD_interesting_items

    p_block = total_blocks[task_idx]
    b_block = total_blocks[task_idx - 1]

    total_user = p_block.user.max() + 1
    total_item = p_block.item.max() + 1
    
    before_total_user = b_block.user.max() + 1
    before_total_item = b_block.item.max() + 1
    
    # train/test/valid split for new users

    num_new_user = total_user - before_total_user    
    num_new_item = total_item - before_total_item
    
    new_user_train_mat = {}
    new_user_valid_mat = {}
    new_user_test_mat = {}
    
    for new_user_id in range(before_total_user, total_user):
        if new_user_id in train_mat.keys():
            new_user_train_mat[new_user_id] = train_mat[new_user_id]
        if new_user_id in valid_mat.keys():
            new_user_valid_mat[new_user_id] = valid_mat[new_user_id]
        if new_user_id in test_mat.keys():
            new_user_test_mat[new_user_id] = test_mat[new_user_id]
    
    R = make_R(total_user, total_item, train_mat)
    b_R = make_R(before_total_user, before_total_item, b_train_mat)

    SNM = get_SNM(total_user, total_item, R, gpu)
    b_SNM = get_SNM(before_total_user, before_total_item, b_R, gpu)

    D_model_path = os.path.join(load_D_model_dir_path, f"TASK_{distilled_idx}.pth")
    D_model_weight, D_score_mat, D_sorted_mat = load_saved_model(D_model_path, gpu)
    
    if type(D_model_weight) != dict:
        D_model_weight = D_model_weight.state_dict()
        
    model_type, model_seed = args.model.split("_")

    # Random Seed
    print(f"random_seed = {model_seed}")
    set_random_seed(int(model_seed))
    
    # Distilled Student model
    D_model = get_model(before_total_user, before_total_item, b_SNM, gpu, args, model_type, D_model_weight).to(gpu)
    del b_R, b_SNM
    print("[D_model]\n", D_model)
    print("\n[D_model Performance]")
    get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, D_sorted_mat, args.k_list, 
                  current_task_idx = args.num_task-1, FB_flag = False, return_value = False, max_task = task_idx)
    Negatvie_exclude = torch.cat([Negatvie_exclude, torch.tensor(D_sorted_mat[:, :40])], dim = 1)
    del D_sorted_mat

    # S, P proxies
    if distilled_idx == 0:
        S_proxy = deepcopy(D_model)
        P_proxy = deepcopy(D_model)
        
        S_score_mat = deepcopy(D_score_mat)
        P_score_mat = deepcopy(D_score_mat)

    else:
        S_proxy_path = os.path.join(load_S_proxy_dir_path, f"TASK_{distilled_idx-1}.pth")
        P_proxy_path = os.path.join(load_P_proxy_dir_path, f"TASK_{distilled_idx-1}.pth")
        
        S_proxy_weight = torch.load(S_proxy_path)["best_model"]
        P_proxy_weight = torch.load(P_proxy_path)["best_model"]
        
        bb_block = total_blocks[task_idx-2]
        bb_total_user = bb_block.user.max() + 1
        bb_total_item = bb_block.item.max() + 1
        
        bb_train_dict = total_train_dataset[f"TASK_{task_idx-2}"]
        bb_train_mat = make_rating_mat(bb_train_dict)
        
        bb_R = make_R(bb_total_user, bb_total_item, bb_train_mat)
        bb_SNM = get_SNM(bb_total_user, bb_total_item, bb_R, gpu)
        
        S_proxy = get_model(bb_total_user, bb_total_item, bb_SNM, gpu, args, model_type, S_proxy_weight).to(gpu)
        P_proxy = get_model(bb_total_user, bb_total_item, bb_SNM, gpu, args, model_type, P_proxy_weight).to(gpu)
        del bb_R, bb_SNM
        
        # print("[Before Merging]")
        # print("[S_proxy]\n", S_proxy)
        # print("\n[P_proxy]\n", P_proxy)
    
        S_proxy = merge_model(S_proxy, D_model, wme = True, b_weight = args.s_weight, p_weight = 1 - args.s_weight).to(gpu)
        P_proxy = merge_model(P_proxy, D_model, wme = True, b_weight = args.p_weight, p_weight = 1 - args.p_weight).to(gpu)

        S_proxy = freeze(S_proxy)
        P_proxy = freeze(P_proxy)
        
        # print("[After Merging]")
        # print("[S_proxy]\n", S_proxy)
        # print("\n[P_proxy]\n", P_proxy)
                
        S_score_mat, S_sorted_mat = get_sorted_score_mat(S_proxy, return_sorted_mat = True)
        P_score_mat, P_sorted_mat = get_sorted_score_mat(P_proxy, return_sorted_mat = True)

        print("\n[S_proxy Performance]")
        get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, S_sorted_mat, args.k_list, 
                      current_task_idx = args.num_task-1, FB_flag = False, return_value = False, max_task = task_idx)
        Negatvie_exclude = torch.cat([Negatvie_exclude, torch.tensor(S_sorted_mat[:, :40])], dim = 1)
        del S_sorted_mat
        
        print("\n[P_proxy Performance]")
        get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, P_sorted_mat, args.k_list, 
                      current_task_idx = args.num_task-1, FB_flag = False, return_value = False, max_task = task_idx)
        Negatvie_exclude = torch.cat([Negatvie_exclude, torch.tensor(P_sorted_mat[:, :40])], dim = 1)
        del P_sorted_mat

################################### Student Update ##########################################################################################################################################################        
    
    print(f"\n## Task_IDX = {task_idx} ##")
    
    # Current Task Dataset #
    BPR_train_dataset = implicit_CF_dataset(total_user, total_item, train_mat, args.nns, train_interaction, Negatvie_exclude)
    train_loader = DataLoader(BPR_train_dataset, batch_size = args.bs, shuffle = True, drop_last = False)
    
    FS_score_mat = filtering_simple(S_score_mat, b_train_dict)
    FS_score_mat = filtering_simple(FS_score_mat, p_train_dict)
    
    if distilled_idx > 0:
        FP_score_mat = filtering_simple(P_score_mat, b_train_dict)
        FP_score_mat = filtering_simple(FP_score_mat, p_train_dict)
    else:
        FP_score_mat = FS_score_mat
    
    if args.BD:
        if args.Using_S:
            S_sig_mat = torch.sigmoid(FS_score_mat)
            S_rank_mat = convert_to_rank_mat(FS_score_mat)
        else:
            S_rank_mat = None
            S_sig_mat = None

        if (distilled_idx > 0 and args.Using_P) or (distilled_idx == 0 and args.Using_P and args.Using_S != True):
            P_rank_mat = convert_to_rank_mat(FP_score_mat)
            P_sig_mat = torch.sigmoid(FP_score_mat)
        else:
            P_rank_mat = None
            P_sig_mat = None
        
        if (distilled_idx == 0 and args.Using_P and args.Using_S != True):
            args.P_sample = args.S_sample
            
        BD_dataset = get_total_BD_dataset(D_score_mat, S_rank_mat, S_sig_mat, P_rank_mat, P_sig_mat, args)

    # model
    model = deepcopy(D_model).to(gpu)
    for param in model.parameters():
        param.requires_grad = True
    
    print("[Before updating]", model)

    # Model size update
    model.update(before_user_ids, before_item_ids,
                 before_user_mapping, before_item_mapping,
                 before_rating_mat, num_new_user, num_new_item,
                 UU, II, present_user_ids, present_item_ids, R, args.random_init, SNM, args.init_topk, args.only_one_hop)
    
    print("[After updating]", model)
    
    if args.model == "LightGCN_0":
        model.base_model.set_layer_index(args.using_layer_index)
        model.base_model.num_layer = args.num_layer
        
        print(f"args.using_layer_index = {args.using_layer_index}")
        print(f"args.num_layer = {args.num_layer}")

    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.reg)
    relu = nn.ReLU()
    criterion = nn.BCEWithLogitsLoss(reduction = 'sum')
    eval_args = {"best_score" : 0, "test_score" : 0, "best_epoch" : 0, "best_model" : None,
                 "score_mat" : None,  "sorted_mat" : None, "patience" : 0, "avg_valid_score" : 0, "avg_test_score": 0}
    total_time = 0.0
    
    # training
    for epoch in range(args.max_epoch):
        print(f"\n[Epoch:{epoch + 1}/{args.max_epoch}]")
        start_time = time.time()    
        train_loader.dataset.negative_sampling()
        epoch_CF_loss = 0.0
        
        model.train()
        for mini_batch in train_loader:
            
            # with torch.cuda.amp.autocast():
            mini_batch = {key : values.to(gpu) for key, values in mini_batch.items()}
            embs = model.base_model.forward(mini_batch)
            batch_loss = model.base_model.get_loss(embs)
            
            # Backward
            optimizer.zero_grad()
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_CF_loss += batch_loss.item()
        
        CF_time = time.time()
        print(f"epoch_CF_loss = {round(epoch_CF_loss / len(train_loader), 4)}, CF_time = {CF_time - start_time:.4f} seconds", end = " ")
        
        # BD
        if args.BD:
            if args.annealing:
                BD_lambda = args.BD_lambda * torch.exp(torch.tensor(-epoch)/args.a_T)
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
                    
                batch_user = torch.tensor(u_list[start : end]).to(gpu)
                batch_item = torch.tensor(i_list[start : end]).to(gpu)
                batch_label = torch.tensor(r_list[start : end]).to(gpu)
                
                user_emb, item_emb = model.base_model.get_embedding()
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
            
            BD_time = time.time()            
            print(f"epoch_BD_loss = {round(epoch_BD_loss/iteration, 4)}, BD_lambda = {BD_lambda:.5f}, BD_time = {BD_time - CF_time:.4f} seconds", end = " ")
        
        # Result (Time)
        end_time = time.time()
        epoch_time = end_time - start_time
        total_time += epoch_time
        print(f"epoch_time = {epoch_time:.4f} seconds, total_time = {total_time:.4f} seconds")

        if epoch % 5 == 0:
            print("\n[Evaluating]")
            model.eval()
            with torch.no_grad():
                CL_score_mat, CL_sorted_mat = get_sorted_score_mat(model, return_sorted_mat = True)
                    
            valid_list, test_list = get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, 
                                                  CL_sorted_mat, args.k_list, current_task_idx = task_idx, FB_flag = False, return_value = True)
            
            avg_valid_score = get_average_score(valid_list, "valid_R20")
            avg_test_score = get_average_score(test_list, "test_R20")
            
            if args.eval_average_acc:
                valid_score = avg_valid_score
                test_score =  avg_test_score
            else:
                valid_score = valid_list[task_idx]["valid_R20"]
                test_score = test_list[task_idx]["test_R20"]
        
            # new_users #
            print("\n[New user result]")
            new_user_results = get_eval_with_mat(new_user_train_mat, new_user_valid_mat, new_user_test_mat, CL_sorted_mat, args.k_list)
            new_user_results_txt = f"valid_R20 = {new_user_results['valid']['R20']}, test_R20 = {new_user_results['test']['R20']}"
            print(f"\t{new_user_results_txt}\n")
                              
            if valid_score > eval_args["best_score"]:
                print(f"[Best Model Changed]\n\tvalid_score = {valid_score:.4f}, test_score = {test_score:.4f}")
                eval_args["best_score"] = valid_score
                eval_args["test_score"] = test_score
                eval_args["avg_valid_score"] = avg_valid_score
                eval_args["avg_test_score"] = avg_test_score
                eval_args["best_epoch"] = epoch
                
                eval_args["best_model"] = deepcopy({k: v.cpu() for k, v in model.state_dict().items()}) #deepcopy(model) # model.state_dict()
                eval_args["score_mat"] = deepcopy(CL_score_mat)
                eval_args["sorted_mat"] = deepcopy(CL_sorted_mat)
                eval_args["patience"] = 0
                best_new_user_results = deepcopy(new_user_results_txt)
            else:
                eval_args["patience"] += 1
                if eval_args["patience"] >= args.early_stop:
                    print("[Early Stopping]")
                    break
            
            if args.BD and epoch > 0:
                BD_dataset = get_total_BD_dataset(D_score_mat, S_rank_mat, S_sig_mat, P_rank_mat, P_sig_mat, args)

        
        # get gpu memory
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"\n[CL_Best_Result for Task_{task_idx}]")
    print(f"\tbest_epoch = {eval_args['best_epoch']}, valid_score = {eval_args['best_score']}, test_score = {eval_args['test_score']}", end = " ")
    print(f"avg_valid_score = {eval_args['avg_valid_score']}, avg_test_score = {eval_args['avg_test_score']}")

    get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, eval_args["sorted_mat"], args.k_list, 
                  current_task_idx = args.num_task, FB_flag = False, return_value = False)
    
    print("\n[New user result]")
    print(f"\t{best_new_user_results}\n")
    
    if args.save:
        print("\n[Model Save]")
        
        if args.save_path is not None:
            save_path = args.save_path
        else:
            save_path = Student_model_path
            
        save_S_proxy_dir_path = f"{save_path}/Stability"
        save_P_proxy_dir_path = f"{save_path}/Plasticity"
        save_CL_model_dir_path = f"{save_path}/CL"
            
        for dir_path in [save_S_proxy_dir_path, save_P_proxy_dir_path, save_CL_model_dir_path]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
        save_model(save_S_proxy_dir_path, distilled_idx, {"best_model" : deepcopy({k: v.cpu() for k, v in S_proxy.state_dict().items()}), "score_mat": FS_score_mat})
        save_model(save_P_proxy_dir_path, distilled_idx, {"best_model" : deepcopy({k: v.cpu() for k, v in P_proxy.state_dict().items()}), "score_mat": FP_score_mat})
        save_model(save_CL_model_dir_path, task_idx, {"best_model" : eval_args["best_model"], "score_mat"  : eval_args["score_mat"]})
            
        print("save_S_proxy_dir_path", save_S_proxy_dir_path)
        print("save_P_proxy_dir_path", save_P_proxy_dir_path)
        print("save_CL_model_dir_path", save_CL_model_dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save", "--s", action = argparse.BooleanOptionalAction, help = "whether saving param or not (--s or --no-s)")
    parser.add_argument('--save_path', "--sp", type = str, default = "../ckpt/Yelp/Student/LightGCN_1/Test")
    
    # etc
    parser.add_argument("--gpu", type = int, default = 0)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument("--reg", type = float, default = 0.0001)
    parser.add_argument("--bs", help = "batch_size", type = int, default = 2048)
    parser.add_argument('--max_epoch', type = int, default = 100)
    parser.add_argument('--rs', type = int, default = 1, help = "random seed")
    parser.add_argument('--early_stop', type = int, default = 2)
    parser.add_argument("--nns", help = "the number of negative sample", type = int, default = 1)
    parser.add_argument("--sd", help = "student_dims", type = int, default = 8)

    # S/P proxies & BD(Bidirectional Distillation)
    parser.add_argument("--BD", action = argparse.BooleanOptionalAction, help = "whether using param or not (--BD or --no-BD)")
    parser.add_argument("--eps", type = float, default = 0.0001)
    parser.add_argument("--a_T", type = float, default = 10.)
    parser.add_argument("--BD_lambda", type = float, default = 0.5)
    parser.add_argument("--s_weight", "--sw", type = float, default = 0.9)
    parser.add_argument("--p_weight", "--pw", type = float, default = 0.9)
    parser.add_argument("--absolute", "--ab", type = int, default = 100)
    parser.add_argument("--S_sample", "--ss",type = int, default = 5, help = "hyper parameter for Ranking Distillation")
    parser.add_argument("--P_sample", "--ps", type = int, default = 5, help = "hyper parameter for Ranking Distillation")
    parser.add_argument("--annealing", "--a", action = "store_true", default = True, help = "using annealing")
    parser.add_argument("--correction", "--c", action = "store_true", default = True, help = "using correction loss")
    
    # initaltation for new users/items
    parser.add_argument('--random_init', "--r", action = "store_true", default = False, help = 'random_initalization for new user/items')
    parser.add_argument("--init_topk", type = int, default = 20) # how many popular neighbors you would like to aggregate 
    parser.add_argument("--only_one_hop", "--ooh", action = argparse.BooleanOptionalAction, help = "whether using param or not (--ooh or --no-ooh)")

    # evalation
    parser.add_argument("--num_task", type = int, default = 6)
    parser.add_argument("--hyper_param", "--h", action = argparse.BooleanOptionalAction, help = "whether saving param or not (--h or --no-h)")
    parser.add_argument("--eval_average_acc", "--eaa", action = argparse.BooleanOptionalAction, help = "whether saving param or not (--eaa or --no-eaa)")
    parser.add_argument('--k_list', type = list, default = [20, 50, 100])

    # LightGCN
    parser.add_argument('--num_layer', type = int, default = 2)
    parser.add_argument('--using_layer_index', type = str, default = "avg", help = "first, last, avg")
    
    # PIW
    parser.add_argument("--nc", help = "num_cluster", type = int, default = 10)
    parser.add_argument("--T", help = "temperature", type = int, default = 5.)
    
    # select
    parser.add_argument("--model", "-m", type = str, help = "LightGCN_0 or BPR_0")
    parser.add_argument('--dataset', "--d", type = str, default = None, help = "Gowalla or Yelp")
    parser.add_argument('--target_task', "--tt", type = int, default = -1)
    
    parser.add_argument("--Using_S", "--US", action = argparse.BooleanOptionalAction, help = "whether saving param or not (--s or --no-s)")
    parser.add_argument("--Using_P", "--UP", action = argparse.BooleanOptionalAction, help = "whether saving param or not (--s or --no-s)")
    parser.add_argument("--Using_D", "--UD", action = argparse.BooleanOptionalAction, help = "whether saving param or not (--s or --no-s)")

    args = parser.parse_args()
    print_command_args(args)
    main(args)
    print_command_args(args)

#python -u S_update.py --d Yelp -m LightGCN_1 --tt 1 --BD --US --UP --ab 50 --ss 1 --ps 0 --sw 1.0 --pw 0.1 --s --max_epoch 10