import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import argparse
import random
from collections import defaultdict
from requests import get

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import gc

from Utils.data_loaders import *
from Utils.utils import *

from Models.BPR import BPR
from Models.LightGCN_V2 import LightGCN
from Models.VAE import VAE

from time import time

def merge_model(before_model, present_model, wme, b_weight = None, p_weight = None):
    
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
    bp_user_emb[:u_size] = p_weight * p_user_emb[:u_size] + b_weight * b_user_emb
    return bp_user_emb

def filtering_simple(mat, filtered_data):
    u_size, i_size = mat.shape
    
    for u, items in filtered_data.items():
        items = torch.tensor(items).long()
        if u < u_size:
            mat[u][items[items < i_size].long()] = -1e8
    return mat

def expand_mat_zeros(mat, expand_size):
    new_mat = torch.zeros(expand_size)
    u_size, i_size = mat.shape
    new_mat[:u_size, :i_size] = mat
    return new_mat

def expand_mat(mat, shape, value = torch.nan):
    expand_mat = torch.full((shape), value)
    b_u_size, b_i_size = mat.size()
    expand_mat[:b_u_size, :b_i_size] = mat
    return expand_mat

def expand_emb_average(emb_weight, expand_size):
    
    new_mat = torch.zeros(expand_size)
    u_size, i_size = emb_weight.shape
    new_mat[:u_size, :i_size] = emb_weight
    new_mat[u_size: ] = torch.mean(emb_weight, dim = 0)
    
    return new_mat

def get_topk_prompt(query, prompt_key: nn.Embedding, topk = 5):

    denominator = query.norm(2, dim = 1).unsqueeze(1) * prompt_key.weight.norm(2, dim = 1).unsqueeze(0)
    numerator = query @ prompt_key.weight.T
    cos_sim = numerator / (denominator + 1e-8)
    cos_values, prompt_idx = torch.topk(cos_sim, k = topk)
    
    return cos_values, prompt_idx
    
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

def get_interesting_items_after_filtering(score_mat, filter_dict, item_side = False):
    score_mat = filtering_simple(score_mat, filter_dict)
    if item_side:
        interesting_things = torch.topk(score_mat.T, k = 40, dim = 1).indices
    else:
        interesting_things = torch.topk(score_mat, k = 40, dim = 1).indices
    return interesting_things

def get_train_valid_test_mat(task_idx, total_train_dataset, total_valid_dataset, total_test_dataset):
    p_train_dict = total_train_dataset[f"TASK_{task_idx}"] # {u_1 : {i_1, i_2, i_3}, ..., }
    p_valid_dict = total_valid_dataset[f"TASK_{task_idx}"]
    p_test_dict = total_test_dataset[f"TASK_{task_idx}"]
    
    train_interaction = make_interaction(p_train_dict)
    train_mat = make_rating_mat(p_train_dict)
    valid_mat = make_rating_mat(p_valid_dict)
    test_mat = make_rating_mat(p_test_dict)
    
    return train_interaction, train_mat, valid_mat, test_mat

def main(args):
    
    # GPU & scaler
    gpu = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    scaler = torch.cuda.amp.GradScaler()
    print(f"[GPU] {gpu}")
    
    
    # dataset
    assert args.dataset in ["Gowalla", "Yelp"]
    assert args.model in ["BPR_0", "BPR_1", "BPR_2", "BPR_3", "BPR_4", "LightGCN_0", "LightGCN_1", "LightGCN_2", "LightGCN_3", "LightGCN_4"]
    
    model_type, random_seed = args.model.split("_")
    # Random Seed
    set_random_seed(int(random_seed))
    
    # Learning
    distilled_idx = args.target_task
    print(f"[Distilled_idx = {distilled_idx}]")
    
    Student_load_path = f"../ckpt/{args.dataset}/Student/{args.model}/Method"
    load_S_model_dir_path = f"{Student_load_path}/Stability"
    load_P_model_dir_path = f"{Student_load_path}/Plasticity"
    load_CL_model_dir_path = f"{Student_load_path}/CL"
    print("Student_load_path", Student_load_path)
    
    RRD_SM_dir = f"../ckpt/{args.dataset}/Teacher/using_{args.model}/Method/Ensemble"
    print("Teacher_load_path", RRD_SM_dir)

    data_path = f"../dataset/{args.dataset}/total_blocks_timestamp.pickle"
    data_dict_path = f"../dataset/{args.dataset}"
    
    total_blocks = load_pickle(data_path)
    total_train_dataset, total_valid_dataset, total_test_dataset, total_item_list = load_data_as_dict(data_dict_path, num_task = args.num_task)
    
    if args.dataset == "Gowalla":
        args.nui = 30000
        args.nuu = 19500
        if args.sd == None:
            args.sd = 16
        
    elif args.dataset == "Yelp":
        args.nui = 10000
        args.nuu = 11500
        if args.sd == None:
            args.sd = 8
    
    p_block = total_blocks[distilled_idx]
    p_total_user = p_block.user.max() + 1
    p_total_item = p_block.item.max() + 1
    
    p_train_dict = total_train_dataset[f"TASK_{distilled_idx}"]
    p_train_interaction, p_train_mat, p_valid_mat, p_test_mat = get_train_valid_test_mat(distilled_idx, total_train_dataset, total_valid_dataset, total_test_dataset)
    R = make_R(p_total_user, p_total_item, p_train_mat)
    
    RRD_SM_path = f"{RRD_SM_dir}/TASK_{distilled_idx}_score_mat.pth"
    if distilled_idx == 0:
        RRD_SM_path = f"../ckpt/{args.dataset}/Teacher/base_teacher/Ensemble/TASK_0_score_mat.pth"
    T_score_mat = torch.load(RRD_SM_path, map_location = gpu)["score_mat"].detach().cpu() # Teacher Score Mat for RRD
    T_sorted_mat = to_np(torch.topk(T_score_mat, k = 1000).indices)
    
    print("[Eval for Teacher]")
    get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, T_sorted_mat, 
                  args.k_list, args.num_task-1)

    if distilled_idx >= 1:
        b_block = total_blocks[distilled_idx - 1]
        b_total_user = b_block.user.max() + 1
        b_total_item = b_block.item.max() + 1

        # train/test/valid for new users/items
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
                
        print("\n[New user result]")
        new_user_results = get_eval_with_mat(new_user_train_mat, new_user_valid_mat, new_user_test_mat, T_sorted_mat, args.k_list)
        print(f"\tvalid_R20 = {new_user_results['valid']['R20']}, test_R20 = {new_user_results['test']['R20']}")
        
    
    Negatvie_exclude = torch.empty((p_total_user, 0))

    # negative exclude
    if distilled_idx > 0:
        S_model_task_path = os.path.join(load_S_model_dir_path, f"TASK_{distilled_idx-1}.pth")
        S_score_mat = torch.load(S_model_task_path, map_location = gpu)["score_mat"].detach().cpu()
        S_interesting_items = get_interesting_items_after_filtering(S_score_mat, p_train_dict)
        S_interesting_items = expand_mat(S_interesting_items, (p_total_user, 40), -1)
        Negatvie_exclude = torch.cat([Negatvie_exclude, S_interesting_items], dim = 1)
        del S_score_mat, S_interesting_items
            
        if distilled_idx >= 2:
            P_model_task_path = os.path.join(load_P_model_dir_path, f"TASK_{distilled_idx-1}.pth")
            P_score_mat = torch.load(P_model_task_path, map_location = gpu)["score_mat"].detach().cpu()
            P_interesting_items = get_interesting_items_after_filtering(P_score_mat, p_train_dict)
            P_interesting_items = expand_mat(P_interesting_items, (p_total_user, 40), -1)
            Negatvie_exclude = torch.cat([Negatvie_exclude, P_interesting_items], dim = 1)
            del P_score_mat, P_interesting_items
            
        CL_model_task_path = os.path.join(load_CL_model_dir_path, f"TASK_{distilled_idx}.pth")
        CL_score_mat = torch.load(CL_model_task_path, map_location = gpu)["score_mat"].detach().cpu()
        CL_interesting_items = get_interesting_items_after_filtering(CL_score_mat, p_train_dict)
        Negatvie_exclude = torch.cat([Negatvie_exclude, CL_interesting_items], dim = 1)
        del CL_score_mat, CL_interesting_items

        
    # RRD interesting_items(users)
    T_RRD_interesting_items = get_interesting_items_after_filtering(T_score_mat, p_train_dict, item_side = False)
    T_RRD_interesting_users = get_interesting_items_after_filtering(T_score_mat, p_train_dict, item_side = True)
    
    # Filtering
    T_score_mat_for_RRD = deepcopy(T_score_mat)

    for user, items in p_train_dict.items(): # train data
        for item in items:
            T_score_mat_for_RRD[user][item] = torch.nan
     
    for user, items in enumerate(Negatvie_exclude): # negative
        T_score_mat_for_RRD[user][items.long()] = torch.nan
        
    for user, items in enumerate(T_RRD_interesting_items): # RRD
        T_score_mat_for_RRD[user][items] = torch.nan
        
    for item, users in enumerate(T_RRD_interesting_users): # IR_RRD
        T_score_mat_for_RRD[users, item] = torch.nan
        
    T_score_mat_for_RRD = torch.where(torch.isnan(T_score_mat_for_RRD), 0.0, 1.0)
    
    # RRD dataset
    RRD_item_ids = torch.arange(p_total_item)
    RRD_train_dataset = RRD_dataset_simple(T_RRD_interesting_items, T_score_mat_for_RRD, num_uninteresting_items = args.nui)
    IR_reg_train_dataset = IR_RRD_dataset_simple(T_RRD_interesting_users, T_score_mat_for_RRD.t(), num_uninteresting_users = args.nuu)    
    
    Negatvie_exclude = torch.cat([Negatvie_exclude, T_RRD_interesting_items], dim = 1)

    # 과거 Task Dataset
    BPR_train_dataset = implicit_CF_dataset(p_total_user, p_total_item, p_train_mat, args.nns, p_train_interaction, Negatvie_exclude) #RRD_train_dataset.interesting_items) # Intersteting items들은 negative sampling에서 제외.        
    train_loader = DataLoader(deepcopy(BPR_train_dataset), batch_size = args.bs, shuffle = True, drop_last = False)
    
    del T_score_mat_for_RRD, Negatvie_exclude, T_RRD_interesting_items, T_RRD_interesting_users, T_score_mat, T_sorted_mat
    # get gpu memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Distilled_model
    if model_type == "LightGCN":
        SNM = get_SNM(p_total_user, p_total_item, R, gpu)
        model_args = [p_total_user, p_total_item, args.sd, gpu, SNM, args.num_layer, args.using_layer_index] # user_count, item_count, dim, gpu, SNM, num_layer, CF_return_average, RRD_return_average
        
    elif model_type == "BPR":
        model_args = [p_total_user, p_total_item, args.sd, gpu]
    
    base_model = eval(model_type)(*model_args)
    base_model = base_model.to(gpu)
    D_Student = PIW_LWCKD(base_model, 
                          LWCKD_flag = False, PIW_flag = False, # task0때는 LWCKD_flag는 False (이전 모델로부터 받을 수가 없음.)
                          temperature = args.T, 
                          num_cluster = args.nc,
                          dim = args.sd, gpu = gpu, model_type = model_type)
    D_Student = D_Student.to(gpu)

    optimizer = optim.Adam(D_Student.parameters(), lr = args.lr, weight_decay = args.reg)
    eval_args = {"best_score" : 0, "test_score" : 0, "best_epoch" : 0, "best_model" : None,
                 "score_mat" : None,  "sorted_mat" : None, "patience" : 0,  "avg_valid_score" : 0, "avg_test_score": 0}
    
    print("[D_Student]\n", D_Student)

############################ Knowledge Distillation ########################################################

    total_time = 0.0
    for epoch in range(args.max_epoch):
        print(f"\n[KD_Epoch : {epoch + 1} / {args.max_epoch}]")
        start_time = time()
        train_loader.dataset.negative_sampling()

        if epoch % args.negative_sample_epoch == 0:
            RRD_train_dataset.sampling_for_uninteresting_items()
            IR_reg_train_dataset.sampling_for_uninteresting_users()
        
        epoch_URRD_loss = 0.0
        epoch_IR_RRD_loss = 0.0
        epoch_CF_loss = 0.0
        
        # CF + RRD
        D_Student.train()
        for mini_batch in train_loader:
            #with torch.cuda.amp.autocast():
            mini_batch = {key : values.to(gpu) for key, values in mini_batch.items()}
            batch_user = mini_batch['user']
            batch_pos_item = mini_batch['pos_item']
            batch_neg_item = mini_batch['neg_item']
            
            ## CF ##
            user_emb, item_emb = D_Student.base_model.get_embedding()
                
            batch_user_emb = user_emb[batch_user]
            batch_pos_item_emb = item_emb[batch_pos_item]
            batch_neg_item_emb = item_emb[batch_neg_item]

            pos_score = (batch_user_emb * batch_pos_item_emb).sum(dim=1, keepdim=True)
            neg_score = (batch_user_emb * batch_neg_item_emb).sum(dim=1, keepdim=True)
            
            base_loss = (pos_score - neg_score).sigmoid()
            base_loss = torch.clamp(base_loss, min = 0.0001, max = 0.9999)
            base_loss = -base_loss.log().sum()
            epoch_CF_loss += base_loss.item()
            
            ## RRD ##
            batch_user = batch_user.unique()
            interesting_items, uninteresting_items = RRD_train_dataset.get_samples(batch_user.cpu())
            
            #batch_user = batch_user.to(gpu)
            interesting_items = interesting_items.to(gpu)#.type(torch.cuda.LongTensor)
            uninteresting_items = uninteresting_items.to(gpu)#.type(torch.cuda.LongTensor)
            
            if model_type == "LightGCN":
                user_emb = D_Student.base_model.user_emb.weight
                item_emb = D_Student.base_model.item_emb.weight
            
            interesting_prediction = forward_multi_items(user_emb, item_emb, batch_user, interesting_items)
            uninteresting_prediction = forward_multi_items(user_emb, item_emb, batch_user, uninteresting_items)
            
            URRD_loss = args.URRD_lambda * relaxed_ranking_loss(interesting_prediction, uninteresting_prediction)
            epoch_URRD_loss += URRD_loss.item()
            
            batch_loss = base_loss + URRD_loss
            
            # backward (Cf + RRD)
            optimizer.zero_grad()
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if args.one_iteration:
                break
          
        ## IR_RRD ##
        iteration = (len(RRD_item_ids) // args.bs) + 1
        shuffle_item_ids = RRD_item_ids[torch.randperm(RRD_item_ids.size(0))]
        for idx in range(iteration):
            
            if idx + 1 == iteration:
                batch_item = shuffle_item_ids[idx * args.bs : ]
            else:
                batch_item = shuffle_item_ids[idx * args.bs : (idx + 1) * args.bs]
                
            # with torch.cuda.amp.autocast():
                
            user_emb, item_emb = D_Student.base_model.get_embedding()
            interesting_users, uninteresting_users = IR_reg_train_dataset.get_samples(batch_item)
            
            batch_item = batch_item.to(gpu)
            interesting_users = interesting_users.to(gpu)
            uninteresting_users = uninteresting_users.to(gpu)
            
            if model_type == "LightGCN":
                user_emb = D_Student.base_model.user_emb.weight
                item_emb = D_Student.base_model.item_emb.weight
            
            interesting_user_prediction = forward_multi_users(user_emb, item_emb, interesting_users, batch_item)
            uninteresting_user_prediction = forward_multi_users(user_emb, item_emb, uninteresting_users, batch_item)

            IR_reg = relaxed_ranking_loss(interesting_user_prediction, uninteresting_user_prediction)
            IR_RRD_loss = args.IR_reg_lmbda * IR_reg
            epoch_IR_RRD_loss += IR_RRD_loss.item()
            
            # backward (IR_RRD)
            optimizer.zero_grad()
            scaler.scale(IR_RRD_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if args.one_iteration:
                break
            
        # Training Result
        end_time = time()
        epoch_time = end_time - start_time
        total_time += epoch_time
        
        epoch_CF_loss = round(epoch_CF_loss / len(train_loader), 4)
        epoch_URRD_loss = round(epoch_URRD_loss / len(train_loader), 4)
        epoch_IR_RRD_loss = round(epoch_IR_RRD_loss / iteration, 4)
            
        print(f"epoch_CF_loss = {epoch_CF_loss}, epoch_URRD_loss = {epoch_URRD_loss}, epoch_IR_RRD_loss = {epoch_IR_RRD_loss}", end = " ")
        print(f"epoch_time = {epoch_time:.4f} seconds, total_time = {total_time:.4f} seconds")

        # Evaluation Result
        if epoch % args.eval_cycle == 0:
            print("\n[Evaluation]")
            D_Student.eval()
            with torch.no_grad():
                D_score_mat, D_sorted_mat = get_sorted_score_mat(D_Student, topk = 1000, return_sorted_mat = True)
            
            valid_list, test_list = get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset,
                                                  D_sorted_mat, args.k_list, current_task_idx = distilled_idx, FB_flag = False, return_value = True)
            
            avg_valid_score = get_average_score(valid_list[:distilled_idx+1], "valid_R20")
            avg_test_score = get_average_score(test_list[:distilled_idx+1], "test_R20")
            
            if args.eval_average_acc:
                valid_score = avg_valid_score
                test_score =  avg_test_score
            else:
                valid_score = valid_list[distilled_idx]["valid_R20"]
                test_score = test_list[distilled_idx]["test_R20"]
            
            if distilled_idx >= 1:
                print("\n[New user result]")
                new_user_results = get_eval_with_mat(new_user_train_mat, new_user_valid_mat, new_user_test_mat, D_sorted_mat, args.k_list)
                new_user_results_txt = f"valid_R20 = {new_user_results['valid']['R20']}, test_R20 = {new_user_results['test']['R20']}"
                print(f"\t{new_user_results_txt}\n")
            else:
                new_user_results_txt = None
                
            if valid_score > eval_args["best_score"]:
                print(f"[Best Model Changed]\n\tvalid_score = {valid_score:.4f}, test_score = {test_score:.4f}")
                eval_args["best_score"] = valid_score
                eval_args["test_score"] = test_score
                eval_args["avg_valid_score"] = avg_valid_score
                eval_args["avg_test_score"] = avg_test_score
                
                eval_args["best_epoch"] = epoch
                eval_args["best_model"] = deepcopy({k: v.cpu() for k, v in D_Student.state_dict().items()}) #D_Student
                eval_args["score_mat"] = deepcopy(D_score_mat)
                eval_args["sorted_mat"] = deepcopy(D_sorted_mat)
                eval_args["patience"] = 0
                
                # best_valid_list = deepcopy(valid_list)
                # best_test_list = deepcopy(test_list)
                best_new_user_results = deepcopy(new_user_results_txt)
            else:
                eval_args["patience"] += 1
                if eval_args["patience"] >= args.early_stop:
                    print("[Early Stopping]")
                    break
         
        # get gpu memory
        gc.collect()
        torch.cuda.empty_cache()
        if args.one_iteration:
            break
    
    # Last Result Report
    print(f"\n[RRD_Best_Result_for_Distilled_idx_{distilled_idx}]")
    print(f"best_epoch = {eval_args['best_epoch']}, valid_score = {eval_args['best_score']}, test_score = {eval_args['test_score']}")
    print(f"avg_valid_score = {eval_args['avg_valid_score']}, avg_test_score = {eval_args['avg_test_score']}")
    
    get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, eval_args["sorted_mat"], args.k_list, 
                  current_task_idx = args.num_task, FB_flag = False, return_value = False)
    
    if distilled_idx >= 1:
        print("\n[New user result]")
        print(f"\t{best_new_user_results}\n")
    
    if args.save:
        if args.save_path is None:        
            save_path = Student_load_path
        else:
            save_path = args.save_path

        save_dir_path = os.path.join(save_path, "Distilled")
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        
        save_path = os.path.join(save_dir_path, f"TASK_{distilled_idx}.pth")
        model_state = {"best_model" : eval_args["best_model"], #.cpu(), 
                       "score_mat"  : eval_args["score_mat"]}
        
        torch.save(model_state, save_path)
        
        print("[Model Saved]\n save_path = ", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # RRD
    parser.add_argument("--RRD_SM_path", help = "the path of the ensembeled teacher's score mat", type = str, 
                        default = "../ckpt/Method/New_Teacher/Ensemble")
    
    parser.add_argument("--negative_sample_epoch", type = int, default = 10)

    # UR-RRD
    parser.add_argument('--URRD_lambda', "--ul", type = float, default = 0.01)
    parser.add_argument("--nii", help = "the number of interesting items", type = int, default = 40)
    parser.add_argument("--nui", help = "the number of uninteresting items", type = int, default = 10000) # 10000)
    
    # IR-RRD
    parser.add_argument('--IR_reg_lmbda', "--il", type = float, default = 0.01)
    parser.add_argument("--niu", help = "the number of interesting users", type = int, default = 40)
    parser.add_argument("--nuu", help = "the number of uninteresting users", type = int, default = 11500) #11500
    
    # LWCKD + PIW
    parser.add_argument("--nc", help = "num_cluster", type = int, default = 10)
    parser.add_argument("--T", help = "temperature", type = int, default = 5.)
    parser.add_argument("--LWCKD_flag", "--lf", action = argparse.BooleanOptionalAction, help = "whether using LWC_KD or not (--lf or --no-lf)")
    parser.add_argument("--PIW_flag", "--pf", action = argparse.BooleanOptionalAction, help = "whether using PIW or not (--pf or --no-pf)")
    parser.add_argument("--LWCKD_lambda", type = float, default = 0.01)
    parser.add_argument("--cluster_lambda", type = float, default = 1)
    
    # setup
    parser.add_argument("--gpu", type = int, default = 0)
    parser.add_argument("--bs", help = "batch_size", type = int, default = 1024)
    parser.add_argument("--nns", help = "the number of negative sample", type = int, default = 1)
    parser.add_argument("--sd", help = "student_dims", type = int, default = None)
    parser.add_argument('--lr', type = float, default = 0.005)
    parser.add_argument("--reg", type = float, default = 0.0001)
    parser.add_argument('--max_epoch', type = int, default = 100)
    parser.add_argument('--early_stop', type = int, default = 2)
    parser.add_argument('--eval_cycle', type = int, default = 5)
    parser.add_argument("--eval_average_acc", "--eaa", action = argparse.BooleanOptionalAction, help = "whether saving param or not (--eaa or --no-eaa)")
    parser.add_argument('--k_list', type = list, default = [20, 50, 100])
    parser.add_argument("--num_task", type = int, default = 6)
    parser.add_argument("--random_init", "--r", action = argparse.BooleanOptionalAction, help = "whether saving param or not (--r or --no-r)")

    # LightGCN
    parser.add_argument('--num_layer', type = int, default = 2)
    parser.add_argument('--using_layer_index', type = str, default = "avg", help = "first, last, avg")

    # SAVE
    parser.add_argument("--one_iteration", "--oi", action = argparse.BooleanOptionalAction, help = "whether saving param or not (--s or --no-s)")
    parser.add_argument("--save", "--s", action = argparse.BooleanOptionalAction, help = "whether saving param or not (--s or --no-s)")
    parser.add_argument('--save_path', "--sp", type = str,
                        default = "../ckpt/Yelp/Student/LightGCN_1/Method_test")
    
    # choose
    parser.add_argument('--dataset', "--d", type = str, default = None, help = "Gowalla or Yelp")
    parser.add_argument("--model", "-m", type = str, help = "LightGCN_0 or BPR_0")
    parser.add_argument('--target_task', "--tt", type = int, default = -1)

    args = parser.parse_args()
    print_command_args(args)
    main(args)
    print_command_args(args)
 
# python KD.py --d Yelp -m LightGCN_1 --tt 1 --s --max_epoch 10