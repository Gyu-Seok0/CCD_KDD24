import argparse
import random
import time
import gc
import sys
import os
from copy import deepcopy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from Utils.data_loaders import *
from Utils.utils import *

def main(args):
    """ Main function for training and evaluation in Stage3:Teacher update """

    gpu = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    scaler = torch.cuda.amp.GradScaler()
    
    # Validate dataset and model
    assert args.dataset in ["Gowalla", "Yelp"], "Dataset must be either 'Gowalla' or 'Yelp'."
    
    model_type, model_seed = args.teacher.split("_")
    
    # Random Seed
    print(f"Random_seed = {model_seed}")
    set_random_seed(int(model_seed))
    
## Load data
    data_path = f"../dataset/{args.dataset}/total_blocks_timestamp.pickle"
    data_dict_path = f"../dataset/{args.dataset}"
    
    total_blocks = load_pickle(data_path)
    max_item = load_pickle(data_path)[-1].item.max() + 1
    total_train_dataset, total_valid_dataset, total_test_dataset, total_item_list = load_data_as_dict(data_dict_path, num_task = args.num_task)
    
    task_idx = args.target_task
    distillation_idx = task_idx - 1
    
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
    
    b_R = make_R(b_total_user, b_total_item, b_train_mat)
    p_R = make_R(p_total_user, p_total_item, p_train_mat)
    
    p_user_ids = torch.tensor(sorted(p_block.user.unique()))
    p_item_ids = torch.tensor(sorted(p_block.item.unique()))
    b_user_ids = torch.tensor(sorted(list(b_train_dict.keys())))#.to(gpu)
    b_item_ids = torch.tensor(sorted(total_item_list[f"TASK_{task_idx - 1}"]))#.to(gpu)
    
    _, b_user_mapping, b_item_mapping, b_rating_mat, UU, II = None, None, None, None, None, None
    
    # Train/test/valid data split for new users/items
    new_user_train_mat, new_user_valid_mat, new_user_test_mat = get_train_valid_test_mat_for_new_users(b_total_user, p_total_user, p_train_mat, p_valid_mat, p_test_mat)
    
## Load Models
    if args.S_load_path is None:
       args.S_load_path = f"../ckpt/{args.dataset}/Student/{args.student}/Method"
    
    if args.T_load_path is None:
        args.T_load_path = f"../ckpt/{args.dataset}/Teacher/using_student_{args.student}/Method" # The student should be specified because the student and teacher collaboratively evolve along the data stream in our proposed CCD framework.
    
    print(f"Student = {args.student} (with low dimensionailty), S_load_path = {args.S_load_path}")
    print(f"Teacher = {args.teacher} (with high dimensionailty), T_load_path = {args.T_load_path}")
    
    # Load the path of student-side models (S_proxy, P_proxy, Student)
    load_D_model_dir_path = f"{args.S_load_path}/Distilled"
    load_S_model_dir_path = f"{args.S_load_path}/Stability"
    load_P_model_dir_path = f"{args.S_load_path}/Plasticity"
    load_CL_model_dir_path = f"{args.S_load_path}/CL"
    
    # Load the path of teacher
    RRD_SM_dir_path = f"{args.T_load_path}/Ensemble"
        
    # Load teacher
    RRD_SM_path = f"{RRD_SM_dir_path}/TASK_{distillation_idx}_score_mat.pth"
    if distillation_idx == 0:
        RRD_SM_path = f"../ckpt/{args.dataset}/Teacher/base_teacher/Ensemble/TASK_0_score_mat.pth"
    T_score_mat = torch.load(RRD_SM_path, map_location = gpu)["score_mat"].detach().cpu() #RRD_SM[f"TASK_{distillation_idx}"]
    FT_score_mat = filtering_simple(T_score_mat, b_train_dict).detach().cpu()
    T_RRD_interesting_items = torch.topk(FT_score_mat, k = 40, dim = 1).indices
    negatvie_exclude = T_RRD_interesting_items.clone().detach()
    del RRD_SM_path, T_score_mat, FT_score_mat, T_RRD_interesting_items
    
    # Load students
    S_score_mat, P_score_mat, CL_score_mat = None, None, None
    
    # S proxy
    if args.Using_S:
        S_model_task_path = os.path.join(load_S_model_dir_path, f"TASK_{distillation_idx}.pth")
        _, S_score_mat, S_sorted_mat = load_saved_model(S_model_task_path, gpu)
        
        print("\n[Evaluation for S_proxy]")
        get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, 
                        S_sorted_mat, args.k_list, current_task_idx = task_idx, FB_flag = False, return_value = False)
        
        negatvie_exclude = torch.cat([negatvie_exclude, torch.tensor(S_sorted_mat[:, :40])], dim = 1)
        del S_sorted_mat
    
    # P proxy
    if (distillation_idx > 0 and args.Using_P) or (distillation_idx == 0 and args.Using_P and args.Using_S != True): #and args.P_model_path != "None":
        P_model_task_path = os.path.join(load_P_model_dir_path, f"TASK_{distillation_idx}.pth")
        _, P_score_mat, P_sorted_mat = load_saved_model(P_model_task_path, gpu)
        
        print("\n[Evaluation for P_proxy]")
        get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, 
                    P_sorted_mat, args.k_list, current_task_idx = task_idx, FB_flag = False, return_value = False)
        
        negatvie_exclude = torch.cat([negatvie_exclude, torch.tensor(P_sorted_mat[:, :40])], dim = 1)
        del P_sorted_mat
        
        if (distillation_idx == 0 and args.Using_P and args.Using_S != True):
            args.P_sample = args.S_sample
    
    # Student via continual update
    if args.Using_CL:    
        CL_model_task_path = os.path.join(load_CL_model_dir_path, f"TASK_{task_idx}.pth")
        _, CL_score_mat, CL_sorted_mat = load_saved_model(CL_model_task_path, gpu)
        
        print("\n[Evaluation for CL_Student]")
        get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, 
                        CL_sorted_mat, args.k_list, current_task_idx = task_idx, FB_flag = False, return_value = False)
        
        u_size, i_size = negatvie_exclude.shape
        negatvie_exclude_expand = torch.full((p_total_user, i_size), -1.0)
        negatvie_exclude_expand[:u_size, :i_size] = negatvie_exclude
        negatvie_exclude = torch.cat([negatvie_exclude_expand, torch.tensor(CL_sorted_mat[:, :40])], dim = 1)
        del CL_sorted_mat
        
    # Dataset / DataLoader
    if model_type in ["BPR", "LightGCN"]:
        train_dataset = implicit_CF_dataset(p_total_user, p_total_item, p_train_mat, args.nns, p_train_interaction, negatvie_exclude)
    elif model_type == "VAE":
        train_dataset = implicit_CF_dataset_AE(p_total_user, max_item, p_train_mat, is_user_side = True)
    train_loader = DataLoader(train_dataset, batch_size = args.bs, shuffle = True, drop_last = False)
        
    # Load Teacher model
    Teacher, T_sorted_mat = get_teacher_model(model_type, b_total_user, b_total_item, b_R, task_idx, max_item, gpu, args) 
    print(f"\n[Teacher]\n{Teacher}")
    
    print("\n[[Before Update] Evalutation for Teacher]")
    get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, 
                    T_sorted_mat, args.k_list, current_task_idx = task_idx, FB_flag = False, return_value = False)
    
    # Increase the model size due to new users/items.
    Teacher, T_score_mat, T_sorted_mat = Teacher_update(model_type, Teacher, b_total_user, b_total_item, b_user_ids, b_item_ids, b_train_dict, 
                                           p_total_user, p_total_item, p_R, p_user_ids, p_item_ids,
                                           num_new_user, num_new_item, gpu, train_loader, args)

    print("\n[[After Update] Evalutation for Teacher]")
    get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, 
                    T_sorted_mat, args.k_list, current_task_idx = task_idx, FB_flag = False, return_value = False)

################################### Compose the initial dataset of replay learning (it adaptively changes through training) ##########################################################################################################################################################

    if args.replay_learning:
        if model_type in ["BPR", "LightGCN"]:
            S_score_mat = torch.sigmoid(S_score_mat) if S_score_mat is not None else None
            P_score_mat = torch.sigmoid(P_score_mat) if P_score_mat is not None else None
            CL_score_mat = torch.sigmoid(CL_score_mat) if CL_score_mat is not None else None
            
        elif model_type == "VAE":
            S_score_mat = F.softmax(S_score_mat, dim=-1) if S_score_mat is not None else None
            P_score_mat = F.softmax(P_score_mat, dim=-1) if P_score_mat is not None else None
            CL_score_mat = F.softmax(CL_score_mat, dim=-1) if CL_score_mat is not None else None
        
        S_rank_mat = convert_to_rank_mat(S_score_mat) if S_score_mat is not None else None
        P_rank_mat = convert_to_rank_mat(P_score_mat) if P_score_mat is not None else None
        CL_rank_mat = convert_to_rank_mat(CL_score_mat) if CL_score_mat is not None else None
        
        replay_learning_dataset = get_total_replay_learning_dataset_Teacher(T_score_mat, S_score_mat, S_rank_mat, P_score_mat, P_rank_mat, CL_score_mat, CL_rank_mat, args)
        
        #If the model is VAE, we use pseudo-labeling by imputing the replay_learning_dataset with args.VAE_replay_learning_value.
        if model_type == "VAE":
            train_loader = get_VAE_replay_learning_loader_integrate_with_R(replay_learning_dataset, p_R, p_total_user, max_item, args)

################################### Stage3: Teacher update ##########################################################################################################################################################        

    if model_type in ["BPR", "LightGCN"]:
        param = [{"params" : Teacher.parameters()}, {"params" : Teacher.cluster}]
        loss_type = ["base", "UI", "IU", "UU", "II", "cluster"]
    
    elif model_type == "VAE":
        param = Teacher.parameters()
        loss_type = ["base", "kl"]
        
    optimizer = optim.Adam(param, lr = args.lr, weight_decay = args.reg)
    criterion = nn.BCEWithLogitsLoss(reduction = 'sum')

    eval_args = {"best_score" : 0, "test_score" : 0, "best_epoch" : 0, "best_model" : None, "score_mat" : None,  "sorted_mat" : None, "patience" : 0,  "avg_valid_score" : 0, "avg_test_score": 0}
    total_time = 0
    
    # Get gpu memory
    gc.collect()
    torch.cuda.empty_cache()

    for epoch in range(args.max_epoch):
        print(f"\n[Epoch:{epoch + 1}/{args.max_epoch}]")
        epoch_loss = {f"epoch_{l}_loss": 0.0 for l in loss_type}
        epoch_replay_learning_loss = 0.0
        start_time = time.time()
        train_loader.dataset.negative_sampling()

        Teacher.train()
        for mini_batch in train_loader:
            # Forward
            if model_type in ["BPR", "LightGCN"]:
                base_loss, UI_loss, IU_loss, UU_loss, II_loss, cluster_loss = Teacher(mini_batch)
                batch_loss = base_loss + args.LWCKD_lambda * (UI_loss + IU_loss + UU_loss + II_loss) + (args.cluster_lambda * cluster_loss)
            
            elif model_type == "VAE":
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
        
        CF_time = time.time()
        print(f"{epoch_loss}, CF_time = {CF_time - start_time:.4f} seconds", end = " ")
                    
############################################ replay_learning ############################################

        if args.replay_learning and model_type in ["BPR", "LightGCN"]:
            if args.annealing:
                replay_learning_lambda = args.replay_learning_lambda * torch.exp(torch.tensor(-epoch)/args.T)
            else:
                replay_learning_lambda = args.replay_learning_lambda
                    
            # Data shuffle
            shuffled_list = random.sample(replay_learning_dataset, len(replay_learning_dataset))
            u_list, i_list, r_list = list(zip(*shuffled_list))
            iteration = (len(shuffled_list) // args.bs) + 1
            epoch_replay_learning_loss = 0.0
                            
            for idx in range(iteration):
                if idx + 1 == iteration:
                    start, end = idx * args.bs , -1
                else:
                    start, end = idx * args.bs , (idx + 1) * args.bs
                
                # Batch    
                batch_user = torch.tensor(u_list[start : end], dtype=torch.long).to(gpu)
                batch_item = torch.tensor(i_list[start : end], dtype=torch.long).to(gpu)
                batch_label = torch.tensor(r_list[start : end], dtype=torch.float16).to(gpu)
            
                user_emb, item_emb = Teacher.base_model.get_embedding()
                batch_user_emb = user_emb[batch_user]
                batch_item_emb = item_emb[batch_item]
                
                # Forward
                output = (batch_user_emb * batch_item_emb).sum(1)
                batch_loss = criterion(output, batch_label)
                batch_loss *= replay_learning_lambda

                # Backward
                optimizer.zero_grad()
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_replay_learning_loss += batch_loss.item()

            replay_learning_time = time.time()
            print(f"epoch_replay_learning_loss = {round(epoch_replay_learning_loss/iteration, 4)}, replay_learning_lambda = {replay_learning_lambda:.5f}, replay_learning_time = {replay_learning_time - CF_time:.4f} seconds", end = " ")

        # Time
        end_time = time.time()
        epoch_time = end_time - start_time
        total_time += epoch_time
        print(f"epoch_time = {epoch_time:.4f} seconds, total_time = {total_time:.4f} seconds")

############################################ EVAL ############################################

        if epoch % 5 == 0:
            print("\n[Evaluation]")
            Teacher.eval()
            with torch.no_grad():
                if model_type in ["BPR", "LightGCN"]:
                    T_score_mat, T_sorted_mat = get_sorted_score_mat(Teacher, topk = 1000, return_sorted_mat = True)
                
                elif model_type == "VAE":
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
                
            print(f"\t[The Result of new users in {task_idx}-th Block]")
            new_user_results = get_eval_with_mat(new_user_train_mat, new_user_valid_mat, new_user_test_mat, T_sorted_mat, args.k_list)
            new_user_results_txt = f"valid_R20 = {new_user_results['valid']['R20']}, test_R20 = {new_user_results['test']['R20']}"
            print(f"\t{new_user_results_txt}\n")

            if valid_score > eval_args["best_score"]:
                print(f"\t[Best Model Changed]\n\tvalid_score = {valid_score:.4f}, test_score = {test_score:.4f}")
                eval_args["best_score"] = valid_score
                eval_args["test_score"] = test_score
                eval_args["avg_valid_score"] = avg_valid_score
                eval_args["avg_test_score"] = avg_test_score
                
                eval_args["best_epoch"] = epoch
                eval_args["best_model"] = deepcopy({k: v.cpu() for k, v in Teacher.state_dict().items()}) #deepcopy(Teacher)
                eval_args["score_mat"] = deepcopy(T_score_mat)
                eval_args["sorted_mat"] = deepcopy(T_sorted_mat)
                eval_args["patience"] = 0
                
                best_new_user_results = deepcopy(new_user_results_txt)
            else:
                eval_args["patience"] += 1
                if eval_args["patience"] >= args.early_stop:
                    print("[Early Stopping]")
                    break
            
            if args.replay_learning and epoch > 0:
                replay_learning_dataset = get_total_replay_learning_dataset_Teacher(T_score_mat, S_score_mat, S_rank_mat, P_score_mat, P_rank_mat, CL_score_mat, CL_rank_mat, args)
                if model_type == "VAE":
                    train_loader =  get_VAE_replay_learning_loader_integrate_with_R(replay_learning_dataset, p_R, p_total_user, max_item, args)  
        
        # Get gpu memory
        gc.collect()
        torch.cuda.empty_cache()
    
    # Final result
    print(f"\n[Final result of teacher's update in the {task_idx}-th data block]")
    print(f"best_epoch = {eval_args['best_epoch']}, valid_score = {eval_args['best_score']}, test_score = {eval_args['test_score']}")
    print(f"avg_valid_score = {eval_args['avg_valid_score']}, avg_test_score = {eval_args['avg_test_score']}")
    
    get_CL_result(total_train_dataset, total_valid_dataset, total_test_dataset, eval_args["sorted_mat"], args.k_list, 
                    current_task_idx = args.num_task, FB_flag = False, return_value = False)
        
    print(f"\t[The Result of new users in {task_idx}-th Block]")
    print(f"\t{best_new_user_results}\n")
    
    # Model save
    if args.save:
        print("\n[Model Save]")
        
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
    parser.add_argument("--annealing", "--a", action = "store_true", default = True, help = "using annealing")
    parser.add_argument('--random_init', "--r", action = "store_true", default = False, help = 'random_initalization for new user/items')
    parser.add_argument("--absolute", type = float, default = 100)
    parser.add_argument("--replay_learning_lambda", type = float, default = 0.5)

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
    parser.add_argument("--VAE_replay_learning_value", type = float, default = 0.1)
    parser.add_argument("--kl_lambda", '-kl', type = float, default = 1.0)
    
    # S & P proxies
    parser.add_argument("--target_task", "--tt", help = "target_task", type = int, default = -1)
    parser.add_argument("--replay_learning", "--rl", action = argparse.BooleanOptionalAction, help = "whether using param or not (--replay_learning or --no-replay_learning)")
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
    
    # Teacher's embedding dimension
    if args.dataset == "Gowalla":
        args.td = 64
    elif args.dataset == "Yelp":
        args.td = 128
        
    print_command_args(args)
    main(args)
    print_command_args(args)
    
    #run code: python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_0 --tt 5 --rl --UCL --US --UP --ab 100 --ss 1 --ps 5 --cs 1 --max_epoch 10
