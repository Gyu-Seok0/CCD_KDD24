# Gyuseok's custom
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import gc

from Utils.data_loaders import *
from utils import *

from Models.BPR import BPR
from Models.VAE import VAE
from Models.LightGCN_V2 import LightGCN



def main(args):
    
    # GPU
    gpu = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[GPU] {gpu}")
    
    # random seed for dataset
    np.random.seed(args.rs)
    
    # dataset
    base_block = load_pickle(args.data_path)[0]
    max_user = load_pickle(args.data_path)[-1].user.max() + 1
    max_item = load_pickle(args.data_path)[-1].item.max() + 1
        
    total_user = base_block.user.max() + 1
    total_item = base_block.item.max() + 1
    
    task_data = load_pickle(args.base_data_dict_path)
    train_dict = task_data["train_dict"]
    valid_dict = task_data["valid_dict"]
    test_dict = task_data["test_dict"]
    item_list = task_data["item_list"]
    
    train_interaction = make_interaction(train_dict)
    train_mat = make_rating_mat(train_dict)
    valid_mat = make_rating_mat(valid_dict)
    test_mat = make_rating_mat(test_dict)
    
    test_dataset = implicit_CF_dataset_test(total_user, total_item, valid_mat, test_mat)
    VAE_train_dataset = implicit_CF_dataset_AE(total_user, max_item, train_mat, is_user_side = True)
    BPR_train_dataset = implicit_CF_dataset(total_user, total_item, train_mat, args.nns, train_interaction)
    
    # Data Statistics
    # for dic in ["train_dict", "valid_dict", "test_dict"]:
    #     print(f"num_{dic} = {len(sum(eval(dic).values(), []))}")

    # Model
    model_type, model_seed = args.model.split("_")
    if model_type == "VAE":
        model_args = [total_user, max_item, args.dim, gpu]
        
    elif model_type == "LightGCN":
        R = make_R(total_user, total_item, train_mat)
        SNM = get_SNM(total_user, total_item, R, gpu)
        model_args = [total_user, total_item, args.dim, gpu, SNM, args.num_layer, args.using_layer_index] # user_count, item_count, dim, gpu, SNM, num_layer, CF_return_average, RRD_return_average
        
    elif model_type == "BPR":
        model_args = [total_user, total_item, args.dim, gpu]
    
    model = eval(model_type)(*model_args)
    model = model.to(gpu)
    print(f"[{args.model}]\n {model}")
    
    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.reg)
    loss_type = ["base"]
    
    # DataLoader
    train_dataset = None
    if model_type in ["BPR", "LightGCN"]:
        train_dataset = BPR_train_dataset
    elif model_type == "VAE":
        train_dataset = VAE_train_dataset
        
    if train_dataset is not None:
        train_loader = DataLoader(train_dataset, batch_size = args.bs, shuffle = True, drop_last = False) # 
        eval_train_loader = DataLoader(train_dataset, batch_size = args.bs, shuffle = False, drop_last = False)
    else:
        raise NotImplementedError("Train_dataset is not assigned")
    
    # save path
    save_model_path = None
    if args.save:
        save_dir_path = os.path.join(args.save_path, args.model)                    
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        save_model_path = os.path.join(save_dir_path, f"Base_Model.pth")
    
    eval_args = {"best_score" : 0, "best_epoch" : 0, "best_model" : None, "test_score" : 0.0, "patience" : 0, 
                 "save_flag" : args.save, "save_path": save_model_path}
    
    # get gpu memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # seed
    set_random_seed(int(model_seed))
    
    for epoch in range(args.max_epoch):
        
        print(f"\n[Epoch:{epoch + 1}/{args.max_epoch}]")
        report = {}

        print("[Training]")
        model.train()
        #train_epoch(train_loader, loss_type, model, model_type, optimizer, scaler, args, gpu, report)
        train_epoch_base_model(train_loader, loss_type, model, optimizer, scaler, gpu, report)

        print("[Evaluating]")
        model.eval()
        eval_epoch(model, gpu, train_loader, test_dataset, args.k_list, report, epoch, eval_args)

        # report
        print("[Report]\n",report)
        
        if eval_args["patience"] >= args.early_stop:
            print("[Early Stopping]")
            break
    
    print("[BASE_MODEL_RESULT]")
    print(f"best_epoch = {eval_args['best_epoch']}, best_valid_score = {eval_args['best_score']}, test_score = {eval_args['test_score']}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type = int, default = 0)
    parser.add_argument('--rs', type = int, default = 0, help = "random seed")

    parser.add_argument("--data_path", type = str, default = "./dataset/Yelp/total_blocks_timestamp.pickle")
    parser.add_argument("--base_data_dict_path", type = str, default = "./dataset/Yelp/TASK_0.pickle")
    parser.add_argument("--nns", help = "the number of negative sample", type = int, default = 2)
    parser.add_argument("--model", "-m", type = str, default = "BPR_0")
    parser.add_argument("--dim", type = int, default = 64)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument("--reg", type = float, default = 0.0001)
    parser.add_argument("--bs", help = "batch_size", type = int, default = 512)

    parser.add_argument("--save", "--s", action = argparse.BooleanOptionalAction)
    parser.add_argument("--save_path", type = str, default = "./ckpt/Yelp/Base_Model")
    
    parser.add_argument('--max_epoch', type = int, default = 100)
    parser.add_argument('--early_stop', type = int, default = 10)
    parser.add_argument('--k_list', type = list, default = [5, 10, 20])
    
    # LightGCN
    parser.add_argument('--num_layer', type = int, default = 2)
    parser.add_argument('--using_layer_index', type = str, default = "avg", help = "first, last, avg")

    args = parser.parse_args()
    
    print_command_args(args)
    main(args)
    print_command_args(args)