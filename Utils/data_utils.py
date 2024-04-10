import numpy as np
import os
import random
import pickle
import time
import torch
import copy
from Utils.evaluation import *

def full_print_result(epoch, max_epoch, train_loss, eval_results, is_improved=False, train_time=0., test_time=0.):

	if is_improved:
		print('Epoch [{}/{}], Train Loss: {:.4f}, Elapsed: Train: {:.2f} Test: {:.2f} *' .format(epoch, max_epoch, train_loss, train_time, test_time))
	else: 
		print('Epoch [{}/{}], Train Loss: {:.4f}, Elapsed: Train: {:.2f} Test: {:.2f}' .format(epoch, max_epoch, train_loss, train_time, test_time))


	for mode in ['valid', 'test']:
		for topk in [10, 20, 50]:
			p = eval_results[mode]['P' + str(topk)]
			r = eval_results[mode]['R' + str(topk)] 
			n = eval_results[mode]['N' + str(topk)] 

			print('{} P@{}: {:.4f}, R@{}: {:.4f}, N@{}: {:.4f}'.format(mode, topk, p, topk, r, topk, n))

		print()

def Euclidian_dist(user_mat, item_mat):
	A = (user_mat ** 2).sum(1, keepdim=True)
	B = (item_mat ** 2).sum(1, keepdim=True)
	
	AB = -2 * torch.matmul(user_mat, item_mat.T)
	
	return torch.sqrt(A + AB + B.T)   


def to_np(x):
	return x.data.cpu().numpy()


def dict_set(base_dict, user_id, item_id, val):
	if user_id in base_dict:
		base_dict[user_id][item_id] = val
	else:
		base_dict[user_id] = {item_id: val}


def is_visited(base_dict, user_id, item_id):
	if user_id in base_dict and item_id in base_dict[user_id]:
		return True
	else:
		return False


# save_pickle(path, 'top_results', sorted_mat[:,:100])
def save_pickle(path, filename, obj):
	with open(path + filename, 'wb') as f:
		pickle.dump(obj, f)


def load_pickle(path, filename):
	with open(path + filename, 'rb') as f:
		obj = pickle.load(f)

	return obj


def list_to_dict(base_list):
	result = {}
	for user_id, item_id, value in base_list:
		dict_set(result, user_id, item_id, value)
	
	return result


def dict_to_list(base_dict):
	result = []

	for user_id in base_dict:
		for item_id in base_dict[user_id]:
			result.append((user_id, item_id, 1))
	
	return result
	
	
def read_citeULike(f):
	
	# 전체 파일 읽기
	total_interactions = []
	user_count, item_count = 0, 0

	for user_id, line in enumerate(f.readlines()):
		items = line.split(' ')[1:]
		
		user_count = max(user_count, user_id)
		for item in items:
			item_id = int(item)
			item_count = max(item_count, item_id)
			
			total_interactions.append((user_id, item_id, 1))

	return user_count + 1, item_count + 1, total_interactions



def get_count_dict(total_interactions, spliter="\t"):

	user_count_dict, item_count_dict = {}, {}

	for line in total_interactions:
		user, item, rating = line
		user, item, rating = int(user), int(item), float(rating)

		if user in user_count_dict:
			user_count_dict[user] += 1
		else: 
			user_count_dict[user] = 1

		if item in item_count_dict:
			item_count_dict[item] += 1
		else: 
			item_count_dict[item] = 1

	return user_count_dict, item_count_dict


def get_total_interactions(total_interaction_tmp, user_count_dict, item_count_dict, is_implicit=True, count_filtering = [10, 10], spliter="\t"):

	# 전체 파일 읽기
	total_interactions = []
	user_dict, item_dict = {}, {}
	user_count, item_count = 0, 0

	for line in total_interaction_tmp:
		user, item, rating = line
		user, item, rating = int(user), int(item), float(rating)

		# count filtering
		if user_count_dict[user] < count_filtering[0]:
			continue
		if item_count_dict[item] < count_filtering[1]:
			continue

		# user indexing
		if user in user_dict:
			user_id = user_dict[user]
		else:
			user_id = user_count
			user_dict[user] = user_id
			user_count += 1

		# item indexing
		if item in item_dict:
			item_id = item_dict[item]
		else:
			item_id = item_count
			item_dict[item] = item_id
			item_count += 1

		if is_implicit:
			rating = 1.

		total_interactions.append((user_id, item_id, rating))

	return user_count + 1, item_count + 1, total_interactions




def get_total_interactions_time(total_interaction_tmp, user_count_dict, item_count_dict, is_implicit=True, count_filtering = [10, 10], spliter="\t"):

	# 전체 파일 읽기
	total_interactions = []
	user_dict, item_dict = {}, {}
	user_count, item_count = 0, 0

	for line in total_interaction_tmp:
		user, item, time = line
		user, item, time = int(user), int(item), int(time)

		# count filtering
		if user_count_dict[user] < count_filtering[0]:
			continue
		if item_count_dict[item] < count_filtering[1]:
			continue

		# user indexing
		if user in user_dict:
			user_id = user_dict[user]
		else:
			user_id = user_count
			user_dict[user] = user_id
			user_count += 1

		# item indexing
		if item in item_dict:
			item_id = item_dict[item]
		else:
			item_id = item_count
			item_dict[item] = item_id
			item_count += 1

		total_interactions.append((user_id, item_id, time))

	return user_count, item_count, total_interactions


def load_citeULike(path, filename, test_ratio=0.2, random_seed=0):
	
	np.random.seed(0)
	
	with open(path + '/' + filename, 'r') as f:
		user_count, item_count, total_interaction_tmp = read_citeULike(f)

	user_count_dict, item_count_dict = get_count_dict(total_interaction_tmp)
	user_count, item_count, total_interactions = get_total_interactions(total_interaction_tmp, user_count_dict, item_count_dict, count_filtering = [10, 10])
	
	total_mat = list_to_dict(total_interactions)

	train_mat, valid_mat, test_mat = {}, {}, {}

	for user in total_mat:
		items = list(total_mat[user].keys())
		np.random.shuffle(items)

		num_test_items = int(len(items) * test_ratio)
		test_items = items[:num_test_items]
		valid_items = items[num_test_items: num_test_items*2]
		train_items = items[num_test_items*2:]

		for item in test_items:
			dict_set(test_mat, user, item, 1)

		for item in valid_items:
			dict_set(valid_mat, user, item, 1)

		for item in train_items:
			dict_set(train_mat, user, item, 1)
			
	train_mat_R = {}

	for user in train_mat:
		for item in train_mat[user]:
			dict_set(train_mat_R, item, user, 1)
			
	for user in list(valid_mat.keys()):
		for item in list(valid_mat[user].keys()):
			if item not in train_mat_R:
				del valid_mat[user][item]
		if len(valid_mat[user]) == 0:
			del valid_mat[user]
			del test_mat[user]
			
	for user in list(test_mat.keys()):
		for item in list(test_mat[user].keys()):
			if item not in train_mat_R:
				del test_mat[user][item]
		if len(test_mat[user]) == 0:
			del test_mat[user]
			del valid_mat[user]
	
	train_interactions = []
	for user in train_mat:
		for item in train_mat[user]:
			train_interactions.append([user, item, 1])
			
	return user_count, item_count, train_mat, train_interactions, valid_mat, test_mat

def get_g_setting(train_mat, alpha=0.2, random_seed=None):
	
	if random_seed:
		np.random.seed(random_seed)

	users = list(train_mat.keys())
	np.random.shuffle(users)
	
	num_remove_users = int(len(users) * alpha)
	remove_users = users[:num_remove_users]
	train_users = users[num_remove_users:]    
	
	train_mat1, train_mat2 = {}, {}
	
	for user in train_mat:
		
		if user in remove_users:
			for item in train_mat[user]:
				dict_set(train_mat2, user, item, 1)    
			continue
		
		items = list(train_mat[user].keys())
		np.random.shuffle(items)
		
		num_remove_items = int(len(items) * alpha)
		remove_items = items[:num_remove_items]
		train_items = items[num_remove_items:]
		
		for item in train_items:
			dict_set(train_mat1, user, item, 1)
			dict_set(train_mat2, user, item, 1) 

		for item in remove_items:
			dict_set(train_mat2, user, item, 1)        
			
	train_interactions1, train_interactions2 = [], []
	for user in train_mat1:
		for item in train_mat1[user]:
			train_interactions1.append([user, item, 1])        
			
	for user in train_mat2:
		for item in train_mat2[user]:
			train_interactions2.append([user, item, 1])        
			 
	return train_mat1, train_mat2, train_interactions1, train_interactions2



def get_user_item_count_dict(interactions):

	user_count_dict = {}
	item_count_dict = {}

	for user, item, _ in interactions:
		if user not in user_count_dict:
			user_count_dict[user] = 1
		else:
			user_count_dict[user] += 1

		if item not in item_count_dict:
			item_count_dict[item] = 1
		else:
			item_count_dict[item] += 1

	return user_count_dict, item_count_dict


def get_adj_mat(user_count, item_count, train_interactions):

	user_count_dict, item_count_dict = get_user_item_count_dict(train_interactions)

	A_indices, A_values = [[], []], []
	A_T_indices, A_T_values = [[], []], []
	for user, item, _ in train_interactions:
		A_indices[0].append(user)
		A_indices[1].append(item)
		A_values.append(1 / (user_count_dict[user] * item_count_dict[item]))

		A_T_indices[0].append(item)
		A_T_indices[1].append(user)
		A_T_values.append(1 / (user_count_dict[user] * item_count_dict[item]))

	A_indices = torch.LongTensor(A_indices)
	A_values = torch.FloatTensor(A_values)

	A = torch.sparse.FloatTensor(A_indices, A_values, torch.Size([user_count, item_count]))

	A_T_indices = torch.LongTensor(A_T_indices)
	A_T_values = torch.FloatTensor(A_T_values)

	A_T = torch.sparse.FloatTensor(A_T_indices, A_T_values, torch.Size([item_count, user_count]))

	return A, A_T


def relaxed_ranking_loss(S1, S2):

	above = S1.sum(1, keepdims=True)

	below1 = S1.flip(-1).exp().cumsum(1)		
	below2 = S2.exp().sum(1, keepdims=True)		

	below = (below1 + below2).log().sum(1, keepdims=True)
	
	return -(above - below).sum()

def ListMLE(S):
	above = S.sum(1, keepdims=True)
	below = (S.flip(-1).exp().cumsum(1) + 1e-5).log().sum(1, keepdims=True)

	# print(float(S.max()), float(S.min()), float(-(above - below).sum()))

	return -(above - below).sum()

def min_max_norm(X):

	X_min = X.min().detach()
	X_max = X.max().detach()

	return (X - X_min)/ (X_max - X_min)


def read_ml1m(f):

	# 전체 파일 읽기
	total_interactions = []
	user_count, item_count = 0, 0

	for line in f.readlines():
		user_id, item_id, rating, timestamp = line.split('::')
		user_id = int(user_id)
		item_id = int(item_id)
		rating = int(rating)
		timestamp = float(timestamp)

		if rating < 4: continue
		
		user_count = max(user_count, user_id)
		item_count = max(item_count, item_id)

		total_interactions.append((user_id, item_id, timestamp))

	return user_count, item_count, total_interactions




def read_ciao(f):

	# read all
	total_interactions = []
	user_count, item_count = 0, 0

	for line in f.readlines():
		user_id, item_id = line.split()
		user_id = int(float(user_id))
		item_id = int(float(item_id))
		
		user_count = max(user_count, user_id)
		item_count = max(item_count, item_id)

		total_interactions.append((user_id, item_id, 1))

	return user_count, item_count, total_interactions

def read_4sq(f):

	# read all
	total_interactions = []
	user_count, item_count = 0, 0

	for line in f.readlines():
		user_id, item_id, _ = line.split()
		user_id = int(user_id)
		item_id = int(item_id)

		user_count = max(user_count, user_id)
		item_count = max(item_count, item_id)

		total_interactions.append((user_id, item_id, 1))

	return user_count, item_count, total_interactions


def read_Amusic(f):

	# 전체 파일 읽기
	total_interactions = []
	user_count, item_count = 0, 0
	user_id_dict = {}
	item_id_dict = {}

	for line in f.readlines():
		user_raw_id, item_raw_id, _, _ = line.strip().split(',')

		if user_raw_id not in user_id_dict:
			user_id_dict[user_raw_id] = len(user_id_dict)

		user_id = user_id_dict[user_raw_id]

		if item_raw_id not in item_id_dict:
			item_id_dict[item_raw_id] = len(item_id_dict)

		item_id = item_id_dict[item_raw_id]

		user_id = int(user_id)
		item_id = int(item_id)

		user_count = max(user_count, user_id)
		item_count = max(item_count, item_id)

		total_interactions.append((user_id, item_id, 1))

	return user_count, item_count, total_interactions







#############
def score_mat_2_rank_mat(train_interactions, score_mat, is_L2=False):
	
	if is_L2:
		row, col = np.asarray(train_interactions)[:,0], np.asarray(train_interactions)[:,1]
		score_mat[row, col] = score_mat.max()
		rank_tmp = np.argsort(score_mat, axis=-1)
	else:
		row, col = np.asarray(train_interactions)[:,0], np.asarray(train_interactions)[:,1]
		score_mat[row, col] = score_mat.min()
		rank_tmp = np.argsort(-score_mat, axis=-1)
	
	rank_mat = np.zeros_like(rank_tmp)
	for i in range(rank_mat.shape[0]):
		row = rank_tmp[i]
		rank_mat[i][row] = torch.LongTensor(np.arange(len(row)))
		
	return rank_mat

def score_mat_2_rank_mat_torch(score_mat, train_interactions, gpu=None):
	
	row, col = torch.LongTensor(np.asarray(train_interactions)[:,0]), torch.LongTensor(np.asarray(train_interactions)[:,1])
	score_mat[row, col] = score_mat.min()
	rank_tmp = torch.argsort(-score_mat, dim=-1)
	
	rank_mat = torch.zeros_like(rank_tmp).to(gpu)
	for i in range(rank_mat.shape[0]):
		row = rank_tmp[i]
		rank_mat[i][row] = torch.LongTensor(np.arange(len(row))).to(gpu)
		
	return rank_mat

def print_eval_result(train_mat, valid_mat, test_mat, sorted_mat):
	metrics = {'P50':[], 'R50':[], 'N50':[], 'P10':[], 'R10':[], 'N10':[], 'P20':[], 'R20':[], 'N20':[]}
	eval_results = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}

	for test_user in test_mat:

		if test_user not in train_mat: continue
		sorted_list = list(sorted_mat[test_user])

		for mode in ['test']:

			sorted_list_tmp = []
			gt_mat = test_mat
			already_seen_items = set(train_mat[test_user].keys()) | set(valid_mat[test_user].keys())

			for item in sorted_list:
				if item not in already_seen_items:
					sorted_list_tmp.append(item)
				if len(sorted_list_tmp) == 50: break

			hit_10 = len(set(sorted_list_tmp[:10]) & set(gt_mat[test_user].keys()))
			hit_20 = len(set(sorted_list_tmp[:20]) & set(gt_mat[test_user].keys()))
			hit_50 = len(set(sorted_list_tmp[:50]) & set(gt_mat[test_user].keys()))
			
			eval_results[mode]['P10'].append(hit_10 / min(10, len(gt_mat[test_user].keys())))
			eval_results[mode]['R10'].append(hit_10 / len(gt_mat[test_user].keys()))

			eval_results[mode]['P20'].append(hit_20 / min(20, len(gt_mat[test_user].keys())))
			eval_results[mode]['R20'].append(hit_20 / len(gt_mat[test_user].keys()))

			eval_results[mode]['P50'].append(hit_50 / min(50, len(gt_mat[test_user].keys())))
			eval_results[mode]['R50'].append(hit_50 / len(gt_mat[test_user].keys()))    

			# ndcg
			denom = np.log2(np.arange(2, 10 + 2))
			dcg_10 = np.sum(np.in1d(sorted_list_tmp[:10], list(gt_mat[test_user].keys())) / denom)
			idcg_10 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 10)])

			denom = np.log2(np.arange(2, 20 + 2))
			dcg_20 = np.sum(np.in1d(sorted_list_tmp[:20], list(gt_mat[test_user].keys())) / denom)
			idcg_20 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 20)])

			denom = np.log2(np.arange(2, 50 + 2))
			dcg_50 = np.sum(np.in1d(sorted_list_tmp[:50], list(gt_mat[test_user].keys())) / denom)
			idcg_50 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 50)])

			eval_results[mode]['N10'].append(dcg_10 / idcg_10)
			eval_results[mode]['N20'].append(dcg_20 / idcg_20)
			eval_results[mode]['N50'].append(dcg_50 / idcg_50)

	# valid, test
	for mode in ['test']:
		for topk in [50, 10, 20]:
			eval_results['test']['P' + str(topk)] = round(np.asarray(eval_results[mode]['P' + str(topk)]).mean(), 4)
			eval_results['test']['R' + str(topk)] = round(np.asarray(eval_results[mode]['R' + str(topk)]).mean(), 4)  
			eval_results['test']['N' + str(topk)] = round(np.asarray(eval_results[mode]['N' + str(topk)]).mean(), 4)   
			eval_results['valid']['P' + str(topk)] = round(np.asarray(eval_results[mode]['P' + str(topk)]).mean(), 4)
			eval_results['valid']['R' + str(topk)] = round(np.asarray(eval_results[mode]['R' + str(topk)]).mean(), 4)  
			eval_results['valid']['N' + str(topk)] = round(np.asarray(eval_results[mode]['N' + str(topk)]).mean(), 4)   
	
	full_print_result(1000, 0., 0., eval_results, is_improved=True, train_time=0, test_time=0.)
	return eval_results['test']['R50']


def avg_con(target_rank_mat, std_mat, is_rank_aware=True):
	std_mat = np.exp(-std_mat/100)
	K = 100
	if not is_rank_aware:
		return std_mat.mean()
	
	tmp = np.where(target_rank_mat < K, std_mat, 0)
	return tmp.sum() / (tmp >0).sum()

def avg_con_p(target_rank_mat, std_mat, is_rank_aware=True):
	std_mat = np.exp(-std_mat/100)
	K = 100
	if not is_rank_aware:
		return std_mat.mean()
	
	tmp = np.where(target_rank_mat < K, std_mat, 0)
	return tmp.sum(1) / (tmp >0).sum(1)


def RCcon(rank_list, std_list, t1, t2):
	
	t_list = []
	s_list = []
	
	for i in range(len(rank_list)):
		t = np.where(rank_list[i] < 3000, rank_list[i], 10000)
		s = np.where(rank_list[i] < 3000, std_list[i], 10000)
		t_list.append(t)
		s_list.append(s)
	
	result = 0
	for i in range(len(t_list)):
		e_t = t_list[i]
		e_s = s_list[i]
		result += (np.exp(-e_t/t1) + np.exp(-e_s/t2))

	return result
