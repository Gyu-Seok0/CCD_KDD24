from Utils.data_utils import *
from Utils.data_loaders import *

import numpy as np
import torch
import copy
import time

def Euclidian_dist(user_mat, item_mat):
	A = (user_mat ** 2).sum(1, keepdim=True)
	B = (item_mat ** 2).sum(1, keepdim=True)
	
	AB = -2 * torch.matmul(user_mat, item_mat.T)
	
	return torch.sqrt(A + AB + B.T)  
	
def to_np(x):
	return x.data.cpu().numpy()
	
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


def full_evaluate(model, gpu, train_loader, test_dataset, return_score_mat=False, return_sorted_mat=False):
	
	metrics = {'P50':[], 'R50':[], 'N50':[], 'P10':[], 'R10':[], 'N10':[], 'P20':[], 'R20':[], 'N20':[]}
	eval_results = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}
	
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat
	num_topk_items = 1000

	if model.sim_type == 'inner product':
		user_emb, item_emb = model.get_embedding()
		score_mat = torch.matmul(user_emb, item_emb.T)
		#sorted_mat = torch.argsort(score_mat, dim=1, descending=True)
		sorted_mat = torch.topk(score_mat, num_topk_items).indices
		score_mat = - score_mat

	elif model.sim_type == 'weighted inner product':
		score_mat = model.forward_full_items(model.user_list)
		#sorted_mat = torch.argsort(score_mat, dim=1, descending=True)
		sorted_mat = torch.topk(score_mat, num_topk_items).indices
		score_mat = - score_mat
		
	elif model.sim_type == 'L2 dist':
		user_emb, item_emb = model.get_embedding()
		score_mat = Euclidian_dist(user_emb, item_emb)#.to('cpu')

		t = 4
		sorted_mats = []
		size = user_emb.size(0)//t
		
		for i in range(t):

			start_idx = size * i
			end_idx = size * (i+1)

			if i == t-1:
				end_idx = user_emb.size(0)

			sorted_mats.append(torch.argsort(score_mat[start_idx: end_idx, :].to(gpu), dim=1, descending=False).to('cpu'))

		sorted_mat = torch.cat(sorted_mats, 0)
	
	elif model.sim_type == 'network':

		score_mat = []
		while True:
			batch_users, is_last_batch = test_dataset.get_next_batch_users()

			total_items = torch.cat(batch_users.size(0) * [model.item_list.unsqueeze(0)], 0)
			score_mat_tmp = model.forward_multiple_items(batch_users.to(gpu), total_items).squeeze(-1)
			score_mat.append(score_mat_tmp)

			if is_last_batch:
				break

		score_mat = torch.cat(score_mat, 0)
		#sorted_mat = torch.argsort(score_mat, dim=1, descending=True)
		sorted_mat = torch.topk(score_mat, num_topk_items).indices


	elif model.sim_type == 'UAE':
		score_mat = torch.zeros(model.user_count, model.item_count)
		for mini_batch in train_loader:
			mini_batch = {key: value.to(gpu) for key, value in mini_batch.items()}

			output = model.forward_eval(mini_batch)
			score_mat[mini_batch['user'], :] = output.cpu()

		#sorted_mat = torch.argsort(score_mat, dim=1, descending=True)
		sorted_mat = torch.topk(score_mat, num_topk_items).indices


	elif model.sim_type == 'IAE':
		score_mat = torch.zeros(model.user_count, model.item_count)
		for mini_batch in train_loader:
			mini_batch = {key: value.to(gpu) for key, value in mini_batch.items()}

			output = model.forward_eval(mini_batch)
			score_mat[mini_batch['user'], :] = output.cpu()

		score_mat = score_mat.T
		#sorted_mat = torch.argsort(score_mat, dim=1, descending=True)
		sorted_mat = torch.topk(score_mat, num_topk_items).indices


	sorted_mat = to_np(sorted_mat)

	# 각 유저에 대해서,
	for test_user in test_mat:
		
		if test_user not in train_mat: continue
		sorted_list = list(sorted_mat[test_user])
		
		for mode in ['valid', 'test']:
			
			sorted_list_tmp = []
			if mode == 'valid':
				gt_mat = valid_mat
				already_seen_items = set(train_mat[test_user].keys()) | set(test_mat[test_user].keys())
			elif mode == 'test':
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
	for mode in ['test', 'valid']:
		for topk in [50, 10, 20]:
			eval_results[mode]['P' + str(topk)] = round(np.asarray(eval_results[mode]['P' + str(topk)]).mean(), 4)
			eval_results[mode]['R' + str(topk)] = round(np.asarray(eval_results[mode]['R' + str(topk)]).mean(), 4)  
			eval_results[mode]['N' + str(topk)] = round(np.asarray(eval_results[mode]['N' + str(topk)]).mean(), 4)   

	if return_score_mat:
		return eval_results, score_mat

	if return_sorted_mat:
		return eval_results, sorted_mat	
	return eval_results



def full_evaluate_g(model, gpu, train_loader, test_dataset, KD_user, return_score_mat=False, return_sorted_mat=False):
	
	metrics = {'P50':[], 'R50':[], 'N50':[], 'P10':[], 'R10':[], 'N10':[], 'P20':[], 'R20':[], 'N20':[]}
	eval_results = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}
	eval_results_KD = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}
	eval_results_noKD = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}
	
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	if model.sim_type == 'inner product':
		user_emb, item_emb = model.get_embedding()
		score_mat = torch.matmul(user_emb, item_emb.T)
		sorted_mat = torch.argsort(score_mat, dim=1, descending=True)
		score_mat = - score_mat

	elif model.sim_type == 'weighted inner product':
		score_mat = model.forward_full_items(model.user_list)
		sorted_mat = torch.argsort(score_mat, dim=1, descending=True)
		score_mat = - score_mat
		
	elif model.sim_type == 'L2 dist':
		user_emb, item_emb = model.get_embedding()
		score_mat = Euclidian_dist(user_emb, item_emb)
		sorted_mat = torch.argsort(score_mat, dim=1, descending=False)
	
	elif model.sim_type == 'network':

		score_mat = []
		while True:
			batch_users, is_last_batch = test_dataset.get_next_batch_users()

			total_items = torch.cat(batch_users.size(0) * [model.item_list.unsqueeze(0)], 0)
			score_mat_tmp = model.forward_multiple_items(batch_users.to(gpu), total_items).squeeze(-1)
			score_mat.append(score_mat_tmp)

			if is_last_batch:
				break

		score_mat = torch.cat(score_mat, 0)
		sorted_mat = torch.argsort(score_mat, dim=1, descending=True)

	elif model.sim_type == 'UAE':
		score_mat = torch.zeros(model.user_count, model.item_count)
		for mini_batch in train_loader:
			mini_batch = {key: value.to(gpu) for key, value in mini_batch.items()}

			output = model.forward_eval(mini_batch)
			score_mat[mini_batch['user'], :] = output.cpu()

		sorted_mat = torch.argsort(score_mat, dim=1, descending=True)	   

	elif model.sim_type == 'IAE':
		score_mat = torch.zeros(model.user_count, model.item_count)
		for mini_batch in train_loader:
			mini_batch = {key: value.to(gpu) for key, value in mini_batch.items()}

			output = model.forward_eval(mini_batch)
			score_mat[mini_batch['user'], :] = output.cpu()

		score_mat = score_mat.T
		sorted_mat = torch.argsort(score_mat, dim=1, descending=True)	   

	sorted_mat = to_np(sorted_mat)

	# 각 유저에 대해서,
	for test_user in test_mat:
		
		if test_user not in train_mat: continue
		sorted_list = list(sorted_mat[test_user])
		
		for mode in ['valid', 'test']:
			
			sorted_list_tmp = []
			if mode == 'valid':
				gt_mat = valid_mat
				already_seen_items = set(train_mat[test_user].keys()) | set(test_mat[test_user].keys())
			elif mode == 'test':
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

			if test_user in KD_user:
				for metric in ['P10', 'P20', 'P50', 'R10', 'R20', 'R50', 'N10', 'N20', 'N50']:
					eval_results_KD[mode][metric].append(eval_results[mode][metric][-1])
			else:
				for metric in ['P10', 'P20', 'P50', 'R10', 'R20', 'R50', 'N10', 'N20', 'N50']:
					eval_results_noKD[mode][metric].append(eval_results[mode][metric][-1])				
			
	
	# valid, test
	for mode in ['test', 'valid']:
		for topk in [50, 10, 20]:
			eval_results[mode]['P' + str(topk)] = round(np.asarray(eval_results[mode]['P' + str(topk)]).mean(), 4)
			eval_results[mode]['R' + str(topk)] = round(np.asarray(eval_results[mode]['R' + str(topk)]).mean(), 4)  
			eval_results[mode]['N' + str(topk)] = round(np.asarray(eval_results[mode]['N' + str(topk)]).mean(), 4)   

			eval_results_KD[mode]['P' + str(topk)] = round(np.asarray(eval_results_KD[mode]['P' + str(topk)]).mean(), 4)
			eval_results_KD[mode]['R' + str(topk)] = round(np.asarray(eval_results_KD[mode]['R' + str(topk)]).mean(), 4)  
			eval_results_KD[mode]['N' + str(topk)] = round(np.asarray(eval_results_KD[mode]['N' + str(topk)]).mean(), 4) 

			eval_results_noKD[mode]['P' + str(topk)] = round(np.asarray(eval_results_noKD[mode]['P' + str(topk)]).mean(), 4)
			eval_results_noKD[mode]['R' + str(topk)] = round(np.asarray(eval_results_noKD[mode]['R' + str(topk)]).mean(), 4)  
			eval_results_noKD[mode]['N' + str(topk)] = round(np.asarray(eval_results_noKD[mode]['N' + str(topk)]).mean(), 4) 

	if return_score_mat:
		return eval_results, eval_results_KD, eval_results_noKD, score_mat

	if return_sorted_mat:
		return eval_results, eval_results_KD, eval_results_noKD, sorted_mat	
	return eval_results, eval_results_KD, eval_results_noKD



def full_evaluate_g_with_rank_mat(rank_mat, gpu, train_loader, test_dataset, KD_user, return_score_mat=False, return_sorted_mat=False):
	
	metrics = {'P50':[], 'R50':[], 'N50':[], 'P10':[], 'R10':[], 'N10':[], 'P20':[], 'R20':[], 'N20':[]}
	eval_results = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}
	eval_results_KD = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}
	eval_results_noKD = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}
	
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	sorted_mat = np.argsort(rank_mat, axis=1)
	#sorted_mat = to_np(sorted_mat)

	# 각 유저에 대해서,
	for test_user in test_mat:
		
		if test_user not in train_mat: continue
		sorted_list = list(sorted_mat[test_user])
		
		for mode in ['valid', 'test']:
			
			sorted_list_tmp = []
			if mode == 'valid':
				gt_mat = valid_mat
				already_seen_items = set(train_mat[test_user].keys()) | set(test_mat[test_user].keys())
			elif mode == 'test':
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

			if test_user in KD_user:
				for metric in ['P10', 'P20', 'P50', 'R10', 'R20', 'R50', 'N10', 'N20', 'N50']:
					eval_results_KD[mode][metric].append(eval_results[mode][metric][-1])
			else:
				for metric in ['P10', 'P20', 'P50', 'R10', 'R20', 'R50', 'N10', 'N20', 'N50']:
					eval_results_noKD[mode][metric].append(eval_results[mode][metric][-1])				
			
	
	# valid, test
	for mode in ['test', 'valid']:
		for topk in [50, 10, 20]:
			eval_results[mode]['P' + str(topk)] = round(np.asarray(eval_results[mode]['P' + str(topk)]).mean(), 4)
			eval_results[mode]['R' + str(topk)] = round(np.asarray(eval_results[mode]['R' + str(topk)]).mean(), 4)  
			eval_results[mode]['N' + str(topk)] = round(np.asarray(eval_results[mode]['N' + str(topk)]).mean(), 4)   

			eval_results_KD[mode]['P' + str(topk)] = round(np.asarray(eval_results_KD[mode]['P' + str(topk)]).mean(), 4)
			eval_results_KD[mode]['R' + str(topk)] = round(np.asarray(eval_results_KD[mode]['R' + str(topk)]).mean(), 4)  
			eval_results_KD[mode]['N' + str(topk)] = round(np.asarray(eval_results_KD[mode]['N' + str(topk)]).mean(), 4) 

			eval_results_noKD[mode]['P' + str(topk)] = round(np.asarray(eval_results_noKD[mode]['P' + str(topk)]).mean(), 4)
			eval_results_noKD[mode]['R' + str(topk)] = round(np.asarray(eval_results_noKD[mode]['R' + str(topk)]).mean(), 4)  
			eval_results_noKD[mode]['N' + str(topk)] = round(np.asarray(eval_results_noKD[mode]['N' + str(topk)]).mean(), 4) 

	if return_score_mat:
		return eval_results, eval_results_KD, eval_results_noKD, score_mat

	if return_sorted_mat:
		return eval_results, eval_results_KD, eval_results_noKD, sorted_mat	
	return eval_results, eval_results_KD, eval_results_noKD



def full_evaluate_analysis(model, epoch, gpu, train_loader, test_dataset, train_dict=None):
	
	metrics = {'P50':[], 'R50':[], 'N50':[], 'P10':[], 'R10':[], 'N10':[], 'P20':[], 'R20':[], 'N20':[]}
	eval_results = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}
	
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	if model.sim_type == 'inner product':
		user_emb, item_emb = model.get_embedding()
		score_mat = torch.matmul(user_emb, item_emb.T)

	elif model.sim_type == 'weighted inner product':
		score_mat = model.forward_full_items(self.user_list)
		sorted_mat = torch.argsort(score_mat, dim=1, descending=True)
		score_mat = - score_mat
		
	elif model.sim_type == 'L2 dist':
		user_emb, item_emb = model.get_embedding()
		score_mat = Euclidian_dist(user_emb, item_emb)
	
	elif model.sim_type == 'network':

		score_mat = []
		while True:
			batch_users, is_last_batch = test_dataset.get_next_batch_users()

			total_items = torch.cat(batch_users.size(0) * [model.item_list.unsqueeze(0)], 0)
			score_mat_tmp = model.forward_multiple_items(batch_users.to(gpu), total_items).squeeze(-1)
			score_mat.append(score_mat_tmp)

			if is_last_batch:
				break

		score_mat = torch.cat(score_mat, 0)

	elif model.sim_type == 'UAE':
		score_mat = torch.zeros(model.user_count, model.item_count)
		for mini_batch in train_loader:
			mini_batch = {key: value.to(gpu) for key, value in mini_batch.items()}

			output = model.forward_eval(mini_batch)
			score_mat[mini_batch['user'], :] = output.cpu()

	elif model.sim_type == 'IAE':
		score_mat = torch.zeros(model.user_count, model.item_count)
		for mini_batch in train_loader:
			mini_batch = {key: value.to(gpu) for key, value in mini_batch.items()}

			output = model.forward_eval(mini_batch)
			score_mat[mini_batch['user'], :] = output.cpu()

		score_mat = score_mat.T 

	with open(train_dict['anal_path'] + "_" + str(epoch) + '.npy', 'wb') as f:
		np.save(f, to_np(score_mat))


def get_sorted_mat(model, gpu, train_loader, test_dataset, return_score_mat=False, return_sorted_mat=False):

	metrics = {'P50':[], 'R50':[], 'N50':[], 'P10':[], 'R10':[], 'N10':[], 'P20':[], 'R20':[], 'N20':[]}
	eval_results = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}

	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	if model.sim_type == 'inner product':
		user_emb, item_emb = model.get_embedding()
		score_mat = torch.matmul(user_emb, item_emb.T)
		sorted_mat = torch.argsort(score_mat, dim=1, descending=True)
		score_mat = - score_mat

	elif model.sim_type == 'weighted inner product':
		score_mat = model.forward_full_items(model.user_list)
		sorted_mat = torch.argsort(score_mat, dim=1, descending=True)
		score_mat = - score_mat

	elif model.sim_type == 'L2 dist':
		user_emb, item_emb = model.get_embedding()
		score_mat = Euclidian_dist(user_emb, item_emb)
		sorted_mat = torch.argsort(score_mat, dim=1, descending=False)

	elif model.sim_type == 'network':

		score_mat = []
		while True:
			batch_users, is_last_batch = test_dataset.get_next_batch_users()

			total_items = torch.cat(batch_users.size(0) * [model.item_list.unsqueeze(0)], 0)
			score_mat_tmp = model.forward_multiple_items(batch_users.to(gpu), total_items).squeeze(-1)
			score_mat.append(score_mat_tmp)

			if is_last_batch:
				break

		score_mat = torch.cat(score_mat, 0)
		sorted_mat = torch.argsort(score_mat, dim=1, descending=True)

	elif model.sim_type == 'UAE':
		score_mat = torch.zeros(model.user_count, model.item_count)
		for mini_batch in train_loader:
			mini_batch = {key: value.to(gpu) for key, value in mini_batch.items()}

			output = model.forward_eval(mini_batch)
			score_mat[mini_batch['user'], :] = output.cpu()

		sorted_mat = torch.argsort(score_mat, dim=1, descending=True)	   

	elif model.sim_type == 'IAE':
		score_mat = torch.zeros(model.user_count, model.item_count)
		for mini_batch in train_loader:
			mini_batch = {key: value.to(gpu) for key, value in mini_batch.items()}

			output = model.forward_eval(mini_batch)
			score_mat[mini_batch['user'], :] = output.cpu()

		score_mat = score_mat.T
		sorted_mat = torch.argsort(score_mat, dim=1, descending=True)	   

	if return_score_mat:
		return eval_results, score_mat

	if return_sorted_mat:
		return eval_results, sorted_mat
	
	return eval_results


def score_mat_2_rank_mat(score_mat, train_interactions, is_L2=False):
	
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
	

def get_eval_result(train_mat, valid_mat, test_mat, sorted_mat):
	metrics = {'P50':[], 'R50':[], 'N50':[], 'P10':[], 'R10':[], 'N10':[], 'P20':[], 'R20':[], 'N20':[]}
	eval_results = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}

	# 각 유저에 대해서,
	for test_user in test_mat:
		
		sorted_list = list(to_np(sorted_mat[test_user]))
		
		for mode in ['valid', 'test']:
			
			sorted_list_tmp = []
			if mode == 'valid':
				gt_mat = valid_mat
				already_seen_items = set(train_mat[test_user].keys()) | set(test_mat[test_user].keys())
			elif mode == 'test':
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
	for mode in ['test', 'valid']:
		for topk in [50, 10, 20]:
			eval_results[mode]['P' + str(topk)] = round(np.asarray(eval_results[mode]['P' + str(topk)]).mean(), 4)
			eval_results[mode]['R' + str(topk)] = round(np.asarray(eval_results[mode]['R' + str(topk)]).mean(), 4)  
			eval_results[mode]['N' + str(topk)] = round(np.asarray(eval_results[mode]['N' + str(topk)]).mean(), 4)   
	
	return eval_results


def get_eval_result_np(train_mat, valid_mat, test_mat, sorted_mat):
	metrics = {'P50':[], 'R50':[], 'N50':[], 'P10':[], 'R10':[], 'N10':[], 'P20':[], 'R20':[], 'N20':[]}
	eval_results = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}

	# 각 유저에 대해서,
	for test_user in test_mat:
		
		sorted_list = list(sorted_mat[test_user])
		
		for mode in ['valid', 'test']:
			
			sorted_list_tmp = []
			if mode == 'valid':
				gt_mat = valid_mat
				already_seen_items = set(train_mat[test_user].keys()) | set(test_mat[test_user].keys())
			elif mode == 'test':
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
	for mode in ['test', 'valid']:
		for topk in [50, 10, 20]:
			eval_results[mode]['P' + str(topk)] = round(np.asarray(eval_results[mode]['P' + str(topk)]).mean(), 4)
			eval_results[mode]['R' + str(topk)] = round(np.asarray(eval_results[mode]['R' + str(topk)]).mean(), 4)  
			eval_results[mode]['N' + str(topk)] = round(np.asarray(eval_results[mode]['N' + str(topk)]).mean(), 4)   
	
	return eval_results



def evaluate_M_Rec(model, gpu, train_loader, test_dataset):
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	# get score_mats
	user = model.user_emb(model.user_list)
	item = model.item_emb(model.item_list)

	H_u = model.forward_bottom_network(user)
	H_i = model.forward_bottom_network(item)

	bpr_score_mat = model.get_bpr_score_mat(H_u[0], H_i[0])
	cml_score_mat = model.get_cml_score_mat(H_u[1], H_i[1])
	bce_score_mat = model.get_bce_score_mat(H_u[2], H_i[2])
	uae_score_mat = model.get_uae_score_mat(H_u[3])
	iae_score_mat = model.get_iae_score_mat(H_i[4])

	bpr_sorted_mat = torch.argsort(bpr_score_mat, dim=1, descending=True)
	cml_sorted_mat = torch.argsort(cml_score_mat, dim=1, descending=True)
	bce_sorted_mat = torch.argsort(bce_score_mat, dim=1, descending=True)
	uae_sorted_mat = torch.argsort(uae_score_mat, dim=1, descending=True)
	iae_sorted_mat = torch.argsort(iae_score_mat, dim=1, descending=True)

	bpr_results = get_eval_result(train_mat, valid_mat, test_mat, bpr_sorted_mat)
	cml_results = get_eval_result(train_mat, valid_mat, test_mat, cml_sorted_mat)
	bce_results = get_eval_result(train_mat, valid_mat, test_mat, bce_sorted_mat)
	uae_results = get_eval_result(train_mat, valid_mat, test_mat, uae_sorted_mat)
	iae_results = get_eval_result(train_mat, valid_mat, test_mat, iae_sorted_mat)

	return bpr_results, cml_results, bce_results, uae_results, iae_results



def evaluate_M_Rec_seperate(model, gpu, train_loader, test_dataset):
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	# get score_mats
	H_u, H_i, _ = model.forward_bottom_network(model.user_list, model.item_list, model.item_list)

	bpr_score_mat = model.get_bpr_score_mat(H_u[0], H_i[0])
	cml_score_mat = model.get_cml_score_mat(H_u[1], H_i[1])
	bce_score_mat = model.get_bce_score_mat(H_u[2], H_i[2])
	uae_score_mat = model.get_uae_score_mat(H_u[3])
	iae_score_mat = model.get_iae_score_mat(H_i[4])

	bpr_sorted_mat = torch.argsort(bpr_score_mat, dim=1, descending=True)
	cml_sorted_mat = torch.argsort(cml_score_mat, dim=1, descending=True)
	bce_sorted_mat = torch.argsort(bce_score_mat, dim=1, descending=True)
	uae_sorted_mat = torch.argsort(uae_score_mat, dim=1, descending=True)
	iae_sorted_mat = torch.argsort(iae_score_mat, dim=1, descending=True)

	bpr_results = get_eval_result(train_mat, valid_mat, test_mat, bpr_sorted_mat)
	cml_results = get_eval_result(train_mat, valid_mat, test_mat, cml_sorted_mat)
	bce_results = get_eval_result(train_mat, valid_mat, test_mat, bce_sorted_mat)
	uae_results = get_eval_result(train_mat, valid_mat, test_mat, uae_sorted_mat)
	iae_results = get_eval_result(train_mat, valid_mat, test_mat, iae_sorted_mat)

	return bpr_results, cml_results, bce_results, uae_results, iae_results



def evaluate_M_Rec_LGCN(model, gpu, train_loader, test_dataset):
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	# get score_mats
	user, item = model.get_LGCN_emb()
	H_u = model.forward_bottom_network(user)
	H_i = model.forward_bottom_network(item)

	bpr_score_mat = model.get_bpr_score_mat(H_u[0], H_i[0])
	cml_score_mat = model.get_cml_score_mat(H_u[1], H_i[1])
	bce_score_mat = model.get_bce_score_mat(H_u[2], H_i[2])
	uae_score_mat = model.get_uae_score_mat(H_u[3])
	iae_score_mat = model.get_iae_score_mat(H_i[4])

	bpr_sorted_mat = torch.argsort(bpr_score_mat, dim=1, descending=True)
	cml_sorted_mat = torch.argsort(cml_score_mat, dim=1, descending=True)
	bce_sorted_mat = torch.argsort(bce_score_mat, dim=1, descending=True)
	uae_sorted_mat = torch.argsort(uae_score_mat, dim=1, descending=True)
	iae_sorted_mat = torch.argsort(iae_score_mat, dim=1, descending=True)

	bpr_results = get_eval_result(train_mat, valid_mat, test_mat, bpr_sorted_mat)
	cml_results = get_eval_result(train_mat, valid_mat, test_mat, cml_sorted_mat)
	bce_results = get_eval_result(train_mat, valid_mat, test_mat, bce_sorted_mat)
	uae_results = get_eval_result(train_mat, valid_mat, test_mat, uae_sorted_mat)
	iae_results = get_eval_result(train_mat, valid_mat, test_mat, iae_sorted_mat)

	return bpr_results, cml_results, bce_results, uae_results, iae_results


def evaluate_M_Rec_seperate_LGCN(model, gpu, train_loader, test_dataset):
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	# get score_mats
	H_u, H_i, _ = model.forward_bottom_network(user, item, item)

	bpr_score_mat = model.get_bpr_score_mat(H_u[0], H_i[0])
	cml_score_mat = model.get_cml_score_mat(H_u[1], H_i[1])
	bce_score_mat = model.get_bce_score_mat(H_u[2], H_i[2])
	sqr_score_mat = model.get_sqr_score_mat(H_u[3], H_i[3])
	uae_score_mat = model.get_uae_score_mat(H_u[4])
	iae_score_mat = model.get_iae_score_mat(H_i[5])

	bpr_sorted_mat = torch.argsort(bpr_score_mat, dim=1, descending=True)
	cml_sorted_mat = torch.argsort(cml_score_mat, dim=1, descending=True)
	bce_sorted_mat = torch.argsort(bce_score_mat, dim=1, descending=True)
	sqr_sorted_mat = torch.argsort(sqr_score_mat, dim=1, descending=True)
	uae_sorted_mat = torch.argsort(uae_score_mat, dim=1, descending=True)
	iae_sorted_mat = torch.argsort(iae_score_mat, dim=1, descending=True)

	bpr_results = get_eval_result(train_mat, valid_mat, test_mat, bpr_sorted_mat)
	cml_results = get_eval_result(train_mat, valid_mat, test_mat, cml_sorted_mat)
	bce_results = get_eval_result(train_mat, valid_mat, test_mat, bce_sorted_mat)
	sqr_results = get_eval_result(train_mat, valid_mat, test_mat, sqr_sorted_mat)
	uae_results = get_eval_result(train_mat, valid_mat, test_mat, uae_sorted_mat)
	iae_results = get_eval_result(train_mat, valid_mat, test_mat, iae_sorted_mat)

	return bpr_results, cml_results, bce_results, sqr_results, uae_results, iae_results



def interval(tic, toc):
	print(toc-tic)


def evaluate_E_Rec(model, gpu, train_loader, test_dataset):
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	# get score_mats
	user = model.user_emb(model.user_list)
	item = model.item_emb(model.item_list)

	H_u = model.forward_bottom_network(user)
	H_i = model.forward_bottom_network(item)

	bpr_score_mat = model.get_bpr_score_mat(H_u[0], H_i[0])
	bpr_sorted_mat = torch.argsort(bpr_score_mat, dim=1, descending=True)
	bpr_results = get_eval_result(train_mat, valid_mat, test_mat, bpr_sorted_mat)
	del bpr_sorted_mat
	bpr_rank_mat = to_np(score_mat_2_rank_mat_torch(bpr_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del bpr_score_mat


	cml_score_mat = model.get_cml_score_mat(H_u[1], H_i[1])
	cml_sorted_mat = torch.argsort(cml_score_mat, dim=1, descending=True)
	cml_results = get_eval_result(train_mat, valid_mat, test_mat, cml_sorted_mat)
	del cml_sorted_mat
	cml_rank_mat = to_np(score_mat_2_rank_mat_torch(cml_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del cml_score_mat

	bce_score_mat = model.get_bce_score_mat(H_u[2], H_i[2])
	bce_sorted_mat = torch.argsort(bce_score_mat, dim=1, descending=True)
	bce_results = get_eval_result(train_mat, valid_mat, test_mat, bce_sorted_mat)
	del bce_sorted_mat
	bce_rank_mat = to_np(score_mat_2_rank_mat_torch(bce_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del bce_score_mat

	sqr_score_mat = model.get_sqr_score_mat(H_u[3], H_i[3])
	sqr_sorted_mat = torch.argsort(sqr_score_mat, dim=1, descending=True)
	sqr_results = get_eval_result(train_mat, valid_mat, test_mat, sqr_sorted_mat)
	del sqr_sorted_mat
	sqr_rank_mat = to_np(score_mat_2_rank_mat_torch(sqr_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del sqr_score_mat

	uae_score_mat = model.get_uae_score_mat(H_u[4])
	uae_sorted_mat = torch.argsort(uae_score_mat, dim=1, descending=True)
	uae_results = get_eval_result(train_mat, valid_mat, test_mat, uae_sorted_mat)
	del uae_sorted_mat
	uae_rank_mat = to_np(score_mat_2_rank_mat_torch(uae_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del uae_score_mat

	# bpr_rank_mat = score_mat_2_rank_mat(to_np(bpr_score_mat), train_loader.dataset.interactions)
	# cml_rank_mat = score_mat_2_rank_mat(to_np(cml_score_mat), train_loader.dataset.interactions)
	# bce_rank_mat = score_mat_2_rank_mat(to_np(bce_score_mat), train_loader.dataset.interactions)
	# sqr_rank_mat = score_mat_2_rank_mat(to_np(sqr_score_mat), train_loader.dataset.interactions)
	# uae_rank_mat = score_mat_2_rank_mat(to_np(uae_score_mat), train_loader.dataset.interactions)
	
	if model.use_IAE:
		iae_score_mat = model.get_iae_score_mat(H_i[5])
		iae_sorted_mat = torch.argsort(iae_score_mat, dim=1, descending=True)
		iae_results = get_eval_result(train_mat, valid_mat, test_mat, iae_sorted_mat)
		del iae_sorted_mat
		# iae_rank_mat = score_mat_2_rank_mat(to_np(iae_score_mat), train_loader.dataset.interactions)
		iae_rank_mat = to_np(score_mat_2_rank_mat_torch(iae_score_mat, train_loader.dataset.interactions, gpu=gpu))

		return (bpr_results, cml_results, bce_results, sqr_results, uae_results, iae_results), (bpr_rank_mat, cml_rank_mat, bce_rank_mat, sqr_rank_mat, uae_rank_mat, iae_rank_mat)
	else:
		return (bpr_results, cml_results, bce_results, sqr_results, uae_results), (bpr_rank_mat, cml_rank_mat, bce_rank_mat, sqr_rank_mat, uae_rank_mat)


def evaluate_E_Rec_seperate(model, gpu, train_loader, test_dataset):
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	# get score_mats
	H_u, H_i, _ = model.forward_bottom_network(model.user_list, model.item_list, model.item_list)

	bpr_score_mat = model.get_bpr_score_mat(H_u[0], H_i[0])
	bpr_sorted_mat = torch.argsort(bpr_score_mat, dim=1, descending=True)
	bpr_results = get_eval_result(train_mat, valid_mat, test_mat, bpr_sorted_mat)
	del bpr_sorted_mat
	bpr_rank_mat = to_np(score_mat_2_rank_mat_torch(bpr_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del bpr_score_mat


	cml_score_mat = model.get_cml_score_mat(H_u[1], H_i[1])
	cml_sorted_mat = torch.argsort(cml_score_mat, dim=1, descending=True)
	cml_results = get_eval_result(train_mat, valid_mat, test_mat, cml_sorted_mat)
	del cml_sorted_mat
	cml_rank_mat = to_np(score_mat_2_rank_mat_torch(cml_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del cml_score_mat

	bce_score_mat = model.get_bce_score_mat(H_u[2], H_i[2])
	bce_sorted_mat = torch.argsort(bce_score_mat, dim=1, descending=True)
	bce_results = get_eval_result(train_mat, valid_mat, test_mat, bce_sorted_mat)
	del bce_sorted_mat
	bce_rank_mat = to_np(score_mat_2_rank_mat_torch(bce_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del bce_score_mat

	sqr_score_mat = model.get_sqr_score_mat(H_u[3], H_i[3])
	sqr_sorted_mat = torch.argsort(sqr_score_mat, dim=1, descending=True)
	sqr_results = get_eval_result(train_mat, valid_mat, test_mat, sqr_sorted_mat)
	del sqr_sorted_mat
	sqr_rank_mat = to_np(score_mat_2_rank_mat_torch(sqr_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del sqr_score_mat

	uae_score_mat = model.get_uae_score_mat(H_u[4])
	uae_sorted_mat = torch.argsort(uae_score_mat, dim=1, descending=True)
	uae_results = get_eval_result(train_mat, valid_mat, test_mat, uae_sorted_mat)
	del uae_sorted_mat
	uae_rank_mat = to_np(score_mat_2_rank_mat_torch(uae_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del uae_score_mat

	# bpr_rank_mat = score_mat_2_rank_mat(to_np(bpr_score_mat), train_loader.dataset.interactions)
	# cml_rank_mat = score_mat_2_rank_mat(to_np(cml_score_mat), train_loader.dataset.interactions)
	# bce_rank_mat = score_mat_2_rank_mat(to_np(bce_score_mat), train_loader.dataset.interactions)
	# sqr_rank_mat = score_mat_2_rank_mat(to_np(sqr_score_mat), train_loader.dataset.interactions)
	# uae_rank_mat = score_mat_2_rank_mat(to_np(uae_score_mat), train_loader.dataset.interactions)
	
	if model.use_IAE:
		iae_score_mat = model.get_iae_score_mat(H_i[5])
		iae_sorted_mat = torch.argsort(iae_score_mat, dim=1, descending=True)
		iae_results = get_eval_result(train_mat, valid_mat, test_mat, iae_sorted_mat)
		del iae_sorted_mat
		# iae_rank_mat = score_mat_2_rank_mat(to_np(iae_score_mat), train_loader.dataset.interactions)
		iae_rank_mat = to_np(score_mat_2_rank_mat_torch(iae_score_mat, train_loader.dataset.interactions, gpu=gpu))

		return (bpr_results, cml_results, bce_results, sqr_results, uae_results, iae_results), (bpr_rank_mat, cml_rank_mat, bce_rank_mat, sqr_rank_mat, uae_rank_mat, iae_rank_mat)
	else:
		return (bpr_results, cml_results, bce_results, sqr_results, uae_results), (bpr_rank_mat, cml_rank_mat, bce_rank_mat, sqr_rank_mat, uae_rank_mat)



def evaluate_E_Rec_seperate_homo(model, gpu, train_loader, test_dataset):
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	# get score_mats
	H_u, H_i, _ = model.forward_bottom_network(model.user_list, model.item_list, model.item_list)

	bpr_score_mat = model.get_bpr_score_mat(H_u[0])
	bpr_sorted_mat = torch.argsort(bpr_score_mat, dim=1, descending=True)
	bpr_results = get_eval_result(train_mat, valid_mat, test_mat, bpr_sorted_mat)
	del bpr_sorted_mat
	bpr_rank_mat = to_np(score_mat_2_rank_mat_torch(bpr_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del bpr_score_mat


	cml_score_mat = model.get_cml_score_mat(H_u[1])
	cml_sorted_mat = torch.argsort(cml_score_mat, dim=1, descending=True)
	cml_results = get_eval_result(train_mat, valid_mat, test_mat, cml_sorted_mat)
	del cml_sorted_mat
	cml_rank_mat = to_np(score_mat_2_rank_mat_torch(cml_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del cml_score_mat

	bce_score_mat = model.get_bce_score_mat(H_u[2])
	bce_sorted_mat = torch.argsort(bce_score_mat, dim=1, descending=True)
	bce_results = get_eval_result(train_mat, valid_mat, test_mat, bce_sorted_mat)
	del bce_sorted_mat
	bce_rank_mat = to_np(score_mat_2_rank_mat_torch(bce_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del bce_score_mat

	sqr_score_mat = model.get_sqr_score_mat(H_u[3])
	sqr_sorted_mat = torch.argsort(sqr_score_mat, dim=1, descending=True)
	sqr_results = get_eval_result(train_mat, valid_mat, test_mat, sqr_sorted_mat)
	del sqr_sorted_mat
	sqr_rank_mat = to_np(score_mat_2_rank_mat_torch(sqr_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del sqr_score_mat

	uae_score_mat = model.get_uae_score_mat(H_u[4])
	uae_sorted_mat = torch.argsort(uae_score_mat, dim=1, descending=True)
	uae_results = get_eval_result(train_mat, valid_mat, test_mat, uae_sorted_mat)
	del uae_sorted_mat
	uae_rank_mat = to_np(score_mat_2_rank_mat_torch(uae_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del uae_score_mat

	# bpr_rank_mat = score_mat_2_rank_mat(to_np(bpr_score_mat), train_loader.dataset.interactions)
	# cml_rank_mat = score_mat_2_rank_mat(to_np(cml_score_mat), train_loader.dataset.interactions)
	# bce_rank_mat = score_mat_2_rank_mat(to_np(bce_score_mat), train_loader.dataset.interactions)
	# sqr_rank_mat = score_mat_2_rank_mat(to_np(sqr_score_mat), train_loader.dataset.interactions)
	# uae_rank_mat = score_mat_2_rank_mat(to_np(uae_score_mat), train_loader.dataset.interactions)
	
	if model.use_IAE:
		iae_score_mat = model.get_iae_score_mat(H_u[5])
		iae_sorted_mat = torch.argsort(iae_score_mat, dim=1, descending=True)
		iae_results = get_eval_result(train_mat, valid_mat, test_mat, iae_sorted_mat)
		del iae_sorted_mat
		# iae_rank_mat = score_mat_2_rank_mat(to_np(iae_score_mat), train_loader.dataset.interactions)
		iae_rank_mat = to_np(score_mat_2_rank_mat_torch(iae_score_mat, train_loader.dataset.interactions, gpu=gpu))

		return (bpr_results, cml_results, bce_results, sqr_results, uae_results, iae_results), (bpr_rank_mat, cml_rank_mat, bce_rank_mat, sqr_rank_mat, uae_rank_mat, iae_rank_mat)
	else:
		return (bpr_results, cml_results, bce_results, sqr_results, uae_results), (bpr_rank_mat, cml_rank_mat, bce_rank_mat, sqr_rank_mat, uae_rank_mat)




def evaluate_PCL(model, gpu, train_loader, test_dataset):
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	# get score_mats
	user = model.user_emb(model.user_list)
	item = model.item_emb(model.item_list)

	H_u = model.forward_bottom_network(user)
	H_i = model.forward_bottom_network(item)

	bpr_score_mat = model.get_bpr_score_mat(H_u[0])
	bpr_sorted_mat = torch.argsort(bpr_score_mat, dim=1, descending=True)
	bpr_results = get_eval_result(train_mat, valid_mat, test_mat, bpr_sorted_mat)

	cml_score_mat = model.get_cml_score_mat(H_u[1])
	cml_sorted_mat = torch.argsort(cml_score_mat, dim=1, descending=True)
	cml_results = get_eval_result(train_mat, valid_mat, test_mat, cml_sorted_mat)

	bce_score_mat = model.get_bce_score_mat(H_u[2])
	bce_sorted_mat = torch.argsort(bce_score_mat, dim=1, descending=True)
	bce_results = get_eval_result(train_mat, valid_mat, test_mat, bce_sorted_mat)

	sqr_score_mat = model.get_sqr_score_mat(H_u[3])
	sqr_sorted_mat = torch.argsort(sqr_score_mat, dim=1, descending=True)
	sqr_results = get_eval_result(train_mat, valid_mat, test_mat, sqr_sorted_mat)

	uae_score_mat = model.get_uae_score_mat(H_u[4])
	uae_sorted_mat = torch.argsort(uae_score_mat, dim=1, descending=True)
	uae_results = get_eval_result(train_mat, valid_mat, test_mat, uae_sorted_mat)

	# bpr_rank_mat = score_mat_2_rank_mat(to_np(bpr_score_mat), train_loader.dataset.interactions)
	# cml_rank_mat = score_mat_2_rank_mat(to_np(cml_score_mat), train_loader.dataset.interactions)
	# bce_rank_mat = score_mat_2_rank_mat(to_np(bce_score_mat), train_loader.dataset.interactions)
	# sqr_rank_mat = score_mat_2_rank_mat(to_np(sqr_score_mat), train_loader.dataset.interactions)
	# uae_rank_mat = score_mat_2_rank_mat(to_np(uae_score_mat), train_loader.dataset.interactions)

	final_score_mat = model.get_final_score_mat(torch.cat([H_u[0], H_u[1], H_u[2], H_u[3], H_u[4]], dim=-1))
	final_sorted_mat = torch.argsort(final_score_mat, dim=1, descending=True)
	final_results = get_eval_result(train_mat, valid_mat, test_mat, final_sorted_mat)

	
	return (bpr_results, cml_results, bce_results, sqr_results, uae_results, final_results)



def evaluate_seperate_PCL(model, gpu, train_loader, test_dataset):
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	# get score_mats
	H_u, H_i, _ = model.forward_bottom_network(model.user_list, model.item_list, model.item_list)


	bpr_score_mat = model.get_bpr_score_mat(H_u[0])
	bpr_sorted_mat = torch.argsort(bpr_score_mat, dim=1, descending=True)
	bpr_results = get_eval_result(train_mat, valid_mat, test_mat, bpr_sorted_mat)

	cml_score_mat = model.get_cml_score_mat(H_u[1])
	cml_sorted_mat = torch.argsort(cml_score_mat, dim=1, descending=True)
	cml_results = get_eval_result(train_mat, valid_mat, test_mat, cml_sorted_mat)

	bce_score_mat = model.get_bce_score_mat(H_u[2])
	bce_sorted_mat = torch.argsort(bce_score_mat, dim=1, descending=True)
	bce_results = get_eval_result(train_mat, valid_mat, test_mat, bce_sorted_mat)

	sqr_score_mat = model.get_sqr_score_mat(H_u[3])
	sqr_sorted_mat = torch.argsort(sqr_score_mat, dim=1, descending=True)
	sqr_results = get_eval_result(train_mat, valid_mat, test_mat, sqr_sorted_mat)

	uae_score_mat = model.get_uae_score_mat(H_u[4])
	uae_sorted_mat = torch.argsort(uae_score_mat, dim=1, descending=True)
	uae_results = get_eval_result(train_mat, valid_mat, test_mat, uae_sorted_mat)

	# bpr_rank_mat = score_mat_2_rank_mat(to_np(bpr_score_mat), train_loader.dataset.interactions)
	# cml_rank_mat = score_mat_2_rank_mat(to_np(cml_score_mat), train_loader.dataset.interactions)
	# bce_rank_mat = score_mat_2_rank_mat(to_np(bce_score_mat), train_loader.dataset.interactions)
	# sqr_rank_mat = score_mat_2_rank_mat(to_np(sqr_score_mat), train_loader.dataset.interactions)
	# uae_rank_mat = score_mat_2_rank_mat(to_np(uae_score_mat), train_loader.dataset.interactions)

	final_score_mat = model.get_final_score_mat(torch.cat([H_u[0], H_u[1], H_u[2], H_u[3], H_u[4]], dim=-1))
	final_sorted_mat = torch.argsort(final_score_mat, dim=1, descending=True)
	final_results = get_eval_result(train_mat, valid_mat, test_mat, final_sorted_mat)

	
	return (bpr_results, cml_results, bce_results, sqr_results, uae_results, final_results)






def evaluate_ONE(model, gpu, train_loader, test_dataset):
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	# get score_mats
	user = model.user_emb(model.user_list)
	item = model.item_emb(model.item_list)

	H_u = model.forward_bottom_network(user)
	H_i = model.forward_bottom_network(item)

	bpr_score_mat = model.get_bpr_score_mat(H_u[0])
	bpr_sorted_mat = torch.argsort(bpr_score_mat, dim=1, descending=True)
	bpr_results = get_eval_result(train_mat, valid_mat, test_mat, bpr_sorted_mat)

	cml_score_mat = model.get_cml_score_mat(H_u[1])
	cml_sorted_mat = torch.argsort(cml_score_mat, dim=1, descending=True)
	cml_results = get_eval_result(train_mat, valid_mat, test_mat, cml_sorted_mat)

	bce_score_mat = model.get_bce_score_mat(H_u[2])
	bce_sorted_mat = torch.argsort(bce_score_mat, dim=1, descending=True)
	bce_results = get_eval_result(train_mat, valid_mat, test_mat, bce_sorted_mat)

	sqr_score_mat = model.get_sqr_score_mat(H_u[3])
	sqr_sorted_mat = torch.argsort(sqr_score_mat, dim=1, descending=True)
	sqr_results = get_eval_result(train_mat, valid_mat, test_mat, sqr_sorted_mat)

	uae_score_mat = model.get_uae_score_mat(H_u[4])
	uae_sorted_mat = torch.argsort(uae_score_mat, dim=1, descending=True)
	uae_results = get_eval_result(train_mat, valid_mat, test_mat, uae_sorted_mat)

	# bpr_rank_mat = score_mat_2_rank_mat(to_np(bpr_score_mat), train_loader.dataset.interactions)
	# cml_rank_mat = score_mat_2_rank_mat(to_np(cml_score_mat), train_loader.dataset.interactions)
	# bce_rank_mat = score_mat_2_rank_mat(to_np(bce_score_mat), train_loader.dataset.interactions)
	# sqr_rank_mat = score_mat_2_rank_mat(to_np(sqr_score_mat), train_loader.dataset.interactions)
	# uae_rank_mat = score_mat_2_rank_mat(to_np(uae_score_mat), train_loader.dataset.interactions)

	final_score_mat = model.get_final_score_mat([H_u[0], H_u[1], H_u[2], H_u[3], H_u[4]])
	final_sorted_mat = torch.argsort(final_score_mat, dim=1, descending=True)
	final_results = get_eval_result(train_mat, valid_mat, test_mat, final_sorted_mat)

	return (bpr_results, cml_results, bce_results, sqr_results, uae_results, final_results)



def evaluate_seperate_ONE(model, gpu, train_loader, test_dataset):
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	# get score_mats
	H_u = model.forward_bottom_network(model.user_list, model.item_list, model.item_list)

	bpr_score_mat = model.get_bpr_score_mat(H_u[0])
	bpr_sorted_mat = torch.argsort(bpr_score_mat, dim=1, descending=True)
	bpr_results = get_eval_result(train_mat, valid_mat, test_mat, bpr_sorted_mat)

	cml_score_mat = model.get_cml_score_mat(H_u[1])
	cml_sorted_mat = torch.argsort(cml_score_mat, dim=1, descending=True)
	cml_results = get_eval_result(train_mat, valid_mat, test_mat, cml_sorted_mat)

	bce_score_mat = model.get_bce_score_mat(H_u[2])
	bce_sorted_mat = torch.argsort(bce_score_mat, dim=1, descending=True)
	bce_results = get_eval_result(train_mat, valid_mat, test_mat, bce_sorted_mat)

	sqr_score_mat = model.get_sqr_score_mat(H_u[3])
	sqr_sorted_mat = torch.argsort(sqr_score_mat, dim=1, descending=True)
	sqr_results = get_eval_result(train_mat, valid_mat, test_mat, sqr_sorted_mat)

	uae_score_mat = model.get_uae_score_mat(H_u[4])
	uae_sorted_mat = torch.argsort(uae_score_mat, dim=1, descending=True)
	uae_results = get_eval_result(train_mat, valid_mat, test_mat, uae_sorted_mat)

	# bpr_rank_mat = score_mat_2_rank_mat(to_np(bpr_score_mat), train_loader.dataset.interactions)
	# cml_rank_mat = score_mat_2_rank_mat(to_np(cml_score_mat), train_loader.dataset.interactions)
	# bce_rank_mat = score_mat_2_rank_mat(to_np(bce_score_mat), train_loader.dataset.interactions)
	# sqr_rank_mat = score_mat_2_rank_mat(to_np(sqr_score_mat), train_loader.dataset.interactions)
	# uae_rank_mat = score_mat_2_rank_mat(to_np(uae_score_mat), train_loader.dataset.interactions)

	final_score_mat = model.get_final_score_mat([H_u[0], H_u[1], H_u[2], H_u[3], H_u[4]])
	final_sorted_mat = torch.argsort(final_score_mat, dim=1, descending=True)
	final_results = get_eval_result(train_mat, valid_mat, test_mat, final_sorted_mat)

	return (bpr_results, cml_results, bce_results, sqr_results, uae_results, final_results)




def evaluate_E_Rec_LGCN(model, gpu, train_loader, test_dataset):
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	# get score_mats
	user, item = model.get_LGCN_emb()
	H_u = model.forward_bottom_network(user)
	H_i = model.forward_bottom_network(item)

	bpr_score_mat = model.get_bpr_score_mat(H_u[0], H_i[0])
	bpr_sorted_mat = torch.argsort(bpr_score_mat, dim=1, descending=True)
	bpr_results = get_eval_result(train_mat, valid_mat, test_mat, bpr_sorted_mat)
	del bpr_sorted_mat
	bpr_rank_mat = to_np(score_mat_2_rank_mat_torch(bpr_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del bpr_score_mat


	cml_score_mat = model.get_cml_score_mat(H_u[1], H_i[1])
	cml_sorted_mat = torch.argsort(cml_score_mat, dim=1, descending=True)
	cml_results = get_eval_result(train_mat, valid_mat, test_mat, cml_sorted_mat)
	del cml_sorted_mat
	cml_rank_mat = to_np(score_mat_2_rank_mat_torch(cml_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del cml_score_mat

	bce_score_mat = model.get_bce_score_mat(H_u[2], H_i[2])
	bce_sorted_mat = torch.argsort(bce_score_mat, dim=1, descending=True)
	bce_results = get_eval_result(train_mat, valid_mat, test_mat, bce_sorted_mat)
	del bce_sorted_mat
	bce_rank_mat = to_np(score_mat_2_rank_mat_torch(bce_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del bce_score_mat

	sqr_score_mat = model.get_sqr_score_mat(H_u[3], H_i[3])
	sqr_sorted_mat = torch.argsort(sqr_score_mat, dim=1, descending=True)
	sqr_results = get_eval_result(train_mat, valid_mat, test_mat, sqr_sorted_mat)
	del sqr_sorted_mat
	sqr_rank_mat = to_np(score_mat_2_rank_mat_torch(sqr_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del sqr_score_mat

	uae_score_mat = model.get_uae_score_mat(H_u[4])
	uae_sorted_mat = torch.argsort(uae_score_mat, dim=1, descending=True)
	uae_results = get_eval_result(train_mat, valid_mat, test_mat, uae_sorted_mat)
	del uae_sorted_mat
	uae_rank_mat = to_np(score_mat_2_rank_mat_torch(uae_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del uae_score_mat

	# bpr_rank_mat = score_mat_2_rank_mat(to_np(bpr_score_mat), train_loader.dataset.interactions)
	# cml_rank_mat = score_mat_2_rank_mat(to_np(cml_score_mat), train_loader.dataset.interactions)
	# bce_rank_mat = score_mat_2_rank_mat(to_np(bce_score_mat), train_loader.dataset.interactions)
	# sqr_rank_mat = score_mat_2_rank_mat(to_np(sqr_score_mat), train_loader.dataset.interactions)
	# uae_rank_mat = score_mat_2_rank_mat(to_np(uae_score_mat), train_loader.dataset.interactions)
	
	if model.use_IAE:
		iae_score_mat = model.get_iae_score_mat(H_i[5])
		iae_sorted_mat = torch.argsort(iae_score_mat, dim=1, descending=True)
		iae_results = get_eval_result(train_mat, valid_mat, test_mat, iae_sorted_mat)
		del iae_sorted_mat
		# iae_rank_mat = score_mat_2_rank_mat(to_np(iae_score_mat), train_loader.dataset.interactions)
		iae_rank_mat = to_np(score_mat_2_rank_mat_torch(iae_score_mat, train_loader.dataset.interactions, gpu=gpu))

		return (bpr_results, cml_results, bce_results, sqr_results, uae_results, iae_results), (bpr_rank_mat, cml_rank_mat, bce_rank_mat, sqr_rank_mat, uae_rank_mat, iae_rank_mat)
	else:
		return (bpr_results, cml_results, bce_results, sqr_results, uae_results), (bpr_rank_mat, cml_rank_mat, bce_rank_mat, sqr_rank_mat, uae_rank_mat)




def evaluate_E_Rec_seperate_LGCN(model, gpu, train_loader, test_dataset):
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	# get score_mats
	H_u, H_i, _ = model.forward_bottom_network(model.user_list, model.item_list, model.item_list)

	bpr_score_mat = model.get_bpr_score_mat(H_u[0], H_i[0])
	bpr_sorted_mat = torch.argsort(bpr_score_mat, dim=1, descending=True)
	bpr_results = get_eval_result(train_mat, valid_mat, test_mat, bpr_sorted_mat)
	del bpr_sorted_mat
	bpr_rank_mat = to_np(score_mat_2_rank_mat_torch(bpr_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del bpr_score_mat


	cml_score_mat = model.get_cml_score_mat(H_u[1], H_i[1])
	cml_sorted_mat = torch.argsort(cml_score_mat, dim=1, descending=True)
	cml_results = get_eval_result(train_mat, valid_mat, test_mat, cml_sorted_mat)
	del cml_sorted_mat
	cml_rank_mat = to_np(score_mat_2_rank_mat_torch(cml_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del cml_score_mat

	bce_score_mat = model.get_bce_score_mat(H_u[2], H_i[2])
	bce_sorted_mat = torch.argsort(bce_score_mat, dim=1, descending=True)
	bce_results = get_eval_result(train_mat, valid_mat, test_mat, bce_sorted_mat)
	del bce_sorted_mat
	bce_rank_mat = to_np(score_mat_2_rank_mat_torch(bce_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del bce_score_mat

	sqr_score_mat = model.get_sqr_score_mat(H_u[3], H_i[3])
	sqr_sorted_mat = torch.argsort(sqr_score_mat, dim=1, descending=True)
	sqr_results = get_eval_result(train_mat, valid_mat, test_mat, sqr_sorted_mat)
	del sqr_sorted_mat
	sqr_rank_mat = to_np(score_mat_2_rank_mat_torch(sqr_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del sqr_score_mat

	uae_score_mat = model.get_uae_score_mat(H_u[4])
	uae_sorted_mat = torch.argsort(uae_score_mat, dim=1, descending=True)
	uae_results = get_eval_result(train_mat, valid_mat, test_mat, uae_sorted_mat)
	del uae_sorted_mat
	uae_rank_mat = to_np(score_mat_2_rank_mat_torch(uae_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del uae_score_mat

	# bpr_rank_mat = score_mat_2_rank_mat(to_np(bpr_score_mat), train_loader.dataset.interactions)
	# cml_rank_mat = score_mat_2_rank_mat(to_np(cml_score_mat), train_loader.dataset.interactions)
	# bce_rank_mat = score_mat_2_rank_mat(to_np(bce_score_mat), train_loader.dataset.interactions)
	# sqr_rank_mat = score_mat_2_rank_mat(to_np(sqr_score_mat), train_loader.dataset.interactions)
	# uae_rank_mat = score_mat_2_rank_mat(to_np(uae_score_mat), train_loader.dataset.interactions)
	
	if model.use_IAE:
		iae_score_mat = model.get_iae_score_mat(H_i[5])
		iae_sorted_mat = torch.argsort(iae_score_mat, dim=1, descending=True)
		iae_results = get_eval_result(train_mat, valid_mat, test_mat, iae_sorted_mat)
		del iae_sorted_mat
		# iae_rank_mat = score_mat_2_rank_mat(to_np(iae_score_mat), train_loader.dataset.interactions)
		iae_rank_mat = to_np(score_mat_2_rank_mat_torch(iae_score_mat, train_loader.dataset.interactions, gpu=gpu))

		return (bpr_results, cml_results, bce_results, sqr_results, uae_results, iae_results), (bpr_rank_mat, cml_rank_mat, bce_rank_mat, sqr_rank_mat, uae_rank_mat, iae_rank_mat)
	else:
		return (bpr_results, cml_results, bce_results, sqr_results, uae_results), (bpr_rank_mat, cml_rank_mat, bce_rank_mat, sqr_rank_mat, uae_rank_mat)





def evaluate_E_Rec_self(model, gpu, train_loader, test_dataset):
	train_mat = train_loader.dataset.rating_mat
	valid_mat = test_dataset.valid_mat
	test_mat = test_dataset.test_mat

	# get score_mats
	H_u, H_i = model.self_embedding()

	bpr_score_mat = model.get_bpr_score_mat(H_u[0], H_i[0])
	bpr_sorted_mat = torch.argsort(bpr_score_mat, dim=1, descending=True)
	bpr_results = get_eval_result(train_mat, valid_mat, test_mat, bpr_sorted_mat)
	del bpr_sorted_mat
	bpr_rank_mat = to_np(score_mat_2_rank_mat_torch(bpr_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del bpr_score_mat


	cml_score_mat = model.get_cml_score_mat(H_u[1], H_i[1])
	cml_sorted_mat = torch.argsort(cml_score_mat, dim=1, descending=True)
	cml_results = get_eval_result(train_mat, valid_mat, test_mat, cml_sorted_mat)
	del cml_sorted_mat
	cml_rank_mat = to_np(score_mat_2_rank_mat_torch(cml_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del cml_score_mat

	bce_score_mat = model.get_bce_score_mat(H_u[2], H_i[2])
	bce_sorted_mat = torch.argsort(bce_score_mat, dim=1, descending=True)
	bce_results = get_eval_result(train_mat, valid_mat, test_mat, bce_sorted_mat)
	del bce_sorted_mat
	bce_rank_mat = to_np(score_mat_2_rank_mat_torch(bce_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del bce_score_mat

	sqr_score_mat = model.get_sqr_score_mat(H_u[3], H_i[3])
	sqr_sorted_mat = torch.argsort(sqr_score_mat, dim=1, descending=True)
	sqr_results = get_eval_result(train_mat, valid_mat, test_mat, sqr_sorted_mat)
	del sqr_sorted_mat
	sqr_rank_mat = to_np(score_mat_2_rank_mat_torch(sqr_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del sqr_score_mat

	uae_score_mat = model.get_uae_score_mat(H_u[4])
	uae_sorted_mat = torch.argsort(uae_score_mat, dim=1, descending=True)
	uae_results = get_eval_result(train_mat, valid_mat, test_mat, uae_sorted_mat)
	del uae_sorted_mat
	uae_rank_mat = to_np(score_mat_2_rank_mat_torch(uae_score_mat, train_loader.dataset.interactions, gpu=gpu))
	del uae_score_mat

	# bpr_rank_mat = score_mat_2_rank_mat(to_np(bpr_score_mat), train_loader.dataset.interactions)
	# cml_rank_mat = score_mat_2_rank_mat(to_np(cml_score_mat), train_loader.dataset.interactions)
	# bce_rank_mat = score_mat_2_rank_mat(to_np(bce_score_mat), train_loader.dataset.interactions)
	# sqr_rank_mat = score_mat_2_rank_mat(to_np(sqr_score_mat), train_loader.dataset.interactions)
	# uae_rank_mat = score_mat_2_rank_mat(to_np(uae_score_mat), train_loader.dataset.interactions)
	
	if model.use_IAE:
		iae_score_mat = model.get_iae_score_mat(H_i[5])
		iae_sorted_mat = torch.argsort(iae_score_mat, dim=1, descending=True)
		iae_results = get_eval_result(train_mat, valid_mat, test_mat, iae_sorted_mat)
		del iae_sorted_mat
		# iae_rank_mat = score_mat_2_rank_mat(to_np(iae_score_mat), train_loader.dataset.interactions)
		iae_rank_mat = to_np(score_mat_2_rank_mat_torch(iae_score_mat, train_loader.dataset.interactions, gpu=gpu))

		return (bpr_results, cml_results, bce_results, sqr_results, uae_results, iae_results), (bpr_rank_mat, cml_rank_mat, bce_rank_mat, sqr_rank_mat, uae_rank_mat, iae_rank_mat)
	else:
		return (bpr_results, cml_results, bce_results, sqr_results, uae_results), (bpr_rank_mat, cml_rank_mat, bce_rank_mat, sqr_rank_mat, uae_rank_mat)





def print_result_M_Rec(epoch, max_epoch, train_loss, eval_results, is_improved=False, train_time=0., test_time=0.):

	if is_improved:
		print('Epoch [{}/{}], Train Loss: {:.4f}, Elapsed: Train: {:.2f} Test: {:.2f} *' .format(epoch, max_epoch, train_loss, train_time, test_time))
	else: 
		print('Epoch [{}/{}], Train Loss: {:.4f}, Elapsed: Train: {:.2f} Test: {:.2f}' .format(epoch, max_epoch, train_loss, train_time, test_time))


	if len(eval_results) == 6:
		for model_idx, model_type in enumerate(['bpr', 'cml', 'bce', 'sqr', 'uae', 'iae']):
			print(model_type)
			for topk in [10, 20, 50]:
				v_r = eval_results[model_idx]['valid']['R' + str(topk)] 
				v_n = eval_results[model_idx]['valid']['N' + str(topk)] 

				t_r = eval_results[model_idx]['test']['R' + str(topk)] 
				t_n = eval_results[model_idx]['test']['N' + str(topk)] 

				print('Valid R@{}: {:.4f} N@{}: {:.4f}'.format(topk, v_r, topk, v_n), 'Test R@{}: {:.4f} N@{}: {:.4f}'.format(topk, t_r, topk, t_n))
			print()
	else:
		for model_idx, model_type in enumerate(['bpr', 'cml', 'bce', 'sqr', 'uae']):
			print(model_type)
			for topk in [10, 20, 50]:
				v_r = eval_results[model_idx]['valid']['R' + str(topk)] 
				v_n = eval_results[model_idx]['valid']['N' + str(topk)] 

				t_r = eval_results[model_idx]['test']['R' + str(topk)] 
				t_n = eval_results[model_idx]['test']['N' + str(topk)] 

				print('Valid R@{}: {:.4f} N@{}: {:.4f}'.format(topk, v_r, topk, v_n), 'Test R@{}: {:.4f} N@{}: {:.4f}'.format(topk, t_r, topk, t_n))
			print()


# score_function 반환
def ensemble_function(Queue_R, Queue_S=None, dataset='citeULike'):

	if dataset == 'citeULike' or dataset =='4sq':
		tau1 = 10
		tau2 = 100
		K = 100
	elif dataset == 'ciao':
		tau1 = 50
		tau2 = 100
		K = 200

	t_list = []
	s_list = []
	for idx in range(len(Queue_R)):
		t = np.where(Queue_R[idx][-1] < K, Queue_R[idx][-1], 10000)
		t_list.append(t)

		if Queue_S != None:
			s = np.where(Queue_R[idx][-1] < K, Queue_S[idx], 10000)
			s_list.append(s)

	tt = 0.
	ss = 0.
	for idx in range(len(Queue_R)):
		tt += np.exp(-t_list[idx]/tau1)

		if Queue_S != None:
			ss += np.exp(-s_list[idx]/tau2)

	result = tt + ss

	return result

