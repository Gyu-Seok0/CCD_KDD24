import torch.nn.functional as F
import torch.nn as nn
import torch
from pdb import set_trace as bp

class LightGCN(nn.Module):
	def __init__(self, user_count, item_count, dim, gpu, A, A_T, keep_prob = 0.8):
		super(LightGCN, self).__init__()
		self.user_count = user_count
		self.item_count = item_count

		if gpu != None:
			self.user_list = torch.LongTensor([i for i in range(user_count)]).to(gpu)
			self.item_list = torch.LongTensor([i for i in range(item_count)]).to(gpu)
		else:
			self.user_list = torch.LongTensor([i for i in range(user_count)])
			self.item_list = torch.LongTensor([i for i in range(item_count)])

		self.user_emb = nn.Embedding(self.user_count, dim)
		self.item_emb = nn.Embedding(self.item_count, dim)

		self.sim_type = 'inner product'
		self.A = A.to(gpu)	# user x item
		self.A_T = A_T.to(gpu)

		self.A.requires_grad = False
		self.A_T.requires_grad = False
		self.keep_prob = keep_prob
		
		nn.init.normal_(self.user_emb.weight, mean=0., std= 0.01)
		nn.init.normal_(self.item_emb.weight, mean=0., std= 0.01)

	def _dropout_x(self, x):
		size = x.size()
		index = x.indices().t()
		values = x.values()
		random_index = torch.rand(len(values)) + self.keep_prob
		random_index = random_index.int().bool()
		index = index[random_index]
		values = values[random_index]/self.keep_prob
		g = torch.sparse.FloatTensor(index.t(), values, size)
		return g
		
	def forward(self, mini_batch):

		dropout_A = self._dropout_x(self.A.coalesce())
		dropout_A_t = dropout_A.t()
		dropout_A.requires_grad = False
		dropout_A_t.requires_grad = False

		u_0 = self.user_emb(self.user_list)	# num. user x dim
		i_0 = self.item_emb(self.item_list)

		i_1 = torch.spmm(dropout_A_t, u_0)		# 유저 평균 -> 아이템
		u_1 = torch.spmm(dropout_A, i_0)		# 아이템 평균 -> 유저
		
		i_2 = torch.spmm(dropout_A_t, u_1)
		u_2 = torch.spmm(dropout_A, i_1)		
		
		user = mini_batch['user']
		pos_item = mini_batch['pos_item']
		neg_item = mini_batch['neg_item']

		user_0 = torch.index_select(u_0, 0, user) 
		user_1 = torch.index_select(u_1, 0, user) 
		#user_2 = torch.index_select(u_2, 0, user) 

		pos_0 = torch.index_select(i_0, 0, pos_item) 
		pos_1 = torch.index_select(i_1, 0, pos_item) 
		#pos_2 = torch.index_select(i_2, 0, pos_item) 

		neg_0 = torch.index_select(i_0, 0, neg_item) 
		neg_1 = torch.index_select(i_1, 0, neg_item) 
		#neg_2 = torch.index_select(i_2, 0, neg_item) 

		# u = (user_0 + user_1 + user_2) / 3
		# i = (pos_0 + pos_1 + pos_2) / 3
		# j = (neg_0 + neg_1 + neg_2) / 3

		u = (user_0 + user_1) / 2
		i = (pos_0 + pos_1) / 2
		j = (neg_0 + neg_1) / 2

		pos_score = (u * i).sum(dim=1, keepdim=True)
		neg_score = (u * j).sum(dim=1, keepdim=True)

		return (pos_score, neg_score)

	def get_embedding(self):

		u_0 = self.user_emb(self.user_list)	# num. user x dim
		i_0 = self.item_emb(self.item_list)

		i_1 = torch.spmm(self.A_T, u_0)		# 유저 평균 -> 아이템
		u_1 = torch.spmm(self.A, i_0)		# 아이템 평균 -> 유저
		
		user = (u_0 + u_1) / 2
		item =  (i_0 + i_1) / 2

		# i_2 = torch.spmm(self.A_T, u_1)
		# u_2 = torch.spmm(self.A, i_1)	

		# user = (u_0 + u_1 + u_2) / 3
		# item = (i_0 + i_1 + i_2) / 3


		return user, item

	def get_loss(self, output):
		pos_score, neg_score = output[0], output[1]
		loss = -(pos_score - neg_score).sigmoid().log().sum()
		
		return loss


	def forward_multiple_items(self, user, item, span_required=True):
		# user : bs x 1
		# item : bs x k
		#print("rd", user.size(), item.size())
		if span_required:
			user = user.unsqueeze(-1)
			user = torch.cat(item.size(1) * [user], 1)
			
		u = self.user_emb(user)		# bs x k x dim
		i = self.item_emb(item)		# bs x k x dim
		
		score = (u * i).sum(dim=-1, keepdim=True)	
		return score


	def forward_full_items(self, batch_user):
		# user : bs x 1	
		user_emb, item_emb = self.get_embedding()
		user = user_emb[batch_user]	# bs x d
		item = item_emb # item x d

		return torch.matmul(user, item.T)