import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, user_count, item_count, dim, gpu, total_anneal_steps=200000, max_anneal=0.2):
        super(VAE, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.latent_size = dim
        self.gpu = gpu

        self.encoder = nn.Linear(self.item_count, self.latent_size)
        
        self.decoder = nn.Linear(self.latent_size, self.item_count)

        self.w_mean = nn.Linear(self.latent_size, self.latent_size)
        self.w_logstd = nn.Linear(self.latent_size, self.latent_size)

        self.anneal_step = 0
        self.total_anneal_steps = total_anneal_steps
        self.max_anneal = max_anneal
        self.sim_type = 'UAE'
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.normal_(m.bias.data)

        if gpu != None:
            self.user_list = torch.LongTensor([i for i in range(user_count)]).to(gpu)
            self.item_list = torch.LongTensor([i for i in range(item_count)]).to(gpu)
        else:
            self.user_list = torch.LongTensor([i for i in range(user_count)])
            self.item_list = torch.LongTensor([i for i in range(item_count)])
        

    def forward(self, inputs, return_score = False):
        user, rating_vec = inputs['user'], inputs['rating_vec']
        user_input = F.normalize(rating_vec, dim=-1)     
        user_hidden = torch.tanh(self.encoder(user_input))
        
        z_mean = self.w_mean(user_hidden)
        z_logstd = self.w_logstd(user_hidden)
        z_sampled = z_mean + torch.rand_like(z_logstd) * torch.exp(z_logstd)
    
        user_output = self.decoder(z_sampled)
        if return_score:
            return user_output
        
        user_output = F.softmax(user_output, dim=-1)
        return user_output, user_input, z_mean, z_logstd

    def compute_kl_weight(self):
        return min(self.max_anneal, 1.*self.anneal_step/self.total_anneal_steps)

    def get_loss(self, output):
        self.anneal_step += 1
        user_output, user_target, z_mean, z_logstd = output
        recon_loss = (- torch.log(user_output + 1e-12) * user_target).sum(dim=-1)
        kl_loss = 0.5 * (torch.pow(z_mean, 2) + torch.exp(2*z_logstd) - 2*z_logstd).sum(dim=-1)
        total_loss = recon_loss + self.compute_kl_weight() * kl_loss
        
        return total_loss.mean(dim=0)

    def forward_eval(self, inputs):
        user, rating_vec = inputs['user'], inputs['rating_vec']
        user_input = F.normalize(rating_vec, dim=-1)
        user_hidden = torch.tanh(self.encoder(user_input))

        z_mean = self.w_mean(user_hidden)
        z_logstd = self.w_logstd(user_hidden)
        z_sampled = z_mean + torch.rand_like(z_logstd) * torch.exp(z_logstd)
        
        user_output = self.decoder(z_sampled)
        user_output = F.softmax(user_output, dim=-1)

        return user_output


class IVAE(nn.Module):
    def __init__(self, user_count, item_count, dim, gpu, total_anneal_steps=200000, max_anneal=0.2):
        super(IVAE, self).__init__()
        self.user_count = item_count
        self.item_count = user_count
        self.latent_size = dim * 2
        self.gpu = gpu

        self.encoder = nn.Linear(self.item_count, self.latent_size)
        self.decoder = nn.Linear(self.latent_size, self.item_count)

        self.w_mean = nn.Linear(self.latent_size, self.latent_size)
        self.w_logstd = nn.Linear(self.latent_size, self.latent_size)

        self.anneal_step = 0
        self.total_anneal_steps = total_anneal_steps
        self.max_anneal = max_anneal
        self.sim_type = 'IAE'
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.normal_(m.bias.data)

        if gpu != None:
            self.user_list = torch.LongTensor([i for i in range(self.user_count)]).to(gpu)
            self.item_list = torch.LongTensor([i for i in range(self.item_count)]).to(gpu)
        else:
            self.user_list = torch.LongTensor([i for i in range(self.user_count)])
            self.item_list = torch.LongTensor([i for i in range(self.item_count)])
        

    def forward(self, inputs):
        user, rating_vec = inputs['user'], inputs['rating_vec']
        user_input = F.normalize(rating_vec, dim=-1)
        user_hidden = torch.tanh(self.encoder(user_input))

        z_mean = self.w_mean(user_hidden)
        z_logstd = self.w_logstd(user_hidden)
        z_sampled = z_mean + torch.rand_like(z_logstd) * torch.exp(z_logstd)
        
        user_output = self.decoder(z_sampled)
        user_output = F.softmax(user_output, dim=-1)

        return user_output, user_input, z_mean, z_logstd

    def compute_kl_weight(self):
        return min(self.max_anneal, 1.*self.anneal_step/self.total_anneal_steps)

    def get_loss(self, output):
        self.anneal_step += 1
        user_output, user_target, z_mean, z_logstd = output
        
        recon_loss = (- torch.log(user_output + 1e-12) * user_target).sum(dim=-1)
        kl_loss = 0.5 * (torch.pow(z_mean, 2) + torch.exp(2*z_logstd) - 2*z_logstd).sum(dim=-1)
        total_loss = recon_loss + self.compute_kl_weight() * kl_loss
        
        return total_loss.mean(dim=0)

    def forward_eval(self, inputs):
        user, rating_vec = inputs['user'], inputs['rating_vec']
        user_input = F.normalize(rating_vec, dim=-1)
     
        user_hidden = torch.tanh(self.encoder(user_input))

        z_mean = self.w_mean(user_hidden)
        z_logstd = self.w_logstd(user_hidden)
        z_sampled = z_mean + torch.rand_like(z_logstd) * torch.exp(z_logstd)
        
        user_output = self.decoder(z_sampled)
        user_output = F.softmax(user_output, dim=-1)

        return user_output


    def get_batch_full_mat(self, batch_user_rating_vec):
        # user : bs x 1	
        user_hidden = torch.tanh(self.encoder(batch_user_rating_vec))

        z_mean = self.w_mean(user_hidden)
        z_logstd = self.w_logstd(user_hidden)
        z_sampled = z_mean + torch.rand_like(z_logstd) * torch.exp(z_logstd)
        
        user_output = self.decoder(z_sampled)

        return user_output