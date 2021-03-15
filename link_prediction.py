import math
import torch
from torch.nn import Parameter, init
from torch import nn
import torch.nn.functional as F
import numpy as np
from config import args
import scipy.sparse as sp

class link_prediction(nn.Module):
    def __init__(self, i_dim, h_dim, num_rels, num_times, use_cuda=False):
        super(link_prediction, self).__init__()

        self.i_dim = i_dim
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.num_times = num_times
        self.use_cuda = use_cuda

        self.ent_init_embeds = nn.Parameter(torch.Tensor(i_dim, h_dim))
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.tim_init_embeds = nn.Parameter(torch.Tensor(1, h_dim))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.generate_mode = Generate_mode(h_dim, h_dim, self.i_dim)
        self.copy_mode = Copy_mode(self.h_dim, self.i_dim, use_cuda)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ent_init_embeds,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.tim_init_embeds,
                                gain=nn.init.calculate_gain('relu'))
    
    def get_init_time(self, quadrupleList):
        T_idx = quadrupleList[:, 3] / args.time_stamp
        init_tim = torch.Tensor(self.num_times, self.h_dim)
        for i in range(self.num_times):
            init_tim[i] = torch.Tensor(self.tim_init_embeds.cpu().detach().numpy().reshape(self.h_dim)) * (i + 1)
        init_tim = init_tim.to('cuda')
        T = init_tim[T_idx]
        return T
    
    def get_raw_m_t(self, quadrupleList):
        h_idx = quadrupleList[:, 0]
        r_idx = quadrupleList[:, 1]
        t_idx = quadrupleList[:, 2]

        h = self.ent_init_embeds[h_idx]
        r = self.w_relation[r_idx]

        return h, r
    
    def get_raw_m_t_sub(self, quadrupleList):
        h_idx = quadrupleList[:, 0]
        r_idx = quadrupleList[:, 1]
        t_idx = quadrupleList[:, 2]

        t = self.ent_init_embeds[t_idx]
        r = self.w_relation[r_idx]

        return t, r


    def forward(self, quadruple, copy_vocabulary, entity):

        if entity == 'object':
            h, r = self.get_raw_m_t(quadruple)
            T = self.get_init_time(quadruple)
            score_g = self.generate_mode(h, r, T, entity)
            score_c = self.copy_mode(h, r, T, copy_vocabulary, entity)

        if entity == 'subject':
            t, r = self.get_raw_m_t_sub(quadruple)
            T = self.get_init_time(quadruple)
            score_g = self.generate_mode(t, r, T, entity)
            score_c = self.copy_mode(t, r, T, copy_vocabulary, entity)

        a = args.alpha
        score = score_c * a + score_g * (1-a)
        score = torch.log(score)

        return score

    def regularization_loss(self, reg_param):
        regularization_loss = torch.mean(self.w_relation.pow(2)) + torch.mean(self.ent_init_embeds.pow(2)) + torch.mean(self.tim_init_embeds.pow(2))
        return regularization_loss * reg_param


class Copy_mode(nn.Module):
    def __init__(self, hidden_dim, output_dim, use_cuda):
        super(Copy_mode, self).__init__()
        self.hidden_dim = hidden_dim

        self.tanh = nn.Tanh()
        self.W_s = nn.Linear(hidden_dim * 3, output_dim)
        self.use_cuda = use_cuda

    def forward(self, ent_embed, rel_embed, time_embed, copy_vocabulary, entity):
        if entity == 'object':
            m_t = torch.cat((ent_embed, rel_embed, time_embed), dim=1)
        if entity == 'subject':
            m_t = torch.cat((rel_embed, ent_embed, time_embed), dim=1)

        q_s = self.tanh(self.W_s(m_t))
        if self.use_cuda:
            encoded_mask = torch.Tensor(np.array(copy_vocabulary.cpu() == 0, dtype=float) * (-100))
            encoded_mask = encoded_mask.to('cuda')
        else:
            encoded_mask = torch.Tensor(np.array(copy_vocabulary == 0, dtype=float) * (-100))

        score_c = q_s + encoded_mask

        return F.softmax(score_c, dim=1)
    
class Generate_mode(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(Generate_mode, self).__init__()
        # weights
        self.W_mlp = nn.Linear(hidden_size * 3, output_dim)

    def forward(self, ent_embed, rel_embed, tim_embed, entity):
        if entity == 'object':
            m_t = torch.cat((ent_embed, rel_embed, tim_embed), dim=1)
        if entity == 'subject':
            m_t = torch.cat((rel_embed, ent_embed, tim_embed), dim=1)

        score_g = self.W_mlp(m_t)

        return F.softmax(score_g, dim=1)
