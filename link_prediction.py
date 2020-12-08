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
        self.use_cuda = use_cuda

        self.ent_init_embeds = nn.Parameter(torch.Tensor(i_dim, h_dim))
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.tim_init_embeds = nn.Parameter(torch.Tensor(num_times, h_dim))

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

    def get_raw_m_t(self, quadrupleList):
        h_idx = quadrupleList[:, 0]
        r_idx = quadrupleList[:, 1]
        t_idx = quadrupleList[:, 2]
        T_idx = quadrupleList[:, 3] / args.time_stamp

        h = self.ent_init_embeds[h_idx]
        r = self.w_relation[r_idx]
        T = self.tim_init_embeds[T_idx]

        return h, r, T


    def forward(self, quadruple, copy_vocabulary):

        h, r, T = self.get_raw_m_t(quadruple)
        score_g = self.generate_mode(h, r, T)
        #score_g = F.softmax(torch.mm(h*r+T, self.ent_init_embeds.permute(1,0)), dim=1)
        
        score_c = self.copy_mode(h, r, copy_vocabulary)
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
        self.W_s = nn.Linear(hidden_dim * 2, output_dim)
        self.use_cuda = use_cuda

    def forward(self, ent_embed, rel_embed, copy_vocabulary):
        m_t = torch.cat((ent_embed, rel_embed), dim=1)
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

    def forward(self, ent_embed, rel_embed, tim_embed):

        m_t = torch.cat((ent_embed, rel_embed, tim_embed), dim=1)
        score_g = self.W_mlp(m_t)

        return F.softmax(score_g, dim=1)