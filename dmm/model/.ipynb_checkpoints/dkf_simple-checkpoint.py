#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

The code in this file is based on:
- “Deep Kalman Filter” arXiv, 2015, Rahul G.Krishnan et al.
- "Structured Inference Networks for Nonlinear State Space Models" AAAI, 2017, Rahul G.Krishnan et al.

DKF refers to the deep Markov model in the second paper, which has two possibilities:
- with only backwrad RNN in inference, it's a Deep Kalman Smoother (DKS),
- with bi-directional RNN in inference, it's a ST-LR

To have consistant expression comparing with other models we change some functions' name:
Emissino Function -> Generation
Gated Transition Fucntion -> Prior
"""


from torch import nn
import torch
from collections import OrderedDict


def build_DKF_simple(cfg, device='cpu'):

    ### Load parameters
    # General
    c_dim = cfg.getint('Network', 'c_dim')
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    tau = cfg.getint('DataFrame', 'tau')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Inference
#     dense_x_gx = [] if cfg.get('Network', 'dense_x_gx') == '' else [int(i) for i in cfg.get('Network', 'dense_x_gx').split(',')]
    dim_RNN_gx = cfg.getint('Network', 'dim_RNN_gx')
    num_RNN_gx = cfg.getint('Network', 'num_RNN_gx')
#     dense_ztm1_g = [] if cfg.get('Network', 'dense_ztm1_g') == '' else [int(i) for i in cfg.get('Network', 'dense_ztm1_g').split(',')]
    #dense_c_g = [] if cfg.get('Network', 'dense_c_g') == '' else [int(i) for i in cfg.get('Network', 'dense_c_g').split(',')]
    # Generation z
    dense_h = [] if cfg.get('Network', 'dense_h') == '' else [int(i) for i in cfg.get('Network', 'dense_h').split(',')]
    # Generation x

    # Beta-vae
    beta = cfg.getfloat('Training', 'beta')
#     gamma = cfg.getfloat('Training', 'gamma')

    # Build model
    model = DKF_simple(c_dim=c_dim, x_dim=x_dim, z_dim=z_dim, tau=tau,
                dim_RNN_gx=dim_RNN_gx, num_RNN_gx=num_RNN_gx, dense_h=dense_h, 
                dropout_p=dropout_p, beta=beta, device=device).to(device)

    return model


class DKF_simple(nn.Module):

    def __init__(self, c_dim, x_dim, z_dim=2, h_gen=16, dense_h=[], tau=4,
                 dim_RNN_gx=128, num_RNN_gx=1, dropout_p = 0, beta=1, device='cpu'):

        super().__init__()
        ### General parameters  
        #self.c_d = c_d
        self.c_dim = c_dim
        self.x_dim = x_dim
        self.y_dim = x_dim
        self.z_dim = z_dim
        self.tau = tau
        self.dropout_p = dropout_p
        self.device = device
        ### Inference
#         self.dense_x_gx = dense_x_gx[-1]
#         self.dim_RNN_gx = dim_RNN_gx
        self.num_RNN_gx = num_RNN_gx
#         self.dense_ztm1_g = dense_ztm1_g[-1]
        ### Generation z
        self.dense_h = dense_h[-1]
        # generation x
        ### Beta-loss
        self.beta = beta
        
        # AR(1) Parameters for Prior p(u_t | u_{t-1})
        # We learn the autocorrelation (phi) and process noise (sigma)
        self.u_phi = nn.Parameter(torch.tensor(0.9)) 
        self.u_logvar_prior = nn.Parameter(torch.tensor(-1.0)) # Small noise
        
        # Decoder loss
        #self.loss = nn.GaussianNLLLoss(reduction = 'sum') 
        self.loss = nn.GaussianNLLLoss(reduction = 'none') 

        self.build()
    

    def build(self):
        
        ###################
        #### Inference ####
        ###################
        # 1. x and c to rnn
        #self.input_embed = nn.Linear((self.x_dim + self.c_dim), self.dense_x_gx)
        self.rnn_gx = nn.LSTM((self.x_dim + self.c_dim), self.dense_h, self.num_RNN_gx)
        self.h_layernorm = nn.LayerNorm(self.dense_h)
        
        # 2. h_t and z_tm1 to ztm1_g
#         self.posterior_mlp = nn.Linear(self.z_dim, self.dense_ztm1_g)
        self.posterior_mlp = nn.Sequential(nn.Linear(self.z_dim, self.dense_h),
                                           nn.ReLU(),
                                          )
        self.posterior_gru = nn.GRUCell(self.dense_h, self.dense_h)
        self.posterior_layernorm = nn.LayerNorm(self.dense_h)
        
        # 3. ztm1_g to mu and cov
#         self.hidden_layer = nn.Sequential(nn.Linear(self.dense_ztm1_g, self.dense_h),
#                                            nn.ReLU(),
#                                           )
        self.inf_mean = nn.Linear(self.dense_h, self.z_dim)
        self.inf_diag = nn.Linear(self.dense_h, self.z_dim)
        self.inf_off_diag = nn.Linear(self.dense_h, self.z_dim*(self.z_dim - 1)//2)

#         self.inf_mean = nn.Sequential(nn.Linear(self.dense_ztm1_g, self.dense_h),
#                                       nn.ReLU(),
#                                       nn.Dropout(p=self.dropout_p),
#                                       nn.Linear(self.dense_h, self.z_dim),
#                                       )
#         self.inf_diag = nn.Sequential(nn.Linear(self.dense_ztm1_g, self.dense_h),
#                                       nn.ReLU(),
#                                       nn.Dropout(p=self.dropout_p),
#                                       nn.Linear(self.dense_h, self.z_dim),
#                                       )
#         self.inf_off_diag = nn.Sequential(nn.Linear(self.dense_ztm1_g, self.dense_h),
#                                           nn.ReLU(),
#                                           nn.Dropout(p=self.dropout_p),
#                                           nn.Linear(self.dense_h, self.z_dim*(self.z_dim - 1)//2),
#                                           )


        ######################
        #### Generation z ####
        ######################
        # 1. FiLM layer and project to self.dense_h
        self.cond_proj = nn.Sequential(nn.Linear(self.c_dim, self.dense_h * 2),
                                       nn.ReLU(),
                                      )
        self.prior_mlp = nn.Sequential(nn.Linear(self.z_dim, self.dense_h),
                                       nn.ReLU(),
                                      )
        # 2. self.dense_h to mu, cov 
#         self.dropout_z = nn.Dropout(p=self.dropout_p)
        self.prior_mean = nn.Linear(self.dense_h, self.z_dim)
        self.prior_diag = nn.Linear(self.dense_h, self.z_dim)
        self.prior_off_diag = nn.Linear(self.dense_h, self.z_dim*(self.z_dim - 1)//2)
#         self.prior_mean = nn.Sequential(nn.Linear(self.dense_h, 64),
#                                         nn.ReLU(),
#                                         nn.Dropout(p=self.dropout_p),
#                                         nn.Linear(64, self.z_dim)
#                                        )
#         self.prior_diag = nn.Sequential(nn.Linear(self.dense_h, 64))
#                                         nn.ReLU(),
#                                         nn.Dropout(p=self.dropout_p),
#                                         nn.Linear(64, self.z_dim)
#                                        )
#         self.prior_off_diag = nn.Sequential(nn.Linear(self.dense_h, 64))
#                                             nn.ReLU(),
#                                             nn.Dropout(p=self.dropout_p),
#                                             nn.Linear(64, self.z_dim*(self.z_dim - 1)//2)
#                                             )
        
        
        ######################
        #### Generation x ####
        ######################
        # project to dense_h and compute mu and logvar
        self.mlp_z_x = nn.Sequential(nn.Linear(self.z_dim, self.dense_h), 
                                     nn.ReLU(), 
                                     nn.Dropout(p=self.dropout_p),
#                                      nn.Linear(self.dense_h, self.dense_h), # <-- NUOVO STRATO
#                                      nn.ReLU(), # nuovo
#                                      nn.LayerNorm(self.dense_h), # nuovo
                                    )

#         self.mlp_z_x = nn.Sequential(
#             nn.Linear(self.z_dim, self.dense_h), 
#             nn.ReLU(), 
#             nn.LayerNorm(self.dense_h), # LayerNorm aiuta a stabilizzare l'addestramento in reti più profonde
#             nn.Dropout(p=self.dropout_p),

#             nn.Linear(self.dense_h, self.dense_h), # <-- NUOVO STRATO
#             nn.ReLU(),
#             nn.LayerNorm(self.dense_h),
#             nn.Dropout(p=self.dropout_p)
#         )
        self.gen_mean = nn.Linear(self.dense_h, self.y_dim)
        self.gen_logvar = nn.Linear(self.dense_h, self.y_dim)
        
    def covariance(self, diag, off_diag):    
        
        dims = len(diag.shape)
        s = diag.shape
        
        # Create a covariance matrix 
        diag = torch.diag_embed(torch.exp(diag)) 
        rows, cols = torch.tril_indices(self.z_dim, self.z_dim, offset=-1)
        if dims == 2:
            output = torch.zeros((s[0], self.z_dim, self.z_dim), device=self.device)
            batch_idx = torch.arange(s[0], device=self.device)[:, None]
            output[batch_idx, rows, cols] = off_diag
        elif dims == 3:
            output = torch.zeros((s[0], s[1], self.z_dim, self.z_dim), device=self.device)
            batch1_idx = torch.arange(s[0], device=self.device)[:, None, None]
            batch2_idx = torch.arange(s[1], device=self.device)[None, :, None]
            output[batch1_idx, batch2_idx, rows, cols] = off_diag
        L = diag + output
        cov = torch.matmul(L, L.transpose(-2, -1)) #shape kx2x2
        #cov = torch.inverse(precision)
        
        return cov, L
   

    def reparameterization_cov(self, mean, L):
        
        eps = torch.randn_like(mean)
        std = torch.matmul(L, eps.unsqueeze(-1))
        z = mean + std.squeeze(-1)
        
        return z
    
    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return torch.addcmul(mean, eps, std)
    

    def inference(self, x, c):
        
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # Create variable holder and send to GPU if needed
        z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_cov = torch.zeros((seq_len, batch_size, self.z_dim, self.z_dim)).to(self.device)
        z_L = torch.zeros((seq_len, batch_size, self.z_dim, self.z_dim)).to(self.device)
        z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_t = torch.zeros((batch_size, self.z_dim)).to(self.device)

        # 1. x_t to g_t, g_t and z_tm1 to z_t
        x_comb = torch.cat([x, c], dim=-1)
        h_seq, _ = self.rnn_gx(x_comb)  
        # Apply dropout and LayerNorm to the LSTM's output sequence
        #h_seq = self.h_dropout(h_seq)
        h_seq = self.h_layernorm(h_seq)
        for t in range(seq_len):
            g_ztm1 = self.posterior_mlp(z_t)     # proietto z_tm1 in modo che abbia dimensione union_dim
            g_z = self.posterior_gru(h_seq[t], g_ztm1)
            h_z = self.posterior_layernorm(g_z)
            # Calcolo media e cov di z_t
            #h_z = self.hidden_layer(g_z)
            z_mean[t,:,:] = self.inf_mean(h_z)           # DA QUESTO g_z RICAVO MEDIA E LOGVAR DI q di phi
            if torch.onnx.is_in_onnx_export():
                z_cov[t,:,:,:] = torch.zeros(batch_size, self.z_dim, self.z_dim)  # oppure None
                z_L[t,:,:,:] = torch.zeros(batch_size, self.z_dim, self.z_dim)
            else:
                z_cov[t,:,:,:], z_L[t,:,:,:] = self.covariance(self.inf_diag(h_z), self.inf_off_diag(h_z))
            z_t = self.reparameterization_cov(z_mean[t,:,:], z_L[t,:,:,:])      # ESTRAGGO z_t DA q di phi
            # z_t = z_mean[t,:,:]
            z[t,:,:] = z_t

        return z, z_mean, z_cov
    
    
    def generation_z(self, z_tm1, c):          # z_tm1 sarebbe z_{t-1}
        
        batch_size = z_tm1.shape[1]
        gamma, beta = self.cond_proj(c).chunk(2, dim=-1)
        h_tm1 = self.prior_mlp(z_tm1)
        h_t = gamma * h_tm1 + beta
#         h_t = self.dropout_z(h_t)
        z_mean_p = self.prior_mean(h_t)
        if torch.onnx.is_in_onnx_export():
            z_cov_p = torch.zeros(batch_size, self.z_dim, self.z_dim)  # oppure None
            z_L_p = torch.zeros(batch_size, self.z_dim, self.z_dim)
        else:
            z_cov_p, z_L_p = self.covariance(self.prior_diag(h_t), self.prior_off_diag(h_t))
        return z_mean_p, z_cov_p

        
    def generation_x(self, z):
        
        # 1. z_t to y_t
        y = self.mlp_z_x(z) 
        gen_mean = self.gen_mean(y)
        gen_logvar = self.gen_logvar(y)
        
        return gen_mean, gen_logvar
    

    def forward(self, x, c):
        
        # need input:  (seq_len, batch_size, x_dim)
        _, batch_size, _ = x.shape
        self.z, self.z_mean, self.z_cov = self.inference(x, c)
        z_0 = torch.zeros(1, batch_size, self.z_dim).to(self.device)
        z_tm1 = torch.cat([z_0, self.z[:-1, :,:]], dim=0)
        self.z_mean_p, self.z_cov_p = self.generation_z(z_tm1, c)  
        gen_mean, gen_logvar = self.generation_x(self.z)

        return gen_mean, gen_logvar

    def loss_rec(self, gen_mean, gen_logvar, x):
        
        # Equivalente ad assumere p(x|z) Gaussiana 
        gen_var = gen_logvar.exp()
        recon_loss = self.loss(x, gen_mean, gen_var)
        return recon_loss 
    
    def get_info(self):
        
        info = []
        info.append("----- Inference -----")
        info.append('>>>> x_t to g_t^x')
#         for layer in self.mlp_x_gx:
#             info.append(layer)
#         info.append(self.rnn_gx)
#         info.append('>>>> z_tm1 to g_x')
#         info.append(self.mlp_ztm1_g)
#         info.append('>>>> g_x to z_t')
#         for layer in self.mlp_g_z:
#             info.append(layer)

#         info.append("----- Bottleneck -----")
#         info.append(self.inf_mean)
#         info.append(self.inf_diag)
#         info.append(self.inf_off_diag)

        info.append("----- Generation x -----")
#         for layer in self.mlp_z_x:
#             info.append(layer)
        info.append(self.gen_mean)
        info.append(self.gen_logvar)
        
        info.append("----- Generation z -----")
#         info.append('>>>> Gating unit')
#         for layer in self.mlp_gate:
#             info.append(layer)
#         info.append('>>>> Proposed mean')
#         for layer in self.mlp_z_prop:
#             info.append(layer)
#         info.append('>>>> Prior mean and logvar')
#         info.append(self.prior_mean)
#         info.append(self.prior_diag)
#         info.append(self.prior_off_diag)

        return info


if __name__ == '__main__':
    x_dim = 96
    z_dim = 2
    device = 'cpu'
    dkf = DKF(x_dim=x_dim, z_dim=z_dim).to(device)

    x = torch.ones((2,96,3))
    y, z_mean, z_logvar, z_mean_p, z_logvar_p, z = dkf.forward(x)

    def loss_function(recon_x, x, mu, logvar, mu_prior, logvar_prior):
        recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 ) 
        KLD = -0.5 * torch.sum(logvar - logvar_prior - torch.div((logvar.exp() + (mu - mu_prior).pow(2)), logvar_prior.exp()))
        return recon + KLD

    loss = loss_function(y,x,z_mean,z_logvar,z_mean_p,z_logvar_p)/6

    print(loss)
