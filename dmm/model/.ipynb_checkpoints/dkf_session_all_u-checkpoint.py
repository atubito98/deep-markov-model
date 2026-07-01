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
import torch.nn.functional as F
from collections import OrderedDict


def build_DKF_session_all_u(cfg, device='cpu'):

    ### Load parameters
    # General
    c_dim = cfg.getint('Network', 'c_dim')
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    u_dim = cfg.getint('Network','u_dim')
    tau = cfg.getint('DataFrame', 'tau')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Inference
    dense_x_gx = [] if cfg.get('Network', 'dense_x_gx') == '' else [int(i) for i in cfg.get('Network', 'dense_x_gx').split(',')]
    dim_RNN_gx = cfg.getint('Network', 'dim_RNN_gx')
    num_RNN_gx = cfg.getint('Network', 'num_RNN_gx')
    dense_ztm1_g = [] if cfg.get('Network', 'dense_ztm1_g') == '' else [int(i) for i in cfg.get('Network', 'dense_ztm1_g').split(',')]
    #dense_c_g = [] if cfg.get('Network', 'dense_c_g') == '' else [int(i) for i in cfg.get('Network', 'dense_c_g').split(',')]
    # Generation z
    dense_h = [] if cfg.get('Network', 'dense_h') == '' else [int(i) for i in cfg.get('Network', 'dense_h').split(',')]
    # Generation x

    # Beta-vae
    beta = cfg.getfloat('Training', 'beta')

    # Build model
    model = DKF_session_all_u(c_dim=c_dim, x_dim=x_dim, z_dim=z_dim, tau=tau,
                dense_x_gx=dense_x_gx, dim_RNN_gx=dim_RNN_gx, u_dim=u_dim,
                num_RNN_gx=num_RNN_gx, dense_h=dense_h, dense_ztm1_g=dense_ztm1_g,
                dropout_p=dropout_p, beta=beta, device=device).to(device)

    return model


class DKF_session_all_u(nn.Module):

    def __init__(self, c_dim, x_dim, z_dim=2, u_dim=3, h_gen=16, dense_h=[], tau=4,
                 dense_x_gx=[], dim_RNN_gx=128, num_RNN_gx=1, dense_ztm1_g=[],
                 dropout_p = 0, beta=1, device='cpu'):

        super().__init__()
        ### General parameters  
        #self.c_d = c_d
        self.c_dim = c_dim
        self.x_dim = x_dim
        self.y_dim = x_dim
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.tau = tau
        self.dropout_p = dropout_p
        self.device = device
        ### Inference
        self.dense_x_gx = dense_x_gx[-1]
        self.dim_RNN_gx = dim_RNN_gx
        self.num_RNN_gx = num_RNN_gx
        self.dense_ztm1_g = dense_ztm1_g[-1]
        ### Generation z
        self.dense_h = dense_h[-1]
        # generation x
        ### Beta-loss
        self.beta = beta
        
        # AR(1) Parameters for Prior p(u_t | u_{t-1})
        # We learn the autocorrelation (phi) and process noise (sigma)
        self.u_phi = nn.Parameter(torch.tensor(0.9)) 
        self.logvar_prior_u = nn.Parameter(torch.tensor(-1.0)) # Small noise
        
        # Decoder loss
        #self.loss = nn.GaussianNLLLoss(reduction = 'sum') 
        self.loss = nn.GaussianNLLLoss(reduction = 'none') 

        self.build()
    

    def build(self):
        
        # --- 1. SESSION ADAPTERS (Gestione dinamica) ---
        # Usiamo ModuleDict per salvare i layer specifici di ogni sessione.
        # Linear(96 -> 64): Impara ad allineare e proiettare i canali nello spazio condiviso
        self.input_adapters = nn.ModuleDict()
        self.output_adapters_mean = nn.ModuleDict()
        self.output_adapters_logvar = nn.ModuleDict()
        
        ###################
        #### Inference ####
        ###################
        # 1. x and c to rnn
        # Input Dropout (Simulate electrode noise)
        self.input_dropout = nn.Dropout(p=self.dropout_p)
        self.encoder_rnn = nn.LSTM((self.dense_x_gx + self.c_dim), self.dense_h, self.num_RNN_gx)
        self.h_layernorm = nn.LayerNorm(self.dense_h)
        
        # 2. h_t and z_tm1 to ztm1_g
        self.posterior_mlp_u = nn.Sequential(nn.Linear(self.u_dim, self.dense_h),
                                           nn.ReLU(),
                                           nn.Dropout(self.dropout_p)
                                          )
        self.posterior_gru_u = nn.GRUCell(self.dense_h, self.dense_h)
        self.posterior_layernorm_u = nn.LayerNorm(self.dense_h)
        
        # 3. ztm1_g to mu and cov
        self.inf_mean_u = nn.Linear(self.dense_h, self.u_dim)
        self.inf_logvar_u = nn.Linear(self.dense_h, self.u_dim)
        
        self.posterior_mlp = nn.Sequential(nn.Linear(self.z_dim, self.dense_h),
                                           nn.ReLU(),
                                           nn.Dropout(self.dropout_p)
                                          )
        self.posterior_gru = nn.GRUCell(self.dense_h, self.dense_h)
        self.posterior_layernorm = nn.LayerNorm(self.dense_h)
        
        # 3. ztm1_g to mu and cov
        self.inf_mean = nn.Linear(self.dense_h, self.z_dim)
        self.inf_diag = nn.Linear(self.dense_h, self.z_dim)
        self.inf_off_diag = nn.Linear(self.dense_h, self.z_dim*(self.z_dim - 1)//2)
        

        ######################
        #### Generation z ####
        ######################
        # 1. FiLM layer and project to self.dense_h
        self.cond_proj = nn.Sequential(nn.Linear(self.c_dim + self.u_dim, self.dense_h * 2),
                                       nn.ReLU(),
                                      )
        self.prior_mlp = nn.Sequential(nn.Linear(self.z_dim, self.dense_h),
                                       nn.ReLU(),
                                       nn.Dropout(self.dropout_p)
                                      )
        # 2. self.dense_h to mu, cov 
        self.prior_mean = nn.Linear(self.dense_h, self.z_dim)
        self.prior_diag = nn.Linear(self.dense_h, self.z_dim)
        self.prior_off_diag = nn.Linear(self.dense_h, self.z_dim*(self.z_dim - 1)//2)
        
        
        ######################
        #### Generation x ####
        ######################
        # project to dense_h and compute mu and logvar
        self.mlp_z_x = nn.Sequential(nn.Linear(self.z_dim, self.dense_x_gx), 
                                     nn.ReLU(), 
                                     nn.Dropout(p=self.dropout_p))

        
    def add_session(self, session_id, init_strategy='mean'):
        """
        Crea i layer di input/output per una nuova sessione.
        init_strategy='mean': Inizializza i pesi come media delle sessioni esistenti (partenza migliore).
        """
        sid = str(session_id)
        
        # Crea nuovi layer
        in_layer = nn.Linear(self.x_dim, self.dense_x_gx)
        out_mean = nn.Linear(self.dense_x_gx, self.x_dim)
        out_logvar = nn.Linear(self.dense_x_gx, self.x_dim)
        
        # Inizializzazione Intelligente (Warm Start)
        if init_strategy == 'mean' and len(self.input_adapters) > 0:
            with torch.no_grad():
                # Calcola media pesi input
                avg_weight_in = torch.mean(torch.stack([layer.weight for layer in self.input_adapters.values()]), dim=0)
                avg_bias_in = torch.mean(torch.stack([layer.bias for layer in self.input_adapters.values()]), dim=0)
                in_layer.weight.copy_(avg_weight_in)
                in_layer.bias.copy_(avg_bias_in)
                
                # Calcola media pesi output
                avg_weight_out_mean = torch.mean(torch.stack([layer.weight for layer in self.output_adapters_mean.values()]), dim=0)
                avg_bias_out_mean = torch.mean(torch.stack([layer.bias for layer in self.output_adapters_mean.values()]), dim=0)
                out_mean.weight.copy_(avg_weight_out_mean)
                out_mean.bias.copy_(avg_bias_out_mean)
                
                # Calcola media pesi output
                avg_weight_out_logvar = torch.mean(torch.stack([layer.weight for layer in self.output_adapters_logvar.values()]), dim=0)
                avg_bias_out_logvar = torch.mean(torch.stack([layer.bias for layer in self.output_adapters_logvar.values()]), dim=0)
                out_logvar.weight.copy_(avg_weight_out_logvar)
                out_logvar.bias.copy_(avg_bias_out_logvar)
        
        # Aggiungi al modello (spostandoli sul device corretto)
        device = next(self.parameters()).device
        self.input_adapters[sid] = in_layer.to(device)
        self.output_adapters_mean[sid] = out_mean.to(device)
        self.output_adapters_logvar[sid] = out_logvar.to(device)
        print(f"Session '{sid}' added. Init strategy: {init_strategy}")
        

    def freeze_core(self, freeze=True):
        """
        Blocca/Sblocca i gradienti per la parte condivisa.
        Usa freeze=True durante la calibrazione di una nuova sessione.
        """
        for name, param in self.named_parameters():
            # Se il parametro NON appartiene agli adapters, bloccalo
            if "adapters" not in name:
                param.requires_grad = not freeze
        
        status = "FROZEN" if freeze else "UNFROZEN"
        print(f"Shared Core is now {status}.")
        
        
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
    

    def inference(self, x, c, session_id):
        
        seq_len, batch_size, _ = x.shape

        # Create variable holder and send to GPU if needed
        z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_cov = torch.zeros((seq_len, batch_size, self.z_dim, self.z_dim)).to(self.device)
        z_L = torch.zeros((seq_len, batch_size, self.z_dim, self.z_dim)).to(self.device)
        u_mean = torch.zeros((seq_len, batch_size, self.u_dim)).to(self.device)
        u_logvar = torch.zeros((seq_len, batch_size, self.u_dim)).to(self.device)
        z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        u = torch.zeros((seq_len, batch_size, self.u_dim)).to(self.device)
        z_t = torch.zeros((batch_size, self.z_dim)).to(self.device)
        u_t = torch.zeros((batch_size, self.u_dim)).to(self.device)
        
        sid = str(session_id)
        if sid not in self.input_adapters:
            raise ValueError(f"Session {sid} not found! Call add_session('{sid}') first.")

        # 1. ADAPT INPUT (Session Specific)
        h_input = F.relu(self.input_adapters[sid](x))
        # 2. concat h_input with context and pass it to the RNN
        x_comb = torch.cat([h_input, c], dim=-1)
        h_seq, _ = self.encoder_rnn(x_comb)  
        # Apply LayerNorm to the LSTM's output sequence
        h_seq = self.h_layernorm(h_seq)
        for t in range(seq_len):
            g_ztm1 = self.posterior_mlp(z_t)     # proietto z_tm1 in modo che abbia dimensione union_dim
            g_utm1 = self.posterior_mlp_u(u_t)
            g_comb_u = torch.cat([h_seq[t], g_ztm1], dim=-1)
            g_u = self.posterior_gru_u(g_comb_u, g_utm1)
            h_u = self.posterior_layernorm_u(g_u)
            u_mean[t,:,:] = self.inf_mean_u(h_u) 
            u_logvar[t,:,:] = self.inf_logvar_u(h_u)
            u_t = self.reparameterization(u_mean[t,:,:], u_logvar[t,:,:])
            u[t,:,:] = u_t
            g_ut = self.posterior_mlp_u(u_t)
            g_z = self.posterior_gru(g_ut, g_ztm1)
            h_z = self.posterior_layernorm(g_z)
            # Calcolo media e cov di z_t
#             h_z = self.hidden_layer(g_z)
            z_mean[t,:,:] = self.inf_mean(h_z)           # DA QUESTO g_z RICAVO MEDIA E LOGVAR DI q di phi
            if torch.onnx.is_in_onnx_export():
                z_cov[t,:,:,:] = torch.zeros(batch_size, self.z_dim, self.z_dim)  # oppure None
                z_L[t,:,:,:] = torch.zeros(batch_size, self.z_dim, self.z_dim)
            else:
                z_cov[t,:,:,:], z_L[t,:,:,:] = self.covariance(self.inf_diag(h_z), self.inf_off_diag(h_z))
            z_t = self.reparameterization_cov(z_mean[t,:,:], z_L[t,:,:,:])      # ESTRAGGO z_t DA q di phi
            # z_t = z_mean[t,:,:]
            z[t,:,:] = z_t

        return z, z_mean, z_cov, u, u_mean, u_logvar
    
    def generation_u(self, u_tm1):
        
        u_mean_p = self.u_phi * u_tm1
        u_logvar_p = self.logvar_prior_u.expand_as(u_mean_p)
        
        return u_mean_p, u_logvar_p
    
    
    def generation_z(self, z_tm1, c, u):          # z_tm1 sarebbe z_{t-1}
        
        batch_size = z_tm1.shape[1]
        c_comb = torch.cat([c, u], dim=-1)
        gamma, beta = self.cond_proj(c_comb).chunk(2, dim=-1)
        h_tm1 = self.prior_mlp(z_tm1)
        h_t = gamma * h_tm1 + beta
        
        z_mean_p = self.prior_mean(h_t)
        if torch.onnx.is_in_onnx_export():
            z_cov_p = torch.zeros(batch_size, self.z_dim, self.z_dim)  # oppure None
            z_L_p = torch.zeros(batch_size, self.z_dim, self.z_dim)
        else:
            z_cov_p, z_L_p = self.covariance(self.prior_diag(h_t), self.prior_off_diag(h_t))
        return z_mean_p, z_cov_p

        
    def generation_x(self, z, session_id):
        
        sid = str(session_id)
        if sid not in self.input_adapters:
            raise ValueError(f"Session {sid} not found! Call add_session('{sid}') first.")
        
        # 1. z_t to y_t
        h_output = self.mlp_z_x(z) 
        # 1. ADAPT OUTPUT (Session Specific)
        gen_mean = self.output_adapters_mean[sid](h_output)
        gen_logvar = self.output_adapters_logvar[sid](h_output)
        
        return gen_mean, gen_logvar
    

    def forward(self, x, c, session_id):
        
        # need input:  (seq_len, batch_size, x_dim)
        seq_len, batch_size, _ = x.shape
        
        self.z, self.z_mean, self.z_cov, self.u, self.u_mean, self.u_cov = self.inference(x, c, session_id)
        z_0 = torch.zeros(1, batch_size, self.z_dim).to(self.device)
        u_0 = torch.zeros(1, batch_size, self.u_dim).to(self.device)
        z_tm1 = torch.cat([z_0, self.z[:-1, :,:]], dim=0)
#         u_tm1 = torch.cat([u_0, self.u[:-1, :,:]], dim=0)
        self.z_mean_p, self.z_cov_p = self.generation_z(z_tm1, c, self.u)  
        self.u_mean_p, self.u_logvar_p = self.generation_u(u_tm1)  
        gen_mean, gen_logvar = self.generation_x(self.z, session_id)

        return gen_mean, gen_logvar

    def loss_rec(self, gen_mean, gen_logvar, x):
        
        # Equivalente ad assumere p(x|z) Gaussiana 
        gen_var = gen_logvar.exp()
        recon_loss = self.loss(x, gen_mean, gen_var)
        return recon_loss.sum() 
    
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
#         info.append(self.gen_mean)
#         info.append(self.gen_logvar)
        
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
    

    # ======================================================
    # 2. ARRIVA UNA NUOVA SESSIONE (Test / Calibrazione)
    # ======================================================

#     new_session_id = 'sess_new_04'
#     # Immaginiamo di avere 600 trials nuovi. Ne prendiamo 50 per calibrare.
#     data_new_session = torch.randn(600, 50, 96).to(device) 
#     context_new_session = torch.randn(600, 50, 4).to(device)

#     calibration_x = data_new_session[:50] # Pochi dati!
#     calibration_c = context_new_session[:50]
#     test_x = data_new_session[50:]

#     # A. Aggiungi la nuova sessione al modello
#     # Usa 'mean' per partire da una media delle proiezioni imparate (spesso funziona bene)
#     model.add_session(new_session_id, init_strategy='mean')

#     # B. Congela il "Cervello" (Shared Core)
#     # Vogliamo solo imparare come mappare i nuovi elettrodi, non cambiare la dinamica
#     model.freeze_core(True)

#     # C. Loop di Calibrazione Veloce
#     optimizer_calib = torch.optim.Adam(
#         # Ottimizza SOLO i parametri della nuova sessione
#         list(model.input_adapters[new_session_id].parameters()) + 
#         list(model.output_adapters[new_session_id].parameters()), 
#         lr=0.01 # Learning rate un po' più alto per fare presto
#     )

#     print(f"Calibrating on {new_session_id}...")
#     model.train()
#     for epoch in range(50): # Pochi epoch sono sufficienti (es. 50-100)
#         optimizer_calib.zero_grad()
#         recon, mu_q, lv_q, mu_p, lv_p = model(calibration_x, calibration_c, new_session_id)

#         # Loss Semplificata o Completa (solitamente basta MSE per allineare)
#         # loss = F.mse_loss(recon, calibration_x) 
#         # Oppure la tua loss completa DKF
#         loss = F.mse_loss(recon, calibration_x) # Esempio semplice

#         loss.backward()
#         optimizer_calib.step()

#     print("Calibration Done.")

#     # D. Test / Generazione
#     # Ora possiamo usare il modello su tutti gli altri dati della nuova sessione
#     model.eval()
#     with torch.no_grad():
#         recon_test, _, _, _, _ = model(test_x, context_new_session[50:], new_session_id)
#         print("Inference on new session successful. Output shape:", recon_test.shape)
