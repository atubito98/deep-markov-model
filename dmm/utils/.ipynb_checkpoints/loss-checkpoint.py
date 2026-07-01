#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

def loss_ISD(x, y):
    y = y + 1e-10
    ret = torch.sum(x/y - torch.log(x/y) - 1)
    return ret

# === helper for multivariate Gaussian KL between q(s)||p(s) (diag case for s) ===
def loss_KLD_diag(mu_q, logvar_q):
    return -0.5 * torch.sum(1 + logvar_q - mu_q.pow(2) - logvar_q.exp())


def loss_KLD(z_mean, z_logvar, z_mean_p=0, z_logvar_p=0):
    ret = -0.5 * torch.sum(z_logvar - z_logvar_p 
                - torch.div(z_logvar.exp() + (z_mean - z_mean_p).pow(2), z_logvar_p.exp()+1e-10))
    return ret

def loss_KLD_cov(z_mean, z_cov, z_mean_p, z_cov_p, z_dim):
    #ret = 0.5 * torch.sum(torch.einsum('...ii->...', torch.matmul(torch.inverse(z_cov), z_cov_p)) +
    #                torch.einsum('bij,bijk,bik->bi', z_mean - z_mean_p, torch.inverse(z_cov), z_mean - z_mean_p) - z_dim +
    #                torch.log(torch.div(torch.det(z_cov), torch.det(z_cov_p)+1e-10)))
    
    
    a = torch.einsum('...ii->...', torch.matmul(torch.inverse(z_cov_p), z_cov))
    b = torch.einsum('bij,bijk,bik->bi', z_mean_p - z_mean, torch.inverse(z_cov_p), z_mean_p - z_mean) - z_dim
    c = torch.log(torch.div(torch.det(z_cov_p), torch.det(z_cov)+1e-10))
    
    ret = 0.5 * torch.sum(a + b + c)
    return ret#, a, b, c

def analyze_latent_dimensions(z_mean, z_cov, z_mean_p, z_cov_p):
    """
    Analyzes the 'activity' of latent dimensions in a sequence model (DMM).
    It ignores off-diagonal correlations to calculate the Marginal KL.
    
    Args:
        z_mean:   Posterior Mean  [n_trials, steps, z_dim]
        z_cov:    Posterior Cov   [n_trials, steps, z_dim, z_dim]
        z_mean_p: Prior Mean      [n_trials, steps, z_dim]
        z_cov_p:  Prior Cov       [n_trials, steps, z_dim, z_dim]
        threshold: Value below which a unit is considered "dead"
        
    Returns:
        kl_per_dim: Tensor of shape [z_dim] containing average KL for each unit.
    """
    
    # 1. Extract Diagonals (Variances)
    # Input: [n_trials, steps, z_dim, z_dim]
    # Output: [n_trials, steps, z_dim]
    # dim1=-2, dim2=-1 refer to the last two dimensions (the matrix part)
    var = torch.diagonal(z_cov, dim1=-2, dim2=-1)
    var_p = torch.diagonal(z_cov_p, dim1=-2, dim2=-1)
    
    # 2. Add epsilon for numerical stability
    eps = 1e-10
    
    # 3. Calculate Gaussian KL (Univariate/Diagonal formula)
    # KL( q(z|x) || p(z) ) = 0.5 * [ log(var_p/var) + (var + (mu - mu_p)^2)/var_p - 1 ]
    
    # Log term: log(var_p) - log(var)
    log_term = torch.log(var_p + eps) - torch.log(var + eps)
    
    # Trace term numerator: var + error^2
    diff_squared = (z_mean - z_mean_p).pow(2)
    trace_term = (var + diff_squared) / (var_p + eps)
    
    # Element-wise KL [n_trials, steps, z_dim]
    kl_elementwise = 0.5 * (log_term + trace_term - 1)
    
    # 4. Average over Trials (dim 0) and Time Steps (dim 1)
    # We want to know if dimension 'j' is active generally, across all time and data.
    kl_per_dim = torch.mean(kl_elementwise, dim=(0, 1)) # Result shape: [z_dim]
    return kl_per_dim.cpu().detach().numpy()

def loss_JointNorm(x, y, nfeats=3):
    seq_len, bs, _ = x.shape
    x = x.reshape(seq_len, bs, -1, nfeats)
    y = y.reshape(seq_len, bs, -1, nfeats)
    ret = torch.sum(torch.norm(x-y, dim=-1))
    return ret

def loss_MPJPE(x, y, nfeats=3):
    seq_len, bs, _ = x.shape
    x = x.reshape(seq_len, bs, -1, nfeats)
    y = y.reshape(seq_len, bs, -1, nfeats)
    ret = (x-y).norm(dim=-1).mean(dim=-1).sum()
    return ret


def loss_rec(x, y):
    ret = ((x-y)**2).sum()
    return ret

# def loss_rec_prob(mu_out, logvar_out, x):
#         # Equivalente ad assumere p(x|z) Gaussiana 
#         var_out = logvar_out.exp()
#         recon_loss = self.loss(x, mu_out, var_out)
#         return recon_loss 

# def loss_ISD(x, y):
#     seq_len, bs, _ = x.shape
#     ret = torch.sum( x/y - torch.log(x/y) - 1)
#     ret = ret / (bs * seq_len)
#     return ret

# def loss_KLD(z_mean, z_logvar, z_mean_p=0, z_logvar_p=0):
#     if len(z_mean.shape) == 3:
#         seq_len, bs, _ = z_mean.shape
#     elif len(z_mean.shape) == 2:
#         seq_len = 1
#         bs, _ = z_mean.shape
#     ret = -0.5 * torch.sum(z_logvar - z_logvar_p 
#                 - torch.div(z_logvar.exp() + (z_mean - z_mean_p).pow(2), z_logvar_p.exp()))
#     ret = ret / (bs * seq_len)
#     return ret

# def loss_JointNorm(x, y, nfeats=3):
#     seq_len, bs, _ = x.shape
#     x = x.reshape(seq_len, bs, -1, nfeats)
#     y = y.reshape(seq_len, bs, -1, nfeats)
#     return torch.mean(torch.norm(x-y, dim=-1))



