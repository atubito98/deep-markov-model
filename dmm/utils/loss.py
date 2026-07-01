#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


# === helper for multivariate Gaussian KL between q(s)||p(s) (diag case for s) ===
def loss_KLD_diag(mu_q, logvar_q):
    return -0.5 * torch.sum(1 + logvar_q - mu_q.pow(2) - logvar_q.exp())


def loss_KLD(z_mean, z_logvar, z_mean_p=0, z_logvar_p=0):
    ret = -0.5 * torch.sum(z_logvar - z_logvar_p 
                - torch.div(z_logvar.exp() + (z_mean - z_mean_p).pow(2), z_logvar_p.exp()+1e-10))
    return ret

def loss_KLD_cov(z_mean, z_cov, z_mean_p, z_cov_p, z_dim):
    
    a = torch.einsum('...ii->...', torch.matmul(torch.inverse(z_cov_p), z_cov))
    b = torch.einsum('bij,bijk,bik->bi', z_mean_p - z_mean, torch.inverse(z_cov_p), z_mean_p - z_mean) - z_dim
    c = torch.log(torch.div(torch.det(z_cov_p), torch.det(z_cov)+1e-10))
    
    ret = 0.5 * torch.sum(a + b + c)
    return ret

def loss_rec(x, y):
    ret = ((x-y)**2).sum()
    return ret



