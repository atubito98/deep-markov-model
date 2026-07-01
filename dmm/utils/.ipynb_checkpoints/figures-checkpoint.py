# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import math
import random
import matplotlib.gridspec as gridspec
from dmm.dataset import one_hot_cont, load_set, peak_filthers, process_data
from .loss import analyze_latent_dimensions

# +
ch2grid_template = {
    # 0 row
    97 : [0,0], 41 : [0,1], 39 : [0,2], 37 : [0,3], 43 : [0,4], 45 : [0,5], 47 : [0,6], 1 : [0,7], 5 : [0,8], 98 : [0,9],
    # 1 row
    96 : [1,0], 73 : [1,1], 95 : [1,2], 25 : [1,3], 33 : [1,4], 24 : [1,5], 22 : [1,6], 3 : [1,7], 7 : [1,8], 9 : [1,9],
    # 2 row
    94 : [2,0], 75 : [2,1], 93 : [2,2], 26 : [2,3], 35 : [2,4], 16 : [2,5], 20 : [2,6], 10 : [2,7], 11 : [2,8], 13 : [2,9],
    # 3 row
    92 : [3,0], 77 : [3,1], 91 : [3,2], 29 : [3,3], 55 : [3,4], 18 : [3,5], 14 : [3,6], 8 : [3,7], 6 : [3,8], 15 : [3,9],
    # 4 row
    90 : [4,0], 79 : [4,1], 89 : [4,2], 31 : [4,3], 49 : [4,4], 57 : [4,5], 61 : [4,6], 12 : [4,7], 4 : [4,8], 17 : [4,9],
    # 5 row
    88 : [5,0], 81 : [5,1], 48 : [5,2], 46 : [5,3], 51 : [5,4], 53 : [5,5], 59 : [5,6], 71 : [5,7], 2 : [5,8], 19 : [5,9],
    # 6 row
    86 : [6,0], 83 : [6,1], 44 : [6,2], 42 : [6,3], 38 : [6,4], 63 : [6,5], 65 : [6,6], 67 : [6,7], 69 : [6,8], 21 : [6,9],
    # 7 row
    84 : [7,0], 85 : [7,1], 50 : [7,2], 40 : [7,3], 36 : [7,4], 34 : [7,5], 32 : [7,6], 30 : [7,7], 28 : [7,8], 23 : [7,9],
    # 8 row
    82 : [8,0], 87 : [8,1], 52 : [8,2], 54 : [8,3], 74 : [8,4], 72 : [8,5], 70 : [8,6], 62 : [8,7], 27 : [8,8], 66 : [8,9],
    # 9 row
    99 : [9,0], 80 : [9,1], 78 : [9,2], 76 : [9,3], 56 : [9,4], 58 : [9,5], 60 : [9,6], 68 : [9,7], 64 : [9,8], 100 : [9,9],
    
  }
ch2grid=dict() # pythoneque indexing
for key, value in ch2grid_template.items():
    ch2grid[(key-1)] = value
 
 
def channel2grid(data):
    n=len(data.shape)
    out_shape = [10, 10]
    out_shape = list(data.shape[:-1]) + out_shape
    grid_data = np.zeros(out_shape)
    
    valid_channels = {k: v for k, v in ch2grid.items() if k < data.shape[-1]}
    for ch, (i, j) in valid_channels.items():
        if n == 1:
            grid_data[i, j] = data[ch]
        elif n == 2:
            grid_data[:, i, j] = data[:, ch]
        elif n == 3:
            grid_data[:, :, i, j] = data[:, :, ch]
        else:  # n == 4
            grid_data[:, :, :, i, j] = data[:, :, :, ch]
    return grid_data

def binary_output(prob_logits):
    prob_pred = torch.sigmoid(prob_logits)
    predicted_labels = (prob_pred > 0.5).float()
    return predicted_labels.cpu().detach().numpy()

def prob_to_RT(output, tau):
    output_clamped = torch.clamp(output, 0, 1)
    output_clamped = output_clamped.cpu().detach().numpy()
    RT = (output_clamped * 200) // tau
    return RT 


def inference_with_trials(dmm, data_set, cont_set, n_trials, input_dim, device, chunk_size=10):#, u_flag=False):
    """
    Esegue l'inferenza in chunk per evitare errori di memoria GPU.

    """
    n_samples, steps, features = data_set.shape
    _, _, c_dim = cont_set.shape
    
    data_set = torch.from_numpy(data_set).float().permute(1, 0, 2)
    cont_set = torch.from_numpy(cont_set).float().permute(1, 0, 2)
    
    z_dim = dmm.z_dim
    z_mean_accum = np.zeros((steps, n_samples, z_dim), dtype=np.float32)
#     if u_flag:
#         u_dim = dmm.u_dim
#         u_mean_accum = np.zeros((steps, n_samples, u_dim), dtype=np.float32)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        batch_size = end - start

        # Estrai chunk
        x_chunk = data_set[:, start:end, :].repeat_interleave(n_trials, dim=1).to(device)
        c_chunk = cont_set[:, start:end, :].repeat_interleave(n_trials, dim=1).to(device)


        # Inferenza
        with torch.no_grad():
#             if u_flag:
#                 z, z_mean, _, u, _, _ = dmm.inference(x_chunk, c_chunk)
#                 u_mean_chunk = u.cpu().numpy().reshape(steps, batch_size, n_trials, u_dim).mean(2)
#                 u_mean_accum[:, start:end, :] = u_mean_chunk
#             else:
            z, z_mean, _ = dmm.inference(x_chunk, c_chunk)

        z_mean_chunk = z.cpu().numpy().reshape(steps, batch_size, n_trials, z_dim).mean(2)

        # Inserisci nel buffer
        z_mean_accum[:, start:end, :] = z_mean_chunk

        torch.cuda.empty_cache()
    z_mean_trasp = np.transpose(z_mean_accum, (1, 0, 2))
    return z_mean_trasp
        

def setup_matplotlib_backend():
    try:
        # Se siamo in Jupyter, abilita la modalità interattiva 3D
        get_ipython().run_line_magic('matplotlib', 'widget')
    except Exception:
        # Se siamo in un file .py normale, usa modalità interattiva standard
        plt.ion()


        
def test_logvar(comm_dict, diff_dict):
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    c_dim = comm_dict["c_dim"]
    
    data = diff_dict["data"]
    cmap = diff_dict["cmap"]
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    
    set_cn = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2)
    cont_cn = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2)#.repeat_interleave(n_trials, dim=1)
    
    _, y_logvar = dmm(set_cn, cont_cn)
    y_std = torch.exp(0.5*y_logvar)
    y_std = y_std.permute(1, 0, 2).cpu().detach().numpy()
    
    y_std_flatten = y_std.flatten()
    
    num_bins = 30
    min_value = 0
    max_value = 1
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].hist(y_std_flatten, bins=bin_edges, color='skyblue', edgecolor='black')
    # Add labels and title
    ax[0].set_xlabel('y_std')
    ax[0].set_ylabel('# of trials')
    ax[0].set_title("Histograms of y_std for correct-nostop trials")
    
    im = ax[1].imshow(y_std.mean(2), cmap=cmap, aspect='auto')
    ax[1].set_xlabel("time steps")
    ax[1].set_ylabel("trials")
    ax[1].set_title("y_std values of reconstruced y, meaned across channels")
    plt.colorbar(im, ax=ax, shrink = 0.7, aspect = 15)
    plt.show()



        
def RT_pred_performance(comm_dict, diff_dict):
    
    RT_detector = comm_dict["RT_detector"]
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    c_dim = comm_dict["c_dim"]
    ar = comm_dict["ar"]
    
    data = diff_dict["data"]
    mean_z = diff_dict["mean_z"]
    n_trials = diff_dict["n_trials"]
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    #RT_min = RT_cn.min()
    RT_min = 50
    samples_cn, steps, features = set_cn.shape
    
    set_cn = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    cont_cn = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    
    z_cn, _, _ = dmm.inference(set_cn, cont_cn)
    
    if mean_z:
        z_cn = z_cn.reshape(steps, samples_cn, n_trials, z_dim)
        z_cn = z_cn.mean(2)
        if ar:
            set_cn = set_cn.reshape(steps, samples_cn, n_trials, 96)
            set_cn = set_cn.mean(2)
            y_mean, y_logvar = dmm.generation_x(z_cn, set_cn)
        else:
            y_mean, y_logvar = dmm.generation_x(z_cn)
        y_rec = dmm.reparameterization(y_mean, y_logvar)
    else:
        if ar:
            y_mean, y_logvar = dmm.generation_x(z_cn, set_cn)
        else:
            y_mean, y_logvar = dmm.generation_x(z_cn)
        y_rec = dmm.reparameterization(y_mean, y_logvar)
        y_rec = y_rec.reshape(steps, samples_cn, n_trials, features)
        y_rec = y_rec.mean(2)
    y_rec = y_rec.permute(1, 0, 2).cpu().detach().numpy()
    MUA_rec = y_rec.mean(2)
    RT_MUA = np.argmax(MUA_rec[:, (56+RT_min)//tau:], axis=1) + (56+RT_min)//tau
    
    z_cn = z_cn.permute(1, 0, 2)
    RT_output = RT_detector(z_cn)
    RT_estimate = prob_to_RT(RT_output, tau)  
    RT_cn = (RT_cn+56)//tau
    
    correlation_MUA = np.corrcoef(RT_cn, RT_MUA)[0, 1]
    # Fit lineare
    A = np.vstack([RT_cn, np.ones(len(RT_cn))]).T
    m, c = np.linalg.lstsq(A, RT_MUA, rcond=None)[0]
    RT_MUA_fit = m * RT_cn + c

    # Plot
    plt.figure(figsize=(6,4))
    plt.scatter(RT_cn, RT_MUA, color='royalblue', alpha=0.6, label='Dati')
    plt.plot(RT_cn, RT_MUA_fit, color='tomato', lw=2, label=f'Fit: y={m:.2f}x+{c:.2f}')
    plt.xlabel('True RT')
    plt.ylabel('Predicted RT')
    plt.title(f'Scatter & Fit. Correlazione Pearson = {correlation_MUA:.2f}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    correlation_detector = np.corrcoef(RT_cn, RT_estimate)[0, 1]
    # Fit lineare
    A = np.vstack([RT_cn, np.ones(len(RT_cn))]).T
    m, c = np.linalg.lstsq(A, RT_estimate, rcond=None)[0]
    RT_estimate_fit = m * RT_cn + c

    # Plot
    plt.figure(figsize=(6,4))
    plt.scatter(RT_cn, RT_estimate, color='royalblue', alpha=0.6, label='Dati')
    plt.plot(RT_cn, RT_estimate_fit, color='tomato', lw=2, label=f'Fit: y={m:.2f}x+{c:.2f}')
    plt.xlabel('True RT')
    plt.ylabel('Predicted RT (using detector)')
    plt.title(f'Scatter & Fit. Correlazione Pearson = {correlation_detector:.2f}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def cs_ws_SSD_hist(diff_dict):
    data = diff_dict["data"]
    SSD_cs = data["SSD_cs_ordSSD"]
    SSD_ws = data["SSD_ws_ordSSD"]
    
    print("mean train_SSD_ws:")
    print(SSD_ws.mean())
    print("mean train_SSD_cs:")
    print(SSD_cs.mean())
    
    print("cs_SSD:")
    print(np.unique(SSD_cs))
    print("ws_SSD:")
    print(np.unique(SSD_ws))
    
    stop_cs = (SSD_cs+56)*5
    stop_ws = (SSD_ws+56)*5
    
    num_bins = 20
    min_value = min(stop_cs.min(), stop_ws.min())
    max_value = max(stop_cs.max(), stop_ws.max())
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)
    
    fig, ax = plt.subplots()
    ax.hist(stop_ws, bins=bin_edges, alpha = 0.5, color='red', edgecolor='black', label = "wrong stop")
    ax.hist(stop_cs, bins=bin_edges, alpha = 0.5, color='skyblue', edgecolor='black', label = "correct stop")
    # Add labels and title
    ax.set_xlabel('SSD')
    ax.set_ylabel('# of trials')
    ax.set_title("Histograms of SSDs for wrong and correct stop trials")
    ax.legend()
    
def cn_ws_RT_hist(diff_dict):
    data = diff_dict["data"]
    RT_cn = data["RT_cn_ordRT"]
    RT_ws = data["RT_ws_ordRT"]
    
    mov_cn = (RT_cn+56)*5
    mov_ws = (RT_ws+56)*5
    
    num_bins = 30
    min_value = min(mov_cn.min(), mov_ws.min())
    max_value = max(mov_cn.max(), mov_ws.max())
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)
    
    fig, ax = plt.subplots()
    ax.hist(mov_cn, bins=bin_edges, alpha = 0.5, color='skyblue', edgecolor='black', label = "cn")
    ax.hist(mov_ws, bins=bin_edges, alpha = 0.5, color='red', edgecolor='black', label = "ws")
    # Add labels and title
    ax.set_xlabel('Simulation start time ($ms$)')
    ax.set_ylabel('# of trials')
    ax.set_title("Histograms of RTs for correct no-stop trials")
    
    
    
def session_RT_hist(diff_dict):
    data = diff_dict["data"]
    RT_cn = data["RT_cn_ordRT"]
    session = data["sess_cn_ordRT"]
    
    session_unique = np.unique(session)
    
    num_bins = 30
    min_value = (56+50)*5
    max_value = 256*5
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)
    
    fig, ax = plt.subplots()
    
    for i in session_unique:
    
        mov = (RT_cn[session==i]+56)*5
        ax.hist(mov, bins=bin_edges, alpha = 0.5, edgecolor='black', label = f"session {i}")
        ax.axvline(mov.mean(), linestyle="--",label=f"session {i}")
        
    # Add labels and title
    ax.set_xlabel('RT time ($ms$)')
    ax.set_ylabel('# of trials')
    ax.set_title("Histograms of RTs for correct no-stop trials of different sessions")


# -

def infer_latent(dmm, data, device, n_trials=1):
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    set_ws = data["set_ws_ordRT"]
    cont_ws = data["cont_ws_ordRT"]
    set_cs = data["set_cs_ordSSD"]
    cont_cs = data["cont_cs_ordSSD"]
    
    c_dim = cont_cs.shape[2]
    samples_cn = set_cn.shape[0]
    samples_cs = set_cs.shape[0]
    samples_ws = set_ws.shape[0]
    steps = set_cn.shape[1]
    
    set_cn = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, samples_cn*n_trials, 96)
    cont_cn = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, samples_cn*n_trials, c_dim)
    set_ws = torch.from_numpy(set_ws).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, samples_ws*n_trials, 96)
    cont_ws = torch.from_numpy(cont_ws).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, samples_ws*n_trials, c_dim)
    set_cs = torch.from_numpy(set_cs).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, samples_cs*n_trials, 96)
    cont_cs = torch.from_numpy(cont_cs).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, samples_cs*n_trials, c_dim)
        
    z_cs, z_mean_cs, _ = dmm.inference(set_cs, cont_cs)
    z_cn, z_mean_cn, _ = dmm.inference(set_cn, cont_cn)
    z_ws, z_mean_ws, _ = dmm.inference(set_ws, cont_ws)

    z_cs = z_cs.cpu().detach().numpy()
    z_cn = z_cn.cpu().detach().numpy()
    z_ws = z_ws.cpu().detach().numpy()
    return z_cn, z_ws, z_cs



# +
def single_RTcorr(comm_dict, diff_dict, sim_start):

    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    c_dim = comm_dict["c_dim"]
    z_dim = comm_dict["z_dim"]
    RT_detector = comm_dict["RT_detector"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    mean_corr = diff_dict["mean_corr"]
    min_n = diff_dict["min_n"]
    
    set_cn = data["set_cn_ordRT"]
    cont_c = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    
    samples, steps, features = set_cn.shape
    RT_cn_ordered = RT_cn//tau
    teacher = sim_start//(5*tau)
    alone = steps - teacher
    # 1. Per ogni sim_start, seleziono solo i trial con RT > sim_start (sono s)
    mask = (RT_cn_ordered + 56//tau) > teacher
    s = mask.sum()
    set_cn = set_cn[mask]
    cont_c = cont_c[mask]
    RT_cn_ordered_filt = RT_cn_ordered[mask]
    
    # 2. calcolo la correlazione solo se il numero di trials con RT > sim_start è maggiore di 20, per avere un minimo di statistica  
    if s > min_n:
    
        set_cn = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1) 
        cont_c = torch.from_numpy(cont_c).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)

        z_cn, _, _ = dmm.inference(set_cn, cont_c)
        z_teach = z_cn[:teacher]
        
        # inferisco fino a sim_start e genero da lì in poi
        for step in range(alone):
            z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher+step].unsqueeze(0))
            z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
            z_teach = torch.cat((z_teach, z_gen), dim=0)
        
        # 3. se non faccio la media delle correlazioni su n_trials, allora uso la media di z su n_trials per calcolare una sola correlazione
        if not mean_corr:
            z_teach = z_teach.reshape(steps, s, n_trials, z_dim)
            z_teach = z_teach.mean(2)
            
        
        # 4. se ho a disposizione l'RT detector, calcolo l'RT della traiettoria generata usando il detector
        RT_output = RT_detector(z_teach.permute(1, 0, 2))
        RT_pred = prob_to_RT(RT_output, tau)

        if mean_corr:
            RT_pred = RT_pred.reshape(s, n_trials)
            # mask peak elimina i trials per cui anche solo una realizzazione (tra le n_trials) ha RT che coincide con teacher
            mask_peak = np.all((RT_pred + 56//tau) != teacher, axis=1)
            RT_cn_ordered_filt = RT_cn_ordered_filt[mask_peak]
            RT_pred_filt = RT_pred[mask_peak]
            non_zero_counts = np.count_nonzero(RT_pred_filt, axis=1)
            K = np.min(non_zero_counts)
            N = RT_pred_filt.shape[0]
            RT_est = np.full((N, K), np.nan)
            # Fill each row with its non-zero elements
            for i in range(N):
                non_zeros = RT_pred_filt[i][RT_pred_filt[i] != 0]  # Get non-zero elements
                RT_est[i] = non_zeros[:K]  # Assign to output array
            #print(K)
            corr = np.array([np.corrcoef(RT_cn_ordered_filt, RT_est[:, j])[0, 1] for j in range(K)])  
            correlation = np.mean(corr)
            corr_std = np.std(corr)
        else:
            mask_peak = (RT_pred + 56//tau) != teacher
            mask_null = RT_pred != 0
            #print("RT diversi da 0:")
            #print(mask_null.sum())
            mask_comb = mask_peak & mask_null
            RT_cn_ordered_filt = RT_cn_ordered_filt[mask_comb]
            RT_pred_filt = RT_pred[mask_comb]
#             if mask_comb.sum() > 20:
            correlation = np.corrcoef(RT_cn_ordered_filt, RT_pred_filt)[0, 1]
            corr_std = None
#             else:
#                 correlation = 2
#                 corr_std = 1
    else:
        RT_cn_ordered_filt = 0
        RT_pred_filt = np.ones((1, 2))
        correlation = 2
        corr_std = 1

    return RT_cn_ordered_filt, RT_pred_filt, correlation, corr_std
   


def correlation_vs_gentime(comm_dict, diff_dict):     
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    saved_path = comm_dict["saved_path"]
    
    data = diff_dict["data"]
    sim_start_array = diff_dict["sim_start_array"]
    n_trials = diff_dict["n_trials"]
    mean_corr = diff_dict["mean_corr"]
    compute = diff_dict["compute"]
    show_comp = diff_dict["show_comp"]
    save = diff_dict["save"]
    saved_dict = diff_dict["saved_dict"]
    dir_comp = diff_dict["dir_comp"]
    color_corr = diff_dict["color_corr"]
    color_true = diff_dict["color_true"]
    color_edge = diff_dict["color_edge"]
    num_bins = diff_dict["num_bins"]
    alpha = diff_dict["alpha"]
    h_hist = diff_dict["h_hist"]
    y_lim = diff_dict["y_lim"]
    figsize =  diff_dict["figsize"]
    
    RT_cn = data["RT_cn_ordRT"]
    
    if compute:
        correlations = np.zeros(len(sim_start_array))
        corr_stds = np.zeros(len(sim_start_array))
        for cont, sim_start in enumerate(sim_start_array):
            teacher = sim_start//(5*tau)

            _, _, correlation, corr_std = single_RTcorr(comm_dict, diff_dict, sim_start)
            #print(correlation)
            correlations[cont] = correlation
            corr_stds[cont] = corr_std

        mask_new = correlations<2
        sim_start_array_new = sim_start_array[mask_new]
        corr_stds_new = corr_stds[mask_new]
        correlations_new = correlations[mask_new]
        if save:
            np.savez(saved_dict + "/RT_correlations.npz", sim_start_array = sim_start_array_new, correlations = correlations_new, corr_stds = corr_stds_new)
    else:
        with np.load(saved_path + "/RT_correlations.npz") as loaded_file:
            correlations_new = loaded_file["correlations"]
            corr_stds_new = loaded_file["corr_stds"]
            sim_start_array_new = loaded_file["sim_start_array"]
    
    fig, ax1 = plt.subplots(figsize=figsize)

    #ax1.errorbar(sim_start_array_new, correlations_new, yerr=corr_stds_new, fmt='o', ls='--', ecolor='black', 
#     ax1.errorbar(sim_start_array_new, correlations_new, yerr=corr_stds_new, fmt='o', ls='--', ecolor='black', 
#                      elinewidth=1, capsize=5, capthick=1)
    ax1.scatter(sim_start_array_new, correlations_new, color=color_corr, edgecolors=color_edge)#, label = "current model")
        
    if show_comp:
        directory = "/raid/home/tubitoal/DMM/saved_model/" + dir_comp
        with np.load(directory + "/RT_correlations.npz") as loaded_file:
            correlations_old = loaded_file["correlations"]
            corr_stds_old = loaded_file["corr_stds"]
            sim_start_array_old = loaded_file["sim_start_array"]

        ax1.errorbar(sim_start_array_old, correlations_old, yerr=corr_stds_old, fmt='o', ecolor='black', 
                         elinewidth=1, capsize=5, capthick=1)
        ax1.scatter(sim_start_array_old, correlations_old, label = dir_comp)
    ax1.axvline(56*5, color="black", linestyle="--")#,label="GO")
    ax1.set_ylim((0, y_lim))
    ax1.set_xlabel("Simulation start ($ms$)")#, fontsize=font_ax)
    ax1.set_xticks([400, 800]) 
    ax1.set_xticklabels([400, 800])#, fontsize=font_tick) 
    ax1.set_ylabel("Correlation")#, fontsize=font_ax)
    ax1.set_yticks([0, 0.5, 1])
    ax1.set_yticklabels([0, 0.5, 1])#, fontsize=font_tick) 
#     ax1.set_title(f"Correlations between true and predicted RT")

    RT_true = (56 + RT_cn) * 5
    min_value = RT_true.min() - 50
    max_value = RT_true.max() + 50
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)

    ax2 = ax1.twinx()
    ax2.hist(RT_true, bins=bin_edges, density=True, alpha=alpha, color=color_true, edgecolor='none')#, label="RT histogram")
#     ax2.set_ylabel("Trial #")#, fontsize=font_ax)
    ax2.set_ylim(0, h_hist)
    ax2.set_yticks([])
 
   
    fig_file = os.path.join(comm_dict["saved_path"], 'RT_correlation.png')
    plt.savefig(fig_file)

    # Display the plot
    print(correlations_new.shape)
    plt.show()
        
    dict_corr = {a: (b, c) for a, b, c in zip(sim_start_array_new, correlations_new, corr_stds_new)}
    print(correlations_new[56//tau])
    
#     start_array = [56, 76, 96, 116, 136, 156]
    
#     for start in start_array:
#         teacher = start*5

#         RT_cn_ordered_filt, RT_pred_filt, corr, _ = single_RTcorr(comm_dict, diff_dict, teacher)
        
#         f, ax = plt.subplots(figsize = (6, 6))
    
#         ax.scatter(RT_cn_ordered_filt, RT_pred_filt)
#         ax.set_xlabel('True RT ($ms$)')#, fontsize=font_ax)
#         ax.set_ylabel('predicted RT')#, fontsize=font_ax)
#         #ax.set_title('Mean RT stop residual vs True RT')
#     #     ax.legend()

#         # Calcola i limiti comuni
#         min_val_x = ax.get_xlim()[0]
#         min_val_y = ax.get_ylim()[0]
#         max_val_x = ax.get_xlim()[1]
#         max_val_y = ax.get_ylim()[1]

#         # Disegna la diagonale y = x
#         ax.plot([min_val_x, max_val_x], [min_val_y, max_val_y], 'r--', label='diagonale')
#         ax.set_title(f"corr: {corr:.2f}")
#         plt.show()
    return dict_corr

def DMM_EV(comm_dict, diff_dict):
    
    with np.load(comm_dict["saved_path"]+"/data_split.npz", allow_pickle=True) as loaded_file:
            train_set = loaded_file["train_set"]
            vali_set = loaded_file["vali_set"]
            test_set = loaded_file["test_set"]
            train_direction = loaded_file["train_direction"]
            vali_direction = loaded_file["vali_direction"]
            test_direction = loaded_file["test_direction"]
            train_SSD = loaded_file["train_SSD"]
            vali_SSD = loaded_file["vali_SSD"]
            test_SSD = loaded_file["test_SSD"]
            
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    c_dim = comm_dict["c_dim"]
    tau = comm_dict["tau"]
    t = comm_dict["t"]

    #n_trials = diff_dict["n_trials"]
    width = diff_dict["width"]
    color = diff_dict["color"]
    colors = diff_dict["colors"]
    x_label = diff_dict["x_label"]
    x_lims = diff_dict["x_lims"]
    y_lims = diff_dict["y_lims"]
    delta_y = diff_dict["delta_y"]
    alpha = diff_dict["alpha"]
    letter = diff_dict["letter"]
 
    n_test, steps, features = test_set.shape
    cont_test = one_hot_cont(test_SSD, test_direction, tau)
    test_data = torch.from_numpy(test_set).float().to(device).permute(1, 0, 2)
    cont_test = torch.from_numpy(cont_test).float().to(device).permute(1, 0, 2)
    z, _, _ = dmm.inference(test_data, cont_test)
    z = z.cpu().detach().numpy().reshape(-1, z_dim) 
    
    z_scaled = z-z.mean(axis=0)

    from sklearn.decomposition import PCA
    # Applica PCA con n_components=3
    pca = PCA(n_components=z_dim)
    z_pca = pca.fit_transform(z_scaled)

    # Ottieni l'explained variance per ciascuna componente
    explained_variance = pca.explained_variance_ratio_
    ev_cm = np.cumsum(explained_variance)
    
    z_vec = (np.arange(z_dim) + 1)/(z_dim+1)
    
    f, ax = plt.subplots()
    
    ax.plot(z_vec, explained_variance, "--", color=color, alpha=alpha)
    ax.scatter(z_vec, explained_variance, color=color)
#     ax.bar(z_vec, ev_cm, color=colors, width=width)
    ax.set_xlim(x_lims)
    ax.set_xticks(z_vec)
    ax.set_xticklabels(x_label)#, fontsize=font_tick)
    #ax.set_title(f"Confronto {i+1}")
    ax.set_ylim(y_lims)
    ax.set_yticks(y_lims)
    ax.set_yticklabels(y_lims)#, fontsize=font_tick)
    #ax.set_xlabel('latent dimension ($ms$)')#, fontsize=font_ax)
    ax.set_ylabel('Explained variance')#, fontsize=font_ax)
    for xi, yi in zip(z_vec, explained_variance):
        ax.text(xi, yi + (delta_y), f"{yi:.2f}", ha='center')#, fontsize=font_tick)
   
    
    fig_file = os.path.join(comm_dict["saved_path"], 'DMM_EV.png')
    plt.savefig(fig_file)
    plt.show()

    

def DMM_vs_dim(comm_dict, diff_dict):
    
    width = diff_dict["width"]
    x = diff_dict["x"]
    y_lim = diff_dict["y_lim"]
    y_ticks = diff_dict["y_ticks"]
    n_trials = diff_dict["n_trials"]
    colors = diff_dict["colors"]
    metric = diff_dict["metric"]
    inset_dim = diff_dict["inset_dim"]
    model_dirs = diff_dict["model_dirs"]
    
    save_models = "/raid/home/tubitoal/DMM/saved_model/"
    data = []

    for i in range(len(model_dirs)):
        
        saved_path = save_models+model_dirs[i]
    
        dir_filename = "/dir_dict_z"
        checkpoint_dir = torch.load(saved_path + dir_filename, weights_only=False, map_location='cpu')
        accuracy_dir = checkpoint_dir["accuracy"]

        dir_filename = "/Non_Markov_dict_z"
        checkpoint_NonMarkov = torch.load(saved_path + dir_filename, weights_only=False, map_location='cpu')

        dir_filename = "/Markov_dict_z"
        checkpoint_Markov = torch.load(saved_path + dir_filename, weights_only=False, map_location='cpu')
        
#         dir_filename = "/move_dict_MLP_z"
#         checkpoint_move = torch.load(saved_path + dir_filename, weights_only=False)
#         accuracy_move = checkpoint_move["accuracy"]
        
        if metric=="mean_NMSE":
            NMSE_NonMarkov = checkpoint_NonMarkov["mean_NMSE"]
            NMSE_Markov = checkpoint_Markov["mean_NMSE"]
        elif metric=="NMSE_mean":
            NMSE_NonMarkov = checkpoint_NonMarkov["NMSE_mean"]
            NMSE_Markov = checkpoint_Markov["NMSE_mean"]
        elif metric=="MSE_pc":
            NMSE_NonMarkov = checkpoint_NonMarkov["MSE_pc"]
            NMSE_Markov = checkpoint_Markov["MSE_pc"]
        
        dir_filename = "/MSE"
        checkpoint_MSE = torch.load(saved_path + dir_filename, weights_only=False, map_location='cpu')
        MSE_rec = checkpoint_MSE["MSE_DMM"]
    
#         dir_odds = accuracy_dir/(1-accuracy_dir)  # direction
        r2_NonMarkov = 1 - NMSE_NonMarkov
        r2_rec = 1 - MSE_rec
        
        data.append([r2_rec, r2_NonMarkov, accuracy_dir])#, accuracy_move])
 
    data = np.array(data).T
    
    y_label = ["$R^2$", "$R^2_D$", "Accuracy"]#, "Accuracy"]
    plot_label = ["rec", "prediction", "direction"]#, "movement"]
#     letter = ["A", "B", "C", "D"]
    for i in range(len(y_label)):
        f, ax = plt.subplots()
        # x positions delle due colonne
        ax.bar(x, data[i], color=colors, width=width)
        ax.set_xlim((0, 1))
        ax.set_xticks(x)
        ax.set_xticklabels(["D=2", "D=3", "D=4"])#, fontsize=font_tick)
        #ax.set_title(f"Confronto {i+1}")
        ax.set_ylabel(y_label[i])#, fontsize=font_ax)
        ax.set_ylim(y_lim[i])
        ax.set_yticks(y_ticks[i])
        ax.set_yticklabels(y_ticks[i])#, fontsize=font_tick)
        
        # mostro anche il numero sopra la barra
        for xi, yi in zip(x, data[i]):
            if i==2:# or i==3:
                delta_y = 0.01
            else: 
                delta_y = 0.02
            ax.text(xi, yi + (delta_y*y_ticks[i][1]), f"{yi:.2f}", ha='center')
        
        fig_file = os.path.join(comm_dict["saved_path"], f'DMM_vs_dim_{plot_label[i]}.png')
        plt.savefig(fig_file)
        plt.show()

    


def PCA_vs_DMM(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    c_dim = comm_dict["c_dim"]
    tau = comm_dict["tau"]
    t = comm_dict["t"]
    saved_path = comm_dict["saved_path"]

    width = diff_dict["width"]
    x = diff_dict["x"]
    y_lim = diff_dict["y_lim"]
    y_ticks = diff_dict["y_ticks"]
    n_trials = diff_dict["n_trials"]
    color_DMM = diff_dict["color_DMM"]
    color_PCA = diff_dict["color_PCA"]
    inset_dim = diff_dict["inset_dim"]
    inset_font = diff_dict["inset_font"]
    metric = diff_dict["metric"]
    
    dir_filename = "/dir_dict_z"
    checkpoint_dir_DMM = torch.load(comm_dict["saved_path"] + dir_filename, weights_only=False, map_location='cpu')
    accuracy_dir_DMM = checkpoint_dir_DMM["accuracy"]
    
    dir_filename = "/dir_dict_PCA"
    checkpoint_dir_PCA = torch.load(comm_dict["saved_path"] + dir_filename, weights_only=False, map_location='cpu')
    accuracy_dir_PCA = checkpoint_dir_PCA["accuracy"]
    
    dir_filename = "/Non_Markov_dict_z"
    checkpoint_NonMarkov_DMM = torch.load(comm_dict["saved_path"] + dir_filename, weights_only=False, map_location='cpu')

    dir_filename = "/Non_Markov_dict_PCA"
    checkpoint_NonMarkov_PCA = torch.load(comm_dict["saved_path"] + dir_filename, weights_only=False, map_location='cpu')
 
    dir_filename = "/Markov_dict_z"
    checkpoint_Markov_DMM = torch.load(comm_dict["saved_path"] + dir_filename, weights_only=False, map_location='cpu')
    
    dir_filename = "/Markov_dict_PCA"
    checkpoint_Markov_PCA = torch.load(comm_dict["saved_path"] + dir_filename, weights_only=False, map_location='cpu')
    
#     if metric=="mean_NMSE":
#         NMSE_NonMarkov_PCA = checkpoint_NonMarkov_PCA["mean_NMSE"]
#         NMSE_Markov_PCA = checkpoint_Markov_PCA["mean_NMSE"]
#         NMSE_NonMarkov_DMM = checkpoint_NonMarkov_DMM["mean_NMSE"]
#         NMSE_Markov_DMM = checkpoint_Markov_DMM["mean_NMSE"]
#     elif metric=="NMSE_mean":
#         NMSE_NonMarkov_PCA = checkpoint_NonMarkov_PCA["NMSE_mean"]
#         NMSE_Markov_PCA = checkpoint_Markov_PCA["NMSE_mean"]
#         NMSE_NonMarkov_DMM = checkpoint_NonMarkov_DMM["NMSE_mean"]
#         NMSE_Markov_DMM = checkpoint_Markov_DMM["NMSE_mean"]
    if metric=="MSE_pc":
        NMSE_NonMarkov_PCA = checkpoint_NonMarkov_PCA["MSE_pc"]
        NMSE_Markov_PCA = checkpoint_Markov_PCA["MSE_pc"]
        NMSE_NonMarkov_DMM = checkpoint_NonMarkov_DMM["MSE_pc"]
        NMSE_Markov_DMM = checkpoint_Markov_DMM["MSE_pc"]
        
#     dir_filename = "/move_dict_MLP_z"
#     checkpoint_move_DMM = torch.load(saved_path + dir_filename, weights_only=False)
#     accuracy_move_DMM = checkpoint_move_DMM["accuracy"]
    
#     dir_filename = "/move_dict_MLP_PCA"
#     checkpoint_move_PCA = torch.load(saved_path + dir_filename, weights_only=False)
#     accuracy_move_PCA = checkpoint_move_PCA["accuracy"]
    
    dir_filename = "/MSE"
    checkpoint_MSE = torch.load(saved_path + dir_filename, weights_only=False, map_location='cpu')
    MSE_rec_DMM = checkpoint_MSE["MSE_DMM"]
    MSE_rec_PCA = checkpoint_MSE["MSE_PCA"]
    
    r2_Markov_PCA = 1-NMSE_Markov_PCA
    r2_Markov_DMM = 1-NMSE_Markov_DMM
    
    r2_NonMarkov_PCA = 1-NMSE_NonMarkov_PCA
    r2_NonMarkov_DMM = 1-NMSE_NonMarkov_DMM
    
#     dir_odds_DMM = accuracy_dir_DMM/(1-accuracy_dir_DMM)
#     dir_odds_PCA = accuracy_dir_PCA/(1-accuracy_dir_PCA)
    
    r2_rec_PCA = 1-MSE_rec_PCA
    r2_rec_DMM = 1-MSE_rec_DMM
    
    print(f"r2_PCA: {r2_rec_PCA:.4f}")
    print(f"r2_DMM: {r2_rec_DMM:.4f}")
    
    data = [
#         (Markov_coeff_DMM, Markov_coeff_PCA),
#         (r2_Markov_DMM, r2_Markov_PCA),
        (r2_rec_DMM, r2_rec_PCA), 
        (r2_NonMarkov_DMM, r2_NonMarkov_PCA),
        (accuracy_dir_DMM, accuracy_dir_PCA),
        #(accuracy_move_DMM, accuracy_move_PCA),
    ]
    
    y_label = ["$R^2$", "$R^2_D$", "Accuracy"]#, "Accuracy"]
    plot_label = ["rec", "prediction", "direction"]#, "movement"]
#     letter = ["A", "B", "C", "D"]
    color = (color_DMM, color_PCA)

    for i in range(len(y_label)):
        f, ax = plt.subplots()
        ax.bar(x, data[i], color=color, width=width)
        ax.set_xlim((0, 1))
        ax.set_xticks(x)
        ax.set_xticklabels(["DMM", "PCA"])
        #ax.set_title(f"Confronto {i+1}")
        ax.set_ylabel(y_label[i])
        ax.set_ylim(y_lim[i])
        ax.set_yticks(y_ticks[i])
        ax.set_yticklabels(y_ticks[i])

        # mostro anche il numero sopra la barra
        for xi, yi in zip(x, data[i]):
            if i==2:
                delta_y = 0.01
            else: 
                delta_y = 0.02
            ax.text(xi, yi + (delta_y*y_ticks[i][1]), f"{yi:.2f}", ha='center')#, fontsize=font_tick)
            
        if i==1:
            # --- Inset axes inside ax ---
            axins = ax.inset_axes(inset_dim)  # [x0, y0, width, height] in Axes fraction [web:17]

            axins.bar(x, (r2_Markov_DMM, r2_Markov_PCA), color=color, width=width)
            axins.set_xlim((0, 1))
            axins.set_xticks(x)
            axins.set_xticklabels(["DMM", "PCA"])#, fontsize=font_tick)
            #ax.set_title(f"Confronto {i+1}")
            axins.set_ylabel("$R^2_D$", fontsize=inset_font)
            axins.set_ylim((0, 1.2))
            axins.set_yticks((0, 1))
            axins.set_yticklabels((0, 1))#, fontsize=font_tick)
            axins.tick_params(labelsize=inset_font)
            
            # mostro anche il numero sopra la barra
            for xi, yi in zip(x, (r2_Markov_DMM, r2_Markov_PCA)):
                axins.text(xi, yi + (0.05*y_ticks[i][1]), f"{yi:.2f}", ha='center', fontsize=inset_font)

        
        fig_file = os.path.join(comm_dict["saved_path"], f'DMM_vs_PCA_{plot_label[i]}.png')
        plt.savefig(fig_file)
        plt.show()
        

def PCA_vs_VAE(comm_dict, diff_dict):
    
    with np.load(comm_dict["saved_path"]+"/data_split.npz", allow_pickle=True) as loaded_file:
            train_set = loaded_file["train_set"]
            vali_set = loaded_file["vali_set"]
            test_set = loaded_file["test_set"]
            train_direction = loaded_file["train_direction"]
            vali_direction = loaded_file["vali_direction"]
            test_direction = loaded_file["test_direction"]
            train_SSD = loaded_file["train_SSD"]
            vali_SSD = loaded_file["vali_SSD"]
            test_SSD = loaded_file["test_SSD"]
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    c_dim = comm_dict["c_dim"]
    tau = comm_dict["tau"]
    t = comm_dict["t"]

    n_trials = diff_dict["n_trials"]
    GO_flag = diff_dict["GO_flag"]
    bins = diff_dict["bins"]
    density = diff_dict["density"]
    data = diff_dict["data"]
    mean = diff_dict["mean"]
    only_cn = diff_dict["only_cn"]
    
    set_cn = data["set_cn_ordRT"]
    dir_cn = data["dir_cn_ordRT"]
    n_train, steps, features = train_set.shape
    n_vali = vali_set.shape[0]
    
    print(test_set.shape)
    cont_test = one_hot_cont(test_SSD, test_direction, tau)

    ############## PCA ###############
    X_train = train_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
    X_vali = vali_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
    if only_cn:
        test_set = set_cn
        cont_test = one_hot_cont(np.zeros(test_set.shape[0]), dir_cn, tau)
        
    n_test = test_set.shape[0]
    X_test = test_set.reshape(-1, features)  # shape = (trials * time_steps, 96)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=z_dim)
    X_train_pca = pca.fit_transform(X_train)  # fit + transform sul train
    #X_vali_pca = pca.transform(X_vali) 
    X_test_pca = pca.transform(X_test)   
    X_test_rec = pca.inverse_transform(X_test_pca)

    #train_data = X_train_pca.reshape(n_train, steps, self.input_dim)
    #vali_data = X_vali_pca.reshape(n_vali, steps, self.input_dim)
    test_rec = X_test_rec.reshape(n_test, steps, features)
    
    ############## DMM ##############
    #cont_test = one_hot_cont(test_SSD, test_direction, tau)
    test_data = torch.from_numpy(test_set).float().to(device).permute(1, 0, 2)
    cont_test = torch.from_numpy(cont_test).float().to(device).permute(1, 0, 2)
    
    z_test, _, _ = dmm.inference(test_data, cont_test)
    z_test = z_test.cpu().detach().numpy()
    print(f"Variance of DMM latents: {z_test.var()}, mean: {z_test.mean()}, std: {z_test.std()}")
    print(f"Variance of PCA latents: {X_test_pca.var()}, mean: {X_test_pca.mean()}, std: {X_test_pca.std()}")
    
    DMM_var = z_test.var()
    PCA_var = X_test_pca.var()
    
    chunk_size = 10
    
    # Output finale
    y_accum = np.zeros((steps, n_test, features), dtype=np.float32)

    for start in range(0, n_test, chunk_size):
        end = min(start + chunk_size, n_test)
        batch_size = end - start

        # Estrai chunk
        x_chunk = test_data[:, start:end, :].repeat_interleave(n_trials, dim=1).to(device)
        c_chunk = cont_test[:, start:end, :].repeat_interleave(n_trials, dim=1).to(device)

        # Inferenza
        with torch.no_grad():
            y_mean, y_logvar = dmm(x_chunk, c_chunk)
            if mean:
                y_pred = y_mean
            else:
                y_pred = dmm.reparameterization(y_mean, y_logvar)

        # Media sui trials (dim=2)
        y_pred = y_pred.cpu().numpy().reshape(steps, batch_size, n_trials, features).mean(2)

        # Inserisci nel buffer
        y_accum[:, start:end, :] = y_pred

        torch.cuda.empty_cache()
    y_inf = np.transpose(y_accum, (1, 0, 2))
    
    # reconstruction        
#     y_mean, y_logvar = dmm(test_data, cont_test)
#     y_inf = dmm.reparameterization(y_mean, y_logvar)
#     if n_trials>1:
#         y_inf = y_inf.reshape(steps, n_test, n_trials, features).mean(2)
#     y_inf = y_inf.permute(1, 0, 2).cpu().detach().numpy()
    
    residual_VAE = test_set - y_inf
    residual_PCA = test_set - test_rec
    
    var_residual_vae = np.var(residual_VAE)
    var_residual_pca = np.var(residual_PCA)
    var_total = np.var(test_set)
    fve_vae = 1 - (var_residual_vae / var_total)
    fve_pca = 1 - (var_residual_pca / var_total)
    
    print(f"Fraction of Variance Explained (FVE) by VAE: {fve_pca}")
    print(f"Fraction of Variance Explained (FVE) by PCA: {fve_pca}")
    
    MSE_vae = np.sqrt(residual_VAE**2).mean(0).mean(1)
    MSE_pca = np.sqrt(residual_PCA**2).mean(0).mean(1)
    
    print(f"MSE for PCA over the entire dataset: {MSE_pca.mean(0)}")
    print(f"MSE for VAE over the entire dataset: {MSE_vae.mean(0)}")
    
    if GO_flag:
        residual_VAE = residual_VAE[:, 56//tau:]
        residual_PCA = residual_PCA[:, 56//tau:]
        
    residuals_vae_mtime = residual_VAE.mean(1)
    residuals_pca_mtime = residual_PCA.mean(1)
    
    residuals_vae_mtrials = residual_VAE.mean(0)
    residuals_pca_mtrials = residual_PCA.mean(0)
    
    # Flatten su trials e time (ogni riga = istante, colonna = canale)
    residuals_vae = residual_VAE.reshape(-1, features)  # [(N*T), F]
    residuals_pca = residual_PCA.reshape(-1, features)  # [(N*T), F]
    
    pca_vae = PCA(n_components=1)
    pca_vae.fit(residuals_vae)
    frac_common_vae = np.sum(pca_vae.explained_variance_ratio_)
    print(f"Frazione di varianza residua (della VAE) spiegata da un drive comune: {frac_common_vae:.2%}")
    
    pca_pca = PCA(n_components=1)
    pca_pca.fit(residuals_pca)
    frac_common_pca = np.sum(pca_pca.explained_variance_ratio_)
    print(f"Frazione di varianza residua (della PCA) spiegata da un drive comune: {frac_common_pca:.2%}")
    
    # Calcolo matrice di correlazione tra canali
    corr_matrix_vae_mtime = np.corrcoef(residuals_vae_mtime, rowvar=False)  # [F, F]
    corr_matrix_pca_mtime = np.corrcoef(residuals_pca_mtime, rowvar=False)  # [F, F]
    
    # Calcolo matrice di correlazione tra canali
    corr_matrix_vae_mtrials = np.corrcoef(residuals_vae_mtrials, rowvar=False)  # [F, F]
    corr_matrix_pca_mtrials = np.corrcoef(residuals_pca_mtrials, rowvar=False)  # [F, F]
    
    # Calcolo matrice di correlazione tra canali
    corr_matrix_vae = np.corrcoef(residuals_vae, rowvar=False)  # [F, F]
    corr_matrix_pca = np.corrcoef(residuals_pca, rowvar=False)  # [F, F]
    
    # Indici della triangolare superiore senza diagonale
    iu = np.triu_indices(features, k=1)
    res_vae = np.abs(corr_matrix_vae[iu])
    res_pca = np.abs(corr_matrix_pca[iu])
    res_vae_mtime = np.abs(corr_matrix_vae_mtime[iu])
    res_pca_mtime = np.abs(corr_matrix_pca_mtime[iu])
    res_vae_mtrials = np.abs(corr_matrix_vae_mtrials[iu])
    res_pca_mtrials = np.abs(corr_matrix_pca_mtrials[iu])
    
    
    print(f"Rapporto correlazione media tra residui VAE e PCA: {res_vae.mean()/res_pca.mean()}")
    print(f"Rapporto correlazione media tra residui (mediati sul tempo) VAE e PCA: {res_vae_mtime.mean()/res_pca_mtime.mean()}")
    print(f"Rapporto correlazione media tra residui (mediati sui trials) VAE e PCA: {res_vae_mtrials.mean()/res_pca_mtrials.mean()}")
    #print(f"Rapporto correlazione tra VAE e PCA in media: {frac_res.mean()}")

    # Plot istogramma
    plt.figure()
    plt.hist(res_pca, bins=bins, density=density, edgecolor="black", color = "red", alpha=0.75, label = "PCA")
    plt.hist(res_vae, bins=bins, density=density, edgecolor="black", color = "skyblue", alpha=0.75, label = "VAE")
    plt.xlabel("Correlazione")
    plt.ylabel("Frequenza" if not density else "Densità")
    plt.title("Istogramma dei valori off-diagonals della matrice di correlazione dei residui tra elettrodi")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    import seaborn as sns
    
    fig, ax = plt.subplots(1, 3, figsize=(17, 6))

    sns.heatmap(
        corr_matrix_vae,
        cmap="Greens",
        square=True,
        cbar_kws={"label": "Residual correlation"},
        ax=ax[0]
    )
    ax[0].set_title("Correlazioni tra residui (co-varying residuals) per VAE")

    sns.heatmap(
        corr_matrix_pca,
        cmap="Greens",
        square=True,
        cbar_kws={"label": "Residual correlation"},
        ax=ax[1]
    )
    ax[1].set_title("Correlazioni tra residui (co-varying residuals) per PCA")
    
    ax[2].hist(res_pca, bins=bins, density=density, edgecolor="black", color = "red", alpha=0.75, label = "PCA")
    ax[2].hist(res_vae, bins=bins, density=density, edgecolor="black", color = "skyblue", alpha=0.75, label = "VAE")
    ax[2].set_xlabel("Correlazione")
    ax[2].set_ylabel("Frequenza" if not density else "Densità")
    ax[2].set_title("Istogramma dei valori off-diagonals della matrice di correlazione dei residui tra elettrodi")
    ax[2].grid(True, linestyle="--", alpha=0.3)
    ax[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # mean on time
    
    fig, ax = plt.subplots(1, 3, figsize=(17, 6))

    sns.heatmap(
        corr_matrix_vae_mtime,
        cmap="Greens",
        square=True,
        cbar_kws={"label": "Residual correlation"},
        ax=ax[0]
    )
    ax[0].set_title("Correlazioni tra residui (mediati sul tempo) per VAE")

    sns.heatmap(
        corr_matrix_pca_mtime,
        cmap="Greens",
        square=True,
        cbar_kws={"label": "Residual correlation"},
        ax=ax[1]
    )
    ax[1].set_title("Correlazioni tra residui (mediati sul tempo) per PCA")
    
    ax[2].hist(res_pca_mtime, bins=bins, density=density, edgecolor="black", color = "red", alpha=0.75, label = "PCA")
    ax[2].hist(res_vae_mtime, bins=bins, density=density, edgecolor="black", color = "skyblue", alpha=0.75, label = "VAE")
    ax[2].set_xlabel("Correlazione")
    ax[2].set_ylabel("Frequenza" if not density else "Densità")
    ax[2].set_title("Istogramma dei valori off-diagonals della matrice di correlazione dei residui tra elettrodi")
    ax[2].grid(True, linestyle="--", alpha=0.3)
    ax[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # mean on trials
    
    fig, ax = plt.subplots(1, 3, figsize=(17, 6))

    sns.heatmap(
        corr_matrix_vae_mtrials,
        cmap="Greens",
        square=True,
        cbar_kws={"label": "Residual correlation"},
        ax=ax[0]
    )
    ax[0].set_title("Correlazioni tra residui (mediati sui trials) per VAE")

    sns.heatmap(
        corr_matrix_pca_mtrials,
        cmap="Greens",
        square=True,
        cbar_kws={"label": "Residual correlation"},
        ax=ax[1]
    )
    ax[1].set_title("Correlazioni tra residui (mediati sui trials) per PCA")
    
    ax[2].hist(res_pca_mtrials, bins=bins, density=density, edgecolor="black", color = "red", alpha=0.75, label = "PCA")
    ax[2].hist(res_vae_mtrials, bins=bins, density=density, edgecolor="black", color = "skyblue", alpha=0.75, label = "VAE")
    ax[2].set_xlabel("Correlazione")
    ax[2].set_ylabel("Frequenza" if not density else "Densità")
    ax[2].set_title("Istogramma dei valori off-diagonals della matrice di correlazione dei residui tra elettrodi")
    ax[2].grid(True, linestyle="--", alpha=0.3)
    ax[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    #####

    
    f, ax = plt.subplots(figsize = (8, 6))
    
    ax.plot(t*5, MSE_vae, c ="r", label = 'VAE')
    ax.plot(t*5, MSE_pca, c ="b", label = 'PCA')
    ax.axvline(56*5, color="black",linestyle="--",label= "GO")
    ax.set_title(f"MSE over time, between Observed and Reconstructed MUA using VAE (red) and PCA (blue)")
    ax.set_xlabel('time ($ms$)')
    ax.set_ylabel('MSE')
    ax.legend(fontsize=7.3)
    
    ex_var = np.empty(96)
    fve_pca = np.empty(96)
    for n in range(1, 96+1):
        pca = PCA(n_components=n)
        X_train_pca = pca.fit_transform(X_train)  # fit + transform sul train
        ex_var[n-1] = np.sum(pca.explained_variance_)
        X_test_pca = pca.transform(X_test)   
        X_test_rec = pca.inverse_transform(X_test_pca)
        test_rec = X_test_rec.reshape(n_test, steps, features)
        residual_PCA = test_set - test_rec
        var_residual_pca = np.var(residual_PCA)
        fve_pca[n-1] = 1 - (var_residual_pca / var_total)  
    
    f, ax = plt.subplots(figsize = (8, 6))
    
    ax.plot(np.arange(1, 97), ex_var, c ="b", label = 'fve on train_set')
    ax.plot(np.arange(1, 97), fve_pca*100, c ="r", label = 'fve on test_set')
    ax.set_title(f"Percentage of Variance Explained over n_components for train and test set")
    ax.set_xlabel('n_components')
    ax.set_ylabel('Percentage of Variance Explained')
    ax.legend(fontsize=7.3)
        
    #return MUA_trials, MUA_pred, MUA_inf

    
    
    
def check_peaks(comm_dict, diff_dict):
    
    with np.load(comm_dict["saved_path"]+"/data_split.npz", allow_pickle=True) as loaded_file:
            train_set = loaded_file["train_set"]
            vali_set = loaded_file["vali_set"]
            test_set = loaded_file["test_set"]
            train_direction = loaded_file["train_direction"]
            vali_direction = loaded_file["vali_direction"]
            test_direction = loaded_file["test_direction"]
            train_SSD = loaded_file["train_SSD"]
            vali_SSD = loaded_file["vali_SSD"]
            test_SSD = loaded_file["test_SSD"]
    
    # Supponiamo di voler confrontare triplette
    channels = [88, 90, 92]
    
    random_channels = [20, 40, 60]
    X_random = train_set[:, :, random_channels]
    X_train = train_set[:, :, channels]  # shape (N, T, 3)
    # Varianza tra canali nel tempo e nei trial
    var_across_channels_train = X_train.var(axis=2)  # shape (N, T)
    var_random = X_random.var(axis=2)
    mean_var_train = var_across_channels_train.mean()
    mean_var_random = var_random.mean()
    print(f"Varianza media tra i 3 canali strani sul training set: {mean_var_train:.6e}")
    print(f"Varianza media tra 3 canali random sul training set: {mean_var_random:.6e}")
    
    X_test = test_set[:, :, channels]  # shape (N, T, 3)
    X_random = test_set[:, :, random_channels]
    # Varianza tra canali nel tempo e nei trial
    var_across_channels_test = X_test.var(axis=2)  # shape (N, T)
    var_random = X_random.var(axis=2)
    mean_var_test = var_across_channels_test.mean()
    mean_var_random = var_random.mean()
    print(f"Varianza media tra i 3 canali strani sul test set: {mean_var_test:.6e}")
    print(f"Varianza media tra 3 canali random sul test set: {mean_var_random:.6e}")
    
#     trial = train_set[q]
#     MUA = trial.mean(1)
    
#     channels_88 = trial[:, 88]
#     channels_90 = trial[:, 90]
#     channels_92 = trial[:, 92]
    
#     rand_channels = trial[:, ]
    
#     f, ax = plt.subplots(figsize = (7, 6))
    
#     ax.plot(t*5, MUA, c ="blue", label = 'mean MUA')
#     ax.set_title(f"Mean MUA for trial n.{q}")
#     ax.set_xlabel('time ($ms$)')
#     ax.set_ylabel('mean MUA activity over channels')
#     ax.legend(fontsize=7.3)
    
#     trial_ch = channel2grid(trial)
    
#     row = 10
#     col = 10
    
#     fig, ax = plt.subplots(row, col, figsize = (15, 15))
#     for i in range(row):
#         for j in range(col):
#             ax[i, j].plot(t*5, trial_ch[:, i, j])
#             ax[i, j].axvline(56*5, color="black",linestyle="--",label="GO")
#             ax[i, j].set_ylim((-2, 2))
#             ax[i, j].set_xticks([])
#             ax[i, j].set_yticks([])
#             #ax[i, j].set_xlabel("time ($ms$)")
#             #ax[i, j].set_ylabel("Correlation between reconstructed and observed activity")
#             ax[i, j].set_title(f"activity plot for channel: {i}, {j}")
#             ax[i, j].legend()



def MUA_pred_inf(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    t = comm_dict["t"]
    RT_detector = comm_dict["RT_detector"]
    move_detector = comm_dict["move_detector"]
    ar = comm_dict["ar"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    sim_start = diff_dict["sim_start"]
    RT_array = diff_dict["RT_array"]
    color = diff_dict["color"]
    
    test_set = data["set_cn_ordRT"]
    test_cont = data["cont_cn_ordRT"]
    test_RT = data["RT_cn_ordRT"]
    
    l = len(RT_array)
    steps = test_set.shape[1]
    teacher = sim_start//(5*tau)
    alone = steps - teacher
    MUA_trials = np.zeros((l, steps, 96))
    MUA_pred = np.zeros((l, steps, n_trials, 96))
    MUA_inf = np.zeros((l, steps, n_trials, 96))
    
    for i, RT in enumerate(RT_array):       
        trial = test_set[test_RT==RT]
        cont_c = test_cont[test_RT==RT]
        if trial.shape[0] == 0:
            print(f"change the trial with RT={RT:d}")
        elif trial.shape[0] != steps:
            trial = trial[0]
            cont_c = cont_c[0]
        else:
            print()
            trial = trial.squeeze(0)
            cont_c = cont_c.squeeze(0)

        MUA_trials[i] = trial
        
        trial = torch.from_numpy(trial).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
        cont_c = torch.from_numpy(cont_c).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)

        # reconstruction        
        y_mean, y_logvar = dmm(trial, cont_c)
        y_inf = dmm.reparameterization(y_mean, y_logvar)
        y_inf = y_inf.cpu().detach().numpy()

        MUA_inf[i] = y_inf

        # generation
        z, z_mean, _ = dmm.inference(trial, cont_c)
        z_teach = z[:teacher]

        for step in range(alone):
            z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher+step].unsqueeze(0))
            z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
            z_teach = torch.cat((z_teach, z_gen), dim=0)
            
        RT_pred_list = np.zeros((l, steps))
        RT_inf_list = np.zeros((l, steps))
        
        z_pred = z_teach.mean(1)
        z_inf = z.mean(1)
        
        move_logit = move_detector(z_pred.unsqueeze(0))
        move_output = binary_output(move_logit)
        move_pred = move_output.squeeze(0).astype(int)

        if move_pred:
            RT_output = RT_detector(z_pred.unsqueeze(0))
            RT_pred = prob_to_RT(RT_output, tau)
            RT_pred = RT_pred.squeeze(0)
            
        move_logit = move_detector(z_inf.unsqueeze(0))
        move_output = binary_output(move_logit)
        move_inf = move_output.squeeze(0).astype(int)

        if move_pred:
            RT_output = RT_detector(z_inf.unsqueeze(0))
            RT_inf = prob_to_RT(RT_output, tau)
            RT_inf = RT_inf.squeeze(0)
    
        RT_pred_list[i] = RT_pred
        RT_inf_list[i] = RT_inf     
        
        if ar:
            y_mean, y_logvar = dmm.generation_x(z_teach, trial)
        else:
            y_mean, y_logvar = dmm.generation_x(z_teach)
        y_pred = dmm.reparameterization(y_mean, y_logvar)
        y_pred = y_pred.cpu().detach().numpy()

        MUA_pred[i] = y_pred
        
    return MUA_trials, MUA_pred, MUA_inf#, RT_pred_list, RT_inf_list


def MUA_pred_inf_plot(comm_dict, diff_dict):
     
    tau = comm_dict["tau"]
    t = comm_dict["t"]
    
    sim_start = diff_dict["sim_start"]
    RT_array = diff_dict["RT_array"]
    MUA_trials = diff_dict["MUA_trials"]
    MUA_pred = diff_dict["MUA_pred"]
    MUA_inf = diff_dict["MUA_inf"]
#     RT_pred_list = diff_dict["RT_pred_list"]
#     RT_inf_list = diff_dict["RT_inf_list"]
    color = diff_dict["color"]
    
    teacher = sim_start//(5*tau)
    l = len(RT_array)
    
    MUA_pred_mean = np.mean(MUA_pred, axis=3)
    MUA_inf_mean = np.mean(MUA_inf, axis=3)
    MUA_pred_trials = np.mean(MUA_pred_mean, axis=2)
    MUA_inf_trials = np.mean(MUA_inf_mean, axis=2)
    MUA_pred_std = np.std(MUA_pred_mean, axis=2)
    MUA_inf_std = np.std(MUA_inf_mean, axis=2)
    MUA_trials = np.mean(MUA_trials, axis=2)

    
    f, (ax1, ax2) = plt.subplots(2, 1, figsize = (9, 12))
    
    ax1.plot(t*5, MUA_trials[0], c =color[0], label = 'Low_RT true')
    ax1.fill_between(t*5, MUA_pred_trials[0] - MUA_pred_std[0], MUA_pred_trials[0] + MUA_pred_std[0], edgecolor = 'none', color = 'grey', alpha = 0.3)
    ax1.plot(t*5, MUA_trials[1], c =color[1], label = 'Mid_RT true')
    ax1.fill_between(t*5, MUA_pred_trials[1] - MUA_pred_std[1], MUA_pred_trials[1] + MUA_pred_std[1], edgecolor = 'none', color = 'grey', alpha = 0.3)
    ax1.plot(t*5, MUA_trials[2], c =color[2], label = 'High_RT true')
    ax1.fill_between(t*5, MUA_pred_trials[2] - MUA_pred_std[2], MUA_pred_trials[2] + MUA_pred_std[2], edgecolor = 'none', color = 'grey', alpha = 0.3)
    ax1.plot(t*5, MUA_pred_trials[0], c =color[0],linestyle="--", label = 'Low_RT predicted')
    ax1.plot(t*5, MUA_pred_trials[1], c =color[1],linestyle="--", label = 'Mid_RT predicted')
    ax1.plot(t*5, MUA_pred_trials[2], c =color[2],linestyle="--", label = 'High_RT predicted')
    ax1.axvline(sim_start, color="black",linestyle="--",label= "Simulation start")
    if sim_start==0:
        ax1.set_title(f"Mean MUA reconstruction from prediction from GO, red $RT$ = {RT_array[0]*5:d}$ms$, green $RT$ = {RT_array[1]*5:d}$ms$, blue $RT$ = {RT_array[2]*5:d}$ms$")
    else:
        ax1.set_title(f"Mean MUA reconstruction from prediction from {sim_start:d}ms, red $RT$ = {RT_array[0]*5:d}$ms$, green $RT$ = {RT_array[1]*5:d}$ms$, blue $RT$ = {RT_array[2]*5:d}$ms$")
    ax1.set_xlabel('time ($ms$)')
    ax1.set_ylabel('mean MUA activity over channels')
    ax1.legend(fontsize=7.3)
    
    
    ax2.plot(t*5, MUA_trials[0], c =color[0], label = 'Low_RT true')
    ax2.fill_between(t*5, MUA_inf_trials[0] - MUA_inf_std[0], MUA_inf_trials[0] + MUA_inf_std[0], edgecolor = 'none', color = 'grey', alpha = 0.3)
    ax2.plot(t*5, MUA_trials[1], c =color[1], label = 'Mid_RT true')
    ax2.fill_between(t*5, MUA_inf_trials[1] - MUA_inf_std[1], MUA_inf_trials[1] + MUA_inf_std[1], edgecolor = 'none', color = 'grey', alpha = 0.3)
    ax2.plot(t*5, MUA_trials[2], c =color[2], label = 'High_RT true')
    ax2.fill_between(t*5, MUA_inf_trials[2] - MUA_inf_std[2], MUA_inf_trials[2] + MUA_inf_std[2], edgecolor = 'none', color = 'grey', alpha = 0.3)
    ax2.plot(t*5, MUA_inf_trials[0], c =color[0],linestyle="--", label = "Low_RT inferred")
    ax2.plot(t*5, MUA_inf_trials[1], c =color[1],linestyle="--", label = "Mid_RT inferred")
    ax2.plot(t*5, MUA_inf_trials[2], c =color[2],linestyle="--", label = "High_RT inferred")
    #ax.axvline(teacher*tau*5, color="black",linestyle="--",label=" Simulation start (GO)")
    ax2.set_title(f"mean MUA reconstruction from inference, red $RT$ = {RT_array[0]*5:d}$ms$, green $RT$ = {RT_array[1]*5:d}$ms$, blue $RT$ = {RT_array[2]*5:d}$ms$")
    ax2.set_xlabel('time ($ms$)')
    ax2.set_ylabel('mean MUA activity over channels')
    ax2.legend(fontsize=7)
    
    
def MUA_pred_inf_corr_ch(comm_dict, diff_dict):
    
    tau = comm_dict["tau"]
    t = comm_dict["t"]
    
    sim_start = diff_dict["sim_start"]
    RT_array = diff_dict["RT_array"]
    MUA_trials = diff_dict["MUA_trials"]
    MUA_pred = diff_dict["MUA_pred"]
    MUA_inf = diff_dict["MUA_inf"]
    mean_corr = diff_dict["mean_corr"]
    
    l = len(RT_array)
    teacher = sim_start//(5*tau)
    steps = MUA_trials.shape[1]
    
    MUA_pred_mean = MUA_pred.mean(2)
    MUA_inf_mean = MUA_inf.mean(2)
    
    correlations_pred = np.zeros((l, steps))
    corr_stds_pred = np.zeros((l, steps))
    for i in range(l):
        for time in range(steps):
            if mean_corr:
                corr_pred = np.array([np.corrcoef(MUA_trials[i, time], MUA_pred[i, time, j])[0, 1] for j in range(MUA_pred.shape[2])])  
                correlations_pred[i, time] = np.mean(corr_pred)
                corr_stds_pred[i, time] = np.std(corr_pred)
            else:
                correlations_pred[i, time] = np.corrcoef(MUA_trials[i, time], MUA_pred_mean[i, time])[0, 1]
                corr_stds_pred[i, time] = None
    
    
    correlations_inf = np.zeros((l, steps))
    corr_stds_inf = np.zeros((l, steps))
    for i in range(l):
        for time in range(steps):
            if mean_corr:
                corr_inf = np.array([np.corrcoef(MUA_trials[i, time], MUA_inf[i, time, j])[0, 1] for j in range(MUA_inf.shape[2])])  
                correlations_inf[i, time] = np.mean(corr_inf)
                corr_stds_inf[i, time] = np.std(corr_inf)
            else:
                correlations_inf[i, time] = np.corrcoef(MUA_trials[i, time], MUA_inf_mean[i, time])[0, 1]
                corr_stds_inf[i, time] = None

    fig, ax = plt.subplots(1, 3, figsize = (18, 5))
    fig.text(0.16, 0.96, f'Correlations between true and predicted ({sim_start}ms after start) channels over time', rotation=0, size=16, fontweight='bold')
    for i in range(l):
        ax[i].errorbar(t*5, correlations_pred[i], yerr=corr_stds_pred[i], fmt='o', ls='--', ecolor='black', 
                         elinewidth=1, capsize=5, capthick=1)
        ax[i].scatter(t*5, correlations_pred[i], label = "inter channel correlation")
        ax[i].axvline(56*5, color="black",linestyle="--",label="GO")
        ax[i].set_ylim((0, 1))
        ax[i].set_xlabel("time ($ms$)")
        ax[i].set_ylabel(f"Correlation")
        ax[i].set_title(f"RT trial = {(RT_array[i]+56)*5}$ms$")

    
    fig, ax = plt.subplots(1, 3, figsize = (18, 5))
    fig.text(0.25, 0.96, 'Correlations between true and inferred channels over time', rotation=0, size=16, fontweight='bold')
    for i in range(l):
        ax[i].errorbar(t*5, correlations_inf[i], yerr=corr_stds_inf[i], fmt='o', ls='--', ecolor='black', 
                         elinewidth=1, capsize=5, capthick=1)
        ax[i].scatter(t*5, correlations_inf[i], label = "inter channel correlation")
        ax[i].axvline(56*5, color="black",linestyle="--",label="GO")
        ax[i].set_ylim((0, 1))
        ax[i].set_xlabel("time ($ms$)")
        ax[i].set_ylabel("Correlation")
        ax[i].set_title(f"RT trial = {(RT_array[i]+56)*5}$ms$")
    
    
def MUA_pred_inf_corr_time(comm_dict, diff_dict):
    
    sim_start = diff_dict["sim_start"]
    RT_array = diff_dict["RT_array"]
    MUA_trials = diff_dict["MUA_trials"]
    MUA_pred = diff_dict["MUA_pred"]
    MUA_inf = diff_dict["MUA_inf"]
    mean_corr = diff_dict["mean_corr"]
    
    l = len(RT_array)

    MUA_pred_mean = MUA_pred.mean(2)
    MUA_inf_mean = MUA_inf.mean(2)
    
    correlations_pred = np.zeros((l, 96))
    corr_stds_pred = np.zeros((l, 96))
    for i in range(l):
        for channel in range(96):
            if mean_corr:
                corr_pred = np.array([np.corrcoef(MUA_trials[i, :, channel], MUA_pred[i, :, j, channel])[0, 1] for j in range(MUA_pred.shape[2])])  
                correlations_pred[i, channel] = np.mean(corr_pred)
                corr_stds_pred[i, channel] = np.std(corr_pred)
            else:
                correlations_pred[i, channel] = np.corrcoef(MUA_trials[i, :, channel], MUA_pred_mean[i, :, channel])[0, 1]
                corr_stds_pred[i, channel] = None             
    correlations_pred = channel2grid(correlations_pred)
    corr_stds_pred = channel2grid(corr_stds_pred)
    
    correlations_inf = np.zeros((l, 96))
    corr_stds_inf = np.zeros((l, 96))
    for i in range(l):
        for channel in range(96):
            if mean_corr:
                corr_inf = np.array([np.corrcoef(MUA_trials[i, :, channel], MUA_inf[i, :, j, channel])[0, 1] for j in range(MUA_inf.shape[2])])  
                correlations_inf[i, channel] = np.mean(corr_inf)
                corr_stds_inf[i, channel] = np.std(corr_inf)
            else:
                correlations_inf[i, channel] = np.corrcoef(MUA_trials[i, :, channel], MUA_inf_mean[i, :, channel])[0, 1]
                corr_stds_inf[i, channel] = None
    correlations_inf = channel2grid(correlations_inf)
    corr_stds_inf = channel2grid(corr_stds_inf)
    
    
    fig, ax = plt.subplots(1, 3, figsize = (16, 7))
    fig.text(0.16, 0.80, f'Correlations between true and predicted ({sim_start}ms after start) channels over time', rotation=0, size=16, fontweight='bold')
    for i in range(l):
        im = ax[i].imshow(correlations_pred[i], cmap = "magma", aspect='equal', vmin=0, vmax=1)
        ax[i].set_xticks([])  # Show every 4th tick to avoid overcrowding
        ax[i].set_yticks([])
        #ax[i].set_xticklabels([])  # Show corresponding labels
        ax[i].set_title(f"RT trial = {(RT_array[i]+56)*5}$ms$")
    plt.colorbar(im, ax=ax, shrink = 0.7, aspect = 15)

    
    fig, ax = plt.subplots(1, 3, figsize = (16, 7))
    fig.text(0.25, 0.80, 'Correlations between true and inferred channels over time', rotation=0, size=16, fontweight='bold')
    for i in range(l):
        im = ax[i].imshow(correlations_inf[i], cmap = "magma", aspect='equal', vmin=0, vmax=1)
        ax[i].set_xticks([])  # Show every 4th tick to avoid overcrowding
        ax[i].set_yticks([])
        #ax[i].set_xticklabels([])  # Show corresponding labels
        ax[i].set_title(f"RT trial = {(RT_array[i]+56)*5}$ms$")
    plt.colorbar(im, ax=ax, shrink = 0.7, aspect = 15)


# -

def MUA_pred_diff_teach(comm_dict, diff_dict, trial_type):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    ar = comm_dict["ar"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    sim_start_array = diff_dict["sim_start_array"]
    
    test_set = data[f"set_{trial_type}_ordRT"]
    test_cont = data[f"cont_{trial_type}_ordRT"]
    test_RT = data[f"RT_{trial_type}_ordRT"]
    test_SSD = data[f"SSD_{trial_type}_ordRT"]
    
    t = np.linspace(0, 255, 256, dtype = int)
    t = t[::tau] 
    
    steps = test_set.shape[1]
    n = test_set.shape[0]
    q = random.randint(0, n - 1)

    trial = test_set[q]
    RT = test_RT[q]
    cont_c = test_cont[q]

    MUA = trial.mean(1)

    trial = torch.from_numpy(trial).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
    cont_c = torch.from_numpy(cont_c).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)

    for sim_start in sim_start_array:
        teacher = sim_start//(5*tau)
        alone = steps - teacher
        z_inf, z_mean, _ = dmm.inference(trial, cont_c)
        z_teach = z_mean[:teacher]
        for step in range(alone):
            if cpast:
                z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont_c[:(teacher+step+1)])
            else:
                z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher+step].unsqueeze(0))
            z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
            z_teach = torch.cat((z_teach, z_gen), dim=0)
        
        if ar:
            y_mean, y_logvar = dmm.generation_x(z_teach, trial)
        else:
            y_mean, y_logvar = dmm.generation_x(z_teach)
        y_pred = dmm.reparameterization(y_mean, y_logvar)
        y_pred = y_pred.cpu().detach().numpy()
        MUA_pred = y_pred.mean(2)
        mean = np.mean(MUA_pred, axis=1)
        std = np.std(MUA_pred, axis=1)


        # confronta true and generated MUA
        f, ax = plt.subplots(figsize = (8, 4))
        ax.plot(t*5, MUA, c = 'r', label = 'MUA')
        ax.fill_between(t*5, mean - std, mean + std, edgecolor = 'none', color = 'grey', alpha = 0.3)
        ax.plot(t*5, mean, c = 'g', label = 'MUA predicted')
        ax.axvline(sim_start, color="r",linestyle="--",label="Simulation start")
        ax.set_title(f"trial n. {q}, RT  = {RT*5:d}$ms$, simulation start at {sim_start:d}$ms$")
        ax.set_xlabel('time ($ms$)')
        ax.set_ylabel('mean MUA activity over channels')
        ax.legend()  


def MUA_type_pred(comm_dict, diff_dict, trial_type, stop):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    ar = comm_dict["ar"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    sim_start = diff_dict["sim_start"]
    
    test_set = data[f"set_{trial_type}_ordRT"]
    test_cont = data[f"cont_{trial_type}_ordRT"]
    test_RT = data[f"RT_{trial_type}_ordRT"]
    test_SSD = data[f"SSD_{trial_type}_ordRT"]
        
    teacher = sim_start//(5*tau)
    steps = test_set.shape[1]   
    alone = steps - teacher
    
    n = test_set.shape[0]
    q = random.randint(0, n - 1)
    #q = 1
    
    trial = test_set[q]
    RT = test_RT[q]
    SSD = test_SSD[q]
    cont_c = test_cont[q]

    MUA = trial.mean(1)
    #MUA_std = trial.std(1)
    
    trial = torch.from_numpy(trial).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
    cont_c = torch.from_numpy(cont_c).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
    
    z, z_mean, _ = dmm.inference(trial, cont_c)
    z_teach = z_mean[:teacher]

    for step in range(alone):
        z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher+step].unsqueeze(0))
        z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
        z_teach = torch.cat((z_teach, z_gen), dim=0)

    if ar:
        y_mean, y_logvar = dmm.generation_x(z_teach, trial)
    else:
        y_mean, y_logvar = dmm.generation_x(z_teach)
    y_pred = dmm.reparameterization(y_mean, y_logvar)
    y_pred = y_pred.cpu().detach().numpy()
    MUA_pred = y_pred.mean(2)
    mean = np.mean(MUA_pred, axis=1)
    std = np.std(MUA_pred, axis=1)
    
    
    # confronta true and generated MUA
    f, ax = plt.subplots(figsize = (8, 4))
    ax.plot(t*5, MUA, c = 'r', label = 'MUA')
    ax.fill_between(t*5, mean - std, mean + std, edgecolor = 'none', color = 'grey', alpha = 0.3)
    ax.plot(t*5, mean, c = 'g', label = 'MUA predicted')
    ax.axvline(sim_start, color="r",linestyle="--",label="simulation start")
    ax.set_xlabel('time ($ms$)')
    ax.set_ylabel('mean MUA activity over channels')
    if stop:
        ax.set_title(f"trial n. {q}, SSD  = {SSD:d}, simulation start at {sim_start:d}$ms$")
    else:
        ax.set_title(f"trial n. {q}, RT  = {RT:d}, simulation start at {sim_start:d}$ms$")
    ax.legend()


# +
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


class TripleMatrix2DEvolutionVisualizer:
    def __init__(self, data1, data2, data3, tau, titles=None):
        plt.close('all')
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        
        if titles is None:
            self.titles = ['Dataset 1', 'Dataset 2', 'Dataset 3']
        else:
            self.titles = titles
            
        # Create figure with 3 rows and 3 columns
        self.fig = plt.figure(figsize=(18, 12))
        self.gs = self.fig.add_gridspec(3, 3, width_ratios=[2, 1, 1])
        
        # Create all axes
        self.axes = []
        for i in range(3):  # For each dataset
            # Main matrix plot
            ax_matrix = self.fig.add_subplot(self.gs[i, 0])
            # Timeline plot
            ax_time = self.fig.add_subplot(self.gs[i, 1])
            # Histogram plot
            ax_hist = self.fig.add_subplot(self.gs[i, 2])
            self.axes.append([ax_matrix, ax_time, ax_hist])
        
        self.fig.suptitle('Comparison of Three Matrix Evolutions', fontsize=16, y=0.95)
        
        # Initialize all plots
        self.matrix_plots = []
        self.time_indicators = []
        vmin = min(data1.min(), data2.min(), data3.min())
        vmax = max(data1.max(), data2.max(), data3.max())
        
        for i, (data, title) in enumerate(zip([self.data1, self.data2, self.data3], self.titles)):
            # Matrix plot
            data = channel2grid(data)
            first_frame = data[0]
            matrix_plot = self.axes[i][0].imshow(first_frame, 
                                                aspect='equal',
                                                cmap='viridis',
                                                interpolation='nearest',
                                                vmin=vmin, vmax=vmax)
            self.matrix_plots.append(matrix_plot)
            self.axes[i][0].set_xticks([])
            self.axes[i][0].set_yticks([])
            #self.axes[i][0].grid(True, which='major', color='w', alpha=0.2)
            self.axes[i][0].set_title(f'{title} - Current State')
            plt.colorbar(matrix_plot, ax=self.axes[i][0])
            
            # Timeline plot
            timeline_data = np.mean(data, axis=(1,2))
            self.axes[i][1].plot(timeline_data, color='gray', alpha=0.5)
            time_indicator = self.axes[i][1].axvline(x=0, color='r', linestyle='-')
            self.time_indicators.append(time_indicator)
            self.axes[i][1].set_xlim(0, 256//tau - 1)
            self.axes[i][1].set_title(f'{title} - Average Over Time')
            self.axes[i][1].set_xlabel('Time Step')
            
            # Histogram initialization
            self.axes[i][2].set_title(f'{title} - Distribution')
            self.axes[i][2].set_xlabel('Value')
            
        plt.tight_layout()
        
    def update(self, frame):
        for i, data in enumerate([self.data1, self.data2, self.data3]):
            # Update matrix plot
            current_frame = data[frame]
            current_frame = channel2grid(current_frame)
            self.matrix_plots[i].set_array(current_frame)
            
            # Update time indicator
            self.time_indicators[i].set_xdata([frame, frame])
            
            # Update histogram
            self.axes[i][2].clear()
            self.axes[i][2].set_title(f'{self.titles[i]} - Distribution')
            self.axes[i][2].set_xlabel('Value')
            self.axes[i][2].set_ylabel('Count')
            self.axes[i][2].hist(data[frame], bins=20, color='skyblue', alpha=0.7)
            
        self.fig.suptitle(f'Comparison of Three Matrix Evolutions (Step {frame}/63)', 
                         fontsize=16, y=0.95)
        
        return self.matrix_plots + self.time_indicators

def visualize_triple_matrix_evolution(comm_dict, diff_dict):
    
    tau = comm_dict["tau"]
    
    data1 = diff_dict["MUA_trials"] 
    data2 = diff_dict["MUA_predicted"] 
    data3 = diff_dict["MUA_inferred"]
    titles = diff_dict["titles"]
    interval = diff_dict["interval"]
    
    visualizer = TripleMatrix2DEvolutionVisualizer(data1, data2, data3, tau, titles)
    
    anim = FuncAnimation(visualizer.fig, 
                        visualizer.update,
                        frames=len(data1),
                        interval=interval,
                        blit=False)
    
    html_animation = HTML(anim.to_jshtml())
    return html_animation



def latent_preRT_behaviour(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    RT_detector = comm_dict["RT_detector"]
    
    data = diff_dict["data"]
    sim_start = diff_dict["sim_start"]
    alpha = diff_dict["alpha"]
    t_dir = diff_dict["t_dir"]
    color_r = diff_dict["color_r"]
    color_l = diff_dict["color_l"]
    bins = diff_dict["bins"]
    dim_an = diff_dict["dim_an"]
    
    dir_cn = data["dir_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    set_cn = data["set_cn_ordRT"]
    dir_cn = data["dir_cn_ordRT"]
    
    RT_step = (RT_cn+56)//tau
    RT_min = RT_step.min()
    RT_max = RT_step.max()
    
    test_set = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2) 
    cont_c = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2)
    
    # Inferred traj
    y_mean, y_logvar = dmm(test_set, cont_c)
    y_inf = dmm.reparameterization(y_mean, y_logvar)
    y_inf = y_inf.cpu().detach().numpy()
    test_set = test_set.cpu().detach().numpy()
    
    z_cn, _, z_cs = infer_latent(dmm, data, device)
    steps = z_cn.shape[0]
    
    #####################################
#     from sklearn.decomposition import PCA

#     pca = PCA(n_components=z_dim)
# #     z_cn_flat = pca.fit_transform(z_cn.reshape(-1, z_dim))  # fit + transform sul train 
# #     z_cs_flat = pca.transform(z_cs.reshape(-1, z_dim))
#     z_cs_flat = pca.fit_transform(z_cs.reshape(-1, z_dim))  
#     z_cn_flat = pca.transform(z_cn.reshape(-1, z_dim))
# #     z_ws_flat = pca.transform(z_ws.reshape(-1, z_dim))
#     z_cn = z_cn_flat.reshape(steps, -1, z_dim)
#     z_cs = z_cs_flat.reshape(steps, -1, z_dim)
# #     z_ws = z_ws_flat.reshape(steps, -1, z_dim)
    
    #####################################
    
    z_cn_r = z_cn[:, dir_cn==1]
    z_cn_l = z_cn[:, dir_cn==0]
    steps, n, z_dim = z_cn.shape
    z_cut = np.stack([z_cn[RT - RT_min:RT, i] for i, RT in enumerate(RT_step)], axis=1)
    z_cut_r = z_cut[:, dir_cn==1]
    z_cut_l = z_cut[:, dir_cn==0]
    test_cut = np.stack([test_set[RT - RT_min:RT, i] for i, RT in enumerate(RT_step)], axis=1)
    y_cut = np.stack([y_inf[RT - RT_min:RT, i] for i, RT in enumerate(RT_step)], axis=1)
    
    lims = (55*5*tau, 120*5*tau)
    
    t_star = np.argmin(z_cn[:, :, 0] + z_cn[:, :, 1], axis=0)
    corr = np.corrcoef(RT_step, t_star)[0, 1]
    plt.title(f"the correlation between true and predicted RT is: {corr:.2f}")
    plt.scatter(RT_step*5*tau, t_star*5*tau)
    plt.xlabel("true RT")
    plt.ylabel("predicted RT")
    plt.xlim(lims)
    plt.ylim(lims)
    
    min_val = min(lims)
    max_val = max(lims)
    # Disegna la diagonale y = x
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    plt.show()
    
    residual = test_cut - y_cut
    std_res = residual.std(1)
    mean_res = residual.mean(1)
    
    print(std_res.shape)
    print(mean_res.shape)
    
    f, ax = plt.subplots(1, 2, figsize = (12, 5))   
    ax[0].plot(np.arange(RT_min), std_res.mean(1), '-', linewidth=2, color='b')
    ax[1].plot(np.arange(RT_min), mean_res.mean(1), '-', linewidth=2, color='b')
    ax[1].set_title('residuals mean over time')
    ax[0].set_title('residuals std over time')
    
    f, ax = plt.subplots(1, z_dim, figsize = (15, 4))   
    ax[0].plot(np.arange(RT_max), z_cn_r[:RT_max, :, 0], '-', linewidth=2, color='g', alpha = alpha)
    ax[0].plot(np.arange(RT_max), z_cn_l[:RT_max, :, 0], '-', linewidth=2, color='r', alpha = alpha)
#     ax[0].plot(np.arange(RT_max), z_cs[:RT_max, :, 0], '-', linewidth=2, color='b', alpha = alpha*2)
    ax[1].plot(np.arange(RT_max), z_cn_r[:RT_max, :, 1], '-', linewidth=2, color='g', alpha = alpha)
    ax[1].plot(np.arange(RT_max), z_cn_l[:RT_max, :, 1], '-', linewidth=2, color='r', alpha = alpha)
#     ax[1].plot(np.arange(RT_max), z_cs[:RT_max, :, 1], '-', linewidth=2, color='b', alpha = alpha*2)
    ax[0].set_title('z_dim 1 inferred')
    ax[1].set_title('z_dim 2 inferred')
    if z_dim==3:
        ax[2].plot(np.arange(RT_max), z_cn_r[:RT_max, :, 2], '-', linewidth=2, color='g', alpha = alpha)
        ax[2].plot(np.arange(RT_max), z_cn_l[:RT_max, :, 2], '-', linewidth=2, color='r', alpha = alpha)
#         ax[2].plot(np.arange(RT_max), z_cs[:RT_max, :, 2], '-', linewidth=2, color='b', alpha = alpha*2)
        ax[2].set_title('z_dim 3 inferred')
    
    
    f, ax = plt.subplots(1, z_dim, figsize = (15, 4))   
    ax[0].plot(np.arange(RT_max), z_cn[:RT_max, :, 0], '-', linewidth=2, color='g', alpha = alpha)
#     ax[0].plot(np.arange(RT_max), z_cn_l[:RT_max, :, 0], '-', linewidth=2, color='r', alpha = alpha)
    ax[0].plot(np.arange(RT_max), z_cs[:RT_max, :, 0], '-', linewidth=2, color='b', alpha = alpha*2)
    ax[1].plot(np.arange(RT_max), z_cn[:RT_max, :, 1], '-', linewidth=2, color='g', alpha = alpha)
#     ax[1].plot(np.arange(RT_max), z_cn_l[:RT_max, :, 1], '-', linewidth=2, color='r', alpha = alpha)
    ax[1].plot(np.arange(RT_max), z_cs[:RT_max, :, 1], '-', linewidth=2, color='b', alpha = alpha*2)
    ax[0].set_title('z_dim 1 inferred')
    ax[1].set_title('z_dim 2 inferred')
    if z_dim==3:
        ax[2].plot(np.arange(RT_max), z_cn[:RT_max, :, 2], '-', linewidth=2, color='g', alpha = alpha)
#         ax[2].plot(np.arange(RT_max), z_cn_l[:RT_max, :, 2], '-', linewidth=2, color='r', alpha = alpha)
        ax[2].plot(np.arange(RT_max), z_cs[:RT_max, :, 2], '-', linewidth=2, color='b', alpha = alpha*2)
        ax[2].set_title('z_dim 3 inferred')
        
    
    f, ax = plt.subplots(1, z_dim, figsize = (15, 4))   
    ax[0].plot(np.arange(RT_min), z_cut_r[:, :, 0], '-', linewidth=2, color='g', alpha = alpha)
    ax[0].plot(np.arange(RT_min), z_cut_l[:, :, 0], '-', linewidth=2, color='r', alpha = alpha)
    ax[1].plot(np.arange(RT_min), z_cut_r[:, :, 1], '-', linewidth=2, color='g', alpha = alpha)
    ax[1].plot(np.arange(RT_min), z_cut_l[:, :, 1], '-', linewidth=2, color='r', alpha = alpha)
    ax[0].set_title('z_dim 1 inferred')
    ax[1].set_title('z_dim 2 inferred')
    if z_dim==3:
        ax[2].plot(np.arange(RT_min), z_cut_r[:, :, 2], '-', linewidth=2, color='g', alpha = alpha)
        ax[2].plot(np.arange(RT_min), z_cut_l[:, :, 2], '-', linewidth=2, color='r', alpha = alpha)
        ax[2].set_title('z_dim 3 inferred')
#         mean_z2 = z_cut[:, :, 2].mean(axis=1)
#         sem_z2 = z_cut[:, :, 2].std(axis=1) / np.sqrt(z_cut.shape[1])
#         ax[2].plot(np.arange(RT_min), mean_z2, color='b', linewidth=3)
#         ax[2].fill_between(np.arange(RT_min), mean_z2 - sem_z2*3, mean_z2 + sem_z2*3, color='gray', alpha=alpha)
    
#     mean_z0 = z_cut[:, :, 0].mean(axis=1)
#     sem_z0 = z_cut[:, :, 0].std(axis=1) / np.sqrt(z_cut.shape[1])
#     ax[0].plot(np.arange(RT_min), mean_z0, color='b', linewidth=3)
#     ax[0].fill_between(np.arange(RT_min), mean_z0 - sem_z0*3, mean_z0 + sem_z0*3, color='gray', alpha=alpha)
    
#     mean_z1 = z_cut[:, :, 1].mean(axis=1)
#     sem_z1 = z_cut[:, :, 1].std(axis=1) / np.sqrt(z_cut.shape[1])
#     ax[1].plot(np.arange(RT_min), mean_z1, color='b', linewidth=3)
#     ax[1].fill_between(np.arange(RT_min), mean_z1 - sem_z1*3, mean_z1 + sem_z1*3, color='gray', alpha=alpha)
    
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    
    traj_right = z_cn_r[:, :, dim_an].T
    traj_left = z_cn_l[:, :, dim_an].T
    
#     traj_right = z_cut_r[:, :, dim_an].T
#     traj_left = z_cut_l[:, :, dim_an].T

    # traj_right: (n, T), traj_left: (m, T); T tempi
    X = np.concatenate([traj_right, traj_left], axis=0)  # (n+m, T)
    y = np.concatenate([np.ones(traj_right.shape[0]), np.zeros(traj_left.shape[0])])        # labels 1=destra, 0=sinistra

    T = X.shape[1]
    
    aucs = np.zeros(T)
    accs = np.zeros(T)
    for t in range(T):
        Xt = X[:, t:t+1]  # feature scalare a t
        aucs[t] = cross_val_score(LogisticRegression(), Xt, y, scoring='roc_auc', cv=5).mean()
        accs[t] = cross_val_score(LogisticRegression(), Xt, y, scoring='accuracy', cv=5).mean()
        
    best_t = T - np.argmax(aucs)
    print(f"best discrimination at {best_t*tau*5}$ms$ before the RT, with AUC={aucs[T-best_t]:.2f} and accuracy={accs[T-best_t]:.2f}.")
    
    z_right = z_cn_r[-best_t, :, dim_an]
    z_left = z_cn_l[-best_t, :, dim_an]
    
#     z_right = z_cut_r[-best_t, :, dim_an]
#     z_left = z_cut_l[-best_t, :, dim_an]
    
    f, ax = plt.subplots()
    
    ax.hist(z_right, bins=bins, density=True, alpha = 0.4, color=color_r, edgecolor="none", label = "true RT")
    ax.hist(z_left, bins=bins, density=True, alpha = 0.4, color=color_l, edgecolor="none", label = "true RT")
    ax.axvline(z_right.mean(), color=color_r,linestyle="--")
    ax.axvline(z_left.mean(), color=color_l,linestyle="--")
    ax.set_xlabel("z2")
    ax.set_ylabel("Counts")
    ax.set_xticks([])
    ax.set_yticks([])
    
    f, ax = plt.subplots(figsize = (6, 5))   
    ax.plot(np.arange(T), z_cn_r[:, :, dim_an], '-', linewidth=2, color='g', alpha = alpha)
    ax.plot(np.arange(T), z_cn_l[:, :, dim_an], '-', linewidth=2, color='r', alpha = alpha)
    ax.axvline(T-best_t, color="black",linestyle="--")
#     ax.plot(np.arange(RT_min), z_cut_r[:, :, 1], '-', linewidth=2, color='g', alpha = alpha)
#     ax.plot(np.arange(RT_min), z_cut_l[:, :, 1], '-', linewidth=2, color='r', alpha = alpha)
    ax.set_title('z_dim 1 inferred')
    ax.set_title('z_dim 2 inferred')
    
    
    # Generated traj
    
    teacher = sim_start//(5*tau)
    alone = steps - teacher
    z_teach = z_cn[:teacher]
    
    z_teach = torch.from_numpy(z_teach).float().to(device)
    
    for step in range(alone):
        z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher+step].unsqueeze(0))
        z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
        z_teach = torch.cat((z_teach, z_gen), dim=0)
        
    RT_output = RT_detector(z_teach.permute(1, 0, 2))
    RT_estimate = prob_to_RT(RT_output, tau)    
    RT_estimate = np.asarray(RT_estimate, dtype=int)
    RT_gen_min = RT_estimate.min()
    
    z_teach = z_teach.cpu().detach().numpy()
    z_gen_cut = np.stack([z_teach[(RT - RT_gen_min):RT, i] for i, RT in enumerate(RT_estimate)], axis=1)
    z_gen_cut_r = z_gen_cut[:, dir_cn==1]
    z_gen_cut_l = z_gen_cut[:, dir_cn==0]
    
    f, ax = plt.subplots(1, z_dim, figsize = (15, 4))   
    ax[0].plot(np.arange(RT_gen_min), z_gen_cut_r[:, :, 0], '-', linewidth=2, color='g', alpha = alpha)
    ax[0].plot(np.arange(RT_gen_min), z_gen_cut_l[:, :, 0], '-', linewidth=2, color='r', alpha = alpha)
    ax[1].plot(np.arange(RT_gen_min), z_gen_cut_r[:, :, 1], '-', linewidth=2, color='g', alpha = alpha)
    ax[1].plot(np.arange(RT_gen_min), z_gen_cut_l[:, :, 1], '-', linewidth=2, color='r', alpha = alpha)
    ax[0].set_title('z_dim 1 generated from GO')
    ax[1].set_title('z_dim 2 generated from GO')
    if z_dim==3:
        ax[2].plot(np.arange(RT_gen_min), z_gen_cut_r[:, :, 2], '-', linewidth=2, color='g', alpha = alpha)
        ax[2].plot(np.arange(RT_gen_min), z_gen_cut_l[:, :, 2], '-', linewidth=2, color='r', alpha = alpha)
        ax[2].set_title('z_dim 3 generated from GO')
        mean_z2 = z_gen_cut[:, :, 2].mean(axis=1)
        sem_z2 = z_gen_cut[:, :, 2].std(axis=1) / np.sqrt(z_cut.shape[1])
        ax[2].plot(np.arange(RT_gen_min), mean_z2, color='b', linewidth=3)
        ax[2].fill_between(np.arange(RT_gen_min), mean_z2 - sem_z2*3, mean_z2 + sem_z2*3, color='gray', alpha=0.3)
    
    mean_gen_z0 = z_gen_cut[:, :, 0].mean(axis=1)
    sem_gen_z0 = z_gen_cut[:, :, 0].std(axis=1) / np.sqrt(z_gen_cut.shape[1])
    ax[0].plot(np.arange(RT_gen_min), mean_gen_z0, color='b', linewidth=3)
    ax[0].fill_between(np.arange(RT_gen_min), mean_gen_z0 - sem_gen_z0*3, mean_gen_z0 + sem_gen_z0*3, color='gray', alpha=0.3)
    
    mean_gen_z1 = z_gen_cut[:, :, 1].mean(axis=1)
    sem_gen_z1 = z_gen_cut[:, :, 1].std(axis=1) / np.sqrt(z_gen_cut.shape[1])
    ax[1].plot(np.arange(RT_gen_min), mean_gen_z1, color='b', linewidth=3)
    ax[1].fill_between(np.arange(RT_gen_min), mean_gen_z1 - sem_gen_z1*3, mean_gen_z1 + sem_gen_z1*3, color='gray', alpha=0.3)
        
# -

def latent_cn_traj_directions(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    
    data = diff_dict["data"]
    axis = diff_dict["axis"]
    alpha = diff_dict["alpha"]
    pca_flag = diff_dict["pca_flag"]
    z_pca = diff_dict["z_pca"]
    n_trials = diff_dict["n_trials"]

    dir_cn = data["dir_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    z_cn, z_ws, z_cs = infer_latent(dmm, data, device)
    
    n_cn = z_cn.shape[1]
    n_cs = z_cs.shape[1]
    n_ws = z_ws.shape[1]
    
    steps, _, _ = z_cn.shape
    
    #####################################
    
    if pca_flag:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=z_dim)
        z_cs_flat = pca.fit_transform(z_cs.reshape(-1, z_dim))  
        z_cn_flat = pca.transform(z_cn.reshape(-1, z_dim))
        z_cn = z_cn_flat.reshape(steps, n_cn, z_dim)

#     pca = PCA(n_components=z_dim)
#     z_cn_flat = pca.fit_transform(z_cn.reshape(-1, z_dim))  # fit + transform sul train 
#     z_cn = z_cn_flat.reshape(steps, s, z_dim)

    ############  Directions   ################
    
    x_r = z_cn[:, dir_cn==1, axis[0]]
    x_l = z_cn[:, dir_cn==0, axis[0]]

    y_r = z_cn[:, dir_cn==1, axis[1]]
    y_l = z_cn[:, dir_cn==0, axis[1]]
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score
    
    X_t = np.vstack([z_cn[(56+20)//tau:(56+100)//tau, dir_cn==1].reshape(-1, z_dim), z_cn[(56+20)//tau:(56+100)//tau, dir_cn==0].reshape(-1, z_dim)])           # shape (n1+n2, 3)
    y_t = np.array([1]*x_r.shape[1]*40 + [2]*x_l.shape[1]*40)
    
    ##############  Stops  ##################
    
    if z_pca:
        with np.load(comm_dict["saved_path"]+"/data_split.npz", allow_pickle=True) as loaded_file:
            train_set = loaded_file["train_set"]
            vali_set = loaded_file["vali_set"]
            test_set = loaded_file["test_set"]
            train_direction = loaded_file["train_direction"]
            vali_direction = loaded_file["vali_direction"]
            test_direction = loaded_file["test_direction"]
            train_SSD = loaded_file["train_SSD"]
            vali_SSD = loaded_file["vali_SSD"]
            test_SSD = loaded_file["test_SSD"]
            train_RT = loaded_file["train_RT"]
            vali_RT = loaded_file["vali_RT"]
            test_RT = loaded_file["test_RT"]
            
        n_train, steps, features = train_set.shape
        X_train = train_set.reshape(-1, features)  # shape = (n_train * time_steps, 96)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=z_dim)
        X_train_pca = pca.fit_transform(X_train)  # fit + transform sul train 
        train_pca = X_train_pca.reshape(n_train, steps, z_dim)
        train_pca = np.transpose(train_pca, (1, 0, 2))

        mask_ws = (train_RT!=0) & (train_SSD!=0)
        mask_cs = train_RT==0
        mask_ns = train_SSD==0

        z_cn = train_pca[:, mask_ns]
        z_cs = train_pca[:, mask_cs]
        z_ws = train_pca[:, mask_ws]

    #     test_rec = test_PCA(train_set, test_set, z_dim)

        SSD_ws = train_SSD[mask_ws]
        RT_ws = train_RT[mask_ws]
        SSD_cs = train_SSD[mask_cs]
    
    else:
        SSD_ws = data["SSD_ws_ordRT"]
        RT_ws = data["RT_ws_ordRT"]
        SSD_cs = data["SSD_cs_ordSSD"]

        z_cn, z_ws, z_cs = infer_latent(dmm, data, device, n_trials)
        if n_trials>1:
            z_cn = z_cn.reshape(steps, n_cn, n_trials, z_dim)
            z_cn = z_cn.mean(2)
            z_ws = z_ws.reshape(steps, n_ws, n_trials, z_dim)
            z_ws = z_ws.mean(2)
            z_cs = z_cs.reshape(steps, n_cs, n_trials, z_dim)
            z_cs = z_cs.mean(2)
    
    z_cs_stop = z_cs[(SSD_cs+56)//tau, np.arange(z_cs.shape[1])]
    z_ws_stop = z_ws[(SSD_ws+56)//tau, np.arange(z_ws.shape[1])]
    
    X_t = np.vstack([z_cs_stop, z_ws_stop])           # shape (n1+n2, 3)
    y_t = np.array([1]*z_cs.shape[1] + [2]*z_ws.shape[1])
    
    #####################################
    
    # LDA binario [web:29][web:32]
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_t, y_t)

    # Predizioni hard (per accuracy)
    y_pred = lda.predict(X_t)
    acc = (y_pred == y_t).mean()
    
    # Score continui per AUC (decision_function è adatto al ROC) [web:32][web:31]
    scores = lda.decision_function(X_t)
    # sklearn vuole label binarie 0/1; rimappiamo 1 -> 0, 2 -> 1
    y_bin = (y_t == 2).astype(int)
    auc = roc_auc_score(y_bin, scores)
    
    print(f"accuracy: {acc:.2f}")
    print(f"AUC: {auc:.2f}")
    
    w = lda.coef_[0]       # shape (3,)
    b = lda.intercept_[0]  # scalare

    # normal = vettore normale al piano
    normal = w / np.linalg.norm(w)
    print("Vettore normale al piano discriminante:", normal)

#     accuracies = []
#     aucs = []
#     ldas = []
#     scalers = []

#     for t in range(steps):
#         # Costruisci dati per tempo t
#         X_t = np.vstack([z_cn[t, dir_cn==1], z_cn[t, dir_cn==0]])           # shape (n1+n2, 3)
#         y_t = np.array([1]*x_r.shape[1] + [2]*x_l.shape[1])  # label 1 vs 2

#         # LDA binario [web:29][web:32]
#         lda = LinearDiscriminantAnalysis()
#         lda.fit(X_t, y_t)

#         # Predizioni hard (per accuracy)
#         y_pred = lda.predict(X_t)
#         acc = (y_pred == y_t).mean()

#         # Score continui per AUC (decision_function è adatto al ROC) [web:32][web:31]
# #         scores = lda.decision_function(X_t_scaled)
#         # sklearn vuole label binarie 0/1; rimappiamo 1 -> 0, 2 -> 1
# #         y_bin = (y_t == 2).astype(int)
# #         auc = roc_auc_score(y_bin, scores)

#         accuracies.append(acc)
# #         aucs.append(auc)
#         ldas.append(lda)

#     accuracies = np.array(accuracies)
# #     aucs = np.array(aucs)

#     # Scegli il tempo migliore (puoi usare AUC o accuracy)
# #     best_t_auc = np.argmax(aucs)
#     best_t_acc = np.argmax(accuracies)

#     print(f"Miglior tempo (accuracy): {best_t_acc*tau*5}$ms$, ACC = {accuracies[best_t_acc]:.2f}")

#     t_star = int(best_t_acc)
#     lda_star = ldas[t_star]

#     w = lda_star.coef_[0]       # shape (3,)
#     b = lda_star.intercept_[0]  # scalare

#     # normal = vettore normale al piano
#     normal = w / np.linalg.norm(w)
#     print("Vettore normale al piano discriminante:", normal)

    #####################################
    
    if z_dim==2:
        
        mask1 = y_t == 1
        mask2 = y_t == 2
        ax.scatter(X_t[mask1, 0], X_t[mask1, 1], c='coral',  label='CS', alpha=0.7, s=30)
        ax.scatter(X_t[mask2, 0], X_t[mask2, 1], c='steelblue', label='WS', alpha=0.7, s=30)
        
        # Project the 3D boundary onto the (x1, x2) plane:
        # Original: w[0]*x1 + w[1]*x2 + w[2]*x3 + b = 0
        # Projected: w[0]*x1 + w[1]*x2 + b = 0
        #   => x2 = -(w[0]*x1 + b) / w[1]
        w0, w1, w2 = w          # unpack 3D normal
        b_scalar    = b          # lda.intercept_[0]

        x1_range = np.linspace(X_t[:, 0].min() - 0.5, X_t[:, 0].max() + 0.5, 200)

        x2_line = -(w0 * x1_range + b_scalar) / w1
        ax.plot(x1_range, x2_line, 'k--', lw=1.5, label='projected LDA boundary')
        ax.set_xlabel('z₁')
        ax.set_ylabel('z₂')
        ax.legend()
        ax.set_title('LDA boundary projected onto first two components')
        plt.tight_layout()
        plt.show()
        
    
    elif z_dim==3:
        
        f, ax = plt.subplots(figsize = (7, 6))
        
        ax.plot(x_r, y_r, '-', linewidth=2, color='g', alpha = alpha)
        ax.plot(x_l, y_l, '-', linewidth=2, color='r', alpha = alpha)
        ax.set_xlabel("first latent component")
        ax.set_ylabel("second latent component")
        ax.set_title(f"right vs left directed trials")
        plt.show()
        
        x_r = z_cn[:, dir_cn==1, 0]
        x_l = z_cn[:, dir_cn==0, 0]

        y_r = z_cn[:, dir_cn==1, 1]
        y_l = z_cn[:, dir_cn==0, 1]  
        
        z_r = z_cn[:, dir_cn==1, 2]
        z_l = z_cn[:, dir_cn==0, 2]  
        
        import plotly.graph_objects as go

        fig = go.Figure()

#         for i in range(x_r.shape[1]):
#             fig.add_trace(go.Scatter3d(
#                 x=x_r[:, i], y=y_r[:, i], z=z_r[:, i],
#                 mode='lines', line=dict(color='green', width=3), opacity=0.3
#             ))
            
#         for i in range(x_l.shape[1]):
#             fig.add_trace(go.Scatter3d(
#                 x=x_l[:, i], y=y_l[:, i], z=z_l[:, i],
#                 mode='lines', line=dict(color='red', width=3), opacity=0.3
#             ))
            
            
        
        fig.add_trace(go.Scatter3d(
            x=z_cs_stop[:, 0], y=z_cs_stop[:, 1], z=z_cs_stop[:, 2],
            mode='markers', marker=dict(color='red', size=3), opacity=0.3
        ))
            
        
        fig.add_trace(go.Scatter3d(
            x=z_ws_stop[:, 0], y=z_ws_stop[:, 1], z=z_ws_stop[:, 2],
            mode='markers', marker=dict(color='green', size=3), opacity=0.3
        ))
            

        # don't remember why keeping it
#         fig.add_trace(go.Scatter3d(
#             x=z_cn[t_star, dir_cn==1, 0], y=z_cn[t_star, dir_cn==1, 1], z=z_cn[t_star, dir_cn==1, 2],
#             mode='markers',  # FIXED: Added mode
#             marker=dict(color='red', size=5),  # FIXED: color moved inside marker dict
#             opacity=0.5,
#             name='right'
#         ))

#         fig.add_trace(go.Scatter3d(
#                 x=z_cn[t_star, dir_cn==0, 0], y=z_cn[t_star, dir_cn==0, 1], z=z_cn[t_star, dir_cn==0, 2],
#                 mode='markers',  # FIXED: Added mode
#                 marker=dict(color='green', size=5),  # FIXED: color moved inside marker dict
#                 opacity=0.5,
#                 name='left'
#             ))

        # Piano discriminante (mesh traslucida)
        z1_min, z1_max = z_cn[:, :, 0].min(), z_cn[:, :, 0].max()
        z2_min, z2_max = z_cn[:, :, 1].min(), z_cn[:, :, 1].max()
        z3_min, z3_max = z_cn[:, :, 2].min(), z_cn[:, :, 2].max()
        
        a, b_plane, c = w
        d_plane = b

#         if abs(c) < 1e-6:
#             # es: risolvi per y: y = (-d - a*x - c*z)/b_plane
#             xx, zz = np.meshgrid(
#                 np.linspace(z1_min, z1_max, 20),
#                 np.linspace(z3_min, z3_max, 20)
#             )
#             yy = (-d_plane - a*xx - c*zz) / b_plane
#         else:
        xx, yy = np.meshgrid(
            np.linspace(z1_min, z1_max, 20),
            np.linspace(z2_min, z2_max, 20)
        )
        zz = (-d_plane - a*xx - b_plane*yy) / c


#         a, b, c = normal  # nx, ny, nz
#         d = b_orig
# #         d = -np.dot(normal, point_on_plane)
#         xx, yy = np.meshgrid(np.linspace(z1_min, z1_max, 20), np.linspace(z2_min, z2_max, 20))
#         zz = (-d - a * xx - b * yy) / c  # Formula piano: ax+by+cz+d=0
        zz = np.clip(zz, z3_min, z3_max)
        
        fig.add_trace(go.Surface(
            x=xx,
            y=yy,
            z=zz,
            opacity=0.25,
            showscale=False,
            colorscale=[[0, 'blue'], [1, 'blue']],
            name='Piano discriminante'
        ))
        fig.update_layout(
            scene=dict(
                xaxis_title='z1', yaxis_title='z2', zaxis_title='z3',
                bgcolor='black'
            ),
            width=700, height=700,
#             title='Right vs Left trajectories'
            title='cs vs ws stops'
        )
        #fig.show()
        return fig



def latent_cn_u_directions(comm_dict, diff_dict):

    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]

    data = diff_dict["data"]
    axis = diff_dict["axis"]
    alpha = diff_dict["alpha"]

    dir_cn = data["dir_cn_ordRT"]
    u_dim = dmm.u_dim

    _, _, _, u_cn, _, _ = infer_latent(dmm, data, device, u_flag=True)

    x_r = u_cn[:, dir_cn==1, axis[0]]
    x_l = u_cn[:, dir_cn==0, axis[0]]

    y_r = u_cn[:, dir_cn==1, axis[1]]
    y_l = u_cn[:, dir_cn==0, axis[1]]  

    if u_dim==2:

        f, ax = plt.subplots(figsize = (7, 6))

        ax.plot(x_r, y_r, '-', linewidth=2, color='g', alpha = alpha)
        ax.plot(x_l, y_l, '-', linewidth=2, color='r', alpha = alpha)
        ax.set_xlabel("first latent component")
        ax.set_ylabel("second latent component")
        ax.set_title(f"right vs left directed trials")
        plt.show()
        return f

    elif u_dim==3:

        f, ax = plt.subplots(figsize = (7, 6))

        ax.plot(x_r, y_r, '-', linewidth=2, color='g', alpha = alpha)
        ax.plot(x_l, y_l, '-', linewidth=2, color='r', alpha = alpha)
        ax.set_xlabel("first u component")
        ax.set_ylabel("second u component")
        ax.set_title(f"right vs left directed trials")
        plt.show()

        x_r = u_cn[:, dir_cn==1, 0]
        x_l = u_cn[:, dir_cn==0, 0]

        y_r = u_cn[:, dir_cn==1, 1]
        y_l = u_cn[:, dir_cn==0, 1]  

        z_r = u_cn[:, dir_cn==1, 2]
        z_l = u_cn[:, dir_cn==0, 2]  

        import plotly.graph_objects as go

        fig = go.Figure()

        for i in range(x_r.shape[1]):
            fig.add_trace(go.Scatter3d(
                x=x_r[:, i], y=y_r[:, i], z=z_r[:, i],
                mode='lines', line=dict(color='green', width=3), opacity=0.3
            ))

        for i in range(x_l.shape[1]):
            fig.add_trace(go.Scatter3d(
                x=x_l[:, i], y=y_l[:, i], z=z_l[:, i],
                mode='lines', line=dict(color='red', width=3), opacity=0.3
            ))

        fig.update_layout(
            scene=dict(
                xaxis_title='u1', yaxis_title='u2', zaxis_title='u3',
                bgcolor='black'
            ),
            width=700, height=700,
            title='Right vs Left trajectories'
        )
        #fig.show()
        return fig


# +
def random_latent_cn_traj(dmm, data, tau, device, n_trials=1):
    
    RT_cn = data["RT_cn_ordRT"]
    steps = 256//tau
    n = len(RT_cn)
    q = random.randint(0, n - 1)
    RT_trial = RT_cn[q]
    
    z_cn, _, _ = infer_latent(dmm, data, device, n_trials=n_trials)
    z_dim = z_cn.shape[2]

    if n_trials>1:
        z_cn = z_cn.reshape(steps, n, n_trials, z_dim)
    
    z_trial = z_cn[:, q]  
    RT = (56 + RT_trial)//tau
    
    return z_trial, RT, q

# def random_latent_cn_u(dmm, data, tau, device, n_trials=1):
    
#     RT_cn = data["RT_cn_ordRT"]
#     steps = 256//tau
#     n = len(RT_cn)
#     q = random.randint(0, n - 1)
#     RT_trial = RT_cn[q]
    
#     _, _, _, u_cn, _, _ = infer_latent(dmm, data, device, n_trials=n_trials, u_flag=True)
#     u_dim = u_cn.shape[2]

#     if n_trials>1:
#         u_cn = u_cn.reshape(steps, n, n_trials, u_dim)
    
#     u_trial = u_cn[:, q]  
#     RT = (56 + RT_trial)//tau
    
#     return u_trial, RT, q


def random_latent_cs_traj(dmm, data, tau, device, n_trials=1):
    
    SSD_cs = data["SSD_cs_ordSSD"]
    steps = 256//tau
    n = len(SSD_cs)
    q = random.randint(0, n - 1)
    SSD_trial = SSD_cs[q]
    
    _, _, z_cs = infer_latent(dmm, data, device, n_trials=n_trials)
    z_dim = z_cs.shape[2]

    if n_trials>1:
        z_cs = z_cs.reshape(steps, n, n_trials, z_dim)
    
    z_trial = z_cs[:, q]  
    SSD = (56 + SSD_trial)//tau
    
    return z_trial, SSD, q

# def random_latent_cs_u(dmm, data, tau, device, n_trials=1):
    
#     SSD_cs = data["SSD_cs_ordSSD"]
#     steps = 256//tau
#     n = len(SSD_cs)
#     q = random.randint(0, n - 1)
#     SSD_trial = SSD_cs[q]
    
#     _, _, _, _, _, u_cs = infer_latent(dmm, data, device, n_trials=n_trials, u_flag=True)
#     u_dim = u_cs.shape[2]

#     if n_trials>1:
#         u_cs = u_cs.reshape(steps, n, n_trials, u_dim)
    
#     u_trial = u_cs[:, q]  
#     SSD = (56 + SSD_trial)//tau
    
#     return u_trial, SSD, q



def latent_traj_ctype(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    t = comm_dict["t"]
    z_dim = comm_dict["z_dim"]
    
    data = diff_dict["data"]
    alpha = diff_dict["alpha"]
    axis = diff_dict["axis"]
    type_list = diff_dict["type_list"]
    type_trial = diff_dict["type_trial"]
    RT_show = diff_dict["RT_show"]
    axis = diff_dict["axis"]
    
    RT_cn = data["RT_cn_ordRT"] 
    RT_ws = data["RT_ws_ordRT"] 
    RT_cn = (RT_cn + 56)//tau
    RT_ws = (RT_ws + 56)//tau
    steps = 256//tau
    
    z_cn, z_ws, z_cs = infer_latent(dmm, data, device)
    
    #####################################
#     from sklearn.decomposition import PCA

#     pca = PCA(n_components=z_dim)
# #     z_cn_flat = pca.fit_transform(z_cn.reshape(-1, z_dim))  # fit + transform sul train 
# #     z_cs_flat = pca.transform(z_cs.reshape(-1, z_dim))
#     z_cs_flat = pca.fit_transform(z_cs.reshape(-1, z_dim))  
#     z_cn_flat = pca.transform(z_cn.reshape(-1, z_dim))
#     z_ws_flat = pca.transform(z_ws.reshape(-1, z_dim))
#     z_cn = z_cn_flat.reshape(steps, -1, z_dim)
#     z_cs = z_cs_flat.reshape(steps, -1, z_dim)
#     z_ws = z_ws_flat.reshape(steps, -1, z_dim)
    
    #####################################
    
    samples_cn = z_cn.shape[1]
    samples_ws = z_ws.shape[1]
    
    x_cs = z_cs[:, :, axis[0]]
    x_cn = z_cn[:, :, axis[0]]
    x_ws = z_ws[:, :, axis[0]]

    y_cs = z_cs[:, :, axis[1]]
    y_cn = z_cn[:, :, axis[1]]
    y_ws = z_ws[:, :, axis[1]]
    
    x_RT_cn = x_cn[RT_cn, np.arange(samples_cn)]
    y_RT_cn = y_cn[RT_cn, np.arange(samples_cn)]
    x_RT_ws = x_ws[RT_ws, np.arange(samples_ws)]
    y_RT_ws = y_ws[RT_ws, np.arange(samples_ws)]
    
    x_GO_cn = x_cn[56//tau]
    y_GO_cn = y_cn[56//tau]
    x_GO_ws = x_ws[56//tau]
    y_GO_ws = y_ws[56//tau]
    
    if type_trial=="cn":
        z_trial, RT, q = random_latent_cn_traj(dmm, data, tau, device)
        set_cn = data["set_cn_ordRT"]
        MUA = set_cn[q]
        x_RT = z_trial[RT, 0]
        y_RT = z_trial[RT, 1]
    elif type_trial=="cs":
        z_trial, SSD, q = random_latent_cs_traj(dmm, data, tau, device)
        set_cs = data["set_cs_ordSSD"]
        MUA = set_cs[q]
        x_SSD = z_trial[SSD, 0]
        y_SSD = z_trial[SSD, 1]

    print(f"traj n.{q}")
        
    x_true_story = z_trial[:, 0]
    y_true_story = z_trial[:, 1]
    x_start = z_trial[0, 0]
    y_start = z_trial[0, 1]
    x_GO = z_trial[56//tau, 0]
    y_GO = z_trial[56//tau, 1]
    
    if z_dim==2:
        f, ax = plt.subplots(figsize = (7, 6))

        color = np.linspace(0, 1, steps)

        # Plot the surface
        if "cn"in type_list:
            ax.plot(x_cn, y_cn, '-', linewidth=2, color='g', alpha = alpha)
        if "ws"in type_list:
            ax.plot(x_ws, y_ws, '-', linewidth=2, color='orange', alpha = alpha)
        if "cs"in type_list:
            ax.plot(x_cs, y_cs, '-', linewidth=2, color='r', alpha = alpha)

        if RT_show:

            #ax.scatter(x_RT_cn, y_RT_cn, s = 15, c = color, cmap = "Blues", alpha = 0.6, label = "cn")
            #ax.scatter(x_RT_ws, y_RT_ws, s = 15, c = RT_ws/RT_cn.max(), cmap = "Purples", alpha = 0.8, label = "ws")
            ax.scatter(x_RT_cn, y_RT_cn, s = 15, c = "b", alpha = 1, label = "cn")
            ax.scatter(x_RT_ws, y_RT_ws, s = 15, c = "purple", alpha = 1, label = "ws")

        ax.plot(x_true_story, y_true_story, '-', linewidth=4, color="brown", alpha = 1, label = "trajectory example")
        ax.scatter(x_GO, y_GO, s = 150, c = "b", alpha = 1, label = "GO")
        if type_trial=="cn":
            ax.scatter(x_RT, y_RT, s = 150, c = 'r', alpha = 1, label = "RT")
        if type_trial=="cs":
            ax.plot(x_SSD, y_SSD, color='red', marker='x', markeredgewidth = 3, markersize = 15, label = "SSD")
            #ax.scatter(x_SSD, y_SSD, s = 150, c = 'r', alpha = 1, label = "SSD")
        ax.scatter(x_start, y_start, s = 150, c = 'y', alpha = 1, label = "start")
        ax.set_xlabel("first latent component")
        ax.set_ylabel("second latent component")
        ax.set_title(f'cs vs ws vs ns, with n.{q} traj')
#         ax.legend()
        plt.show()


#         f, ax = plt.subplots(figsize = (8, 6))
#         ax.plot(t*5, MUA.mean(1), c = "green", label = 'mean MUA of the traj above')
#         ax.axvline(56*5, color="black",linestyle="--",label= "GO")
#         if type_trial=="cn":
#             ax.axvline(RT*tau*5, color="blue",linestyle="--",label= "RT")
#         if type_trial=="cs":
#             ax.axvline(SSD*tau*5, color="red",linestyle="--",label= "SSD")
#         ax.set_xlabel('time ($ms$)')
#         ax.set_ylabel('mean MUA activity over channels')
#         ax.legend(fontsize=7.3)
#         plt.show()
    
    
    elif z_dim==3:
        
        f, ax = plt.subplots(figsize = (7, 6))

        # Plot the surface
        if "cn"in type_list:
            ax.plot(x_cn, y_cn, '-', linewidth=2, color='g', alpha = alpha)
        if "ws"in type_list:
            ax.plot(x_ws, y_ws, '-', linewidth=2, color='orange', alpha = alpha)
        if "cs"in type_list:
            ax.plot(x_cs, y_cs, '-', linewidth=2, color='r', alpha = alpha)

        if RT_show:

            #ax.scatter(x_RT_cn, y_RT_cn, s = 15, c = color, cmap = "Blues", alpha = 0.6, label = "cn")
            #ax.scatter(x_RT_ws, y_RT_ws, s = 15, c = RT_ws/RT_cn.max(), cmap = "Purples", alpha = 0.8, label = "ws")
            ax.scatter(x_RT_cn, y_RT_cn, s = 15, c = "b", alpha = 1, label = "cn")
            ax.scatter(x_RT_ws, y_RT_ws, s = 15, c = "purple", alpha = 1, label = "ws")

        ax.plot(x_true_story, y_true_story, '-', linewidth=4, color="brown", alpha = 1, label = "trajectory example")
        ax.scatter(x_GO, y_GO, s = 150, c = "b", alpha = 1, label = "GO")
        if type_trial=="cn":
            ax.scatter(x_RT, y_RT, s = 150, c = 'r', alpha = 1, label = "RT")
        if type_trial=="cs":
            ax.plot(x_SSD, y_SSD, color='red', marker='x', markeredgewidth = 3, markersize = 15, label = "SSD")
            #ax.scatter(x_SSD, y_SSD, s = 150, c = 'r', alpha = 1, label = "SSD")
        ax.scatter(x_start, y_start, s = 150, c = 'y', alpha = 1, label = "start")
        ax.set_xlabel("first latent component")
        ax.set_ylabel("second latent component")
        ax.set_title(f'cs vs ws vs ns, with n.{q} traj')
#         ax.legend()
        plt.show()
        
        
        
#         z_cs = z_cs[:, :, 2]
#         z_cn = z_cn[:, :, 2]
#         z_ws = z_ws[:, :, 2]
        
#         z_true_story = z_trial[:, 2]
#         z_start = z_trial[0, 2]
#         z_GO = z_trial[56//tau, 2]
#         #z_RT = RT_z[2]
        
        import plotly.graph_objects as go

        fig = go.Figure()

        
        if "cn"in type_list:
            for i in range(z_cn.shape[1]):
                fig.add_trace(go.Scatter3d(
                    x=z_cn[:, i, 0], y=z_cn[:, i, 1], z=z_cn[:, i, 2],
                    mode='lines', line=dict(color='green', width=3), opacity=0.3
                ))
        if "ws"in type_list:
            for i in range(z_ws.shape[1]):
                fig.add_trace(go.Scatter3d(
                    x=z_ws[:, i, 0], y=z_ws[:, i, 1], z=z_ws[:, i, 2],
                    mode='lines', line=dict(color='orange', width=3), opacity=0.3
                ))
        if "cs"in type_list:
            for i in range(z_cs.shape[1]):
                fig.add_trace(go.Scatter3d(
                    x=z_cs[:, i, 0], y=z_cs[:, i, 1], z=z_cs[:, i, 2],
                    mode='lines', line=dict(color='red', width=3), opacity=0.3
                ))

        fig.update_layout(
            scene=dict(
                xaxis_title='z1', yaxis_title='z2', zaxis_title='z3',
                bgcolor='black'
            ),
            width=700, height=700,
            title='cs vs ws vs ns'
        )
        #fig.show()
        return fig
    
    
def latent_traj_example(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    t = comm_dict["t"]
    z_dim = comm_dict["z_dim"]
    z_ticks = comm_dict["z_ticks"]
    
    data = diff_dict["data"]
#     alpha = diff_dict["alpha"]
#     RT_show = diff_dict["RT_show"]
    color_line = diff_dict["color_line"]
    color_arrows = diff_dict["color_arrows"]
    n_arrows = diff_dict["n_arrows"]
    azim = diff_dict["azim"]
    elev = diff_dict["elev"]
    scale = diff_dict["scale"]
    origin = diff_dict["origin"]
    offset = diff_dict["offset"]
    labelsize = diff_dict["labelsize"]
    labelpad = diff_dict["labelpad"]
    fontsize = diff_dict["fontsize"]
    lw_axis = diff_dict["lw_axis"]
    lw_arrows = diff_dict["lw_arrows"]
    lw_line = diff_dict["lw_line"]
    
    set_cn = data["set_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"] 
    RT_cn = (RT_cn+56)//tau
    steps = 256//tau
    
    z_trial, RT, q = random_latent_cn_traj(dmm, data, tau, device)
    MUA = set_cn[q]
             
    x_RT = z_trial[RT, 0]
    y_RT = z_trial[RT, 1]
    z_RT = z_trial[RT, 2]

    print(f"traj n.{q}")
        
    x_true_story = z_trial[:, 0]
    y_true_story = z_trial[:, 1]
    z_true_story = z_trial[:, 2]
    x_start = z_trial[0, 0]
    y_start = z_trial[0, 1]
    z_start = z_trial[0, 2]
    x_GO = z_trial[56//tau, 0]
    y_GO = z_trial[56//tau, 1]
    z_GO = z_trial[56//tau, 2]
    
    
    # -------------------------
    # Create figure
    # -------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Trajectory
    ax.plot(x_true_story, y_true_story, z_true_story, color=color_line, lw=lw_line)

    # -------------------------
    # Arrows (direction of time)
    # -------------------------
    idx = np.linspace(0, len(x_true_story) - 2, n_arrows).astype(int)

    dx = np.diff(x_true_story)[idx]
    dy = np.diff(y_true_story)[idx]
    dz = np.diff(z_true_story)[idx]

    ax.quiver(
        x_true_story[idx], y_true_story[idx], z_true_story[idx],
        dx, dy, dz,
#         length=0.15,
#         normalize=True,
        color=color_arrows,
        linewidth=lw_arrows,
    )

#     # -------------------------
#     # Axis labels (small, reference only)
#     # -------------------------
#     ax.set_xlabel("z1", labelpad=labelpad)
#     ax.set_ylabel("z2", labelpad=labelpad)
#     ax.set_zlabel("z3", labelpad=labelpad)
#     ax.set_xticks(z_ticks[0])
#     ax.set_yticks(z_ticks[1])
#     ax.set_zticks(z_ticks[2])

#     ax.tick_params(labelsize=labelsize)

    # -------------------------
    # Clean 3D appearance
    # -------------------------
    ax.grid(False)

    # Remove background panes
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_visible(False)
        axis.line.set_alpha(0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    lims = np.max(np.abs(z_trial), axis=0) * scale
    
    print(lims)
    
    ax.quiver(*origin, lims[0], 0, 0, color="black", linewidth=lw_axis)
    ax.quiver(*origin, 0, lims[1], 0, color="black", linewidth=lw_axis)
    ax.quiver(*origin, 0, 0, lims[2], color="black", linewidth=lw_axis)
    
    ax.text(lims[0]*offset + origin[0], origin[1], origin[2], "$z_1$", fontsize=fontsize)
    ax.text(origin[0], lims[1]*offset + origin[1], origin[2], "$z_2$", fontsize=fontsize)
    ax.text(origin[0], origin[1], lims[2]*offset + origin[2], "$z_3$", fontsize=fontsize)

    # Remove pane edges
#     ax.xaxis.pane.set_edgecolor("w")
#     ax.yaxis.pane.set_edgecolor("w")
#     ax.zaxis.pane.set_edgecolor("w")

#     # Make axes lines thinner
#     for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
#         axis.line.set_linewidth(1.2)

    # -------------------------
    # Equal aspect ratio
    # -------------------------
    ranges = np.ptp(z_trial, axis=0)
    ax.set_box_aspect(ranges)

    # -------------------------
    # View angle (change freely)
    # -------------------------
    ax.view_init(elev=elev, azim=azim)

#     plt.tight_layout()
#     fig.subplots_adjust(left=0.1, right=0.90, bottom=0.1, top=0.90)
#     plt.show()
    fig_file = os.path.join(comm_dict["saved_path"], 'trajectory.png')
    plt.savefig(fig_file)
    
    
    
    
def latent_u_ctype(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    t = comm_dict["t"]
    u_dim = dmm.u_dim
    
    data = diff_dict["data"]
    alpha = diff_dict["alpha"]
    axis = diff_dict["axis"]
    type_list = diff_dict["type_list"]
    type_trial = diff_dict["type_trial"]
    RT_show = diff_dict["RT_show"]
    axis = diff_dict["axis"]
    
    RT_cn = data["RT_cn_ordRT"] 
    RT_ws = data["RT_ws_ordRT"] 
    RT_cn = (RT_cn + 56)//tau
    RT_ws = (RT_ws + 56)//tau
    steps = 256//tau
    
    _, _, _, u_cn, u_ws, u_cs = infer_latent(dmm, data, device, u_flag=True)
    samples_cn = u_cn.shape[1]
    samples_ws = u_ws.shape[1]
    
    x_cs = u_cs[:, :, axis[0]]
    x_cn = u_cn[:, :, axis[0]]
    x_ws = u_ws[:, :, axis[0]]

    y_cs = u_cs[:, :, axis[1]]
    y_cn = u_cn[:, :, axis[1]]
    y_ws = u_ws[:, :, axis[1]]
    
    x_RT_cn = x_cn[RT_cn, np.arange(samples_cn)]
    y_RT_cn = y_cn[RT_cn, np.arange(samples_cn)]
    x_RT_ws = x_ws[RT_ws, np.arange(samples_ws)]
    y_RT_ws = y_ws[RT_ws, np.arange(samples_ws)]
    
    x_GO_cn = x_cn[56//tau]
    y_GO_cn = y_cn[56//tau]
    x_GO_ws = x_ws[56//tau]
    y_GO_ws = y_ws[56//tau]
    
    if type_trial=="cn":
        u_trial, RT, q = random_latent_cn_u(dmm, data, tau, device)
        set_cn = data["set_cn_ordRT"]
        MUA = set_cn[q]
        x_RT = u_trial[RT, 0]
        y_RT = u_trial[RT, 1]
    elif type_trial=="cs":
        u_trial, SSD, q = random_latent_cs_u(dmm, data, tau, device)
        set_cs = data["set_cs_ordSSD"]
        MUA = set_cs[q]
        x_SSD = u_trial[SSD, 0]
        y_SSD = u_trial[SSD, 1]

    print(f"traj n.{q}")
        
    x_true_story = u_trial[:, 0]
    y_true_story = u_trial[:, 1]
    x_start = u_trial[0, 0]
    y_start = u_trial[0, 1]
    x_GO = u_trial[56//tau, 0]
    y_GO = u_trial[56//tau, 1]
    
    if u_dim==2:
        f, ax = plt.subplots(figsize = (7, 6))

        color = np.linspace(0, 1, steps)

        # Plot the surface
        if "cn"in type_list:
            ax.plot(x_cn, y_cn, '-', linewidth=2, color='g', alpha = alpha)
        if "ws"in type_list:
            ax.plot(x_ws, y_ws, '-', linewidth=2, color='orange', alpha = alpha)
        if "cs"in type_list:
            ax.plot(x_cs, y_cs, '-', linewidth=2, color='r', alpha = alpha)

        if RT_show:

            #ax.scatter(x_RT_cn, y_RT_cn, s = 15, c = color, cmap = "Blues", alpha = 0.6, label = "cn")
            #ax.scatter(x_RT_ws, y_RT_ws, s = 15, c = RT_ws/RT_cn.max(), cmap = "Purples", alpha = 0.8, label = "ws")
            ax.scatter(x_RT_cn, y_RT_cn, s = 15, c = "b", alpha = 1, label = "cn")
            ax.scatter(x_RT_ws, y_RT_ws, s = 15, c = "purple", alpha = 1, label = "ws")

        ax.plot(x_true_story, y_true_story, '-', linewidth=4, color="brown", alpha = 1, label = "trajectory example")
        ax.scatter(x_GO, y_GO, s = 150, c = "b", alpha = 1, label = "GO")
        if type_trial=="cn":
            ax.scatter(x_RT, y_RT, s = 150, c = 'r', alpha = 1, label = "RT")
        if type_trial=="cs":
            ax.plot(x_SSD, y_SSD, color='red', marker='x', markeredgewidth = 3, markersize = 15, label = "SSD")
            #ax.scatter(x_SSD, y_SSD, s = 150, c = 'r', alpha = 1, label = "SSD")
        ax.scatter(x_start, y_start, s = 150, c = 'y', alpha = 1, label = "start")
        ax.set_xlabel("first latent component")
        ax.set_ylabel("second latent component")
        ax.set_title(f'cs vs ws vs ns, with n.{q} traj')
        ax.legend()
        plt.show()


#         f, ax = plt.subplots(figsize = (8, 6))
#         ax.plot(t*5, MUA.mean(1), c = "green", label = 'mean MUA of the traj above')
#         ax.axvline(56*5, color="black",linestyle="--",label= "GO")
#         if type_trial=="cn":
#             ax.axvline(RT*tau*5, color="blue",linestyle="--",label= "RT")
#         if type_trial=="cs":
#             ax.axvline(SSD*tau*5, color="red",linestyle="--",label= "SSD")
#         ax.set_xlabel('time ($ms$)')
#         ax.set_ylabel('mean MUA activity over channels')
#         ax.legend(fontsize=7.3)
#         plt.show()
    
    
    elif u_dim==3:
        
        f, ax = plt.subplots(figsize = (7, 6))

        # Plot the surface
        if "cn"in type_list:
            ax.plot(x_cn, y_cn, '-', linewidth=2, color='g', alpha = alpha)
        if "ws"in type_list:
            ax.plot(x_ws, y_ws, '-', linewidth=2, color='orange', alpha = alpha)
        if "cs"in type_list:
            ax.plot(x_cs, y_cs, '-', linewidth=2, color='r', alpha = alpha)

        if RT_show:

            #ax.scatter(x_RT_cn, y_RT_cn, s = 15, c = color, cmap = "Blues", alpha = 0.6, label = "cn")
            #ax.scatter(x_RT_ws, y_RT_ws, s = 15, c = RT_ws/RT_cn.max(), cmap = "Purples", alpha = 0.8, label = "ws")
            ax.scatter(x_RT_cn, y_RT_cn, s = 15, c = "b", alpha = 1, label = "cn")
            ax.scatter(x_RT_ws, y_RT_ws, s = 15, c = "purple", alpha = 1, label = "ws")

        ax.plot(x_true_story, y_true_story, '-', linewidth=4, color="brown", alpha = 1, label = "trajectory example")
        ax.scatter(x_GO, y_GO, s = 150, c = "b", alpha = 1, label = "GO")
        if type_trial=="cn":
            ax.scatter(x_RT, y_RT, s = 150, c = 'r', alpha = 1, label = "RT")
        if type_trial=="cs":
            ax.plot(x_SSD, y_SSD, color='red', marker='x', markeredgewidth = 3, markersize = 15, label = "SSD")
            #ax.scatter(x_SSD, y_SSD, s = 150, c = 'r', alpha = 1, label = "SSD")
        ax.scatter(x_start, y_start, s = 150, c = 'y', alpha = 1, label = "start")
        ax.set_xlabel("first latent component")
        ax.set_ylabel("second latent component")
        ax.set_title(f'cs vs ws vs ns, with n.{q} traj')
        ax.legend()
        plt.show()
        
        
        
        z_cs = u_cs[:, :, 2]
        z_cn = u_cn[:, :, 2]
        z_ws = u_ws[:, :, 2]
        
        z_true_story = u_trial[:, 2]
        z_start = u_trial[0, 2]
        z_GO = u_trial[56//tau, 2]
        #z_RT = RT_z[2]
        
        import plotly.graph_objects as go

        fig = go.Figure()

        
        if "cn"in type_list:
            for i in range(x_cn.shape[1]):
                fig.add_trace(go.Scatter3d(
                    x=x_cn[:, i], y=y_cn[:, i], z=z_cn[:, i],
                    mode='lines', line=dict(color='green', width=3), opacity=0.3
                ))
        if "ws"in type_list:
            for i in range(x_ws.shape[1]):
                fig.add_trace(go.Scatter3d(
                    x=x_ws[:, i], y=y_ws[:, i], z=z_ws[:, i],
                    mode='lines', line=dict(color='orange', width=3), opacity=0.3
                ))
        if "cs"in type_list:
            for i in range(x_cs.shape[1]):
                fig.add_trace(go.Scatter3d(
                    x=x_cs[:, i], y=y_cs[:, i], z=z_cs[:, i],
                    mode='lines', line=dict(color='red', width=3), opacity=0.3
                ))

        fig.update_layout(
            scene=dict(
                xaxis_title='z1', yaxis_title='z2', zaxis_title='z3',
                bgcolor='black'
            ),
            width=700, height=700,
            title='cs vs ws vs ns'
        )
        #fig.show()
        return fig


# +
def latent_traj_csession(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    dir_flag = comm_dict["dir_flag"]
    
    session_list = diff_dict["session_list"]
    monkey = diff_dict["monkey"]
    
    if monkey=="Piero":
        data_path_array = ['/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Piero_20131202.npz',
                             '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Piero_20140109.npz',
                             '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Piero_20140116.npz',
                             '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Piero_20140606.npz',  ###
                             '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Piero_20140701.npz',  ###
                             '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Piero_20140922.npz'   ###
                                ]
    elif monkey=="Cornelio":
        data_path_array = ['/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Cornelio_20140424.npz',  
#                        '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Cornelio_20140515.npz', # sessione da 175 trials
#                        '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Cornelio_20140520.npz',  ###
                           '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Cornelio_20140527.npz',  
                           '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Cornelio_20140528.npz',  
#                        '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Cornelio_20140529.npz',  ###
#                        '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Cornelio_20140601.npz',  ###
#                        '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Cornelio_20140606.npz',  ###
                          ]

    sessions = [data_path_array[i] for i in session_list]

    selection ="((y_stop == True) | ((y_stop == False)&(y_reward == True)))"# selezioni tutto il movimento &(y_reward == True)


    data, RT, SSD, direction, session, subject = load_set(sessions, selection)

    if peak_filthers:
        mask = peak_filthers(data)
        data = data[mask, :, :]
        RT = RT[mask]
        SSD = SSD[mask]
        session = session[mask]
        direction = direction[mask]
        subject = subject[mask]
        print("peak_filthered")

    if dir_flag:
        cont_c = one_hot_cont(SSD, direction, tau)
    else:
        cont_c = cont(SSD, tau)

    data = data[:, ::tau, :]  
    data = torch.from_numpy(data).float().to(device).permute(1, 0, 2)
    cont_c = torch.from_numpy(cont_c).float().to(device).permute(1, 0, 2)
    
    z, z_mean, _ = dmm.inference(data, cont_c)
    z = z.cpu().detach().numpy()
    
    return z, session, RT, SSD



def plot_different_sessions(diff_dict, session_list):
    
    z = diff_dict["z"] 
    session = diff_dict["session"] 
    RT = diff_dict["RT"]
    SSD = diff_dict["SSD"]
    type_list = diff_dict["type_list"]
    
    x, y = z[:, :, 0], z[:, :, 1]

    colors = ['r', 'g', 'y', 'b', 'purple', 'black']
    alphas = [0.2, 0.3, 0.2, 0.1, 0.1, 0.2]
    type_masks = {
        'cn': SSD == 0,
        'cs': RT == 0,
        'ws': (RT != 0) & (SSD != 0)
    }
    
    combined_type_mask = np.zeros_like(RT, dtype=bool)
    for trial_type in type_list:
        combined_type_mask |= type_masks[trial_type]

    f, ax = plt.subplots(figsize=(7, 6))

    for i in session_list:
        mask = (session==i) & combined_type_mask
        ax.plot(x[:, mask], y[:, mask], '-', linewidth=2, color=colors[i], alpha=alphas[i])

    ax.set_xlabel("first latent component")
    ax.set_ylabel("second latent component")
# -



def plot_zGO_RTgrad(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z1_min, z1_max, z2_min, z2_max = comm_dict["z_lims"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    points = diff_dict["points"]
    time = diff_dict["time"]
    bins = diff_dict["bins"]
    GO_plot = diff_dict["GO_plot"]
    trial = diff_dict["trial"]
    cmap = diff_dict["cmap"]
    mean_trials = diff_dict["mean_trials"]
    #alpha = diff_dict["alpha"]
    #c_norm = diff_dict["c_norm"]
    
    
    RT_cn = data["RT_cn_ordRT"]
    z_cn, _, _ = infer_latent(dmm, data, device, n_trials)
    z_dim = z_cn.shape[2]
    steps = z_cn.shape[0]
    s = z_cn.shape[1]
    samples = len(RT_cn)
    RT_cn_rep = np.expand_dims(RT_cn, axis=1)
    RT_cn_rep = RT_cn_rep.repeat(n_trials, axis=1).reshape(-1)

    z1_edges = np.linspace(z1_min, z1_max, bins + 1)
    z2_edges = np.linspace(z2_min, z2_max, bins + 1)

    if mean_trials:
        z_cn = z_cn.reshape(steps, samples, n_trials, z_dim)
        z_cn = z_cn.mean(2)
    
    z_GO = z_cn[time//(5*tau)]
    #z_RT = z_cn[(RT_cn + 56)//tau, np.arange(samples)]
    if mean_trials:
        color = RT_cn/RT_cn.max()
    else:
        color = RT_cn_rep/RT_cn_rep.max()
    
    mask = color > 0.4  # Limitiamo il contour plot alla regione RT/RT_max > 0.4
    
    # Creiamo una griglia più fitta per interpolare i dati
    z1_vals = np.linspace(z1_min, z1_max, points)
    z2_vals = np.linspace(z2_min, z2_max, points)
    z1_grid, z2_grid = np.meshgrid(z1_vals, z2_vals)
    
    interpolated = griddata((z_GO[mask, 0], z_GO[mask, 1]), color[mask], (z1_grid, z2_grid), method='cubic')
    
    fig = plt.figure(figsize=(fig_size[0] + 0.5, fig_size[1]))
    gs = gridspec.GridSpec(1, 2, width_ratios=[fig_size[0], 0.5])
    
    ax = plt.subplot(gs[0])
    cax = plt.subplot(gs[1])
    
    ax.set_xlabel("z1", fontsize=font_ax)
    ax.set_ylabel("z2", fontsize=font_ax)
    ax.set_xticks([-1, 2, 5]) 
    ax.set_xticklabels([-1, 2, 5], fontsize=font_tick)
    ax.set_yticks([-2, 1, 4])
    ax.set_yticklabels([-2, 1, 4], fontsize=font_tick)
    
    # Contour plot con livelli continui
    levels = np.linspace(0.4, 1, 10)
    contour = ax.contourf(z1_grid, z2_grid, interpolated, levels=levels, cmap=cmap)
    
    cbar = plt.colorbar(contour, cax=cax)
    cbar.set_ticks([0.4, 0.6, 0.8, 1])
    cbar.set_label('Mean RT/RTmax', fontsize=font_ax)
    cbar.ax.tick_params(labelsize=font_tick)
    
    plt.show()



    RT_cn = data["RT_cn_ordRT"]
    z_cn, _, _ = infer_latent(dmm, data, device, n_trials)
    z_dim = z_cn.shape[2]
    steps = z_cn.shape[0]
    s = z_cn.shape[1]
    samples = len(RT_cn)
    RT_cn_rep = np.expand_dims(RT_cn, axis=1)
    RT_cn_rep = RT_cn_rep.repeat(n_trials, axis=1).reshape(-1)

    z1_edges = np.linspace(z1_min, z1_max, bins + 1)
    z2_edges = np.linspace(z2_min, z2_max, bins + 1)

    if mean_trials:
        z_cn = z_cn.reshape(steps, samples, n_trials, z_dim)
        z_cn = z_cn.mean(2)

    z_GO = z_cn[time//(5*tau)]
    #z_RT = z_cn[(RT_cn + 56)//tau, np.arange(samples)]
    if mean_trials:
        z_RT = z_cn[(RT_cn + 56)//tau, np.arange(samples)]
        color = RT_cn/RT_cn.max()
    else:
        z_RT = z_cn[(RT_cn_rep + 56)//tau, np.arange(s)]
        color = RT_cn_rep/RT_cn_rep.max()

    if GO_plot:
        hist, x_edges, y_edges, binnumber = binned_statistic_2d(z_GO[:, 0], z_GO[:, 1], color, statistic='mean', bins=[z1_edges, z2_edges])
        text = 'GO'
    else:
        hist, x_edges, y_edges, binnumber = binned_statistic_2d(z_RT[:, 0], z_RT[:, 1], color, statistic='mean', bins=[z1_edges, z2_edges])
        text = 'RT'

    # Create a mesh grid for plotting
    z1, z2 = np.meshgrid(z1_edges[:-1] + np.diff(z1_edges)/2, z2_edges[:-1] + np.diff(z2_edges)/2)

    # Plot the density map
    fig = plt.figure(figsize=(fig_size[0]+0.5, fig_size[1]))
    gs = gridspec.GridSpec(1, 2, width_ratios=[fig_size[0], 0.5])  # Plot più largo, colorbar stretta

    ax = plt.subplot(gs[0])
    cax = plt.subplot(gs[1])
    ax.set_xlabel("z1", fontsize=font_ax)
    ax.set_ylabel("z2", fontsize=font_ax)
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]], fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]], fontsize=font_tick)
    im=ax.pcolormesh(z1, z2, hist.T, cmap=cmap, shading='auto')
    #ax.set_title(f'Density Plot of {text} points, coloured from low to high RT true')
    if trial:
        ax.plot(x_true_story, y_true_story, '-', linewidth=2, color='brown', alpha = 0.5)
        n_arrows = 15
        arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
        for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
            dx = x_true_story[i+1] - x_true_story[i]
            dy = y_true_story[i+1] - y_true_story[i]
            ax.arrow(x_true_story[i], y_true_story[i], dx, dy,
                    head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)

        #plt.scatter(x_GO, y_GO, s = 50, c = 'b', alpha = 1, label = "GO")
        #plt.scatter(x_RT, y_RT, s = 50, c = 'r', alpha = 1, label = "RT")
        #plt.scatter(x_start, y_start, s = 50, c = 'y', alpha = 1, label = "start")
        #ax.plot(x_SSD, y_SSD, 'x', c = 'red', marker='x', markeredgewidth = 3, markersize = 15)
        #ax.set_title(f"RT: {SSD:d}")
        #plt.legend(loc="upper right", fontsize=font_leg)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([0, 0.5, 1])  # Specify exact tick locations
    #cbar.set_ticklabels([0, 0.5, 1])  # Custom tick labels
    cbar.set_label('Mean RT/RTmax', fontsize=font_ax)
    cbar.ax.tick_params(labelsize=font_tick)
    #plt.colorbar(im, ax=ax, label='Mean RT/RTmax')
    plt.show()  




from scipy.stats import binned_statistic_2d

def plot_z_cRTtrue(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    time = diff_dict["time"]
    bins = diff_dict["bins"]
    GO_plot = diff_dict["GO_plot"]
    trial = diff_dict["trial"]
    cmap = diff_dict["cmap"]
    axis = diff_dict["axis"]
    mean_trials = diff_dict["mean_trials"]
    show_all = diff_dict["show_all"]
    alpha = diff_dict["alpha"]
    l = diff_dict["l"]
    #c_norm = diff_dict["c_norm"]
    
    
    RT_cn = data["RT_cn_ordRT"]
    z_cn, _, _ = infer_latent(dmm, data, device, n_trials)
    
    steps, s, _ = z_cn.shape
    samples = len(RT_cn)    # s = samples * n_trials

    z1_edges = np.linspace(z_lims[axis[0], 0], z_lims[axis[0], 1], bins + 1)
    z2_edges = np.linspace(z_lims[axis[1], 0], z_lims[axis[1], 1], bins + 1)
    

    if mean_trials:
        z_cn = z_cn.reshape(steps, samples, n_trials, z_dim)
        z_cn = z_cn.mean(2)
        z_RT = z_cn[(RT_cn + 56)//tau, np.arange(samples)]
        color = (RT_cn - RT_cn.min())/(RT_cn.max() - RT_cn.min())
    else:
        RT_cn_rep = np.repeat(RT_cn, n_trials)
        z_RT = z_cn[(RT_cn_rep + 56)//tau, np.arange(s)]
        color = (RT_cn_rep - RT_cn_rep.min())/(RT_cn_rep.max() - RT_cn_rep.min())
    z_GO = z_cn[time//(5*tau)]
    
    z, RT, q = random_latent_cn_traj(dmm, data, tau, device)
    
    print(f"traj n.{q}")
    
    x_true_story = z[:, axis[0]]
    y_true_story = z[:, axis[1]]
    x_start = z[0, axis[0]]
    y_start = z[0, axis[1]]
    x_GO = z[56//tau, axis[0]]
    y_GO = z[56//tau, axis[1]]
    x_RT = z[RT, axis[0]]
    y_RT = z[RT, axis[1]]
    
    if GO_plot:
        hist, x_edges, y_edges, binnumber = binned_statistic_2d(z_GO[:, axis[0]], z_GO[:, axis[1]], color, statistic='mean', bins=[z1_edges, z2_edges])
        text = 'GO'
    else:
        hist, x_edges, y_edges, binnumber = binned_statistic_2d(z_RT[:, axis[0]], z_RT[:, axis[1]], color, statistic='mean', bins=[z1_edges, z2_edges])
        text = 'RT'

    from sklearn.linear_model import LinearRegression
    
    # --- STIMA DIREZIONE ---
    reg = LinearRegression().fit(z_GO, color)
    direction = reg.coef_
    direction /= np.linalg.norm(direction)  # normalizzazione unit vector

    print("Direzione di variazione (aumenta):", direction)
    print("Direzione di diminuzione:", -direction)
        
    # Create a mesh grid for plotting
    z1, z2 = np.meshgrid(z1_edges[:-1] + np.diff(z1_edges)/2, z2_edges[:-1] + np.diff(z2_edges)/2)

    # Plot the density map
    fig = plt.figure(figsize=(fig_size[0]+0.5, fig_size[1]))
    gs = gridspec.GridSpec(1, 2, width_ratios=[fig_size[0], 0.5])  # Plot più largo, colorbar stretta

    ax = plt.subplot(gs[0])
    cax = plt.subplot(gs[1])
    ax.set_xlabel("z1", fontsize=font_ax)
    ax.set_ylabel("z2", fontsize=font_ax)
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]], fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]], fontsize=font_tick)
    if show_all:
        ax.plot(z_cn[:, :, axis[0]], z_cn[:, :, axis[1]], color="green", alpha=alpha)
    im=ax.pcolormesh(z1, z2, hist.T, cmap=cmap, shading='auto')
    # Punto medio per disegnare la direzione
    center = z_GO.mean(axis=0)
    length = l  # lunghezza del vettore per visualizzazione

    # Freccia nella direzione di diminuzione
    ax.arrow(center[axis[0]], center[axis[1]],
              -direction[axis[0]]*length, -direction[axis[1]]*length,
              color='red', width=0.05, head_width=0.1, label='direzione diminuzione')
    #ax.set_title(f'Density Plot of {text} points, coloured from low to high RT true')
    if trial:
        ax.plot(x_true_story, y_true_story, '-', linewidth=2, color='brown', alpha = 0.5)
        n_arrows = 15
        arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
        for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
            dx = x_true_story[i+1] - x_true_story[i]
            dy = y_true_story[i+1] - y_true_story[i]
            ax.arrow(x_true_story[i], y_true_story[i], dx, dy,
                    head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
        
        #plt.scatter(x_GO, y_GO, s = 50, c = 'b', alpha = 1, label = "GO")
        #plt.scatter(x_RT, y_RT, s = 50, c = 'r', alpha = 1, label = "RT")
        #plt.scatter(x_start, y_start, s = 50, c = 'y', alpha = 1, label = "start")
        #ax.plot(x_SSD, y_SSD, 'x', c = 'red', marker='x', markeredgewidth = 3, markersize = 15)
        #ax.set_title(f"RT: {SSD:d}")
        #plt.legend(loc="upper right", fontsize=font_leg)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([0, 0.5, 1])  # Specify exact tick locations
    #cbar.set_ticklabels([0, 0.5, 1])  # Custom tick labels
    cbar.set_label('Mean RT/RTmax', fontsize=font_ax)
    cbar.ax.tick_params(labelsize=font_tick)
    #plt.colorbar(im, ax=ax, label='Mean RT/RTmax')
    plt.show()  

    
import torch
from torch.autograd.functional import jvp

# decoder: callable torch module mapping z (batch? or single) -> x
# z0: tensor shape (latent_dim,), requires_grad=False
# d: unit direction in latent space (latent_dim,)
# l: desired length (scalar)

def decoder_mean(z):   
    x_mean, _ = dmm.generation_x(z)
    return x_mean

def delta_x_via_jvp(dmm, z0, d, l, data=None):
    device = z0.device
    z0 = z0.detach().clone().requires_grad_(True)   # enable autograd        
    delta_z = torch.from_numpy(-d * l).float().to(device)   # se vuoi diminuire lungo d; + per aumentare
    if z0.ndim == 2:
        delta_z = torch.tile(delta_z, (z0.shape[0], 1))
    # jvp expects tuples for inputs/outputs
    y, jvp_out = jvp(decoder_mean, (z0,), (delta_z,))
    jvp_out = jvp_out.cpu().detach().numpy()
    # jvp_out has shape of x (. e.g. data_dim)
    #print(y.shape)
    return jvp_out  # appross delta_x

def plot_cycle_shift_short(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    RT_detector = comm_dict["RT_detector"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    mean = diff_dict["mean"]
    step = diff_dict["step"]
    axis = diff_dict["axis"]
    bins = diff_dict["bins"]
    cmap = diff_dict["cmap"]
    alpha = diff_dict["alpha"]
    stimolate = diff_dict["stimolate"]
    l1_ratio = diff_dict["l1_ratio"]
    alpha_L1 = diff_dict["alpha_L1"]
    n_iter = diff_dict["n_iter"]
    stimulation_steps = diff_dict["stimulation_steps"]
    mean_trials = diff_dict["mean_trials"]
    l = diff_dict["l"]
    f = diff_dict["f"]
    compute = diff_dict["compute"]
    multi_direction = diff_dict["multi_direction"]
    add_residual = diff_dict["add_residual"]
    alpha_point = diff_dict["alpha_point"]
    markersize = diff_dict["markersize"]
    markeredgewidth = diff_dict["markeredgewidth"]
    color_edge = diff_dict["color_edge"]
#     cbar_ticks = diff_dict["cbar_ticks"]
    cmap_stim = diff_dict["cmap_stim"]
    thr = diff_dict["thr"]
#     fraction = diff_dict["fraction"]
    
    RT_cn = data["RT_cn_ordRT"]
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    z_cn, _, _ = infer_latent(dmm, data, device, n_trials)
    
    steps, s, _ = z_cn.shape
    samples = len(RT_cn)    # s = samples * n_trials
    RT_min = 50
    
    if mean_trials:
        z_cn = z_cn.reshape(steps, samples, n_trials, z_dim)
        z_cn = z_cn.mean(2)
        z_RT = z_cn[(RT_cn + 56)//tau, np.arange(samples)]
        color = RT_cn/RT_cn.max()
    else:
        set_cn = set_cn.repeat(n_trials, 1, 1)
        RT_cn_rep = np.repeat(RT_cn, n_trials)
        z_RT = z_cn[(RT_cn_rep + 56)//tau, np.arange(s)]
        color = RT_cn_rep/RT_cn_rep.max()
    z_GO = z_cn[step//tau] # z_76
    
    long_index = int(z_GO.shape[0] * (1-f))
    short_index = int(z_GO.shape[0] * f)
#     print(long_index)
#     print(short_index)
    
    set_cn_long = set_cn[long_index:]
    cont_cn_long = cont_cn[long_index:]
    
    n_long, _, features = set_cn_long.shape
    
    z_GO_long = z_GO[long_index:]
#     z_GO_short = z_GO[:short_index]
    
    trial_true = torch.from_numpy(set_cn_long).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)  # fino a 76 step
    cont_RTlong = torch.from_numpy(cont_cn_long).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.plot(z_cn[:, :, axis[0]], z_cn[:, :, axis[1]], color="green", alpha=alpha)
#     ax.scatter(z_GO_long[:, axis[0]], z_GO_long[:, axis[1]], c="blue", s = 30, alpha=alpha_point, label = 'GO of long RT')
#     ax.scatter(z_GO_short[:, axis[0]], z_GO_short[:, axis[1]], c="red", s = 30, alpha=alpha_point, label = 'GO of short RT')
#     ax.plot(z_GO_short[:, axis[0]].mean(), z_GO_short[:, axis[1]].mean(), c = 'red', marker='*', markeredgewidth = 4, markersize = 20)
#     ax.plot(z_GO_long[:, axis[0]].mean(), z_GO_long[:, axis[1]].mean(), c = 'blue', marker='*', markeredgewidth = 4, markersize = 20)
    
    start_sim = step//tau + 1 - stimulation_steps
    if stimolate:
        text = ""
    else:
        text = "_onech"
        
    if mean:
        text_mean = "_mean"
    else:
        text_mean = ""
    
    if compute:
    
        from sklearn.linear_model import LinearRegression

        # --- STIMA DIREZIONE ---
        reg = LinearRegression().fit(z_GO, color)
        direction = reg.coef_
        direction /= np.linalg.norm(direction)  # normalizzazione unit vector
        delta_z_short = torch.from_numpy(-direction * l).float().to(device)
#         delta_z_long = -delta_z_short
        print("Direzione di variazione (aumenta):", direction)

        delta_z_short = torch.zeros(stimulation_steps, z_dim).to(device)
#         delta_z_long = torch.zeros(stimulation_steps, z_dim).to(device)
        if multi_direction:
            for t in range(stimulation_steps):
                reg = LinearRegression().fit(z_cn[start_sim + t], color)
                direction = reg.coef_
                direction /= np.linalg.norm(direction)  # normalizzazione unit vector
                delta_z_short[t] = torch.from_numpy(-direction * l).float().to(device)
#                 delta_z_long[t] = torch.from_numpy(direction * l).float().to(device)

     
        trial_clone = trial_true[:(step//tau + 1)].clone()
        y_mean, y_logvar = dmm(trial_clone, cont_RTlong[:(step//tau + 1)])
        y_pred = dmm.reparameterization(y_mean, y_logvar)
        if mean:
            y_mean = y_mean.reshape(-1, n_long, n_trials, features).mean(2)
            residual = trial_clone - y_mean.repeat_interleave(n_trials, dim=1)
        else:
            y_pred = y_pred.reshape(-1, n_long, n_trials, features).mean(2)
            residual = trial_clone - y_pred.repeat_interleave(n_trials, dim=1)
            
        trial_modified = trial_true[:(step//tau + 1)].clone() 
        trial_modified_min = trial_true[:(step//tau + 1)].clone()
 
        dx_fix = torch.full((set_cn_long.shape[0]*n_trials,), l).to(device)
        dx_fix_min = torch.full((set_cn_long.shape[0]*n_trials,), 0.0001).to(device)
        dx_array = np.empty([stimulation_steps, 96])
        stim_effect = torch.zeros([stimulation_steps, set_cn_long.shape[0]*n_trials, 96]).to(device)
        stim_effect_min = torch.zeros([stimulation_steps, set_cn_long.shape[0]*n_trials, 96]).to(device)
        for t in range(stimulation_steps):
            print(t)
            # x -> x + dx
            if stimolate:
                #trial_modified, dx = compute_dx_for_last_step_Tikhonov(dmm, trial_modified, cont_RTlong[:(step//tau + 1)], delta_z, alpha = alpha_stim, n_iter=n_iter)
                _, dx = compute_dx_for_last_step_ElasticNet(dmm, 
                                                             trial_modified[:(start_sim + t + 1)], 
                                                             cont_RTlong[:(start_sim + t + 1)], 
                                                             delta_z_short[t] if multi_direction else delta_z_short, 
                                                             alpha_l1=alpha_L1, 
                                                             l1_ratio=l1_ratio, 
                                                             n_iter=1)
                dx_array[t] = dx.mean(0)
                trial_modified[start_sim + t] += torch.from_numpy(dx).float().to(device)
            else:
                trial_modified[start_sim + t, :, 63] += dx_fix
                trial_modified_min[start_sim + t, :, 63] += dx_fix_min
            if t+1 == stimulation_steps:
                continue
            trial_cut = trial_modified.clone()
            # z_t = enc(x+dx)
            z_modified, _, _ = dmm.inference(trial_cut[:(start_sim + t + 1)], cont_RTlong[:(start_sim + t + 1)])
            # z_{t+1} = prop(z_t)
            z_mean_gen, z_cov_gen = dmm.generation_z(z_modified[-1].unsqueeze(0), cont_RTlong[start_sim + t + 1].unsqueeze(0))
            z_gen_last = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
            # x_{t+1} = dec(z_{t+1})
            y_mean, y_logvar = dmm.generation_x(z_gen_last)
            y_pred = dmm.reparameterization(y_mean, y_logvar)
            if mean:
                y_gen = y_mean
            else:
                y_gen = y_pred
            #y_pred = dmm.reparameterization(y_mean, y_logvar)
            trial_modified[start_sim + t + 1] = y_gen[0]
            if add_residual:
                trial_modified[start_sim + t + 1] += residual[start_sim + t + 1]
            stim_effect[t] = trial_modified[start_sim + t + 1] - trial_true[start_sim + t + 1]
            
            trial_cut_min = trial_modified_min.clone()
            # z_t = enc(x+dx)
            z_modified, _, _ = dmm.inference(trial_cut_min[:(start_sim + t + 1)], cont_RTlong[:(start_sim + t + 1)])
            # z_{t+1} = prop(z_t)
            z_mean_gen, z_cov_gen = dmm.generation_z(z_modified[-1].unsqueeze(0), cont_RTlong[start_sim + t + 1].unsqueeze(0))
            z_gen_last = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
            # x_{t+1} = dec(z_{t+1})
            y_mean, y_logvar = dmm.generation_x(z_gen_last)
            y_pred = dmm.reparameterization(y_mean, y_logvar)
            if mean:
                y_gen = y_mean
            else:
                y_gen = y_pred
            #y_pred = dmm.reparameterization(y_mean, y_logvar)
            trial_modified_min[start_sim + t + 1] = y_gen[0]
            if add_residual:
                trial_modified_min[start_sim + t + 1] += residual[start_sim + t + 1]
            stim_effect_min[t] = trial_modified_min[start_sim + t + 1] - trial_true[start_sim + t + 1]

        stim_effect = stim_effect.cpu().detach().numpy()
        stim_effect_min = stim_effect_min.cpu().detach().numpy()
        trial_stim = trial_modified.cpu().detach().numpy()
    
        np.savez(comm_dict["saved_path"] + f"/shorter_RT_{l}{text}{text_mean}.npz", trial_stim=trial_stim, stim_effect=stim_effect, stim_effect_min=stim_effect_min, dx_array=dx_array)
    else:
        with np.load(comm_dict["saved_path"] + f"/shorter_RT_{l}{text}{text_mean}.npz") as loaded_file:
            trial_stim = loaded_file["trial_stim"]
            dx_array = loaded_file["dx_array"]
            if not stimolate:
                stim_effect = loaded_file["stim_effect"]
                stim_effect_min = loaded_file["stim_effect_min"]
                diff_stim = stim_effect - stim_effect_min
            
        trial_modified = torch.from_numpy(trial_stim).float().to(device)
    
    k = math.ceil(math.sqrt(step//tau))
    # Plot the density map
#     fig, ax = plt.subplots(k, k, figsize=(16, 15))
    
    vmin = (trial_stim.mean(1)).min()
    vmax = (trial_stim.mean(1)).max()

    trial_stim = channel2grid(trial_stim)
#     for i in range(k):
#         for j in range(k):
#             if k*i + j > step//tau:
#                 continue
                
#             if k*i + j == start_sim:
#                 color_edge = 'g'
#             shift_plot = ax[i, j].imshow(trial_stim[k*i+j].mean(0), 
#                                   aspect='equal',
#                                   cmap=cmap,
#                                   interpolation='nearest',
#                                   vmin=vmin, vmax=vmax)
#             ax[i, j].plot(0, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
#             ax[i, j].plot(9, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
#             ax[i, j].plot(0, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
#             ax[i, j].plot(9, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
#             ax[i, j].set_xticks([])
#             ax[i, j].set_yticks([])
#             ax[i, j].grid(True, which='major', color='w', alpha=0.2)
#     #ax.set_title(f'Neural stimulus to shorten the RT')
#     plt.colorbar(shift_plot, ax=ax, fraction=0.025, pad=0.06)    
    
    
    m = math.ceil(math.sqrt(stimulation_steps))
    # Plot the density map
    fig, ax = plt.subplots(m, m, figsize=(16, 15))
    
    if stimolate:

        vmin = dx_array.min()
        vmax = dx_array.max()

        color_edge = 'r'
        dx_array = channel2grid(dx_array)
        for i in range(m):
            for j in range(m):
                if m*i + j >= stimulation_steps:
                    continue
                last_stim = dx_array[m*i+j].copy()

                shift_plot = ax[i, j].imshow(dx_array[m*i+j], 
                                      aspect='equal',
                                      cmap=cmap,
                                      interpolation='nearest',
                                      vmin=vmin, vmax=vmax)
                ax[i, j].plot(0, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
                ax[i, j].plot(9, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
                ax[i, j].plot(0, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
                ax[i, j].plot(9, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                ax[i, j].grid(True, which='major', color='w', alpha=0.2)
        #ax.set_title(f'Neural stimulus to shorten the RT')
        cbar = plt.colorbar(shift_plot, ax=ax, fraction=0.025, pad=0.06)    
#         cbar.set_ticks(cbar_ticks)  # Specify exact tick locations
        #cbar.set_ticklabels([0, 0.5, 1])  # Custom tick labels
#         cbar.set_label('Stimulation')#, fontsize=font_ax)
#         cbar.ax.tick_params(labelsize=font_tick)

        last_stim[last_stim < thr] = 0

    else:
        
        mean_stim_effect = diff_stim.mean(1)
        vmin = mean_stim_effect.min()
        vmax = mean_stim_effect.max()

        color_edge = 'r'
        mean_stim_effect = channel2grid(mean_stim_effect)
        for i in range(m):
            for j in range(m):
                if m*i + j >= stimulation_steps:
                    continue

                shift_plot = ax[i, j].imshow(mean_stim_effect[m*i+j], 
                                      aspect='equal',
                                      cmap=cmap,
                                      interpolation='nearest',
                                      vmin=vmin, vmax=vmax)
                ax[i, j].plot(0, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
                ax[i, j].plot(9, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
                ax[i, j].plot(0, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
                ax[i, j].plot(9, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                ax[i, j].grid(True, which='major', color='w', alpha=0.2)
        #ax.set_title(f'Neural stimulus to shorten the RT')
        plt.colorbar(shift_plot, ax=ax, fraction=0.025, pad=0.06)    


    teacher = step//tau + 1
    
    make_RThist(comm_dict, diff_dict, last_stim, trial_modified, trial_true, cont_RTlong, z_cn, l, 'shorter')
    
    
    
    
def plot_cycle_shift_long(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    RT_detector = comm_dict["RT_detector"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    mean = diff_dict["mean"]
    step = diff_dict["step"]
    axis = diff_dict["axis"]
    bins = diff_dict["bins"]
    cmap = diff_dict["cmap"]
    alpha = diff_dict["alpha"]
    stimolate = diff_dict["stimolate"]
    l1_ratio = diff_dict["l1_ratio"]
    alpha_L1 = diff_dict["alpha_L1"]
    n_iter = diff_dict["n_iter"]
    stimulation_steps = diff_dict["stimulation_steps"]
    mean_trials = diff_dict["mean_trials"]
    l = diff_dict["l"]
    f = diff_dict["f"]
    compute = diff_dict["compute"]
    multi_direction = diff_dict["multi_direction"]
    add_residual = diff_dict["add_residual"]
    alpha_point = diff_dict["alpha_point"]
#     cbar_ticks = diff_dict["cbar_ticks"]
    markersize = diff_dict["markersize"]
    markeredgewidth = diff_dict["markeredgewidth"]
    color_edge = diff_dict["color_edge"]
    cmap_stim = diff_dict["cmap_stim"]
    thr = diff_dict["thr"]
#     fraction = diff_dict["fraction"]
    
    RT_cn = data["RT_cn_ordRT"]
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    z_cn, _, _ = infer_latent(dmm, data, device, n_trials)
    
    steps, s, _ = z_cn.shape
    samples = len(RT_cn)    # s = samples * n_trials
    RT_min = 50
    
    if mean_trials:
        z_cn = z_cn.reshape(steps, samples, n_trials, z_dim)
        z_cn = z_cn.mean(2)
        z_RT = z_cn[(RT_cn + 56)//tau, np.arange(samples)]
        color = RT_cn/RT_cn.max()
    else:
        set_cn = set_cn.repeat(n_trials, 1, 1)
        RT_cn_rep = np.repeat(RT_cn, n_trials)
        z_RT = z_cn[(RT_cn_rep + 56)//tau, np.arange(s)]
        color = RT_cn_rep/RT_cn_rep.max()
    z_GO = z_cn[step//tau]
    
    long_index = int(z_GO.shape[0] // (1/(1-f)))
    short_index = int(z_GO.shape[0] // (1/f))
#     print(long_index)
#     print(short_index)
    
    set_cn_short = set_cn[:short_index]
    cont_cn_short = cont_cn[:short_index]
    
    n_short, _, features = set_cn_short.shape
    z_GO_short = z_GO[:short_index]
    
    trial_true = torch.from_numpy(set_cn_short).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)  # fino a 76 step
    cont_RTshort = torch.from_numpy(cont_cn_short).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    
#     trial_true = torch.from_numpy(set_cn_short[:, :(step//tau + 1)]).float().to(device).permute(1, 0, 2)  #[no_peak_mask]
#     cont_RTshort = torch.from_numpy(cont_cn_short).float().to(device).permute(1, 0, 2)
#     trial_modified = trial_true.clone()

#     trial_modified = trial_modified.repeat(1, n_trials, 1)
#     trial_true = trial_true.repeat(1, n_trials, 1)
#     cont_RTlong = cont_RTlong.repeat(1, n_trials, 1)  #.repeat(1, int(1/f), 1)
    
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.plot(z_cn[:, :, axis[0]], z_cn[:, :, axis[1]], color="green", alpha=alpha)
#     ax.scatter(z_GO_long[:, axis[0]], z_GO_long[:, axis[1]], c="blue", s = 30, alpha=alpha_point, label = 'GO of long RT')
#     ax.scatter(z_GO_short[:, axis[0]], z_GO_short[:, axis[1]], c="red", s = 30, alpha=alpha_point, label = 'GO of short RT')
#     ax.plot(z_GO_short[:, axis[0]].mean(), z_GO_short[:, axis[1]].mean(), c = 'red', marker='*', markeredgewidth = 4, markersize = 20)
#     ax.plot(z_GO_long[:, axis[0]].mean(), z_GO_long[:, axis[1]].mean(), c = 'blue', marker='*', markeredgewidth = 4, markersize = 20)
    
    start_sim = step//tau + 1 - stimulation_steps
    if stimolate:
        text = ""
    else:
        text = "_onech"
    
    if compute:
        
        from sklearn.linear_model import LinearRegression

        # --- STIMA DIREZIONE ---
        reg = LinearRegression().fit(z_GO, color)
        direction = reg.coef_
        direction /= np.linalg.norm(direction)  # normalizzazione unit vector
        delta_z_long = torch.from_numpy(direction * l).float().to(device)
        print("Direzione di variazione (aumenta):", direction)

        delta_z_long = torch.zeros(stimulation_steps, z_dim).to(device)
        if multi_direction:
            for t in range(stimulation_steps):
                reg = LinearRegression().fit(z_cn[start_sim + t], color)
                direction = reg.coef_
                direction /= np.linalg.norm(direction)  # normalizzazione unit vector
                delta_z_long[t] = torch.from_numpy(direction * l).float().to(device) 
                
        trial_clone = trial_true[:(step//tau + 1)].clone()
        y_mean, y_logvar = dmm(trial_clone, cont_RTshort[:(step//tau + 1)])
        y_pred = dmm.reparameterization(y_mean, y_logvar)
        if mean:
            y_mean = y_mean.reshape(-1, n_short, n_trials, features).mean(2)
            residual = trial_clone - y_mean.repeat_interleave(n_trials, dim=1)
        else:
            y_pred = y_pred.reshape(-1, n_short, n_trials, features).mean(2)
            residual = trial_clone - y_pred.repeat_interleave(n_trials, dim=1)
            
        trial_modified = trial_true[:(step//tau + 1)].clone() 
        trial_modified_min = trial_true[:(step//tau + 1)].clone()
 
        dx_fix = torch.full((set_cn_short.shape[0]*n_trials,), l).to(device)
        dx_fix_min = torch.full((set_cn_short.shape[0]*n_trials,), 0.001).to(device)
        dx_array = np.empty([stimulation_steps, 96])
        stim_effect = torch.zeros([stimulation_steps, set_cn_short.shape[0]*n_trials, 96]).to(device)
        stim_effect_min = torch.zeros([stimulation_steps, set_cn_short.shape[0]*n_trials, 96]).to(device)
        for t in range(stimulation_steps):
            print(t)
            # x -> x + dx
            if stimolate:
                #trial_modified, dx = compute_dx_for_last_step_Tikhonov(dmm, trial_modified, cont_RTlong[:(step//tau + 1)], delta_z, alpha = alpha_stim, n_iter=n_iter)
                _, dx = compute_dx_for_last_step_ElasticNet(dmm, 
                                                             trial_modified[:(start_sim + t + 1)], 
                                                             cont_RTshort[:(start_sim + t + 1)], 
                                                             delta_z_long[t] if multi_direction else delta_z_long, 
                                                             alpha_l1=alpha_L1, 
                                                             l1_ratio=l1_ratio, 
                                                             n_iter=1)
                dx_array[t] = dx.mean(0)
                trial_modified[start_sim + t] += torch.from_numpy(dx).float().to(device)
            else:
                trial_modified[start_sim + t, :, 63] += dx_fix
                trial_modified_min[start_sim + t, :, 63] += dx_fix_min
            if t+1 == stimulation_steps:
                continue
            trial_cut = trial_modified.clone()
            # z_t = enc(x+dx)
            z_modified, _, _ = dmm.inference(trial_cut[:(start_sim + t + 1)], cont_RTshort[:(start_sim + t + 1)])
            # z_{t+1} = prop(z_t)
            z_mean_gen, z_cov_gen = dmm.generation_z(z_modified[-1].unsqueeze(0), cont_RTshort[start_sim + t + 1].unsqueeze(0))
            z_gen_last = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
            # x_{t+1} = dec(z_{t+1})
            y_mean, y_logvar = dmm.generation_x(z_gen_last)
            y_pred = dmm.reparameterization(y_mean, y_logvar)
            if mean:
                y_gen = y_mean
            else:
                y_gen = y_pred
            #y_pred = dmm.reparameterization(y_mean, y_logvar)
            trial_modified[start_sim + t + 1] = y_gen[0]
            if add_residual:
                trial_modified[start_sim + t + 1] += residual[start_sim + t + 1]
            stim_effect[t] = trial_modified[start_sim + t + 1] - trial_true[start_sim + t + 1]
            
            
            trial_cut_min = trial_modified_min.clone()
            # z_t = enc(x+dx)
            z_modified, _, _ = dmm.inference(trial_cut_min[:(start_sim + t + 1)], cont_RTshort[:(start_sim + t + 1)])
            # z_{t+1} = prop(z_t)
            z_mean_gen, z_cov_gen = dmm.generation_z(z_modified[-1].unsqueeze(0), cont_RTshort[start_sim + t + 1].unsqueeze(0))
            z_gen_last = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
            # x_{t+1} = dec(z_{t+1})
            y_mean, y_logvar = dmm.generation_x(z_gen_last)
            y_pred = dmm.reparameterization(y_mean, y_logvar)
            if mean:
                y_gen = y_mean
            else:
                y_gen = y_pred
            #y_pred = dmm.reparameterization(y_mean, y_logvar)
            trial_modified_min[start_sim + t + 1] = y_gen[0]
            if add_residual:
                trial_modified_min[start_sim + t + 1] += residual[start_sim + t + 1]
            stim_effect_min[t] = trial_modified_min[start_sim + t + 1] - trial_true[start_sim + t + 1]

        stim_effect = stim_effect.cpu().detach().numpy()
        stim_effect_min = stim_effect_min.cpu().detach().numpy()
        trial_stim = trial_modified.cpu().detach().numpy()
    
        np.savez(comm_dict["saved_path"] + f"/longer_RT_{l}{text}.npz", trial_stim = trial_stim, stim_effect_min=stim_effect_min, stim_effect = stim_effect, dx_array = dx_array)
    else:
        with np.load(comm_dict["saved_path"] + f"/longer_RT_{l}{text}.npz") as loaded_file:
            trial_stim = loaded_file["trial_stim"]
            dx_array = loaded_file["dx_array"]
            if not stimolate:
                stim_effect = loaded_file["stim_effect"]
                stim_effect_min = loaded_file["stim_effect_min"]
                diff_stim = stim_effect - stim_effect_min
                
        trial_modified = torch.from_numpy(trial_stim).float().to(device)
    
    k = math.ceil(math.sqrt(step//tau))
    # Plot the density map
#     fig, ax = plt.subplots(k, k, figsize=(16, 15))
    
    vmin = (trial_stim.mean(1)).min()
    vmax = (trial_stim.mean(1)).max()

    trial_stim = channel2grid(trial_stim)
#     for i in range(k):
#         for j in range(k):
#             if k*i + j > step//tau:
#                 continue
                
#             if k*i + j == start_sim:
#                 color_edge = 'g'
#             shift_plot = ax[i, j].imshow(trial_stim[k*i+j].mean(0), 
#                                   aspect='equal',
#                                   cmap=cmap,
#                                   interpolation='nearest',
#                                   vmin=vmin, vmax=vmax)
#             ax[i, j].plot(0, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
#             ax[i, j].plot(9, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
#             ax[i, j].plot(0, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
#             ax[i, j].plot(9, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
#             ax[i, j].set_xticks([])
#             ax[i, j].set_yticks([])
#             ax[i, j].grid(True, which='major', color='w', alpha=0.2)
#             ax[i, j].spines['top'].set_visible(True)
#             ax[i, j].spines['right'].set_visible(True)
#     #ax.set_title(f'Neural stimulus to shorten the RT')
#     plt.colorbar(shift_plot, ax=ax, fraction=0.025, pad=0.06)    
    
    m = math.ceil(math.sqrt(stimulation_steps))
    # Plot the density map
    fig, ax = plt.subplots(m, m, figsize=(16, 15))
    
    if stimolate:

        vmin = dx_array.min()
        vmax = dx_array.max()

        color_edge = 'r'
        dx_array = channel2grid(dx_array)
        for i in range(m):
            for j in range(m):
                if m*i + j >= stimulation_steps:
                    continue

                last_stim = dx_array[m*i+j].copy()
                    
                shift_plot = ax[i, j].imshow(dx_array[m*i+j], 
                                      aspect='equal',
                                      cmap=cmap,
                                      interpolation='nearest',
                                      vmin=vmin, vmax=vmax)
                ax[i, j].plot(0, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
                ax[i, j].plot(9, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
                ax[i, j].plot(0, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
                ax[i, j].plot(9, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                ax[i, j].grid(True, which='major', color='w', alpha=0.2)
        #ax.set_title(f'Neural stimulus to shorten the RT')
        cbar = plt.colorbar(shift_plot, ax=ax, fraction=0.025, pad=0.06)    
#         cbar.set_ticks(cbar_ticks)  # Specify exact tick locations
        #cbar.set_ticklabels([0, 0.5, 1])  # Custom tick labels
#         cbar.set_label('Stimulation')#, fontsize=font_ax)
#         cbar.ax.tick_params(labelsize=font_tick)
    
        last_stim[last_stim > thr] = 0

    else:
        
        mean_stim_effect = diff_stim.mean(1)
        vmin = mean_stim_effect.min()
        vmax = mean_stim_effect.max()

        color_edge = 'r'
        mean_stim_effect = channel2grid(mean_stim_effect)
        for i in range(m):
            for j in range(m):
                if m*i + j >= stimulation_steps:
                    continue

                shift_plot = ax[i, j].imshow(mean_stim_effect[m*i+j], 
                                      aspect='equal',
                                      cmap=cmap,
                                      interpolation='nearest',
                                      vmin=vmin, vmax=vmax)
                ax[i, j].plot(0, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
                ax[i, j].plot(9, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
                ax[i, j].plot(0, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
                ax[i, j].plot(9, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                ax[i, j].grid(True, which='major', color='w', alpha=0.2)
        #ax.set_title(f'Neural stimulus to shorten the RT')
        plt.colorbar(shift_plot, ax=ax, fraction=0.025, pad=0.06)  


    teacher = step//tau + 1
    
    make_RThist(comm_dict, diff_dict, last_stim, trial_modified, trial_true, cont_RTshort, z_cn, l, 'longer')
    
    
    


def plot_channel_shift(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    ar = comm_dict["ar"]
    RT_detector = comm_dict["RT_detector"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    step = diff_dict["step"]
    bins = diff_dict["bins"]
    cmap = diff_dict["cmap"]
    alpha = diff_dict["alpha"]
    mean_trials = diff_dict["mean_trials"]
    l = diff_dict["l"]
    f = diff_dict["f"]
    axis = diff_dict["axis"]
    alpha_point = diff_dict["alpha_point"]
    markersize = 15
    markeredgewidth = 3
    
    RT_cn = data["RT_cn_ordRT"]
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    z_cn, _, _ = infer_latent(dmm, data, device, n_trials)
    
    steps, s, _ = z_cn.shape
    samples = len(RT_cn)    # s = samples * n_trials
    RT_min = 50
    
    if mean_trials:
        z_cn = z_cn.reshape(steps, samples, n_trials, z_dim)
        z_cn = z_cn.mean(2)
        z_RT = z_cn[(RT_cn + 56)//tau, np.arange(samples)]
        color = RT_cn/RT_cn.max()
    else:
        set_cn = set_cn.repeat(n_trials, 1, 1)
        RT_cn_rep = np.repeat(RT_cn, n_trials)
        z_RT = z_cn[(RT_cn_rep + 56)//tau, np.arange(s)]
        color = RT_cn_rep/RT_cn_rep.max()
    z_GO = z_cn[step//tau]
    
    long_index = int(z_GO.shape[0] // (1/(1-f)))
    short_index = int(z_GO.shape[0] // (1/f))
#     print(long_index)
#     print(short_index)
    
    set_cn_long = set_cn[long_index:]
    cont_cn_long = cont_cn[long_index:]
    RT_cn_long = RT_cn[long_index:]
    z_GO_long = z_GO[long_index:]
    z_GO_short = z_GO[:short_index]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(z_cn[:, :, axis[0]], z_cn[:, :, axis[1]], color="green", alpha=alpha)
    ax.scatter(z_GO_long[:, axis[0]], z_GO_long[:, axis[1]], c="blue", s = 30, alpha=alpha_point, label = 'GO of long RT')
    ax.scatter(z_GO_short[:, axis[0]], z_GO_short[:, axis[1]], c="red", s = 30, alpha=alpha_point, label = 'GO of short RT')
    ax.plot(z_GO_short[:, axis[0]].mean(), z_GO_short[:, axis[1]].mean(), c = 'red', marker='*', markeredgewidth = 4, markersize = 20)
    ax.plot(z_GO_long[:, axis[0]].mean(), z_GO_long[:, axis[1]].mean(), c = 'blue', marker='*', markeredgewidth = 4, markersize = 20)
    
    trial_true = torch.from_numpy(set_cn_long[:, :(step//tau + 1)]).float().to(device).permute(1, 0, 2)  #[no_peak_mask]
    cont_RTlong = torch.from_numpy(cont_cn_long).float().to(device).permute(1, 0, 2)
    trial_modified = trial_true.clone()
    
    dx = torch.full((set_cn_long.shape[0],), l).to(device)
    for t in range(step//tau + 1):
        trial_modified[t, :, 63] = trial_true[t, :, 63] + t*dx
        
    trial_stim = trial_modified.cpu().detach().numpy()
    
    k = math.ceil(math.sqrt(step//tau))
    # Plot the density map
    fig, ax = plt.subplots(k, k, figsize=(16, 15))
    
    vmin = trial_stim.min()
    vmax = trial_stim.max()

    
    trial_stim = channel2grid(trial_stim)
    for i in range(k):
        for j in range(k):
            if k*i + j >= step//tau:
                continue
            shift_plot = ax[i, j].imshow(trial_stim[k*i+j].mean(0), 
                                  aspect='equal',
                                  cmap=cmap,
                                  interpolation='nearest',
                                  vmin=vmin, vmax=vmax)
            ax[i, j].plot(0, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
            ax[i, j].plot(9, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
            ax[i, j].plot(0, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
            ax[i, j].plot(9, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].grid(True, which='major', color='w', alpha=0.2)
    #ax.set_title(f'Neural stimulus to shorten the RT')
    plt.colorbar(shift_plot, ax=ax, fraction=0.025, pad=0.06)    
                                                           
    trial_modified = trial_modified.repeat(1, n_trials, 1)
    trial_true = trial_true.repeat(1, n_trials, 1)
    cont_RTlong = cont_RTlong.repeat(1, n_trials, 1)  #.repeat(1, int(1/f), 1)

    teacher = step//tau + 1
    
    make_RThist(comm_dict, diff_dict, trial_modified, trial_true, cont_RTlong, z_cn)
    
    
def plot_z_shift(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    ar = comm_dict["ar"]
    RT_detector = comm_dict["RT_detector"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    step = diff_dict["step"]
    bins = diff_dict["bins"]
    cmap = diff_dict["cmap"]
    q = diff_dict["q"]
    no_peak_lim = diff_dict["no_peak_lim"]
    alpha = diff_dict["alpha"]
    stimolate = diff_dict["stimolate"]
    alpha_stim = diff_dict["alpha_stim"]
    alpha_L1 = diff_dict["alpha_L1"]
    n_iter = diff_dict["n_iter"]
    mean_trials = diff_dict["mean_trials"]
    l = diff_dict["l"]
    l1_ratio = diff_dict["l1_ratio"]
    f = diff_dict["f"]
    axis = diff_dict["axis"]
    alpha_point = diff_dict["alpha_point"]
    markersize = 15
    markeredgewidth = 3
    
    RT_cn = data["RT_cn_ordRT"]
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    z_cn, _, _ = infer_latent(dmm, data, device, n_trials)
    
    steps, s, _ = z_cn.shape
    samples = len(RT_cn)    # s = samples * n_trials
    RT_min = 50
    
#     z1_edges = np.linspace(z1_min, z1_max, bins + 1)
#     z2_edges = np.linspace(z2_min, z2_max, bins + 1)

    if mean_trials:
        z_cn = z_cn.reshape(steps, samples, n_trials, z_dim)
        z_cn = z_cn.mean(2)
        z_RT = z_cn[(RT_cn + 56)//tau, np.arange(samples)]
        color = RT_cn/RT_cn.max()
    else:
        set_cn = set_cn.repeat(n_trials, 1, 1)
        RT_cn_rep = np.repeat(RT_cn, n_trials)
        z_RT = z_cn[(RT_cn_rep + 56)//tau, np.arange(s)]
        color = RT_cn_rep/RT_cn_rep.max()
    z_GO = z_cn[step//tau]
    
    long_index = int(z_GO.shape[0] // (1/(1-f)))
    short_index = int(z_GO.shape[0] // (1/f))
#     print(long_index)
#     print(short_index)
    
    set_cn_long = set_cn[long_index:]
    cont_cn_long = cont_cn[long_index:]
    RT_cn_long = RT_cn[long_index:]
    z_GO_long = z_GO[long_index:]
    z_GO_short = z_GO[:short_index]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(z_cn[:, :, axis[0]], z_cn[:, :, axis[1]], color="green", alpha=alpha)
    ax.scatter(z_GO_long[:, axis[0]], z_GO_long[:, axis[1]], c="blue", s = 30, alpha=alpha_point, label = 'GO of long RT')
    ax.scatter(z_GO_short[:, axis[0]], z_GO_short[:, axis[1]], c="red", s = 30, alpha=alpha_point, label = 'GO of short RT')
    ax.plot(z_GO_short[:, axis[0]].mean(), z_GO_short[:, axis[1]].mean(), c = 'red', marker='*', markeredgewidth = 4, markersize = 20)
    ax.plot(z_GO_long[:, axis[0]].mean(), z_GO_long[:, axis[1]].mean(), c = 'blue', marker='*', markeredgewidth = 4, markersize = 20)

    from sklearn.linear_model import LinearRegression
    
    # --- STIMA DIREZIONE ---
    reg = LinearRegression().fit(z_GO, color)
    direction = reg.coef_
    direction /= np.linalg.norm(direction)  # normalizzazione unit vector
    delta_z = torch.from_numpy(-direction * l).float().to(device)
    print("Direzione di variazione (aumenta):", direction)
    
    trial_true = torch.from_numpy(set_cn_long[:, :(step//tau + 1)]).float().to(device).permute(1, 0, 2)  #[no_peak_mask]
    cont_RTlong = torch.from_numpy(cont_cn_long).float().to(device).permute(1, 0, 2)
    trial_modified = trial_true.clone()
    
    if single_stim:
        dx = torch.full((set_cn_long.shape[0]), l).to(device)
        for t in range():
            trial_modified[t, :, 63] = trial_true[t, :, 63] + t*dx
    
    if stimolate:
        #trial_modified, dx = compute_dx_for_last_step_Tikhonov(dmm, trial_modified, cont_RTlong[:(step//tau + 1)], delta_z, alpha = alpha_stim, n_iter=n_iter)
        trial_modified, dx = compute_dx_for_last_step_ElasticNet(dmm, trial_modified, cont_RTlong[:(step//tau + 1)], delta_z, alpha_l1=alpha_L1, l1_ratio=l1_ratio, n_iter=1)
        
    # Plot the density map
    fig, ax = plt.subplots(figsize=(6, 6))
    
    vmin = dx.min()
    vmax = dx.max()
    
    dx = channel2grid(dx)
    shift_plot = ax.imshow(dx.mean(axis=0), 
                          aspect='equal',
                          cmap=cmap,
                          interpolation='nearest',
                          vmin=vmin, vmax=vmax)
    ax.plot(0, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
    ax.plot(9, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
    ax.plot(0, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
    ax.plot(9, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, which='major', color='w', alpha=0.2)
    ax.set_title(f'Neural stimulus to shorten the RT')
    plt.colorbar(shift_plot, ax=ax, fraction=0.025, pad=0.06)  
                                                           
    trial_modified = trial_modified.repeat(1, n_trials, 1)
    trial_true = trial_true.repeat(1, n_trials, 1)
    cont_RTlong = cont_RTlong.repeat(1, n_trials, 1)  #.repeat(1, int(1/f), 1)
    
    teacher = step//tau + 1
    
    make_RThist(comm_dict, diff_dict, trial_modified, trial_true, cont_RTlong, z_cn)
    
    
    if mean_y:
        RT_output = RT_detector(z_teach.permute(1, 0, 2))
        RT_rec = prob_to_RT(RT_output, tau) 
        RT_rec = RT_rec.reshape(s, n_trials)
        RT_estimate = RT_rec.mean(1).astype(int)
        y_mean, y_logvar = dmm.generation_x(z_teach)
        y_pred = dmm.reparameterization(y_mean, y_logvar)
        y_pred = y_pred.reshape(steps, s, n_trials, 96)
        y_pred = y_pred.mean(2)
    else:
        z_teach= z_teach.reshape(steps, s, n_trials, z_dim)
        z_teach = z_teach.mean(2)
        RT_output = RT_detector(z_teach.permute(1, 0, 2))
        RT_estimate = prob_to_RT(RT_output, tau) 
        y_mean, y_logvar = dmm.generation_x(z_teach)
        y_pred = dmm.reparameterization(y_mean, y_logvar)
    y_pred = y_pred.permute(1, 0, 2).cpu().detach().numpy()
    MUA_pred = y_pred.mean(2)
    RT_estimate = np.argmax(MUA_pred[:, ((RT_min+56)//tau):], axis=1)
    RT_est_sort = np.argsort(RT_estimate)

    num_bins = 25
    min_value = min(RT_true.min(), RT_gen_true.min())
    max_value = max(RT_true.max(), RT_gen_true.max())
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))
    ax1.hist(RT_true, bins=bin_edges, alpha = 0.5, density=True, color='skyblue', edgecolor='black', label = "true RT")
    ax1.axvline(RT_true.mean(), color="skyblue",linestyle="--",label= "mean true RT")
    ax1.hist(RT_gen_true, bins=bin_edges, alpha = 0.5, density=True, color='red', edgecolor='black', label = "gen RT")
    ax1.axvline(RT_gen_true.mean(), color="red",linestyle="--",label= "mean gen RT")
    y_max = int(ax1.get_ylim()[1])
    x_min, x_max = ax1.get_xlim()
    delta_x = x_max - x_min
    ax1.set_xticks([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))])
    ax1.set_xticklabels([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))], fontsize=font_tick)
    ax1.set_yticks([0, y_max//2, y_max])
    ax1.set_yticklabels([0, y_max//2, y_max], fontsize=font_tick)
    # Add labels and title
    ax1.set_xlabel('Reaction Time ($ms$)', fontsize=font_ax)
    ax1.set_ylabel('# of trials', fontsize=font_ax)
    ax1.set_title(f"Histograms of true and predicted RTs generated from GO stimulation")
    ax1.legend(fontsize=font_leg)
    
    #fig, ax = plt.subplots(figsize = fig_size)
    ax2.hist(RT_true, bins=bin_edges, cumulative=True, density=True, alpha = 0.5, color='skyblue', edgecolor='black', label = "true RT")
    ax2.hist(RT_gen_true, bins=bin_edges, cumulative=True, density=True, alpha = 0.5, color='red', edgecolor='black', label = "simulated RT")
    # Add labels and title
    y_max = int(ax2.get_ylim()[1])
    x_min, x_max = ax2.get_xlim()
    delta_x = x_max - x_min
    ax2.set_xticks([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))])
    ax2.set_xticklabels([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))], fontsize=font_tick)
    ax2.set_yticks([0, y_max//2, y_max])
    ax2.set_yticklabels([0, y_max//2, y_max], fontsize=font_tick)
    ax2.set_xlabel('Reaction Time ($ms$)', fontsize=font_ax)
    ax2.set_ylabel('# of trials', fontsize=font_ax)
    ax2.set_title(f"Cumulative Histograms of true and predicted RTs generated from GO stimulation")
    ax2.legend(fontsize=font_leg)

    num_bins = 25
    min_value = min(RT_true.min(), RT_gen.min())
    max_value = max(RT_true.max(), RT_gen.max())
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))
    ax1.hist(RT_true, bins=bin_edges, alpha = 0.5, density=True, color='skyblue', edgecolor='black', label = "true RT")
    ax1.axvline(RT_true.mean(), color="skyblue",linestyle="--",label= "mean true RT")
    ax1.hist(RT_gen, bins=bin_edges, alpha = 0.5, density=True, color='red', edgecolor='black', label = "gen RT")
    ax1.axvline(RT_gen.mean(), color="red",linestyle="--",label= "mean gen RT")
    y_max = int(ax1.get_ylim()[1])
    x_min, x_max = ax1.get_xlim()
    delta_x = x_max - x_min
    ax1.set_xticks([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))])
    ax1.set_xticklabels([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))], fontsize=font_tick)
    ax1.set_yticks([0, y_max//2, y_max])
    ax1.set_yticklabels([0, y_max//2, y_max], fontsize=font_tick)
    # Add labels and title
    ax1.set_xlabel('Reaction Time ($ms$)', fontsize=font_ax)
    ax1.set_ylabel('# of trials', fontsize=font_ax)
    ax1.set_title(f"Histograms of true and predicted RTs generated from GO stimulation")
    ax1.legend(fontsize=font_leg)
    
    #fig, ax = plt.subplots(figsize = fig_size)
    ax2.hist(RT_true, bins=bin_edges, cumulative=True, density=True, alpha = 0.5, color='skyblue', edgecolor='black', label = "true RT")
    ax2.hist(RT_gen, bins=bin_edges, cumulative=True, density=True, alpha = 0.5, color='red', edgecolor='black', label = "simulated RT")
    # Add labels and title
    y_max = int(ax2.get_ylim()[1])
    x_min, x_max = ax2.get_xlim()
    delta_x = x_max - x_min
    ax2.set_xticks([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))])
    ax2.set_xticklabels([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))], fontsize=font_tick)
    ax2.set_yticks([0, y_max//2, y_max])
    ax2.set_yticklabels([0, y_max//2, y_max], fontsize=font_tick)
    ax2.set_xlabel('Reaction Time ($ms$)', fontsize=font_ax)
    ax2.set_ylabel('# of trials', fontsize=font_ax)
    ax2.set_title(f"Cumulative Histograms of true and predicted RTs generated from GO stimulation")
    ax2.legend(fontsize=font_leg)




def delta_x_via_jvp(dmm, z0, d, l, data=None):
    device = z0.device
    z0 = z0.detach().clone().requires_grad_(True)   # enable autograd        
    delta_z = torch.from_numpy(-d * l).float().to(device)   # se vuoi diminuire lungo d; + per aumentare
    if z0.ndim == 2:
        delta_z = torch.tile(delta_z, (z0.shape[0], 1))
    # jvp expects tuples for inputs/outputs
    def decoder_mean(z):   # (z, data)
#         if data:
#             x_mean, _ = dmm.generation_x(z, data)
#         else:
        x_mean, _ = dmm.generation_x(z)
        return x_mean
    y, jvp_out = jvp(decoder_mean, (z0,), (delta_z,))
    jvp_out = jvp_out.cpu().detach().numpy()
    # jvp_out has shape of x (. e.g. data_dim)
    #print(y.shape)
    print(jvp_out.shape)
    return jvp_out  # appross delta_x


from sklearn.linear_model import ElasticNet

def compute_dx_for_last_step_ElasticNet(dmm, x_seq, c_seq, delta_z, alpha_l1=1e-3, l1_ratio=1, n_iter=1):
    """
    Variante L1 (LASSO) della stima di Δx_t che soddisfa J_h(x_t) Δx ≈ Δz,
    generalizzata per batch N>1.
    
    Args:
        dmm: modello con metodo inference(x_seq, c_seq) -> (z, z_mean, z_cov)
        x_seq: torch.tensor (T, N, D)
        c_seq: torch.tensor (T, N, c_dim)
        delta_z: torch.tensor (N, d)
        alpha_l1: forza regolarizzazione L1 (per LASSO)
        n_iter: numero di iterazioni (relinearizzazioni)
    
    Returns:
        x_seq_new: torch.tensor (T, N, D)
        dx_last: torch.tensor (N, D)
    """
    device = x_seq.device
    dmm.train()
    T, N, D = x_seq.shape
    d = dmm.z_dim
    
    delta_z = delta_z.detach().clone().to(device).repeat(N, 1).view(N, d)

    for it in range(n_iter):
        x_curr = x_seq.clone().detach().requires_grad_(True)
        c_curr = c_seq.clone().detach()

        # Inferenza -> (T, N, d)
        _, z_mean, _ = dmm.inference(x_curr, c_curr)
        z_mean_t = z_mean[-1]  # (N, d)

        dx_last_list = []

        for n in range(N):
            # Calcolo Jacobiano J_n: (d, D)
            J_rows = []
            for i in range(d):
                grad_i = torch.autograd.grad(
                    z_mean_t[n, i],
                    x_curr,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True
                )[0]
                grad_last = grad_i[-1, n, :].unsqueeze(0)  # (1, D)
                J_rows.append(grad_last)
            J = torch.cat(J_rows, dim=0).detach().cpu().numpy()  # (d, D)

            # Δz per il campione n
            delta_z_np = delta_z[n].detach().cpu().numpy().flatten()

            # LASSO: J Δx ≈ Δz
#             lasso = Lasso(alpha=alpha_l1, fit_intercept=False, max_iter=10000)
#             lasso.fit(J, delta_z_np)
            model = ElasticNet(alpha=alpha_l1, l1_ratio=l1_ratio, fit_intercept=False, max_iter=10000)
            model.fit(J, delta_z_np)
            dx_last_n = torch.from_numpy(model.coef_).float().to(device).unsqueeze(0)  # (1, D)

            dx_last_list.append(dx_last_n)

        dx_last = torch.cat(dx_last_list, dim=0)  # (N, D)

        # Aggiorna solo ultimo timestep
        x_seq[-1] = x_seq[-1] + dx_last
        dx_last = dx_last.cpu().detach().numpy()
    return x_seq, dx_last



def compute_dx_for_last_step_Tikhonov(dmm, x_seq, c_seq, delta_z, alpha=0.1, n_iter=10, lam=1e-5):
    """
    Calcola Δx_t (vettore N×D) che linearmente porta z_mean_t -> z_mean_t + delta_z,
    mantenendo invariati x_{1:t-1} e c_{1:t}.

    Args:
        dmm: modello con metodo `inference(x_seq, c_seq)` che ritorna (z, z_mean, z_cov)
        x_seq: torch.tensor shape (T, N, D)
        c_seq: torch.tensor shape (T, N, c_dim)
        delta_z: torch.tensor shape (N, z_dim)
        lam: regolarizzazione Tikhonov (λ)
        alpha: passo (scalare)
        n_iter: numero di iterazioni

    Returns:
        x_seq_new: tensor shape (T, N, D) (solo ultimo passo modificato)
        dx_last: tensor shape (N, D)
    """
    device = x_seq.device
    dmm.train()
    T, N, D = x_seq.shape
    d = dmm.z_dim

    delta_z = delta_z.detach().clone().to(device).repeat(N, 1).view(N, d, 1)

    for it in range(n_iter):
        x_curr = x_seq.clone().detach().requires_grad_(True)
        c_curr = c_seq.clone().detach()

        # Run inference -> (T, N, d)
        z_sampled, z_mean, z_cov = dmm.inference(x_curr, c_curr)
        z_mean_t = z_mean[-1]  # (N, d)

        dx_last_list = []
        J_norms = []

        for n in range(N):
            # Costruisci Jacobiano J_n: (d, D) per ogni elemento nel batch
            J_rows = []
            for i in range(d):
                grad_i = torch.autograd.grad(
                    z_mean_t[n, i],
                    x_curr,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True
                )[0]
                grad_last = grad_i[-1, n, :].unsqueeze(0)  # (1, D)
                J_rows.append(grad_last)

            J = torch.cat(J_rows, dim=0)  # (d, D)
            JJt = J @ J.t() + lam * torch.eye(d, device=device)
            y = torch.linalg.solve(JJt, delta_z[n])  # (d,1)
            dx_last_n = (J.t() @ y).view(1, D)       # (1,D)

            dx_last_list.append(dx_last_n)
            J_norms.append(torch.linalg.norm(J).item())

        dx_last = torch.cat(dx_last_list, dim=0)  # (N, D)

        # Applica update solo all'ultimo step
        x_seq[-1] = x_seq[-1] + alpha * dx_last
        dx_last = dx_last.cpu().detach().numpy()
        print(f"[Iter {it+1}] Norme Jacobiane medie: {sum(J_norms)/N:.4f}")

    return x_seq, dx_last



def compute_dx_for_last_step_Tikhonov(dmm, x_seq, c_seq, delta_z, alpha=0.1, n_iter=10, lam=1e-5):
    """
    Calcola Delta x_t (vettore D) che, linearmente, porta z_mean_t -> z_mean_t + delta_z,
    mantenendo invariati x_{1:t-1} e c_{1:t}.
    - dmm: il tuo modello, con metodo `inference(x_seq, c_seq)` che ritorna (z, z_mean, z_cov)
    - x_seq: torch.tensor shape (T, 1, D)  (single trial)
    - c_seq: torch.tensor shape (T, 1, c_dim)
    - delta_z: torch.tensor shape (z_dim,)
    - lam: regularization lambda
    - alpha: passo (moltiplica la soluzione lineare)
    - n_iter: numero di iterazioni (relinearizza e aggiorna per robustezza)
    - device: 'cpu' or 'cuda'
    Returns:
      x_seq_new: tensor (T, 1, D) dove solo l'ultimo passo è modificato
      dx_last: tensor (D,)
    """
    device = x_seq.device
    #dmm = dmm.to(device)
    dmm.train()  # serve il graph (no dropout ideally)
    T, _, D = x_seq.shape
    d = dmm.z_dim

    # ensure shapes batch-first as required by your inference: (T, batch, D)
#     x_curr = x_seq.detach().clone().to(device)
#     c_curr = c_seq.detach().clone().to(device)
    delta_z = delta_z.detach().clone().to(device).view(d, 1)

    for it in range(n_iter):
        # require grad on the whole sequence (we'll only use last timestep gradient)
        x_curr = x_seq.clone().detach().requires_grad_(True)
        c_seq = c_seq.clone().detach()

        # run inference to get z_mean (graph intact)
        z_sampled, z_mean, z_cov = dmm.inference(x_curr, c_seq)  # shapes (T,1,d)...
        z_mean_t = z_mean[-1, 0]   # (d,)

        # compute Jacobian rows: each row i is grad of z_mean_t[i] wrt x_var (T,D)
        J_rows = []
        for i in range(d):
            # grad returns same shape as x_var: (T, D) because x_var is (T,D)
            grad_i = torch.autograd.grad(z_mean_t[i], x_curr, retain_graph=True, create_graph=False, allow_unused=True)[0]  # (T, D)
            # extract last time-step
            grad_last = grad_i[-1, 0, :].unsqueeze(0)   # (1, D)
            J_rows.append(grad_last)
        # stack -> d x D
        J = torch.cat(J_rows, dim=0)  # (d, D)

        # Build JJt (d x d) and solve
        JJt = J @ J.t()               # (d, d)
        A = JJt + lam * torch.eye(d, device=device)  # (d, d)
        # solve A * y = delta_z  -> y shape (d,1)
        y = torch.linalg.solve(A, delta_z)           # (d, 1)
        dx_last = (J.t() @ y).view(-1)               # (D,)

        # apply step alpha to last timepoint
        x_seq[:, -1] = x_seq[:, -1] + alpha * dx_last.unsqueeze(0)
#         print(x_seq.shape)
#         print(dx_last.shape)
        # optional: clamp to realistic firing ranges, e.g.
        # x_curr[-1] = torch.clamp(x_curr[-1], min_val, max_val)
        J_norm = torch.linalg.norm(J)
        print("Norma Jacobiana rispetto a x_t:", J_norm.item())
        #print(x_curr.grad[-1, 0, :].abs().sum())
        # next iteration will re-linearize around new x_curr
        dx_last = dx_last.cpu().detach().numpy() 

    return x_seq, dx_last


def single_cycle_shift(comm_dict, diff_dict):
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    ar = comm_dict["ar"]
    RT_detector = comm_dict["RT_detector"]

    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    step = diff_dict["step"]
    bins = diff_dict["bins"]
    cmap = diff_dict["cmap"]
    q = diff_dict["q"]
    alpha = diff_dict["alpha"]
    stimolate = diff_dict["stimolate"]
    l1_ratio = diff_dict["l1_ratio"]
    alpha_L1 = diff_dict["alpha_L1"]
    n_iter = diff_dict["n_iter"]
    mean = diff_dict["mean"]
    stimulation_steps = diff_dict["stimulation_steps"]
    mean_trials = diff_dict["mean_trials"]
    l = diff_dict["l"]
    f = diff_dict["f"]
    axis = diff_dict["axis"]
    multi_direction = diff_dict["multi_direction"]
    add_residual = diff_dict["add_residual"]
    alpha_point = diff_dict["alpha_point"]
    markersize = 15
    markeredgewidth = 3

    RT_cn = data["RT_cn_ordRT"]
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    z_cn, _, _ = infer_latent(dmm, data, device)

    steps, s, _ = z_cn.shape
    samples = len(RT_cn)    # s = samples * n_trials
    RT_min = 50

    z_RT = z_cn[(RT_cn + 56)//tau, np.arange(samples)]
    color = RT_cn/RT_cn.max()

    z_GO = z_cn[step//tau]

    print(z_GO.shape[0])

    long_index = int(z_GO.shape[0] // (1/(1-f)))
    short_index = int(z_GO.shape[0] // (1/f))
#     print(long_index)
#     print(short_index)

    set_cn_long = set_cn[long_index:]
    cont_cn_long = cont_cn[long_index:]
    RT_cn_long = RT_cn[long_index:]
    z_GO_long = z_GO[long_index:]
    z_GO_short = z_GO[:short_index]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(z_cn[:, :, axis[0]], z_cn[:, :, axis[1]], color="green", alpha=alpha)
    ax.scatter(z_GO_long[:, axis[0]], z_GO_long[:, axis[1]], c="blue", s = 30, alpha=alpha_point, label = 'GO of long RT')
    ax.scatter(z_GO_short[:, axis[0]], z_GO_short[:, axis[1]], c="red", s = 30, alpha=alpha_point, label = 'GO of short RT')
    ax.plot(z_GO_short[:, axis[0]].mean(), z_GO_short[:, axis[1]].mean(), c = 'red', marker='*', markeredgewidth = 4, markersize = 20)
    ax.plot(z_GO_long[:, axis[0]].mean(), z_GO_long[:, axis[1]].mean(), c = 'blue', marker='*', markeredgewidth = 4, markersize = 20)

    from sklearn.linear_model import LinearRegression

    # --- STIMA DIREZIONE ---
    reg = LinearRegression().fit(z_GO, color)
    direction = reg.coef_
    direction /= np.linalg.norm(direction)  # normalizzazione unit vector
    delta_z = torch.from_numpy(-direction * l).float().to(device)
    print("Direzione di variazione (aumenta):", direction)

    start_sim = step//tau + 1 - stimulation_steps 
    delta_z = torch.zeros(stimulation_steps, z_dim).to(device)
    if multi_direction:
        for t in range(stimulation_steps):
            reg = LinearRegression().fit(z_cn[start_sim + t], color)
            direction = reg.coef_
            direction /= np.linalg.norm(direction)  # normalizzazione unit vector
            delta_z[t] = torch.from_numpy(-direction * l).float().to(device)
            print("Direzione di variazione (aumenta):", direction)

    trial_true = torch.from_numpy(set_cn_long[q, :(step//tau + 1)]).float().to(device).unsqueeze(1)  #[no_peak_mask]
    cont_RTlong = torch.from_numpy(cont_cn_long[q]).float().to(device).unsqueeze(1)
    trial_modified = trial_true.clone()

    trial_modified = trial_modified.repeat(1, n_trials, 1)
    trial_true = trial_true.repeat(1, n_trials, 1)
    cont_RTlong = cont_RTlong.repeat(1, n_trials, 1)  #.repeat(1, int(1/f), 1)

    trial_clone = trial_modified.clone()
    y_mean, y_logvar = dmm(trial_clone, cont_RTlong[:(step//tau + 1)])
    y_pred = dmm.reparameterization(y_mean, y_logvar)
    if mean:
        residual = trial_clone - y_mean
    else:
        residual = trial_clone - y_pred


    start_sim = step//tau + 1 - stimulation_steps 
    dx_fix = torch.full((n_trials,), l).to(device)
    dx_array = np.empty([stimulation_steps, 96])
    for t in range(stimulation_steps):
        print(t)
        # x -> x + dx
        if stimolate:
            #trial_modified, dx = compute_dx_for_last_step_Tikhonov(dmm, trial_modified, cont_RTlong[:(step//tau + 1)], delta_z, alpha = alpha_stim, n_iter=n_iter)
            _, dx = compute_dx_for_last_step_ElasticNet(dmm, 
                                                         trial_modified[:(start_sim + t + 1)], 
                                                         cont_RTlong[:(start_sim + t + 1)], 
                                                         delta_z[t] if multi_direction else delta_z, 
                                                         alpha_l1=alpha_L1, 
                                                         l1_ratio=l1_ratio, 
                                                         n_iter=1)
            dx_array[t] = dx.mean(0)
            trial_modified[start_sim + t] += torch.from_numpy(dx).float().to(device)
        else:
            trial_modified[start_sim + t, :, 63] += dx_fix
        if t+1 == stimulation_steps:
            continue
        trial_cut = trial_modified.clone()
        # z_t = enc(x+dx)
        z_modified, _, _ = dmm.inference(trial_cut[:(start_sim + t + 1)], cont_RTlong[:(start_sim + t + 1)])
        # z_{t+1} = prop(z_t)
        z_mean_gen, z_cov_gen = dmm.generation_z(z_modified[-1].unsqueeze(0), cont_RTlong[start_sim + t + 1].unsqueeze(0))
        z_gen_last = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
        # x_{t+1} = dec(z_{t+1})
        y_mean, y_logvar = dmm.generation_x(z_gen_last)
        y_pred = dmm.reparameterization(y_mean, y_logvar)
        if mean:
            y_gen = y_mean
        else:
            y_gen = y_pred
        #y_pred = dmm.reparameterization(y_mean, y_logvar)
        trial_modified[start_sim + t + 1] = y_gen[0]
        if add_residual:
            trial_modified[start_sim + t + 1] += residual[start_sim + t + 1]


    trial_stim = trial_modified.cpu().detach().numpy()

    k = math.ceil(math.sqrt(step//tau))
    # Plot the density map
    fig, ax = plt.subplots(k, k, figsize=(16, 15))

    vmin = trial_stim.min()
    vmax = trial_stim.max()

    color_edge = 'r'
    trial_stim = channel2grid(trial_stim)
    for i in range(k):
        for j in range(k):
            if k*i + j > step//tau:
                continue

            if k*i + j == start_sim:
                color_edge = 'g'
            shift_plot = ax[i, j].imshow(trial_stim[k*i+j].mean(0), 
                                  aspect='equal',
                                  cmap=cmap,
                                  interpolation='nearest',
                                  vmin=vmin, vmax=vmax)
            ax[i, j].plot(0, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
            ax[i, j].plot(9, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
            ax[i, j].plot(0, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
            ax[i, j].plot(9, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].grid(True, which='major', color='w', alpha=0.2)
    #ax.set_title(f'Neural stimulus to shorten the RT')
    plt.colorbar(shift_plot, ax=ax, fraction=0.025, pad=0.06)    

    if stimolate:
        m = math.ceil(math.sqrt(stimulation_steps))
        # Plot the density map
        fig, ax = plt.subplots(m, m, figsize=(16, 15))

        vmin = dx_array.min()
        vmax = dx_array.max()

        color_edge = 'r'
        dx_array = channel2grid(dx_array)
        for i in range(m):
            for j in range(m):
                if m*i + j >= stimulation_steps:
                    continue

                shift_plot = ax[i, j].imshow(dx_array[m*i+j], 
                                      aspect='equal',
                                      cmap=cmap,
                                      interpolation='nearest',
                                      vmin=vmin, vmax=vmax)
                ax[i, j].plot(0, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
                ax[i, j].plot(9, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
                ax[i, j].plot(0, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
                ax[i, j].plot(9, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                ax[i, j].grid(True, which='major', color='w', alpha=0.2)
        #ax.set_title(f'Neural stimulus to shorten the RT')
        plt.colorbar(shift_plot, ax=ax, fraction=0.025, pad=0.06)    

    teacher = step//tau + 1
    RT_true = (RT_cn_long[q]+56)*5

    make_RThist(comm_dict, diff_dict, trial_modified, trial_true, cont_RTlong, z_cn, RT_true)



def single_channel_shift(comm_dict, diff_dict):
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    ar = comm_dict["ar"]
    RT_detector = comm_dict["RT_detector"]

    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    step = diff_dict["step"]
    bins = diff_dict["bins"]
    cmap = diff_dict["cmap"]
    q = diff_dict["q"]
    alpha = diff_dict["alpha"]
    mean_trials = diff_dict["mean_trials"]
    l = diff_dict["l"]
    f = diff_dict["f"]
    axis = diff_dict["axis"]
    alpha_point = diff_dict["alpha_point"]
    markersize = 15
    markeredgewidth = 3

    RT_cn = data["RT_cn_ordRT"]
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    z_cn, _, _ = infer_latent(dmm, data, device)

    steps, s, _ = z_cn.shape
    samples = len(RT_cn)    # s = samples * n_trials
    RT_min = 50

    z_RT = z_cn[(RT_cn + 56)//tau, np.arange(samples)]
    color = RT_cn/RT_cn.max()

    z_GO = z_cn[step//tau]

    print(z_GO.shape[0])

    long_index = int(z_GO.shape[0] // (1/(1-f)))
    short_index = int(z_GO.shape[0] // (1/f))
#     print(long_index)
#     print(short_index)

    set_cn_long = set_cn[long_index:]
    cont_cn_long = cont_cn[long_index:]
    RT_cn_long = RT_cn[long_index:]
    z_GO_long = z_GO[long_index:]
    z_GO_short = z_GO[:short_index]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(z_cn[:, :, axis[0]], z_cn[:, :, axis[1]], color="green", alpha=alpha)
    ax.scatter(z_GO_long[:, axis[0]], z_GO_long[:, axis[1]], c="blue", s = 30, alpha=alpha_point, label = 'GO of long RT')
    ax.scatter(z_GO_short[:, axis[0]], z_GO_short[:, axis[1]], c="red", s = 30, alpha=alpha_point, label = 'GO of short RT')
    ax.plot(z_GO_short[:, axis[0]].mean(), z_GO_short[:, axis[1]].mean(), c = 'red', marker='*', markeredgewidth = 4, markersize = 20)
    ax.plot(z_GO_long[:, axis[0]].mean(), z_GO_long[:, axis[1]].mean(), c = 'blue', marker='*', markeredgewidth = 4, markersize = 20)

    trial_true = torch.from_numpy(set_cn_long[q, :(step//tau + 1)]).float().to(device).unsqueeze(1)  #[no_peak_mask]
    cont_RTlong = torch.from_numpy(cont_cn_long[q]).float().to(device).unsqueeze(1)
    trial_modified = trial_true.clone()

    dx = torch.full((1,), l).to(device)
    for t in range(step//tau + 1):
        trial_modified[t, :, 63] = trial_true[t, :, 63] + t*dx

    trial_stim = trial_modified.cpu().detach().numpy()

    k = math.ceil(math.sqrt(step//tau))
    # Plot the density map
    fig, ax = plt.subplots(k, k, figsize=(16, 15))

    vmin = trial_stim.min()
    vmax = trial_stim.max()


    trial_stim = channel2grid(trial_stim)
    for i in range(k):
        for j in range(k):
            if k*i + j >= step//tau:
                continue
            shift_plot = ax[i, j].imshow(trial_stim[k*i+j, 0], 
                                  aspect='equal',
                                  cmap=cmap,
                                  interpolation='nearest',
                                  vmin=vmin, vmax=vmax)
            ax[i, j].plot(0, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
            ax[i, j].plot(9, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
            ax[i, j].plot(0, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
            ax[i, j].plot(9, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].grid(True, which='major', color='w', alpha=0.2)
    #ax.set_title(f'Neural stimulus to shorten the RT')
    plt.colorbar(shift_plot, ax=ax, fraction=0.025, pad=0.06)

    RT_true = (RT_cn_long[q]+56)*5

    trial_modified = trial_modified.repeat(1, n_trials, 1)
    trial_true = trial_true.repeat(1, n_trials, 1)
    cont_RTlong = cont_RTlong.repeat(1, n_trials, 1)  #.repeat(1, int(1/f), 1)

    make_RThist(comm_dict, diff_dict, trial_modified, trial_true, cont_RTlong, z_cn, RT_true)

    

def make_RThist(comm_dict, diff_dict, last_stim, trial_modified, trial_true, cont_RTlong, z_cn, l, text, RT_true=None):
    
    dmm = comm_dict["dmm"]
    tau = comm_dict["tau"]
    RT_detector = comm_dict["RT_detector"]
    
    n_trials = diff_dict["n_trials"]
    alpha = diff_dict["alpha"]
    alpha_point = diff_dict["alpha_point"]
    step = diff_dict["step"]
    bins = diff_dict["bins"]
    axis = diff_dict["axis"]
    color_true = diff_dict["color_true"]
    color_pred = diff_dict["color_pred"]
    x_ticks_hist = diff_dict["x_ticks_hist"]
    inset_dim = diff_dict["inset_dim"]
    markersize = diff_dict["markersize"]
    markeredgewidth = diff_dict["markeredgewidth"]
    color_edge = diff_dict["color_edge"]
    cmap_stim = diff_dict["cmap_stim"]
        
    steps = 256//tau
    teacher = step//tau + 1
                                                          
    z_generated, _, _ = dmm.inference(trial_true[:teacher], cont_RTlong[:teacher])
    z_modified, _, _ = dmm.inference(trial_modified, cont_RTlong[:teacher])
    z_GO_true = z_generated[-1].cpu().detach().numpy()                                                       
    z_GO_modified = z_modified[-1].cpu().detach().numpy()

#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.plot(z_cn[:, :, axis[0]], z_cn[:, :, axis[1]], color="green", alpha=alpha_point)
#     ax.scatter(z_GO_true[:, axis[0]], z_GO_true[:, axis[1]], c="blue", s = 30, alpha=alpha, label = 'GO of the inferred traj')
#     ax.scatter(z_GO_modified[:, axis[0]], z_GO_modified[:, axis[1]], c="red", s = 30, alpha=alpha, label = 'GO after addition of stimulus')
#     if not RT_true:
#         ax.plot(z_GO_modified[:, axis[0]].mean(), z_GO_modified[:, axis[1]].mean(), c = 'red', marker='*', markeredgewidth = 4, markersize = 20)
#         ax.plot(z_GO_true[:, axis[0]].mean(), z_GO_true[:, axis[1]].mean(), c = 'blue', marker='*', markeredgewidth = 4, markersize = 20)


    #s = set_cn.shape[0]
    alone = steps - teacher
    
    for cycle_step in range(alone):
        
        z_mean_gen, z_cov_gen = dmm.generation_z(z_modified[-1].unsqueeze(0), cont_RTlong[teacher+cycle_step].unsqueeze(0))
        z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
        z_modified = torch.cat((z_modified, z_gen), dim=0)
        
        z_mean_gen, z_cov_gen = dmm.generation_z(z_generated[-1].unsqueeze(0), cont_RTlong[teacher+cycle_step].unsqueeze(0))
        z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
        z_generated = torch.cat((z_generated, z_gen), dim=0)
        

    RT_output = RT_detector(z_modified.permute(1, 0, 2))
    RT_estimate = prob_to_RT(RT_output, tau)   

    RT_gen = RT_detector(z_generated.permute(1, 0, 2))
    RT_estimate_gen = prob_to_RT(RT_gen, tau)  
    
    RT_gen = (RT_estimate*tau)*5
    RT_gen_true = (RT_estimate_gen*tau)*5
    
    num_bins = bins
    min_value = min(RT_gen_true.min(), RT_gen.min())
    max_value = max(RT_gen_true.max(), RT_gen.max())
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)
    
    print(f"True mean RT: {RT_gen_true.mean()}")
    print(f"Generated mean RT: {RT_gen.mean()}")
    
    
    fig, ax = plt.subplots()
    ax.hist(RT_gen_true, bins=bin_edges, alpha = alpha, density=True, color=color_true, edgecolor='none')#, label = "true RT")
    ax.axvline(RT_gen_true.mean(), color=color_true,linestyle="--")#,label= "mean mod RT")
    ax.hist(RT_gen, bins=bin_edges, alpha = alpha, density=True, color=color_pred, edgecolor='none')#, label = "gen RT")
    ax.axvline(RT_gen.mean(), color=color_pred,linestyle="--")#,label= "mean gen RT")
    if RT_true:
        ax.axvline(RT_true, color="k",linestyle="--")#,label= "true RT")
#     x_min, x_max = ax.get_xlim()
#     dx = (x_max - x_min)//3
#     ax.set_xticks([int(((x_min + dx)//100)*100), int(((x_min + 2*dx)//100)*100)])
#     ax.set_xticklabels([int(((x_min + dx)//100)*100), int(((x_min + 2*dx)//100)*100)], fontsize=font_tick)
    ax.set_xticks(x_ticks_hist)
    ax.set_xticklabels(x_ticks_hist)#, fontsize=font_tick)
    ax.set_yticks([])
    #ax1.set_yticklabels([], fontsize=font_tick)
    # Add labels and title
    ax.set_xlabel('Reaction Time ($ms$)')#, fontsize=font_ax)
    ax.set_ylabel('Counts')#, fontsize=font_ax)
    
    vmax = abs(last_stim).max()
    
    if vmax < 0.1:
        vmax=0.3
#     vmin = last_stim.min()
#     vmax = last_stim.max()
    
    axins = ax.inset_axes(inset_dim) 

    stim_plot = axins.imshow(last_stim, 
                          aspect='equal',
                          cmap=cmap_stim,
                          interpolation='nearest',
                          vmin=-vmax, vmax=vmax)
    axins.plot(0, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
    axins.plot(9, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
    axins.plot(0, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
    axins.plot(9, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
    axins.set_xticks([])
    axins.set_yticks([])
    axins.grid(True, which='major', color='w', alpha=0.2)

#     cbar = plt.colorbar(stim_plot, ax=ax, fraction=fraction)#, pad=0.06)    
#     cbar.set_ticks(cbar_ticks)  # Specify exact tick locations
#     #cbar.set_ticklabels([0, 0.5, 1])  # Custom tick labels
#     cbar.set_label('Stimulation')#, fontsize=font_ax)

#     ax1.set_title(f"Histograms of generated and modified RTs generated from the GO of a single short RT trial.")
#     ax1.legend(fontsize=font_leg)
    fig_file = os.path.join(comm_dict["saved_path"], f'{text}_RT_{l}.png')
    plt.savefig(fig_file)

#     ax2.hist(RT_gen_true, bins=bin_edges, cumulative=True, density=True, alpha = alpha, color=color_true, edgecolor='none', label = "true RT")
#     ax2.hist(RT_gen, bins=bin_edges, cumulative=True, density=True, alpha = alpha, color=color_pred, edgecolor='none', label = "simulated RT")
#     # Add labels and title
#     y_max = int(ax2.get_ylim()[1])
#     ax2.set_xticks([int(((x_min + dx)//100)*100), int(((x_min + 2*dx)//100)*100)])
#     ax2.set_xticklabels([int(((x_min + dx)//100)*100), int(((x_min + 2*dx)//100)*100)], fontsize=font_tick)
# #     ax2.set_yticks([0, y_max//2, y_max])
# #     ax2.set_yticklabels([0, y_max//2, y_max], fontsize=font_tick)
#     ax2.set_xlabel('Reaction Time ($ms$)', fontsize=font_ax)
#     ax2.set_ylabel('fraction of trials', fontsize=font_ax)
# #     ax2.set_title(f"Cumulative Histograms of generated and modified RTs generated from GO stimulation")
# #     ax2.legend(fontsize=font_leg)


def single_shift(comm_dict, diff_dict):
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    ar = comm_dict["ar"]
    RT_detector = comm_dict["RT_detector"]

    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    step = diff_dict["step"]
    bins = diff_dict["bins"]
    cmap = diff_dict["cmap"]
    q = diff_dict["q"]
    stimolate = diff_dict["stimolate"]
    #stimulation_steps = diff_dict["stimulation_steps"]
    no_peak_lim = diff_dict["no_peak_lim"]
    alpha = diff_dict["alpha"]
    alpha_stim = diff_dict["alpha_stim"]
    alpha_L1 = diff_dict["alpha_L1"]
    n_iter = diff_dict["n_iter"]
    mean_trials = diff_dict["mean_trials"]
    l = diff_dict["l"]
    l1_ratio = diff_dict["l1_ratio"]
    f = diff_dict["f"]
    axis = diff_dict["axis"]
    alpha_point = diff_dict["alpha_point"]
    markersize = 15
    markeredgewidth = 3

    RT_cn = data["RT_cn_ordRT"]
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    z_cn, _, _ = infer_latent(dmm, data, device)

    steps, s, _ = z_cn.shape
    samples = len(RT_cn)    # s = samples * n_trials
    RT_min = 50

    z_RT = z_cn[(RT_cn + 56)//tau, np.arange(samples)]
    color = RT_cn/RT_cn.max()

#     z1_edges = np.linspace(z1_min, z1_max, bins + 1)
#     z2_edges = np.linspace(z2_min, z2_max, bins + 1)

#     if mean_trials:
#         z_cn = z_cn.reshape(steps, samples, n_trials, z_dim)
#         z_cn = z_cn.mean(2)
#         z_RT = z_cn[(RT_cn + 56)//tau, np.arange(samples)]
#         color = RT_cn/RT_cn.max()
#     else:
#         set_cn = set_cn.repeat(n_trials, 1, 1)
#         RT_cn_rep = np.repeat(RT_cn, n_trials)
#         z_RT = z_cn[(RT_cn_rep + 56)//tau, np.arange(s)]
#         color = RT_cn_rep/RT_cn_rep.max()
    z_GO = z_cn[step//tau]

    print(z_GO.shape[0])

    long_index = int(z_GO.shape[0] // (1/(1-f)))
    short_index = int(z_GO.shape[0] // (1/f))
#     print(long_index)
#     print(short_index)

    set_cn_long = set_cn[long_index:]
    cont_cn_long = cont_cn[long_index:]
    RT_cn_long = RT_cn[long_index:]
    z_GO_long = z_GO[long_index:]
    z_GO_short = z_GO[:short_index]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(z_cn[:, :, axis[0]], z_cn[:, :, axis[1]], color="green", alpha=alpha)
    ax.scatter(z_GO_long[:, axis[0]], z_GO_long[:, axis[1]], c="blue", s = 30, alpha=alpha_point, label = 'GO of long RT')
    ax.scatter(z_GO_short[:, axis[0]], z_GO_short[:, axis[1]], c="red", s = 30, alpha=alpha_point, label = 'GO of short RT')
    ax.plot(z_GO_short[:, axis[0]].mean(), z_GO_short[:, axis[1]].mean(), c = 'red', marker='*', markeredgewidth = 4, markersize = 20)
    ax.plot(z_GO_long[:, axis[0]].mean(), z_GO_long[:, axis[1]].mean(), c = 'blue', marker='*', markeredgewidth = 4, markersize = 20)

    from sklearn.linear_model import LinearRegression

    # --- STIMA DIREZIONE ---
    reg = LinearRegression().fit(z_GO, color)
    direction = reg.coef_
    direction /= np.linalg.norm(direction)  # normalizzazione unit vector
    delta_z = torch.from_numpy(-direction * l).float().to(device)
    print("Direzione di variazione (aumenta):", direction)

    trial_true = torch.from_numpy(set_cn_long[q, :(step//tau + 1)]).float().to(device).unsqueeze(1)  #[no_peak_mask]
    cont_RTlong = torch.from_numpy(cont_cn_long[q]).float().to(device).unsqueeze(1)
    trial_modified = trial_true.clone()

    if stimolate:
        #trial_modified, dx = compute_dx_for_last_step_Tikhonov(dmm, trial_modified, cont_RTlong[:(step//tau + 1)], delta_z, alpha = alpha_stim, n_iter=n_iter)
        trial_modified, dx = compute_dx_for_last_step_ElasticNet(dmm, trial_modified, cont_RTlong[:(step//tau + 1)], delta_z, alpha_l1=alpha_L1, l1_ratio=l1_ratio, n_iter=1)

    # Plot the density map
    fig, ax = plt.subplots(figsize=(6, 6))

    vmin = dx.min()
    vmax = dx.max()

    dx = channel2grid(dx)
    shift_plot = ax.imshow(dx.mean(0), 
                          aspect='equal',
                          cmap=cmap,
                          interpolation='nearest',
                          vmin=vmin, vmax=vmax)
    ax.plot(0, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
    ax.plot(9, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
    ax.plot(0, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
    ax.plot(9, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, which='major', color='w', alpha=0.2)
    ax.set_title(f'Neural stimulus to shorten the RT')
    plt.colorbar(shift_plot, ax=ax, fraction=0.025, pad=0.06)  

    trial_modified = trial_modified.repeat(1, n_trials, 1)
    trial_true = trial_true.repeat(1, n_trials, 1)
    cont_RTlong = cont_RTlong.repeat(1, n_trials, 1)  #.repeat(1, int(1/f), 1)

    teacher = step//tau + 1
    RT_true = (RT_cn_long[q]+56)*5

    make_RThist(comm_dict, diff_dict, trial_modified, trial_true, cont_RTlong, z_cn, RT_true)



def true_vs_pred_zcn(comm_dict, diff_dict):

    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    c_dim = comm_dict["c_dim"]

    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    sim_start = diff_dict["sim_start"]
    bins = diff_dict["bins"]
    cmap = diff_dict["cmap"]
    c_norm = diff_dict["c_norm"]
    color = diff_dict["color"]
    alpha = diff_dict["alpha"]
    dir_on = diff_dict["dir_on"]
    axis = diff_dict["axis"]

    cont_c = data["cont_cn_ordRT"]
    dir_cn = data["dir_cn_ordRT"]
    z_cn, _, _ = infer_latent(dmm, data, device, n_trials)

    s = z_cn.shape[1]
    steps = z_cn.shape[0]
    teacher = sim_start//(5*tau)
    alone = steps - teacher
    c = np.full(s*steps, c_norm/s)

    dir_cn = dir_cn.repeat(n_trials)

    cont_c = torch.from_numpy(cont_c).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, s, c_dim)
    z_teach = torch.from_numpy(z_cn[:teacher]).float().to(device)

    for step in range(alone):
        z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher+step].unsqueeze(0))
        z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
        z_teach = torch.cat((z_teach, z_gen), dim=0)
    z_teach = z_teach.cpu().detach().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    if dir_on:
        z_cn_l = z_cn[:, dir_cn==0]
        z_cn_r = z_cn[:, dir_cn==1]
        ax[0].plot(z_cn_l[:, :, axis[0]], z_cn_l[:, :, axis[1]], color="blue", alpha=alpha)
        ax[0].plot(z_cn_r[:, :, axis[0]], z_cn_r[:, :, axis[1]], color="red", alpha=alpha)
    else:
        ax[0].plot(z_cn[:, :, axis[0]], z_cn[:, :, axis[1]], color=color[0], alpha=alpha)
    #ax[0].set_title('Density Plot of inferred traj in latent space')
    ax[0].set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    ax[0].set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax[0].set_xticks(z_ticks[axis[0]]) 
    ax[0].set_xticklabels(z_ticks[axis[0]], fontsize=font_tick)  # Show corresponding labels
    ax[0].set_yticks(z_ticks[axis[1]])
    ax[0].set_yticklabels(z_ticks[axis[1]], fontsize=font_tick)
    ax[0].set_xlabel('z1', fontsize=font_ax)
    ax[0].set_ylabel('z2', fontsize=font_ax)

    # Plot the second density map with extended limits
    if dir_on:
        z_teach_l = z_teach[:, dir_cn==0]
        z_teach_r = z_teach[:, dir_cn==1]
        ax[1].plot(z_teach_l[:, :, axis[0]], z_teach_l[:, :, axis[1]], color="blue", alpha=alpha)
        ax[1].plot(z_teach_r[:, :, axis[0]], z_teach_r[:, :, axis[1]], color="red", alpha=alpha)
    else:
        ax[1].plot(z_teach[:, :, axis[0]], z_teach[:, :, axis[1]], color=color[1], alpha=alpha)
    #ax[1].set_title(f'Density Plot of predicted (from {sim_start}$ms$ after start) traj in latent space')
    ax[1].set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    ax[1].set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax[1].set_xticks(z_ticks[axis[0]]) 
    ax[1].set_xticklabels(z_ticks[axis[0]], fontsize=font_tick)  # Show corresponding labels
    ax[1].set_yticks(z_ticks[axis[1]])
    ax[1].set_yticklabels(z_ticks[axis[1]], fontsize=font_tick)
    ax[1].set_xlabel('z1', fontsize=font_ax)
    ax[1].set_ylabel('z2', fontsize=font_ax)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(z_cn[:, :, axis[0]], z_cn[:, :, axis[1]], color=color[0], alpha=alpha)
    #ax.plot(z_teach[:, :, 0], z_teach[:, :, 1], color=color[1], alpha=alpha)
    #ax[0].set_title('Density Plot of inferred traj in latent space')
    ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]], fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]], fontsize=font_tick)
    ax.set_xlabel('z1', fontsize=font_ax)
    ax.set_ylabel('z2', fontsize=font_ax)

    z_teach = z_teach.reshape((-1, 2))
    z_cn = z_cn.reshape((-1, 2))

    # Crea edges comuni per entrambi i plot
    z1_edges = np.linspace(z_lims[axis[0], 0], z_lims[axis[0], 1], bins + 1)
    z2_edges = np.linspace(z_lims[axis[1], 0], z_lims[axis[1], 1], bins + 1)

    hist_inf, z1_edges_inf, z2_edges_inf, binnumber_inf = binned_statistic_2d(
        z_cn[:, axis[0]], z_cn[:, axis[1]], c, 
        statistic='sum', 
        bins=[z1_edges, z2_edges]
    )

    hist_pred, z1_edges_pred, z2_edges_pred, binnumber_pred = binned_statistic_2d(
        z_teach[:, axis[0]], z_teach[:, axis[1]], c, 
        statistic='sum', 
        bins=[z1_edges, z2_edges]
    )

    vmin = 0#min(hist_inf.min(), hist_pred.min())
    vmax = 1#max(hist_inf.max(), hist_pred.max())

    # Create a mesh grid for plotting
    z1, z2 = np.meshgrid(z1_edges[:-1] + np.diff(z1_edges)/2, z2_edges[:-1] + np.diff(z2_edges)/2)

    # Plot the density map
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].pcolormesh(z1, z2, 1-np.exp(-hist_inf.T), cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    #ax[0].set_title('Density Plot of inferred traj in latent space')
    ax[0].set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    ax[0].set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax[0].set_xticks(z_ticks[axis[0]]) 
    ax[0].set_xticklabels(z_ticks[axis[0]], fontsize=font_tick)  # Show corresponding labels
    ax[0].set_yticks(z_ticks[axis[0]])
    ax[0].set_yticklabels(z_ticks[axis[0]], fontsize=font_tick)
    ax[0].set_xlabel('z1', fontsize=font_ax)
    ax[0].set_ylabel('z2', fontsize=font_ax)

    # Plot the second density map with extended limits
    im=ax[1].pcolormesh(z1, z2, 1-np.exp(-hist_pred.T), cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    #ax[1].set_title(f'Density Plot of predicted (from {sim_start}$ms$ after start) traj in latent space')
    ax[1].set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    ax[1].set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax[1].set_xticks(z_ticks[axis[0]]) 
    ax[1].set_xticklabels(z_ticks[axis[0]], fontsize=font_tick)  # Show corresponding labels
    ax[1].set_yticks(z_ticks[axis[1]])
    ax[1].set_yticklabels(z_ticks[axis[1]], fontsize=font_tick)
    ax[1].set_xlabel('z1', fontsize=font_ax)
    ax[1].set_ylabel('z2', fontsize=font_ax)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([0, 0.5, 1])  # Specify exact tick locations
    #cbar.set_ticklabels([0, 0.5, 1])  # Custom tick labels
    cbar.set_label('n. of traj passing from that bin', fontsize=font_ax)
    cbar.ax.tick_params(labelsize=font_tick)



def plot_covariance_ellipse(ax, mean, cov, color='b', n_std=1.0):
    """
    Plotta un'ellisse di covarianza.
    
    :param ax: Asse matplotlib su cui disegnare
    :param mean: Vettore media [x, y]
    :param cov: Matrice di covarianza 2x2
    :param color: Colore dell'ellisse
    :param n_std: Numero di deviazioni standard per l'ellisse
    """
    lambda_, v = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    
    width, height = 2 * n_std * np.sqrt(lambda_)
    ellipse = plt.matplotlib.patches.Ellipse(
        xy=mean, width=width, height=height, angle=angle, 
        fill=False, color=color, linewidth=2
    )
    ax.add_artist(ellipse)

    
def inference_variance(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    intervals = diff_dict["intervals"]
    
    z_cn, R, q = random_latent_cn_traj(dmm, data, tau, device, n_trials)
    
    print(f"traj n.{q}")
    
    GO_cn = z_cn[intervals[0]//(5*tau)]
    afterGO_cn = z_cn[intervals[1]//(5*tau)]
    lot_afterGO_cn = z_cn[intervals[2]//(5*tau)]
    
    q = 0
    x_true_story = z_cn[:, q, 0]
    y_true_story = z_cn[:, q, 1]
    #x_start = z_cn[0, q, axes[0]]
    #y_start = z_cn[0, q, axes[1]]
    #x_GO = z_cn[56//tau, q, axes[0]]
    #y_GO = z_cn[56//tau, q, axes[1]]
    #x_RT = z_cn[RT, q, axes[0]]
    #y_RT = z_cn[RT, q, axes[1]]
    
    fig, ax = plt.subplots(figsize = (7, 6))
    #im = ax.scatter(GO_x_cn, GO_y_cn, c=color_GO, s = 15, alpha = 0.5, label = 'GO from low to high RT predicted from it')
    plot_covariance_ellipse(ax, GO_cn.mean(0), np.cov(GO_cn.T), color='purple', n_std=2)
    plot_covariance_ellipse(ax, afterGO_cn.mean(0), np.cov(afterGO_cn.T), color='purple', n_std=2)
    plot_covariance_ellipse(ax, lot_afterGO_cn.mean(0), np.cov(lot_afterGO_cn.T), color='purple', n_std=2)
    ax.plot(x_true_story, y_true_story, '-', linewidth=2, color='brown')
    n_arrows = 15
    arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
    for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = x_true_story[i+1] - x_true_story[i]
        dy = y_true_story[i+1] - y_true_story[i]
        ax.arrow(x_true_story[i], y_true_story[i], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    ax.scatter(GO_cn[:, 0], GO_cn[:, 1], s = 5, c = 'b', alpha = 1, label = "GO")
    ax.scatter(afterGO_cn[:, 0], afterGO_cn[:, 1], s = 5, c = 'g', alpha = 1, label = "GO + 300ms")
    ax.scatter(lot_afterGO_cn[:, 0], lot_afterGO_cn[:, 1], s = 5, c = 'r', alpha = 1, label = "GO + 600ms")
    #ax.scatter(x_GO, y_GO, s = 50, c = 'b', alpha = 1, label = "GO")
    #ax.scatter(x_RT, y_RT, s = 50, c = 'r', alpha = 1, label = "RT")
    #ax.plot(x_SSD, y_SSD, 'x', c = 'red', marker='x', markeredgewidth = 3, markersize = 15)
    #ax.set_title(f"RT: {SSD:d}")
    ax.set_xlabel("first latent component")
    ax.set_ylabel("second latent component")
    ax.set_title("Variability of single point reconstruction, for GO, GO+300ms and GO+600ms")
    ax.legend()
    #fig.colorbar(im)




def cs_ws_stop_plot(comm_dict, diff_dict):

    with np.load(comm_dict["saved_path"]+"/data_split.npz", allow_pickle=True) as loaded_file:
            train_set = loaded_file["train_set"]
            vali_set = loaded_file["vali_set"]
            test_set = loaded_file["test_set"]
            train_direction = loaded_file["train_direction"]
            vali_direction = loaded_file["vali_direction"]
            test_direction = loaded_file["test_direction"]
            train_SSD = loaded_file["train_SSD"]
            vali_SSD = loaded_file["vali_SSD"]
            test_SSD = loaded_file["test_SSD"]
            train_RT = loaded_file["train_RT"]
            vali_RT = loaded_file["vali_RT"]
            test_RT = loaded_file["test_RT"]

    ws_train_mask = (train_RT>0) & (train_SSD>0)
    ws_test_mask = (test_RT>0) & (test_SSD>0)
    cs_train_mask = (train_RT==0) & (train_SSD>0)
    cs_test_mask = (test_RT==0) & (test_SSD>0)

    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]

    data = diff_dict["data"]
    axis = diff_dict["axis"]
    axis_pca = diff_dict["axis_pca"]
    n_pc = diff_dict["n_pc"]
    j = diff_dict["j"]
    color_ws = diff_dict["color_ws"]
    color_cs = diff_dict["color_cs"]
    pca_flag = diff_dict["pca_flag"]
    delta_t = diff_dict["delta_t"]
    RT_thr = diff_dict["RT_thr"]
    cmap_RT = diff_dict["cmap_RT"]
    z1_list = diff_dict["z1_list"]
    color_list = diff_dict["color_list"]
    alpha = diff_dict["alpha"]
    RT_groups = diff_dict["RT_groups"]

    SSD_cs = data["SSD_cs_ordSSD"]
    SSD_ws = data["SSD_ws_ordSSD"]
    RT_ws = data["RT_ws_ordSSD"]
    RT_cn = data["RT_cn_ordRT"]

    s, steps, features = test_set.shape
    X_train = train_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
    X_test = test_set.reshape(-1, features)  # shape = (trials * time_steps, 96)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_pc)
    X_train_pca = pca.fit_transform(X_train)  # fit + transform sul train 
    X_test_pca = pca.transform(X_test)   
#     X_test_rec = pca.inverse_transform(X_test_pca)
#     X_train_rec = pca.inverse_transform(X_train_pca)
    test_rec = X_test_pca.reshape(s, steps, n_pc)
    train_rec = X_train_pca.reshape(-1, steps, n_pc)

    ws_train_pca = train_rec[ws_train_mask]
    ws_test_pca = test_rec[ws_test_mask]
    cs_train_pca = train_rec[cs_train_mask]
    cs_test_pca = test_rec[cs_test_mask]

    train_ws_SSD = train_SSD[ws_train_mask]
    test_ws_SSD = test_SSD[ws_test_mask]
    train_cs_SSD = train_SSD[cs_train_mask]
    test_cs_SSD = test_SSD[cs_test_mask]

    x_cs_train = cs_train_pca[np.arange(cs_train_pca.shape[0]), (train_cs_SSD+56)//tau, axis_pca[0]]
    x_ws_train = ws_train_pca[np.arange(ws_train_pca.shape[0]), (train_ws_SSD+56)//tau, axis_pca[0]]

    x_cs_test = cs_test_pca[np.arange(cs_test_pca.shape[0]), (test_cs_SSD+56)//tau, axis_pca[0]]
    x_ws_test = ws_test_pca[np.arange(ws_test_pca.shape[0]), (test_ws_SSD+56)//tau, axis_pca[0]]

    y_cs_train = cs_train_pca[np.arange(cs_train_pca.shape[0]), (train_cs_SSD+56)//tau, axis_pca[1]]
    y_ws_train = ws_train_pca[np.arange(ws_train_pca.shape[0]), (train_ws_SSD+56)//tau, axis_pca[1]]

    y_cs_test = cs_test_pca[np.arange(cs_test_pca.shape[0]), (test_cs_SSD+56)//tau, axis_pca[1]]
    y_ws_test = ws_test_pca[np.arange(ws_test_pca.shape[0]), (test_ws_SSD+56)//tau, axis_pca[1]]

    trial = ws_train_pca[j]

    z_cn, z_ws, z_cs = infer_latent(dmm, data, device)

    #####################################

    if pca_flag:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=z_dim)
        z_cs_flat = pca.fit_transform(z_cs.reshape(-1, z_dim))  
        z_ws_flat = pca.transform(z_ws.reshape(-1, z_dim))
        z_cs = z_cs_flat.reshape(steps, -1, z_dim)
        z_ws = z_ws_flat.reshape(steps, -1, z_dim)

#     pca = PCA(n_components=z_dim)
#     z_cn_flat = pca.fit_transform(z_cn.reshape(-1, z_dim))  # fit + transform sul train 
#     z_cn = z_cn_flat.reshape(steps, s, z_dim)

    #####################################
    f1, ax1 = plt.subplots()
    f2, ax2 = plt.subplots()
    for i, z1_thr in enumerate(z1_list):
        t_star = np.argmax(z_cn[:, :, 0] < z1_thr, axis=0)
        t_std_list = []
        t_mean_list = []
        RT_list = []
        n_group = len(RT_cn) // RT_groups if (len(RT_cn)%RT_groups)==0 else (len(RT_cn) // RT_groups) + 1
        for k in range(RT_groups):
            start = k*n_group
            end = min(start + n_group, len(RT_cn) + 1)
            # calcolo SSRT per ogni trial, e poi ne prendo la media per ogni gruppo di trials
            t_group = t_star[start:end]
            t_mean = np.nanmean(t_group)
            t_std = t_group.std()/math.sqrt(end-start)
            # calcolo l'RT medio di ogni gruppo di trials
            RT_group = RT_cn[start:end]
            mean_RT_group = RT_group.mean()
            # calcolo la media del gruppo di trials, e ne calcolo l'SSRt
            # riempo le liste
            t_std_list.append(t_std)
            t_mean_list.append(t_mean)
            RT_list.append(mean_RT_group)
        # converto le liste in array
        t_std_array = np.array(t_std_list)
        t_mean_array = np.array(t_mean_list)
        RT_array = np.array(RT_list)


#         ax1.scatter(RT_cn, t_star, color = color_list[i], alpha=alpha)
        ax1.errorbar(RT_array, RT_array-t_mean_array, yerr=t_std_array, fmt='o', color=color_list[i], ecolor='black', 
                                 elinewidth=1, linestyle='none',capsize=10, ms=8)
#         ax.plot(RT_time_masked, pred_SSRT, color = "r", 
#                    label = f"linear fit: m={reg.coef_.item():.2f}, q={reg.intercept_.item():.2f}")
        ax1.set_xlabel('RT ($ms$)')#, fontsize=font_ax)
        ax1.set_ylabel('t_star')#, fontsize=font_ax)
        ax1.set_title('t_star vs RT')

        ax2.errorbar(RT_array, t_mean_array, yerr=t_std_array, fmt='o', color=color_list[i], ecolor='black', 
                                 elinewidth=1, linestyle='none',capsize=10, ms=8)
#         ax.plot(RT_array, pred_SSRT_mean, "--", color = color_line, 
#                    label = f"linear fit: m={reg_mean.coef_.item():.2f}, q={reg_mean.intercept_.item():.2f}")

        # Calcola i limiti comuni
#         min_val_x = ax.get_xlim()[0]
#         min_val_y = ax.get_ylim()[0]
#         max_val_x = ax.get_xlim()[1]
#         max_val_y = ax.get_ylim()[1]

#         dx = (max_val_x - min_val_x)//3
#         dy = (max_val_y - min_val_y)//3

        ax2.set_xlabel('mean RT ($ms$)')#, fontsize=font_ax)
        ax2.set_ylabel('mean t_star')#, fontsize=font_ax)
#         ax2.set_xticks([int(((min_val_x + dx)//100 + 1)*100), int(((min_val_x + 2*dx)//100 + 1)*100)])
#         ax2.set_xticklabels([int(((min_val_x + dx)//100 + 1)*100), int(((min_val_x + 2*dx)//100 + 1)*100)])#, fontsize=font_tick)
#         ax2.set_yticks([int(((min_val_y + dy)//10)*10), int(((min_val_y + 2*dy)//10 + 1)*10)])
#         ax2.set_yticklabels([int(((min_val_y + dy)//10)*10), int(((min_val_y + 2*dy)//10 + 1)*10)])#, fontsize=font_tick)


    mask_highRT = RT_cn > (RT_thr//5)
    print(f"trials with high RT: {mask_highRT.sum():d}, trials with low RT: {len(RT_cn)-mask_highRT.sum():d}")
    RT_high = (RT_cn[mask_highRT] + 56)//tau 
    RT_low = (RT_cn[~mask_highRT] + 56)//tau 
    z_highRT = z_cn[:, mask_highRT]
    z_lowRT = z_cn[:, ~mask_highRT]
    z_SSRT_highRT = z_highRT[RT_high - 20, np.arange(z_highRT.shape[1])]
    z_before_highRT = z_highRT[RT_high - (20 + delta_t), np.arange(z_highRT.shape[1])]
    color_highRT = (RT_high - RT_high.min())/(RT_high.max() - RT_high.min())
    z_SSRT_lowRT = z_lowRT[RT_low - 20, np.arange(z_lowRT.shape[1])]
    z_before_lowRT = z_lowRT[RT_low - (20 + delta_t), np.arange(z_lowRT.shape[1])]
    color_lowRT = (RT_low - RT_low.min())/(RT_low.max() - RT_low.min())

    x_cs = z_cs[(SSD_cs+56)//tau, np.arange(z_cs.shape[1]), axis[0]]
    x_ws = z_ws[(SSD_ws+56)//tau, np.arange(z_ws.shape[1]), axis[0]]

    y_cs = z_cs[(SSD_cs+56)//tau, np.arange(z_cs.shape[1]), axis[1]]
    y_ws = z_ws[(SSD_ws+56)//tau, np.arange(z_ws.shape[1]), axis[1]]

    z_cn, RT, q = random_latent_cn_traj(dmm, data, tau, device)

    if pca_flag:
        z_cn = pca.transform(z_cn) # if PCA

    print(f"traj n.{q}")

    x_true_story = z_cn[:, axis[0]]
    y_true_story = z_cn[:, axis[1]]
    x_start = z_cn[0, axis[0]]
    y_start = z_cn[0, axis[1]]
    x_GO = z_cn[56//tau, axis[0]]
    y_GO = z_cn[56//tau, axis[1]]
    x_RT = z_cn[RT, axis[0]]
    y_RT = z_cn[RT, axis[1]]


    x_true_pc = trial[:, axis_pca[0]]
    y_true_pc = trial[:, axis_pca[1]]
    x_start_pc = trial[0, axis_pca[0]]
    y_start_pc = trial[0, axis_pca[1]]
    x_GO_pc = trial[56//tau, axis_pca[0]]
    y_GO_pc = trial[56//tau, axis_pca[1]]
    x_RT_pc = trial[RT, axis_pca[0]]
    y_RT_pc = trial[RT, axis_pca[1]]

    f, ax = plt.subplots()

    color = (RT_ws-RT_ws.min())/(RT_ws.max()-RT_ws.min())

    # Plot the surface
    ax.scatter(x_cs, y_cs, s = 10, c = 'r', alpha = 0.8, label = "cs stops")
    ax.scatter(x_ws, y_ws, s = 10, c = color, cmap = "Greens", alpha = 0.8, label = "ws stops")
    ax.plot(x_true_story, y_true_story, '-', linewidth=2, color='brown', label = "trajectory example")
    n_arrows = 15
    arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
    for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = x_true_story[i+1] - x_true_story[i]
        dy = y_true_story[i+1] - y_true_story[i]
        ax.arrow(x_true_story[i], y_true_story[i], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    ax.scatter(x_GO, y_GO, s = 50, c = 'b', alpha = 1, label = "GO")
    ax.scatter(x_RT, y_RT, s = 50, c = 'r', alpha = 1, label = "RT")
    ax.scatter(x_start, y_start, s = 50, c = 'purple', alpha = 1, label = "start")
    ax.set_xlim(z_lims[axis[0]])
    ax.set_ylim(z_lims[axis[1]])
    ax.set_xlabel("z1")#, fontsize=font_ax)
    ax.set_ylabel("z2")#, fontsize=font_ax)
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)
#     ax.legend()

    f, ax = plt.subplots(1, 2, figsize=(14, 7))
    # Plot the surface
    ax[0].scatter(z_SSRT_highRT[:, axis[0]], z_SSRT_highRT[:, axis[1]], s = 10, c = color_highRT, cmap = cmap_RT, alpha = 0.8, label = "cs stops")
    ax[0].plot(x_true_story, y_true_story, '-', linewidth=2, color='brown', label = "trajectory example")
    ax[0].set_xlim(z_lims[axis[0]])
    ax[0].set_ylim(z_lims[axis[1]])
    ax[0].set_xlabel("z1")#, fontsize=font_ax)
    ax[0].set_ylabel("z2")#, fontsize=font_ax)
    ax[0].set_xticks(z_ticks[axis[0]]) 
    ax[0].set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
    ax[0].set_yticks(z_ticks[axis[1]])
    ax[0].set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)

    ax[1].scatter(z_before_highRT[:, axis[0]], z_before_highRT[:, axis[1]], s = 10, c = color_highRT, cmap = cmap_RT, alpha = 0.8, label = "ws stops")
    ax[1].plot(x_true_story, y_true_story, '-', linewidth=2, color='brown', label = "trajectory example")
    ax[1].set_xlim(z_lims[axis[0]])
    ax[1].set_ylim(z_lims[axis[1]])
    ax[1].set_xlabel("z1")#, fontsize=font_ax)
    ax[1].set_ylabel("z2")#, fontsize=font_ax)
    ax[1].set_xticks(z_ticks[axis[0]]) 
    ax[1].set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
    ax[1].set_yticks(z_ticks[axis[1]])
    ax[1].set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)


    f, ax = plt.subplots(1, 2, figsize=(14, 7))
    # Plot the surface
    ax[0].scatter(z_SSRT_lowRT[:, axis[0]], z_SSRT_lowRT[:, axis[1]], s = 10, c = color_lowRT, cmap = cmap_RT, alpha = 0.8, label = "cs stops")
    ax[0].plot(x_true_story, y_true_story, '-', linewidth=2, color='brown', label = "trajectory example")
    ax[0].set_xlim(z_lims[axis[0]])
    ax[0].set_ylim(z_lims[axis[1]])
    ax[0].set_xlabel("z1")#, fontsize=font_ax)
    ax[0].set_ylabel("z2")#, fontsize=font_ax)
    ax[0].set_xticks(z_ticks[axis[0]]) 
    ax[0].set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
    ax[0].set_yticks(z_ticks[axis[1]])
    ax[0].set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)

    ax[1].scatter(z_before_lowRT[:, axis[0]], z_before_lowRT[:, axis[1]], s = 10, c = color_lowRT, cmap = cmap_RT, alpha = 0.8, label = "ws stops")
    ax[1].plot(x_true_story, y_true_story, '-', linewidth=2, color='brown', label = "trajectory example")
    ax[1].set_xlim(z_lims[axis[0]])
    ax[1].set_ylim(z_lims[axis[1]])
    ax[1].set_xlabel("z1")#, fontsize=font_ax)
    ax[1].set_ylabel("z2")#, fontsize=font_ax)
    ax[1].set_xticks(z_ticks[axis[0]]) 
    ax[1].set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
    ax[1].set_yticks(z_ticks[axis[1]])
    ax[1].set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)


    f, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))

    # Plot the surface
    ax1.scatter(x_cs_train, y_cs_train, s = 10, c = 'r', alpha = 0.8, label = "cs stops train")
    ax1.scatter(x_ws_train, y_ws_train, s = 10, c = 'g', alpha = 0.8, label = "ws stops train")
    ax1.plot(x_true_pc, y_true_pc, '-', linewidth=2, color='brown', label = "trajectory example")
    n_arrows = 15
    arrow_indices = np.arange(0, len(x_true_pc), len(x_true_pc)//n_arrows)  # Place n_arrows arrows along the path
    for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = x_true_pc[i+1] - x_true_pc[i]
        dy = y_true_pc[i+1] - y_true_pc[i]
        ax1.arrow(x_true_pc[i], y_true_pc[i], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    ax1.scatter(x_GO_pc, y_GO_pc, s = 50, c = 'b', alpha = 1, label = "GO")
    ax1.scatter(x_RT_pc, y_RT_pc, s = 50, c = 'r', alpha = 1, label = "RT")
    ax1.scatter(x_start_pc, y_start_pc, s = 50, c = 'purple', alpha = 1, label = "start")
#     ax1.set_xlim(z_lims[0, 0], z_lims[0, 1])
#     ax1.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax1.set_xlabel(f"PC {axis_pca[0]}")#, fontsize=font_ax)
    ax1.set_ylabel(f"PC {axis_pca[1]}")#, fontsize=font_ax)
#     ax1.set_xticks(z_ticks[axis[0]]) 
#     ax1.set_xticklabels(z_ticks[axis[0]], fontsize=font_tick)  # Show corresponding labels
#     ax1.set_yticks(z_ticks[axis[1]])
#     ax1.set_yticklabels(z_ticks[axis[1]], fontsize=font_tick)
#     ax1.legend()


    # Plot the surface
    ax2.scatter(x_cs_test, y_cs_test, s = 10, c = 'r', alpha = 0.8, label = "cs stops test")
    ax2.scatter(x_ws_test, y_ws_test, s = 10, c = 'g', alpha = 0.8, label = "ws stops test")
    ax2.plot(x_true_pc, y_true_pc, '-', linewidth=2, color='brown', label = "trajectory example")
    n_arrows = 15
    arrow_indices = np.arange(0, len(x_true_pc), len(x_true_pc)//n_arrows)  # Place n_arrows arrows along the path
    for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = x_true_pc[i+1] - x_true_pc[i]
        dy = y_true_pc[i+1] - y_true_pc[i]
        ax2.arrow(x_true_pc[i], y_true_pc[i], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    ax2.scatter(x_GO_pc, y_GO_pc, s = 50, c = 'b', alpha = 1, label = "GO")
    ax2.scatter(x_RT_pc, y_RT_pc, s = 50, c = 'r', alpha = 1, label = "RT")
    ax2.scatter(x_start_pc, y_start_pc, s = 50, c = 'purple', alpha = 1, label = "start")
#     ax1.set_xlim(z_lims[0, 0], z_lims[0, 1])
#     ax1.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax2.set_xlabel("PC1")#, fontsize=font_ax)
    ax2.set_ylabel("PC2")#, fontsize=font_ax)
#     ax1.set_xticks(z_ticks[axis[0]]) 
#     ax1.set_xticklabels(z_ticks[axis[0]], fontsize=font_tick)  # Show corresponding labels
#     ax1.set_yticks(z_ticks[axis[1]])
#     ax1.set_yticklabels(z_ticks[axis[1]], fontsize=font_tick)
#     ax1.legend()

def GO_types(dmm, data, n_trials, tau, device):
    samples_cn = data["set_cn_ordRT"].shape[0]
    samples_ws = data["set_ws_ordSSD"].shape[0]
    samples_cs = data["set_cs_ordSSD"].shape[0]
    steps = 256//tau

    z_cn, z_ws, z_cs = infer_latent(dmm, data, device, n_trials)
    z_dim = z_cn.shape[2]
    z_cn = z_cn.reshape(steps, samples_cn, n_trials, z_dim)
    z_ws = z_ws.reshape(steps, samples_ws, n_trials, z_dim)
    z_cs = z_cs.reshape(steps, samples_cs, n_trials, z_dim)

    GO_cs = z_cs[56//tau].mean(1)
    GO_ws = z_ws[56//tau].mean(1)
    GO_cn = z_cn[56//tau].mean(1)

    return GO_cs, GO_ws, GO_cn


def cs_ws_GO(comm_dict, diff_dict):

    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]

    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    axis = diff_dict["axis"]

    steps = 256//tau
    SSD_cs = data["SSD_cs_ordSSD"]
    SSD_ws = data["SSD_ws_ordSSD"]
    GO_cs, GO_ws, GO_cn = GO_types(dmm, data, n_trials, tau, device)
    z_trial, RT, q = random_latent_cn_traj(dmm, data, tau, device)

    print(f"traj n.{q}")

    x_true_story = z_trial[:, axis[0]]
    y_true_story = z_trial[:, axis[1]]
    x_start = z_trial[0, axis[0]]
    y_start = z_trial[0, axis[1]]
    x_GO = z_trial[56//tau, axis[0]]
    y_GO = z_trial[56//tau, axis[1]]
    x_RT = z_trial[RT, axis[0]]
    y_RT = z_trial[RT, axis[1]]


    fig, ax = plt.subplots(figsize = (7, 6))

    color = np.linspace(0, 1, steps)

    ax.scatter(GO_cn[:, axis[0]], GO_cn[:, axis[1]], c ="green", s = 15, alpha = 0.4, label = 'cs GO')
    ax.scatter(GO_ws[:, axis[0]], GO_ws[:, axis[1]], c ="orange", s = 15, alpha = 0.5, label = 'ws GO')
    ax.scatter(GO_cs[:, axis[0]], GO_cs[:, axis[1]], c ="red", s = 15, alpha = 0.5, label = 'cs GO')
    #ax.scatter(x_true_story, y_true_story, s = 15, c = color, cmap = 'Greens_r', alpha = 1, label = "trajectory example")
    ax.plot(x_true_story, y_true_story, '-', linewidth=2, color='brown', label = "trajectory example")
    n_arrows = 15
    arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
    for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = x_true_story[i+1] - x_true_story[i]
        dy = y_true_story[i+1] - y_true_story[i]
        ax.arrow(x_true_story[i], y_true_story[i], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    ax.scatter(x_GO, y_GO, s = 50, c = 'b', alpha = 1, label = "GO")
    ax.scatter(x_RT, y_RT, s = 50, c = 'r', alpha = 1, label = "RT")
    ax.scatter(x_start, y_start, s = 50, c = 'purple', alpha = 1, label="start")
    #ax.set_xlim((-3, 12))
    #ax.set_ylim((-6, 10))
    ax.set_title(f"cs, ws and cn GO")#the smallest STOP")
    #ax.set_title("cs, ws and cn GO with all traj given")
    ax.set_xlabel("first latent component")
    ax.set_ylabel("second latent component")
    ax.legend()

    fig, ax = plt.subplots(figsize = (7, 6))

    color_cs = SSD_cs/SSD_ws.max()
    color_ws = SSD_ws/SSD_ws.max()

    #ax.scatter(z_GO[:, 0], z_GO[:, 1], c ="b", s = 15, alpha = 0.2, label = 'correct nostop_GO')
    ax.scatter(GO_cs[:, axis[0]], GO_cs[:, axis[1]], c=color_cs, cmap = "Reds", s = 15, alpha = 0.7, label = 'cs GO')
    ax.scatter(GO_ws[:, axis[0]], GO_ws[:, axis[1]], c=color_ws, cmap = "Blues", s = 15, alpha = 0.7, label = 'ws GO')
    #ax.scatter(x_true_story, y_true_story, s = 15, c = color, cmap = 'Greens_r', alpha = 1, label = "trajectory example")
    ax.plot(x_true_story, y_true_story, '-', linewidth=2, color='brown', label = "trajectory example")
    n_arrows = 15
    arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
    for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = x_true_story[i+1] - x_true_story[i]
        dy = y_true_story[i+1] - y_true_story[i]
        ax.arrow(x_true_story[i], y_true_story[i], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    ax.scatter(x_GO, y_GO, s = 50, c = 'b', alpha = 1, label = "GO")
    ax.scatter(x_RT, y_RT, s = 50, c = 'r', alpha = 1, label = "RT")
    ax.scatter(x_start, y_start, s = 50, c = 'purple', alpha = 1, label="start")
    #ax.set_xlim((-3, 12))
    #ax.set_ylim((-6, 10))
    ax.set_title(f"cs, ws and cn GO")#the smallest STOP")
    #ax.set_title("cs, ws and cn GO with all traj given")
    ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax.set_xlabel("z1", fontsize=font_ax)
    ax.set_ylabel("z2", fontsize=font_ax)
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]], fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]], fontsize=font_tick)
    ax.legend()
    #plt.colorbar()


from matplotlib.patches import Ellipse

def get_correlation_ellipse_params(x, y):
    """
    Calculate the parameters for a correlation ellipse from points.
    
    Parameters:
    x, y: arrays of coordinates
    
    Returns:
    width, height, angle (in degrees) of the ellipse
    """
    # Calculate covariance matrix
    cov = np.cov(x, y)
    
    # Calculate eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Get the index of the largest eigenvalue
    largest_eigval_ind = np.argmax(eigenvals)
    largest_eigval = eigenvals[largest_eigval_ind]
    largest_eigvec = eigenvecs[:, largest_eigval_ind]
    smallest_eigval = eigenvals[1 - largest_eigval_ind]
    
    # Calculate angle
    angle = np.degrees(np.arctan2(largest_eigvec[1], largest_eigvec[0]))
    
    # Calculate width and height (2 * standard deviation)
    width = 2 * np.sqrt(largest_eigval)
    height = 2 * np.sqrt(smallest_eigval)
    
    return width, height, angle

def plot_trajectory_with_2d_uncertainty(trajectories):
    """
    Plot mean trajectory with 2D standard deviation ellipses.
    
    Parameters:
    trajectories: numpy array of shape (n_trials, n_timesteps, 2) containing 2D trajectories
    """
    # Calculate mean and std across trials
    mean_trajectory = np.mean(trajectories, axis=0)  # shape: (n_timesteps, 2)
    std_trajectory = np.std(trajectories, axis=0)    # shape: (n_timesteps, 2)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot mean trajectory as a line
    ax.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], 
            'b-', linewidth=2, label='Mean Trajectory')
    
    # Plot ellipses for each timestep
    for t in range(mean_trajectory.shape[0]):
        x_points = z_teach[t, :, 0]
        y_points = z_teach[t, :, 1]

        # Calculate ellipse parameters
        width, height, angle = get_correlation_ellipse_params(x_points, y_points)

        ellipse = Ellipse(xy=(mean_trajectory[t, 0], mean_trajectory[t, 1]),
                         width=width,
                         height=height,
                         angle=angle,
                         alpha=0.1,
                         color='gray')
        ax.add_patch(ellipse)
    # Add a sample ellipse for the legend
    legend_ellipse = Ellipse((0, 0), 1, 1, alpha=0.1, color='gray',
                            label='±1 Standard Deviation')
    ax.add_patch(legend_ellipse)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Mean Trajectory with 2D Standard Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')  # Make sure the aspect ratio is 1:1
    
    return fig


# -


from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors

def add_gradient_line(ax, x, y, cmap="viridis", lw=5, vmin=0.4, vmax=1.0, alpha=1):

    # segmenti tra punti consecutivi
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

    # gradiente (valori associati ai segmenti)
    t = np.linspace(vmin, vmax, len(x) - 1)

    lc = LineCollection(
        segs,
        cmap=cmap,
        norm=mcolors.Normalize(vmin, vmax),
        linewidths=lw,
        alpha=alpha,
    )
    lc.set_array(t)
    ax.add_collection(lc)
    return lc

def change_trial_cs(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    move_detector = comm_dict["move_detector"]
    RT_detector = comm_dict["RT_detector"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    SSD = diff_dict["SSD"]
    axis = diff_dict["axis"]
    min_grad = diff_dict["min_grad"]
    lw = diff_dict["lw"]
    cmap = diff_dict["cmap"]
    cmap_go = diff_dict["cmap_go"]
    n_arrows_init = diff_dict["n_arrows_init"]
    color_stop = diff_dict["color_stop"]
    
    set_cs = data["set_cs_ordSSD"]
    cont_cs = data["cont_cs_ordSSD"]
    SSD_cs = data["SSD_cs_ordSSD"]
    steps = set_cs.shape[1]
    t = np.linspace(0, 255, 256, dtype = int)
    t = t[::tau]
    
    f, ax = plt.subplots()
    
    trial = set_cs[SSD_cs==SSD]
    cont_c = cont_cs[SSD_cs==SSD]
    if trial.shape[0] == 0:
        print(f"change the trial with SSD={SSD:d}")
    elif trial.shape[0] != steps:
        print("many trials")
        trial = trial[0]
        cont_c = cont_c[0]
    else:
        trial = trial.squeeze(0)
        cont_c = cont_c.squeeze(0)

    trial = torch.from_numpy(trial).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
    cont_c = torch.from_numpy(cont_c).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
    # generation
    z, z_mean, _ = dmm.inference(trial, cont_c)

    teacher = ((56+SSD)//tau) 
    alone = steps-teacher
    min_grad_mod = (alone*(1-min_grad))/steps
    z_teach = z[:teacher]
    for step in range(alone):
        z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher].unsqueeze(0))
        z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
        z_teach = torch.cat((z_teach, z_gen), dim=0)

    mu_z = z_teach.mean(1)

#         move_logit = move_detector(mu_z.unsqueeze(0))
#         move_output = binary_output(move_logit)
#         move_pred = move_output.squeeze(0).astype(int)
#         gen_string = ["does not", ""]
#         RT_pred = 0

#         if move_pred:
#             RT_output = RT_detector(mu_z.unsqueeze(0))
#             RT_pred = prob_to_RT(RT_output, tau)
#             RT_pred = RT_pred.squeeze(0)
#             gen_string = ["", f" with RT = {RT_pred*tau*5}ms"]

#         mask_nan = ~np.isnan(RT_pred)
#         RT_pred = RT_pred[mask_nan]  
#         RT_cn_ordered_filt = RT_cn_ordered_filt[mask_nan]

    mu_z = mu_z[teacher:].cpu().detach().numpy()
    mu_x = mu_z[:, axis[0]]
    mu_y = mu_z[:, axis[1]]

#     ax.plot(mu_x, mu_y, c ="g", label = f'Fake trial, detected RT={RT_pred*5*tau}$ms$')      
    true_trial = z.mean(1)

#     move_logit = move_detector(true_trial.unsqueeze(0))
#     move_output = binary_output(move_logit)
#     move_rec = move_output.squeeze(0).astype(int)
#     RT_rec = 0

#     if move_rec:
#         RT_output = RT_detector(true_trial.unsqueeze(0))
#         RT_rec = prob_to_RT(RT_output, tau)
#         RT_rec = RT_rec.squeeze(0)

    true_trial = true_trial.cpu().detach().numpy()
    x_true_story = true_trial[:, axis[0]]
    y_true_story = true_trial[:, axis[1]]
    
    x_SSD = true_trial[teacher-1, axis[0]]
    y_SSD = true_trial[teacher-1, axis[1]]
    
    lc1 = add_gradient_line(ax, x_true_story, y_true_story, cmap=cmap, lw=lw, vmin=min_grad, vmax=1.0)
    lc2 = add_gradient_line(ax, mu_x, mu_y, cmap=cmap_go, lw=lw, vmin=min_grad_mod, vmax=1.0)
#     ax.plot(x_true_story, y_true_story, linewidth=2, c = "r", label = f'True traj, SSD={((SSD+56)*5):d}$ms$, detected RT={RT_rec*5*tau}$ms$') 
    n_arrows = int(n_arrows_init*(alone/steps))
    arrow_indices = np.arange(0, len(mu_x), len(mu_x)//n_arrows)  # Place n_arrows arrows along the path
    for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = mu_x[k+1] - mu_x[k]
        dy = mu_y[k+1] - mu_y[k]
        ax.arrow(mu_x[k], mu_y[k], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    n_arrows = n_arrows_init
    arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
    for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = x_true_story[k+1] - x_true_story[k]
        dy = y_true_story[k+1] - y_true_story[k]
        ax.arrow(x_true_story[k], y_true_story[k], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    ax.plot(x_SSD, y_SSD, color=color_stop, marker='x', markeredgewidth = 3, markersize = 15)
    #ax[i].set_title(f"Virtual experiment: changing correct stops trajectories to no stop")#the smallest STOP")
    ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax.set_xlabel("z1")#, fontsize=font_ax)
    ax.set_ylabel("z2")#, fontsize=font_ax)
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)
    
    fig_file = os.path.join(comm_dict["saved_path"], 'cs_to_ws.png')
    plt.savefig(fig_file)


def plot_cs_ns(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    t = comm_dict["t"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    move_detector = comm_dict["move_detector"]
    RT_detector = comm_dict["RT_detector"]
    
    data = diff_dict["data"]
    pca_flag = diff_dict["pca_flag"]
    s = diff_dict["s"]
    s_pc = diff_dict["s_pc"]
    alpha = diff_dict["alpha"]
    n_trials = diff_dict["n_trials"]
    SSD_early = diff_dict["SSD_early"]
    SSD_late = diff_dict["SSD_late"]
    axis = diff_dict["axis"]
    min_grad = diff_dict["min_grad"]
    lw = diff_dict["lw"]
    lw_pc = diff_dict["lw_pc"]
    n_arrows = diff_dict["n_arrows"]
    cmap_ws = diff_dict["cmap_ws"]
    cmap_cs = diff_dict["cmap_cs"]
    color_stop_ws = diff_dict["color_stop_ws"]
    color_stop_cs = diff_dict["color_stop_cs"]
    color_line = diff_dict["color_line"]
    alpha_line = diff_dict["alpha_line"]
    lw = diff_dict["lw"]
    axis_pca = diff_dict["axis_pca"]
    alpha_pc = diff_dict["alpha_pc"]
    inset_dim = diff_dict["inset_dim"]
    inset_font = diff_dict["inset_font"]
    x_min, x_max = diff_dict["xlims"]
    x_min_pc, x_max_pc = diff_dict["xlims_pc"]
    
    SSD_ws = data["SSD_ws_ordRT"]
    RT_ws = data["RT_ws_ordRT"]
    SSD_cs = data["SSD_cs_ordSSD"]
    set_cn = data["set_cn_ordRT"]
    set_ws = data["set_ws_ordRT"]
    set_cs = data["set_cs_ordSSD"]
    
    z_cn, z_ws, z_cs = infer_latent(dmm, data, device)
    
#     if pca_flag:
#         from sklearn.decomposition import PCA

#         pca = PCA(n_components=z_dim)
#         z_cs_flat = pca.fit_transform(z_cs.reshape(-1, z_dim))  
#         z_ws_flat = pca.transform(z_ws.reshape(-1, z_dim))
#         z_cs = z_cs_flat.reshape(steps, -1, z_dim)
#         z_ws = z_ws_flat.reshape(steps, -1, z_dim)
    
    f, ax = plt.subplots()
    
    q_ws = np.where(SSD_ws == SSD_late)[0]
    if q_ws.size == 0:
        print(f"change the trial with SSD={SSD_late:d}")
    elif q_ws.size > 1:
        q_ws = q_ws[0]
    trial_ws = z_ws[:, q_ws]
    
    q_cs = np.where(SSD_cs == SSD_early)[0]
    if q_cs.size == 0:
        print(f"change the trial with SSD={SSD_early:d}")
    elif q_cs.size > 1:
        q_cs = q_cs[0]
    trial_cs = z_cs[:, q_cs]

    
    ws_x = trial_ws[:, axis[0]]
    ws_y = trial_ws[:, axis[1]]
    
    cs_x = trial_cs[:, axis[0]]
    cs_y = trial_cs[:, axis[1]]

    ws_SSD_x = trial_ws[(56+SSD_late)//tau, axis[0]]
    ws_SSD_y = trial_ws[(56+SSD_late)//tau, axis[1]]
    
    cs_SSD_x = trial_cs[(56+SSD_late)//tau, axis[0]]
    cs_SSD_y = trial_cs[(56+SSD_late)//tau, axis[1]]
    
    lc1 = add_gradient_line(ax, ws_x, ws_y, cmap=cmap_ws, lw=lw, vmin=min_grad, vmax=1.0)
    lc2 = add_gradient_line(ax, cs_x, cs_y, cmap=cmap_cs, lw=lw, vmin=min_grad, vmax=1.0)

    arrow_indices = np.arange(0, len(ws_x), len(ws_x)//n_arrows)  # Place n_arrows arrows along the path
    for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = ws_x[k+1] - ws_x[k]
        dy = ws_y[k+1] - ws_y[k]
        ax.arrow(ws_x[k], ws_y[k], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
        
    arrow_indices = np.arange(0, len(cs_x), len(cs_x)//n_arrows)  # Place n_arrows arrows along the path
    for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = cs_x[k+1] - cs_x[k]
        dy = cs_y[k+1] - cs_y[k]
        ax.arrow(cs_x[k], cs_y[k], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
        
    ax.plot(ws_SSD_x, ws_SSD_y, color=color_stop_ws, marker='x', markeredgewidth = 3, markersize = 15)
    ax.plot(cs_SSD_x, cs_SSD_y, color=color_stop_cs, marker='x', markeredgewidth = 3, markersize = 15)
    #ax[i].set_title(f"Virtual experiment: changing correct stops trajectories to no stop")#the smallest STOP")
    ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax.set_xlabel("$z_1$")#, fontsize=font_ax)
    ax.set_ylabel("$z_2$")#, fontsize=font_ax)
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)
    
    fig_file = os.path.join(comm_dict["saved_path"], 'ws_and_cs.png')
    plt.savefig(fig_file)
    
    
    
    
    #------------------ Discriminator line for z  -----------------
    
    z_cn, z_ws, z_cs = infer_latent(dmm, data, device, n_trials)
    if n_trials>1:
        z_cn = z_cn.reshape(steps, n_cn, n_trials, z_dim)
        z_cn = z_cn.mean(2)
        z_ws = z_ws.reshape(steps, n_ws, n_trials, z_dim)
        z_ws = z_ws.mean(2)
        z_cs = z_cs.reshape(steps, n_cs, n_trials, z_dim)
        z_cs = z_cs.mean(2)
    
    z_cs_stop = z_cs[(SSD_cs+56)//tau, np.arange(z_cs.shape[1])]
    z_ws_stop = z_ws[(SSD_ws+56)//tau, np.arange(z_ws.shape[1])]
    
    X_t = np.vstack([z_cs_stop, z_ws_stop])           # shape (n1+n2, 3)
    y_t = np.array([1]*z_cs.shape[1] + [2]*z_ws.shape[1])
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score
    
    # LDA binario 
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_t, y_t)

    # Predizioni hard (per accuracy)
    y_pred = lda.predict(X_t)
    acc = (y_pred == y_t).mean()
    
    # Score continui per AUC (decision_function è adatto al ROC)
    scores = lda.decision_function(X_t)
    # sklearn vuole label binarie 0/1; rimappiamo 1 -> 0, 2 -> 1
    y_bin = (y_t == 2).astype(int)
    auc = roc_auc_score(y_bin, scores)
    
    print(f"accuracy: {acc:.2f}")
    print(f"AUC: {auc:.2f}")
    
    w = lda.coef_[0]       # shape (3,)
    b = lda.intercept_[0]  # scalare

    wi = w[axis[0]]
    wj = w[axis[1]]
    xx = np.linspace(x_min, x_max, 200)
    yy = -(wi * xx + b) / wj
    
    f, ax = plt.subplots()
    
    lc1 = add_gradient_line(ax, ws_x, ws_y, cmap=cmap_ws, lw=lw, vmin=min_grad, vmax=1.0, alpha=alpha)
    lc2 = add_gradient_line(ax, cs_x, cs_y, cmap=cmap_cs, lw=lw, vmin=min_grad, vmax=1.0, alpha=alpha)

    arrow_indices = np.arange(0, len(ws_x), len(ws_x)//n_arrows)  # Place n_arrows arrows along the path
    for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = ws_x[k+1] - ws_x[k]
        dy = ws_y[k+1] - ws_y[k]
        ax.arrow(ws_x[k], ws_y[k], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=alpha)
        
    arrow_indices = np.arange(0, len(cs_x), len(cs_x)//n_arrows)  # Place n_arrows arrows along the path
    for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = cs_x[k+1] - cs_x[k]
        dy = cs_y[k+1] - cs_y[k]
        ax.arrow(cs_x[k], cs_y[k], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=alpha)
    ax.scatter(z_cs_stop[:, axis[0]], z_cs_stop[:, axis[1]], s = s, c = 'r', alpha = 0.8, label = "cs stops")
    ax.scatter(z_ws_stop[:, axis[0]], z_ws_stop[:, axis[1]], s = s, c = 'g', alpha = 0.8, label = "ws stops")
    ax.plot(xx, yy, color = color_line, lw=lw, alpha=alpha_line)
    ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax.set_xlabel("$z_1$")#, fontsize=font_ax)
    ax.set_ylabel("$z_2$")#, fontsize=font_ax)
    ax.set_xticks(z_ticks[axis[0]])
    ax.set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)
    
    
    
    #------------ inset PCA plot ------------------
    
    axins = ax.inset_axes(inset_dim)
    
    with np.load(comm_dict["saved_path"]+"/data_split.npz", allow_pickle=True) as loaded_file:
        train_set = loaded_file["train_set"]
        test_set = loaded_file["test_set"]
        train_direction = loaded_file["train_direction"]
        test_direction = loaded_file["test_direction"]
        train_SSD = loaded_file["train_SSD"]
        test_SSD = loaded_file["test_SSD"]
        train_RT = loaded_file["train_RT"]
        test_RT = loaded_file["test_RT"]
            
    n_train, steps, features = train_set.shape
    X_train = train_set.reshape(-1, features)  # shape = (n_train * time_steps, 96)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=z_dim)
    pca.fit(X_train)  # fit + transform sul train 
    
    pc_cn = pca.transform(set_cn.reshape(-1, features)).reshape(-1, steps, z_dim).transpose(1, 0, 2)
    pc_cs = pca.transform(set_cs.reshape(-1, features)).reshape(-1, steps, z_dim).transpose(1, 0, 2)
    pc_ws = pca.transform(set_ws.reshape(-1, features)).reshape(-1, steps, z_dim).transpose(1, 0, 2)
    
    pc_cs_stop = pc_cs[(SSD_cs+56)//tau, np.arange(pc_cs.shape[1])]
    pc_ws_stop = pc_ws[(SSD_ws+56)//tau, np.arange(pc_ws.shape[1])]
    
    X_pc = np.vstack([pc_cs_stop, pc_ws_stop])           # shape (n1+n2, 3)
    y_pc = np.array([1]*pc_cs.shape[1] + [2]*pc_ws.shape[1])
    
    # LDA binario 
    lda_pc = LinearDiscriminantAnalysis()
    lda_pc.fit(X_pc, y_pc)

    # Predizioni hard (per accuracy)
    y_pred_pc = lda_pc.predict(X_pc)
    acc_pc = (y_pred_pc == y_pc).mean()
    
    # Score continui per AUC (decision_function è adatto al ROC)
    scores_pc = lda_pc.decision_function(X_pc)
    # sklearn vuole label binarie 0/1; rimappiamo 1 -> 0, 2 -> 1
    y_bin_pc = (y_pc == 2).astype(int)
    auc_pc = roc_auc_score(y_bin_pc, scores_pc)
    
    print(f"accuracy: {acc_pc:.2f}")
    print(f"AUC: {auc_pc:.2f}")
    
    w_pc = lda_pc.coef_[0]       # shape (3,)
    b_pc = lda_pc.intercept_[0]  # scalare
    
    wi_pc = w_pc[axis_pca[0]]
    wj_pc = w_pc[axis_pca[1]]
    xx_pc = np.linspace(x_min_pc, x_max_pc, 200)
    yy_pc = -(wi_pc * xx_pc + b_pc) / wj_pc
    
    trial_cs = pc_cs[:, q_cs]
    trial_ws = pc_ws[:, q_ws]
    
    ws_x = trial_ws[:, axis_pca[0]]
    ws_y = trial_ws[:, axis_pca[1]]
    
    cs_x = trial_cs[:, axis_pca[0]]
    cs_y = trial_cs[:, axis_pca[1]]
    
    lc1 = add_gradient_line(axins, ws_x, ws_y, cmap=cmap_ws, lw=lw_pc, vmin=min_grad, vmax=1.0, alpha=alpha)
    lc2 = add_gradient_line(axins, cs_x, cs_y, cmap=cmap_cs, lw=lw_pc, vmin=min_grad, vmax=1.0, alpha=alpha)

    arrow_indices = np.arange(0, len(ws_x), len(ws_x)//n_arrows)  # Place n_arrows arrows along the path
    for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = ws_x[k+1] - ws_x[k]
        dy = ws_y[k+1] - ws_y[k]
        axins.arrow(ws_x[k], ws_y[k], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=alpha)
        
    arrow_indices = np.arange(0, len(cs_x), len(cs_x)//n_arrows)  # Place n_arrows arrows along the path
    for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = cs_x[k+1] - cs_x[k]
        dy = cs_y[k+1] - cs_y[k]
        axins.arrow(cs_x[k], cs_y[k], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=alpha)
    axins.scatter(pc_cs_stop[:, axis_pca[0]], pc_cs_stop[:, axis_pca[1]], s = s_pc, c = 'r', alpha = alpha_pc, label = "cs stops")
    axins.scatter(pc_ws_stop[:, axis_pca[0]], pc_ws_stop[:, axis_pca[1]], s = s_pc, c = 'g', alpha = alpha_pc, label = "ws stops")
    axins.plot(xx_pc, yy_pc, color = color_line, lw=lw_pc, alpha=alpha_line)
#     axins.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
#     axins.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    axins.set_xlabel(f"$PC_{axis_pca[0]+1}$", fontsize=inset_font)
    axins.set_ylabel(f"$PC_{axis_pca[1]+1}$", fontsize=inset_font)

    # Calcola i limiti comuni
    min_val_x_c = axins.get_xlim()[0]
    min_val_y_c = axins.get_ylim()[0]
    max_val_x_c = axins.get_xlim()[1]
    max_val_y_c = axins.get_ylim()[1]

    dx_c = (max_val_x_c - min_val_x_c)//3
    dy_c = (max_val_y_c - min_val_y_c)//3

#     axins.set_xlabel('mean RT ($ms$)', fontsize=inset_font)
#     axins.set_ylabel('SSRT', fontsize=inset_font)
    axins.set_xticks([])
    axins.set_yticks([])
#     axins.set_xticks([int(((min_val_x_c + dx_c)//2)*2), int(((min_val_x_c + 2*dx_c)//2 + 1)*2)])
#     axins.set_xticklabels([int(((min_val_x_c + dx_c)//2)*2), int(((min_val_x_c + 2*dx_c)//2 + 1)*2)])#, fontsize=font_tick)
#     axins.set_yticks([int(((min_val_y_c + dy_c)//2 + 1)*2), int(((min_val_y_c + 2*dy_c)//2 + 2)*2)])
#     axins.set_yticklabels([int(((min_val_y_c + dy_c)//2 + 1)*2), int(((min_val_y_c + 2*dy_c)//2 + 2)*2)])#, fontsize=font_tick)
    axins.tick_params(labelsize=inset_font)
    
    fig_file = os.path.join(comm_dict["saved_path"], 'ws_vs_cs_stop.png')
    plt.savefig(fig_file)
    
    return xx, yy




def plot_cs_ns_pca(comm_dict, diff_dict):
    
    with np.load(comm_dict["saved_path"]+"/data_split.npz", allow_pickle=True) as loaded_file:
            train_set = loaded_file["train_set"]
            vali_set = loaded_file["vali_set"]
            test_set = loaded_file["test_set"]
            train_direction = loaded_file["train_direction"]
            vali_direction = loaded_file["vali_direction"]
            test_direction = loaded_file["test_direction"]
            train_SSD = loaded_file["train_SSD"]
            vali_SSD = loaded_file["vali_SSD"]
            test_SSD = loaded_file["test_SSD"]
            train_RT = loaded_file["train_RT"]
            vali_RT = loaded_file["vali_RT"]
            test_RT = loaded_file["test_RT"]
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    t = comm_dict["t"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    move_detector = comm_dict["move_detector"]
    RT_detector = comm_dict["RT_detector"]
    
    data = diff_dict["data"]
    pca_flag = diff_dict["pca_flag"]
    s = diff_dict["s"]
    alpha = diff_dict["alpha"]
    n_trials = diff_dict["n_trials"]
    SSD_early = diff_dict["SSD_early"]
    SSD_late = diff_dict["SSD_late"]
    axis = diff_dict["axis"]
    min_grad = diff_dict["min_grad"]
    lw = diff_dict["lw"]
    n_arrows = diff_dict["n_arrows"]
    cmap_ws = diff_dict["cmap_ws"]
    cmap_cs = diff_dict["cmap_cs"]
    color_stop_ws = diff_dict["color_stop_ws"]
    color_stop_cs = diff_dict["color_stop_cs"]
    color_line = diff_dict["color_line"]
    alpha_line = diff_dict["alpha_line"]
    lw = diff_dict["lw"]
    x_min, x_max = diff_dict["xlims"]
    
    print(train_set.shape)
    n_train, steps, features = train_set.shape
    X_train = train_set.reshape(-1, features)  # shape = (n_train * time_steps, 96)
   
    from sklearn.decomposition import PCA
    pca = PCA(n_components=z_dim)
    X_train_pca = pca.fit_transform(X_train)  # fit + transform sul train 
    train_pca = X_train_pca.reshape(n_train, steps, z_dim)
    train_pca = np.transpose(train_pca, (1, 0, 2))
    
    mask_ws = (train_RT!=0) & (train_SSD!=0)
    mask_cs = train_RT==0
    mask_ns = train_SSD==0
    
    z_cn = train_pca[:, mask_ns]
    z_cs = train_pca[:, mask_cs]
    z_ws = train_pca[:, mask_ws]
    
#     test_rec = test_PCA(train_set, test_set, z_dim)
    
    SSD_ws = train_SSD[mask_ws]
    RT_ws = train_RT[mask_ws]
    SSD_cs = train_SSD[mask_cs]
       
    if pca_flag:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=z_dim)
        z_cs_flat = pca.fit_transform(z_cs.reshape(-1, z_dim))  
        z_ws_flat = pca.transform(z_ws.reshape(-1, z_dim))
        z_cs = z_cs_flat.reshape(steps, -1, z_dim)
        z_ws = z_ws_flat.reshape(steps, -1, z_dim)
    
    f, ax = plt.subplots()
    
    trial_ws = z_ws[:, SSD_ws==SSD_late]
    if trial_ws.shape[1] == 0:
        print(f"change the trial with SSD={SSD_late:d}")
    elif trial_ws.shape[1] != 1:
        print("many trials")
        trial_ws = trial_ws[:, 0]
    else:
        trial_ws = trial_ws.squeeze(1)
        
    trial_cs = z_cs[:, SSD_cs==SSD_early]
    if trial_cs.shape[1] == 0:
        print(f"change the trial with SSD={SSD_early:d}")
    elif trial_cs.shape[1] != 1:
        print("many trials")
        trial_cs = trial_cs[:, 0]
    else:
        trial_cs = trial_cs.squeeze(1)

    ws_x = trial_ws[:, axis[0]]
    ws_y = trial_ws[:, axis[1]]
    
    cs_x = trial_cs[:, axis[0]]
    cs_y = trial_cs[:, axis[1]]

    ws_SSD_x = trial_ws[(56+SSD_late)//tau, axis[0]]
    ws_SSD_y = trial_ws[(56+SSD_late)//tau, axis[1]]
    
    cs_SSD_x = trial_cs[(56+SSD_late)//tau, axis[0]]
    cs_SSD_y = trial_cs[(56+SSD_late)//tau, axis[1]]
    
    lc1 = add_gradient_line(ax, ws_x, ws_y, cmap=cmap_ws, lw=lw, vmin=min_grad, vmax=1.0)
    lc2 = add_gradient_line(ax, cs_x, cs_y, cmap=cmap_cs, lw=lw, vmin=min_grad, vmax=1.0)

    arrow_indices = np.arange(0, len(ws_x), len(ws_x)//n_arrows)  # Place n_arrows arrows along the path
    for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = ws_x[k+1] - ws_x[k]
        dy = ws_y[k+1] - ws_y[k]
        ax.arrow(ws_x[k], ws_y[k], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
        
    arrow_indices = np.arange(0, len(cs_x), len(cs_x)//n_arrows)  # Place n_arrows arrows along the path
    for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = cs_x[k+1] - cs_x[k]
        dy = cs_y[k+1] - cs_y[k]
        ax.arrow(cs_x[k], cs_y[k], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
        
    ax.plot(ws_SSD_x, ws_SSD_y, color=color_stop_ws, marker='x', markeredgewidth = 3, markersize = 15)
    ax.plot(cs_SSD_x, cs_SSD_y, color=color_stop_cs, marker='x', markeredgewidth = 3, markersize = 15)
    #ax[i].set_title(f"Virtual experiment: changing correct stops trajectories to no stop")#the smallest STOP")
#     ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
#     ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax.set_xlabel("z1")#, fontsize=font_ax)
    ax.set_ylabel("z2")#, fontsize=font_ax)
#     ax.set_xticks(z_ticks[axis[0]]) 
#     ax.set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
#     ax.set_yticks(z_ticks[axis[1]])
#     ax.set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)
    
    fig_file = os.path.join(comm_dict["saved_path"], 'ws_and_cs_pca.png')
    plt.savefig(fig_file)
    
    z_cs_stop = z_cs[(SSD_cs+56)//tau, np.arange(z_cs.shape[1])]
    z_ws_stop = z_ws[(SSD_ws+56)//tau, np.arange(z_ws.shape[1])]
    
    x_cs = z_cs_stop[:, axis[0]]
    x_ws = z_ws_stop[:, axis[0]]

    y_cs = z_cs_stop[:, axis[1]]
    y_ws = z_ws_stop[:, axis[1]]
    
    X_cs = np.column_stack((x_cs, y_cs))
    X_ws = np.column_stack((x_ws, y_ws))
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score
    
    X_t = np.vstack([X_cs, X_ws])           # shape (n1+n2, 3)
    y_t = np.array([1]*X_cs.shape[0] + [2]*X_ws.shape[0])
    
    # LDA binario [web:29][web:32]
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_t, y_t)

    # Predizioni hard (per accuracy)
    y_pred = lda.predict(X_t)
    acc = (y_pred == y_t).mean()
    
    # Score continui per AUC (decision_function è adatto al ROC) [web:32][web:31]
    scores = lda.decision_function(X_t)
    # sklearn vuole label binarie 0/1; rimappiamo 1 -> 0, 2 -> 1
    y_bin = (y_t == 2).astype(int)
    auc = roc_auc_score(y_bin, scores)
    
    print(f"accuracy: {acc:.2f}")
    print(f"AUC: {auc:.2f}")
    
    # la retta trovta ha la forma ax + cy + b = 0 
    w = lda.coef_[0]       # shape (2,)
    b = lda.intercept_[0]  # scalare

    # normal = vettore normale al piano
    normal = w / np.linalg.norm(w)
    print("Vettore normale al piano discriminante:", normal)

#     x_min, x_max = min(z_cs_stop[:, 0].min(), z_ws_stop[:, 0].min()), max(z_cs_stop[:, 0].max(), z_ws_stop[:, 0].max())
#     y_min, y_max = min(z_cs_stop[:, 1].min(), z_ws_stop[:, 1].min()), max(z_cs_stop[:, 1].max(), z_ws_stop[:, 1].max())

    a, c = w  
    xx = np.linspace(x_min, x_max, 200)
    yy = (-b - a * xx) / c
    
    f, ax = plt.subplots()
    
    lc1 = add_gradient_line(ax, ws_x, ws_y, cmap=cmap_ws, lw=lw, vmin=min_grad, vmax=1.0, alpha=alpha)
    lc2 = add_gradient_line(ax, cs_x, cs_y, cmap=cmap_cs, lw=lw, vmin=min_grad, vmax=1.0, alpha=alpha)

    arrow_indices = np.arange(0, len(ws_x), len(ws_x)//n_arrows)  # Place n_arrows arrows along the path
    for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = ws_x[k+1] - ws_x[k]
        dy = ws_y[k+1] - ws_y[k]
        ax.arrow(ws_x[k], ws_y[k], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=alpha)
        
    arrow_indices = np.arange(0, len(cs_x), len(cs_x)//n_arrows)  # Place n_arrows arrows along the path
    for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = cs_x[k+1] - cs_x[k]
        dy = cs_y[k+1] - cs_y[k]
        ax.arrow(cs_x[k], cs_y[k], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=alpha)
    ax.scatter(x_cs, y_cs, s = s, c = 'r', alpha = 0.8, label = "cs stops")
    ax.scatter(x_ws, y_ws, s = s, c = 'g', alpha = 0.8, label = "ws stops")
    ax.plot(xx, yy, color = color_line, lw=lw, alpha=alpha_line)
#     ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
#     ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax.set_xlabel("z1")#, fontsize=font_ax)
    ax.set_ylabel("z2")#, fontsize=font_ax)
#     ax.set_xticks(z_ticks[axis[0]]) 
#     ax.set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
#     ax.set_yticks(z_ticks[axis[1]])
#     ax.set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)
    
    fig_file = os.path.join(comm_dict["saved_path"], 'ws_vs_cs_stop_pca.png')
    plt.savefig(fig_file)
    
    return xx, yy



def change_trial_ws(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    move_detector = comm_dict["move_detector"]
    RT_detector = comm_dict["RT_detector"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    SSD = diff_dict["SSD"]
    axis = diff_dict["axis"]
    min_grad = diff_dict["min_grad"]
    lw = diff_dict["lw"]
    n_arrows_init = diff_dict["n_arrows_init"]
    cmap = diff_dict["cmap"]
    cmap_go = diff_dict["cmap_go"]
    color_fstop = diff_dict["color_fstop"]
    color_stop = diff_dict["color_stop"]
    anticipation = diff_dict["anticipation"]
    
    set_ws = data["set_ws_ordSSD"]
    cont_ws = data["cont_ws_ordSSD"]
    SSD_ws = data["SSD_ws_ordSSD"]
    steps = set_ws.shape[1]
    t = np.linspace(0, 255, 256, dtype = int)
    t = t[::tau]
    
    f, ax = plt.subplots()
    
    trial = set_ws[SSD_ws==SSD]
    cont_c = cont_ws[SSD_ws==SSD]
    if trial.shape[0] == 0:
        print(f"change the trial with SSD={SSD:d}")
    elif trial.shape[0] != 1:
        print("many trials")
        trial = trial[0]
        cont_c = cont_c[0]
    else:
        trial = trial.squeeze(0)
        cont_c = cont_c.squeeze(0)

    trial = torch.from_numpy(trial).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
    cont_c = torch.from_numpy(cont_c).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
    
    cont = torch.zeros((1, n_trials, 4)).to(device)
    cont[:, :, 3] = 1
    
    # generation
    z, z_mean, _ = dmm.inference(trial, cont_c)

    teacher = ((56+SSD)//tau) - anticipation//(5*tau)
    alone = steps-teacher
    min_grad_mod = (alone*(1-min_grad))/steps
    z_teach = z[:teacher]
    for step in range(alone):
        z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont)
        z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
        z_teach = torch.cat((z_teach, z_gen), dim=0)

    mu_z = z_teach.mean(1)

#         move_logit = move_detector(mu_z.unsqueeze(0))
#         move_output = binary_output(move_logit)
#         move_pred = move_output.squeeze(0).astype(int)
#         gen_string = ["does not", ""]
#         RT_pred = 0

#         if move_pred:
#             RT_output = RT_detector(mu_z.unsqueeze(0))
#             RT_pred = prob_to_RT(RT_output, tau)
#             RT_pred = RT_pred.squeeze(0)
#             gen_string = ["", f" with RT = {RT_pred*tau*5}ms"]

#         mask_nan = ~np.isnan(RT_pred)
#         RT_pred = RT_pred[mask_nan]  
#         RT_cn_ordered_filt = RT_cn_ordered_filt[mask_nan]

    mu_z = mu_z[teacher:].cpu().detach().numpy()
    mu_x = mu_z[:, axis[0]]
    mu_y = mu_z[:, axis[1]]

#     ax.plot(mu_x, mu_y, c ="g", label = f'Fake trial, detected RT={RT_pred*5*tau}$ms$')      
    true_trial = z.mean(1)

#     move_logit = move_detector(true_trial.unsqueeze(0))
#     move_output = binary_output(move_logit)
#     move_rec = move_output.squeeze(0).astype(int)
#     RT_rec = 0

#     if move_rec:
#         RT_output = RT_detector(true_trial.unsqueeze(0))
#         RT_rec = prob_to_RT(RT_output, tau)
#         RT_rec = RT_rec.squeeze(0)

    true_trial = true_trial.cpu().detach().numpy()
    x_true_story = true_trial[:, axis[0]]
    y_true_story = true_trial[:, axis[1]]

#     x_SSD = true_trial[teacher-1, axis[0]]
#     y_SSD = true_trial[teacher-1, axis[1]]

    x_SSD = true_trial[(56+SSD)//tau, axis[0]]
    y_SSD = true_trial[(56+SSD)//tau, axis[1]]
    
    x_stop = true_trial[teacher-1, axis[0]]
    y_stop = true_trial[teacher-1, axis[1]]
    
    
    
    lc1 = add_gradient_line(ax, x_true_story, y_true_story, cmap=cmap, lw=lw, vmin=min_grad, vmax=1.0)
    lc2 = add_gradient_line(ax, mu_x, mu_y, cmap=cmap_go, lw=lw, vmin=min_grad_mod, vmax=1.0)
#     ax.plot(x_true_story, y_true_story, linewidth=2, c = "r", label = f'True traj, SSD={((SSD+56)*5):d}$ms$, detected RT={RT_rec*5*tau}$ms$') 
    n_arrows = int(n_arrows_init*(alone/steps))
    arrow_indices = np.arange(0, len(mu_x), len(mu_x)//n_arrows)  # Place n_arrows arrows along the path
    for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = mu_x[k+1] - mu_x[k]
        dy = mu_y[k+1] - mu_y[k]
        ax.arrow(mu_x[k], mu_y[k], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
        
    arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows_init)  # Place n_arrows arrows along the path
    for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = x_true_story[k+1] - x_true_story[k]
        dy = y_true_story[k+1] - y_true_story[k]
        ax.arrow(x_true_story[k], y_true_story[k], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    ax.plot(x_SSD, y_SSD, color=color_stop, marker='x', markeredgewidth = 3, markersize = 15)
    ax.plot(x_stop, y_stop, color=color_fstop, marker='x', markeredgewidth = 3, markersize = 15)
    #ax[i].set_title(f"Virtual experiment: changing correct stops trajectories to no stop")#the smallest STOP")
    ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax.set_xlabel("z1")#, fontsize=font_ax)
    ax.set_ylabel("z2")#, fontsize=font_ax)
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)
    
    fig_file = os.path.join(comm_dict["saved_path"], 'ws_to_cs.png')
    plt.savefig(fig_file)


def change_trial_cn(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    RT_detector = comm_dict["RT_detector"]
    move_detector = comm_dict["move_detector"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    stop_array = diff_dict["stop_array"]
#     color = diff_dict["color"]
    RT = diff_dict["RT"]
    axis = diff_dict["axis"]
#     color_true = diff_dict["color_true"]
    min_grad = diff_dict["min_grad"]
    min_grad_mod = diff_dict["min_grad_mod"]
    lw = diff_dict["lw"]
    n_arrows_init = diff_dict["n_arrows_init"]
    cmap = diff_dict["cmap"]
    cmap_stop = diff_dict["cmap_stop"]
    color_stop = diff_dict["color_stop"]
    color_line = diff_dict["color_line"]
    alpha_line = diff_dict["alpha_line"]
    lw = diff_dict["lw"]
    xx, yy = diff_dict["line"]
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    set_cs = data["set_cs_ordSSD"]
    steps = set_cn.shape[1]
    l = len(stop_array)
    t = np.linspace(0, 255, 256, dtype = int)
    t = t[::tau]
        
    trial = set_cn[RT_cn==RT]
    cont_c = cont_cn[RT_cn==RT]
    if trial.shape[0] == 0:
        print(f"change the trial with RT={RT:d}")
    elif trial.shape[0] != steps:
        trial = trial[0]
        cont_c = cont_c[0]
    else:
        trial = trial.squeeze(0)
        cont_c = cont_c.squeeze(0)

    cont = torch.zeros((1, n_trials, 4)).to(device)
    cont[:, :, 3] = 1

    trial = torch.from_numpy(trial).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
    cont_c = torch.from_numpy(cont_c).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
    # generation
    z, z_mean, _ = dmm.inference(trial, cont_c)
    
    true_trial = z.mean(1)
            
    move_logit = move_detector(true_trial.unsqueeze(0))
    move_output = binary_output(move_logit)
    move_rec = move_output.squeeze(0).astype(int)
    RT_rec = 0

    if move_rec:
        RT_output = RT_detector(true_trial.unsqueeze(0))
        RT_rec = prob_to_RT(RT_output, tau)
        RT_rec = RT_rec.squeeze(0)

    true_trial = true_trial.cpu().detach().numpy()
    x_true_story = true_trial[:, axis[0]]
    y_true_story = true_trial[:, axis[1]]
    
    plot_label = ["ns_to_cs", "ns_to_ws"]
        
    for i, stop in enumerate(stop_array):           
        teacher = 56//tau + stop//(5*tau)
        alone = steps-teacher
        frac_mod = alone/steps
#         min_grad_mod = 1-(frac_mod*(1-min_grad))
        z_teach = z[:teacher]
        for step in range(alone):
            z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont)
            z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
            z_teach = torch.cat((z_teach, z_gen), dim=0)

        mu_z = z_teach.mean(1)

#         move_logit = move_detector(mu_z.unsqueeze(0))
#         move_output = binary_output(move_logit)
#         move_pred = move_output.squeeze(0).astype(int)
#         RT_pred = 0

#         if move_pred:
#             RT_output = RT_detector(mu_z.unsqueeze(0))
#             RT_pred = prob_to_RT(RT_output, tau)
#             RT_pred = RT_pred.squeeze(0)

        mu_z = mu_z[teacher:].cpu().detach().numpy()
        mu_x = mu_z[:, axis[0]]
        mu_y = mu_z[:, axis[1]]
        
        f, ax = plt.subplots()
        
        n_arrows = int(n_arrows_init*(alone/steps))
        arrow_indices = np.arange(0, len(mu_x), len(mu_x)//n_arrows)  # Place n_arrows arrows along the path
        
        lc1 = add_gradient_line(ax, x_true_story, y_true_story, cmap=cmap, lw=lw, vmin=min_grad, vmax=1.0)
        lc2 = add_gradient_line(ax, mu_x, mu_y, cmap=cmap_stop, lw=lw, vmin=min_grad_mod, vmax=1.0)
#         ax[i].plot(mu_x, mu_y, c=color[i], label = f'Fake traj, detected RT={RT_pred*5*tau}$ms$')
#         ax[i].plot(x_true_story, y_true_story, linewidth=2, c = color_true, label = f'True traj, true RT={((RT+56)*5):d}$ms$, Detected RT={RT_rec*5*tau}$ms$') 
        #ax[i].plot(t*5, RT_prob, linewidth=2, c = color[i], label = f'estimate RT={RT_pred*5*tau}$ms$') 
        for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
            dx = mu_x[k+1] - mu_x[k]
            dy = mu_y[k+1] - mu_y[k]
            ax.arrow(mu_x[k], mu_y[k], dx, dy,
                    head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)    
        arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows_init)  # Place n_arrows arrows along the path
        for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
            dx = x_true_story[k+1] - x_true_story[k]
            dy = y_true_story[k+1] - y_true_story[k]
            ax.arrow(x_true_story[k], y_true_story[k], dx, dy,
                    head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
            
        x_SSD = true_trial[teacher-1, axis[0]]
        y_SSD = true_trial[teacher-1, axis[1]]
        #ax[i].set_title(f"Virtual experiment: changing no stops trajectories for different stops signals")#the smallest STOP")
        ax.plot(x_SSD, y_SSD, color=color_stop, marker='x', markeredgewidth = 3, markersize = 15)
        ax.plot(xx, yy, color = color_line, lw=lw, alpha=alpha_line)
        ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
        ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
        ax.set_xlabel("$z_1$")#, fontsize=font_ax)
        ax.set_ylabel("$z_2$")#, fontsize=font_ax)
        ax.set_xticks(z_ticks[axis[0]]) 
        ax.set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
        ax.set_yticks(z_ticks[axis[1]])
        ax.set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)
#     ax.autoscale_view()
        fig_file = os.path.join(comm_dict["saved_path"], f'{plot_label[i]}.png')
        plt.savefig(fig_file)
        plt.show()


# def change_trial_cn(comm_dict, diff_dict):

#     dmm = comm_dict["dmm"]
#     device = comm_dict["device"]
#     tau = comm_dict["tau"]
#     z_lims = comm_dict["z_lims"]
#     z_ticks = comm_dict["z_ticks"]
#     font_ax = comm_dict["font_ax"]
#     font_tick = comm_dict["font_tick"]
#     font_leg = comm_dict["font_leg"]
#     fig_size = comm_dict["fig_size"]
#     RT_detector = comm_dict["RT_detector"]
#     move_detector = comm_dict["move_detector"]

#     data = diff_dict["data"]
#     n_trials = diff_dict["n_trials"]
#     stop_array = diff_dict["stop_array"]
#     color = diff_dict["color"]
#     RT_array = diff_dict["RT_array"]
#     n_cont = diff_dict["n_cont"]
#     axis = diff_dict["axis"]
#     color_true = diff_dict["color_true"]

#     set_cn = data["set_cn_ordRT"]
#     cont_cn = data["cont_cn_ordRT"]
#     RT_cn = data["RT_cn_ordRT"]
#     set_cs = data["set_cs_ordSSD"]
#     steps = set_cn.shape[1]
#     n = len(RT_array)
#     l = len(stop_array)
#     t = np.linspace(0, 255, 256, dtype = int)
#     t = t[::tau]

#     if n>1:
#         f, ax = plt.subplots(1, len(RT_array), figsize = (7*len(RT_array), 6))
#     else:
#         f, ax = plt.subplots(figsize = (7, 6))
#     f.suptitle("Virtual experiment: changing correct stops trajectories to no-stop")

#     for i, RT in enumerate(RT_array):     
#         trial = set_cn[RT_cn==RT]
#         cont_c = cont_cn[RT_cn==RT]
#         if trial.shape[0] == 0:
#             print(f"change the trial with RT={RT:d}")
#         elif trial.shape[0] != steps:
#             trial = trial[0]
#             cont_c = cont_c[0]
#         else:
#             trial = trial.squeeze(0)
#             cont_c = cont_c.squeeze(0)

# #         if n_cont == 3:
# #             cont = torch.zeros((1, n_trials, 2)).to(device)
# #         else:
# #             cont = torch.ones((1, n_trials, 2)).to(device)

#         cont = torch.zeros((1, n_trials, 4)).to(device)
#         cont[:, :, 3] = 1

#         trial = torch.from_numpy(trial).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
#         cont_c = torch.from_numpy(cont_c).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
#         # generation
#         z, z_mean, _ = dmm.inference(trial, cont_c)

#         for j, stop in enumerate(stop_array):   
#             teacher = stop//(5*tau)
#             alone = steps-teacher
#             z_teach = z[:teacher]
#             for step in range(alone):
#                 z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont)
#                 z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
#                 z_teach = torch.cat((z_teach, z_gen), dim=0)

#             mu_z = z_teach.mean(1)

#             move_logit = move_detector(mu_z.unsqueeze(0))
#             move_output = binary_output(move_logit)
#             move_pred = move_output.squeeze(0).astype(int)
#             RT_pred = 0

#             if move_pred:
#                 RT_output = RT_detector(mu_z.unsqueeze(0))
#                 RT_pred = prob_to_RT(RT_output, tau)
#                 RT_pred = RT_pred.squeeze(0)

#             mu_z = mu_z.cpu().detach().numpy()
#             mu_x = mu_z[:, axis[0]]
#             mu_y = mu_z[:, axis[1]]

#             n_arrows = 15
#             arrow_indices = np.arange(0, len(mu_x), len(mu_x)//n_arrows)  # Place n_arrows arrows along the path

#             if n>1:
#                 ax[i].plot(mu_x, mu_y, c=color[j], label = f'Fake traj, detected RT={RT_pred*5*tau}$ms$')
#                 #ax[i].plot(t*5, RT_prob, linewidth=2, c = color[j], label = f'estimate RT={RT_pred*5*tau}$ms$') 
#                 for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
#                     dx = mu_x[k+1] - mu_x[k]
#                     dy = mu_y[k+1] - mu_y[k]
#                     ax[i].arrow(mu_x[k], mu_y[k], dx, dy,
#                             head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
#             else:
#                 ax.plot(mu_x, mu_y, c=color[j], label = f'Fake traj, detected RT={RT_pred*5*tau}$ms$')
#                 #ax[i].plot(t*5, RT_prob, linewidth=2, c = color[j], label = f'estimate RT={RT_pred*5*tau}$ms$') 
#                 for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
#                     dx = mu_x[k+1] - mu_x[k]
#                     dy = mu_y[k+1] - mu_y[k]
#                     ax.arrow(mu_x[k], mu_y[k], dx, dy,
#                             head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)

#         true_trial = z.mean(1)

#         move_logit = move_detector(true_trial.unsqueeze(0))
#         move_output = binary_output(move_logit)
#         move_rec = move_output.squeeze(0).astype(int)
#         RT_rec = 0

#         if move_rec:
#             RT_output = RT_detector(true_trial.unsqueeze(0))
#             RT_rec = prob_to_RT(RT_output, tau)
#             RT_rec = RT_rec.squeeze(0)

#         true_trial = true_trial.cpu().detach().numpy()
#         x_true_story = true_trial[:, axis[0]]
#         y_true_story = true_trial[:, axis[1]]
#         if n>1:
#             ax[i].plot(x_true_story, y_true_story, linewidth=2, c = color_true, label = f'True traj, true RT={((RT+56)*5):d}$ms$, Detected RT={RT_rec*5*tau}$ms$')  
#             n_arrows = 15
#             arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
#             for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
#                 dx = x_true_story[k+1] - x_true_story[k]
#                 dy = y_true_story[k+1] - y_true_story[k]
#                 ax[i].arrow(x_true_story[k], y_true_story[k], dx, dy,
#                         head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
#             #ax[i].set_title(f"Virtual experiment: changing no stops trajectories for different stops signals")#the smallest STOP")
#             ax[i].set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
#             ax[i].set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
#             ax[i].set_xlabel("z1", fontsize=font_ax)
#             ax[i].set_ylabel("z2", fontsize=font_ax)
#             ax[i].set_xticks(z_ticks[axis[0]]) 
#             ax[i].set_xticklabels(z_ticks[axis[0]], fontsize=font_tick)  # Show corresponding labels
#             ax[i].set_yticks(z_ticks[axis[1]])
#             ax[i].set_yticklabels(z_ticks[axis[1]], fontsize=font_tick)
#     #         ax[i].legend()
#         else:
#             ax.plot(x_true_story, y_true_story, linewidth=2, c = color_true, label = f'True traj, true RT={((RT+56)*5):d}$ms$, Detected RT={RT_rec*5*tau}$ms$')  
#             n_arrows = 15
#             arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
#             for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
#                 dx = x_true_story[k+1] - x_true_story[k]
#                 dy = y_true_story[k+1] - y_true_story[k]
#                 ax.arrow(x_true_story[k], y_true_story[k], dx, dy,
#                         head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
#             #ax[i].set_title(f"Virtual experiment: changing no stops trajectories for different stops signals")#the smallest STOP")
#             ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
#             ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
#             ax.set_xlabel("z1", fontsize=font_ax)
#             ax.set_ylabel("z2", fontsize=font_ax)
#             ax.set_xticks(z_ticks[axis[0]]) 
#             ax.set_xticklabels(z_ticks[axis[0]], fontsize=font_tick)  # Show corresponding labels
#             ax.set_yticks(z_ticks[axis[1]])
#             ax.set_yticklabels(z_ticks[axis[1]], fontsize=font_tick)        





def RTgen_vs_SSD(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    t = comm_dict["t"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    RT_detector = comm_dict["RT_detector"]
    move_detector = comm_dict["move_detector"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    x_ticks = diff_dict["x_ticks"]
    elinewidth = diff_dict["elinewidth"]
    capsize = diff_dict["capsize"]
    ms = diff_dict["ms"]
    compute = diff_dict["compute"]
    mean_z = diff_dict["mean_z"]
    min_ns = diff_dict["min_ns"]
    axis = diff_dict["axis"]
    color_point = diff_dict["color_point"]
    color_line = diff_dict["color_line"]
    y_ticks = diff_dict["y_ticks"]
    chunk_size = diff_dict["chunk_size"]
    size = diff_dict["size"]
    x_ticks_hist = diff_dict["x_ticks_hist"]
    bins = diff_dict["bins"]
    alpha = diff_dict["alpha"]
    xlims = diff_dict["xlims"]
    ylims = diff_dict["ylims"]
    SSD_list = diff_dict["SSD_list"]
    figsize = diff_dict["figsize"]
    simulate_go = diff_dict["simulate_go"]
    ylims_inset = diff_dict["ylims_inset"]
    y_ticks_inset = diff_dict["y_ticks_inset"]
    inset_font = diff_dict["inset_font"]
    inset_dim = diff_dict["inset_dim"]
    
#     ms_c = diff_dict["ms_c"]
#     size_point = diff_dict["size_point"]
    
#     SSD_cs = data["SSD_cs_ordSSD"]
#     SSD_ws = data["SSD_ws_ordSSD"]
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
#     set_cs = data["set_cs_ordSSD"]
    
    n, steps, features = set_cn.shape
    RT_cn_step = (RT_cn+56)//tau
    RT_time = RT_cn*5
    n_batch = n_trials // chunk_size if (n_trials%chunk_size)==0 else (n_trials // chunk_size) + 1
    text = "_sim" if simulate_go else ""  
    
    # Input per batch
    test_data = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).repeat_interleave(chunk_size, dim=1)
    cont_data = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2).repeat_interleave(chunk_size, dim=1)
    
    cont = torch.zeros((1, n*chunk_size, 4)).to(device)
    cont[:, :, 3] = 1

    if compute:
        move_perc_list = []
        with torch.inference_mode():
            RT_pred_list = []
            RT_time_list = []
            pmove_list = []
            # 🔁 Ora ciclo esterno sul tempo t
            for SSD in SSD_list:
                teacher = (SSD//5 + 56)//tau
                alone = steps - teacher - 1
                
                move_pred_t_list = []
#                 RT_rec_list = []
                mu_go_list = []
                mu_z_list = []
                # 🔁 Ciclo interno sui batch
                for batch in range(n_batch):

                    # Input per batch
                    test_set = test_data.clone()
                    cont_c = cont_data.clone()

                    # --- Inference
                    if mean_z:
                        _, z, _ = dmm.inference(test_set, cont_c)
                    else:
                        z, _, _ = dmm.inference(test_set, cont_c)

                    # Teacher e z iniziale
                    z_tmp = z[:teacher+1]
                    mu_z = z_tmp.clone()
                    mu_go = z_tmp.clone()
                    for step in range(alone):
                        z_mean_gen, z_cov_gen = dmm.generation_z(mu_go[-1].unsqueeze(0), cont_c[teacher+step].unsqueeze(0))
                        z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                        mu_go = torch.cat((mu_go, z_gen), dim=0)
                        
                    mu_go_list.append(mu_go.reshape(mu_go.shape[0], -1, chunk_size, z_dim))
                    
                    for step in range(alone):
                        z_mean_gen, z_cov_gen = dmm.generation_z(mu_z[-1].unsqueeze(0), cont)
                        z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                        mu_z = torch.cat((mu_z, z_gen), dim=0)
                    mu_z = mu_z.permute(1, 0, 2)
                        
                    # --- Move detector
                    move_logit = move_detector(mu_z)
                    move_output = binary_output(move_logit)
                    move_pred = move_output.reshape(-1, chunk_size)   # n x batch 
                    
#                     RT_output = RT_detector(mu_z)
#                     RT_rec = prob_to_RT(RT_output, tau)
#                     RT_rec = RT_rec.reshape(-1, chunk_size) # n x batch 
#                     RT_rec = RT_rec*tau*5
                
                    move_pred_t_list.append(move_pred)
                    mu_z_list.append(mu_z.reshape(-1, chunk_size, mu_z.shape[1], z_dim))
#                     RT_rec_list.append(RT_rec)

                # Concatena tutti i batch per questo tempo t
                move_pred_t = np.concatenate(move_pred_t_list, axis=1)  # n x n_trials
                mu_z_array = torch.cat(mu_z_list, dim=1)
#                 RT_pred_array = np.concatenate(RT_rec_list, axis=1)  # n x n_trials
                mu_go_array = torch.cat(mu_go_list, dim=2)  # n x n_trials
                
                mu_mean_go = mu_go_array.mean(2)
                RT_output = RT_detector(mu_mean_go.permute(1, 0, 2))
                RT_go = prob_to_RT(RT_output, tau)
                RT_go = RT_go*tau*5
                
                mask_ns = move_pred_t > 0.5
                hist = mask_ns.sum(axis=1)     # numero di sim che hanno portato a un ws, per trial
                valid_rows = hist >= min_ns    # trial con numero di simulazioni ws sufficienti
            
#                 RT_masked = RT_pred_array[valid_rows]
                mu_z_masked = mu_z_array[valid_rows]
                mask_valid = mask_ns[valid_rows]
                RT_cn_valid = RT_go[valid_rows] if simulate_go else RT_time[valid_rows] 
                
#                 # For each valid row, we need the indices of the first k True values.
#                 # argsort on booleans: False < True, so True values come last.
#                 # Using a stable sort + flipping gives True-first column order.
#                 order = np.argsort(~mask_valid, axis=1, kind="stable")  # (l, m)
#                 # Gather values in that order, then keep only the first k columns
#                 RT_pred_valid = RT_masked[np.arange(len(RT_masked))[:, None], order[:, :min_ns]]  # (l, k)
#                 RT_pred_mean = RT_pred_valid.mean(1)
                
#                 num = np.(mask_valid, RT_masked, 0).sum(axis=1)
#                 den = mask_valid.sum(axis=1)
#                 RT_pred_mean = num / den
                print(mu_z_masked.shape)
#                 print(mask_valid.shape)
                mask_valid = torch.from_numpy(mask_valid).float().to(device)
                mask_valid = mask_valid.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, mu_z_masked.shape[2], mu_z_masked.shape[3])
                num = (mu_z_masked * mask_valid).sum(dim=1)
                den = mask_valid.sum(dim=1)
                mu_z_mean = num / den
            
                RT_output = RT_detector(mu_z_mean)
                RT_rec = prob_to_RT(RT_output, tau)
#                 RT_rec = RT_rec.reshape(-1, chunk_size) # n x batch 
                RT_pred_mean = RT_rec*tau*5
    
                # aggiungere codice che filtra i Nan
                mask_notnan = ~np.isnan(RT_pred_mean)
                RT_cn_valid = RT_cn_valid[mask_notnan]
                RT_pred_mean = RT_pred_mean[mask_notnan]
                
                RT_pred_list.append(RT_pred_mean)
                RT_time_list.append(RT_cn_valid)
                pmove_list.append(mask_ns.mean(1))
                print(f"SSD={SSD} retain {mask_valid.shape[0]} trials out of {n}")
                
        RT_pred_dict = {i: arr for i, arr in enumerate(RT_pred_list)}
        RT_time_dict = {i: arr for i, arr in enumerate(RT_time_list)}
        pmove_dict = {i: arr for i, arr in enumerate(pmove_list)}

        np.savez(comm_dict["saved_path"] + f"/RTgen_vs_SSD{text}_{n_trials}.npz", RT_pred_dict=RT_pred_dict, RT_time_dict=RT_time_dict, pmove_dict=pmove_dict)
    else:
        with np.load(comm_dict["saved_path"] + f"/RTgen_vs_SSD{text}_{n_trials}.npz", allow_pickle=True) as loaded_file:
            RT_pred_dict = loaded_file["RT_pred_dict"].item()
            RT_time_dict = loaded_file["RT_time_dict"].item()
            pmove_dict = loaded_file["pmove_dict"].item()
            

    import math
    import matplotlib.gridspec as gridspec

    # ── Font sizes (override mplstyle for this multiplot) ──────────────────────────
    TITLE_FS   = 14   # suptitle
    LABEL_FS   = 10   # x/y axis labels
    TICK_FS    = 8    # tick labels
    LEGEND_FS  = 8    # legend text
    LINEWIDTH  = 1.2  # mean-line width

    fig, axes = plt.subplots(
        3, 3,
        figsize=(10, 8),          # width x height in inches
        constrained_layout=True,  # avoids overlapping labels
    )
    fig.suptitle("Distribution of $RT_{pred} - RT_{true}$", fontsize=TITLE_FS, fontweight="bold")

    axes_flat = axes.flatten()
    RT_diff_mean = []
    RT_diff_std = []
    for i in range(len(SSD_list)):
        ax = axes_flat[i]
        diff_RT = RT_pred_dict[i] - RT_time_dict[i]
        RT_diff_mean.append(diff_RT.mean())
        RT_diff_std.append(diff_RT.std()/math.sqrt(len(diff_RT)))

        # ── Histogram ──────────────────────────────────────────────────────────────
        ax.hist(
            diff_RT,
            bins=bins,
            alpha=alpha,
            density=True,
            color="skyblue",
            edgecolor="none",
        )

        # ── Mean line ──────────────────────────────────────────────────────────────
        ax.axvline(
            diff_RT.mean(),
            color="black",
            linestyle="--",
            linewidth=LINEWIDTH,
            label=f"Mean = {diff_RT.mean():.0f} ms",
        )

        # ── Axes limits & ticks ───────────────────────────────────────────────────
        ax.set_xlim(xlims)
        ax.set_xticks(x_ticks_hist)
        ax.set_xticklabels(x_ticks_hist, fontsize=TICK_FS)
        ax.set_yticks([])

        # ── Labels ────────────────────────────────────────────────────────────────
#         ax.set_xlabel(r"$RT_go - RT_stop$ (ms)", fontsize=LABEL_FS)
        ax.set_ylabel("Counts", fontsize=LABEL_FS)

        # ── Legend ────────────────────────────────────────────────────────────────
        ax.legend(fontsize=LEGEND_FS, frameon=False)

    # ── Hide unused panels (if SSD_list has fewer than 9 entries) ─────────────────
    for j in range(len(SSD_list), 9):
        axes_flat[j].set_visible(False)
        
    #--------------- same thing but with pMove---------------
    
    fig, axes = plt.subplots(
        2, 3,
        figsize=(10, 8),          # width x height in inches
        constrained_layout=True,  # avoids overlapping labels
    )
    fig.suptitle("Distribution of $pMove$", fontsize=TITLE_FS, fontweight="bold")

    axes_flat = axes.flatten()
#     RT_diff_mean = []
#     RT_diff_std = []
    for i in range(len(SSD_list)):
        ax = axes_flat[i]

        # ── Histogram ──────────────────────────────────────────────────────────────
        ax.hist(
            pmove_dict[i],
            bins=bins,
            alpha=alpha,
            density=True,
            color="skyblue",
            edgecolor="none",
        )

        # ── Mean line ──────────────────────────────────────────────────────────────
        ax.axvline(
            pmove_dict[i].mean(),
            color="black",
            linestyle="--",
            linewidth=LINEWIDTH,
            label=f"Mean = {pmove_dict[i].mean():.0f}",
        )

        # ── Axes limits & ticks ───────────────────────────────────────────────────
        ax.set_xlim([0, 1])
        ax.set_xticks([0.5])
        ax.set_xticklabels([0.5], fontsize=TICK_FS)
        ax.set_yticks([])

        # ── Labels ────────────────────────────────────────────────────────────────
#         ax.set_xlabel(r"$RT_go - RT_stop$ (ms)", fontsize=LABEL_FS)
        ax.set_ylabel("Counts", fontsize=LABEL_FS)

        # ── Legend ────────────────────────────────────────────────────────────────
        ax.legend(fontsize=LEGEND_FS, frameon=False)

    # ── Hide unused panels (if SSD_list has fewer than 9 entries) ─────────────────
    for j in range(len(SSD_list), 6):
        axes_flat[j].set_visible(False)
    
    
    # ------------------ diff RT vs SSD --------------------------
    
    fig, ax = plt.subplots(figsize=figsize)
    print(RT_diff_mean)
    ax.plot(SSD_list, RT_diff_mean, color="black")
    ax.errorbar(SSD_list, RT_diff_mean, yerr=RT_diff_std, fmt='o', color="black", ecolor='black', 
                             elinewidth=elinewidth, linestyle='none',capsize=capsize, ms=ms)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_ylim(ylims)
    ax.set_xlabel(r"$SSD\ (\mathrm{ms})$")
    ax.set_ylabel(r"$RT_{\mathrm{stop\ failure}} - RT_{\mathrm{no-stop}} \, (\mathrm{ms})$")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    
    with np.load("/raid/home/tubitoal/DMM/saved_model/2026-02-18-23h02_DKF_b4_3Cz3_w3" + f"/RTgen_vs_SSD{text}_{n_trials}.npz", allow_pickle=True) as loaded_file:
            RT_pred_dict = loaded_file["RT_pred_dict"].item()
            RT_time_dict = loaded_file["RT_time_dict"].item()
            pmove_dict = loaded_file["pmove_dict"].item()
            
    RT_diff_mean = []
    RT_diff_std = []
    for i in range(len(SSD_list)):
        diff_RT = RT_pred_dict[i] - RT_time_dict[i]
        RT_diff_mean.append(diff_RT.mean())
        RT_diff_std.append(diff_RT.std()/math.sqrt(len(diff_RT)))
    
    axins = ax.inset_axes(inset_dim)

    axins.plot(SSD_list, RT_diff_mean, color="black")
    axins.errorbar(SSD_list, RT_diff_mean, yerr=RT_diff_std, fmt='o', color="black", ecolor='black', 
                             elinewidth=elinewidth, linestyle='none',capsize=capsize//2, ms=ms//2)
    axins.axhline(0, color="red", linestyle="--")
    axins.set_ylim(ylims_inset)
        
    axins.set_xticks(x_ticks)
    axins.set_xticklabels(x_ticks)
    axins.set_yticks(y_ticks_inset)
    axins.set_yticklabels(y_ticks_inset)
    axins.tick_params(labelsize=inset_font)
    
    # ── Save ──────────────────────────────────────────────────────────────────────
    fig_file = os.path.join(comm_dict["saved_path"], f'RTgen_vs_SSD{text}.png')
    plt.savefig(fig_file)
    plt.show()
    
    
def early_SSRT(comm_dict, diff_dict):

    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    t = comm_dict["t"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    RT_detector = comm_dict["RT_detector"]
    move_detector = comm_dict["move_detector"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    simuldistr = diff_dict["simuldistr"]
    color_Piero = diff_dict["color_Piero"]
    y_ticks = diff_dict["y_ticks"]
    ylims = diff_dict["ylims"]
    figsize = diff_dict["figsize"]
    slice_start = diff_dict["slice_start"]
    slice_end = diff_dict["slice_end"]
    mean_z = diff_dict["mean_z"]
    color_humans = diff_dict["color_humans"]
    alpha = diff_dict["alpha"]
    color_Cornelio = diff_dict["color_Cornelio"]
    chunk_size = diff_dict["chunk_size"]
    compute = diff_dict["compute"]
    SSD_short_list = diff_dict["SSD_short_list"]
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    
    n, steps, features = set_cn.shape
    RT_cn_step = (RT_cn+56)//tau
    RT_time = RT_cn*5
    
    text = "_sim" if simuldistr else ""
    mean = "_mean" if mean_z else ""
    n_batch = n*n_trials // chunk_size if (n*n_trials%chunk_size)==0 else (n*n_trials // chunk_size) + 1
    
    # Input per batch
    test_data = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    cont_data = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    
    cont = torch.zeros((1, chunk_size, 4)).to(device)
    cont[:, :, 3] = 1
    
    if compute:
        
#         z_cn, _, _ = infer_latent(dmm, data, device, n_trials)
        RT_pred_list = []
        move_perc_list = []
        for i, SSD_short in enumerate(SSD_short_list):
            print(i)
            move_pred_list = []
            # 🔁 Ciclo interno sui batch
            for batch in range(n_batch):
                start = batch*chunk_size
                end = min(start + chunk_size, n*n_trials) 
                interval = end-start

                # Input per batch
                test_set = test_data[:, start:end]
                cont_c = cont_data[:, start:end]

                # --- Inference
                if mean_z:
                    _, z_cn, _ = dmm.inference(test_set, cont_c)
                else:
                    z_cn, _, _ = dmm.inference(test_set, cont_c)
                

                # Simulo tanti Go trials
                if simuldistr and i==0:
                    teacher = 56//tau
                    alone = steps-teacher
                    z_teach = z_cn[:teacher]
                    for step in range(alone):
                        z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont_c[56//tau+step].unsqueeze(0))
                        z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                        z_teach = torch.cat((z_teach, z_gen), dim=0)

                    RT_output = RT_detector(z_teach.permute(1, 0, 2))
                    RT_pred = prob_to_RT(RT_output, tau)
                    RT_pred_list.extend(RT_pred) 
                    
                    del z_teach, z_gen
                    torch.cuda.empty_cache()

                stop = (56+SSD_short)//tau
                alone_go = SSD_short//tau
                z_stop = z_cn[:56//tau]
                for step in range(alone_go):
                    z_mean_gen, z_cov_gen = dmm.generation_z(z_stop[-1].unsqueeze(0), cont_c[56//tau+step].unsqueeze(0))
                    z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                    z_stop = torch.cat((z_stop, z_gen), dim=0)

                alone_stop = steps - stop                          
                for step in range(alone_stop):
#                     print(z_stop.shape, cont.shape)
                    z_mean_gen, z_cov_gen = dmm.generation_z(z_stop[-1].unsqueeze(0), cont[:, :interval])
                    z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                    z_stop = torch.cat((z_stop, z_gen), dim=0) 

                move_logit = move_detector(z_stop.permute(1, 0, 2))
                move_output = binary_output(move_logit).astype(int)
                move_pred = move_output.astype(int)

                move_pred_list.extend(move_pred)
                
                del move_pred, z_stop
                torch.cuda.empty_cache()

            # Concatena tutti i batch per questo tempo t
            move_pred_tot = np.array(move_pred_list)  # n x n_trials
            move_perc_list.append(move_pred_tot.mean())  # media su n_trial (asse 1)
            
        move_perc_arr = np.array(move_perc_list)
        RT_pred_arr = np.array(RT_pred_list)
     
        np.savez(comm_dict["saved_path"] + f"/SSRT_short_{n_trials}{text}{mean}.npz", move_perc_arr=move_perc_arr, RT_pred_arr=RT_pred_arr)
    else:
        with np.load(comm_dict["saved_path"] + f"/SSRT_short_{n_trials}{text}{mean}.npz", allow_pickle=True) as loaded_file:
            move_perc_arr = loaded_file["move_perc_arr"]
            RT_pred_arr = loaded_file["RT_pred_arr"]

    perc_mov = np.array([0.1, 0.3, 0.6, 0.8])
    
    RT_val = np.quantile(RT_pred_arr, move_perc_arr)
    SSRT_val = (RT_val*tau - SSD_short_list)*5
    
    SSRT_val = SSRT_val[slice_start:slice_end]
    move_perc_arr = move_perc_arr[slice_start:slice_end]
    SSD_short_list_sliced = SSD_short_list[slice_start:slice_end]

    idx = np.abs(move_perc_arr[:, None] - perc_mov[None, :]).argmin(axis=0)
    SSRT_Piero = SSRT_val[idx]
    for i in idx:
        print(SSD_short_list_sliced[i])
    print()
    
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.plot(np.array(SSD_short_list_sliced)*5, SSRT_val, color="black")
#     ax.errorbar(np.array(SSD_short_list_sliced)*5, SSRT_val, fmt='o', color="black", ecolor='black') 
# #                                  elinewidth=elinewidth, linestyle='none',capsize=capsize, ms=ms)
#     ax.axhline(SSRT_val.mean(), color="red", linestyle="--")
#     ax.set_ylim(ylims)
#     ax.set_xlabel(r"$SSD\ (\mathrm{ms})$")
#     ax.set_ylabel(r"$SSRT\ (\mathrm{ms})$")
#     ax.set_xticks(x_ticks)
#     ax.set_xticklabels(x_ticks)
#     ax.set_yticks(y_ticks)
#     ax.set_yticklabels(y_ticks)

    with np.load("/raid/home/tubitoal/DMM/saved_model/2026-02-18-23h02_DKF_b4_3Cz3_w3" + f"/SSRT_short_{n_trials}{text}{mean}.npz", allow_pickle=True) as loaded_file:
        move_perc_arr = loaded_file["move_perc_arr"]
        RT_pred_arr = loaded_file["RT_pred_arr"]

    RT_val = np.quantile(RT_pred_arr, move_perc_arr)
    SSRT_val = (RT_val*tau - SSD_short_list)*5
    
    move_perc_arr = move_perc_arr[slice_start:slice_end]
    SSRT_val = SSRT_val[slice_start:slice_end]
    
    idx = np.abs(move_perc_arr[:, None] - perc_mov[None, :]).argmin(axis=0)
    SSRT_Cornelio = SSRT_val[idx]
    for i in idx:
        print(SSD_short_list_sliced[i])
    
#     axins = ax.inset_axes(inset_dim)

#     axins.plot(np.array(SSD_short_list_sliced)*5, SSRT_val, color="black")
#     axins.errorbar(np.array(SSD_short_list_sliced)*5, SSRT_val, fmt='o', color="black", ecolor='black')
# #                                  elinewidth=elinewidth, linestyle='none',capsize=capsize//2, ms=ms//2)
#     axins.axhline(SSRT_val.mean(), color="red", linestyle="--")
#     axins.set_ylim(ylims)

#     axins.set_xticks(x_ticks)
#     axins.set_xticklabels(x_ticks)
#     axins.set_yticks(y_ticks)
#     axins.set_yticklabels(y_ticks)
#     axins.tick_params(labelsize=inset_font)

#     # ── Save ──────────────────────────────────────────────────────────────────────
#     fig_file = os.path.join(comm_dict["saved_path"], f'SSRT_vs_SSD{text}{mean}.png')
#     plt.savefig(fig_file)
#     plt.show()                      
    
    SSRTs = np.array([[307, 182, 165, 154], [275, 192, 155, 148], [200, 190, 176, 171], [286, 243, 203, 197], [179, 151, 128, 123], [270, 188, 171, 179]])
#     SSDs = np.array([[0, 140, 180, 220], [0, 80, 160, 240], [100, 120, 150, 200], [30, 80, 150, 170], [90, 130, 180, 240], [40, 130, 190, 240]])
#     labels = ["CS", "DS", "EH", "IM", "IW", "SW"]
    xlabel = ["SSD 1", "SSD 2", "SSD 3", "SSD 4"]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xlabel, SSRT_Piero, color=color_Piero, zorder=2)  # Piero
    ax.plot(xlabel, SSRT_Cornelio, color=color_Cornelio, zorder=2)

    for i in range(6):
        ax.plot(xlabel, SSRTs[i], color=color_humans, alpha=alpha, zorder=1)
        
    ax.scatter(xlabel, SSRT_Piero, marker="d", color='black', zorder=3)
    ax.scatter(xlabel, SSRT_Cornelio, marker="d", color='black', zorder=3)

#     ax.xlabel("SSD")# corresponding to the SSD chosen")
    ax.set_ylabel("SSRT")
    ax.set_ylim(ylims)
    ax.set_yticks(y_ticks)
#     ax.legend(fontsize="xx-small")
    
    # ── Save ──────────────────────────────────────────────────────────────────────
    fig_file = os.path.join(comm_dict["saved_path"], f'SSRT_vs_SSD{text}{mean}.png')
    plt.savefig(fig_file)
    plt.show()      
                                            

def SSRT_plot(comm_dict, diff_dict): 
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    t = comm_dict["t"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    RT_detector = comm_dict["RT_detector"]
    move_detector = comm_dict["move_detector"]
    
    data = diff_dict["data"]
    plot_data = diff_dict["plot_data"]
    n_trials = diff_dict["n_trials"]
    cmap = diff_dict["cmap"]
    elinewidth = diff_dict["elinewidth"]
    capsize = diff_dict["capsize"]
    ms = diff_dict["ms"]
    SSD_interval = diff_dict["SSD_interval"]
    RT_groups = diff_dict["RT_groups"]
    mean_z_flag = diff_dict["mean_z_flag"]
    move_frac = diff_dict["move_frac"]
    compute = diff_dict["compute"]
    logit_move = diff_dict["logit_move"]
    cut_tail = diff_dict["cut_tail"]
    frac_tail = diff_dict["frac_tail"]
    axis = diff_dict["axis"]
    no_zero = diff_dict["no_zero"]
    color_point = diff_dict["color_point"]
    color_line = diff_dict["color_line"]
    color_vline = diff_dict["color_vline"]
    chunk_size = diff_dict["chunk_size"]
    size = diff_dict["size"]
    x_ticks_hist = diff_dict["x_ticks_hist"]
    bins = diff_dict["bins"]
    alpha = diff_dict["alpha"]
    xlims = diff_dict["xlims"]
    opacity = diff_dict["opacity"]
    opacity_stop = diff_dict["opacity_stop"]
    opacity_ssrt = diff_dict["opacity_ssrt"]
    inset_dim = diff_dict["inset_dim"]
    inset_font = diff_dict["inset_font"]
    show_comp = diff_dict["show_comp"]
    figsize = diff_dict["figsize"]
#     ms_c = diff_dict["ms_c"]
#     size_point = diff_dict["size_point"]
    
    SSD_cs = data["SSD_cs_ordSSD"]
    SSD_ws = data["SSD_ws_ordSSD"]
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    set_cs = data["set_cs_ordSSD"]
    
    n, steps, features = set_cn.shape
    RT_cn_step = (RT_cn+56)//tau
    RT_time = RT_cn*5
    
    # Preallocazione risultati
    t_start, t_end = SSD_interval
    t_interval = t_end - t_start
    n_batch = n_trials // chunk_size if (n_trials%chunk_size)==0 else (n_trials // chunk_size) + 1
    
    # Input per batch
    test_data = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).repeat_interleave(chunk_size, dim=1)
    cont_data = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2).repeat_interleave(chunk_size, dim=1)
    
    cont = torch.zeros((1, n*chunk_size, 4)).to(device)
    cont[:, :, 3] = 1
    
    if compute:

        move_perc_list = []
        with torch.inference_mode():
            # 🔁 Ora ciclo esterno sul tempo t
            for idx_t, t in enumerate(range(t_start, t_end)):
                if idx_t % 5 ==0:
                    print(f"time {idx_t}/{t_interval}")
                move_pred_t_list = []
                teacher = np.repeat(RT_cn_step, chunk_size) - t
                alone = steps - teacher[0]

                # 🔁 Ciclo interno sui batch
                for batch in range(n_batch):

                    # Input per batch
                    test_set = test_data.clone()
                    cont_c = cont_data.clone()

                    # --- Inference
#                     z, _, _ = dmm.inference(test_set, cont_c)
                    _, z, _ = dmm.inference(test_set, cont_c)
#                     z = z_mu_arr[batch]

                    # Teacher e z iniziale
                    z_gen = z[teacher, torch.arange(len(teacher), device=z.device)].unsqueeze(0)
                    z_tmp = z_gen.clone()

                    # --- Generazione z_teach
                    z_teach_list = []
                    for _ in range(alone):
                        z_mean_gen, z_cov_gen = dmm.generation_z(z_tmp, cont)
                        z_tmp = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                        z_teach_list.append(z_tmp)
                    z_teach = torch.cat(z_teach_list, dim=0)

                    # --- Ricostruzione mu_z
                    mu_z_list = []
                    for i in range(z_teach.shape[1]):
                        diff = teacher[i] - teacher[0] + 1
                        z_final_i = torch.cat((z[:(teacher[i] + 1), i], z_teach[:-diff, i]), dim=0)
                        mu_z_list.append(z_final_i)
                    mu_z = torch.stack(mu_z_list, dim=1)  # [T, n*chunk, latent_dim]

                    # --- Move detector
                    move_logit = move_detector(mu_z.permute(1, 0, 2))
                    if logit_move:
                        move_output = torch.sigmoid(move_logit).cpu().detach().numpy()
                    else:
                        move_output = binary_output(move_logit)

                    move_pred = move_output.reshape(-1, chunk_size)   # n x batch 
                    move_pred_t_list.append(move_pred)

                # Concatena tutti i batch per questo tempo t
                move_pred_t = np.concatenate(move_pred_t_list, axis=1)  # n x n_trials
                move_perc_list.append(move_pred_t.mean(1))  # media su n_trial (asse 1)

            # Stack finale su tutti i tempi
            move_perc = np.stack(move_perc_list, axis=1)  # (n, t_end - t_start)

            print(move_perc.shape[0])

            # Pulizia NaN
            nan_mask = np.isnan(move_perc).any(axis=1)
            nan_indices = np.where(nan_mask)[0]
            move_perc = move_perc[~nan_mask]
            RT_time = RT_time[~nan_mask]
            RT_cn_step = RT_cn_step[~nan_mask]
            set_cn = set_cn[~nan_mask]
            cont_cn = cont_cn[~nan_mask]
            ##################################
#             RT_pred_step = RT_pred_step[~nan_mask]

#             RT_ind = np.argsort(RT_pred_step)
#             RT_pred_step = RT_pred_step[RT_ind]
#             move_perc = move_perc[RT_ind]
            ##################################
            mask = move_perc <= 0.5
            SSRT_critic = mask.argmax(axis=1)
            critic_t = RT_cn_step - SSRT_critic   # modified
#             critic_t = RT_pred_step - SSRT_critic

            test_cn = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2)#.repeat_interleave(n_trials, dim=1)
            test_c = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2)#.repeat_interleave(n_trials, dim=1)

            z_test, _, _ = dmm.inference(test_cn, test_c)
            z_critic = z_test[critic_t, torch.arange(set_cn.shape[0])]

            z_critic = z_critic.cpu().detach().numpy()
            print(z_critic.shape)

            cont = torch.zeros((1, n_trials, 4)).to(device)
            cont[:, :, 3] = 1
            print()
            print(len(SSRT_critic), move_perc.shape[0])
            RT_stop = np.zeros(len(SSRT_critic))
            RT_go = np.zeros(len(SSRT_critic))
            z_mean_stop = np.zeros((len(SSRT_critic), steps, z_dim))
            z_mean_go = np.zeros((len(SSRT_critic), steps, z_dim))
            frac_realizations = np.zeros(len(SSRT_critic))
            for i in range(len(SSRT_critic)):
                if i % 30==0:
                    print(f"{i}/{len(SSRT_critic)}")

                teacher = RT_cn_step[i] - SSRT_critic[i] + 1

                # Input per batch
                test_set = torch.from_numpy(set_cn[i]).float().to(device).unsqueeze(1).repeat_interleave(n_trials, dim=1)####
                cont_c = torch.from_numpy(cont_cn[i]).float().to(device).unsqueeze(1).repeat_interleave(n_trials, dim=1)

#                 # --- Inference
                z, _, _ = dmm.inference(test_set, cont_c)###
#                 z = z_mu_go[:, i]
                z_gen = z[:teacher]
                cont_go = cont_c[teacher].unsqueeze(0)

                z_tmp_stop = z_gen.clone()
                z_tmp_go = z_gen.clone()

                # --- Generazione z_teach
                alone = steps - teacher
                #print(z_tmp_stop.shape, z_tmp_stop.device)
                for _ in range(alone):
                    z_mean_gen, z_cov_gen = dmm.generation_z(z_tmp_stop[-1].unsqueeze(0), cont)
                    z_rep_stop = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                    z_tmp_stop = torch.cat((z_tmp_stop, z_rep_stop), dim=0)

                    z_mean_gen, z_cov_gen = dmm.generation_z(z_tmp_go[-1].unsqueeze(0), cont_go)
                    z_rep_go = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                    z_tmp_go = torch.cat((z_tmp_go, z_rep_go), dim=0)

                z_tmp_stop = z_tmp_stop.permute(1, 0, 2)
                z_tmp_go = z_tmp_go.permute(1, 0, 2)

                # --- Move detector
                move_logit = move_detector(z_tmp_stop)
                if logit_move:
                    move_output = torch.sigmoid(move_logit).cpu().detach().numpy()
                else:
                    move_output = binary_output(move_logit)

                mask = move_output > move_frac
                if not mask.any():
                    continue
                else:
                    z_tmp_stop_masked=z_tmp_stop[mask]
                    z_tmp_go_masked=z_tmp_go[mask]
                    
                    RT_prob_stop = RT_detector(z_tmp_stop_masked)
                    RT_estimate_stop = prob_to_RT(RT_prob_stop, tau)  
                    RT_gen_stop = (RT_estimate_stop*tau)*5

                    RT_prob_go = RT_detector(z_tmp_go_masked)
                    RT_estimate_go = prob_to_RT(RT_prob_go, tau)  
                    RT_gen_go = (RT_estimate_go*tau)*5
                    
                    frac_realizations[i] = mask.sum()/n_trials 
                    RT_stop[i] = RT_gen_stop.mean()
                    RT_go[i] = RT_gen_go.mean()
                    z_mean_stop[i] = z_tmp_stop_masked.cpu().detach().numpy().mean(0)
                    z_mean_go[i] = z_tmp_go_masked.cpu().detach().numpy().mean(0)
        np.savez(comm_dict["saved_path"] + f"/SSRT_plot_{n_trials}.npz", move_perc=move_perc, RT_stop=RT_stop, RT_go=RT_go, RT_time=RT_time,
                SSRT_critic=SSRT_critic, z_mean_stop=z_mean_stop, z_mean_go=z_mean_go, z_critic=z_critic, frac_realizations=frac_realizations)#, #RT_pred_step=RT_pred_step)
    else:
        with np.load(comm_dict["saved_path"] + f"/SSRT_plot_{n_trials}.npz", allow_pickle=True) as loaded_file:
            move_perc = loaded_file["move_perc"]
#             RT_pred_step = loaded_file["RT_pred_step"]
            RT_stop = loaded_file["RT_stop"]
            RT_go = loaded_file["RT_go"]
            RT_time = loaded_file["RT_time"]#
            SSRT_critic = loaded_file["SSRT_critic"]
            z_mean_stop = loaded_file["z_mean_stop"]
            z_mean_go = loaded_file["z_mean_go"]
            z_critic = loaded_file["z_critic"]
            frac_realizations = loaded_file["frac_realizations"]
    
    
    RT_time_masked = RT_time
    if cut_tail:
        cut = int(len(RT_time) * frac_tail)
        move_perc = move_perc[cut:-cut]
        SSRT_critic = SSRT_critic[cut:-cut]
        RT_stop = RT_stop[cut:-cut]
        RT_go = RT_go[cut:-cut]
        RT_time_masked = RT_time_masked[cut:-cut]
        z_mean_stop = z_mean_stop[cut:-cut]
        z_mean_go = z_mean_go[cut:-cut]
        z_critic = z_critic[cut:-cut]
        frac_realizations = frac_realizations[cut:-cut]
    
    mask = SSRT_critic != 0
    print(len(mask), mask.sum())
    
    if no_zero:
        move_perc = move_perc[mask]
        RT_time_masked = RT_time_masked[mask]
        SSRT_critic = SSRT_critic[mask]
        RT_stop = RT_stop[mask]
        RT_go = RT_go[mask]
        z_mean_stop = z_mean_stop[mask]
        z_mean_go = z_mean_go[mask]
        z_critic = z_critic[mask]
        frac_realizations = frac_realizations[mask]
    
    n = move_perc.shape[0]

    mean_move_perc = move_perc.mean(0)
    std_move_perc = move_perc.std(0)
    mean_SSRT_all = np.argmax(mean_move_perc <= 0.5)*tau*5
#     left_lim = np.argmax(mean_move_perc <= 0.85)*tau*5
#     right_lim = np.argmax(mean_move_perc <= 0.15)*tau*5

    SSRT_std_list = []
    SSRT_mean_list = []
    mean_SSRT_list = []
    RT_list = []
    n_group = len(RT_time_masked) // RT_groups if (len(RT_time_masked)%RT_groups)==0 else (len(RT_time_masked) // RT_groups) + 1
    SSRT_trials = np.argmax(move_perc <= 0.5, axis=1)*tau*5
    print(f"L'SSRT medio è: {SSRT_trials.mean()}")
    for k in range(RT_groups):
        start = k*n_group
        end = min(start + n_group, len(RT_time_masked) + 1)
        # calcolo SSRT per ogni trial, e poi ne prendo la media per ogni gruppo di trials
        SSRT_group = SSRT_trials[start:end]
#         SSRT_mean = SSRT_group.mean()
        SSRT_mean = np.nanmean(SSRT_group)
        SSRT_std = SSRT_group.std()/math.sqrt(end-start)
        # calcolo l'RT medio di ogni gruppo di trials
        RT_group = RT_time_masked[start:end]
        mean_RT_group = RT_group.mean()
        # calcolo la media del gruppo di trials, e ne calcolo l'SSRT
        move_group = move_perc[start:end]
        mean_move_group = move_group.mean(0)
        mean_SSRT = np.argmax(mean_move_group <= 0.5)*tau*5
        # riempo le liste
        SSRT_std_list.append(SSRT_std)
        SSRT_mean_list.append(SSRT_mean)
        mean_SSRT_list.append(mean_SSRT)
        RT_list.append(mean_RT_group)
    # converto le liste in array
    SSRT_std_array = np.array(SSRT_std_list)
    SSRT_mean_array = np.array(SSRT_mean_list)
    mean_SSRT_array = np.array(mean_SSRT_list)
    RT_array = np.array(RT_list)

    from sklearn.linear_model import LinearRegression
    
    # --- STIMA DIREZIONE ---
    reg = LinearRegression().fit(RT_time_masked[:, None], SSRT_trials)
    reg_mean = LinearRegression().fit(RT_array[:, None], SSRT_mean_array)
    mean_reg = LinearRegression().fit(RT_array[:, None], mean_SSRT_array)
    pred_SSRT = reg.predict(RT_time_masked[:, None])
    pred_SSRT_mean = reg_mean.predict(RT_array[:, None])
    pred_mean_SSRT = mean_reg.predict(RT_array[:, None])
    
    f, ax = plt.subplots()
    
    x_positions = np.arange(move_perc.shape[1])
    x_labels = x_positions * (tau*5)
    y_labels = np.arange(n)

    #ax1.plot(t*5, MUA, c ='r', label = 'mean MUA')
    im = ax.imshow(move_perc, cmap = cmap, aspect='auto', vmin=0, vmax=1)
    ax.scatter(SSRT_critic, np.arange(move_perc.shape[0]), color='red', s=size)#, label='first < 0.5')
    ax.set_xticks([0, 20, 40]) 
    ax.set_xticklabels([0, 200, 400])#, fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks([(n//300)*100, (n//300)*200])
    ax.set_yticklabels([(n//300)*100, (n//300)*200])#, fontsize=font_tick)
    ax.set_xlabel('Time before RT ($ms$)')#, fontsize=font_ax)
    ax.set_ylabel('Trials ordered by RT')#, fontsize=font_ax)
#     ax.set_title('Percentage of movement onset as a function of RT and stop time going backward from the RT')
    cbar=plt.colorbar(im, ax=ax)
    cbar.set_ticks([0, 0.5, 1])
#     cbar.ax.tick_params(labelsize=font_tick)
    
    
    f, ax = plt.subplots()
    #ax.plot(RT_array, SSRT_mean_array, color = "r", label = "mean SSRT of each group")
    ax.scatter(RT_time_masked, SSRT_trials, color = "r", label = "mean SSRT of each group")
    ax.plot(RT_time_masked, pred_SSRT, color = "r", 
               label = f"linear fit: m={reg.coef_.item():.2f}, q={reg.intercept_.item():.2f}")
#     ax.errorbar(np.arange(RT_groups), mean_move_perc, yerr=std_move_perc, fmt='o', ls='--', ecolor='black', 
#                              elinewidth=1, capsize=5, capthick=1)
#     ax.axvline(mean_SSRT, color=color[k],linestyle="--",label= f"mean SSRT = {mean_SSRT}")
#         ax.axvline(left_lim, color="b",linestyle="--",label= f"left_lim = {left_lim}")
#         ax.axvline(right_lim, color="b",linestyle="--",label= f"right_lim = {right_lim}")
    ax.set_xlabel('RT ($ms$)')#, fontsize=font_ax)
    ax.set_ylabel('SSRT')#, fontsize=font_ax)
    ax.set_title('SSRT vs RT')
#     ax[0].set_xlim(0, RT_array.max())
#     ax[0].set_ylim(0, SSRT_mean_array.max())
    ax.legend()
    
    
    
    f, ax = plt.subplots(figsize=figsize)
    #ax.plot(RT_array, SSRT_mean_array, color = "r", label = "mean SSRT of each group")
    ax.errorbar(RT_array, SSRT_mean_array, yerr=SSRT_std_array, fmt='o', color=color_point, ecolor='black', 
                             elinewidth=elinewidth, linestyle='none',capsize=capsize, ms=ms)
#     ax[0].scatter(RT_array, SSRT_mean_array, color = color_point, label = "mean SSRT of each group")
    ax.plot(RT_array, pred_SSRT_mean, "--", color = color_line, 
               label = f"linear fit: m={reg_mean.coef_.item():.2f}, q={reg_mean.intercept_.item():.2f}")
    #ax.plot(RT_array, mean_SSRT_array, color = "b", label = "SSRT of the mean of each group")
#     ax[1].scatter(RT_array, mean_SSRT_array, color = color_point, label = "SSRT of the mean of each group")
#     ax[1].plot(RT_array, pred_mean_SSRT, "--", color = color_line,
#                label = f"linear fit: m={mean_reg.coef_.item():.2f}, q={mean_reg.intercept_.item():.2f}")

    # Calcola i limiti comuni
    min_val_x = ax.get_xlim()[0]
    min_val_y = ax.get_ylim()[0]
    max_val_x = ax.get_xlim()[1]
    max_val_y = ax.get_ylim()[1]
    
    dx = (max_val_x - min_val_x)//3
    dy = (max_val_y - min_val_y)//3

    ax.set_xlabel('mean RT ($ms$)')#, fontsize=font_ax)
    ax.set_ylabel('SSRT')#, fontsize=font_ax)
    ax.set_xticks([int(((min_val_x + dx)//100 + 1)*100), int(((min_val_x + 2*dx)//100 + 1)*100)])
    ax.set_xticklabels([int(((min_val_x + dx)//100 + 1)*100), int(((min_val_x + 2*dx)//100 + 1)*100)])#, fontsize=font_tick)
    ax.set_yticks([int(((min_val_y + dy)//10)*10), int(((min_val_y + 2*dy)//10 + 1)*10)])
    ax.set_yticklabels([int(((min_val_y + dy)//10)*10), int(((min_val_y + 2*dy)//10 + 1)*10)])#, fontsize=font_tick)
    
    # 2026-02-18-23h02_DKF_b4_3Cz3_w3
    # 2026-02-16-07h47_DKF_b12_3Cz3_w3
    
    if show_comp:
        with np.load("/raid/home/tubitoal/DMM/saved_model/2026-02-18-23h02_DKF_b4_3Cz3_w3" + f"/SSRT_plot_{n_trials}.npz", allow_pickle=True) as loaded_file:
                move_perc_c = loaded_file["move_perc"]
    #             RT_pred_step = loaded_file["RT_pred_step"]
                RT_stop_c = loaded_file["RT_stop"]
                RT_go_c = loaded_file["RT_go"]
                RT_time_c = loaded_file["RT_time"]#
                SSRT_critic_c = loaded_file["SSRT_critic"]

        RT_time_masked_c = RT_time_c
        if cut_tail:
            cut_c = int(len(RT_time_c) * frac_tail)
            move_perc_c = move_perc_c[cut_c:-cut_c]
            SSRT_critic_c = SSRT_critic_c[cut_c:-cut_c]
            RT_stop_c = RT_stop_c[cut_c:-cut_c]
            RT_go_c = RT_go_c[cut_c:-cut_c]
            RT_time_masked_c = RT_time_masked_c[cut_c:-cut_c]

        mask_c = SSRT_critic_c != 0
        print(len(mask_c), mask_c.sum())

        if no_zero:
            move_perc_c = move_perc_c[mask_c]
            RT_time_masked_c = RT_time_masked_c[mask_c]
            SSRT_critic_c = SSRT_critic_c[mask_c]
            RT_stop_c = RT_stop_c[mask_c]
            RT_go_c = RT_go_c[mask_c]

        n_c = move_perc_c.shape[0]

        mean_move_perc_c = move_perc_c.mean(0)
        std_move_perc_c = move_perc_c.std(0)
        mean_SSRT_all_c = np.argmax(mean_move_perc_c <= 0.5)*tau*5

        SSRT_std_list_c = []
        SSRT_mean_list_c = []
        mean_SSRT_list_c = []
        RT_list_c = []
        n_group_c = len(RT_time_masked_c) // RT_groups if (len(RT_time_masked_c)%RT_groups)==0 else (len(RT_time_masked_c) // RT_groups) + 1
        SSRT_trials_c = np.argmax(move_perc_c <= 0.5, axis=1)*tau*5
        for k in range(RT_groups):
            start = k*n_group_c
            end = min(start + n_group, len(RT_time_masked_c) + 1)
            # calcolo SSRT per ogni trial, e poi ne prendo la media per ogni gruppo di trials
            SSRT_group_c = SSRT_trials_c[start:end]
    #         SSRT_mean = SSRT_group.mean()
            SSRT_mean_c = np.nanmean(SSRT_group_c)
            SSRT_std_c = SSRT_group_c.std()/math.sqrt(end-start)
            # calcolo l'RT medio di ogni gruppo di trials
            RT_group_c = RT_time_masked_c[start:end]
            mean_RT_group_c = RT_group_c.mean()
            # calcolo la media del gruppo di trials, e ne calcolo l'SSRT
            move_group_c = move_perc_c[start:end]
            mean_move_group_c = move_group_c.mean(0)
            mean_SSRT_c = np.argmax(mean_move_group_c <= 0.5)*tau*5
            # riempo le liste
            SSRT_std_list_c.append(SSRT_std_c)
            SSRT_mean_list_c.append(SSRT_mean_c)
            mean_SSRT_list_c.append(mean_SSRT_c)
            RT_list_c.append(mean_RT_group_c)
        # converto le liste in array
        SSRT_std_array_c = np.array(SSRT_std_list_c)
        SSRT_mean_array_c = np.array(SSRT_mean_list_c)
        mean_SSRT_array_c = np.array(mean_SSRT_list_c)
        RT_array_c = np.array(RT_list_c)

        from sklearn.linear_model import LinearRegression

        # --- STIMA DIREZIONE ---
        reg_c = LinearRegression().fit(RT_time_masked_c[:, None], SSRT_trials_c)
        reg_mean_c = LinearRegression().fit(RT_array_c[:, None], SSRT_mean_array_c)
        mean_reg_c = LinearRegression().fit(RT_array_c[:, None], mean_SSRT_array_c)
        pred_SSRT_c = reg.predict(RT_time_masked_c[:, None])
        pred_SSRT_mean_c = reg_mean_c.predict(RT_array_c[:, None])
        pred_mean_SSRT_c = mean_reg_c.predict(RT_array_c[:, None])
    
    
        axins = ax.inset_axes(inset_dim)

        axins.errorbar(RT_array_c, SSRT_mean_array_c, yerr=SSRT_std_array_c, fmt='o', color=color_point, ecolor='black', 
                                 elinewidth=elinewidth/2, linestyle='none',capsize=capsize/2, ms=ms/2)
    #     ax[0].scatter(RT_array, SSRT_mean_array, color = color_point, label = "mean SSRT of each group")
        axins.plot(RT_array_c, pred_SSRT_mean_c, "--", color = color_line)#, label = f"linear fit: m={reg_mean.coef_.item():.2f}, q={reg_mean_c.intercept_.item():.2f}")

        # Calcola i limiti comuni
        min_val_x_c = axins.get_xlim()[0]
        min_val_y_c = axins.get_ylim()[0]
        max_val_x_c = axins.get_xlim()[1]
        max_val_y_c = axins.get_ylim()[1]

        dx_c = (max_val_x_c - min_val_x_c)//3
        dy_c = (max_val_y_c - min_val_y_c)//3

    #     axins.set_xlabel('mean RT ($ms$)', fontsize=inset_font)
    #     axins.set_ylabel('SSRT', fontsize=inset_font)
        axins.set_xticks([500, 700])
        axins.set_xticklabels([500, 700])#, fontsize=font_tick)
        axins.set_yticks([int(((min_val_y_c + dy_c)//10)*10), int(((min_val_y_c + 2*dy_c)//10 + 1)*10)])
        axins.set_yticklabels([int(((min_val_y_c + dy_c)//10)*10), int(((min_val_y_c + 2*dy_c)//10 + 1)*10)])#, fontsize=font_tick)
        axins.tick_params(labelsize=inset_font)
    
    fig_file = os.path.join(comm_dict["saved_path"], 'SSRT_vs_RT.png')
    plt.savefig(fig_file)
#     ax[0].set_title('SSRT of the groups mean vs RT')
#     ax[0].set_xlim(0, RT_array.max())
#     ax[0].set_ylim(0, SSRT_mean_array.max())
#     ax[0].legend()

    # Calcola i limiti comuni
#     min_val_x = ax[1].get_xlim()[0]
#     min_val_y = ax[1].get_ylim()[0]
#     max_val_x = ax[1].get_xlim()[1]
#     max_val_y = ax[1].get_ylim()[1]

#     dx = (max_val_x - min_val_x)//3
#     dy = (max_val_y - min_val_y)//3

#     ax[1].set_xlabel('mean RT of the groups ($ms$)', fontsize=font_ax)
#     ax[1].set_ylabel('SSRT', fontsize=font_ax)
#     ax[1].set_xticks([800, 1000])
#     ax[1].set_xticklabels([800, 1000], fontsize=font_tick)
#     ax[1].set_yticks([((min_val_y + dy)//10)*10, ((min_val_y + 2*dy)//10)*10])
#     ax[1].set_yticklabels([((min_val_y + dy)//10)*10, ((min_val_y + 2*dy)//10)*10], fontsize=font_tick)

    corr = np.corrcoef(RT_array, mean_SSRT_array)[0, 1]
    print(f"the correlation between RT and SSRT is {corr:.2f}")

    f, ax = plt.subplots()
    ax.scatter(RT_array, mean_SSRT_array, color = color_point)#, label = "mean SSRT of each group")
#     ax.errorbar(RT_array, mean_SSRT_array, yerr=SSRT_std_array, fmt='o', color=color_point, ecolor='black', 
#                              elinewidth=elinewidth, linestyle='none',capsize=capsize, ms=ms)
#     ax[0].scatter(RT_array, SSRT_mean_array, color = color_point, label = "mean SSRT of each group")
    ax.plot(RT_array, pred_mean_SSRT, "--", color = color_line, 
               label = f"linear fit: m={reg_mean.coef_.item():.2f}, q={reg_mean.intercept_.item():.2f}")
    #ax.plot(RT_array, mean_SSRT_array, color = "b", label = "SSRT of the mean of each group")
#     ax[1].scatter(RT_array, mean_SSRT_array, color = color_point, label = "SSRT of the mean of each group")
#     ax[1].plot(RT_array, pred_mean_SSRT, "--", color = color_line,
#                label = f"linear fit: m={mean_reg.coef_.item():.2f}, q={mean_reg.intercept_.item():.2f}")

    # Calcola i limiti comuni
    min_val_x = ax.get_xlim()[0]
    min_val_y = ax.get_ylim()[0]
    max_val_x = ax.get_xlim()[1]
    max_val_y = ax.get_ylim()[1]
    
    dx = (max_val_x - min_val_x)//3
    dy = (max_val_y - min_val_y)//3

    ax.set_xlabel('Mean RT ($ms$)')#, fontsize=font_ax)
    ax.set_ylabel('SSRT')#, fontsize=font_ax)
    ax.set_xticks([int(((min_val_x + dx)//100)*100), int(((min_val_x + 2*dx)//100 + 1)*100)])
    ax.set_xticklabels([int(((min_val_x + dx)//100)*100), int(((min_val_x + 2*dx)//100 + 1)*100)])#, fontsize=font_tick)
    ax.set_yticks([int(((min_val_y + dy)//10)*10), int(((min_val_y + 2*dy)//10 + 1)*10)])
    ax.set_yticklabels([int(((min_val_y + dy)//10)*10), int(((min_val_y + 2*dy)//10 + 1)*10)])#, fontsize=font_tick)

    
    f, ax = plt.subplots()
    
    print(f"SSRT medio: {mean_SSRT_all}ms")
    t_backward = -((np.arange(t_interval) + t_start) * (5 * tau))   # same expression you use
    
    t0 = -mean_SSRT_all
    idx_v = np.argmin(np.abs(t_backward - t0))
    t_v = t_backward[idx_v]
    move_perc_v = mean_move_perc[idx_v]

#     move_perc0 = 0.5
#     # find index where y is closest to 0.5
#     idx_h = np.argmin(np.abs(mean_move_perc - move_perc0))
#     t_h = t_backward[idx_h]
#     move_perc_h = mean_move_perc[idx_h]

    ax.plot(t_backward, mean_move_perc, color=color_line)
#     ax.errorbar(-((np.arange(t_interval) + t_start)*(5*tau)), mean_move_perc, mfc=color_point, mec='black', fmt='o', ecolor='black',  elinewidth=1, capsize=5, capthick=1)
    ax.scatter(-((np.arange(t_interval) + t_start)*(5*tau)),
               mean_move_perc,
               c=color_point,       # Corrisponde a mfc (marker face color)
               edgecolors='black',  # Corrisponde a mec (marker edge color)
               marker='o',          # Corrisponde a fmt='o'
#                linewidths=1,        # Spessore del bordo (opzionale, default simile ai markers)
               zorder=2)            # Utile se vuoi assicurarti che i punti stiano "sopra" eventuali linee
    ax.vlines(t_v, ymin=0, ymax=move_perc_v, color=color_vline, linestyle="--", zorder=1)#,label= f"mean SSRT = {mean_SSRT_all}")
    ax.hlines(move_perc_v, xmin=t_backward.min(), xmax=t_v, color=color_vline, linestyle="--", zorder=1)
#         ax.axvline(left_lim, color="b",linestyle="--",label= f"left_lim = {left_lim}")
#         ax.axvline(right_lim, color="b",linestyle="--",label= f"right_lim = {right_lim}")
    ax.set_xlabel('Time before RT ($ms$)')#, fontsize=font_ax)
    ax.set_ylabel('Probability of stop failure')#, fontsize=font_ax)
    ax.set_xticks([-200, -400])
    ax.set_xticklabels([-200, -400])#, fontsize=font_tick)
    ax.set_ylim((0, 1))
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([0, 0.5, 1])#, fontsize=font_tick)

#     ax.text(
#         -0.15, 1.05, "B",
#         transform=ax.transAxes,
#         fontsize=22,
#         fontweight="bold",
#         va="top",
#         ha="left"
#     )

    fig_file = os.path.join(comm_dict["saved_path"], 'pMove_vs_t.png')
    plt.savefig(fig_file)
#     ax.set_title('Mean percentage of movement onset vs stop time going backward from the RT')
#     ax.legend()

#     f, ax = plt.subplots(1, 2, figsize = (12, 5))
#     #ax.plot(RT_array, SSRT_mean_array, color = "r", label = "mean SSRT of each group")
#     ax[0].scatter(RT_array, diff_RT_array, color = "r", label = "mean RT residual of each group")
#     ax[0].plot(RT_array, pred_diff_RT, color = "r", 
#                label = f"linear fit: m={reg.coef_.item():.2f}, q={reg.intercept_.item():.2f}")
# #     ax.errorbar(np.arange(RT_groups), mean_move_perc, yerr=std_move_perc, fmt='o', ls='--', ecolor='black', 
# #                              elinewidth=1, capsize=5, capthick=1)
# #     ax.axvline(mean_SSRT, color=color[k],linestyle="--",label= f"mean SSRT = {mean_SSRT}")
# #         ax.axvline(left_lim, color="b",linestyle="--",label= f"left_lim = {left_lim}")
# #         ax.axvline(right_lim, color="b",linestyle="--",label= f"right_lim = {right_lim}")
#     ax[0].set_xlabel('mean RT of the groups ($ms$)', fontsize=font_ax)
#     ax[0].set_ylabel('RT stop residual', fontsize=font_ax)
#     ax[0].set_title('RT stop residual of the groups mean vs RT')
# #     ax[0].set_xlim(0, RT_array.max())
# #     ax[0].set_ylim(0, SSRT_mean_array.max())
#     ax[0].legend()


#     #ax.plot(RT_array, SSRT_mean_array, color = "r", label = "mean SSRT of each group")
#     ax[1].scatter(RT_true_array, diff_gen_array, color = "r", label = "mean RT residual of each group")
#     ax[1].plot(RT_true_array, pred_diff_gen, color = "r", 
#                label = f"linear fit: m={reg_true.coef_.item():.2f}, q={reg_true.intercept_.item():.2f}")
# #     ax.errorbar(np.arange(RT_groups), mean_move_perc, yerr=std_move_perc, fmt='o', ls='--', ecolor='black', 
# #                              elinewidth=1, capsize=5, capthick=1)
# #     ax.axvline(mean_SSRT, color=color[k],linestyle="--",label= f"mean SSRT = {mean_SSRT}")
# #         ax.axvline(left_lim, color="b",linestyle="--",label= f"left_lim = {left_lim}")
# #         ax.axvline(right_lim, color="b",linestyle="--",label= f"right_lim = {right_lim}")
#     ax[1].set_xlabel('mean RT of the groups ($ms$)', fontsize=font_ax)
#     ax[1].set_ylabel('RT go residual', fontsize=font_ax)
#     ax[1].set_title('RT go residual of the groups mean vs RT')
# #     ax[0].set_xlim(0, RT_array.max())
# #     ax[0].set_ylim(0, SSRT_mean_array.max())
#     ax[1].legend()


    mask_f = RT_stop > 0
    move_perc = move_perc[mask_f]
    SSRT_critic = SSRT_critic[mask_f]
    print(RT_stop.shape[0], mask_f.sum())
    RT_stop = RT_stop[mask_f]
    RT_go = RT_go[mask_f]
#     RT_pred = (RT_pred_step*tau)*5
#     RT_time = RT_pred ###############################
    RT_time_masked = RT_time_masked[mask_f]
    z_mean_stop = z_mean_stop[mask_f]
    z_mean_go = z_mean_go[mask_f]
    z_critic = z_critic[mask_f]
    frac_realizations = frac_realizations[mask_f]
    
    diff_RT = RT_time_masked - RT_stop 
    diff_gen = RT_go - RT_stop
    diff_true = RT_time_masked - RT_go

    x_critic = z_critic[:, axis[0]]
    y_critic = z_critic[:, axis[1]]
    color_critic =  (RT_time_masked - RT_time_masked.min()) / (RT_time_masked.max() - RT_time_masked.min())
    color_stop =  (RT_stop - RT_stop.min()) / (RT_stop.max() - RT_stop.min())

    mask_slow = diff_gen < 0
    z_critic_slow = z_critic[mask_slow]
    z_critic_fast = z_critic[~mask_slow]
    RT_stop_slow = RT_stop[mask_slow]
    RT_stop_fast = RT_stop[~mask_slow]

    x_critic_slow = z_critic_slow[:, axis[0]]
    y_critic_slow = z_critic_slow[:, axis[1]]
    x_critic_fast = z_critic_fast[:, axis[0]]
    y_critic_fast = z_critic_fast[:, axis[1]]

    x_mean_stop_slow = z_mean_stop[mask_slow, :, axis[0]]
    y_mean_stop_slow = z_mean_stop[mask_slow, :, axis[1]]
    x_mean_stop_fast = z_mean_stop[~mask_slow, :, axis[0]]
    y_mean_stop_fast = z_mean_stop[~mask_slow, :, axis[1]]
    x_mean_go = z_mean_go[:, :, axis[0]]
    y_mean_go = z_mean_go[:, :, axis[1]]

    mean_diff_RT = diff_RT.mean()
    std_diff_RT = diff_RT.std()

    mean_diff_gen = diff_gen.mean()
    std_diff_gen = diff_gen.std()

    mean_diff_true = diff_true.mean()
    std_diff_true = diff_true.std()
    
    
    
    
    f, ax = plt.subplots(3, 3, figsize = (16, 16))
    
    ax[0, 0].scatter(RT_time_masked, diff_RT)
#     ax[0, 0].axhline(mean_diff_RT, color="r",linestyle="--",label= f"mean residual")
#     ax[0].errorbar(diff_RT, mean_diff_RT, yerr=std_diff_RT, fmt='o', ls='--', ecolor='black', 
#                              elinewidth=1, capsize=5, capthick=1)
#     ax.axvline(mean_SSRT_all, color="r",linestyle="--",label= f"mean SSRT = {mean_SSRT_all}")
#         ax.axvline(left_lim, color="b",linestyle="--",label= f"left_lim = {left_lim}")
#         ax.axvline(right_lim, color="b",linestyle="--",label= f"right_lim = {right_lim}")
    ax[0, 0].set_xlabel('True RT ($ms$)')#, fontsize=font_ax)
    ax[0, 0].set_ylabel('RT stop residual')#, fontsize=font_ax)
    ax[0, 0].set_title('Mean RT stop residual vs True RT')
#     ax.legend()

    # Calcola i limiti comuni
    min_val_x = ax[0, 0].get_xlim()[0]
    min_val_y = ax[0, 0].get_ylim()[0]
    max_val_x = ax[0, 0].get_xlim()[1]
    max_val_y = ax[0, 0].get_ylim()[1]

    # Disegna la diagonale y = x
    ax[0, 0].plot([min_val_x, max_val_x], [min_val_y, max_val_y], 'r--', label='diagonalee')

    ax[0, 1].scatter(RT_time_masked, diff_gen)
#     ax[0, 1].axhline(mean_diff_gen, color="r",linestyle="--",label= f"mean residual")
#     ax[1].errorbar((np.arange(t_interval) + t_start)*(5*tau), mean_diff_gen, yerr=std_diff_gen, fmt='o', ls='--', ecolor='black', 
#                              elinewidth=1, capsize=5, capthick=1)
#     ax.axvline(mean_SSRT_all, color="r",linestyle="--",label= f"mean SSRT = {mean_SSRT_all}")
#         ax.axvline(left_lim, color="b",linestyle="--",label= f"left_lim = {left_lim}")
#         ax.axvline(right_lim, color="b",linestyle="--",label= f"right_lim = {right_lim}")
    ax[0, 1].set_xlabel('True RT ($ms$)')#, fontsize=font_ax)
    ax[0, 1].set_ylabel('RT stop residual')#, fontsize=font_ax)
    ax[0, 1].set_title('Mean RT stop residual vs True RT')
    
    # Calcola i limiti comuni
    min_val_x = ax[0, 1].get_xlim()[0]
    min_val_y = ax[0, 1].get_ylim()[0]
    max_val_x = ax[0, 1].get_xlim()[1]
    max_val_y = ax[0, 1].get_ylim()[1]

    # Disegna la diagonale y = x
    ax[0, 1].plot([min_val_x, max_val_x], [min_val_y, max_val_y], 'r--', label='diagonalee')
    
    ax[0, 2].scatter(RT_time_masked, diff_true)
#     ax[0, 2].axhline(mean_diff_true, color="r",linestyle="--",label= f"mean residual")
#     ax[1].errorbar((np.arange(t_interval) + t_start)*(5*tau), mean_diff_gen, yerr=std_diff_gen, fmt='o', ls='--', ecolor='black', 
#                              elinewidth=1, capsize=5, capthick=1)
#     ax.axvline(mean_SSRT_all, color="r",linestyle="--",label= f"mean SSRT = {mean_SSRT_all}")
#         ax.axvline(left_lim, color="b",linestyle="--",label= f"left_lim = {left_lim}")
#         ax.axvline(right_lim, color="b",linestyle="--",label= f"right_lim = {right_lim}")
    ax[0, 2].set_xlabel('True RT ($ms$)')#, fontsize=font_ax)
    ax[0, 2].set_ylabel('RT go residual')#, fontsize=font_ax)
    ax[0, 2].set_title('Mean RT go residual vs True RT')
    
    # Calcola i limiti comuni
    min_val_x = ax[0, 2].get_xlim()[0]
    min_val_y = ax[0, 2].get_ylim()[0]
    max_val_x = ax[0, 2].get_xlim()[1]
    max_val_y = ax[0, 2].get_ylim()[1]

    # Disegna la diagonale y = x
    ax[0, 2].plot([min_val_x, max_val_x], [min_val_y, max_val_y], 'r--', label='diagonalee')
    
    ax[1, 0].scatter(RT_time_masked, frac_realizations)
    ax[1, 0].axhline(frac_realizations.mean(), color="r",linestyle="--",label= f"mean residual")
#     ax[1].errorbar((np.arange(t_interval) + t_start)*(5*tau), mean_diff_gen, yerr=std_diff_gen, fmt='o', ls='--', ecolor='black', 
#                              elinewidth=1, capsize=5, capthick=1)
#     ax.axvline(mean_SSRT_all, color="r",linestyle="--",label= f"mean SSRT = {mean_SSRT_all}")
#         ax.axvline(left_lim, color="b",linestyle="--",label= f"left_lim = {left_lim}")
#         ax.axvline(right_lim, color="b",linestyle="--",label= f"right_lim = {right_lim}")
    ax[1, 0].set_xlabel('True RT ($ms$)')#, fontsize=font_ax)
    ax[1, 0].set_ylabel('fraction of trials kept')#, fontsize=font_ax)
    ax[1, 0].set_title('Fraction of trials kept vs True RT')

    ax[1, 1].scatter(RT_go, diff_gen)
    ax[1, 1].axhline(mean_diff_gen, color="r",linestyle="--",label= f"mean residual")
#     ax[1].errorbar((np.arange(t_interval) + t_start)*(5*tau), mean_diff_gen, yerr=std_diff_gen, fmt='o', ls='--', ecolor='black', 
#                              elinewidth=1, capsize=5, capthick=1)
#     ax.axvline(mean_SSRT_all, color="r",linestyle="--",label= f"mean SSRT = {mean_SSRT_all}")
#         ax.axvline(left_lim, color="b",linestyle="--",label= f"left_lim = {left_lim}")
#         ax.axvline(right_lim, color="b",linestyle="--",label= f"right_lim = {right_lim}")
    ax[1, 1].set_xlabel('Predicted RT ($ms$)')#, fontsize=font_ax)
    ax[1, 1].set_ylabel('RT stop residual')#, fontsize=font_ax)
    ax[1, 1].set_title('Mean RT stop residual vs predicted RT')
    
    ax[1, 2].scatter(RT_go, diff_true)
    ax[1, 2].axhline(mean_diff_true, color="r",linestyle="--",label= f"mean residual")
#     ax[1].errorbar((np.arange(t_interval) + t_start)*(5*tau), mean_diff_gen, yerr=std_diff_gen, fmt='o', ls='--', ecolor='black', 
#                              elinewidth=1, capsize=5, capthick=1)
#     ax.axvline(mean_SSRT_all, color="r",linestyle="--",label= f"mean SSRT = {mean_SSRT_all}")
#         ax.axvline(left_lim, color="b",linestyle="--",label= f"left_lim = {left_lim}")
#         ax.axvline(right_lim, color="b",linestyle="--",label= f"right_lim = {right_lim}")
    ax[1, 2].set_xlabel('Predicted RT ($ms$)')#, fontsize=font_ax)
    ax[1, 2].set_ylabel('RT go residual')#, fontsize=font_ax)
    ax[1, 2].set_title('Mean RT go residual vs predicted RT')
    
    
    ax[2, 0].scatter(RT_time_masked, RT_stop)
    ax[2, 0].axhline(RT_stop.mean(), color="r",linestyle="--",label= f"mean RT_stop")
#     ax[0].errorbar(diff_RT, mean_diff_RT, yerr=std_diff_RT, fmt='o', ls='--', ecolor='black', 
#                              elinewidth=1, capsize=5, capthick=1)
#     ax.axvline(mean_SSRT_all, color="r",linestyle="--",label= f"mean SSRT = {mean_SSRT_all}")
#         ax.axvline(left_lim, color="b",linestyle="--",label= f"left_lim = {left_lim}")
#         ax.axvline(right_lim, color="b",linestyle="--",label= f"right_lim = {right_lim}")
    ax[2, 0].set_xlabel('True RT ($ms$)')#, fontsize=font_ax)
    ax[2, 0].set_ylabel('RT stop')#, fontsize=font_ax)
    ax[2, 0].set_title('Mean RT stop vs True RT')
#     ax.legend()

    ax[2, 1].scatter(RT_time_masked, RT_go)
    #ax[2, 1].axhline(RT_go.mean(), color="r",linestyle="--",label= f"mean RT_go")
#     ax[1].errorbar((np.arange(t_interval) + t_start)*(5*tau), mean_diff_gen, yerr=std_diff_gen, fmt='o', ls='--', ecolor='black', 
#                              elinewidth=1, capsize=5, capthick=1)
#     ax.axvline(mean_SSRT_all, color="r",linestyle="--",label= f"mean SSRT = {mean_SSRT_all}")
#         ax.axvline(left_lim, color="b",linestyle="--",label= f"left_lim = {left_lim}")
#         ax.axvline(right_lim, color="b",linestyle="--",label= f"right_lim = {right_lim}")
    ax[2, 1].set_xlabel('True RT ($ms$)')#, fontsize=font_ax)
    ax[2, 1].set_ylabel('RT go')#, fontsize=font_ax)
    ax[2, 1].set_title('Mean RT go vs True RT')
    
    # Calcola i limiti comuni
    min_val = min(ax[2, 1].get_xlim()[0], ax[2, 1].get_ylim()[0])
    max_val = max(ax[2, 1].get_xlim()[1], ax[2, 1].get_ylim()[1])

    # Disegna la diagonale y = x
    ax[2, 1].plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    
    corr_go_stop = np.corrcoef(RT_go, RT_stop)[0, 1]
    
    ax[2, 2].scatter(RT_go, RT_stop)#, s=size_point)
    #ax[2, 2].axhline(RT_stop.mean(), color="r",linestyle="--",label= f"mean residual")
#     ax[1].errorbar((np.arange(t_interval) + t_start)*(5*tau), mean_diff_gen, yerr=std_diff_gen, fmt='o', ls='--', ecolor='black', 
#                              elinewidth=1, capsize=5, capthick=1)
#     ax.axvline(mean_SSRT_all, color="r",linestyle="--",label= f"mean SSRT = {mean_SSRT_all}")
#         ax.axvline(left_lim, color="b",linestyle="--",label= f"left_lim = {left_lim}")
#         ax.axvline(right_lim, color="b",linestyle="--",label= f"right_lim = {right_lim}")
    ax[2, 2].set_xlabel('RT GO gen ($ms$)')#, fontsize=font_ax)
    ax[2, 2].set_ylabel('RT STOP gen')#, fontsize=font_ax)
    ax[2, 2].set_title(f'Correlation={corr_go_stop:.2f}')
    
    # Calcola i limiti comuni
    min_val = min(ax[2, 2].get_xlim()[0], ax[2, 2].get_ylim()[0])
    max_val = max(ax[2, 2].get_xlim()[1], ax[2, 2].get_ylim()[1])

    # Disegna la diagonale y = x
    ax[2, 2].set_xlim([400, 820])
    ax[2, 2].set_ylim([400, 820])
    ax[2, 2].plot([400, 820], [400, 820], 'r--', label='y = x')
    ax[2, 2].set_xticks([500, 700]) 
    ax[2, 2].set_xticklabels([500, 700])  # Show corresponding labels
    ax[2, 2].set_yticks([500, 700])
    ax[2, 2].set_yticklabels([500, 700])
    
    
    plt.tight_layout()
    plt.show()
    
    
    f, ax = plt.subplots()
    
    ax.scatter(RT_go, RT_stop)#, s=size_point)
    #ax.axhline(RT_stop.mean(), color="r",linestyle="--",label= f"mean residual")
#     ax.errorbar((np.arange(t_interval) + t_start)*(5*tau), mean_diff_gen, yerr=std_diff_gen, fmt='o', ls='--', ecolor='black', 
#                              elinewidth=1, capsize=5, capthick=1)
#     ax.axvline(mean_SSRT_all, color="r",linestyle="--",label= f"mean SSRT = {mean_SSRT_all}")
#         ax.axvline(left_lim, color="b",linestyle="--",label= f"left_lim = {left_lim}")
#         ax.axvline(right_lim, color="b",linestyle="--",label= f"right_lim = {right_lim}")
    ax.set_xlabel('RT GO gen ($ms$)')#, fontsize=font_ax)
    ax.set_ylabel('RT STOP gen')#, fontsize=font_ax)
    ax.set_title(f'Correlation={corr_go_stop:.2f}')
    
    # Calcola i limiti comuni
    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])

    # Disegna la diagonale y = x
    ax.set_xlim([400, 820])
    ax.set_ylim([400, 820])
    ax.plot([400, 820], [400, 820], 'r--', label='y = x')
    ax.set_xticks([500, 700]) 
    ax.set_xticklabels([500, 700])  # Show corresponding labels
    ax.set_yticks([500, 700])
    ax.set_yticklabels([500, 700])
    
    fig_file = os.path.join(comm_dict["saved_path"], 'RTstop_vs_RTgo.png')
    plt.savefig(fig_file)
    
    
    f, ax = plt.subplots()
    
    diff_RT = RT_go - RT_stop
    ax.hist(diff_RT, bins=bins, alpha = alpha, density=True, color="skyblue", edgecolor='none')#, label = "true RT")
    ax.axvline(diff_RT.mean(), color="black",linestyle="--")#,label= "mean mod RT")
#     x_min, x_max = ax.get_xlim()
#     dx = (x_max - x_min)//3
#     ax.set_xticks([int(((x_min + dx)//100)*100), int(((x_min + 2*dx)//100)*100)])
#     ax.set_xticklabels([int(((x_min + dx)//100)*100), int(((x_min + 2*dx)//100)*100)], fontsize=font_tick)
    ax.set_xlim(xlims)
    ax.set_xticks(x_ticks_hist)
    ax.set_xticklabels(x_ticks_hist)#, fontsize=font_tick)
    ax.set_yticks([])
    #ax1.set_yticklabels([], fontsize=font_tick)
    # Add labels and title
    ax.set_xlabel('RTgo-RTstop ($ms$)')#, fontsize=font_ax)
    ax.set_ylabel('Counts')#, fontsize=font_ax)
    ax.set_title(f"Mean diff_RT={diff_RT.mean():.0f}$ms$")
#     ax1.legend(fontsize=font_leg)
    fig_file = os.path.join(comm_dict["saved_path"], f'diffRT_hist.png')
    plt.savefig(fig_file)
    
    
    
    
    #########################################################################################
    
    _, z_ws, z_cs = infer_latent(dmm, data, device)
    
    x_cs = z_cs[(SSD_cs+56)//tau, np.arange(z_cs.shape[1]), axis[0]]
    x_ws = z_ws[(SSD_ws+56)//tau, np.arange(z_ws.shape[1]), axis[0]]

    y_cs = z_cs[(SSD_cs+56)//tau, np.arange(z_cs.shape[1]), axis[1]]
    y_ws = z_ws[(SSD_ws+56)//tau, np.arange(z_ws.shape[1]), axis[1]]
    
    z_cn, RT, q = random_latent_cn_traj(dmm, data, tau, device)
    
    print(f"traj n.{q}")
    
    x_true_story = z_cn[:, axis[0]]
    y_true_story = z_cn[:, axis[1]]
    x_start = z_cn[0, axis[0]]
    y_start = z_cn[0, axis[1]]
    x_GO = z_cn[56//tau, axis[0]]
    y_GO = z_cn[56//tau, axis[1]]
    x_RT = z_cn[RT, axis[0]]
    y_RT = z_cn[RT, axis[1]]
    
    f, ax = plt.subplots(figsize = (7, 6))
    
    # Plot the surface
    ax.plot(x_mean_go.T, y_mean_go.T, '-', linewidth=2, color='g', alpha=0.1)
    ax.plot(x_true_story, y_true_story, '-', linewidth=2, color='brown', label = "trajectory example")
    n_arrows = 15
    arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
    for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = x_true_story[i+1] - x_true_story[i]
        dy = y_true_story[i+1] - y_true_story[i]
        ax.arrow(x_true_story[i], y_true_story[i], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    ax.scatter(x_GO, y_GO, s = 50, c = 'b', alpha = 1, label = "GO")
    ax.scatter(x_RT, y_RT, s = 50, c = 'r', alpha = 1, label = "RT")
    ax.scatter(x_start, y_start, s = 50, c = 'purple', alpha = 1, label = "start")
    ax.set_xlim(z_lims[axis[0]])
    ax.set_ylim(z_lims[axis[1]])
    ax.set_xlabel("z1")#, fontsize=font_ax)
    ax.set_ylabel("z2")#, fontsize=font_ax)
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)
    ax.set_title("x_mean_go")
#     ax.legend()
    
    f, ax = plt.subplots(1, 2, figsize = (12, 6))
    
    # Plot the surface
    ax[0].scatter(x_critic_slow, y_critic_slow, s = 10, c = "b", alpha = 0.2, label = "slow stops")
    ax[0].scatter(x_critic_fast, y_critic_fast, s = 10, c = "r", alpha = 0.2, label = "fast stops")
    ax[0].plot(x_true_story, y_true_story, '-', linewidth=2, color='brown', label = "trajectory example")
    n_arrows = 15
    arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
    for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = x_true_story[i+1] - x_true_story[i]
        dy = y_true_story[i+1] - y_true_story[i]
        ax[0].arrow(x_true_story[i], y_true_story[i], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    ax[0].scatter(x_GO, y_GO, s = 50, c = 'b', alpha = 1, label = "GO")
    ax[0].scatter(x_RT, y_RT, s = 50, c = 'r', alpha = 1, label = "RT")
    ax[0].scatter(x_start, y_start, s = 50, c = 'purple', alpha = 1, label = "start")
    ax[0].set_xlim(z_lims[axis[0]])
    ax[0].set_ylim(z_lims[axis[1]])
    ax[0].set_xlabel("z1")#, fontsize=font_ax)
    ax[0].set_ylabel("z2")#, fontsize=font_ax)
    ax[0].set_xticks(z_ticks[axis[0]]) 
    ax[0].set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
    ax[0].set_yticks(z_ticks[axis[1]])
    ax[0].set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)
    ax[0].set_title("slow (blue) vs fast (red) ssrt (RTgo - RTstop)")
#     ax[0].legend()
    

    # Plot the surface
    ax[1].plot(x_mean_stop_fast.T, y_mean_stop_fast.T, '-', linewidth=2, color='r', alpha=0.1)
    ax[1].plot(x_mean_stop_slow.T, y_mean_stop_slow.T, '-', linewidth=2, color='b', alpha=0.1)
    ax[1].plot(x_true_story, y_true_story, '-', linewidth=2, color='brown', label = "trajectory example")
    n_arrows = 15
    arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
    for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = x_true_story[i+1] - x_true_story[i]
        dy = y_true_story[i+1] - y_true_story[i]
        ax[1].arrow(x_true_story[i], y_true_story[i], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    ax[1].scatter(x_GO, y_GO, s = 50, c = 'b', alpha = 1, label = "GO")
    ax[1].scatter(x_RT, y_RT, s = 50, c = 'r', alpha = 1, label = "RT")
    ax[1].scatter(x_start, y_start, s = 50, c = 'purple', alpha = 1, label = "start")
    ax[1].set_xlim(z_lims[axis[0]])
    ax[1].set_ylim(z_lims[axis[1]])
    ax[1].set_xlabel("z1")#, fontsize=font_ax)
    ax[1].set_ylabel("z2")#, fontsize=font_ax)
    ax[1].set_xticks(z_ticks[axis[0]]) 
    ax[1].set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
    ax[1].set_yticks(z_ticks[axis[1]])
    ax[1].set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)
    ax[0].set_title("slow (blue) vs fast (red) stop traj (RTgo - RTstop)")
#     ax[1].legend()
    
    
    f, ax = plt.subplots(figsize = (7, 6))

    # Plot the surface
    ax.scatter(x_cs, y_cs, s = 20, c = 'r', alpha = 0.8, label = "cs stops")
    ax.scatter(x_ws, y_ws, s = 20, c = 'g', alpha = 0.8, label = "ws stops")
    ax.scatter(x_critic, y_critic, s = 10, c = 'b', alpha = 0.2, label = "ssd critical")
    ax.plot(x_true_story, y_true_story, '-', linewidth=2, color='brown', label = "trajectory example")
    n_arrows = 15
    arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
    for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = x_true_story[i+1] - x_true_story[i]
        dy = y_true_story[i+1] - y_true_story[i]
        ax.arrow(x_true_story[i], y_true_story[i], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    ax.scatter(x_GO, y_GO, s = 50, c = 'b', alpha = 1, label = "GO")
    ax.scatter(x_RT, y_RT, s = 50, c = 'r', alpha = 1, label = "RT")
    ax.scatter(x_start, y_start, s = 50, c = 'purple', alpha = 1, label = "start")
    ax.set_xlim(z_lims[axis[0]])
    ax.set_ylim(z_lims[axis[1]])
    ax.set_xlabel("z1")#, fontsize=font_ax)
    ax.set_ylabel("z2")#, fontsize=font_ax)
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)
    ax.set_title("cs vs ws stop")
#     ax.legend()
    
    f, ax = plt.subplots(1, 2, figsize = (12, 6))

    # Plot the surface
#     ax.scatter(x_cs, y_cs, s = 20, c = 'r', alpha = 0.8, label = "cs stops")
#     ax.scatter(x_ws, y_ws, s = 20, c = 'g', alpha = 0.8, label = "ws stops")
    ax[0].scatter(x_critic, y_critic, s = 15, c = color_critic, cmap="Blues", alpha = 1, label = "ssd critical")
    ax[0].plot(x_true_story, y_true_story, '-', linewidth=2, color='brown', label = "trajectory example")
    n_arrows = 15
    arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
    for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = x_true_story[i+1] - x_true_story[i]
        dy = y_true_story[i+1] - y_true_story[i]
        ax[0].arrow(x_true_story[i], y_true_story[i], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    ax[0].scatter(x_GO, y_GO, s = 50, c = 'b', alpha = 1, label = "GO")
    ax[0].scatter(x_RT, y_RT, s = 50, c = 'r', alpha = 1, label = "RT")
    ax[0].scatter(x_start, y_start, s = 50, c = 'purple', alpha = 1, label = "start")
    ax[0].set_xlim(z_lims[axis[0]])
    ax[0].set_ylim(z_lims[axis[1]])
    ax[0].set_xlabel("z1")#, fontsize=font_ax)
    ax[0].set_ylabel("z2")#, fontsize=font_ax)
    ax[0].set_xticks(z_ticks[axis[0]]) 
    ax[0].set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
    ax[0].set_yticks(z_ticks[axis[1]])
    ax[0].set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)
    ax[0].set_title("ssrt coloured by true RT")
#     ax[0].legend()
    
    # Plot the surface
    ax[1].scatter(x_critic, y_critic, s = 10, c = color_stop, cmap="Reds", alpha = 1, label = "ssd critical")
    ax[1].plot(x_true_story, y_true_story, '-', linewidth=2, color='brown', label = "trajectory example")
    n_arrows = 15
    arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
    for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = x_true_story[i+1] - x_true_story[i]
        dy = y_true_story[i+1] - y_true_story[i]
        ax[1].arrow(x_true_story[i], y_true_story[i], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    ax[1].scatter(x_GO, y_GO, s = 50, c = 'b', alpha = 1, label = "GO")
    ax[1].scatter(x_RT, y_RT, s = 50, c = 'r', alpha = 1, label = "RT")
    ax[1].scatter(x_start, y_start, s = 50, c = 'purple', alpha = 1, label = "start")
    ax[1].set_xlim(z_lims[axis[0]])
    ax[1].set_ylim(z_lims[axis[1]])
    ax[1].set_xlabel("z1")#, fontsize=font_ax)
    ax[1].set_ylabel("z2")#, fontsize=font_ax)
    ax[1].set_xticks(z_ticks[axis[0]]) 
    ax[1].set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
    ax[1].set_yticks(z_ticks[axis[1]])
    ax[1].set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)
    ax[1].set_title("ssrt coloured by stop RT")
#     ax[1].legend()

    z_cn, _, _ = infer_latent(dmm, plot_data, device)
    
    point_cs = z_cs[(SSD_cs+56)//tau, np.arange(z_cs.shape[1])]
    point_ws = z_ws[(SSD_ws+56)//tau, np.arange(z_ws.shape[1])]

    import plotly.graph_objects as go

    fig = go.Figure()

    for i in range(z_cn.shape[1]):
        fig.add_trace(go.Scatter3d(
            x=z_cn[:, i, 0], y=z_cn[:, i, 1], z=z_cn[:, i, 2],
            mode='lines', line=dict(color="blue", width=3), opacity=opacity, showlegend=False
        ))

    fig.add_trace(go.Scatter3d(
        x=point_cs[:, 0], y=point_cs[:, 1], z=point_cs[:, 2],
        mode='markers',  # FIXED: Added mode
        marker=dict(color='red', size=5),  # FIXED: color moved inside marker dict
        opacity=opacity_stop,
        name='CS Points'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=point_ws[:, 0], y=point_ws[:, 1], z=point_ws[:, 2],
        mode='markers',  # FIXED: Added mode
        marker=dict(color='green', size=5),  # FIXED: color moved inside marker dict; "g" -> "green"
        opacity=opacity_stop,
        name='WS Points'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=z_critic[:, 0], y=z_critic[:, 1], z=z_critic[:, 2],
        mode='markers',  # FIXED: Added mode
        marker=dict(color='purple', size=5),  # FIXED: color moved inside marker dict
        opacity=opacity_ssrt,
        name='SSRT Points'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='z1', yaxis_title='z2', zaxis_title='z3',
            bgcolor='black'
        ),
        width=700, height=700,
#         title='Right vs Left trajectories'
    )
    return fig


def RT_stop_dependence(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    t = comm_dict["t"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    RT_detector = comm_dict["RT_detector"]
    move_detector = comm_dict["move_detector"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    cmap = diff_dict["cmap"]
    SSD_interval = diff_dict["SSD_interval"]
    RT_groups = diff_dict["RT_groups"]
    mean_z_flag = diff_dict["mean_z_flag"]
    #axis = diff_dict["axis"]
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    set_cs = data["set_cs_ordSSD"]
    
    # Preallocazione risultati
    n, steps, features = set_cn.shape
    RT_cn_step = (RT_cn+56)//tau
    RT_time = (RT_cn+56)*5
    
    t_start, t_end = SSD_interval
    t_interval = t_end - t_start
    chunk_size = 10
    n_batch = n_trials // chunk_size if (n_trials%chunk_size)==0 else (n_trials // chunk_size) + 1
    
    cont = torch.zeros((1, n*chunk_size, 4)).to(device)
    cont[:, :, 3] = 1
    
    RT_mean_stop_list = []
    RT_mean_go_list = []
    mean_RT_list = []
    with torch.no_grad():
        # 🔁 Ora ciclo esterno sul tempo t
        for idx_t, t in enumerate(range(t_start, t_end)):
            if idx_t % 5 ==0:
                print(f"time {idx_t}/{t_interval}")
            move_batch_list = []
            RT_gen_stop_list = []
            RT_gen_go_list = []
            mean_z_list = []
            teacher = np.repeat(RT_cn_step, chunk_size) - t
            alone = steps - teacher[0]

            # 🔁 Ciclo interno sui batch
            for batch in range(n_batch):

                # Input per batch
                test_set = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).repeat_interleave(chunk_size, dim=1)
                cont_c = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2).repeat_interleave(chunk_size, dim=1)

                # --- Inference
                z, z_mean, _ = dmm.inference(test_set, cont_c)

                # Teacher e z iniziale
                z_gen = z[teacher, torch.arange(len(teacher), device=z.device)].unsqueeze(0)
                cont_go = cont_c[teacher, torch.arange(len(teacher), device=z.device)].unsqueeze(0)
                z_tmp_stop = z_gen.clone()
                z_tmp_go = z_gen.clone()

                # --- Generazione z_teach
                z_teach_stop_list = []
                z_teach_go_list = []
                for _ in range(alone):
                    z_mean_gen, z_cov_gen = dmm.generation_z(z_tmp_stop, cont)
                    z_tmp_stop = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                    z_teach_stop_list.append(z_tmp_stop)
                    
                    z_mean_gen, z_cov_gen = dmm.generation_z(z_tmp_go, cont_go)
                    z_tmp_go = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                    z_teach_go_list.append(z_tmp_go)
                    
                z_teach_stop = torch.cat(z_teach_stop_list, dim=0)
                z_teach_go = torch.cat(z_teach_go_list, dim=0)

                # --- Ricostruzione mu_z
                mu_z_stop_list = []
                mu_z_go_list = []
                for i in range(z_teach_go.shape[1]):
                    diff = teacher[i] - teacher[0] + 1
                    z_final_stop = torch.cat((z[:(teacher[i] + 1), i], z_teach_stop[:-diff, i]), dim=0)
                    z_final_go = torch.cat((z[:(teacher[i] + 1), i], z_teach_go[:-diff, i]), dim=0)
                    mu_z_stop_list.append(z_final_stop)
                    mu_z_go_list.append(z_final_go)
                
                mu_z_stop = torch.stack(mu_z_stop_list, dim=1)  # [steps, n*batch, latent_dim]
                mu_z_stop = mu_z_stop.permute(1, 0, 2)          # [n*batch, steps, latent_dim]
                mu_z_go = torch.stack(mu_z_go_list, dim=1)  # [steps, n*batch, latent_dim]
                mu_z_go = mu_z_go.permute(1, 0, 2)          # [n*batch, steps, latent_dim]
                
                mean_z = mu_z_stop.reshape(n, chunk_size, steps, z_dim).cpu().detach().numpy() # [n, batch, steps, latent_dim]
                
                # --- Move detector
                move_logit = move_detector(mu_z_stop)
                move_output = binary_output(move_logit)
                move_batch = move_output.reshape(-1, chunk_size)   # n x batch 
                
                RT_prob_stop = RT_detector(mu_z_stop)
                RT_estimate_stop = prob_to_RT(RT_prob_stop, tau)  
                RT_estimate_stop = RT_estimate_stop.reshape(-1, chunk_size)   # n x batch 
                RT_gen_stop = (RT_estimate_stop*tau)*5
                
                RT_prob_go = RT_detector(mu_z_go)
                RT_estimate_go = prob_to_RT(RT_prob_go, tau)  
                RT_estimate_go = RT_estimate_go.reshape(-1, chunk_size)   # n x batch 
                RT_gen_go = (RT_estimate_go*tau)*5
                
                move_batch_list.append(move_batch)
                RT_gen_stop_list.append(RT_gen_stop)
                RT_gen_go_list.append(RT_gen_go)
                mean_z_list.append(mean_z)
            
            move_trials = np.concatenate(move_batch_list, axis=1) # n x n_trials 
            RT_gen_stop_trials = np.concatenate(RT_gen_stop_list, axis=1) # n x n_trials
            RT_gen_go_trials = np.concatenate(RT_gen_go_list, axis=1) # n x n_trials
            mean_z_batch = np.concatenate(mean_z_list, axis=1)  # n x n_trials x steps x z_dim 
            mean_z_trials = mean_z_batch.mean(1)   # n x steps x z_dim
            
            mask = move_trials > 0.5
            # metti NaN dove la maschera è False)
            RT_gen_masked = np.where(mask, RT_gen_stop_trials, np.nan)  # n x n_trials 
            # Calcola la media lungo l’asse 1 ignorando i NaN
            RT_mean_gen_stop = np.nanmean(RT_gen_masked, axis=1) # n 
            RT_mean_gen_go = np.mean(RT_gen_go_trials, axis=1)
            
            mean_z_trials = torch.from_numpy(mean_z_trials).float().to(device)
            move_mean_logit = move_detector(mean_z_trials)
            move_mean_output = binary_output(move_mean_logit) # n
            idx = np.where(move_mean_output == 0)[0]    
            if idx.size > 0:
                print(f"Generated trials n.{idx.tolist()} are correct-stop with a stop occurring at {t} steps before RT")
            else:
                RT_prob = RT_detector(mean_z_trials)
                RT_estimate = prob_to_RT(RT_prob, tau)  
                mean_RT_gen = (RT_estimate*tau)*5
                mean_RT_list.append(mean_RT_gen)
                
            RT_mean_stop_list.append(RT_mean_gen_stop)
            RT_mean_go_list.append(RT_mean_gen_go)
            mean_RT_list.append(mean_RT_gen)

        # Stack finale su tutti i tempi
        RT_mean_stop_array = np.stack(RT_mean_stop_list, axis=1)  # (n, t_end - t_start)
        RT_mean_go_array = np.stack(RT_mean_go_list, axis=1)  # (n, t_end - t_start)
        mean_RT_array = np.stack(mean_RT_list, axis=1)  # (n, t_end - t_start)
        
        idx = np.where(np.isnan(RT_mean_stop_array))[0]    
        if idx.size > 0:
            print(f"Generated trials n.{idx.tolist()} are nan")

        
        if mean_z_flag:
            #diff_RT = RT_time[:, None] - mean_RT_array
            diff_RT = RT_mean_go_array - mean_RT_array 
        else:
            #diff_RT = RT_time[:, None] - RT_mean_stop_array
            diff_RT = RT_mean_go_array - RT_mean_stop_array 
        diff_gen = RT_time[:, None] - RT_mean_go_array
        mean_diff_RT = diff_RT.mean(0)
        std_diff_RT = diff_RT.std(0)
        
        mean_diff_gen = diff_gen.mean(0)
        std_diff_gen = diff_gen.std(0)
   
        diff_RT_list = []
        diff_gen_list = []
        RT_list = []
        RT_true_list = []
        n_group = len(RT_time) // RT_groups if (len(RT_time)%RT_groups)==0 else (len(RT_time) // RT_groups) + 1
        for k in range(RT_groups):
            start = k*n_group
            end = start + n_group
            # calcolo l'RT medio di ogni gruppo di trials
            RT_group = RT_mean_go_array[start:end]
            mean_RT_group = RT_group.mean(0)
            
            RT_true = RT_time[start:end]
            mean_RT_true = RT_true.mean(0)
            # calcolo la media del gruppo di trials, e ne calcolo l'SSRT
            diff_RT_group = diff_RT[start:end]
            mean_diff_RT_group = diff_RT_group.mean(0)
            
            diff_gen_group = diff_gen[start:end]
            mean_diff_gen_group = diff_gen_group.mean(0)
            # riempo le liste
            diff_RT_list.append(mean_diff_RT_group.mean())
            diff_gen_list.append(mean_diff_gen_group.mean())
            RT_list.append(mean_RT_group.mean())
            RT_true_list.append(mean_RT_true)
        # converto le liste in array
        diff_RT_array = np.array(diff_RT_list)
        diff_gen_array = np.array(diff_gen_list)
        RT_array = np.array(RT_list)
        RT_true_array = np.array(RT_true_list)

#     idx = np.where(np.isnan(diff_RT_array))[0]    
#     if idx.size > 0:
#         print(f"Generated trials n.{idx.tolist()} are nan")

    from sklearn.linear_model import LinearRegression
    
    # --- STIMA DIREZIONE ---
    reg = LinearRegression().fit(RT_array[:, None], diff_RT_array)
    pred_diff_RT = reg.predict(RT_array[:, None])
    
    reg_true = LinearRegression().fit(RT_array[:, None], diff_gen_array)
    pred_diff_gen = reg_true.predict(RT_array[:, None])
    
    f, ax = plt.subplots(1, 2, figsize = (12, 6))
    
    x_positions = np.arange(diff_RT.shape[1])
    x_labels = x_positions * (tau*5)
    y_labels = np.arange(n)

    #ax1.plot(t*5, MUA, c ='r', label = 'mean MUA')
    im1 = ax[0].imshow(diff_RT, cmap = cmap, aspect='auto')#, vmin=0, vmax=1)
    ax[0].set_xticks(x_positions[::5]) 
    ax[0].set_xticklabels(x_labels[::5], fontsize=font_tick)  # Show corresponding labels
    ax[0].set_yticks([n//2, n])
    ax[0].set_yticklabels([n//2, n], fontsize=font_tick)
    ax[0].set_xlabel('time before RT ($ms$)', fontsize=font_ax)
    ax[0].set_ylabel('Trials ordered by RT', fontsize=font_ax)
    ax[0].set_title('RT stop residuals as a function of RT and stop time going backward from the RT')
    cbar=plt.colorbar(im1, ax=ax[0])
    cbar.ax.tick_params(labelsize=font_tick)
    
    im2 = ax[1].imshow(diff_gen, cmap = cmap, aspect='auto')#, vmin=0, vmax=1)
    ax[1].set_xticks(x_positions[::5]) 
    ax[1].set_xticklabels(x_labels[::5], fontsize=font_tick)  # Show corresponding labels
    ax[1].set_yticks([n//2, n])
    ax[1].set_yticklabels([n//2, n], fontsize=font_tick)
    ax[1].set_xlabel('time before RT ($ms$)', fontsize=font_ax)
    ax[1].set_ylabel('Trials ordered by RT', fontsize=font_ax)
    ax[1].set_title('RT go residuals as a function of RT and stop time going backward from the RT')
    cbar=plt.colorbar(im2, ax=ax[1])
    cbar.ax.tick_params(labelsize=font_tick)
    
    f, ax = plt.subplots(1, 2, figsize = (12, 5))
    #ax.plot(RT_array, SSRT_mean_array, color = "r", label = "mean SSRT of each group")
    ax[0].scatter(RT_array, diff_RT_array, color = "r", label = "mean RT residual of each group")
    ax[0].plot(RT_array, pred_diff_RT, color = "r", 
               label = f"linear fit: m={reg.coef_.item():.2f}, q={reg.intercept_.item():.2f}")
#     ax.errorbar(np.arange(RT_groups), mean_move_perc, yerr=std_move_perc, fmt='o', ls='--', ecolor='black', 
#                              elinewidth=1, capsize=5, capthick=1)
#     ax.axvline(mean_SSRT, color=color[k],linestyle="--",label= f"mean SSRT = {mean_SSRT}")
#         ax.axvline(left_lim, color="b",linestyle="--",label= f"left_lim = {left_lim}")
#         ax.axvline(right_lim, color="b",linestyle="--",label= f"right_lim = {right_lim}")
    ax[0].set_xlabel('mean RT of the groups ($ms$)', fontsize=font_ax)
    ax[0].set_ylabel('RT stop residual', fontsize=font_ax)
    ax[0].set_title('RT stop residual of the groups mean vs RT')
#     ax[0].set_xlim(0, RT_array.max())
#     ax[0].set_ylim(0, SSRT_mean_array.max())
    ax[0].legend()
    
    
    #ax.plot(RT_array, SSRT_mean_array, color = "r", label = "mean SSRT of each group")
    ax[1].scatter(RT_true_array, diff_gen_array, color = "r", label = "mean RT residual of each group")
    ax[1].plot(RT_true_array, pred_diff_gen, color = "r", 
               label = f"linear fit: m={reg_true.coef_.item():.2f}, q={reg_true.intercept_.item():.2f}")
#     ax.errorbar(np.arange(RT_groups), mean_move_perc, yerr=std_move_perc, fmt='o', ls='--', ecolor='black', 
#                              elinewidth=1, capsize=5, capthick=1)
#     ax.axvline(mean_SSRT, color=color[k],linestyle="--",label= f"mean SSRT = {mean_SSRT}")
#         ax.axvline(left_lim, color="b",linestyle="--",label= f"left_lim = {left_lim}")
#         ax.axvline(right_lim, color="b",linestyle="--",label= f"right_lim = {right_lim}")
    ax[1].set_xlabel('mean RT of the groups ($ms$)', fontsize=font_ax)
    ax[1].set_ylabel('RT go residual', fontsize=font_ax)
    ax[1].set_title('RT go residual of the groups mean vs RT')
#     ax[0].set_xlim(0, RT_array.max())
#     ax[0].set_ylim(0, SSRT_mean_array.max())
    ax[1].legend()
    
    
    f, ax = plt.subplots(1, 2, figsize = (12, 5))
    
    ax[0].plot((np.arange(t_interval) + t_start)*(5*tau), mean_diff_RT)
    ax[0].errorbar((np.arange(t_interval) + t_start)*(5*tau), mean_diff_RT, yerr=std_diff_RT, fmt='o', ls='--', ecolor='black', 
                             elinewidth=1, capsize=5, capthick=1)
#     ax.axvline(mean_SSRT_all, color="r",linestyle="--",label= f"mean SSRT = {mean_SSRT_all}")
#         ax.axvline(left_lim, color="b",linestyle="--",label= f"left_lim = {left_lim}")
#         ax.axvline(right_lim, color="b",linestyle="--",label= f"right_lim = {right_lim}")
    ax[0].set_xlabel('time before RT ($ms$)', fontsize=font_ax)
    ax[0].set_ylabel('RT stop residual', fontsize=font_ax)
    ax[0].set_title('Mean RT stop residual vs stop time going backward from the RT')
#     ax.legend()

    ax[1].plot((np.arange(t_interval) + t_start)*(5*tau), mean_diff_gen)
    ax[1].errorbar((np.arange(t_interval) + t_start)*(5*tau), mean_diff_gen, yerr=std_diff_gen, fmt='o', ls='--', ecolor='black', 
                             elinewidth=1, capsize=5, capthick=1)
#     ax.axvline(mean_SSRT_all, color="r",linestyle="--",label= f"mean SSRT = {mean_SSRT_all}")
#         ax.axvline(left_lim, color="b",linestyle="--",label= f"left_lim = {left_lim}")
#         ax.axvline(right_lim, color="b",linestyle="--",label= f"right_lim = {right_lim}")
    ax[1].set_xlabel('time before RT ($ms$)', fontsize=font_ax)
    ax[1].set_ylabel('RT go residual', fontsize=font_ax)
    ax[1].set_title('Mean RT go residual vs stop time going backward from the RT')

    


def examinate_traj(dmm, data, type_trial, q, axes, tau, device):
    if type_trial=="cn":
        set_c = data["set_cn_ordRT"]
        cont = data["cont_cn_ordRT"]
        RT = data["RT_cn_ordRT"]
    elif type_trial=="ws":
        set_c = data["set_ws_ordRT"]
        cont = data["cont_ws_ordRT"]
        RT = data["RT_ws_ordRT"]
    elif type_trial=="cs":
        set_c = data["set_cs_ordSSD"]
        cont = data["cont_cs_ordSSD"]
        SSD = data["SSD_cs_ordSSD"]

    MUA = set_c[q].mean(1)

    set_c = torch.from_numpy(set_c).float().to(device).permute(1, 0, 2)
    cont = torch.from_numpy(cont).float().to(device).permute(1, 0, 2)

    z, z_mean, _ = dmm.inference(set_c, cont)
    z = z[:, q].cpu().detach().numpy()

    x_true_story = z[:, axes[0]]
    y_true_story = z[:, axes[1]]
    x_start = z[0, axes[0]]
    y_start = z[0, axes[1]]
    x_GO = z[56//tau, axes[0]]
    y_GO = z[56//tau, axes[1]]


    t = np.linspace(0, 255, 256, dtype = int)        
    t = t[::tau]
    steps = set_c.shape[1]

    f, (ax1, ax2) = plt.subplots(2, 1, figsize = (8, 14))

    ax1.plot(t*5, MUA, c ='r', label = 'mean MUA')
    ax1.axvline(56*5, color="black",linestyle="--",label= "GO")
    if type_trial=="cs":
        ax1.axvline((SSD[q]+56)*5, color="r",linestyle="--",label= "SSD")
    ax1.set_title(f"Mean MUA of {type_trial} trial n.{q}")
    ax1.set_xlabel('time ($ms$)')
    ax1.set_ylabel('mean MUA activity over channels')
    ax1.legend(fontsize=7.3)

    ax2.plot(x_true_story, y_true_story, '-', linewidth=4, color="brown", alpha = 1, label = "trajectory example")
    ax2.scatter(x_GO, y_GO, s = 150, c = "b", alpha = 1, label = "GO")
    if type_trial=="cn":
        x_RT = z[(RT[q]+56)//tau, axes[0]]
        y_RT = z[(RT[q]+56)//tau, axes[1]]
        ax2.scatter(x_RT, y_RT, s = 150, c = 'r', alpha = 1, label = "RT")
    if type_trial=="cs":
        x_SSD = z[(SSD[q]+56)//tau, axes[0]]
        y_SSD = z[(SSD[q]+56)//tau, axes[1]]
        ax2.plot(x_SSD, y_SSD, color='red', marker='x', markeredgewidth = 3, markersize = 15, label = "SSD")
    ax2.scatter(x_start, y_start, s = 150, c = 'y', alpha = 1, label = "start")
    ax2.set_xlabel("first latent component")
    ax2.set_ylabel("second latent component")
    ax2.set_title(f'latent traj of {type_trial} trial n.{q}')
    ax2.legend()
    plt.show()


import math

def gen_field(comm_dict, diff_dict):

    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    c_dim = comm_dict["c_dim"]
    
    data = diff_dict["data"]
    points = diff_dict["points"]
    cont_type = diff_dict["cont_type"]
    stop_trial = diff_dict["stop_trial"]
    axis = diff_dict["axis"]
    
    cont_cn = data["cont_cn_ordRT"]
    cont_cs = data["cont_cs_ordSSD"]
    dir_cn = data["dir_cn_ordRT"]
    
    x_lim_l = -5.5
    x_lim_r = 7
    y_lim_l = -5
    y_lim_r = 10
#     z_lim_l = -10
#     z_lim_r = 10

    x_points = np.linspace(x_lim_l, x_lim_r, points)   # punti per freccie generiche
    y_points = np.linspace(y_lim_l, y_lim_r, points)
#     z_points = np.linspace(z_lim_l, z_lim_r, points)

    X_points, Y_points = np.meshgrid(x_points, y_points)  
    X_grid = X_points[1:points-1, 1:points-1]
    Y_grid = Y_points[1:points-1, 1:points-1]

    points_x = X_grid.flatten()
    points_y = Y_grid.flatten()

    points_z = np.column_stack((points_x, points_y))
    points_z = torch.from_numpy(points_z).float().to(device)

    if cont_type=="LEFT":
        mask_left = dir_cn==0  # prendo solo trial col contesto scelto 
        cont_left = cont_cn[mask_left][0][56//tau + 2][np.newaxis, :]   # ottengo un array (1, c_dim) del contesto scelto
        cont_z = cont_left.repeat(points_z.shape[0], axis=0)
    elif cont_type=="RIGHT":
        mask_right = dir_cn==1  # prendo solo trial col contesto scelto 
        cont_right = cont_cn[mask_right][0][56//tau + 2][np.newaxis, :]   # ottengo un array (1, c_dim) del contesto scelto
        cont_z = cont_right.repeat(points_z.shape[0], axis=0)
    elif cont_type=="STOP":
        cont_stop = cont_cs[0][-1][np.newaxis, :]   # ottengo un array (1, c_dim) del contesto scelto
        cont_z = cont_stop.repeat(points_z.shape[0], axis=0)
    elif cont_type=="preGO":
        cont_stop = cont_cn[0][0][np.newaxis, :]   # ottengo un array (1, c_dim) del contesto scelto
        cont_z = cont_stop.repeat(points_z.shape[0], axis=0)
    
    cont_z = torch.from_numpy(cont_z).float().to(device)

    z_mean, _ = dmm.generation_z(points_z, cont_z)
    shift = z_mean - points_z
    shift = shift.cpu().detach().numpy()#.squeeze(1).cpu().detach().numpy()

    u = shift[:, 0]
    v = shift[:, 1]
    
    #renormalize the vectors
    magnitude = np.sqrt(u**2 + v**2)
    d = 0.01
    u = u*(math.sqrt(d)/magnitude)
    v = v*(math.sqrt(d)/magnitude)
    
    if stop_trial:
        z_trial, SSD, q = random_latent_cs_traj(dmm, data, tau, device)
        x_SSD = z_trial[SSD, 0]
        y_SSD = z_trial[SSD, 1]
    else:
        z_trial, RT, q = random_latent_cn_traj(dmm, data, tau, device)
        x_RT = z_trial[RT, 0]
        y_RT = z_trial[RT, 1]
    print(f"traj n.{q}")
    
    x_true_story = z_trial[:, 0]
    y_true_story = z_trial[:, 1]
    x_start = z_trial[0, 0]
    y_start = z_trial[0, 1]
    x_GO = z_trial[56//tau, 0]
    y_GO = z_trial[56//tau, 1]
    
    x_lim_inf = x_true_story.min() - 0.5
    x_lim_sup = x_true_story.max() + 0.5
    y_lim_inf = y_true_story.min() - 0.5
    y_lim_sup = y_true_story.max() + 0.5

    f, ax = plt.subplots(figsize = (8, 7))

    ax.quiver(points_x, points_y, u, v, color = 'lime', angles='xy', scale_units='xy', scale=0.5, width = 0.003, alpha = 0.5)
    ax.plot(x_true_story, y_true_story, '-', linewidth=2, color='brown', label = "trajectory example")
    n_arrows = 15
    arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
    for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = x_true_story[i+1] - x_true_story[i]
        dy = y_true_story[i+1] - y_true_story[i]
        ax.arrow(x_true_story[i], y_true_story[i], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    ax.scatter(x_GO, y_GO, s = 50, c = 'b', alpha = 1, label = "GO")
    if stop_trial:
        ax.scatter(x_SSD, y_SSD, color='red', marker='x', s=150, linewidth=3, label = "SSD")
        ax.set_title(f'Mean Markov generation field with {cont_type} context, in 2D latent space: test trial n.{q} with SSD = {SSD*5*tau}$ms$')
    else:
        ax.scatter(x_RT, y_RT, s = 50, c = 'r', alpha = 1, label = "RT")
        ax.set_title(f'Mean Markov generation field with {cont_type} context, in 2D latent space: test trial n.{q} with RT = {RT*5*tau}$ms$')
    ax.scatter(x_start, y_start, s = 50, c = 'purple', alpha = 1)
    ax.set_xlim((x_lim_inf, x_lim_sup))
    ax.set_ylim((y_lim_inf, y_lim_sup))
    ax.set_xlabel('first latent dimension')
    ax.set_ylabel('second latent dimension')
    #ax.set_title(f'Mean Markov generation field on 2D latent space: trial n.{q} with RT = {RT}')




def rec_prove(comm_dict, diff_dict):

    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    ar = comm_dict["ar"]

    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    RT = diff_dict["RT"]
    moments = diff_dict["moments"]
    mean_corr = diff_dict["mean_corr"]
    mean_y = diff_dict["mean_y"]

    test_set = data["set_cn_ordRT"]
    test_cont = data["cont_cn_ordRT"]
    test_RT = data["RT_cn_ordRT"]
    steps = test_set.shape[1]

    trial = test_set[test_RT==RT]
    cont_c = test_cont[test_RT==RT]
    if trial.shape[0] == 0:
        print(f"change the trial with RT={RT:d}")
    elif trial.shape[0] != steps:
        trial = trial[0]
        cont_c = cont_c[0]
    else:
        print()
        trial = trial.squeeze(0)
        cont_c = cont_c.squeeze(0)

    MUA_trial = trial
    MUA_true = MUA_trial.mean(1)

    trial = torch.from_numpy(trial).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
    cont_c = torch.from_numpy(cont_c).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)

    y_mean, y_logvar = dmm(trial, cont_c)
    y_inf = dmm.reparameterization(y_mean, y_logvar)
    y_inf = y_inf.cpu().detach().numpy()
    MUA_inf = y_inf.mean(1)
    MUA_inf_std = y_inf.std(1)
    correlation = np.zeros(steps)
    corr_std = np.zeros(steps)

    if mean_corr:
        MUA_std = MUA_inf_std.mean(1)
        for time in range(steps):
            corr = np.array([np.corrcoef(MUA_trial[time], y_inf[time, i])[0, 1] for i in range(y_inf.shape[1])])  
            correlation[time] = np.mean(corr)
            corr_std[time] = np.std(corr)
    else:
        if mean_y:
            MUA_std = MUA_inf_std.mean(1)
        else:
            z, _, _ = dmm.inference(trial, cont_c)
            if ar:
                y_mu, y_logv = dmm.generation_x(z.mean(1), trial.mean(1))
            else:
                y_mu, y_logv = dmm.generation_x(z.mean(1))
            y_infer = dmm.reparameterization(y_mu, y_logv)
            MUA_inf = y_infer.cpu().detach().numpy()

        for time in range(steps):
            correlation[time] = np.corrcoef(MUA_trial[time], MUA_inf[time])[0, 1]
            corr_std[time] = None

    MUA_mean = MUA_inf.mean(1)

    # y_mean=True, mean_corr=False: tanti z, tanti y, media y, singola corr su MUA vero
    # y_mean=True/False, mean_corr=True: tanti z, tanti y, tante corr, media e std corr
    # y_mean=False, mean_corr=False: tanti z, media z, singola y, singola corr

    t = np.linspace(0, 255, 256, dtype = int)        
    t = t[::tau]

    fig, ax = plt.subplots(2, 5, figsize = (18, 8))

    fig.text(0.40, 0.88, 'True MUA', rotation=0, size=20, fontweight='bold')
    fig.text(0.35, 0.46, 'Reconstructed MUA', rotation=0, size=20, fontweight='bold')

    vmin = min(MUA_mean.min(), MUA_trial.min())
    vmax = max(MUA_mean.max(), MUA_trial.max())

    for i in range(5):
        MUA_rec = MUA_trial[moments[i]//(5*tau)]
        MUA_rec = channel2grid(MUA_rec)
        rec_plot = ax[0, i].imshow(MUA_rec, 
                                      aspect='equal',
                                      cmap='viridis',
                                      interpolation='nearest',
                                      vmin=vmin, vmax=vmax)
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        #ax[0, i].grid(True, which='major', color='w', alpha=0.2)
        ax[0, i].set_title(f'{moments[i]}$ms$')

        MUA_inferred = MUA_inf[moments[i]//(5*tau)]
        MUA_inferred = channel2grid(MUA_inferred)
        inf_plot = ax[1, i].imshow(MUA_inferred, 
                                      aspect='equal',
                                      cmap='viridis',
                                      interpolation='nearest',
                                      vmin=vmin, vmax=vmax)
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])
        #ax[1, i].grid(True, which='major', color='w', alpha=0.2)
        ax[1, i].set_title(f'{moments[i]}$ms$')
    plt.colorbar(inf_plot, ax=ax)


    f, ax1 = plt.subplots(figsize = (12, 5))

    ax1.plot(t*5, MUA_true, c ='b', label = 'True mean MUA')
    if mean_y or mean_corr:
        ax1.fill_between(t*5, MUA_mean - MUA_std, MUA_mean + MUA_std, edgecolor = 'none', color = 'grey', alpha = 0.3)
    ax1.plot(t*5, MUA_mean, c ='r',linestyle="--", label = "Reconstructed mean MUA")
    #ax.axvline(teacher*tau*5, color="black",linestyle="--",label=" Simulation start (GO)")
    ax1.set_title(f"mean MUA reconstructed: comparison with its real value, $RT$ = {((RT+56)*5):d}$ms$, mean_corr={mean_corr}, mean_y={mean_y}")
    ax1.set_xlabel('time ($ms$)')
    ax1.set_ylabel('mean MUA activity over channels')

    ax2 = ax1.twinx()

    ax2.errorbar(t*5, correlation, yerr=corr_std, fmt='o', ls='--', ecolor='black', 
                         elinewidth=1, capsize=5, capthick=1)
    ax2.scatter(t*5, correlation, label = "correlations")
    #ax1.axvline(56*5, color="black",linestyle="--",label="GO")
    ax2.set_ylim((0, 1))
    ax2.set_ylabel("Correlation between predicted and true RT")

    # Optionally, if you want to add a legend for both plots
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    

def consistent_sim_stop(comm_dict, diff_dict):
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    c_dim = comm_dict["c_dim"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    RT_detector = comm_dict["RT_detector"]
    move_detector = comm_dict["move_detector"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    sim_start = diff_dict["sim_start"]
    cmap = diff_dict["cmap"]
    chunk_size = diff_dict["chunk_size"]
    #n_ticks = diff_dict["n_ticks"]
    bins = diff_dict["bins"]
    compute = diff_dict["compute"]
    mean_z = diff_dict["mean_z"]
    
    set_ws = data["set_ws_ordSSD"]
    cont_ws = data["cont_ws_ordSSD"]
    RT_ws = data["RT_ws_ordSSD"]
    SSD_ws = data["SSD_ws_ordSSD"]
    dir_ws = data["dir_ws_ordSSD"]
    
    set_cs = data["set_cs_ordSSD"]
    cont_cs = data["cont_cs_ordSSD"]
    SSD_cs = data["SSD_cs_ordSSD"]
    dir_cs = data["dir_cs_ordSSD"]

#     RT_min = RT_ws.min()

    n_cs, steps, features = set_cs.shape
    n_ws, steps, features = set_ws.shape
    
    dir_cs_rep = dir_cs.repeat(n_trials)
    dir_ws_rep = dir_ws.repeat(n_trials)
    dir_stop_single = np.concatenate((dir_cs, dir_ws), axis=0) 
    
    l_perc = (dir_stop_single==0).sum()/len(dir_stop_single)
    r_perc = (dir_stop_single==1).sum()/len(dir_stop_single)
    
    print(f"Il {r_perc*100:.1f}% degli stop trial è right, il {l_perc*100:.1f}% è left")
    
    dir_stop_rep = np.concatenate((dir_cs_rep, dir_ws_rep), axis=0)
    
    if compute:
    
        teacher = sim_start//(5*tau)
        alone = steps - teacher

        cont_cs_rep = torch.from_numpy(cont_cs).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
        cont_ws_rep = torch.from_numpy(cont_ws).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)

        set_cs = torch.from_numpy(set_cs).float().permute(1, 0, 2).to(device)
        cont_cs = torch.from_numpy(cont_cs).float().permute(1, 0, 2).to(device)
        set_ws = torch.from_numpy(set_ws).float().permute(1, 0, 2).to(device)
        cont_ws = torch.from_numpy(cont_ws).float().permute(1, 0, 2).to(device)
        # Output finale
        z_cs = torch.zeros(steps, n_cs*n_trials, z_dim).to(device)
        z_ws = torch.zeros(steps, n_ws*n_trials, z_dim).to(device)

        for start in range(0, n_cs, chunk_size):
            end = min(start + chunk_size, n_cs)
            batch_size = end - start

            # Estrai chunk
            x_chunk = set_cs[:, start:end, :].repeat_interleave(n_trials, dim=1)
            c_chunk = cont_cs[:, start:end, :].repeat_interleave(n_trials, dim=1)

            # Inferenza
            with torch.no_grad():
                z, z_mean, _ = dmm.inference(x_chunk, c_chunk)

            # Inserisci nel buffer
            z_cs[:, start*n_trials:end*n_trials, :] = z

            torch.cuda.empty_cache()


        for start in range(0, n_ws, chunk_size):
            end = min(start + chunk_size, n_ws)
            batch_size = end - start

            # Estrai chunk
            x_chunk = set_ws[:, start:end, :].repeat_interleave(n_trials, dim=1)
            c_chunk = cont_ws[:, start:end, :].repeat_interleave(n_trials, dim=1)
            #c_chunk = cont_ws[:, start:end, :].unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, batch_size*n_trials, c_dim)

            # Inferenza
            with torch.no_grad():
                z, z_mean, _ = dmm.inference(x_chunk, c_chunk)

            # Inserisci nel buffer
            z_ws[:, start*n_trials:end*n_trials, :] = z

            torch.cuda.empty_cache()

        #z_cn, _, _ = dmm.inference(set_cn, cont_c)
        #z_cn = inference_with_trials(dmm, test_set, test_cont, n_trials, z_dim, device, chunk_size=10)
        #z_cn = torch.from_numpy(z_cn).float().to(device).permute(1, 0, 2)
        if mean_z:
            z_cs = z_cs.reshape(steps, n_cs, n_trials, z_dim)
            z_cs = z_cs.mean(2)
            z_ws = z_ws.reshape(steps, n_ws, n_trials, z_dim)
            z_ws = z_ws.mean(2)
            cont_cs_rep = cont_cs
            cont_ws_rep = cont_ws
            dir_cs_rep = dir_cs
            dir_ws_rep = dir_ws
            dir_stop_rep = dir_stop_single

        z_teach_cs = z_cs[:teacher]
        z_teach_ws = z_ws[:teacher]

        for step in range(alone):
            z_mean_gen, z_cov_gen = dmm.generation_z(z_teach_cs[-1].unsqueeze(0), cont_cs_rep[teacher+step].unsqueeze(0))
            z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
            z_teach_cs = torch.cat((z_teach_cs, z_gen), dim=0)

            z_mean_gen, z_cov_gen = dmm.generation_z(z_teach_ws[-1].unsqueeze(0), cont_ws_rep[teacher+step].unsqueeze(0))
            z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
            z_teach_ws = torch.cat((z_teach_ws, z_gen), dim=0)

        z_stop = torch.cat((z_teach_cs, z_teach_ws), dim=1)
        z_stop = z_stop.permute(1, 0, 2)

        move_logit = move_detector(z_stop)
        move_output = binary_output(move_logit)

        mask_move = move_output > 0.5
        z_move = z_stop[mask_move]
        RT_output = RT_detector(z_move)   # [no_peak]
        RT_estimate = prob_to_RT(RT_output, tau)  

        np.savez(comm_dict["saved_path"] + f"/sim_stop_{n_trials}.npz", move_output=move_output, RT_estimate=RT_estimate, mask_move=mask_move)

    else:
        with np.load(comm_dict["saved_path"] + f"/sim_stop_{n_trials}.npz", allow_pickle=True) as loaded_file:
            move_output = loaded_file["move_output"]
            RT_estimate = loaded_file["RT_estimate"]
            mask_move = loaded_file["mask_move"]
            
    dir_move = dir_stop_rep[mask_move]
    dir_stop = dir_stop_rep[~mask_move]

    true_ws_l = ((dir_ws==0).sum())/n_ws   # Number of true wrong stop trials that are left
    true_ws_r = ((dir_ws==1).sum())/n_ws   # Number of true wrong stop trials that are right
    
    true_cs_l = ((dir_cs==0).sum())/n_cs   # Number of true correct stop trials that are left
    true_cs_r = ((dir_cs==1).sum())/n_cs   # Number of true correct stop trials that are right
    
    pred_ws_l = ((dir_move==0).sum())/len(dir_move)  # Number of predicted wrong stop trials that are left
    pred_ws_r = ((dir_move==1).sum())/len(dir_move)  # Number of predicted wrong stop trials that are right
     
    pred_cs_l = ((dir_stop==0).sum())/len(dir_stop)  # Number of predicted correct stop trials that are left
    pred_cs_r = ((dir_stop==1).sum())/len(dir_stop)  # Number of predicted correct stop trials that are left
    
    
    true_l_ws = ((dir_ws_rep==0).sum())/((dir_stop_rep==0).sum())
    true_l_cs = ((dir_cs_rep==0).sum())/((dir_stop_rep==0).sum())
    
    true_r_ws = ((dir_ws_rep==1).sum())/((dir_stop_rep==1).sum())
    true_r_cs = ((dir_cs_rep==1).sum())/((dir_stop_rep==1).sum())
    
    pred_l_ws = ((dir_move==0).sum())/((dir_stop_rep==0).sum())
    pred_l_cs = ((dir_stop==0).sum())/((dir_stop_rep==0).sum())
    
    pred_r_ws = ((dir_move==1).sum())/((dir_stop_rep==1).sum())
    pred_r_cs = ((dir_stop==1).sum())/((dir_stop_rep==1).sum())
    
    true_frac = n_ws/(n_ws + n_cs)
    pred_frac = len(RT_estimate)/len(move_output)
    
    print(f"{true_ws_l*100:.1f}% of true wrong stop trials are left, {true_ws_r*100:.1f}% are right")
    print(f"{pred_ws_l*100:.1f}% of predicted wrong stop trials are left, {pred_ws_r*100:.1f}% are right")
    
    print(f"{true_cs_l*100:.2f}% of true correct stop trials are left, {true_cs_r*100:.1f}% are right")
    print(f"{pred_cs_l*100:.1f}% of true predicted stop trials are left, {pred_cs_r*100:.1f}% are right")
    
    print(f"{true_r_cs*100:.1f}% of true right trials are correct stop, {true_r_ws*100:.1f}% are wrong stop")
    print(f"{true_l_cs*100:.1f}% of true left trials are correct stop, {true_l_ws*100:.1f}% are wrong stop")
    
    print(f"{pred_r_cs*100:.1f}% of predicted right trials are correct stop, {pred_r_ws*100:.1f}% are wrong stop")
    print(f"{pred_l_cs*100:.1f}% of predicted left trials are correct stop, {pred_l_ws*100:.1f}% are wrong stop")
    
    print(f"The true fraction of wrong stop trials is: {true_frac:.3f}")
    print(f"The simulated fraction of wrong stop trials is: {pred_frac:.3f}")
    
    import seaborn as sns
    from scipy.stats import ks_2samp, wasserstein_distance
    
    if not mean_z:
        RT_ws = RT_ws.repeat(n_trials)
    RT_true = (RT_ws+56)*5    # [no_peak.cpu().detach().numpy()]
    RT_pred = RT_estimate*tau*5
    
    statistic, p_value = ks_2samp(RT_true, RT_pred)
    print("\n--- Test di Kolmogorov-Smirnov (K-S) ---")
    print(f"Statistica del test: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    # La funzione in scipy si chiama wasserstein_distance per il caso 1D
    emd = wasserstein_distance(RT_true, RT_pred)
    
    print("\n--- Earth Mover's Distance (EMD) / Wasserstein-1 ---")
    print(f"Distanza: {emd:.4f}")
    print("Interpretazione: Questo valore rappresenta il 'costo' per trasformare la distribuzione generata in quella reale.")
    print(" -> Valori più bassi indicano maggiore somiglianza. Non c'è una soglia fissa, si usa per confrontare modelli.")
    
    num_bins = bins
    min_value = min(RT_true.min(), RT_pred.min())
    max_value = max(RT_true.max(), RT_pred.max())
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))
    ax1.hist(RT_true, bins=bin_edges, density=True, alpha = 0.5, color='skyblue', edgecolor='black', label = "true RT")
    ax1.hist(RT_pred, bins=bin_edges, density=True, alpha = 0.5, color='red', edgecolor='black', label = "simulated RT")
    y_max = int(ax1.get_ylim()[1])
    x_min, x_max = ax1.get_xlim()
    delta_x = x_max - x_min
    ax1.set_xticks([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))])
    ax1.set_xticklabels([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))], fontsize=font_tick)
    ax1.set_yticks([0, y_max//2, y_max])
    ax1.set_yticklabels([0, y_max//2, y_max], fontsize=font_tick)
    # Add labels and title
    ax1.set_xlabel('Reaction Time ($ms$)', fontsize=font_ax)
    ax1.set_ylabel('# of trials', fontsize=font_ax)
    ax1.set_title(f"Histograms of true and simulated (from {sim_start}$ms$) RTs of wrong stop trials")
    ax1.legend(fontsize=font_leg)
    
    #fig, ax = plt.subplots(figsize = fig_size)
    ax2.hist(RT_true, bins=bin_edges, cumulative=True, density=True, alpha = 0.5, color='skyblue', edgecolor='black', label = "true RT")
    ax2.hist(RT_pred, bins=bin_edges, cumulative=True, density=True, alpha = 0.5, color='red', edgecolor='black', label = "simulated RT")
    # Add labels and title
    y_max = int(ax2.get_ylim()[1])
    x_min, x_max = ax2.get_xlim()
    delta_x = x_max - x_min
    ax2.set_xticks([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))])
    ax2.set_xticklabels([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))], fontsize=font_tick)
    ax2.set_yticks([0, y_max//2, y_max])
    ax2.set_yticklabels([0, y_max//2, y_max], fontsize=font_tick)
    ax2.set_xlabel('Reaction Time ($ms$)', fontsize=font_ax)
    ax2.set_ylabel('# of trials', fontsize=font_ax)
    ax2.set_title(f"Cumulative Histograms of true and simulated (from {sim_start}$ms$) RTs of wrong stop trials")
    ax2.legend(fontsize=font_leg)
    plt.show()
    
    RT_true_l = RT_true[dir_ws_rep==0]
    RT_true_r = RT_true[dir_ws_rep==1]
    
    RT_pred_l = RT_pred[dir_move==0]
    RT_pred_r = RT_pred[dir_move==1]
    
    statistic, p_value = ks_2samp(RT_true_l, RT_true_r)
    print("\n--- Test di Kolmogorov-Smirnov (K-S) true right vs left ---")
    print(f"Statistica del test: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    statistic, p_value = ks_2samp(RT_pred_l, RT_pred_r)
    print("\n--- Test di Kolmogorov-Smirnov (K-S) pred right vs left ---")
    print(f"Statistica del test: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))

    ax1.hist(RT_true_l, bins=bin_edges, density=True, alpha = 0.5, color='skyblue', edgecolor='black', label = "true left")
    ax1.hist(RT_true_r, bins=bin_edges, density=True, alpha = 0.5, color='red', edgecolor='black', label = "true right")
    y_max = int(ax1.get_ylim()[1])
    x_min, x_max = ax1.get_xlim()
    delta_x = x_max - x_min
    ax1.set_xticks([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))])
    ax1.set_xticklabels([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))], fontsize=font_tick)
    ax1.set_yticks([0, y_max//2, y_max])
    ax1.set_yticklabels([0, y_max//2, y_max], fontsize=font_tick)
    # Add labels and title
    ax1.set_xlabel('Reaction Time ($ms$)', fontsize=font_ax)
    ax1.set_ylabel('# of trials', fontsize=font_ax)
    ax1.set_title(f"Histograms of true RTs of right and left wrong stop trials")
    ax1.legend(fontsize=font_leg)
    
    ax2.hist(RT_pred_l, bins=bin_edges, density=True, alpha = 0.5, color='skyblue', edgecolor='black', label = "pred left")
    ax2.hist(RT_pred_r, bins=bin_edges, density=True, alpha = 0.5, color='red', edgecolor='black', label = "pred right")
    y_max = int(ax1.get_ylim()[1])
    x_min, x_max = ax1.get_xlim()
    delta_x = x_max - x_min
    ax2.set_xticks([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))])
    ax2.set_xticklabels([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))], fontsize=font_tick)
    ax2.set_yticks([0, y_max//2, y_max])
    ax2.set_yticklabels([0, y_max//2, y_max], fontsize=font_tick)
    # Add labels and title
    ax2.set_xlabel('Reaction Time ($ms$)', fontsize=font_ax)
    ax2.set_ylabel('# of trials', fontsize=font_ax)
    ax2.set_title(f"Histograms of simulated (from {sim_start}$ms$) RTs of right and left wrong stop trials")
    ax2.legend(fontsize=font_leg)
    plt.show()
    
    statistic, p_value = ks_2samp(RT_true_l, RT_pred_l)
    print("\n--- Test di Kolmogorov-Smirnov (K-S) left true vs pred ---")
    print(f"Statistica del test: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    statistic, p_value = ks_2samp(RT_true_r, RT_pred_r)
    print("\n--- Test di Kolmogorov-Smirnov (K-S) right true vs pred ---")
    print(f"Statistica del test: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")


#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))

#     ax1.hist(RT_true_l, bins=bin_edges, density=True, alpha = 0.5, color='skyblue', edgecolor='black', label = "true left")
#     ax1.hist(RT_pred_l, bins=bin_edges, density=True, alpha = 0.5, color='red', edgecolor='black', label = "pred left")
#     y_max = int(ax1.get_ylim()[1])
#     x_min, x_max = ax1.get_xlim()
#     delta_x = x_max - x_min
#     ax1.set_xticks([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))])
#     ax1.set_xticklabels([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))], fontsize=font_tick)
#     ax1.set_yticks([0, y_max//2, y_max])
#     ax1.set_yticklabels([0, y_max//2, y_max], fontsize=font_tick)
#     # Add labels and title
#     ax1.set_xlabel('Reaction Time ($ms$)', fontsize=font_ax)
#     ax1.set_ylabel('# of trials', fontsize=font_ax)
#     ax1.set_title(f"Histograms of true and simulated (from {sim_start}$ms$) RTs of left wrong stop trials")
#     ax1.legend(fontsize=font_leg)

#     ax2.hist(RT_true_r, bins=bin_edges, density=True, alpha = 0.5, color='skyblue', edgecolor='black', label = "true right")
#     ax2.hist(RT_pred_r, bins=bin_edges, density=True, alpha = 0.5, color='red', edgecolor='black', label = "pred right")
#     y_max = int(ax1.get_ylim()[1])
#     x_min, x_max = ax1.get_xlim()
#     delta_x = x_max - x_min
#     ax2.set_xticks([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))])
#     ax2.set_xticklabels([int(x_min+(delta_x//3)), int(x_min+(2*delta_x//3))], fontsize=font_tick)
#     ax2.set_yticks([0, y_max//2, y_max])
#     ax2.set_yticklabels([0, y_max//2, y_max], fontsize=font_tick)
#     # Add labels and title
#     ax2.set_xlabel('Reaction Time ($ms$)', fontsize=font_ax)
#     ax2.set_ylabel('# of trials', fontsize=font_ax)
#     ax2.set_title(f"Histograms of true and simulated (from {sim_start}$ms$) RTs of right wrong stop trials")
#     ax2.legend(fontsize=font_leg)
#     plt.show()

#     # Calcolo dei quartili e mediane
#     def get_percentiles(data):
#         return np.percentile(data, [25, 50, 75])

#     p25_l_true, med_l_true, p75_l_true = get_percentiles(RT_true_l)
#     p25_l_pred, med_l_pred, p75_l_pred = get_percentiles(RT_pred_l)
#     p25_r_true, med_r_true, p75_r_true = get_percentiles(RT_true_r)
#     p25_r_pred, med_r_pred, p75_r_pred = get_percentiles(RT_pred_r)

#     # Figure: 2 righe, 1 colonna, assi X condivisi
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

#     # --- PLOT SINISTRO ---
#     ax1.hist(RT_true_l, bins=bin_edges, density=True, alpha=0.5,
#              color='skyblue', edgecolor='black', label="True Left")
#     ax1.hist(RT_pred_l, bins=bin_edges, density=True, alpha=0.5,
#              color='red', edgecolor='black', label="Pred Left")

#     # Linee verticali: mediana e quartili
#     for val, style, col in [
#         (p25_l_true, '--', 'blue'), (med_l_true, '-', 'blue'), (p75_l_true, '--', 'blue'),
#         (p25_l_pred, '--', 'darkred'), (med_l_pred, '-', 'darkred'), (p75_l_pred, '--', 'darkred')
#     ]:
#         ax1.axvline(val, linestyle=style, color=col, alpha=0.8, lw=1.5)

#     ax1.set_ylabel('Density', fontsize=12)
#     ax1.set_title(f"RT distributions – Left wrong-stop trials (sim start: {sim_start} ms)",
#                   fontsize=13, pad=10)
#     ax1.legend(fontsize=10)
#     ax1.grid(alpha=0.2)

#     # --- PLOT DESTRO ---
#     ax2.hist(RT_true_r, bins=bin_edges, density=True, alpha=0.5,
#              color='skyblue', edgecolor='black', label="True Right")
#     ax2.hist(RT_pred_r, bins=bin_edges, density=True, alpha=0.5,
#              color='red', edgecolor='black', label="Pred Right")

#     # Linee verticali: mediana e quartili
#     for val, style, col in [
#         (p25_r_true, '--', 'blue'), (med_r_true, '-', 'blue'), (p75_r_true, '--', 'blue'),
#         (p25_r_pred, '--', 'darkred'), (med_r_pred, '-', 'darkred'), (p75_r_pred, '--', 'darkred')
#     ]:
#         ax2.axvline(val, linestyle=style, color=col, alpha=0.8, lw=1.5)

#     ax2.set_xlabel('Reaction Time (ms)', fontsize=12)
#     ax2.set_ylabel('Density', fontsize=12)
#     ax2.set_title(f"RT distributions – Right wrong-stop trials (sim start: {sim_start} ms)",
#                   fontsize=13, pad=10)
#     ax2.legend(fontsize=10)
#     ax2.grid(alpha=0.2)

#     # --- Layout e stile ---
#     plt.tight_layout()
#     plt.show()


    # Funzione per calcolare media e quartili
    def get_stats(data):
        return np.mean(data), np.percentile(data, [25, 75])

    mean_l_true, (p25_l_true, p75_l_true) = get_stats(RT_true_l)
    mean_l_pred, (p25_l_pred, p75_l_pred) = get_stats(RT_pred_l)
    mean_r_true, (p25_r_true, p75_r_true) = get_stats(RT_true_r)
    mean_r_pred, (p25_r_pred, p75_r_pred) = get_stats(RT_pred_r)

    # Figure 2x1 con assi X condivisi
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # --- PLOT SINISTRO ---
    ax1.hist(RT_true_l, bins=bin_edges, density=True, alpha=0.5,
             color='skyblue', edgecolor='black', label="True Left")
    ax1.hist(RT_pred_l, bins=bin_edges, density=True, alpha=0.5,
             color='lightcoral', edgecolor='black', label="Pred Left")

    # Linee verticali: quartili e media
    ax1.axvline(p25_l_true, linestyle='--', color='blue', lw=1.5, label='True Q1 (25%)')
    ax1.axvline(mean_l_true, linestyle='-', color='blue', lw=2, label='True Mean')
    ax1.axvline(p75_l_true, linestyle='--', color='blue', lw=1.5, label='True Q3 (75%)')

    ax1.axvline(p25_l_pred, linestyle='--', color='darkred', lw=1.5, label='Pred Q1 (25%)')
    ax1.axvline(mean_l_pred, linestyle='-', color='darkred', lw=2, label='Pred Mean')
    ax1.axvline(p75_l_pred, linestyle='--', color='darkred', lw=1.5, label='Pred Q3 (75%)')

    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f"RT distributions – Left wrong-stop trials (sim start: {sim_start} ms)",
                  fontsize=13, pad=10)
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(alpha=0.2)

    # --- PLOT DESTRO ---
    ax2.hist(RT_true_r, bins=bin_edges, density=True, alpha=0.5,
             color='skyblue', edgecolor='black', label="True Right")
    ax2.hist(RT_pred_r, bins=bin_edges, density=True, alpha=0.5,
             color='lightcoral', edgecolor='black', label="Pred Right")

    # Linee verticali: quartili e media
    ax2.axvline(p25_r_true, linestyle='--', color='blue', lw=1.5, label='True Q1 (25%)')
    ax2.axvline(mean_r_true, linestyle='-', color='blue', lw=2, label='True Mean')
    ax2.axvline(p75_r_true, linestyle='--', color='blue', lw=1.5, label='True Q3 (75%)')

    ax2.axvline(p25_r_pred, linestyle='--', color='darkred', lw=1.5, label='Pred Q1 (25%)')
    ax2.axvline(mean_r_pred, linestyle='-', color='darkred', lw=2, label='Pred Mean')
    ax2.axvline(p75_r_pred, linestyle='--', color='darkred', lw=1.5, label='Pred Q3 (75%)')

    ax2.set_xlabel('Reaction Time (ms)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f"RT distributions – Right wrong-stop trials (sim start: {sim_start} ms)",
                  fontsize=13, pad=10)
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(alpha=0.2)

    # --- Layout finale ---
    plt.tight_layout()
    plt.show()

    



def consistent_sim(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    c_dim = comm_dict["c_dim"]
    tau = comm_dict["tau"]
    RT_detector = comm_dict["RT_detector"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    sim_start = diff_dict["sim_start"]
    cmap = diff_dict["cmap"]
    n_ticks = diff_dict["n_ticks"]
    bins = diff_dict["bins"]
    #mean_y = diff_dict["mean_y"]
    mean_z = diff_dict["mean_z"]
    color_true = diff_dict["color_true"]
    color_pred = diff_dict["color_pred"]
    eps = diff_dict["eps"]
    alpha = diff_dict["alpha"]
    compute = diff_dict["compute"]
    x_ticks_hist = diff_dict["x_ticks_hist"]
    inset_dim = diff_dict["inset_dim"]
    inset_font = diff_dict["inset_font"]
    
    test_set = data["set_cn_ordRT"]
    test_cont = data["cont_cn_ordRT"]
    test_RT = data["RT_cn_ordRT"]
    RT_min = test_RT.min()
    
    n_samples, steps, features = test_set.shape
    
    n_samples, steps, features = test_set.shape
    teacher = sim_start//(5*tau)
    alone = steps - teacher
    
    MUA_true = test_set.mean(2)
    
    if compute:
    
        z_cn, _, _ = infer_latent(dmm, data, device)
        z_cn = torch.from_numpy(z_cn).float().to(device)

        cont_c = torch.from_numpy(test_cont).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)

#         data_set = torch.from_numpy(test_set).float().permute(1, 0, 2).to(device)
#         cont_set = torch.from_numpy(test_cont).float().permute(1, 0, 2).to(device)
#         chunk_size=10
#         # Output finale
#         z_cn = torch.zeros(steps, n_samples*n_trials, z_dim).to(device)

#         for start in range(0, n_samples, chunk_size):
#             end = min(start + chunk_size, n_samples)
#             batch_size = end - start

#             # Estrai chunk
#             x_chunk = data_set[:, start:end, :].repeat_interleave(n_trials, dim=1)
#             c_chunk = cont_set[:, start:end, :].repeat_interleave(n_trials, dim=1)

#             # Inferenza
#             with torch.no_grad():
#                 z, z_mean, _ = dmm.inference(x_chunk, c_chunk)

#             # Media sui trials (dim=2)
#             #z_mean_chunk = z.reshape(steps, batch_size, n_trials, input_dim).mean(2)

#             # Inserisci nel buffer
#             z_cn[:, start*n_trials:end*n_trials, :] = z

#             torch.cuda.empty_cache()

#         if mean_z:
#             z_cn = z_cn.reshape(steps, n_samples, n_trials, z_dim)
#             z_cn = z_cn.mean(2)
#             cont_c = cont_c.reshape(steps, n_samples, n_trials, c_dim)
#             cont_c = cont_c.mean(2)
        z_teach = z_cn[:teacher]
   
        for step in range(alone):
            z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher+step].unsqueeze(0))
            z_gen = dmm.reparameterization_cov(z_mean_gen, eps*z_cov_gen)
            z_teach = torch.cat((z_teach, z_gen), dim=0)

        y_mean, y_logvar = dmm.generation_x(z_teach)
        y_pred = dmm.reparameterization(y_mean, y_logvar)
    #     y_mean, y_logvar = dmm.generation_x(z_teach)
        z_teach = z_teach.permute(1, 0, 2)
        RT_output = RT_detector(z_teach)   # [no_peak]
        RT_estimate = prob_to_RT(RT_output, tau)    

        y_pred = y_pred.permute(1, 0, 2)
        y_pred = y_pred.cpu().detach().numpy()
        MUA_pred = y_pred.mean(2)
        #RT_estimate = np.argmax(MUA_pred[:, ((RT_min+56)//tau):], axis=1)
        RT_est_sort = np.argsort(RT_estimate)
        MUA_pred = MUA_pred[RT_est_sort]
        
        if not mean_z:
            test_RT = test_RT.repeat(n_trials)
        RT_true = test_RT*5    # [no_peak.cpu().detach().numpy()]
        #RT_pred = ((RT_estimate*tau)+56+RT_min)*5
        RT_pred = RT_estimate*tau*5
        
        np.savez(comm_dict["saved_path"] + f"/RT_distribution_{n_trials}.npz", MUA_pred=MUA_pred, RT_true=RT_true, RT_pred=RT_pred)
    else:
        with np.load(comm_dict["saved_path"] + f"/RT_distribution_{n_trials}.npz", allow_pickle=True) as loaded_file:
            MUA_pred = loaded_file["MUA_pred"]
            RT_true = loaded_file["RT_true"]
            RT_pred = loaded_file["RT_pred"]

    vmin = -3#min(MUA_pred.min(), MUA_true.min())
    vmax = 3#max(MUA_pred.max(), MUA_true.max())

    x_positions = np.arange(200//tau) 
    x_labels = x_positions * (tau*5) 

    x_ticks = len(x_positions)
    y_ticks = n_samples

    n_xticks, n_yticks = n_ticks

    fig, ax = plt.subplots()

    #fig.text(0.14, 0.92, 'Comparison of True vs Generated trials ordered w.r.t. RT', rotation=0, size=15, fontweight='bold')

    im1 = ax.imshow(MUA_true[:, 56//tau:], cmap = cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(x_positions[::x_ticks//n_xticks]) 
    ax.set_xticklabels(x_labels[::x_ticks//n_xticks])#, fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks([])#[n_samples//2, n_samples])
    ax.set_yticklabels([])#[n_samples//2, n_samples], fontsize=font_tick)
    ax.set_xlabel('Time from GO signal ($ms$)')#, fontsize=font_tick)
    ax.set_ylabel('Trial #')#, fontsize=font_tick)

    cbar=plt.colorbar(im1, ax=ax)
    cbar.set_ticks([vmin, 0, vmax])
#     cbar.ax.tick_params(labelsize=font_tick)
    
    fig_file = os.path.join(comm_dict["saved_path"], 'MUA_true.png')
    plt.savefig(fig_file)
    
    fig, ax = plt.subplots()
#     ax[0].set_ylabel('real trials ordered by RT', fontsize=font_ax)
    im2 = ax.imshow(MUA_pred[:, 56//tau:], cmap = cmap, aspect='auto', vmin=vmin, vmax=vmax)
    #ax[1].axvline(56//tau, color="g",linestyle="--",label=" Simulation start (GO)")
    ax.set_xticks(x_positions[::x_ticks//n_xticks]) 
    ax.set_xticklabels(x_labels[::x_ticks//n_xticks])#, fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks([])#[n_samples//2, n_samples])
    ax.set_yticklabels([])#[n_samples//2, n_samples], fontsize=font_tick)
    ax.set_xlabel('Time from GO signal ($ms$)')#, fontsize=font_ax)
    ax.set_ylabel('Trial #')#, fontsize=font_tick)

    cbar=plt.colorbar(im1, ax=ax)
    cbar.set_ticks([vmin, 0, vmax])
#     cbar.ax.tick_params(labelsize=font_tick)
    
    fig_file = os.path.join(comm_dict["saved_path"], 'MUA_gen.png')
    plt.savefig(fig_file)

    import seaborn as sns
    from scipy.stats import ks_2samp, wasserstein_distance
    
    statistic, p_value = ks_2samp(RT_true, RT_pred)
    print("\n--- Test di Kolmogorov-Smirnov (K-S) ---")
    print(f"Statistica del test: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    # La funzione in scipy si chiama wasserstein_distance per il caso 1D
    emd = wasserstein_distance(RT_true, RT_pred)
    
    print("\n--- Earth Mover's Distance (EMD) / Wasserstein-1 ---")
    print(f"Distanza: {emd:.4f}")
    print("Interpretazione: Questo valore rappresenta il 'costo' per trasformare la distribuzione generata in quella reale.")
    print(" -> Valori più bassi indicano maggiore somiglianza. Non c'è una soglia fissa, si usa per confrontare modelli.")
    
    num_bins = bins
    min_value = RT_true.min() - 50
    max_value = RT_true.max() + 50
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)
    
    fig, ax = plt.subplots()
    ax.hist(RT_true, bins=bin_edges, density=True, alpha = alpha, color=color_true, edgecolor='none')#, label = "true RT")
    ax.hist(RT_pred, bins=bin_edges, density=True, alpha = alpha, color=color_pred, edgecolor='none')#, label = "simulated RT")
    ax.set_xticks(x_ticks_hist)
    ax.set_xticklabels(x_ticks_hist)#, fontsize=font_tick)
    ax.set_yticks([])
    #ax1.set_yticklabels([0, y_max//2, y_max], fontsize=font_tick)
    # Add labels and title
    ax.set_xlabel('Reaction Time ($ms$)')#, fontsize=font_ax)
    ax.set_ylabel('Counts')#, fontsize=font_ax)
    
    # 2026-02-18-23h02_DKF_b4_3Cz3_w3
    # 2026-02-16-07h47_DKF_b12_3Cz3_w3
    
    with np.load("/raid/home/tubitoal/DMM/saved_model/2026-02-18-23h02_DKF_b4_3Cz3_w3" + f"/RT_distribution_{n_trials}.npz", allow_pickle=True) as loaded_file:
            MUA_pred = loaded_file["MUA_pred"]
            RT_true = loaded_file["RT_true"]
            RT_pred = loaded_file["RT_pred"]
    
    
    statistic, p_value = ks_2samp(RT_true, RT_pred)
    print("\n--- Test di Kolmogorov-Smirnov (K-S) Cornelio ---")
    print(f"Statistica del test: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    # La funzione in scipy si chiama wasserstein_distance per il caso 1D
    emd = wasserstein_distance(RT_true, RT_pred)
    
    print("\n--- Earth Mover's Distance (EMD) / Wasserstein-1 Cornelio ---")
    print(f"Distanza: {emd:.4f}")
    print("Interpretazione: Questo valore rappresenta il 'costo' per trasformare la distribuzione generata in quella reale.")
    print(" -> Valori più bassi indicano maggiore somiglianza. Non c'è una soglia fissa, si usa per confrontare modelli.")
    
    min_value = RT_true.min() - 50
    max_value = RT_true.max() + 50
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)
    
    axins = ax.inset_axes(inset_dim)
    
    axins.hist(RT_true, bins=bin_edges, density=True, alpha = alpha, color=color_true, edgecolor='none')#, label = "true RT")
    axins.hist(RT_pred, bins=bin_edges, density=True, alpha = alpha, color=color_pred, edgecolor='none')#, label = "simulated RT")
    
    axins.set_xticks(x_ticks_hist)
    axins.set_xticklabels(x_ticks_hist)#, fontsize=font_tick)
    axins.set_yticks([])
    #ax1.set_yticklabels([0, y_max//2, y_max], fontsize=font_tick)
    # Add labels and title
    axins.set_xlabel('Reaction Time ($ms$)', fontsize=inset_font)
    axins.set_ylabel('Counts', fontsize=inset_font)
    axins.tick_params(labelsize=inset_font)

    fig_file = os.path.join(comm_dict["saved_path"], 'RT_distribution.png')
    plt.savefig(fig_file)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))
    ax1.hist(RT_true, bins=bin_edges, density=True, alpha = alpha, color=color_true, edgecolor='none', label = "true RT")
    ax1.hist(RT_pred, bins=bin_edges, density=True, alpha = alpha, color=color_pred, edgecolor='none', label = "simulated RT")
    y_max = int(ax1.get_ylim()[1])
    x_min, x_max = ax1.get_xlim()
    delta_x = x_max - x_min
    ax1.set_xticks([800, 1000])
    ax1.set_xticklabels([800, 1000])#, fontsize=font_tick)
    ax1.set_yticks([])
    #ax1.set_yticklabels([0, y_max//2, y_max], fontsize=font_tick)
    # Add labels and title
    ax1.set_xlabel('Reaction Time ($ms$)')#, fontsize=font_ax)
#     ax1.set_ylabel('# of trials', fontsize=font_ax)
    #ax1.set_title(f"Histograms of true and predicted RTs from {sim_start}$ms$")
    #ax1.legend(fontsize=font_leg)
    
    #fig, ax = plt.subplots(figsize = fig_size)
    ax2.hist(RT_true, bins=bin_edges, cumulative=True, density=True, alpha = alpha, color=color_true, edgecolor='none', label = "true RT")
    ax2.hist(RT_pred, bins=bin_edges, cumulative=True, density=True, alpha = alpha, color=color_pred, edgecolor='none', label = "simulated RT")
    # Add labels and title
    y_max = int(ax2.get_ylim()[1])
    x_min, x_max = ax2.get_xlim()
    delta_x = x_max - x_min
    ax2.set_xticks([800, 1000])
    ax2.set_xticklabels([800, 1000])#, fontsize=font_tick)
#     ax2.set_yticks([0, y_max//2, y_max])
#     ax2.set_yticklabels([0, y_max//2, y_max], fontsize=font_tick)
    ax2.set_xlabel('Reaction Time ($ms$)')#, fontsize=font_ax)
#     ax2.set_ylabel('# of trials', fontsize=font_ax)
    #ax2.set_title(f"Cumulative Histograms of true and predicted RTs from {sim_start}$ms$")
    #ax2.legend(fontsize=font_leg)


def latent_diff_RT(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    color = diff_dict["color"]
    RT_array = diff_dict["RT_array"]
    alpha = diff_dict["alpha"]
    bins = diff_dict["bins"]
    axis = diff_dict["axis"]
    """
    cmap = diff_dict["cmap"]
    c_norm = diff_dict["c_norm"]"""
    
    test_set = data["set_cn_ordRT"]
    test_cont = data["cont_cn_ordRT"]
    test_RT = data["RT_cn_ordRT"]
    
    t = np.linspace(0, 255, 256, dtype = int)        
    t = t[::tau]
    l = len(RT_array)
    steps = test_set.shape[1]
    MUA_inf = np.zeros((l, steps, n_trials, 96))
    
    """fig = plt.figure(figsize=(fig_size[0]+0.5, fig_size[1]))
    gs = gridspec.GridSpec(1, 2, width_ratios=[fig_size[0], 0.5])  # Plot più largo, colorbar stretta

    ax = plt.subplot(gs[0])
    cax = plt.subplot(gs[1])"""
    
    fig, ax = plt.subplots(figsize = fig_size)
    ax.set_xlabel("z1", fontsize=font_ax)
    ax.set_ylabel("z2", fontsize=font_ax)
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]], fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]], fontsize=font_tick)
    #ax.set_title(f'Example of ws trajectory')
    
    """h_min = np.zeros(2)
    h_max = np.zeros(2)
    c = np.full((steps-(56//tau))*n_trials, c_norm/n_trials)"""
    
    for i, RT in enumerate(RT_array):       
        trial = test_set[test_RT==RT]
        cont_c = test_cont[test_RT==RT]
        if trial.shape[0] == 0:
            print(f"change the trial with RT={RT:d}")
        elif trial.shape[0] != steps:
            trial = trial[0]
            cont_c = cont_c[0]
        else:
            print()
            trial = trial.squeeze(0)
            cont_c = cont_c.squeeze(0)
        
        trial = torch.from_numpy(trial).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
        cont_c = torch.from_numpy(cont_c).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)

        z, _, _ = dmm.inference(trial, cont_c)
        mean_z = z.mean(1)
    
        z_mean = mean_z[56//tau:].cpu().detach().numpy()
        z = z[56//tau:].cpu().detach().numpy()
        #z = z.reshape(-1, 2)
        
        #x_start = z_mean[0, 0]
        #y_start = z_mean[0, 1]
        x_GO = z_mean[0, axis[0]]
        y_GO = z_mean[0, axis[1]]
        x_RT = z_mean[RT//tau, axis[0]]
        y_RT = z_mean[RT//tau, axis[1]]
        #x_SSD = z_mean[SSD, 0]
        #y_SSD = z_mean[SSD, 1]
    
        z1_edges = np.linspace(z_lims[axis[0], 0], z_lims[axis[0], 1], bins + 1)
        z2_edges = np.linspace(z_lims[axis[1], 0], z_lims[axis[1], 1], bins + 1)

        """hist, z1_edges, z2_edges, binnumber = binned_statistic_2d(
            z[:, 0], z[:, 1], c, 
            statistic='sum', 
            bins=[z1_edges, z2_edges]
        )

        h_min[i] = hist.min()
        h_max[i] = hist.max()
        
        # Create a mesh grid for plotting
        z1, z2 = np.meshgrid(z1_edges[:-1] + np.diff(z1_edges)/2, z2_edges[:-1] + np.diff(z2_edges)/2)

        # Plot the density map
        im[i] = ax.pcolormesh(z1, z2, 1-np.exp(-hist.T), cmap=cmap[i], vmin=0, vmax=1, shading='auto')"""
        ax.plot(z[:, :, axis[0]], z[:, :, axis[1]], c=color[i], alpha=alpha)
        ax.plot(z_mean[:, axis[0]], z_mean[:, axis[1]], c=color[i], linewidth=3)
        # Add arrows along the mean trajectory
        n_arrows = 12
        arrow_indices = np.arange(0, len(z_mean), len(z_mean)//n_arrows)  # Place n_arrows arrows along the path
        for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
            dx = z_mean[i+1, 0] - z_mean[i, 0]
            dy = z_mean[i+1, 1] - z_mean[i, 1]
            ax.arrow(z_mean[i, axis[0]], z_mean[i, axis[1]], dx, dy,
                    head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
        #ax.fill_between(t*5, MUA_mean - MUA_std, MUA_mean + MUA_std, edgecolor = 'none', color = 'grey', alpha = 0.3)
        ax.scatter(x_GO, y_GO, s = 100, c = "black", marker='o', alpha = 1)# label = "GO")
        ax.scatter(x_RT, y_RT, s = 150, c = 'black', marker='*', alpha = 1)#, label = "RT")
        ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
        ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
        #ax.plot(x_SSD, y_SSD, color='red', marker='x', markeredgewidth = 2, markersize = 10, linestyle='None', label = "SSD")
        #ax.scatter(x_SSD, y_SSD, s = 80, c = 'r', alpha = 1, label = "SSD")
        #ax.scatter(x_start, y_start, s = 80, c = 'y', alpha = 1, label = "start")
        """cbar[i] = plt.colorbar(im[i], cax=cax)
        cbar[i].set_ticks([0, 0.5, 1])  # Specify exact tick locations
        #cbar.set_ticklabels([0, 0.5, 1])  # Custom tick labels
        cbar[i].set_label('n. of traj passing from that bin', fontsize=font_ax)
        cbar[i].ax.tick_params(labelsize=font_tick)"""


def enc_dec_show(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    ar = comm_dict["ar"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    cmap = diff_dict["cmap"]
    markersize = diff_dict["markersize"]
    markeredgewidth = diff_dict["markeredgewidth"]
    leg = diff_dict["leg"]
    axis = diff_dict["axis"]
    
    test_set = data["set_ws_ordRT"]
    test_cont = data["cont_ws_ordRT"]
    test_RT = data["RT_ws_ordRT"]
    test_SSD = data["SSD_ws_ordRT"]
    
    steps = test_set.shape[1]
    samples = test_set.shape[0]
    q = random.randint(0, samples - 1)
    trial = test_set[q]
    cont_c = test_cont[q]
    RT_trial = test_RT[q]
    RT = (RT_trial + 56)//tau
    SSD_trial = test_SSD[q]
    SSD = (SSD_trial + 56)//tau
    moments = np.array([2, 56//tau, SSD, RT])*(5*tau)
    l = len(moments)
    
    MUA_trial = trial

    trial = torch.from_numpy(trial).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
    cont_c = torch.from_numpy(cont_c).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)

    # reconstruction        
    z, _, _ = dmm.inference(trial, cont_c)
    mean_z = z.mean(1)
    
    if ar:
        y_mean, y_logvar = dmm.generation_x(mean_z, trial.mean(1))
    else:
        y_mean, y_logvar = dmm.generation_x(mean_z)
    y_pred = dmm.reparameterization(y_mean, y_logvar)
    y_pred = y_pred.cpu().detach().numpy()
    z_mean = mean_z.cpu().detach().numpy()
    z = z.cpu().detach().numpy()
    MUA_inf = y_pred
    
    x_start = z_mean[0, axis[0]]
    y_start = z_mean[0, axis[1]]
    x_GO = z_mean[56//tau, axis[0]]
    y_GO = z_mean[56//tau, axis[1]]
    x_RT = z_mean[RT, axis[0]]
    y_RT = z_mean[RT, axis[1]]
    x_SSD = z_mean[SSD, axis[0]]
    y_SSD = z_mean[SSD, axis[1]]
    
    t = np.linspace(0, 255, 256, dtype = int)        
    t = t[::tau]
    
    vmin = -3 #min(MUA_inf.min(), MUA_trial.min())
    vmax = 3 #max(MUA_inf.max(), MUA_trial.max())

    fig, ax = plt.subplots(1, 4, figsize = (18, 4))
    fig.text(0.36, 0.86, 'Reconstructed MUA', rotation=0, size=20, fontweight='bold')
    
    for i in range(l):
        j = i
        MUA_rec = MUA_trial[moments[j]//(5*tau)]
        MUA_rec = channel2grid(MUA_rec)
        rec_plot = ax[i].imshow(MUA_rec, 
                                      aspect='equal',
                                      cmap=cmap,
                                      interpolation='nearest',
                                      vmin=vmin, vmax=vmax)
        ax[i].plot(0, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
        ax[i].plot(9, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
        ax[i].plot(0, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
        ax[i].plot(9, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].grid(True, which='major', color='w', alpha=0.2)
        ax[i].set_title(f'{moments[j]}$ms$')
    plt.colorbar(rec_plot, ax=ax, fraction=0.025, pad=0.06)
    
    # Adjust spacing for first figure
    plt.subplots_adjust(top=0.75, right=0.85)  # Reduce space above plots
    
    f, ax = plt.subplots(figsize = (8, 8))
    
    ax.plot(z[:, :, axis[0]], z[:, :, axis[1]], c ='grey', alpha=0.05)
    ax.plot(z_mean[:, axis[0]], z_mean[:, axis[1]], c ='brown', linewidth=3)
    # Add arrows along the mean trajectory
    n_arrows = 10
    arrow_indices = np.arange(0, len(z_mean), len(z_mean)//n_arrows)  # Place n_arrows arrows along the path
    for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = z_mean[i+1, axis[0]] - z_mean[i, axis[0]]
        dy = z_mean[i+1, axis[1]] - z_mean[i, axis[1]]
        ax.arrow(z_mean[i, axis[0]], z_mean[i, axis[1]], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    #ax.fill_between(t*5, MUA_mean - MUA_std, MUA_mean + MUA_std, edgecolor = 'none', color = 'grey', alpha = 0.3)
    ax.scatter(x_GO, y_GO, s = 100, c = "black", marker='o', alpha = 1, label = "GO")
    ax.scatter(x_RT, y_RT, s = 150, c = "black", marker='*', alpha = 1, label = "RT")
    ax.plot(x_SSD, y_SSD, color='black', marker='x', markeredgewidth = 3, markersize = 10, linestyle='None', label = "SSD")
    #ax.scatter(x_SSD, y_SSD, s = 80, c = 'r', alpha = 1, label = "SSD")
    #ax.scatter(x_start, y_start, s = 80, c = 'y', alpha = 1, label = "start")
    ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax.set_xlabel("z1", fontsize=font_ax)
    ax.set_ylabel("z2", fontsize=font_ax)
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]], fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]], fontsize=font_tick)
    #ax.set_title(f'Example of ws trajectory')
    if leg:
        ax.legend(loc='best', fontsize=font_leg)
    
    # Adjust spacing for central figure
    plt.subplots_adjust(bottom=0.2, top=0.9)  # Adjust vertical spacing
        
    fig, ax = plt.subplots(1, 4, figsize = (18, 4))
    fig.text(0.40, 0.12, 'Observed MUA', rotation=0, size=20, fontweight='bold')
        
    for i in range(4):
        j = i
        MUA_inferred = MUA_inf[moments[j]//(5*tau)]
        MUA_inferred = channel2grid(MUA_inferred)
        inf_plot = ax[i].imshow(MUA_inferred, 
                                      aspect='equal',
                                      cmap=cmap,
                                      interpolation='nearest',
                                      vmin=vmin, vmax=vmax)
        ax[i].plot(0, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
        ax[i].plot(9, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
        ax[i].plot(0, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
        ax[i].plot(9, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].grid(True, which='major', color='w', alpha=0.2)
        ax[i].set_xlabel(f'{moments[j]}$ms$')
    plt.colorbar(inf_plot, ax=ax, fraction=0.025, pad=0.06)
    
    # Adjust spacing for bottom figure
    plt.subplots_adjust(bottom=0.25, right=0.85)  # Reduce space below plots
    
    plt.show()


def corr_rand_elec(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    c_dim = comm_dict["c_dim"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    mean_y = diff_dict["mean_y"]
    
    set_cn = data["set_cn_ordRT"]
    cont_c = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    
    steps = set_cn.shape[1]
    s = len(RT_cn)
    
    test_cn = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, s*n_trials, 96) 
    cont_c = torch.from_numpy(cont_c).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, s*n_trials, c_dim)

    y_mean, y_logvar = dmm(test_cn, cont_c)
    y_pred = dmm.reparameterization(y_mean, y_logvar)
    y_pred = y_pred.reshape(steps, s, n_trials, 96)
    y_pred = y_pred.cpu().detach().numpy()
    time = np.linspace(0, 63, 64, dtype = int)        
    
    correlations = np.zeros((96, steps))
    corr_stds = np.zeros((96, steps))
    dict_couple = np.zeros(96)
    for channel in range(96):
        q = random.randint(0, 95)
        dict_couple[channel] = q
        el_rec = y_pred[:, :, :, q]
        el_true = set_cn[:, :, channel]
        for t in range(steps):
            act_rec = el_rec[t]
            act_true = el_true[:, t]
            if mean_y:
                correlations[channel, t] = np.corrcoef(act_true, act_rec.mean(1))[0, 1] 
                corr_stds[channel, t] = None
            else:
                corr = np.array([np.corrcoef(act_true, act_rec[:, trial])[0, 1] for trial in range(act_rec.shape[1])])  
                correlations[channel, t] = np.mean(corr)
                corr_stds[channel, t] = np.std(corr)
    
    mean_corr = correlations.mean(0)
    mean_std = corr_stds.mean(0)    
        
    fig, ax = plt.subplots(figsize = (8, 7))
    ax.errorbar(time*(5*tau), mean_corr, yerr=mean_std, fmt='o', ls='--', ecolor='black', 
                         elinewidth=0.5, capsize=2, capthick=1)
    ax.scatter(time*(5*tau), mean_corr, s = 0.2, label = f"correlazione media sugli elettrodi")
    ax.axvline(56*5, color="black",linestyle="--",label="GO")
    ax.set_ylim((0, 1))
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_xlabel("time ($ms$)")
    ax.set_ylabel("Correlation between reconstructed and observed activity")
    ax.set_title(f"Mean ordered electrodes correlation plot")
    ax.legend()
    
    elec_map = channel2grid(dict_couple)
    correlations = channel2grid(correlations.T)
    corr_stds = channel2grid(corr_stds.T)
    
    row = 10
    col = 10
    
    fig, ax = plt.subplots(row, col, figsize = (15, 15))
    for i in range(row):
        for j in range(col):
            ax[i, j].errorbar(time*(5*tau), correlations[:, i, j], yerr=corr_stds[:, i, j], fmt='o', ls='--', ecolor='black', 
                             elinewidth=1, capsize=5, capthick=1)
            ax[i, j].scatter(time*(5*tau), correlations[:, i, j], s=1, label = f"{10*row + col} e {elec_map[i, j]}")
            ax[i, j].axvline(56*5, color="black",linestyle="--",label="GO")
            ax[i, j].set_ylim((0, 1))
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            #ax[i, j].set_xlabel("time ($ms$)")
            #ax[i, j].set_ylabel("Correlation between reconstructed and observed activity")
            #ax[i, j].set_title(f"correlation plot for channel: {channel}")
            ax[i, j].legend()


def corr_rec_elec(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    c_dim = comm_dict["c_dim"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    mean_y = diff_dict["mean_y"]
    markersize = diff_dict["markersize"]
    markeredgewidth = diff_dict["markeredgewidth"]
    sampling = diff_dict["sampling"]
    size = diff_dict["size"]
    
    set_cn = data["set_cn_ordRT"]
    cont_c = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    
    steps = set_cn.shape[1]
    s = len(RT_cn)
    
    test_cn = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, s*n_trials, 96) 
    cont_c = torch.from_numpy(cont_c).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, s*n_trials, c_dim)

    y_mean, y_logvar = dmm(test_cn, cont_c)
    y_pred = dmm.reparameterization(y_mean, y_logvar)
    y_pred = y_pred.reshape(steps, s, n_trials, 96)
    y_pred = y_pred.cpu().detach().numpy()
    time = np.linspace(0, 63, 64, dtype = int)        
    
    correlations = np.zeros((96, steps))
    corr_stds = np.zeros((96, steps))
    for channel in range(96):
        el_rec = y_pred[:, :, :, channel]
        el_true = set_cn[:, :, channel]
        for t in range(steps):
            act_rec = el_rec[t]
            act_true = el_true[:, t]
            if mean_y:
                correlations[channel, t] = np.corrcoef(act_true, act_rec.mean(1))[0, 1] 
                corr_stds[channel, t] = None
            else:
                corr = np.array([np.corrcoef(act_true, act_rec[:, trial])[0, 1] for trial in range(act_rec.shape[1])])  
                correlations[channel, t] = np.mean(corr)
                corr_stds[channel, t] = np.std(corr)
        
    mean_el_corr = correlations.mean(0)
    mean_el_std = corr_stds.mean(0)    
        
    fig, ax = plt.subplots(figsize = (8, 7))
    ax.errorbar(time[::sampling]*(5*tau), mean_el_corr[::sampling], yerr=mean_el_std[::sampling], fmt='o', ls='--', ecolor='black', 
                         elinewidth=0.5, capsize=2, capthick=1)
    ax.scatter(time[::sampling]*(5*tau), mean_el_corr[::sampling], s = size)
    ax.axvline(56*5, color="black",linestyle="--",label="GO")
    ax.set_ylim((0, 1))
    ax.set_xticks(time[::31]*(5*tau)) 
    ax.set_xticklabels(time[::31]*(5*tau), fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([0, 0.5, 1], fontsize=font_tick)
    ax.set_xlabel("time ($ms$)", fontsize=font_ax)
    ax.set_ylabel("Correlation", fontsize=font_ax)
    #ax.set_title(f"Mean ordered electrodes correlation plot")
    ax.legend(fontsize = font_leg)
    
    correlations = channel2grid(correlations.T)
    corr_stds = channel2grid(corr_stds.T)
    
    mean_t_corr = correlations.mean(0)
    mean_t_std = corr_stds.mean(0)
    
    fig, ax = plt.subplots(figsize = (8, 7))
    im = ax.imshow(mean_t_corr, cmap = "Blues", aspect='equal', vmin=0, vmax=1)
    ax.plot(0, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
    ax.plot(9, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
    ax.plot(0, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
    ax.plot(9, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
    ax.set_xticks([])  # Show every 4th tick to avoid overcrowding
    ax.set_yticks([])
    #ax[i].set_xticklabels([])  # Show corresponding labels
    ax.set_title(f"Correlations between true and reconstructed channels over time")
    plt.colorbar(im, ax=ax, shrink = 0.9, aspect = 15)
    
    row = 10
    col = 10
    
    fig, ax = plt.subplots(row, col, figsize = (15, 15))
    for i in range(row):
        for j in range(col):
            ax[i, j].errorbar(time*(5*tau), correlations[:, i, j], yerr=corr_stds[:, i, j], fmt='o', ls='--', ecolor='black', 
                             elinewidth=1, capsize=5, capthick=1)
            ax[i, j].scatter(time*(5*tau), correlations[:, i, j], s=1)#, label = f"canale: {channel}")
            ax[i, j].axvline(56*5, color="black",linestyle="--",label="GO")
            ax[i, j].set_ylim((0, 1))
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            #ax[i, j].set_xlabel("time ($ms$)")
            #ax[i, j].set_ylabel("Correlation between reconstructed and observed activity")
            #ax[i, j].set_title(f"correlation plot for channel: {channel}")
            ax[i, j].legend()


def mean_MUA_trial(data, tau):
    tau = comm_dict["tau"]
    data = diff_dict["data"]
    
    set_cn = data["set_cn_ordRT"]
    set_cs = data["set_cs_ordSSD"]
    set_ws = data["set_ws_ordRT"]
    
    l_cn = set_cn.shape[0]
    l_cs = set_cs.shape[0]
    l_ws = set_ws.shape[0]
    
    q_cn = random.randint(0, l_cn-1)
    q_cs = random.randint(0, l_cs-1)
    q_ws = random.randint(0, l_ws-1)
    
    MUA_cn = set_cn[q_cn].mean(1)
    MUA_cs = set_cs[q_cs].mean(1)
    MUA_ws = set_ws[q_ws].mean(1)
    
    t = np.linspace(0, 255, 256, dtype = int)        
    t = t[::tau]
    
    f, ax = plt.subplots(figsize = (8, 4))
    ax.plot(t*5, MUA_cn, c = 'g', label = 'cn')
    ax.plot(t*5, MUA_cs, c = 'r', label = 'cs')
    ax.plot(t*5, MUA_ws, c = 'orange', label = 'ws')
    ax.axvline(56*5, color="black",linestyle="--",label="GO")
    ax.set_xlabel('time ($ms$)')
    ax.set_ylabel('mean MUA activity over channels')
    ax.legend()


def white_noise_deviation(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    t = comm_dict["t"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]

    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    cmap = diff_dict["cmap"]
    markersize = diff_dict["markersize"]
    markeredgewidth = diff_dict["markeredgewidth"]
    noise_level = diff_dict["noise_level"]
    max_lag = diff_dict["max_lag"]
    leg = diff_dict["leg"]
    axis = diff_dict["axis"]
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    
    steps = set_cn.shape[1]
    n = set_cn.shape[0]
    q = random.randint(0, n - 1)
    trial = set_cn[q]
    cont_c = cont_cn[q]
    RT_trial = RT_cn[q]
    RT = (RT_trial + 56)//tau
    
    white_noise = np.random.normal(0, noise_level, trial.shape)
    noisy_trial = trial + white_noise
    
    
    #samples = test_set.shape[0]
    RT_trial = RT_cn[q]
    RT = (RT_trial + 56)//tau
    
    moments = np.array([2, 56//tau, RT])*(5*tau)
    l = len(moments)
    rand_chann = np.random.choice(np.arange(96), size=3, replace=False)
    
    fig, ax = plt.subplots(1, 3, figsize = (14, 4))
    for i, q in enumerate(rand_chann):
        ax[i].plot(t*5, trial[:, q], 'r', linewidth = "2.5", label = "true")
        ax[i].plot(t*5, noisy_trial[:, q], 'b', ls='-', label = "noisy")
        ax[i].set_xlabel("time from start")
        ax[i].set_ylabel("channel activity")
        ax[i].set_title(f"true channel n.{q}")
        ax[i].set_ylim((-3, 3))
        ax[i].legend(loc='best')
    vmin = -3 #min(MUA_inf.min(), MUA_trial.min())
    vmax = 3 #max(MUA_inf.max(), MUA_trial.max())

    fig, ax = plt.subplots(1, 3, figsize = (14, 4))
    #fig.text(0.36, 0.86, 'Reconstructed MUA', rotation=0, size=20, fontweight='bold')
    
    for i in range(l):
        j = i
        MUA = trial[moments[j]//(5*tau)]
        MUA = channel2grid(MUA)
        true_plot = ax[i].imshow(MUA, 
                                      aspect='equal',
                                      cmap=cmap,
                                      interpolation='nearest',
                                      vmin=vmin, vmax=vmax)
        ax[i].plot(0, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
        ax[i].plot(9, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
        ax[i].plot(0, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
        ax[i].plot(9, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].grid(True, which='major', color='w', alpha=0.2)
        ax[i].set_title(f'{moments[j]}$ms$')
    plt.colorbar(true_plot, ax=ax, fraction=0.025, pad=0.06)
    
    fig, ax = plt.subplots(1, 3, figsize = (14, 4))
    for i in range(l):
        j = i
        MUA_noisy = noisy_trial[moments[j]//(5*tau)]
        MUA_noisy = channel2grid(MUA_noisy)
        noisy_plot = ax[i].imshow(MUA_noisy, 
                                      aspect='equal',
                                      cmap=cmap,
                                      interpolation='nearest',
                                      vmin=vmin, vmax=vmax)
        ax[i].plot(0, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
        ax[i].plot(9, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
        ax[i].plot(0, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
        ax[i].plot(9, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].grid(True, which='major', color='w', alpha=0.2)
        ax[i].set_title(f'{moments[j]}$ms$')
    plt.colorbar(noisy_plot, ax=ax, fraction=0.025, pad=0.06)
    
    
    
    descr = ["traiettoria vera", "traiettoria con rumore"]
    mean_autocorr = np.zeros(max_lag + 1)
    for d, data in enumerate([trial, noisy_trial]):
        for channel in range(96):
            series = data[:, channel]
            # Calcoliamo l'autocorrelazione fino al lag massimo
            autocorr = acf(series, nlags=max_lag, fft=True)
            #all_autocorrs[channel] = autocorr
            mean_autocorr += autocorr

        # Calcoliamo la media su tutte le serie e dimensioni
        mean_autocorr /= 96

        # Visualizziamo l'autocorrelazione media
        plt.figure(figsize=(8, 6))
        plt.stem(range(max_lag + 1), mean_autocorr)
        plt.title(f'Autocorrelazione media in funzione del lag per {descr[d]}')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelazione')
        #plt.grid(True)
        # Aggiungiamo una linea orizzontale a zero
        plt.axhline(y=0, color='r', linestyle='-')
        # Aggiungiamo i limiti di confidenza (circa ±1.96/√n)
        plt.axhline(y=1.96/np.sqrt(64), color='k', linestyle='--')
        plt.axhline(y=-1.96/np.sqrt(64), color='k', linestyle='--')
        plt.show()
    
    #trial = torch.from_numpy(trial).float().to(device).unsqueeze(1).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, n_trials, 96)
    #cont_c = torch.from_numpy(cont_c).float().to(device).unsqueeze(1).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, n_trials, 2)
    #noisy_trial = torch.from_numpy(noisy_trial).float().to(device).unsqueeze(1).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, n_trials, 96)
    
    trial = torch.from_numpy(trial).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
    cont_c = torch.from_numpy(cont_c).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
    noisy_trial = torch.from_numpy(noisy_trial).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
    
    print(trial.shape)
    print(noisy_trial.shape)
    
    z, _, _ = dmm.inference(trial, cont_c)
    z_noisy, _, _ = dmm.inference(noisy_trial, cont_c)
    
    z = z.cpu().detach().numpy()
    z_noisy = z_noisy.cpu().detach().numpy()
    z_mean = z.mean(1)
    z_noisy_mean = z_noisy.mean(1)
    
    x_start = z_mean[0, 0]
    y_start = z_mean[0, 1]
    x_GO = z_mean[56//tau, 0]
    y_GO = z_mean[56//tau, 1]
    x_RT = z_mean[RT, 0]
    y_RT = z_mean[RT, 1]
    
    f, ax = plt.subplots(figsize = fig_size)
    
    ax.plot(z[:, :, 0], z[:, :, 1], c ='grey', alpha=0.05)
    ax.plot(z_mean[:, 0], z_mean[:, 1], c ='brown', linewidth=3, label = "true")
    ax.plot(z_noisy_mean[:, 0], z_noisy_mean[:, 1], c ='r', linewidth=3, label = "noise")
    # Add arrows along the mean trajectory
    n_arrows = 10
    arrow_indices = np.arange(0, len(z_mean), len(z_mean)//n_arrows)  # Place n_arrows arrows along the path
    for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = z_mean[i+1, 0] - z_mean[i, 0]
        dy = z_mean[i+1, 1] - z_mean[i, 1]
        ax.arrow(z_mean[i, 0], z_mean[i, 1], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    #ax.fill_between(t*5, MUA_mean - MUA_std, MUA_mean + MUA_std, edgecolor = 'none', color = 'grey', alpha = 0.3)
    ax.scatter(x_GO, y_GO, s = 100, c = "black", marker='o', alpha = 1)#, label = "GO")
    ax.scatter(x_RT, y_RT, s = 150, c = "black", marker='*', alpha = 1)#, label = "RT")
    #ax.plot(x_SSD, y_SSD, color='black', marker='x', markeredgewidth = 3, markersize = 10, linestyle='None', label = "SSD")
    #ax.scatter(x_SSD, y_SSD, s = 80, c = 'r', alpha = 1, label = "SSD")
    #ax.scatter(x_start, y_start, s = 80, c = 'y', alpha = 1, label = "start")
    ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax.set_xlabel("z1", fontsize=font_ax)
    ax.set_ylabel("z2", fontsize=font_ax)
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]], fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]], fontsize=font_tick)
    ax.set_title(f'Comparison of the same mean latent trajectory with and without noise (module={noise_level}) in the observation')
    if leg:
        ax.legend(loc='best', fontsize=font_leg)

# +
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def compute_stat_prob(comm_dict, diff_dict):
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    
    data = diff_dict["data"]
    points = diff_dict["points"]
    n_neighbours = diff_dict["n_neighbours"]
    cont_type = diff_dict["cont_type"]
    alpha = diff_dict["alpha"]
    axis = diff_dict["axis"]
    #min_points_in_bin = diff_dict["min_points_in_bin"]
    #max_frac_val = diff_dict["max_frac_val"]
    
    cont_cn = data["cont_cn_ordRT"]
    cont_cs = data["cont_cs_ordSSD"]
    dir_cn = data["dir_cn_ordRT"]
    
    x_lim_l = z_lims[axis[0], 0] #-1
    x_lim_r = z_lims[axis[0], 1] # 5
    y_lim_l = z_lims[axis[1], 0] #-4
    y_lim_r = z_lims[axis[1], 1] # 4
    
    # divido lo spazio in celle
    n_cells = points * points
    cell_size_x = (x_lim_r - x_lim_l) / points
    cell_size_y = (y_lim_r - y_lim_l) / points
    
    # Generazione delle coordinate del centro delle celle
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_lim_l + cell_size_x / 2, x_lim_r - cell_size_x / 2, points),
        np.linspace(y_lim_l + cell_size_y / 2, y_lim_r - cell_size_y / 2, points)
    )
    cell_centers = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    # preparo le coordinate dei centri e i contesti, a passare per il modello
    z_centers = torch.tensor(cell_centers, dtype=torch.float32, device=device)
    print(z_centers.shape)
    if cont_type=="LEFT":
        mask_left = dir_cn==0  # prendo solo trial col contesto scelto 
        cont_left = cont_cn[mask_left][0][56//tau + 2][np.newaxis, :]   # ottengo un array (1, c_dim) del contesto scelto
        cont_z = cont_left.repeat(z_centers.shape[0], axis=0)
    elif cont_type=="RIGHT":
        mask_right = dir_cn==1  # prendo solo trial col contesto scelto 
        cont_right = cont_cn[mask_right][0][56//tau + 2][np.newaxis, :]   # ottengo un array (1, c_dim) del contesto scelto
        cont_z = cont_right.repeat(z_centers.shape[0], axis=0)
    elif cont_type=="STOP":
        cont_stop = cont_cs[0][-1][np.newaxis, :]   # ottengo un array (1, c_dim) del contesto scelto
        cont_z = cont_stop.repeat(z_centers.shape[0], axis=0)
    elif cont_type=="preGO":
        cont_stop = cont_cn[0][0][np.newaxis, :]   # ottengo un array (1, c_dim) del contesto scelto
        cont_z = cont_stop.repeat(z_centers.shape[0], axis=0)
        
    cont_z = torch.from_numpy(cont_z).float().to(device)
    
    # calcolo media e covarianza dei punti generati 
    means, covariances = dmm.generation_z(z_centers, cont_z)
    means = means.cpu().detach().numpy()
    covariances = covariances.cpu().detach().numpy()
    inv_covariances = np.linalg.inv(covariances)
    
    # Costruzione della matrice sparsa di transizione
    rows, cols, values = [], [], []
    for idx, (mean, inv_cov) in enumerate(zip(means, inv_covariances)):
        i, j = divmod(idx, points)

        # Probabilità di transizione: distribuzione gaussiana discreta sulle celle vicine
        for di in range(-n_neighbours, n_neighbours+1):
            for dj in range(-n_neighbours, n_neighbours+1):
                ni, nj = i + di, j + dj
                if 0 <= ni < points and 0 <= nj < points:  # Restiamo nei limiti
                    xj, yj = x_lim_l + (ni + 0.5) * cell_size_x, y_lim_l + (nj + 0.5) * cell_size_y
                    diff = np.array([xj - mean[0], yj - mean[1]])
                    prob = np.exp(-0.5 * diff @ inv_cov @ diff) / (2 * np.pi * np.sqrt(np.linalg.det(covariances[idx])))
                    row = i*points + j
                    col = ni*points + nj
                    rows.append(row)
                    cols.append(col)
                    values.append(prob)

    print("Transition matrix calculated")
    # Normalizzazione delle probabilità per ogni riga
    M = sp.csr_matrix((values, (rows, cols)), shape=(n_cells, n_cells))
    M = M.multiply(1 / M.sum(axis=1))  # Normalizzazione per rendere le righe somme a 1

    print("Computing its eigenvector")
    # Calcolo della distribuzione stazionaria (autovettore di M associato a λ=1)
    eigvals, eigvecs = spla.eigs(M.T, k=1, which='LM')
    print(eigvals[0])
    P_stationary = eigvecs[:, 0].real
    P_stationary /= P_stationary.sum()  # Normalizzazione a probabilità
    E = -np.log(P_stationary)

    # Reshape per visualizzazione
    E_grid = E.reshape((points, points))
    P_stationary_grid = P_stationary.reshape((points, points))
    x_values = np.linspace(x_lim_l, x_lim_r, points)
    y_values = np.linspace(y_lim_l, y_lim_r, points)
    X, Y = np.meshgrid(x_values, y_values)

    print("plotting...")
    # Plot della distribuzione stazionaria
    plt.figure(figsize=(fig_size[0]+2, fig_size[1]))
    plt.contourf(X, Y, P_stationary_grid, cmap="viridis")
    plt.colorbar(label="Probabilità stazionaria")
    plt.xlabel('z1', fontsize=font_ax)
    plt.ylabel('z2', fontsize=font_ax)
    plt.xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    plt.ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    plt.xticks(z_ticks[axis[0]], fontsize=font_tick)
    plt.yticks(z_ticks[axis[0]], fontsize=font_tick)
    plt.title("Distribuzione stazionaria della dinamica")
    plt.show()
    
    # Plot della distribuzione stazionaria
    plt.figure(figsize=(fig_size[0]+2, fig_size[1]))
    plt.contourf(X, Y, E_grid, cmap="viridis")
    plt.colorbar(label="- Log Probabilità stazionaria")
    plt.xlabel('z1', fontsize=font_ax)
    plt.ylabel('z2', fontsize=font_ax)
    plt.xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    plt.ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    plt.xticks(z_ticks[axis[0]], fontsize=font_tick)
    plt.yticks(z_ticks[axis[1]], fontsize=font_tick)
    plt.title("-Log della distribuzione stazionaria della dinamica")
    plt.show()


# +
from scipy.interpolate import griddata

def compute_fraction(z_cov):
    eigenvalues = np.linalg.eigvals(z_cov)
    return np.max(eigenvalues) / np.min(eigenvalues)

def compute_shift_std_ratio(shift, z_cov):    
    shift_magnitude = np.linalg.norm(shift, axis=1)
    total_std = np.sqrt(np.sum(np.linalg.eigvals(z_cov), axis=1))
    return shift_magnitude / total_std

def compute_alignement_vec(shift, z_cov):
    max_std_directions = []
    for cov_matrix in z_cov:
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        max_var_idx = np.argmax(eigenvalues)  # Trova l'indice dell'autovalore più grande
        max_std_directions.append(eigenvectors[:, max_var_idx] * np.sqrt(eigenvalues[max_var_idx]))  # Prendi l'autovettore corrispondente

    # Converti la lista in un array numpy
    max_std_directions = np.array(max_std_directions)

    # Passo 1: Normalizzare i vettori (convertirli in vettori unitari)
    norm_shift = shift / np.linalg.norm(shift, axis=1, keepdims=True)
    norm_std = max_std_directions / np.linalg.norm(max_std_directions, axis=1, keepdims=True)
    return norm_shift, norm_std

def plot_contour(xedges, yedges, hist_masked, title, z1_min, z1_max, z2_min, z2_max, font_ax, font_tick):
    plt.clf()
    plt.pcolormesh(xedges, yedges, hist_masked.T, cmap='viridis')
    plt.xlabel('z1', fontsize=font_ax)
    plt.ylabel('z2', fontsize=font_ax)
    plt.title(title)
    plt.xlim(z1_min, z1_max)
    plt.ylim(z2_min, z2_max)
    plt.xticks(z_ticks[axis[0]], fontsize=font_tick)
    plt.yticks(z_ticks[axis[1]], fontsize=font_tick)
    plt.colorbar()


def fraction_contour(comm_dict, diff_dict):

    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    t = comm_dict["t"]
    z_lims = comm_dict["z_lims"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    
    data = diff_dict["data"]
    points = diff_dict["points"]
    n_trials = diff_dict["n_trials"]
    cont_type = diff_dict["cont_type"]
    stop_trial = diff_dict["stop_trial"]
    n_bins = diff_dict["n_bins"]
    alpha = diff_dict["alpha"]
    min_points_in_bin = diff_dict["min_points_in_bin"]
    max_frac_val = diff_dict["max_frac_val"]
    mean_trials = diff_dict["mean_trials"]
    time = diff_dict["time"]
    GO_plot = diff_dict["GO_plot"]
    trial = diff_dict["trial"]
    axis = diff_dict["axis"]
    #vector_density = diff_dict["vector_density"]    
    
    cont_cn = data["cont_cn_ordRT"]
    cont_cs = data["cont_cs_ordSSD"]
    dir_cn = data["dir_cn_ordRT"]
    
    x_lim_l = z_lims[axis[0], 0] - 0.5 #-5.5
    x_lim_r = z_lims[axis[0], 1] + 0.5 #7
    y_lim_l = z_lims[axis[1], 0] - 0.5 #-5
    y_lim_r = z_lims[axis[1], 1] + 0.5 #10
    #z_lim_l = -10
    #z_lim_r = 10
    
    # CREO LA MASCHERA DEL CONTOUR PLOT RELATIVA AI PUNTI DEL DATASET
    
    z_cn, _, _ = infer_latent(dmm, data, device, n_trials=n_trials)
    steps, s, _ = z_cn.shape
    z = z_cn.reshape(-1, z_dim) 
    
    
    RT_cn = data["RT_cn_ordRT"]
    samples = len(RT_cn)
    RT_cn_rep = np.expand_dims(RT_cn, axis=1)
    RT_cn_rep = RT_cn_rep.repeat(n_trials, axis=1).reshape(-1)

    z1_edges = np.linspace(z_lims[axis[0], 0], z_lims[axis[0], 1], n_bins + 1)
    z2_edges = np.linspace(z_lims[axis[1], 0], z_lims[axis[1], 1], n_bins + 1)

    if mean_trials:
        z_cn = z_cn.reshape(steps, samples, n_trials, z_dim)
        z_cn = z_cn.mean(2)
    
    z_GO = z_cn[time//(5*tau)]
    #z_RT = z_cn[(RT_cn + 56)//tau, np.arange(samples)]
    if mean_trials:
        z_RT = z_cn[(RT_cn + 56)//tau, np.arange(samples)]
        color = RT_cn/RT_cn.max()
    else:
        z_RT = z_cn[(RT_cn_rep + 56)//tau, np.arange(s)]
        color = RT_cn_rep/RT_cn_rep.max()
    
    if GO_plot:
        hist_GO, _, _, binnumber = binned_statistic_2d(z_GO[:, 0], z_GO[:, 1], color, statistic='mean', bins=[z1_edges, z2_edges])
        text = 'GO'
    else:
        hist_RT, x_edges, y_edges, binnumber = binned_statistic_2d(z_RT[:, 0], z_RT[:, 1], color, statistic='mean', bins=[z1_edges, z2_edges])
        text = 'RT'

    #print(hist_GO[hist_GO < 1.1])
    #print(hist_GO > 0.9)
    
    plt.figure(figsize=(fig_size[0]+2, fig_size[1]))
    hist, xedges, yedges, _ = plt.hist2d(z[:, 0], z[:, 1], bins=[points, points], range=[[x_lim_l, x_lim_r], [y_lim_l, y_lim_r]])
    x_centers, y_centers = (xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
    
    points_z = torch.tensor(np.column_stack((X.flatten(), Y.flatten())), dtype=torch.float32, device=device)
    cont_z = torch.zeros_like(points_z, device=device)
    if cont_type=="LEFT":
        mask_left = dir_cn==0  # prendo solo trial col contesto scelto 
        cont_left = cont_cn[mask_left][0][56//tau + 2][np.newaxis, :]   # ottengo un array (1, c_dim) del contesto scelto
        cont_z = cont_left.repeat(points_z.shape[0], axis=0)
    elif cont_type=="RIGHT":
        mask_right = dir_cn==1  # prendo solo trial col contesto scelto 
        cont_right = cont_cn[mask_right][0][56//tau + 2][np.newaxis, :]   # ottengo un array (1, c_dim) del contesto scelto
        cont_z = cont_right.repeat(points_z.shape[0], axis=0)
    elif cont_type=="STOP":
        cont_stop = cont_cs[0][-1][np.newaxis, :]   # ottengo un array (1, c_dim) del contesto scelto
        cont_z = cont_stop.repeat(points_z.shape[0], axis=0)
    elif cont_type=="preGO":
        cont_stop = cont_cn[0][0][np.newaxis, :]   # ottengo un array (1, c_dim) del contesto scelto
        cont_z = cont_stop.repeat(points_z.shape[0], axis=0)
    
    cont_z = torch.from_numpy(cont_z).float().to(device)
    z_mean, z_cov = dmm.generation_z(points_z, cont_z)
    shift = (z_mean - points_z).cpu().detach().numpy()
    z_cov = z_cov.cpu().detach().numpy()
    print(z_cov.shape)
    
    
    total_fraction = np.array([compute_fraction(cov) for cov in z_cov]).reshape(X.shape)
    mask = (hist >= min_points_in_bin) & (total_fraction <= max_frac_val)
    
    z, RT, q = random_latent_cn_traj(dmm, data, tau, device)
    
    print(f"traj n.{q}")
    
    x_true_story = z[:, 0]
    y_true_story = z[:, 1]
    x_start = z[0, 0]
    y_start = z[0, 1]
    x_GO = z[56//tau, 0]
    y_GO = z[56//tau, 1]
    x_RT = z[RT, 0]
    y_RT = z[RT, 1]
    
    
    #hist_GO = np.minimum(hist_GO, 0.85)  # Cap at 0.85
    hist_masked = np.ma.masked_where(~mask, hist_GO)
    plt.figure(figsize=(fig_size[0]+2, fig_size[1]))
    plot_contour(xedges, yedges, hist_masked, f'Histogram of RT values of the z_GO', z_lims[axis[0], 0], z_lims[axis[0], 1], z_lims[axis[1], 0], z_lims[axis[1], 1], font_ax, font_tick)
    
    if trial:
        plt.plot(x_true_story, y_true_story, '-', linewidth=2, color='brown', alpha = 0.5)
        n_arrows = 15
        arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
        for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
            dx = x_true_story[i+1] - x_true_story[i]
            dy = y_true_story[i+1] - y_true_story[i]
            plt.arrow(x_true_story[i], y_true_story[i], dx, dy,
                    head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    plt.show()
    """# Plot the density map
    fig = plt.figure(figsize=(fig_size[0]+0.5, fig_size[1]))
    gs = gridspec.GridSpec(1, 2, width_ratios=[fig_size[0], 0.5])  # Plot più largo, colorbar stretta

    ax = plt.subplot(gs[0])
    cax = plt.subplot(gs[1])
    ax.set_xlabel("z1", fontsize=font_ax)
    ax.set_ylabel("z2", fontsize=font_ax)
    ax.set_xticks([-1, 2, 5]) 
    ax.set_xticklabels([-1, 2, 5], fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks([-2, 1, 4])
    ax.set_yticklabels([-2, 1, 4], fontsize=font_tick)
    im=ax.pcolormesh(z1, z2, hist_masked.T, cmap=cmap, shading='auto')
    #ax.set_title(f'Density Plot of {text} points, coloured from low to high RT true')
    if trial:
        ax.plot(x_true_story, y_true_story, '-', linewidth=2, color='brown', alpha = 0.5)
        n_arrows = 15
        arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
        for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
            dx = x_true_story[i+1] - x_true_story[i]
            dy = y_true_story[i+1] - y_true_story[i]
            ax.arrow(x_true_story[i], y_true_story[i], dx, dy,
                    head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
        
        #plt.scatter(x_GO, y_GO, s = 50, c = 'b', alpha = 1, label = "GO")
        #plt.scatter(x_RT, y_RT, s = 50, c = 'r', alpha = 1, label = "RT")
        #plt.scatter(x_start, y_start, s = 50, c = 'y', alpha = 1, label = "start")
        #ax.plot(x_SSD, y_SSD, 'x', c = 'red', marker='x', markeredgewidth = 3, markersize = 15)
        #ax.set_title(f"RT: {SSD:d}")
        #plt.legend(loc="upper right", fontsize=font_leg)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([0, 0.5, 1])  # Specify exact tick locations
    #cbar.set_ticklabels([0, 0.5, 1])  # Custom tick labels
    cbar.set_label('Mean RT/RTmax', fontsize=font_ax)
    cbar.ax.tick_params(labelsize=font_tick)
    #plt.colorbar(im, ax=ax, label='Mean RT/RTmax')
    plt.show()  """
    
    plt.figure(figsize=(fig_size[0]+2, fig_size[1]))
    hist_masked = np.ma.masked_where(~mask, total_fraction)
    plot_contour(xedges, yedges, hist_masked, f'Rapporto tra modulo eigenvalues - {cont_type}', z_lims[axis[0], 0], z_lims[axis[0], 1], z_lims[axis[1], 0], z_lims[axis[1], 1], font_ax, font_tick)
    plt.show()
    
    shift_std_ratio = compute_shift_std_ratio(shift, z_cov).reshape(X.shape)
    hist_masked = np.ma.masked_where(~mask, shift_std_ratio)
    plt.figure(figsize=(fig_size[0]+2, fig_size[1]))
    plot_contour(xedges, yedges, hist_masked, f'Rapporto tra modulo eigenvalues - {cont_type}', z_lims[axis[0], 0], z_lims[axis[0], 1], z_lims[axis[1], 0], z_lims[axis[1], 1], font_ax, font_tick)
    plt.show()
    
    norm_shift, norm_std = compute_alignement_vec(shift, z_cov)
    cosine_similarities = np.sum(norm_shift * norm_std, axis=1)
    alignment_grid = np.absolute(cosine_similarities.reshape(X.shape))
    hist_masked = np.ma.masked_where(~mask, alignment_grid)
    plt.figure(figsize=(fig_size[0]+2, fig_size[1]))
    plot_contour(xedges, yedges, hist_masked, f'Rapporto tra modulo eigenvalues - {cont_type}', z_lims[axis[0], 0], z_lims[axis[0], 1], z_lims[axis[1], 0], z_lims[axis[1], 1], font_ax, font_tick)
    
    vector_density = 100
    vector_grid_points = int(np.sqrt(vector_density))
       
    x_vector_points = np.linspace(x_lim_l, x_lim_r, vector_grid_points)
    y_vector_points = np.linspace(y_lim_l, y_lim_r, vector_grid_points)
    X_vector, Y_vector = np.meshgrid(x_vector_points, y_vector_points)
    vector_positions = np.column_stack((X_vector.flatten(), Y_vector.flatten()))
    original_positions = np.column_stack((X.flatten(), Y.flatten()))
    
    shift_x = griddata(original_positions, norm_shift[:, 0], vector_positions, method='linear')
    shift_y = griddata(original_positions, norm_shift[:, 1], vector_positions, method='linear')
    interp_shift = np.column_stack((shift_x, shift_y))
    
    var_x = griddata(original_positions, norm_std[:, 0], vector_positions, method='linear')
    var_y = griddata(original_positions, norm_std[:, 1], vector_positions, method='linear')
    interp_var = np.column_stack((var_x, var_y))
    
    interp_shift /= np.linalg.norm(interp_shift, axis=1, keepdims=True)
    interp_var /= np.linalg.norm(interp_var, axis=1, keepdims=True)
    
    mask_interp = griddata(original_positions, mask.flatten(), vector_positions, method='nearest')
    valid_vectors = mask_interp.astype(bool)
    
    plt.quiver(vector_positions[valid_vectors, 0], vector_positions[valid_vectors, 1], 
               interp_shift[valid_vectors, 0], interp_shift[valid_vectors, 1], 
               color='red', scale=15, label='Vettore spostamento')
    plt.quiver(vector_positions[valid_vectors, 0], vector_positions[valid_vectors, 1], 
               interp_var[valid_vectors, 0], interp_var[valid_vectors, 1], 
               color='blue', scale=15, label='Direzione max varianza')
    
    plt.legend()
    plt.show()


# -

def worst_RTpred(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    RT_detector = comm_dict["RT_detector"]
    z_dim = comm_dict["z_dim"]
    c_dim = comm_dict["c_dim"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    detector = diff_dict["detector"]
    #trial = diff_dict["trial"]
    leg = diff_dict["leg"]
    sim_start = diff_dict["sim_start"]
    axis = diff_dict["axis"]
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    
    RT_cn = (RT_cn + 56)//tau
    #z_dim = set_cn.shape[2]
    steps = set_cn.shape[1]
    s = set_cn.shape[0]
    teacher = sim_start//(5*tau) 
    alone = steps - teacher
    
    set_cn = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, s*n_trials, 96) 
    cont_cn = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, s*n_trials, c_dim)

    z_cn, _, _ = dmm.inference(set_cn, cont_cn)
    z_teach = z_cn[:teacher]

    for step in range(alone):
        z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont_cn[teacher+step].unsqueeze(0))
        z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
        z_teach = torch.cat((z_teach, z_gen), dim=0)

    z_teach = z_teach.reshape(steps, s, n_trials, z_dim)
    z_teach = z_teach.mean(2)
        
    print(s)
    print(z_teach.shape)
    
    
    if detector:
        RT_output = RT_detector(z_teach.permute(1, 0, 2))
        RT_rec = prob_to_RT(RT_output, tau)
        #if mean_corr:
        #    RT_pred = RT_pred.reshape(s, n_trials)
    else:
        y_mean, y_logvar = dmm.generation_x(z_teach)
        y_pred = dmm.reparameterization(y_mean, y_logvar)
        y_pred = y_pred[teacher:].cpu().detach().numpy()
        MUA_pred = y_pred
        RT_pred = np.argmax(MUA_pred, axis=0) + teacher
        
    z_cn = z_cn.reshape(steps, s, n_trials, 2)
    z_mean = z_cn.mean(2)
    z_mean = z_mean.cpu().detach().numpy()
    
    diff_RT = np.abs(RT_cn - RT_pred)
    
    # Trova gli indici dei 10 valori più grandi
    top_10_indices = np.argsort(diff_RT)[-10:]
    print(top_10_indices)
    print(diff_RT[top_10_indices])
    # Crea una maschera di falsi della stessa dimensione dell'array originale
    worst_mask = np.zeros(diff_RT.shape, dtype=bool)
    
    # Imposta a True gli elementi corrispondenti ai 10 valori più grandi
    worst_mask[top_10_indices] = True
    
    z_worst = z_mean[:, worst_mask]
    z_GO_worst = z_worst[56//tau]
    
    z, RT, q = random_latent_cn_traj(dmm, data, tau, device)
    
    print(f"traj n.{q}")
    
    x_true_story = z[:, 0]
    y_true_story = z[:, 1]
    x_start = z[0, 0]
    y_start = z[0, 1]
    x_GO = z[56//tau, 0]
    y_GO = z[56//tau, 1]
    x_RT = z[RT, 0]
    y_RT = z[RT, 1]
    
    
    f, ax = plt.subplots(figsize = (8, 8))
    
    ax.plot(z_worst[:, :, 0], z_worst[:, :, 1], c ='b', linewidth=2, alpha=0.2)
    ax.scatter(z_GO_worst[:, 0], z_GO_worst[:, 1], c = 'r')
    ax.plot(x_true_story, y_true_story, c ='brown', linewidth=3)
    # Add arrows along the mean trajectory
    n_arrows = 10
    arrow_indices = np.arange(0, len(z), len(z)//n_arrows)  # Place n_arrows arrows along the path
    for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = z[i+1, 0] - z[i, 0]
        dy = z[i+1, 1] - z[i, 1]
        ax.arrow(z[i, 0], z[i, 1], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    #ax.fill_between(t*5, MUA_mean - MUA_std, MUA_mean + MUA_std, edgecolor = 'none', color = 'grey', alpha = 0.3)
    ax.scatter(x_GO, y_GO, s = 100, c = "black", marker='o', alpha = 1, label = "GO")
    ax.scatter(x_RT, y_RT, s = 150, c = "black", marker='*', alpha = 1, label = "RT")
    #ax.plot(x_SSD, y_SSD, color='black', marker='x', markeredgewidth = 3, markersize = 10, linestyle='None', label = "SSD")
    #ax.scatter(x_SSD, y_SSD, s = 80, c = 'r', alpha = 1, label = "SSD")
    #ax.scatter(x_start, y_start, s = 80, c = 'y', alpha = 1, label = "start")
    ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax.set_xlabel("z1", fontsize=font_ax)
    ax.set_ylabel("z2", fontsize=font_ax)
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]], fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]], fontsize=font_tick)
    #ax.set_title(f'Example of ws trajectory')
    if leg:
        ax.legend(loc='best', fontsize=font_leg)

# !pip install statsmodels

# +
from statsmodels.tsa.stattools import acf

# def whiteness_ratio(data, max_lag=50, demeaned=False):
#     """
#     Compute whiteness ratio for model residuals relative to true signal.
#     x_true, x_rec: arrays (n_trials, T, n_channels)
#     max_lag: number of lags for autocorrelation
#     """
#     s, T, D = data.shape

# Mean autocorrelation (excluding lag 0)
def acf_mat(data, max_lag=50, demeaned=False):
    s, T, D = data.shape
    #total = np.zeros(max_lag)
    acf_array = np.zeros((max_lag, D, s))
    for i in range(s):
        for j in range(D):
            trial = data[i, :, j]
            if demeaned:
                trial = trial - trial.mean()
            r = acf(trial, nlags=max_lag, fft=True)
            #total += r[1:]  # skip lag 0
            acf_array[:, j, i] = r[1:]
    return acf_array#, np.abs(total / (s * D))

#     acf_data = mean_abs_acf(x_true)
#     acf_rec = mean_abs_acf(x_rec)
#     acf_resid = mean_abs_acf(residuals)

    #whiteness = 1 - (acf_resid.sum() / acf_data.sum())
#     return acf_data, acf_resid


from scipy.spatial import distance_matrix

def morans_I(matrix, W):
    """
    Calcola Moran's I per una singola matrice 2D.
    matrix: (10, 10)
    W: matrice di pesi spaziali (NxN)
    """
    x = matrix.flatten()
    x_mean = x.mean()
    x_c = x - x_mean
    num = np.dot(x_c, W.dot(x_c))
    den = np.sum(x_c**2)
    return (len(x) / W.sum()) * (num / den)

def make_weight_matrix(grid_size=10, tau_m=None, row_standardize=False, exclude_corners=True):
    """
    Costruisce matrice di adiacenza 8-neighbors per una griglia 10x10.
    """
    coords = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)])
    dist = distance_matrix(coords, coords)
    if tau_m:
        W = np.exp(-dist / tau_m)
        # 4. Moran's I richiede w_ii = 0 (nessuna auto-correlazione)
        np.fill_diagonal(W, 0.0)
    else:
        W = (dist > 0) & (dist <= np.sqrt(2))  # 8-neighbors
        W = W.astype(float)

    if exclude_corners:
        # Imposta a zero righe e colonne dei 4 angoli spenti
        corner_idx = [0, grid_size-1, (grid_size-1)*grid_size, grid_size**2 - 1]
        for idx in corner_idx:
            W[idx, :] = 0
            W[:, idx] = 0
            
    if row_standardize:
        row_sums = W.sum(axis=1)
        # evita divisione per zero
        nz = row_sums != 0
        W[nz, :] = W[nz, :] / row_sums[nz][:, None]
    return W

def spatial_autocorr_residuals(residuals, tau_m=None, row_standardize=False, m_trials=False, m_time=False):
    """
    residuals: array (n_trials, T, 10, 10)
    Ritorna Moran's I medio (± std)
    """
    W = make_weight_matrix(10, tau_m=tau_m, row_standardize=row_standardize, exclude_corners=True)
    if m_trials:
        T, _, _ = residuals.shape
        morans = np.empty(T)
        for t in range(T):
            morans[t] = morans_I(residuals[t], W)
    elif m_time:
        n_trials, _, _ = residuals.shape
        morans = np.empty(n_trials)
        for i in range(n_trials):
            morans[i] = morans_I(residuals[i], W)
    else:
        n_trials, T, _, _ = residuals.shape
        morans = np.empty([n_trials, T])
        for i in range(n_trials):
            for t in range(T):
                morans[i, t] = morans_I(residuals[i, t], W)
                
    return morans


def reconstruct_vae(dmm, z, n_trials=1, mean_z=False):#, ar=False):
    steps, _, z_dim = z.shape
    with torch.inference_mode():
        if mean_z:
            z = z.reshape(steps, -1, n_trials, z_dim)
            z = z.mean(2)
            y_mean, y_logvar = dmm.generation_x(z)
            y_pred = dmm.reparameterization(y_mean, y_logvar)
        else:
            y_mean, y_logvar = dmm.generation_x(z)
            y_pred = dmm.reparameterization(y_mean, y_logvar)

            y_mean = y_mean.reshape(steps, -1, n_trials, 96)
            y_mean = torch.nanmean(y_mean, dim=2)
            y_pred = y_pred.reshape(steps, -1, n_trials, 96)
            y_pred = torch.nanmean(y_pred, dim=2)
        y_mean = y_mean.permute(1, 0, 2).cpu().detach().numpy()
        y_pred = y_pred.permute(1, 0, 2).cpu().detach().numpy()
    return y_mean, y_pred


def test_PCA(train_set, test_set, z_dim):
    
    s, steps, features = test_set.shape
    X_train = train_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
    X_test = test_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
    
    from sklearn.decomposition import PCA

    pca = PCA(n_components=z_dim)
    X_train_pca = pca.fit_transform(X_train)  # fit + transform sul train 
    X_test_pca = pca.transform(X_test)   
    X_test_rec = pca.inverse_transform(X_test_pca)
    test_rec = X_test_rec.reshape(s, steps, features)
    return test_rec


def gen_PCA(comm_dict, train_set, test_set, cont_pca, teacher):
    
    z_dim = comm_dict["z_dim"]
    saved_path = comm_dict["saved_path"]
    device = comm_dict["device"]
    
    from dmm.learning_markov_detector import Learning_markov_detector
    
    markov_filename = "/Markov_dict_PCA"
    checkpoint_markov = torch.load(saved_path + markov_filename, weights_only=False)
    markov_learning_algo = Learning_markov_detector(params=checkpoint_markov["params"])
    markov_detector = markov_learning_algo.load(markov_filename, markov=True).to(device)
    
    non_markov_filename = "/Non_Markov_dict_PCA"
    checkpoint_non_markov = torch.load(saved_path + non_markov_filename, weights_only=False)
    non_markov_learning_algo = Learning_markov_detector(params=checkpoint_non_markov["params"])
    non_markov_detector = non_markov_learning_algo.load(non_markov_filename, markov=False).to(device)
    
    s, steps, features = test_set.shape
    X_train = train_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
    X_test = test_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
    
    from sklearn.decomposition import PCA

    pca = PCA(n_components=z_dim)
    X_train_pca = pca.fit_transform(X_train)  # fit + transform sul train 
    pca_test_flat = pca.transform(X_test)   
    pca_test = pca_test_flat.reshape(s, steps, z_dim)
    non_markov_pca = torch.from_numpy(pca_test[:, :teacher]).float().to(device)
    markov_pca = torch.from_numpy(pca_test[:, :teacher]).float().to(device)
    alone = steps - teacher
    for step in range(alone):
    
        next_non_markov_pca = non_markov_detector(non_markov_pca, cont_pca[:, 1:(teacher + step + 1)])
        non_markov_pca = torch.cat((non_markov_pca, next_non_markov_pca[:, -1].unsqueeze(1)), dim=1)
        
        next_markov_pca = markov_detector(markov_pca[:, -1], cont_pca[:, (teacher + step)])
        markov_pca = torch.cat((markov_pca, next_markov_pca.unsqueeze(1)), dim=1)
        
    non_markov_pca_flat = non_markov_pca.reshape(-1, z_dim)
    markov_pca_flat = markov_pca.reshape(-1, z_dim)
    non_markov_pca_flat = non_markov_pca_flat.cpu().detach().numpy()
    markov_pca_flat = markov_pca_flat.cpu().detach().numpy()
    X_non_markov_rec_flat = pca.inverse_transform(non_markov_pca_flat)
    X_markov_rec_flat = pca.inverse_transform(markov_pca_flat)
    X_non_markov_rec = X_non_markov_rec_flat.reshape(s, steps, features)
    X_markov_rec = X_markov_rec_flat.reshape(s, steps, features)
    return X_non_markov_rec, X_markov_rec




from statsmodels.tsa.arima.model import ARIMA


def preprocess_residuals(residuals):
    """ residuals: (trials, T, features)
        returns flattened zero-mean residual series 
    """
    r = residuals.reshape(-1)
    return r - np.mean(r)

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import bds
from scipy.stats import combine_pvalues
from tqdm import tqdm    # progress bar (optional)

def test_per_trial_channel(residuals, ljung_box_lags=10, demeaned = False, bds_max_dim=4, bds_eps_factors=(0.5,1.0, 1.5)):
    """
    Run Ljung-Box and BDS on each (trial, channel) time series.
    residuals: np.array shape (n_trials, T, n_channels)
    Returns:
      - lb_p_mat: (n_trials, n_channels) p-values
      - bds_p_mat: (n_trials, n_channels) min p-value across eps_factors (or np.nan if failed)
      - summary dict: aggregated numbers (median, mean, fraction>0.05)
    """
    n_trials, T, n_channels = residuals.shape
    lb_p_mat = np.full((n_trials, n_channels), np.nan, dtype=float)
    lb_p_mat_shuf = np.full((n_trials, n_channels), np.nan, dtype=float)
    bds_p_mat = np.full((n_trials, n_channels), np.nan, dtype=float)
    bds_p_mat_shuf = np.full((n_trials, n_channels), np.nan, dtype=float)

    for i in range(n_trials):
        if i%10==0:
            print(i)
        # vectorize per-channel Ljung-Box with loop on channels (T small)
        for c in range(n_channels):
            r = residuals[i, :, c].astype(float)
            if demeaned:
                r = r - np.mean(r)          # zero-mean per time series
            # ensure length > lags
            r_shuf = np.random.permutation(r)
            try:
                lb = acorr_ljungbox(r, lags=ljung_box_lags, return_df=True)
                lb_p_mat[i, c] = float(lb.iloc[-1]['lb_pvalue'])
            except Exception as e:
                lb_p_mat[i, c] = np.nan
            
            try:
                lb_shuf = acorr_ljungbox(r_shuf, lags=ljung_box_lags, return_df=True)
                lb_p_mat_shuf[i, c] = float(lb_shuf.iloc[-1]['lb_pvalue'])
            except Exception as e:
                lb_p_mat_shuf[i, c] = np.nan

            # BDS: try multiple eps factors and take min p-value
            std_r = np.std(r, ddof=1)
            pvals = []
            pvals_shuf = []
            for f in bds_eps_factors:
                eps = f * std_r
                try:
                    _, p = bds(r, max_dim=bds_max_dim, epsilon=eps)
                    pvals.append(np.min(p))   # p is array for dims 2..max_dim
                except Exception:
                    pvals.append(np.nan)
                    
                try:
                    _, p_shuf = bds(r_shuf, max_dim=bds_max_dim, epsilon=eps)
                    pvals_shuf.append(np.min(p_shuf))   # p is array for dims 2..max_dim
                except Exception:
                    pvals_shuf.append(np.nan)
            # choose min non-nan p
            pvals = np.array(pvals)
            pvals_shuf = np.array(pvals_shuf)
            if np.all(np.isnan(pvals)):
                bds_p_mat[i, c] = np.nan
            else:
                bds_p_mat[i, c] = float(np.nanmin(pvals))
                
            if np.all(np.isnan(pvals_shuf)):
                bds_p_mat_shuf[i, c] = np.nan
            else:
                bds_p_mat_shuf[i, c] = float(np.nanmin(pvals_shuf))

    # summary
    def summarize(mat):
        valid = mat[~np.isnan(mat)]
        if valid.size == 0:
            return {'median': np.nan, 'mean': np.nan, 'fraction_gt_0.05': np.nan}
        return {'median': float(np.median(valid)),
                'mean': float(np.mean(valid)),
                'fraction_gt_0.05': float(np.mean(valid > 0.05))}

    summary = {
        'LB': summarize(lb_p_mat),
        'LB_shuf': summarize(lb_p_mat_shuf),
        'BDS': summarize(bds_p_mat),
        'BDS_shuf': summarize(bds_p_mat_shuf)
    }

    # Global combined p-value via Fisher on flattened (ignore NaN)
    lb_flat = lb_p_mat[~np.isnan(lb_p_mat)]
    lb_flat_shuf = lb_p_mat_shuf[~np.isnan(lb_p_mat_shuf)]
    bds_flat = bds_p_mat[~np.isnan(bds_p_mat)]
    bds_flat_shuf = bds_p_mat_shuf[~np.isnan(bds_p_mat_shuf)]
    fisher_lb = combine_pvalues(lb_flat) if lb_flat.size>0 else (np.nan, np.nan)
    fisher_lb_shuf = combine_pvalues(lb_flat_shuf) if lb_flat_shuf.size>0 else (np.nan, np.nan)
    fisher_bds = combine_pvalues(bds_flat) if bds_flat.size>0 else (np.nan, np.nan)
    fisher_bds_shuf = combine_pvalues(bds_flat_shuf) if bds_flat_shuf.size>0 else (np.nan, np.nan)

    summary['fisher_lb'] = {'stat': float(fisher_lb[0]), 'pval': float(fisher_lb[1])}
    summary['fisher_lb_shuf'] = {'stat': float(fisher_lb_shuf[0]), 'pval': float(fisher_lb_shuf[1])}
    summary['fisher_bds'] = {'stat': float(fisher_bds[0]), 'pval': float(fisher_bds[1])}
    summary['fisher_bds_shuf'] = {'stat': float(fisher_bds_shuf[0]), 'pval': float(fisher_bds_shuf[1])}

    return lb_p_mat, bds_p_mat, lb_p_mat_shuf, bds_p_mat_shuf, summary


# def apply_arima(r, arima_order=(5,0,0)):
#     """ Fit ARIMA to flattened residuals and return ARIMA residuals. """
#     #r = preprocess_residuals(residuals)

#     model = ARIMA(r, order=arima_order).fit()
#     r_arima = model.resid

#     return r_arima - np.mean(r_arima)

def fit_arima_on_residuals(residuals, arima_order=(2,0,2)):
    """
    Fit ARIMA per trial x channel.
    residuals: (trials, T, channels)
    Returns arima_residuals: same shape, ARIMA-residuals
    """
    n_trials, T, C = residuals.shape
    arima_res = np.zeros_like(residuals)
    count=0

    for i in range(n_trials):
        print()
        print()
        print()
        print()
        print(i)
        for c in range(C):
            r = residuals[i, :, c].astype(float)

            # ARIMA may fail on constant sequences → catch exception
            try:
                model = ARIMA(r, order=arima_order)
                fit = model.fit()
                arima_res[i, :, c] = fit.resid
            except Exception:
                count += 1
                # fallback: no ARIMA applied
                arima_res[i, :, c] = r - np.mean(r)
    print(f"exception count: {count}")
    return arima_res

def plot_pvalue_histograms(lb_p_mat, bds_p_mat, lb_p_mat_shuf, bds_p_mat_shuf, color, bins=30):
    # Remove NaN values
    lb_vals = lb_p_mat[~np.isnan(lb_p_mat)]
    bds_vals = bds_p_mat[~np.isnan(bds_p_mat)]
    
    lb_vals_shuf = lb_p_mat_shuf[~np.isnan(lb_p_mat_shuf)]
    bds_vals_shuf = bds_p_mat_shuf[~np.isnan(bds_p_mat_shuf)]
    
    from sklearn.metrics import roc_auc_score

    labels_bds = np.concatenate([np.ones(len(bds_vals_shuf)), np.zeros(len(bds_vals))])
    scores_bds = np.concatenate([bds_vals_shuf, bds_vals])

    auroc_bds = roc_auc_score(labels_bds, scores_bds)
    
    labels_lb = np.concatenate([np.ones(len(lb_vals_shuf)), np.zeros(len(lb_vals))])
    scores_lb = np.concatenate([lb_vals_shuf, lb_vals])

    auroc_lb = roc_auc_score(labels_lb, scores_lb)
    
    print(f"AUROC per LB: {auroc_lb:.2f}, per BDS: {auroc_bds:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    axes[0].hist(lb_vals, bins=bins, density=True, edgecolor='black', color=color, alpha=0.4, label="True")
    axes[0].hist(lb_vals_shuf, bins=bins, density=True, edgecolor='black', color="red", alpha=0.4, label="random")
    #axes[0].set_title("Ljung–Box p-values (no NaNs)")
    axes[0].set_xlabel("p-value")
    axes[0].set_ylabel("count")
    #axes[0].legend()       

    axes[1].hist(bds_vals, bins=bins, density=True, edgecolor='black', color=color, alpha=0.4, label="True")
    axes[1].hist(bds_vals_shuf, bins=bins, density=True, edgecolor='black', color="red", alpha=0.4, label="random")
    axes[1].set_title("BDS p-values (no NaNs)")
    axes[1].set_xlabel("p-value"); axes[1].set_ylabel("count")
    axes[1].legend()  

    plt.tight_layout()
    plt.show()
    

def autocorr(comm_dict, diff_dict):
    
    with np.load(comm_dict["saved_path"]+"/data_split.npz", allow_pickle=True) as loaded_file:
            train_set = loaded_file["train_set"]
            vali_set = loaded_file["vali_set"]
            test_set = loaded_file["test_set"]
            train_direction = loaded_file["train_direction"]
            vali_direction = loaded_file["vali_direction"]
            test_direction = loaded_file["test_direction"]
            train_SSD = loaded_file["train_SSD"]
            vali_SSD = loaded_file["vali_SSD"]
            test_SSD = loaded_file["test_SSD"]
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    ar = comm_dict["ar"]
    font_tick = comm_dict["font_tick"]
    font_ax = comm_dict["font_ax"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    max_lag = diff_dict["max_lag"]
    mean_z = diff_dict["mean_z"]
    mean = diff_dict["mean"]
    mean_gen = diff_dict["mean_gen"]
    compute = diff_dict["compute"]
    demeaned = diff_dict["demeaned"]
    post_GO = diff_dict["post_GO"]
    cmap = diff_dict["cmap"]
    bds_max_dim = diff_dict["bds_max_dim"]
    idx_eps = diff_dict["idx_eps"]
    tau_m = diff_dict["tau_m"]
    bins = diff_dict["bins"]
    row_standardize = diff_dict["row_standardize"]
    #bds_max_points = diff_dict["bds_max_points"]
    ljung_box_lags = diff_dict["ljung_box_lags"]
    trial_type = diff_dict["trial_type"]
    color_PCA = diff_dict["color_PCA"]
    color_DMM = diff_dict["color_DMM"]
    teacher = diff_dict["teacher"]
    
    markersize = 15
    markeredgewidth = 3
    
    if mean:
        text = "_mean"
    else:
        text = ""
    
    #if compute:
#     if trial_type == "cn": 
#         test_set = data["set_cn_ordRT"]
#         cont_c = data["cont_cn_ordRT"]
#     else: 
#         test_set = data["set_cs_ordSSD"]
#         cont_c = data["cont_cs_ordSSD"]

    cont_test = one_hot_cont(test_SSD, test_direction, tau)
    test_data = torch.from_numpy(test_set).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1).to(device)
    cont_pca = torch.from_numpy(cont_test).float().to(device)
    cont_test = cont_pca.permute(1, 0, 2).repeat_interleave(n_trials, dim=1)

    s, steps, features = test_set.shape
    #cont_pca = torch.from_numpy(cont_c).float().to(device)
    #cont_c = torch.from_numpy(cont_c).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)

#     if trial_type == "cn":
#         z, _, _ = infer_latent(dmm, data, device, n_trials=n_trials)
#     else:
#         _, _, z = infer_latent(dmm, data, device, n_trials=n_trials)
#     z = torch.from_numpy(z).float().to(device)
    with torch.inference_mode():
        z, _, _ = dmm.inference(test_data, cont_test)
        if mean_z:
            z_meaned = z.reshape(steps, s, n_trials, z_dim)
            z_meaned = z_meaned.mean(2)
            cont_test = cont_pca.permute(1, 0, 2)
        
#     chunk_size = 10
#     # Output finale
#     z_mean_accum = torch.zeros((steps, s, n_trials, z_dim))
#     for start in range(0, s, chunk_size):
#         end = min(start + chunk_size, s)
#         batch_size = end - start

#         # Estrai chunk
#         x_chunk = test_data[:, start:end, :].repeat_interleave(n_trials, dim=1).to(device)
#         c_chunk = cont_test[:, start:end, :].repeat_interleave(n_trials, dim=1).to(device)

#         # Inferenza
#         with torch.no_grad():
#             z, z_mean, _ = dmm.inference(x_chunk, c_chunk)

#         # Media sui trials (dim=2)
#         z_mean_chunk = z.reshape(steps, batch_size, n_trials, z_dim)

#         # Inserisci nel buffer
#         z_mean_accum[:, start:end] = z_mean_chunk

#         torch.cuda.empty_cache()
#     z = z_mean_accum.reshape(steps, s*n_trials, z_dim).to(device)
    
#     cont_test = cont_test.repeat_interleave(n_trials, dim=1)
    
    z_teach = z_meaned[:teacher]
    print(torch.isnan(z_teach).any())
    alone = steps - teacher
    with torch.inference_mode():
        for step in range(alone):
            z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont_test[teacher+step].unsqueeze(0))
            z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
            z_teach = torch.cat((z_teach, z_gen), dim=0)

        print(torch.isnan(z_teach).sum())
        if mean_z:
            mask = torch.isnan(z_teach).sum(axis=(0, 2))<1
            z_masked = z_teach[:, mask]
        else:
            z_reshaped = z_teach.reshape(steps, -1, n_trials, z_dim)
            mask = torch.isnan(z_reshaped).sum(axis=(0, 2, 3))<1
            z_masked = z_reshaped[:, mask]
            z_masked = z_masked.reshape(steps, -1, z_dim)
        print(torch.isnan(z_masked).any())

        y_mean, y_logvar = dmm.generation_x(z_teach)
        y_pred = dmm.reparameterization(y_mean, y_logvar)
        if not mean_z:
            y_mean = y_mean.reshape(steps, -1, n_trials, 96)
            y_mean = y_mean.mean(2)
            y_pred = y_pred.reshape(steps, -1, n_trials, 96)
            y_pred = y_pred.mean(2)
        y_mean = y_mean.permute(1, 0, 2).cpu().detach().numpy()
        y_pred = y_pred.permute(1, 0, 2).cpu().detach().numpy()
    
    print(np.isnan(y_mean).any())
    print(np.isnan(y_pred).any())
    test_set_vae = test_set[mask.cpu().detach().numpy()]
    
    diff_vae_gen_mean_tot = test_set_vae - y_mean
    diff_vae_gen_tot = test_set_vae - y_pred
    
    diff_vae_gen_mean = diff_vae_gen_mean_tot[:, teacher:]
    diff_vae_gen = diff_vae_gen_tot[:, teacher:]
    
    print(diff_vae_gen.shape)
    print(np.isnan(diff_vae_gen).any())
    
    if post_GO:
        z = z[56//tau:]
        test_set = test_set[:, 56//tau:]

    y_mean, y_pred = reconstruct_vae(dmm, z, n_trials=n_trials)#, mean_z=mean_z)#, ar=ar)
    diff_vae_mean_tot = test_set - y_mean
    diff_vae_tot = test_set - y_pred
    diff_vae_mean = diff_vae_mean_tot#[:, teacher:]
    diff_vae = diff_vae_tot#[:, teacher:]

    test_rec = test_PCA(train_set, test_set, z_dim)
    diff_pca = test_set - test_rec

#     non_markov_pca, markov_pca = gen_PCA(comm_dict, train_set, test_set, cont_pca, teacher)
#     diff_pca_non_markov_tot = test_set - non_markov_pca
#     diff_pca_markov_tot = test_set - markov_pca
    
#     diff_pca_non_markov = diff_pca_non_markov_tot[:, teacher:]
#     diff_pca_markov = diff_pca_markov_tot[:, teacher:]
    
    rng = np.random.default_rng(42)
    vec = diff_vae_mean.flatten()
    vec_perm = rng.permutation(vec)
    diff_vae_shuff = vec_perm.reshape(diff_vae_mean.shape)
    
    res_mat_vae = channel2grid(diff_vae_mean)
    res_mat_shuff = channel2grid(diff_vae_shuff)
    res_mat_pca = channel2grid(diff_pca)
    
    res_mat_vae_mtrials = channel2grid(diff_vae_mean.mean(0))
    res_mat_shuff_mtrials = channel2grid(diff_vae_shuff.mean(0))
    res_mat_pca_mtrials = channel2grid(diff_pca.mean(0))
    
    res_mat_vae_mtime = channel2grid(diff_vae_mean.mean(1))
    res_mat_shuff_mtime = channel2grid(diff_vae_shuff.mean(1))
    res_mat_pca_mtime = channel2grid(diff_pca.mean(1))

    all_I_vae = spatial_autocorr_residuals(res_mat_vae, tau_m=tau_m, row_standardize=row_standardize)
    all_I_shuff = spatial_autocorr_residuals(res_mat_shuff, tau_m=tau_m, row_standardize=row_standardize)
    all_I_pca = spatial_autocorr_residuals(res_mat_pca, tau_m=tau_m, row_standardize=row_standardize)

    all_I_vae_mtrials = spatial_autocorr_residuals(res_mat_vae_mtrials, tau_m=tau_m, row_standardize=row_standardize, m_trials=True)
    all_I_shuff_mtrials = spatial_autocorr_residuals(res_mat_shuff_mtrials, tau_m=tau_m, row_standardize=row_standardize, m_trials=True)
    all_I_pca_mtrials = spatial_autocorr_residuals(res_mat_pca_mtrials, tau_m=tau_m, row_standardize=row_standardize, m_trials=True)
    
    all_I_vae_mtime = spatial_autocorr_residuals(res_mat_vae_mtime, tau_m=tau_m, row_standardize=row_standardize, m_time=True)
    all_I_shuff_mtime = spatial_autocorr_residuals(res_mat_shuff_mtime, tau_m=tau_m, row_standardize=row_standardize, m_time=True)
    all_I_pca_mtime = spatial_autocorr_residuals(res_mat_pca_mtime, tau_m=tau_m, row_standardize=row_standardize, m_time=True)
    
    print(f"Moran's I medio per VAE: {all_I_vae.mean():.4f} ± {all_I_vae.std():.4f}")
    print(f"Moran's I medio per VAE: {all_I_shuff.mean():.4f} ± {all_I_shuff.std():.4f}")
    print(f"Moran's I medio per PCA: {all_I_pca.mean():.4f} ± {all_I_pca.std():.4f}")
    
    print(f"Moran's I medio per VAE (residui mediati sui trials): {all_I_vae_mtrials.mean():.4f} ± {all_I_vae_mtrials.std():.4f}")
    print(f"Moran's I medio per VAE (residui mediati sui trials): {all_I_shuff_mtrials.mean():.4f} ± {all_I_shuff_mtrials.std():.4f}")
    print(f"Moran's I medio per PCA (residui mediati sui trials): {all_I_pca_mtrials.mean():.4f} ± {all_I_pca_mtrials.std():.4f}")
    
    print(f"Moran's I medio per VAE (residui mediati sui tempi): {all_I_vae_mtime.mean():.4f} ± {all_I_vae_mtime.std():.4f}")
    print(f"Moran's I medio per VAE (residui mediati sui tempi): {all_I_shuff_mtime.mean():.4f} ± {all_I_shuff_mtime.std():.4f}")
    print(f"Moran's I medio per PCA (residui mediati sui tempi): {all_I_pca_mtime.mean():.4f} ± {all_I_pca_mtime.std():.4f}")
    
    # Visualizziamo il Moran I medio (sui trials) dei residui vae e pca, nel tempo
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(np.arange(all_I_pca.shape[1])*tau*5, all_I_pca.mean(0), color = color_PCA, label = "PCA")
    ax[0].plot(np.arange(all_I_vae.shape[1])*tau*5, all_I_vae.mean(0), color = color_DMM, label = "DMM")
    ax[0].plot(np.arange(all_I_shuff.shape[1])*tau*5, all_I_shuff.mean(0), color = "g", label = "shuff")
    ax[0].set_title('Moran I medio dei residui vae e pca, in funzione del tempo')
    ax[0].set_xlabel('tempo')
    ax[0].set_ylabel('Morans I')
    #plt.ylim(0, 1)
    ax[0].axvline(56*5, color='k', linestyle='--', label = "GO")
    ax[0].legend()
    
    # Visualizziamo il Moran I medio (sui trials) dei residui vae e pca, nel tempo
    ax[1].plot(np.arange(all_I_pca_mtrials.shape[0])*tau*5, all_I_pca_mtrials, color = color_PCA, label = "PCA")
    ax[1].plot(np.arange(all_I_vae_mtrials.shape[0])*tau*5, all_I_vae_mtrials, color = color_DMM, label = "DMM")
    ax[1].plot(np.arange(all_I_shuff_mtrials.shape[0])*tau*5, all_I_shuff_mtrials, color = "g", label = "shuff")
    ax[1].set_title('Moran I medio (sui trials) dei residui vae e pca, in funzione del tempo')
    ax[1].set_xlabel('tempo')
    ax[1].set_ylabel('Morans I')
    #plt.ylim(0, 1)
    ax[1].axvline(56*5, color='k', linestyle='--', label = "GO")
    ax[1].legend()

    plt.show()
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].hist(all_I_pca.mean(1), bins=bins, density=True, edgecolor='black', color=color_PCA, alpha=0.4, label="PCA")
    ax[0].hist(all_I_vae.mean(1), bins=bins, density=True, edgecolor='black', color=color_DMM, alpha=0.4, label="DMM")
    ax[0].hist(all_I_shuff.mean(1), bins=bins, density=True, edgecolor='black', color="g", alpha=0.4, label="shuff")
    ax[0].set_title("mean over time of the Moran's I of the residuals")
    ax[0].set_xlabel("Moran's I"); ax[1].set_ylabel("count")
    ax[0].legend()  
    
    ax[1].hist(all_I_pca_mtime, bins=bins, density=True, edgecolor='black', color=color_PCA, alpha=0.4, label="PCA")
    ax[1].hist(all_I_vae_mtime, bins=bins, density=True, edgecolor='black', color=color_DMM, alpha=0.4, label="DMM")
    ax[1].hist(all_I_shuff_mtime, bins=bins, density=True, edgecolor='black', color="g", alpha=0.4, label="shuff")
    ax[1].set_title("Moran's I of the residuals meaned over time")
    ax[1].set_xlabel("Moran's I"); ax[1].set_ylabel("count")
    ax[1].legend()
    
    plt.show()

    
#     acf_data = acf_mat(test_set, max_lag=max_lag, demeaned=demeaned)
#     acf_resid_pca = acf_mat(diff_pca, max_lag=max_lag, demeaned=demeaned)
#     acf_resid_vae = acf_mat(diff_vae, max_lag=max_lag, demeaned=demeaned)
#     acf_resid_vae_mean = acf_mat(diff_vae_mean, max_lag=max_lag, demeaned=demeaned)
        
#         np.savez(comm_dict["saved_path"] + f"/acf_file.npz", diff_vae=diff_vae, diff_pca=diff_pca,
#                     all_I_vae=all_I_vae, all_I_pca=all_I_pca, acf_data=acf_data, acf_resid_pca=acf_resid_pca, acf_resid_vae=acf_resid_vae)
        
#         print(diff_vae.shape)
    
#     r_vae_gen_arima = fit_arima_on_residuals(diff_vae_gen_mean)
    
#     if compute:
#         r_vae_arima = fit_arima_on_residuals(diff_vae)
#         r_pca_arima = fit_arima_on_residuals(diff_pca)
        
#         np.savez(comm_dict["saved_path"] + f"/arima_fit{text}.npz", r_vae_arima=r_vae_arima, r_pca_arima=r_pca_arima, diff_vae=diff_vae, diff_pca=diff_pca,
#                     all_I_vae=all_I_vae, all_I_pca=all_I_pca, acf_data=acf_data, acf_resid_pca=acf_resid_pca, acf_resid_vae=acf_resid_vae)
                 
#         bds_max_dim_list = [4]
#         bds_eps_factors_list = [(0.5, 1.0, 1.5)]
                 
#         for bds_max_dim in bds_max_dim_list:
#             for idx_eps, bds_eps_factors in enumerate(bds_eps_factors_list):                
#                 # VRNN
#                 lb_vae, bds_vae, lb_vae_shuf, bds_vae_shuf, summary_vae = test_per_trial_channel(diff_vae, ljung_box_lags=ljung_box_lags, demeaned=demeaned, 
#                                                                                                  bds_max_dim=bds_max_dim, bds_eps_factors=bds_eps_factors)
#                 lb_vae_ARIMA, bds_vae_ARIMA, lb_vae_ARIMA_shuf, bds_vae_ARIMA_shuf, summary_vae_ARIMA = test_per_trial_channel(r_vae_arima, ljung_box_lags=ljung_box_lags, demeaned=demeaned, bds_max_dim=bds_max_dim, bds_eps_factors=bds_eps_factors)

#                 # PCA
#                 lb_pca, bds_pca, lb_pca_shuf, bds_pca_shuf, summary_pca = test_per_trial_channel(diff_pca, ljung_box_lags=ljung_box_lags, demeaned=demeaned, 
#                                                                                                  bds_max_dim=bds_max_dim, bds_eps_factors=bds_eps_factors)
#                 lb_pca_ARIMA, bds_pca_ARIMA, lb_pca_ARIMA_shuf, bds_pca_ARIMA_shuf, summary_pca_ARIMA = test_per_trial_channel(r_pca_arima, ljung_box_lags=ljung_box_lags, demeaned=demeaned, bds_max_dim=bds_max_dim, bds_eps_factors=bds_eps_factors)

#                 np.savez(comm_dict["saved_path"] + f"/whiteness_m{bds_max_dim}_eps{idx_eps}{text}.npz", 
#                          lb_vae=lb_vae, bds_vae=bds_vae, summary_vae=summary_vae, lb_vae_shuf=lb_vae_shuf, bds_vae_shuf=bds_vae_shuf, lb_vae_ARIMA=lb_vae_ARIMA,
#                          bds_vae_ARIMA=bds_vae_ARIMA, summary_vae_ARIMA=summary_vae_ARIMA, lb_vae_ARIMA_shuf=lb_vae_ARIMA_shuf, bds_vae_ARIMA_shuf=bds_vae_ARIMA_shuf,
#                          lb_pca=lb_pca, bds_pca=bds_pca, summary_pca=summary_pca, lb_pca_shuf=lb_pca_shuf, bds_pca_shuf=bds_pca_shuf, lb_pca_ARIMA=lb_pca_ARIMA,
#                          bds_pca_ARIMA=bds_pca_ARIMA, summary_pca_ARIMA=summary_pca_ARIMA, lb_pca_ARIMA_shuf=lb_pca_ARIMA_shuf, bds_pca_ARIMA_shuf=bds_pca_ARIMA_shuf)
#     else:        
#         with np.load(comm_dict["saved_path"]+f"/arima_fit{text}.npz", allow_pickle=True) as loaded_file:
#             r_vae_arima = loaded_file["r_vae_arima"]
#             r_pca_arima = loaded_file["r_pca_arima"]
#             r_vae_gen_arima = loaded_file['r_vae_gen_arima']
#             diff_vae = loaded_file["diff_vae"]
#             diff_pca = loaded_file["diff_pca"]
#             all_I_vae = loaded_file["all_I_vae"]
#             all_I_pca = loaded_file["all_I_pca"]
            
#             acf_data = loaded_file["acf_data"]
#             acf_resid_pca = loaded_file["acf_resid_pca"]
#             acf_resid_vae = loaded_file["acf_resid_vae"]
            
#         with np.load(comm_dict["saved_path"]+f"/whiteness_m{bds_max_dim}_eps{idx_eps}{text}.npz", allow_pickle=True) as loaded_file:
#             lb_vae = loaded_file["lb_vae"]
#             bds_vae = loaded_file["bds_vae"]
#             lb_vae_shuf = loaded_file["lb_vae_shuf"]
#             bds_vae_shuf = loaded_file["bds_vae_shuf"]
#             summary_vae = loaded_file["summary_vae"]
            
#             lb_vae_ARIMA = loaded_file["lb_vae_ARIMA"]
#             bds_vae_ARIMA = loaded_file["bds_vae_ARIMA"]
#             lb_vae_ARIMA_shuf = loaded_file["lb_vae_ARIMA_shuf"]
#             bds_vae_ARIMA_shuf = loaded_file["bds_vae_ARIMA_shuf"]
#             summary_vae_ARIMA = loaded_file["summary_vae_ARIMA"]
            
#             lb_pca = loaded_file["lb_pca"]
#             bds_pca = loaded_file["bds_pca"]
#             lb_pca_shuf = loaded_file["lb_pca_shuf"]
#             bds_pca_shuf = loaded_file["bds_pca_shuf"]
#             summary_pca = loaded_file["summary_pca"]
            
#             lb_pca_ARIMA = loaded_file["lb_pca_ARIMA"]
#             bds_pca_ARIMA = loaded_file["bds_pca_ARIMA"]
#             lb_pca_ARIMA_shuf = loaded_file["lb_pca_ARIMA_shuf"]
#             bds_pca_ARIMA_shuf = loaded_file["bds_pca_ARIMA_shuf"]
#             summary_pca_ARIMA = loaded_file["summary_pca_ARIMA"]
            
    
#     # 1. Carica il file NPZ esistente
#     loaded = np.load(comm_dict["saved_path"] + f"/whiteness_m{bds_max_dim}_eps0{text}.npz", allow_pickle=True)
#     # 2. Converti in dizionario e aggiungi i nuovi array
#     data_dict = dict(loaded)
    
#     # GEN VRNN
#     bds_max_dim = 4
#     bds_eps_factors = (0.5, 1.0, 1.5)
#     lb_vae_gen, bds_vae_gen, lb_vae_gen_shuf, bds_vae_gen_shuf, summary_vae_gen = test_per_trial_channel(diff_vae_gen_mean, ljung_box_lags=ljung_box_lags, demeaned=demeaned, bds_max_dim=bds_max_dim, bds_eps_factors=bds_eps_factors)
#     lb_vae_gen_ARIMA, bds_vae_gen_ARIMA, lb_vae_gen_ARIMA_shuf, bds_vae_gen_ARIMA_shuf, summary_vae_gen_ARIMA = test_per_trial_channel(r_vae_gen_arima, ljung_box_lags=ljung_box_lags, demeaned=demeaned, bds_max_dim=bds_max_dim, bds_eps_factors=bds_eps_factors)
#     # 3. Aggiungi i nuovi array al dizionario
#     data_dict['lb_vae_gen'] = lb_vae_gen
#     data_dict['bds_vae_gen'] = bds_vae_gen
#     data_dict['lb_vae_gen_shuf'] = lb_vae_gen_shuf
#     data_dict['bds_vae_gen_shuf'] = bds_vae_gen_shuf
#     data_dict['summary_vae_gen'] = summary_vae_gen
#     data_dict['lb_vae_gen_ARIMA'] = lb_vae_gen_ARIMA
#     data_dict['bds_vae_gen_ARIMA'] = bds_vae_gen_ARIMA
#     data_dict['lb_vae_gen_ARIMA_shuf'] = lb_vae_gen_ARIMA_shuf
#     data_dict['bds_vae_gen_ARIMA_shuf'] = bds_vae_gen_ARIMA_shuf
#     data_dict['summary_vae_gen_ARIMA'] = summary_vae_gen_ARIMA
#     # 4. Salva tutto insieme (vecchi + nuovi array)
#     np.savez(comm_dict["saved_path"] + f"/whiteness_m{bds_max_dim}_eps{idx_eps}{text}.npz", **data_dict)
#     # 5. IMPORTANTE: chiudi il file caricato
#     loaded.close()



            
#     acf_ARIMA_pca = acf_mat(r_pca_arima, max_lag=max_lag, demeaned=demeaned)
#     acf_ARIMA_vae = acf_mat(r_vae_arima, max_lag=max_lag, demeaned=demeaned)
#     acf_gen_vae = acf_mat(diff_vae_gen, max_lag=max_lag, demeaned=demeaned)
#     acf_gen_vae_mean = acf_mat(diff_vae_gen_mean, max_lag=max_lag, demeaned=demeaned)
#     acf_pca_non_markov = acf_mat(diff_pca_non_markov, max_lag=max_lag, demeaned=demeaned)
#     acf_pca_markov = acf_mat(diff_pca_markov, max_lag=max_lag, demeaned=demeaned)
    
#     print(acf_gen_vae.shape)
#     print(np.isnan(acf_gen_vae).any())
    
#     print(f"Moran's I medio per VAE: {all_I_vae.mean():.4f} ± {all_I_vae.std():.4f}")
#     print(f"Moran's I medio per PCA: {all_I_pca.mean():.4f} ± {all_I_pca.std():.4f}")
    
#     print(f"VAE residuals: mean={diff_vae.mean()}, std={diff_vae.std()}")
#     print(f"PCA residuals: mean={diff_pca.mean()}, std={diff_pca.std()}")
    
#     acf_data_channels = acf_data.mean(2)
#     acf_pca_channels = acf_resid_pca.mean(2)
#     acf_vae_channels = acf_resid_vae.mean(2)
#     acf_vae_mean_channels = acf_resid_vae_mean.mean(2)
    
#     acf_data_tot = acf_data_channels.mean(1)
#     acf_pca_tot = acf_pca_channels.mean(1)
#     acf_inf_vae_tot = acf_vae_channels.mean(1)
#     acf_inf_vae_mean_tot = acf_vae_mean_channels.mean(1)
    
#     acf_ARIMA_pca_chann = acf_ARIMA_pca.mean(2)
#     acf_ARIMA_vae_chann = acf_ARIMA_vae.mean(2)
#     acf_gen_vae_mean_chann = acf_gen_vae_mean.mean(2)
#     acf_gen_vae_chann = acf_gen_vae.mean(2)
#     acf_pca_non_markov_chann = acf_pca_non_markov.mean(2)
#     acf_pca_markov_chann = acf_pca_markov.mean(2)
    
#     acf_ARIMA_pca_tot = acf_ARIMA_pca_chann.mean(1)
#     acf_ARIMA_vae_tot = acf_ARIMA_vae_chann.mean(1)
#     acf_gen_vae_mean_tot = acf_gen_vae_mean_chann.mean(1)
#     acf_gen_vae_tot = acf_gen_vae_chann.mean(1)
#     acf_pca_non_markov_tot = acf_pca_non_markov_chann.mean(1)
#     acf_pca_markov_tot = acf_pca_markov_chann.mean(1)
    
#     whiteness_ch_pca = 1 - (acf_pca_channels.sum() / acf_data_channels.sum())
#     whiteness_ch_vae = 1 - (acf_vae_channels.sum() / acf_data_channels.sum())
    
#     whiteness_tot_pca = 1 - (acf_pca_tot.sum() / acf_data_tot.sum())
#     whiteness_tot_vae = 1 - (acf_inf_vae_tot.sum() / acf_data_tot.sum())
    
#     print(f"Whiteness PCA: {whiteness_tot_pca:.3f} ({whiteness_ch_pca:.3f}),  Whiteness VAE: {whiteness_tot_vae:.3f} ({whiteness_ch_vae:.3f})")
        
#     print("PCA")
#     print(summary_pca)
#     print("VAE")
#     print(summary_vae)
#     print("VAE gen")
#     print(summary_vae_gen)
#     print("PCA and ARIMA")
#     print(summary_pca_ARIMA)
#     print("VAE and ARIMA")
#     print(summary_vae_ARIMA)
#     print("VAE gen and ARIMA")
#     print(summary_vae_gen_ARIMA)
    
#     plot_pvalue_histograms(lb_pca, bds_pca, lb_pca_shuf, bds_pca_shuf, color=color_PCA)
#     plot_pvalue_histograms(lb_vae, bds_vae, lb_vae_shuf, bds_vae_shuf, color=color_DMM)
#     #plot_pvalue_histograms(lb_vae_gen, bds_vae_gen, lb_vae_gen_shuf, bds_vae_gen_shuf, color=color_DMM)
#     plot_pvalue_histograms(lb_pca_ARIMA, bds_pca_ARIMA, lb_pca_ARIMA_shuf, bds_pca_ARIMA_shuf, color=color_PCA)
#     plot_pvalue_histograms(lb_vae_ARIMA, bds_vae_ARIMA, lb_vae_ARIMA_shuf, bds_vae_ARIMA_shuf, color=color_DMM)
#     #plot_pvalue_histograms(lb_vae_gen_ARIMA, bds_vae_gen_ARIMA, lb_vae_gen_ARIMA_shuf, bds_vae_gen_ARIMA_shuf, color=color_DMM)


#     # Visualizziamo l'autocorrelazione media del dato osservato
#     plt.figure(figsize=(10, 6))
#     plt.stem(range(max_lag), acf_data)
#     plt.title(f'Autocorrelazione media in funzione del lag per dato osservato')
#     plt.xlabel('Lag')
#     plt.ylabel('Autocorrelazione')
#     plt.ylim(0, 1)
#     #plt.grid(True)
#     # Aggiungiamo una linea orizzontale a zero
#     #plt.axhline(y=0, color='r', linestyle='-')
#     # Aggiungiamo i limiti di confidenza (circa ±1.96/√n)
#     plt.axhline(y=1.96/np.sqrt(256//tau), color='k', linestyle='--')
#     #plt.axhline(y=-1.96/np.sqrt(256//tau), color='k', linestyle='--')
#     plt.show()


    # Visualizziamo l'autocorrelazione media del dato ricostruito con pca e vae
#     plt.figure(figsize=(10, 6))
#     plt.stem(range(max_lag), acf_rec_pca, linefmt='C0-', markerfmt='C0o', basefmt='k-', label = "PCA")
#     plt.stem(range(max_lag), acf_rec_vae, linefmt='C1-', markerfmt='C1s', basefmt='k-', label = "VAE")
#     plt.title(f'Autocorrelazione media in funzione del lag per dato ricostruito')
#     plt.xlabel('Lag')
#     plt.ylabel('Autocorrelazione')
#     plt.ylim(0, 1)
#     #plt.grid(True)
#     # Aggiungiamo una linea orizzontale a zero
#     #plt.axhline(y=0, color='r', linestyle='-')
#     # Aggiungiamo i limiti di confidenza (circa ±1.96/√n)
#     plt.axhline(y=1.96/np.sqrt(256//tau), color='k', linestyle='--')
#     #plt.axhline(y=-1.96/np.sqrt(256//tau), color='k', linestyle='--')
#     plt.legend()
#     plt.show()
#     acf_resid_pca = np.insert(acf_resid_pca, 0, 1)
#     acf_resid_vae = np.insert(acf_resid_vae, 0, 1)


#     k = math.ceil(math.sqrt(max_lag))
#     # Plot the density map
#     fig, ax = plt.subplots(k, k, figsize=(16, 15))
    
#     vmin = acf_channels_pca.min()
#     vmax = acf_channels_pca.max()

    
#     acf_ch_pca = channel2grid(acf_channels_pca)
#     for i in range(k):
#         for j in range(k):
#             if k*i + j >= max_lag:
#                 continue
#             shift_plot_pca = ax[i, j].imshow(acf_ch_pca[k*i+j], 
#                                   aspect='equal',
#                                   cmap=cmap,
#                                   interpolation='nearest',
#                                   vmin=vmin, vmax=vmax)
#             ax[i, j].plot(0, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
#             ax[i, j].plot(9, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
#             ax[i, j].plot(0, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
#             ax[i, j].plot(9, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
#             ax[i, j].set_xticks([])
#             ax[i, j].set_yticks([])
#             ax[i, j].grid(True, which='major', color='w', alpha=0.2)
#     #ax.set_title(f'Neural stimulus to shorten the RT')
#     plt.colorbar(shift_plot_pca, ax=ax, fraction=0.025, pad=0.06)   
#     plt.show()
    
    
#     fig, ax = plt.subplots(k, k, figsize=(16, 15))
    
#     vmin = acf_channels_vae.min()
#     vmax = acf_channels_vae.max()

    
#     acf_ch_vae = channel2grid(acf_channels_vae)
#     for i in range(k):
#         for j in range(k):
#             if k*i + j >= max_lag:
#                 continue
#             shift_plot_vae = ax[i, j].imshow(acf_ch_vae[k*i+j], 
#                                   aspect='equal',
#                                   cmap=cmap,
#                                   interpolation='nearest',
#                                   vmin=vmin, vmax=vmax)
#             ax[i, j].plot(0, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a sinistra
#             ax[i, j].plot(9, 0, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Alto a destra
#             ax[i, j].plot(0, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a sinistra
#             ax[i, j].plot(9, 9, 'x', color='r', markersize=markersize, markeredgewidth=markeredgewidth)  # Basso a destra
#             ax[i, j].set_xticks([])
#             ax[i, j].set_yticks([])
#             ax[i, j].grid(True, which='major', color='w', alpha=0.2)
#     #ax.set_title(f'Neural stimulus to shorten the RT')
#     plt.colorbar(shift_plot_vae, ax=ax, fraction=0.025, pad=0.06)   
#     plt.show()
    
#     lags = np.arange(len(acf_pca_tot))

#     # definizione funzione esponenziale con offset
#     def exp_decay(lag, A, tau):
#         return A * np.exp(-lag / tau)
# #     def exp_decay(lag, tau, C):
# #         return np.exp(-lag / tau) + C

#     # uso solo la parte di decadimento (escludo eventuale rumore per lag grandi)
#     fit_mask = (lags > 0) & (lags < max_lag)  # esempio: fino a lag=50 (~500 ms)
#     from scipy.optimize import curve_fit
#     popt_pca, pcov_pca = curve_fit(exp_decay, lags[fit_mask], acf_tot_pca[fit_mask], p0=(1, 1))
#     popt_vae, pcov_vae = curve_fit(exp_decay, lags[fit_mask], acf_tot_vae[fit_mask], p0=(1, 1))

#     A_fit_pca, tau_fit_pca = popt_pca
#     A_fit_vae, tau_fit_vae = popt_vae
# #     tau_fit_pca, C_fit_pca = popt_pca
# #     tau_fit_vae, C_fit_vae = popt_vae


#     print(f"Parametri stimati fit autocorr residui PCA: A={A_fit_pca:.3f}, tau={tau_fit_pca:.3f}")
#     print(f"Parametri stimati fit autocorr residui VAE: A={A_fit_vae:.3f}, tau={tau_fit_vae:.3f}")
# #     print(f"Parametri stimati fit autocorr residui PCA: tau={tau_fit_pca:.3f}, C={C_fit_pca:.3f}")
# #     print(f"Parametri stimati fit autocorr residui VAE: tau={tau_fit_vae:.3f}, C={C_fit_vae:.3f}")

    # Visualizziamo l'autocorrelazione media del residuo di pca e vae
#     plt.figure(figsize=(10, 6))
#     plt.stem(lags, acf_ARIMA_pca_tot, linefmt=color_PCA, markerfmt='o', basefmt='k-', label = "PCA")  # o
#     #plt.plot(lags, exp_decay(lags, *popt_pca), '--', color=color_PCA, label=f'Fit exp PCA: tau={tau_fit_pca:.2f}')
#     plt.stem(lags, acf_ARIMA_vae_tot, linefmt=color_DMM, markerfmt='o', basefmt='k-', label = "VAE") # s
#     #plt.plot(lags, exp_decay(lags, *popt_vae), '--', color=color_DMM, label=f'Fit exp VAE: tau={tau_fit_vae:.2f}')
#     #plt.title(f'Autocorrelazione media in funzione del lag per residuo tra dato vero e ricostruito')
#     plt.xlabel('Lag', fontsize=font_ax)
#     plt.ylabel('Autocorrelation', fontsize=font_ax)
#     plt.ylim(0, 1)
    
#     plt.xticks([2, 5, 8], [3, 6, 9], fontsize=font_tick)
#     plt.yticks([0.50], [0.50], fontsize=font_tick)

#     #plt.grid(True)
#     # Aggiungiamo una linea orizzontale a zero
#     #plt.axhline(y=0, color='r', linestyle='-')
#     # Aggiungiamo i limiti di confidenza (circa ±1.96/√n)
#     plt.axhline(y=1.96/np.sqrt(256//tau), color='k', linestyle='--')
#     #plt.axhline(y=-1.96/np.sqrt(256//tau), color='k', linestyle='--')
#     #plt.legend()
#     plt.show()
    
#     print("acf gen:")
#     print(acf_gen_vae_tot)
    
#     # Visualizziamo l'autocorrelazione media del residuo di pca e vae
#     plt.figure(figsize=(10, 6))
#     #plt.stem(lags, acf_tot_pca, linefmt=color_PCA, markerfmt='o', basefmt='k-', label = "PCA")  # o
#     #plt.plot(lags, exp_decay(lags, *popt_pca), '--', color=color_PCA, label=f'Fit exp PCA: tau={tau_fit_pca:.2f}')
#     plt.stem(lags, acf_gen_vae_mean_tot, linefmt=color_PCA, markerfmt='o', basefmt='k-', label = "DMM gen mean")
#     plt.stem(lags, acf_gen_vae_tot, linefmt=color_DMM, markerfmt='o', basefmt='k-', label = "DMM gen noisy") # s
#     #plt.plot(lags, exp_decay(lags, *popt_vae), '--', color=color_DMM, label=f'Fit exp VAE: tau={tau_fit_vae:.2f}')
#     #plt.title(f'Autocorrelazione media in funzione del lag per residuo tra dato vero e ricostruito')
#     plt.xlabel('Lag', fontsize=font_ax)
#     plt.ylabel('Autocorrelation', fontsize=font_ax)
#     plt.ylim(0, 1)
    
#     plt.xticks([2, 5, 8], [3, 6, 9], fontsize=font_tick)
#     plt.yticks([0.50], [0.50], fontsize=font_tick)

#     #plt.grid(True)
#     # Aggiungiamo una linea orizzontale a zero
#     #plt.axhline(y=0, color='r', linestyle='-')
#     # Aggiungiamo i limiti di confidenza (circa ±1.96/√n)
#     plt.axhline(y=1.96/np.sqrt(256//tau), color='k', linestyle='--')
#     #plt.axhline(y=-1.96/np.sqrt(256//tau), color='k', linestyle='--')
#     plt.legend()
#     plt.show()
    
    # Visualizziamo l'autocorrelazione media del residuo di pca e vae
#     plt.figure(figsize=(10, 6))
#     #plt.stem(lags, acf_tot_pca, linefmt=color_PCA, markerfmt='o', basefmt='k-', label = "PCA")  # o
#     #plt.plot(lags, exp_decay(lags, *popt_pca), '--', color=color_PCA, label=f'Fit exp PCA: tau={tau_fit_pca:.2f}')
#     plt.stem(lags, acf_inf_vae_mean_tot, linefmt=color_PCA, markerfmt='o', basefmt='k-', label = "DMM inf mean")
#     plt.stem(lags, acf_inf_vae_tot, linefmt=color_DMM, markerfmt='o', basefmt='k-', label = "DMM inf noisy") # s
#     #plt.plot(lags, exp_decay(lags, *popt_vae), '--', color=color_DMM, label=f'Fit exp VAE: tau={tau_fit_vae:.2f}')
#     #plt.title(f'Autocorrelazione media in funzione del lag per residuo tra dato vero e ricostruito')
#     plt.xlabel('Lag', fontsize=font_ax)
#     plt.ylabel('Autocorrelation', fontsize=font_ax)
#     plt.ylim(0, 1)
    
#     plt.xticks([2, 5, 8], [3, 6, 9], fontsize=font_tick)
#     plt.yticks([0.50], [0.50], fontsize=font_tick)

#     #plt.grid(True)
#     # Aggiungiamo una linea orizzontale a zero
#     #plt.axhline(y=0, color='r', linestyle='-')
#     # Aggiungiamo i limiti di confidenza (circa ±1.96/√n)
#     plt.axhline(y=1.96/np.sqrt(256//tau), color='k', linestyle='--')
#     #plt.axhline(y=-1.96/np.sqrt(256//tau), color='k', linestyle='--')
#     plt.legend()
#     plt.show()
    
    
    # Visualizziamo l'autocorrelazione media del residuo di pca e vae
#     plt.figure(figsize=(10, 6))
#     #plt.stem(lags, acf_tot_pca, linefmt=color_PCA, markerfmt='o', basefmt='k-', label = "PCA")  # o
#     #plt.plot(lags, exp_decay(lags, *popt_pca), '--', color=color_PCA, label=f'Fit exp PCA: tau={tau_fit_pca:.2f}')
#     plt.stem(lags, acf_pca_markov_tot, linefmt=color_PCA, markerfmt='o', basefmt='k-', label = "PCA markov")
#     plt.stem(lags, acf_gen_vae_mean_tot, linefmt=color_DMM, markerfmt='o', basefmt='k-', label = "DMM gen mean") # s
#     #plt.plot(lags, exp_decay(lags, *popt_vae), '--', color=color_DMM, label=f'Fit exp VAE: tau={tau_fit_vae:.2f}')
#     #plt.title(f'Autocorrelazione media in funzione del lag per residuo tra dato vero e ricostruito')
#     plt.xlabel('Lag', fontsize=font_ax)
#     plt.ylabel('Autocorrelation', fontsize=font_ax)
#     plt.ylim(0, 1)
    
#     plt.xticks([2, 5, 8], [3, 6, 9], fontsize=font_tick)
#     plt.yticks([0.50], [0.50], fontsize=font_tick)

#     #plt.grid(True)
#     # Aggiungiamo una linea orizzontale a zero
#     #plt.axhline(y=0, color='r', linestyle='-')
#     # Aggiungiamo i limiti di confidenza (circa ±1.96/√n)
#     plt.axhline(y=1.96/np.sqrt(256//tau), color='k', linestyle='--')
#     #plt.axhline(y=-1.96/np.sqrt(256//tau), color='k', linestyle='--')
#     plt.legend()
#     plt.show()
    
    
    # Visualizziamo l'autocorrelazione media del residuo di pca e vae
#     plt.figure(figsize=(10, 6))
#     #plt.stem(lags, acf_tot_pca, linefmt=color_PCA, markerfmt='o', basefmt='k-', label = "PCA")  # o
#     #plt.plot(lags, exp_decay(lags, *popt_pca), '--', color=color_PCA, label=f'Fit exp PCA: tau={tau_fit_pca:.2f}')
#     plt.stem(lags, acf_pca_non_markov_tot, linefmt=color_PCA, markerfmt='o', basefmt='k-', label = "PCA non markov")
#     plt.stem(lags, acf_gen_vae_mean_tot, linefmt=color_DMM, markerfmt='o', basefmt='k-', label = "DMM gen mean") # s
#     #plt.plot(lags, exp_decay(lags, *popt_vae), '--', color=color_DMM, label=f'Fit exp VAE: tau={tau_fit_vae:.2f}')
#     #plt.title(f'Autocorrelazione media in funzione del lag per residuo tra dato vero e ricostruito')
#     plt.xlabel('Lag', fontsize=font_ax)
#     plt.ylabel('Autocorrelation', fontsize=font_ax)
#     plt.ylim(0, 1)
    
#     plt.xticks([2, 5, 8], [3, 6, 9], fontsize=font_tick)
#     plt.yticks([0.50], [0.50], fontsize=font_tick)

#     #plt.grid(True)
#     # Aggiungiamo una linea orizzontale a zero
#     #plt.axhline(y=0, color='r', linestyle='-')
#     # Aggiungiamo i limiti di confidenza (circa ±1.96/√n)
#     plt.axhline(y=1.96/np.sqrt(256//tau), color='k', linestyle='--')
#     #plt.axhline(y=-1.96/np.sqrt(256//tau), color='k', linestyle='--')
#     plt.legend()
#     plt.show()
    
    
    # Visualizziamo l'autocorrelazione media del residuo di pca e vae
#     plt.figure(figsize=(10, 6))
#     #plt.stem(lags, acf_tot_pca, linefmt=color_PCA, markerfmt='o', basefmt='k-', label = "PCA")  # o
#     #plt.plot(lags, exp_decay(lags, *popt_pca), '--', color=color_PCA, label=f'Fit exp PCA: tau={tau_fit_pca:.2f}')
#     plt.stem(lags, acf_gen_vae_mean_tot, linefmt=color_PCA, markerfmt='o', basefmt='k-', label = "DMM gen mean")
#     plt.stem(lags, acf_inf_vae_mean_tot, linefmt=color_DMM, markerfmt='o', basefmt='k-', label = "DMM inf mean") # s
#     #plt.plot(lags, exp_decay(lags, *popt_vae), '--', color=color_DMM, label=f'Fit exp VAE: tau={tau_fit_vae:.2f}')
#     #plt.title(f'Autocorrelazione media in funzione del lag per residuo tra dato vero e ricostruito')
#     plt.xlabel('Lag', fontsize=font_ax)
#     plt.ylabel('Autocorrelation', fontsize=font_ax)
#     plt.ylim(0, 1)
    
#     plt.xticks([2, 5, 8], [3, 6, 9], fontsize=font_tick)
#     plt.yticks([0.50], [0.50], fontsize=font_tick)

#     #plt.grid(True)
#     # Aggiungiamo una linea orizzontale a zero
#     #plt.axhline(y=0, color='r', linestyle='-')
#     # Aggiungiamo i limiti di confidenza (circa ±1.96/√n)
#     plt.axhline(y=1.96/np.sqrt(256//tau), color='k', linestyle='--')
#     #plt.axhline(y=-1.96/np.sqrt(256//tau), color='k', linestyle='--')
#     plt.legend()
#     plt.show()
    
    
    # Visualizziamo l'autocorrelazione media del residuo di pca e vae
#     plt.figure(figsize=(10, 6))
#     #plt.stem(lags, acf_tot_pca, linefmt=color_PCA, markerfmt='o', basefmt='k-', label = "PCA")  # o
#     #plt.plot(lags, exp_decay(lags, *popt_pca), '--', color=color_PCA, label=f'Fit exp PCA: tau={tau_fit_pca:.2f}')
#     plt.stem(lags, acf_gen_vae_tot, linefmt=color_PCA, markerfmt='o', basefmt='k-', label = "DMM gen noisy")
#     plt.stem(lags, acf_inf_vae_tot, linefmt=color_DMM, markerfmt='o', basefmt='k-', label = "DMM inf noisy") # s
#     #plt.plot(lags, exp_decay(lags, *popt_vae), '--', color=color_DMM, label=f'Fit exp VAE: tau={tau_fit_vae:.2f}')
#     #plt.title(f'Autocorrelazione media in funzione del lag per residuo tra dato vero e ricostruito')
#     plt.xlabel('Lag', fontsize=font_ax)
#     plt.ylabel('Autocorrelation', fontsize=font_ax)
#     plt.ylim(0, 1)
    
#     plt.xticks([2, 5, 8], [3, 6, 9], fontsize=font_tick)
#     plt.yticks([0.50], [0.50], fontsize=font_tick)

#     #plt.grid(True)
#     # Aggiungiamo una linea orizzontale a zero
#     #plt.axhline(y=0, color='r', linestyle='-')
#     # Aggiungiamo i limiti di confidenza (circa ±1.96/√n)
#     plt.axhline(y=1.96/np.sqrt(256//tau), color='k', linestyle='--')
#     #plt.axhline(y=-1.96/np.sqrt(256//tau), color='k', linestyle='--')
#     plt.legend()
#     plt.show()


def whiteness_vs_lag(comm_dict, diff_dict):
    
    with np.load(comm_dict["saved_path"]+"/data_split.npz", allow_pickle=True) as loaded_file:
            train_set = loaded_file["train_set"]
            vali_set = loaded_file["vali_set"]
            test_set = loaded_file["test_set"]
            train_direction = loaded_file["train_direction"]
            vali_direction = loaded_file["vali_direction"]
            test_direction = loaded_file["test_direction"]
            train_SSD = loaded_file["train_SSD"]
            vali_SSD = loaded_file["vali_SSD"]
            test_SSD = loaded_file["test_SSD"]
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    ar = comm_dict["ar"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    max_lag = diff_dict["max_lag"]
    mean_z = diff_dict["mean_z"]
    demeaned = diff_dict["demeaned"]
    trial_type = diff_dict["trial_type"]
    
    if trial_type == "cn": 
        test_set = data["set_cn_ordRT"]
    else: 
        test_set = data["set_cs_ordSSD"]
    
    s, steps, features = test_set.shape
    
    descr = ["traj vera", "traj rec", "differenza tra traj vera e rec con VAE", "differenza tra traj vera e rec con PCA"]
    
    if trial_type == "cn":
        z, _, _ = infer_latent(dmm, data, device, n_trials=n_trials)
    else:
        _, _, z = infer_latent(dmm, data, device, n_trials=n_trials)
    z = torch.from_numpy(z).float().to(device)
    
    y_mean, y_pred = reconstruct_vae(dmm, z, n_trials=n_trials, mean_z=mean_z)#, ar=ar)
    if mean:
        y_rec = y_mean
    else:
        y_rec = y_pred
    diff_vae = test_set - y_rec
    
    test_rec = test_PCA(train_set, test_set, z_dim)
    diff_pca = test_set - test_rec

    w_pca = []
    w_vae = []
    for lag in range(2, max_lag):
        acf_data = acf_mat(test_set, max_lag=lag, demeaned=demeaned)
        acf_res_pca = acf_mat(diff_pca, max_lag=lag, demeaned=demeaned)
        acf_res_vae = acf_mat(diff_vae, max_lag=lag, demeaned=demeaned)
        acf_tot_data = acf_data.mean(1, 2)
        acf_tot_pca = acf_res_pca.mean(1, 2)
        acf_tot_vae = acf_res_vae.mean(1, 2)
        whiteness_pca = 1 - (acf_tot_pca.sum() / acf_tot_data.sum())
        whiteness_vae = 1 - (acf_tot_vae.sum() / acf_tot_data.sum())
        w_pca.append(whiteness_pca)
        w_vae.append(whiteness_vae)

    w_pca = np.array(w_pca)
    w_vae = np.array(w_vae)
    frac = w_vae/w_pca

    plt.figure(figsize=(7, 5))
    plt.title("Whiteness score as a function of max_lag")
    plt.plot(np.arange(2, max_lag), w_pca, color = "blue", label = "PCA")
    plt.plot(np.arange(2, max_lag), w_vae, color = "red", label = "VAE")
    plt.legend()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.title("Fraction of VAE and PCA whiteness score as a function of max_lag")
    plt.plot(np.arange(2, max_lag), frac)
    plt.show()

    
    
#     var_residual_vae = np.var(diff_vae)
#     var_residual_pca = np.var(diff_pca)
#     var_total = np.var(set_cn)
#     fve_vae = 1 - (var_residual_vae / var_total)
#     fve_pca = 1 - (var_residual_vae / var_total)
#     print(f"Fraction of Variance Explained (FVE) by VAE: {fve_vae}")
#     print(f"Fraction of Variance Explained (FVE) by PCA: {fve_pca}")

# +
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def calculate_reconstruction_metrics(comm_dict, diff_dict):
    """
    Calculates reconstruction metrics between real and reconstructed time series data.

    Args:
        real_data (np.ndarray): The original dataset (n_trials, n_timesteps, n_features).
                                Assumed to be normalized.
        reconstructed_data (np.ndarray): The VAE's reconstructed dataset with the same shape.
        data_range (float): The range of the normalized data (e.g., 1.0 for [0, 1] or [-0.5, 0.5],
                            2.0 for [-1, 1]). Used for PSNR calculation.

    Returns:
        dict: A dictionary containing calculated metrics:
              'mse': Mean Squared Error (overall)
              'rmse': Root Mean Squared Error (overall)
              'mae': Mean Absolute Error (overall)
              'psnr': Peak Signal-to-Noise Ratio (overall, in dB)
              'cosine_similarity_avg': Average Cosine Similarity per trial
              'mse_per_feature': Average MSE for each feature channel
              'mse_per_timestep': Average MSE for each time step
    """
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    c_dim = comm_dict["c_dim"]
    z_lims = comm_dict["z_lims"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    #axis = diff_dict["axis"]
    #trial = diff_dict["trial"]
    leg = diff_dict["leg"]
    
    set_cn_ordRT = data["set_cn_ordRT"]
    cont_cn_ordRT = data["cont_cn_ordRT"]
    #RT_cn = data["RT_cn_ordRT"]
    
    s, steps, n_features = set_cn_ordRT.shape
    
    set_cn = torch.from_numpy(set_cn_ordRT).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, s*n_trials, n_features) 
    cont_cn = torch.from_numpy(cont_cn_ordRT).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, s*n_trials, c_dim)

    z_cn, _, _ = dmm.inference(set_cn, cont_cn)
    z_cn = z_cn.reshape(steps, s, n_trials, z_dim)
    z_mean = z_cn.mean(2)
    y_mean, y_logvar = dmm.generation_x(z_mean)
    y_pred = dmm.reparameterization(y_mean, y_logvar)
    y_pred = y_pred.permute(1, 0, 2).cpu().detach().numpy()
    
    real_data = set_cn_ordRT
    reconstructed_data = y_pred
    data_range=1.0
    
    if real_data.shape != reconstructed_data.shape:
        raise ValueError("Real and reconstructed data must have the same shape.")

    # --- Overall Metrics ---
    # Flatten the data to calculate overall metrics easily
    real_flat = real_data.reshape(s, -1) # Shape: (n_trials, n_timesteps * n_features)
    recon_flat = reconstructed_data.reshape(s, -1)

    # Using sklearn functions (calculates mean over all elements by default)
    # Note: sklearn's mse/mae compute the mean over all elements if multi-output.
    # Alternatively, calculate per sample and average if needed, but global is standard.
    mse_global = mean_squared_error(real_data.flatten(), reconstructed_data.flatten())
    mae_global = mean_absolute_error(real_data.flatten(), reconstructed_data.flatten())
    rmse_global = math.sqrt(mse_global)

    # PSNR Calculation
    if mse_global == 0:
        psnr_global = float('inf') # Perfect reconstruction
    else:
        psnr_global = 10 * math.log10((data_range**2) / mse_global)

    # --- Per-Trial Cosine Similarity ---
    cos_sims = []
    for i in range(s):
        real_trial_flat = real_flat[i]
        recon_trial_flat = recon_flat[i]

        # Handle potential zero vectors
        norm_real = np.linalg.norm(real_trial_flat)
        norm_recon = np.linalg.norm(recon_trial_flat)

        if norm_real == 0 or norm_recon == 0:
            # Define similarity based on context:
            # If both are zero, similarity is 1 (perfect match)
            # If only one is zero, similarity is 0 (no match)
            sim = 1.0 if norm_real == norm_recon else 0.0
        else:
            sim = np.dot(real_trial_flat, recon_trial_flat) / (norm_real * norm_recon)
        cos_sims.append(sim)
    avg_cosine_similarity = np.mean(cos_sims)


    # --- Granular Metrics ---
    # MSE per feature (averaged over trials and time steps)
    mse_per_feature = np.mean((real_data - reconstructed_data)**2, axis=(0, 1)) # Average over n_trials and n_timesteps

    # MSE per time step (averaged over trials and features)
    mse_per_timestep = np.mean((real_data - reconstructed_data)**2, axis=(0, 2)) # Average over n_trials and n_features


    metrics = {
        'mse': mse_global,
        'rmse': rmse_global,
        'mae': mae_global,
        'psnr': psnr_global,
        'cosine_similarity_avg': avg_cosine_similarity,
        'mse_per_feature': mse_per_feature, # Array of size (n_features,)
        'mse_per_timestep': mse_per_timestep # Array of size (n_timesteps,)
    }

    print(f"mse: {metrics['mse']}")
    print(f"rmse: {metrics['rmse']}")
    print(f"mae: {metrics['mae']}")
    print(f"psnr: {metrics['psnr']}")
    print(f"cosine_similarity_avg: {metrics['cosine_similarity_avg']}")
    
    # Crea un array di indici da 0 a 95 per le posizioni delle barre
    x = np.arange(96)
    
    # You can plot mse_per_feature and mse_per_timestep to see patterns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(x, metrics['mse_per_feature'], width=1.0)
    plt.title('MSE per Feature')
    plt.xlabel('Feature Index')
    plt.ylabel('MSE')
    plt.subplot(1, 2, 2)
    plt.plot(metrics['mse_per_timestep'])
    plt.title('MSE per Timestep')
    plt.xlabel('Timestep Index')
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.show()


# -

def calculate_PCA_metrics(comm_dict, diff_dict):
    """
    Calculates reconstruction metrics between real and reconstructed time series data.

    Args:
        real_data (np.ndarray): The original dataset (n_trials, n_timesteps, n_features).
                                Assumed to be normalized.
        reconstructed_data (np.ndarray): The VAE's reconstructed dataset with the same shape.
        data_range (float): The range of the normalized data (e.g., 1.0 for [0, 1] or [-0.5, 0.5],
                            2.0 for [-1, 1]). Used for PSNR calculation.

    Returns:
        dict: A dictionary containing calculated metrics:
              'mse': Mean Squared Error (overall)
              'rmse': Root Mean Squared Error (overall)
              'mae': Mean Absolute Error (overall)
              'psnr': Peak Signal-to-Noise Ratio (overall, in dB)
              'cosine_similarity_avg': Average Cosine Similarity per trial
              'mse_per_feature': Average MSE for each feature channel
              'mse_per_timestep': Average MSE for each time step
    """
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    z_lims = comm_dict["z_lims"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    
    data = diff_dict["data"]
    n_components_max = diff_dict["n_components_max"]
    
    set_cn_ordRT = data["set_cn_ordRT"]
    cont_cn_ordRT = data["cont_cn_ordRT"]
    #RT_cn = data["RT_cn_ordRT"]
    
    s, steps, channels = set_cn_ordRT.shape
    
    from sklearn.decomposition import PCA
    real_data = set_cn_ordRT
    set_cn_flatten = set_cn_ordRT.reshape(-1, channels)
    
    n_components_array = np.arange(1, n_components_max+1)
    rec_accuracy = np.zeros(n_components_max)
    mse = np.zeros(n_components_max)
    for n_components in n_components_array:
        
        print(F"RISULTATI PER PCA CON {n_components} COMPONENTI")
        
        # 1. Applichiamo la PCA per ridurre a 3 dimensioni
        pca = PCA(n_components=(n_components))
        reduced_data_flatten = pca.fit_transform(set_cn_flatten)  # shape: (1000, 3)
        
        # 2. Ricostruiamo i dati originali da quelli ridotti
        reconstructed_data_flatten = pca.inverse_transform(reduced_data_flatten)  # shape: (1000, 96)
        reconstructed_data = reconstructed_data_flatten.reshape(s, steps, channels)
        
        
        data_range=1.0
    
        if real_data.shape != reconstructed_data.shape:
            raise ValueError("Real and reconstructed data must have the same shape.")

        # --- Overall Metrics ---
        # Flatten the data to calculate overall metrics easily
        real_flat = real_data.reshape(s, -1) # Shape: (n_trials, n_timesteps * n_features)
        recon_flat = reconstructed_data.reshape(s, -1)

        # Using sklearn functions (calculates mean over all elements by default)
        # Note: sklearn's mse/mae compute the mean over all elements if multi-output.
        # Alternatively, calculate per sample and average if needed, but global is standard.
        mse_global = mean_squared_error(real_data.flatten(), reconstructed_data.flatten())
        mae_global = mean_absolute_error(real_data.flatten(), reconstructed_data.flatten())
        rmse_global = math.sqrt(mse_global)

        # PSNR Calculation
        if mse_global == 0:
            psnr_global = float('inf') # Perfect reconstruction
        else:
            psnr_global = 10 * math.log10((data_range**2) / mse_global)

        # --- Per-Trial Cosine Similarity ---
        cos_sims = []
        for i in range(s):
            real_trial_flat = real_flat[i]
            recon_trial_flat = recon_flat[i]

            # Handle potential zero vectors
            norm_real = np.linalg.norm(real_trial_flat)
            norm_recon = np.linalg.norm(recon_trial_flat)

            if norm_real == 0 or norm_recon == 0:
                # Define similarity based on context:
                # If both are zero, similarity is 1 (perfect match)
                # If only one is zero, similarity is 0 (no match)
                sim = 1.0 if norm_real == norm_recon else 0.0
            else:
                sim = np.dot(real_trial_flat, recon_trial_flat) / (norm_real * norm_recon)
            cos_sims.append(sim)
        avg_cosine_similarity = np.mean(cos_sims)


        # --- Granular Metrics ---
        # MSE per feature (averaged over trials and time steps)
        mse_per_feature = np.mean((real_data - reconstructed_data)**2, axis=(0, 1)) # Average over n_trials and n_timesteps

        # MSE per time step (averaged over trials and features)
        mse_per_timestep = np.mean((real_data - reconstructed_data)**2, axis=(0, 2)) # Average over n_trials and n_features


        metrics = {
            'mse': mse_global,
            'rmse': rmse_global,
            'mae': mae_global,
            'psnr': psnr_global,
            'cosine_similarity_avg': avg_cosine_similarity,
            'mse_per_feature': mse_per_feature, # Array of size (n_features,)
            'mse_per_timestep': mse_per_timestep # Array of size (n_timesteps,)
        }

        print(f"mse: {metrics['mse']}")
        print(f"rmse: {metrics['rmse']}")
        print(f"mae: {metrics['mae']}")
        print(f"psnr: {metrics['psnr']}")
        print(f"cosine_similarity_avg: {metrics['cosine_similarity_avg']}")

        # Crea un array di indici da 0 a 95 per le posizioni delle barre
        x = np.arange(96)

        # You can plot mse_per_feature and mse_per_timestep to see patterns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.bar(x, metrics['mse_per_feature'], width=1.0)
        plt.title('MSE per Feature')
        plt.xlabel('Feature Index')
        plt.ylabel('MSE')
        plt.subplot(1, 2, 2)
        plt.plot(metrics['mse_per_timestep'])
        plt.title('MSE per Timestep')
        plt.xlabel('Timestep Index')
        plt.ylabel('MSE')
        plt.tight_layout()
        plt.show()


def generate_long_traj(comm_dict, diff_dict):    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    c_dim = comm_dict["c_dim"]
    RT_detector = comm_dict["RT_detector"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    cont_type = diff_dict["cont_type"]
    leg = diff_dict["leg"]
    points = diff_dict["points"]
    mean_prop = diff_dict["mean_prop"]
    only_GO = diff_dict["only_GO"]
    #mean_corr = diff_dict["mean_corr"]
    sim_start = diff_dict["sim_start"]
    tot_steps = diff_dict["tot_steps"]
    norm = diff_dict["norm"]
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    dir_cn = data["dir_cn_ordRT"]
    cont_cs = data["cont_cs_ordSSD"]
    
    samples, steps, channels = set_cn.shape
    
    x_lim_l = z_lims[axis[0], 0] - 0.5 #-5.5
    x_lim_r = z_lims[axis[0], 1] + 0.5 #7
    y_lim_l = z_lims[axis[1], 0] - 0.5 #-5
    y_lim_r = z_lims[axis[1], 1] + 0.5 #10
    
    x_points = np.linspace(x_lim_l, x_lim_r, points)   # punti per freccie generiche
    y_points = np.linspace(y_lim_l, y_lim_r, points)

    X_points, Y_points = np.meshgrid(x_points, y_points)  
    X_grid = X_points[1:points-1, 1:points-1]
    Y_grid = Y_points[1:points-1, 1:points-1]

    points_x = X_grid.flatten()
    points_y = Y_grid.flatten()

    points_z = np.column_stack((points_x, points_y))
    points_z = torch.from_numpy(points_z).float().to(device)
    cont_z = torch.zeros_like(points_z).float().to(device)
    cont_z = torch.zeros(points_z.shape[0], c_dim).float().to(device)
    if cont_type=="LEFT":
        mask_left = dir_cn==0  # prendo solo trial col contesto scelto 
        cont_left = cont_cn[mask_left][0][56//tau + 2][np.newaxis, :]   # ottengo un array (1, c_dim) del contesto scelto
        cont_z = cont_left.repeat(points_z.shape[0], axis=0)
    elif cont_type=="RIGHT":
        mask_right = dir_cn==1  # prendo solo trial col contesto scelto 
        cont_right = cont_cn[mask_right][0][56//tau + 2][np.newaxis, :]   # ottengo un array (1, c_dim) del contesto scelto
        cont_z = cont_right.repeat(points_z.shape[0], axis=0)
    elif cont_type=="STOP":
        cont_stop = cont_cs[0][-1][np.newaxis, :]   # ottengo un array (1, c_dim) del contesto scelto
        cont_z = cont_stop.repeat(points_z.shape[0], axis=0)
    elif cont_type=="preGO":
        cont_stop = cont_cn[0][0][np.newaxis, :]   # ottengo un array (1, c_dim) del contesto scelto
        cont_z = cont_stop.repeat(points_z.shape[0], axis=0)
        
    cont_z = torch.from_numpy(cont_z).float().to(device)
    
    z_mean, _ = dmm.generation_z(points_z, cont_z)
    shift = z_mean - points_z
    shift = shift.cpu().detach().numpy()#.squeeze(1).cpu().detach().numpy()

    u = shift[:, axis[0]]
    v = shift[:, axis[1]]
    
    """#renormalize the vectors
    magnitude = np.sqrt(u**2 + v**2)
    d = 0.01
    u = u*(math.sqrt(d)/magnitude)
    v = v*(math.sqrt(d)/magnitude)"""
    
    u = u/norm
    v = v/norm
    
    
    RT_cn_ordered = (RT_cn + 56)//tau
    teacher = sim_start//(5*tau) 
    q = random.randint(0, samples - 1)
    trial = set_cn[q]
    cont_c = cont_cn[q]
    
    trial = torch.from_numpy(trial).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
    cont_c = torch.from_numpy(cont_c).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
    
    z_cn, z_mean, _ = dmm.inference(trial, cont_c)
    z_teach = z_mean[:teacher] if mean_prop else z_cn[:teacher]
    
    block_preGO = torch.zeros(56//tau, n_trials, 2)
    block_postGO = torch.zeros(200//tau, n_trials, 2)
    if only_GO:
        block_preGO[:, :, 1] = 1
    block_postGO[:, :, 1] = 1
    blocks = tot_steps//(256//tau)
    cont_tot = torch.cat([block_preGO, block_postGO] * blocks, dim=0).to(device)
    #cont_tot = torch.zeros((tot_steps, n_trials, 2), device=device)

    for step in range(teacher, tot_steps):
        z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont_tot[step].unsqueeze(0))
        z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
        z_teach = torch.cat((z_teach, z_mean_gen), dim=0) if mean_prop else torch.cat((z_teach, z_gen), dim=0)
        
    z_teach = z_teach.cpu().detach().numpy()
    mu_z = z_teach.mean(1)
    mu_x = mu_z[:, 0]
    mu_y = mu_z[:, 1]

    fig, ax = plt.subplots(figsize = fig_size)
    ax.plot(z_teach[:, :, 0], z_teach[:, :, 1], c ='grey', alpha=0.05)
    ax.quiver(points_x, points_y, u, v, color = 'lime', angles='xy', scale_units='xy', scale=0.5, width = 0.003, alpha = 0.5)
    ax.plot(mu_x, mu_y, c="brown", label = f'tot_steps={tot_steps}')
    n_arrows = 15
    arrow_indices = np.arange(0, len(mu_x), len(mu_x)//n_arrows)  # Place n_arrows arrows along the path
    for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = mu_x[i+1] - mu_x[i]
        dy = mu_y[i+1] - mu_y[i]
        ax.arrow(mu_x[i], mu_y[i], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
        
    ax.set_xlim(z1_min, z1_max)
    ax.set_ylim(z2_min, z2_max)
    ax.set_xlabel("z1", fontsize=font_ax)
    ax.set_ylabel("z2", fontsize=font_ax)
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]], fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]], fontsize=font_tick)
    #ax.set_title(f'Example of ws trajectory')
    if leg:
        ax.legend(loc='best', fontsize=font_leg)


def dir_accuracy_gen_traj(comm_dict, diff_dict, sim_start):
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    c_dim = comm_dict["c_dim"]
    dir_detector = comm_dict["dir_detector"]
    z_lims = comm_dict["z_lims"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    ar = comm_dict["ar"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    mean_z = diff_dict["mean_z"]
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    dir_cn = data["dir_cn_ordRT"]
    
    #dir_cn = dir_cn.astype(int)
    samples, steps, channels = set_cn.shape
    teacher = sim_start//(5*tau) 
    alone = steps - teacher
    
    cont_dir = torch.zeros((steps, 2*samples*n_trials, c_dim)).float().to(device)
    cont_dir[:, :samples*n_trials, 1] = 1  # RIGHT
    cont_dir[:, samples*n_trials:, 0] = 1  # LEFT
    
    set_cn = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, samples*n_trials, 96) 
    cont_cn = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, samples*n_trials, c_dim)

    z_cn, _, _ = dmm.inference(set_cn, cont_cn)
    z_teach = torch.cat((z_cn[:teacher], z_cn[:teacher]), dim=1)

    for step in range(alone):
        z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont_dir[teacher+step].unsqueeze(0))
        z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
        z_teach = torch.cat((z_teach, z_gen), dim=0)

    if mean_z:
        z_teach = z_teach.reshape(steps, 2*samples, n_trials, z_dim)
        z_teach = z_teach.mean(2)
        gen_dir = np.zeros(2*samples)
        gen_dir[:samples] = 1
        gen_dir[samples:] = 0
    else:
        gen_dir = np.zeros(2*samples*n_trials)
        gen_dir[:samples*n_trials] = 1
        gen_dir[samples*n_trials:] = 0
        
    if ar:
        y_mean, y_logvar = dmm.generation_x(z_cn, torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2))
    else:
        y_mean, y_logvar = dmm.generation_x(z_cn)
    y_rec = dmm.reparameterization(y_mean, y_logvar)
    if n_trials > 1:
        y_rec = y_rec.reshape(steps, samples, n_trials, channels)
        y_rec = y_rec.mean(2)
    y_rec = y_rec.permute(1, 0, 2)
        
    if ar:
        y_mean, y_logvar = dmm.generation_x(z_teach, torch.from_numpy(set_cn[:, 56//tau:]).float().to(device).permute(1, 0, 2).repeat(1, 2, 1))
    else:
        y_mean, y_logvar = dmm.generation_x(z_teach)
    y_pred = dmm.reparameterization(y_mean, y_logvar)
#     if not mean_z:
#         y_pred = y_pred.reshape(steps, 2*samples, n_trials, channels)
#         y_pred = y_pred.mean(2)
        
    y_pred = y_pred.permute(1, 0, 2)#.cpu().detach().numpy()
        
    set_cn = set_cn.reshape(steps, samples, n_trials, channels)
    set_cn = set_cn.mean(2).permute(1, 0, 2)
    
    gen_pred_dir_prob = dir_detector(y_pred)
    rec_pred_dir_prob = dir_detector(y_rec)
    true_pred_dir_prob = dir_detector(set_cn)
    
    gen_pred_dir = binary_output(gen_pred_dir_prob)
    rec_pred_dir = binary_output(rec_pred_dir_prob)
    true_pred_dir = binary_output(true_pred_dir_prob)
        
    true_accuracy = np.array(true_pred_dir.astype(int) == dir_cn.astype(int)).mean()
    rec_accuracy = np.array(rec_pred_dir.astype(int) == dir_cn.astype(int)).mean()
    gen_accuracy = np.array(gen_pred_dir.astype(int) == gen_dir.astype(int)).mean()
    #print(f"The accuracy on true data is: {true_accuracy*100:.2f}%")
    #print(f"The accuracy on rec data is: {rec_accuracy*100:.2f}%")
    return true_accuracy, rec_accuracy, gen_accuracy


n_components_array = np.arange(1, 97)
print(n_components_array)
def plot_time_channel_pca(comm_dict, diff_dict):
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    #dir_detector = comm_dict["dir_detector"]
    z_lims = comm_dict["z_lims"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    
    data = diff_dict["data"]
    n_components_max = diff_dict["n_components_max"]

    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    dir_cn = data["dir_cn_ordRT"]
    
    #dir_cn = dir_cn.astype(int)
    samples, steps, channels = set_cn.shape
    
    set_cn = torch.from_numpy(set_cn).float().to(device)
    # true_pred_dir_prob = dir_detector(set_cn)
    # true_pred_dir = true_pred_dir_prob > 0.5
    # true_pred_dir = true_pred_dir.cpu().detach().numpy()
    # true_accuracy = np.array(true_pred_dir.astype(int) == dir_cn.astype(int)).mean()
    print(f"The accuracy on true data is: {true_accuracy*100:.2f}%")

    from sklearn.decomposition import PCA

    set_cn = set_cn.cpu().detach().numpy()
    set_cn_flatten = set_cn.reshape(-1, channels)
    
    n_components_array = np.arange(1, n_components_max+1)
    rec_accuracy = np.zeros(n_components_max)
    mse = np.zeros(n_components_max)
    for n_components in n_components_array:
        # 1. Applichiamo la PCA per ridurre a 3 dimensioni
        pca = PCA(n_components=(n_components))
        reduced_data_flatten = pca.fit_transform(set_cn_flatten)  # shape: (1000, 3)
        
        # 2. Ricostruiamo i dati originali da quelli ridotti
        reconstructed_data_flatten = pca.inverse_transform(reduced_data_flatten)  # shape: (1000, 96)
        reconstructed_data = reconstructed_data_flatten.reshape(samples, steps, channels)

        # 3. Calcoliamo l'errore di ricostruzione (opzionale)
        mse[n_components-1] = np.mean((set_cn - reconstructed_data) ** 2)
        
        reconstructed_data = torch.from_numpy(reconstructed_data).float().to(device)
        rec_pred_dir_prob = dir_detector(reconstructed_data)
        rec_pred_dir = rec_pred_dir_prob > 0.5
        rec_pred_dir = rec_pred_dir.cpu().detach().numpy()
        rec_accuracy[n_components-1] = np.array(rec_pred_dir.astype(int) == dir_cn.astype(int)).mean()
        #print(f"The accuracy on rec data is: {rec_accuracy*100:.2f}%")
        
    fig, ax = plt.subplots(figsize = fig_size)
    ax1 = ax.twinx()
    ax1.plot(n_components_array, mse, c ='r', label = "MSE between reconstructed and true data")
    ax1.set_ylabel("MSE", fontsize=font_ax)
    ax.plot(n_components_array, rec_accuracy, c ='b', label = "accuracy of the direction detector")
    ax.set_xlabel("number of principal components", fontsize=font_ax)
    ax.set_ylabel("detector accuracy", fontsize=font_ax)
    ax.set_title("Accuracy of the direction detector and MSE between reconstructed and true data as the number of principal components grow")


# def dir_accuracy_pca_traj(comm_dict, diff_dict):
#     dmm = comm_dict["dmm"]
#     device = comm_dict["device"]
#     tau = comm_dict["tau"]
#     z_dim = comm_dict["z_dim"]
#     dir_detector = comm_dict["dir_detector"]
#     z1_min, z1_max, z2_min, z2_max = comm_dict["z_lims"]
#     font_ax = comm_dict["font_ax"]
#     font_tick = comm_dict["font_tick"]
#     font_leg = comm_dict["font_leg"]
#     fig_size = comm_dict["fig_size"]

#     data = diff_dict["data"]
#     n_components_max = diff_dict["n_components_max"]

#     set_cn = data["set_cn_ordRT"]
#     cont_cn = data["cont_cn_ordRT"]
#     RT_cn = data["RT_cn_ordRT"]
#     dir_cn = data["dir_cn_ordRT"]

#     #dir_cn = dir_cn.astype(int)
#     samples, steps, channels = set_cn.shape

#     set_cn = torch.from_numpy(set_cn).float().to(device)
#     true_pred_dir_prob = dir_detector(set_cn)
#     true_pred_dir = true_pred_dir_prob > 0.5
#     true_pred_dir = true_pred_dir.cpu().detach().numpy()
#     true_accuracy = np.array(true_pred_dir.astype(int) == dir_cn.astype(int)).mean()
#     print(f"The accuracy on true data is: {true_accuracy*100:.2f}%")

#     from sklearn.decomposition import PCA

#     set_cn = set_cn.cpu().detach().numpy()
#     set_cn_flatten = set_cn.reshape(-1, channels)

#     n_components_array = np.arange(1, n_components_max+1)
#     rec_accuracy = np.zeros(n_components_max)
#     mse = np.zeros(n_components_max)
#     for n_components in n_components_array:
#         # 1. Applichiamo la PCA per ridurre a 3 dimensioni
#         pca = PCA(n_components=(n_components))
#         reduced_data_flatten = pca.fit_transform(set_cn_flatten)  # shape: (1000, 3)

#         # 2. Ricostruiamo i dati originali da quelli ridotti
#         reconstructed_data_flatten = pca.inverse_transform(reduced_data_flatten)  # shape: (1000, 96)
#         reconstructed_data = reconstructed_data_flatten.reshape(samples, steps, channels)

#         # 3. Calcoliamo l'errore di ricostruzione (opzionale)
#         mse[n_components-1] = np.mean((set_cn - reconstructed_data) ** 2)

#         reconstructed_data = torch.from_numpy(reconstructed_data).float().to(device)
#         rec_pred_dir_prob = dir_detector(reconstructed_data)
#         rec_pred_dir = rec_pred_dir_prob > 0.5
#         rec_pred_dir = rec_pred_dir.cpu().detach().numpy()
#         rec_accuracy[n_components-1] = np.array(rec_pred_dir.astype(int) == dir_cn.astype(int)).mean()
#         #print(f"The accuracy on rec data is: {rec_accuracy*100:.2f}%")

#     fig, ax = plt.subplots(figsize = fig_size)
#     ax1 = ax.twinx()
#     ax1.plot(n_components_array, mse, c ='r', label = "MSE between reconstructed and true data")
#     ax1.set_ylabel("MSE", fontsize=font_ax)
#     ax.plot(n_components_array, rec_accuracy, c ='b', label = "accuracy of the direction detector")
#     ax.set_xlabel("number of principal components", fontsize=font_ax)
#     ax.set_ylabel("detector accuracy", fontsize=font_ax)
#     ax.set_title("Accuracy of the direction detector and MSE between reconstructed and true data as the number of principal components grow")
