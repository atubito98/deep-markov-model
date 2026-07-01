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
import random
import matplotlib.gridspec as gridspec
from dvae.dataset import one_hot_cont, load_set, peak_filthers

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
ch2grid=dict()# pythoneque indexing
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
    RT = (output_clamped * 200 + 56) // tau
    return RT 

def sess_sublist(session_set, session_spec=None):
    if session_spec:
        session_list = np.where(np.isin(session_set, session_spec), session_set, np.nan)
        mask = ~np.isnan(session_list)
    else:
        session_list = session_set
        mask = np.ones(session_set.shape[0], dtype=bool)
    return session_list, mask


def inference_with_trials_s(dvae, data_set, cont_set, session_list, n_trials, z_dim, device, chunk_size=128):
    """
    Esegue l'inferenza in chunk per evitare errori di memoria GPU.

    """
    
#     data_set = torch.from_numpy(data_set).float().permute(1, 0, 2)
#     cont_set = torch.from_numpy(cont_set).float().permute(1, 0, 2)
        
    sessions = np.unique(session_list)
#     session_list = torch.from_numpy(session_list)
    z_mean_sess = np.zeros((data_set.shape[0], data_set.shape[1], z_dim))
    # Output finale
    for idx, sess in enumerate(sessions):
        print(sess)
        mask = session_list == sess
        data_sess = data_set[mask]
        cont_sess = cont_set[mask]
        
        data_sess = torch.from_numpy(data_sess).float().permute(1, 0, 2)
        cont_sess = torch.from_numpy(cont_sess).float().permute(1, 0, 2)
        steps, n_samples, features = data_sess.shape
        z_mean_accum = np.zeros((steps, n_samples, z_dim), dtype=np.float32)
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            batch_size = end - start

            # Estrai chunk
            x_chunk = data_sess[:, start:end].repeat_interleave(n_trials, dim=1)
            c_chunk = cont_sess[:, start:end].repeat_interleave(n_trials, dim=1)
#             s_chunk = session_list[start:end].repeat_interleave(n_trials, dim=0)

            # Porta su GPU
            x_chunk = x_chunk.to(device)
            c_chunk = c_chunk.to(device)
            #s_chunk = s_chunk.to(device)

            # Inferenza
            with torch.no_grad():
                z, z_mean, _ = dvae.inference(x_chunk, c_chunk, sess)

            # Media sui trials (dim=2)
            z_mean_chunk = z.cpu().numpy().reshape(steps, batch_size, n_trials, z_dim).mean(2)

            # Inserisci nel buffer
            z_mean_accum[:, start:end, :] = z_mean_chunk

#             torch.cuda.empty_cache()
        z_mean_sess[mask] = np.transpose(z_mean_accum, (1, 0, 2))    
    return z_mean_sess
        

def setup_matplotlib_backend():
    try:
        # Se siamo in Jupyter, abilita la modalità interattiva 3D
        get_ipython().run_line_magic('matplotlib', 'widget')
    except Exception:
        # Se siamo in un file .py normale, usa modalità interattiva standard
        plt.ion()
        
def RT_pred_performance_e_all(comm_dict, diff_dict):
    
    RT_detector = comm_dict["RT_detector"]
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    c_dim = comm_dict["c_dim"]
    s_dim = comm_dict["s_dim"]
    
    data = diff_dict["data"]
    mean_z = diff_dict["mean_z"]
    n_trials = diff_dict["n_trials"]
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    session_cn = data["sess_cn_ordRT"]
    #RT_min = RT_cn.min()
    RT_min = 50
    samples_cn, steps, features = set_cn.shape
    
    set_cn = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    cont_cn = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    
    session_embed = dvae.session_embeddings(torch.from_numpy(session_cn).long().to(device).repeat_interleave(n_trials))
    session_cn_flat = session_embed.unsqueeze(0).expand(steps, -1, -1)
    session_cn = session_cn_flat.reshape(steps, samples_cn, n_trials, s_dim)
    
    z_cn, _, _ = dvae.inference(set_cn, cont_cn, session_cn_flat)
    
    if mean_z:
        z_cn = z_cn.reshape(steps, samples_cn, n_trials, z_dim)
        z_cn = z_cn.mean(2)
        y_mean, y_logvar = dvae.generation_x(z_cn, session_cn.mean(2))
        y_rec = dvae.reparameterization(y_mean, y_logvar)
    else:
        y_mean, y_logvar = dvae.generation_x(z_cn, session_cn_flat)
        y_rec = dvae.reparameterization(y_mean, y_logvar)
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
    plt.ylabel('Predicted RT (using reconstructed MUA)')
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

# +
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
    
    mov_0 = (RT_cn[session==0]+56)*5
    mov_1 = (RT_cn[session==1]+56)*5
    mov_2 = (RT_cn[session==2]+56)*5
    
    num_bins = 30
    min_value = min(mov_0.min(), mov_1.min(), mov_2.min())
    max_value = max(mov_0.max(), mov_1.max(), mov_2.max())
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)
    
    fig, ax = plt.subplots()
    ax.hist(mov_0, bins=bin_edges, alpha = 0.5, color='skyblue', edgecolor='black', label = "0")
    ax.axvline(mov_0.mean(), color="skyblue",linestyle="--",label="0")
    ax.hist(mov_1, bins=bin_edges, alpha = 0.5, color='red', edgecolor='black', label = "1")
    ax.axvline(mov_1.mean(), color="red",linestyle="--",label="1")
    ax.hist(mov_2, bins=bin_edges, alpha = 0.5, color='green', edgecolor='black', label = "2")
    ax.axvline(mov_2.mean(), color="green",linestyle="--",label="2")
    
    # Add labels and title
    ax.set_xlabel('RT time ($ms$)')
    ax.set_ylabel('# of trials')
    ax.set_title("Histograms of RTs for correct no-stop trials of different sessions")


# -

def infer_latent(dvae, data, device, n_trials=1):
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    session_cn = data["sess_cn_ordRT"]
    set_ws = data["set_ws_ordRT"]
    cont_ws = data["cont_ws_ordRT"]
    session_ws = data["sess_ws_ordRT"]
    set_cs = data["set_cs_ordSSD"]
    cont_cs = data["cont_cs_ordSSD"]
    session_cs = data["sess_cs_ordSSD"]
    
    c_dim = cont_cs.shape[2]
    samples_cn = set_cn.shape[0]
    samples_cs = set_cs.shape[0]
    samples_ws = set_ws.shape[0]
    steps = set_cn.shape[1]
    
    set_cn = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    cont_cn = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    session_cn_embed = dvae.session_embeddings(torch.from_numpy(session_cn).long().to(device).repeat_interleave(n_trials))
    session_cn_flat = session_cn_embed.unsqueeze(0).expand(steps, -1, -1)
    
    set_ws = torch.from_numpy(set_ws).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    cont_ws = torch.from_numpy(cont_ws).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    session_ws_embed = dvae.session_embeddings(torch.from_numpy(session_ws).long().to(device).repeat_interleave(n_trials))
    session_ws_flat = session_ws_embed.unsqueeze(0).expand(steps, -1, -1)
    
    set_cs = torch.from_numpy(set_cs).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    cont_cs = torch.from_numpy(cont_cs).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    session_cs_embed = dvae.session_embeddings(torch.from_numpy(session_cs).long().to(device).repeat_interleave(n_trials))
    session_cs_flat = session_cs_embed.unsqueeze(0).expand(steps, -1, -1)
    
    z_cs, z_mean_cs, _ = dvae.inference(set_cs, cont_cs, session_cs_flat)
    z_cn, z_mean_cn, _ = dvae.inference(set_cn, cont_cn, session_cn_flat)
    z_ws, z_mean_ws, _ = dvae.inference(set_ws, cont_ws, session_ws_flat)
     
    z_cs = z_cs.cpu().detach().numpy()
    z_cn = z_cn.cpu().detach().numpy()
    z_ws = z_ws.cpu().detach().numpy()

    return z_cn, z_ws, z_cs



# +
def single_RTcorr(comm_dict, diff_dict, sim_start):

    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    c_dim = comm_dict["c_dim"]
    z_dim = comm_dict["z_dim"]
    s_dim = comm_dict["s_dim"]
    RT_detector = comm_dict["RT_detector"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    detector = diff_dict["detector"]
    mean_corr = diff_dict["mean_corr"]
    
    set_cn = data["set_cn_ordRT"]
    cont_c = data["cont_cn_ordRT"]
    session_cn = data["sess_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    
    samples, steps, features = set_cn.shape
    RT_cn_ordered = (RT_cn + 56)//tau
    teacher = sim_start//(5*tau) 
    alone = steps - teacher
    # 1. Per ogni sim_start, seleziono solo i trial con RT > sim_start (sono s)
    mask = RT_cn_ordered > teacher 
    s = mask.sum()
    set_cn = set_cn[mask]
    cont_c = cont_c[mask]
    session_cn = session_cn[mask]
    RT_cn_ordered_filt = RT_cn_ordered[mask]
    
    # 2. calcolo la correlazione solo se il numero di trials con RT > sim_start è maggiore di 20, per avere un minimo di statistica  
    if s > 100:
    
        set_cn = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1) 
        cont_c = torch.from_numpy(cont_c).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
        session_cn_embed = dvae.session_embeddings(torch.from_numpy(session_cn).long().to(device)).unsqueeze(0).expand(steps, -1, -1) #steps x s x s_dim
        session_cn_rep = session_cn_embed.repeat_interleave(n_trials, dim=1) #steps x s*n_trials x s_dim
        
        z_cn, _, _ = dvae.inference(set_cn, cont_c, session_cn_rep)
        z_teach = z_cn[:teacher]
        
        # inferisco fino a sim_start e genero da lì in poi
        for step in range(alone):
            z_mean_gen, z_cov_gen = dvae.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher+step].unsqueeze(0), session_cn_rep[0].unsqueeze(0))
            z_gen = dvae.reparameterization_cov(z_mean_gen, z_cov_gen)
            z_teach = torch.cat((z_teach, z_gen), dim=0)
        
        # 3. se non faccio la media delle correlazioni su n_trials, allora uso la media di z su n_trials per calcolare una sola correlazione
        if not mean_corr:
            z_teach = z_teach.reshape(steps, s, n_trials, z_dim)
            z_teach = z_teach.mean(2)
            session_cn_rep = session_cn_embed
        
        # 4. se ho a disposizione l'RT detector, calcolo l'RT della traiettoria generata usando il detector
        if detector:
            RT_output = RT_detector(z_teach.permute(1, 0, 2))
            RT_pred = prob_to_RT(RT_output, tau)
            #RT_pred, n, count = gate_to_RT(RT_prob)
            if mean_corr:
                RT_pred = RT_pred.reshape(s, n_trials)
                
            if RT_pred.ndim == 1:
                mask_nan = ~np.isnan(RT_pred)
            elif RT_pred.ndim == 2:
                mask_nan = ~np.isnan(RT_pred).any(axis=1)        # True dove il dato è valido
            RT_pred = RT_pred[mask_nan]  
            RT_cn_ordered_filt = RT_cn_ordered_filt[mask_nan]  
        # 5. altrimenti, nel caso in cui non ho il detector (in fase di training), ricostruisco la traiettoria z 
        # nello spazio delle x e uso come stima dell'RT il picco della traiettoria y mediata sui canali 
        else:
            y_mean, y_logvar = dvae.generation_x(z_teach, session_cn_rep)
            y_pred = dvae.reparameterization(y_mean, y_logvar)
            # 6. considero la traiettoria solo da teacher in poi, riducendo la probabilità che il picco con cui 
            # stimo l'RT coincida con uno dei picchi spuri nel dataset (generalmente accadono prima dell'RT)
            y_pred = y_pred[teacher:].cpu().detach().numpy()
            if not mean_corr:
                MUA_pred = y_pred.mean(2)
            else:
                y_pred = y_pred.reshape(alone, s, n_trials, 96)
                MUA_pred = y_pred.mean(3)
            # Calcolo quindi l'RT come l'istante di picco della traiettoria rimanente, + teacher
            RT_pred = np.argmax(MUA_pred, axis=0) + teacher

        if mean_corr:
            # mask peak elimina i trials per cui anche solo una realizzazione (tra le n_trials) ha RT che coincide con teacher
            mask_peak = np.all(RT_pred != teacher, axis=1)
            RT_cn_ordered_filt = RT_cn_ordered_filt[mask_peak]
            RT_pred_filt = RT_pred[mask_peak]
            if mask_peak.sum() > 10:
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
                correlation = 2
                corr_std = 1
        else:
            mask_peak = RT_pred != teacher
            mask_null = RT_pred != 0
            #print("RT diversi da 0:")
            #print(mask_null.sum())
            mask_comb = mask_peak & mask_null
            RT_cn_ordered_filt = RT_cn_ordered_filt[mask_comb]
            RT_pred_filt = RT_pred[mask_comb]
            if mask_comb.sum() > 50:
                correlation = np.corrcoef(RT_cn_ordered_filt, RT_pred_filt)[0, 1]
                corr_std = None
            else:
                correlation = 2
                corr_std = 1
    else:
        RT_cn_ordered_filt = 0
        RT_pred_filt = np.ones((1, 2))
        correlation = 2
        corr_std = 1

    return RT_cn_ordered_filt, RT_pred_filt, correlation, corr_std
    #return RT_cn_ordered_filt, RT_pred_filt.mean(1), correlation, corr_std



def correlation_vs_gentime_e_all(comm_dict, diff_dict):     
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    saved_path = comm_dict["saved_path"]
    
    data = diff_dict["data"]
    sim_start_array = diff_dict["sim_start_array"]
    n_trials = diff_dict["n_trials"]
    detector = diff_dict["detector"]
    mean_corr = diff_dict["mean_corr"]
    compute = diff_dict["compute"]
    show_comp = diff_dict["show_comp"]
    save = diff_dict["save"]
    saved_dict = diff_dict["saved_dict"]
    dir_comp = diff_dict["dir_comp"]
    
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
    
    fig, ax1 = plt.subplots(figsize = (8, 6))

    #ax1.errorbar(sim_start_array_new, correlations_new, yerr=corr_stds_new, fmt='o', ls='--', ecolor='black', 
    ax1.errorbar(sim_start_array_new, correlations_new, yerr=corr_stds_new, fmt='o', ls='--', ecolor='black', 
                     elinewidth=1, capsize=5, capthick=1)
    ax1.scatter(sim_start_array_new, correlations_new, label = "current model")
        
    if show_comp:
        directory = "/raid/home/tubitoal/DVAE/saved_model/" + dir_comp
        with np.load(directory + "/RT_correlations.npz") as loaded_file:
            correlations_old = loaded_file["correlations"]
            corr_stds_old = loaded_file["corr_stds"]
            sim_start_array_old = loaded_file["sim_start_array"]

        ax1.errorbar(sim_start_array_old, correlations_old, yerr=corr_stds_old, fmt='o', ls='--', ecolor='black', 
                         elinewidth=1, capsize=5, capthick=1)
        ax1.scatter(sim_start_array_old, correlations_old, label = dir_comp)
    ax1.axvline(56*5, color="black",linestyle="--",label="GO")
    ax1.set_ylim((0, 1))
    ax1.set_xlabel("Simulation start time ($ms$)")
    ax1.set_ylabel("Correlation between predicted and true RT")
    ax1.set_title(f"correlations between true and predicted RT, mean_corr={mean_corr}, detector={detector}")

    ax2 = ax1.twinx()
    ax2.hist((RT_cn + 56) * 5, bins=20, alpha=0.5, color='skyblue', edgecolor='black', label="RT histogram")
    ax2.set_ylabel("# of trials")
    ax2.set_ylim((0, 50))

    # Optionally, if you want to add a legend for both plots
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Display the plot
    plt.show()
        
    dict_corr = {a: (b, c) for a, b, c in zip(sim_start_array_new, correlations_new, corr_stds_new)}
    
    return dict_corr


def PCA_vs_VAE_e_all(comm_dict, diff_dict):
    
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
            session_train = loaded_file["session_train"]
            session_vali = loaded_file["session_vali"]
            session_test = loaded_file["session_test"]
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    c_dim = comm_dict["c_dim"]
    tau = comm_dict["tau"]
    t = comm_dict["t"]

    n_trials = diff_dict["n_trials"]
    
    n_train, steps, features = train_set.shape
    n_vali = vali_set.shape[0]
    n_test = test_set.shape[0]
    
    print(test_set.shape)

    ############## PCA ###############
    X_train = train_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
    X_vali = vali_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
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
    
    ############## DVAE ##############
    cont_test = one_hot_cont(test_SSD, test_direction, tau)
    test_data = torch.from_numpy(test_set).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    cont_test = torch.from_numpy(cont_test).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    session_cn = torch.from_numpy(session_test).long().to(device).repeat_interleave(n_trials) #steps x s x s_dim
    #session_cn_rep = session_cn_embed.repeat_interleave(n_trials, dim=1) #steps x s*n_trials x s_dim

    # reconstruction        
    y_mean, y_logvar = dvae(test_data, cont_test, session_cn)
    y_inf = dvae.reparameterization(y_mean, y_logvar)
    if n_trials>1:
        y_inf = y_inf.reshape(steps, n_test, n_trials, features).mean(2)
    y_inf = y_inf.permute(1, 0, 2).cpu().detach().numpy()
    
    MSE_vae = np.sqrt((y_inf - test_set)**2).mean(0).mean(1)
    MSE_pca = np.sqrt((test_rec - test_set)**2).mean(0).mean(1)
    
    print(f"MSE for PCA over the entire dataset: {MSE_pca.mean(0)}")
    print(f"MSE for VAE over the entire dataset: {MSE_vae.mean(0)}")
    
    f, ax = plt.subplots(figsize = (8, 6))
    
    ax.plot(t*5, MSE_vae, c ="r", label = 'VAE')
    ax.plot(t*5, MSE_pca, c ="b", label = 'PCA')
    ax.axvline(56*5, color="black",linestyle="--",label= "GO")
    ax.set_title(f"MSE over time, between Observed and Reconstructed MUA using VAE (red) and PCA (blue)")
    ax.set_xlabel('time ($ms$)')
    ax.set_ylabel('MSE')
    ax.legend(fontsize=7.3)
        
    #return MUA_trials, MUA_pred, MUA_inf


# +
def MUA_pred_inf(comm_dict, diff_dict):
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    t = comm_dict["t"]
    RT_detector = comm_dict["RT_detector"]
    move_detector = comm_dict["move_detector"]
    
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
        y_mean, y_logvar = dvae(trial, cont_c)
        y_inf = dvae.reparameterization(y_mean, y_logvar)
        y_inf = y_inf.cpu().detach().numpy()

        MUA_inf[i] = y_inf

        # generation
        z, z_mean, _ = dvae.inference(trial, cont_c)
        z_teach = z[:teacher]

        for step in range(alone):
            z_mean_gen, z_cov_gen = dvae.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher+step].unsqueeze(0))
            z_gen = dvae.reparameterization_cov(z_mean_gen, z_cov_gen)
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
        
        y_mean, y_logvar = dvae.generation_x(z_teach)
        y_pred = dvae.reparameterization(y_mean, y_logvar)
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
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    
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
        z_inf, z_mean, _ = dvae.inference(trial, cont_c)
        z_teach = z_mean[:teacher]
        for step in range(alone):
            if cpast:
                z_mean_gen, z_cov_gen = dvae.generation_z(z_teach[-1].unsqueeze(0), cont_c[:(teacher+step+1)])
            else:
                z_mean_gen, z_cov_gen = dvae.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher+step].unsqueeze(0))
            z_gen = dvae.reparameterization_cov(z_mean_gen, z_cov_gen)
            z_teach = torch.cat((z_teach, z_gen), dim=0)

        y_mean, y_logvar = dvae.generation_x(z_teach)
        y_pred = dvae.reparameterization(y_mean, y_logvar)
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
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    
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
    
    z, z_mean, _ = dvae.inference(trial, cont_c)
    z_teach = z_mean[:teacher]

    for step in range(alone):
        z_mean_gen, z_cov_gen = dvae.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher+step].unsqueeze(0))
        z_gen = dvae.reparameterization_cov(z_mean_gen, z_cov_gen)
        z_teach = torch.cat((z_teach, z_gen), dim=0)

    y_mean, y_logvar = dvae.generation_x(z_teach)
    y_pred = dvae.reparameterization(y_mean, y_logvar)
    y_pred = y_pred.cpu().detach().numpy()
    MUA_pred = y_pred.mean(2)
    mean = np.mean(MUA_pred, axis=1)
    std = np.std(MUA_pred, axis=1)
    
    '''
    # PLOT
    fig, ax = plt.subplots(figsize = (6, 5))

    curva = []    
    for step in range(test_trial.shape[0]):    
        contatore = 0
        for i in range(96):
            if test_trial[step, i] > 2:
                contatore += 1
        curva.append(contatore)

    ax.plot(t, curva)
    ax.set_ylim(0, 90)
    '''
    
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


# -

def latent_cn_traj_directions_e_all(comm_dict, diff_dict):
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    
    data = diff_dict["data"]
    alpha = diff_dict["alpha"]
    
    dir_cn = data["dir_cn_ordRT"]
    
    z_cn, _, _ = infer_latent(dvae, data, device)
    
    print(z_cn.shape)
    
    x_r = z_cn[:, dir_cn==1, 0]
    x_l = z_cn[:, dir_cn==0, 0]

    y_r = z_cn[:, dir_cn==1, 1]
    y_l = z_cn[:, dir_cn==0, 1]    
    
    if z_dim==2:
        
        f, ax = plt.subplots(figsize = (7, 6))
        
        ax.plot(x_r, y_r, '-', linewidth=2, color='g', alpha = alpha)
        ax.plot(x_l, y_l, '-', linewidth=2, color='r', alpha = alpha)
        ax.set_xlabel("first latent component")
        ax.set_ylabel("second latent component")
        ax.set_title(f"right vs left directed trials")
        return f
    
    elif z_dim==3:
        
        z_r = z_cn[:, dir_cn==1, 2]
        z_l = z_cn[:, dir_cn==0, 2]  
        
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
                xaxis_title='z1', yaxis_title='z2', zaxis_title='z3',
                bgcolor='black'
            ),
            width=700, height=700,
            title='Right vs Left trajectories'
        )
        #fig.show()
        return fig


# import plotly.graph_objects as go
# from sklearn.cluster import KMeans

# import numpy as np
# import plotly.graph_objects as go

# def latent_cn_traj_directions(comm_dict, diff_dict, n_show=8):
#     dvae = comm_dict["dvae"]
#     device = comm_dict["device"]
#     z_dim = comm_dict["z_dim"]
#     data = diff_dict["data"]
#     dir_cn = data["dir_cn_ordRT"]

#     z_cn, _, _ = infer_latent(dvae, data, device)  # (T, N, z_dim)
#     print("Latent shape:", z_cn.shape)

#     if z_dim != 3:
#         raise ValueError("This version supports only z_dim == 3 for 3D visualization.")

#     # Split by condition (right vs left)
#     z_r = z_cn[:, dir_cn == 1, :]   # shape (T, N_r, 3)
#     z_l = z_cn[:, dir_cn == 0, :]   # shape (T, N_l, 3)

#     # Pick a few representative trajectories to avoid clutter
#     n_show_r = min(n_show, z_r.shape[1])
#     n_show_l = min(n_show, z_l.shape[1])
#     idx_r = np.linspace(0, z_r.shape[1]-1, n_show_r, dtype=int)
#     idx_l = np.linspace(0, z_l.shape[1]-1, n_show_l, dtype=int)

#     z_r = z_r[:, idx_r, :]
#     z_l = z_l[:, idx_l, :]

#     fig = go.Figure()

#     def add_trajs(z_set, color, name):
#         """Add trajectories with direction arrows."""
#         for i in range(z_set.shape[1]):  # loop over selected trials
#             traj = z_set[:, i, :]  # (T, 3)
#             x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]

#             # Main line
#             fig.add_trace(go.Scatter3d(
#                 x=x, y=y, z=z,
#                 mode='lines',
#                 line=dict(color=color, width=4),
#                 opacity=0.6,
#                 name=name if i == 0 else None
#             ))

#             # Add arrows (every ~10% of trajectory)
#             step = max(1, len(x)//10)
#             for j in range(0, len(x)-step, step):
#                 fig.add_trace(go.Cone(
#                     x=[x[j]], y=[y[j]], z=[z[j]],
#                     u=[x[j+step]-x[j]], v=[y[j+step]-y[j]], w=[z[j+step]-z[j]],
#                     sizemode="absolute", sizeref=0.15,
#                     anchor="tail",
#                     colorscale=[[0, color], [1, color]],
#                     showscale=False,
#                     opacity=0.8
#                 ))

#     add_trajs(z_r, "green", "Right trials")
#     add_trajs(z_l, "red", "Left trials")

#     # Layout and style
#     fig.update_layout(
#         scene=dict(
#             xaxis_title='Latent dim 1',
#             yaxis_title='Latent dim 2',
#             zaxis_title='Latent dim 3',
#             xaxis=dict(backgroundcolor='white', gridcolor='lightgray'),
#             yaxis=dict(backgroundcolor='white', gridcolor='lightgray'),
#             zaxis=dict(backgroundcolor='white', gridcolor='lightgray'),
#         ),
#         title=dict(
#             text="Latent Trajectories (Right vs Left)",
#             x=0.5,
#             font=dict(size=18)
#         ),
#         width=850,
#         height=800,
#         showlegend=True,
#         paper_bgcolor="white",
#         plot_bgcolor="white",
#         margin=dict(l=0, r=0, b=0, t=50)
#     )

#     #fig.show()
#     return fig



# +
def random_latent_cn_traj(dvae, data, tau, device, n_trials=1):
    
    RT_cn = data["RT_cn_ordRT"]
    steps = 256//tau
    n = len(RT_cn)
    q = random.randint(0, n - 1)
    RT_trial = RT_cn[q]
    
    z_cn, _, _ = infer_latent(dvae, data, device, n_trials=n_trials)
    z_dim = z_cn.shape[2]

    if n_trials>1:
        z_cn = z_cn.reshape(steps, n, n_trials, z_dim)
    
    z_trial = z_cn[:, q]  
    RT = (56 + RT_trial)//tau
    
    return z_trial, RT, q


def random_latent_cs_traj(dvae, data, tau, device, n_trials=1):
    
    SSD_cs = data["SSD_cs_ordSSD"]
    steps = 256//tau
    n = len(SSD_cs)
    q = random.randint(0, n - 1)
    SSD_trial = SSD_cs[q]
    
    _, _, z_cs = infer_latent(dvae, data, device, n_trials=n_trials)
    z_dim = z_cs.shape[2]

    if n_trials>1:
        z_cs = z_cs.reshape(steps, n, n_trials, z_dim)
    
    z_trial = z_cs[:, q]  
    SSD = (56 + SSD_trial)//tau
    
    return z_trial, SSD, q



def latent_traj_ctype_e_all(comm_dict, diff_dict):
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    t = comm_dict["t"]
    z_dim = comm_dict["z_dim"]
    
    data = diff_dict["data"]
    alfa = diff_dict["alfa"]
    type_list = diff_dict["type_list"]
    type_trial = diff_dict["type_trial"]
    RT_show = diff_dict["RT_show"]
    
    RT_cn = data["RT_cn_ordRT"] 
    RT_ws = data["RT_ws_ordRT"] 
    RT_cn = (RT_cn + 56)//tau
    RT_ws = (RT_ws + 56)//tau
    steps = 256//tau
    
    z_cn, z_ws, z_cs = infer_latent(dvae, data, device)
    samples_cn = z_cn.shape[1]
    samples_ws = z_ws.shape[1]
    
    x_cs = z_cs[:, :, 0]
    x_cn = z_cn[:, :, 0]
    x_ws = z_ws[:, :, 0]

    y_cs = z_cs[:, :, 1]
    y_cn = z_cn[:, :, 1]
    y_ws = z_ws[:, :, 1]
    
    x_RT_cn = x_cn[RT_cn, np.arange(samples_cn)]
    y_RT_cn = y_cn[RT_cn, np.arange(samples_cn)]
    x_RT_ws = x_ws[RT_ws, np.arange(samples_ws)]
    y_RT_ws = y_ws[RT_ws, np.arange(samples_ws)]
    
    x_GO_cn = x_cn[56//tau]
    y_GO_cn = y_cn[56//tau]
    x_GO_ws = x_ws[56//tau]
    y_GO_ws = y_ws[56//tau]
    
    if type_trial=="cn":
        z_trial, RT, q = random_latent_cn_traj(dvae, data, tau, device)
        set_cn = data["set_cn_ordRT"]
        MUA = set_cn[q]
        x_RT = z_trial[RT, 0]
        y_RT = z_trial[RT, 1]
    elif type_trial=="cs":
        z_trial, SSD, q = random_latent_cs_traj(dvae, data, tau, device)
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
            ax.plot(x_cn, y_cn, '-', linewidth=2, color='g', alpha = 0.1*alfa)
        if "ws"in type_list:
            ax.plot(x_ws, y_ws, '-', linewidth=2, color='orange', alpha = 0.2*alfa)
        if "cs"in type_list:
            ax.plot(x_cs, y_cs, '-', linewidth=2, color='r', alpha = 0.2*alfa)

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


        f, ax = plt.subplots(figsize = (8, 6))
        ax.plot(t*5, MUA.mean(1), c = "green", label = 'mean MUA of the traj above')
        ax.axvline(56*5, color="black",linestyle="--",label= "GO")
        if type_trial=="cn":
            ax.axvline(RT*tau*5, color="blue",linestyle="--",label= "RT")
        if type_trial=="cs":
            ax.axvline(SSD*tau*5, color="red",linestyle="--",label= "SSD")
        ax.set_xlabel('time ($ms$)')
        ax.set_ylabel('mean MUA activity over channels')
        ax.legend(fontsize=7.3)
        plt.show()
    
    
    elif z_dim==3:
        
        z_cs = z_cs[:, :, 2]
        z_cn = z_cn[:, :, 2]
        z_ws = z_ws[:, :, 2]
        
        z_true_story = z_trial[:, 2]
        z_start = z_trial[0, 2]
        z_GO = z_trial[56//tau, 2]
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
    
    """elif dim==3:
        
        z_cs = z_cs[:, :, axes[2]].reshape(-1).cpu().detach().numpy()
        z_cn = z_cn[:, :, axes[2]].reshape(-1).cpu().detach().numpy()
        z_ws = z_ws[:, :, axes[2]].reshape(-1).cpu().detach().numpy()
        
        z_true_story = true_story[:, axes[2]]
        z_start = start[axes[2]]
        z_GO = GO[axes[2]]
        z_RT = RT_z[axes[2]]
        from mpl_toolkits.mplot3d import Axes3D

        #%matplotlib notebook

        fig = plt.figure(figsize = (7, 7))
        ax = fig.add_subplot(111, projection='3d')


        ax.plot3D(x_cn, y_cn, z_cn, '-', linewidth=2, color='g', alpha = 0.2)
        ax.plot3D(x_ws, y_ws, z_ws, '-', linewidth=2, color='orange', alpha = 0.3)
        ax.plot3D(x_cs, y_cs, z_cs, '-', linewidth=2, color='r', alpha = 0.2)
        ax.scatter3D(x_true_story, y_true_story, z_true_story, s = 15, c = color, cmap = 'copper', alpha = 1, label = "trajectory example")
        ax.scatter3D(x_GO, y_GO, z_GO, s = 50, c = 'b', alpha = 1, label = "GO")
        ax.scatter3D(x_RT, y_RT, z_RT, s = 50, c = 'r', alpha = 1, label = "RT")
        ax.scatter(x_start, y_start, z_start, s = 50, c = 'purple', alpha = 1, label = "start")
        ax.patch.set_facecolor('black')

        ax.set_xlabel('z1')
        ax.set_ylabel('z2')
        ax.set_zlabel('z3')
        ax.set_title(f'cs vs ws vs ns')
        ax.view_init(40, 45)

        plt.show()"""


# +
def latent_traj_csession(comm_dict, diff_dict):
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    dir_flag = comm_dict["dir_flag"]
    
    session_list = diff_dict["session_list"]
    
    piero_data_path_array = ['/raid/home/tubitoal/DVAE/dvae/dataset/MUA/data/Piero_20131202.npz',
                         '/raid/home/tubitoal/DVAE/dvae/dataset/MUA/data/Piero_20140109.npz',
                         '/raid/home/tubitoal/DVAE/dvae/dataset/MUA/data/Piero_20140116.npz',
                         '/raid/home/tubitoal/DVAE/dvae/dataset/MUA/data/Piero_20140606.npz',
                         '/raid/home/tubitoal/DVAE/dvae/dataset/MUA/data/Piero_20140701.npz',
                         '/raid/home/tubitoal/DVAE/dvae/dataset/MUA/data/Piero_20140922.npz']

    piero_sessions = [piero_data_path_array[i] for i in session_list]

    selection ="((y_stop == True) | ((y_stop == False)&(y_reward == True)))"# selezioni tutto il movimento &(y_reward == True)


    data, RT, SSD, direction, session, subject = load_set(piero_sessions, selection)

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
    
    z, z_mean, _ = dvae.inference(data, cont_c)
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



# def plot_zGO_RTgrad(comm_dict, diff_dict):
#     
#     dvae = comm_dict["dvae"]
#     device = comm_dict["device"]
#     tau = comm_dict["tau"]
#     z1_min, z1_max, z2_min, z2_max = comm_dict["z_lims"]
#     font_ax = comm_dict["font_ax"]
#     font_tick = comm_dict["font_tick"]
#     font_leg = comm_dict["font_leg"]
#     fig_size = comm_dict["fig_size"]
#     
#     data = diff_dict["data"]
#     n_trials = diff_dict["n_trials"]
#     points = diff_dict["points"]
#     time = diff_dict["time"]
#     bins = diff_dict["bins"]
#     GO_plot = diff_dict["GO_plot"]
#     trial = diff_dict["trial"]
#     cmap = diff_dict["cmap"]
#     mean_trials = diff_dict["mean_trials"]
#     #alpha = diff_dict["alpha"]
#     #c_norm = diff_dict["c_norm"]
#     
#     
#     RT_cn = data["RT_cn_ordRT"]
#     z_cn, _, _ = infer_latent(dvae, data, device, n_trials)
#     z_dim = z_cn.shape[2]
#     steps = z_cn.shape[0]
#     s = z_cn.shape[1]
#     samples = len(RT_cn)
#     RT_cn_rep = np.expand_dims(RT_cn, axis=1)
#     RT_cn_rep = RT_cn_rep.repeat(n_trials, axis=1).reshape(-1)
#
#     z1_edges = np.linspace(z1_min, z1_max, bins + 1)
#     z2_edges = np.linspace(z2_min, z2_max, bins + 1)
#
#     if mean_trials:
#         z_cn = z_cn.reshape(steps, samples, n_trials, z_dim)
#         z_cn = z_cn.mean(2)
#     
#     z_GO = z_cn[time//(5*tau)]
#     #z_RT = z_cn[(RT_cn + 56)//tau, np.arange(samples)]
#     if mean_trials:
#         color = RT_cn/RT_cn.max()
#     else:
#         color = RT_cn_rep/RT_cn_rep.max()
#     
#     mask = color > 0.4  # Limitiamo il contour plot alla regione RT/RT_max > 0.4
#     
#     # Creiamo una griglia più fitta per interpolare i dati
#     z1_vals = np.linspace(z1_min, z1_max, points)
#     z2_vals = np.linspace(z2_min, z2_max, points)
#     z1_grid, z2_grid = np.meshgrid(z1_vals, z2_vals)
#     
#     interpolated = griddata((z_GO[mask, 0], z_GO[mask, 1]), color[mask], (z1_grid, z2_grid), method='cubic')
#     
#     fig = plt.figure(figsize=(fig_size[0] + 0.5, fig_size[1]))
#     gs = gridspec.GridSpec(1, 2, width_ratios=[fig_size[0], 0.5])
#     
#     ax = plt.subplot(gs[0])
#     cax = plt.subplot(gs[1])
#     
#     ax.set_xlabel("z1", fontsize=font_ax)
#     ax.set_ylabel("z2", fontsize=font_ax)
#     ax.set_xticks([-1, 2, 5]) 
#     ax.set_xticklabels([-1, 2, 5], fontsize=font_tick)
#     ax.set_yticks([-2, 1, 4])
#     ax.set_yticklabels([-2, 1, 4], fontsize=font_tick)
#     
#     # Contour plot con livelli continui
#     levels = np.linspace(0.4, 1, 10)
#     contour = ax.contourf(z1_grid, z2_grid, interpolated, levels=levels, cmap=cmap)
#     
#     cbar = plt.colorbar(contour, cax=cax)
#     cbar.set_ticks([0.4, 0.6, 0.8, 1])
#     cbar.set_label('Mean RT/RTmax', fontsize=font_ax)
#     cbar.ax.tick_params(labelsize=font_tick)
#     
#     plt.show()



    RT_cn = data["RT_cn_ordRT"]
    z_cn, _, _ = infer_latent(dvae, data, device, n_trials)
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
    ax.set_xticks([-1, 2, 5]) 
    ax.set_xticklabels([-1, 2, 5], fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks([-2, 1, 4])
    ax.set_yticklabels([-2, 1, 4], fontsize=font_tick)
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



# +
from scipy.stats import binned_statistic_2d

def plot_z_cRTtrue_e_all(comm_dict, diff_dict):
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    z1_min, z1_max, z2_min, z2_max = comm_dict["z_lims"]
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
    mean_trials = diff_dict["mean_trials"]
    show_all = diff_dict["show_all"]
    alpha = diff_dict["alpha"]
    l = diff_dict["l"]
    #c_norm = diff_dict["c_norm"]
    
    
    RT_cn = data["RT_cn_ordRT"]
    z_cn, _, _ = infer_latent(dvae, data, device, n_trials)
    
    steps, s, _ = z_cn.shape
    samples = len(RT_cn)    # s = samples * n_trials

    z1_edges = np.linspace(z1_min, z1_max, bins + 1)
    z2_edges = np.linspace(z2_min, z2_max, bins + 1)

    if mean_trials:
        z_cn = z_cn.reshape(steps, samples, n_trials, z_dim)
        z_cn = z_cn.mean(2)
        z_RT = z_cn[(RT_cn + 56)//tau, np.arange(samples)]
        color = RT_cn/RT_cn.max()
    else:
        RT_cn_rep = np.repeat(RT_cn, n_trials)
        z_RT = z_cn[(RT_cn_rep + 56)//tau, np.arange(s)]
        color = RT_cn_rep/RT_cn_rep.max()
    z_GO = z_cn[time//(5*tau)]
    
    z, RT, q = random_latent_cn_traj(dvae, data, tau, device)
    
    print(f"traj n.{q}")
    
    x_true_story = z[:, 0]
    y_true_story = z[:, 1]
    x_start = z[0, 0]
    y_start = z[0, 1]
    x_GO = z[56//tau, 0]
    y_GO = z[56//tau, 1]
    x_RT = z[RT, 0]
    y_RT = z[RT, 1]
    
    if GO_plot:
        hist, x_edges, y_edges, binnumber = binned_statistic_2d(z_GO[:, 0], z_GO[:, 1], color, statistic='mean', bins=[z1_edges, z2_edges])
        text = 'GO'
    else:
        hist, x_edges, y_edges, binnumber = binned_statistic_2d(z_RT[:, 0], z_RT[:, 1], color, statistic='mean', bins=[z1_edges, z2_edges])
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
    ax.set_xticks([-1, 2, 5]) 
    ax.set_xticklabels([-1, 2, 5], fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks([-2, 1, 4])
    ax.set_yticklabels([-2, 1, 4], fontsize=font_tick)
    if show_all:
        ax.plot(z_cn[:, :, 0], z_cn[:, :, 1], color="green", alpha=alpha)
    im=ax.pcolormesh(z1, z2, hist.T, cmap=cmap, shading='auto')
    # Punto medio per disegnare la direzione
    center = z_GO.mean(axis=0)
    length = l  # lunghezza del vettore per visualizzazione

    # Freccia nella direzione di diminuzione
    ax.arrow(center[0], center[1],
              -direction[0]*length, -direction[1]*length,
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

def delta_x_via_jvp(dvae, z0, d, l):
    device = z0.device
    z0 = z0.detach().clone().requires_grad_(True)   # enable autograd        
    delta_z = torch.from_numpy(-d * l).float().to(device)   # se vuoi diminuire lungo d; + per aumentare
    if z0.ndim == 2:
        delta_z = torch.tile(delta_z, (z0.shape[0], 1))
    # jvp expects tuples for inputs/outputs
    def decoder_mean(z):
        x_mean, _ = dvae.generation_x(z)
        return x_mean
    y, jvp_out = jvp(decoder_mean, (z0,), (delta_z,))
    jvp_out = jvp_out.cpu().detach().numpy()
    # jvp_out has shape of x (. e.g. data_dim)
    #print(y.shape)
    print(jvp_out.shape)
    return jvp_out  # appross delta_x
    
    
def plot_z_shift(comm_dict, diff_dict):
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    z1_min, z1_max, z2_min, z2_max = comm_dict["z_lims"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    RT_detector = comm_dict["RT_detector"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    #bins = diff_dict["bins"]
    cmap = diff_dict["cmap"]
    q = diff_dict["q"]
    alpha = diff_dict["alpha"]
    mean_trials = diff_dict["mean_trials"]
    l = diff_dict["l"]
    alpha_point = diff_dict["alpha_point"]
    markersize = 15
    markeredgewidth = 3
    
    RT_cn = data["RT_cn_ordRT"]
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    z_cn, _, _ = infer_latent(dvae, data, device, n_trials)
    
    steps, s, _ = z_cn.shape
    samples = len(RT_cn)    # s = samples * n_trials

#     z1_edges = np.linspace(z1_min, z1_max, bins + 1)
#     z2_edges = np.linspace(z2_min, z2_max, bins + 1)

    if mean_trials:
        z_cn = z_cn.reshape(steps, samples, n_trials, z_dim)
        z_cn = z_cn.mean(2)
        z_RT = z_cn[(RT_cn + 56)//tau, np.arange(samples)]
        color = RT_cn/RT_cn.max()
    else:
        RT_cn_rep = np.repeat(RT_cn, n_trials)
        z_RT = z_cn[(RT_cn_rep + 56)//tau, np.arange(s)]
        color = RT_cn_rep/RT_cn_rep.max()
    z_GO = z_cn[56//tau]

    from sklearn.linear_model import LinearRegression
    
    # --- STIMA DIREZIONE ---
    reg = LinearRegression().fit(z_GO, color)
    direction = reg.coef_
    direction /= np.linalg.norm(direction)  # normalizzazione unit vector

    print("Direzione di variazione (aumenta):", direction)
    
    #RT_mask = (z_GO.shape[0]//2):-q
    no_peak_mask = z_GO[:, 0]>-1
    z_GO_true = z_GO[no_peak_mask]
    z_GO_device = torch.from_numpy(z_GO_true).float().to(device)
    trial_modified = set_cn[:, :(56//tau + 1)][no_peak_mask]
    
    dx = delta_x_via_jvp(dvae, z_GO_device, direction, l=l)
    
    vmin = dx.min()
    vmax = dx.max()
    
    trial_modified[:, 56//tau] = trial_modified[:, 56//tau] + dx
    trial_modified = torch.from_numpy(trial_modified).float().to(device).permute(1, 0, 2)
    cont_RTlong = torch.from_numpy(cont_cn[no_peak_mask]).float().to(device).permute(1, 0, 2)
    z_modified, _, _ = dvae.inference(trial_modified, cont_RTlong[:(56//tau + 1)])
    z_GO_modified = z_modified[-1].cpu().detach().numpy()

    # Plot the density map
    fig, ax = plt.subplots(figsize=(6, 6))
    
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
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(z_cn[:, :, 0], z_cn[:, :, 1], color="green", alpha=alpha)
    ax.scatter(z_GO_true[:, 0], z_GO_true[:, 1], c="blue", s = 30, alpha=alpha_point, label = 'GO of the inferred traj')
    ax.scatter(z_GO_modified[:, 0], z_GO_modified[:, 1], c="red", s = 30, alpha=alpha_point, label = 'GO after addition of stimulus')
    ax.plot(z_GO_modified[:, 0].mean(), z_GO_modified[:, 1].mean(), c = 'red', marker='*', markeredgewidth = 4, markersize = 20)
    ax.plot(z_GO_true[:, 0].mean(), z_GO_true[:, 1].mean(), c = 'blue', marker='*', markeredgewidth = 4, markersize = 20)
    
    
    #s = set_cn.shape[0]
    steps = set_cn.shape[1]
    teacher = 56//tau + 1
    alone = steps - teacher

    for step in range(alone):
        z_mean_gen, z_cov_gen = dvae.generation_z(z_modified[-1].unsqueeze(0), cont_RTlong[teacher+step].unsqueeze(0))
        z_gen = dvae.reparameterization_cov(z_mean_gen, z_cov_gen)
        z_modified = torch.cat((z_modified, z_gen), dim=0)
        
    RT_output = RT_detector(z_modified.permute(1, 0, 2))
    RT_estimate = prob_to_RT(RT_output, tau)   
    #RT_est_sort = np.argsort(RT_estimate)
    RT_gen = RT_estimate*5*tau
    RT_true = (RT_cn[no_peak_mask]+56)*5
    
    
#     if mean_y:
#         RT_output = RT_detector(z_teach.permute(1, 0, 2))
#         RT_rec = prob_to_RT(RT_output, tau) 
#         RT_rec = RT_rec.reshape(s, n_trials)
#         RT_estimate = RT_rec.mean(1).astype(int)
#         y_mean, y_logvar = dvae.generation_x(z_teach)
#         y_pred = dvae.reparameterization(y_mean, y_logvar)
#         y_pred = y_pred.reshape(steps, s, n_trials, 96)
#         y_pred = y_pred.mean(2)
#     else:
#         z_teach= z_teach.reshape(steps, s, n_trials, z_dim)
#         z_teach = z_teach.mean(2)
#         RT_output = RT_detector(z_teach.permute(1, 0, 2))
#         RT_estimate = prob_to_RT(RT_output, tau) 
#         y_mean, y_logvar = dvae.generation_x(z_teach)
#         y_pred = dvae.reparameterization(y_mean, y_logvar)
#     y_pred = y_pred.permute(1, 0, 2).cpu().detach().numpy()
#     MUA_pred = y_pred.mean(2)
#     RT_estimate = np.argmax(MUA_pred[:, ((RT_min+56)//tau):], axis=1)
#     RT_est_sort = np.argsort(RT_estimate)

    num_bins = 25
    min_value = min(RT_true.min(), RT_gen.min())
    max_value = max(RT_true.max(), RT_gen.max())
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))
    ax1.hist(RT_true, bins=bin_edges, alpha = 0.5, color='skyblue', edgecolor='black', label = "true RT")
    ax1.axvline(RT_true.mean(), color="skyblue",linestyle="--",label= "mean true RT")
    ax1.hist(RT_gen, bins=bin_edges, alpha = 0.5, color='red', edgecolor='black', label = "gen RT")
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
    ax2.hist(RT_true, bins=bin_edges, cumulative=True, alpha = 0.5, color='skyblue', edgecolor='black', label = "true RT")
    ax2.hist(RT_gen, bins=bin_edges, cumulative=True, alpha = 0.5, color='red', edgecolor='black', label = "simulated RT")
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

# -

def true_vs_pred_zcn(comm_dict, diff_dict):
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z1_min, z1_max, z2_min, z2_max = comm_dict["z_lims"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    fig_double = comm_dict["fig_double"]
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
    
    cont_c = data["cont_cn_ordRT"]
    dir_cn = data["dir_cn_ordRT"]
    z_cn, _, _ = infer_latent(dvae, data, device, n_trials)
    
    s = z_cn.shape[1]
    steps = z_cn.shape[0]
    teacher = sim_start//(5*tau)
    alone = steps - teacher
    c = np.full(s*steps, c_norm/s)
    
    dir_cn = dir_cn.repeat(n_trials)
    
    cont_c = torch.from_numpy(cont_c).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, s, c_dim)
    z_teach = torch.from_numpy(z_cn[:teacher]).float().to(device)

    for step in range(alone):
        z_mean_gen, z_cov_gen = dvae.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher+step].unsqueeze(0))
        z_gen = dvae.reparameterization_cov(z_mean_gen, z_cov_gen)
        z_teach = torch.cat((z_teach, z_gen), dim=0)
    z_teach = z_teach.cpu().detach().numpy()
    
    fig, ax = plt.subplots(1, 2, figsize=fig_double)
    if dir_on:
        z_cn_l = z_cn[:, dir_cn==0]
        z_cn_r = z_cn[:, dir_cn==1]
        ax[0].plot(z_cn_l[:, :, 0], z_cn_l[:, :, 1], color="blue", alpha=alpha)
        ax[0].plot(z_cn_r[:, :, 0], z_cn_r[:, :, 1], color="red", alpha=alpha)
    else:
        ax[0].plot(z_cn[:, :, 0], z_cn[:, :, 1], color=color[0], alpha=alpha)
    #ax[0].set_title('Density Plot of inferred traj in latent space')
    ax[0].set_xlim(z1_min, z1_max)
    ax[0].set_ylim(z2_min, z2_max)
    ax[0].set_xticks([-1, 2, 5]) 
    ax[0].set_xticklabels([-1, 2, 5], fontsize=font_tick)  # Show corresponding labels
    ax[0].set_yticks([-2, 1, 4])
    ax[0].set_yticklabels([-2, 1, 4], fontsize=font_tick)
    ax[0].set_xlabel('z1', fontsize=font_ax)
    ax[0].set_ylabel('z2', fontsize=font_ax)
    
    # Plot the second density map with extended limits
    if dir_on:
        z_teach_l = z_teach[:, dir_cn==0]
        z_teach_r = z_teach[:, dir_cn==1]
        ax[1].plot(z_teach_l[:, :, 0], z_teach_l[:, :, 1], color="blue", alpha=alpha)
        ax[1].plot(z_teach_r[:, :, 0], z_teach_r[:, :, 1], color="red", alpha=alpha)
    else:
        ax[1].plot(z_teach[:, :, 0], z_teach[:, :, 1], color=color[1], alpha=alpha)
    #ax[1].set_title(f'Density Plot of predicted (from {sim_start}$ms$ after start) traj in latent space')
    ax[1].set_xlim(z1_min, z1_max)
    ax[1].set_ylim(z2_min, z2_max)
    ax[1].set_xticks([-1, 2, 5]) 
    ax[1].set_xticklabels([-1, 2, 5], fontsize=font_tick)  # Show corresponding labels
    ax[1].set_yticks([-2, 1, 4])
    ax[1].set_yticklabels([-2, 1, 4], fontsize=font_tick)
    ax[1].set_xlabel('z1', fontsize=font_ax)
    ax[1].set_ylabel('z2', fontsize=font_ax)
    
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(z_cn[:, :, 0], z_cn[:, :, 1], color=color[0], alpha=alpha)
    #ax.plot(z_teach[:, :, 0], z_teach[:, :, 1], color=color[1], alpha=alpha)
    #ax[0].set_title('Density Plot of inferred traj in latent space')
    ax.set_xlim(z1_min, z1_max)
    ax.set_ylim(z2_min, z2_max)
    ax.set_xticks([-1, 2, 5]) 
    ax.set_xticklabels([-1, 2, 5], fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks([-2, 1, 4])
    ax.set_yticklabels([-2, 1, 4], fontsize=font_tick)
    ax.set_xlabel('z1', fontsize=font_ax)
    ax.set_ylabel('z2', fontsize=font_ax)
    
    z_teach = z_teach.reshape((-1, 2))
    z_cn = z_cn.reshape((-1, 2))

    # Crea edges comuni per entrambi i plot
    z1_edges = np.linspace(z1_min, z1_max, bins + 1)
    z2_edges = np.linspace(z2_min, z2_max, bins + 1)

    hist_inf, z1_edges_inf, z2_edges_inf, binnumber_inf = binned_statistic_2d(
        z_cn[:, 0], z_cn[:, 1], c, 
        statistic='sum', 
        bins=[z1_edges, z2_edges]
    )

    hist_pred, z1_edges_pred, z2_edges_pred, binnumber_pred = binned_statistic_2d(
        z_teach[:, 0], z_teach[:, 1], c, 
        statistic='sum', 
        bins=[z1_edges, z2_edges]
    )
    
    vmin = 0#min(hist_inf.min(), hist_pred.min())
    vmax = 1#max(hist_inf.max(), hist_pred.max())
    
    # Create a mesh grid for plotting
    z1, z2 = np.meshgrid(z1_edges[:-1] + np.diff(z1_edges)/2, z2_edges[:-1] + np.diff(z2_edges)/2)

    # Plot the density map
    fig, ax = plt.subplots(1, 2, figsize=fig_double)
    ax[0].pcolormesh(z1, z2, 1-np.exp(-hist_inf.T), cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    #ax[0].set_title('Density Plot of inferred traj in latent space')
    ax[0].set_xlim(z1_min, z1_max)
    ax[0].set_ylim(z2_min, z2_max)
    ax[0].set_xticks([-1, 2, 5]) 
    ax[0].set_xticklabels([-1, 2, 5], fontsize=font_tick)  # Show corresponding labels
    ax[0].set_yticks([-2, 1, 4])
    ax[0].set_yticklabels([-2, 1, 4], fontsize=font_tick)
    ax[0].set_xlabel('z1', fontsize=font_ax)
    ax[0].set_ylabel('z2', fontsize=font_ax)
    
    # Plot the second density map with extended limits
    im=ax[1].pcolormesh(z1, z2, 1-np.exp(-hist_pred.T), cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    #ax[1].set_title(f'Density Plot of predicted (from {sim_start}$ms$ after start) traj in latent space')
    ax[1].set_xlim(z1_min, z1_max)
    ax[1].set_ylim(z2_min, z2_max)
    ax[1].set_xticks([-1, 2, 5]) 
    ax[1].set_xticklabels([-1, 2, 5], fontsize=font_tick)  # Show corresponding labels
    ax[1].set_yticks([-2, 1, 4])
    ax[1].set_yticklabels([-2, 1, 4], fontsize=font_tick)
    ax[1].set_xlabel('z1', fontsize=font_ax)
    ax[1].set_ylabel('z2', fontsize=font_ax)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([0, 0.5, 1])  # Specify exact tick locations
    #cbar.set_ticklabels([0, 0.5, 1])  # Custom tick labels
    cbar.set_label('n. of traj passing from that bin', fontsize=font_ax)
    cbar.ax.tick_params(labelsize=font_tick)


# +
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
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    intervals = diff_dict["intervals"]
    
    z_cn, R, q = random_latent_cn_traj(dvae, data, tau, device, n_trials)
    
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


# -

def cs_ws_stop_plot_e_all(comm_dict, diff_dict):
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    
    data = diff_dict["data"]
    
    SSD_cs = data["SSD_cs_ordSSD"]
    SSD_ws = data["SSD_ws_ordSSD"]
    steps = 256//tau
    
    _, z_ws, z_cs = infer_latent(dvae, data, device)
    
    x_cs = z_cs[(SSD_cs+56)//tau, np.arange(z_cs.shape[1]), 0]
    x_ws = z_ws[(SSD_ws+56)//tau, np.arange(z_ws.shape[1]), 0]

    y_cs = z_cs[(SSD_cs+56)//tau, np.arange(z_cs.shape[1]), 1]
    y_ws = z_ws[(SSD_ws+56)//tau, np.arange(z_ws.shape[1]), 1]
    
    z_cn, RT, q = random_latent_cn_traj(dvae, data, tau, device)
    
    print(f"traj n.{q}")
    
    x_true_story = z_cn[:, 0]
    y_true_story = z_cn[:, 1]
    x_start = z_cn[0, 0]
    y_start = z_cn[0, 1]
    x_GO = z_cn[56//tau, 0]
    y_GO = z_cn[56//tau, 1]
    x_RT = z_cn[RT, 0]
    y_RT = z_cn[RT, 1]
    
    f, ax = plt.subplots(figsize = (7, 6))

    color = np.linspace(0, 1, steps)

    # Plot the surface
    ax.scatter(x_cs, y_cs, s = 10, c = 'r', alpha = 0.8, label = "cs stops")
    ax.scatter(x_ws, y_ws, s = 10, c = 'g', alpha = 0.8, label = "ws stops")
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
    ax.set_xlabel("first latent component")
    ax.set_ylabel("second latent component")
    ax.legend()


def GO_types(dvae, data, n_trials, tau, device):
    samples_cn = data["set_cn_ordRT"].shape[0]
    samples_ws = data["set_ws_ordSSD"].shape[0]
    samples_cs = data["set_cs_ordSSD"].shape[0]
    steps = 256//tau
    
    z_cn, z_ws, z_cs = infer_latent(dvae, data, device, n_trials)
    z_dim = z_cn.shape[2]
    z_cn = z_cn.reshape(steps, samples_cn, n_trials, z_dim)
    z_ws = z_ws.reshape(steps, samples_ws, n_trials, z_dim)
    z_cs = z_cs.reshape(steps, samples_cs, n_trials, z_dim)
    
    GO_cs = z_cs[56//tau].mean(1)
    GO_ws = z_ws[56//tau].mean(1)
    GO_cn = z_cn[56//tau].mean(1)
    
    return GO_cs, GO_ws, GO_cn


def cs_ws_GO_e_all(comm_dict, diff_dict):
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    
    steps = 256//tau
    SSD_cs = data["SSD_cs_ordSSD"]
    SSD_ws = data["SSD_ws_ordSSD"]
    GO_cs, GO_ws, GO_cn = GO_types(dvae, data, n_trials, tau, device)
    z_trial, RT, q = random_latent_cn_traj(dvae, data, tau, device)
    
    print(f"traj n.{q}")
    
    x_true_story = z_trial[:, 0]
    y_true_story = z_trial[:, 1]
    x_start = z_trial[0, 0]
    y_start = z_trial[0, 1]
    x_GO = z_trial[56//tau, 0]
    y_GO = z_trial[56//tau, 1]
    x_RT = z_trial[RT, 0]
    y_RT = z_trial[RT, 1]
    
    
    fig, ax = plt.subplots(figsize = (7, 6))
    
    color = np.linspace(0, 1, steps)
    
    ax.scatter(GO_cn[:, 0], GO_cn[:, 1], c ="green", s = 15, alpha = 0.4, label = 'cs GO')
    ax.scatter(GO_ws[:, 0], GO_ws[:, 1], c ="orange", s = 15, alpha = 0.5, label = 'ws GO')
    ax.scatter(GO_cs[:, 0], GO_cs[:, 1], c ="red", s = 15, alpha = 0.5, label = 'cs GO')
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
    ax.scatter(GO_cs[:, 0], GO_cs[:, 1], c=color_cs, cmap = "Reds", s = 15, alpha = 0.7, label = 'cs GO')
    ax.scatter(GO_ws[:, 0], GO_ws[:, 1], c=color_ws, cmap = "Blues", s = 15, alpha = 0.7, label = 'ws GO')
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
    #plt.colorbar()

# +
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

def change_trial_cs_e_all(comm_dict, diff_dict):
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    move_detector = comm_dict["move_detector"]
    RT_detector = comm_dict["RT_detector"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    SSD_array = diff_dict["SSD_array"]
    
    set_cs = data["set_cs_ordSSD"]
    cont_cs = data["cont_cs_ordSSD"]
    SSD_cs = data["SSD_cs_ordSSD"]
    session_cs = data["sess_cs_ordSSD"]
    
    steps = set_cs.shape[1]
    t = np.linspace(0, 255, 256, dtype = int)
    t = t[::tau]
    
    f, ax = plt.subplots(1, 3, figsize = (18, 6))
    f.suptitle("Virtual experiment: changing correct stops trajectories to no-stop")
    
    for i, SSD in enumerate(SSD_array):     
        trial = set_cs[SSD_cs==SSD]
        cont_c = cont_cs[SSD_cs==SSD]
        session_c = session_cs[SSD_cs==SSD]
        if trial.shape[0] == 0:
            print(f"change the trial with SSD={SSD:d}")
        elif trial.shape[0] != steps:
            print("many trials")
            trial = trial[0]
            cont_c = cont_c[0]
            session_c = session_cn[0]
        else:
            trial = trial.squeeze(0)
            cont_c = cont_c.squeeze(0)
            #session_c = session_c.squeeze(0)

        trial = torch.from_numpy(trial).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
        cont_c = torch.from_numpy(cont_c).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
        session_c = dvae.session_embeddings(torch.from_numpy(session_c).long().to(device)).repeat(n_trials, 1).unsqueeze(0).expand(steps, -1, -1)

        # generation
        z, z_mean, _ = dvae.inference(trial, cont_c, session_c)
        
        teacher = ((56+SSD)//tau) - 2
        alone = steps-teacher
        z_teach = z[:teacher]
        for step in range(alone):
            z_mean_gen, z_cov_gen = dvae.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher].unsqueeze(0), session_c[0].unsqueeze(0))
            z_gen = dvae.reparameterization_cov(z_mean_gen, z_cov_gen)
            z_teach = torch.cat((z_teach, z_gen), dim=0)

        mu_z = z_teach.mean(1)
        
        move_logit = move_detector(mu_z.unsqueeze(0))
        move_output = binary_output(move_logit)
        move_pred = move_output.squeeze(0).astype(int)
        gen_string = ["does not", ""]
        RT_pred = 0
        
        if move_pred:
            RT_output = RT_detector(mu_z.unsqueeze(0))
            RT_pred = prob_to_RT(RT_output, tau)
            RT_pred = RT_pred.squeeze(0)
            gen_string = ["", f" with RT = {RT_pred*tau*5}ms"]

#         mask_nan = ~np.isnan(RT_pred)
#         RT_pred = RT_pred[mask_nan]  
#         RT_cn_ordered_filt = RT_cn_ordered_filt[mask_nan]

        mu_z = mu_z.cpu().detach().numpy()
        mu_x = mu_z[:, 0]
        mu_y = mu_z[:, 1]

        ax[i].plot(mu_x, mu_y, c ="g", label = f'Fake trial, detected RT={RT_pred*5*tau}$ms$')
        n_arrows = 15
        arrow_indices = np.arange(0, len(mu_x), len(mu_x)//n_arrows)  # Place n_arrows arrows along the path
        for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
            dx = mu_x[k+1] - mu_x[k]
            dy = mu_y[k+1] - mu_y[k]
            ax[i].arrow(mu_x[k], mu_y[k], dx, dy,
                    head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
            
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
        x_true_story = true_trial[:, 0]
        y_true_story = true_trial[:, 1]
        ax[i].plot(x_true_story, y_true_story, linewidth=2, c = "r", label = f'True traj, SSD={((SSD+56)*5):d}$ms$, detected RT={RT_rec*5*tau}$ms$') 
        n_arrows = 15
        arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
        for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
            dx = x_true_story[k+1] - x_true_story[k]
            dy = y_true_story[k+1] - y_true_story[k]
            ax[i].arrow(x_true_story[k], y_true_story[k], dx, dy,
                    head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
        #ax[i].set_title(f"Virtual experiment: changing correct stops trajectories to no stop")#the smallest STOP")
        ax[i].set_xlabel("first latent component")
        ax[i].set_ylabel("second latent component")
        ax[i].legend()


# def change_strange_cs(comm_dict, diff_dict):

#     dvae = comm_dict["dvae"]
#     device = comm_dict["device"]
#     tau = comm_dict["tau"]
#     z_dim = comm_dict["z_dim"]
#     t = comm_dict["t"]
#     RT_detector = comm_dict["RT_detector"]

#     data = diff_dict["data"]
#     n_trials = diff_dict["n_trials"]
#     SSD_array = diff_dict["SSD_array"]
#     lim_array = diff_dict["lim_array"]

#     #set_cs = data["set_cs_ordSSD"]
#     cont_cs = data["cont_cs_ordSSD"]
#     SSD_cs = data["SSD_cs_ordSSD"]
#     samples_cs = cont_cs.shape[0]
#     steps = cont_cs.shape[1]

#     z_cn, z_ws, z_cs = infer_latent(dvae, data, device, n_trials)
#     z_cs = z_cs.reshape(steps, samples_cs, n_trials, z_dim)
#     z_cs_mean = z_cs.mean(2)
#     z_GO = z_cs_mean[56//tau]

#     mask_x = (z_GO[:, 0] < lim_array[1]).astype("bool") & (z_GO[:, 0] > lim_array[0]).astype("bool") 
#     mask_y = (z_GO[:, 1] < lim_array[3]).astype("bool") & (z_GO[:, 1] > lim_array[2]).astype("bool")
#     mask_all = mask_x & mask_y

#     z_cs_masked = z_cs[:, mask_all] 
#     cont_cs = cont_cs[mask_all]
#     SSD_cs = SSD_cs[mask_all]
#     n_strange = z_cs_masked.shape[1]
#     q = random.randint(0, n_strange - 1)
#     print(f"trial n.{q} out of {n_strange}")

#     z = z_cs_masked[:, q]
#     cont_c = cont_cs[q]
#     SSD = SSD_cs[q]

#     z = torch.from_numpy(z).float().to(device)
#     cont_c = torch.from_numpy(cont_c).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)


#     f, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))

#     teacher = ((56+SSD)//tau) - 2
#     alone = steps-teacher
#     z_teach = z[:teacher]
#     for step in range(alone):
#         z_mean_gen, z_cov_gen = dvae.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher].unsqueeze(0))
#         z_gen = dvae.reparameterization_cov(z_mean_gen, z_cov_gen)
#         z_teach = torch.cat((z_teach, z_gen), dim=0)

#     mu_z = z_teach.mean(1)
#     RT_prob = RT_detector(mu_z.unsqueeze(1))
#     RT_prob = RT_prob.squeeze(1).cpu().detach().numpy()
#     RT_pred_arr = np.where(RT_prob>0.5)[0]
#     if RT_pred_arr.any():
#         RT_pred = RT_pred_arr[0]
#     else:
#         RT_pred = 0

#     mu_z = mu_z.cpu().detach().numpy()
#     mu_x = mu_z[:, 0]
#     mu_y = mu_z[:, 1]

#     ax1.plot(mu_x, mu_y, c ="g", label = f'no stop trial')
#     n_arrows = 15
#     arrow_indices = np.arange(0, len(mu_x), len(mu_x)//n_arrows)  # Place n_arrows arrows along the path
#     for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
#         dx = mu_x[i+1] - mu_x[i]
#         dy = mu_y[i+1] - mu_y[i]
#         ax1.arrow(mu_x[i], mu_y[i], dx, dy,
#                 head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
#     '''
#     x_points = z_teach[t, :, 0]
#     y_points = z_teach[t, :, 1]

#     # Calculate ellipse parameters
#     width, height, angle = get_correlation_ellipse_params(x_points, y_points)

#     ellipse = Ellipse(xy=(mean_trajectory[t, 0], mean_trajectory[t, 1]),
#                      width=width,
#                      height=height,
#                      angle=angle,
#                      alpha=0.1,
#                      color='gray')
#     ax.add_patch(ellipse)
#     '''

#     true_trial = z.mean(1)
#     RT_prob_true = RT_detector(true_trial.unsqueeze(1))
#     RT_prob_true = RT_prob_true.squeeze(1).cpu().detach().numpy()
#     true_trial = true_trial.cpu().detach().numpy()
#     x_true_story = true_trial[:, 0]
#     y_true_story = true_trial[:, 1]
#     ax1.plot(x_true_story, y_true_story, linewidth=2, c = "r", label = f'True traj, SSD={((SSD+56)*5):d}$ms$') 
#     n_arrows = 15
#     arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
#     for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
#         dx = x_true_story[i+1] - x_true_story[i]
#         dy = y_true_story[i+1] - y_true_story[i]
#         ax1.arrow(x_true_story[i], y_true_story[i], dx, dy,
#                 head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
#     ax1.set_title(f"Virtual experiment: changing correct stops trajectories to no stop")#the smallest STOP")
#     ax1.set_xlabel("first latent component")
#     ax1.set_ylabel("second latent component")
#     ax1.legend()

#     ax2.plot(t*5, RT_prob_true, linewidth=2, c = "r", label = f'True traj, SSD={((SSD+56)*5):d}$ms$') 
#     ax2.plot(t*5, RT_prob, linewidth=2, c = "g", label = f'no stop trial, estimate RT={RT_pred*5*tau}$ms$') 
#     ax2.axvline((SSD+56)*5, color="r",linestyle="--",label= "SSD")
#     ax2.axvline(56*5, color="black",linestyle="--",label= "GO")
#     ax2.set_title(f"Virtual RT change")#the smallest STOP")
#     ax2.set_xlabel("time ($ms$)")
#     ax2.set_ylabel("probability that a movement occured")
#     ax2.legend()


def change_trial_cn_e_all(comm_dict, diff_dict):
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    RT_detector = comm_dict["RT_detector"]
    move_detector = comm_dict["move_detector"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    stop_array = diff_dict["stop_array"]
    color = diff_dict["color"]
    RT_array = diff_dict["RT_array"]
    n_cont = diff_dict["n_cont"]
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    set_cs = data["set_cs_ordSSD"]
    session_cn = data["sess_cn_ordRT"]
    
    steps = set_cn.shape[1]
    l = len(stop_array)
    t = np.linspace(0, 255, 256, dtype = int)
    t = t[::tau]
    
    f, ax = plt.subplots(1, 3, figsize = (15, 5))
    f.suptitle("Virtual experiment: changing correct stops trajectories to no-stop")
    
    for i, RT in enumerate(RT_array):     
        trial = set_cn[RT_cn==RT]
        cont_c = cont_cn[RT_cn==RT]
        session_c = session_cn[RT_cn==RT]
        if trial.shape[0] == 0:
            print(f"change the trial with RT={RT:d}")
        elif trial.shape[0] != steps:
            trial = trial[0]
            cont_c = cont_c[0]
        else:
            trial = trial.squeeze(0)
            cont_c = cont_c.squeeze(0)

#         if n_cont == 3:
#             cont = torch.zeros((1, n_trials, 2)).to(device)
#         else:
#             cont = torch.ones((1, n_trials, 2)).to(device)

        cont = torch.zeros((1, n_trials, 4)).to(device)
        cont[:, :, 3] = 1

        trial = torch.from_numpy(trial).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
        cont_c = torch.from_numpy(cont_c).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
        session_c = dvae.session_embeddings(torch.from_numpy(session_c).long().to(device)).repeat(n_trials, 1).unsqueeze(0).expand(steps, -1, -1)
        
        # generation
        z, z_mean, _ = dvae.inference(trial, cont_c, session_c)
        
        for j, stop in enumerate(stop_array):   
            teacher = stop//(5*tau)
            alone = steps-teacher
            z_teach = z[:teacher]
            for step in range(alone):
                z_mean_gen, z_cov_gen = dvae.generation_z(z_teach[-1].unsqueeze(0), cont, session_c[0].unsqueeze(0))
                z_gen = dvae.reparameterization_cov(z_mean_gen, z_cov_gen)
                z_teach = torch.cat((z_teach, z_gen), dim=0)
                
            mu_z = z_teach.mean(1)
            
            move_logit = move_detector(mu_z.unsqueeze(0))
            move_output = binary_output(move_logit)
            move_pred = move_output.squeeze(0).astype(int)
            RT_pred = 0

            if move_pred:
                RT_output = RT_detector(mu_z.unsqueeze(0))
                RT_pred = prob_to_RT(RT_output, tau)
                RT_pred = RT_pred.squeeze(0)
            
            mu_z = mu_z.cpu().detach().numpy()
            mu_x = mu_z[:, 0]
            mu_y = mu_z[:, 1]
            
            ax[i].plot(mu_x, mu_y, c=color[j], label = f'Fake traj, detected RT={RT_pred*5*tau}$ms$')
            #ax[i].plot(t*5, RT_prob, linewidth=2, c = color[j], label = f'estimate RT={RT_pred*5*tau}$ms$') 
            n_arrows = 15
            arrow_indices = np.arange(0, len(mu_x), len(mu_x)//n_arrows)  # Place n_arrows arrows along the path
            for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
                dx = mu_x[k+1] - mu_x[k]
                dy = mu_y[k+1] - mu_y[k]
                ax[i].arrow(mu_x[k], mu_y[k], dx, dy,
                        head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
        
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
        x_true_story = true_trial[:, 0]
        y_true_story = true_trial[:, 1]
        ax[i].plot(x_true_story, y_true_story, linewidth=2, c = "black", label = f'True traj, true RT={((RT+56)*5):d}$ms$, Detected RT={RT_rec*5*tau}$ms$')  
        n_arrows = 15
        arrow_indices = np.arange(0, len(x_true_story), len(x_true_story)//n_arrows)  # Place n_arrows arrows along the path
        for k in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
            dx = x_true_story[k+1] - x_true_story[k]
            dy = y_true_story[k+1] - y_true_story[k]
            ax[i].arrow(x_true_story[k], y_true_story[k], dx, dy,
                    head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
        #ax[i].set_title(f"Virtual experiment: changing no stops trajectories for different stops signals")#the smallest STOP")
        ax[i].set_xlabel("first latent component")
        ax[i].set_ylabel("second latent component")
        ax[i].legend()



def examinate_traj(dvae, data, type_trial, q, axes, tau, device):
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
    
    z, z_mean, _ = dvae.inference(set_c, cont)
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

# +
import math

def gen_field_e_all(comm_dict, diff_dict):

    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    c_dim = comm_dict["c_dim"]
    
    data = diff_dict["data"]
    points = diff_dict["points"]
    cont_type = diff_dict["cont_type"]
    session_type = diff_dict["session_type"]
    stop_trial = diff_dict["stop_trial"]
    
    cont_cn = data["cont_cn_ordRT"]
    cont_cs = data["cont_cs_ordSSD"]
    dir_cn = data["dir_cn_ordRT"]
    
    x_lim_l = -5.5
    x_lim_r = 7
    y_lim_l = -5
    y_lim_r = 10
    #z_lim_l = -10
    #z_lim_r = 10

    x_points = np.linspace(x_lim_l, x_lim_r, points)   # punti per freccie generiche
    y_points = np.linspace(y_lim_l, y_lim_r, points)

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
    
    session = dvae.session_embeddings(torch.Tensor(session_type).long().to(device)).unsqueeze(0).expand(points_z.shape[0], -1)

    z_mean, _ = dvae.generation_z(points_z, cont_z, session)
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
        z_trial, SSD, q = random_latent_cs_traj(dvae, data, tau, device)
        x_SSD = z_trial[SSD, 0]
        y_SSD = z_trial[SSD, 1]
    else:
        z_trial, RT, q = random_latent_cn_traj(dvae, data, tau, device)
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


# -

def rec_prove(comm_dict, diff_dict):
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    
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
    
    y_mean, y_logvar = dvae(trial, cont_c)
    y_inf = dvae.reparameterization(y_mean, y_logvar)
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
            z, _, _ = dvae.inference(trial, cont_c)
            y_mu, y_logv = dvae.generation_x(z.mean(1))
            y_infer = dvae.reparameterization(y_mu, y_logv)
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


def consistent_sim_e_all(comm_dict, diff_dict):
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    RT_detector = comm_dict["RT_detector"]
    font_ax = comm_dict["font_ax"]
    c_dim = comm_dict["c_dim"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    sim_start = diff_dict["sim_start"]
    cmap = diff_dict["cmap"]
    n_ticks = diff_dict["n_ticks"]
    mean_y = diff_dict["mean_y"]
    fig_double = diff_dict["fig_double"]
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    session_cn = data["sess_cn_ordRT"]
    RT_min = RT_cn.min()
    
    n_samples, steps, features = set_cn.shape
    teacher = sim_start//(5*tau)
    alone = steps - teacher
    
    #MUA_true = test_set.mean(2)

#     set_cn = torch.from_numpy(test_set).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, n_samples*n_trials, 96)
    cont_c = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    
    data_set = torch.from_numpy(set_cn).float().permute(1, 0, 2)
    cont_set = torch.from_numpy(cont_cn).float().permute(1, 0, 2)
    session_set = dvae.session_embeddings(torch.from_numpy(session_cn).long().to(device)).unsqueeze(0).expand(steps, -1, -1) #steps x s x s_dim
    session_cn_flat = session_set.repeat_interleave(n_trials, dim=1)
    
    chunk_size=10
    # Output finale
    z_cn = torch.zeros(steps, n_samples*n_trials, z_dim).to(device)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        batch_size = end - start

        # Estrai chunk
        x_chunk = data_set[:, start:end, :].repeat_interleave(n_trials, dim=1)
        c_chunk = cont_set[:, start:end, :].repeat_interleave(n_trials, dim=1)
        s_chunk = session_set[:, start:end, :].repeat_interleave(n_trials, dim=1)
        
        # Porta su GPU
        x_chunk = x_chunk.to(device)
        c_chunk = c_chunk.to(device)
        s_chunk = s_chunk.to(device)
        
        # Inferenza
        with torch.no_grad():
            z, z_mean, _ = dvae.inference(x_chunk, c_chunk, s_chunk)

        # Inserisci nel buffer
        z_cn[:, start*n_trials:end*n_trials, :] = z

        torch.cuda.empty_cache()

   
    z_teach = z_cn[:teacher]
    no_peak = z_teach[-1, :, 0]>-10

    for step in range(alone):
        z_mean_gen, z_cov_gen = dvae.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher+step].unsqueeze(0), session_cn_flat[0].unsqueeze(0))
        z_gen = dvae.reparameterization_cov(z_mean_gen, z_cov_gen)
        z_teach = torch.cat((z_teach, z_gen), dim=0)

#     y_mean, y_logvar = dvae.generation_x(z_teach)
    z_teach = z_teach.permute(1, 0, 2)
    RT_output = RT_detector(z_teach[no_peak])
    RT_estimate = prob_to_RT(RT_output, tau)    

#     if mean_y:
#         RT_output = RT_detector(z_teach.permute(1, 0, 2))
#         RT_rec = prob_to_RT(RT_output, tau) 
#         RT_rec = RT_rec.reshape(s, n_trials)
#         RT_estimate = RT_rec.mean(1).astype(int)
#         y_mean, y_logvar = dvae.generation_x(z_teach)
#         y_pred = dvae.reparameterization(y_mean, y_logvar)
#         y_pred = y_pred.reshape(steps, s, n_trials, 96)
#         y_pred = y_pred.mean(2)
#     else:
#         z_teach= z_teach.reshape(steps, s, n_trials, z_dim)
#         z_teach = z_teach.mean(2)
#         RT_output = RT_detector(z_teach.permute(1, 0, 2))
#         RT_estimate = prob_to_RT(RT_output, tau) 
#         y_mean, y_logvar = dvae.generation_x(z_teach)
#         y_pred = dvae.reparameterization(y_mean, y_logvar)

#     y_pred = dvae.reparameterization(y_mean, y_logvar)
#     y_pred = y_pred.permute(1, 0, 2)
#     y_pred = y_pred[no_peak].cpu().detach().numpy()
#     MUA_pred = y_pred.mean(2)
    #RT_estimate = np.argmax(MUA_pred[:, ((RT_min+56)//tau):], axis=1)
#     RT_est_sort = np.argsort(RT_estimate)
#     MUA_pred = MUA_pred[RT_est_sort]
#     print(MUA_pred.shape[0])
#     print(s)

#     vmin = -3#min(MUA_pred.min(), MUA_true.min())
#     vmax = 3#max(MUA_pred.max(), MUA_true.max())

#     y_positions = np.arange(s)
#     x_positions = np.arange(256//tau) 
#     x_positions = x_positions[56//tau:]

#     x_labels = x_positions * 20 
#     y_labels = y_positions

#     x_ticks = len(x_positions)
#     y_ticks = s

#     x_start = 56//tau
#     x_end = 256//tau
#     y_start = 0
#     y_end = n_samples

#     n_xticks, n_yticks = n_ticks

#     fig, ax = plt.subplots(1, 2, figsize = fig_double)

    #fig.text(0.14, 0.92, 'Comparison of True vs Generated trials ordered w.r.t. RT', rotation=0, size=15, fontweight='bold')

#     im1 = ax[0].imshow(MUA_true, extent=[x_start, x_end, y_start, y_end], cmap = cmap, aspect='auto', vmin=vmin, vmax=vmax)
#     ax[0].set_xticks(x_positions[::x_ticks//n_xticks]) 
#     ax[0].set_xticklabels(x_labels[::x_ticks//n_xticks], fontsize=font_tick)  # Show corresponding labels
#     ax[0].set_yticks([s//2, s])
#     ax[0].set_yticklabels([s//2, s], fontsize=font_tick)
#     ax[0].set_xlabel('time from start ($ms$)', fontsize=font_ax)
#     ax[0].set_ylabel('Real trials ordered by RT', fontsize=font_ax)
#     im2 = ax[1].imshow(MUA_pred, extent=[x_start, x_end, y_start, y_end], cmap = cmap, aspect='auto', vmin=vmin, vmax=vmax)
#     #ax[1].axvline(56//tau, color="g",linestyle="--",label=" Simulation start (GO)")
#     ax[1].set_xticks(x_positions[::x_ticks//n_xticks]) 
#     ax[1].set_xticklabels(x_labels[::x_ticks//n_xticks], fontsize=font_tick)  # Show corresponding labels
#     ax[1].set_yticks([s//2, s])
#     ax[1].set_yticklabels([s//2, s], fontsize=font_tick)
#     ax[1].set_xlabel('time from start ($ms$)', fontsize=font_ax)
#     ax[1].set_ylabel('Generated trials ordered by RT', fontsize=font_ax)
#     cbar=plt.colorbar(im1, ax=ax)
#     cbar.ax.tick_params(labelsize=font_tick)

    import seaborn as sns
    from scipy.stats import ks_2samp, wasserstein_distance
    
    RT_cn = RT_cn.repeat(n_trials)
    RT_true = (RT_cn[no_peak.cpu().detach().numpy()]+56)*5
    #RT_pred = ((RT_estimate*tau)+56+RT_min)*5
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
    
    num_bins = 25
    min_value = min(RT_true.min(), RT_pred.min())
    max_value = max(RT_true.max(), RT_pred.max())
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))
    ax1.hist(RT_true, bins=bin_edges, alpha = 0.5, color='skyblue', edgecolor='black', label = "true RT")
    ax1.hist(RT_pred, bins=bin_edges, alpha = 0.5, color='red', edgecolor='black', label = "simulated RT")
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
    ax1.set_title(f"Histograms of true and predicted RTs from {sim_start}$ms$")
    ax1.legend(fontsize=font_leg)
    
    #fig, ax = plt.subplots(figsize = fig_size)
    ax2.hist(RT_true, bins=bin_edges, cumulative=True, alpha = 0.5, color='skyblue', edgecolor='black', label = "true RT")
    ax2.hist(RT_pred, bins=bin_edges, cumulative=True, alpha = 0.5, color='red', edgecolor='black', label = "simulated RT")
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
    ax2.set_title(f"Cumulative Histograms of true and predicted RTs from {sim_start}$ms$")
    ax2.legend(fontsize=font_leg)


def latent_diff_RT(comm_dict, diff_dict):
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z1_min, z1_max, z2_min, z2_max = comm_dict["z_lims"]
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
    ax.set_xticks([-1, 2, 5]) 
    ax.set_xticklabels([-1, 2, 5], fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks([-2, 1, 4])
    ax.set_yticklabels([-2, 1, 4], fontsize=font_tick)
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

        z, _, _ = dvae.inference(trial, cont_c)
        mean_z = z.mean(1)
    
        z_mean = mean_z[56//tau:].cpu().detach().numpy()
        z = z[56//tau:].cpu().detach().numpy()
        #z = z.reshape(-1, 2)
        
        #x_start = z_mean[0, 0]
        #y_start = z_mean[0, 1]
        x_GO = z_mean[0, 0]
        y_GO = z_mean[0, 1]
        x_RT = z_mean[RT//tau, 0]
        y_RT = z_mean[RT//tau, 1]
        #x_SSD = z_mean[SSD, 0]
        #y_SSD = z_mean[SSD, 1]
    
        z1_edges = np.linspace(z1_min, z1_max, bins + 1)
        z2_edges = np.linspace(z2_min, z2_max, bins + 1)

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
        ax.plot(z[:, :, 0], z[:, :, 1], c=color[i], alpha=alpha)
        ax.plot(z_mean[:, 0], z_mean[:, 1], c=color[i], linewidth=3)
        # Add arrows along the mean trajectory
        n_arrows = 12
        arrow_indices = np.arange(0, len(z_mean), len(z_mean)//n_arrows)  # Place n_arrows arrows along the path
        for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
            dx = z_mean[i+1, 0] - z_mean[i, 0]
            dy = z_mean[i+1, 1] - z_mean[i, 1]
            ax.arrow(z_mean[i, 0], z_mean[i, 1], dx, dy,
                    head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
        #ax.fill_between(t*5, MUA_mean - MUA_std, MUA_mean + MUA_std, edgecolor = 'none', color = 'grey', alpha = 0.3)
        ax.scatter(x_GO, y_GO, s = 100, c = "black", marker='o', alpha = 1)# label = "GO")
        ax.scatter(x_RT, y_RT, s = 150, c = 'black', marker='*', alpha = 1)#, label = "RT")
        ax.set_xlim(z1_min, z1_max)
        ax.set_ylim(z2_min, z2_max)
        #ax.plot(x_SSD, y_SSD, color='red', marker='x', markeredgewidth = 2, markersize = 10, linestyle='None', label = "SSD")
        #ax.scatter(x_SSD, y_SSD, s = 80, c = 'r', alpha = 1, label = "SSD")
        #ax.scatter(x_start, y_start, s = 80, c = 'y', alpha = 1, label = "start")
        """cbar[i] = plt.colorbar(im[i], cax=cax)
        cbar[i].set_ticks([0, 0.5, 1])  # Specify exact tick locations
        #cbar.set_ticklabels([0, 0.5, 1])  # Custom tick labels
        cbar[i].set_label('n. of traj passing from that bin', fontsize=font_ax)
        cbar[i].ax.tick_params(labelsize=font_tick)"""


def enc_dec_show(comm_dict, diff_dict):
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z1_min, z1_max, z2_min, z2_max = comm_dict["z_lims"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    cmap = diff_dict["cmap"]
    markersize = diff_dict["markersize"]
    markeredgewidth = diff_dict["markeredgewidth"]
    leg = diff_dict["leg"]
    
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
    z, _, _ = dvae.inference(trial, cont_c)
    mean_z = z.mean(1)
    
    y_mean, y_logvar = dvae.generation_x(mean_z)
    y_pred = dvae.reparameterization(y_mean, y_logvar)
    y_pred = y_pred.cpu().detach().numpy()
    z_mean = mean_z.cpu().detach().numpy()
    z = z.cpu().detach().numpy()
    MUA_inf = y_pred
    
    x_start = z_mean[0, 0]
    y_start = z_mean[0, 1]
    x_GO = z_mean[56//tau, 0]
    y_GO = z_mean[56//tau, 1]
    x_RT = z_mean[RT, 0]
    y_RT = z_mean[RT, 1]
    x_SSD = z_mean[SSD, 0]
    y_SSD = z_mean[SSD, 1]
    
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
    
    ax.plot(z[:, :, 0], z[:, :, 1], c ='grey', alpha=0.05)
    ax.plot(z_mean[:, 0], z_mean[:, 1], c ='brown', linewidth=3)
    # Add arrows along the mean trajectory
    n_arrows = 10
    arrow_indices = np.arange(0, len(z_mean), len(z_mean)//n_arrows)  # Place n_arrows arrows along the path
    for i in arrow_indices[:-1]:  # Exclude last point to avoid arrow at the end
        dx = z_mean[i+1, 0] - z_mean[i, 0]
        dy = z_mean[i+1, 1] - z_mean[i, 1]
        ax.arrow(z_mean[i, 0], z_mean[i, 1], dx, dy,
                head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
    #ax.fill_between(t*5, MUA_mean - MUA_std, MUA_mean + MUA_std, edgecolor = 'none', color = 'grey', alpha = 0.3)
    ax.scatter(x_GO, y_GO, s = 100, c = "black", marker='o', alpha = 1, label = "GO")
    ax.scatter(x_RT, y_RT, s = 150, c = "black", marker='*', alpha = 1, label = "RT")
    ax.plot(x_SSD, y_SSD, color='black', marker='x', markeredgewidth = 3, markersize = 10, linestyle='None', label = "SSD")
    #ax.scatter(x_SSD, y_SSD, s = 80, c = 'r', alpha = 1, label = "SSD")
    #ax.scatter(x_start, y_start, s = 80, c = 'y', alpha = 1, label = "start")
    ax.set_xlim(z1_min, z1_max)
    ax.set_ylim(z2_min, z2_max)
    ax.set_xlabel("z1", fontsize=font_ax)
    ax.set_ylabel("z2", fontsize=font_ax)
    ax.set_xticks([-1, 2, 5]) 
    ax.set_xticklabels([-1, 2, 5], fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks([-2, 1, 4])
    ax.set_yticklabels([-2, 1, 4], fontsize=font_tick)
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
    
    dvae = comm_dict["dvae"]
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

    y_mean, y_logvar = dvae(test_cn, cont_c)
    y_pred = dvae.reparameterization(y_mean, y_logvar)
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
    
    dvae = comm_dict["dvae"]
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

    y_mean, y_logvar = dvae(test_cn, cont_c)
    y_pred = dvae.reparameterization(y_mean, y_logvar)
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
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    t = comm_dict["t"]
    z1_min, z1_max, z2_min, z2_max = comm_dict["z_lims"]
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
    
    z, _, _ = dvae.inference(trial, cont_c)
    z_noisy, _, _ = dvae.inference(noisy_trial, cont_c)
    
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
    ax.set_xlim(z1_min, z1_max)
    ax.set_ylim(z2_min, z2_max)
    ax.set_xlabel("z1", fontsize=font_ax)
    ax.set_ylabel("z2", fontsize=font_ax)
    ax.set_xticks([-1, 2, 5]) 
    ax.set_xticklabels([-1, 2, 5], fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks([-2, 1, 4])
    ax.set_yticklabels([-2, 1, 4], fontsize=font_tick)
    ax.set_title(f'Comparison of the same mean latent trajectory with and without noise (module={noise_level}) in the observation')
    if leg:
        ax.legend(loc='best', fontsize=font_leg)

# +
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def compute_stat_prob(comm_dict, diff_dict):
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    z1_min, z1_max, z2_min, z2_max = comm_dict["z_lims"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    
    data = diff_dict["data"]
    points = diff_dict["points"]
    n_neighbours = diff_dict["n_neighbours"]
    cont_type = diff_dict["cont_type"]
    alpha = diff_dict["alpha"]
    #min_points_in_bin = diff_dict["min_points_in_bin"]
    #max_frac_val = diff_dict["max_frac_val"]
    
    cont_cn = data["cont_cn_ordRT"]
    cont_cs = data["cont_cs_ordSSD"]
    dir_cn = data["dir_cn_ordRT"]
    
    x_lim_l = z1_min #-1
    x_lim_r = z1_max # 5
    y_lim_l = z2_min #-4
    y_lim_r = z2_max # 4
    
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
    means, covariances = dvae.generation_z(z_centers, cont_z)
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
    plt.xlim(z1_min, z1_max)
    plt.ylim(z2_min, z2_max)
    plt.xticks([-1, 2, 5], fontsize=font_tick)
    plt.yticks([-2, 1, 4], fontsize=font_tick)
    plt.title("Distribuzione stazionaria della dinamica")
    plt.show()
    
    # Plot della distribuzione stazionaria
    plt.figure(figsize=(fig_size[0]+2, fig_size[1]))
    plt.contourf(X, Y, E_grid, cmap="viridis")
    plt.colorbar(label="- Log Probabilità stazionaria")
    plt.xlabel('z1', fontsize=font_ax)
    plt.ylabel('z2', fontsize=font_ax)
    plt.xlim(z1_min, z1_max)
    plt.ylim(z2_min, z2_max)
    plt.xticks([-1, 2, 5], fontsize=font_tick)
    plt.yticks([-2, 1, 4], fontsize=font_tick)
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
    plt.xticks([-1, 2, 5], fontsize=font_tick)
    plt.yticks([-2, 1, 4], fontsize=font_tick)
    plt.colorbar()


def fraction_contour(comm_dict, diff_dict):

    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    t = comm_dict["t"]
    z1_min, z1_max, z2_min, z2_max = comm_dict["z_lims"]
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
    #vector_density = diff_dict["vector_density"]    
    
    cont_cn = data["cont_cn_ordRT"]
    cont_cs = data["cont_cs_ordSSD"]
    dir_cn = data["dir_cn_ordRT"]
    
    x_lim_l = z1_min - 0.5 #-5.5
    x_lim_r = z1_max + 0.5 #7
    y_lim_l = z2_min - 0.5 #-5
    y_lim_r = z2_max + 0.5 #10
    #z_lim_l = -10
    #z_lim_r = 10
    
    # CREO LA MASCHERA DEL CONTOUR PLOT RELATIVA AI PUNTI DEL DATASET
    
    z_cn, _, _ = infer_latent(dvae, data, device, n_trials=n_trials)
    steps, s, _ = z_cn.shape
    z = z_cn.reshape(-1, z_dim) 
    
    
    RT_cn = data["RT_cn_ordRT"]
    samples = len(RT_cn)
    RT_cn_rep = np.expand_dims(RT_cn, axis=1)
    RT_cn_rep = RT_cn_rep.repeat(n_trials, axis=1).reshape(-1)

    z1_edges = np.linspace(z1_min, z1_max, n_bins + 1)
    z2_edges = np.linspace(z2_min, z2_max, n_bins + 1)

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
    z_mean, z_cov = dvae.generation_z(points_z, cont_z)
    shift = (z_mean - points_z).cpu().detach().numpy()
    z_cov = z_cov.cpu().detach().numpy()
    print(z_cov.shape)
    
    
    total_fraction = np.array([compute_fraction(cov) for cov in z_cov]).reshape(X.shape)
    mask = (hist >= min_points_in_bin) & (total_fraction <= max_frac_val)
    
    z, RT, q = random_latent_cn_traj(dvae, data, tau, device)
    
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
    plot_contour(xedges, yedges, hist_masked, f'Histogram of RT values of the z_GO', z1_min, z1_max, z2_min, z2_max, font_ax, font_tick)
    
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
    plot_contour(xedges, yedges, hist_masked, f'Rapporto tra modulo eigenvalues - {cont_type}', z1_min, z1_max, z2_min, z2_max, font_ax, font_tick)
    plt.show()
    
    shift_std_ratio = compute_shift_std_ratio(shift, z_cov).reshape(X.shape)
    hist_masked = np.ma.masked_where(~mask, shift_std_ratio)
    plt.figure(figsize=(fig_size[0]+2, fig_size[1]))
    plot_contour(xedges, yedges, hist_masked, f'Rapporto tra modulo eigenvalues - {cont_type}', z1_min, z1_max, z2_min, z2_max, font_ax, font_tick)
    plt.show()
    
    norm_shift, norm_std = compute_alignement_vec(shift, z_cov)
    cosine_similarities = np.sum(norm_shift * norm_std, axis=1)
    alignment_grid = np.absolute(cosine_similarities.reshape(X.shape))
    hist_masked = np.ma.masked_where(~mask, alignment_grid)
    plt.figure(figsize=(fig_size[0]+2, fig_size[1]))
    plot_contour(xedges, yedges, hist_masked, f'Rapporto tra modulo eigenvalues - {cont_type}', z1_min, z1_max, z2_min, z2_max, font_ax, font_tick)
    
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
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z1_min, z1_max, z2_min, z2_max = comm_dict["z_lims"]
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

    z_cn, _, _ = dvae.inference(set_cn, cont_cn)
    z_teach = z_cn[:teacher]

    for step in range(alone):
        z_mean_gen, z_cov_gen = dvae.generation_z(z_teach[-1].unsqueeze(0), cont_cn[teacher+step].unsqueeze(0))
        z_gen = dvae.reparameterization_cov(z_mean_gen, z_cov_gen)
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
        y_mean, y_logvar = dvae.generation_x(z_teach)
        y_pred = dvae.reparameterization(y_mean, y_logvar)
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
    
    z, RT, q = random_latent_cn_traj(dvae, data, tau, device)
    
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
    ax.set_xlim(z1_min, z1_max)
    ax.set_ylim(z2_min, z2_max)
    ax.set_xlabel("z1", fontsize=font_ax)
    ax.set_ylabel("z2", fontsize=font_ax)
    ax.set_xticks([-1, 2, 5]) 
    ax.set_xticklabels([-1, 2, 5], fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks([-2, 1, 4])
    ax.set_yticklabels([-2, 1, 4], fontsize=font_tick)
    #ax.set_title(f'Example of ws trajectory')
    if leg:
        ax.legend(loc='best', fontsize=font_leg)

# !pip install statsmodels

# +
from statsmodels.tsa.stattools import acf
#import seaborn as sns

def autocorr_e_all(comm_dict, diff_dict):
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    leg = diff_dict["leg"]
    max_lag = diff_dict["max_lag"]
    mean_z = diff_dict["mean_z"]
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    session_cn = data["sess_cn_ordRT"]
    
    s, steps, features = set_cn.shape
    
    descr = ["traj vera", "traj rec", "differenza tra traj vera e rec"]
    
    cont_c = torch.from_numpy(cont_c).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    session_cn_embed = dvae.session_embeddings(torch.from_numpy(session_cn).long().to(device)).unsqueeze(0).expand(steps, -1, -1) #steps x s x s_dim
    session_cn_rep = session_cn_embed.repeat_interleave(n_trials, dim=1) #steps x s*n_trials x s_dim

    z_cn, _, _ = infer_latent(dvae, data, device, n_trials=n_trials)
    z_cn = torch.from_numpy(z_cn).float().to(device)
    
    if mean_z:
        if n_trials>1:
            z_cn = z_cn.reshape(steps, s, n_trials, z_dim)
            z_cn = z_cn.mean(2)
        y_mean, y_logvar = dvae.generation_x(z_cn, session_cn_embed)
        y_pred = dvae.reparameterization(y_mean, y_logvar)
    else:
        y_mean, y_logvar = dvae.generation_x(z_cn, session_cn_rep)
        y_pred = dvae.reparameterization(y_mean, y_logvar)
        y_pred= y_pred.reshape(steps, s, n_trials, 96)
        y_pred = y_pred.mean(2)
    y_rec = y_pred.permute(1, 0, 2).cpu().detach().numpy()

    diff_y = set_cn - y_rec

    for d, data in enumerate([set_cn, y_rec, diff_y]):
        # Calcoliamo l'autocorrelazione per ogni serie e dimensione
        all_autocorrs = np.zeros((s, 96, max_lag + 1))
        mean_autocorr = np.zeros(max_lag + 1)

        for i in range(s):
            for j in range(96):
                series = data[i, :, j]
                # Calcoliamo l'autocorrelazione fino al lag massimo
                autocorr = acf(series, nlags=max_lag, fft=True)
                all_autocorrs[i, j, :] = autocorr
                mean_autocorr += autocorr

        # Calcoliamo la media su tutte le serie e dimensioni
        mean_autocorr /= (s * 96)

        # Visualizziamo l'autocorrelazione media
        plt.figure(figsize=(10, 6))
        plt.stem(range(max_lag + 1), mean_autocorr)
        plt.title(f'Autocorrelazione media in funzione del lag per {descr[d]}')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelazione')
        #plt.grid(True)
        # Aggiungiamo una linea orizzontale a zero
        plt.axhline(y=0, color='r', linestyle='-')
        # Aggiungiamo i limiti di confidenza (circa ±1.96/√n)
        plt.axhline(y=1.96/np.sqrt(256//tau), color='k', linestyle='--')
        plt.axhline(y=-1.96/np.sqrt(256//tau), color='k', linestyle='--')
        plt.show()



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
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    c_dim = comm_dict["c_dim"]
    z1_min, z1_max, z2_min, z2_max = comm_dict["z_lims"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    #trial = diff_dict["trial"]
    leg = diff_dict["leg"]
    
    set_cn_ordRT = data["set_cn_ordRT"]
    cont_cn_ordRT = data["cont_cn_ordRT"]
    #RT_cn = data["RT_cn_ordRT"]
    
    s, steps, n_features = set_cn_ordRT.shape
    
    set_cn = torch.from_numpy(set_cn_ordRT).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, s*n_trials, n_features) 
    cont_cn = torch.from_numpy(cont_cn_ordRT).float().to(device).permute(1, 0, 2).unsqueeze(2).repeat(1, 1, n_trials, 1).reshape(steps, s*n_trials, c_dim)

    z_cn, _, _ = dvae.inference(set_cn, cont_cn)
    z_cn = z_cn.reshape(steps, s, n_trials, z_dim)
    z_mean = z_cn.mean(2)
    y_mean, y_logvar = dvae.generation_x(z_mean)
    y_pred = dvae.reparameterization(y_mean, y_logvar)
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
    
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    z1_min, z1_max, z2_min, z2_max = comm_dict["z_lims"]
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
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    c_dim = comm_dict["c_dim"]
    RT_detector = comm_dict["RT_detector"]
    z1_min, z1_max, z2_min, z2_max = comm_dict["z_lims"]
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
    
    x_lim_l = z1_min - 0.5 #-5.5
    x_lim_r = z1_max + 0.5 #7
    y_lim_l = z2_min - 0.5 #-5
    y_lim_r = z2_max + 0.5 #10
    
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
    
    z_mean, _ = dvae.generation_z(points_z, cont_z)
    shift = z_mean - points_z
    shift = shift.cpu().detach().numpy()#.squeeze(1).cpu().detach().numpy()

    u = shift[:, 0]
    v = shift[:, 1]
    
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
    
    z_cn, z_mean, _ = dvae.inference(trial, cont_c)
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
        z_mean_gen, z_cov_gen = dvae.generation_z(z_teach[-1].unsqueeze(0), cont_tot[step].unsqueeze(0))
        z_gen = dvae.reparameterization_cov(z_mean_gen, z_cov_gen)
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
    ax.set_xticks([-1, 2, 5]) 
    ax.set_xticklabels([-1, 2, 5], fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks([-2, 1, 4])
    ax.set_yticklabels([-2, 1, 4], fontsize=font_tick)
    #ax.set_title(f'Example of ws trajectory')
    if leg:
        ax.legend(loc='best', fontsize=font_leg)


def dir_accuracy_gen_traj_e_all(comm_dict, diff_dict, sim_start):
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    c_dim = comm_dict["c_dim"]
    dir_detector = comm_dict["dir_detector"]
    z1_min, z1_max, z2_min, z2_max = comm_dict["z_lims"]
    font_ax = comm_dict["font_ax"]
    font_tick = comm_dict["font_tick"]
    font_leg = comm_dict["font_leg"]
    fig_size = comm_dict["fig_size"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    mean_z = diff_dict["mean_z"]
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    dir_cn = data["dir_cn_ordRT"]
    session_cn = data["sess_cn_ordRT"]
    
    #dir_cn = dir_cn.astype(int)
    samples, steps, channels = set_cn.shape
    teacher = sim_start//(5*tau) 
    alone = steps - teacher
    
    cont_dir = torch.zeros((steps, 2*samples*n_trials, c_dim)).float().to(device)
    cont_dir[:, :samples*n_trials, 1] = 1  # RIGHT
    cont_dir[:, samples*n_trials:, 0] = 1  # LEFT
    
    set_cn = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    cont_cn = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    session_cn_embed = dvae.session_embeddings(torch.from_numpy(session_cn).long().to(device)).unsqueeze(0).expand(steps, -1, -1) #steps x s x s_dim
    session_cn_rep = session_cn_embed.repeat_interleave(n_trials, dim=1) #steps x s*n_trials x s_dim
        
    z_cn, _, _ = dvae.inference(set_cn, cont_cn, session_cn_rep)
    z_teach = torch.cat((z_cn[:teacher], z_cn[:teacher]), dim=1)

    for step in range(alone):
        z_mean_gen, z_cov_gen = dvae.generation_z(z_teach[-1].unsqueeze(0), cont_dir[teacher+step].unsqueeze(0), session_cn_rep[0].unsqueeze(0))
        z_gen = dvae.reparameterization_cov(z_mean_gen, z_cov_gen)
        z_teach = torch.cat((z_teach, z_gen), dim=0)

    if mean_z:
        z_teach = z_teach.reshape(steps, 2*samples, n_trials, z_dim)
        z_teach = z_teach.mean(2)
        gen_dir = np.zeros(2*samples)
        gen_dir[:samples] = 1
        gen_dir[samples:] = 0
        session_cn_double = session_cn_embed.repeat(1, 2, 1)
    else:
        gen_dir = np.zeros(2*samples*n_trials)
        gen_dir[:samples*n_trials] = 1
        gen_dir[samples*n_trials:] = 0
        session_cn_double = session_cn_rep.repeat(1, 2, 1)
    y_mean, y_logvar = dvae.generation_x(z_cn, session_cn_rep)
    y_rec = dvae.reparameterization(y_mean, y_logvar)
    if n_trials > 1:
        y_rec = y_rec.reshape(steps, samples, n_trials, channels)
        y_rec = y_rec.mean(2)
    y_rec = y_rec.permute(1, 0, 2)
        
    y_mean, y_logvar = dvae.generation_x(z_teach, session_cn_double)
    y_pred = dvae.reparameterization(y_mean, y_logvar)
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
    dvae = comm_dict["dvae"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    #dir_detector = comm_dict["dir_detector"]
    z1_min, z1_max, z2_min, z2_max = comm_dict["z_lims"]
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
#     dvae = comm_dict["dvae"]
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
