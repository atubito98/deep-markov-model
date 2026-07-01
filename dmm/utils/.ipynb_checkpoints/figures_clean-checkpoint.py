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
        

# def setup_matplotlib_backend():
#     try:
#         # Se siamo in Jupyter, abilita la modalità interattiva 3D
#         get_ipython().run_line_magic('matplotlib', 'widget')
#     except Exception:
#         # Se siamo in un file .py normale, usa modalità interattiva standard
#         plt.ion()


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
    bins = diff_dict["bins"]
    alpha = diff_dict["alpha"]
    h_hist = diff_dict["h_hist"]
    y_lim = diff_dict["y_lim"]
    
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
    
    fig, ax1 = plt.subplots()

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

    ax2 = ax1.twinx()
    ax2.hist((RT_cn + 56) * 5, bins=bins, density=True, alpha=alpha, color=color_true, edgecolor='none')#, label="RT histogram")
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
        checkpoint_dir = torch.load(saved_path + dir_filename, weights_only=False)
        accuracy_dir = checkpoint_dir["accuracy"]

        dir_filename = "/Non_Markov_dict_z"
        checkpoint_NonMarkov = torch.load(saved_path + dir_filename, weights_only=False)

        dir_filename = "/Markov_dict_z"
        checkpoint_Markov = torch.load(saved_path + dir_filename, weights_only=False)
        
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
        checkpoint_MSE = torch.load(saved_path + dir_filename, weights_only=False)
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
        
        fig_file = os.path.join(comm_dict["saved_path"], f'DMMF_vs_dim_{plot_label[i]}.png')
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
    checkpoint_dir_DMM = torch.load(comm_dict["saved_path"] + dir_filename, weights_only=False)
    accuracy_dir_DMM = checkpoint_dir_DMM["accuracy"]
    
    dir_filename = "/dir_dict_PCA"
    checkpoint_dir_PCA = torch.load(comm_dict["saved_path"] + dir_filename, weights_only=False)
    accuracy_dir_PCA = checkpoint_dir_PCA["accuracy"]
    
    dir_filename = "/Non_Markov_dict_z"
    checkpoint_NonMarkov_DMM = torch.load(comm_dict["saved_path"] + dir_filename, weights_only=False)

    dir_filename = "/Non_Markov_dict_PCA"
    checkpoint_NonMarkov_PCA = torch.load(comm_dict["saved_path"] + dir_filename, weights_only=False)
 
    dir_filename = "/Markov_dict_z"
    checkpoint_Markov_DMM = torch.load(comm_dict["saved_path"] + dir_filename, weights_only=False)
    
    dir_filename = "/Markov_dict_PCA"
    checkpoint_Markov_PCA = torch.load(comm_dict["saved_path"] + dir_filename, weights_only=False)
    
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
    checkpoint_MSE = torch.load(saved_path + dir_filename, weights_only=False)
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

    dir_cn = data["dir_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    z_cn, _, z_cs = infer_latent(dmm, data, device)
    steps, s, _ = z_cn.shape
    
    #####################################
    
    if pca_flag:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=z_dim)
        z_cs_flat = pca.fit_transform(z_cs.reshape(-1, z_dim))  
        z_cn_flat = pca.transform(z_cn.reshape(-1, z_dim))
        z_cn = z_cn_flat.reshape(steps, s, z_dim)

#     pca = PCA(n_components=z_dim)
#     z_cn_flat = pca.fit_transform(z_cn.reshape(-1, z_dim))  # fit + transform sul train 
#     z_cn = z_cn_flat.reshape(steps, s, z_dim)
    
    #####################################
    
    x_r = z_cn[:, dir_cn==1, axis[0]]
    x_l = z_cn[:, dir_cn==0, axis[0]]

    y_r = z_cn[:, dir_cn==1, axis[1]]
    y_l = z_cn[:, dir_cn==0, axis[1]]
    
    
    #####################################
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#     from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    
    X_t = np.vstack([z_cn[(56+20)//tau:(56+100)//tau, dir_cn==1].reshape(-1, z_dim), z_cn[(56+20)//tau:(56+100)//tau, dir_cn==0].reshape(-1, z_dim)])           # shape (n1+n2, 3)
    y_t = np.array([1]*x_r.shape[1]*40 + [2]*x_l.shape[1]*40)
    
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
        
        f, ax = plt.subplots(figsize = (7, 6))
        
        ax.plot(x_r, y_r, '-', linewidth=2, color='g', alpha = alpha)
        ax.plot(x_l, y_l, '-', linewidth=2, color='r', alpha = alpha)
        ax.set_xlabel("first latent component")
        ax.set_ylabel("second latent component")
        ax.set_title(f"right vs left directed trials")
        return f
        
    
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
    
    ax.text(lims[0]*offset + origin[0], origin[1], origin[2], "z1", fontsize=fontsize)
    ax.text(origin[0], lims[1]*offset + origin[1], origin[2], "z2", fontsize=fontsize)
    ax.text(origin[0], origin[1], lims[2]*offset + origin[2], "z3", fontsize=fontsize)

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
    
    SSD_ws = data["SSD_ws_ordRT"]
    RT_ws = data["RT_ws_ordRT"]
    SSD_cs = data["SSD_cs_ordSSD"]
    
    z_cn, z_ws, z_cs = infer_latent(dmm, data, device)
    
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
    ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax.set_xlabel("z1")#, fontsize=font_ax)
    ax.set_ylabel("z2")#, fontsize=font_ax)
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)
    
    fig_file = os.path.join(comm_dict["saved_path"], 'ws_and_cs.png')
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
    ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax.set_xlabel("z1")#, fontsize=font_ax)
    ax.set_ylabel("z2")#, fontsize=font_ax)
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)
    
    fig_file = os.path.join(comm_dict["saved_path"], 'ws_vs_cs_stop.png')
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
        ax.set_xlabel("z1")#, fontsize=font_ax)
        ax.set_ylabel("z2")#, fontsize=font_ax)
        ax.set_xticks(z_ticks[axis[0]]) 
        ax.set_xticklabels(z_ticks[axis[0]])#, fontsize=font_tick)  # Show corresponding labels
        ax.set_yticks(z_ticks[axis[1]])
        ax.set_yticklabels(z_ticks[axis[1]])#, fontsize=font_tick)
#     ax.autoscale_view()
        fig_file = os.path.join(comm_dict["saved_path"], f'{plot_label[i]}.png')
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
    
    
    
    f, ax = plt.subplots()
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
    
    if show_comp:
        with np.load("/raid/home/tubitoal/DMM/saved_model/2026-02-16-07h47_DKF_b12_3Cz3_w3" + f"/SSRT_plot_{n_trials}.npz", allow_pickle=True) as loaded_file:
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
    min_value = min(RT_true.min(), RT_pred.min())
    max_value = max(RT_true.max(), RT_pred.max())
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
    
    with np.load("/raid/home/tubitoal/DMM/saved_model/2026-02-16-07h47_DKF_b12_3Cz3_w3" + f"/RT_distribution_{n_trials}.npz", allow_pickle=True) as loaded_file:
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
    
    min_value = min(RT_true.min(), RT_pred.min())
    max_value = max(RT_true.max(), RT_pred.max())
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
