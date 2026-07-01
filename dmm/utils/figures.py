import os
import math
import random
import numpy as np
import torch
from torch.autograd.functional import jvp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp, wasserstein_distance
from dmm.dataset import one_hot_cont

 
def channel2grid(data):
    n=len(data.shape)
    out_shape = [10, 10]
    out_shape = list(data.shape[:-1]) + out_shape
    grid_data = np.zeros(out_shape)
    
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


def inference_with_trials(dmm, data_set, cont_set, n_trials, device, chunk_size=10):
    """
    Esegue l'inferenza in chunk per evitare errori di memoria GPU.

    """
    n_samples, steps, features = data_set.shape
    
    data_set = torch.from_numpy(data_set).float().permute(1, 0, 2)
    cont_set = torch.from_numpy(cont_set).float().permute(1, 0, 2)
    
    z_dim = dmm.z_dim
    z_mean_accum = np.zeros((steps, n_samples, z_dim), dtype=np.float32)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        batch_size = end - start

        # Estrai chunk
        x_chunk = data_set[:, start:end, :].repeat_interleave(n_trials, dim=1).to(device)
        c_chunk = cont_set[:, start:end, :].repeat_interleave(n_trials, dim=1).to(device)

        # Inferenza
        with torch.no_grad():
            z, z_mean, _ = dmm.inference(x_chunk, c_chunk)

        z_mean_chunk = z.cpu().numpy().reshape(steps, batch_size, n_trials, z_dim).mean(2)

        # Inserisci nel buffer
        z_mean_accum[:, start:end, :] = z_mean_chunk

        torch.cuda.empty_cache()
    z_mean_trasp = np.transpose(z_mean_accum, (1, 0, 2))
    return z_mean_trasp
        
        
def RT_pred_performance(comm_dict, diff_dict):
    
    RT_detector = comm_dict["RT_detector"]
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    
    set_cn = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    cont_cn = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    
    z_cn, _, _ = dmm.inference(set_cn, cont_cn)
    
    z_cn = z_cn.permute(1, 0, 2)
    RT_output = RT_detector(z_cn)
    RT_estimate = prob_to_RT(RT_output, tau)  
    RT_cn = (RT_cn+56)//tau
    
    correlation_detector = np.corrcoef(RT_cn, RT_estimate)[0, 1]
    # Fit lineare
    A = np.vstack([RT_cn, np.ones(len(RT_cn))]).T
    m, c = np.linalg.lstsq(A, RT_estimate, rcond=None)[0]
    RT_estimate_fit = m * RT_cn + c

    # Plot
    plt.figure(figsize=(9, 7))
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
    
    print(f"mean train_SSD_ws: {SSD_ws.mean()*5}ms")
    print(f"mean train_SSD_cs: {SSD_cs.mean()*5}ms")
    
    print("cs_SSD:", np.unique(SSD_cs))
    print("ws_SSD:", np.unique(SSD_ws))
    
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
    ax.hist(mov_cn, bins=bin_edges, density=True, alpha = 0.4, color='skyblue', edgecolor='black', label = "cn")
    ax.hist(mov_ws, bins=bin_edges, density=True, alpha = 0.4, color='red', edgecolor='black', label = "ws")
    # Add labels and title
    ax.set_xlabel('Simulation start time ($ms$)')
    ax.set_ylabel('# of trials')
    ax.set_title("Histograms of RTs for correct no-stop trials")


def infer_latent(dmm, data, device, n_trials=1):
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    set_ws = data["set_ws_ordRT"]
    cont_ws = data["cont_ws_ordRT"]
    set_cs = data["set_cs_ordSSD"]
    cont_cs = data["cont_cs_ordSSD"]
    
    set_cn = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    cont_cn = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    set_ws = torch.from_numpy(set_ws).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    cont_ws = torch.from_numpy(cont_ws).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    set_cs = torch.from_numpy(set_cs).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    cont_cs = torch.from_numpy(cont_cs).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
        
    z_cs, _, _ = dmm.inference(set_cs, cont_cs)
    z_cn, _, _ = dmm.inference(set_cn, cont_cn)
    z_ws, _, _ = dmm.inference(set_ws, cont_ws)

    z_cs = z_cs.cpu().detach().numpy()
    z_cn = z_cn.cpu().detach().numpy()
    z_ws = z_ws.cpu().detach().numpy()
    
    return z_cn, z_ws, z_cs


# +
def single_RTcorr(comm_dict, diff_dict, sim_start):

    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    RT_detector = comm_dict["RT_detector"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    mean_corr = diff_dict["mean_corr"]
    min_n = diff_dict["min_n"]
    
    set_cn = data["set_cn_ordRT"]
    cont_c = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    
    _, steps, _ = set_cn.shape
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
            corr = np.array([np.corrcoef(RT_cn_ordered_filt, RT_est[:, j])[0, 1] for j in range(K)])  
            correlation = np.mean(corr)
            corr_std = np.std(corr)
        else:
            mask_peak = (RT_pred + 56//tau) != teacher
            mask_null = RT_pred != 0
            mask_comb = mask_peak & mask_null
            RT_cn_ordered_filt = RT_cn_ordered_filt[mask_comb]
            RT_pred_filt = RT_pred[mask_comb]
            correlation = np.corrcoef(RT_cn_ordered_filt, RT_pred_filt)[0, 1]
            corr_std = None
    else:
        RT_cn_ordered_filt = 0
        RT_pred_filt = np.ones((1, 2))
        correlation = 2
        corr_std = 1

    return RT_cn_ordered_filt, RT_pred_filt, correlation, corr_std
   


def correlation_vs_gentime(comm_dict, diff_dict):     
    
    tau = comm_dict["tau"]
    saved_path = comm_dict["saved_path"]
    
    data = diff_dict["data"]
    sim_start_array = diff_dict["sim_start_array"]
    compute = diff_dict["compute"]
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
            _, _, correlation, corr_std = single_RTcorr(comm_dict, diff_dict, sim_start)
            correlations[cont] = correlation
            corr_stds[cont] = corr_std

        mask_new = correlations<2
        sim_start_array_new = sim_start_array[mask_new]
        corr_stds_new = corr_stds[mask_new]
        correlations_new = correlations[mask_new]
        np.savez(saved_path / "RT_correlations.npz", sim_start_array = sim_start_array_new, correlations = correlations_new, corr_stds = corr_stds_new)
    else:
        with np.load(saved_path / "RT_correlations.npz") as loaded_file:
            correlations_new = loaded_file["correlations"]
            corr_stds_new = loaded_file["corr_stds"]
            sim_start_array_new = loaded_file["sim_start_array"]
    
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.scatter(sim_start_array_new, correlations_new, color=color_corr, edgecolors=color_edge)
    ax1.axvline(56*5, color="black", linestyle="--")
    ax1.set_ylim((0, y_lim))
    ax1.set_xlabel("Simulation start ($ms$)")
    ax1.set_xticks([400, 800]) 
    ax1.set_xticklabels([400, 800])
    ax1.set_ylabel("Correlation")
    ax1.set_yticks([0, 0.5, 1])
    ax1.set_yticklabels([0, 0.5, 1])

    RT_true = (56 + RT_cn) * 5
    min_value = RT_true.min() - 50
    max_value = RT_true.max() + 50
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)

    ax2 = ax1.twinx()
    ax2.hist(RT_true, bins=bin_edges, density=True, alpha=alpha, color=color_true, edgecolor='none')
    ax2.set_ylim(0, h_hist)
    ax2.set_yticks([])
   
    fig_file = saved_path / 'RT_correlation.png'
    plt.savefig(fig_file)
    plt.show()
        
    dict_corr = {a: (b, c) for a, b, c in zip(sim_start_array_new, correlations_new, corr_stds_new)}
    print(f"RT correlations at the GO cue: {correlations_new[56//tau]}")
    
    return dict_corr

def DMM_EV(comm_dict, diff_dict):
            
    device = comm_dict["device"]
    saved_path = comm_dict["saved_path"]
    
    color = diff_dict["color"]
    x_label = diff_dict["x_label"]
    x_lims = diff_dict["x_lims"]
    y_lims = diff_dict["y_lims"]
    delta_y = diff_dict["delta_y"]
    alpha = diff_dict["alpha"]
    model_dir = diff_dict["model_dir"]
    
    model_path = saved_path.parent / model_dir
    with np.load(model_path / "data_split.npz", allow_pickle=True) as loaded_file:
            test_set = loaded_file["test_set"]
            test_direction = loaded_file["test_direction"]
            test_SSD = loaded_file["test_SSD"]
            
    params = {
        "cfg": model_path / "config.ini",
        "device": device,
        "saved_dir": model_path
    }
    
    from dmm.learning_algo_dir import LearningAlgorithm_dir
    
    weights_dir = "DKF_simple_final_epoch149.pt"
    learning_algo = LearningAlgorithm_dir(params=params)
    dmm = learning_algo.model
    dmm.load_state_dict(torch.load(model_path / weights_dir, map_location=device))
    dmm.eval()
    z_dim = dmm.z_dim
 
    cont_test = one_hot_cont(test_SSD, test_direction, dmm.tau)
    test_data = torch.from_numpy(test_set).float().to(device).permute(1, 0, 2)
    cont_test = torch.from_numpy(cont_test).float().to(device).permute(1, 0, 2)
    z, _, _ = dmm.inference(test_data, cont_test)
    z = z.cpu().detach().numpy().reshape(-1, z_dim) 
    
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)

    # Applica PCA con n_components=3
    pca = PCA(n_components=z_dim)
    pca.fit(z_scaled)

    # Ottieni l'explained variance per ciascuna componente
    explained_variance = pca.explained_variance_ratio_
    z_vec = (np.arange(z_dim) + 1)/(z_dim+1)
    
    f, ax = plt.subplots()
    
    ax.plot(z_vec, explained_variance, "--", color=color, alpha=alpha)
    ax.scatter(z_vec, explained_variance, color=color)
    ax.set_xlim(x_lims)
    ax.set_xticks(z_vec)
    ax.set_xticklabels(x_label)
    ax.set_ylim(y_lims)
    ax.set_yticks(y_lims)
    ax.set_yticklabels(y_lims)
    ax.set_ylabel('Explained variance')
    for xi, yi in zip(z_vec, explained_variance):
        ax.text(xi, yi + (delta_y), f"{yi:.2f}", ha='center')
   
    fig_file = model_path / 'DMM_EV.png'
    plt.savefig(fig_file)
    plt.show()

    

def DMM_vs_dim(comm_dict, diff_dict):
    
    device = comm_dict["device"]
    saved_path = comm_dict["saved_path"]
    
    width = diff_dict["width"]
    x = diff_dict["x"]
    y_lim = diff_dict["y_lim"]
    y_ticks = diff_dict["y_ticks"]
    colors = diff_dict["colors"]
    model_dirs = diff_dict["model_dirs"]
    
    data = []
    for i in range(len(model_dirs)):
        
        model_path = comm_dict["saved_path"].parent / model_dirs[i]
        
        dir_filename = "dir_dict_z"
        checkpoint_dir = torch.load(model_path / dir_filename, weights_only=False, map_location=device)
        accuracy_dir = checkpoint_dir["accuracy"]

        dir_filename = "Non_Markov_dict_z"
        checkpoint_NonMarkov = torch.load(model_path / dir_filename, weights_only=False, map_location=device)
        
        dir_filename = "MSE"
        checkpoint_MSE = torch.load(model_path / dir_filename, weights_only=False, map_location=device)
        MSE_rec = checkpoint_MSE["MSE_DMM"]
    
        NMSE_NonMarkov = checkpoint_NonMarkov["MSE_pc"]
        r2_NonMarkov = 1 - NMSE_NonMarkov
        r2_rec = 1 - MSE_rec
        
        data.append([r2_rec, r2_NonMarkov, accuracy_dir])
 
    data = np.array(data).T
    
    y_label = ["$R^2$", "$R^2_D$", "Accuracy"]
    plot_label = ["rec", "prediction", "direction"]
    for i in range(len(y_label)):
        f, ax = plt.subplots()
        # x positions delle due colonne
        ax.bar(x, data[i], color=colors, width=width)
        ax.set_xlim((0, 1))
        ax.set_xticks(x)
        ax.set_xticklabels(["D=2", "D=3", "D=4"])
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
            ax.text(xi, yi + (delta_y*y_ticks[i][1]), f"{yi:.2f}", ha='center')
        
        fig_file = saved_path / f'DMM_vs_dim_{plot_label[i]}.png'
        plt.savefig(fig_file)
        plt.show()


def PCA_vs_DMM(comm_dict, diff_dict):
 
    saved_path = comm_dict["saved_path"]
    device = comm_dict["device"]

    width = diff_dict["width"]
    x = diff_dict["x"]
    y_lim = diff_dict["y_lim"]
    y_ticks = diff_dict["y_ticks"]
    color_DMM = diff_dict["color_DMM"]
    color_PCA = diff_dict["color_PCA"]
    inset_dim = diff_dict["inset_dim"]
    inset_font = diff_dict["inset_font"]
    
    dir_filename = "dir_dict_z"
    checkpoint_dir_DMM = torch.load(saved_path / dir_filename, weights_only=False, map_location=device)
    accuracy_dir_DMM = checkpoint_dir_DMM["accuracy"]
    
    dir_filename = "dir_dict_PCA"
    checkpoint_dir_PCA = torch.load(saved_path / dir_filename, weights_only=False, map_location=device)
    accuracy_dir_PCA = checkpoint_dir_PCA["accuracy"]
    
    dir_filename = "Non_Markov_dict_z"
    checkpoint_NonMarkov_DMM = torch.load(saved_path / dir_filename, weights_only=False, map_location=device)

    dir_filename = "Non_Markov_dict_PCA"
    checkpoint_NonMarkov_PCA = torch.load(saved_path / dir_filename, weights_only=False, map_location=device)
 
    dir_filename = "Markov_dict_z"
    checkpoint_Markov_DMM = torch.load(saved_path / dir_filename, weights_only=False, map_location=device)
    
    dir_filename = "Markov_dict_PCA"
    checkpoint_Markov_PCA = torch.load(saved_path / dir_filename, weights_only=False, map_location=device)
    
    NMSE_NonMarkov_PCA = checkpoint_NonMarkov_PCA["MSE_pc"]
    NMSE_Markov_PCA = checkpoint_Markov_PCA["MSE_pc"]
    NMSE_NonMarkov_DMM = checkpoint_NonMarkov_DMM["MSE_pc"]
    NMSE_Markov_DMM = checkpoint_Markov_DMM["MSE_pc"]
    
    dir_filename = "MSE"
    checkpoint_MSE = torch.load(saved_path / dir_filename, weights_only=False, map_location=device)
    MSE_rec_DMM = checkpoint_MSE["MSE_DMM"]
    MSE_rec_PCA = checkpoint_MSE["MSE_PCA"]
    
    r2_Markov_PCA = 1-NMSE_Markov_PCA
    r2_Markov_DMM = 1-NMSE_Markov_DMM
    
    r2_NonMarkov_PCA = 1-NMSE_NonMarkov_PCA
    r2_NonMarkov_DMM = 1-NMSE_NonMarkov_DMM
      
    r2_rec_PCA = 1-MSE_rec_PCA
    r2_rec_DMM = 1-MSE_rec_DMM
    
    print(f"r2_PCA: {r2_rec_PCA:.4f}")
    print(f"r2_DMM: {r2_rec_DMM:.4f}")
    
    data = [
        (r2_rec_DMM, r2_rec_PCA), 
        (r2_NonMarkov_DMM, r2_NonMarkov_PCA),
        (accuracy_dir_DMM, accuracy_dir_PCA),
    ]
    
    y_label = ["$R^2$", "$R^2_D$", "Accuracy"]
    plot_label = ["rec", "prediction", "direction"]
    color = (color_DMM, color_PCA)

    for i in range(len(y_label)):
        f, ax = plt.subplots()
        ax.bar(x, data[i], color=color, width=width)
        ax.set_xlim((0, 1))
        ax.set_xticks(x)
        ax.set_xticklabels(["DMM", "PCA"])
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
            ax.text(xi, yi + (delta_y*y_ticks[i][1]), f"{yi:.2f}", ha='center')
            
        if i==1:
            # --- Inset axes inside ax ---
            axins = ax.inset_axes(inset_dim)  # [x0, y0, width, height] in Axes fraction

            axins.bar(x, (r2_Markov_DMM, r2_Markov_PCA), color=color, width=width)
            axins.set_xlim((0, 1))
            axins.set_xticks(x)
            axins.set_xticklabels(["DMM", "PCA"])
            axins.set_ylabel("$R^2_D$", fontsize=inset_font)
            axins.set_ylim((0, 1.2))
            axins.set_yticks((0, 1))
            axins.set_yticklabels((0, 1))
            axins.tick_params(labelsize=inset_font)
            
            # mostro anche il numero sopra la barra
            for xi, yi in zip(x, (r2_Markov_DMM, r2_Markov_PCA)):
                axins.text(xi, yi + (0.05*y_ticks[i][1]), f"{yi:.2f}", ha='center', fontsize=inset_font)
        
        fig_file = saved_path / f'DMM_vs_PCA_{plot_label[i]}.png'
        plt.savefig(fig_file)
        plt.show()


def random_latent_cn_traj(comm_dict, data, n_trials=1):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    
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

    
def latent_traj_example(comm_dict, diff_dict):
    
    saved_path = comm_dict["saved_path"]
    
    data = diff_dict["data"]
    color_line = diff_dict["color_line"]
    color_arrows = diff_dict["color_arrows"]
    n_arrows = diff_dict["n_arrows"]
    azim = diff_dict["azim"]
    elev = diff_dict["elev"]
    scale = diff_dict["scale"]
    origin = diff_dict["origin"]
    offset = diff_dict["offset"]
    fontsize = diff_dict["fontsize"]
    lw_axis = diff_dict["lw_axis"]
    lw_arrows = diff_dict["lw_arrows"]
    lw_line = diff_dict["lw_line"]
    
    z_trial, RT, q = random_latent_cn_traj(comm_dict, data,)

    print(f"traj n.{q}")
        
    x_true_story = z_trial[:, 0]
    y_true_story = z_trial[:, 1]
    z_true_story = z_trial[:, 2]
    
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
        color=color_arrows,
        linewidth=lw_arrows,
    )

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
    
    ax.quiver(*origin, lims[0], 0, 0, color="black", linewidth=lw_axis)
    ax.quiver(*origin, 0, lims[1], 0, color="black", linewidth=lw_axis)
    ax.quiver(*origin, 0, 0, lims[2], color="black", linewidth=lw_axis)
    
    ax.text(lims[0]*offset + origin[0], origin[1], origin[2], "$z_1$", fontsize=fontsize)
    ax.text(origin[0], lims[1]*offset + origin[1], origin[2], "$z_2$", fontsize=fontsize)
    ax.text(origin[0], origin[1], lims[2]*offset + origin[2], "$z_3$", fontsize=fontsize)

    # -------------------------
    # Equal aspect ratio
    # -------------------------
    ranges = np.ptp(z_trial, axis=0)
    ax.set_box_aspect(ranges)

    # -------------------------
    # View angle (change freely)
    # -------------------------
    ax.view_init(elev=elev, azim=azim)

    fig_file = saved_path / 'trajectory.png'
    plt.savefig(fig_file)


# decoder: callable torch module mapping z (batch? or single) -> x
# z0: tensor shape (latent_dim,), requires_grad=False
# d: unit direction in latent space (latent_dim,)
# l: desired length (scalar)
def delta_x_via_jvp(dmm, z0, d, l, data=None):
    device = z0.device
    z0 = z0.detach().clone().requires_grad_(True)   # enable autograd        
    delta_z = torch.from_numpy(-d * l).float().to(device)   # se vuoi diminuire lungo d; + per aumentare
    if z0.ndim == 2:
        delta_z = torch.tile(delta_z, (z0.shape[0], 1))
    # jvp expects tuples for inputs/outputs
    def decoder_mean(z):   # (z, data)
        x_mean, _ = dmm.generation_x(z)
        return x_mean
    y, jvp_out = jvp(decoder_mean, (z0,), (delta_z,))
    jvp_out = jvp_out.cpu().detach().numpy()
    # jvp_out has shape of x (. e.g. data_dim)
    #print(y.shape)
    print(jvp_out.shape)
    return jvp_out  # appross delta_x


def plot_cycle_shift_short(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    saved_path = comm_dict["saved_path"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    mean = diff_dict["mean"]
    step = diff_dict["step"]
    stimolate = diff_dict["stimolate"]
    l1_ratio = diff_dict["l1_ratio"]
    alpha_L1 = diff_dict["alpha_L1"]
    stimulation_steps = diff_dict["stimulation_steps"]
    mean_trials = diff_dict["mean_trials"]
    l = diff_dict["l"]
    f = diff_dict["f"]
    compute = diff_dict["compute"]
    multi_direction = diff_dict["multi_direction"]
    add_residual = diff_dict["add_residual"]
    thr = diff_dict["thr"]
    
    RT_cn = data["RT_cn_ordRT"]
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    z_cn, _, _ = infer_latent(dmm, data, device, n_trials)
    
    steps, s, _ = z_cn.shape
    samples = len(RT_cn)    # s = samples * n_trials
    
    if mean_trials:
        z_cn = z_cn.reshape(steps, samples, n_trials, z_dim)
        z_cn = z_cn.mean(2)
        color = RT_cn/RT_cn.max()
    else:
        set_cn = set_cn.repeat(n_trials, 1, 1)
        RT_cn_rep = np.repeat(RT_cn, n_trials)
        color = RT_cn_rep/RT_cn_rep.max()
    z_GO = z_cn[step//tau]
    
    long_index = int(z_GO.shape[0] * (1-f))
    set_cn_long = set_cn[long_index:]
    cont_cn_long = cont_cn[long_index:]
    
    n_long, _, features = set_cn_long.shape
  
    trial_true = torch.from_numpy(set_cn_long).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    cont_RTlong = torch.from_numpy(cont_cn_long).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    
    start_sim = step//tau + 1 - stimulation_steps
    
    text = "" if stimolate else "_onech"
    text_mean = "_mean" if mean else ""
    
    if compute:
        # --- STIMA DIREZIONE ---
        reg = LinearRegression().fit(z_GO, color)
        direction = reg.coef_
        direction /= np.linalg.norm(direction)  # normalizzazione unit vector
        delta_z_short = torch.from_numpy(-direction * l).float().to(device)
        print("Direzione di variazione (aumenta):", direction)

        delta_z_short = torch.zeros(stimulation_steps, z_dim).to(device)
        if multi_direction:
            for t in range(stimulation_steps):
                reg = LinearRegression().fit(z_cn[start_sim + t], color)
                direction = reg.coef_
                direction /= np.linalg.norm(direction)  # normalizzazione unit vector
                delta_z_short[t] = torch.from_numpy(-direction * l).float().to(device)

     
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
            trial_modified_min[start_sim + t + 1] = y_gen[0]
            if add_residual:
                trial_modified_min[start_sim + t + 1] += residual[start_sim + t + 1]
            stim_effect_min[t] = trial_modified_min[start_sim + t + 1] - trial_true[start_sim + t + 1]

        stim_effect = stim_effect.cpu().detach().numpy()
        stim_effect_min = stim_effect_min.cpu().detach().numpy()
        trial_stim = trial_modified.cpu().detach().numpy()
    
        np.savez(saved_path / f"shorter_RT_{l}{text}{text_mean}.npz", trial_stim=trial_stim, stim_effect=stim_effect, stim_effect_min=stim_effect_min, dx_array=dx_array)
    else:
        with np.load(saved_path / f"shorter_RT_{l}{text}{text_mean}.npz") as loaded_file:
            trial_stim = loaded_file["trial_stim"]
            dx_array = loaded_file["dx_array"]
            if not stimolate:
                stim_effect = loaded_file["stim_effect"]
                stim_effect_min = loaded_file["stim_effect_min"]
                diff_stim = stim_effect - stim_effect_min
            
        trial_modified = torch.from_numpy(trial_stim).float().to(device)
    
    if stimolate:  
        dx_array = channel2grid(dx_array)
        last_stim = dx_array[stimulation_steps].copy()
        last_stim[last_stim < thr] = 0
    else:
        mean_stim_effect = diff_stim.mean(1)
        last_stim = channel2grid(mean_stim_effect)
    
    make_RThist(comm_dict, diff_dict, last_stim, trial_modified, trial_true, cont_RTlong, l, 'shorter')
    
    
def plot_cycle_shift_long(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    saved_path = comm_dict["saved_path"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    mean = diff_dict["mean"]
    step = diff_dict["step"]
    stimolate = diff_dict["stimolate"]
    l1_ratio = diff_dict["l1_ratio"]
    alpha_L1 = diff_dict["alpha_L1"]
    stimulation_steps = diff_dict["stimulation_steps"]
    mean_trials = diff_dict["mean_trials"]
    l = diff_dict["l"]
    f = diff_dict["f"]
    compute = diff_dict["compute"]
    multi_direction = diff_dict["multi_direction"]
    add_residual = diff_dict["add_residual"]
    thr = diff_dict["thr"]
    
    RT_cn = data["RT_cn_ordRT"]
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    z_cn, _, _ = infer_latent(dmm, data, device, n_trials)
    
    steps, _, _ = z_cn.shape
    samples = len(RT_cn)    # s = samples * n_trials
    
    if mean_trials:
        z_cn = z_cn.reshape(steps, samples, n_trials, z_dim)
        z_cn = z_cn.mean(2)
        color = RT_cn/RT_cn.max()
    else:
        set_cn = set_cn.repeat(n_trials, 1, 1)
        RT_cn_rep = np.repeat(RT_cn, n_trials)
        color = RT_cn_rep/RT_cn_rep.max()
    z_GO = z_cn[step//tau]
    
    short_index = int(z_GO.shape[0] // (1/f))
    
    set_cn_short = set_cn[:short_index]
    cont_cn_short = cont_cn[:short_index]
    
    n_short, _, features = set_cn_short.shape
    
    trial_true = torch.from_numpy(set_cn_short).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1) 
    cont_RTshort = torch.from_numpy(cont_cn_short).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    
    start_sim = step//tau + 1 - stimulation_steps
    text = "" if stimolate else "_onech"
    
    if compute:
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
            # x -> x + dx
            if stimolate:
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
    
        np.savez(saved_path / f"longer_RT_{l}{text}.npz", trial_stim = trial_stim, stim_effect_min=stim_effect_min, stim_effect = stim_effect, dx_array = dx_array)
    else:
        with np.load(saved_path / f"longer_RT_{l}{text}.npz") as loaded_file:
            trial_stim = loaded_file["trial_stim"]
            dx_array = loaded_file["dx_array"]
            if not stimolate:
                stim_effect = loaded_file["stim_effect"]
                stim_effect_min = loaded_file["stim_effect_min"]
                diff_stim = stim_effect - stim_effect_min
                
        trial_modified = torch.from_numpy(trial_stim).float().to(device)
    
    if stimolate:  
        dx_array = channel2grid(dx_array)
        last_stim = dx_array[stimulation_steps].copy()
        last_stim[last_stim < thr] = 0
    else:
        mean_stim_effect = diff_stim.mean(1)
        last_stim = channel2grid(mean_stim_effect)
    
    make_RThist(comm_dict, diff_dict, last_stim, trial_modified, trial_true, cont_RTshort, l, 'longer')


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

    for _ in range(n_iter):
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
            model = ElasticNet(alpha=alpha_l1, l1_ratio=l1_ratio, fit_intercept=False, max_iter=10000)
            model.fit(J, delta_z_np)
            dx_last_n = torch.from_numpy(model.coef_).float().to(device).unsqueeze(0)  # (1, D)

            dx_last_list.append(dx_last_n)

        dx_last = torch.cat(dx_last_list, dim=0)  # (N, D)

        # Aggiorna solo ultimo timestep
        x_seq[-1] = x_seq[-1] + dx_last
        dx_last = dx_last.cpu().detach().numpy()
    return x_seq, dx_last
    

def make_RThist(comm_dict, diff_dict, last_stim, trial_modified, trial_true, cont_RTlong, l, text, RT_true=None):
    
    dmm = comm_dict["dmm"]
    tau = comm_dict["tau"]
    RT_detector = comm_dict["RT_detector"]
    saved_path = comm_dict["saved_path"]
    
    alpha = diff_dict["alpha"]
    step = diff_dict["step"]
    bins = diff_dict["bins"]
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
    ax.hist(RT_gen_true, bins=bin_edges, alpha = alpha, density=True, color=color_true, edgecolor='none')
    ax.axvline(RT_gen_true.mean(), color=color_true,linestyle="--")
    ax.hist(RT_gen, bins=bin_edges, alpha = alpha, density=True, color=color_pred, edgecolor='none')
    ax.axvline(RT_gen.mean(), color=color_pred,linestyle="--")
    if RT_true:
        ax.axvline(RT_true, color="k",linestyle="--")
    ax.set_xticks(x_ticks_hist)
    ax.set_xticklabels(x_ticks_hist)
    ax.set_yticks([])
    # Add labels and title
    ax.set_xlabel('Reaction Time ($ms$)')
    ax.set_ylabel('Counts')
    
    vmax = abs(last_stim).max()
    
    if vmax < 0.1:
        vmax=0.3
    
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

    fig_file = saved_path / f'{text}_RT_{l}.png'
    plt.savefig(fig_file)
    

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
    saved_path = comm_dict["saved_path"]
    
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
    z, _, _ = dmm.inference(trial, cont_c)

    teacher = ((56+SSD)//tau) 
    alone = steps-teacher
    min_grad_mod = (alone*(1-min_grad))/steps
    z_teach = z[:teacher]
    for _ in range(alone):
        z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher].unsqueeze(0))
        z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
        z_teach = torch.cat((z_teach, z_gen), dim=0)

    mu_z = z_teach.mean(1)
    mu_z = mu_z[teacher:].cpu().detach().numpy()
    mu_x = mu_z[:, axis[0]]
    mu_y = mu_z[:, axis[1]]
 
    true_trial = z.mean(1)
    true_trial = true_trial.cpu().detach().numpy()
    x_true_story = true_trial[:, axis[0]]
    y_true_story = true_trial[:, axis[1]]
    
    x_SSD = true_trial[teacher-1, axis[0]]
    y_SSD = true_trial[teacher-1, axis[1]]
    
    lc1 = add_gradient_line(ax, x_true_story, y_true_story, cmap=cmap, lw=lw, vmin=min_grad, vmax=1.0)
    lc2 = add_gradient_line(ax, mu_x, mu_y, cmap=cmap_go, lw=lw, vmin=min_grad_mod, vmax=1.0)
 
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
    ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]])  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]])
    
    fig_file = saved_path / 'cs_to_ws.png'
    plt.savefig(fig_file)


def plot_cs_ns(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_dim = comm_dict["z_dim"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    saved_path = comm_dict["saved_path"]
    
    data = diff_dict["data"]
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
    n_cn, n_ws, n_cs = z_cn.shape[1], z_ws.shape[1], z_cs.shape[1]
    
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
    ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]])  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]])
    
    fig_file = saved_path / 'ws_and_cs.png'
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
    
    X_t = np.vstack([z_cs_stop, z_ws_stop])     # shape (n1+n2, 3)
    y_t = np.array([1]*z_cs.shape[1] + [2]*z_ws.shape[1])
    
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
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_xticks(z_ticks[axis[0]])
    ax.set_xticklabels(z_ticks[axis[0]])  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]])
    
    #------------ inset PCA plot ------------------
    
    axins = ax.inset_axes(inset_dim)
    
    with np.load(saved_path / "data_split.npz", allow_pickle=True) as loaded_file:
        train_set = loaded_file["train_set"]
            
    n_train, steps, features = train_set.shape
    X_train = train_set.reshape(-1, features)  # shape = (n_train * time_steps, 96)

    pca = PCA(n_components=z_dim)
    pca.fit(X_train)  # fit + transform sul train 
    
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
    axins.set_xlabel(f"$PC_{axis_pca[0]+1}$", fontsize=inset_font)
    axins.set_ylabel(f"$PC_{axis_pca[1]+1}$", fontsize=inset_font)

    axins.set_xticks([])
    axins.set_yticks([])
    axins.tick_params(labelsize=inset_font)
    
    fig_file = saved_path / 'ws_vs_cs_stop.png'
    plt.savefig(fig_file)
    
    return xx, yy


def change_trial_ws(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    saved_path = comm_dict["saved_path"]
    
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
    z, _, _ = dmm.inference(trial, cont_c)

    teacher = ((56+SSD)//tau) - anticipation//(5*tau)
    alone = steps-teacher
    min_grad_mod = (alone*(1-min_grad))/steps
    z_teach = z[:teacher]
    for _ in range(alone):
        z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont)
        z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
        z_teach = torch.cat((z_teach, z_gen), dim=0)

    mu_z = z_teach.mean(1)
    mu_z = mu_z[teacher:].cpu().detach().numpy()
    mu_x = mu_z[:, axis[0]]
    mu_y = mu_z[:, axis[1]]

    true_trial = z.mean(1)
    true_trial = true_trial.cpu().detach().numpy()
    x_true_story = true_trial[:, axis[0]]
    y_true_story = true_trial[:, axis[1]]

    x_SSD = true_trial[(56+SSD)//tau, axis[0]]
    y_SSD = true_trial[(56+SSD)//tau, axis[1]]
    
    x_stop = true_trial[teacher-1, axis[0]]
    y_stop = true_trial[teacher-1, axis[1]]
    
    lc1 = add_gradient_line(ax, x_true_story, y_true_story, cmap=cmap, lw=lw, vmin=min_grad, vmax=1.0)
    lc2 = add_gradient_line(ax, mu_x, mu_y, cmap=cmap_go, lw=lw, vmin=min_grad_mod, vmax=1.0)
    
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
    ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
    ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_xticks(z_ticks[axis[0]]) 
    ax.set_xticklabels(z_ticks[axis[0]])  # Show corresponding labels
    ax.set_yticks(z_ticks[axis[1]])
    ax.set_yticklabels(z_ticks[axis[1]])
    
    fig_file = saved_path / 'ws_to_cs.png'
    plt.savefig(fig_file)


def change_trial_cn(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    z_lims = comm_dict["z_lims"]
    z_ticks = comm_dict["z_ticks"]
    RT_detector = comm_dict["RT_detector"]
    move_detector = comm_dict["move_detector"]
    saved_path = comm_dict["saved_path"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    stop_array = diff_dict["stop_array"]
    RT = diff_dict["RT"]
    axis = diff_dict["axis"]
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
    cont[:, :, 3] = 1  # Stop context

    trial = torch.from_numpy(trial).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
    cont_c = torch.from_numpy(cont_c).float().to(device).unsqueeze(1).repeat(1, n_trials, 1)
    
    # generation
    z, _, _ = dmm.inference(trial, cont_c)
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
        z_teach = z[:teacher]
        for _ in range(alone):
            z_mean_gen, z_cov_gen = dmm.generation_z(z_teach[-1].unsqueeze(0), cont)
            z_gen = dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
            z_teach = torch.cat((z_teach, z_gen), dim=0)

        mu_z = z_teach.mean(1)
        mu_z = mu_z[teacher:].cpu().detach().numpy()
        mu_x = mu_z[:, axis[0]]
        mu_y = mu_z[:, axis[1]]
        
        f, ax = plt.subplots()
        
        n_arrows = int(n_arrows_init*(alone/steps))
        arrow_indices = np.arange(0, len(mu_x), len(mu_x)//n_arrows)  # Place n_arrows arrows along the path
        
        lc1 = add_gradient_line(ax, x_true_story, y_true_story, cmap=cmap, lw=lw, vmin=min_grad, vmax=1.0)
        lc2 = add_gradient_line(ax, mu_x, mu_y, cmap=cmap_stop, lw=lw, vmin=min_grad_mod, vmax=1.0)
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
        ax.plot(x_SSD, y_SSD, color=color_stop, marker='x', markeredgewidth = 3, markersize = 15)
        ax.plot(xx, yy, color = color_line, lw=lw, alpha=alpha_line)
        ax.set_xlim(z_lims[axis[0], 0], z_lims[axis[0], 1])
        ax.set_ylim(z_lims[axis[1], 0], z_lims[axis[1], 1])
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")
        ax.set_xticks(z_ticks[axis[0]]) 
        ax.set_xticklabels(z_ticks[axis[0]]) # Show corresponding labels
        ax.set_yticks(z_ticks[axis[1]])
        ax.set_yticklabels(z_ticks[axis[1]])
        fig_file = saved_path / f'{plot_label[i]}.png'
        plt.savefig(fig_file)
        plt.show()


def RTgen_vs_SSD(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    RT_detector = comm_dict["RT_detector"]
    move_detector = comm_dict["move_detector"]
    saved_path = comm_dict["saved_path"]
    with_Cornelio = comm_dict["with_Cornelio"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    x_ticks = diff_dict["x_ticks"]
    elinewidth = diff_dict["elinewidth"]
    capsize = diff_dict["capsize"]
    ms = diff_dict["ms"]
    compute = diff_dict["compute"]
    mean_z = diff_dict["mean_z"]
    min_ns = diff_dict["min_ns"]
    y_ticks = diff_dict["y_ticks"]
    chunk_size = diff_dict["chunk_size"]
    ylims = diff_dict["ylims"]
    SSD_list = diff_dict["SSD_list"]
    figsize = diff_dict["figsize"]
    simulate_go = diff_dict["simulate_go"]
    ylims_inset = diff_dict["ylims_inset"]
    y_ticks_inset = diff_dict["y_ticks_inset"]
    inset_font = diff_dict["inset_font"]
    inset_dim = diff_dict["inset_dim"]
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
   
    n, steps, _ = set_cn.shape
    RT_time = RT_cn*5
    n_batch = n_trials // chunk_size if (n_trials%chunk_size)==0 else (n_trials // chunk_size) + 1
    text = "_sim" if simulate_go else ""  
    
    # Input per batch
    test_data = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).repeat_interleave(chunk_size, dim=1)
    cont_data = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2).repeat_interleave(chunk_size, dim=1)
    
    cont = torch.zeros((1, n*chunk_size, 4)).to(device)
    cont[:, :, 3] = 1

    if compute:
        with torch.inference_mode():
            RT_pred_list = []
            RT_time_list = []
            pmove_list = []
            # 🔁 Ora ciclo esterno sul tempo t
            for SSD in SSD_list:
                teacher = (SSD//5 + 56)//tau
                alone = steps - teacher - 1
                
                move_pred_t_list = []
                mu_go_list = []
                mu_z_list = []
                # 🔁 Ciclo interno sui batch
                for _ in range(n_batch):

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
                
                    move_pred_t_list.append(move_pred)
                    mu_z_list.append(mu_z.reshape(-1, chunk_size, mu_z.shape[1], z_dim))

                # Concatena tutti i batch per questo tempo t
                move_pred_t = np.concatenate(move_pred_t_list, axis=1)  # n x n_trials
                mu_z_array = torch.cat(mu_z_list, dim=1)
                mu_go_array = torch.cat(mu_go_list, dim=2)  # n x n_trials
                
                mu_mean_go = mu_go_array.mean(2)
                RT_output = RT_detector(mu_mean_go.permute(1, 0, 2))
                RT_go = prob_to_RT(RT_output, tau)
                RT_go = RT_go*tau*5
                
                mask_ns = move_pred_t > 0.5
                hist = mask_ns.sum(axis=1)     # numero di sim che hanno portato a un ws, per trial
                valid_rows = hist >= min_ns    # trial con numero di simulazioni ws sufficienti
            
                mu_z_masked = mu_z_array[valid_rows]
                mask_valid = mask_ns[valid_rows]
                RT_cn_valid = RT_go[valid_rows] if simulate_go else RT_time[valid_rows] 
                
                print(mu_z_masked.shape)
                mask_valid = torch.from_numpy(mask_valid).float().to(device)
                mask_valid = mask_valid.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, mu_z_masked.shape[2], mu_z_masked.shape[3])
                num = (mu_z_masked * mask_valid).sum(dim=1)
                den = mask_valid.sum(dim=1)
                mu_z_mean = num / den
            
                RT_output = RT_detector(mu_z_mean)
                RT_rec = prob_to_RT(RT_output, tau)
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

        np.savez(saved_path / f"RTgen_vs_SSD{text}_{n_trials}.npz", RT_pred_dict=RT_pred_dict, RT_time_dict=RT_time_dict, pmove_dict=pmove_dict)
    else:
        with np.load(saved_path / f"RTgen_vs_SSD{text}_{n_trials}.npz", allow_pickle=True) as loaded_file:
            RT_pred_dict = loaded_file["RT_pred_dict"].item()
            RT_time_dict = loaded_file["RT_time_dict"].item()
            pmove_dict = loaded_file["pmove_dict"].item()

    # ------------------ diff RT vs SSD --------------------------
    
    RT_diff_mean = []
    RT_diff_std = []
    for i in range(len(SSD_list)):
        diff_RT = RT_pred_dict[i] - RT_time_dict[i]
        RT_diff_mean.append(diff_RT.mean())
        RT_diff_std.append(diff_RT.std()/math.sqrt(len(diff_RT)))
    
    fig, ax = plt.subplots(figsize=figsize)
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
    
    if with_Cornelio:
        with np.load(saved_path.parent / with_Cornelio / f"RTgen_vs_SSD{text}_{n_trials}.npz", allow_pickle=True) as loaded_file:
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
    fig_file = saved_path / f'RTgen_vs_SSD{text}.png'
    plt.savefig(fig_file)
    plt.show()
    
    
def early_SSRT(comm_dict, diff_dict):

    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    RT_detector = comm_dict["RT_detector"]
    move_detector = comm_dict["move_detector"]
    saved_path = comm_dict["saved_path"]
    with_Cornelio = comm_dict["with_Cornelio"]
    
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
    
    n, steps, _ = set_cn.shape
    
    text = "_sim" if simuldistr else ""
    mean = "_mean" if mean_z else ""
    n_batch = n*n_trials // chunk_size if (n*n_trials%chunk_size)==0 else (n*n_trials // chunk_size) + 1
    
    # Input per batch
    test_data = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    cont_data = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
    
    cont = torch.zeros((1, chunk_size, 4)).to(device)
    cont[:, :, 3] = 1
    
    if compute:
        RT_pred_list = []
        move_perc_list = []
        for i, SSD_short in enumerate(SSD_short_list):
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
     
        np.savez(saved_path / f"SSRT_short_{n_trials}{text}{mean}.npz", move_perc_arr=move_perc_arr, RT_pred_arr=RT_pred_arr)
    else:
        with np.load(saved_path / f"SSRT_short_{n_trials}{text}{mean}.npz", allow_pickle=True) as loaded_file:
            move_perc_arr = loaded_file["move_perc_arr"]
            RT_pred_arr = loaded_file["RT_pred_arr"]

    perc_mov = np.array([0.1, 0.3, 0.6, 0.8])
    
    RT_val = np.quantile(RT_pred_arr, move_perc_arr)
    SSRT_val = (RT_val*tau - SSD_short_list)*5
    
    SSRT_val = SSRT_val[slice_start:slice_end]
    move_perc_arr = move_perc_arr[slice_start:slice_end]
    SSD_short_list_sliced = SSD_short_list[slice_start:slice_end]


    idxs = np.abs(move_perc_arr[:, None] - perc_mov[None, :]).argmin(axis=0)
    SSRT_Piero = SSRT_val[idxs]
    for i, idx in enumerate(idxs):
        print(f"SSD corresponding to inhibitory probability of {(1-perc_mov[i])*100:.0f}% for Monkey P: {SSD_short_list_sliced[idx]*5}ms")
    
    #labels = ["CS", "DS", "EH", "IM", "IW", "SW"]
    SSRTs = np.array([[307, 182, 165, 154], [275, 192, 155, 148], [200, 190, 176, 171], [286, 243, 203, 197], [179, 151, 128, 123], [270, 188, 171, 179]])
    xlabel = ["SSD 1", "SSD 2", "SSD 3", "SSD 4"]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xlabel, SSRT_Piero, color=color_Piero, zorder=2)  # Piero
    ax.scatter(xlabel, SSRT_Piero, marker="d", color='black', zorder=3)
    
    if with_Cornelio:
        with np.load(saved_path.parent / with_Cornelio / f"SSRT_short_{n_trials}{text}{mean}.npz", allow_pickle=True) as loaded_file:
            move_perc_arr = loaded_file["move_perc_arr"]
            RT_pred_arr = loaded_file["RT_pred_arr"]

        RT_val = np.quantile(RT_pred_arr, move_perc_arr)
        SSRT_val = (RT_val*tau - SSD_short_list)*5
        
        move_perc_arr = move_perc_arr[slice_start:slice_end]
        SSRT_val = SSRT_val[slice_start:slice_end]
        
        idxs = np.abs(move_perc_arr[:, None] - perc_mov[None, :]).argmin(axis=0)
        SSRT_Cornelio = SSRT_val[idxs]
        for i, idx in enumerate(idxs):
            print(f"SSD corresponding to inhibitory probability of {(1-perc_mov[i])*100:.0f}% for Monkey C: {SSD_short_list_sliced[idx]*5}ms")       
    
        ax.plot(xlabel, SSRT_Cornelio, color=color_Cornelio, zorder=2)
        ax.scatter(xlabel, SSRT_Cornelio, marker="d", color='black', zorder=3) # Cornelio

    for i in range(6):
        ax.plot(xlabel, SSRTs[i], color=color_humans, alpha=alpha, zorder=1)

    ax.set_ylabel("SSRT")
    ax.set_ylim(ylims)
    ax.set_yticks(y_ticks)
    
    # ── Save ──────────────────────────────────────────────────────────────────────
    fig_file = saved_path / f'SSRT_vs_SSD{text}{mean}.png'
    plt.savefig(fig_file)
    plt.show()      
                                            

def SSRT_plot(comm_dict, diff_dict): 
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    t = comm_dict["t"]
    z_dim = comm_dict["z_dim"]
    tau = comm_dict["tau"]
    RT_detector = comm_dict["RT_detector"]
    move_detector = comm_dict["move_detector"]
    saved_path = comm_dict["saved_path"]
    with_Cornelio = comm_dict["with_Cornelio"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    elinewidth = diff_dict["elinewidth"]
    capsize = diff_dict["capsize"]
    ms = diff_dict["ms"]
    SSD_interval = diff_dict["SSD_interval"]
    RT_groups = diff_dict["RT_groups"]
    move_frac = diff_dict["move_frac"]
    compute = diff_dict["compute"]
    logit_move = diff_dict["logit_move"]
    cut_tail = diff_dict["cut_tail"]
    frac_tail = diff_dict["frac_tail"]
    no_zero = diff_dict["no_zero"]
    color_point = diff_dict["color_point"]
    color_line = diff_dict["color_line"]
    chunk_size = diff_dict["chunk_size"]
    inset_dim = diff_dict["inset_dim"]
    inset_font = diff_dict["inset_font"]
    figsize = diff_dict["figsize"]
    mean_z = diff_dict["mean_z"]
    
    set_cn = data["set_cn_ordRT"]
    cont_cn = data["cont_cn_ordRT"]
    RT_cn = data["RT_cn_ordRT"]
    
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
                    if mean_z:
                        _, z, _ = dmm.inference(test_set, cont_c)
                    else:
                        z, _, _ = dmm.inference(test_set, cont_c)

                    # Teacher e z iniziale
                    z_gen = z[teacher, torch.arange(len(teacher), device=device)].unsqueeze(0)
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

            # Pulizia NaN
            nan_mask = np.isnan(move_perc).any(axis=1)
            move_perc = move_perc[~nan_mask]
            RT_time = RT_time[~nan_mask]
            RT_cn_step = RT_cn_step[~nan_mask]
            set_cn = set_cn[~nan_mask]
            cont_cn = cont_cn[~nan_mask]
            
            mask = move_perc <= 0.5
            SSRT_critic = mask.argmax(axis=1)
            critic_t = RT_cn_step - SSRT_critic 

            test_cn = torch.from_numpy(set_cn).float().to(device).permute(1, 0, 2)
            test_c = torch.from_numpy(cont_cn).float().to(device).permute(1, 0, 2)

            z_test, _, _ = dmm.inference(test_cn, test_c)
            z_critic = z_test[critic_t, torch.arange(set_cn.shape[0])]
            z_critic = z_critic.cpu().detach().numpy()

            cont = torch.zeros((1, n_trials, 4)).to(device)
            cont[:, :, 3] = 1
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
                test_set = torch.from_numpy(set_cn[i]).float().to(device).unsqueeze(1).repeat_interleave(n_trials, dim=1)
                cont_c = torch.from_numpy(cont_cn[i]).float().to(device).unsqueeze(1).repeat_interleave(n_trials, dim=1)

                # --- Inference
                if mean_z:
                    _, z, _ = dmm.inference(test_set, cont_c)
                else:
                    z, _, _ = dmm.inference(test_set, cont_c)
                z_gen = z[:teacher]
                cont_go = cont_c[teacher].unsqueeze(0)

                z_tmp_stop = z_gen.clone()
                z_tmp_go = z_gen.clone()

                # --- Generazione z_teach
                alone = steps - teacher
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
        np.savez(saved_path / f"SSRT_plot_{n_trials}.npz", move_perc=move_perc, RT_stop=RT_stop, RT_go=RT_go, RT_time=RT_time,
                SSRT_critic=SSRT_critic, z_mean_stop=z_mean_stop, z_mean_go=z_mean_go, z_critic=z_critic, frac_realizations=frac_realizations)
    else:
        with np.load(saved_path / f"SSRT_plot_{n_trials}.npz", allow_pickle=True) as loaded_file:
            move_perc = loaded_file["move_perc"]
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
    print(f"Of the {len(mask)} trials of Monkey P kept, {len(mask)-mask.sum()} are removed as they result in SSRT=0")
    
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
    mean_SSRT_all = np.argmax(mean_move_perc <= 0.5)*tau*5
    
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
    
    # --- STIMA DIREZIONE ---
    reg_mean = LinearRegression().fit(RT_array[:, None], SSRT_mean_array)
    pred_SSRT_mean = reg_mean.predict(RT_array[:, None])
    
    f, ax = plt.subplots(figsize=figsize)
    ax.errorbar(RT_array, SSRT_mean_array, yerr=SSRT_std_array, fmt='o', color=color_point, ecolor='black', 
                             elinewidth=elinewidth, linestyle='none',capsize=capsize, ms=ms)
    ax.plot(RT_array, pred_SSRT_mean, "--", color = color_line, 
               label = f"linear fit: m={reg_mean.coef_.item():.2f}, q={reg_mean.intercept_.item():.2f}")
    
    # Calcola i limiti comuni
    min_val_x = ax.get_xlim()[0]
    min_val_y = ax.get_ylim()[0]
    max_val_x = ax.get_xlim()[1]
    max_val_y = ax.get_ylim()[1]
    
    dx = (max_val_x - min_val_x)//3
    dy = (max_val_y - min_val_y)//3

    ax.set_xlabel('mean RT ($ms$)')
    ax.set_ylabel('SSRT')
    ax.set_xticks([int(((min_val_x + dx)//100 + 1)*100), int(((min_val_x + 2*dx)//100 + 1)*100)])
    ax.set_xticklabels([int(((min_val_x + dx)//100 + 1)*100), int(((min_val_x + 2*dx)//100 + 1)*100)])
    ax.set_yticks([int(((min_val_y + dy)//10)*10), int(((min_val_y + 2*dy)//10 + 1)*10)])
    ax.set_yticklabels([int(((min_val_y + dy)//10)*10), int(((min_val_y + 2*dy)//10 + 1)*10)])

    
    if with_Cornelio:
        with np.load(saved_path.parent / with_Cornelio / f"SSRT_plot_{n_trials}.npz", allow_pickle=True) as loaded_file:
                move_perc_c = loaded_file["move_perc"]
                RT_stop_c = loaded_file["RT_stop"]
                RT_go_c = loaded_file["RT_go"]
                RT_time_c = loaded_file["RT_time"]
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
        print(f"Of the {len(mask_c)} trials of Monkey C kept, {len(mask_c)-mask_c.sum()} are removed as they result in SSRT=0")

        if no_zero:
            move_perc_c = move_perc_c[mask_c]
            RT_time_masked_c = RT_time_masked_c[mask_c]
            SSRT_critic_c = SSRT_critic_c[mask_c]
            RT_stop_c = RT_stop_c[mask_c]
            RT_go_c = RT_go_c[mask_c]

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
        RT_array_c = np.array(RT_list_c)

        # --- STIMA DIREZIONE ---
        reg_mean_c = LinearRegression().fit(RT_array_c[:, None], SSRT_mean_array_c)
        pred_SSRT_mean_c = reg_mean_c.predict(RT_array_c[:, None])
    
        axins = ax.inset_axes(inset_dim)
        axins.errorbar(RT_array_c, SSRT_mean_array_c, yerr=SSRT_std_array_c, fmt='o', color=color_point, ecolor='black', 
                                 elinewidth=elinewidth/2, linestyle='none',capsize=capsize/2, ms=ms/2)
        axins.plot(RT_array_c, pred_SSRT_mean_c, "--", color = color_line)

        # Calcola i limiti comuni
        min_val_y_c = axins.get_ylim()[0]
        max_val_y_c = axins.get_ylim()[1]

        dy_c = (max_val_y_c - min_val_y_c)//3

        axins.set_xticks([500, 700])
        axins.set_xticklabels([500, 700])
        axins.set_yticks([int(((min_val_y_c + dy_c)//10)*10), int(((min_val_y_c + 2*dy_c)//10 + 1)*10)])
        axins.set_yticklabels([int(((min_val_y_c + dy_c)//10)*10), int(((min_val_y_c + 2*dy_c)//10 + 1)*10)])
        axins.tick_params(labelsize=inset_font)
    
    fig_file = saved_path / 'SSRT_vs_RT.png'
    plt.savefig(fig_file)

    corr = np.corrcoef(RT_array, mean_SSRT_array)[0, 1]
    print(f"the correlation between RT and SSRT is {corr:.2f}")
    print(f"SSRT medio: {mean_SSRT_all}ms")


def consistent_sim(comm_dict, diff_dict):
    
    dmm = comm_dict["dmm"]
    device = comm_dict["device"]
    tau = comm_dict["tau"]
    RT_detector = comm_dict["RT_detector"]
    saved_path = comm_dict["saved_path"]
    with_Cornelio = comm_dict["with_Cornelio"]
    
    data = diff_dict["data"]
    n_trials = diff_dict["n_trials"]
    sim_start = diff_dict["sim_start"]
    cmap = diff_dict["cmap"]
    n_ticks = diff_dict["n_ticks"]
    bins = diff_dict["bins"]
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
    
    _, steps, _ = test_set.shape
    
    _, steps, _ = test_set.shape
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
        z_teach = z_teach.permute(1, 0, 2)
        RT_output = RT_detector(z_teach)
        RT_estimate = prob_to_RT(RT_output, tau)    

        y_pred = y_pred.permute(1, 0, 2)
        y_pred = y_pred.cpu().detach().numpy()
        MUA_pred = y_pred.mean(2)
        RT_est_sort = np.argsort(RT_estimate)
        MUA_pred = MUA_pred[RT_est_sort]
        
        if not mean_z:
            test_RT = test_RT.repeat(n_trials)
        RT_true = test_RT*5
        RT_pred = RT_estimate*tau*5
        
        np.savez(saved_path / f"RT_distribution_{n_trials}.npz", MUA_pred=MUA_pred, RT_true=RT_true, RT_pred=RT_pred)
    else:
        with np.load(saved_path / f"RT_distribution_{n_trials}.npz", allow_pickle=True) as loaded_file:
            MUA_pred = loaded_file["MUA_pred"]
            RT_true = loaded_file["RT_true"]
            RT_pred = loaded_file["RT_pred"]

    vmin = -3
    vmax = 3

    x_positions = np.arange(200//tau) 
    x_labels = x_positions * (tau*5) 

    x_ticks = len(x_positions)
    n_xticks, _ = n_ticks

    #---------- FIG 4A ----------#
    fig, ax = plt.subplots()
    im1 = ax.imshow(MUA_true[:, 56//tau:], cmap = cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(x_positions[::x_ticks//n_xticks]) 
    ax.set_xticklabels(x_labels[::x_ticks//n_xticks])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlabel('Time from GO signal ($ms$)')
    ax.set_ylabel('Trial #')
    cbar=plt.colorbar(im1, ax=ax)
    cbar.set_ticks([vmin, 0, vmax])
    
    fig_file = saved_path / 'MUA_true.png'
    plt.savefig(fig_file)
    
    #---------- FIG 4B ----------#
    fig, ax = plt.subplots()
    im2 = ax.imshow(MUA_pred[:, 56//tau:], cmap = cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(x_positions[::x_ticks//n_xticks]) 
    ax.set_xticklabels(x_labels[::x_ticks//n_xticks])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlabel('Time from GO signal ($ms$)')
    ax.set_ylabel('Trial #')
    cbar=plt.colorbar(im1, ax=ax)
    cbar.set_ticks([vmin, 0, vmax])
    
    fig_file = saved_path / 'MUA_gen.png'
    plt.savefig(fig_file)
    
    
    #---------- FIG 4A ----------#
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
    ax.hist(RT_true, bins=bin_edges, density=True, alpha = alpha, color=color_true, edgecolor='none')
    ax.hist(RT_pred, bins=bin_edges, density=True, alpha = alpha, color=color_pred, edgecolor='none')
    ax.set_xticks(x_ticks_hist)
    ax.set_xticklabels(x_ticks_hist)
    ax.set_yticks([])
    # Add labels and title
    ax.set_xlabel('Reaction Time ($ms$)')
    ax.set_ylabel('Counts')
    
    if with_Cornelio:
        with np.load(saved_path.parent / with_Cornelio / f"RT_distribution_{n_trials}.npz", allow_pickle=True) as loaded_file:
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
        
        axins.hist(RT_true, bins=bin_edges, density=True, alpha = alpha, color=color_true, edgecolor='none')
        axins.hist(RT_pred, bins=bin_edges, density=True, alpha = alpha, color=color_pred, edgecolor='none')
        
        axins.set_xticks(x_ticks_hist)
        axins.set_xticklabels(x_ticks_hist)
        axins.set_yticks([])
        # Add labels and title
        axins.set_xlabel('Reaction Time ($ms$)', fontsize=inset_font)
        axins.set_ylabel('Counts', fontsize=inset_font)
        axins.tick_params(labelsize=inset_font)

    fig_file = saved_path / 'RT_distribution.png'
    plt.savefig(fig_file)