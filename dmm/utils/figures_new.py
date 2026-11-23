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

# =====================================================================
# Standalone Utility Functions
# =====================================================================

def channel2grid(data):
    """
    Maps 1D/2D/3D/4D channel data into a 10x10 grid representation based on spatial layout.

    Parameters
    ----------
    data : np.ndarray
        Array containing channel data. The last dimension must represent channels.

    Returns
    -------
    np.ndarray
        Array with the channel dimension replaced by a (10, 10) grid.
    """
    n = len(data.shape)
    out_shape = [10, 10]
    out_shape = list(data.shape[:-1]) + out_shape
    grid_data = np.zeros(out_shape)
    
    ch2grid_template = {
        97: [0, 0], 41: [0, 1], 39: [0, 2], 37: [0, 3], 43: [0, 4], 45: [0, 5], 47: [0, 6], 1: [0, 7], 5: [0, 8], 98: [0, 9],
        96: [1, 0], 73: [1, 1], 95: [1, 2], 25: [1, 3], 33: [1, 4], 24: [1, 5], 22: [1, 6], 3: [1, 7], 7: [1, 8], 9: [1, 9],
        94: [2, 0], 75: [2, 1], 93: [2, 2], 26: [2, 3], 35: [2, 4], 16: [2, 5], 20: [2, 6], 10: [2, 7], 11: [2, 8], 13: [2, 9],
        92: [3, 0], 77: [3, 1], 91: [3, 2], 29: [3, 3], 55: [3, 4], 18: [3, 5], 14: [3, 6], 8: [3, 7], 6: [3, 8], 15: [3, 9],
        90: [4, 0], 79: [4, 1], 89: [4, 2], 31: [4, 3], 49: [4, 4], 57: [4, 5], 61: [4, 6], 12: [4, 7], 4: [4, 8], 17: [4, 9],
        88: [5, 0], 81: [5, 1], 48: [5, 2], 46: [5, 3], 51: [5, 4], 53: [5, 5], 59: [5, 6], 71: [5, 7], 2: [5, 8], 19: [5, 9],
        86: [6, 0], 83: [6, 1], 44: [6, 2], 42: [6, 3], 38: [6, 4], 63: [6, 5], 65: [6, 6], 67: [6, 7], 69: [6, 8], 21: [6, 9],
        84: [7, 0], 85: [7, 1], 50: [7, 2], 40: [7, 3], 36: [7, 4], 34: [7, 5], 32: [7, 6], 30: [7, 7], 28: [7, 8], 23: [7, 9],
        82: [8, 0], 87: [8, 1], 52: [8, 2], 54: [8, 3], 74: [8, 4], 72: [8, 5], 70: [8, 6], 62: [8, 7], 27: [8, 8], 66: [8, 9],
        99: [9, 0], 80: [9, 1], 78: [9, 2], 76: [9, 3], 56: [9, 4], 58: [9, 5], 60: [9, 6], 68: [9, 7], 64: [9, 8], 100: [9, 9],
    }
    
    ch2grid = {(key - 1): value for key, value in ch2grid_template.items()}
    valid_channels = {k: v for k, v in ch2grid.items() if k < data.shape[-1]}
    
    for ch, (i, j) in valid_channels.items():
        if n == 1:
            grid_data[i, j] = data[ch]
        elif n == 2:
            grid_data[:, i, j] = data[:, ch]
        elif n == 3:
            grid_data[:, :, i, j] = data[:, :, ch]
        else:
            grid_data[:, :, :, i, j] = data[:, :, :, ch]
            
    return grid_data


def binary_output(prob_logits):
    """
    Converts probabilities/logits into binary labels (0.0 or 1.0) using sigmoid and a 0.5 threshold.

    Parameters
    ----------
    prob_logits : torch.Tensor
        Input logits or raw network outputs.

    Returns
    -------
    np.ndarray
        Binary labels.
    """
    prob_pred = torch.sigmoid(prob_logits)
    predicted_labels = (prob_pred > 0.5).float()
    return predicted_labels.cpu().detach().numpy()


def prob_to_RT(output, tau):
    """
    Calculates the reaction time index from probability outputs.

    Parameters
    ----------
    output : torch.Tensor
        Probability outputs.
    tau : int or float
        Time resolution parameter.

    Returns
    -------
    np.ndarray
        Reaction time step indices.
    """
    output_clamped = torch.clamp(output, 0, 1)
    output_clamped = output_clamped.cpu().detach().numpy()
    RT = (output_clamped * 200) // tau
    return RT


def add_gradient_line(ax, x, y, cmap="viridis", lw=5, vmin=0.4, vmax=1.0, alpha=1):
    """
    Adds a line segment collection to the axes with a color gradient.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the line on.
    x : np.ndarray
        X coordinates.
    y : np.ndarray
        Y coordinates.
    cmap : str
        Colormap name.
    lw : float
        Line width.
    vmin : float
        Min normalizer value.
    vmax : float
        Max normalizer value.
    alpha : float
        Transparency.

    Returns
    -------
    matplotlib.collections.LineCollection
        The added line collection object.
    """
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
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


def delta_x_via_jvp(dmm, z0, d, l, data=None):
    """
    Uses Jacobian-Vector Product (JVP) to approximate the change in data space (delta_x)
    required to move along direction d in the latent space by length l.

    Parameters
    ----------
    dmm : nn.Module
        The deep Markov model instance.
    z0 : torch.Tensor
        Reference latent states.
    d : np.ndarray
        Latent space direction vector.
    l : float
        Perturbation magnitude.
    data : optional
        Unused placeholder.

    Returns
    -------
    np.ndarray
        Approximated delta_x in observation space.
    """
    device = z0.device
    z0 = z0.detach().clone().requires_grad_(True)
    delta_z = torch.from_numpy(-d * l).float().to(device)
    if z0.ndim == 2:
        delta_z = torch.tile(delta_z, (z0.shape[0], 1))

    def decoder_mean(z):
        x_mean, _ = dmm.generation_x(z)
        return x_mean

    y, jvp_out = jvp(decoder_mean, (z0,), (delta_z,))
    jvp_out = jvp_out.cpu().detach().numpy()
    print(jvp_out.shape)
    return jvp_out


def compute_dx_for_last_step_ElasticNet(dmm, x_seq, c_seq, delta_z, alpha_l1=1e-3, l1_ratio=1, n_iter=1):
    """
    Computes delta_x for the last sequence timestep satisfying J_h(x_t) delta_x ≈ delta_z
    using ElasticNet regularization.

    Parameters
    ----------
    dmm : nn.Module
        Deep Markov model.
    x_seq : torch.Tensor
        Sequence of observations (T, N, D).
    c_seq : torch.Tensor
        Sequence of contexts (T, N, c_dim).
    delta_z : torch.Tensor
        Target latent perturbation (N, d).
    alpha_l1 : float
        L1 regularization coefficient.
    l1_ratio : float
        ElasticNet mixing parameter.
    n_iter : int
        Number of relinearization steps.

    Returns
    -------
    tuple
        Updated x_seq (T, N, D) and last-timestep delta_x (N, D).
    """
    device = x_seq.device
    dmm.train()
    T, N, D = x_seq.shape
    d = dmm.z_dim
    
    delta_z = delta_z.detach().clone().to(device).repeat(N, 1).view(N, d)

    for _ in range(n_iter):
        x_curr = x_seq.clone().detach().requires_grad_(True)
        c_curr = c_seq.clone().detach()

        _, z_mean, _ = dmm.inference(x_curr, c_curr)
        z_mean_t = z_mean[-1]

        dx_last_list = []

        for n in range(N):
            J_rows = []
            for i in range(d):
                grad_i = torch.autograd.grad(
                    z_mean_t[n, i],
                    x_curr,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True
                )[0]
                grad_last = grad_i[-1, n, :].unsqueeze(0)
                J_rows.append(grad_last)
            J = torch.cat(J_rows, dim=0).detach().cpu().numpy()

            delta_z_np = delta_z[n].detach().cpu().numpy().flatten()

            model = ElasticNet(alpha=alpha_l1, l1_ratio=l1_ratio, fit_intercept=False, max_iter=10000)
            model.fit(J, delta_z_np)
            dx_last_n = torch.from_numpy(model.coef_).float().to(device).unsqueeze(0)

            dx_last_list.append(dx_last_n)

        dx_last = torch.cat(dx_last_list, dim=0)
        x_seq[-1] = x_seq[-1] + dx_last
        dx_last = dx_last.cpu().detach().numpy()
        
    return x_seq, dx_last


# =====================================================================
# Visualizer Class
# =====================================================================

class DMMVisualizer:
    """
    Encapsulates state and helper functions for plotting visualizations
    and running analysis metrics for the Deep Markov Model.
    """
    def __init__(self, dmm=None, device=None, tau=None, saved_path=None, z_dim=None,
                 RT_detector=None, move_detector=None, with_Cornelio=None):
        self.dmm = dmm
        self.device = device
        self.tau = tau
        self.saved_path = saved_path
        self.z_dim = z_dim
        self.RT_detector = RT_detector
        self.move_detector = move_detector
        self.with_Cornelio = with_Cornelio

    @classmethod
    def from_dict(cls, comm_dict):
        """Creates DMMVisualizer using keys from comm_dict."""
        if comm_dict is None:
            return cls()
        return cls(
            dmm=comm_dict.get("dmm"),
            device=comm_dict.get("device"),
            tau=comm_dict.get("tau"),
            saved_path=comm_dict.get("saved_path"),
            z_dim=comm_dict.get("z_dim"),
            RT_detector=comm_dict.get("RT_detector"),
            move_detector=comm_dict.get("move_detector"),
            with_Cornelio=comm_dict.get("with_Cornelio")
        )

    def inference_with_trials(self, data_set, cont_set, n_trials, chunk_size=10):
        """
        Executes model inference in chunks to prevent GPU out-of-memory errors.
        """
        n_samples, steps, features = data_set.shape
        data_set_t = torch.from_numpy(data_set).float().permute(1, 0, 2)
        cont_set_t = torch.from_numpy(cont_set).float().permute(1, 0, 2)
        
        z_mean_accum = np.zeros((steps, n_samples, self.z_dim), dtype=np.float32)

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            batch_size = end - start

            x_chunk = data_set_t[:, start:end, :].repeat_interleave(n_trials, dim=1).to(self.device)
            c_chunk = cont_set_t[:, start:end, :].repeat_interleave(n_trials, dim=1).to(self.device)

            with torch.no_grad():
                z, _, _ = self.dmm.inference(x_chunk, c_chunk)

            z_mean_chunk = z.cpu().numpy().reshape(steps, batch_size, n_trials, self.z_dim).mean(2)
            z_mean_accum[:, start:end, :] = z_mean_chunk
            torch.cuda.empty_cache()

        return np.transpose(z_mean_accum, (1, 0, 2))

    def RT_pred_performance(self, diff_dict):
        """
        Plots scatter plots and regression fit for predicted vs true Reaction Times.
        """
        data = diff_dict["data"]
        n_trials = diff_dict["n_trials"]
        
        set_cn = data["set_cn_ordRT"]
        cont_cn = data["cont_cn_ordRT"]
        RT_cn = data["RT_cn_ordRT"]
        
        set_cn_t = torch.from_numpy(set_cn).float().to(self.device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
        cont_cn_t = torch.from_numpy(cont_cn).float().to(self.device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
        
        z_cn, _, _ = self.dmm.inference(set_cn_t, cont_cn_t)
        z_cn = z_cn.permute(1, 0, 2)
        RT_output = self.RT_detector(z_cn)
        RT_estimate = prob_to_RT(RT_output, self.tau)  
        RT_cn_norm = (RT_cn + 56) // self.tau
        
        correlation = np.corrcoef(RT_cn_norm, RT_estimate)[0, 1]
        A = np.vstack([RT_cn_norm, np.ones(len(RT_cn_norm))]).T
        m, c = np.linalg.lstsq(A, RT_estimate, rcond=None)[0]
        RT_estimate_fit = m * RT_cn_norm + c

        plt.figure(figsize=(9, 7))
        plt.scatter(RT_cn_norm, RT_estimate, color='royalblue', alpha=0.6, label='Dati')
        plt.plot(RT_cn_norm, RT_estimate_fit, color='tomato', lw=2, label=f'Fit: y={m:.2f}x+{c:.2f}')
        plt.xlabel('True RT')
        plt.ylabel('Predicted RT (using detector)')
        plt.title(f'Scatter & Fit. Correlazione Pearson = {correlation:.2f}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def cs_ws_SSD_hist(self, diff_dict):
        """
        Plots histograms of Stop Signal Delays (SSD) for correct stop and wrong stop trials.
        """
        data = diff_dict["data"]
        SSD_cs = data["SSD_cs_ordSSD"]
        SSD_ws = data["SSD_ws_ordSSD"]
        
        print(f"mean train_SSD_ws: {SSD_ws.mean()*5}ms")
        print(f"mean train_SSD_cs: {SSD_cs.mean()*5}ms")
        print("cs_SSD:", np.unique(SSD_cs))
        print("ws_SSD:", np.unique(SSD_ws))
        
        stop_cs = (SSD_cs + 56) * 5
        stop_ws = (SSD_ws + 56) * 5
        
        num_bins = 20
        min_value = min(stop_cs.min(), stop_ws.min())
        max_value = max(stop_cs.max(), stop_ws.max())
        bin_edges = np.linspace(min_value, max_value, num_bins + 1)
        
        _, ax = plt.subplots()
        ax.hist(stop_ws, bins=bin_edges, alpha=0.5, color='red', edgecolor='black', label="wrong stop")
        ax.hist(stop_cs, bins=bin_edges, alpha=0.5, color='skyblue', edgecolor='black', label="correct stop")
        ax.set_xlabel('SSD')
        ax.set_ylabel('# of trials')
        ax.set_title("Histograms of SSDs for wrong and correct stop trials")
        ax.legend()
        plt.show()

    def cn_ws_RT_hist(self, diff_dict):
        """
        Plots histograms comparing RT distributions of correct no-stop and wrong stop trials.
        """
        data = diff_dict["data"]
        RT_cn = data["RT_cn_ordRT"]
        RT_ws = data["RT_ws_ordRT"]
        
        mov_cn = (RT_cn + 56) * 5
        mov_ws = (RT_ws + 56) * 5
        
        num_bins = 30
        min_value = min(mov_cn.min(), mov_ws.min())
        max_value = max(mov_cn.max(), mov_ws.max())
        bin_edges = np.linspace(min_value, max_value, num_bins + 1)
        
        _, ax = plt.subplots()
        ax.hist(mov_cn, bins=bin_edges, density=True, alpha=0.4, color='skyblue', edgecolor='black', label="cn")
        ax.hist(mov_ws, bins=bin_edges, density=True, alpha=0.4, color='red', edgecolor='black', label="ws")
        ax.set_xlabel('Simulation start time ($ms$)')
        ax.set_ylabel('# of trials')
        ax.set_title("Histograms of RTs for correct no-stop trials")
        ax.legend()
        plt.show()

    def infer_latent(self, data, n_trials=1):
        """
        Infers latent vectors from data observations for correct no-stop, wrong stop,
        and correct stop conditions.
        """
        set_cn = torch.from_numpy(data["set_cn_ordRT"]).float().to(self.device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
        cont_cn = torch.from_numpy(data["cont_cn_ordRT"]).float().to(self.device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
        set_ws = torch.from_numpy(data["set_ws_ordRT"]).float().to(self.device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
        cont_ws = torch.from_numpy(data["cont_ws_ordRT"]).float().to(self.device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
        set_cs = torch.from_numpy(data["set_cs_ordSSD"]).float().to(self.device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
        cont_cs = torch.from_numpy(data["cont_cs_ordSSD"]).float().to(self.device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
            
        z_cs, _, _ = self.dmm.inference(set_cs, cont_cs)
        z_cn, _, _ = self.dmm.inference(set_cn, cont_cn)
        z_ws, _, _ = self.dmm.inference(set_ws, cont_ws)

        return z_cn.cpu().detach().numpy(), z_ws.cpu().detach().numpy(), z_cs.cpu().detach().numpy()

    def single_RTcorr(self, diff_dict, sim_start):
        """
        Computes the correlation between predicted and true Reaction Times for a specific simulation start index.
        """
        data = diff_dict["data"]
        n_trials = diff_dict["n_trials"]
        mean_corr = diff_dict["mean_corr"]
        min_n = diff_dict["min_n"]
        
        set_cn = data["set_cn_ordRT"]
        cont_c = data["cont_cn_ordRT"]
        RT_cn = data["RT_cn_ordRT"]
        
        _, steps, _ = set_cn.shape
        RT_cn_ordered = RT_cn // self.tau
        teacher = sim_start // (5 * self.tau)
        alone = steps - teacher

        mask = (RT_cn_ordered + 56 // self.tau) > teacher
        s = mask.sum()
        set_cn = set_cn[mask]
        cont_c = cont_c[mask]
        RT_cn_ordered_filt = RT_cn_ordered[mask]
        
        if s > min_n:
            set_cn_t = torch.from_numpy(set_cn).float().to(self.device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1) 
            cont_c_t = torch.from_numpy(cont_c).float().to(self.device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)

            z_cn, _, _ = self.dmm.inference(set_cn_t, cont_c_t)
            z_teach = z_cn[:teacher]
            for step in range(alone):
                z_mean_gen, z_cov_gen = self.dmm.generation_z(z_teach[-1].unsqueeze(0), cont_c_t[teacher + step].unsqueeze(0))
                z_gen = self.dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                z_teach = torch.cat((z_teach, z_gen), dim=0)
            
            if not mean_corr:
                z_teach = z_teach.reshape(steps, s, n_trials, self.z_dim).mean(2)
                
            RT_output = self.RT_detector(z_teach.permute(1, 0, 2))
            RT_pred = prob_to_RT(RT_output, self.tau)

            if mean_corr:
                RT_pred = RT_pred.reshape(s, n_trials)
                mask_peak = np.all((RT_pred + 56 // self.tau) != teacher, axis=1)
                RT_cn_ordered_filt = RT_cn_ordered_filt[mask_peak]
                RT_pred_filt = RT_pred[mask_peak]
                non_zero_counts = np.count_nonzero(RT_pred_filt, axis=1)
                K = np.min(non_zero_counts)
                N = RT_pred_filt.shape[0]
                RT_est = np.full((N, K), np.nan)
                for i in range(N):
                    non_zeros = RT_pred_filt[i][RT_pred_filt[i] != 0]
                    RT_est[i] = non_zeros[:K]
                corr = np.array([np.corrcoef(RT_cn_ordered_filt, RT_est[:, j])[0, 1] for j in range(K)])  
                correlation = np.mean(corr)
                corr_std = np.std(corr)
            else:
                mask_peak = (RT_pred + 56 // self.tau) != teacher
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

    def correlation_vs_gentime(self, diff_dict):     
        """
        Plots simulation start time vs RT prediction correlation.
        """
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
        figsize = diff_dict["figsize"]
        
        RT_cn = data["RT_cn_ordRT"]
        
        if compute:
            correlations = np.zeros(len(sim_start_array))
            corr_stds = np.zeros(len(sim_start_array))
            for cont, sim_start in enumerate(sim_start_array):
                _, _, correlation, corr_std = self.single_RTcorr(diff_dict, sim_start)
                correlations[cont] = correlation
                corr_stds[cont] = corr_std

            mask_new = correlations < 2
            sim_start_array_new = sim_start_array[mask_new]
            corr_stds_new = corr_stds[mask_new]
            correlations_new = correlations[mask_new]
            np.savez(self.saved_path / "RT_correlations.npz", sim_start_array=sim_start_array_new, correlations=correlations_new, corr_stds=corr_stds_new)
        else:
            with np.load(self.saved_path / "RT_correlations.npz") as loaded_file:
                correlations_new = loaded_file["correlations"]
                corr_stds_new = loaded_file["corr_stds"]
                sim_start_array_new = loaded_file["sim_start_array"]
        
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.scatter(sim_start_array_new, correlations_new, color=color_corr, edgecolors=color_edge)
        ax1.axvline(56 * 5, color="black", linestyle="--")
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
       
        fig_file = self.saved_path / 'RT_correlation.png'
        plt.savefig(fig_file)
        plt.show()
            
        dict_corr = {a: (b, c) for a, b, c in zip(sim_start_array_new, correlations_new, corr_stds_new)}
        print(f"RT correlations at the GO cue: {correlations_new[56 // self.tau]}")
        return dict_corr

    def DMM_EV(self, diff_dict):
        """
        Plots explained variance of DMM latent space using PCA.
        """
        color = diff_dict["color"]
        x_label = diff_dict["x_label"]
        x_lims = diff_dict["x_lims"]
        y_lims = diff_dict["y_lims"]
        delta_y = diff_dict["delta_y"]
        alpha = diff_dict["alpha"]
        model_dir = diff_dict["model_dir"]
        
        model_path = self.saved_path.parent / model_dir
        with np.load(model_path / "data_split.npz", allow_pickle=True) as loaded_file:
            test_set = loaded_file["test_set"]
            test_direction = loaded_file["test_direction"]
            test_SSD = loaded_file["test_SSD"]
                
        params = {
            "cfg": model_path / "config.ini",
            "device": self.device,
            "saved_dir": model_path
        }
        
        from dmm.learning_algo_dir import LearningAlgorithm_dir
        
        weights_dir = "DKF_simple_final_epoch149.pt"
        learning_algo = LearningAlgorithm_dir(params=params)
        dmm_model = learning_algo.model
        dmm_model.load_state_dict(torch.load(model_path / weights_dir, map_location=self.device))
        dmm_model.eval()
        z_dim = dmm_model.z_dim
     
        cont_test = one_hot_cont(test_SSD, test_direction, dmm_model.tau)
        test_data = torch.from_numpy(test_set).float().to(self.device).permute(1, 0, 2)
        cont_test = torch.from_numpy(cont_test).float().to(self.device).permute(1, 0, 2)
        z, _, _ = dmm_model.inference(test_data, cont_test)
        z = z.cpu().detach().numpy().reshape(-1, z_dim) 
        
        scaler = StandardScaler()
        z_scaled = scaler.fit_transform(z)

        pca = PCA(n_components=z_dim)
        pca.fit(z_scaled)

        explained_variance = pca.explained_variance_ratio_
        z_vec = (np.arange(z_dim) + 1) / (z_dim + 1)
        
        _, ax = plt.subplots()
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
            ax.text(xi, yi + delta_y, f"{yi:.2f}", ha='center')
       
        fig_file = model_path / 'DMM_EV.png'
        plt.savefig(fig_file)
        plt.show()

    def DMM_vs_dim(self, diff_dict):
        """
        Compares DMM performance metrics across different dimensions (bar plots).
        """
        width = diff_dict["width"]
        x = diff_dict["x"]
        y_lim = diff_dict["y_lim"]
        y_ticks = diff_dict["y_ticks"]
        colors = diff_dict["colors"]
        model_dirs = diff_dict["model_dirs"]
        
        data = []
        for i in range(len(model_dirs)):
            model_path = self.saved_path.parent / model_dirs[i]
            
            checkpoint_dir = torch.load(model_path / "dir_dict_z", weights_only=False, map_location=self.device)
            accuracy_dir = checkpoint_dir["accuracy"]

            checkpoint_NonMarkov = torch.load(model_path / "Non_Markov_dict_z", weights_only=False, map_location=self.device)
            checkpoint_MSE = torch.load(model_path / "MSE", weights_only=False, map_location=self.device)
            
            MSE_rec = checkpoint_MSE["MSE_DMM"]
            NMSE_NonMarkov = checkpoint_NonMarkov["MSE_pc"]
            r2_NonMarkov = 1 - NMSE_NonMarkov
            r2_rec = 1 - MSE_rec
            
            data.append([r2_rec, r2_NonMarkov, accuracy_dir])
     
        data = np.array(data).T
        
        y_label = ["$R^2$", "$R^2_D$", "Accuracy"]
        plot_label = ["rec", "prediction", "direction"]
        for i in range(len(y_label)):
            _, ax = plt.subplots()
            ax.bar(x, data[i], color=colors, width=width)
            ax.set_xlim((0, 1))
            ax.set_xticks(x)
            ax.set_xticklabels(["D=2", "D=3", "D=4"])
            ax.set_ylabel(y_label[i])
            ax.set_ylim(y_lim[i])
            ax.set_yticks(y_ticks[i])
            ax.set_yticklabels(y_ticks[i])
            
            for xi, yi in zip(x, data[i]):
                delta_y = 0.01 if i == 2 else 0.02
                ax.text(xi, yi + (delta_y * y_ticks[i][1]), f"{yi:.2f}", ha='center')
            
            fig_file = self.saved_path / f'DMM_vs_dim_{plot_label[i]}.png'
            plt.savefig(fig_file)
            plt.show()

    def PCA_vs_DMM(self, diff_dict):
        """
        Plots comparative bar charts showing R^2 and accuracy differences between DMM and PCA.
        """
        width = diff_dict["width"]
        x = diff_dict["x"]
        y_lim = diff_dict["y_lim"]
        y_ticks = diff_dict["y_ticks"]
        color_DMM = diff_dict["color_DMM"]
        color_PCA = diff_dict["color_PCA"]
        inset_dim = diff_dict["inset_dim"]
        inset_font = diff_dict["inset_font"]
        
        checkpoint_dir_DMM = torch.load(self.saved_path / "dir_dict_z", weights_only=False, map_location=self.device)
        accuracy_dir_DMM = checkpoint_dir_DMM["accuracy"]
        
        checkpoint_dir_PCA = torch.load(self.saved_path / "dir_dict_PCA", weights_only=False, map_location=self.device)
        accuracy_dir_PCA = checkpoint_dir_PCA["accuracy"]
        
        checkpoint_NonMarkov_DMM = torch.load(self.saved_path / "Non_Markov_dict_z", weights_only=False, map_location=self.device)
        checkpoint_NonMarkov_PCA = torch.load(self.saved_path / "Non_Markov_dict_PCA", weights_only=False, map_location=self.device)
     
        checkpoint_Markov_DMM = torch.load(self.saved_path / "Markov_dict_z", weights_only=False, map_location=self.device)
        checkpoint_Markov_PCA = torch.load(self.saved_path / "Markov_dict_PCA", weights_only=False, map_location=self.device)
        
        NMSE_NonMarkov_PCA = checkpoint_NonMarkov_PCA["MSE_pc"]
        NMSE_Markov_PCA = checkpoint_Markov_PCA["MSE_pc"]
        NMSE_NonMarkov_DMM = checkpoint_NonMarkov_DMM["MSE_pc"]
        NMSE_Markov_DMM = checkpoint_Markov_DMM["MSE_pc"]
        
        checkpoint_MSE = torch.load(self.saved_path / "MSE", weights_only=False, map_location=self.device)
        MSE_rec_DMM = checkpoint_MSE["MSE_DMM"]
        MSE_rec_PCA = checkpoint_MSE["MSE_PCA"]
        
        r2_Markov_PCA = 1 - NMSE_Markov_PCA
        r2_Markov_DMM = 1 - NMSE_Markov_DMM
        r2_NonMarkov_PCA = 1 - NMSE_NonMarkov_PCA
        r2_NonMarkov_DMM = 1 - NMSE_NonMarkov_DMM
        r2_rec_PCA = 1 - MSE_rec_PCA
        r2_rec_DMM = 1 - MSE_rec_DMM
        
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
            _, ax = plt.subplots()
            ax.bar(x, data[i], color=color, width=width)
            ax.set_xlim((0, 1))
            ax.set_xticks(x)
            ax.set_xticklabels(["DMM", "PCA"])
            ax.set_ylabel(y_label[i])
            ax.set_ylim(y_lim[i])
            ax.set_yticks(y_ticks[i])
            ax.set_yticklabels(y_ticks[i])

            for xi, yi in zip(x, data[i]):
                delta_y = 0.01 if i == 2 else 0.02
                ax.text(xi, yi + (delta_y * y_ticks[i][1]), f"{yi:.2f}", ha='center')
                
            if i == 1:
                axins = ax.inset_axes(inset_dim)
                axins.bar(x, (r2_Markov_DMM, r2_Markov_PCA), color=color, width=width)
                axins.set_xlim((0, 1))
                axins.set_xticks(x)
                axins.set_xticklabels(["DMM", "PCA"])
                axins.set_ylabel("$R^2_D$", fontsize=inset_font)
                axins.set_ylim((0, 1.2))
                axins.set_yticks((0, 1))
                axins.set_yticklabels((0, 1))
                axins.tick_params(labelsize=inset_font)
                
                for xi, yi in zip(x, (r2_Markov_DMM, r2_Markov_PCA)):
                    axins.text(xi, yi + (0.05 * y_ticks[i][1]), f"{yi:.2f}", ha='center', fontsize=inset_font)
            
            fig_file = self.saved_path / f'DMM_vs_PCA_{plot_label[i]}.png'
            plt.savefig(fig_file)
            plt.show()

    def random_latent_cn_traj(self, data, n_trials=1):
        """
        Retrieves the latent trajectory of a single random trial.
        """
        RT_cn = data["RT_cn_ordRT"]
        steps = 256 // self.tau
        n = len(RT_cn)
        q = random.randint(0, n - 1)
        RT_trial = RT_cn[q]
        
        z_cn, _, _ = self.infer_latent(data, n_trials=n_trials)
        z_dim = z_cn.shape[2]

        if n_trials > 1:
            z_cn = z_cn.reshape(steps, n, n_trials, z_dim)
        
        z_trial = z_cn[:, q]  
        RT = (56 + RT_trial) // self.tau
        return z_trial, RT, q

    def latent_traj_example(self, diff_dict):
        """
        Plots a 3D example trajectory of latent variables over time.
        """
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
        
        z_trial, _, q = self.random_latent_cn_traj(data)
        print(f"traj n.{q}")
            
        x_true_story = z_trial[:, 0]
        y_true_story = z_trial[:, 1]
        z_true_story = z_trial[:, 2]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x_true_story, y_true_story, z_true_story, color=color_line, lw=lw_line)

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

        ax.grid(False)
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
        
        ax.text(lims[0] * offset + origin[0], origin[1], origin[2], "$z_1$", fontsize=fontsize)
        ax.text(origin[0], lims[1] * offset + origin[1], origin[2], "$z_2$", fontsize=fontsize)
        ax.text(origin[0], origin[1], lims[2] * offset + origin[2], "$z_3$", fontsize=fontsize)

        ranges = np.ptp(z_trial, axis=0)
        ax.set_box_aspect(ranges)
        ax.view_init(elev=elev, azim=azim)

        fig_file = self.saved_path / 'trajectory.png'
        plt.savefig(fig_file)
        plt.show()

    def plot_cycle_shift_short(self, diff_dict):
        """
        Shifts the latent states to produce shorter RT distributions and visualizes the results.
        """
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
        z_cn, _, _ = self.infer_latent(data, n_trials)
        
        steps, s, _ = z_cn.shape
        samples = len(RT_cn)
        
        if mean_trials:
            z_cn = z_cn.reshape(steps, samples, n_trials, self.z_dim).mean(2)
            color = RT_cn / RT_cn.max()
        else:
            set_cn = set_cn.repeat(n_trials, 1, 1)
            RT_cn_rep = np.repeat(RT_cn, n_trials)
            color = RT_cn_rep / RT_cn_rep.max()
        z_GO = z_cn[step // self.tau]
        
        long_index = int(z_GO.shape[0] * (1 - f))
        set_cn_long = set_cn[long_index:]
        cont_cn_long = cont_cn[long_index:]
        
        n_long, _, features = set_cn_long.shape
        trial_true = torch.from_numpy(set_cn_long).float().to(self.device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
        cont_RTlong = torch.from_numpy(cont_cn_long).float().to(self.device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
        
        start_sim = step // self.tau + 1 - stimulation_steps
        text = "" if stimolate else "_onech"
        text_mean = "_mean" if mean else ""
        
        if compute:
            reg = LinearRegression().fit(z_GO, color)
            direction = reg.coef_
            direction /= np.linalg.norm(direction)
            delta_z_short = torch.from_numpy(-direction * l).float().to(self.device)
            print("Direzione di variazione (aumenta):", direction)

            delta_z_short = torch.zeros(stimulation_steps, self.z_dim).to(self.device)
            if multi_direction:
                for t in range(stimulation_steps):
                    reg = LinearRegression().fit(z_cn[start_sim + t], color)
                    direction = reg.coef_
                    direction /= np.linalg.norm(direction)
                    delta_z_short[t] = torch.from_numpy(-direction * l).float().to(self.device)
         
            trial_clone = trial_true[:(step // self.tau + 1)].clone()
            y_mean, y_logvar = self.dmm(trial_clone, cont_RTlong[:(step // self.tau + 1)])
            y_pred = self.dmm.reparameterization(y_mean, y_logvar)
            if mean:
                y_mean = y_mean.reshape(-1, n_long, n_trials, features).mean(2)
                residual = trial_clone - y_mean.repeat_interleave(n_trials, dim=1)
            else:
                y_pred = y_pred.reshape(-1, n_long, n_trials, features).mean(2)
                residual = trial_clone - y_pred.repeat_interleave(n_trials, dim=1)
                
            trial_modified = trial_true[:(step // self.tau + 1)].clone() 
            trial_modified_min = trial_true[:(step // self.tau + 1)].clone()
     
            dx_fix = torch.full((set_cn_long.shape[0] * n_trials,), l).to(self.device)
            dx_fix_min = torch.full((set_cn_long.shape[0] * n_trials,), 0.0001).to(self.device)
            dx_array = np.empty([stimulation_steps, 96])
            stim_effect = torch.zeros([stimulation_steps, set_cn_long.shape[0] * n_trials, 96]).to(self.device)
            stim_effect_min = torch.zeros([stimulation_steps, set_cn_long.shape[0] * n_trials, 96]).to(self.device)
            
            for t in range(stimulation_steps):
                print(t)
                if stimolate:
                    _, dx = compute_dx_for_last_step_ElasticNet(
                        self.dmm, 
                        trial_modified[:(start_sim + t + 1)], 
                        cont_RTlong[:(start_sim + t + 1)], 
                        delta_z_short[t] if multi_direction else delta_z_short, 
                        alpha_l1=alpha_L1, 
                        l1_ratio=l1_ratio, 
                        n_iter=1
                    )
                    dx_array[t] = dx.mean(0)
                    trial_modified[start_sim + t] += torch.from_numpy(dx).float().to(self.device)
                else:
                    trial_modified[start_sim + t, :, 63] += dx_fix
                    trial_modified_min[start_sim + t, :, 63] += dx_fix_min
                if t + 1 == stimulation_steps:
                    continue
                
                trial_cut = trial_modified.clone()
                z_modified, _, _ = self.dmm.inference(trial_cut[:(start_sim + t + 1)], cont_RTlong[:(start_sim + t + 1)])
                z_mean_gen, z_cov_gen = self.dmm.generation_z(z_modified[-1].unsqueeze(0), cont_RTlong[start_sim + t + 1].unsqueeze(0))
                z_gen_last = self.dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                y_mean, y_logvar = self.dmm.generation_x(z_gen_last)
                y_pred = self.dmm.reparameterization(y_mean, y_logvar)
                y_gen = y_mean if mean else y_pred
                trial_modified[start_sim + t + 1] = y_gen[0]
                if add_residual:
                    trial_modified[start_sim + t + 1] += residual[start_sim + t + 1]
                stim_effect[t] = trial_modified[start_sim + t + 1] - trial_true[start_sim + t + 1]
                
                trial_cut_min = trial_modified_min.clone()
                z_modified, _, _ = self.dmm.inference(trial_cut_min[:(start_sim + t + 1)], cont_RTlong[:(start_sim + t + 1)])
                z_mean_gen, z_cov_gen = self.dmm.generation_z(z_modified[-1].unsqueeze(0), cont_RTlong[start_sim + t + 1].unsqueeze(0))
                z_gen_last = self.dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                y_mean, y_logvar = self.dmm.generation_x(z_gen_last)
                y_pred = self.dmm.reparameterization(y_mean, y_logvar)
                y_gen = y_mean if mean else y_pred
                trial_modified_min[start_sim + t + 1] = y_gen[0]
                if add_residual:
                    trial_modified_min[start_sim + t + 1] += residual[start_sim + t + 1]
                stim_effect_min[t] = trial_modified_min[start_sim + t + 1] - trial_true[start_sim + t + 1]

            stim_effect = stim_effect.cpu().detach().numpy()
            stim_effect_min = stim_effect_min.cpu().detach().numpy()
            trial_stim = trial_modified.cpu().detach().numpy()
            np.savez(self.saved_path / f"shorter_RT_{l}{text}{text_mean}.npz", trial_stim=trial_stim, stim_effect=stim_effect, stim_effect_min=stim_effect_min, dx_array=dx_array)
        else:
            with np.load(self.saved_path / f"shorter_RT_{l}{text}{text_mean}.npz") as loaded_file:
                trial_stim = loaded_file["trial_stim"]
                dx_array = loaded_file["dx_array"]
                if not stimolate:
                    stim_effect = loaded_file["stim_effect"]
                    stim_effect_min = loaded_file["stim_effect_min"]
                    diff_stim = stim_effect - stim_effect_min
                
            trial_modified = torch.from_numpy(trial_stim).float().to(self.device)
        
        if stimolate:  
            dx_array = channel2grid(dx_array)
            last_stim = dx_array[stimulation_steps].copy()
            last_stim[last_stim < thr] = 0
        else:
            mean_stim_effect = diff_stim.mean(1)
            last_stim = channel2grid(mean_stim_effect)
        
        self.make_RThist(diff_dict, last_stim, trial_modified, trial_true, cont_RTlong, l, 'shorter')

    def plot_cycle_shift_long(self, diff_dict):
        """
        Shifts the latent states to produce longer RT distributions and visualizes the results.
        """
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
        z_cn, _, _ = self.infer_latent(data, n_trials)
        
        steps, _, _ = z_cn.shape
        samples = len(RT_cn)
        
        if mean_trials:
            z_cn = z_cn.reshape(steps, samples, n_trials, self.z_dim).mean(2)
            color = RT_cn / RT_cn.max()
        else:
            set_cn = set_cn.repeat(n_trials, 1, 1)
            RT_cn_rep = np.repeat(RT_cn, n_trials)
            color = RT_cn_rep / RT_cn_rep.max()
        z_GO = z_cn[step // self.tau]
        
        short_index = int(z_GO.shape[0] // (1 / f))
        set_cn_short = set_cn[:short_index]
        cont_cn_short = cont_cn[:short_index]
        
        n_short, _, features = set_cn_short.shape
        trial_true = torch.from_numpy(set_cn_short).float().to(self.device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1) 
        cont_RTshort = torch.from_numpy(cont_cn_short).float().to(self.device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
        
        start_sim = step // self.tau + 1 - stimulation_steps
        text = "" if stimolate else "_onech"
        
        if compute:
            reg = LinearRegression().fit(z_GO, color)
            direction = reg.coef_
            direction /= np.linalg.norm(direction)
            delta_z_long = torch.from_numpy(direction * l).float().to(self.device)
            print("Direzione di variazione (aumenta):", direction)

            delta_z_long = torch.zeros(stimulation_steps, self.z_dim).to(self.device)
            if multi_direction:
                for t in range(stimulation_steps):
                    reg = LinearRegression().fit(z_cn[start_sim + t], color)
                    direction = reg.coef_
                    direction /= np.linalg.norm(direction)
                    delta_z_long[t] = torch.from_numpy(direction * l).float().to(self.device) 
                    
            trial_clone = trial_true[:(step // self.tau + 1)].clone()
            y_mean, y_logvar = self.dmm(trial_clone, cont_RTshort[:(step // self.tau + 1)])
            y_pred = self.dmm.reparameterization(y_mean, y_logvar)
            if mean:
                y_mean = y_mean.reshape(-1, n_short, n_trials, features).mean(2)
                residual = trial_clone - y_mean.repeat_interleave(n_trials, dim=1)
            else:
                y_pred = y_pred.reshape(-1, n_short, n_trials, features).mean(2)
                residual = trial_clone - y_pred.repeat_interleave(n_trials, dim=1)
                
            trial_modified = trial_true[:(step // self.tau + 1)].clone() 
            trial_modified_min = trial_true[:(step // self.tau + 1)].clone()
     
            dx_fix = torch.full((set_cn_short.shape[0] * n_trials,), l).to(self.device)
            dx_fix_min = torch.full((set_cn_short.shape[0] * n_trials,), 0.001).to(self.device)
            dx_array = np.empty([stimulation_steps, 96])
            stim_effect = torch.zeros([stimulation_steps, set_cn_short.shape[0] * n_trials, 96]).to(self.device)
            stim_effect_min = torch.zeros([stimulation_steps, set_cn_short.shape[0] * n_trials, 96]).to(self.device)
            
            for t in range(stimulation_steps):
                if stimolate:
                    _, dx = compute_dx_for_last_step_ElasticNet(
                        self.dmm, 
                        trial_modified[:(start_sim + t + 1)], 
                        cont_RTshort[:(start_sim + t + 1)], 
                        delta_z_long[t] if multi_direction else delta_z_long, 
                        alpha_l1=alpha_L1, 
                        l1_ratio=l1_ratio, 
                        n_iter=1
                    )
                    dx_array[t] = dx.mean(0)
                    trial_modified[start_sim + t] += torch.from_numpy(dx).float().to(self.device)
                else:
                    trial_modified[start_sim + t, :, 63] += dx_fix
                    trial_modified_min[start_sim + t, :, 63] += dx_fix_min
                if t + 1 == stimulation_steps:
                    continue
                
                trial_cut = trial_modified.clone()
                z_modified, _, _ = self.dmm.inference(trial_cut[:(start_sim + t + 1)], cont_RTshort[:(start_sim + t + 1)])
                z_mean_gen, z_cov_gen = self.dmm.generation_z(z_modified[-1].unsqueeze(0), cont_RTshort[start_sim + t + 1].unsqueeze(0))
                z_gen_last = self.dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                y_mean, y_logvar = self.dmm.generation_x(z_gen_last)
                y_pred = self.dmm.reparameterization(y_mean, y_logvar)
                y_gen = y_mean if mean else y_pred
                trial_modified[start_sim + t + 1] = y_gen[0]
                if add_residual:
                    trial_modified[start_sim + t + 1] += residual[start_sim + t + 1]
                stim_effect[t] = trial_modified[start_sim + t + 1] - trial_true[start_sim + t + 1]
                
                trial_cut_min = trial_modified_min.clone()
                z_modified, _, _ = self.dmm.inference(trial_cut_min[:(start_sim + t + 1)], cont_RTshort[:(start_sim + t + 1)])
                z_mean_gen, z_cov_gen = self.dmm.generation_z(z_modified[-1].unsqueeze(0), cont_RTshort[start_sim + t + 1].unsqueeze(0))
                z_gen_last = self.dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                y_mean, y_logvar = self.dmm.generation_x(z_gen_last)
                y_pred = self.dmm.reparameterization(y_mean, y_logvar)
                y_gen = y_mean if mean else y_pred
                trial_modified_min[start_sim + t + 1] = y_gen[0]
                if add_residual:
                    trial_modified_min[start_sim + t + 1] += residual[start_sim + t + 1]
                stim_effect_min[t] = trial_modified_min[start_sim + t + 1] - trial_true[start_sim + t + 1]

            stim_effect = stim_effect.cpu().detach().numpy()
            stim_effect_min = stim_effect_min.cpu().detach().numpy()
            trial_stim = trial_modified.cpu().detach().numpy()
            np.savez(self.saved_path / f"longer_RT_{l}{text}.npz", trial_stim=trial_stim, stim_effect_min=stim_effect_min, stim_effect=stim_effect, dx_array=dx_array)
        else:
            with np.load(self.saved_path / f"longer_RT_{l}{text}.npz") as loaded_file:
                trial_stim = loaded_file["trial_stim"]
                dx_array = loaded_file["dx_array"]
                if not stimolate:
                    stim_effect = loaded_file["stim_effect"]
                    stim_effect_min = loaded_file["stim_effect_min"]
                    diff_stim = stim_effect - stim_effect_min
                    
            trial_modified = torch.from_numpy(trial_stim).float().to(self.device)
        
        if stimolate:  
            dx_array = channel2grid(dx_array)
            last_stim = dx_array[stimulation_steps].copy()
            last_stim[last_stim < thr] = 0
        else:
            mean_stim_effect = diff_stim.mean(1)
            last_stim = channel2grid(mean_stim_effect)
        
        self.make_RThist(diff_dict, last_stim, trial_modified, trial_true, cont_RTshort, l, 'longer')

    def make_RThist(self, diff_dict, last_stim, trial_modified, trial_true, cont_RTlong, l, text, RT_true=None):
        """
        Plots the true and modified Reaction Time histograms side-by-side with spatial stimulation grid.
        """
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
            
        steps = 256 // self.tau
        teacher = step // self.tau + 1
                                                              
        z_generated, _, _ = self.dmm.inference(trial_true[:teacher], cont_RTlong[:teacher])
        z_modified, _, _ = self.dmm.inference(trial_modified, cont_RTlong[:teacher])

        alone = steps - teacher
        for cycle_step in range(alone):
            z_mean_gen, z_cov_gen = self.dmm.generation_z(z_modified[-1].unsqueeze(0), cont_RTlong[teacher + cycle_step].unsqueeze(0))
            z_gen = self.dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
            z_modified = torch.cat((z_modified, z_gen), dim=0)
            
            z_mean_gen, z_cov_gen = self.dmm.generation_z(z_generated[-1].unsqueeze(0), cont_RTlong[teacher + cycle_step].unsqueeze(0))
            z_gen = self.dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
            z_generated = torch.cat((z_generated, z_gen), dim=0)
            
        RT_output = self.RT_detector(z_modified.permute(1, 0, 2))
        RT_estimate = prob_to_RT(RT_output, self.tau)   

        RT_gen = self.RT_detector(z_generated.permute(1, 0, 2))
        RT_estimate_gen = prob_to_RT(RT_gen, self.tau)  
        
        RT_gen = (RT_estimate * self.tau) * 5
        RT_gen_true = (RT_estimate_gen * self.tau) * 5
        
        num_bins = bins
        min_value = min(RT_gen_true.min(), RT_gen.min())
        max_value = max(RT_gen_true.max(), RT_gen.max())
        bin_edges = np.linspace(min_value, max_value, num_bins + 1)
        
        print(f"True mean RT: {RT_gen_true.mean()}")
        print(f"Generated mean RT: {RT_gen.mean()}")
        
        fig, ax = plt.subplots()
        ax.hist(RT_gen_true, bins=bin_edges, alpha=alpha, density=True, color=color_true, edgecolor='none')
        ax.axvline(RT_gen_true.mean(), color=color_true, linestyle="--")
        ax.hist(RT_gen, bins=bin_edges, alpha=alpha, density=True, color=color_pred, edgecolor='none')
        ax.axvline(RT_gen.mean(), color=color_pred, linestyle="--")
        if RT_true:
            ax.axvline(RT_true, color="k", linestyle="--")
        ax.set_xticks(x_ticks_hist)
        ax.set_xticklabels(x_ticks_hist)
        ax.set_yticks([])
        ax.set_xlabel('Reaction Time ($ms$)')
        ax.set_ylabel('Counts')
        
        vmax = abs(last_stim).max()
        if vmax < 0.1:
            vmax = 0.3
        
        axins = ax.inset_axes(inset_dim) 
        axins.imshow(last_stim, aspect='equal', cmap=cmap_stim, interpolation='nearest', vmin=-vmax, vmax=vmax)
        axins.plot(0, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)
        axins.plot(9, 0, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)
        axins.plot(0, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)
        axins.plot(9, 9, 'x', color=color_edge, markersize=markersize, markeredgewidth=markeredgewidth)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.grid(True, which='major', color='w', alpha=0.2)

        fig_file = self.saved_path / f'{text}_RT_{l}.png'
        plt.savefig(fig_file)
        plt.show()

    def change_trial_cs(self, diff_dict):
        """
        Visualizes the latent trajectory during correct stop vs modified trajectories.
        """
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
        
        f, ax = plt.subplots()
        
        trial = set_cs[SSD_cs == SSD]
        cont_c = cont_cs[SSD_cs == SSD]
        if trial.shape[0] == 0:
            print(f"change the trial with SSD={SSD:d}")
        elif trial.shape[0] != steps:
            print("many trials")
            trial = trial[0]
            cont_c = cont_c[0]
        else:
            trial = trial.squeeze(0)
            cont_c = cont_c.squeeze(0)

        trial = torch.from_numpy(trial).float().to(self.device).unsqueeze(1).repeat(1, n_trials, 1)
        cont_c = torch.from_numpy(cont_c).float().to(self.device).unsqueeze(1).repeat(1, n_trials, 1)
        z, _, _ = self.dmm.inference(trial, cont_c)

        teacher = ((56 + SSD) // self.tau) 
        alone = steps - teacher
        min_grad_mod = (alone * (1 - min_grad)) / steps
        z_teach = z[:teacher]
        for _ in range(alone):
            z_mean_gen, z_cov_gen = self.dmm.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher].unsqueeze(0))
            z_gen = self.dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
            z_teach = torch.cat((z_teach, z_gen), dim=0)

        mu_z = z_teach.mean(1).cpu().detach().numpy()
        mu_x = mu_z[teacher:, axis[0]]
        mu_y = mu_z[teacher:, axis[1]]
     
        true_trial = z.mean(1).cpu().detach().numpy()
        x_true_story = true_trial[:, axis[0]]
        y_true_story = true_trial[:, axis[1]]
        
        x_SSD = true_trial[teacher - 1, axis[0]]
        y_SSD = true_trial[teacher - 1, axis[1]]
        
        add_gradient_line(ax, x_true_story, y_true_story, cmap=cmap, lw=lw, vmin=min_grad, vmax=1.0)
        add_gradient_line(ax, mu_x, mu_y, cmap=cmap_go, lw=lw, vmin=min_grad_mod, vmax=1.0)
     
        n_arrows = int(n_arrows_init * (alone / steps))
        arrow_indices = np.arange(0, len(mu_x), len(mu_x) // n_arrows)
        for k in arrow_indices[:-1]:
            dx = mu_x[k + 1] - mu_x[k]
            dy = mu_y[k + 1] - mu_y[k]
            ax.arrow(mu_x[k], mu_y[k], dx, dy, head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)

        arrow_indices = np.arange(0, len(x_true_story), len(x_true_story) // n_arrows_init)
        for k in arrow_indices[:-1]:
            dx = x_true_story[k + 1] - x_true_story[k]
            dy = y_true_story[k + 1] - y_true_story[k]
            ax.arrow(x_true_story[k], y_true_story[k], dx, dy, head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)

        ax.plot(x_SSD, y_SSD, color=color_stop, marker='x', markeredgewidth=3, markersize=15)
        ax.set_xlim(diff_dict["z_lims"][axis[0], 0], diff_dict["z_lims"][axis[0], 1])
        ax.set_ylim(diff_dict["z_lims"][axis[1], 0], diff_dict["z_lims"][axis[1], 1])
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_xticks(diff_dict["z_ticks"][axis[0]]) 
        ax.set_xticklabels(diff_dict["z_ticks"][axis[0]])
        ax.set_yticks(diff_dict["z_ticks"][axis[1]])
        ax.set_yticklabels(diff_dict["z_ticks"][axis[1]])
        
        fig_file = self.saved_path / 'cs_to_ws.png'
        plt.savefig(fig_file)
        plt.show()

    def plot_cs_ns(self, diff_dict):
        """
        Plots correct stop and wrong stop trials with classification line inside latent space.
        """
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
        axis_pca = diff_dict["axis_pca"]
        alpha_pc = diff_dict["alpha_pc"]
        inset_dim = diff_dict["inset_dim"]
        inset_font = diff_dict["inset_font"]
        x_min, x_max = diff_dict["xlims"]
        x_min_pc, x_max_pc = diff_dict["xlims_pc"]
        
        SSD_ws = data["SSD_ws_ordRT"]
        SSD_cs = data["SSD_cs_ordSSD"]
        set_cs = data["set_cs_ordSSD"]
        set_ws = data["set_ws_ordRT"]
        
        z_cn, z_ws, z_cs = self.infer_latent(data)
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

        ws_SSD_x = trial_ws[(56 + SSD_late) // self.tau, axis[0]]
        ws_SSD_y = trial_ws[(56 + SSD_late) // self.tau, axis[1]]
        cs_SSD_x = trial_cs[(56 + SSD_late) // self.tau, axis[0]]
        cs_SSD_y = trial_cs[(56 + SSD_late) // self.tau, axis[1]]
        
        add_gradient_line(ax, ws_x, ws_y, cmap=cmap_ws, lw=lw, vmin=min_grad, vmax=1.0)
        add_gradient_line(ax, cs_x, cs_y, cmap=cmap_cs, lw=lw, vmin=min_grad, vmax=1.0)

        for arrow_path in [ws_x, cs_x]:
            arrow_y = ws_y if arrow_path is ws_x else cs_y
            arrow_indices = np.arange(0, len(arrow_path), len(arrow_path) // n_arrows)
            for k in arrow_indices[:-1]:
                dx = arrow_path[k + 1] - arrow_path[k]
                dy = arrow_y[k + 1] - arrow_y[k]
                ax.arrow(arrow_path[k], arrow_y[k], dx, dy, head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
            
        ax.plot(ws_SSD_x, ws_SSD_y, color=color_stop_ws, marker='x', markeredgewidth=3, markersize=15)
        ax.plot(cs_SSD_x, cs_SSD_y, color=color_stop_cs, marker='x', markeredgewidth=3, markersize=15)
        ax.set_xlim(diff_dict["z_lims"][axis[0], 0], diff_dict["z_lims"][axis[0], 1])
        ax.set_ylim(diff_dict["z_lims"][axis[1], 0], diff_dict["z_lims"][axis[1], 1])
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")
        ax.set_xticks(diff_dict["z_ticks"][axis[0]]) 
        ax.set_xticklabels(diff_dict["z_ticks"][axis[0]])
        ax.set_yticks(diff_dict["z_ticks"][axis[1]])
        ax.set_yticklabels(diff_dict["z_ticks"][axis[1]])
        
        fig_file = self.saved_path / 'ws_and_cs.png'
        plt.savefig(fig_file)    
        
        # Discriminator line
        z_cn, z_ws, z_cs = self.infer_latent(data, n_trials)
        steps = z_cn.shape[0]
        if n_trials > 1:
            z_cn = z_cn.reshape(steps, n_cn, n_trials, self.z_dim).mean(2)
            z_ws = z_ws.reshape(steps, n_ws, n_trials, self.z_dim).mean(2)
            z_cs = z_cs.reshape(steps, n_cs, n_trials, self.z_dim).mean(2)
        
        z_cs_stop = z_cs[(SSD_cs + 56) // self.tau, np.arange(z_cs.shape[1])]
        z_ws_stop = z_ws[(SSD_ws + 56) // self.tau, np.arange(z_ws.shape[1])]
        
        X_t = np.vstack([z_cs_stop, z_ws_stop])
        y_t = np.array([1] * z_cs.shape[1] + [2] * z_ws.shape[1])
        
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_t, y_t)
        y_pred = lda.predict(X_t)
        acc = (y_pred == y_t).mean()
        
        scores = lda.decision_function(X_t)
        y_bin = (y_t == 2).astype(int)
        auc = roc_auc_score(y_bin, scores)
        
        print(f"accuracy: {acc:.2f}")
        print(f"AUC: {auc:.2f}")
        
        w = lda.coef_[0]
        b = lda.intercept_[0]
        wi, wj = w[axis[0]], w[axis[1]]
        xx = np.linspace(x_min, x_max, 200)
        yy = -(wi * xx + b) / wj
        
        f, ax = plt.subplots()
        add_gradient_line(ax, ws_x, ws_y, cmap=cmap_ws, lw=lw, vmin=min_grad, vmax=1.0, alpha=alpha)
        add_gradient_line(ax, cs_x, cs_y, cmap=cmap_cs, lw=lw, vmin=min_grad, vmax=1.0, alpha=alpha)

        for arrow_path in [ws_x, cs_x]:
            arrow_y = ws_y if arrow_path is ws_x else cs_y
            arrow_indices = np.arange(0, len(arrow_path), len(arrow_path) // n_arrows)
            for k in arrow_indices[:-1]:
                dx = arrow_path[k + 1] - arrow_path[k]
                dy = arrow_y[k + 1] - arrow_y[k]
                ax.arrow(arrow_path[k], arrow_y[k], dx, dy, head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=alpha)

        ax.scatter(z_cs_stop[:, axis[0]], z_cs_stop[:, axis[1]], s=s, c='r', alpha=0.8, label="cs stops")
        ax.scatter(z_ws_stop[:, axis[0]], z_ws_stop[:, axis[1]], s=s, c='g', alpha=0.8, label="ws stops")
        ax.plot(xx, yy, color=color_line, lw=lw, alpha=alpha_line)
        ax.set_xlim(diff_dict["z_lims"][axis[0], 0], diff_dict["z_lims"][axis[0], 1])
        ax.set_ylim(diff_dict["z_lims"][axis[1], 0], diff_dict["z_lims"][axis[1], 1])
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")
        ax.set_xticks(diff_dict["z_ticks"][axis[0]])
        ax.set_xticklabels(diff_dict["z_ticks"][axis[0]])
        ax.set_yticks(diff_dict["z_ticks"][axis[1]])
        ax.set_yticklabels(diff_dict["z_ticks"][axis[1]])
        
        # PCA Inset
        axins = ax.inset_axes(inset_dim)
        with np.load(self.saved_path / "data_split.npz", allow_pickle=True) as loaded_file:
            train_set = loaded_file["train_set"]
                
        n_train, steps, features = train_set.shape
        X_train = train_set.reshape(-1, features)

        pca = PCA(n_components=self.z_dim)
        pca.fit(X_train)
        
        pc_cs = pca.transform(set_cs.reshape(-1, features)).reshape(-1, steps, self.z_dim).transpose(1, 0, 2)
        pc_ws = pca.transform(set_ws.reshape(-1, features)).reshape(-1, steps, self.z_dim).transpose(1, 0, 2)
        
        pc_cs_stop = pc_cs[(SSD_cs + 56) // self.tau, np.arange(pc_cs.shape[1])]
        pc_ws_stop = pc_ws[(SSD_ws + 56) // self.tau, np.arange(pc_ws.shape[1])]
        
        X_pc = np.vstack([pc_cs_stop, pc_ws_stop])
        y_pc = np.array([1] * pc_cs.shape[1] + [2] * pc_ws.shape[1])
        
        lda_pc = LinearDiscriminantAnalysis()
        lda_pc.fit(X_pc, y_pc)
        y_pred_pc = lda_pc.predict(X_pc)
        acc_pc = (y_pred_pc == y_pc).mean()
        
        scores_pc = lda_pc.decision_function(X_pc)
        y_bin_pc = (y_pc == 2).astype(int)
        auc_pc = roc_auc_score(y_bin_pc, scores_pc)
        
        print(f"accuracy: {acc_pc:.2f}")
        print(f"AUC: {auc_pc:.2f}")
        
        w_pc = lda_pc.coef_[0]
        b_pc = lda_pc.intercept_[0]
        wi_pc, wj_pc = w_pc[axis_pca[0]], w_pc[axis_pca[1]]
        xx_pc = np.linspace(x_min_pc, x_max_pc, 200)
        yy_pc = -(wi_pc * xx_pc + b_pc) / wj_pc
        
        trial_cs = pc_cs[:, q_cs]
        trial_ws = pc_ws[:, q_ws]
        ws_x = trial_ws[:, axis_pca[0]]
        ws_y = trial_ws[:, axis_pca[1]]
        cs_x = trial_cs[:, axis_pca[0]]
        cs_y = trial_cs[:, axis_pca[1]]
        
        add_gradient_line(axins, ws_x, ws_y, cmap=cmap_ws, lw=lw_pc, vmin=min_grad, vmax=1.0, alpha=alpha)
        add_gradient_line(axins, cs_x, cs_y, cmap=cmap_cs, lw=lw_pc, vmin=min_grad, vmax=1.0, alpha=alpha)

        for arrow_path in [ws_x, cs_x]:
            arrow_y = ws_y if arrow_path is ws_x else cs_y
            arrow_indices = np.arange(0, len(arrow_path), len(arrow_path) // n_arrows)
            for k in arrow_indices[:-1]:
                dx = arrow_path[k + 1] - arrow_path[k]
                dy = arrow_y[k + 1] - arrow_y[k]
                axins.arrow(arrow_path[k], arrow_y[k], dx, dy, head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=alpha)
                
        axins.scatter(pc_cs_stop[:, axis_pca[0]], pc_cs_stop[:, axis_pca[1]], s=s_pc, c='r', alpha=alpha_pc, label="cs stops")
        axins.scatter(pc_ws_stop[:, axis_pca[0]], pc_ws_stop[:, axis_pca[1]], s=s_pc, c='g', alpha=alpha_pc, label="ws stops")
        axins.plot(xx_pc, yy_pc, color=color_line, lw=lw_pc, alpha=alpha_line)
        axins.set_xlabel(f"$PC_{axis_pca[0]+1}$", fontsize=inset_font)
        axins.set_ylabel(f"$PC_{axis_pca[1]+1}$", fontsize=inset_font)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.tick_params(labelsize=inset_font)
        
        fig_file = self.saved_path / 'ws_vs_cs_stop.png'
        plt.savefig(fig_file)
        return xx, yy

    def change_trial_ws(self, diff_dict):
        """
        Visualizes wrong stop trajectories and target-switching scenarios in latent space.
        """
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
        
        f, ax = plt.subplots()
        
        trial = set_ws[SSD_ws == SSD]
        cont_c = cont_ws[SSD_ws == SSD]
        if trial.shape[0] == 0:
            print(f"change the trial with SSD={SSD:d}")
        elif trial.shape[0] != 1:
            print("many trials")
            trial = trial[0]
            cont_c = cont_c[0]
        else:
            trial = trial.squeeze(0)
            cont_c = cont_c.squeeze(0)

        trial = torch.from_numpy(trial).float().to(self.device).unsqueeze(1).repeat(1, n_trials, 1)
        cont_c = torch.from_numpy(cont_c).float().to(self.device).unsqueeze(1).repeat(1, n_trials, 1)
        
        cont = torch.zeros((1, n_trials, 4)).to(self.device)
        cont[:, :, 3] = 1
        z, _, _ = self.dmm.inference(trial, cont_c)

        teacher = ((56 + SSD) // self.tau) - anticipation // (5 * self.tau)
        alone = steps - teacher
        min_grad_mod = (alone * (1 - min_grad)) / steps
        z_teach = z[:teacher]
        for _ in range(alone):
            z_mean_gen, z_cov_gen = self.dmm.generation_z(z_teach[-1].unsqueeze(0), cont)
            z_gen = self.dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
            z_teach = torch.cat((z_teach, z_gen), dim=0)

        mu_z = z_teach.mean(1).cpu().detach().numpy()
        mu_x = mu_z[teacher:, axis[0]]
        mu_y = mu_z[teacher:, axis[1]]

        true_trial = z.mean(1).cpu().detach().numpy()
        x_true_story = true_trial[:, axis[0]]
        y_true_story = true_trial[:, axis[1]]

        x_SSD = true_trial[(56 + SSD) // self.tau, axis[0]]
        y_SSD = true_trial[(56 + SSD) // self.tau, axis[1]]
        x_stop = true_trial[teacher - 1, axis[0]]
        y_stop = true_trial[teacher - 1, axis[1]]
        
        add_gradient_line(ax, x_true_story, y_true_story, cmap=cmap, lw=lw, vmin=min_grad, vmax=1.0)
        add_gradient_line(ax, mu_x, mu_y, cmap=cmap_go, lw=lw, vmin=min_grad_mod, vmax=1.0)
        
        n_arrows = int(n_arrows_init * (alone / steps))
        arrow_indices = np.arange(0, len(mu_x), len(mu_x) // n_arrows)
        for k in arrow_indices[:-1]:
            dx = mu_x[k + 1] - mu_x[k]
            dy = mu_y[k + 1] - mu_y[k]
            ax.arrow(mu_x[k], mu_y[k], dx, dy, head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
            
        arrow_indices = np.arange(0, len(x_true_story), len(x_true_story) // n_arrows_init)
        for k in arrow_indices[:-1]:
            dx = x_true_story[k + 1] - x_true_story[k]
            dy = y_true_story[k + 1] - y_true_story[k]
            ax.arrow(x_true_story[k], y_true_story[k], dx, dy, head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)

        ax.plot(x_SSD, y_SSD, color=color_stop, marker='x', markeredgewidth=3, markersize=15)
        ax.plot(x_stop, y_stop, color=color_fstop, marker='x', markeredgewidth=3, markersize=15)
        ax.set_xlim(diff_dict["z_lims"][axis[0], 0], diff_dict["z_lims"][axis[0], 1])
        ax.set_ylim(diff_dict["z_lims"][axis[1], 0], diff_dict["z_lims"][axis[1], 1])
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_xticks(diff_dict["z_ticks"][axis[0]]) 
        ax.set_xticklabels(diff_dict["z_ticks"][axis[0]])
        ax.set_yticks(diff_dict["z_ticks"][axis[1]])
        ax.set_yticklabels(diff_dict["z_ticks"][axis[1]])
        
        fig_file = self.saved_path / 'ws_to_cs.png'
        plt.savefig(fig_file)
        plt.show()

    def change_trial_cn(self, diff_dict):
        """
        Applies stop perturbations to correct no-stop trials at various SSDs and plots latent trajectories.
        """
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
        xx, yy = diff_dict["line"]
        
        set_cn = data["set_cn_ordRT"]
        cont_cn = data["cont_cn_ordRT"]
        RT_cn = data["RT_cn_ordRT"]
        steps = set_cn.shape[1]
        
        trial = set_cn[RT_cn == RT]
        cont_c = cont_cn[RT_cn == RT]
        if trial.shape[0] == 0:
            print(f"change the trial with RT={RT:d}")
        elif trial.shape[0] != steps:
            trial = trial[0]
            cont_c = cont_c[0]
        else:
            trial = trial.squeeze(0)
            cont_c = cont_c.squeeze(0)

        cont = torch.zeros((1, n_trials, 4)).to(self.device)
        cont[:, :, 3] = 1

        trial = torch.from_numpy(trial).float().to(self.device).unsqueeze(1).repeat(1, n_trials, 1)
        cont_c = torch.from_numpy(cont_c).float().to(self.device).unsqueeze(1).repeat(1, n_trials, 1)
        z, _, _ = self.dmm.inference(trial, cont_c)
        true_trial = z.mean(1)
                
        move_logit = self.move_detector(true_trial.unsqueeze(0))
        move_output = binary_output(move_logit)
        move_rec = move_output.squeeze(0).astype(int)

        if move_rec:
            RT_output = self.RT_detector(true_trial.unsqueeze(0))
            RT_rec = prob_to_RT(RT_output, self.tau)
            RT_rec = RT_rec.squeeze(0)

        true_trial = true_trial.cpu().detach().numpy()
        x_true_story = true_trial[:, axis[0]]
        y_true_story = true_trial[:, axis[1]]
        
        plot_label = ["ns_to_cs", "ns_to_ws"]
            
        for i, stop in enumerate(stop_array):           
            teacher = 56 // self.tau + stop // (5 * self.tau)
            alone = steps - teacher
            z_teach = z[:teacher]
            for _ in range(alone):
                z_mean_gen, z_cov_gen = self.dmm.generation_z(z_teach[-1].unsqueeze(0), cont)
                z_gen = self.dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                z_teach = torch.cat((z_teach, z_gen), dim=0)

            mu_z = z_teach.mean(1).cpu().detach().numpy()
            mu_x = mu_z[teacher:, axis[0]]
            mu_y = mu_z[teacher:, axis[1]]
            
            f, ax = plt.subplots()
            
            n_arrows = int(n_arrows_init * (alone / steps))
            arrow_indices = np.arange(0, len(mu_x), len(mu_x) // n_arrows)
            
            add_gradient_line(ax, x_true_story, y_true_story, cmap=cmap, lw=lw, vmin=min_grad, vmax=1.0)
            add_gradient_line(ax, mu_x, mu_y, cmap=cmap_stop, lw=lw, vmin=min_grad_mod, vmax=1.0)
            
            for k in arrow_indices[:-1]:
                dx = mu_x[k + 1] - mu_x[k]
                dy = mu_y[k + 1] - mu_y[k]
                ax.arrow(mu_x[k], mu_y[k], dx, dy, head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)    
                
            arrow_indices = np.arange(0, len(x_true_story), len(x_true_story) // n_arrows_init)
            for k in arrow_indices[:-1]:
                dx = x_true_story[k + 1] - x_true_story[k]
                dy = y_true_story[k + 1] - y_true_story[k]
                ax.arrow(x_true_story[k], y_true_story[k], dx, dy, head_width=0.1, head_length=0.1, fc='brown', ec='brown', alpha=0.6)
                
            x_SSD = true_trial[teacher - 1, axis[0]]
            y_SSD = true_trial[teacher - 1, axis[1]]
            ax.plot(x_SSD, y_SSD, color=color_stop, marker='x', markeredgewidth=3, markersize=15)
            ax.plot(xx, yy, color=color_line, lw=lw, alpha=alpha_line)
            ax.set_xlim(diff_dict["z_lims"][axis[0], 0], diff_dict["z_lims"][axis[0], 1])
            ax.set_ylim(diff_dict["z_lims"][axis[1], 0], diff_dict["z_lims"][axis[1], 1])
            ax.set_xlabel("$z_1$")
            ax.set_ylabel("$z_2$")
            ax.set_xticks(diff_dict["z_ticks"][axis[0]]) 
            ax.set_xticklabels(diff_dict["z_ticks"][axis[0]])
            ax.set_yticks(diff_dict["z_ticks"][axis[1]])
            ax.set_yticklabels(diff_dict["z_ticks"][axis[1]])
            
            fig_file = self.saved_path / f'{plot_label[i]}.png'
            plt.savefig(fig_file)
            plt.show()

    def RTgen_vs_SSD(self, diff_dict):
        """
        Plots generated RT failure versus SSD and verifies stopping failures vs SSD.
        """
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
        RT_time = RT_cn * 5
        n_batch = n_trials // chunk_size if (n_trials % chunk_size) == 0 else (n_trials // chunk_size) + 1
        text = "_sim" if simulate_go else ""  
        
        test_data = torch.from_numpy(set_cn).float().to(self.device).permute(1, 0, 2).repeat_interleave(chunk_size, dim=1)
        cont_data = torch.from_numpy(cont_cn).float().to(self.device).permute(1, 0, 2).repeat_interleave(chunk_size, dim=1)
        cont = torch.zeros((1, n * chunk_size, 4)).to(self.device)
        cont[:, :, 3] = 1

        if compute:
            with torch.inference_mode():
                RT_pred_list = []
                RT_time_list = []
                pmove_list = []
                for SSD in SSD_list:
                    teacher = (SSD // 5 + 56) // self.tau
                    alone = steps - teacher - 1
                    
                    move_pred_t_list = []
                    mu_go_list = []
                    mu_z_list = []
                    for _ in range(n_batch):
                        test_set = test_data.clone()
                        cont_c = cont_data.clone()

                        if mean_z:
                            _, z, _ = self.dmm.inference(test_set, cont_c)
                        else:
                            z, _, _ = self.dmm.inference(test_set, cont_c)

                        z_tmp = z[:teacher + 1]
                        mu_z = z_tmp.clone()
                        mu_go = z_tmp.clone()
                        for step in range(alone):
                            z_mean_gen, z_cov_gen = self.dmm.generation_z(mu_go[-1].unsqueeze(0), cont_c[teacher + step].unsqueeze(0))
                            z_gen = self.dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                            mu_go = torch.cat((mu_go, z_gen), dim=0)
                            
                        mu_go_list.append(mu_go.reshape(mu_go.shape[0], -1, chunk_size, self.z_dim))
                        
                        for step in range(alone):
                            z_mean_gen, z_cov_gen = self.dmm.generation_z(mu_z[-1].unsqueeze(0), cont)
                            z_gen = self.dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                            mu_z = torch.cat((mu_z, z_gen), dim=0)
                        mu_z = mu_z.permute(1, 0, 2)
                            
                        move_logit = self.move_detector(mu_z)
                        move_output = binary_output(move_logit)
                        move_pred = move_output.reshape(-1, chunk_size)
                    
                        move_pred_t_list.append(move_pred)
                        mu_z_list.append(mu_z.reshape(-1, chunk_size, mu_z.shape[1], self.z_dim))

                    move_pred_t = np.concatenate(move_pred_t_list, axis=1)
                    mu_z_array = torch.cat(mu_z_list, dim=1)
                    mu_go_array = torch.cat(mu_go_list, dim=2)
                    
                    mu_mean_go = mu_go_array.mean(2)
                    RT_output = self.RT_detector(mu_mean_go.permute(1, 0, 2))
                    RT_go = prob_to_RT(RT_output, self.tau)
                    RT_go = RT_go * self.tau * 5
                    
                    mask_ns = move_pred_t > 0.5
                    hist = mask_ns.sum(axis=1)
                    valid_rows = hist >= min_ns
                
                    mu_z_masked = mu_z_array[valid_rows]
                    mask_valid = mask_ns[valid_rows]
                    RT_cn_valid = RT_go[valid_rows] if simulate_go else RT_time[valid_rows]
                    
                    print(mu_z_masked.shape)
                    mask_valid_t = torch.from_numpy(mask_valid).float().to(self.device)
                    mask_valid_t = mask_valid_t.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, mu_z_masked.shape[2], mu_z_masked.shape[3])
                    num = (mu_z_masked * mask_valid_t).sum(dim=1)
                    den = mask_valid_t.sum(dim=1)
                    mu_z_mean = num / den
                
                    RT_output = self.RT_detector(mu_z_mean)
                    RT_rec = prob_to_RT(RT_output, self.tau)
                    RT_pred_mean = RT_rec * self.tau * 5
        
                    mask_notnan = ~np.isnan(RT_pred_mean)
                    RT_cn_valid = RT_cn_valid[mask_notnan]
                    RT_pred_mean = RT_pred_mean[mask_notnan]
                    
                    RT_pred_list.append(RT_pred_mean)
                    RT_time_list.append(RT_cn_valid)
                    pmove_list.append(mask_ns.mean(1))
                    print(f"SSD={SSD} retain {mask_valid_t.shape[0]} trials out of {n}")
                    
            RT_pred_dict = {i: arr for i, arr in enumerate(RT_pred_list)}
            RT_time_dict = {i: arr for i, arr in enumerate(RT_time_list)}
            pmove_dict = {i: arr for i, arr in enumerate(pmove_list)}

            np.savez(self.saved_path / f"RTgen_vs_SSD{text}_{n_trials}.npz", RT_pred_dict=RT_pred_dict, RT_time_dict=RT_time_dict, pmove_dict=pmove_dict)
        else:
            with np.load(self.saved_path / f"RTgen_vs_SSD{text}_{n_trials}.npz", allow_pickle=True) as loaded_file:
                RT_pred_dict = loaded_file["RT_pred_dict"].item()
                RT_time_dict = loaded_file["RT_time_dict"].item()
                pmove_dict = loaded_file["pmove_dict"].item()

        RT_diff_mean = []
        RT_diff_std = []
        for i in range(len(SSD_list)):
            diff_RT = RT_pred_dict[i] - RT_time_dict[i]
            RT_diff_mean.append(diff_RT.mean())
            RT_diff_std.append(diff_RT.std() / math.sqrt(len(diff_RT)))
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(SSD_list, RT_diff_mean, color="black")
        ax.errorbar(SSD_list, RT_diff_mean, yerr=RT_diff_std, fmt='o', color="black", ecolor='black', 
                    elinewidth=elinewidth, linestyle='none', capsize=capsize, ms=ms)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_ylim(ylims)
        ax.set_xlabel(r"$SSD\ (\mathrm{ms})$")
        ax.set_ylabel(r"$RT_{\mathrm{stop\ failure}} - RT_{\mathrm{no-stop}} \, (\mathrm{ms})$")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)
        
        if self.with_Cornelio:
            with np.load(self.saved_path.parent / self.with_Cornelio / f"RTgen_vs_SSD{text}_{n_trials}.npz", allow_pickle=True) as loaded_file:
                RT_pred_dict = loaded_file["RT_pred_dict"].item()
                RT_time_dict = loaded_file["RT_time_dict"].item()
                pmove_dict = loaded_file["pmove_dict"].item()
                    
            RT_diff_mean = []
            RT_diff_std = []
            for i in range(len(SSD_list)):
                diff_RT = RT_pred_dict[i] - RT_time_dict[i]
                RT_diff_mean.append(diff_RT.mean())
                RT_diff_std.append(diff_RT.std() / math.sqrt(len(diff_RT)))
            
            axins = ax.inset_axes(inset_dim)
            axins.plot(SSD_list, RT_diff_mean, color="black")
            axins.errorbar(SSD_list, RT_diff_mean, yerr=RT_diff_std, fmt='o', color="black", ecolor='black', 
                           elinewidth=elinewidth / 2, linestyle='none', capsize=capsize // 2, ms=ms // 2)
            axins.axhline(0, color="red", linestyle="--")
            axins.set_ylim(ylims_inset)
            axins.set_xticks(x_ticks)
            axins.set_xticklabels(x_ticks)
            axins.set_yticks(y_ticks_inset)
            axins.set_yticklabels(y_ticks_inset)
            axins.tick_params(labelsize=inset_font)
        
        fig_file = self.saved_path / f'RTgen_vs_SSD{text}.png'
        plt.savefig(fig_file)
        plt.show()

    def early_SSRT(self, diff_dict):
        """
        Plots early SSRT (Stop Signal Reaction Time) metrics against human data comparison.
        """
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
        
        n, steps, _ = set_cn.shape
        text = "_sim" if simuldistr else ""
        mean = "_mean" if mean_z else ""
        n_batch = n * n_trials // chunk_size if (n * n_trials % chunk_size) == 0 else (n * n_trials // chunk_size) + 1
        
        test_data = torch.from_numpy(set_cn).float().to(self.device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
        cont_data = torch.from_numpy(cont_cn).float().to(self.device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
        cont = torch.zeros((1, chunk_size, 4)).to(self.device)
        cont[:, :, 3] = 1
        
        if compute:
            RT_pred_list = []
            move_perc_list = []
            for i, SSD_short in enumerate(SSD_short_list):
                move_pred_list = []
                for batch in range(n_batch):
                    start = batch * chunk_size
                    end = min(start + chunk_size, n * n_trials) 
                    interval = end - start

                    test_set = test_data[:, start:end]
                    cont_c = cont_data[:, start:end]

                    if mean_z:
                        _, z_cn, _ = self.dmm.inference(test_set, cont_c)
                    else:
                        z_cn, _, _ = self.dmm.inference(test_set, cont_c)
                    
                    if simuldistr and i == 0:
                        teacher = 56 // self.tau
                        alone = steps - teacher
                        z_teach = z_cn[:teacher]
                        for step in range(alone):
                            z_mean_gen, z_cov_gen = self.dmm.generation_z(z_teach[-1].unsqueeze(0), cont_c[56 // self.tau + step].unsqueeze(0))
                            z_gen = self.dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                            z_teach = torch.cat((z_teach, z_gen), dim=0)

                        RT_output = self.RT_detector(z_teach.permute(1, 0, 2))
                        RT_pred = prob_to_RT(RT_output, self.tau)
                        RT_pred_list.extend(RT_pred) 
                        
                        del z_teach, z_gen
                        torch.cuda.empty_cache()

                    stop = (56 + SSD_short) // self.tau
                    alone_go = SSD_short // self.tau
                    z_stop = z_cn[:56 // self.tau]
                    for step in range(alone_go):
                        z_mean_gen, z_cov_gen = self.dmm.generation_z(z_stop[-1].unsqueeze(0), cont_c[56 // self.tau + step].unsqueeze(0))
                        z_gen = self.dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                        z_stop = torch.cat((z_stop, z_gen), dim=0)

                    alone_stop = steps - stop                          
                    for step in range(alone_stop):
                        z_mean_gen, z_cov_gen = self.dmm.generation_z(z_stop[-1].unsqueeze(0), cont[:, :interval])
                        z_gen = self.dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                        z_stop = torch.cat((z_stop, z_gen), dim=0) 

                    move_logit = self.move_detector(z_stop.permute(1, 0, 2))
                    move_output = binary_output(move_logit).astype(int)
                    move_pred = move_output.astype(int)
                    move_pred_list.extend(move_pred)
                    
                    del move_pred, z_stop
                    torch.cuda.empty_cache()

                move_pred_tot = np.array(move_pred_list)
                move_perc_list.append(move_pred_tot.mean())
                
            move_perc_arr = np.array(move_perc_list)
            RT_pred_arr = np.array(RT_pred_list)
            np.savez(self.saved_path / f"SSRT_short_{n_trials}{text}{mean}.npz", move_perc_arr=move_perc_arr, RT_pred_arr=RT_pred_arr)
        else:
            with np.load(self.saved_path / f"SSRT_short_{n_trials}{text}{mean}.npz", allow_pickle=True) as loaded_file:
                move_perc_arr = loaded_file["move_perc_arr"]
                RT_pred_arr = loaded_file["RT_pred_arr"]

        perc_mov = np.array([0.1, 0.3, 0.6, 0.8])
        RT_val = np.quantile(RT_pred_arr, move_perc_arr)
        SSRT_val = (RT_val * self.tau - SSD_short_list) * 5
        
        SSRT_val = SSRT_val[slice_start:slice_end]
        move_perc_arr = move_perc_arr[slice_start:slice_end]
        SSD_short_list_sliced = SSD_short_list[slice_start:slice_end]

        idxs = np.abs(move_perc_arr[:, None] - perc_mov[None, :]).argmin(axis=0)
        SSRT_Piero = SSRT_val[idxs]
        for i, idx in enumerate(idxs):
            print(f"SSD corresponding to inhibitory probability of {(1 - perc_mov[i]) * 100:.0f}% for Monkey P: {SSD_short_list_sliced[idx] * 5}ms")
        
        SSRTs = np.array([
            [307, 182, 165, 154], [275, 192, 155, 148], [200, 190, 176, 171],
            [286, 243, 203, 197], [179, 151, 128, 123], [270, 188, 171, 179]
        ])
        xlabel = ["SSD 1", "SSD 2", "SSD 3", "SSD 4"]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(xlabel, SSRT_Piero, color=color_Piero, zorder=2)
        ax.scatter(xlabel, SSRT_Piero, marker="d", color='black', zorder=3)
        
        if self.with_Cornelio:
            with np.load(self.saved_path.parent / self.with_Cornelio / f"SSRT_short_{n_trials}{text}{mean}.npz", allow_pickle=True) as loaded_file:
                move_perc_arr_c = loaded_file["move_perc_arr"]
                RT_pred_arr_c = loaded_file["RT_pred_arr"]

            RT_val_c = np.quantile(RT_pred_arr_c, move_perc_arr_c)
            SSRT_val_c = (RT_val_c * self.tau - SSD_short_list) * 5
            
            move_perc_arr_c = move_perc_arr_c[slice_start:slice_end]
            SSRT_val_c = SSRT_val_c[slice_start:slice_end]
            
            idxs_c = np.abs(move_perc_arr_c[:, None] - perc_mov[None, :]).argmin(axis=0)
            SSRT_Cornelio = SSRT_val_c[idxs_c]
            for i, idx in enumerate(idxs_c):
                print(f"SSD corresponding to inhibitory probability of {(1 - perc_mov[i]) * 100:.0f}% for Monkey C: {SSD_short_list_sliced[idx] * 5}ms")       
        
            ax.plot(xlabel, SSRT_Cornelio, color=color_Cornelio, zorder=2)
            ax.scatter(xlabel, SSRT_Cornelio, marker="d", color='black', zorder=3)

        for i in range(6):
            ax.plot(xlabel, SSRTs[i], color=color_humans, alpha=alpha, zorder=1)

        ax.set_ylabel("SSRT")
        ax.set_ylim(ylims)
        ax.set_yticks(y_ticks)
        
        fig_file = self.saved_path / f'SSRT_vs_SSD{text}{mean}.png'
        plt.savefig(fig_file)
        plt.show()

    def SSRT_plot(self, diff_dict): 
        """
        Calculates and plots Stop Signal Reaction Time (SSRT) vs RT distribution.
        """
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
        RT_cn_step = (RT_cn + 56) // self.tau
        RT_time = RT_cn * 5
        
        t_start, t_end = SSD_interval
        t_interval = t_end - t_start
        n_batch = n_trials // chunk_size if (n_trials % chunk_size) == 0 else (n_trials // chunk_size) + 1
        
        test_data = torch.from_numpy(set_cn).float().to(self.device).permute(1, 0, 2).repeat_interleave(chunk_size, dim=1)
        cont_data = torch.from_numpy(cont_cn).float().to(self.device).permute(1, 0, 2).repeat_interleave(chunk_size, dim=1)
        cont = torch.zeros((1, n * chunk_size, 4)).to(self.device)
        cont[:, :, 3] = 1
        
        if compute:
            move_perc_list = []
            with torch.inference_mode():
                for idx_t, t_val in enumerate(range(t_start, t_end)):
                    if idx_t % 5 == 0:
                        print(f"time {idx_t}/{t_interval}")
                    move_pred_t_list = []
                    teacher = np.repeat(RT_cn_step, chunk_size) - t_val
                    alone = steps - teacher[0]

                    for _ in range(n_batch):
                        test_set = test_data.clone()
                        cont_c = cont_data.clone()

                        if mean_z:
                            _, z, _ = self.dmm.inference(test_set, cont_c)
                        else:
                            z, _, _ = self.dmm.inference(test_set, cont_c)

                        z_gen = z[teacher, torch.arange(len(teacher), device=self.device)].unsqueeze(0)
                        z_tmp = z_gen.clone()

                        z_teach_list = []
                        for _ in range(alone):
                            z_mean_gen, z_cov_gen = self.dmm.generation_z(z_tmp, cont)
                            z_tmp = self.dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                            z_teach_list.append(z_tmp)
                        z_teach = torch.cat(z_teach_list, dim=0)

                        mu_z_list = []
                        for i in range(z_teach.shape[1]):
                            diff = teacher[i] - teacher[0] + 1
                            z_final_i = torch.cat((z[:(teacher[i] + 1), i], z_teach[:-diff, i]), dim=0)
                            mu_z_list.append(z_final_i)
                        mu_z = torch.stack(mu_z_list, dim=1)

                        move_logit = self.move_detector(mu_z.permute(1, 0, 2))
                        if logit_move:
                            move_output = torch.sigmoid(move_logit).cpu().detach().numpy()
                        else:
                            move_output = binary_output(move_logit)

                        move_pred = move_output.reshape(-1, chunk_size)
                        move_pred_t_list.append(move_pred)

                    move_pred_t = np.concatenate(move_pred_t_list, axis=1)
                    move_perc_list.append(move_pred_t.mean(1))

                move_perc = np.stack(move_perc_list, axis=1)

                nan_mask = np.isnan(move_perc).any(axis=1)
                move_perc = move_perc[~nan_mask]
                RT_time = RT_time[~nan_mask]
                RT_cn_step = RT_cn_step[~nan_mask]
                set_cn = set_cn[~nan_mask]
                cont_cn = cont_cn[~nan_mask]
                
                mask = move_perc <= 0.5
                SSRT_critic = mask.argmax(axis=1)
                critic_t = RT_cn_step - SSRT_critic 

                test_cn = torch.from_numpy(set_cn).float().to(self.device).permute(1, 0, 2)
                test_c = torch.from_numpy(cont_cn).float().to(self.device).permute(1, 0, 2)

                z_test, _, _ = self.dmm.inference(test_cn, test_c)
                z_critic = z_test[critic_t, torch.arange(set_cn.shape[0])].cpu().detach().numpy()

                cont_n_t = torch.zeros((1, n_trials, 4)).to(self.device)
                cont_n_t[:, :, 3] = 1
                RT_stop = np.zeros(len(SSRT_critic))
                RT_go = np.zeros(len(SSRT_critic))
                z_mean_stop = np.zeros((len(SSRT_critic), steps, self.z_dim))
                z_mean_go = np.zeros((len(SSRT_critic), steps, self.z_dim))
                frac_realizations = np.zeros(len(SSRT_critic))
                for i in range(len(SSRT_critic)):
                    if i % 30 == 0:
                        print(f"{i}/{len(SSRT_critic)}")

                    teacher = RT_cn_step[i] - SSRT_critic[i] + 1
                    test_set = torch.from_numpy(set_cn[i]).float().to(self.device).unsqueeze(1).repeat_interleave(n_trials, dim=1)
                    cont_c = torch.from_numpy(cont_cn[i]).float().to(self.device).unsqueeze(1).repeat_interleave(n_trials, dim=1)

                    if mean_z:
                        _, z, _ = self.dmm.inference(test_set, cont_c)
                    else:
                        z, _, _ = self.dmm.inference(test_set, cont_c)
                    z_gen = z[:teacher]
                    cont_go = cont_c[teacher].unsqueeze(0)

                    z_tmp_stop = z_gen.clone()
                    z_tmp_go = z_gen.clone()

                    alone = steps - teacher
                    for _ in range(alone):
                        z_mean_gen, z_cov_gen = self.dmm.generation_z(z_tmp_stop[-1].unsqueeze(0), cont_n_t)
                        z_rep_stop = self.dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                        z_tmp_stop = torch.cat((z_tmp_stop, z_rep_stop), dim=0)

                        z_mean_gen, z_cov_gen = self.dmm.generation_z(z_tmp_go[-1].unsqueeze(0), cont_go)
                        z_rep_go = self.dmm.reparameterization_cov(z_mean_gen, z_cov_gen)
                        z_tmp_go = torch.cat((z_tmp_go, z_rep_go), dim=0)

                    z_tmp_stop = z_tmp_stop.permute(1, 0, 2)
                    z_tmp_go = z_tmp_go.permute(1, 0, 2)

                    move_logit = self.move_detector(z_tmp_stop)
                    if logit_move:
                        move_output = torch.sigmoid(move_logit).cpu().detach().numpy()
                    else:
                        move_output = binary_output(move_logit)

                    mask_move = move_output > move_frac
                    if not mask_move.any():
                        continue
                    else:
                        z_tmp_stop_masked = z_tmp_stop[mask_move]
                        z_tmp_go_masked = z_tmp_go[mask_move]
                        
                        RT_prob_stop = self.RT_detector(z_tmp_stop_masked)
                        RT_estimate_stop = prob_to_RT(RT_prob_stop, self.tau)  
                        RT_gen_stop = (RT_estimate_stop * self.tau) * 5

                        RT_prob_go = self.RT_detector(z_tmp_go_masked)
                        RT_estimate_go = prob_to_RT(RT_prob_go, self.tau)  
                        RT_gen_go = (RT_estimate_go * self.tau) * 5
                        
                        frac_realizations[i] = mask_move.sum() / n_trials 
                        RT_stop[i] = RT_gen_stop.mean()
                        RT_go[i] = RT_gen_go.mean()
                        z_mean_stop[i] = z_tmp_stop_masked.cpu().detach().numpy().mean(0)
                        z_mean_go[i] = z_tmp_go_masked.cpu().detach().numpy().mean(0)
            np.savez(self.saved_path / f"SSRT_plot_{n_trials}.npz", move_perc=move_perc, RT_stop=RT_stop, RT_go=RT_go, RT_time=RT_time,
                     SSRT_critic=SSRT_critic, z_mean_stop=z_mean_stop, z_mean_go=z_mean_go, z_critic=z_critic, frac_realizations=frac_realizations)
        else:
            with np.load(self.saved_path / f"SSRT_plot_{n_trials}.npz", allow_pickle=True) as loaded_file:
                move_perc = loaded_file["move_perc"]
                RT_stop = loaded_file["RT_stop"]
                RT_go = loaded_file["RT_go"]
                RT_time = loaded_file["RT_time"]
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
        
        mask_valid = SSRT_critic != 0
        print(f"Of the {len(mask_valid)} trials of Monkey P kept, {len(mask_valid)-mask_valid.sum()} are removed as they result in SSRT=0")
        
        if no_zero:
            move_perc = move_perc[mask_valid]
            RT_time_masked = RT_time_masked[mask_valid]
            SSRT_critic = SSRT_critic[mask_valid]
            RT_stop = RT_stop[mask_valid]
            RT_go = RT_go[mask_valid]
            z_mean_stop = z_mean_stop[mask_valid]
            z_mean_go = z_mean_go[mask_valid]
            z_critic = z_critic[mask_valid]
            frac_realizations = frac_realizations[mask_valid]
        
        mean_move_perc = move_perc.mean(0)
        mean_SSRT_all = np.argmax(mean_move_perc <= 0.5) * self.tau * 5
        
        SSRT_std_list = []
        SSRT_mean_list = []
        mean_SSRT_list = []
        RT_list = []
        n_group = len(RT_time_masked) // RT_groups if (len(RT_time_masked) % RT_groups) == 0 else (len(RT_time_masked) // RT_groups) + 1
        SSRT_trials = np.argmax(move_perc <= 0.5, axis=1) * self.tau * 5
        print(f"L'SSRT medio è: {SSRT_trials.mean()}")
        for k in range(RT_groups):
            start = k * n_group
            end = min(start + n_group, len(RT_time_masked) + 1)
            SSRT_group = SSRT_trials[start:end]
            SSRT_mean = np.nanmean(SSRT_group)
            SSRT_std = SSRT_group.std() / math.sqrt(end - start)
            RT_group = RT_time_masked[start:end]
            mean_RT_group = RT_group.mean()
            move_group = move_perc[start:end]
            mean_move_group = move_group.mean(0)
            mean_SSRT = np.argmax(mean_move_group <= 0.5) * self.tau * 5
            SSRT_std_list.append(SSRT_std)
            SSRT_mean_list.append(SSRT_mean)
            mean_SSRT_list.append(mean_SSRT)
            RT_list.append(mean_RT_group)
            
        SSRT_std_array = np.array(SSRT_std_list)
        SSRT_mean_array = np.array(SSRT_mean_list)
        mean_SSRT_array = np.array(mean_SSRT_list)
        RT_array = np.array(RT_list)
        
        reg_mean = LinearRegression().fit(RT_array[:, None], SSRT_mean_array)
        pred_SSRT_mean = reg_mean.predict(RT_array[:, None])
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.errorbar(RT_array, SSRT_mean_array, yerr=SSRT_std_array, fmt='o', color=color_point, ecolor='black', 
                    elinewidth=elinewidth, linestyle='none', capsize=capsize, ms=ms)
        ax.plot(RT_array, pred_SSRT_mean, "--", color=color_line, 
                label=f"linear fit: m={reg_mean.coef_.item():.2f}, q={reg_mean.intercept_.item():.2f}")
        
        min_val_x = ax.get_xlim()[0]
        min_val_y = ax.get_ylim()[0]
        max_val_x = ax.get_xlim()[1]
        max_val_y = ax.get_ylim()[1]
        
        dx = (max_val_x - min_val_x) // 3
        dy = (max_val_y - min_val_y) // 3

        ax.set_xlabel('mean RT ($ms$)')
        ax.set_ylabel('SSRT')
        ax.set_xticks([int(((min_val_x + dx) // 100 + 1) * 100), int(((min_val_x + 2 * dx) // 100 + 1) * 100)])
        ax.set_xticklabels([int(((min_val_x + dx) // 100 + 1) * 100), int(((min_val_x + 2 * dx) // 100 + 1) * 100)])
        ax.set_yticks([int(((min_val_y + dy) // 10) * 10), int(((min_val_y + 2 * dy) // 10 + 1) * 10)])
        ax.set_yticklabels([int(((min_val_y + dy) // 10) * 10), int(((min_val_y + 2 * dy) // 10 + 1) * 10)])

        if self.with_Cornelio:
            with np.load(self.saved_path.parent / self.with_Cornelio / f"SSRT_plot_{n_trials}.npz", allow_pickle=True) as loaded_file:
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
            n_group_c = len(RT_time_masked_c) // RT_groups if (len(RT_time_masked_c) % RT_groups) == 0 else (len(RT_time_masked_c) // RT_groups) + 1
            SSRT_trials_c = np.argmax(move_perc_c <= 0.5, axis=1) * self.tau * 5
            for k in range(RT_groups):
                start = k * n_group_c
                end = min(start + n_group, len(RT_time_masked_c) + 1)
                SSRT_group_c = SSRT_trials_c[start:end]
                SSRT_mean_c = np.nanmean(SSRT_group_c)
                SSRT_std_c = SSRT_group_c.std() / math.sqrt(end - start)
                RT_group_c = RT_time_masked_c[start:end]
                mean_RT_group_c = RT_group_c.mean()
                move_group_c = move_perc_c[start:end]
                mean_move_group_c = move_group_c.mean(0)
                mean_SSRT_c = np.argmax(mean_move_group_c <= 0.5) * self.tau * 5
                SSRT_std_list_c.append(SSRT_std_c)
                SSRT_mean_list_c.append(SSRT_mean_c)
                mean_SSRT_list_c.append(mean_SSRT_c)
                RT_list_c.append(mean_RT_group_c)
                
            SSRT_std_array_c = np.array(SSRT_std_list_c)
            SSRT_mean_array_c = np.array(SSRT_mean_list_c)
            RT_array_c = np.array(RT_list_c)

            reg_mean_c = LinearRegression().fit(RT_array_c[:, None], SSRT_mean_array_c)
            pred_SSRT_mean_c = reg_mean_c.predict(RT_array_c[:, None])
        
            axins = ax.inset_axes(inset_dim)
            axins.errorbar(RT_array_c, SSRT_mean_array_c, yerr=SSRT_std_array_c, fmt='o', color=color_point, ecolor='black', 
                           elinewidth=elinewidth / 2, linestyle='none', capsize=capsize / 2, ms=ms / 2)
            axins.plot(RT_array_c, pred_SSRT_mean_c, "--", color=color_line)

            min_val_y_c = axins.get_ylim()[0]
            max_val_y_c = axins.get_ylim()[1]
            dy_c = (max_val_y_c - min_val_y_c) // 3

            axins.set_xticks([500, 700])
            axins.set_xticklabels([500, 700])
            axins.set_yticks([int(((min_val_y_c + dy_c) // 10) * 10), int(((min_val_y_c + 2 * dy_c) // 10 + 1) * 10)])
            axins.set_yticklabels([int(((min_val_y_c + dy_c) // 10) * 10), int(((min_val_y_c + 2 * dy_c) // 10 + 1) * 10)])
            axins.tick_params(labelsize=inset_font)
        
        fig_file = self.saved_path / 'SSRT_vs_RT.png'
        plt.savefig(fig_file)

        corr = np.corrcoef(RT_array, mean_SSRT_array)[0, 1]
        print(f"the correlation between RT and SSRT is {corr:.2f}")
        print(f"SSRT medio: {mean_SSRT_all}ms")
        plt.show()

    def consistent_sim(self, diff_dict):
        """
        Simulates model trajectories consistently starting from the GO cue and compares predictions.
        """
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
        teacher = sim_start // (5 * self.tau)
        alone = steps - teacher
        
        MUA_true = test_set.mean(2)
        
        if compute:
            z_cn, _, _ = self.infer_latent(data)
            z_cn_t = torch.from_numpy(z_cn).float().to(self.device)

            cont_c = torch.from_numpy(test_cont).float().to(self.device).permute(1, 0, 2).repeat_interleave(n_trials, dim=1)
            z_teach = z_cn_t[:teacher]
       
            for step in range(alone):
                z_mean_gen, z_cov_gen = self.dmm.generation_z(z_teach[-1].unsqueeze(0), cont_c[teacher + step].unsqueeze(0))
                z_gen = self.dmm.reparameterization_cov(z_mean_gen, eps * z_cov_gen)
                z_teach = torch.cat((z_teach, z_gen), dim=0)

            y_mean, y_logvar = self.dmm.generation_x(z_teach)
            y_pred = self.dmm.reparameterization(y_mean, y_logvar)
            z_teach = z_teach.permute(1, 0, 2)
            RT_output = self.RT_detector(z_teach)
            RT_estimate = prob_to_RT(RT_output, self.tau)    

            y_pred = y_pred.permute(1, 0, 2).cpu().detach().numpy()
            MUA_pred = y_pred.mean(2)
            RT_est_sort = np.argsort(RT_estimate)
            MUA_pred = MUA_pred[RT_est_sort]
            
            if not mean_z:
                test_RT = test_RT.repeat(n_trials)
            RT_true = test_RT * 5
            RT_pred = RT_estimate * self.tau * 5
            
            np.savez(self.saved_path / f"RT_distribution_{n_trials}.npz", MUA_pred=MUA_pred, RT_true=RT_true, RT_pred=RT_pred)
        else:
            with np.load(self.saved_path / f"RT_distribution_{n_trials}.npz", allow_pickle=True) as loaded_file:
                MUA_pred = loaded_file["MUA_pred"]
                RT_true = loaded_file["RT_true"]
                RT_pred = loaded_file["RT_pred"]

        vmin = -3
        vmax = 3

        x_positions = np.arange(200 // self.tau) 
        x_labels = x_positions * (self.tau * 5) 

        x_ticks = len(x_positions)
        n_xticks, _ = n_ticks

        # FIG 4A
        fig, ax = plt.subplots()
        im1 = ax.imshow(MUA_true[:, 56 // self.tau:], cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_xticks(x_positions[::x_ticks // n_xticks]) 
        ax.set_xticklabels(x_labels[::x_ticks // n_xticks])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xlabel('Time from GO signal ($ms$)')
        ax.set_ylabel('Trial #')
        cbar = plt.colorbar(im1, ax=ax)
        cbar.set_ticks([vmin, 0, vmax])
        fig_file = self.saved_path / 'MUA_true.png'
        plt.savefig(fig_file)
        plt.show()
        
        # FIG 4B
        fig, ax = plt.subplots()
        ax.imshow(MUA_pred[:, 56 // self.tau:], cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_xticks(x_positions[::x_ticks // n_xticks]) 
        ax.set_xticklabels(x_labels[::x_ticks // n_xticks])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xlabel('Time from GO signal ($ms$)')
        ax.set_ylabel('Trial #')
        cbar = plt.colorbar(im1, ax=ax)
        cbar.set_ticks([vmin, 0, vmax])
        fig_file = self.saved_path / 'MUA_gen.png'
        plt.savefig(fig_file)
        plt.show()
        
        statistic, p_value = ks_2samp(RT_true, RT_pred)
        print("\n--- Test di Kolmogorov-Smirnov (K-S) ---")
        print(f"Statistica del test: {statistic:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        emd = wasserstein_distance(RT_true, RT_pred)
        print("\n--- Earth Mover's Distance (EMD) / Wasserstein-1 ---")
        print(f"Distanza: {emd:.4f}")
        
        num_bins = bins
        min_value = RT_true.min() - 50
        max_value = RT_true.max() + 50
        bin_edges = np.linspace(min_value, max_value, num_bins + 1)
        
        fig, ax = plt.subplots()
        ax.hist(RT_true, bins=bin_edges, density=True, alpha=alpha, color=color_true, edgecolor='none')
        ax.hist(RT_pred, bins=bin_edges, density=True, alpha=alpha, color=color_pred, edgecolor='none')
        ax.set_xticks(x_ticks_hist)
        ax.set_xticklabels(x_ticks_hist)
        ax.set_yticks([])
        ax.set_xlabel('Reaction Time ($ms$)')
        ax.set_ylabel('Counts')
        
        if self.with_Cornelio:
            with np.load(self.saved_path.parent / self.with_Cornelio / f"RT_distribution_{n_trials}.npz", allow_pickle=True) as loaded_file:
                RT_true_c = loaded_file["RT_true"]
                RT_pred_c = loaded_file["RT_pred"]
        
            statistic_c, p_value_c = ks_2samp(RT_true_c, RT_pred_c)
            print("\n--- Test di Kolmogorov-Smirnov (K-S) Cornelio ---")
            print(f"Statistica del test: {statistic_c:.4f}")
            print(f"P-value: {p_value_c:.4f}")
            
            emd_c = wasserstein_distance(RT_true_c, RT_pred_c)
            print("\n--- Earth Mover's Distance (EMD) / Wasserstein-1 Cornelio ---")
            print(f"Distanza: {emd_c:.4f}")
            
            min_value_c = RT_true_c.min() - 50
            max_value_c = RT_true_c.max() + 50
            bin_edges_c = np.linspace(min_value_c, max_value_c, num_bins + 1)
            
            axins = ax.inset_axes(inset_dim)
            axins.hist(RT_true_c, bins=bin_edges_c, density=True, alpha=alpha, color=color_true, edgecolor='none')
            axins.hist(RT_pred_c, bins=bin_edges_c, density=True, alpha=alpha, color=color_pred, edgecolor='none')
            
            axins.set_xticks(x_ticks_hist)
            axins.set_xticklabels(x_ticks_hist)
            axins.set_yticks([])
            axins.set_xlabel('Reaction Time ($ms$)', fontsize=inset_font)
            axins.set_ylabel('Counts', fontsize=inset_font)
            axins.tick_params(labelsize=inset_font)

        fig_file = self.saved_path / 'RT_distribution.png'
        plt.savefig(fig_file)
        plt.show()


# =====================================================================
# Backward-Compatible Wrapper Functions
# =====================================================================

def inference_with_trials(dmm, data_set, cont_set, n_trials, device, chunk_size=10):
    """Legacy wrapper for DMMVisualizer.inference_with_trials."""
    viz = DMMVisualizer(dmm=dmm, device=device, z_dim=dmm.z_dim)
    return viz.inference_with_trials(data_set, cont_set, n_trials, chunk_size)

def RT_pred_performance(comm_dict, diff_dict):
    """Legacy wrapper for DMMVisualizer.RT_pred_performance."""
    DMMVisualizer.from_dict(comm_dict).RT_pred_performance(diff_dict)

def cs_ws_SSD_hist(diff_dict):
    """Legacy wrapper for DMMVisualizer.cs_ws_SSD_hist."""
    DMMVisualizer().cs_ws_SSD_hist(diff_dict)

def cn_ws_RT_hist(diff_dict):
    """Legacy wrapper for DMMVisualizer.cn_ws_RT_hist."""
    DMMVisualizer().cn_ws_RT_hist(diff_dict)

def infer_latent(dmm, data, device, n_trials=1):
    """Legacy wrapper for DMMVisualizer.infer_latent."""
    viz = DMMVisualizer(dmm=dmm, device=device)
    return viz.infer_latent(data, n_trials)

def single_RTcorr(comm_dict, diff_dict, sim_start):
    """Legacy wrapper for DMMVisualizer.single_RTcorr."""
    return DMMVisualizer.from_dict(comm_dict).single_RTcorr(diff_dict, sim_start)

def correlation_vs_gentime(comm_dict, diff_dict):
    """Legacy wrapper for DMMVisualizer.correlation_vs_gentime."""
    return DMMVisualizer.from_dict(comm_dict).correlation_vs_gentime(diff_dict)

def DMM_EV(comm_dict, diff_dict):
    """Legacy wrapper for DMMVisualizer.DMM_EV."""
    DMMVisualizer.from_dict(comm_dict).DMM_EV(diff_dict)

def DMM_vs_dim(comm_dict, diff_dict):
    """Legacy wrapper for DMMVisualizer.DMM_vs_dim."""
    DMMVisualizer.from_dict(comm_dict).DMM_vs_dim(diff_dict)

def PCA_vs_DMM(comm_dict, diff_dict):
    """Legacy wrapper for DMMVisualizer.PCA_vs_DMM."""
    DMMVisualizer.from_dict(comm_dict).PCA_vs_DMM(diff_dict)

def random_latent_cn_traj(comm_dict, data, n_trials=1):
    """Legacy wrapper for DMMVisualizer.random_latent_cn_traj."""
    return DMMVisualizer.from_dict(comm_dict).random_latent_cn_traj(data, n_trials)

def latent_traj_example(comm_dict, diff_dict):
    """Legacy wrapper for DMMVisualizer.latent_traj_example."""
    DMMVisualizer.from_dict(comm_dict).latent_traj_example(diff_dict)

def plot_cycle_shift_short(comm_dict, diff_dict):
    """Legacy wrapper for DMMVisualizer.plot_cycle_shift_short."""
    DMMVisualizer.from_dict(comm_dict).plot_cycle_shift_short(diff_dict)

def plot_cycle_shift_long(comm_dict, diff_dict):
    """Legacy wrapper for DMMVisualizer.plot_cycle_shift_long."""
    DMMVisualizer.from_dict(comm_dict).plot_cycle_shift_long(diff_dict)

def change_trial_cs(comm_dict, diff_dict):
    """Legacy wrapper for DMMVisualizer.change_trial_cs."""
    DMMVisualizer.from_dict(comm_dict).change_trial_cs(diff_dict)

def plot_cs_ns(comm_dict, diff_dict):
    """Legacy wrapper for DMMVisualizer.plot_cs_ns."""
    return DMMVisualizer.from_dict(comm_dict).plot_cs_ns(diff_dict)

def change_trial_ws(comm_dict, diff_dict):
    """Legacy wrapper for DMMVisualizer.change_trial_ws."""
    DMMVisualizer.from_dict(comm_dict).change_trial_ws(diff_dict)

def change_trial_cn(comm_dict, diff_dict):
    """Legacy wrapper for DMMVisualizer.change_trial_cn."""
    DMMVisualizer.from_dict(comm_dict).change_trial_cn(diff_dict)

def RTgen_vs_SSD(comm_dict, diff_dict):
    """Legacy wrapper for DMMVisualizer.RTgen_vs_SSD."""
    DMMVisualizer.from_dict(comm_dict).RTgen_vs_SSD(diff_dict)

def early_SSRT(comm_dict, diff_dict):
    """Legacy wrapper for DMMVisualizer.early_SSRT."""
    DMMVisualizer.from_dict(comm_dict).early_SSRT(diff_dict)

def SSRT_plot(comm_dict, diff_dict):
    """Legacy wrapper for DMMVisualizer.SSRT_plot."""
    DMMVisualizer.from_dict(comm_dict).SSRT_plot(diff_dict)

def consistent_sim(comm_dict, diff_dict):
    """Legacy wrapper for DMMVisualizer.consistent_sim."""
    DMMVisualizer.from_dict(comm_dict).consistent_sim(diff_dict)