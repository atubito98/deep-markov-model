#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt
"""


import os
import random
import shutil
import socket
import datetime
import pickle
import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from .utils import correlation_vs_gentime, latent_traj_ctype, MUA_pred_inf_plot, MUA_pred_inf, myconf, get_logger, loss_KLD, loss_KLD_cov, loss_rec
from .dataset import process_data, data_split, MUA_dataset, BalancedHierarchicalSampler, BalancedHomogeneousSampler, one_hot_cont
from .model import build_VAE, build_DKF_cx_causal_chot, build_DKF_session_all, build_DKF_session_e_dec

class LearningAlgorithm_session_all():

    """
    Basical class for model building, including:
    - read common parameters for different models
    - define data loader
    - define loss function as a class member
    """

    def __init__(self, params):
        # Load config parser
        self.params = params
        self.config_file = self.params['cfg']
        if not os.path.isfile(self.config_file):
            raise ValueError('Invalid config file path')    
        self.cfg = myconf()
        self.cfg.read(self.config_file)
        self.model_name = self.cfg.get('Network', 'name')

        # Get host name and date
        self.hostname = socket.gethostname()
        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%M")
        
        # Load model parameters
        self.use_cuda = self.cfg.getboolean('Training', 'use_cuda')
        self.device = self.params['device'] if torch.cuda.is_available() and self.use_cuda else 'cpu'


    def build_model(self):
        if self.model_name == 'VAE':
            self.model = build_VAE(cfg=self.cfg, device=self.device)
        elif self.model_name == 'DKF_session_all':
            self.model = build_DKF_session_all(cfg=self.cfg, device=self.device)
            self.best_model = build_DKF_session_all(cfg=self.cfg, device=self.device) 
        elif self.model_name == 'DKF_session_e_dec':
            self.model = build_DKF_session_e_dec(cfg=self.cfg, device=self.device)
            self.best_model = build_DKF_session_e_dec(cfg=self.cfg, device=self.device) 
        

    def init_optimizer(self, train_dataloader, epochs):
        optimization  = self.cfg.get('Training', 'optimization')
        lr = self.cfg.getfloat('Training', 'lr')
        decay_lr = self.cfg.getfloat('Training', 'decay_lr')
        
        # --- MODIFICA 1: Scaling del Learning Rate per gli Adapters ---
        # Gli adapter vedono molti meno dati del core condiviso, quindi devono imparare più in fretta.
        
#         adapter_params = []
#         core_params = []
        
#         for name, param in self.model.named_parameters():
# #             if not param.requires_grad:
# #                 continue  # SKIP FROZEN PARAMETERS (Input Adapters)
            
#             # Now separate Adapters from Shared Core
#             if 'adapters' in name:
#                 adapter_params.append(param)
#             else:
#                 core_params.append(param)
        
#         # Assegna un LR 10 volte più alto agli adapters
#         param_groups = [
#             {'params': core_params, 'lr': lr},
#             {'params': adapter_params, 'lr': lr * 1.0} 
#         ]
        
        milestones = [] if self.cfg.get('Training', 'milestones') == '' else [int(i) for i in self.cfg.get('Training', 'milestones').split(',')]
        if optimization == 'adam': # could be extend to other optimizers
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
#             optimizer = optim.Adam(param_groups)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
#             optimizer = torch.optim.AdamW(param_groups)
#         lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=decay_lr)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            epochs=epochs, 
            steps_per_epoch=len(train_dataloader),
            #total_steps=5000,      # target number of steps S
            pct_start=0.3,         # 30% warmup, then decay
            anneal_strategy='cos'
        )
        return optimizer, lr_scheduler


    def get_basic_info(self):
        basic_info = []
        basic_info.append('HOSTNAME: ' + self.hostname)
        basic_info.append('Time: ' + self.date)
        basic_info.append('Device for training: ' + self.device)
        if self.device != 'cpu':
            basic_info.append('Cuda verion: {}'.format(torch.version.cuda))
        basic_info.append('Model name: {}'.format(self.model_name))
        basic_info.append('Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters()) / 1000000.0))
        
        return basic_info
    
    def align_adapters_with_pca(self, train_set, train_labels, train_session, bottleneck_dim=64):
        """
        Implementa l'inizializzazione PCA-Regression stile LFADS.
        1. Calcola i PSTH (medie per condizione) per ogni sessione.
        2. Sceglie una sessione di riferimento ed esegue PCA sui suoi PSTH -> Target Latent Space.
        3. Per ogni altra sessione, allinea i suoi PSTH al Target tramite Regressione Lineare.
        4. Inizializza i pesi degli adapter con le matrici di regressione.
        """
        print("Initializing Adapters using PCA-Regression Alignment...")
        
        unique_sessions = np.unique(train_session)
        psths = {} # Dizionario per salvare i PSTH concatenati di ogni sessione
        
        # 1. Calcolo Condition-Averaged Firing Rates (PSTH)
        # Condizioni: 0 (Correct Stop), 1 (Correct No-Stop), 2 (Wrong Stop)
        conditions = [0, 1, 2] 
        
        for sess in unique_sessions:
            idx_s = np.where(train_session == sess)[0]
            sess_psths = []
            
            # Seleziona i dati di questa sessione
            X_sess = train_set[idx_s] # [Trials, Time, Channels]
            y_sess = train_labels[idx_s]
            
            for cond in conditions:
                idx_c = np.where(y_sess == cond)[0]
                if len(idx_c) > 0:
                    # Media sui trial (Condition Average) -> [Time, Channels]
                    avg_cond = np.mean(X_sess[idx_c], axis=0)
                    sess_psths.append(avg_cond)
                else:
                    # Se manca una condizione, aggiungi zeri o salta (qui aggiungo zeri per mantenere shape)
                    # print(f"Warning: Session {sess} missing condition {cond}")
                    _, time_bins, n_ch = X_sess.shape
                    sess_psths.append(np.zeros((time_bins, n_ch)))
            
            # Concatena le condizioni nel tempo -> [3 * Time, Channels]
            # Questa matrice rappresenta la "Dinamica Tipica" di quella sessione
            psths[sess] = np.concatenate(sess_psths, axis=0)

        # 2. PCA sulla Sessione di Riferimento
        # Scegliamo la prima sessione come "Reference" (Target Space)
        ref_session = unique_sessions[0]
        X_ref = psths[ref_session] # [T_total, Ch_ref]
        
        # Eseguiamo PCA usando SVD di PyTorch
        # Centriamo i dati
        X_ref_tensor = torch.tensor(X_ref, dtype=torch.float32)
        mean_ref = torch.mean(X_ref_tensor, dim=0)
        X_ref_centered = X_ref_tensor - mean_ref
        
        # U, S, V = torch.svd(X_ref_centered)
        # PC Scores (Proiezioni nello spazio latente) = X @ V
        # V ha shape [Ch, Ch]. Prendiamo le prime 'bottleneck_dim' colonne.
        try:
            U, S, V = torch.svd(X_ref_centered)
            # V[:, :bottleneck_dim] sono i componenti principali
            # Y_target è la traiettoria latente che vogliamo che TUTTE le sessioni imparino a produrre
            Y_target = X_ref_centered @ V[:, :bottleneck_dim] # [T_total, 64]
        except:
            print("SVD failed on reference session, skipping alignment.")
            return

        # 3. Regressione per ogni Sessione
        for sess in unique_sessions:
            X_s = torch.tensor(psths[sess], dtype=torch.float32)
            
            # Vogliamo trovare W_in tale che: X_s @ W_in ~= Y_target
            # Ridge Regression: W = (X^T X + lambda I)^-1 X^T Y
            ridge_lambda = 0.01
            
            XTX = X_s.T @ X_s
            reg = ridge_lambda * torch.eye(XTX.shape[0])
            
            # W_in: [Channels, 64]
            # Usiamo lstsq o calcolo manuale dell'inversa per stabilità
            try:
                W_in = torch.inverse(XTX + reg) @ (X_s.T @ Y_target)
            except RuntimeError:
                # Fallback pseudoinverse se singolare
                W_in = torch.pinverse(X_s) @ Y_target

            # 4. Assegnazione ai Pesi del Modello
            # Input Adapter
            sid = str(sess)
            if sid in self.model.input_adapters:
                with torch.no_grad():
                    # Aggiorna W_in (Transpose perché Linear usa W x Input^T)
                    self.model.input_adapters[sid].weight.copy_(W_in.T)
                    self.model.input_adapters[sid].bias.zero_() # Bias a zero per pulizia
                    
                    # Output Adapter Mean (Inizializzato come Pseudo-Inversa di W_in)
                    # W_out ~ W_in_pinv. 
                    # W_in shape [Ch, 64]. W_out shape [64, Ch].
                    W_out = torch.pinverse(W_in) 
                    
                    self.model.output_adapters_mean[sid].weight.copy_(W_out.T) # Transpose per Linear layer
                    self.model.output_adapters_mean[sid].bias.zero_()
                    
                    # Output Adapter LogVar
                    # Inizializziamo a valori piccoli random (varianza bassa all'inizio)
                    nn.init.uniform_(self.model.output_adapters_logvar[sid].weight, -0.01, 0.01)
                    nn.init.constant_(self.model.output_adapters_logvar[sid].bias, 0.0) # log(var) = -5 -> piccola varianza

        print("Adapter Alignment Complete.")


    def train(self):
        ############
        ### Init ###
        ############

        # Build model
        self.build_model()

        # Set module.training = True
        torch.autograd.set_detect_anomaly(True)

        # Create directory for results
        if not self.params['reload']:
            saved_root = self.cfg.get('User', 'saved_root')
            c_dim = self.cfg.getint('Network','c_dim')
            z_dim = self.cfg.getint('Network','z_dim')
            x_dim = self.cfg.getint('Network','x_dim')
            tau = self.cfg.getint('DataFrame','tau')
            GO_flag = self.cfg.getboolean('DataFrame','GO_flag')
            shift = self.cfg.getint('DataFrame', 'shift')
            tag = self.cfg.get('Network', 'tag')
            filename = "{}_{}".format(self.date, tag)
            save_dir = os.path.join(saved_root, filename)
            if not(os.path.isdir(save_dir)):
                os.makedirs(save_dir)
        else:
            c_dim = self.cfg.getint('Network','c_dim')
            z_dim = self.cfg.getint('Network','z_dim')
            x_dim = self.cfg.getint('Network','x_dim')
            tau = self.cfg.getint('DataFrame','tau')
            GO_flag = self.cfg.getboolean('DataFrame','GO_flag')
            shift = self.cfg.getint('DataFrame', 'shift')
            tag = self.cfg.get('Network', 'tag')
            save_dir = self.params['model_dir']
            

        # Save the model configuration
        save_cfg = os.path.join(save_dir, 'config.ini')
        shutil.copy(self.config_file, save_cfg)

        # Create logger
        log_file = os.path.join(save_dir, 'log.txt')
        logger_type = self.cfg.getint('User', 'logger_type')
        logger = get_logger(log_file, logger_type)

        # Print basical infomation
        for log in self.get_basic_info():
            logger.info(log)
        logger.info('In this experiment, result will be saved in: ' + save_dir)

        # Print model infomation (optional)
        if self.cfg.getboolean('User', 'print_model'):
            for log in self.model.get_info():
                logger.info(log)

        # Create data loader
        train_set, vali_set, test_set, train_RT, vali_RT, test_RT, train_SSD, vali_SSD, test_SSD, train_direction, \
        vali_direction, test_direction, session_train, session_vali, session_test, seed, data_path_array = data_split.split(self.cfg)
        
        all_name = []
        for j in range(len(data_path_array)):
            name = os.path.splitext(os.path.basename(data_path_array[j]))[0]
            self.model.add_session(name)
            all_name.append(name)
        print(all_name)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Totale parametri: {total_params}")
        
        train_session = np.array([all_name[int(i)] for i in session_train])
        vali_session = np.array([all_name[int(i)] for i in session_vali])
        test_session = np.array([all_name[int(i)] for i in session_test])
        
        gate_cont_train = one_hot_cont(train_SSD, train_direction, tau)
        gate_cont_vali = one_hot_cont(vali_SSD, vali_direction, tau)
        gate_cont_test = one_hot_cont(test_SSD, test_direction, tau)
            
        print(gate_cont_train.shape)
        samples, steps, _ = gate_cont_train.shape

        train_num = train_set.shape[0] 
        val_num = vali_set.shape[0] 
        test_num = test_set.shape[0] 
            
        test_correct_stop = test_set[test_RT == 0]
        test_correct_nostop = test_set[test_SSD == 0]
        test_wrong_stop = test_set[(test_RT != 0) & (test_SSD != 0)]
            
        vali_correct_stop = vali_set[vali_RT == 0]
        vali_correct_nostop = vali_set[vali_SSD == 0]
        vali_wrong_stop = vali_set[(vali_RT != 0) & (vali_SSD != 0)]
            
        train_correct_stop = train_set[train_RT == 0]
        train_correct_nostop = train_set[train_SSD == 0]
        train_wrong_stop = train_set[(train_RT != 0) & (train_SSD != 0)]
            
        train_labels = np.zeros(train_set.shape[0])#.to(self.device)
        train_labels[train_RT == 0] = 0
        train_labels[train_SSD == 0] = 1
        train_labels[(train_RT != 0) & (train_SSD != 0)] = 2
        train_labels = train_labels.astype(int)
            
        vali_labels = np.zeros(vali_set.shape[0])#.to(self.device)
        vali_labels[vali_RT == 0] = 0
        vali_labels[vali_SSD == 0] = 1
        vali_labels[(vali_RT != 0) & (vali_SSD != 0)] = 2
        vali_labels = vali_labels.astype(int)
            
        shuffle = self.cfg.getboolean('DataFrame', 'shuffle')
        batch_size = self.cfg.getint('DataFrame', 'batch_size')
        num_workers = self.cfg.getint('DataFrame', 'num_workers')
        
#         def collate_fn_safe(batch):
#             # Separa i componenti
#             batch_data = [item[0] for item in batch]
#             batch_gate = [item[1] for item in batch]
#             batch_sess = [str(item[2]) for item in batch] # Forza str() python

#             # Stack tensori (assumendo siano numpy array o tensori)
#             # Se sono numpy:
#             data_tensor = torch.as_tensor(np.stack(batch_data))
#             gate_tensor = torch.as_tensor(np.stack(batch_gate))

#             return data_tensor, gate_tensor, batch_sess
            
        #train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        #val_dataloader = DataLoader(dataset=vali_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        #test_dataloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            
        train_dataset = MUA_dataset(train_set, gate_cont_train, train_labels, train_session)
        vali_dataset = MUA_dataset(vali_set, gate_cont_vali, vali_labels, vali_session)

#         train_sampler = BalancedHierarchicalSampler(train_dataset)
        batch_sampler = BalancedHomogeneousSampler(train_dataset, batch_size=batch_size, drop_last=True)
        train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=num_workers)
        #train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
        #val_dataloader = DataLoader(dataset=vali_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        self.align_adapters_with_pca(train_set, train_labels, train_session, bottleneck_dim=self.model.dense_x_gx)
        
#         # 2. FREEZE INPUT ADAPTERS (The Fix)
#         # We force the model to use the PCA-aligned input space as the ground truth.
#         print("Freezing Input Adapters to anchor latent space...")
#         for param in self.model.input_adapters.parameters():
#             param.requires_grad = False
            
        np.savez(save_dir + "/data_split.npz", all_name = all_name, train_set = train_set, vali_set = vali_set, test_set = test_set, \
                    train_RT = train_RT, vali_RT = vali_RT, test_RT = test_RT, train_SSD = train_SSD, vali_SSD = vali_SSD, \
                    test_SSD = test_SSD, train_direction = train_direction, vali_direction = vali_direction, \
                    test_direction = test_direction, train_session = train_session, vali_session = vali_session, 
                    session_test = session_test, data_path_array = data_path_array)
        
        # Load data
        #with np.load(save_dir + "/data_split.npz") as loaded_file:
            #data = {key: loaded_file[key] for key in loaded_file.files}
            #locals().update(data)  # Unpack all variables to local namespace

        # Process test and train data
        test_data = process_data(test_set, test_RT, test_SSD, test_direction, test_session, tau)
        train_data = process_data(train_set, train_RT, train_SSD, train_direction, train_session, tau)
        
        
        logger.info('Split operated with random seed: {}'.format(seed))  
        logger.info('Sessions used: {}'.format(all_name))  
        logger.info('Training samples: {}, of which {} correct stop, {} correct no stop, {} wrong stop'.format(train_num, \
                        train_correct_stop.shape[0], train_correct_nostop.shape[0], train_wrong_stop.shape[0]))
        logger.info('Validation samples: {}, of which {} correct stop, {} correct no stop, {} wrong stop'.format(val_num, \
                        vali_correct_stop.shape[0], vali_correct_nostop.shape[0], vali_wrong_stop.shape[0]))
        logger.info('Test samples: {}, of which {} correct stop, {} correct no stop, {} wrong stop'.format(test_num, \
                        test_correct_stop.shape[0], test_correct_nostop.shape[0], test_wrong_stop.shape[0]))
        
        ######################
        ### Batch Training ###
        ######################

        # Load training parameters
        epochs = self.cfg.getint('Training', 'epochs')
        early_stop_patience = self.cfg.getint('Training', 'early_stop_patience')
        save_frequency = self.cfg.getint('Training', 'save_frequency')
        beta = self.cfg.getfloat('Training', 'beta')
        kl_slope = self.cfg.getfloat('Training', 'kl_slope')
        kl_warm_steps = self.cfg.getint('Training', 'kl_warm_steps')
        kl_warm = 0
        last_epoch = 0
    
        optimizer, lr_scheduler = self.init_optimizer(train_dataloader, epochs)

        # Create python list for loss
        if not self.params['reload']:
            train_loss = np.zeros((epochs,))
            val_loss = np.zeros((epochs,))
            train_recon = np.zeros((epochs,))
            train_kl = np.zeros((epochs,))
            val_recon = np.zeros((epochs,))
            val_kl = np.zeros((epochs,))
            best_val_loss = np.inf
            cpt_patience = 0
            cur_best_epoch = epochs
            best_state_dict = self.model.state_dict()
            best_optim_dict = optimizer.state_dict()
            start_epoch = -1
        else:
            cp_file = os.path.join(save_dir, '{}_checkpoint.pt'.format(self.model_name))
            checkpoint = torch.load(cp_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            start_epoch = checkpoint['epoch']
            loss_log = checkpoint['loss_log']
            kl_warm = checkpoint['kl_warm']
            C_warm = checkpoint['C_warm']
            logger.info('Resuming trainning: epoch: {}'.format(start_epoch))
            train_loss = np.pad(loss_log['train_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            val_loss = np.pad(loss_log['val_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            train_recon = np.pad(loss_log['train_recon'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            train_kl = np.pad(loss_log['train_kl'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            val_recon = np.pad(loss_log['val_recon'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            val_kl = np.pad(loss_log['val_kl'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            best_val_loss = checkpoint['best_val_loss']
            cpt_patience = 0
            cur_best_epoch = start_epoch
            best_state_dict = self.model.state_dict()
            best_optim_dict = optimizer.state_dict()
            
            
        # Train with mini-batch SGD
        for epoch in range(start_epoch+2, epochs+1):
            
            start_time = datetime.datetime.now()

            # KL warm-up
            if epoch % kl_warm_steps == 0 and kl_warm < 1:
                kl_warm = (epoch // kl_warm_steps) * kl_slope
                logger.info('KL warm-up, anneal coeff: {}'.format(kl_warm))
              
            # Batch training
            self.model.train()
            
            num_train = 0
            # Reset counters for epoch statistics
            epoch_loss_tot = 0
            epoch_loss_recon = 0
            epoch_loss_kl = 0

            #for _, batch_data in enumerate(train_dataloader):
            for _, (batch_data, batch_gate_cont, batch_sessions) in enumerate(train_dataloader):
                
                batch_data = batch_data.float().to(self.device)
                batch_gate_cont = batch_gate_cont.float().to(self.device)
                
                bs, seq_len, _ = batch_data.shape
                num_train += bs
                # Check Session ID (Sampler guarantees they are all same)
                sess_idx = batch_sessions[0].item() 
                sess_id = train_dataset.idx_to_sess[sess_idx]
                
                # Permute
                batch_data = batch_data.permute(1, 0, 2)
                batch_gate_cont = batch_gate_cont.permute(1, 0, 2)
                
                recon_mean_data, recon_logvar_data = self.model(batch_data, batch_gate_cont, sess_id)
                loss_recon = self.model.loss_rec(recon_mean_data, recon_logvar_data, batch_data)

                loss_recon = loss_recon.sum() / (seq_len * bs )

                #two = torch.full((self.model.z_mean.shape[0], self.model.z_mean.shape[1]), z_dim).to(self.device)
                loss_kl = loss_KLD_cov(self.model.z_mean, self.model.z_cov, self.model.z_mean_p, self.model.z_cov_p, z_dim)
                loss_kl = loss_kl / (seq_len * bs )

                loss_tot = loss_recon + kl_warm * beta * loss_kl

                loss_tot.backward() # Accumula i gradienti
                    
                # Logging (accumuliamo i valori puri per le statistiche)
                epoch_loss_tot += loss_tot.item() * bs 
                epoch_loss_recon += loss_recon.item() * bs
                epoch_loss_kl += loss_kl.item() * bs

                optimizer.step()

            # Aggiorna gli array per i plot
            train_loss[epoch-1] = epoch_loss_tot / num_train
            train_recon[epoch-1] = epoch_loss_recon / num_train
            train_kl[epoch-1] = epoch_loss_kl / num_train
            
            # Validation
            self.model.eval()
            epoch_val_loss_tot = 0
            epoch_val_loss_recon = 0
            epoch_val_loss_kl = 0
            num_val = 0
            unique_val_sessions = np.unique(vali_session)
            with torch.no_grad():
                for sess_id in unique_val_sessions:
                    
                    # 1. Select all data for this session
                    indices = np.where(vali_session == sess_id)[0]
                    if len(indices) == 0: continue
                        
                    # Extract data
                    val_data_s = torch.tensor(vali_set[indices]).float().to(self.device)
                    val_gate_s = torch.tensor(gate_cont_vali[indices]).float().to(self.device)
                    
                    # Permute for model [Seq, Batch, Dim]
                    val_data_perm = val_data_s.permute(1, 0, 2)
                    val_gate_perm = val_gate_s.permute(1, 0, 2)
                    
                    bs_s, seq_len, _ = val_data_s.shape
                    num_val += bs_s

                    recon_mean_data, recon_logvar_data = self.model(val_data_perm, val_gate_perm, str(sess_id))
                    loss_recon = self.model.loss_rec(recon_mean_data, recon_logvar_data, val_data_perm)

                    loss_recon = loss_recon.sum() / (seq_len * bs_s )

                    #two = torch.full((self.model.z_mean.shape[0], self.model.z_mean.shape[1]), z_dim).to(self.device)
                    loss_kl = loss_KLD_cov(self.model.z_mean, self.model.z_cov, self.model.z_mean_p, self.model.z_cov_p, z_dim)
                    loss_kl = loss_kl / (seq_len * bs_s )

                    loss_tot = loss_recon + kl_warm * beta * loss_kl

                    # Logging (accumuliamo i valori puri per le statistiche)
                    epoch_val_loss_tot += loss_tot.item() * bs_s 
                    epoch_val_loss_recon += loss_recon.item() * bs_s
                    epoch_val_loss_kl += loss_kl.item() * bs_s

            # Normalizzazione Validation
            val_loss[epoch-1] = epoch_val_loss_tot / num_val
            val_recon[epoch-1] = epoch_val_loss_recon / num_val
            val_kl[epoch-1] = epoch_val_loss_kl / num_val

            current_lr = lr_scheduler.get_last_lr()[0]
                
            # Early stop patiance
            if val_loss[epoch-1] < best_val_loss or kl_warm < 1 or epoch==(kl_warm_steps*5): #C_warm < 1:  #sostituire con kl se non usi C
                best_val_loss = val_loss[epoch-1]                  # ho cambiato da or kl_warm <1 a and kl_warm==1
                cpt_patience = 0
                best_state_dict = self.model.state_dict()
                best_optim_dict = optimizer.state_dict()
                cur_best_epoch = epoch
            else:
                cpt_patience += 1
            

            # Save model parameters regularly
            if epoch % save_frequency == 0:
                loss_log = {'train_loss': train_loss[:cur_best_epoch],
                            'val_loss': val_loss[:cur_best_epoch],
                            'train_recon': train_recon[:cur_best_epoch],
                            'train_kl': train_kl[:cur_best_epoch],
                            'val_recon': val_recon[:cur_best_epoch], 
                            'val_kl': val_kl[:cur_best_epoch],
                            'kl_warm': kl_warm 
                            }
                save_file = os.path.join(save_dir, self.model_name + '_checkpoint.pt' + f'{epoch}')

                # QUESTO VA NEL FILE CHECKPOINT.PT
                torch.save({'epoch': cur_best_epoch,
                            'best_val_loss': best_val_loss,
                            'cpt_patience': cpt_patience,
                            'model_state_dict': best_state_dict,
                            'optim_state_dict': best_optim_dict,
                            'kl_warm': kl_warm,
                            'loss_log': loss_log
                        }, save_file)
                logger.info('Epoch: {} ===> checkpoint stored with current best epoch: {}'.format(epoch, cur_best_epoch))

                
                ################################################################################        


            # Training time
            end_time = datetime.datetime.now()
            interval = (end_time - start_time).seconds / 60
            logger.info('Epoch: {} training time {:.2f}m'.format(epoch, interval))
            #logger.info('Train => tot: {:.2f}, recon: {:.2f}, KL: {:.2f}; Val => tot: {:.2f}, recon: {:.2f}, KL: {:.2f}'.format(train_loss[epoch], train_recon[epoch], train_kl[epoch], val_loss[epoch], val_recon[epoch], val_kl[epoch]))
            logger.info('Train => tot: {:.2f}, recon: {:.2f}, KL: {:.2f}; Val => tot: {:.2f}, recon: {:.2f}, KL: {:.2f}, lr: {:.2f}'.format(train_loss[epoch-1], train_recon[epoch-1], train_kl[epoch-1], val_loss[epoch-1], val_recon[epoch-1], val_kl[epoch-1], current_lr))

            # Stop traning if early-stop triggers
            if cpt_patience == early_stop_patience and kl_warm >= 1.0:   #sostituire con kl se non usi RT
                logger.info('Early stop patience achieved')
                break

            # update learning rate schedule
            lr_scheduler.step()
        
        # Save the final weights of network with the best validation loss
        save_file = os.path.join(save_dir, self.model_name + '_final_epoch' + str(cur_best_epoch) + '.pt')
        torch.save(best_state_dict, save_file)
        
        # Save the training loss and validation loss
        train_loss = train_loss[:epoch]
        val_loss = val_loss[:epoch]
        train_recon = train_recon[:epoch]
        train_kl = train_kl[:epoch]
        val_recon = val_recon[:epoch]
        val_kl = val_kl[:epoch]
        loss_file = os.path.join(save_dir, 'loss_model.pckl')
        with open(loss_file, 'wb') as f:
            #pickle.dump([train_loss, val_loss, train_recon, train_kl, val_recon, val_kl], f)
            pickle.dump([train_loss, val_loss, train_recon, train_kl, val_recon, val_kl], f)

        # Save the loss figure
        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_loss, label='training loss')
        plt.plot(val_loss, label='validation loss')
        plt.legend(fontsize=16, title=self.model_name, title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        plt.semilogy()
        fig_file = os.path.join(save_dir, 'loss_{}.png'.format(tag))
        plt.savefig(fig_file)

        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_recon, label='Training')
        plt.plot(val_recon, label='Validation')
        plt.legend(fontsize=16, title='{}: Recon. Loss'.format(self.model_name), title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        plt.semilogy()
        fig_file = os.path.join(save_dir, 'loss_recon_{}.png'.format(tag))
        plt.savefig(fig_file) 

        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_kl, label='Training')
        plt.plot(val_kl, label='Validation')
        plt.legend(fontsize=16, title='{}: KL Divergence'.format(self.model_name), title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        plt.semilogy()
        fig_file = os.path.join(save_dir, 'loss_KLD_{}.png'.format(tag))
        plt.savefig(fig_file)


