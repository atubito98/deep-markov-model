#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import random
import shutil
import socket
import datetime
import pickle
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from .utils import myconf, get_logger, loss_KLD, loss_KLD_cov, loss_rec
from .dataset import process_data, data_split, MUA_dataset, BalancedDataLoader, one_hot_cont
from .model import build_VAE, build_DKF_simple, build_DKF_cx_causal_chot


class LearningAlgorithm_dir():

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
        elif self.model_name == 'DKF_simple':
            self.model = build_DKF_simple(cfg=self.cfg, device=self.device)
            self.best_model = build_DKF_simple(cfg=self.cfg, device=self.device) 
        elif self.model_name == 'DKF_cx_causal_chot':
            self.model = build_DKF_cx_causal_chot(cfg=self.cfg, device=self.device)
            self.best_model = build_DKF_cx_causal_chot(cfg=self.cfg, device=self.device) 
        

    def init_optimizer(self, train_dataloader, epochs):
        optimization  = self.cfg.get('Training', 'optimization')
        lr = self.cfg.getfloat('Training', 'lr')
        decay_lr = self.cfg.getfloat('Training', 'decay_lr')
        weight_decay = self.cfg.getfloat('Training', 'weight_decay', fallback=1e-4)
#         milestones = [] if self.cfg.get('Training', 'milestones') == '' else [int(i) for i in self.cfg.get('Training', 'milestones').split(',')]
        if optimization == 'adam': # could be extend to other optimizers
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        #lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=decay_lr)
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


    def train(self):
        ############
        ### Init ###
        ############

        # Build model
        self.build_model()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Totale parametri: {total_params}")

        # Set module.training = True
        torch.autograd.set_detect_anomaly(True)

        saved_root = self.cfg.get('User', 'saved_root')
        #RT_dim = self.cfg.getint('Network','RT_dim')
        c_dim = self.cfg.getint('Network','c_dim')
        z_dim = self.cfg.getint('Network','z_dim')
        x_dim = self.cfg.getint('Network','x_dim')
        tau = self.cfg.getint('DataFrame','tau')
        #GO_flag = self.cfg.getboolean('DataFrame','GO_flag')
        #dir_flag = self.cfg.getboolean('DataFrame','dir_flag')
        shift = self.cfg.getint('DataFrame', 'shift')
        tag = self.cfg.get('Network', 'tag')
        filename = "{}_{}".format(self.date, tag)
        save_dir = os.path.join(saved_root, filename)
        if not(os.path.isdir(save_dir)):
            os.makedirs(save_dir)
            

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
        
        gate_RT_train = np.zeros(train_set.shape[0])
        gate_RT_vali = np.zeros(vali_set.shape[0])
        gate_RT_test = np.zeros(test_set.shape[0])
        
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
            
            
        train_dataset = MUA_dataset(train_set, gate_cont_train, train_labels, session_train)
        vali_dataset = MUA_dataset(vali_set, gate_cont_vali, vali_labels, session_vali)
            
        train_dataloader = BalancedDataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        #train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        #val_dataloader = BalancedDataLoader(dataset=vali_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_dataloader = DataLoader(dataset=vali_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            
        np.savez(save_dir + "/data_split.npz", train_set = train_set, vali_set = vali_set, test_set = test_set, \
                    train_RT = train_RT, vali_RT = vali_RT, test_RT = test_RT, train_SSD = train_SSD, vali_SSD = vali_SSD, \
                    test_SSD = test_SSD, train_direction = train_direction, vali_direction = vali_direction, \
                    test_direction = test_direction, session_train = session_train, session_vali = session_vali, 
                    session_test = session_test, data_path_array = data_path_array)

        # Process test and train data
        test_data = process_data(test_set, test_RT, test_SSD, test_direction, session_test, tau)
        train_data = process_data(train_set, train_RT, train_SSD, train_direction, session_train, tau)
        
        
        logger.info('Split operated with random seed: {}'.format(seed))        
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
        #gamma = self.cfg.getfloat('Training', 'gamma')
        #C = self.cfg.getfloat('Training', 'C')
        kl_warm_steps = self.cfg.getint('Training', 'kl_warm_steps')
        kl_slope = self.cfg.getfloat('Training', 'kl_slope')
        # kl_slope = 0.2
        # kl_warm_steps = 5
        p_warm_steps = 6
        #C_warm = 0.01
        p_warm = 0
        kl_warm = 0
        last_epoch = 0
        
        # Init optimizer
        optimizer, lr_scheduler = self.init_optimizer(train_dataloader, epochs)
    
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
        
        # Train with mini-batch SGD
        for epoch in range(start_epoch+2, epochs+1):
            
            start_time = datetime.datetime.now()

            # KL warm-up
            if epoch % kl_warm_steps == 0 and kl_warm < 1:
                kl_warm = (epoch // kl_warm_steps) * kl_slope
                logger.info('KL warm-up, anneal coeff: {}'.format(kl_warm))
              
            if epoch % p_warm_steps == 0 and p_warm < 0.5:
                p_warm = (epoch // p_warm_steps) * 0.1
                logger.info('p warm-up, anneal coeff: {}'.format(p_warm))
                
            # Batch training
            self.model.train()
            
            #for _, batch_data in enumerate(train_dataloader):
            for _, (batch_data,batch_gate_cont, _) in enumerate(train_dataloader):
                
                batch_data = batch_data.float().to(self.device)
                batch_gate_cont = batch_gate_cont.float().to(self.device)
                
                bs, seq_len, _ = batch_data.shape
                optimizer.zero_grad()
                #num_train += bs
            
                # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
                batch_data = batch_data.permute(1, 0, 2)
                batch_gate_cont = batch_gate_cont.permute(1, 0, 2)
                
                recon_mean_data, recon_logvar_data = self.model(batch_data, batch_gate_cont)
                loss_recon = self.model.loss_rec(recon_mean_data, recon_logvar_data, batch_data)
                loss_recon = loss_recon.sum() / (seq_len * bs )

                loss_kl = loss_KLD_cov(self.model.z_mean, self.model.z_cov, self.model.z_mean_p, self.model.z_cov_p, z_dim)
                loss_kl = loss_kl / (seq_len * bs )
                #loss_kl = gamma * torch.abs((loss_kl / (seq_len * bs )) - C_warm)
                loss_tot = loss_recon + kl_warm * beta * loss_kl 
                
                loss_tot.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                # update learning rate schedule
                lr_scheduler.step()

                train_loss[epoch-1] += loss_tot.item() * bs
                train_recon[epoch-1] += loss_recon.item() * bs
                train_kl[epoch-1] += loss_kl.item() * bs
    
            # Validation
            self.model.eval()
            
            with torch.no_grad():
                #for _, batch_data in enumerate(val_dataloader):
                for _, (batch_data, batch_gate_cont, _) in enumerate(val_dataloader):

                    batch_data = batch_data.float().to(self.device)
                    batch_gate_cont = batch_gate_cont.float().to(self.device)

                    bs, seq_len, _ = batch_data.shape
                    #num_val += bs

                    # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
                    batch_data = batch_data.permute(1, 0, 2)
                    batch_gate_cont = batch_gate_cont.permute(1, 0, 2)
                    
                    recon_mean_data, recon_logvar_data = self.model(batch_data, batch_gate_cont)
                    loss_recon = self.model.loss_rec(recon_mean_data, recon_logvar_data, batch_data)
                    loss_recon = loss_recon.sum() / (seq_len * bs )

                    loss_kl = loss_KLD_cov(self.model.z_mean, self.model.z_cov, self.model.z_mean_p, self.model.z_cov_p, z_dim)
                    loss_kl = loss_kl / (seq_len * bs )
                    #loss_kl = gamma * torch.abs((loss_kl / (seq_len * bs )) - C_warm) 
                    loss_tot = loss_recon + kl_warm * beta * loss_kl 

                    val_loss[epoch-1] += loss_tot.item() * bs
                    val_recon[epoch-1] += loss_recon.item() * bs
                    val_kl[epoch-1] += loss_kl.item() * bs
                    
                
            # Loss normalization
            train_loss[epoch-1] = train_loss[epoch-1]/ train_num
            val_loss[epoch-1] = val_loss[epoch-1] / val_num
            train_recon[epoch-1] = train_recon[epoch-1] / train_num 
            train_kl[epoch-1] = train_kl[epoch-1]/ train_num
            val_recon[epoch-1] = val_recon[epoch-1] / val_num       
            val_kl[epoch-1] = val_kl[epoch-1] / val_num

            current_lr = lr_scheduler.get_last_lr()[0]
                
            # Early stop patiance
            if val_loss[epoch-1] < best_val_loss or kl_warm < 1 or epoch==int(kl_warm_steps/kl_slope): #C_warm < 1:  #sostituire con kl se non usi C
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

                #############################################################################################      


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
            #lr_scheduler.step()
        
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
        
        n_test, steps, x_dim = test_set.shape
        ############## PCA ###############
        X_train = train_set.reshape(-1, x_dim)  # shape = (trials * time_steps, 96)
        X_vali = vali_set.reshape(-1, x_dim)  # shape = (trials * time_steps, 96)
        X_test = test_set.reshape(-1, x_dim)  # shape = (trials * time_steps, 96)

        from sklearn.decomposition import PCA

        pca = PCA(n_components=z_dim)
        X_train_pca = pca.fit_transform(X_train)  # fit + transform sul train
        X_test_pca = pca.transform(X_test)   
        X_test_rec = pca.inverse_transform(X_test_pca)

        test_rec = X_test_rec.reshape(n_test, steps, x_dim)

        ############## DMM ##############
        test_data = torch.from_numpy(test_set).float().to(self.device).permute(1, 0, 2)
        cont_test = torch.from_numpy(gate_cont_test).float().to(self.device).permute(1, 0, 2)
        
        # Output finale
        chunk_size = 10
        n_trials = 10
        y_accum = np.zeros((steps, n_test, x_dim), dtype=np.float32)

        for start in range(0, n_test, chunk_size):
            end = min(start + chunk_size, n_test)
            batch_size = end - start

            # Estrai chunk
            x_chunk = test_data[:, start:end, :].repeat_interleave(n_trials, dim=1)
            c_chunk = cont_test[:, start:end, :].repeat_interleave(n_trials, dim=1)

            # Inferenza
            with torch.no_grad():
                y_mean, y_logvar = self.model(x_chunk, c_chunk)
                y_inf = self.model.reparameterization(y_mean, y_logvar)

            # Media sui trials (dim=2)
            y_inf = y_inf.cpu().numpy().reshape(steps, batch_size, n_trials, x_dim).mean(2)

            # Inserisci nel buffer
            y_accum[:, start:end, :] = y_inf

            torch.cuda.empty_cache()
        y_inf = np.transpose(y_accum, (1, 0, 2))

        # compute residual MSE

        residual_DMM = test_set - y_inf
        residual_PCA = test_set - test_rec

        MSE_DMM = np.mean(residual_DMM**2)
        MSE_PCA = np.mean(residual_PCA**2)
        
        torch.save({
                    'n_trials': n_trials,
                    'MSE_PCA': MSE_PCA,
                    'MSE_DMM': MSE_DMM,
                }, save_dir + "/MSE")

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


