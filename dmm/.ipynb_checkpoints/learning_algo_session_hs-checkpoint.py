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
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from .utils import correlation_vs_gentime, latent_traj_ctype, MUA_pred_inf_plot, MUA_pred_inf, myconf, get_logger, loss_KLD, loss_KLD_cov, loss_KLD_diag, loss_rec
from .dataset import process_data, data_split, MUA_dataset, BalancedHierarchicalSampler, one_hot_cont
from .model import build_VAE, build_DKF_cx_causal_chot, build_DKF_session_hs

class LearningAlgorithm_session_hs():

    """
    Basical class for model building, including:
    - read common parameters for different models
    - define data loader
    - define loss function as a class memberdata_split
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
        elif self.model_name == 'DKF_cx_causal_chot':
            self.model = build_DKF_cx_causal_chot(cfg=self.cfg, device=self.device)
            self.best_model = build_DKF_cx_causal_chot(cfg=self.cfg, device=self.device) 
        elif self.model_name == 'DKF_session_hs':
            self.model = build_DKF_session_hs(cfg=self.cfg, device=self.device)
            self.best_model = build_DKF_session_hs(cfg=self.cfg, device=self.device)
        

    def init_optimizer(self):
        optimization  = self.cfg.get('Training', 'optimization')
        lr = self.cfg.getfloat('Training', 'lr')
        decay_lr = self.cfg.getfloat('Training', 'decay_lr')
        milestones = [] if self.cfg.get('Training', 'milestones') == '' else [int(i) for i in self.cfg.get('Training', 'milestones').split(',')]
        if optimization == 'adam': # could be extend to other optimizers
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=decay_lr)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=decay_lr)
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

        # Set module.training = True
        torch.autograd.set_detect_anomaly(True)

        # Create directory for results
        if not self.params['reload']:
            saved_root = self.cfg.get('User', 'saved_root')
            #RT_dim = self.cfg.getint('Network','RT_dim')
            c_dim = self.cfg.getint('Network','c_dim')
            z_dim = self.cfg.getint('Network','z_dim')
            x_dim = self.cfg.getint('Network','x_dim')
            s_dim = self.cfg.getint('Network','s_dim')
            tau = self.cfg.getint('DataFrame','tau')
            GO_flag = self.cfg.getboolean('DataFrame','GO_flag')
            dir_flag = self.cfg.getboolean('DataFrame','dir_flag')
            shift = self.cfg.getint('DataFrame', 'shift')
            tag = self.cfg.get('Network', 'tag')
            filename = "{}_{}".format(self.date, tag)
            save_dir = os.path.join(saved_root, filename)
            if not(os.path.isdir(save_dir)):
                os.makedirs(save_dir)
        else:
            #RT_dim = self.cfg.getint('Network','RT_dim')
            c_dim = self.cfg.getint('Network','c_dim')
            z_dim = self.cfg.getint('Network','z_dim')
            x_dim = self.cfg.getint('Network','x_dim')
            s_dim = self.cfg.getint('Network','s_dim')
            tau = self.cfg.getint('DataFrame','tau')
            GO_flag = self.cfg.getboolean('DataFrame','GO_flag')
            dir_flag = self.cfg.getboolean('DataFrame','dir_flag')
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

        # Init optimizer
        optimizer, lr_scheduler = self.init_optimizer()

        # Create data loader
        train_set, vali_set, test_set, train_RT, vali_RT, test_RT, train_SSD, vali_SSD, test_SSD, train_direction, \
        vali_direction, test_direction, session_train, session_vali, session_test, seed = data_split.split(self.cfg)
        
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
            
        #train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        #val_dataloader = DataLoader(dataset=vali_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        #test_dataloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            
        train_dataset = MUA_dataset(train_set, gate_cont_train, train_labels, session_train)
        vali_dataset = MUA_dataset(vali_set, gate_cont_vali, vali_labels, session_vali)

            
        train_sampler = BalancedHierarchicalSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
        #train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        #val_dataloader = BalancedDataLoader(dataset=vali_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_dataloader = DataLoader(dataset=vali_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            
        np.savez(save_dir + "/data_split.npz", train_set = train_set, vali_set = vali_set, test_set = test_set, \
                    train_RT = train_RT, vali_RT = vali_RT, test_RT = test_RT, train_SSD = train_SSD, vali_SSD = vali_SSD, \
                    test_SSD = test_SSD, train_direction = train_direction, vali_direction = vali_direction, \
                    test_direction = test_direction, session_train = session_train, session_vali = session_vali, 
                    session_test = session_test)
        
        # Load data
        #with np.load(save_dir + "/data_split.npz") as loaded_file:
            #data = {key: loaded_file[key] for key in loaded_file.files}
            #locals().update(data)  # Unpack all variables to local namespace

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
        kl_warm_steps = 5
        kl_warm = 0
        last_epoch = 0
    

        # Create python list for loss
        if not self.params['reload']:
            train_loss = np.zeros((epochs,))
            val_loss = np.zeros((epochs,))
            train_recon = np.zeros((epochs,))
            train_loss_session = np.zeros((epochs,))
            train_kl_z = np.zeros((epochs,))
            train_kl_s = np.zeros((epochs,))
            val_recon = np.zeros((epochs,))
            val_loss_session = np.zeros((epochs,))
            val_kl_z = np.zeros((epochs,))
            val_kl_s = np.zeros((epochs,))
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
            train_loss_session = np.pad(loss_log['train_loss_session'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            train_kl_z = np.pad(loss_log['train_kl_z'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            train_kl_s = np.pad(loss_log['train_kl_s'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            val_recon = np.pad(loss_log['val_recon'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            val_loss_session = np.pad(loss_log['val_loss_session'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            val_kl_z = np.pad(loss_log['val_kl_z'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            val_kl_s = np.pad(loss_log['val_kl_s'], (0, epochs-start_epoch), mode='constant', constant_values=0)
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
                kl_warm = (epoch // kl_warm_steps) * 0.2
                logger.info('KL warm-up, anneal coeff: {}'.format(kl_warm))
              
            # Batch training
            self.model.train()
            num_train = 0
            num_val = 0
            #for _, batch_data in enumerate(train_dataloader):
            for _, (batch_data, batch_gate_cont, batch_session) in enumerate(train_dataloader):
                
                batch_data = batch_data.float().to(self.device)
                batch_gate_cont = batch_gate_cont.float().to(self.device)
                batch_session = batch_session.float().to(self.device)
                
                bs, seq_len, _ = batch_data.shape
                num_train += bs
                
                # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
                batch_data = batch_data.permute(1, 0, 2)
                batch_gate_cont = batch_gate_cont.permute(1, 0, 2)
                batch_session = batch_session.permute(1, 0, 2)
                
                recon_mean_data, recon_logvar_data = self.model(batch_data, batch_gate_cont, sample_for_s=(batch_data, batch_gate_cont))
                loss_recon = self.model.loss_rec(recon_mean_data, recon_logvar_data, batch_data) / (seq_len * bs)
                #loss_recon = loss_recon.sum() / (seq_len * bs )

                #two = torch.full((self.model.z_mean.shape[0], self.model.z_mean.shape[1]), z_dim).to(self.device)
                loss_kl_z = loss_KLD_cov(self.model.z_mean, self.model.z_cov, self.model.z_mean_p, self.model.z_cov_p, z_dim) / (seq_len * bs)
                #loss_kl_z = loss_kl_z / (seq_len * bs )
                #loss_kl = gamma * torch.abs((loss_kl / (seq_len * bs )) - C_warm)
                
                loss_kl_s = loss_KLD_diag(self.model.s_mean, self.model.s_logvar) / (seq_len * bs)
                
                sess_embed = self.session_embeddings(batch_session)
                loss_session = self.model.loss_rec(self.model.s_mean, self.model.s_logvar, sess_embed) / (seq_len * bs)
                
                loss_tot = loss_recon + kl_warm * beta * loss_kl_z + kl_warm * loss_kl_s + kl_warm * loss_session
                
                optimizer.zero_grad()
                loss_tot.backward()
                optimizer.step()

                train_loss[epoch-1] += loss_tot.item() * bs
                train_recon[epoch-1] += loss_recon.item() * bs
                train_kl_z[epoch-1] += loss_kl_z.item() * bs
                train_kl_s[epoch-1] += loss_kl_s.item() * bs
                train_loss_session[epoch-1] += loss_session.item() * bs
            
            # Validation
            self.model.eval()
            
            with torch.no_grad():
                #for _, batch_data in enumerate(val_dataloader):
                for _, (batch_data, batch_gate_cont, batch_session) in enumerate(val_dataloader):

                    batch_data = batch_data.float().to(self.device)
                    batch_gate_cont = batch_gate_cont.float().to(self.device)
                    batch_session = batch_session.float().to(self.device)

                    bs, seq_len, _ = batch_data.shape
                    num_val += bs

                    # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
                    batch_data = batch_data.permute(1, 0, 2)
                    batch_gate_cont = batch_gate_cont.permute(1, 0, 2)
                    batch_session = batch_session.permute(1, 0, 2)
                    
                    recon_mean_data, recon_logvar_data = self.model(batch_data, batch_gate_cont, sample_for_s=(batch_data, batch_gate_cont))
                    loss_recon = self.model.loss_rec(recon_mean_data, recon_logvar_data, batch_data) / (seq_len * bs)

                    loss_kl_z = loss_KLD_cov(self.model.z_mean, self.model.z_cov, self.model.z_mean_p, self.model.z_cov_p, z_dim) / (seq_len * bs)
                    #loss_kl_z = loss_kl_z / (seq_len * bs )
                    #loss_kl = gamma * torch.abs((loss_kl / (seq_len * bs )) - C_warm)

                    loss_kl_s = loss_KLD_diag(self.model.s_mean, self.model.s_logvar) / (seq_len * bs)

                    sess_embed = self.session_embeddings(batch_session)
                    loss_session = self.model.loss_rec(self.model.s_mean, self.model.s_logvar, sess_embed) / (seq_len * bs)

                    loss_tot = loss_recon + kl_warm * beta * loss_kl_z + kl_warm * loss_kl_s + kl_warm * loss_session

                    val_loss[epoch-1] += loss_tot.item() * bs
                    val_recon[epoch-1] += loss_recon.item() * bs
                    val_kl_z[epoch-1] += loss_kl_z.item() * bs
                    val_kl_s[epoch-1] += loss_kl_s.item() * bs
                    val_loss_session[epoch-1] += loss_session.item() * bs
                
            # Loss normalization
            train_loss[epoch-1] = train_loss[epoch-1]/ num_train
            val_loss[epoch-1] = val_loss[epoch-1] / num_val
            train_recon[epoch-1] = train_recon[epoch-1] / num_train 
            train_kl_z[epoch-1] = train_kl_z[epoch-1]/ num_train
            train_loss_session[epoch-1] = train_loss_session[epoch-1] / num_train 
            train_kl_s[epoch-1] = train_kl_s[epoch-1]/ num_train
            val_recon[epoch-1] = val_recon[epoch-1] / num_val       
            val_kl_z[epoch-1] = val_kl_z[epoch-1] / num_val
            val_loss_session[epoch-1] = val_loss_session[epoch-1] / num_val       
            val_kl_s[epoch-1] = val_kl_s[epoch-1] / num_val

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
                            'train_kl_z': train_kl_z[:cur_best_epoch],
                            'train_loss_session': train_loss_session[:cur_best_epoch],
                            'train_kl_s': train_kl_s[:cur_best_epoch],
                            'val_recon': val_recon[:cur_best_epoch], 
                            'val_kl_z': val_kl_z[:cur_best_epoch],
                            'val_loss_session': val_loss_session[:cur_best_epoch], 
                            'val_kl_s': val_kl_s[:cur_best_epoch],
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
            logger.info('Train => tot: {:.2f}, recon: {:.2f}, KL_z: {:.2f}, KL_s: {:.2f}, rec_s: {:.2f}; Val => tot: {:.2f}, recon: {:.2f}, KL_z: {:.2f}, KL_s: {:.2f}, rec_s: {:.2f}, lr: {:.2f}'.format(train_loss[epoch-1], train_recon[epoch-1], train_kl_z[epoch-1], train_kl_s[epoch-1], train_loss_session[epoch-1], val_loss[epoch-1], val_recon[epoch-1], val_kl_z[epoch-1], val_kl_s[epoch-1], train_loss_session[epoch-1], current_lr))

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
        train_kl_z = train_kl_z[:epoch]
        train_kl_s = train_kl_s[:epoch]
        train_loss_session = train_loss_session[:epoch]
        val_recon = val_recon[:epoch]
        val_kl_z = val_kl_z[:epoch]
        val_kl_s = val_kl_s[:epoch]
        val_loss_session = val_loss_session[:epoch]
        loss_file = os.path.join(save_dir, 'loss_model.pckl')
        with open(loss_file, 'wb') as f:
            #pickle.dump([train_loss, val_loss, train_recon, train_kl, val_recon, val_kl], f)
            pickle.dump([train_loss, val_loss, train_recon, train_kl_z, train_kl_s, train_loss_session, val_recon, val_kl_z, val_kl_s, val_loss_session], f)

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


        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_kl_s, label='Training')
        plt.plot(val_kl_s, label='Validation')
        plt.legend(fontsize=16, title='{}: KL Divergence for session latent space'.format(self.model_name), title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        plt.semilogy()
        fig_file = os.path.join(save_dir, 'loss_KLD_s_{}.png'.format(tag))
        plt.savefig(fig_file)
        
        
        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_loss_session, label='Training')
        plt.plot(val_loss_session, label='Validation')
        plt.legend(fontsize=16, title='{}: Recon. loss for session latent space'.format(self.model_name), title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        plt.semilogy()
        fig_file = os.path.join(save_dir, 'loss_recon_s_{}.png'.format(tag))
        plt.savefig(fig_file)