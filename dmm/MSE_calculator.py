#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from sklearn.decomposition import PCA
from .dataset import one_hot_cont
from .learning_algo_dir import LearningAlgorithm_dir


class Computing_MSE():
    def __init__(self, params, device=None, saved_dir=None):
        # Load config parser
        self.params = params
        # Hyperparameters
        self.n_trials = self.params['n_trials']
        self.chunk_size = self.params['chunk_size']
        self.weights_dir = self.params['weights_dir']
        self.device = device if device is not None else self.params["device"]
        self.saved_dir = self.params["saved_dir"] if saved_dir is None else saved_dir
        
        dmm_params = {
            "cfg": self.saved_dir / "config.ini",
            "device": self.device,
            "saved_dict": self.saved_dir,
        }
                
        learning_algo = LearningAlgorithm_dir(params=dmm_params)
        self.dmm = learning_algo.model
        self.dmm.load_state_dict(torch.load(self.saved_dir / self.weights_dir, map_location=self.device))
        self.dmm.eval()
        self.tau = self.dmm.tau
        self.z_dim = self.dmm.z_dim
        
    def generate_and_rec(self, data_set, cont_set):
        n_data, steps, features = data_set.shape
        data = torch.from_numpy(data_set).float().to(self.device).permute(1, 0, 2)
        cont = torch.from_numpy(cont_set).float().to(self.device).permute(1, 0, 2)
        # Output finale
        chunk_size = self.chunk_size
        y_accum = np.zeros((steps, n_data, features), dtype=np.float32)
        for start in range(0, n_data, chunk_size):
            end = min(start + chunk_size, n_data)
            batch_size = end - start

            # Estrai chunk
            x_chunk = data[:, start:end, :].repeat_interleave(self.n_trials, dim=1)
            c_chunk = cont[:, start:end, :].repeat_interleave(self.n_trials, dim=1)

            # Inferenza
            with torch.no_grad():
                y_mean, y_logvar = self.dmm(x_chunk, c_chunk)
                y_inf = self.dmm.reparameterization(y_mean, y_logvar)

            # Media sui trials (dim=2)
            y_inf = y_inf.cpu().numpy().reshape(steps, batch_size, self.n_trials, features).mean(2)
            # Inserisci nel buffer
            y_accum[:, start:end, :] = y_inf

            torch.cuda.empty_cache()
        y_inf = np.transpose(y_accum, (1, 0, 2))
        return y_inf
    
        
    def compute(self):
        
        with np.load(self.saved_dir / "data_split.npz", allow_pickle=True) as loaded_file:
            train_set = loaded_file["train_set"]
            vali_set = loaded_file["vali_set"]
            test_set = loaded_file["test_set"]
            vali_direction = loaded_file["vali_direction"]
            test_direction = loaded_file["test_direction"]
            vali_SSD = loaded_file["vali_SSD"]
            test_SSD = loaded_file["test_SSD"]
            
        cont_vali = one_hot_cont(vali_SSD, vali_direction, self.tau)
        cont_test = one_hot_cont(test_SSD, test_direction, self.tau)
        
        _, steps, features = train_set.shape
        
        ############## PCA ###############
        X_train = train_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
        X_vali = vali_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
        X_test = test_set.reshape(-1, features)  # shape = (trials * time_steps, 96)

        pca = PCA(n_components=self.z_dim)
        pca.fit(X_train)  # fit + transform sul train
        X_vali_pca = pca.transform(X_vali)
        X_test_pca = pca.transform(X_test)   
        vali_rec = pca.inverse_transform(X_vali_pca).reshape(-1, steps, features)
        test_rec = pca.inverse_transform(X_test_pca).reshape(-1, steps, features)
    
        ############## DMM ##############
        y_vali = self.generate_and_rec(vali_set, cont_vali)
        y_test = self.generate_and_rec(test_set, cont_test)

        # compute residual MSE
        residual_DMM_vali = vali_set - y_vali
        residual_DMM_test = test_set - y_test
        
        residual_PCA_vali = vali_set - vali_rec
        residual_PCA_test = test_set - test_rec
        
        MSE_DMM_vali = np.mean(residual_DMM_vali**2)
        MSE_DMM_test = np.mean(residual_DMM_test**2)
        MSE_PCA_vali = np.mean(residual_PCA_vali**2)
        MSE_PCA_test = np.mean(residual_PCA_test**2)
        
        torch.save({
                    'params': self.params,
                    'MSE_PCA_vali': MSE_PCA_vali,
                    'MSE_PCA_test': MSE_PCA_test,
                    'MSE_DMM_vali': MSE_DMM_vali,
                    'MSE_DMM_test': MSE_DMM_test,
                }, self.saved_dir / "MSE")
        
        print(f"MSE_PCA_vali = {MSE_PCA_vali}, MSE_PCA_test = {MSE_PCA_test}, MSE_DMM_vali = {MSE_DMM_vali}, MSE_DMM_test = {MSE_DMM_test}")

    # Function to load model weights
    def load(self):
        saved_dict = self.saved_dir / "MSE"
        if os.path.exists(saved_dict):
            checkpoint = torch.load(saved_dict, map_location=self.device)
            MSE_PCA_vali = checkpoint['MSE_PCA_vali']
            MSE_PCA_test = checkpoint['MSE_PCA_test']
            MSE_DMM_vali = checkpoint['MSE_DMM_vali']
            MSE_DMM_test = checkpoint['MSE_DMM_test']
            return MSE_DMM_vali, MSE_DMM_test, MSE_PCA_vali, MSE_PCA_test
        else:
            print("No saved MSE found")