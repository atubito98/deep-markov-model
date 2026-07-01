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
        
        
    def compute(self):
        
        with np.load(self.saved_dir / "data_split.npz", allow_pickle=True) as loaded_file:
            train_set = loaded_file["train_set"]
            vali_set = loaded_file["vali_set"]
            test_set = loaded_file["test_set"]
            test_direction = loaded_file["test_direction"]
            test_SSD = loaded_file["test_SSD"]
            
        cont_test = one_hot_cont(test_SSD, test_direction, self.tau)
        
        _, steps, features = train_set.shape
        n_test, _, _ = cont_test.shape
        
        ############## PCA ###############
        X_train = train_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
        X_vali = vali_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
        X_test = test_set.reshape(-1, features)  # shape = (trials * time_steps, 96)

        pca = PCA(n_components=self.z_dim)
        X_train_pca = pca.fit_transform(X_train)  # fit + transform sul train
        X_test_pca = pca.transform(X_test)   
        X_test_rec = pca.inverse_transform(X_test_pca)
        test_rec = X_test_rec.reshape(n_test, steps, features)

        ############## DMM ##############
        test_data = torch.from_numpy(test_set).float().to(self.device).permute(1, 0, 2)
        cont_test = torch.from_numpy(cont_test).float().to(self.device).permute(1, 0, 2)
        
        # Output finale
        chunk_size = self.chunk_size
        y_accum = np.zeros((steps, n_test, features), dtype=np.float32)

        for start in range(0, n_test, chunk_size):
            end = min(start + chunk_size, n_test)
            batch_size = end - start

            # Estrai chunk
            x_chunk = test_data[:, start:end, :].repeat_interleave(self.n_trials, dim=1)
            c_chunk = cont_test[:, start:end, :].repeat_interleave(self.n_trials, dim=1)

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

        # compute residual MSE
        residual_DMM = test_set - y_inf
        residual_PCA = test_set - test_rec
        
        MSE_DMM = np.mean(residual_DMM**2)
        MSE_PCA = np.mean(residual_PCA**2)
        
        torch.save({
                    'params': self.params,
                    'MSE_PCA': MSE_PCA,
                    'MSE_DMM': MSE_DMM,
                }, self.saved_dir / "MSE")
        
        print(f"MSE_PCA = {MSE_PCA}, MSE_DMM = {MSE_DMM}")

    # Function to load model weights
    def load(self):
        saved_dict = self.saved_dir / "MSE"
        if os.path.exists(saved_dict):
            checkpoint = torch.load(saved_dict, map_location=self.device)
            MSE_PCA = checkpoint['MSE_PCA']
            MSE_DMM = checkpoint['MSE_DMM']
            return MSE_DMM, MSE_PCA
        else:
            print("No saved MSE found")