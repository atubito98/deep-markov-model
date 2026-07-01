#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from .dataset import one_hot_cont
from .learning_algo_dir import LearningAlgorithm_dir


class Computing_MSE():
    def __init__(self, params, saved_dir=None):
        # Load config parser
        self.params = params
        # Hyperparameters
        self.net = self.params["net"]
        if self.net == "sess" or self.net == "sess_u":
            self.train_sess = self.params["train_sess"]
            self.test_sess = self.params["test_sess"]
        self.n_trials = self.params['n_trials']
        self.chunk_size = self.params['chunk_size']
        self.weights_dir = self.params['weights_dir']
        self.device = self.params['device']
        if saved_dir is None:
            self.saved_dir = params["saved_dir"]
        else:
            self.saved_dir = saved_dir
        
        dmm_params = {
            "cpast": False,
            "cfg": self.saved_dir + "/config.ini",
            "ss": False,
            "device": self.device,
            "saved_dict": self.saved_dir,
        }
        
#         if self.net == "c_hot":
        learning_algo = LearningAlgorithm_dir(params=dmm_params)
#             self.u_flag = False
#         elif self.net == "sess":
#             learning_algo = LearningAlgorithm_s(params=dmm_params)
#             self.u_flag = False
#         elif self.net == "u":
#             learning_algo = LearningAlgorithm_u(params=dmm_params)
#             self.u_flag = True
#         elif self.net == "sess_u":
#             learning_algo = LearningAlgorithm_su(params=dmm_params)
#             self.u_flag = True
        learning_algo.build_model()
        self.dmm = learning_algo.model
#         if self.net == "sess": 
#             with np.load(self.saved_dir+"/data_split.npz", allow_pickle=True) as loaded_file:
#                 all_name = loaded_file["all_name"]
#             for sess in all_name:
#                 self.dmm.add_session(sess)
        self.dmm.load_state_dict(torch.load(self.saved_dir + self.weights_dir, map_location='cpu'))
        self.dmm.eval()
        self.tau = self.dmm.tau
        self.z_dim = self.dmm.z_dim
        
        
    def compute(self):
        
        with np.load(self.saved_dir+"/data_split.npz", allow_pickle=True) as loaded_file:
            train_set = loaded_file["train_set"]
            vali_set = loaded_file["vali_set"]
            test_set = loaded_file["test_set"]
            train_direction = loaded_file["train_direction"]
            vali_direction = loaded_file["vali_direction"]
            test_direction = loaded_file["test_direction"]
            train_SSD = loaded_file["train_SSD"]
            vali_SSD = loaded_file["vali_SSD"]
            test_SSD = loaded_file["test_SSD"]
#             if self.net == "sess" or self.net == "sess_u":
#                 session_train = loaded_file["train_session"]
#                 session_vali = loaded_file["vali_session"]
#                 session_test = loaded_file["test_session"]
                
#                 session_list_train, mask_sess_train = sess_sublist(session_train, session_spec=self.train_sess)
#                 session_list_vali, mask_sess_vali = sess_sublist(session_vali, session_spec=self.train_sess)
#                 session_list_test, mask_sess_test = sess_sublist(session_test, session_spec=self.test_sess)
#             else:
            mask_sess_train = np.ones(train_set.shape[0], dtype=bool)
            mask_sess_vali = np.ones(vali_set.shape[0], dtype=bool)
            mask_sess_test = np.ones(test_set.shape[0], dtype=bool)
            
#         cont_train = one_hot_cont(train_SSD, train_direction, self.tau)
#         cont_vali = one_hot_cont(vali_SSD, vali_direction, self.tau)
        cont_test = one_hot_cont(test_SSD, test_direction, self.tau)
        
#         train_set = train_set[mask_sess_train]
#         vali_set = vali_set[mask_sess_vali]
        test_set = test_set[mask_sess_test]
        
#         cont_train = cont_train[mask_sess_train]
#         cont_vali = cont_vali[mask_sess_vali]
        cont_test = cont_test[mask_sess_test]
        
        n_train, steps, features = train_set.shape
        n_test, _, c_dim = cont_test.shape
        
        ############## PCA ###############
        X_train = train_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
        X_vali = vali_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
        X_test = test_set.reshape(-1, features)  # shape = (trials * time_steps, 96)

        from sklearn.decomposition import PCA

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
                if self.net == "c_hot" or self.net == "u":
                    y_mean, y_logvar = self.dmm(x_chunk, c_chunk)
#                 elif self.net == "sess" or self.net == "sess_u":
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
                }, self.saved_dir + "/MSE")
        
        print(f"MSE_PCA = {MSE_PCA}, MSE_DMM = {MSE_DMM}")

    # Function to load model weights
    def load(self):
        saved_dict = self.saved_dir + "/MSE"
        if os.path.exists(saved_dict):
            checkpoint = torch.load(saved_dict)
            MSE_PCA = checkpoint['MSE_PCA']
            MSE_DMM = checkpoint['MSE_DMM']
            return MSE_DMM, MSE_PCA
        else:
            print("No saved MSE found")
