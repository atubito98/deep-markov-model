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
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from .dataset.dataloader import load_test_set
import torch.nn.functional as F
from .utils import inference_with_trials_e_all, inference_with_trials
from .dataset import one_hot_cont, process_data, data_split, MUA_dataset
from .learning_algo_session_e import LearningAlgorithm_session_e


# ==== Markovian MLP ====
class MarkovMLP(nn.Module):
    def __init__(self, x_dim, c_dim, num_sessions, hidden_dim=128):
        super().__init__()
        self.num_sessions = num_sessions
        self.net = nn.Sequential(
            nn.Linear(x_dim + c_dim + num_sessions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim)
        )

    def forward(self, x, c, s):
        s = F.one_hot(s, num_classes=self.num_sessions)
        inp = torch.cat([x, c, s], dim=-1)
        return self.net(inp)


# ==== LSTM (non-Markovian) model ====
class NonMarkovLSTM(nn.Module):
    def __init__(self, x_dim, c_dim, num_sessions, hidden_dim=128, num_layers=1):
        super().__init__()
        self.num_sessions = num_sessions
        self.lstm = nn.LSTM(x_dim + c_dim + num_sessions, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, x_dim)

    def forward(self, x, c, s):
        steps = x.shape[1]
        s = s.unsqueeze(1).expand(-1, steps)
        s = F.one_hot(s, num_classes=self.num_sessions)
        inp = torch.cat([x, c, s], dim=-1)  # concatenazione interna
        out, _ = self.lstm(inp)
        return self.fc(out)  # predizioni per ogni step


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_samples = 0
    for x, c, s, y in loader:
        x = x.to(device)
        c = c.to(device)
        y = y.to(device)
        s = s.to(device).long()
        optimizer.zero_grad()
        pred = model(x, c, s)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
        total_samples += x.shape[0]
    return total_loss / total_samples

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for x, c, s, y in loader:
            x = x.to(device)
            c = c.to(device)
            y = y.to(device)
            s = s.to(device).long()
            pred = model(x, c, s)
            loss = criterion(pred, y)
            total_loss += loss.item() * len(x)
            total_samples += x.shape[0]
    return total_loss / total_samples

def build_seq_data(X, C):
    X_in = X[:, :-1, :]   # x_{1:t}
    C_in = C[:, 1:, :]   # c_{1:t}
    Y_out = X[:, 1:, :]   # x_{2:t+1}
    return X_in, C_in, Y_out

def build_seq_data(X, C):
    X_in = X[:, :-1, :]   # x_{1:t}
    C_in = C[:, 1:, :]   # c_{1:t}
    Y_out = X[:, 1:, :]   # x_{2:t+1}
    return X_in, C_in, Y_out




class Learning_markov_detector_e():
    def __init__(self, params):
        # Load config parser
        self.params = params
        # Hyperparameters
        self.type = self.params["type"]
        self.batch_size_seq = self.params['batch_size_seq']
        self.batch_size_m = self.params['batch_size_m']
        self.learning_rate = self.params['learning_rate']
        #self.hidden_size = self.params['hidden_size']
        #self.num_layers = self.params['num_layers']
        self.num_epochs = self.params['num_epochs']
        self.n_trials = self.params['n_trials']
        #self.dropout = self.params['dropout']
        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%M")
        self.saved_dir = self.params['saved_dir']
        self.weights_dir = self.params['weights_dir']
        self.hidden_dim = self.params['hidden_dim']
        self.device = self.params['device']
        
        dvae_params = {
            "type": "dir",
            "cpast": False,
            "dir_flag": True,
            "cfg": self.saved_dir + "/config.ini",
            "ss": False,
            "device": self.device,
            "saved_dict": self.saved_dir,
        }

        learning_algo = LearningAlgorithm_session_e(params=dvae_params)
        learning_algo.build_model()
        self.dvae = learning_algo.model
        self.dvae.load_state_dict(torch.load(self.saved_dir + self.weights_dir, map_location='cpu'))
        self.dvae.eval()
        self.tau = self.dvae.tau
        self.c_dim = self.dvae.c_dim
        self.z_dim = self.dvae.z_dim
        self.x_dim = self.dvae.x_dim
        self.num_sessions = self.dvae.num_sessions
        self.input_dim = self.x_dim if self.type == "x" else self.z_dim
        
        self.model_m = MarkovMLP(self.input_dim, self.c_dim, self.num_sessions, self.hidden_dim).to(self.device)  # Use same device as data
        self.model_seq = NonMarkovLSTM(self.input_dim, self.c_dim, self.num_sessions, self.hidden_dim).to(self.device)  
        self.criterion = nn.MSELoss()
        _, _, _, _, _, _, self.MSE_prop = self.create_loaders()
        
        
    def create_loaders(self):
        
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
            session_train = loaded_file["session_train"]
            session_vali = loaded_file["session_vali"]
            session_test = loaded_file["session_test"]
        
        cont_train = one_hot_cont(train_SSD, train_direction, self.tau)
        cont_vali = one_hot_cont(vali_SSD, vali_direction, self.tau)
        cont_test = one_hot_cont(test_SSD, test_direction, self.tau)
        
        n_train, steps, features = train_set.shape
        n_vali, _, c_dim = cont_vali.shape
        n_test = test_set.shape[0]
        
        if self.type == "x":
            train_data = train_set  # n_train x steps x features 
            vali_data = vali_set
            test_data = test_set

        
        ####################### z_dataset #######################
        
        if self.type == "z":
            
            train_data = inference_with_trials_e_all(self.dvae, train_set, cont_train, session_train, self.n_trials, self.input_dim, self.device, chunk_size=128)
            vali_data = inference_with_trials_e_all(self.dvae, vali_set, cont_vali, session_vali, self.n_trials, self.input_dim, self.device, chunk_size=128)
            test_data = inference_with_trials_e_all(self.dvae, test_set, cont_test, session_test, self.n_trials, self.input_dim, self.device, chunk_size=128)
               
        ################## PCA ######################
        
        if self.type == "PCA":

            X_train = train_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
            X_vali = vali_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
            X_test = test_set.reshape(-1, features)  # shape = (trials * time_steps, 96)

            from sklearn.decomposition import PCA

            pca = PCA(n_components=self.z_dim)
            X_train_pca = pca.fit_transform(X_train)  # fit + transform sul train
            X_vali_pca = pca.transform(X_vali) 
            X_test_pca = pca.transform(X_test)   

            train_data = X_train_pca.reshape(n_train, steps, self.z_dim)
            vali_data = X_vali_pca.reshape(n_vali, steps, self.z_dim)
            test_data = X_test_pca.reshape(n_test, steps, self.z_dim)
        
        ################# Dataloader creation ##################
            
        X_train_seq, C_train_seq, Y_train_seq = build_seq_data(train_data, cont_train)
        X_vali_seq, C_vali_seq, Y_vali_seq = build_seq_data(vali_data, cont_vali)
        X_test_seq, C_test_seq, Y_test_seq = build_seq_data(test_data, cont_test)
        
        X_train_seq = torch.from_numpy(X_train_seq).float()
        C_train_seq = torch.from_numpy(C_train_seq).float()
        Y_train_seq = torch.from_numpy(Y_train_seq).float()

        X_vali_seq = torch.from_numpy(X_vali_seq).float()
        C_vali_seq = torch.from_numpy(C_vali_seq).float()
        Y_vali_seq = torch.from_numpy(Y_vali_seq).float()

        X_test_seq = torch.from_numpy(X_test_seq).float()
        C_test_seq = torch.from_numpy(C_test_seq).float()
        Y_test_seq = torch.from_numpy(Y_test_seq).float()

        
        X_train_m, C_train_m, Y_train_m = X_train_seq.reshape(-1,self.input_dim), C_train_seq.reshape(-1,self.c_dim), Y_train_seq.reshape(-1,self.input_dim)
        X_vali_m, C_vali_m, Y_vali_m = X_vali_seq.reshape(-1, self.input_dim), C_vali_seq.reshape(-1, self.c_dim), Y_vali_seq.reshape(-1, self.input_dim)
        X_test_m, C_test_m, Y_test_m = X_test_seq.reshape(-1, self.input_dim), C_test_seq.reshape(-1, self.c_dim), Y_test_seq.reshape(-1, self.input_dim)
        
        session_train_seq = torch.from_numpy(session_train).long()
        session_vali_seq = torch.from_numpy(session_vali).long()
        session_test_seq = torch.from_numpy(session_test).long()
        
        session_train_m = session_train_seq.unsqueeze(1).expand(-1, 256//self.tau - 1).reshape(-1)
        session_vali_m = session_vali_seq.unsqueeze(1).expand(-1, 256//self.tau - 1).reshape(-1)
        session_test_m = session_test_seq.unsqueeze(1).expand(-1, 256//self.tau - 1).reshape(-1)
          
        train_loader_m = DataLoader(TensorDataset(X_train_m, C_train_m, session_train_m, Y_train_m), batch_size=self.batch_size_m, shuffle=True)
        vali_loader_m = DataLoader(TensorDataset(X_vali_m, C_vali_m, session_vali_m, Y_vali_m), batch_size=self.batch_size_m)
        test_loader_m = DataLoader(TensorDataset(X_test_m, C_test_m, session_test_m, Y_test_m), batch_size=self.batch_size_m)
        
        train_loader_seq = DataLoader(TensorDataset(X_train_seq, C_train_seq, session_train_seq, Y_train_seq), batch_size=self.batch_size_seq, shuffle=True)
        vali_loader_seq = DataLoader(TensorDataset(X_vali_seq, C_vali_seq, session_vali_seq, Y_vali_seq), batch_size=self.batch_size_seq)
        test_loader_seq = DataLoader(TensorDataset(X_test_seq, C_test_seq, session_test_seq, Y_test_seq), batch_size=self.batch_size_seq)
        
        # Calcola MSE del propagatore della dvae sul test_set, per vedere se le performance sono simili a quelle del Markov predictor (MLP)
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for x, c, s, y in test_loader_m:
                x = x.to(self.device)
                c = c.to(self.device)
                y = y.to(self.device)
                s = s.to(self.device).long()
                #print(s.shape)
                s = self.dvae.session_embeddings(s)
                #print(s.shape)
                #print(c.shape)
                pred, _ = self.dvae.generation_z(x, c, s)
#                 pred_mean, pred_cov = dvae.generation_z(x, c)
#                 pred = dvae.reparameterization_cov(pred_mean, pred_cov)
                loss = self.criterion(pred, y)
                total_loss += loss.item() * len(x)
                total_samples += x.shape[0]
        MSE_prop = total_loss / total_samples
        
        return train_loader_m, vali_loader_m, test_loader_m, train_loader_seq, vali_loader_seq, test_loader_seq, MSE_prop

    
    def train(self):
        
        train_loader_m, vali_loader_m, test_loader_m, train_loader_seq, vali_loader_seq, test_loader_seq, _ = self.create_loaders()
        optimizer_m = torch.optim.Adam(self.model_m.parameters(), lr=self.learning_rate)
        optimizer_seq = torch.optim.Adam(self.model_seq.parameters(), lr=self.learning_rate)
    
        ################# Markov training ###################
    
        train_curve = []
        vali_curve = []
        MSE_curve = []
        best_epoch_curve = []
        best_loss = float('inf')
        for epoch in range(self.params["num_epochs"]):
            train_loss = train_epoch(self.model_m, train_loader_m, optimizer_m, self.criterion, self.device)
            vali_loss = evaluate(self.model_m, vali_loader_m, self.criterion, self.device)
            train_curve.append(train_loss)
            vali_curve.append(vali_loss)

            if vali_loss < best_loss:
                best_loss = vali_loss
                MSE_test = evaluate(self.model_m, test_loader_m, self.criterion, self.device)
                MSE_curve.append(MSE_test)
                best_epoch_curve.append(epoch)
                # Save the model weights
                torch.save({
                    'params': self.params,
                    'epoch': epoch,
                    'train_curve': train_curve,
                    'vali_curve': vali_curve,
                    'MSE_curve': MSE_curve,
                    'best_epoch_curve': best_epoch_curve,
                    'model_state_dict': self.model_m.state_dict(),
                    'optimizer_state_dict': optimizer_m.state_dict(),
                    'loss': best_loss,
                    'MSE': MSE_test,
                }, self.saved_dir + "/Markov_dict_" + self.type + self.date)

            print(f"[Markov] Epoch {epoch+1}: train={train_loss:.4f}, val={vali_loss:.4f}, MSE_test={MSE_test:.4f}")
        
        ################# Non-Markov training ###################
        
        train_curve = []
        vali_curve = []
        MSE_curve = []
        best_epoch_curve = []
        best_loss = float('inf')
        for epoch in range(self.params["num_epochs"]):
            train_loss = train_epoch(self.model_seq, train_loader_seq, optimizer_seq, self.criterion, self.device)
            vali_loss = evaluate(self.model_seq, vali_loader_seq, self.criterion, self.device)
            train_curve.append(train_loss)
            vali_curve.append(vali_loss)

            if vali_loss < best_loss:
                best_loss = vali_loss
                MSE_test = evaluate(self.model_seq, test_loader_seq, self.criterion, self.device)
                MSE_curve.append(MSE_test)
                best_epoch_curve.append(epoch)
                # Save the model weights
                torch.save({
                    'params': self.params,
                    'epoch': epoch,
                    'train_curve': train_curve,
                    'vali_curve': vali_curve,
                    'MSE_curve': MSE_curve,
                    'best_epoch_curve': best_epoch_curve,
                    'model_state_dict': self.model_seq.state_dict(),
                    'optimizer_state_dict': optimizer_seq.state_dict(),
                    'loss': best_loss,
                    'MSE': MSE_test,
                }, self.saved_dir + "/Non_Markov_dict_" + self.type + self.date)

            print(f"[Non-Markov] Epoch {epoch+1}: train={train_loss:.4f}, val={vali_loss:.4f}, MSE_test={MSE_test:.4f}")
        
        
        # ==== Final comparison ====
        test_loss_markov = evaluate(self.model_m, test_loader_m, self.criterion, self.device)
        test_loss_lstm = evaluate(self.model_seq, test_loader_seq, self.criterion, self.device)

        print(f"\n=== Comparison ===")
        print(f"Markov MLP test MSE: {test_loss_markov:.4f}")
        print(f"LSTM (non-Markov) test MSE: {test_loss_lstm:.4f}")
        print(f"percentage of difference: {abs(test_loss_markov - test_loss_lstm) / test_loss_markov}")
    

    # Function to load model weights
    def load(self, saved_dict):
        if os.path.exists(self.saved_dir):
            checkpoint = torch.load(self.saved_dir + saved_dict, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            MSE = checkpoint["MSE"]
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"Loaded model from epoch {epoch} with loss {loss:.4f} and MSE: {MSE:.4f}%")
        else:
            print("No saved model found")
        return self.model
