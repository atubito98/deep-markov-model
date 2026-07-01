#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
import numpy as np
import torch
from torch import nn
from torch import optim
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from .learning_algo_dir import LearningAlgorithm_dir
from .utils import inference_with_trials
from .dataset import one_hot_cont


# ==== Markovian MLP ====
class MarkovMLP(nn.Module):
    def __init__(self, x_dim, c_dim, hidden_dim=128):
        super().__init__()
        self.x_dim = x_dim
        self.net = nn.Sequential(
            nn.Linear(x_dim + c_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim)
        )

    def forward(self, x, c):
        inp = torch.cat([x, c], dim=-1)
        return self.net(inp)
    
    
# ==== LSTM (non-Markovian) model ====
class NonMarkovLSTM(nn.Module):
    def __init__(self, x_dim, c_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        self.x_dim = x_dim
        self.lstm = nn.LSTM(x_dim + c_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, x_dim)

    def forward(self, x, c):
        inp = torch.cat([x, c], dim=-1)  # concatenazione interna
        out, _ = self.lstm(inp)
        return self.fc(out)  # predizioni per ogni step


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_samples = 0
    for x, c, y in loader:
        x = x.to(device)
        c = c.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(x, c)
        loss = criterion(pred, y)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
        total_samples += x.shape[0]
    return total_loss / total_samples

def evaluate(model, loader, criterion, device):
    model.eval()
#     total_loss = 0
    total_loss = np.zeros(model.x_dim)
    total_samples = 0
    delta_pred_list = []
    with torch.no_grad():
        for x, c, y in loader:
            x = x.to(device)
            c = c.to(device)
            y = y.to(device)
            pred = model(x, c)
            loss = criterion(pred, y)
            loss = loss.cpu().detach().numpy()
            loss = loss.reshape(-1, x.shape[-1]).mean(0)
            pred = pred.cpu().detach().numpy()
            x = x.cpu().detach().numpy()
            delta_pred = pred-x
            delta_pred_list.extend(delta_pred)
            total_loss += loss * len(x)
            total_samples += x.shape[0]
        delta_pred_arr = np.array(delta_pred_list)
    return total_loss / total_samples, delta_pred_arr

def build_seq_data(X, C):
    X_in = X[:, :-1, :]   # x_{1:t}
    C_in = C[:, 1:, :]   # c_{1:t}
    Y_out = X[:, 1:, :]   # x_{2:t+1}
    return X_in, C_in, Y_out



class Learning_markov_detector():
    def __init__(self, params, device=None, saved_dir=None):
        # Load config parser
        self.params = params
        # Hyperparameters
        self.type = self.params["type"]
        self.batch_size_seq = self.params['batch_size_seq']
        self.batch_size_m = self.params['batch_size_m']
        self.learning_rate = self.params['learning_rate']
        self.num_epochs = self.params['num_epochs']
        self.n_trials = self.params['n_trials']
        self.chunk_size = self.params['chunk_size']
        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%M")
        self.hidden_dim = self.params['hidden_dim']
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
        self.c_dim = self.dmm.c_dim
        self.z_dim = self.dmm.z_dim
        self.x_dim = self.dmm.x_dim
        self.input_dim = self.x_dim if self.type == "x" else self.z_dim
        
        self.model_m = MarkovMLP(self.input_dim, self.c_dim, self.hidden_dim).to(self.device)  # Use same device as data
        self.model_seq = NonMarkovLSTM(self.input_dim, self.c_dim, self.hidden_dim).to(self.device)  
        self.criterion = nn.MSELoss(reduction='none')
        
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
            
        cont_train = one_hot_cont(train_SSD, train_direction, self.tau)
        cont_vali = one_hot_cont(vali_SSD, vali_direction, self.tau)
        cont_test = one_hot_cont(test_SSD, test_direction, self.tau)
        
        n_train, steps, features = train_set.shape
        n_vali, _, _ = cont_vali.shape
        n_test = test_set.shape[0]
        
        if self.type == "x":
            train_data = train_set  # n_train x steps x features 
            vali_data = vali_set
            test_data = test_set

        
        ####################### z_dataset #######################
        
        if self.type == "z":
            
            train_data = inference_with_trials(self.dmm, train_set, cont_train, self.n_trials, self.device, chunk_size=self.chunk_size)
            vali_data = inference_with_trials(self.dmm, vali_set, cont_vali, self.n_trials, self.device, chunk_size=self.chunk_size)
            test_data = inference_with_trials(self.dmm, test_set, cont_test, self.n_trials, self.device, chunk_size=self.chunk_size)
           
        ################## PCA ######################
        
        if self.type == "PCA":

            X_train = train_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
            X_vali = vali_set.reshape(-1, features)  # shape = (trials * time_steps, 96)
            X_test = test_set.reshape(-1, features)  # shape = (trials * time_steps, 96)

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
        
        self.Y_test_m = Y_test_m
        self.X_test_m = X_test_m
        
        train_loader_m = DataLoader(TensorDataset(X_train_m, C_train_m, Y_train_m), batch_size=self.batch_size_m, shuffle=True)
        vali_loader_m = DataLoader(TensorDataset(X_vali_m, C_vali_m, Y_vali_m), batch_size=self.batch_size_m)
        test_loader_m = DataLoader(TensorDataset(X_test_m, C_test_m, Y_test_m), batch_size=self.batch_size_m)
        
        train_loader_seq = DataLoader(TensorDataset(X_train_seq, C_train_seq, Y_train_seq), batch_size=self.batch_size_seq, shuffle=True)
        vali_loader_seq = DataLoader(TensorDataset(X_vali_seq, C_vali_seq, Y_vali_seq), batch_size=self.batch_size_seq)
        test_loader_seq = DataLoader(TensorDataset(X_test_seq, C_test_seq, Y_test_seq), batch_size=self.batch_size_seq)
        
        return train_loader_m, vali_loader_m, test_loader_m, train_loader_seq, vali_loader_seq, test_loader_seq

    
    def train(self):
        
        train_loader_m, vali_loader_m, test_loader_m, train_loader_seq, vali_loader_seq, test_loader_seq = self.create_loaders()
        optimizer_m = torch.optim.Adam(self.model_m.parameters(), lr=self.learning_rate)
        optimizer_seq = torch.optim.Adam(self.model_seq.parameters(), lr=self.learning_rate)
        
        delta_z = self.Y_test_m - self.X_test_m
        delta_z = delta_z.cpu().detach().numpy()
        test_var = np.var(delta_z, axis=0)
        
        pca_dz = PCA(n_components=self.z_dim)
        delta_pc = pca_dz.fit_transform(delta_z)  # PCA center data internally, so there is no need to do it manually before
        pc_std = delta_pc.std(axis=0)
        delta_norm = delta_pc/pc_std
    
        ################# Markov training ###################
        markov_file = "Markov_dict_" + self.type
        train_curve = []
        vali_curve = []
        mean_NMSE_curve = []
        NMSE_mean_curve = [] 
        MSE_pc_curve = []
        best_epoch_curve = []
        best_loss = float('inf')
        for epoch in range(self.params["num_epochs"]):
            train_loss = train_epoch(self.model_m, train_loader_m, optimizer_m, self.criterion, self.device)
            vali_loss_tensor, delta_pred_arr = evaluate(self.model_m, vali_loader_m, self.criterion, self.device)
            vali_loss = vali_loss_tensor.mean()
            train_curve.append(train_loss)
            vali_curve.append(vali_loss)

            if vali_loss < best_loss:
                best_loss = vali_loss
                MSE_test_tensor, delta_pred_arr = evaluate(self.model_m, test_loader_m, self.criterion, self.device)
                delta_pred_norm = pca_dz.transform(delta_pred_arr)/pc_std
                MSE_pc = ((delta_norm - delta_pred_norm)**2).mean()
                MSE_pc_curve.append(MSE_pc)
                mean_NMSE_test = (MSE_test_tensor/test_var).mean()
                NMSE_mean_test = MSE_test_tensor.mean()/test_var.mean()
                mean_NMSE_curve.append(mean_NMSE_test)
                NMSE_mean_curve.append(NMSE_mean_test)
                best_epoch_curve.append(epoch)
                # Save the model weights
                torch.save({
                    'params': self.params,
                    'epoch': epoch,
                    'train_curve': train_curve,
                    'vali_curve': vali_curve,
                    'mean_NMSE_curve': mean_NMSE_curve,
                    'NMSE_mean_curve': NMSE_mean_curve,
                    'MSE_pc_curve': MSE_pc_curve,
                    'best_epoch_curve': best_epoch_curve,
                    'model_state_dict': self.model_m.state_dict(),
                    'optimizer_state_dict': optimizer_m.state_dict(),
                    'loss': best_loss,
                    'test_var': test_var,
                    'MSE_pc': MSE_pc,
                    'NMSE_mean': NMSE_mean_test,
                    'mean_NMSE': mean_NMSE_test,
                }, self.saved_dir / markov_file)

            print(f"[Markov] Epoch {epoch+1}: train={train_loss:.4f}, val={vali_loss:.4f}, NMSE_mean_test={NMSE_mean_test:.4f}, mean_NMSE_test={mean_NMSE_test:.4f}, MSE_pc={MSE_pc:.4f}")
        
        ################# Non-Markov training ###################
        non_markov_file = "Non_Markov_dict_" + self.type
        train_curve = []
        vali_curve = []
        mean_NMSE_curve = []
        NMSE_mean_curve = [] 
        MSE_pc_curve = []
        best_epoch_curve = []
        best_loss = float('inf')
        for epoch in range(self.params["num_epochs"]):
            train_loss = train_epoch(self.model_seq, train_loader_seq, optimizer_seq, self.criterion, self.device)
            vali_loss_tensor, delta_pred_arr = evaluate(self.model_seq, vali_loader_seq, self.criterion, self.device)
            vali_loss = vali_loss_tensor.mean()
            train_curve.append(train_loss)
            vali_curve.append(vali_loss)

            if vali_loss < best_loss:
                best_loss = vali_loss
                MSE_test_tensor, delta_pred_arr = evaluate(self.model_seq, test_loader_seq, self.criterion, self.device)
                delta_pred_arr = delta_pred_arr.reshape(-1, self.z_dim)
                delta_pred_norm = pca_dz.transform(delta_pred_arr)/pc_std
                MSE_pc = ((delta_norm - delta_pred_norm)**2).mean()
                MSE_pc_curve.append(MSE_pc)
                mean_NMSE_test = (MSE_test_tensor/test_var).mean()
                NMSE_mean_test = MSE_test_tensor.mean()/test_var.mean()
                mean_NMSE_curve.append(mean_NMSE_test)
                NMSE_mean_curve.append(NMSE_mean_test)
                best_epoch_curve.append(epoch)
                # Save the model weights
                torch.save({
                    'params': self.params,
                    'epoch': epoch,
                    'train_curve': train_curve,
                    'vali_curve': vali_curve,
                    'mean_NMSE_curve': mean_NMSE_curve,
                    'NMSE_mean_curve': NMSE_mean_curve,
                    'MSE_pc_curve': MSE_pc_curve,
                    'best_epoch_curve': best_epoch_curve,
                    'model_state_dict': self.model_seq.state_dict(),
                    'optimizer_state_dict': optimizer_seq.state_dict(),
                    'loss': best_loss,
                    'MSE_pc': MSE_pc,
                    'NMSE_mean': NMSE_mean_test,
                    'mean_NMSE': mean_NMSE_test,
                    'test_var': test_var,
                }, self.saved_dir / non_markov_file)

            print(f"[Non-Markov] Epoch {epoch+1}: train={train_loss:.4f}, val={vali_loss:.4f}, NMSE_mean_test={NMSE_mean_test:.4f}, mean_NMSE_test={mean_NMSE_test:.4f}, MSE_pc={MSE_pc:.4f}")
        
        # ==== Final comparison ====
        test_loss_Markov, delta_pred_arr = evaluate(self.model_m, test_loader_m, self.criterion, self.device)
        delta_pred_norm = pca_dz.transform(delta_pred_arr)/pc_std
        Markov_MSE_pc = ((delta_norm - delta_pred_norm)**2).mean()
        test_loss_NonMarkov, delta_pred_arr = evaluate(self.model_seq, test_loader_seq, self.criterion, self.device)
        delta_pred_arr = delta_pred_arr.reshape(-1, self.z_dim)
        delta_pred_norm = pca_dz.transform(delta_pred_arr)/pc_std
        NonMarkov_MSE_pc = ((delta_norm - delta_pred_norm)**2).mean()

        NMSE_Markov = test_loss_Markov.mean()/test_var.mean()
        NMSE_NonMarkov = test_loss_NonMarkov.mean()/test_var.mean()
        
        print(f"\n=== Comparison ===")
        print(f"Markov MSE_pc: {Markov_MSE_pc:.4f}")
        print(f"LSTM (non-Markov) MSE_pc: {NonMarkov_MSE_pc:.4f}")
        print(f"Markov MLP test NMSE: {NMSE_Markov:.4f}")
        print(f"LSTM (non-Markov) test NMSE: {NMSE_NonMarkov:.4f}")
        print(f"fraction NonMarkov/Markov: {NMSE_NonMarkov / NMSE_Markov}")
    

    # Function to load model weights
    def load(self, saved_dict, markov=True):
        if os.path.exists(self.saved_dir):
            checkpoint = torch.load(self.saved_dir / saved_dict, weights_only=False, map_location=self.device)
            if markov:
                self.model_m.load_state_dict(checkpoint['model_state_dict'])
                model = self.model_m
            else: 
                self.model_seq.load_state_dict(checkpoint['model_state_dict'])
                model = self.model_seq
            NMSE = checkpoint["NMSE"]
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"Loaded model from epoch {epoch} with loss {loss:.4f} andNMSE: {NMSE:.4f}%")
        else:
            print("No saved model found")
        model.eval()
        return model
