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
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from .dataset.dataloader import load_test_set
from .utils import inference_with_trials#, sess_sublist
from .dataset import one_hot_cont, process_data, data_split, MUA_dataset
from .learning_algo_dir import LearningAlgorithm_dir


class Dir_Detector(nn.Module):
    def __init__(self, input_dim=96, encoded_dim = 2, hidden_dim=128, num_layers=2, dropout=0.2, bidirectional=False):
        super(Dir_Detector, self).__init__()
        
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, encoded_dim),
#             nn.ReLU(),
#         )
        
        self.lstm = nn.LSTM(
            input_size=input_dim,#encoded_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # (batch, time, features)
            bidirectional=bidirectional
        )
        
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Output: (batch, 256, hidden_dim) → use the last time step
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
            # Non usiamo Sigmoid qui perché BCEWithLogitsLoss è più stabile
        )
        
    def forward(self, x):
        
        # x shape: (batch_size, 256, 96)
        #enc = self.encoder(x)
        out, (hn, cn) = self.lstm(x)  # out shape: (batch, 256, hidden_dim)
        last_hidden = out[:, -1, :]   # take the last time step's output
        logits = self.fc(last_hidden)  # shape: (batch_size, 1)
        return logits.squeeze(1)       # shape: (batch_size,)


# +
class TrialDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(x)
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
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item() * len(x)
            total_samples += x.shape[0]
    return total_loss / total_samples
    

    

class Learning_dir_detector():
    def __init__(self, params, saved_dir=None):
        # Load config parser
        self.params = params
        # Hyperparameters
        self.net = self.params["net"]
        self.type = self.params["type"]
        self.batch_size = self.params['batch_size']
        self.num_epochs = self.params['num_epochs']
        self.n_trials = self.params['n_trials']
        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%M")
        self.weights_dir = self.params['weights_dir']
        self.device = self.params['device']
        if saved_dir is None:
            self.saved_dir = params["saved_dir"]
        else:
            self.saved_dir = saved_dir
            
        dmm_params = {
            "cfg": self.saved_dir + "/config.ini",
            "device": self.device,
            "saved_dict": self.saved_dir,
        }

        learning_algo = LearningAlgorithm_dir(params=dmm_params)
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
        self.x_dim = self.dmm.x_dim
        self.input_dim = self.x_dim if self.type == "x" else self.z_dim
        
        self.model = Dir_Detector(
                            input_dim=self.input_dim,
                            encoded_dim = self.params['encoded_dim'],
                            hidden_dim=self.params['hidden_dim'],
                            num_layers=self.params['num_layers'],
                            dropout=self.params['dropout'],
                            bidirectional=self.params['bidirectional'],
                        ).to(self.device)  # Use same device as data
        
        
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
#             if self.net == "sess" or self.net == "sess_u":
#                 session_train = loaded_file["train_session"]
#                 session_vali = loaded_file["vali_session"]
#                 session_test = loaded_file["test_session"]
                
#                 session_list_train, mask_sess_train = sess_sublist(session_train, session_spec=self.train_sess)
#                 session_list_vali, mask_sess_vali = sess_sublist(session_vali, session_spec=self.train_sess)
#                 session_list_test, mask_sess_test = sess_sublist(session_test, session_spec=self.test_sess)
#             else:
#                 mask_sess_train = np.ones(train_set.shape[0], dtype=bool)
#                 mask_sess_vali = np.ones(vali_set.shape[0], dtype=bool)
#                 mask_sess_test = np.ones(test_set.shape[0], dtype=bool)
            
        cont_train = one_hot_cont(train_SSD, train_direction, self.tau)
        cont_vali = one_hot_cont(vali_SSD, vali_direction, self.tau)
        cont_test = one_hot_cont(test_SSD, test_direction, self.tau)
                
#         train_set = train_set[mask_sess_train]
#         vali_set = vali_set[mask_sess_vali]
#         test_set = test_set[mask_sess_test]
        
#         cont_train = cont_train[mask_sess_train]
#         cont_vali = cont_vali[mask_sess_vali]
#         cont_test = cont_test[mask_sess_test]
        
#         train_direction = train_direction[mask_sess_train]
#         vali_direction = vali_direction[mask_sess_vali]
#         test_direction = test_direction[mask_sess_test]
            
        n_train, steps, features = train_set.shape
        n_vali, _, c_dim = cont_vali.shape
        n_test = test_set.shape[0]
        
        if self.type == "x":
            train_data = train_set  # n_train x steps x features 
            vali_data = vali_set
            test_data = test_set

        
        ####################### z_dataset #######################
        
        if self.type == "z":
            
            train_data = inference_with_trials(self.dmm, train_set, cont_train, self.n_trials, self.input_dim, self.device, chunk_size=128)
            vali_data = inference_with_trials(self.dmm, vali_set, cont_vali, self.n_trials, self.input_dim, self.device, chunk_size=128)
            test_data = inference_with_trials(self.dmm, test_set, cont_test, self.n_trials, self.input_dim, self.device, chunk_size=128)

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

        ################## Datasets & Loaders ######################
        
        train_dataset = TrialDataset(train_data, train_direction)
        vali_dataset = TrialDataset(vali_data, vali_direction)
        test_dataset = TrialDataset(test_data, test_direction)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        vali_loader = DataLoader(vali_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        return train_loader, vali_loader, test_loader
        
    
    def train(self):
        
        train_dataloader, vali_dataloader, test_dataloader = self.create_loaders()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])
    
        self.model.train()

        train_curve = []
        vali_curve = []
        accuracy_curve = []
        best_epoch_curve = []
        best_loss = float('inf')
        for epoch in range(self.num_epochs):
            train_loss = train_epoch(self.model, train_dataloader, optimizer, criterion, self.device)
            vali_loss = evaluate(self.model, vali_dataloader, criterion, self.device)
            train_curve.append(train_loss)
            vali_curve.append(vali_loss)
        
            if vali_loss < best_loss:
                best_loss = vali_loss
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_data, batch_labels in test_dataloader:
                        batch_data = batch_data.float().to(self.device)
                        batch_labels = batch_labels.float().to(self.device)
                        outputs = self.model(batch_data)
                        prob = torch.sigmoid(outputs)
                        predictions = (prob > 0.5).float()

                        # Calculate accuracy
                        correct += (predictions == batch_labels).sum().item()
                        total += batch_labels.numel()

                    accuracy = correct / total
                accuracy_curve.append(accuracy)
                best_epoch_curve.append(epoch)
                # Save the model weights
                torch.save({
                    'params': self.params,
                    'epoch': epoch,
                    'train_curve': train_curve,
                    'vali_curve': vali_curve,
                    'accuracy_curve': accuracy_curve,
                    'best_epoch_curve': best_epoch_curve,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'accuracy': accuracy,
                }, self.saved_dir + "/dir_dict_" + self.type)

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train_Loss: {train_loss:.4f}, Vali_Loss: {vali_loss:.4f}, Accuracy: {accuracy:.4f}')


    # Function to load model weights
    def load(self, saved_dict):
        if os.path.exists(self.saved_dir):
            checkpoint = torch.load(self.saved_dir + saved_dict, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            accuracy = checkpoint["accuracy"]
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"Loaded model from epoch {epoch} with loss {loss:.4f} and accuracy: {accuracy*100:.4f}%")
        else:
            print("No saved model found")
        return self.model
