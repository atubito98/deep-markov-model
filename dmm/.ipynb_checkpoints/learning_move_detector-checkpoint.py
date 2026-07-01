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
from .utils import inference_with_trials#, sess_sublist
from .dataset import one_hot_cont, process_data, data_split, MUA_dataset
from .learning_algo_dir import LearningAlgorithm_dir


class moveDataset(Dataset):
    def __init__(self, trajectories, move_labels):
        self.trajectories = torch.from_numpy(trajectories).float()
        self.move_labels = torch.from_numpy(move_labels).float()

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx], self.move_labels[idx]
    
    
class MovePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0, bidirectional=True):
        super(MovePredictor, self).__init__()
        
        # Backbone: un LSTM per processare la sequenza
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional) # Importante!
        
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Testa di Classificazione: predice la probabilità di movimento
        self.classification_head = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
            # Non usiamo Sigmoid qui perché BCEWithLogitsLoss è più stabile
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # outputs shape: (batch, seq_len, hidden_size)
        # hidden shape: (num_layers, batch, hidden_size)
        outputs, (hidden, cell) = self.lstm(x)
        
        # Usiamo l'ultimo stato nascosto dell'ultimo layer come embedding
        # L'ultimo stato nascosto riassume l'intera sequenza
        final_hidden_state = outputs[:, -1, :] # Shape: (batch, hidden_size)

        # Passa l'embedding a entrambe le teste
        prob_move_logits = self.classification_head(final_hidden_state)
        
        return prob_move_logits.squeeze(1)
    
    
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_samples = 0
    for batch_trajectories, batch_move_labels in loader:
        optimizer.zero_grad()
        batch_trajectories = batch_trajectories.to(device)
        batch_move_labels = batch_move_labels.to(device)
        prob_move_logits = model(batch_trajectories)
        loss = criterion(prob_move_logits, batch_move_labels)
        # Backward pass e ottimizzazione
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_move_labels)
        total_samples += batch_move_labels.shape[0]
    avg_total_loss = total_loss / total_samples
    return avg_total_loss


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    for batch_trajectories, batch_move_labels in loader:

        batch_trajectories = batch_trajectories.to(device)
        batch_move_labels = batch_move_labels.to(device)
        prob_move_logits = model(batch_trajectories)
        loss = criterion(prob_move_logits, batch_move_labels)
        total_loss += loss.item() * len(batch_move_labels)
        total_samples += batch_move_labels.shape[0]
    avg_total_loss = total_loss / total_samples
    return avg_total_loss



class Learning_move_detector():
    def __init__(self, params, saved_dir=None):
        # Load config parser
        self.params = params
        # Hyperparameters
        self.net = self.params["net"]
        self.type = self.params["type"]
        self.batch_size = self.params['batch_size']
        self.num_epochs = self.params['num_epochs']
        self.n_trials = self.params['n_trials']
        self.chunk_size = self.params['chunk_size']
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
        self.dmm.load_state_dict(torch.load(self.saved_dir + self.weights_dir, map_location='cpu'))
        self.dmm.eval()
        self.tau = self.dmm.tau
        self.z_dim = self.dmm.z_dim
        self.x_dim = self.dmm.x_dim
        self.input_dim = self.x_dim if self.type == "x" else self.z_dim
        
        self.model = MovePredictor(
                            input_dim = self.input_dim,
                            hidden_dim=self.params['hidden_dim'],
                            num_layers=self.params['num_layers'],
                            dropout = self.params['dropout'],
                            bidirectional = self.params['bidirectional'] 
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
            train_RT = loaded_file["train_RT"]
            vali_RT = loaded_file["vali_RT"]
            test_RT = loaded_file["test_RT"]
            
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
            
            train_data = inference_with_trials(self.dmm, train_set, cont_train, self.n_trials, self.input_dim, self.device, chunk_size=self.chunk_size)
            vali_data = inference_with_trials(self.dmm, vali_set, cont_vali, self.n_trials, self.input_dim, self.device, chunk_size=self.chunk_size)
            test_data = inference_with_trials(self.dmm, test_set, cont_test, self.n_trials, self.input_dim, self.device, chunk_size=self.chunk_size)
         
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
        
        # seleziono solo cs e ws trials
        
        mask_train = train_SSD > 0
        mask_vali = vali_SSD > 0
        mask_test = test_SSD > 0
        
        train_data = train_data[mask_train]
        vali_data = vali_data[mask_vali]
        test_data = test_data[mask_test]
        
        train_RT = train_RT[mask_train]
        vali_RT = vali_RT[mask_vali]
        test_RT = test_RT[mask_test]
        
        train_SSD = train_SSD[mask_train]
        vali_SSD = vali_SSD[mask_vali]
        test_SSD = test_SSD[mask_test]
        
        # creo le labels
        
        train_move = (train_RT > 0)   # 1 per nostop, 0 per correct-stop
        vali_move = (vali_RT > 0)
        test_move = (test_RT > 0)
        
        train_dataset = moveDataset(train_data, train_move)
        vali_dataset = moveDataset(vali_data, vali_move)
        test_dataset = moveDataset(test_data, test_move)
        
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        vali_dataloader = DataLoader(vali_dataset, batch_size=self.batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        return train_dataloader, vali_dataloader, test_dataloader
        
    def train(self):
        
        train_dataloader, vali_dataloader, test_dataloader = self.create_loaders()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])
        criterion = nn.BCEWithLogitsLoss()
    
        ################# Markov training ###################
        threshold = 0.5
        train_loss_curve = []
        vali_loss_curve = []
        accuracy_curve = []
        best_epoch_curve = []
        best_loss = float('inf')
        for epoch in range(self.params["num_epochs"]):
            loss_train = train_epoch(self.model, train_dataloader, optimizer, criterion, self.device)
            loss_vali = evaluate(self.model, vali_dataloader, criterion, self.device)
            train_loss_curve.append(loss_train)
            vali_loss_curve.append(loss_vali)

            if loss_vali < best_loss:
                best_loss = loss_vali
                self.model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_trajectories, batch_move_labels in test_dataloader:

                        batch_trajectories = batch_trajectories.to(self.device)
                        batch_move_labels = batch_move_labels.to(self.device)
                        prob_move_logits = self.model(batch_trajectories)
                        prob_move = torch.sigmoid(prob_move_logits)
                        predicted_labels = (prob_move > threshold).float()
                        correct += (predicted_labels == batch_move_labels).sum().item()
                        total += batch_move_labels.numel()

                    accuracy = correct / total
                accuracy_curve.append(accuracy)
                best_epoch_curve.append(epoch)
                # Save the model weights
                torch.save({
                    'params': self.params,
                    'epoch': epoch,
                    'train_loss_curve': train_loss_curve,
                    'vali_loss_curve': vali_loss_curve,
                    'accuracy_curve': accuracy_curve,
                    'best_epoch_curve': best_epoch_curve,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    #'MSE': MSE_test,
                }, self.saved_dir + "/move_dict_" + self.type)

            print(f"Epoch {epoch+1}: train={loss_train:.4f}, val={loss_vali:.4f}, test_accuracy={accuracy:.4f}")
        


    # Function to load model weights
    def load(self, saved_dict):
        if os.path.exists(self.saved_dir):
            checkpoint = torch.load(self.saved_dir + saved_dict)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            accuracy_curve = checkpoint['accuracy_curve']
            print(f"Loaded model from epoch {epoch} with loss {loss:.4f} and accuracy {accuracy_curve[-1]:.4f}")
        else:
            print("No saved model found")
        return self.model
