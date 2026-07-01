#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# %%


import scipy


# %%


from scipy import stats


# %%


from sklearn.model_selection import train_test_split, KFold


# %%


#######
#PARAMS
#######
#TODO:move to config file
ramp_param={'sig':0.05, 'dt': 5*10**-3 , 't_go':56, 't_max':200, 'L':256, "dt_data" : 5*10**-3}
data_path = '/raid/home/tubitoal/MUA/data/'


# %%


def deterministic_ramp_generator(param,rt):
    mu = 1.0 / rt
    sig=param["sig"]
    dt =param["dt"] 
    dt_data= param["dt_data"] 
    t_go = param["t_go"] * dt_data
    t_max = param["t_max"] * dt_data

    n_t_max = int(np.round( (t_max + t_go) / dt) )
    #print("ntmax ",n_t_max)
    n_t_go = int(np.round(t_go / dt))
    #print("n_t_go ",n_t_go)

    y = np.zeros((n_t_max,))
    y[n_t_go:] = mu * dt
    y[:] = y.cumsum()
    #y += sig * np.random.randn(n_t_max)

    return y


# %%
def single_corr_plot(RT_cn_ordered_filt, RT_pred_filt, corr, llim, rlim):

    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(RT_cn_ordered_filt[:, np.newaxis], RT_pred_filt)
    # Predict the labels using the model
    retta_pred = model.predict(np.linspace(llim, rlim, num=150)[:, np.newaxis])

    #corr = np.corrcoef(RT_cn_ordered_filt, RT_pred_filt)[0, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))
    #ax1.plot(RT_cn_ordered_filt, label = 'true_RT')
    ax1.plot(np.linspace(llim, rlim, num=150), retta_pred, label = 'linear fit')
    #ax1.scatter(np.arange(samp_filt, dtype=int), RT_pred_filt)
    ax1.scatter(RT_cn_ordered_filt, RT_pred_filt, label = 'predicted from GO+100ms')
    ax1.set_xlabel("RT_true")
    ax1.set_ylabel("RT_pred")
    ax1.set_xlim(llim, rlim)
    ax1.set_ylim(llim, rlim)
    ax1.legend()
    ax1.set_title(f'correlation = {corr:.2f}')

    #fig, ax = plt.subplots(figsize = (7, 6))
    ax2.hist(RT_pred_filt - RT_cn_ordered_filt, bins=30, color='skyblue', edgecolor='black')
    ax2.set_title("Difference between true and predicted RT")


# %%
def corr_RT_maxMUA(data, RT, SSD):
    data = data[SSD==0]
    RT = RT[SSD==0]
    RT_min = RT.min()
    print(RT_min)
    start = RT_min + 56 
    data = data[:, start:]
    MUA_mean = data.mean(2)
    print(MUA_mean.shape)
    RT_estimate = np.argmax(MUA_mean, axis=1) + RT_min
    mask = (RT_estimate<195) & (RT_estimate>RT_min+5)
    corr = np.corrcoef(RT[mask], RT_estimate[mask])[0, 1]
    single_corr_plot(RT[mask], RT_estimate[mask], corr, llim=0, rlim=256)
    #print(f"the correlation between RT and the maximum of the mean MUA is: {corr}")

# %%


def load_session_data(data_path, selection,session, DEMEANED=False, NORMALIZED=True):
   

    with np.load(data_path) as andreaData:

        X_trial=np.array(andreaData["X_trial"]) 
        y_stop=np.array(andreaData["y_stop"])
        y_reward=np.array(andreaData["y_reward"])
        y_direction=np.array(andreaData["y_direction"])
        y_RT=np.array(andreaData["y_RT"])
        y_SSD=np.array(andreaData["y_SSD"])


    #selec no stop
    if selection == "all":
        data=X_trial
        RT=y_RT
        SSD=y_SSD
        direction=y_direction
    else:
        sel=eval(selection)
        data=X_trial[sel,:,:]
        RT=y_RT[sel]
        SSD=y_SSD[sel]
        direction=y_direction[sel]
    session =session*np.ones(data.shape[0])
    subject = np.array([data_path.split("/")[-1][0] == "P"]*data.shape[0])  # 1 is piero, 0 Cornelio
    direction_binary = direction == "right"  # 1 right, 0 left
    
   # pca.fit_transform(data.reshape([-1,data.shape[-1]]))
    if DEMEANED:
        data=data-np.expand_dims( data.mean(axis=-1),axis=-1)
    if NORMALIZED:
        #data=scipy.stats.zscore(data, axis=2, ddof=0, nan_policy='propagate')
        
#         data_preGO = data[:, :56, :]
#         mean = data_preGO.mean(axis=(0,1), keepdims=True)
#         std = data_preGO.std(axis=(0,1), keepdims=True)
#         data = (data - mean) / std
        
        # data_trasp = np.transpose(data, (2, 1, 0))
        # data_norm=scipy.stats.zscore(data_trasp, axis=1, ddof=0, nan_policy='propagate')
        # data = np.transpose(data_norm, (2, 1, 0))
        
        mean = data.mean(axis=(0,1), keepdims=True)
        std = data.std(axis=(0,1), keepdims=True)
        data = (data - mean) / std
        
    return data,RT,SSD,direction_binary,session,subject


# %%


def load_test_set(path_array,selection):
    for i,data_path in enumerate(path_array):
        
        if i==0:
            data,RT,SSD,direction,session,subject = load_session_data(data_path, selection,i,NORMALIZED=True)
            print(len(RT))
        
        else:
            data_tmp,RT_tmp,SSD_tmp,direction_tmp,session_tmp,subject_tmp = load_session_data(data_path, selection, i,NORMALIZED=True)
        
            data=np.concatenate((data,data_tmp),axis=0)
            RT=np.concatenate((RT,RT_tmp),axis=0)
            SSD=np.concatenate((SSD,SSD_tmp),axis=0)
            direction=np.concatenate((direction,direction_tmp),axis=0)
            session=np.concatenate((session,session_tmp),axis=0)
            subject=np.concatenate((subject,subject_tmp),axis=0)
            
            
   
    test_size=data.shape[0]
    print("test size :" ,test_size)
    print("mean test RT  {:3.2f} s".format(RT.mean()*0.005))
    
    return data, RT, SSD,direction,session,subject


# %%
def load_set(path_array,selection, random_state=0):
    
    for i,data_path in enumerate(path_array):
        
        if i==0:
            data,RT,SSD,direction,session,subject = load_session_data(data_path, selection,i, NORMALIZED=True)
        
        else:
            data_tmp,RT_tmp,SSD_tmp,direction_tmp,session_tmp,subject_tmp = load_session_data(data_path, selection, i, NORMALIZED=True)
        
            data=np.concatenate((data,data_tmp),axis=0)
            RT=np.concatenate((RT,RT_tmp),axis=0)
            SSD=np.concatenate((SSD,SSD_tmp),axis=0)
            direction=np.concatenate((direction,direction_tmp),axis=0)
            session=np.concatenate((session,session_tmp),axis=0)
            subject=np.concatenate((subject,subject_tmp),axis=0)
    
    mask = (RT < 200) & (SSD < 200)
    data = data[mask]
    direction = direction[mask]
    session = session[mask]
    subject = subject[mask]
    RT = RT[mask]
    SSD = SSD[mask]
    
    data = np.array(data)
    RT = np.array(RT)
    SSD = np.array(SSD)
    direction = np.array(direction)
    session = np.array(session)
    subject = np.array(subject)
    
    corr_RT_maxMUA(data, RT, SSD)
    
    print("mean RT  {:3.2f} s".format(RT.mean()*0.005))

    return data, RT, SSD, direction, session, subject

# %%


# def load_train_set(path_array,selection, train_split=0.80,vali_split = 0.12, random_state=0):
    
#     for i,data_path in enumerate(path_array):
        
#         if i==0:
#             data,RT,SSD,direction,session,subject = load_session_data(data_path, selection,i, NORMALIZED=True)
        
#         else:
#             data_tmp,RT_tmp,SSD_tmp,direction_tmp,session_tmp,subject_tmp = load_session_data(data_path, selection, i, NORMALIZED=True)
        
#             data=np.concatenate((data,data_tmp),axis=0)
#             RT=np.concatenate((RT,RT_tmp),axis=0)
#             SSD=np.concatenate((SSD,SSD_tmp),axis=0)
#             direction=np.concatenate((direction,direction_tmp),axis=0)
#             session=np.concatenate((session,session_tmp),axis=0)
#             subject=np.concatenate((subject,subject_tmp),axis=0)
    
#     mask = (RT < 200) & (SSD < 200)
#     print("mask dtype:", mask.dtype)
#     print("mask shape:", mask.shape)
#     print("mask sample:", mask[:10])
#     data = data[mask]
#     direction = direction[mask]
#     session = session[mask]
#     subject = subject[mask]
#     RT = RT[mask]
#     SSD = SSD[mask]
    
#     test_split = 1-train_split-vali_split
#     effective_vali_split = vali_split/(vali_split+test_split)
#     train_set, vali_set, train_RT, vali_RT, train_SSD, vali_SSD, train_direction, vali_direction, session_train,session_vali, subject_train, subject_vali  = train_test_split(data, RT, SSD,direction,session, subject, train_size=train_split, random_state=random_state)
#     vali_set, test_set, vali_RT, test_RT, vali_SSD, test_SSD, vali_direction, test_direction, session_vali,session_test, subject_vali, subject_test  = train_test_split(vali_set, vali_RT, vali_SSD,vali_direction,session_vali, subject_vali, train_size=effective_vali_split, random_state=random_state)
    
#     train_set = np.array(train_set)
#     vali_set = np.array(vali_set)
#     test_set = np.array(test_set)
#     train_RT = np.array(train_RT)
#     vali_RT = np.array(vali_RT)
#     test_RT = np.array(test_RT)
#     train_SSD = np.array(train_SSD)
#     vali_SSD = np.array(vali_SSD)
#     test_SSD = np.array(test_SSD)
#     train_direction = np.array(train_direction)
#     vali_direction = np.array(vali_direction)
#     test_direction = np.array(test_direction)
#     session_train = np.array(session_train)
#     session_vali = np.array(session_vali)
#     session_test = np.array(session_test)
#     subject_train = np.array(subject_train)
#     subject_vali = np.array(subject_vali)
#     subject_test = np.array(subject_test)
    
#     print("mean training RT  {:3.2f} s".format(train_RT.mean()*0.005))
#     print("mean test RT  {:3.2f} s".format(test_RT.mean()*0.005))

#     return train_set, vali_set, test_set, train_RT, vali_RT, test_RT, train_SSD, vali_SSD, test_SSD, \
#             train_direction, vali_direction, test_direction, session_train,session_vali,session_test, \
#             subject_train, subject_vali, subject_test


# %%


# from collections import defaultdict
# import warnings

def load_train_set(path_array,selection, train_split=0.80, vali_split=0.12, random_state=0):
    """
    Split dei dati preservando, per ogni sessione, le proporzioni per classe di trial.
    - data: (N, ...) np.array
    - RT, SSD, direction, session, subject: arrays length N
    - train_split, vali_split: frazioni (test = 1 - train - vali)
    - restituisce gli stessi 18 elementi come nella tua versione precedente,
      ma con splits costruiti per sessione.
    Note: se una sessione non ha esempi per una certa classe, non è possibile
    farci nulla: quella classe non sarà presente per quella sessione.
    """
    
    for i,data_path in enumerate(path_array):
        
        if i==0:
            data,RT,SSD,direction,session,subject = load_session_data(data_path, selection,i, NORMALIZED=True)
        
        else:
            data_tmp,RT_tmp,SSD_tmp,direction_tmp,session_tmp,subject_tmp = load_session_data(data_path, selection, i, NORMALIZED=True)
        
            data=np.concatenate((data,data_tmp),axis=0)
            RT=np.concatenate((RT,RT_tmp),axis=0)
            SSD=np.concatenate((SSD,SSD_tmp),axis=0)
            direction=np.concatenate((direction,direction_tmp),axis=0)
            session=np.concatenate((session,session_tmp),axis=0)
            subject=np.concatenate((subject,subject_tmp),axis=0)
    
    mask = (RT < 200) & (SSD < 200)
    data = np.array(data[mask])
    direction = np.array(direction[mask])
    session = np.array(session[mask])
    subject = np.array(subject[mask])
    RT = np.array(RT[mask])
    SSD = np.array(SSD[mask])
    
    rng = np.random.RandomState(random_state)
    n = data.shape[0]
    # costruisci label tri-classe: 0 no-stop (RT==0), 1 correct-no-stop (SSD==0), 2 wrong-stop
    labels = np.zeros(n, dtype=int)
    labels[RT == 0] = 0
    labels[SSD == 0] = 1
    labels[(RT != 0) & (SSD != 0)] = 2

    unique_sessions = np.unique(session)
    train_idx = []
    vali_idx = []
    test_idx = []

    for sess in unique_sessions:
        sess_mask = (session == sess)
        sess_indices = np.nonzero(sess_mask)[0]
#         if len(sess_indices) == 0:
#             continue

        # per classe, raccogli indici e split
        sess_train = []
        sess_vali = []
        sess_test = []

        for cls in [0, 1, 2]:
            cls_idx = sess_indices[labels[sess_indices] == cls]
            m = len(cls_idx)
#             if m == 0:
#                 continue
            # shuffle
            perm = rng.permutation(m)
            cls_idx_shuffled = cls_idx[perm]

            # calcolo conteggi per split (flooring), lasciando rest al test
            n_train = int(np.floor(m * train_split))
            n_vali = int(np.floor(m * vali_split))
            n_test = m - n_train - n_vali

            # slice
            start = 0
            end = n_train
            sess_train.extend(cls_idx_shuffled[start:end].tolist())

            start = end
            end = end + n_vali
            sess_vali.extend(cls_idx_shuffled[start:end].tolist())

            start = end
            sess_test.extend(cls_idx_shuffled[start:].tolist())

        train_idx.extend(sess_train)
        vali_idx.extend(sess_vali)
        test_idx.extend(sess_test)

    # final global shuffle per mix dei batch
    rng.shuffle(train_idx)
    rng.shuffle(vali_idx)
    rng.shuffle(test_idx)

    # costruisci gli splits
    train_set = data[train_idx]
    vali_set = data[vali_idx]
    test_set = data[test_idx]

    train_RT = RT[train_idx]
    vali_RT = RT[vali_idx]
    test_RT = RT[test_idx]

    train_SSD = SSD[train_idx]
    vali_SSD = SSD[vali_idx]
    test_SSD = SSD[test_idx]

    train_direction = direction[train_idx]
    vali_direction = direction[vali_idx]
    test_direction = direction[test_idx]

    session_train = session[train_idx]
    session_vali = session[vali_idx]
    session_test = session[test_idx]

    subject_train = subject[train_idx]
    subject_vali = subject[vali_idx]
    subject_test = subject[test_idx]

    # convert to np.array already are
    return (train_set, vali_set, test_set,
            train_RT, vali_RT, test_RT,
            train_SSD, vali_SSD, test_SSD,
            train_direction, vali_direction, test_direction,
            session_train, session_vali, session_test,
            subject_train, subject_vali, subject_test)


def build_ramps(train_RT,ramp_param):
    
    ramps = np.zeros([train_RT.shape[0],ramp_param["L"]])
    
    for i in range(train_RT.shape[0]):
        ramp=deterministic_ramp_generator(ramp_param,train_RT[i]*ramp_param['dt_data'])
        ramps[i,:]=ramp
        
    return list(ramps)  


# %%


class trialGenerator(Dataset):
  

    def __init__(self, MUA,ramps,RT,SSD,session,subject, SATURATE = True,DEMEAN=False):
        self.MUA = np.array(MUA)
        if DEMEAN:
            self.MUA = self.MUA - self.MUA.mean(-1,keepdims=True)
        self.ramps = ramps
        self.RT = RT
        self.SSD = SSD
        self.session = session
        self.subject = subject.astype("int")
        if SATURATE:
            self.ramps[self.ramps>1]=1

    def __len__(self):
        return len(self.MUA)

    def __getitem__(self, idx):
        
        trial = self.MUA[idx]
        ramp = self.ramps[idx]
        RT = self.RT[idx]
        SSD = self.SSD[idx]
        session = self.session[idx]
        subject = self.subject[idx]
        
        trial = torch.from_numpy(trial).transpose(1,0).float()
        RT = torch.Tensor([RT]).float()
        SSD = torch.Tensor([SSD]).float()
        session = torch.Tensor([session]).long()#.unsqueeze(-1)
        subject = torch.Tensor([subject]).float()#.unsqueeze(-1)
        
        ramp = torch.from_numpy(ramp).float().unsqueeze(0)
   
       
        
        return trial, ramp, RT, SSD, session, subject


# %%


class BalancedDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True,num_workers=0,prefetch_factor=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # If sampler is not provided, create a new one
    
        # Count the number of samples in each class
        class_counts = torch.bincount(torch.tensor(dataset.subject))

        # Compute the weight of each sample
        weights_class = 1.0 / class_counts[torch.tensor(dataset.subject)]# has the same length on the data array
        weigth_RT = compute_w_RT(torch.Tensor(dataset.RT))
        weights = weights_class*weigth_RT

        # Create a sampler that samples each class with equal probability
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        super().__init__(dataset, batch_size=batch_size, sampler=sampler,num_workers=num_workers,prefetch_factor=prefetch_factor)


# %%


def compute_w_RT(RT):
    num_bins = 8
    min_value = RT.min()
    max_value = RT.max()

    # Compute the bin widths and the bin indices for each value
    bin_width = (max_value - min_value) / num_bins
    bin_indices = torch.floor((RT - min_value) / bin_width).long()
    bin_indices[(bin_indices==0) | (bin_indices==1)] = 2
    bin_indices[(bin_indices==8) | (bin_indices==7) | (bin_indices==6)] = 5
    bin_indices =bin_indices-2
    # Compute the histogram of bin indices using torch.bincount
    histogram = torch.bincount(bin_indices)
    
    rt_w = 1. / histogram[bin_indices]
    
    return rt_w


# %%

# %%
