#!/usr/bin/env python
# coding: utf-8
# %%

import numpy as np


def load_session_data(data_path, selection,session, DEMEANED=False, NORMALIZED=True):
   

    with np.load(data_path) as andreaData:

        X_trial=np.array(andreaData["X_trial"]) 
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
    
    if DEMEANED:
        data=data-np.expand_dims( data.mean(axis=-1),axis=-1)
    if NORMALIZED:       
        mean = data.mean(axis=(0,1), keepdims=True)
        std = data.std(axis=(0,1), keepdims=True)
        data = (data - mean) / std
        
    return data,RT,SSD,direction_binary,session,subject


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

        # per classe, raccogli indici e split
        sess_train = []
        sess_vali = []
        sess_test = []

        for cls in [0, 1, 2]:
            cls_idx = sess_indices[labels[sess_indices] == cls]
            m = len(cls_idx)

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