
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .dataloader import load_train_set


#  left: (1, 0, 0, 0)
# right: (0, 1, 0, 0)
# preGO: (0, 0, 1, 0)
#  STOP: (0, 0, 0, 1)
def one_hot_cont(x, direction, tau):
    direct = direction.astype(int)
    steps = 256//tau
    GO_step = 56
    batch_size = x.shape[0]
    tensor = np.zeros((batch_size, steps, 4))  # preGO: (0, 0, 1, 0)
    tensor[:, :(GO_step//tau), 2] = 1 
    tensor[np.arange(batch_size), (GO_step//tau):, direct] = 1     # left(0): (1, 0, 0, 0), right(1): (0, 1, 0, 0)
    for i, val in enumerate(x): 
        if val != 0:    
            tensor[i, ((val + GO_step)//tau):, direct] = 0
            tensor[i, ((val + GO_step)//tau):, 3] = 1      # STOP: (0, 0, 0, 1)
    return tensor


def process_data(data_set, RT, SSD, direction, session, tau):
    masks = {
        'cs': RT == 0,
        'cn': SSD == 0,
        'ws': (RT != 0) & (SSD != 0)
    }
    
    data = {}
    for mask_type, mask in masks.items():
        # Apply mask
        set_masked = data_set[mask]
        rt_masked = RT[mask]
        ssd_masked = SSD[mask]
        dir_masked = direction[mask]
        sess_masked = session[mask]
        
        # PER MODELLO NUOVO USARE QUESTA RIGA
        cont_masked = one_hot_cont(ssd_masked, dir_masked, tau)
            
        if mask_type == 'cn' or mask_type == 'ws':
            # Sort by RT
            rt_indices = np.argsort(rt_masked)
            data[f"set_{mask_type}_ordRT"] = set_masked[rt_indices]
            data[f"cont_{mask_type}_ordRT"] = cont_masked[rt_indices]
            data[f"RT_{mask_type}_ordRT"] = rt_masked[rt_indices]
            data[f"SSD_{mask_type}_ordRT"] = ssd_masked[rt_indices]
            data[f"dir_{mask_type}_ordRT"] = dir_masked[rt_indices]
            data[f"sess_{mask_type}_ordRT"] = sess_masked[rt_indices]
        if mask_type == 'cs' or mask_type == 'ws':
            # Sort by SSD
            ssd_indices = np.argsort(ssd_masked)
            data[f"set_{mask_type}_ordSSD"] = set_masked[ssd_indices]
            data[f"cont_{mask_type}_ordSSD"] = cont_masked[ssd_indices]
            data[f"RT_{mask_type}_ordSSD"] = rt_masked[ssd_indices]
            data[f"SSD_{mask_type}_ordSSD"] = ssd_masked[ssd_indices]
            data[f"dir_{mask_type}_ordSSD"] = dir_masked[ssd_indices]
            data[f"sess_{mask_type}_ordSSD"] = sess_masked[ssd_indices]

    return data


def split(cfg):
    tau = cfg.getint('DataFrame', 'tau')
    train_split = cfg.getfloat('DataFrame', 'train_split')
    vali_split = cfg.getfloat('DataFrame', 'vali_split')
    if cfg.get('DataFrame', 'seed') == '':
        random_seed = random.randint(0, 1000)
    else:
        random_seed = cfg.getint('DataFrame', 'seed')

    data_path_array = [
                       '/dmm/dataset/MUA/data/Piero_20131202.npz',    #####
                       '/dmm/dataset/MUA/data/Piero_20140109.npz',    #####
                       'dmm/dataset/MUA/data/Piero_20140116.npz',    #####
#                        'dmm/dataset/MUA/data/Piero_20140606.npz',
#                        'dmm/dataset/MUA/data/Piero_20140701.npz',
#                        'dmm/dataset/MUA/data/Piero_20140922.npz',
                       'dmm/dataset/MUA/data/Cornelio_20140424.npz',  ######
#                        'dmm/dataset/MUA/data/Cornelio_20140515.npz', 
#                        'dmm/dataset/MUA/data/Cornelio_20140520.npz',    
                       'dmm/dataset/MUA/data/Cornelio_20140527.npz',  ######
                       'dmm/dataset/MUA/data/Cornelio_20140528.npz',  ######
#                        'dmm/dataset/MUA/data/Cornelio_20140529.npz',
#                        'dmm/dataset/MUA/data/Cornelio_20140601.npz',
#                        'dmm/dataset/MUA/data/Cornelio_20140606.npz',
    ]
   
    selection ="((y_stop == True) | ((y_stop == False)&(y_reward == True)))"# selezioni tutto il movimento &(y_reward == True)
    

    (train_set,
    vali_set,
    test_set,
    train_RT,
    vali_RT,
    test_RT,
    train_SSD,
    vali_SSD,
    test_SSD,
    train_direction,
    vali_direction,
    test_direction,
    session_train,
    session_vali,
    session_test,
    _, _, _) = load_train_set(data_path_array,selection, train_split=train_split, vali_split = vali_split, random_state=random_seed)
    
    
    
    train_set = train_set[:, ::tau, :]
    test_set = test_set[:, ::tau, :]
    vali_set = vali_set[:, ::tau, :]  
        
    print(train_set.shape)
    print(vali_set.shape)
    print(test_set.shape)
      
    return train_set, vali_set, test_set, train_RT, vali_RT, test_RT, train_SSD, vali_SSD, test_SSD, train_direction, \
            vali_direction, test_direction, session_train, session_vali, session_test, random_seed, data_path_array


class MUA_dataset(Dataset):
    def __init__(self, trials, gates_cont, labels, sessions):
        self.trials = trials
        self.gates_cont = gates_cont
        self.labels = labels
        self.sessions = sessions
        
        # Mappa stringa -> intero
        unique_sess = np.unique(sessions)
        self.sess_to_idx = {s: i for i, s in enumerate(unique_sess)}
        self.idx_to_sess = {i: s for i, s in enumerate(unique_sess)}
        
        # Converti sessions in interi per getitem
        self.session_indices = np.array([self.sess_to_idx[s] for s in sessions])

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return (self.trials[idx], self.gates_cont[idx], self.session_indices[idx])

from collections import defaultdict


class BalancedDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True,num_workers = 4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # If sampler is not provided, create a new one
    
        # Count the number of samples in each class
        class_counts = torch.bincount(torch.tensor(dataset.labels))

        # Compute the weight of each sample
        weights = 1.0 / class_counts[dataset.labels]# has the same length on the data array

        # Create a sampler that samples each class with equal probability
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        super().__init__(dataset, batch_size=batch_size, sampler=sampler,num_workers =num_workers )