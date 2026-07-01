
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from .dataloader import load_train_set#, BalancedDataLoader


#  left(0): (1, 0, 0, 0)
# right(1): (0, 1, 0, 0)
#    preGO: (0, 0, 1, 0)
#     STOP: (0, 0, 0, 1)
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
        
        # Calculate control signals (PER MODELLO VECCHIO USARE QUESTA RIGA)
        #cont_masked = cont_dir(ssd_masked, dir_masked, tau) if dir_flag else cont(ssd_masked, tau)
        #gate_masked = gate(rt_masked, tau, GO_flag)
        
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
            #data[f"gate_{mask_type}_ordRT"] = gate_masked[rt_indices]
        if mask_type == 'cs' or mask_type == 'ws':
            # Sort by SSD
            ssd_indices = np.argsort(ssd_masked)
            data[f"set_{mask_type}_ordSSD"] = set_masked[ssd_indices]
            data[f"cont_{mask_type}_ordSSD"] = cont_masked[ssd_indices]
            data[f"RT_{mask_type}_ordSSD"] = rt_masked[ssd_indices]
            data[f"SSD_{mask_type}_ordSSD"] = ssd_masked[ssd_indices]
            data[f"dir_{mask_type}_ordSSD"] = dir_masked[ssd_indices]
            data[f"sess_{mask_type}_ordSSD"] = sess_masked[ssd_indices]
            #data[f"gate_{mask_type}_ssd"] = gate_masked[ssd_indices]

    return data


def peak_filthers(dataset, window = 5, threshold = 1):
    lenght = len(dataset)
    #window = 5
    #threshold = 1
    no_peak = 0
    no_peak_list = []
    for q in range(lenght):
        counts = 0
        same_peak = 400
        dataset_trial = dataset[q, :, :]
        MUA = dataset_trial.mean(1)
        for t in range(256-window):
            gradient = MUA[t+window] - MUA[t]
            if gradient > threshold:
                if t != (same_peak + 1):
                    counts += 1
                    same_peak = t
                else: 
                    same_peak = t
        if counts == 0: 
            no_peak += 1
            no_peak_list.append(q)
    
    return no_peak_list


def split(cfg):
    tau = cfg.getint('DataFrame', 'tau')
    peak_filther = cfg.getboolean('DataFrame', 'peak_filther')
    train_split = cfg.getfloat('DataFrame', 'train_split')
    vali_split = cfg.getfloat('DataFrame', 'vali_split')
    if cfg.get('DataFrame', 'seed') == '':
        random_seed = random.randint(0, 1000)
    else:
        random_seed = cfg.getint('DataFrame', 'seed')


    # Piero data\n",
    data_path_array = [
#                        '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Piero_20131202.npz',    #####
#                        '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Piero_20140109.npz',    #####
#                        '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Piero_20140116.npz',    #####
#                        '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Piero_20140606.npz',
#                        '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Piero_20140701.npz',
#                        '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Piero_20140922.npz',
                       '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Cornelio_20140424.npz',  ######
#                        '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Cornelio_20140515.npz', # sessione da 175 trials
#                        '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Cornelio_20140520.npz',    
                       '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Cornelio_20140527.npz',  ######
                       '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Cornelio_20140528.npz',  ######
#                        '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Cornelio_20140529.npz',
#                        '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Cornelio_20140601.npz',
#                        '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Cornelio_20140606.npz',
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
    subject_train,
    subject_vali, 
    subject_test) = load_train_set(data_path_array,selection, train_split=train_split, vali_split = vali_split, random_state=random_seed)
    
#     if peak_filther:
#         train_mask = peak_filthers(train_set)
#         train_set = train_set[train_mask, :, :]
#         train_RT = train_RT[train_mask]
#         train_SSD = train_SSD[train_mask]
#         train_direction = train_direction[train_mask]
#         session_train = session_train[train_mask]
#         subject_train = subject_train[train_mask]
#         vali_mask = peak_filthers(vali_set)
#         vali_set = vali_set[vali_mask, :, :]
#         vali_RT = vali_RT[vali_mask]
#         vali_SSD = vali_SSD[vali_mask]
#         vali_direction = vali_direction[vali_mask]
#         session_vali = session_vali[vali_mask]
#         subject_vali = subject_vali[vali_mask]
#         test_mask = peak_filthers(test_set)
#         test_set = test_set[test_mask, :, :]
#         test_RT = test_RT[test_mask]
#         test_SSD = test_SSD[test_mask]
#         test_direction = test_direction[test_mask]
#         session_test = session_test[test_mask]
#         subject_test = subject_test[test_mask]
#         print("peak_filthered")
    
    
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
#         return (self.trials[idx], self.gates_cont[idx], self.sessions[idx])


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
        

# class BalancedHierarchicalSampler(torch.utils.data.Sampler):
#     """
#     Bilanciamento gerarchico:
#       - uguale numero di sessioni
#       - dentro ciascuna sessione, uguale numero di classi
#     """
#     def __init__(self, dataset, samples_per_class=8):
#         self.dataset = dataset
#         self.samples_per_class = samples_per_class
#         self.indices = self._make_indices()

#     def _make_indices(self):
#         idx_per_sess_class = defaultdict(list)
#         for idx, sess in enumerate(self.dataset.sessions):
#             label = self.dataset.labels[idx]
#             idx_per_sess_class[(sess, label)].append(idx)

#         selected = []
#         for (sess, label), idxs in idx_per_sess_class.items():
#             n = min(len(idxs), self.samples_per_class)
#             selected.extend(random.sample(idxs, n))
#         return selected

#     def __iter__(self):
#         random.shuffle(self.indices)
#         return iter(self.indices)

#     def __len__(self):
#         return len(self.indices)

# class BalancedHomogeneousSampler(Sampler):
#     """
#     1. Homogeneous: Each batch contains data from ONE session.
#     2. Balanced: Inside that batch, we aim for equal distribution of classes (Stop/No-Stop).
#     """
#     def __init__(self, dataset, batch_size, drop_last=False):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.drop_last = drop_last
        
#         # Structure: indices[session_id][class_label] = [idx1, idx2...]
#         self.indices_tree = defaultdict(lambda: defaultdict(list))
        
#         # Populate tree
#         unique_sessions = np.unique(dataset.sessions)
#         for idx in range(len(dataset)):
#             sess = dataset.sessions[idx]
#             label = int(dataset.labels[idx]) # Ensure int for keys
#             self.indices_tree[sess][label].append(idx)
            
#         self.batches = self._generate_batches()

#     def _generate_batches(self):
#         final_batches = []
        
#         for sess in self.indices_tree.keys():
#             # Get indices for this session
#             class_indices = self.indices_tree[sess] # Dict {0: [...], 1: [...], 2: [...]}
#             classes = list(class_indices.keys())
            
#             # Shuffle indices within each class
#             for c in classes:
#                 random.shuffle(class_indices[c])
            
#             # Calculate number of batches for this session
#             # We determine total samples available
#             total_samples = sum(len(lst) for lst in class_indices.values())
#             n_batches = total_samples // self.batch_size
            
#             # We use iterators to pull samples round-robin style
#             iterators = {c: iter(class_indices[c]) for c in classes}
            
#             for _ in range(n_batches):
#                 batch = []
#                 # Attempt to fill batch with equal number of samples per class
#                 samples_per_class = self.batch_size // len(classes)
#                 remainder = self.batch_size % len(classes)
                
#                 # Fill batch
#                 for c in classes:
#                     # Take 'samples_per_class' from this label
#                     count = samples_per_class + (1 if remainder > 0 else 0)
#                     remainder -= 1
                    
#                     for _ in range(count):
#                         try:
#                             batch.append(next(iterators[c]))
#                         except StopIteration:
#                             # If we run out of a class, fill from ANY remaining class in this session
#                             # Fallback logic to ensure batch is full
#                             pass
                            
#                 # If balanced filling failed (run out of one class), fill with random remaining from session
#                 while len(batch) < self.batch_size:
#                     # Find a class that still has data
#                     valid_iterators = [it for it in iterators.values()]
#                     if not valid_iterators: break # Session exhausted
#                     try:
#                         # Simple strategy: just pick next available
#                         for it in valid_iterators:
#                             try:
#                                 batch.append(next(it))
#                                 if len(batch) == self.batch_size: break
#                             except StopIteration:
#                                 continue
#                     except:
#                         break
                
#                 if len(batch) == self.batch_size:
#                     random.shuffle(batch) # Shuffle inside batch so classes aren't ordered
#                     final_batches.append(batch)
#                 elif not self.drop_last:
#                     final_batches.append(batch)

#         # Shuffle the ORDER of batches (Session A, Session C, Session A...)
#         random.shuffle(final_batches)
#         return final_batches

#     def __iter__(self):
#         # Regenerate batches every epoch to ensure new shuffling
#         self.batches = self._generate_batches()
#         for batch in self.batches:
#             yield batch

#     def __len__(self):
#         return len(self.batches)


# def data_attr():
#     # Piero data\n",
#     piero_data_path_array = [
#                            '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Piero_20131202.npz',
#                            '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Piero_20140109.npz',
#                            '/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Piero_20140116.npz',
#                            #'/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Piero_20140606.npz',
#                            #'/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Piero_20140701.npz',
#                            #'/raid/home/tubitoal/DMM/dmm/dataset/MUA/data/Piero_20140922.npz',
#     ]
#     piero_date_array = [
#                      # 'P-20131202',
#                       # 'P-20140109',
#                        #'P-20140116',
#                       #'P-20140606',
#                       #'P-20140701',
#                       #'P-20140922'
#     ]

#     #+test_date_array = test_date_array + ['C-20140527']

#     data_path_array = piero_data_path_array 
#     date_array = piero_date_array

#     selection ="((y_stop == True) | ((y_stop == False)&(y_reward == True)))"# selezioni tutto il movimento
    
#     return data_path_array, selection
