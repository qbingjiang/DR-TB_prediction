
import numpy as np
import json
import pandas as pd
from torchsampler import ImbalancedDatasetSampler   ## need to check how to use this package ImbalancedDatasetSampler
from torch.utils.data.sampler import Sampler
import itertools
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset 
# from dataset_2timePoints import data_set
from dataset import data_set 

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    ref::https://github.com/HiLab-git/SSL4MIS/blob/master/code/dataloaders/brats2019.py
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0
    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )
    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size  ## decide the final 

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)



def make_dataloader(X_train_path, y_train, bs=10, ifshuffle=True, iftransform=False, ifbatchSampler=True, ifrandomCrop=True): 
    if ifbatchSampler: 
        idxs_sec = list(np.where(np.array(y_train)==0)[1] ) 
        idxs = list(np.where(np.array(y_train)==1)[1] ) 
        batch_sampler = TwoStreamBatchSampler(idxs_sec, idxs, bs, bs - bs//2)

    if len(X_train_path)>1: 
        dataset_list = [data_set(X_train_path[i], y_train[i], ifSaveDatasetTemp=True, ifReadDatasetTemp=True, iftransform=iftransform, ifrandomCrop=ifrandomCrop) for i in range(len(X_train_path))]
        dataset = ConcatDataset(dataset_list)
        if ifshuffle and not ifbatchSampler: ##ValueError: sampler option is mutually exclusive with shuffle pytorch
            dataloader = DataLoader(dataset=dataset, batch_size=bs, num_workers=0, shuffle=ifshuffle ) 
        elif ifbatchSampler: 
            dataloader = DataLoader(dataset=dataset, num_workers=0, batch_sampler=batch_sampler, pin_memory=True) 
        else: 
            dataloader = DataLoader(dataset=dataset, batch_size=bs, num_workers=0, shuffle=ifshuffle ) 
    else: 
        X_train_path = X_train_path[0] 
        y_train = y_train[0] 
        dataset = data_set(X_train_path, y_train, ifSaveDatasetTemp=True, ifReadDatasetTemp=True, iftransform=iftransform, ifrandomCrop=ifrandomCrop) 
        if ifshuffle and not ifbatchSampler: ##ValueError: sampler option is mutually exclusive with shuffle pytorch
            dataloader = DataLoader(dataset=dataset, batch_size=bs, num_workers=0, shuffle=ifshuffle, ) 
        elif ifbatchSampler: 
            dataloader = DataLoader(dataset=dataset, num_workers=0, batch_sampler=batch_sampler, pin_memory=True ) 
        else: 
            dataloader = DataLoader(dataset=dataset, batch_size=bs, num_workers=0, shuffle=ifshuffle ) 
    return dataloader 


if __name__ == '__main__':
    bs=20
    X_train_path = []
    y_train = []
    X_test_path, y_test = [], []
    testData_list = []
    trainloader = make_dataloader(X_train_path, y_train, bs=bs, ifshuffle=True, iftransform=False, ifbatchSampler=True, ifrandomCrop=True ) 

    testloader0 = make_dataloader(X_train_path, y_train, bs=bs, ifshuffle=False, iftransform=False, ifbatchSampler=False, ifrandomCrop=False ) 
    testloader = make_dataloader(X_test_path, y_test, bs=bs, ifshuffle=False,  iftransform=False, ifbatchSampler=False, ifrandomCrop=False ) 
    
    if testData_list is not None: 
        ExtraTestloader=[] 
        for t_i in range(len(testData_list)): 
            ExtraTestloader.append(make_dataloader(testData_list[t_i][0:1], testData_list[t_i][1:2], bs=bs, ifshuffle=False, iftransform=False, ifbatchSampler=False, ifrandomCrop=False ) )


