import os
import numpy as np
import pickle

import torch
import torch_geometric as pyg
import torch.nn.functional as F


def get_dataloaders(cfgs):
    if cfgs['debug']:
        mu = 1000
    else:
        mu = int(10e9)

    cache_tr = os.path.join(cfgs['cache_dir'], "train.txt")
    cache_te = os.path.join(cfgs['cache_dir'], "test.txt")
    cache_va = os.path.join(cfgs['cache_dir'], "validation.txt")

    datalist_tr = pickle.load(open(cache_tr, "rb"))
    datalist_va = pickle.load(open(cache_va, "rb"))
    datalist_te = pickle.load(open(cache_te, "rb"))

    train_dataset = MultiSessionsGraph('./dataset', phrase='train', cfgs=cfgs)
    train_loader = pyg.data.DataLoader(train_dataset,
                                      batch_size=cfgs['batch_size'],
                                      shuffle=True,
                                      drop_last=True)
    val_dataset = MultiSessionsGraph('./dataset', phrase='validation', cfgs=cfgs)
    val_loader = pyg.data.DataLoader(val_dataset,
                                     batch_size=cfgs['batch_size'],
                                     shuffle=True,
                                     drop_last=True)
    test_dataset = MultiSessionsGraph('./dataset', phrase='test', cfgs=cfgs)
    test_loader = pyg.data.DataLoader(test_dataset,
                                      batch_size=cfgs['batch_size'],
                                      shuffle=True,
                                      drop_last=True)
    return train_loader, val_loader, test_loader


def custom_collate(features, cfgs):
    """Returns a [batch x seq x feats] tensor.

    Args:
        feats: [padded_positions,positions,inputs_ts,items,users,targets,targets_ts]
    """
    feat_len = len(features)
    batch_seq = torch.zeros(cfgs['seq_len'], feat_len, dtype=torch.long)
    for feature_idx, seq in enumerate(features):
        for point_idx, feat in enumerate(seq):
            batch_seq[point_idx, feature_idx] = feat
    return batch_seq


class MultiSessionsGraph(pyg.data.InMemoryDataset):
    """Every session is a graph."""
    def __init__(self, root, phrase, transform=None, pre_transform=None, cfgs=None):
        """
        Args:
            root: 'sample', 'yoochoose1_4', 'yoochoose1_64' or 'diginetica'
            phrase: 'train' or 'test'
        """
        assert phrase in ['train', 'validation', 'test']
        self.phrase = phrase
        self.cfgs = cfgs
        super(MultiSessionsGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.phrase + '.txt']

    @property
    def processed_file_names(self):
        return [self.phrase + '.pt']

    def process(self):
        data = pickle.load(
            open(self.raw_dir + '/' + self.raw_file_names[0],'rb'))
        data_list = []

        for datapoint in data:
            sequences, y = datapoint[3], datapoint[5]
            i = 0
            nodes = {}    # dict{15: 0, 16: 1, 18: 2, ...}
            senders = []
            x = []

            for node in sequences:
                if node not in nodes:
                    nodes[node] = i
                    x.append([node])
                    i += 1
                senders.append(nodes[node])
            receivers = senders[:]
            del senders[-1]    # the last item is a receiver
            del receivers[0]    # the first item is a sender
            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor(y, dtype=torch.long)
            data_list.append(pyg.data.Data(x=x,
                                           edge_index=edge_index,
                                           y=y,
                                           liverec_data=custom_collate(datapoint, self.cfgs)))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
