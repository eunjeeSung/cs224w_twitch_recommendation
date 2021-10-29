import argparse
import os
import pickle
import yaml

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm


def load_data(cfgs):
    INFILE = os.path.join(cfgs['dataset'],'100k.csv')
    #user,stream,streamer_id,start,stop
    cols = ["user", "stream", "streamer", "start", "stop"]
    data_fu = pd.read_csv(INFILE, header=None, names=cols)
    
    # Add one for padding
    data_fu.user = pd.factorize(data_fu.user)[0]+1
    data_fu['streamer_raw'] = data_fu.streamer
    data_fu.streamer = pd.factorize(data_fu.streamer)[0]+1
    print("Num users: ", data_fu.user.nunique())
    print("Num streamers: ", data_fu.streamer.nunique())
    print("Num interactions: ", len(data_fu))
    print("Estimated watch time: ", (data_fu['stop']-data_fu['start']).sum() * 5 / 60.0)
    
    cfgs['M'] = data_fu.user.max()+1 # users
    cfgs['N'] = data_fu.streamer.max()+2 # items
    print(f'M: {cfgs["M"]} / N: {cfgs["N"]}')
    
    data_temp = data_fu.drop_duplicates(subset=['streamer','streamer_raw'])
    umap      = dict(zip(data_temp.streamer_raw.tolist(),data_temp.streamer.tolist()))
    
    # Splitting and caching
    max_step = max(data_fu.start.max(),data_fu.stop.max())
    print("Num timesteps: ", max_step)
    cfgs['max_step'] = max_step
    cfgs['pivot_1']  = max_step-500
    cfgs['pivot_2']  = max_step-250
    
    print("caching availability")
    ts = {}
    max_avail = 0
    for s in range(max_step+1):
        all_av = data_fu[(data_fu.start<=s) & (data_fu.stop>s)].streamer.unique().tolist()
        ts[s] = all_av
        max_avail = max(max_avail,len(ts[s]))
    cfgs['max_avail'] = max_avail
    cfgs['ts'] = ts
    print("max_avail: ", max_avail)
    
    # Compute availability matrix of size (num_timesteps x max_available)
    max_av   = max([len(v) for k,v in cfgs['ts'].items()])
    max_step = max([k for k,v in cfgs['ts'].items()])+1
    av_tens = torch.zeros(max_step,max_av).type(torch.long)
    for k,v in cfgs['ts'].items():
        av_tens[k,:len(v)] = torch.LongTensor(v)
    cfgs['av_tens'] = av_tens.to(cfgs['device'])
    return data_fu

def preprocess_data(data_fu, cfgs):
    if cfgs['debug']:
        mu = 1000
    else:
        mu = int(10e9)
 
    cache_tr = os.path.join(cfgs['cache_dir'], "train.txt")
    cache_te = os.path.join(cfgs['cache_dir'], "test.txt")
    cache_va = os.path.join(cfgs['cache_dir'], "validation.txt")

    datalist_tr = get_sequences(data_fu, 0, cfgs['pivot_1'], cfgs, mu)
    datalist_va = get_sequences(data_fu, cfgs['pivot_1'], cfgs['pivot_2'], cfgs, mu)
    datalist_te = get_sequences(data_fu, cfgs['pivot_2'], cfgs['max_step'], cfgs, mu)

    pickle.dump(datalist_te, open(cache_te, "wb"))
    pickle.dump(datalist_tr, open(cache_tr, "wb"))
    pickle.dump(datalist_va, open(cache_va, "wb"))

def get_sequences(_data, _p1, _p2, cfgs, max_u=int(10e9)):
    data_list = []

    _data = _data[_data.stop<_p2].copy()
    
    grouped = _data.groupby('user')
    for user_id, group in tqdm(grouped):
        group = group.sort_values('start')
        group = group.tail(cfgs['seq_len']+1)
        if len(group)<2: continue

        group = group.reset_index(drop=True) 
        
        # Get last interaction
        last_el = group.tail(1)
        yt = last_el.start.values[0]
        group.drop(last_el.index,inplace=True)

        # avoid including train in test/validation
        if yt < _p1 or yt >= _p2: continue

        padlen = cfgs['seq_len'] - len(group)

        # sequence input features
        positions  = torch.LongTensor(group.index.values)
        inputs_ts  = torch.LongTensor(group.start.values)
        items      = torch.LongTensor(group['streamer'].values)
        users      = torch.LongTensor(group.user.values)
        bpad       = torch.LongTensor(group.index.values + padlen)

        # sequence output features
        targets    = torch.LongTensor(items[1:].tolist() + [last_el.streamer.values[0]])
        targets_ts = torch.LongTensor(inputs_ts[1:].tolist() + [last_el.start.values[0]])

        data_list.append([bpad, positions, inputs_ts, items, users, targets, targets_ts])

        # stop if user limit is reached
        if len(data_list)>max_u: break
    return data_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to the config file (.yaml)')
    args = parser.parse_args()

    with open(args.config) as file:
        cfgs = yaml.load(file, Loader=yaml.FullLoader)

    data_fu = load_data(cfgs)
    preprocess_data(data_fu, cfgs)