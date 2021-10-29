import random
from tqdm import tqdm
import torch
from sampling import *
from data import *
import torch.nn.functional as F
import numpy as np


def save_scores(scores, cfgs):
    with open("logs.txt", 'a') as fout:
        fout.write('{};{};{};{:.5f};{:.5f};{};{}\n'.format(
                   cfgs['K'],
                   cfgs['fr_ctx'],
                   cfgs['fr_rep'],
                   cfgs['lr'],
                   cfgs['l2'],
                   cfgs['seq_len'],
                   cfgs['topk_att']))
        for k in ['all','new','rep']:
            fout.write('{};{:.5f};{:.5f};{:.5f};{:.5f};{:.5f};{:.5f}\n'.format(
                                                        k,
                                                        scores[k]['h01'],
                                                        scores[k]['h05'],
                                                        scores[k]['h10'],
                                                        scores[k]['ndcg01'],
                                                        scores[k]['ndcg05'],
                                                        scores[k]['ndcg10'],
                                                      ))
        if cfgs['model']=="BERT":
            fout.write("mask_prob: %.2f\n"  % (cfgs['mask_prob']))
        fout.write('\n')
 

def print_scores(scores):
    for k in ['all','new','rep']:
        print('{}: h@1: {:.5f} h@5: {:.5f} h@10: {:.5f} ndcg@1: {:.5f} ndcg@5: {:.5f} ndcg@10: {:.5f}'.format(
                                                        k,
                                                        scores[k]['h01'],
                                                        scores[k]['h05'],
                                                        scores[k]['h10'],
                                                        scores[k]['ndcg01'],
                                                        scores[k]['ndcg05'],
                                                        scores[k]['ndcg10'],
                                                      ))
    print("ratio: ", scores['ratio'])


def metrics(a):
    a   = np.array(a)
    tot = float(len(a))

    return {
      'h01': (a<1).sum()/tot,
      'h05': (a<5).sum()/tot,
      'h10': (a<10).sum()/tot,
      'ndcg01': np.sum([1 / np.log2(rank + 2) for rank in a[a<1]])/tot,
      'ndcg05': np.sum([1 / np.log2(rank + 2) for rank in a[a<5]])/tot,
      'ndcg10': np.sum([1 / np.log2(rank + 2) for rank in a[a<10]])/tot,
    }


def compute_recall(model, _loader, cfgs, writer, epoch, maxit=100000):
    store = {'rrep': [],'rnew': [],'rall': [], 'ratio': []}

    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(_loader)):
            data = data.to(cfgs['device'])
            store = model.compute_rank(data, store, k=10)
            if i > maxit: break

    loss = model.train_step(data)
    writer.add_scalar(f'val/loss', loss, epoch)
    return {
            'rep': metrics(store['rrep']),
            'new': metrics(store['rnew']),
            'all': metrics(store['rall']),
            'ratio': np.mean(store['ratio']),
           }