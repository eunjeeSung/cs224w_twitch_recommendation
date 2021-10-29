import random

import torch


def sample_av(p, t, cfgs):
    # availability sampling
    av = cfgs['ts'][t]
    while True:
        ridx = random.randint(0,len(av)-1)
        ri   = av[ridx]
        if p!=ri:
            return ri
 
def sample_uni(p, t, cfgs):
    # uniform sampling
    while True:
        ri = random.randint(0, cfgs['N']-1)
        if p != ri:
            return ri

def sample_negs(data, cfgs):
    pos = data[:, :, 5]
    xts = data[:, :, 6]
    neg = torch.zeros_like(pos)

    ci = torch.nonzero(pos, as_tuple=False)
    ps = pos[ci[:, 0], ci[:, 1]].tolist()
    ts = xts[ci[:, 0], ci[:, 1]].tolist()

    for i in range(ci.shape[0]):
        p, t = ps[i], ts[i]

        if cfgs['uniform']:
            neg[ci[i, 0], ci[i, 1]] = sample_uni(p, t, cfgs)
        else:
            neg[ci[i, 0], ci[i, 1]] = sample_av(p, t, cfgs)

    return neg
