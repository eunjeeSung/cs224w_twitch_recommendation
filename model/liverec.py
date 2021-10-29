import numpy as np
import torch
from torch import nn

from model import attention, embedding, gnn
import sampling


class LiveRec(nn.Module):
    def __init__(self, cfgs):
        super(LiveRec, self).__init__()
        self.cfgs = cfgs

        self.item_embedding = nn.Embedding(cfgs['N']+1, cfgs['K'], padding_idx=0)
        self.pos_emb = nn.Embedding(cfgs['seq_len'], cfgs['K']) 
        self.emb_dropout = nn.Dropout(p=0.2)

        # Sequence encoding attention
        self.att = attention.Attention(cfgs, 
                                       cfgs['num_att'],
                                       cfgs['num_heads'],
                                       causality=True)

        # Availability attention
        self.att_ctx = attention.Attention(cfgs, 
                                           cfgs['num_att_ctx'],
                                           cfgs['num_heads_ctx'],
                                           causality=False)

        # Time interval embedding
        # 24h cycles, except for the first one set to 12h
        self.boundaries = torch.LongTensor([0]+list(range(77, 3000+144, 144))).to(cfgs['device'])
        self.rep_emb = nn.Embedding(len(self.boundaries)+2, cfgs['K'], padding_idx=0)

        # GNN encoder
        self.gnn = gnn.GNNEncoder(hidden_size=cfgs['K'],
                                  n_node=cfgs['N']+1,
                                  cfgs=cfgs)

        # From GNNEncoder
        self.loss_function = nn.CrossEntropyLoss()

    def compute_rank(self, data, store, k=10):
        target_reshape_tuple = (self.cfgs['batch_size'], -1, data.liverec_data.size(1))
        liverec_data_tensor = torch.reshape(data.liverec_data, target_reshape_tuple)
        inputs = liverec_data_tensor[:, :, 3] # inputs 
        pos    = liverec_data_tensor[:, :, 5] # targets
        xtsy   = liverec_data_tensor[:, :, 6] # targets ts

        feats = self.get_feats(data)

        # Add time interval embeddings
        if self.cfgs['fr_ctx']:
            ctx, batch_inds = self.get_ctx_att(liverec_data_tensor, feats)

        # identify repeated interactions in the batch 
        mask = torch.ones_like(pos[:, -1]).type(torch.bool)
        for b in range(pos.shape[0]):
            avt = pos[b, :-1]
            avt = avt[avt != 0]
            mask[b] = pos[b, -1] in avt
            store['ratio'] += [float(pos[b, -1] in avt)]

        for b in range(inputs.shape[0]):
            step = xtsy[b, -1].item()
            av = torch.LongTensor(self.cfgs['ts'][step]).to(self.cfgs['device'])
            av_embs = self.item_embedding(av)

            if self.cfgs['fr_ctx']:
                ctx_expand = torch.zeros(self.cfgs['av_tens'].shape[1],
                                         self.cfgs['K'],
                                         device=self.cfgs['device'])
                ctx_expand[batch_inds[b, -1, :], :] = ctx[b, -1, :, :]
                scores = (feats[b, -1, :] * ctx_expand).sum(-1)
                scores = scores[:len(av)]
            else:
                scores = (feats[b, -1, :] * av_embs).sum(-1)

            iseq = pos[b, -1] == av
            idx  = torch.where(iseq)[0]
            if len(idx) == 0:
                continue
            rank = torch.where(torch.argsort(scores, descending=True) == idx)[0].item()

            if mask[b]:
                store['rrep'] += [rank]
            else:
                store['rnew'] += [rank]
            store['rall'] += [rank]
        return store

    def get_ctx_att(self, data, feats, neg=None):
        if not self.cfgs['fr_ctx']:
            return None

        inputs,pos,xtsy = data[:, :, 3], data[:, :, 5], data[:, :, 6] 

        # unbatch indices
        ci = torch.nonzero(inputs, as_tuple=False)
        flat_xtsy = xtsy[ci[:, 0], ci[:, 1]]

        av = self.cfgs['av_tens'][flat_xtsy,:]
        av_embs = self.item_embedding(av)

        # repeat consumption: time interval embeddings
        if self.cfgs['fr_rep']:
            av_rep_batch = self.get_av_rep(data)
            av_rep_flat  = av_rep_batch[ci[:,0], ci[:,1]]
            rep_enc = self.rep_emb(av_rep_flat)
            av_embs += rep_enc

        flat_feats = feats[ci[:, 0], ci[:, 1], :]
        flat_feats = flat_feats.unsqueeze(1).expand(flat_feats.shape[0],
                                                    self.cfgs['av_tens'].shape[-1],
                                                    flat_feats.shape[1])

        scores = (av_embs * flat_feats).sum(-1)
        inds   = scores.topk(self.cfgs['topk_att'], dim=1).indices
        
        # embed selected items
        seqs = torch.gather(av_embs, 1, inds.unsqueeze(2) \
                    .expand(-1,-1,self.cfgs['K']))

        seqs = self.att_ctx(seqs)

        def expand_att(items):
            av_pos = torch.where(av==items[ci[:,0], ci[:,1]].unsqueeze(1))[1]
            is_in = torch.any(inds == av_pos.unsqueeze(1),1)
            
            att_feats = torch.zeros(av.shape[0],
                                    self.cfgs['K']).to(self.cfgs['device'])
            att_feats[is_in,:] = seqs[is_in,torch.where(av_pos.unsqueeze(1) == inds)[1],:]
            
            out = torch.zeros(inputs.shape[0],
                              inputs.shape[1],
                              self.cfgs['K']).to(self.cfgs['device'])
            out[ci[:,0],ci[:,1],:] = att_feats
            return out

        # training
        if pos != None and neg != None:
            return expand_att(pos), expand_att(neg)
        # testing
        else:
            out = torch.zeros(inputs.shape[0],
                              inputs.shape[1],
                              seqs.shape[1],
                              self.cfgs['K']).to(self.cfgs['device'])
            out[ci[:,0],ci[:,1],:] = seqs
            batch_inds = torch.zeros(inputs.shape[0],
                                     inputs.shape[1],
                                     inds.shape[1],
                                     dtype=torch.long).to(self.cfgs['device'])
            batch_inds[ci[:,0],ci[:,1],:] = inds
            return out, batch_inds

    def get_av_rep(self, data):
        bs     = data.shape[0]
        inputs = data[:,:,3] # inputs 
        xtsb   = data[:,:,2] # inputs ts
        xtsy   = data[:,:,6] # targets ts

        av_batch  = self.cfgs['av_tens'][xtsy.view(-1),:]
        av_batch  = av_batch.view(xtsy.shape[0],xtsy.shape[1],-1)
        av_batch *= (xtsy!=0).unsqueeze(2) # masking pad inputs
        av_batch  = av_batch.to(self.cfgs['device'])

        mask_caus = 1 - torch.tril(torch.ones(self.cfgs['seq_len'],
                                             self.cfgs['seq_len']),
                                   diagonal=-1)
        mask_caus = mask_caus.unsqueeze(0).unsqueeze(3)
        mask_caus = mask_caus.expand(bs, -1, -1,
                                     self.cfgs['av_tens'].shape[-1])
        mask_caus = mask_caus.type(torch.bool).to(self.cfgs['device'])
       
        tile = torch.arange(self.cfgs['seq_len']).unsqueeze(0).repeat(bs,1).to(self.cfgs['device'])
 
        bm   = (inputs.unsqueeze(2).unsqueeze(3)
                == av_batch.unsqueeze(1).expand(-1,
                                                self.cfgs['seq_len'], -1, -1))
        bm  &= mask_caus

        # **WARNING** this is a hacky way to get the last non-zero element in the sequence.
        # It works with pytorch 1.8.1 but might break in the future. 
        sm   = bm.type(torch.int).argmax(1)
        sm   = torch.any(bm,1) * sm
        
        sm   = (torch.gather(xtsy, 1, tile).unsqueeze(2) - 
                torch.gather(xtsb.unsqueeze(2).expand(-1, -1, self.cfgs['av_tens'].shape[-1]), 1, sm))
        sm   = torch.bucketize(sm, self.boundaries)+1
        sm   = torch.any(bm,1) * sm
        
        sm  *= av_batch!=0
        sm  *= inputs.unsqueeze(2)!=0
        return sm

    def get_feats(self, data):
        feats = self.gnn(data)
        return feats

    def predict(self, feats, inputs, items, ctx, data):
        if ctx != None:
            i_embs = ctx
        else:
            i_embs = self.item_embedding(items)
        return (feats * i_embs).sum(dim=-1)

    def train_step(self, data):
        feats = self.get_feats(data)

        target_reshape_tuple = (self.cfgs['batch_size'], self.cfgs['seq_len'], -1)
        liverec_data_tensor = torch.reshape(data.liverec_data, target_reshape_tuple)
        inputs = liverec_data_tensor[:, :, 3]
        pos = liverec_data_tensor[:, :, 5]
        neg = sampling.sample_negs(liverec_data_tensor, self.cfgs).to(self.cfgs['device'])

        ctx_pos, ctx_neg = None, None
        if self.cfgs['fr_ctx']:
            ctx_pos, ctx_neg = self.get_ctx_att(liverec_data_tensor, feats, neg)

        pos_logits = self.predict(feats, inputs, pos, ctx_pos, liverec_data_tensor)
        neg_logits = self.predict(feats, inputs, neg, ctx_neg, liverec_data_tensor)

        loss  = (-torch.log(pos_logits[inputs != 0].sigmoid() + 1e-24)
                 -torch.log(1-neg_logits[inputs != 0].sigmoid() + 1e-24)).sum()
        return loss