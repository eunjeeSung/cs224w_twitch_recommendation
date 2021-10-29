import torch
from torch import nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class Attention(nn.Module):
    def __init__(self, cfgs, num_att, num_heads, causality=False):
        super(Attention, self).__init__()
        self.cfgs = cfgs
        self.causality = causality

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(cfgs['K'], eps=1e-8)
        
        for _ in range(num_att):
            new_attn_layernorm = nn.LayerNorm(cfgs['K'], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  nn.MultiheadAttention(cfgs['K'],
                                                    num_heads,
                                                    0.2)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(cfgs['K'], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(cfgs['K'], 0.2)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, seqs, timeline_mask=None):
        if self.causality:
            tl = seqs.shape[1] # time dim len for enforce causality
            attention_mask = ~torch.tril(torch.ones((tl, tl), 
                                         dtype=torch.bool, 
                                         device=self.cfgs['device']))
        else: attention_mask = None
        
        if timeline_mask != None:
            seqs *= ~timeline_mask.unsqueeze(-1)

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            if timeline_mask != None:
                seqs *=  ~timeline_mask.unsqueeze(-1)

        return self.last_layernorm(seqs)