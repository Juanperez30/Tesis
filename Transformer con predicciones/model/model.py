from torch import nn
import torch
import math
import numpy as np

def subsequent_mask(size):
    #"Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

    

class Attention(nn.Module): 
    def __init__(self, attention_dropout=0):
        super(Attention, self).__init__()
        
        self.dropout = nn.Dropout(p=attention_dropout)

    def forward(self, q, k, v, mask=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1)) #batch *batch(Q,K)
        scores = scores.masked_fill(mask == 0, 0) if mask is not None else scores  

        return torch.matmul(scores, v),scores



class AttentionLayer(nn.Module):
    def __init__(self,length,dp):
        super(AttentionLayer, self).__init__()

        self.Wq = nn.Linear(1, length,bias=False)
        self.Wk = nn.Linear(1, length,bias=False)
        self.Wv = nn.Linear(1, 1,bias=False)

        self.attention = Attention(dp)

    def forward(self, x, mask=None):

        q,k,v = self.Wq(x),self.Wk(x),self.Wv(x)
        return self.attention(q, k, v, mask)


class Encoder(nn.Module):
    def __init__(self, dx, length,dp):
        super(Encoder, self).__init__()
        self.dx = dx
        self.dropout = nn.Dropout(p=dp)

        self.linear = nn.Linear(dx,dx)

        self.attentionLayer = AttentionLayer(length,dp)
        self.norm = nn.LayerNorm([dx,1],elementwise_affine=True,eps=1e-5)

        
        self.f = nn.ELU()

    def forward(self, x):

        x = self.f(self.linear(x))
        x = self.dropout(x)

        x = x.view(-1,1,self.dx,1)
        t,scores =self.attentionLayer(x)
        x = self.norm(x + t)


        return x


class Decoder(nn.Module):
    def __init__(self, dy, length,dp):
        super(Decoder, self).__init__()
        self.dy = dy
        self.dropout = nn.Dropout(p=dp)

        self.linear1 = nn.Linear(dy, dy)

        self.mask = subsequent_mask(dy)
        self.attentionLayer = AttentionLayer(length,dp)
        self.norm1 = nn.LayerNorm([dy,1],elementwise_affine=True,eps=1e-5)

        self.EDAttention = Attention(dp)
        self.norm2 = nn.LayerNorm([dy,1],elementwise_affine=True,eps=1e-5)

        self.linear2 = nn.Linear(dy, dy)

        self.f = nn.ELU()

    def forward(self, c, x):

        zero = torch.zeros(x.shape[0],1,1,1)
        x= torch.cat((zero, x),dim=3)

        x = self.f(self.linear1(x))
        x = self.dropout(x)

        x = x.view(-1,1,self.dy,1)
        t,scores = self.attentionLayer(x, self.mask)
        x = self.norm1(x + t)

        t,scores= self.EDAttention(x, c, c)#(query y)(key,value x)
        x = self.norm2(x + t)

        x = x.view(-1,1,1,self.dy)
        x = self.f(self.linear2(x))
        x = self.dropout(x)

        return x


class AAL(nn.Module):
    def __init__(self, dx, dy,length,dp):
        super(AAL, self).__init__()

        self.encoder = Encoder(dx,length,dp)
        self.decoder = Decoder(dy,length,dp)


    def forward(self, x_enc, x_dec):

        c = self.encoder(x_enc)
        out = self.decoder(c, x_dec)

        return out