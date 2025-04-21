import torch 
import torch.nn as nn
import math
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias
        assert n_embd % n_head == 0
        
        self.head_size = n_embd // n_head
        
        self.q_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=bias)
        
        self.o_proj = nn.Linear(n_embd, n_embd, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None
                ):
        
        batch_size, seq_len_q, n_embd_q = query.shape
        _, seq_len_k, n_embd_k = key.shape
        _, seq_len_v, n_embd_v = value.shape
        
        assert n_embd_q == self.n_embd
        assert n_embd_k == self.n_embd
        assert n_embd_v == self.n_embd

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # (B,T,C) -> (B,T,h,hs) -> (B,h,T,hs)        
        Q = Q.view(batch_size, seq_len_q, self.n_head, -1).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.n_head, -1).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.n_head, -1).transpose(1, 2)
        
        # (B,h,Tq,hs) @ (B,h,hs,Tk) -> (B,h,Tq,Tk)
        scores = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(self.head_size) 
        
        if key_padding_mask is not None:
            # key_padding_mask:(B,seq_len_k) - > (B,1,1,seq_len_k)
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # (Tq,Tk)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # (B, Tq, Tk)
                attn_mask = attn_mask.unsqueeze(1)
            scores = scores.masked_fill(attn_mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        #(B,h,Tq,Tk) @ (B,h,Tk,hs) -> (B,h,Tq,hs)
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1,2).contiguous().view(batch_size, seq_len_q, -1)
        
        output = self.o_proj(output)
        output = self.resid_dropout(output)
        return output, attn_weights
        
        