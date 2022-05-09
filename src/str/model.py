import torch as T
from torch import nn, Tensor, functional as F
import math
from typing import List, Dict, Tuple

class PositionalEncoding(nn.Module):
    """
    Positional encoding from 'Attention is All You Need'
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pos = T.arange(max_len).unsqueeze(1)
        div = T.exp(T.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = T.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = T.sin(pos * div)
        pe[:, 0, 1::2] = T.cos(pos * div)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (sequence length, batch size, d_model)
        # x.size(0) -> sequence length
        # self.pe[x.size(0)] -> get positional encoding up to seq length
        return self.dropout(x + self.pe[:x.size(0)])
    
class Model(nn.Module):
    """
    This model learns to infer a program that describes a set of strings.
    
    Inputs: (S, f^, S^) where
      - S is a set of strings
      - f^ is the models current best guess
      - S^ is the set described by the model's current best guess
    
    Output: delta - a letter to add   
    """
    
    PROGRAM_START = "P_START"
    PROGRAM_END = "P_END"
    PADDING = "PADDING"

    def __init__(self, N: int, lexicon: List[str], d_model=512):
        super().__init__()
        self.N = N
        self.d_model = d_model
        
        self.lexicon = [Model.PROGRAM_START, Model.PROGRAM_END, Model.PADDING] + lexicon
        self.lex_size = len(self.lexicon)
        self.token_to_index = {token: i for i, token in enumerate(self.lexicon)}
        self.PROGRAM_START = self.token_to_index[Model.PROGRAM_START]
        self.PROGRAM_END = self.token_to_index[Model.PROGRAM_END]
        self.PADDING = self.token_to_index[Model.PADDING]
        
        self.pos_encoding = PositionalEncoding(self.d_model)
        self.program_embedding = nn.Embedding(num_embeddings=self.lex_size,
                                              embedding_dim=self.d_model)
        
        
        
        
        
        
        
        
if __name__ == '__main__':
    pass