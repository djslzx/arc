"""
Implement a two-headed model (value, policy)
"""
import pdb
import math
import itertools as it
import torch as T
import torch.nn as nn
import torch.nn.functional as F

dev_name = 'cuda:0' if T.cuda.is_available() else 'cpu'
device = T.device(dev_name)

class SrcEncoding(nn.Module):
    """
    Add learned encoding to identify data sources
    """
    
    def __init__(self, d_model: int, source_sizes: list[int], dropout=0.1):
        super().__init__()
        n_sources = len(source_sizes)
        self.dropout = nn.Dropout(p=dropout)
        self.src_embedding = nn.Embedding(n_sources, d_model)
        
        encoding = T.zeros(sum(source_sizes), d_model)
        start = 0
        for i, source_sz in enumerate(source_sizes):
            encoding[start:start + source_sz, :] = self.src_embedding[i]
            start += source_sz
        self.encoding = encoding
        pdb.set_trace()
    
    def forward(self, x):
        return self.dropout(x + self.encoding)


class PositionalEncoding(nn.Module):
    """
    Use positional encoding from 'Attention is All You Need'
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
        # x: (sequence length, batch size)
        # x.size(0) -> sequence length
        # self.pe[x.size(0)] -> get positional encoding up to seq length
        return self.dropout(x + self.pe[:x.size(0)])


class Model(nn.Module):
    
    def __init__(self, N: int, H: int, W: int, lexicon: list[str],
                 d_model=512,
                 n_conv_layers=6, n_conv_channels=16,
                 n_tf_encoder_layers=6, n_tf_decoder_layers=6,
                 n_value_heads=3, n_value_ff_layers=6,
                 max_program_length=50,
                 batch_size=32):
        super().__init__()
        self.N = N
        self.H = H
        self.W = W
        self.d_model = d_model
        self.n_conv_layers = n_conv_layers
        self.n_conv_channels = n_conv_channels
        self.n_tf_encoder_layers = n_tf_encoder_layers
        self.n_tf_decoder_layers = n_tf_decoder_layers
        self.n_value_heads = n_value_heads
        self.n_value_ff_layers = n_value_ff_layers
        self.max_program_length = max_program_length
        self.batch_size = batch_size
        
        # program embedding: embed program tokens as d_model-size tensors
        self.lexicon        = lexicon + ["PROGRAM_START", "PROGRAM_END", "PADDING", "LINE_END"]
        self.n_tokens       = len(lexicon)
        self.token_to_index = {s: i for i, s in enumerate(lexicon)}
        self.PADDING        = self.token_to_index["PADDING"]
        self.PROGRAM_START  = self.token_to_index["PROGRAM_START"]
        self.PROGRAM_END    = self.token_to_index["PROGRAM_END"]
        self.LINE_END       = self.token_to_index["LINE_END"]
        self.p_embedding    = nn.Embedding(num_embeddings=self.n_tokens, embedding_dim=self.d_model)
        self.pos_encoding   = PositionalEncoding(self.d_model)
        
        # bitmap embedding: convolve HxW bitmaps into d_model-size tensors
        conv_stack = [
            nn.Conv2d(1, n_conv_channels, 3, padding='same'),
            nn.BatchNorm2d(n_conv_channels),
            nn.ReLU(),
        ]
        for _ in range(n_conv_layers - 1):
            conv_stack.extend([
                nn.Conv2d(n_conv_channels, n_conv_channels, 3, padding='same'),
                nn.BatchNorm2d(n_conv_channels),
                nn.ReLU(),
            ])
        conv_out_dim = n_conv_channels * H * W
        self.conv = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            nn.Linear(conv_out_dim, d_model),
        )
        # Add embedding to track source of different tensors passed into the transformer encoder
        self.src_encoding = SrcEncoding(d_model=d_model, source_sizes=[self.N, self.N, self.max_program_length])
        self.tf_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8),
                                                num_layers=self.n_tf_encoder_layers)
        
        # policy head: receives transformer encoding of the triple (B, B', f') and generates an embedding
        self.tf_decoder = nn.TransformerDecoder(decoder_layer=nn.TransformerDecoderLayer(d_model=self.d_model, nhead=8),
                                                num_layers=self.n_tf_decoder_layers)

        # value head: receives code from tf_encoder and maps to evaluation of program
        value_layers = []
        for _ in range(n_value_ff_layers):
            value_layers.extend([
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
            ])
        self.value_fn = nn.Sequential(
            *value_layers,
            nn.Linear(self.d_model, self.n_value_heads)
        )

    def embed_bitmaps(self, b, batch_size):
        """
        Map the batched bitmap set b (of shape [batch, H, W])
        to bitmap embeddings (of shape [N, batch, d_model])
        """
        b_flat = b.to(device).reshape(-1, 1, self.H, self.W) # add channel dim, flatten bitmap collection
        e_b = self.conv(b_flat).reshape(batch_size, -1, self.d_model)  # apply conv, portion embeddings into batches
        e_b = e_b.transpose(0, 1)  # swap batch and sequence dims
        return e_b
        
    def embed_programs(self, f):
        """
        Map the batched program list (represented as a 2D tensor of program indices)
        to a tensor of program embeddings
        """
        f_embeddings = T.stack([self.p_embedding(p) for p in f.to(device)]).transpose(0, 1)
        return f_embeddings

    def forward(self, b, b_hat, f_hat):
        # pass B, B' through CNN -> e_B, e_B'
        # pass f' through program embedding and positional encoding -> pe_f'
        # note: assumes that f' is a list of *indices*, not tokens
        batch_size = b.shape[0]
        e_b = self.embed_bitmaps(b, batch_size)
        e_b_hat = self.embed_bitmaps(b_hat, batch_size)
        pe_f_hat = self.pos_encoding(self.embed_programs(f_hat))
        
        # concatenate e_B, e_B', e_f' into a single sequence and pass through source encoding -> src
        src = self.src_encoding(T.cat((e_b, e_b_hat, pe_f_hat)))

        # pass src through tf_encoder -> tf_encoding
        tf_code = self.tf_encoder(src)

        # pass tf_encoding through value fn and tf_decoder -> value, policy
        policy = self.tf_decoder(tf_code)
        value = self.value_fn(tf_code)
        
        return policy, value
    
    
def prep_data(raw_programs):
    """
    Convert a list of (program, bitmap set) pairs into:
     - batches of programs (F),
     - prefix-delta pairs (f', d') associated with a program (f)
     - value function outputs
    """
    pass





