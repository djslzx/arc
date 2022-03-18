"""
Implement a two-headed model
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
    Use positional encoding to mark some
    """
    
    def __init__(self, N, d_model, dropout=0.1):
        super().__init__()
        self.N = N
        self.dropout = nn.Dropout(p=dropout)
        self.src_embedding = nn.Embedding(2, d_model)
        
        encoding = T.zeros(N * 2, d_model)
        encoding[:N, :] = self.src_embedding[0]
        encoding[N:, :] = self.src_embedding[1]
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
    
    def __init__(self, N, H, W, lexicon,
                 d_model=512,
                 n_conv_layers=6, n_conv_channels=16,
                 n_tf_encoder_layers=6, n_tf_decoder_layers=6,
                 n_value_heads=3, n_value_ff_layers=6,
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

        # program embedding: embed program tokens as d_model-size tensors
        self.lexicon        = lexicon + ["PROGRAM_START", "PROGRAM_END", "PADDING", "LINE_END"]
        self.n_tokens       = len(lexicon)
        self.token_to_index = {s: i for i, s in enumerate(lexicon)}
        self.PAD            = self.token_to_index["PADDING"]
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
        self.conv = nn.Sequential(*conv_stack)
        self.conv_lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_dim, d_model),
        )
        
        # policy net: receives (B, B', f') and generates an embedding
        # Q: how should B, B' and f' be fed into the transformer?
        #   - add learned encoding for B, B', and f' that identifies what they are
        #   - each elt of B, B', and f' is a [1, d_model] tensor (bitmap embedding, program embedding)
        self.tf_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8),
                                                num_layers=self.n_tf_encoder_layers)
        self.tf_decoder = nn.TransformerDecoder(decoder_layer=nn.TransformerDecoderLayer(d_model=self.d_model, nhead=8),
                                                num_layers=self.n_tf_decoder_layers)
        
        # value function: receives embedding from tf_encoder and maps to evaluation of program
        # Q: should this look anything like a transformerdecoder? I guess we're not generating a seq, so probably not?
        value_layers = []
        for _ in range(n_value_ff_layers):
            value_layers.extend([
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
            ])
        self.value_fn = nn.Sequential(*value_layers, nn.Linear(self.d_model, self.n_value_heads))
        
        