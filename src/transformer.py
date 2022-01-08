import math
import numpy as np
import time

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import torch.optim as optim
import torch.utils.tensorboard as tb

from grammar import *

# device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
device=T.device('cpu')
print('Using ' + ('GPU' if T.cuda.is_available() else 'CPU'))

PATH='./transformer_model.pt'


# TODO: pay attention to batch dim/sequence length dim
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
        # x: (sequence length, batch size) ***
        return self.dropout(x + self.pe[:x.size(0)])


class ArcTransformer(nn.Module):

    def __init__(self, 
                 N, H, W,        # bitmap count, height, width
                 lexicon,        # list of program grammar components
                 d_model=512,
                 n_conv_layers=6, 
                 n_conv_channels=6,
                 batch_size=16):

        super().__init__()
        self.N, self.H, self.W = N, H, W
        self.d_model = d_model
        self.batch_size = batch_size
        self.n_conv_layers = n_conv_layers
        self.n_conv_channels = n_conv_channels

        # program embedding
        lexicon             = lexicon + ["START", "END", "PAD"] # add start/end tokens to lex
        self.lexicon        = lexicon
        self.n_tokens       = len(lexicon)
        print('lex size:', self.n_tokens)
        self.token_to_index = {s: i for i, s in enumerate(lexicon)}
        self.p_embedding    = nn.Embedding(self.n_tokens, self.d_model)
        
        # positional encoding (for program embedding)
        self.pos_encoder = PositionalEncoding(self.d_model)

        # bitmap embedding
        conv_stack = []
        for i in range(n_conv_layers):
            conv_stack.extend([
                nn.Conv2d((1 if i==0 else n_conv_channels), n_conv_channels, 3, padding='same'),
                nn.BatchNorm2d(n_conv_channels),
                nn.ReLU(),
            ])
        k = n_conv_channels * H * W
        self.conv = nn.Sequential(*conv_stack)
        self.linear = nn.Sequential(
            nn.Linear(k, k),
            nn.ReLU(),
            nn.Linear(k, k),
            nn.ReLU(),
            nn.Linear(k, k),
            nn.ReLU(),
            nn.Linear(k, d_model),
        )

        # transformer
        self.transformer = nn.Transformer(self.d_model, num_encoder_layers=1)

        # output linear+softmax
        self.out = nn.Sequential(
            nn.Linear(self.d_model, self.n_tokens),
            nn.LogSoftmax(dim=-1),
        )

    def padding_mask(self, mat, pad_token=-1):
        return mat == pad_token

    def tokens_to_indices(self, tokens):
        return T.tensor([self.token_to_index["START"]] + 
                        [self.token_to_index[token] for token in tokens] + 
                        [self.token_to_index["END"]])

    def forward(self, bmp_sets, progs):
        """
        # TODO: batchify
        bmp_sets: a batch of bitmap sets each represented as tensors w/ shape (b, N, 1, H, W) where b is the batch size
        progs: a batch of programs represented as tensors of indices
        """
        src = bmp_sets.to(device)
        tgt = progs.to(device)

        # compute bitmap embedddings
        # 1. convolve all bitmaps
        src = src.reshape(-1, 1, self.H, self.W)
        src = self.conv(src)
        src = src.reshape(self.batch_size, self.N, self.n_conv_channels, self.H, self.W)
        # 2. max pool conv results w.r.t. each set of bitmaps 
        src = src.max(dim=1).values # shape: (b, 1, H, W)
        # 3. turn each summary bitmap into an embedding of size d_model
        src = src.flatten(start_dim=1)
        src = self.linear(src)
        src = src.transpose(0,1)
        print(src.shape)

        # compute program embeddings w/ positional encoding
        print('tgt', tgt.shape)
        print(tgt[0], self.p_embedding(tgt[0]))
        tgt = T.stack([self.p_embedding(p) for p in tgt])
        tgt = tgt.transpose(0,1)
        tgt = self.pos_encoder(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[0]) # FIXME?

        print(src.shape, tgt.shape)
        out = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask)
        out = self.out(out)
        return out

    def make_dataloader(self, data):
        pad_token = self.token_to_index["PAD"]
        max_p_len = max(len(p) for bmps, p in data)
        B, P = [], []
        for bmps, p in data:
            # process bitmaps: add channel dimension and turn list of bmps into tensor
            bmps = T.stack(bmps).unsqueeze(1)

            # process progs: turn into indices and add padding
            p = self.tokens_to_indices(p)
            p = F.pad(p, pad=(0, max_p_len - len(p)), value=pad_token)

            B.append(bmps)
            P.append(p)

        dataset = T.utils.data.TensorDataset(T.stack(B), T.stack(P))
        dataloader = T.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def train_model(self, dataloader, epochs=1000):
        criterion = nn.CrossEntropyLoss() # TODO: check that this plays nice with LogSoftmax
        optimizer = T.optim.Adam(self.parameters(), lr=0.001)
        self.to(device)
        self.train()            # mark as train mode
        start_time = time.time()
        
        for i in range(1, epochs+1):
            self.train_epoch(dataloader, criterion, optimizer)

            print('[%d/%d] loss: %.5f' % (i, epochs, loss.item()), end='\r')
            writer.add_scalar('training loss', loss.item(), i)

            if loss.item() == 0:
                print('[%d/%d] loss: %.5f' % (i, epochs, loss.item()))
                break

        T.save(model.state_dict(), PATH)
        print('Finished training')

    def train_epoch(self, dataloader, criterion, optimizer):
        loss = 0
        for B, P in dataloader:
            B.to(device)
            P.to(device)

            P_input    = P[:, :-1].to(device)
            P_expected = P[:, 1:].to(device)
            out = self(B, P_input)
            loss = criterion(out, P_expected)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss += loss.detach().item()
        return loss/len(dataloader)

if __name__ == '__main__':
    lexicon = [f'z_{i}' for i in range(LIB_SIZE)] + \
              [f'S_{i}' for i in range(LIB_SIZE)] + \
              [i for i in range(Z_LO, Z_HI + 1)] + \
              ['~', '+', '-', '*', '<', '&', '?',
               'P', 'L', 'R', 
               'H', 'V', 'T', '#', 'o', '@', '!', '{', '}',]

    model = ArcTransformer(N=100, H=B_H, W=B_W, lexicon=lexicon).to(device)
    data = util.load('../data/exs.dat')
    # bmps, progs = util.unzip(data)
    # print(bmps[0][0], progs[0])
    dataloader = model.make_dataloader(data)
    model.train_model(dataloader, epochs=1)
    
