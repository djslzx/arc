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

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
print('Using ' + ('GPU' if T.cuda.is_available() else 'CPU'))

PATH='./transformer_model.pt'

import os
dirname = os.path.dirname(__file__)

def rel_to_abspath(relpath):
    return os.path.join(dirname, relpath)

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
        self.n_conv_layers = n_conv_layers
        self.n_conv_channels = n_conv_channels
        self.batch_size = batch_size

        # program embedding
        lexicon             = lexicon + ["START", "END", "PAD"] # add start/end tokens to lex
        self.lexicon        = lexicon
        self.n_tokens       = len(lexicon)
        self.token_to_index = {s: i for i, s in enumerate(lexicon)}
        self.pad_token      = self.token_to_index["PAD"]
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
        self.conv = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
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
        self.out = nn.Linear(self.d_model, self.n_tokens)

    def padding_mask(self, mat):
        return mat == self.pad_token

    def tokens_to_indices(self, tokens):
        return T.tensor([self.token_to_index["START"]] + 
                        [self.token_to_index[token] for token in tokens] + 
                        [self.token_to_index["END"]])

    def forward(self, B, P):
        """
        bmp_sets: a batch of bitmap sets each represented as tensors w/ shape (b, N, c, H, W) 
                  where b is the batch size and c is the number of channels
        progs: a batch of programs represented as tensors of indices
        """
        # compute bitmap embedddings
        batch_size = B.shape[0]
        src = B.to(device)
        src = src.reshape(-1, 1, self.H, self.W)
        src = self.conv(src)
        src = src.reshape(batch_size, self.N, self.d_model)
        src = src.transpose(0,1)

        # compute program embeddings w/ positional encoding
        tgt = P.to(device)
        tgt = T.stack([self.p_embedding(p) for p in tgt])
        tgt = tgt.transpose(0, 1)
        tgt = self.pos_encoder(tgt)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[0]).to(device)
        tgt_padding_mask = self.padding_mask(P).to(device)

        out = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask,
                               tgt_key_padding_mask=tgt_padding_mask)
        out = self.out(out)
        return out

    def make_dataloader(self, data):
        B, P = [], []
        max_p_len = max(len(p) for bmps, p in data)
        for bmps, p in data:
            # process bitmaps: add channel dimension and turn list of bmps into tensor
            bmps = T.stack(bmps).unsqueeze(1)

            # process progs: turn into indices and add padding
            p = self.tokens_to_indices(p)
            p = F.pad(p, pad=(0, max_p_len - len(p) + 1), value=self.pad_token) # add 1 to compensate for truncation later

            B.append(bmps)
            P.append(p)

        dataset = T.utils.data.TensorDataset(T.stack(B), T.stack(P))
        dataloader = T.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def train_epoch(self, dataloader, criterion, optimizer):
        epoch_loss = 0
        for B, P in dataloader:
            B.to(device)
            P.to(device)
            P_input    = P[:, :-1].to(device)
            P_expected = F.one_hot(P[:, 1:], num_classes=self.n_tokens).float().transpose(0, 1).to(device)
            out = self(B, P_input)
            # print('P_input shape:', P_input.shape, 
            #       'P_expected shape:', P_expected.shape, 
            #       'output shape:', out.shape)
            loss = criterion(out, P_expected)
            # print(f" minibatch loss: {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        return epoch_loss/len(dataloader)

    def train_model(self, dataloader, epochs=10000):
        criterion = nn.CrossEntropyLoss()
        optimizer = T.optim.Adam(self.parameters(), lr=0.001)
        self.to(device)
        self.train()            # mark as train mode
        start_t = time.time()
        writer = tb.SummaryWriter()
        
        for i in range(1, epochs+1):
            epoch_start_t = time.time()
            loss = self.train_epoch(dataloader, criterion, optimizer)
            epoch_end_t = time.time()

            print(f'[{i}/{epochs}] loss: {loss:.5f},',
                  f'epoch took {epoch_end_t - epoch_start_t:.2f}s,', 
                  f'{epoch_end_t -start_t:.2f}s total')
            writer.add_scalar('training loss', loss, i)
            if loss == 0: break

        end_t = time.time()
        # print(f'[{i}/{epochs}] loss: {loss.item():.5f}, took {end_t - start_t:.2f}s')
        T.save(model.state_dict(), PATH)
        print('Finished training')


if __name__ == '__main__':
    lexicon = [f'z_{i}' for i in range(LIB_SIZE)] + \
              [f'S_{i}' for i in range(LIB_SIZE)] + \
              [i for i in range(Z_LO, Z_HI + 1)] + \
              ['~', '+', '-', '*', '<', '&', '?',
               'P', 'L', 'R', 
               'H', 'V', 'T', '#', 'o', '@', '!', '{', '}',]

    data = util.load(rel_to_abspath('../data/exs.dat'))
    # print('max program length:', max_p_len)

    model = ArcTransformer(N=100, H=B_H, W=B_W, lexicon=lexicon, batch_size=32).to(device)
    dataloader = model.make_dataloader(data)
    model.train_model(dataloader)
    
