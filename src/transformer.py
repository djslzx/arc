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


# TODO: pay attention to batch dim/sequence length dim
class PositionalEncoding(nn.Module):
    
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
        # x : sequence length, batch size ***
        return self.dropout(x + self.pe[:x.size(0)])


class ArcTransformer(nn.Module):

    def __init__(self, 
                 N, H, W,        # bitmap count, height, width
                 lexicon,        # list of program grammar components
                 d_model=512,
                 max_p_len=100,   # max program length in tokens
                 n_conv_layers=6, 
                 n_conv_channels=6):

        super().__init__()
        self.N, self.H, self.W = N, H, W
        self.d_model = d_model
        self.max_p_len = max_p_len

        # program embedding
        self.lexicon        = lexicon
        self.n_tokens       = len(lexicon)
        self.token_to_index = {s: i for i, s in enumerate(lexicon)}
        self.p_embedding    = nn.Embedding(self.n_tokens, self.d_model)
        
        # positional encoding (for program embedding)
        self.pos_encoder = PositionalEncoding(self.d_model)

        # bitmap embedding
        layers = []
        for i in range(n_conv_layers):
            layers.extend([
                nn.Conv2d((1 if i==0 else n_conv_channels), n_conv_channels, 3, padding='same'),
                nn.BatchNorm2d(n_conv_channels),
                nn.ReLU(),
            ])
        self.bmp_embedding = nn.Sequential(*layers)

        # transformer
        self.transformer = nn.Transformer(self.d_model, num_encoder_layers=1)

        # output linear+softmax
        self.out = nn.Sequential(
            nn.Linear(self.d_model, self.n_tokens),
            nn.Softmax(dim=0),
        )

    def tokens_to_indices(self, tokens):
        return F.one_hot(T.tensor([self.token_to_index[token] for token in tokens]))

    def programs_to_tensors(self, programs):
        return pad_sequence([self.tokens_to_indices(program) for program in programs], batch_first=True)

    def forward(self, bmps, p):
        '''
        bmps: a set of bitmaps represented as a tensor w/ shape (N x 1 x H x W)
        p: a program represented as a tensor of token indices
        '''
        src = self.bmp_embedding(bmps.squeeze(dim=0))
        tgt = self.pos_encoder(self.p_embedding(p))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[0])
        out = self.transformer(src=src, tgt=tgt, 
                               tgt_mask=tgt_mask)
        out = self.out(out)
        return out

    def train_model(self, bmp_sets, progs, epochs=1000):
        criterion = nn.CrossEntropyLoss()
        optimizer = T.optim.Adam(self.parameters(), lr=0.001)

        srcs = T.stack([T.stack(bmp_set).unsqueeze(1) for bmp_set in bmp_sets]).to(device)
        tgts = self.programs_to_tensors(progs).to(device)
        print('srcs shape:', srcs.shape, 'tgts shape:', tgts.shape)

        dataset = T.utils.data.TensorDataset(srcs, tgts)
        dataloader = T.utils.data.DataLoader(dataset, shuffle=True)

        self.to(device)
        self.train()            # train mode
        loss = 0
        start_time = time.time()
        
        for epoch in range(1, epochs+1):
            for i, (x, y) in enumerate(dataloader, 1):
                out = self(x, y)
                loss = criterion(out, y.squeeze(0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('[%d/%d] loss: %.10f' % (epoch, epochs, loss.item()), end='\r')
            writer.add_scalar('training loss', loss.item(), epoch)

            if loss.item() == 0:
                print('[%d/%d] loss: %.3f' % (epoch, epochs, loss.item()))
                break

        T.save(self.state_dict(), PATH)
        print('Finished training')
        

if __name__ == '__main__':
    lexicon = [f'z_{i}' for i in range(LIB_SIZE)] + \
              [f'S_{i}' for i in range(LIB_SIZE)] + \
              [i for i in range(Z_LO, Z_HI + 1)] + \
              ['~', '+', '-', '*', '<', '&', '?',
               'P', 'L', 'R', 
               'H', 'V', 'T', '#', 'o', '@', '!', '{', '}',]
    print('lex size:', len(lexicon))

    transformer = ArcTransformer(N=100, H=B_H, W=B_W, lexicon=lexicon).to(device)
    data = util.load('../data/exs.dat')
    progs, bmp_sets = util.unzip(data)

    transformer.train_model(bmp_sets, progs)
    
