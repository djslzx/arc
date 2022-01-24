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
T.set_printoptions(threshold=10_000)

class MLP(nn.Module):

    def __init__(self, 
                 N, H, W,       # bitmap count, height, width
                 lexicon,       # list of program grammar components
                 hidden=512,    # size of hidden layers
                 max_program_size=50,
                 batch_size=16,
                 path='./mlp.pt'):

        super().__init__()
        self.N, self.H, self.W = N, H, W
        self.hidden = hidden
        self.max_program_size = max_program_size
        self.batch_size = batch_size
        self.path = path

        # program embedding
        lexicon             = lexicon + ["START", "END", "PAD"] # add start/end tokens to lex
        self.lexicon        = lexicon
        self.n_tokens       = len(lexicon)
        self.token_to_index = {s: i for i, s in enumerate(lexicon)}
        self.pad_token      = self.token_to_index["PAD"]

        # MLP
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.N * self.H * self.W, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.n_tokens * self.max_program_size),
        )

    def tokens_to_indices(self, tokens):
        def lookup(key):
            # k = key if not isinstance(key, T.Tensor) else key.item()
            return self.token_to_index[key]

        return T.tensor([lookup("START")] + 
                        [lookup(token) for token in tokens] + 
                        [lookup("END")])

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.to(device)
        # print('x shape:', x.shape)
        out = self.mlp(x)
        # print('out shape:', out.shape)
        return out

    def make_dataloader(self, data):
        B, P = [], []
        for bmps, p in data:
            bmps = T.stack(bmps).unsqueeze(1) # add channel dimension and turn list of bmps into tensor
            p = self.tokens_to_indices(p)
            p = F.pad(p, pad=(0, self.max_program_size - len(p)), value=self.pad_token)
            p = F.one_hot(p).flatten().float()

            B.append(bmps)
            P.append(p)

        dataset = T.utils.data.TensorDataset(T.stack(B), T.stack(P))
        dataloader = T.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def train_epoch(self, dataloader, criterion, optimizer):
        epoch_loss = 0
        for B, P in dataloader:
            x = B.to(device)
            y = P.to(device)
            # print(f'x: {x.shape}, y: {y.shape}')
            out = self(x)
            loss = criterion(out, y)

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

            if i % 100 == 0:
                print(f'[{i}/{epochs}] loss: {loss:.5f},',
                      f'epochs took {(epoch_end_t - epoch_start_t)*100:.2f}s,', 
                      f'{epoch_end_t -start_t:.2f}s total')
                writer.add_scalar('training loss', loss, i)
                if loss == 0: break

        end_t = time.time()
        # print(f'[{i}/{epochs}] loss: {loss.item():.5f}, took {end_t - start_t:.2f}s')
        T.save(self.state_dict(), self.path)
        print('Finished training')


def train(lex, datafile, N=7, batch_size=2, epochs=1000000):
    data = util.load(datafile)
    model = MLP(N=N, H=B_H, W=B_W, lexicon=lexicon, batch_size=batch_size).to(device)
    dataloader = model.make_dataloader(data)
    model.train_model(dataloader, epochs)


if __name__ == '__main__':
    lexicon = [f'z_{i}' for i in range(LIB_SIZE)] + \
              [f'S_{i}' for i in range(LIB_SIZE)] + \
              [i for i in range(Z_LO, Z_HI + 1)] + \
              ['~', '+', '-', '*', '<', '&', '?',
               'P', 'L', 'R', 
               'H', 'V', 'T', '#', 'o', '@', '!', '{', '}',]

    train(lexicon, '../data/small-exs.dat', epochs=10000000)
    
