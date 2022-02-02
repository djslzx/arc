import pdb

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
import viz

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
print('Using ' + ('GPU' if T.cuda.is_available() else 'CPU'))

PATH='./tf_model.pt'

T.set_printoptions(threshold=10_000)

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
                 d_model=16,
                 n_conv_layers=6, 
                 n_conv_channels=16,
                 batch_size=16):

        super().__init__()
        self.N, self.H, self.W = N, H, W
        self.d_model = d_model
        self.n_conv_layers = n_conv_layers
        self.n_conv_channels = n_conv_channels
        self.batch_size = batch_size
        self.run = int(time.time())

        # program embedding
        lexicon             = lexicon + ["START", "END", "PAD"] # add start/end tokens to lex
        self.lexicon        = lexicon
        self.n_tokens       = len(lexicon)
        self.token_to_index = {s: i for i, s in enumerate(lexicon)}
        self.pad_token      = self.token_to_index["PAD"]
        self.start_token    = self.token_to_index["START"]
        self.end_token      = self.token_to_index["END"]
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

    def indices_to_tokens(self, indices):
        return [self.lexicon[i] for i in indices]
    
    def model_path(self, epoch):
        return f'tf_model_{self.run}_{epoch}.pt'

    def forward(self, B, P):
        """
        B: a batch of bitmap sets each represented as tensors w/ shape (b, N, c, H, W) 
                  where b is the batch size and c is the number of channels
        P: a batch of programs represented as tensors of indices
        """
        # compute bitmap embedddings
        batch_size = B.shape[0]
        src = B.to(device)
        src = src.reshape(-1, 1, self.H, self.W)
        src = self.conv(src)
        src = self.linear(src)
        src = src.reshape(batch_size, self.N, self.d_model)
        src = src.transpose(0,1)

        # compute program embeddings w/ positional encoding
        tgt = P.to(device)
        tgt = T.stack([self.p_embedding(p) for p in tgt])
        tgt = tgt.transpose(0, 1)
        tgt = self.pos_encoder(tgt)
        # compute masks
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[0]).to(device)
        tgt_padding_mask = self.padding_mask(P).to(device)

        out = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask,
                               tgt_key_padding_mask=tgt_padding_mask)
        out = self.out(out)
        return out

    def make_dataloaders(self, data_loc):
        B, P = [], []
        max_p_len = max(len(p) for bmps, p in util.load_incremental(data_loc))
        for bmps, p in util.load_incremental(data_loc):
            # process bitmaps: add channel dimension and turn list of bmps into tensor
            bmps = T.stack(bmps).unsqueeze(1)
            B.append(bmps)
            # process progs: turn into indices and add padding
            p = self.tokens_to_indices(p)
            p = F.pad(p, pad=(0, max_p_len + 2 - len(p)), value=self.pad_token) # add 2 to compensate for START/END tokens
            P.append(p)

        B = T.stack(B)
        P = T.stack(P)
        # 80/20 split
        tB, vB = B.split((B.shape[0]*4)//5)
        tP, vP = P.split((P.shape[0]*4)//5)

        tloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(tB, tP),
                                          batch_size=self.batch_size, shuffle=True)
        vloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(vB, vP),
                                          batch_size=self.batch_size, shuffle=True)
        return tloader, vloader

    @staticmethod
    def loss(expected, actual):
        return -T.mean(T.sum(F.log_softmax(actual, dim=-1) * expected, dim=-1))
        
    def train_epoch(self, dataloader, optimizer):
        self.train()
        epoch_loss = 0
        for B, P in dataloader:
            B.to(device)
            P.to(device)
            # pdb.set_trace()
            P_input    = P[:, :-1].to(device)
            P_expected = F.one_hot(P[:, 1:], num_classes=self.n_tokens).float().transpose(0, 1).to(device)
            out = self(B, P_input)
            """
            log_softmax: get probabilities
            sum at dim=-1: pull out nonzero components
            mean: avg diff
            negation: max the mean (a score) by minning -mean (a loss)
            """
            loss = ArcTransformer.loss(P_expected, out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        return epoch_loss/len(dataloader)

    def validate_epoch(self, dataloader, optimizer):
        self.eval()
        epoch_loss = 0
        with T.no_grad():
            for B, P in dataloader:
                B.to(device)
                P.to(device)
                P_input    = P[:, :-1].to(device)
                P_expected = F.one_hot(P[:, 1:], num_classes=self.n_tokens).float().transpose(0, 1).to(device)
                out = self(B, P_input)
                loss = ArcTransformer.loss(P_expected, out)
                epoch_loss += loss.detach().item()
        return epoch_loss / len(dataloader)

    def learn(self, tloader, vloader, epochs, threshold=0):
        self.to(device)
        optimizer = T.optim.Adam(self.parameters(), lr=0.001)
        writer = tb.SummaryWriter()
        start_t = time.time()
        current_hour = 0

        for epoch in range(1, epochs+1):
            epoch_start_t = time.time()
            tloss = self.train_epoch(tloader, optimizer)
            epoch_end_t = time.time()
            vloss = self.validate_epoch(vloader, optimizer)

            print(f'[{epoch}/{epochs}] training loss: {tloss:.3f}, validation loss: {vloss:.3f};',
                  f'epoch took {epoch_end_t - epoch_start_t:.3f}s,', 
                  f'{epoch_end_t -start_t:.3f}s total')
            writer.add_scalar('training loss', tloss, i)
            writer.add_scalar('validation loss', vloss, i)

            if (epoch_end_t - start_t)//3600 > current_hour:
                T.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training_loss': tloss,
                    'validation loss': vloss,
                }, self.model_path(epoch))
                current_hour += 1

            if vloss <= threshold or tloss <= threshold: break

        T.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training loss': tloss,
            'validation loss': vloss,
        }, self.model_path(epoch))
        print('Finished training')

    def infer(self, bitmaps, max_length):
        self.eval()
        prompt = T.tensor([[self.start_token]]).long().to(device)
        for _ in range(max_length):
            # pdb.set_trace()
            out = self(bitmaps.unsqueeze(0), prompt)
            indices = out.topk(1).indices
            next_index = indices[-1]
            prompt = T.cat((prompt, next_index), dim=1)
            if next_index == self.end_token:
                break
        return prompt

def train_tf(data_loc, lexicon, epochs, batch_size):
    model = ArcTransformer(N=9, H=B_H, W=B_W, lexicon=lexicon, batch_size=batch_size).to(device)
    tloader, vloader = model.make_dataloaders(data_loc)
    model.learn(tloader, vloader, epochs)

def test_inference(model_state_loc, data_loc):
    def strip(tokens):
        return [t for t in tokens if t not in ['START', 'END', 'PAD']]

    model = ArcTransformer(N=9, H=B_H, W=B_W, lexicon=lexicon, batch_size=16).to(device)
    model.load_state_dict(T.load(model_state_loc))
    tloader, vloader = model.make_dataloaders(data_loc)

    n_envs = 7
    for B, P in tloader:
        for bitmaps, program in zip(B, P):
            out = model.infer(bitmaps, max_length=30)
            out = model.indices_to_tokens(out[0])
            out = strip(out)
            out = deserialize(out)
            ans = strip(model.indices_to_tokens(program))
            ans = deserialize(ans)

            text = f'wanted: {ans}\ngot: {out}\n'
            print(text)

            bmps = []
            while len(bmps) < n_envs:
                env = {'z': seed_zs(), 'sprites': seed_sprites()}
                try:
                    bmps.append((ans.eval(env), out.eval(env)))
                except:
                    pass

            x, y = util.unzip(bmps)
            viz.viz_sample(x, y, text=text)


if __name__ == '__main__':
    lexicon = [f'z_{i}' for i in range(LIB_SIZE)] + \
              [f'S_{i}' for i in range(LIB_SIZE)] + \
              [i for i in range(Z_LO, Z_HI + 1)] + \
              ['~', '+', '-', '*', '<', '&', '?',
               'P', 'L', 'R', 
               'H', 'V', 'T', '#', 'o', '@', '!', '{', '}',]

    # TODO: update/improve model state save loc 
    # TODO: command line args; would enable running multiple models in tandem without editing source files
    # TODO: make N flexible - adapt to datasets with variable-size bitmap example sets

    data_loc = '../data/tiny-exs.dat'
    train_tf(data_loc, lexicon, epochs=1_000_000, batch_size=16)
    # test_inference(model_state_loc='../models/tf_med.pt', 
    #                data_loc='../data/tiny-exs.dat')
    
