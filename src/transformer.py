import pdb
import sys
import math
import numpy as np
import time
import random

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler, BatchSampler, TensorDataset, DataLoader
import torch.utils.tensorboard as tb
import matplotlib.pyplot as plt

from grammar import *

# dev_name = 'cpu'
dev_name = 'cuda:0' if T.cuda.is_available() else 'cpu'
device = T.device(dev_name)
print(f'Using {dev_name}')

PATH='./tf_model.pt'

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
            nn.Linear(k, d_model),
        )

        # transformer
        self.transformer = nn.Transformer(self.d_model,
                                          num_encoder_layers=6,
                                          num_decoder_layers=6)

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
        tloader = DataLoader(TensorDataset(tB, tP),
                             batch_size=self.batch_size, shuffle=True)
        vloader = DataLoader(TensorDataset(vB, vP),
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

    def sample_model(self, writer, dataloader, epoch, n_samples, max_length):
        self.eval()
        B, P = dataloader.dataset[:self.batch_size] # first batch
        samples = self.sample_programs(B, P, max_length)
        expected_tokens = samples['expected tokens']
        expected_exprs = samples['expected exprs']
        out_tokens = samples['out tokens']
        out_exprs = samples['out exprs']

        n_well_formed = 0       # count the number of well-formed programs over time
        n_non_blank = 0         # count the number of programs with non-blank renders

        def well_formed(expr):
            try:
                if expr is not None:
                    return True
            except: pass        # handle weird exceptions
            return False

        for e_toks, e_expr, o_toks, o_expr in zip(expected_tokens, expected_exprs,
                                                  out_tokens, out_exprs):
            # print(f'Ground truth: {e_expr}')
            # print(f'Sampled program: {o_toks}')

            # record program
            writer.add_text(f'program sample for {e_expr}',
                            f'tokens: {o_toks}, expr: {o_expr}',
                            epoch)
            # record sampled bitmaps (if possible)
            if well_formed(o_expr):
                print("well-formed program, making bitmap...")
                n_well_formed += 1
                any_non_blank = False
                bmps = []
                for i in range(self.N):
                    env = {'z': seed_zs(), 'sprites': seed_sprites()}
                    try:
                        bmp = out_expr.eval(env)
                        any_non_blank = True
                    except:
                        bmp = T.zeros(B_H, B_W).unsqueeze(0) # blank canvas
                    bmps.append(bmp)

                n_non_blank += any_non_blank
                if any_non_blank:
                    writer.add_images(f'bmp samples for {e_expr}', T.stack(bmps), epoch)
                else:
                    print('blank bitmap, skipped')

        # record number of well-formed/non-blank programs found
        writer.add_scalar(f'well-formed', n_well_formed, epoch)
        writer.add_scalar(f'non-blank', n_non_blank, epoch)

    def sample_programs(self, B, P, max_length):
        assert len(B) > 0
        assert len(P) > 0

        def strip(tokens):
            start = int(tokens[0] == 'START')
            end = None
            for i, tok in enumerate(tokens):
                if tok == 'END':
                    end = i
                    break
            return tokens[start:end]

        expected_tokens = [self.indices_to_tokens(p) for p in P]
        expected_exprs = [deserialize(strip(toks)) for toks in expected_tokens]
        out = self.sample(B, max_length)
        out_tokens = [self.indices_to_tokens(o) for o in out]
        out_exprs = []
        for toks in out_tokens:
            try:
                stripped = strip(toks)
                d = deserialize(stripped)
                print(f'in: {toks}, stripped: {stripped}, deserialized: {d}')
                out_exprs.append(d)
            except:
                out_exprs.append(None)
        return {
            'expected tokens': expected_tokens,
            'expected exprs':  expected_exprs,
            'out tokens':      out_tokens,
            'out exprs':       out_exprs,
        }

    def learn(self, tloader, vloader, epochs, threshold=0, sample_freq=10):
        self.to(device)
        optimizer = T.optim.Adam(self.parameters(), lr=0.001)
        writer = tb.SummaryWriter()
        start_t = time.time()
        checkpoint_no = 1       # only checkpoint after first 5 hr period

        for epoch in range(1, epochs+1):
            epoch_start_t = time.time()
            tloss = self.train_epoch(tloader, optimizer)
            epoch_end_t = time.time()
            vloss = self.validate_epoch(vloader, optimizer)

            print(f'[{epoch}/{epochs}] training loss: {tloss:.3f}, validation loss: {vloss:.3f};',
                  f'epoch took {epoch_end_t - epoch_start_t:.3f}s,', 
                  f'{epoch_end_t -start_t:.3f}s total')
            writer.add_scalar('training loss', tloss, epoch)
            writer.add_scalar('validation loss', vloss, epoch)

            if (epoch_end_t - start_t)//(3600 * 5) > checkpoint_no: # checkpoint every 5 hours
                T.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training loss': tloss,
                    'validation loss': vloss,
                }, self.model_path(epoch))
                checkpoint_no += 1

            if epoch % sample_freq == 0:
                self.sample_model(writer, vloader, epoch, n_samples=10, max_length=50)

            if vloss <= threshold or tloss <= threshold: break

        T.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training loss': tloss,
            'validation loss': vloss,
        }, self.model_path(epoch))
        print('Finished training')

    def sample(self, B, max_length, p=0.95):
        # B: [b, N, 1, H, W]
        # max length: cap length of generated seq

        self.eval()
        batch_size = B.shape[0]

        def filter_top_p(v):
            assert v.shape[0] == batch_size
            values, indices = T.sort(v, descending=True)
            sums = T.cumsum(values, dim=-1)
            mask = sums >= p
            # right-shift indices to keep first sum >= p
            mask[..., 1:] = mask[..., :-1].clone() 
            mask[..., 0] = False
            # filter out elements in v
            for b in range(batch_size):
                v[b, indices[b, mask[b]]] = 0
            return v

        prompt = T.tensor([[self.start_token]] * batch_size).long().to(device) # [b, 1]
        for i in range(max_length):
            # TODO: don't run CNN repeatedly
            outs = self(B, prompt)   # [i, b, L] where L is size of alphabet
            outs = T.softmax(outs, 2) # softmax across L dim
            indices = [T.multinomial(filter_top_p(out.clone()), 1) for out in outs] # sample from distribution of predictions
            next_index = indices[-1]
            prompt = T.cat((prompt, next_index), dim=1)
        return prompt

def train_tf(data_loc, lexicon, epochs, N, batch_size):
    model = ArcTransformer(N=N, H=B_H, W=B_W, lexicon=lexicon, batch_size=batch_size).to(device)
    tloader, vloader = model.make_dataloaders(data_loc)
    model.learn(tloader, vloader, epochs)

def test_sampling(checkpoint_loc, data_loc, max_length=100):
    model = ArcTransformer(N=9, H=B_H, W=B_W, lexicon=lexicon, batch_size=16).to(device)
    tloader, vloader = model.make_dataloaders(data_loc)
    checkpoint = T.load(checkpoint_loc)
    tloss = checkpoint['training loss']
    vloss = checkpoint['validation loss']
    model_state = checkpoint['model_state_dict']

    n_envs = 7
    for B, P in tloader:
        samples = self.sample_programs(B, P, max_length)
        expected_tokens = samples['expected tokens']
        expected_exprs = samples['expected exprs']
        out_tokens = samples['out tokens']
        out_tokens = samples['out exprs']

        n_matches = 0
        n_compares = len(P)
        for x_toks, x_expr, y_toks, y_expr in zip(expected_tokens, expected_exprs, 
                                                  out_tokens, out_exprs):
            print(f'expected tokens : {x_toks}')
            print(f'expected exprs: {x_expr}')
            print('-----')
            print(f'actual tokens: {y_toks}')
            print(f'actual exprs: {y_expr}')
            print('-----')
            matching = x_expr == y_expr
            print(f'matching? {matching}')
            n_matches += matching
            if not matching:
                m = max(len(strip(x_toks)), len(strip(y_toks)))
                tok_matches = 0
                for x, y in zip(x_toks[:m], y_toks[:m]):
                    tok_matches += x == y and x not in ['PAD', 'START', 'END']
                print(f'token matches: {tok_matches}/{m}')
            print()

        print(f'matches: {n_matches}/{n_compares} [{n_matches/n_compares * 100}%]')


if __name__ == '__main__':
    lexicon = [f'z_{i}' for i in range(LIB_SIZE)] + \
              [f'S_{i}' for i in range(LIB_SIZE)] + \
              [i for i in range(Z_LO, Z_HI + 1)] + \
              ['~', '+', '-', '*', '<', '&', '?',
               'P', 'L', 'R', 
               'H', 'V', 'T', '#', 'o', '@', '!', '{', '}',]

    # TODO: update/improve model state save loc 
    # TODO: make N flexible - adapt to datasets with variable-size bitmap example sets

    if len(sys.argv) < 2:
        print("Usage: transformer.py train | sample")
        exit(1)

    if sys.argv[1] == 'sample':
        if len(sys.argv) != 4:
            print("Usage: transformer.py sample checkpoint_loc data_loc")
            exit(1)

        checkpoint_loc, data_loc = sys.argv[2:]
        test_sampling(checkpoint_loc, data_loc)        

    elif sys.argv[1] == 'train':
        if len(sys.argv) != 4:
            print("Usage: transformer.py train data_loc N")
            exit(1)

        data_loc = sys.argv[2]
        N = int(sys.argv[3])
        train_tf(data_loc, lexicon, N=N, epochs=1_000_000, batch_size=16)        
    else:
        print("Usage: transformer.py train | sample")
        exit(1)
