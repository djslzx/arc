import pdb
import time
import math
import itertools as it

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.tensorboard as tb

from grammar import *
import viz

# dev_name = 'cpu'
dev_name = 'cuda:0' if T.cuda.is_available() else 'cpu'
device = T.device(dev_name)

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
                 name,          # identifier for tracking separate runs
                 N, H, W,       # bitmap count, height, width
                 lexicon,       # list of program grammar components
                 d_model=512,
                 n_conv_layers=6, 
                 n_conv_channels=16,
                 batch_size=16):

        super().__init__()
        self.name = name
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
        return f'tf_model_{self.name}_{epoch}.pt'

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

    def make_dataloader(self, get_data, blind=False):
        B, P = [], []
        max_p_len = max(len(p) for bmps, p in get_data())
        for bmps, p in get_data():
            # process bmps
            if not blind:
                # add channel dimension
                bmps = T.stack(bmps).unsqueeze(1)
            else:
                bmps = T.zeros(self.N, 1, self.H, self.W)
            B.append(bmps)

            # process progs: turn into indices and add padding
            indices = self.tokens_to_indices(p)
            # add 2 to compensate for START/END tokens
            padded_indices = F.pad(indices, pad=(0, max_p_len + 2 - len(indices)), value=self.pad_token)
            P.append(padded_indices)

        B = T.stack(B)
        P = T.stack(P)
        return DataLoader(TensorDataset(B, P), batch_size=self.batch_size, shuffle=True)

    @staticmethod
    def token_loss(expected, actual):
        """
        log_softmax: get probabilities
        sum at dim=-1: pull out nonzero components
        mean: avg diff
        negation: max the mean (a score) by minning -mean (a loss)
        """
        return -T.mean(T.sum(F.log_softmax(actual, dim=-1) * expected, dim=-1))

    @staticmethod
    def word_loss(expected, actual):
        """
        x = log_softmax(actual, -1) : turn logits into probabilities
        x = (x * expected)          : pull out values from `actual` at nonzero locations in `expected`
        x = T.sum(x, -1)            : remove zeros
        x = T.sum(x, 0)             : take sum of log-probabilities for each example in the batch
        x = T.mean(x)               : compute mean probability of correctly generating each sequence in the batch
        x = -x                      : minimize loss (-mean) to maximize mean pr of
        """
        log_prs = T.sum(F.log_softmax(actual, dim=-1) * expected, dim=-1)
        return -T.mean(T.sum(log_prs, dim=0))
    
    def train_epoch(self, dataloader, optimizer):
        self.train()
        epoch_loss = 0
        for B, P in dataloader:
            B.to(device)
            P.to(device)
            P_input    = P[:, :-1].to(device)
            P_expected = F.one_hot(P[:, 1:], num_classes=self.n_tokens).float().transpose(0, 1).to(device)
            out = self(B, P_input)

            loss = ArcTransformer.word_loss(P_expected, out)
            # loss = ArcTransformer.token_loss(P_expected, out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        return epoch_loss/len(dataloader)

    def validate_epoch(self, dataloader):
        self.eval()
        epoch_loss = 0
        with T.no_grad():
            for B, P in dataloader:
                B.to(device)
                P.to(device)
                P_input    = P[:, :-1].to(device)
                P_expected = F.one_hot(P[:, 1:], num_classes=self.n_tokens).float().transpose(0, 1).to(device)
                out = self(B, P_input)
                loss = ArcTransformer.word_loss(P_expected, out)
                epoch_loss += loss.detach().item()
        return epoch_loss / len(dataloader)

    def sample_model(self, writer, B, P, step, max_length):
        self.eval()
        samples = self.sample_programs(B, P, max_length)
        expected_tokens = samples['expected tokens']
        expected_exprs = samples['expected exprs']
        out_tokens = samples['out tokens']
        out_exprs = samples['out exprs']

        n_well_formed = 0       # count the number of well-formed programs over time
        n_non_blank = 0         # count the number of programs with non-blank renders

        def well_formed(expr):
            try:
                if expr is not None:  # expr is None when parsing incomplete
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
                            step)
            # record sampled bitmaps (if possible)
            print(f'output for program {e_expr}:')
            if not well_formed(o_expr):
                print(f"  malformed: {o_toks}")
            else:
                print(f"  well-formed: {o_expr}")
                n_well_formed += 1
                any_non_blank = False
                bmps = []
                for i in range(self.N):
                    env = {'z': seed_zs(), 'sprites': seed_sprites()}
                    try:
                        bmp = o_expr.eval(env).unsqueeze(0)
                        any_non_blank = True
                    except (AssertionError, AttributeError):
                        bmp = T.zeros(B_H, B_W).unsqueeze(0) # blank canvas
                    bmps.append(bmp/10)                      # scale color value to fit in [0,1]

                n_non_blank += any_non_blank
                if any_non_blank:
                    bmp_stack = T.stack(bmps)
                    writer.add_images(f'bmp samples for {e_expr}', bmp_stack, step, dataformats='NCHW')
                else:
                    print('all bitmaps blank, skipped')

        # record number of well-formed/non-blank programs found
        writer.add_scalar(f'well-formed', n_well_formed, step)
        writer.add_scalar(f'non-blank', n_non_blank, step)

    @staticmethod
    def strip(tokens):
        start = int(tokens[0] == 'START')
        end = None
        for i, tok in enumerate(tokens):
            if tok == 'END':
                end = i
                break
        return tokens[start:end]

    def sample_tokens(self, B, P, max_length):
        assert len(B) > 0 and len(P) > 0, f'len(B)={len(B)}, len(P)={len(P)}'
        expected_tok_vecs = [self.indices_to_tokens(p) for p in P]
        out = self.sample(B, max_length)
        out_tok_vecs = [self.indices_to_tokens(o) for o in out]
        return expected_tok_vecs, out_tok_vecs
    
    def sample_programs(self, B, P, max_length):
        assert len(B) > 0 and len(P) > 0, f'len(B)={len(B)}, len(P)={len(P)}'

        def tok_vecs_to_exprs(tok_vecs):
            for tok_vec in tok_vecs:
                try:
                    yield deserialize(ArcTransformer.strip(tok_vec))
                except:
                    yield None

        expected_tok_vecs, out_tok_vecs = self.sample_tokens(B, P, max_length)
        expected_exprs = list(tok_vecs_to_exprs(expected_tok_vecs))
        out_exprs = list(tok_vecs_to_exprs(out_tok_vecs))
        assert(len(out_exprs) == len(expected_exprs))
        return {
            'expected tokens': expected_tok_vecs,
            'expected exprs':  expected_exprs,
            'out tokens':      out_tok_vecs,
            'out exprs':       out_exprs,
        }

    def learn(self, tloader, vloader, epochs, learning_rate=10 ** -4, threshold=0, vloss_margin=1,
              sample_freq=10, log_freq=1):
        self.to(device)
        optimizer = T.optim.Adam(self.parameters(), lr=learning_rate)
        writer = tb.SummaryWriter(comment=f'_{self.name}')
        start_t = time.time()
        checkpoint_no = 1       # only checkpoint after first 5 hr period

        it = iter(vloader)
        sample_B, sample_P = it.next()
        min_vloss = T.inf

        for epoch in range(1, epochs+1):
            epoch_start_t = time.time()
            tloss = self.train_epoch(tloader, optimizer)
            epoch_end_t = time.time()
            vloss = self.validate_epoch(vloader)

            print(f'[{epoch}/{epochs}] training loss: {tloss:.3f}, validation loss: {vloss:.3f}; '
                  f'epoch took {epoch_end_t - epoch_start_t:.3f}s '
                  f'on {len(tloader)} batch{"es" if len(tloader) > 1 else ""} of size {self.batch_size}, '
                  f'{epoch_end_t -start_t:.3f}s total')
            writer.add_scalar('training loss', tloss, epoch)
            writer.add_scalar('validation loss', vloss, epoch)

            # write a checkpoint every `log_freq` hours
            if (epoch_end_t - start_t)//(3600 * log_freq) > checkpoint_no: 
                T.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training loss': tloss,
                    'validation loss': vloss,
                }, self.model_path(epoch))
                checkpoint_no += 1

            if epoch % sample_freq == 0:
                self.sample_model(writer, sample_B, sample_P, epoch, max_length=50)

            # exit when error is within threshold of 0
            if vloss <= threshold or tloss <= threshold: break
            
            # exit when validation error starts ticking back up
            if vloss >= min_vloss + vloss_margin: break

            if vloss < min_vloss:
                print(f"Achieved new minimum validation loss: {min_vloss} -> {vloss}")
                min_vloss = vloss

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

def recover_model(checkpoint_loc, name, N, H, W, d_model, batch_size, lexicon=LEXICON):
    model = ArcTransformer(name=name, lexicon=lexicon, N=N, H=H, W=W, d_model=d_model, batch_size=batch_size).to(device)
    checkpoint = T.load(checkpoint_loc)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_lines(seq):
    try:
        return seq.bmps
    except AttributeError:
        return []

def eval_expr(expr, env):
    try:
        return expr.eval(env).unsqueeze(0)
    except (AssertionError, AttributeError):
        return T.zeros(B_H, B_W).unsqueeze(0)
    
def track_stats(model, dataloader, envs, max_length=50):
    """
    Track the number of matches out of total (B, P) pairs, and measure distance between guessed bitmaps
    and ground-truth bitmaps.
    
    Track accuracy/distance by program size (number of objects in scene).
    """
    stats = {}
    
    def update(k, n, v):
        if k not in stats:
            stats[k] = {n: 0 for n in range(1, 6)}
        else:
            stats[k][n] += v
    
    def updates(d, n):
        for k, v in d.items():
            update(k, n, v)
    
    for i, (B, P) in enumerate(dataloader):
        sample = model.sample_programs(B, P, max_length)
        e_exprs, o_exprs = sample['expected exprs'], sample['out exprs']
        for j, (e_expr, o_expr) in enumerate(zip(e_exprs, o_exprs)):
            e_lines = get_lines(e_expr)
            o_lines = get_lines(o_expr)
            e_len = len(e_lines)
            o_len = len(o_lines)
            max_len = max(o_len, e_len)
            
            matching = int(o_expr == e_expr)
            prefix = int(util.is_prefix(o_lines, e_lines))
            n_common = len(set.intersection(set(e_lines), set(o_lines)))
            len_diff = o_len - e_len
            o_bmps = T.stack([eval_expr(o_expr, env) for env in envs])
            e_bmps = T.stack([eval_expr(e_expr, env) for env in envs])
            bmps_diff = (o_bmps == e_bmps).logical_not().sum()
            bmps_colorblind_diff = ((o_bmps > 0) == (e_bmps > 0)).logical_not().sum()
            
            updates({
                'n_exprs': 1,
                'matches': matching,
                'prefixes': prefix,
                'overlap': n_common/max_len,
                'len diff': len_diff,
                'bmp color-aware diff': bmps_diff,
                'bmp color-blind diff': bmps_colorblind_diff,
            }, e_len)
            
            if o_expr != e_expr:
                print(i, j)
                print(e_expr)
                print(o_expr)
                print(f'n_common={n_common}/{max_len}, len_diff={len_diff}, '
                      f'bmp diff={bmps_diff}, bmp colorblind diff={bmps_colorblind_diff}')
                print()
                # if bmps_diff > 0 and bmps_nonzero_diff == 0:
                #     viz.viz_sample(xs=e_bmps, ys=o_bmps, text=f'expected (left): {e_expr}, out (right): {o_expr}')

    print("totals:")
    for k in stats.keys():
        print(f'  {k}: {sum(stats[k].values())}')
    print()

    print("by size:")
    for n in range(1, 6):
        print(f'  n={n}:')
        for k in stats.keys():
            print(f'    {k}: {stats[k][n]}')
    print()

def train_models(training_data_loc, test_data_loc, batch_size=64):
    N = 5
    for d_model, learning_rate_exp in it.product([256, 512, 1024], [-4, -5, -6]):
        learning_rate = 10 ** learning_rate_exp
        name = f'{d_model}m{learning_rate}lr'
        model = ArcTransformer(
            name=name,
            lexicon=LEXICON,
            N=N, H=B_H, W=B_W,
            d_model=d_model,
            batch_size=batch_size,
        ).to(device)
        
        # train models
        tloader = model.make_dataloader(lambda: util.load_multi_incremental(training_data_loc), blind=False)
        vloader = model.make_dataloader(lambda: util.load_multi_incremental(test_data_loc), blind=False)
        model.learn(
            tloader=tloader,
            vloader=vloader,
            epochs=1_000_000,
            learning_rate=learning_rate,
            threshold=10 ** -3,
            sample_freq=3,  # epochs btwn samples (bmp/txt)
            log_freq=3,  # hours btwn logs
        )

def test_model(name, d_model, N, batch_size, data_loc, envs_loc, checkpoint_loc):
    model = ArcTransformer(
        name=name,
        lexicon=LEXICON,
        N=N, H=B_H, W=B_W,
        d_model=d_model,
        batch_size=batch_size,
    ).to(device)
    checkpoint = T.load(checkpoint_loc)
    model.load_state_dict(checkpoint['model_state_dict'])
    envs = util.load(envs_loc)['envs']
    dataloader = model.make_dataloader(lambda: util.load_multi_incremental(data_loc))
    track_stats(model, dataloader, envs)


if __name__ == '__main__':
    # TODO: make N flexible - adapt to datasets with variable-size bitmap example sets

    print(f'lexicon: {LEXICON}')
    print(f'Using {dev_name}')
    test_model(
        name='1mil-0z-test',
        d_model=1024, N=5, batch_size=16,
        data_loc='../data/10-1~5r0~1z5e-tf/*.tf.exs',
        checkpoint_loc='../models/tf_model_1mil-1~5r0z1e_123.pt',
        envs_loc='../data/10-1~5r0~1z5e-tf/*.cmps',
    )
    # train_models(
    #     training_data_loc='../data/10000-1~5r0z1e-train*.tf.exs',
    #     test_data_loc='../data/10000-1~5r0z1e-test*.tf.exs'
    # )
    