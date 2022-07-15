"""
Implement a two-headed model (value, policy)
"""
# import pdb
import math
import pickle
import time
from glob import glob
# import matplotlib.pyplot as plt

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import torch.utils.tensorboard as tb
from typing import Optional, Iterable, List, Callable, Tuple
from collections import namedtuple

# import arc_data
import grammar as g
import util
import viz

dev_name = 'cuda:0' if T.cuda.is_available() else 'cpu'
dev = T.device(dev_name)
print(f"Using {dev_name}")

PADDING = "PADDING"
PROGRAM_START = "PROGRAM_START"
PROGRAM_END = "PROGRAM_END"


class SrcEncoding(nn.Module):
    """Add learned encoding to identify data sources."""

    def __init__(self, d_model: int, source_sizes: List[int], dropout=0.1):
        super().__init__()
        n_sources = len(source_sizes)
        self.d_model = d_model
        self.source_sizes = source_sizes
        self.dropout = nn.Dropout(p=dropout)
        self.src_embedding = nn.Embedding(num_embeddings=n_sources,
                                          embedding_dim=d_model)

    def encoding(self):
        encoding = T.zeros(sum(self.source_sizes), 1, self.d_model).to(dev)
        start = 0
        for i, source_sz in enumerate(self.source_sizes):
            encoding[start:start + source_sz, :] = self.src_embedding(T.tensor([i]).to(dev))
            start += source_sz
        return encoding

    def forward(self, x):
        # need to cut off end b/c program might not be full length
        return self.dropout(x + self.encoding()[:x.size(0)])


class PositionalEncoding(nn.Module):
    """Positional encoding from 'Attention is All You Need'."""

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


Rollout = namedtuple('Result', 'indices completed well_formed')


class Model(nn.Module):

    def __init__(self,
                 name: str,
                 N: int, H: int, W: int,
                 Z_LO: int, Z_HI: int, lexicon: List[str],
                 C=10,
                 d_model=512,
                 n_conv_channels=16,
                 n_tf_encoder_layers=6,
                 n_tf_decoder_layers=6,
                 n_value_heads=3,
                 max_program_length=203,
                 max_line_length=17):
        super().__init__()
        self.name = name

        # grammar/bitmap parameters
        self.N = N        # number of rendered bitmaps per sample
        self.H = H        # height (in pixels) of each bitmap
        self.C = C        # color channel (one-hot)
        self.W = W        # width of each bitmap
        self.Z_LO = Z_LO  # minimum value of z
        self.Z_HI = Z_HI  # max value of z

        # model hyperparameters
        self.d_model = d_model
        self.n_conv_channels = n_conv_channels
        self.n_tf_encoder_layers = n_tf_encoder_layers
        self.n_tf_decoder_layers = n_tf_decoder_layers
        self.n_value_heads = n_value_heads
        self.max_line_length = max_line_length
        self.max_program_length = max_program_length

        # program embedding: embed program tokens as d_model-size tensors
        # lexicon includes both the program alphabet and the range of valid z's
        self.lexicon        = [PADDING, PROGRAM_START, PROGRAM_END, g.Z_IGNORE] + lexicon
        self.n_tokens       = len(self.lexicon)
        self.token_to_index = {s: i for i, s in enumerate(self.lexicon)}
        self.PADDING        = self.token_to_index[PADDING]
        self.PROGRAM_START  = self.token_to_index[PROGRAM_START]
        self.PROGRAM_END    = self.token_to_index[PROGRAM_END]
        self.Z_IGNORE       = self.token_to_index[g.Z_IGNORE]
        self.pos_encoding   = PositionalEncoding(self.d_model)
        self.p_embedding    = nn.Embedding(num_embeddings=self.n_tokens, embedding_dim=self.d_model)
        
        src_sizes = [N, N, max_program_length]
        
        # bitmap embedding: convolve HxW bitmaps into d_model-size tensors
        def conv_block(in_channels: int, out_channels: int, kernel_size=3) -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        self.conv = nn.Sequential(
            conv_block(self.C, n_conv_channels),
            conv_block(n_conv_channels, n_conv_channels),
            conv_block(n_conv_channels, n_conv_channels),
            conv_block(n_conv_channels, n_conv_channels),
            conv_block(n_conv_channels, n_conv_channels),
            conv_block(n_conv_channels, n_conv_channels),
            nn.Flatten(),
            nn.Linear(n_conv_channels * H * W, d_model),  # [..., C', H, W] --> [..., d_model]
        )
        # Add embedding to track source of different tensors passed into the transformer encoder
        self.src_encoding = SrcEncoding(d_model, src_sizes)
        self.tf_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=8),
                                                num_layers=n_tf_encoder_layers)
        
        # policy heads: receive transformer encoding of the triple (B, B', f') and
        # generate a program embedding + random params
        self.tf_decoder = nn.TransformerDecoder(decoder_layer=nn.TransformerDecoderLayer(d_model=d_model, nhead=8),
                                                num_layers=n_tf_decoder_layers)
        # take the d_model-dimensional vector output from the decoder and map it to a probability distribution over
        # the program/parameter alphabet
        self.tf_out = nn.Linear(self.d_model, self.n_tokens)
        
        # feed in z's appended to the end of each program

        # value head: receives code from tf_encoder and maps to evaluation of program
        def lin_block(in_features: int, out_features: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU(),
            )
        self.value_fn = nn.Sequential(
            lin_block(sum(src_sizes) * self.d_model, self.d_model),
            lin_block(self.d_model, self.d_model),
            lin_block(self.d_model, self.d_model),
            nn.Linear(self.d_model, self.n_value_heads)
        )

    def model_path(self, dir, epoch):
        return f'{dir}/model_{self.name}_{epoch}.pt'

    def load_model(self, checkpoint_loc: str):
        checkpoint = T.load(checkpoint_loc, map_location=dev)
        self.load_state_dict(checkpoint['model_state_dict'])

    def to_index(self, token):
        return self.token_to_index[token]

    def to_indices(self, tokens: Iterable, decorate=False) -> T.Tensor:
        indices = [self.to_index(t) for t in tokens]
        if decorate:
            indices = [self.PROGRAM_START] + indices + [self.PROGRAM_END]
        return T.tensor(indices).to(dev)

    def to_token(self, index):
        return self.lexicon[index]

    def to_tokens(self, indices: Iterable):
        return [self.to_token(i) for i in indices]

    def embed_bitmaps(self, bmps, batch_size):
        """
        Map the batched bitmap set b (of shape [batch, H, W])
        to bitmap embeddings (of shape [N, batch, d_model])
        """
        # add channel dim: [b, n, h, w] -> [b, n, c, h, w]
        bmps_w_channel = util.add_channels(bmps.long()).float()
        # flatten: [b, n, c, h, w] -> [b * n, c, h, w]
        bmps_flat = bmps_w_channel.reshape(-1, self.C, self.H, self.W)  # flatten bitmap collection
        # apply conv, batch: [b * n, c, h, w] -> [b * n, d_model] -> [b, n, d_model]
        bmps_enc = self.conv(bmps_flat).reshape(batch_size, -1, self.d_model)
        # swap batch and sequence dims [b, n, d_model] -> [n, b, d_model]
        bmps_enc = bmps_enc.transpose(0, 1)
        return bmps_enc

    def forward(self, f_bmps, p_bmps, p, d):
        # compute embeddings for bitmaps, programs
        batch_size = f_bmps.shape[0]  # need this b/c last batch might not be the full size
        e_f_bmps = self.embed_bitmaps(f_bmps, batch_size)
        e_p_bmps = self.embed_bitmaps(p_bmps, batch_size)
        e_p = self.pos_encoding(self.p_embedding(p).transpose(0, 1))
        e_d = self.pos_encoding(self.p_embedding(d).transpose(0, 1))

        # concat e_B, e_B', e_p', add source encoding, then pass through tf_encoder
        src = T.cat((e_f_bmps, e_p_bmps, e_p))
        src = self.src_encoding.forward(src)
        tf_code = self.tf_encoder.forward(src)

        # pass tf_code through v and pi -> value, policy
        padding_mask = T.eq(d, self.PADDING).to(dev)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(sz=e_d.shape[0]).to(dev)

        # pass in delta tokens to transformer as tgt
        policy = self.tf_out(self.tf_decoder.forward(tgt=e_d,
                                                     memory=tf_code,
                                                     tgt_mask=tgt_mask,
                                                     tgt_key_padding_mask=padding_mask))

        # flatten tf_code and pass into value fn
        flat_tf_code = tf_code.transpose(0, 1).flatten(start_dim=1)
        value = self.value_fn.forward(flat_tf_code)
        
        return policy, value

    @staticmethod
    def word_loss(target, output):
        """
        x = log_softmax(actual, -1) : turn logits into probabilities
        x = (x * expected)          : pull out values from `actual` at nonzero locations in `expected`
        x = T.sum(x, -1)            : pull out nonzero values
        x = T.sum(x, 0)             : take sum of log-probabilities over tokens for each example in the batch
        x = T.mean(x)               : compute mean probability of correctly generating each sequence in the batch
        x = -x                      : minimize loss (-mean) to maximize mean pr
        """
        log_prs = T.sum(F.log_softmax(output, dim=-1) * target, dim=-1)
        return -T.mean(T.sum(log_prs, dim=0))

    def program_params_loss(self, target, output, mask):
        """
        fz: f cat f_envs
        pz: p cat p_envs
        """
        envs_sz = self.N * g.LIB_SIZE - 1  # -1 to account for shifting
        p_sz = target.shape[0] - envs_sz
        
        # split target/output into program/env
        target_p, target_z = target.split((p_sz, envs_sz))
        output_p, output_z = output.split((p_sz, envs_sz))

        # compute losses separately
        f_loss = Model.word_loss(target_p, output_p)
        z_loss = Model.word_loss(target_z * mask, output_z * mask)
        return f_loss, z_loss
    
    def env_mask(self, envs):
        """
        Take the natural mask for envs (mask out tokens eq to Z_IGNORE), then stretch it to cover
        params represented as probabilities
        """
        mask: T.Tensor = (envs != self.Z_IGNORE)[:, 1:]  # compensate for shifting - match tgt_out
        mask = mask.unsqueeze(-1) * T.ones(1, self.n_tokens).to(dev)
        mask = mask.transpose(0, 1)
        return mask
    
    def stretched_env_mask(self, envs):
        """
        Take the natural mask for envs (mask out tokens eq to Z_IGNORE), then stretch it to cover
        deltas that include params, represented as probabilities
        """
        batch = envs.shape[0]
        mask: T.Tensor = (envs != self.Z_IGNORE)[:, 1:]  # compensate for shifting - match tgt_out
        # [b, N*L - 1] => [|d| + N*L - 1, b, n_tokens]
        mask = T.cat((T.ones(batch, self.max_line_length).to(dev), mask), dim=1)  # [b, |d| + N*L - 1]
        mask = mask.unsqueeze(-1) * T.ones(1, self.n_tokens).to(dev)  # [b, |d| + N*L - 1, n_toks]
        mask = mask.transpose(0, 1)  # [|d| + N*L - 1, b, n_toks]
        return mask
    
    def pretrain_policy(self,
                        save_dir: str,
                        tloader: DataLoader, vloader: DataLoader,
                        epochs: int, lr=10 ** -4,
                        assess_freq=10_000, checkpoint_freq=100_000,
                        tloss_thresh: float = 0, vloss_thresh: float = 0,
                        check_vloss_gap=True, vloss_gap: float = 1):
        """
        Pretrain the policy network, exiting when
        (a) the validation or training loss reaches the correctness threshold, or
        (b) the validation loss creeps upwards of `exit_dist_from_min` away from its lowest point.
        """
        T.save('test', self.model_path(save_dir, 0))  # Test save_dir early to make sure it'll work
        assess_freq = min(assess_freq, len(tloader))  # bound assess_freq by dataloader size

        self.to(dev)
        optimizer = T.optim.Adam(self.parameters(), lr=lr)
        writer = tb.SummaryWriter(comment=f'_{self.name}_pretrain')

        self.train()
        t_start = time.time()
        step, tloss, vloss = 0, 0, 0
        min_vloss = T.inf
        training_complete = False
        for epoch in range(epochs):
            round_tloss = 0
            for (p, p_bmps), (f, f_bmps), d in tloader:
                step += 1
                
                d_in = d[:, :-1]
                d_out = d[:, 1:]
                policy_out, value_out = self.forward(f_bmps, p_bmps, p, d_in)
                target = self.to_probabilities(d_out).transpose(0, 1)  # [seq-len, batch, lexicon-size]
                loss = self.word_loss(target, policy_out)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                round_tloss += loss.item()
        
                # assess and record current performance
                if step % assess_freq == 0:
                    tloss = round_tloss / assess_freq
                    vloss = self.pretrain_validate(vloader, n_examples=assess_freq)
                    self.train()
                    if vloss < min_vloss: min_vloss = vloss
                    
                    # record losses
                    print(f" [step={step}, epoch={epoch}]: "
                          f"training loss={tloss:.5f}, validation loss={vloss:.5f}, "
                          f" {time.time() - t_start:.2f}s elapsed")
                    writer.add_scalar('training loss', tloss, step)
                    writer.add_scalar('validation loss', vloss, step)
                    round_tloss = 0
            
                    # check exit conditions (train for at least one epoch)
                    if epoch > 0 and ((check_vloss_gap and vloss > min_vloss + vloss_gap)
                                      or tloss <= tloss_thresh or vloss <= vloss_thresh):
                        training_complete = True
                        break
            
                # write a checkpoint
                if step % checkpoint_freq == 0:
                    print(f"Logging checkpoint at step {step}...")
                    T.save({'step': step,
                            'model_state_dict': self.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'training loss': tloss,
                            'validation loss': vloss,
                            }, self.model_path(save_dir, step))
            
            # TODO: sample from model during training and record using tensorboard
            if training_complete: break
        
        path = self.model_path(save_dir, step)
        print(f"Finished training at step {step} with tloss={tloss}, vloss={vloss},"
              f"Saving to {path}...")
        T.save({
            'step': step,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training loss': tloss,
            'validation loss': vloss,
        }, path)
        
    def to_probabilities(self, indices):
        return F.one_hot(indices, num_classes=self.n_tokens).float()
    
    def pretrain_validate(self, dataloader, n_examples=None):
        """Compute validation loss of the policy network on the (validation data) dataloader."""
        self.eval()
        epoch_loss = 0
        n_examples = len(dataloader) if n_examples is None else min(n_examples, len(dataloader))
        i = 0
        total_toks = 0
        for (p, p_bmps), (f, f_bmps), d in dataloader:
            i += 1
            if i > n_examples: break
            d_in = d[:, :-1]
            d_out = d[:, 1:]
            policy_out, _ = self.forward(f_bmps, p_bmps, p, d_in)
            target = self.to_probabilities(d_out).transpose(0, 1)
            loss = self.word_loss(target, policy_out)
            epoch_loss += loss.detach().item()
            
            # record lengths of programs observed in dataloader
            total_toks += T.ne(f, self.PADDING).sum()
        
        avg_toks = total_toks / (n_examples * dataloader.batch_size)
        avg_lines = (avg_toks - 4)/6  # remove START, END, {, }; each rect takes 6 tokens (R, color, 2 corners)
        print(f"Validation loss computed with {avg_toks:.2f} tokens per example, "
              f"or about {avg_lines:.2f} lines")
        return epoch_loss / n_examples

    def estimate_value(self, delta, f_bmps, p_bmps, p, f, batch_size):
        """
        (1) I[f' is a prefix of f]
        (2) min{g: f'g ~ f} |g|
           i.e., minimum number of lines that need to be added to f' to yield the same behavior as f
        (3) log P(B|f')
            P(B|f') = (sum_{b in B} log sum_{z in Z} I[f'(z) = b]) - |B| log |Z|
        (4) min{ d1..dk st f'd1..dk ~ f} - log P(d1 | B, f', B'(f'))
         - log P(d2 | B, f'd1, B'(f'd1))
         - log P(d3 | B, f'd1d2, B'(f'd1d2)) ...
         i.e., maximize probability of deltas being the right lines to add
        (5) pixel difference between b and b_hat
        """
        # new_fs = []
        # for f, d in zip(f_hat, delta):
        #     p = self.append_delta(f, d).indices
        #     pp = self.pad_program(p)
        #     new_fs.append(pp)
        # new_fs = T.stack(new_fs)
        v = [
            [self.is_prefix(x, y) for x, y in zip(p, f)],
            [self.min_lines(x, y) for x, y in zip(p, f)],
            # [self.approx_likelihood(x, y) for x, y in zip(b, f_hat)],
            # [self.approx_likelihood(b, new_fs)],
        ]  # [n_heads, b]
        return T.tensor(v + [[0] * batch_size] * (self.n_value_heads - len(v))).transpose(0, 1).to(dev)
    
    def is_prefix(self, f_hat, f):
        return float(T.equal(f_hat, f[:f_hat.shape[0]]))
    
    def min_lines(self, f_hat, f):
        try:
            p = self.to_program(f)
            p_hat = self.to_program(f_hat)
            return len(p.lines()) - len(p_hat.lines())
        except AssertionError:
            return -1

    def approx_likelihood(self, b: T.Tensor, f_hat: T.Tensor, n_samples=1000) -> float:
        """
        Approximate P(B | f'), where
          P(B | f') = (sum_{b in B} log sum_{z in Z} I[f'(z) = b]) - |B| log |Z|
          
        NB: This is always 0 for programs f_hat that are unable to generate bitmaps b
        """
        # log |Z| = log(|[z_lo, z_hi]| ** lib_size) = lib_size * log(|[z_lo, z_hi]|) ~= 8^10 = 2^30 ~= 10^9
        log_Z = g.LIB_SIZE * math.log(g.Z_HI - g.Z_LO + 1)
        prior_term = self.N * log_Z  # |B| log |Z|
    
        # take fixed number of samples (~1000?) to approximate log sum_{z in Z} I[f'(z) = b] for each b in B
        i_term = 0
        # print(f'f_hat={f_hat}, {self.to_tokens(f_hat)}, log z={log_Z}, n_samples={n_samples}')
        for bitmap in b:
            s = 1  # to avoid taking log of 0
            for sample in range(n_samples):
                f_hat_render = self.render(f_hat, envs=g.seed_envs(self.N))
                s += int(T.equal(f_hat_render, bitmap))
            # log sum_z I[f'(z) = b] ~= log (sum_{i in 1..k, z_i random} I[f'(z_i) = b] * |Z|/k)
            # = log (sum ..) + log |Z| - log k
            i_term += math.log(s) + log_Z - math.log(n_samples)
        return i_term - prior_term

    def value_loss(self, target, actual) -> float:
        """Compute loss btwn expected delta and actual delta (delta_hat)"""
        target[:, 1] *= target[:, 0]  # if f_hat is not a prefix of f, ignore min_lines prediction
        criterion = nn.MSELoss()
        return criterion(actual, target)

    def train_value(self, dataloader: DataLoader, epochs: int, lr=10 ** -4,
                    assess_freq=10_000, checkpoint_freq=100_000):
        self.train()
        optimizer = T.optim.Adam(self.parameters(), lr=lr)
        writer = tb.SummaryWriter(comment=f'_{self.name}_value')
        
        step = 0
        t_start = time.time()
        for epoch in range(epochs):
            epoch_loss = 0
            for (p, p_bmps), (f, f_bmps), d in dataloader:
                step += 1
                # TODO: sample multiple rollouts per input in policy dataloader
                # TODO: modify delta sampling so we can get multiple rollouts per instance in each batch
                # pdb.set_trace()
                lines = self.sample_line_rollouts(f_bmps, p_bmps, p).to(dev)  # [b, l]
                # new_fs = [self.append_delta(f, d).indices for f, d in zip(f_hat, lines)]  # [b, l]
                # new_bs = T.stack([self.render(f, k=10) for f in new_fs])
        
                # TODO: use rollouts instead of dataset values
                value_expected = self.estimate_value(lines, f_bmps, p_bmps, p, f)
                _, value_out = self.forward(f_bmps, p_bmps, p, lines)
                loss = self.value_loss(target=value_expected, actual=value_out)
                
                print(f"loss: {loss}")
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
                # TODO: logging
            print(f"Epoch loss = {epoch_loss}, {time.time() - t_start:.2f}s elapsed")
    
    @staticmethod
    def strip(tokens):
        start = int(tokens[0] == PROGRAM_START)  # skip start token if present
        end = None                                     # end at last token by default
        for i, token in enumerate(tokens):
            if token == PROGRAM_END:
                end = i
                break
        return tokens[start:end]
        
    def to_program(self, indices) -> Optional[g.Expr]:
        tokens = Model.strip(self.to_tokens(indices))
        try:
            return g.deserialize(tokens)
        except (AssertionError, IndexError):
            return None

    def render(self, indices, envs=None, k=1, check_env_size=True) -> T.Tensor:
        def try_render(f) -> Optional[T.Tensor]:
            try:
                return f.eval(env)
            except AssertionError:
                return None

        if envs is None:
            envs = g.seed_envs(self.N * k)
        elif check_env_size:
            assert len(envs) >= self.N, \
                f'When evaluating a program on multiple environments, ' \
                f'the number of environments should be the same as N, ' \
                f'the number of bitmaps rendered per program, but got ' \
                f'len(envs)={len(envs)}, while self.N={self.N}.'
        n_envs = len(envs)
        program = self.to_program(indices)
        if program is None: program = g.Seq()

        bitmaps = []
        for env in envs:
            if (bitmap := try_render(program)) is not None:
                bitmaps.append(bitmap)
                if len(bitmaps) == n_envs:
                    break
        # add blank bitmaps until we reach the quota
        if len(bitmaps) < n_envs:
            bitmaps += [T.zeros(self.H, self.W)] * (n_envs - len(bitmaps))

        assert(len(bitmaps) == n_envs)
        return T.stack(bitmaps).to(dev)
        
    def sample_line_rollouts(self, f_bmps, p_bmps, p):
        self.eval()
        batch_size = f_bmps.shape[0]
        rollouts = T.tensor([[self.PROGRAM_START]] * batch_size).long().to(dev)  # [b, 1]
        for i in range(self.max_line_length - 1):
            p_outs, _ = self.forward(f_bmps, p_bmps, p, rollouts)  # [i, b, n_tokens]
            next_prs = p_outs.softmax(-1)[-1]  # convert p_outs to probabilities -> [i, b]
            next_indices = T.multinomial(util.filter_top_p(next_prs), 1)
            rollouts = T.cat((rollouts, next_indices), dim=1)
        return rollouts

    def append_delta(self, indices: T.Tensor, delta_indices: T.Tensor, line_cap=4) -> Rollout:
        """
        Append a 'delta' (a new line) to a program.
        """
        p = self.to_program(indices)
        # cap length of p by number of lines
        if p is None:
            return Rollout(indices=indices, completed=False, well_formed=False)
        if len(p.lines()) >= line_cap:
            return Rollout(indices=indices, completed=True, well_formed=True)
        delta = self.to_program(delta_indices)
        if delta is None:
            return Rollout(indices=indices, completed=True, well_formed=False)
        if delta == g.Seq():  # a bit hacky: use the empty program to denote an empty delta
            # TODO: add an end-program token?
            return Rollout(indices=indices, completed=True, well_formed=True)
        return Rollout(indices=self.to_indices(p.add_line(delta).serialize()), completed=False, well_formed=True)

    def pad_program(self, indices: T.Tensor) -> T.Tensor:
        return util.pad(indices, self.max_program_length, self.PADDING).to(dev)
    
    def sample_program_rollouts(self, bitmaps: T.Tensor, line_cap=4) -> List[T.Tensor]:
        """Take samples from the model wrt a (batched) set of bitmaps"""
        self.eval()
        batch_size = bitmaps.shape[0]
        # track whether a rollout is completed in tandem with the rollout data
        rollouts: List[Rollout] = [Rollout(indices=self.to_indices(['{', '}'], decorate=True),
                                           completed=False,
                                           well_formed=True)
                                   for _ in range(batch_size)]
        while any(not r.completed for r in rollouts):
            rollout_renders = T.stack([self.render(r.indices) for r in rollouts])
            padded_rollouts = T.stack([self.pad_program(r.indices) for r in rollouts])
            deltas = self.sample_line_rollouts(f_bmps=bitmaps, p_bmps=rollout_renders, p=padded_rollouts)
            rollouts = [self.append_delta(r.indices, delta, line_cap=line_cap)
                        if r.well_formed and not r.completed else r
                        for r, delta in zip(rollouts, deltas)]
        return [r.indices for r in rollouts]
    
    def make_policy_dataloader(self, data_src: str, batch_size: int):
        """
        Make a dataloader for policy network pre-training. (batching, tokens -> indices, add channel dim, etc)
        """
        dataset = PolicyDataset(data_src,
                                to_indices=self.to_indices,
                                padding=self.PADDING,
                                program_length=self.max_program_length,
                                line_length=self.max_line_length)
        return DataLoader(dataset, batch_size=batch_size)


class PolicyDataset(IterableDataset):

    def __init__(self, src_glob: str, to_indices: Callable[[List, bool], T.Tensor],
                 padding: int, program_length: int, line_length: int):
        super(PolicyDataset).__init__()
        self.files = util.shuffled(glob(src_glob))  # shuffle file order bc IterableDatasets can't be shuffled
        assert len(self.files) > 0, f"Found empty glob: {src_glob} => {self.files}"
        self.to_indices = to_indices
        self.padding = padding
        self.program_length = program_length
        self.line_length = line_length
        self.length = None
        
    def __len__(self):
        if self.length is None:
            n_objs = 0
            for file in self.files:
                with open(file, 'rb') as fp:
                    while True:
                        try:
                            pickle.load(fp)
                            n_objs += 1
                        except EOFError:
                            break
            self.length = n_objs
        return self.length
    
    def __iter__(self):
        for file in self.files:
            with open(file, 'rb') as fp:
                while True:
                    try:
                        (p_toks, p_bmps), (f_toks, f_bmps), d_toks = pickle.load(fp)
                        f_bmps = f_bmps.to(dev)
                        p_bmps = p_bmps.to(dev)
                        
                        # # convert envs into vectors of length N * lib_size
                        # p_envs_indices = self.to_indices([util.unwrap_tensor(z)
                        #                                   for env in p_envs for z in env['z']], False).to(dev)
                        # f_envs_indices = self.to_indices([util.unwrap_tensor(z)
                        #                                   for env in f_envs for z in env['z']], False).to(dev)
                        
                        # convert program tokens into padded vectors of indices
                        p_indices = util.pad(self.to_indices(p_toks, True), self.program_length, self.padding).to(dev)
                        f_indices = util.pad(self.to_indices(f_toks, True), self.program_length, self.padding).to(dev)
                        d_indices = util.pad(self.to_indices(d_toks, True), self.line_length, self.padding).to(dev)

                        yield (p_indices, p_bmps), (f_indices, f_bmps), d_indices

                    except EOFError:
                        break

def sample_model_on_policy_data(model: Model, dataloader: DataLoader):
    for (p, p_bmps), (f, f_bmps), d in dataloader:
        batch_size = f.size(0)
        print(f[0])
        print(model.to_tokens(f[0]))
        n_lines = len(model.to_program(f[0]).lines())
        rollouts = model.sample_program_rollouts(f_bmps, line_cap=n_lines)
        envs = g.seed_envs(model.N ** 2)
        print(f'envs={envs}')
        # batched_envs = [[{'z': t} for t in batch.split(g.LIB_SIZE)]
        #                 for batch in f_envs]
        current_bmps = None
        for i in range(batch_size):
            output = model.to_program(rollouts[i])
            expected = model.to_program(f[i])
            print(f'expected={expected}, actual={output}')
            
            in_bmps = f_bmps.cpu()[i]
            if current_bmps is not None and T.equal(in_bmps, current_bmps):
                continue
            else:
                current_bmps = in_bmps
            
            out_bmps = model.render(rollouts[i], envs=envs, check_env_size=False).cpu()
            bmps = T.cat((in_bmps, out_bmps)).reshape(-1, model.N, model.H, model.W)
            text = f'expected={expected}\n'\
                   f'actual={output}'
            viz.viz_grid(bmps, text)

def sample_model_on_bitmaps(model: Model, bitmaps: Iterable[T.Tensor], line_cap=8):
    for bitmap in bitmaps:
        # reshape bitmaps -> [b=1, N, H, W, C]
        bitmap_stack = T.stack([util.pad_mat(bitmap, h=model.H, w=model.W, padding_token=0)
                                for _ in range(model.N)]).unsqueeze(0).to(dev)
        rollout = model.sample_program_rollouts(bitmap_stack, line_cap=line_cap)[0]
        envs = g.seed_envs(3 ** 2 - 1)
        out_program = model.to_program(rollout)
        out_bitmaps = model.render(rollout, envs=envs, check_env_size=False).cpu()
        bmps = T.cat((bitmap.unsqueeze(0), out_bitmaps)).reshape(3, 3, model.H, model.W)
        viz.viz_grid(bmps, text=out_program)
        
def make_model(model_code: str, lexicon: List[str]) -> Model:
    return Model(
        name=model_code,
        N=5, H=g.B_H, W=g.B_W,
        Z_HI=g.Z_HI, Z_LO=g.Z_LO,
        lexicon=lexicon,
        d_model=512,
        n_conv_channels=12,
        n_tf_encoder_layers=6,
        n_tf_decoder_layers=6,
        n_value_heads=5,
        max_program_length=53, # FIXME: this should be longer to account b/c we added more items
        max_line_length=17
    ).to(dev)

def recover_model(code: str, lexicon: List[str], directory: str, n_steps: int,
                  test_with: Optional[Tuple[str, int]] = None) -> Model:
    """
    Loads in a pretrained model.
    :param code: Identifies the model.
    :param lexicon: The lexicon used by the model
    :param directory: The directory where the model is stored.
    :param n_steps: The number of steps the model was trained
    :param test_with: Optional: If provided, will be used to test the recovered
                      model on a few examples (100).
                      This is a tuple (src, batch_size) that specifies the
                      filename and batch size of the dataloader to make.
    :return: The pretrained model.
    """
    model = make_model(code, lexicon)
    model.load_model(f'{directory}/model_{code}_{n_steps}.pt')
    if test_with:
        data_src, batch_size = test_with
        dataloader = model.make_policy_dataloader(data_src, batch_size)
        loss = model.pretrain_validate(dataloader, n_examples=100)
        print(f"Model loaded. Loss={loss}")
    return model


if __name__ == '__main__':
    # n = 10
    # model = recover_model(
    #     code='50k-R-5e1~20l0z',
    #     lexicon=g.SIMPLE_LEXICON,
    #     directory='../models',
    #     n_steps=200_000,
    #     test_with=(f'../data/10-R-5e{n}l0~5z/july15/train/deltas_*.dat', 16)
    # )

    # # test on ARC data
    # bitmaps = arc_data.task_bitmaps(arc_data.SEQ_FEASIBLE_TASK_NAMES)
    # for cap in [3, 6, 9]:
    #     sample_model_on_bitmaps(model, bitmaps, line_cap=cap)

    # pretraining model
    # data_dir = '/home/djl328/arc/data'
    data_dir = '../data'
    # model_dir = '/home/djl328/arc/models'
    model_dir = '../models'
    data_code = '10-R-5e6l0~5z'
    data_t = 'july15'
    model = make_model(
        model_code='CS,S,SR,LL-5e6l0~5z',
        lexicon=g.SIMPLE_LEXICON,
    )
    model.pretrain_policy(
        save_dir=model_dir,
        tloader=model.make_policy_dataloader(
            f'{data_dir}/{data_code}/{data_t}/train/deltas_*.dat',
            batch_size=32
        ),
        vloader=model.make_policy_dataloader(
            f'{data_dir}/{data_code}/{data_t}/test/deltas_*.dat',
            batch_size=32
        ),
        epochs=100_000,
        lr=10 ** -5,
        assess_freq=10_000,
        checkpoint_freq=100_000,
        tloss_thresh=10 ** -4,
        vloss_thresh=10 ** -3,
        check_vloss_gap=False,
        # vloss_gap=2
    )

    # # training value function
    # model.train_value(
    #     dataloader=model.make_policy_dataloader(f'..', batch_size=32),
    #     epochs=100_000,
    #     lr=10 ** -5,
    #     assess_freq=10_000,
    #     checkpoint_freq=100_000
    # )
