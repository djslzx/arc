"""
Implement a two-headed model (value, policy)
"""
import pdb
import math
import pickle
import time
from glob import glob

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, IterableDataset, ChainDataset
import torch.utils.tensorboard as tb
from typing import Optional, Iterable, List, Dict, Callable

import grammar as g
import util

dev_name = 'cuda:0' if T.cuda.is_available() else 'cpu'
device = T.device(dev_name)
print(f"Using {dev_name}")

class SrcEncoding(nn.Module):
    """
    Add learned encoding to identify data sources
    """
    
    def __init__(self, d_model: int, source_sizes: List[int], dropout=0.1):
        super().__init__()
        n_sources = len(source_sizes)
        self.d_model = d_model
        self.source_sizes = source_sizes
        self.dropout = nn.Dropout(p=dropout)
        self.src_embedding = nn.Embedding(n_sources, d_model)
        
    def encoding(self):
        encoding = T.zeros(sum(self.source_sizes), 1, self.d_model).to(device)
        start = 0
        for i, source_sz in enumerate(self.source_sizes):
            encoding[start:start + source_sz, :] = self.src_embedding(T.tensor([i]).to(device))
            start += source_sz
        return encoding
        
    def forward(self, x):
        return self.dropout(x + self.encoding()[:x.size(0)])


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
    
    PADDING = "PADDING"
    PROGRAM_START = "PROGRAM_START"
    PROGRAM_END = "PROGRAM_END"
    LINE_START = "("
    LINE_END = ")"
    
    def __init__(self,
                 name: str,
                 N: int, H: int, W: int, lexicon: List[str],
                 d_model=512,
                 n_conv_layers=6, n_conv_channels=16,
                 n_tf_encoder_layers=6, n_tf_decoder_layers=6,
                 n_value_heads=3, n_value_ff_layers=6,
                 max_program_length=50, max_line_length=15,
                 batch_size=32, save_dir='.'):
        super().__init__()
        self.name = name
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
        self.max_line_length = max_line_length
        self.max_program_length = max_program_length
        self.batch_size = batch_size
        self.save_dir = save_dir
        
        # program embedding: embed program tokens as d_model-size tensors
        self.lexicon        = lexicon + ["PROGRAM_START", "PROGRAM_END", "PADDING"]
        self.n_tokens       = len(self.lexicon)
        self.token_to_index = {s: i for i, s in enumerate(self.lexicon)}
        self.PADDING        = self.token_to_index[Model.PADDING]
        self.PROGRAM_START  = self.token_to_index[Model.PROGRAM_START]
        self.PROGRAM_END    = self.token_to_index[Model.PROGRAM_END]
        self.LINE_START     = self.token_to_index[Model.LINE_START]
        self.LINE_END       = self.token_to_index[Model.LINE_END]
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
        self.src_encoding = SrcEncoding(d_model=d_model,
                                        source_sizes=[self.N, self.N, self.max_program_length])
        self.tf_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8),
                                                num_layers=self.n_tf_encoder_layers)
        
        # policy head: receives transformer encoding of the triple (B, B', f') and generates an embedding
        self.tf_decoder = nn.TransformerDecoder(decoder_layer=nn.TransformerDecoderLayer(d_model=self.d_model, nhead=8),
                                                num_layers=self.n_tf_decoder_layers)
        self.tf_out = nn.Linear(self.d_model, self.n_tokens)

        # value head: receives code from tf_encoder and maps to evaluation of program
        value_layers = [
            nn.Linear((2 * self.N + self.max_program_length) * self.d_model,
                      self.d_model),
            nn.ReLU(),
        ]
        assert self.n_value_ff_layers >= 1
        for _ in range(self.n_value_ff_layers - 1):
            value_layers.extend([
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
            ])
        self.value_fn = nn.Sequential(
            *value_layers,
            nn.Linear(self.d_model, self.n_value_heads)
        )

    def model_path(self, epoch):
        return f'{self.save_dir}/model_{self.name}_{epoch}.pt'

    def load_model(self, checkpoint_loc: str):
        checkpoint = T.load(checkpoint_loc, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])

    def to_index(self, token):
        return self.token_to_index[token]

    def to_indices(self, tokens: Iterable, add_markers=False):
        indices = [self.to_index(t) for t in tokens]
        if add_markers:
            indices = [self.PROGRAM_START] + indices + [self.PROGRAM_END]
        return T.tensor(indices)

    def to_token(self, index):
        return self.lexicon[index]

    def to_tokens(self, indices: Iterable):
        return [self.to_token(i) for i in indices]

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
        programs = f.to(device)
        f_embeddings = T.stack([self.p_embedding(p) for p in programs]).transpose(0, 1)
        return f_embeddings

    def forward(self, b, b_hat, p_hat, delta):
        # compute embeddings for bitmaps, programs
        batch_size = b.shape[0]
        e_b = self.embed_bitmaps(b, batch_size)
        e_b_hat = self.embed_bitmaps(b_hat, batch_size)
        e_p_hat = self.pos_encoding(self.embed_programs(p_hat))
        e_delta = self.pos_encoding(self.embed_programs(delta))

        # concat e_B, e_B', e_p', add source encoding, then pass through tf_encoder
        src = T.cat((e_b, e_b_hat, e_p_hat))
        src = self.src_encoding.forward(src)
        tf_code = self.tf_encoder.forward(src)

        # pass tf_code through v and pi -> value, policy
        padding_mask = T.eq(delta, self.PADDING).to(device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(sz=e_delta.shape[0]).to(device)
        policy = self.tf_out(self.tf_decoder.forward(tgt=e_delta, memory=tf_code,
                                                     tgt_mask=tgt_mask,
                                                     tgt_key_padding_mask=padding_mask))

        # flatten tf_code and pass into value fn
        flat_tf_code = tf_code.transpose(0, 1).flatten(start_dim=1)
        value = self.value_fn.forward(flat_tf_code)
        
        return policy, value

    @staticmethod
    def word_loss(expected, actual):
        """
        x = log_softmax(actual, -1) : turn logits into probabilities
        x = (x * expected)          : pull out values from `actual` at nonzero locations in `expected`
        x = T.sum(x, -1)            : pull out nonzero values
        x = T.sum(x, 0)             : take sum of log-probabilities for each example in the batch
        x = T.mean(x)               : compute mean probability of correctly generating each sequence in the batch
        x = -x                      : minimize loss (-mean) to maximize mean pr
        """
        log_prs = T.sum(F.log_softmax(actual, dim=-1) * expected, dim=-1)
        return -T.mean(T.sum(log_prs, dim=0))

    def pretrain_policy(self, tloader: DataLoader, vloader: DataLoader, epochs: int, lr=10 ** -4,
                        assess_freq=10_000, checkpoint_freq=1, vloss_gap=1, tloss_thresh=0, vloss_thresh=0):
        """
        Pretrain the policy network, exiting when
        (a) the validation or training loss reaches the correctness threshold, or
        (b) the validation loss creeps upwards of `exit_dist_from_min` away from its lowest point.
        """
        self.to(device)
        optimizer = T.optim.Adam(self.parameters(), lr=lr)
        writer = tb.SummaryWriter(comment=f'_{self.name}')

        self.train()
        t_start = time.time()
        step, tloss, vloss = 0, 0, 0
        checkpoint = 1
        min_vloss = T.inf
        assess_freq = min(assess_freq, len(tloader))  # bound assess_freq by dataloader size
        training_complete = False
        for epoch in range(epochs):
            round_tloss = 0
            for i, (B, B_hat, P_hat, D) in enumerate(tloader, 1):
                step = epoch * assess_freq + i
                
                # batch dim first, seq-len dim second in dataloader
                D_in = D[:, 1:]
                D_out = D[:, :-1]
                policy_out, value_out = self.forward(b=B, b_hat=B_hat, p_hat=P_hat, delta=D_in)
                expected = self.to_probabilities(D_out).transpose(0, 1)  # [seq-len, batch, lexicon-size]
                loss = self.word_loss(expected=expected, actual=policy_out)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                round_tloss += loss.item()
        
                # assess and record current performance
                if i % assess_freq == 0:
                    tloss = round_tloss / assess_freq
                    vloss = self.pretrain_validate(vloader, n_steps=assess_freq); self.train()
                    if vloss < min_vloss: min_vloss = vloss
                    
                    # record losses
                    print(f" [{step}]: training loss={tloss:.3f}, validation loss={vloss:.3f},"
                          f" {time.time() - t_start:.2f}s elapsed")
                    writer.add_scalar('training loss', tloss, step)
                    writer.add_scalar('validation loss', vloss, step)
                    round_tloss = 0
            
                    # check exit conditions
                    if vloss > min_vloss + vloss_gap or tloss <= tloss_thresh or vloss <= vloss_thresh:
                        training_complete = True
                        break
            
                # write a checkpoint
                if (time.time() - t_start) // (3600 * checkpoint_freq) > checkpoint:
                    T.save({'step': step,
                            'model_state_dict': self.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'training loss': tloss,
                            'validation loss': vloss,
                            }, self.model_path(step))
                    checkpoint += 1
            
            # TODO: sample from model
            if training_complete: break
        
        path = self.model_path(step)
        print(f"Finished training at step {step} with tloss={tloss}, vloss={vloss}. Saving to {path}...")
        T.save({
            'step': step,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training loss': tloss,
            'validation loss': vloss,
        }, path)
            
    def to_probabilities(self, indices):
        return F.one_hot(indices, num_classes=self.n_tokens).float()
    
    def pretrain_validate(self, dataloader, n_steps=None):
        """Compute validation loss of the policy network on the (validation data) dataloader."""
        self.eval()
        epoch_loss = 0
        if n_steps is None: n_steps = len(dataloader)
        for i, (B, B_hat, P_hat, D) in zip(range(n_steps), dataloader):
            D_in = D[:, 1:]
            D_out = D[:, :-1]
            p, v = self.forward(b=B, b_hat=B_hat, p_hat=P_hat, delta=D_in)
            expected_d = self.to_probabilities(D_out).transpose(0, 1)
            loss = self.word_loss(expected_d, p)
            epoch_loss += loss.detach().item()
        return epoch_loss/n_steps

    @staticmethod
    def strip(tokens):
        start = int(tokens[0] == Model.PROGRAM_START)  # skip start token if present
        end = None                                     # end at last token by default
        for i, token in enumerate(tokens):
            if token == Model.PROGRAM_END:
                end = i
                break
        return tokens[start:end]
        
    def to_program(self, indices, default=None) -> Optional[g.Expr]:
        tokens = self.to_tokens(indices)
        try:
            return g.deserialize(Model.strip(tokens))
        except (AssertionError, IndexError):
            return default

    def eval_program(self, indices, envs=None) -> T.Tensor:
        if envs is None:
            envs = g.seed_libs(self.N)
        else:
            assert len(envs) == self.N, \
                f'When evaluating a program on multiple environments, ' \
                f'the number of environments should be the same as N, ' \
                f'the number of bitmaps rendered per program.'
        program = self.to_program(indices, default=g.Seq())
        bitmaps = []
        for env in envs:
            try:
                bitmap = program.eval(env)
            except AssertionError:
                bitmap = T.zeros(self.H, self.W)
            bitmaps.append(bitmap.unsqueeze(0))  # add channel dimension
        return T.stack(bitmaps)
    
    def append_delta(self, indices: T.tensor, delta: T.Tensor) -> T.Tensor:
        """Append a 'delta' (a new line) to a program."""
        p_program = self.to_program(indices, default=g.Seq())
        p_delta = self.to_program(delta)
        print(f'p_program: {p_program}')
        print(f'p_delta: {p_delta}')
        if p_delta is None:
            p_out = p_program
        else:
            p_out = p_program.add_line(p_delta)
        return self.to_indices(p_out.serialize())
    
    def sample_line_rollouts(self, b, b_hat, p_hat):
        print("Sampling line rollouts...")
        self.eval()
        batch_size = b.shape[0]
        rollouts = T.tensor([[self.PROGRAM_START]] * batch_size).long().to(device)  # [b, 1]
        for i in range(self.max_line_length):
            p_outs, _ = self.forward(b=b, b_hat=b_hat, p_hat=p_hat, delta=rollouts)  # [i, b, n_tokens]
            p_prs = p_outs.softmax(-1)  # convert p_outs to probabilities -> [i, b]
            next_indices = T.multinomial(util.filter_top_p(p_prs[-1]), 1)
            rollouts = T.cat((rollouts, next_indices), dim=1)
        return rollouts
    
    def sample_program_rollouts(self, bitmaps):
        """Take samples from the model wrt a (batched) set of bitmaps"""
        print("Sampling program rollouts...")
        self.eval()
        batch_size = bitmaps.shape[0]
        rollouts = T.stack([self.to_indices(['{', '}'], add_markers=True)] * batch_size).to(device)
        print(f"rollouts: {rollouts}")
        while all(len(rollout) <= self.max_program_length for rollout in rollouts):
            rollout_renders = T.stack([self.eval_program(rollout) for rollout in rollouts])
            padded_rollouts = T.stack([util.pad(rollout, self.max_program_length, self.PADDING)
                                       for rollout in rollouts])
            deltas = self.sample_line_rollouts(b=bitmaps, b_hat=rollout_renders, p_hat=padded_rollouts)
            print(f"deltas: {deltas}")
            rollouts = T.stack([self.append_delta(r, d) for r, d in zip(rollouts, deltas)])
            print(f"rollouts: {rollouts}")
        return rollouts
    
    def make_policy_dataloader(self, data_src: str):
        """
        Make a dataloader for policy network pre-training. (batching, tokens -> indices, add channel dim, etc)
        """
        dataset = PolicyDataset(data_src,
                                to_indices=lambda tokens: self.to_indices(tokens, add_markers=True),
                                padding=self.PADDING,
                                program_length=self.max_program_length,
                                line_length=self.max_line_length)
        return DataLoader(dataset, batch_size=self.batch_size)


class PolicyDataset(IterableDataset):

    def __init__(self, src_glob: str, to_indices: Callable[[List], T.Tensor],
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
                        bitmaps, partial_bitmaps, f_prefix_tokens, delta = pickle.load(fp)
                        # add channel dimension to bitmaps
                        b = bitmaps.unsqueeze(1).to(device)
                        b_hat = partial_bitmaps.unsqueeze(1).to(device)
                        # convert program tokens into padded vectors of indices
                        f_hat = util.pad(self.to_indices(f_prefix_tokens), self.program_length, self.padding).to(device)
                        d = util.pad(self.to_indices(delta), self.line_length, self.padding).to(device)
                        yield b, b_hat, f_hat, d
                    except EOFError:
                        break


if __name__ == '__main__':
    prefix = '../data/policy-pretraining'
    data_code = '10-RLP-5e1~3l0~2z'
    model_code = data_code
    data_t = 'Apr13_22_11-35-52'
    model_t = 'Apr13_22_11-41-54'
    # model_t = util.now_str()

    model = Model(name=f'{model_code}_{model_t}', N=5, H=g.B_H, W=g.B_W, lexicon=g.SIMPLE_LEXICON,
                  d_model=512, n_conv_layers=6, n_conv_channels=12,
                  n_tf_encoder_layers=6, n_tf_decoder_layers=6,
                  n_value_heads=1, n_value_ff_layers=2,
                  max_program_length=50, batch_size=16,
                  save_dir='../models').to(device)
    
    print("Making dataloaders...")
    tloader = model.make_policy_dataloader(f'{prefix}/{data_code}/training_{data_t}/joined_deltas.dat')
    vloader = model.make_policy_dataloader(f'{prefix}/{data_code}/validation_{data_t}/joined_deltas.dat')

    print("Pretraining policy....")
    epochs = 100
    model.pretrain_policy(tloader=tloader, vloader=vloader, epochs=epochs)

    # print("Loading trained policy...")
    # steps = 160
    # model.load_model(f'../models/model_{model_code}_{model_t}_{steps}.pt')
    vloss = model.pretrain_validate(vloader, n_steps=1000)
    print(f"Model loaded. Validation loss={vloss}")

    # sample rollouts
    sample_B, sample_B_hat, sample_P_hat, D = iter(tloader).next()
    print("Sampling rollouts...")
    print(f"sample_B: {sample_B.shape}")
    print(model.sample_program_rollouts(sample_B))
    
