"""
Implement a two-headed model (value, policy)
"""
import pdb
import math
import time

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.tensorboard as tb

import grammar as g
import util

dev_name = 'cuda:0' if T.cuda.is_available() else 'cpu'
device = T.device(dev_name)

class SrcEncoding(nn.Module):
    """
    Add learned encoding to identify data sources
    """
    
    def __init__(self, d_model: int, source_sizes: list[int], dropout=0.1):
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
            encoding[start:start + source_sz, :] = self.src_embedding(T.tensor([i]))
            start += source_sz
        return encoding
        
    def forward(self, x):
        return self.dropout(x + self.encoding())


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
    
    def __init__(self,
                 name: str,
                 N: int, H: int, W: int, lexicon: list[str],
                 d_model=512,
                 n_conv_layers=6, n_conv_channels=16,
                 n_tf_encoder_layers=6, n_tf_decoder_layers=6,
                 n_value_heads=3, n_value_ff_layers=6,
                 max_program_length=50,
                 batch_size=32):
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
        self.max_program_length = max_program_length
        self.batch_size = batch_size
        
        # program embedding: embed program tokens as d_model-size tensors
        self.lexicon        = lexicon + ["PROGRAM_START", "PROGRAM_END", "PADDING", "LINE_END"]
        self.n_tokens       = len(self.lexicon)
        self.token_to_index = {s: i for i, s in enumerate(self.lexicon)}
        self.PADDING        = self.token_to_index["PADDING"]
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
        return f'model_{self.name}_{epoch}.pt'

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
        # pass B, B' through CNN -> e_B, e_B'
        # compute embeddings for bitmaps, programs
        batch_size = b.shape[0]
        e_b = self.embed_bitmaps(b, batch_size)
        e_b_hat = self.embed_bitmaps(b_hat, batch_size)
        e_p_hat = self.pos_encoding(self.embed_programs(p_hat))
        e_delta = self.pos_encoding(self.embed_programs(delta))

        # concat e_B, e_B', e_p', add source encoding, then pass through tf_encoder
        src = self.src_encoding(T.cat((e_b, e_b_hat, e_p_hat)))
        tf_code = self.tf_encoder.forward(src)

        # pass tf_code through v and pi -> value, policy
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(sz=e_delta.shape[0])
        padding_mask: T.Tensor = T.eq(delta, self.PADDING)
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
        x = T.sum(x, -1)            : remove zeros
        x = T.sum(x, 0)             : take sum of log-probabilities for each example in the batch
        x = T.mean(x)               : compute mean probability of correctly generating each sequence in the batch
        x = -x                      : minimize loss (-mean) to maximize mean pr of
        """
        log_prs = T.sum(F.log_softmax(actual, dim=-1) * expected, dim=-1)
        return -T.mean(T.sum(log_prs, dim=0))

    def pretrain_policy(self, tloader: DataLoader, vloader: DataLoader, epochs: int,
                        lr=10 ** -4, correctness_threshold=0, exit_dist_from_min=1,
                        epochs_per_sample=10, hours_between_checkpoints=1):
        """
        Pretrain the policy network, exiting when
        (a) the validation or training loss reaches the correctness threshold, or
        (b) the validation loss creeps upwards of `exit_dist_from_min` away from its lowest point.
        """
        self.to(device)
        optimizer = T.optim.Adam(self.parameters(), lr=lr)
        writer = tb.SummaryWriter(comment=f'_{self.name}')

        epoch, tloss, vloss = 0, 0, 0
        checkpoint = 1
        min_vloss = T.inf
        t_start = time.time()
        for epoch in range(1, epochs + 1):
            t_epoch_start = time.time()
            tloss = self.pretrain_epoch_tl(tloader, optimizer)
            vloss = self.pretrain_epoch_vl(vloader)
            t_epoch_end = time.time()
            
            if vloss < min_vloss: min_vloss = vloss

            t_epoch = t_epoch_end - t_epoch_start
            t_total = t_epoch_end - t_start
            print(f'training loss: {tloss:.3f}, validation loss: {vloss:.3f}; '
                  f'took {t_epoch:.2f}s, {t_total:.2f}s total')
            writer.add_scalar('training loss', tloss, epoch)
            writer.add_scalar('validation loss', vloss, epoch)
            
            # write a checkpoint every `hours_between_checkpoints` hours
            if (t_epoch_end - t_start)//(3600 * hours_between_checkpoints) > checkpoint:
                T.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training loss': tloss,
                    'validation loss': vloss,
                }, self.model_path(epoch))
                checkpoint += 1
                
            # TODO: sample from model
            if min(tloss, vloss) <= correctness_threshold or \
               vloss >= min_vloss + exit_dist_from_min: break
        
        print(f"Finished training at epoch {epoch} with tloss={tloss}")
        T.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training loss': tloss,
            'validation loss': vloss,
        }, self.model_path(epoch))
            
    def to_probabilities(self, indices):
        return F.one_hot(indices, num_classes=self.n_tokens).float()
    
    def pretrain_epoch_tl(self, dataloader, optimizer):
        """Pretrain the policy network."""
        self.train()
        epoch_loss = 0
        for i, (B, B_hat, P_hat, D) in enumerate(dataloader):
            optimizer.zero_grad()
            # batch dim first, seq-len dim second in dataloader
            policy_out, value_out = self.forward(b=B, b_hat=B_hat, p_hat=P_hat, delta=D)
            expected = self.to_probabilities(D).transpose(0, 1)
            loss = self.word_loss(expected=expected, actual=policy_out)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        return epoch_loss/len(dataloader)
    
    def pretrain_epoch_vl(self, dataloader):
        """Compute validation loss of the policy network on the (validation data) dataloader."""
        self.eval()
        epoch_loss = 0
        for (B, B_hat, P_hat, D) in dataloader:
            p, v = self.forward(b=B, b_hat=B_hat, p_hat=P_hat, delta=D)
            expected_d = self.to_probabilities(D).transpose(0, 1)
            epoch_loss += self.word_loss(expected_d, p).detach().item()
        return epoch_loss/len(dataloader)
        
    def to_indices(self, tokens):
        def to_index(token):
            if token == '|':
                return self.LINE_END
            else:
                return self.token_to_index[token]
        return T.tensor([self.PROGRAM_START] + [to_index(t) for t in tokens] + [self.PROGRAM_END])
    
    @staticmethod
    def pad(v, padded_length: int, padding_value: int):
        """Pad v to length `padded_length`."""
        return F.pad(v, pad=(0, padded_length - len(v)), value=padding_value)
    
    def make_policy_dataloader(self, data_src: str, blind: bool = False):
        """
        Make a dataloader for policy network pre-training. (batching, tokens -> indices, add channel dim, etc)
        """
        B, B_hat, P_hat, D = [], [], [], []
        for (bitmaps, partial_bitmaps, f_prefix_tokens, delta) in util.load_incremental(data_src):
            # process bitmaps
            if blind:
                bitmaps = T.zeros(self.N, 1, self.H, self.W)
                partial_bitmaps = T.zeros(self.N, 1, self.H, self.W)
            else:
                bitmaps = bitmaps.unsqueeze(1)  # add channel dimension
                partial_bitmaps = partial_bitmaps.unsqueeze(1)
            B.append(bitmaps)
            B_hat.append(partial_bitmaps)

            # process programs: turn tokens into tensors of indices, add padding
            delta_indices = self.pad(self.to_indices(delta), self.max_program_length, self.PADDING)
            p_prefix_indices = self.pad(self.to_indices(f_prefix_tokens), self.max_program_length, self.PADDING)
            D.append(delta_indices)
            P_hat.append(p_prefix_indices)

        dataset = TensorDataset(T.stack(B).to(device), T.stack(B_hat).to(device),
                                T.stack(P_hat).to(device), T.stack(D).to(device))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


if __name__ == '__main__':
    model = Model(name='test', N=5, H=g.B_H, W=g.B_W, lexicon=g.LEXICON,
                  d_model=512, n_conv_layers=6, n_conv_channels=12,
                  n_tf_encoder_layers=6, n_tf_decoder_layers=6,
                  n_value_heads=1, n_value_ff_layers=2,
                  max_program_length=50, batch_size=16).to(device)
    prefix = '../data/policy-pretraining'
    code = '10-RLP-5e1~3l0~1z'
    tloader = model.make_policy_dataloader(f'{prefix}/{code}-training-exs.dat')
    vloader = model.make_policy_dataloader(f'{prefix}/{code}-validation-exs.dat')
    model.pretrain_policy(tloader=tloader, vloader=vloader, epochs=100)
    