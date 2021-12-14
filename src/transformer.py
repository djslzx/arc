import torch as T
import torch.nn.functional as F

def self_attn(x):
    raw_weights = T.bmm(x, x.transpose(1,2))
    weights = F.softmax(raw_weights, dim=2)
    y = T.bmm(weights, x)
    return y


class SelfAttention(T.nn.Module):

    def __init__(self, k, heads=8):
        # h attention heads
        # k x k 

        super().__init__()
        self.k, self.heads = k, heads

        self.tokeys    = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
	self.tovalues  = nn.Linear(k, k * heads, bias=False)

	self.unifyheads = nn.Linear(heads * k, k)

    def forward(self, x):
         b, t, k = x.size()
         h = self.heads

         # reshape linear module output (b, t, hk) to (b, t, h, k)
         queries = self.toqueries(x).view(b, t, h, k)
         keys    = self.tokeys(x)   .view(b, t, h, k)
         values  = self.tovalues(x) .view(b, t, h, k)
    


if __name__ == '__main__':
    pass
