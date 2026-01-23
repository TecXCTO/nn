# Attention
# Multi‑Head Attention (simplified)
class SimpleAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1,2).contiguous().view(B, T, -1)
        return self.W_o(out)