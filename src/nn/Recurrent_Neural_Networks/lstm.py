# LSTM – Solving vanishing gradient
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, hidden):
        h, c = hidden
        gates = self.fc(torch.cat([x, h], dim=1))
        i, f, g, o = gates.chunk(4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next