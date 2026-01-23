# Recurrent Neural Networks (RNNs)
# Simple RNN (Vanilla)
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.Wx = nn.Linear(input_size, hidden_size)
        self.Wh = nn.Linear(hidden_size, hidden_size)
        self.Wy = nn.Linear(hidden_size, output_size)

    def forward(self, seq):  # seq: [T, B, input_size]
        h = torch.zeros(seq.size(1), self.Wh.out_features)
        outputs = []
        for t in range(seq.size(0)):
            h = torch.tanh(self.Wx(seq[t]) + self.Wh(h))
            y = self.Wy(h)
            outputs.append(y)
        return torch.stack(outputs)