import torch.nn as nn


class MLPLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, k_neighbors=30):
        super(MLPLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.k_neighbors = k_neighbors
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(k_neighbors, 1, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.ReLU()

        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )

    def forward(self, h_V, h_E, edge_idx, batch_id=None):
        B, D = h_V.shape
        if h_E.shape[0] // B != self.k_neighbors:
            return h_V
        reshaped_h_E = h_E.reshape(B, -1, self.k_neighbors)
        h_E = self.act(self.W1(reshaped_h_E).squeeze(-1))
        h_E = self.act(self.W11(h_E))
        dh = self.W3(self.act(self.W2(h_V + h_E)))
        h_V = self.norm1(h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout(dh))
        return h_V