import torch.nn as nn
from .nat_structgnn_model import NATStructGNN_Model
from modules import TransformerLayer


class NATGraphTrans_Model(NATStructGNN_Model):
    def __init__(self, args):
        NATStructGNN_Model.__init__(self, args)

        layer = TransformerLayer
        self.encoder_layers = nn.ModuleList([
            layer(self.hidden_dim, self.hidden_dim*2, dropout=args.dropout)
            for _ in range(args.num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([
            layer(self.hidden_dim, self.hidden_dim*2, dropout=args.dropout)
            for _ in range(args.num_decoder_layers)])
        self.W_out = nn.Linear(self.hidden_dim, args.vocab_size, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)