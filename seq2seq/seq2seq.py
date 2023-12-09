import numpy as np
from torch import nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y):
        encoder_hidden, encoder_cell = self.encoder(x)
        decoder_output = self.decoder(y, encoder_hidden, encoder_cell)
        return decoder_output
