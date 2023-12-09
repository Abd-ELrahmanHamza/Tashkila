import numpy as np
from torch import nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y):
        output_size = y.shape[0]

        encoder_hidden, encoder_cell = self.encoder(x)

        # TODO: check y[0, :] is the correct way to get the first word it should be <SOS>
        decoder_input, decoder_hidden_input, decoder_cell_input = y[0], encoder_hidden, encoder_cell
        decoder_output = self.decoder(y, encoder_hidden, encoder_cell)

        # TODO: check if this is the correct way
        results = np.zeros((output_size, 000000))
        for i in range(1, output_size):
            decoder_output, decoder_hidden_input, decoder_cell_input = self.decoder(decoder_input, decoder_hidden_input, decoder_cell_input)

            results[i] = decoder_output

            # Teacher forcing
            decoder_input = y[i]
        return decoder_output
