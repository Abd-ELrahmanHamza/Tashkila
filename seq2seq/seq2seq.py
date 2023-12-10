import numpy as np
from torch import nn
import torch.nn.functional as F # nn.functional give us access to the activation and loss functions.
from torch.optim import Adam # optim contains many optimizers. This time we're using Adam
import lightning as L # lightning has tons of cool tools that make neural networks easier


class Seq2Seq(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_inputs, decoder_inputs):
        encoder_hidden, encoder_cell = self.encoder(encoder_inputs)
        decoder_output = self.decoder(decoder_inputs, encoder_hidden, encoder_cell)
        return decoder_output
    
    def training_step(self, batch, batch_idx):
        encoder_inputs, decoder_inputs, decoder_targets = batch
        decoder_outputs = self.forward(encoder_inputs, decoder_inputs)
        loss = self.decoder.loss(decoder_outputs.view(-1, decoder_outputs.size(-1)), decoder_targets.view(-1))
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)
    
