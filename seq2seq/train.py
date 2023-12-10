import torch
import torch.nn as nn
import lightning as L # lightning has tons of cool tools that make neural networks easier
from torch.utils.data import TensorDataset, DataLoader # these are needed for the training data

from encoder import Encoder
from decoder import Decoder
from seq2seq.seq2seq import Seq2Seq

# Hyperparameters
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HIDDEN_DIM = 256
EMBEDDING_DIM = 100
NUM_LAYERS = 2
DROPOUT = 1

# Define the models
encoder = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
decoder = Decoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
model = Seq2Seq(encoder, decoder)

# Define the data loaders encoder, decoder inputs and targets
dataset = TensorDataset(encoder_inputs, decoder_inputs, decoder_targets)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the trainer
trainer = L.Trainer(max_epochs=100, accelerator="auto", devices="auto")

# Train the model
trainer.fit(model, train_dataloaders=train_loader)
