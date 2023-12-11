import numpy as np
import torch
import torch.nn as nn
import lightning as L  # lightning has tons of cool tools that make neural networks easier
from torch.utils.data import TensorDataset, DataLoader  # these are needed for the training data

from encoder import Encoder
from decoder import Decoder
from seq2seq.byte_pair_encoding import BPE
from seq2seq.seq2seq import Seq2Seq
from words_hot_enconding import create_word_index, one_hot_encode
from characters_hot_encoding import hot_encoding, char2idx, idx2char, vocab
from seq2seq.tokenize import tokenize

def preprocess_data():
    expected_output, char_input, word_input = tokenize()
    word_index = create_word_index(word_input)
    word_hot_encoding = {word:one_hot_encode(word, word_index) for word in word_input}
    #TODO: need to perform character hot encoding
    character_hot_encoding = None
    # decoder_inputs = [[char for char in word] for sentence in corpus for word in sentence]
    return expected_output, word_hot_encoding, character_hot_encoding, char_input, word_input

if __name__ == '__main__':
    # Preprocess the data
    encoder_inputs, decoder_inputs = preprocess_data()
    decoder_targets = None # TODO: PROBLEM==== HOW TO GET THE TASHKIL FOR INPUT WORDS
    # 8- Define the hyperparameters
    ENCODER_INPUT_DIM = None  # TODO: replace None with the correct value
    DECODER_INPUT_DIM = None  # TODO: replace None with the correct value
    OUTPUT_DIM = None  # TODO: replace None with the correct value
    HIDDEN_DIM = 256
    EMBEDDING_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 1

    # Define the models
    encoder = Encoder(ENCODER_INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = Decoder(DECODER_INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model = Seq2Seq(encoder, decoder)

    # Define the data loaders encoder, decoder inputs and targets
    dataset = TensorDataset(encoder_inputs, decoder_inputs, decoder_targets)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define the trainer
    trainer = L.Trainer(max_epochs=100, accelerator="auto", devices="auto")

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader)
