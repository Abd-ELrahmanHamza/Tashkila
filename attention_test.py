import torch
import torch.nn as nn
import torch.optim as optim

# Define the toy dataset
input_sequence = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
target_sequence = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11], [12, 13, 14, 15, 0]])

# Define the encoder-decoder model with attention
class AttentionEncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionEncoderDecoder, self).__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input_sequence, target_sequence):
        # Encoder
        embedded_input = self.embedding(input_sequence)
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder_lstm(embedded_input)

        # Decoder with attention
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        loss = 0

        for i in range(target_sequence.size(1)):
            embedded_target = self.embedding(target_sequence[:, i:i+1])
            attention_weights = self.calculate_attention_weights(encoder_output, decoder_hidden)
            attention_vectors = attention_weights * encoder_output
            context_vector = torch.sum(attention_vectors, dim=1, keepdim=True)
            decoder_input = torch.cat((embedded_target, context_vector), dim=2)

            decoder_output, (decoder_hidden, decoder_cell) = self.decoder_lstm(decoder_input, (decoder_hidden, decoder_cell))
            output = self.linear(decoder_output.squeeze(1))
            loss += nn.CrossEntropyLoss()(output, target_sequence[:, i])

        return loss

    def calculate_attention_weights(self, encoder_output, decoder_hidden):
        # encoder output: [batch size, seq len, hidden size]
        # decoder hidden: [1, batch size, hidden size]
        # attention weights: [batch size, seq len, 1]
        decoder_hidden_permuted = decoder_hidden.permute(1, 2, 0)
        attention_weights = torch.bmm(encoder_output, decoder_hidden_permuted)
        attention_weights = self.softmax(attention_weights)
        return attention_weights

# Initialize the model, optimizer, and criterion
input_size = 16
hidden_size = 8
output_size = 16

model = AttentionEncoderDecoder(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1000

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = model(input_sequence, target_sequence)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    test_input = torch.tensor([[1, 2, 3, 4, 5]])
    test_target = torch.tensor([[2, 3, 4, 5, 6]])
    test_loss = model(test_input, test_target)
    print(f'Test Loss: {test_loss.item():.4f}')