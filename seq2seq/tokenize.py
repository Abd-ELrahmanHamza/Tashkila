from seq2seq.byte_pair_encoding import Byte_Pair_Encoding
from letters_dataset import read_data, find_width_99_percentile

def tokenize():
    input, expected_output = read_data("./clean_out/X.csv", "./clean_out/Y.csv", True)
    max_word_len = 0
    for word in input:
        if len(word) > max_word_len:
            max_word_len = len(word)
    Padding_token = "P" * max_word_len

    # Find the max sentence length
    max_sentence_length = find_width_99_percentile(input)

    bpe = Byte_Pair_Encoding(max_sentence_length,Padding_token)
    bpe.train("./clean_out/merged.txt")

    char_input = input.copy()
    tokenized_word_input = input.copy()
    for i in range(len(input)):
        word = ''.join(input[i])+"P"*(max_word_len-len(input[i]))
        tokenized_word_input[i] = bpe.encode(word)
    return expected_output, char_input, tokenized_word_input