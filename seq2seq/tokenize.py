from byte_pair_encoding import Byte_Pair_Encoding
from words_dataset import read_data, find_width_99_percentile


def tokenize():
    input, expected_output = read_data(
        "../clean_out/X.csv", "../clean_out/Y.csv", True)

    # Find the max sentence length
    max_sentence_length = 300

    bpe = Byte_Pair_Encoding(max_sentence_length)
    bpe.train("./clean_out/merged.txt")

    char_input = input.copy()
    tokenized_word_input = input.copy()
    for i in range(len(input)):
        word = ''.join(input[i])
        tokenized_word_input[i] = bpe.encode(word)
    return expected_output, char_input, tokenized_word_input


if __name__ == '__main__':
    expected_output, char_input, tokenized_word_input = tokenize()
    # print("Expected output: ", expected_output)
    # print("Char input: ", char_input)
    print("Tokenized word input: ", tokenized_word_input[0])
