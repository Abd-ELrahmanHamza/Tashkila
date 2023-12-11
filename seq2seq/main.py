from byte_pair_encoding import Byte_Pair_Encoding
from letters_dataset import read_data, find_width_99_percentile

if __name__ == "__main__":
    input, expected_output = read_data("./clean_out/X.csv", "./clean_out/Y.csv", True)

    # Find the max sentence length
    max_sentence_length = find_width_99_percentile(input)

    bpe = Byte_Pair_Encoding(max_sentence_length)
    bpe.train("./clean_out/merged.txt")

    char_input = input.copy()
    tokenized_word_input = input.copy()
    for i in range(len(input)):
        tokenized_word_input[i] = bpe.encode(''.join(input[i]))
    print(tokenized_word_input[0])
