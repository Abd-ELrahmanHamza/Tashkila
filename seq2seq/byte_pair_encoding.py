import re
from collections import defaultdict


class BPE:
    def __get_vocab(self, data):
        """
        Given a list of strings, returns a dictionary of words mapping to their frequency
        count in the data.

        Parameters:
        data: list of strings

        Returns:
        dictionary of words mapping to their frequency count in the data

        Example:
        If the input 'data' is ['low', 'lower'], the function returns:
        {'l o w </w>': 1, 'l o w e r </w>': 1}
        """
        vocab = defaultdict(int)
        for line in data:
            for word in line.split():
                vocab[' '.join(list(word)) + ' </w>'] += 1
        return vocab

    def __get_stats(self, vocab):
        """
        Given a vocabulary (a dictionary mapping words to frequency counts), this function returns a dictionary of tuples representing the frequency count of pairs of characters in the vocabulary.

        Parameters:
            vocab (dict[str, int]): A dictionary where keys are words, and values are their frequency counts.

        Returns:
            dict[tuple[str, str], int]: A dictionary where keys are tuples of two characters, and values are the frequency count of those character pairs.

        Example:
            If the input 'vocab' is {'l o w </w>': 5, 'l o w e r </w>': 2}, the function returns:
            {('l', 'o'): 7, ('o', 'w'): 7, ('w', '</w>'): 5, ('w', 'e'): 2, ('e', 'r'): 2, ('r', '</w>'): 2}
        """
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def __merge_vocab(self, pair, v_in):
        """
        Given a pair of characters and a vocabulary, returns a new vocabulary with the
        pair of characters merged together wherever they appear.

        Parameters:
            pair: tuple of two characters
            v_in: dictionary of words mapping to their frequency count in the data

        Returns:
            V_out: dictionary of words mapping to their frequency count in the data with the pair merged

        Example:
            If the input 'pair' is ('e', 'r') and the input 'v_in' is {'l o w </w>': 5, 'l o w e r </w>': 2},
            the function returns:
            {'l o w </w>': 5, 'l o w er </w>': 2}
        """
        v_out = {}
        bigram = re.escape(' '.join(pair))
        #  Negative Lookbehind (?<!\S) => it matches a position where the character before it is a whitespace character or the beginning of the string
        #  Negative Lookahead (?!\S)  => it matches a position where the character after it is a whitespace character or the end of the string
        # \S => non white space [^\s]
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def byte_pair_encoding(self, data, n):
        """
        Given a list of strings and an integer n, returns a list of n merged pairs
        of characters found in the vocabulary of the input data.
        """
        vocab = self.__get_vocab(data)
        for i in range(n):
            pairs = self.__get_stats(vocab)
            best = max(pairs, key=pairs.get)
            vocab = self.__merge_vocab(best, vocab)
        return vocab
