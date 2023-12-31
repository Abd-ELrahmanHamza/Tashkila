from constants import *
import pickle

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
START_TOKEN = '<s>'
END_TOKEN = '</s>'


def read_pickle_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


DS_ARABIC_LETTERS: list[str] = ARABIC_LETTERS + \
    [' ', '،', '-']

NON_HARAKA = ''

DS_HARAKAT = [
    FATHA,
    DAMMA,
    KASRA,
    SUKUN,
    FATHATAN,
    DAMMATAN,
    KASRATAN,
    SHADDA,
    SHADDA + FATHA,
    SHADDA + DAMMA,
    SHADDA + KASRA,
    SHADDA + FATHATAN,
    SHADDA + DAMMATAN,
    SHADDA + KASRATAN,
    NON_HARAKA
]
harakat2id = read_pickle_file('./delivery/diacritic2id.pickle')
