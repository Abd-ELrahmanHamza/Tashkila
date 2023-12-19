from constants import *

DS_ARABIC_LETTERS: list[str] = ARABIC_LETTERS + \
    [' '] + ARABIC_PUNCTUATIONS + ['.']

NON_HARAKA = '$'

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

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
START_TOKEN = '<s>'
END_TOKEN = '</s>'
