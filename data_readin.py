import numpy as np
from collections import Counter


# tokenize函数，把一篇文本转化成一个个单词
def word_tokenize(text):
    return text.split()


# 数据读取
def data_readin(path, num_words):
    with open(path, "r") as fin:
        text = fin.read()

    text = [w for w in word_tokenize(text.lower())]
    vocab = dict(Counter(text).most_common(num_words - 1))
    vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
    idx_to_word = [word for word in vocab.keys()]
    word_to_idx = {word: i for word, i in enumerate(idx_to_word)}

    return text, vocab, idx_to_word, word_to_idx
