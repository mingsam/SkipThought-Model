
import torch.utils.data as tud
import torch


class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx,idx_to_word, word_freqs, word_counts, vocab_size, window_size, num_neg_words):
        super(WordEmbeddingDataset,self).__init__()
        self.window_size = window_size
        self.num_neg_words = num_neg_words
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_count = torch.Tensor(word_counts)
        self.text_encode = [word_to_idx.get(t, vocab_size) for t in text]
        self.text_encode = torch.Tensor(self.text_encode).long()


    def __len__(self):
        return len(self.text_encode)


    def __getitem__(self, idx):
        center_word = self.text_encode[idx]
        near_pos = list(range(idx-self.window_size,idx))+list(range(idx+1, self.window_size+1))
        near_pos = [i % len(self.text_encode) for i in near_pos]
        near_word = self.text_encode[near_pos]
        neg_words = torch.multinomial(self.word_freqs, self.num_neg_words * near_word.shape[0], True)

        return center_word, near_word, neg_words

