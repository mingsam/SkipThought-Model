from typing import Optional, Any

import numpy as np

def freq_revise(vocab):
    word_counts = np.array([count for count in vocab.values()],dtype=np.float32)
    word_freqs = word_counts / sum(word_counts)
    word_freqs = word_freqs ** (3./4.)
    #word_freqs: Optional[Any] = word_freqs / np.sum(word_freqs)
    
    return word_freqs, word_counts