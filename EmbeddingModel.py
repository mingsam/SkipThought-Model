
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        initrange = 0.5
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.out_embed.weight.data.uniform_(-initrange, initrange)
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.in_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, inout_labels, near_labels, neg_labels):
        batch_size = inout_labels.size(0)

        input_embedding = self.in_embed(inout_labels)
        near_embedding = self.out_embed(near_labels)
        neg_embedding = self.out_embed(neg_labels)

        log_near = torch.bmm(near_embedding, input_embedding.unsqueeze(2)).squeeze()
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze()

        log_near = F.logsigmoid(log_near).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)

        loss = log_near + log_neg

        return -loss

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()
