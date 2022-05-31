import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
class Padding(BaseEstimator, TransformerMixin):
    def __init__(self, seq_length=None):
        self.seq_length = seq_length
    def fit(self, X, y=None):
        self.embed_dim = len(X[0][0])
        self.seq_length = self.seq_length or int(np.quantile([len(i) for i in X], 0.99))
        return self
    def transform(self, X):
        self.padding_to_embeddings = np.zeros((len(X), self.seq_length, self.embed_dim), dtype=np.float32)
        for i, row in enumerate(X):
            self.padding_to_embeddings[i, :len(row)] = np.array(row)[:self.seq_length]
        return self.padding_to_embeddings
