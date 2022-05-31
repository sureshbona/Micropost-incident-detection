import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class GloveTwitterEmbeddings(BaseEstimator, TransformerMixin):
    def __init__(self, transformTo, seed=42):
        # Takes in a dictionary of words and vectors as input
        self.word2vec = None
        self.dimensions = None
        self.transformTo = transformTo
        self.seed = seed
        self.npr_ = np.random.RandomState(self.seed)

    def fit(self, X, y=None):
        total_vocabulary = set(word for text in X for word in text)
        # len(total_vocabulary)
        embeddings_dict = {}
        with open("../../Datasets/Embeddings/Glove_Twitter_27B/glove.twitter.27B.200d.txt", 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in total_vocabulary:
                    vector = np.asarray(values[1:], "float32")
                    embeddings_dict[word] = vector
        self.word2vec = embeddings_dict
        self.dimensions = len(self.word2vec[next(iter(self.word2vec))])
        return self

    def transform(self, X):
        if self.transformTo == "mean":
            return np.array([
                np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                        or [self.npr_.randn(self.dimensions)], axis=0) for words in X])

        elif self.transformTo == "concat":
            self.uniform_distribution_range = np.sqrt(
                6 / self.dimensions)  # taken from c-bigru paper
            return np.array([
                [self.word2vec[w] for w in words if w in self.word2vec]
                or [self.npr_.uniform(low=-self.uniform_distribution_range,
                                      high=self.uniform_distribution_range,
                                      size=self.dimensions)]
                for words in X], dtype=object,)
