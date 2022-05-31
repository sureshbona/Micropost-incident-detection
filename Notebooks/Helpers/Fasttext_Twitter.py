import numpy as np
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
import fasttext


class FasttextTwitter(BaseEstimator, TransformerMixin):
    def __init__(self, transformTo, seed=42):
        # Takes in a dictionary of words and vectors as input
        self.word2vecDict = None
        self.dimensions = None
        self.unk = 'unk'
        self.transformTo = transformTo
        self.seed = seed
        self.npr_ = np.random.RandomState(self.seed)

    def fit(self, X, y=None):
        self.total_vocabulary = set(word for text in X for word in text)
        self.embeddings_dict = {}
        self.word2vec = fasttext.load_model("../../Datasets/Embeddings/fasttext_twitter/fasttext_twitter_tokens.bin")

        self.dimensions = self.word2vec.get_dimension()

        for word in self.word2vec.words:
            if word == self.unk:
                self.vector = np.zeros(self.dimensions)
            elif word in self.total_vocabulary:
                self.vector = self.word2vec[word]
            else:
                continue
            self.embeddings_dict[word] = self.vector

        self.word2vecDict = self.embeddings_dict

        return self

    def transform(self, X):
        self.uniform_distribution_range = np.sqrt(
                6 / self.dimensions)  # taken from c-bigru paper
        if self.transformTo == "mean":
            return np.array([
                np.mean([self.word2vecDict[w] for w in words if w in self.word2vecDict]
                        or [self.npr_.uniform(low=-self.uniform_distribution_range,
                                      high=self.uniform_distribution_range,
                                      size=self.dimensions)], axis=0) for words in X])

        elif self.transformTo == "concat":
            return np.array([
                [self.word2vecDict[w] for w in words if w in self.word2vecDict]
                or [self.npr_.uniform(low=-self.uniform_distribution_range,
                                      high=self.uniform_distribution_range,
                                      size=self.dimensions)]
                for words in X], dtype=object,)
