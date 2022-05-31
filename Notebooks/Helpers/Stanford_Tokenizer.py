import stanza
from sklearn.base import BaseEstimator, TransformerMixin

class StanfordTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_pretokenized=True, tokenize_no_split=True, verbose=False)

    def tokenize(self, text):
      doc = self.nlp(text)
      for i, sentence in enumerate(doc.sentences):
          tokens = [token.text for token in sentence.tokens]
          return tokens

    def fit(self, X=None, Y=None):
        return self

    def transform(self, X):
        X = X.apply(self.tokenize)
        return X
