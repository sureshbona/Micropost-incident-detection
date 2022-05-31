from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TweetTokenize(BaseEstimator, TransformerMixin):
    def __init__(self):
        # self.stopwords_eng = set(stopwords.words('english'))
        self.unk = 'unk'
        self.andPattern = re.compile('&amp;')
        self.gtPattern = re.compile('&gt;')
        self.ltPattern = re.compile('&lt;')
        self.tkz = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    def tokenize(self, text):
        text = re.sub(self.andPattern, "and", text)
        text = re.sub(self.gtPattern, ">", text)
        text = re.sub(self.ltPattern, "<", text)

        self.output = []
        tokens = self.tkz.tokenize(text)
        for token in tokens:
            if len(token) > 1:
                if token[0] == '#':
                    self.output.append('#')
                    token = token[1:]
                    #output.append(stemmer.stem(token))
                subtoken = token.split('-')
                if len(subtoken) > 1:
                    for t in subtoken:
                        self.output.append(t)
                else:
                    self.output.append(token)
        return self.output

    def fit(self, X, Y=None):
        self.token_set = (set([token for tweet in X for token in self.tokenize(tweet)]))
        self.t2count = dict.fromkeys(self.token_set, 0)
        for tweet in X:
            for token in self.tokenize(tweet):
                self.t2count[token] = self.t2count[token] + 1
        self.t2pluscount = [k for k, v in self.t2count.items() if v > 1]
        return self

    def transform(self, X):
        self.X = [[t if t in self.t2pluscount else self.unk for t in self.tokenize(tweet)]
                  for tweet in X]

        return self.X
