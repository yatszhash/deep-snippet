from keras.preprocessing import text, sequence
from sklearn.base import BaseEstimator, TransformerMixin


class KerasTextTokenizer(BaseEstimator, TransformerMixin):

    def __init__(self, num_words):
        self.tokenizer = text.Tokenizer(num_words=num_words)

    def fit(self, x):
        self.tokenizer.fit_on_texts(list(x))
        return self

    def transform(self, x):
        return self.tokenizer.texts_to_sequences(x)

    def fit_transform(self, x, y=None, **fit_params):
        return self.fit(x).transform(x)


class KerasTextPadding(BaseEstimator, TransformerMixin):

    def __init__(self, maxlen, dtype='int32', padding='pre', truncating='pre', value=0.0):
        self.maxlen = maxlen
        self.dtype = dtype
        self.padding = padding
        self.truncating = truncating
        self.value = value

    def fit(self, x):
        return self

    def transform(self, x):
        return sequence.pad_sequences(x, maxlen=self.maxlen,
                                      dtype=self.dtype, padding=self.padding,
                                      truncating=self.truncating, value=self.value)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)
