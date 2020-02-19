from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from text_cleaner import TextCleaner

from docopt import docopt

MATH_TOKEN = 'MATHEQTOKEN'  # note this should be alphanumeric

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Returns series from pandas dataframe."""
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.column]


class Cleaner(BaseEstimator, TransformerMixin):
    """Cleans abstracts/titles of unwanted math expressions."""
    def __init__(self, math_replacement=''):
        self.math_replacement = math_replacement

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        f = lambda abstract: TextCleaner.get_clean_words(abstract, math_token=MATH_TOKEN)
        return X.apply(f)


if __name__ == '__main__':
    args = docopt(__doc__)

    print('read in source sentences: %s' % args['--train-src'])
    print('read in target sentences: %s' % args['--train-tgt'])

    src_sents = read_corpus(args['--train-src'], source='src')
    tgt_sents = read_corpus(args['--train-tgt'], source='tgt')

    vocab = Vocab.build(src_sents, tgt_sents, int(args['--size']), int(args['--freq-cutoff']))
    print('generated vocabulary, source %d words, target %d words' % (len(vocab.src), len(vocab.tgt)))

    vocab.save(args['VOCAB_FILE'])
    print('vocabulary saved to %s' % args['VOCAB_FILE'])