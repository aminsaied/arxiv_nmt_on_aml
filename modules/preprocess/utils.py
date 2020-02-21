#!/usr/bin/env python3
"""Used to process abstracts and titles of papers.
"""
import re

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Returns series from pandas dataframe."""
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.column]

class TextCleaner(BaseEstimator, TransformerMixin):
    """Cleans abstracts/titles in preparation for NMT model."""
    def __init__(self, column):
        fname = column.lower() + ".txt"
        self.path = os.path.join(os.path.abspath('.'), 'data', 'arxiv', fname)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        f = lambda abstract: self.clean(abstract, math_token=MATH_TOKEN)
        X_cleaned = X.apply(f)
        
        # write series to txt file line by line
        with open(self.path, 'a') as f:
            for _, line in X_cleaned.items():
                f.write(line + '\n')
                
        return X_cleaned
    
    @classmethod
    def clean(cls, series):
        """Apply text cleaning to a pandas series.
        """
        f = lambda raw_text: cls._clean_text(raw_text)
        return series.apply(f)

    @classmethod
    def _clean_text(cls, text, math_token='MATHEQUATIONTOKEN'):
        """
        Removes math and non-alpha characters
        (including tabs and new line symbols) from the text.
        Input:
          -- text : string
        Output:
          string, 'cleaned' text
        """
        replaced_math = cls._strip_math(text, math_token)
        clean_text = cls._strip_non_alphas(replaced_math)
        return clean_text

    @staticmethod
    def _strip_math(text, math_token=MATH_TOKEN):
        """
        Replaces all the LaTeX math in a given text with given symbol.
        Input:
          -- text : string
        Output:
          string : text with math removed
        """
        replace_dollar_signs = re.sub( r'\$+[^\$]+\$+', math_token, text)
        replace_math = re.sub( r'\\\[[^\]]+\]', math_token, replace_dollar_signs)
        return replace_math

    @staticmethod
    def _strip_non_alphas(text):
        """
        Removes non-letter symbols, replacing them with spaces.
        Note: also removes tabs and new line symbols.
        Input:
          -- text : string
        Output:
          string : text with all non-letter symbols removed
        """
        # remove newline symbol '\n'
        no_newline_pattern = text.replace('\n', ' ')
        
        # remove apostophy
        no_apos_pattern = re.compile('[\']+')
        no_apos = re.sub(no_apos_pattern, '', no_newline_pattern)
        
        # replace dash with space
        no_dash_apos = no_apos.replace('-', ' ')
        
        # remove non-alpha characters - keeping periods
        noalpha_pattern = re.compile('[^a-zA-Z\\s\.]+')
        noalpha = re.sub(noalpha_pattern, '', no_dash_apos)
        
        # pad period with whitespace
        periodpadded = noalpha.replace('.', ' . ')
        
        # remove multiple whitespace
        nomultiplewhitespace = ' '.join(periodpadded.split())
        
        # remove repeated periods
        no_multiple_period_pattern = re.compile('(\s*\.\s*){2,}')
        return re.sub(no_multiple_period_pattern, ' . ', nomultiplewhitespace)


def train_valid_test_split(df, train_proportion):
    """Splits the dataframe into train/valid/test.

    Train will consist of `train_proportion` many of the rows, valid and test
    split the rest evenly.
    """

    train_size = int(train_proportion * len(df))
    valid_size = (len(df) - train_size) // 2

    train = df[:train_size]
    valid = df[train_size:train_size + valid_size]
    test = df[train_size + valid_size:]

    return train, valid, test