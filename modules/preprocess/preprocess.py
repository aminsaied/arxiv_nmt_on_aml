import os
import argparse
import random

from utils import DataFrameSelector, TextCleaner, train_valid_test_split

INPUT_ = "Abstract"
OUTPUT_ = "Title"
ARXIV_RAW_DIR = os.path.join(os.path.abspath('.'), 'data', 'arxiv_raw')

MATH_TOKEN = 'MATHEQUATIONTOKEN'

# hyperparameters
MIN_ABSTR = 10
MIN_TITLE = 5

# define transformers

        
# Define arguments
parser = argparse.ArgumentParser(description='Web scraping arg parser')
parser.add_argument('--raw_data_dir', type=str, help='Directory where raw data is stored')
parser.add_argument('--train_proportion', type=float, default=0.8, help='Proportion of data used to train')
parser.add_argument('--train_dir', type=str, help='Directory to output the processed training data')
parser.add_argument('--valid_dir', type=str, help='Directory to output the processed valid data')
parser.add_argument('--test_dir', type=str, help='Directory to output the processed test data')
parser.add_argument('--input_col', type=str, default='Abstract', help='Name of the input data column')
parser.add_argument('--output_col', type=str, default='Title', help='Name of the output data column')
args = parser.parse_args()

# Get arguments from parser
raw_data_dir = args.raw_data_dir
train_proportion = args.train_proportion
train_dir = args.train_dir
valid_dir = args.valid_dir
test_dir = args.test_dir
input_col = args.input_col
output_col = args.output_col

assert 0 <= train_proportion and train_proportion <= 1

# Make train, valid, test directories
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Read raw data
raw_files = [f for f in os.listdir(raw_data_dir) if os.path.isfile(os.path.join(raw_data_dir, f))]

for file in raw_files:
    # load the raw data
    arxiv = pd.read_pickle(file)

    # split the data into train/valid/test sets
    train, valid, test = train_valid_test_split(arxiv, train_proportion)

    # process the text
    names = ['train', 'valid', 'test']
    dirs = [train_dir, valid_dir, test_dir]
    dataframes = [train, valid, test]

    for dir_, name, df in zip(dirs, names, dataframes):
        # clean input/output text
        input_series_raw = df[input_col]
        output_series_raw = df[output_col]
        input_series_clean = TextCleaner.clean(input_series_raw)
        output_series_clean = TextCleaner.clean(output_series_raw)
        
        # write input to txt file line by line
        input_filepath = dir_ + name + input_col.lower()
        with open(input_filepath, 'a+') as file:
            for _, line in input_series_clean.items():
                file.write(line + '\n')

        # write output to txt file line by line
        output_filepath = dir_ + name + output_col.lower()
        with open(output_filepath, 'a+') as file:
            for _, line in output_series_clean.items():
                file.write(line + '\n')