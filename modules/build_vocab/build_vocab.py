import argparse

from utils import read_corpus
from vocab import Vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess text data arg parser')
    parser.add_argument('--train_dir', type=str, help='Directory where training data is stored')
    parser.add_argument('--input_col', type=str, help='Input column name for training data')
    parser.add_argument('--output_col', type=str, help='Output column name for training data')
    parser.add_argument('--vocab_dir', type=str, help='Directory to output the vocab')
    parser.add_argument('--size', type=int, default=50000, help='Vocab size')
    parser.add_argument('--freq_cutoff', type=int, default=2, help='Frequency cutoff')
    args = parser.parse_args()

    train_src = os.path.join(args.train_dir, args.input_col.lower())
    train_tgt = os.path.join(args.train_dir, args.output_col.lower()) 

    print('read in source sentences: %s' % train_src)
    print('read in target sentences: %s' % train_tgt)

    src_sents = read_corpus(train_src, source='src')
    tgt_sents = read_corpus(train_tgt, source='tgt')

    vocab = Vocab.build(src_sents, tgt_sents, int(args.size), int(args.freq_cutoff))
    print('generated vocabulary, source %d words, target %d words' % (len(vocab.src), len(vocab.tgt)))

    vocab.save(args.vocab_dir)
    print('vocabulary saved to %s' % args.vocab_dir)
