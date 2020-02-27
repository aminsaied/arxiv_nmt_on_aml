import sys
import os
import time
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union
import argparse
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

import torch

from nmt import NMT
from utils import read_corpus

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """

    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score

def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_src_dir = os.path.join(args.test_dir, args.input_col.lower())
    test_tgt_dir = os.path.join(args.test_dir, args.output_col.lower())

    print(f"load test source sentences from [{test_src_dir}]", file=sys.stderr)
    test_data_src = read_corpus(test_src_dir, source='src')
    if test_tgt_dir:
        print(f"load test target sentences from [{test_tgt_dir}]", file=sys.stderr)
        test_data_tgt = read_corpus(test_tgt_dir, source='tgt')

    model_path = os.path.join(args.model_dir, 'model.bin')
    print(f"load model from {model_path}", file=sys.stderr)
    model = NMT.load(model_path)

    if args.cuda:
        model = model.to(torch.device("cuda:0"))

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args.beam_size),
                             max_decoding_time_step=int(args.max_decoding_time_step))

    top_hypotheses = [hyps[0] for hyps in hypotheses]
    bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
    print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    output_path = os.path.join(args.eval_dir, 'decode.txt')
    with open(output_path, 'w') as f:
        f.write(str(bleu_score))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set arguments for training NMT model')
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--input_col', type=str)
    parser.add_argument('--output_col', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--eval_dir', type=str)
    parser.add_argument('--cuda', type=bool)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--beam_size', type=int)
    parser.add_argument('--max_decoding_time_step', type=int)

    args = parser.parse_args()

    # seed the random number generators
    seed = int(args.seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if not os.path.exists(args.eval_dir):
        os.makedirs(args.eval_dir)

    decode(args)