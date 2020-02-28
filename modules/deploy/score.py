import json
import os
import torch
from collections import namedtuple
from azureml.core.model import Model

from nmt import NMT
from text_cleaner import TextCleaner

BEAM_SIZE = 5
MAX_DECODING_TIME_STEP = 70
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

def read_input_data(input_sentence):
    processed_input_sentence = TextCleaner._clean_text(input_sentence)
    return processed_input_sentence.strip().split(' ')

def beam_search(model, input_sent, beam_size, max_decoding_time_step):
    hypotheses = []
    with torch.no_grad():
        hypothesis = model.beam_search(input_sent, beam_size, max_decoding_time_step)
    return hypothesis

def init():
    global model
    model_dir = Model.get_model_path('arxiv-nmt-pipeline')
    model_path = os.path.join(model_dir, 'model.bin')
    model = NMT.load(model_path)
    model.eval()
    
def run(input_data):
    input_sentence = read_input_data(json.loads(input_data)['data'])

    # get prediction
    hypothesis = beam_search(model, input_sentence,
                             beam_size=BEAM_SIZE,
                             max_decoding_time_step=MAX_DECODING_TIME_STEP)

    s1 = hypothesis[0]
    s2 = hypothesis[1] 
    s3 = hypothesis[2]

    result = json.dumps({"sentence1": s1.value, "sentence2": s2.value, "sentence3": s3.value})
    return result