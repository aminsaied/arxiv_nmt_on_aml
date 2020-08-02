# Generate paper titles

We train a model that generates titles for mathematical papers based on their abstracts. This proof-of-concept repo will demonstrate AzureML workflow for a typical data science project. 

Disclaimers:
- The NMT model we use here is strictly for demonstration purposes and was forked from here: https://github.com/pcyin/pytorch_basic_nmt. In their words:

> This is a basic implementation of attentional neural machine translation (Bahdanau et al., 2015, Luong et al., 2015) in Pytorch 0.4.
It implements the model described in [Luong et al., 2015](https://arxiv.org/abs/1508.04025), and supports label smoothing, beam-search decoding and random sampling.
With 256-dimensional LSTM hidden size, it achieves 28.13 BLEU score on the IWSLT 2014 Germen-English dataset (Ranzato et al., 2015).

- The structure for this pipeline was taken from this example: https://github.com/Azure/aml-object-classification-pipeline

## Usage - build and run Azure ML pipeline

Run `$ python arxiv_nmt_pipeline.py` to build an AzureML pipeline that performs the following steps

- Ingest: Grab raw data from the arXiv.
- Process: Prepare the text for the model, split into train/valid/test sets.
- Build vocab: Construct a vocabulary from the training data.
- Train: Train a simple NMT attention model built in PyTorch.
- Evaluate: Evaluate the model on the test data.
- Deploy: Deploy the model to a webservice.

The source code for these steps can be found in the `modules` directory. Subdirectories there correspond to steps in the pipeline. Those subdirectories have the following structure:

- `<task>.py`: the script itself
- `<task>_step.py`: builds the AzureML pipeline step wrapping the script
- additional "helper" modules used in `<task>.py`

## Usage - invoke the model endpoint

The final result of the pipeline is a endpoint that can be used to generate a title for your paper - use at your own risk ;-)

Run `$ python invoke.py` for an example that submits an abstract to the endpoint and unpacks it's response.

## Pipeline Structure

This code builds the following pipeline in AzureML.

![](docs/PipelineGraph.PNG)

The results from individual steps are cached, and only recomputed if there are changes with that step (or anywhere upstream!).


### License

This work is licensed under a Creative Commons Attribution 4.0 International License.
