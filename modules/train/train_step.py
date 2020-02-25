import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.runconfig import DEFAULT_GPU_IMAGE
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter
        
def train_step(datastore, train_dir, valid_dir, vocab_dir, compute_target):
    '''
    This step will take the raw data downloaded from the previous step,
    preprocess it, and split into train, valid, and test directories.
    
    :param datastore: The datastore that will be used
    :type datastore: Datastore
    :param train_dir: The reference to the directory containing the training data
    :type train_src: DataReference
    :param valid_dir: The reference to the directory containing the validation data
    :type valid_dir: DataReference
    :param vocab_dir: The reference to the directory containing the vocab data
    :type vocab_dir: DataReference
    :param compute_target: The compute target to run the step on
    :type compute_target: ComputeTarget
    
    :return: The training step, step outputs dictionary (keys: model_dir)
    :rtype: PythonScriptStep, dict
    '''

    run_config = RunConfiguration()
    run_config.environment.docker.enabled = True
    run_config.environment.docker.base_image = DEFAULT_GPU_IMAGE
    run_config.environment.python.user_managed_dependencies = False
    conda_packages = ['pytorch', 'tqdm', 'nltk']
    run_config.environment.python.conda_dependencies = CondaDependencies.create(
        conda_packages=conda_packages
        )

    # set hyperparameters of the model training step
    input_col = PipelineParameter(name='input_col', default_value='Abstract')
    output_col = PipelineParameter(name='output_col', default_value='Title')
    cuda = PipelineParameter(name='cuda', default_value=True)
    seed = PipelineParameter(name='seed', default_value=0)
    batch_size = PipelineParameter(name='batch_size', default_value=32)
    embed_size = PipelineParameter(name='embed_size', default_value=256)
    hidden_size = PipelineParameter(name='hidden_size', default_value=256)
    clip_grad = PipelineParameter(name='clip_grad', default_value=5.0)
    label_smoothing = PipelineParameter(name='label_smoothing', default_value=0.0)
    log_every = PipelineParameter(name='log_every', default_value=10)
    max_epoch = PipelineParameter(name='max_epoch', default_value=2)
    input_feed = PipelineParameter(name='input_feed', default_value=True)
    patience = PipelineParameter(name='patience', default_value=5)
    max_num_trial = PipelineParameter(name='max_num_trial', default_value=5)
    lr_decay = PipelineParameter(name='lr_decay', default_value=0.5)
    beam_size = PipelineParameter(name='beam_size', default_value=5)
    sample_size = PipelineParameter(name='sample_size', default_value=5)
    lr = PipelineParameter(name='lr', default_value=0.001)
    uniform_init = PipelineParameter(name='uniform_init', default_value=0.1)
    valid_niter = PipelineParameter(name='valid_niter', default_value=2000)
    dropout = PipelineParameter(name='dropout', default_value=0.3)
    max_decoding_time_step = PipelineParameter(name='max_decoding_time_step', default_value=70)


    model_dir = PipelineData(
        name='model_dir',
        pipeline_output_name='model_dir',
        datastore=datastore,
        output_mode='mount',
        is_directory=True)

    outputs = [model_dir]
    outputs_map = { 
        'model_dir': model_dir,
    }

    step = PythonScriptStep(
        name="Train",
        script_name='train.py',
        arguments=[
            '--train_dir', train_dir,
            '--valid_dir', train_dir,
            '--input_col', input_col,
            '--output_col', output_col,
            '--vocab_dir', vocab_dir,
            '--model_dir', model_dir,
            '--input_col', input_col,
            '--output_col', output_col,
            '--cuda', cuda,
            '--seed', seed,
            '--batch_size', batch_size,
            '--embed_size', embed_size,
            '--hidden_size', hidden_size,
            '--clip_grad', clip_grad,
            '--label_smoothing', label_smoothing,
            '--log_every', log_every,
            '--max_epoch', max_epoch,
            '--input_feed', input_feed,
            '--patience', patience,
            '--max_num_trial', max_num_trial,
            '--lr_decay', lr_decay,
            '--beam_size', beam_size,
            '--sample_size', sample_size,
            '--lr', lr,
            '--uniform_init', uniform_init,
            '--valid_niter', valid_niter,
            '--dropout', dropout,
            '--max_decoding_time_step', max_decoding_time_step,
        ],
        inputs=[train_dir, valid_dir, vocab_dir],
        outputs=outputs,
        compute_target=compute_target,
        runconfig=run_config,
        source_directory=os.path.dirname(os.path.abspath(__file__)),
        allow_reuse=True
    )

    return step, outputs_map