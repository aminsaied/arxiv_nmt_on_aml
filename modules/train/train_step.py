import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.steps import EstimatorStep
# from azureml.pipeline.steps import MpiStep
from azureml.contrib.pipeline.steps import ParallelRunConfig
from azureml.contrib.pipeline.steps import ParallelRunStep
from azureml.core import Environment
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

    # parallel_cd = CondaDependencies()

    # parallel_cd.add_channel("pytorch")
    # parallel_cd.add_conda_package("pytorch")
    # parallel_cd.add_conda_package("torchvision")
    # parallel_cd.add_conda_package("tqdm")
    # parallel_cd.add_conda_package("nltk")

    # parallel_env = Environment(name="styleenvironment")
    # parallel_env.python.conda_dependencies=parallel_cd
    # parallel_env.docker.base_image = DEFAULT_GPU_IMAGE

    # parallel_run_config = ParallelRunConfig(
    #                     environment=parallel_env,
    #                     entry_script='train.py',
    #                     output_action='summary_only',
    #                     mini_batch_size="1",
    #                     error_threshold=1,
    #                     source_directory=os.path.dirname(os.path.abspath(__file__)),
    #                     compute_target=compute_target, 
    #                     node_count=3)


    # set hyperparameters of the model training step
    input_col = PipelineParameter(name='input_col', default_value='Title')
    output_col = PipelineParameter(name='output_col', default_value='Abstract')
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

    estimator = PyTorch(
        compute_target=compute_target,
        entry_script='train.py',
        script_params=script_params,
        node_count=2,
        distributed_training=Mpi(),
        source_directory=os.path.dirname(os.path.abspath(__file__)),
        use_gpu=True,
        conda_packages=['nltk'])

    est_step = EstimatorStep(
        name="Estimator_Train", 
        estimator=est, 
        estimator_entry_script_arguments=["--datadir", input_data, "--output", output],
        runconfig_pipeline_params=None, 
        inputs=[input_data], 
        outputs=[output], 
        compute_target=cpu_cluster)

    return step, outputs_map