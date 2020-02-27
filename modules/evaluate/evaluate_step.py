import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.runconfig import DEFAULT_GPU_IMAGE
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter
        
def evaluate_step(datastore, test_dir, model_dir, compute_target):
    '''
    This step will take the raw data downloaded from the previous step,
    preprocess it, and split into train, valid, and test directories.
    
    :param datastore: The datastore that will be used
    :type datastore: Datastore
    :param test_dir: The reference to the directory containing the test data
    :type test_dir: DataReference
    :param model_dir: The reference to the directory containing the NMT model
    :type model_dir: DataReference
    :param compute_target: The compute target to run the step on
    :type compute_target: ComputeTarget
    
    :return: The evaluate step, step outputs dictionary (keys: eval_dir)
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
    beam_size = PipelineParameter(name='beam_size', default_value=5)
    max_decoding_time_step = PipelineParameter(name='max_decoding_time_step', default_value=70)


    eval_dir = PipelineData(
        name='eval_dir',
        pipeline_output_name='eval_dir',
        datastore=datastore,
        output_mode='mount',
        is_directory=True)

    outputs = [eval_dir]
    outputs_map = { 
        'eval_dir': eval_dir,
    }

    step = PythonScriptStep(
        name="Evaluate",
        script_name='evaluate.py',
        arguments=[
            '--test_dir', test_dir,
            '--model_dir', model_dir,
            '--input_col', input_col,
            '--output_col', output_col,
            '--cuda', cuda,
            '--seed', seed,
            '--beam_size', beam_size,
            '--max_decoding_time_step', max_decoding_time_step,
            '--eval_dir', eval_dir
        ],
        inputs=[test_dir, model_dir],
        outputs=outputs,
        compute_target=compute_target,
        runconfig=run_config,
        source_directory=os.path.dirname(os.path.abspath(__file__)),
        allow_reuse=True
    )

    return step, outputs_map