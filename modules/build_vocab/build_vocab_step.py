import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter
        
def build_vocab_step(train_dir, compute_target):
    '''
    This step will take the raw data downloaded from the previous step,
    preprocess it, and split into train, valid, and test directories.
    
    :param train_dir: The reference to the directory containing the training data
    :type train_src: DataReference
    :param compute_target: The compute target to run the step on
    :type compute_target: ComputeTarget
    
    :return: The preprocess step, step outputs dictionary (keys: vocab_dir)
    :rtype: PythonScriptStep, dict
    '''

    run_config = RunConfiguration()
    run_config.environment.docker.enabled = True
    run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE
    run_config.environment.python.user_managed_dependencies = False
    conda_packages = ['pytorch']
    run_config.environment.python.conda_dependencies = CondaDependencies.create(
        conda_packages=conda_packages
        )

    input_col = PipelineParameter(name='input_col', default_value='Abstract')
    output_col = PipelineParameter(name='output_col', default_value='Title')
    size = PipelineParameter(name='size', default_value=50000)
    freq_cutoff = PipelineParameter(name='freq_cutoff', default_value=2)

    vocab_dir = PipelineData(
        name='vocab_dir',
        pipeline_output_name='vocab_dir',
        datastore=train_dir.datastore,
        output_mode='mount',
        is_directory=True)

    outputs = [vocab_dir]
    outputs_map = { 
        'vocab_dir': vocab_dir,
    }

    step = PythonScriptStep(
        name="Build Vocab",
        script_name='build_vocab.py',
        arguments=[
            '--train_dir', train_dir, 
            '--vocab_dir', vocab_dir,
            '--input_col', input_col,
            '--output_col', output_col,
            '--size', size,
            '--freq_cutoff', freq_cutoff,
        ],
        inputs=[train_dir],
        outputs=outputs,
        compute_target=compute_target,
        runconfig=run_config,
        source_directory=os.path.dirname(os.path.abspath(__file__)),
        allow_reuse=True
    )

    return step, outputs_map