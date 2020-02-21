import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter

def ingest_step(datastore, compute_target):
    '''
    Collects metadata of all the papers published between two given days (both
    days given as YYYY-MM-DD strings). The metadata we collect is:

    -- 'arXiv_id' , unique id, of the format 'int.int' or just 'int'
    -- 'Title' , the title of the paper
    -- 'Authors' , as a string of last and first names separated by a delimiter
    -- 'Date' , date of publication, YYYY-MM-DD
    -- 'MSCs' , string of 5-character codes, separated by a delimiter.
    -- 'Abstract' , short description of the content of the paper

    :param datastore: The datastore that will be used
    :type datastore: Datastore
    :param compute_target: The compute target to run the step on
    :type compute_target: ComputeTarget
    
    :return: The ingestion step, step outputs dictionary (keys: raw_data_dir)
    :rtype: PythonScriptStep, dict
    '''

    run_config = RunConfiguration()
    run_config.environment.environment_variables = {
        'AZURE_REGION': datastore._workspace.location
        }
    run_config.environment.docker.enabled = True

    start_date = PipelineParameter(name='start_date', default_value='2015-01-01')
    end_date = PipelineParameter(name='end_date', default_value='2015-06-01')

    raw_data_dir = PipelineData(
        name='raw_data_dir', 
        pipeline_output_name='raw_data_dir',
        datastore=datastore,
        output_mode='mount',
        is_directory=True)

    outputs = [raw_data_dir]
    outputs_map = { 'raw_data_dir': raw_data_dir }

    step = PythonScriptStep(
        name="Ingest",
        script_name='ingest.py',
        arguments=[
            '--output_dir', raw_data_dir, 
            '--start_date', start_date, 
            '--end_date', end_date
            ],
        outputs=outputs,
        compute_target=compute_target,
        source_directory=os.path.dirname(os.path.abspath(__file__)),
        runconfig=run_config,
        allow_reuse=True
    )

    return step, outputs_map