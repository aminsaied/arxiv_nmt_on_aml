from azureml.core import Workspace
from azureml.core.environment import Environment
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.core import Experiment
from azureml.pipeline.core import Pipeline
from modules.ingest.ingest_step import ingest_step
from modules.preprocess.preprocess_step import preprocess_step
from modules.train.build_vocab_step import build_vocab_step
from modules.train.train_step import train_step
from modules.evaluate.evaluate_step import evaluate_step
# from modules.deploy.deploy_step import
from azureml.core.compute import AmlCompute, ComputeTarget

# Get workspace, datastores, and compute targets
print('Connecting to Workspace ...')
workspace = Workspace.from_config()
datastore = workspace.get_default_datastore()

# Create CPU compute target
print('Creating CPU compute target ...')
cpu_cluster_name = 'cpucluster'
cpu_compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2', 
                                                           idle_seconds_before_scaledown=1200,
                                                           min_nodes=0, 
                                                           max_nodes=2)
cpu_compute_target = ComputeTarget.create(workspace, cpu_cluster_name, cpu_compute_config)
cpu_compute_target.wait_for_completion(show_output=True)

# # Create Run Configuration
# # Create a new runconfig object
# run_amlcompute = RunConfiguration()
# run_amlcompute.target = cpu_compute_target
# run_amlcompute.environment.docker.enabled = True
# run_amlcompute.environment.docker.base_image = DEFAULT_CPU_IMAGE
# run_amlcompute.environment.python.user_managed_dependencies = False
# conda_packages = ['beautifulsoup4']
# run_amlcompute.environment.python.conda_dependencies = CondaDependencies.create(conda_packages=conda_packages)

# TODO: Use env.yml to specify conda dependencies instead of
# manually specifying list of required packages
# # Set up conda environment
# conda_env = Environment.from_conda_specification(name="cornetto", file_path="env.yml")
# # Attach conda environment specified above to run config
# runconfig.run_config.environment = conda_env


# Create GPU compute target
print('Creating GPU compute target ...')
gpu_cluster_name = 'gpucluster'
gpu_compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', 
                                                           idle_seconds_before_scaledown=1200,
                                                           min_nodes=0, 
                                                           max_nodes=2)
gpu_compute_target = ComputeTarget.create(workspace, gpu_cluster_name, gpu_compute_config)
gpu_compute_target.wait_for_completion(show_output=True)

# Step 1: Data ingestion 
ingest_step, ingest_outputs = ingest_step(datastore, cpu_compute_target)

# Step 2: Data preprocessing 
preprocess_step, preprocess_outputs = preprocess_step(ingest_outputs['raw_data_dir'], cpu_compute_target)

# Step 3: Create vocab from training data
build_vocab_step, build_vocab_outputs = build_vocab_step(preprocess_outputs['train_dir'], cpu_compute_target)

# Step 4: Train Model
train_step, train_outputs = train_step(
    datastore,
    preprocess_outputs['train_dir'],
    preprocess_outputs['valid_dir'],
    build_vocab_outputs['vocab_dir'],
    gpu_compute_target)

# Step 5: Evaluate Model
evaluate_step, evaluate_outputs = evaluate_step(train_outputs['model_dir'], data_preprocess_outputs['test_dir'], gpu_compute_target)

# # Step 6: Deploy Model
# deploy_step, deploy_outputs = deploy_step(train_outputs['model_dir'], evaluate_outputs['accuracy_file'], data_preprocess_outputs['test_dir'], cpu_compute_target)

# Submit pipeline
print('Submitting pipeline ...')
pipeline_parameters = {
    'start_date': '2015-01-01',
    'end_date': '2015-01-02',
    'input_col': 'Abstract',
    'output_col': 'Title',
    'train_proportion': 0.8,
    'max_epoch': 1,
}

pipeline = Pipeline(workspace=workspace, steps=[ingest_step, preprocess_step, build_vocab_step, train_step, evaluate_step])
pipeline_run = Experiment(workspace, 'arXiv-NMT').submit(pipeline, pipeline_parameters=pipeline_parameters)