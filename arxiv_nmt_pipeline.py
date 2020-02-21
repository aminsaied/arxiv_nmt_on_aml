from azureml.core import Workspace
from azureml.core import Experiment
from azureml.pipeline.core import Pipeline
from modules.ingestion.ingestion_step import ingestion_step
# from modules.preprocess.data_preprocess_step import data_preprocess_step
# from modules.train.train_step import train_step
# from modules.evaluate.evaluate_step import evaluate_step
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

# # Create GPU compute target
# print('Creating GPU compute target ...')
# gpu_cluster_name = 'gpucluster'
# gpu_compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', 
#                                                            idle_seconds_before_scaledown=1200,
#                                                            min_nodes=0, 
#                                                            max_nodes=2)
# gpu_compute_target = ComputeTarget.create(workspace, gpu_cluster_name, gpu_compute_config)
# gpu_compute_target.wait_for_completion(show_output=True)

# Step 1: Data ingestion 
ingestion_step, ingestion_outputs = ingestion_step(datastore, cpu_compute_target)

# # Step 2: Data preprocessing 
# data_preprocess_step, data_preprocess_outputs = data_preprocess_step(data_ingestion_outputs['raw_data_dir'], cpu_compute_target)

# # Step 3: Train Model
# train_step, train_outputs = train_step(data_preprocess_outputs['train_dir'], data_preprocess_outputs['valid_dir'], gpu_compute_target)

# # Step 4: Evaluate Model
# evaluate_step, evaluate_outputs = evaluate_step(train_outputs['model_dir'], data_preprocess_outputs['test_dir'], gpu_compute_target)

# # Step 5: Deploy Model
# deploy_step, deploy_outputs = deploy_step(train_outputs['model_dir'], evaluate_outputs['accuracy_file'], data_preprocess_outputs['test_dir'], cpu_compute_target)

# Submit pipeline
print('Submitting pipeline ...')
pipeline_parameters = {
    'start_date': '2015-01-01',
    'end_date': '2015-02-01',
}
pipeline = Pipeline(workspace=workspace, steps=[ingestion_step])
pipeline_run = Experiment(workspace, 'arXiv NMT model').submit(pipeline, pipeline_parameters=pipeline_parameters)