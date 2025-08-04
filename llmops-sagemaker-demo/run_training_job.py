#Executed by CodeBuild to launch the SageMaker training job.
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFace
import os

# Fetch environment variables provided by CodeBuild
# Ensure you have uploaded your training data to this S3 path
s3_bucket = os.environ['S3_BUCKET']
sagemaker_role = os.environ['SAGEMAKER_ROLE']
s3_train_path = f"s3://{s3_bucket}/llmops-demo/data/train"

hyperparameters = {
    'epochs': 2,
    'train_batch_size': 4,
    'model_name':'distilbert-base-uncased',
    'learning_rate': 5e-5,
}

huggingface_estimator = HuggingFace(
    entry_point='train.py',
    source_dir='./scripts',
    instance_type='ml.g4dn.xlarge',
    instance_count=1,
    role=sagemaker_role,
    transformers_version='4.28.1',
    pytorch_version='2.0.0',
    py_version='py310',
    hyperparameters=hyperparameters
)

print("Starting SageMaker training job...")
# Use wait=True to ensure the pipeline waits for the job to finish
huggingface_estimator.fit({'train': s3_train_path}, wait=True)
print("Training job completed.")
