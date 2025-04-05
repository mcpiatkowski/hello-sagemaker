# Using Custom SciKit-Learn Container

This guide demonstrates how to use your existing training and inference scripts with a custom SciKit-Learn container.

## AWS SageMaker Pre-built Container

Amazon SageMaker provides pre-built Docker images that support scikit-learn. You can use them without creating a custom container:

```python
from sagemaker.sklearn.estimator import SKLearn

sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version='0.23-1'  # Choose appropriate version
)
```

## Custom Container Approach

For more control, you can create a custom container:

### 1. Create Dockerfile

```dockerfile
FROM python:3.8

# Install dependencies
RUN pip install scikit-learn==0.23.2 pandas numpy joblib sagemaker-training

# Set working directory
WORKDIR /opt/ml/code

# Copy your scripts
COPY train.py /opt/ml/code/train.py
COPY inference.py /opt/ml/code/inference.py

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# Set entrypoints for training and serving
ENTRYPOINT ["python", "/opt/ml/code/train.py"]
```

### 2. Build and Push Container

```bash
# Build container
docker build -t sagemaker-sklearn-custom .

# Tag and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker tag sagemaker-sklearn-custom ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/sagemaker-sklearn-custom:latest
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/sagemaker-sklearn-custom:latest
```

### 3. Use Custom Container with SageMaker

```python
from sagemaker.estimator import Estimator

custom_estimator = Estimator(
    image_uri='ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/sagemaker-sklearn-custom:latest',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    hyperparameters={
        'n-estimators': 100,
        'max-depth': 10,
        'random-state': 42
    }
)

# Start training job
custom_estimator.fit({
    'train': 's3://bucket-name/data/train',
    'test': 's3://bucket-name/data/test'
})
```

## Using AWS Pre-built Container with Your Scripts

The simplest approach is using AWS's containers with your scripts:

```python
from sagemaker.sklearn.estimator import SKLearn

# Use AWS's container but with your scripts
sklearn_estimator = SKLearn(
    entry_point='train.py',
    source_dir='.',  # Directory containing your scripts
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version='0.23-1',
    hyperparameters={
        'n-estimators': 100,
        'max-depth': 10,
        'random-state': 42
    }
)

# Start training job
sklearn_estimator.fit({
    'train': 's3://bucket-name/data/train',
    'test': 's3://bucket-name/data/test'
})

# Deploy model
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)
```

## Registering a Model with Custom Container

### 1. Create an inference container (separate Dockerfile)

```dockerfile
FROM python:3.8

# Install dependencies
RUN pip install scikit-learn==0.23.2 pandas numpy joblib sagemaker-inference

# Set working directory
WORKDIR /opt/program

# Copy inference script
COPY inference.py /opt/program/inference.py

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set entrypoint for serving
ENTRYPOINT ["python", "-m", "sagemaker_inference.serving"]
```

### 2. Build and push the inference container

```bash
# Build inference container
docker build -t sagemaker-sklearn-inference -f Dockerfile.inference .

# Tag and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker tag sagemaker-sklearn-inference ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/sagemaker-sklearn-inference:latest
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/sagemaker-sklearn-inference:latest
```

### 3. Register the model in Model Registry

```python
import boto3
import sagemaker
from sagemaker import ModelPackage
from sagemaker.model import Model
from time import gmtime, strftime

# Initialize clients
sm_client = boto3.client('sagemaker')
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Model data location (S3 URI from training job)
model_data = 's3://bucket-name/model-artifacts/model.tar.gz'

# Create a model
model = Model(
    image_uri='ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/sagemaker-sklearn-inference:latest',
    model_data=model_data,
    role=role,
    sagemaker_session=sagemaker_session
)

# Create model package group if it doesn't exist
model_package_group_name = 'IrisClassifierGroup'
try:
    sm_client.create_model_package_group(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageGroupDescription='Iris classifier models'
    )
except sm_client.exceptions.ResourceInUseException:
    print(f"Model package group {model_package_group_name} already exists")

# Create model package
model_package = model.register(
    model_package_group_name=model_package_group_name,
    content_types=["text/csv", "application/json"],
    response_types=["text/csv", "application/json"],
    inference_instances=["ml.t2.medium", "ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_metrics={
        'Accuracy': {
            'Value': 0.95,  # Example metric value
            'Standard': 0.9  # Minimum acceptable value
        }
    },
    approval_status="Approved",
    description="Iris classifier with custom container"
)

print(f"Model package ARN: {model_package.model_package_arn}")
```

## Batch Inference with Custom Container Model

### 1. Create a Transform Job with the Registered Model

```python
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker import ModelPackage

# Initialize session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Create model from model package
model_package_arn = 'arn:aws:sagemaker:us-east-1:ACCOUNT_ID:model-package/IrisClassifierGroup/1'
model = ModelPackage(
    role=role,
    model_package_arn=model_package_arn,
    sagemaker_session=sagemaker_session
)

# Create transformer for batch inference
transformer = model.transformer(
    instance_count=1,
    instance_type='ml.m5.large',
    output_path='s3://bucket-name/batch-output/',
    assemble_with='Line',
    accept='application/json'
)

# Run batch transform job
transformer.transform(
    data='s3://bucket-name/batch-input/test-data.csv',
    data_type='S3Prefix',
    content_type='text/csv',
    split_type='Line'
)

# Wait for the batch job to complete
transformer.wait()

print("Batch transform job completed. Results are in:", transformer.output_path)
```

### 2. Python Script to Run Batch Inference

Save this as `batch_inference.py`:

```python
import boto3
import sagemaker
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-package-arn', type=str, required=True, 
                        help='ARN of model package')
    parser.add_argument('--input-data', type=str, required=True,
                        help='S3 URI of input data')
    parser.add_argument('--output-path', type=str, required=True,
                        help='S3 URI for output data')
    parser.add_argument('--content-type', type=str, default='text/csv',
                        choices=['text/csv', 'application/json'])
    parser.add_argument('--accept', type=str, default='application/json',
                        choices=['text/csv', 'application/json'])
    parser.add_argument('--instance-type', type=str, default='ml.m5.large')
    args = parser.parse_args()
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    # Create model from model package
    model = sagemaker.ModelPackage(
        role=role,
        model_package_arn=args.model_package_arn,
        sagemaker_session=sagemaker_session
    )
    
    # Create transformer
    timestamp = sagemaker_session.current_time_millis()
    transformer = model.transformer(
        instance_count=1,
        instance_type=args.instance_type,
        output_path=args.output_path,
        base_transform_job_name=f'iris-batch-{timestamp}',
        assemble_with='Line',
        accept=args.accept
    )
    
    # Run batch inference
    transformer.transform(
        data=args.input_data,
        data_type='S3Prefix',
        content_type=args.content_type,
        split_type='Line'
    )
    
    print(f"Started batch transform job: {transformer.latest_transform_job.job_name}")
    
    # Wait for the batch job to complete
    transformer.wait()
    
    print(f"Batch transform job completed. Results are in: {args.output_path}")

if __name__ == '__main__':
    main()
```

Example usage:

```bash
python batch_inference.py \
    --model-package-arn "arn:aws:sagemaker:us-east-1:ACCOUNT_ID:model-package/IrisClassifierGroup/1" \
    --input-data "s3://bucket-name/batch-input/test-data.csv" \
    --output-path "s3://bucket-name/batch-output/" \
    --content-type "text/csv" \
    --accept "application/json" \
    --instance-type "ml.m5.large"
```

## Adapting Your Scripts

Your existing `train.py` and `inference.py` scripts should work with minimal modifications:

1. Ensure your scripts use the SageMaker environment variables:
   - `SM_MODEL_DIR`: Where model should be saved
   - `SM_CHANNEL_TRAIN`: Path to training data
   - `SM_CHANNEL_TEST`: Path to test data
   - `SM_OUTPUT_DATA_DIR`: Path for output artifacts

2. Make sure your `inference.py` implements the required functions:
   - `model_fn(model_dir)`: Load model
   - `input_fn(request_body, content_type)`: Parse input
   - `predict_fn(input_data, model)`: Generate predictions
   - `output_fn(predictions, accept)`: Format output

## Using Public ECR Container Registry

AWS provides public sklearn containers you can use directly:

```python
from sagemaker.image_uris import retrieve

# Get the sklearn container image URI
sklearn_image_uri = retrieve(
    framework='sklearn',
    region='us-east-1',
    version='0.23-1',
    py_version='py3',
    instance_type='ml.m5.large'
)

# Create estimator with the container
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri=sklearn_image_uri,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    entry_point='train.py',
    hyperparameters={
        'n-estimators': 100,
        'max-depth': 10
    }
)

# Train and deploy as usual
estimator.fit({'train': train_input, 'test': test_input})
```