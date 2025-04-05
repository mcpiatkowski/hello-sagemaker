#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import boto3
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions using a model from SageMaker Model Registry')
    
    parser.add_argument('--model-package-arn', type=str, required=True, 
                        help='The ARN of the model package version to use')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to the input data file (CSV format)')
    parser.add_argument('--content-type', type=str, default='text/csv',
                        choices=['text/csv', 'application/json'],
                        help='The content type of the input data')
    parser.add_argument('--accept', type=str, default='application/json',
                        choices=['text/csv', 'application/json'],
                        help='The expected output format')
    parser.add_argument('--region', type=str, default='us-east-1',
                        help='AWS region where the model is deployed')
    parser.add_argument('--output-file', type=str, default='predictions.json',
                        help='Path to save the prediction results')
    
    return parser.parse_args()

def load_data(input_file, content_type):
    """Load and prepare the input data for prediction."""
    if content_type == 'text/csv':
        df = pd.read_csv(input_file)
        # Return CSV string without header
        return df.to_csv(index=False, header=False)
    elif content_type == 'application/json':
        df = pd.read_csv(input_file)
        # Convert to JSON format
        return json.dumps(df.to_dict(orient='records'))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def create_model_from_registry(model_package_arn, region):
    """Create a SageMaker model from a model package in the registry."""
    client = boto3.client('sagemaker', region_name=region)
    
    # Get model package details
    model_package = client.describe_model_package(
        ModelPackageName=model_package_arn
    )
    
    # Create model from the model package
    model_name = f"model-{model_package_arn.split('/')[-1]}"
    
    response = client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'ModelPackageName': model_package_arn
        },
        ExecutionRoleArn=model_package['InferenceSpecification']['Containers'][0]['ModelDataUrl']
    )
    
    return model_name

def create_transform_job(model_name, input_file, content_type, accept, region):
    """Create a batch transform job for prediction."""
    client = boto3.client('sagemaker', region_name=region)
    s3 = boto3.client('s3', region_name=region)
    
    # Upload input data to S3
    bucket_name = 'sagemaker-{}-{}'.format(region, boto3.client('sts').get_caller_identity().get('Account'))
    input_prefix = 'batch-transform/input'
    output_prefix = 'batch-transform/output'
    
    input_key = f"{input_prefix}/{input_file.split('/')[-1]}"
    s3.upload_file(input_file, bucket_name, input_key)
    
    # Create transform job
    job_name = f"transform-job-{model_name}"
    response = client.create_transform_job(
        TransformJobName=job_name,
        ModelName=model_name,
        MaxConcurrentTransforms=1,
        BatchStrategy='MultiRecord',
        TransformInput={
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': f"s3://{bucket_name}/{input_key}"
                }
            },
            'ContentType': content_type,
        },
        TransformOutput={
            'S3OutputPath': f"s3://{bucket_name}/{output_prefix}",
            'Accept': accept,
        },
        TransformResources={
            'InstanceType': 'ml.m5.large',
            'InstanceCount': 1
        }
    )
    
    # Wait for the job to complete
    waiter = client.get_waiter('transform_job_completed_or_stopped')
    waiter.wait(TransformJobName=job_name)
    
    # Get the job status
    job = client.describe_transform_job(TransformJobName=job_name)
    
    # If successful, download the results
    if job['TransformJobStatus'] == 'Completed':
        output_key = f"{output_prefix}/{input_file.split('/')[-1]}.out"
        output_file = 'predictions.json'
        s3.download_file(bucket_name, output_key, output_file)
        return output_file
    else:
        raise Exception(f"Transform job failed: {job['FailureReason']}")

def make_realtime_prediction(model_package_arn, input_data, content_type, accept, region):
    """Make a real-time prediction using SageMaker runtime."""
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    # Get the endpoint name from the model package
    sm_client = boto3.client('sagemaker', region_name=region)
    model_package = sm_client.describe_model_package(
        ModelPackageName=model_package_arn
    )
    
    # Check if there's an endpoint already available
    endpoints = sm_client.list_endpoints()
    endpoint_name = None
    
    for endpoint in endpoints['Endpoints']:
        config = sm_client.describe_endpoint_config(EndpointConfigName=endpoint['EndpointConfigName'])
        for variant in config['ProductionVariants']:
            if variant['ModelName'] == model_package_arn.split('/')[-1]:
                endpoint_name = endpoint['EndpointName']
                break
        if endpoint_name:
            break
    
    if not endpoint_name:
        # You would need to deploy the model to an endpoint first
        # This is a more complex process not covered in this script
        raise ValueError("No endpoint found for this model. Please deploy the model first.")
    
    # Make prediction
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Accept=accept,
        Body=input_data
    )
    
    # Parse and return the response
    if accept == 'application/json':
        return json.loads(response['Body'].read().decode())
    else:
        return response['Body'].read().decode()

def main():
    args = parse_args()
    
    # Load and prepare input data
    input_data = load_data(args.input_file, args.content_type)
    
    try:
        # Create model from registry
        model_name = create_model_from_registry(args.model_package_arn, args.region)
        
        # Run batch transform
        output_file = create_transform_job(model_name, args.input_file, 
                                          args.content_type, args.accept, args.region)
        
        print(f"Predictions successfully saved to {output_file}")
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()