# SageMaker Hello World

This project demonstrates a complete machine learning workflow using Amazon SageMaker with scikit-learn.

## Project Plan

### 1. Model Training with scikit-learn
- Use scikit-learn (version compatible with SageMaker ~0.21) to train a classification model
- Use the Iris dataset from sklearn as a simple demonstration case
- [Optional] Implement data preprocessing, feature engineering, and model selection
- [Optional] Evaluate model performance using appropriate metrics (accuracy, F1, etc.)

### 2. Create Training Script
- Develop `train.py` script that will be executed in SageMaker training job
- Implement data loading, preprocessing, model training, and evaluation
- Save the trained model using the SageMaker model saving conventions
- Add proper logging and error handling

### 3. Create Inference Script
- Develop `inference.py` script for model serving in SageMaker
- Implement model loading, input preprocessing, prediction, and output formatting
- Ensure compatibility with both real-time and batch inference
- Add proper error handling and logging

### 4. Package the Model for SageMaker
- Create the necessary directory structure for SageMaker model packaging
- Package the model artifacts, inference script, and dependencies
- Configure model metadata and requirements

### 5. Push Model to Model Registry
- Create a model package in the SageMaker Model Registry
- Add appropriate tags and version information
- Document model metadata, metrics, and intended use

### 6. Deploy for Batch Inference
- Configure a SageMaker batch transform job
- Set up input data location and output destination
- Configure instance type and count for batch processing
- Set up necessary IAM roles and permissions

### 7. Execute Batch Inference
- Run the batch transform job on test data
- Monitor the inference job progress and resources
- Validate the output format and predictions
- Analyze inference performance metrics

## Using the Scripts

See the [INSTRUCTIONS.md](INSTRUCTIONS.md) file for detailed instructions on:
- Training the model locally and on SageMaker
- Registering the model in the SageMaker Model Registry
- Running batch inference with the trained model
- Making predictions from a model in the registry using `predict_from_registry.py`

## Getting Started

1. Clone this repository
2. Set up your AWS credentials
3. Install dependencies: `pip install scikit-learn pandas numpy boto3 sagemaker`
4. Follow instructions in [INSTRUCTIONS.md](INSTRUCTIONS.md)