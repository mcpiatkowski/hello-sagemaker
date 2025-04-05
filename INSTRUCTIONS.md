# Using the SageMaker Scripts

## Local Training

```bash
# Install dependencies
pip install scikit-learn joblib pandas numpy

# Create data directories
mkdir -p data/train data/test

# Train the model locally (simulating SageMaker environment)
export SM_MODEL_DIR=./model
export SM_CHANNEL_TRAIN=./data/train
export SM_CHANNEL_TEST=./data/test
export SM_OUTPUT_DATA_DIR=./output

python train.py --n-estimators 100 --max-depth 10
```

## SageMaker Training

```python
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# Create SKLearn Estimator
sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=role,
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
```

## Model Registration

```python
# Register model in Model Registry
model_package = sklearn_estimator.register(
    content_types=["text/csv", "application/json"],
    response_types=["text/csv", "application/json"],
    inference_instances=["ml.t2.medium", "ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="IrisClassifier",
    approval_status="Approved"
)
```

## Batch Inference

```python
# Create transformer for batch inference
transformer = sklearn_estimator.transformer(
    instance_count=1,
    instance_type='ml.m5.large',
    assemble_with='Line',
    accept='application/json',
    output_path='s3://bucket-name/output/'
)

# Run batch transform job
transformer.transform(
    data='s3://bucket-name/test/test.csv',
    content_type='text/csv',
    split_type='Line'
)
```

## Local Inference Testing

```python
# Load model
import joblib
model = joblib.load('./model/model.joblib')

# Make predictions
import pandas as pd
test_data = pd.read_csv('./data/test/test.csv')
X_test = test_data.drop('target', axis=1)  # Adjust column name as needed
predictions = model.predict(X_test)
```