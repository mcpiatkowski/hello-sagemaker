#!/usr/bin/env python
# coding=utf-8

import os
import json
import joblib
import numpy as np
import pandas as pd

# inference functions ---------------


def model_fn(model_dir):
    """Load the model from disk and prepare for inference."""
    print("Loading model...")

    # Load the model
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)

    # Load feature names if saved
    features_path = os.path.join(model_dir, "features.txt")
    if os.path.exists(features_path):
        with open(features_path, "r") as f:
            feature_names = f.read().strip().split(",")
        print(f"Loaded feature names: {feature_names}")

    return model


def input_fn(request_body, request_content_type):
    """Parse input data payload."""
    print(f"Received request with content type: {request_content_type}")

    if request_content_type == "application/json":
        input_data = json.loads(request_body)

        # Handle both single prediction and batch predictions
        if isinstance(input_data, dict):
            # Single prediction case
            df = pd.DataFrame([input_data])
        else:
            # Batch prediction case
            df = pd.DataFrame(input_data)

        return df

    elif request_content_type == "text/csv":
        # Handle CSV format
        return pd.read_csv(pd.io.StringIO(request_body))

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Make prediction with model based on the input data."""
    print("Performing prediction...")

    # Make predictions
    predictions = model.predict(input_data)

    # Add probabilities if classifier supports it
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)
        return {"predictions": predictions, "probabilities": probabilities}

    return {"predictions": predictions}


def output_fn(prediction_output, accept):
    """Format prediction output according to the response accept header."""
    print(f"Formatting output to {accept}")

    if accept == "application/json":
        # Convert predictions to appropriate format
        predictions = prediction_output["predictions"]

        response = {"predictions": predictions.tolist()}

        # Include probabilities if available
        if "probabilities" in prediction_output:
            response["probabilities"] = prediction_output["probabilities"].tolist()

        return json.dumps(response)

    elif accept == "text/csv":
        # For CSV, just return the predictions as a single column
        predictions = prediction_output["predictions"]
        return ",".join([str(p) for p in predictions])

    else:
        raise ValueError(f"Unsupported accept type: {accept}")
