#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Hyperparameters and model parameters
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    
    # SageMaker specific arguments
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    train_data_path = os.path.join(args.train, "train.csv")
    test_data_path = os.path.join(args.test, "test.csv") if args.test else None
    
    train_df = pd.read_csv(train_data_path)
    
    # For Iris dataset example
    # Assuming format: sepal_length,sepal_width,petal_length,petal_width,species
    if "species" in train_df.columns:  # For Iris dataset
        X = train_df.drop(["species"], axis=1)
        y = train_df["species"]
    else:  # For other datasets, we'd need to determine the target column
        # This is a placeholder; adjust based on your dataset
        X = train_df.iloc[:, :-1]
        y = train_df.iloc[:, -1]
    
    # Split data if test set not provided
    if test_data_path and os.path.exists(test_data_path):
        test_df = pd.read_csv(test_data_path)
        if "species" in test_df.columns:  # For Iris dataset
            X_test = test_df.drop(["species"], axis=1)
            y_test = test_df["species"]
        else:
            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.random_state
        )
        X, X_test, y, y_test = X_train, X_test, y_train, y_test
    
    # Train the model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    model.fit(X, y)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    print(f"Saving model to {args.model_dir}")
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    
    # Save feature names (for inference)
    with open(os.path.join(args.model_dir, "features.txt"), "w") as f:
        f.write(",".join(X.columns.tolist()))
    
    # Save additional artifacts if needed
    if args.output_data_dir:
        # Example: Save feature importances
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        os.makedirs(args.output_data_dir, exist_ok=True)
        feature_importances.to_csv(
            os.path.join(args.output_data_dir, "feature_importances.csv"),
            index=False
        )
        
    print("Training complete!")