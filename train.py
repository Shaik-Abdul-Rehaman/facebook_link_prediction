import sys
import os
import pandas as pd
from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from config import *

def main():
    print("=" * 60)
    print("LINK PREDICTION MODEL - TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Data Processing
    print("\n[1/4] DATA PROCESSING")
    print("-" * 60)
    processor = DataProcessor()
    processor.load_or_create_graph()
    processor.load_or_create_missing_edges()
    processor.load_or_compute_centrality_measures()
    processor.compute_svd_features()

    # Step 2: Prepare training data
    print("\n[2/4] PREPARING TRAINING DATA")
    print("-" * 60)
    X_train, X_test = processor.prepare_training_data()
    y_train = X_train['indicator_link'].values
    y_test = X_test['indicator_link'].values
    X_train = X_train.drop('indicator_link', axis=1)
    X_test = X_test.drop('indicator_link', axis=1)

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # Step 3: Feature Engineering
    print("\n[3/4] FEATURE ENGINEERING")
    print("-" * 60)
    feature_engineer = FeatureEngineer(
        graph=processor.train_graph,
        pagerank=processor.pagerank,
        katz=processor.katz,
        hits=processor.hits,
        wcc=processor.wcc,
        adj_dict=processor.adj_dict,
        svd_u=processor.svd_u,
        svd_v=processor.svd_v
    )

    features_train = feature_engineer.engineer_features(X_train)
    features_test = feature_engineer.engineer_features(X_test)
    #savve features
    features_train.to_csv(FEATURES_TRAIN_CSV, index=False)
    features_test.to_csv(FEATURES_TEST_CSV, index=False)

    print(f"Generated {features_train.shape[1]} features")
    print(f"Features shape - Train: {features_train.shape}, Test: {features_test.shape}")

    # Step 4: Model Training and Evaluation
    print("\n[4/4] MODEL TRAINING AND EVALUATION")
    print("-" * 60)
    trainer = ModelTrainer()
    models, metrics = trainer.train_all_models(features_train, y_train, features_test, y_test)

    # Save models
    trainer.save_models()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the API server: python run_api.py")
    print("2. Test the API: python test_api.py")
    print("3. Or use curl to make predictions")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
