import os
from src.api import app
from config import MODELS_DIR, FEATURES_DIR

if __name__ == '__main__':
    # Check if models exist
    if not os.path.exists(MODELS_DIR / 'random_forest.p'):
        print("ERROR: Models not found!")
        print("Please run 'python train.py' first to train the models.")
        exit(1)

    print("=" * 60)
    print("STARTING FLASK API SERVER")
    print("=" * 60)
    print("\nAPI Documentation:")
    print("  - Health Check: GET http://localhost:5000/health")
    print("  - Predict: POST http://localhost:5000/api/v1/predict")
    print("  - Batch Predict: POST http://localhost:5000/api/v1/batch-predict")
    print("  - Model Info: GET http://localhost:5000/api/v1/model-info")
    print("  - Metrics: GET http://localhost:5000/api/v1/metrics")
    print("\nServer running on http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
