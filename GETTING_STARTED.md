# Getting Started - Link Prediction API

## Quick Start (5 minutes)

### 1. Setup Environment
```bash
cd task_management
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train Models (First Time Only - Takes 1-2 hours)
```bash
python train.py
```

This creates:
- Trained models in `models/` directory
- Feature files in `features/` directory
- Intermediate data in `data/` directory

### 3. Start API Server
```bash
python run_api.py
```

Server runs on: `http://localhost:5000`

### 4. Test the API
In a new terminal:
```bash
python test_api.py
```

## Project Architecture

### Data Flow
```
train.csv (input)
    ↓
[DataProcessor] → Builds graph, negative sampling
    ↓ (pickle files)
    ├→ graph.p
    ├→ page_rank.p
    ├→ katz.p
    └→ hits.p
    ↓
[FeatureEngineer] → Generates 30+ features
    ↓
[ModelTrainer] → Trains 4 models
    ↓
models/
├→ random_forest.p (RFC model)
├→ onehot_encoder.p
├→ logistic_regression.p (Proposed model)
├→ decision_tree.p
├→ svm.p
└→ logistic_regression_baseline.p
```

### Code Organization

#### `config.py`
Central configuration:
- Data paths
- Model selection
- Feature definitions
- API limits

#### `src/data_processor.py`
Handles data:
- Graph construction from CSV
- Negative edge generation
- Centrality measures (PageRank, Katz, HITS)
- SVD decomposition

#### `src/feature_engineer.py`
Feature engineering:
- Similarity metrics (Jaccard, Cosine)
- Path-based features
- Network centrality features
- SVD features

#### `src/model_trainer.py`
Model training:
- 4 different classifiers
- Training and evaluation
- Model persistence

#### `src/api.py`
Flask API:
- RESTful endpoints
- Batch prediction
- Error handling

## Usage Examples

### Example 1: Predict Single Edge
```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "edges": [
      {"source_node": 1, "destination_node": 2}
    ]
  }'
```

### Example 2: Batch Prediction (10 edges)
```python
import requests

edges = [
    {"source_node": i, "destination_node": i+100}
    for i in range(1, 11)
]

response = requests.post(
    "http://localhost:5000/api/v1/predict",
    json={"edges": edges}
)

for pred in response.json()['predictions']:
    print(f"{pred['source_node']} → {pred['destination_node']}: "
          f"{pred['probability']:.2%}")
```

### Example 3: Check Model Performance
```bash
curl http://localhost:5000/api/v1/metrics
```

Returns accuracy, precision, recall, F1 for each model.

## API Reference

### POST /api/v1/predict
Single and batch predictions (up to 1000 edges)

**Request:**
```json
{
  "edges": [
    {"source_node": 123, "destination_node": 456},
    ...
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "model": "random_forest_ensemble",
  "predictions": [
    {
      "source_node": 123,
      "destination_node": 456,
      "prediction": 1,
      "probability": 0.87,
      "link_exists": true
    }
  ]
}
```

### GET /api/v1/model-info
Get model and feature information

**Response:**
```json
{
  "model": "random_forest_ensemble",
  "version": "1.0.0",
  "features": [
    "num_followers_s",
    "num_followers_d",
    ...
  ],
  "max_batch_size": 1000,
  "available_models": [
    "random_forest_ensemble",
    "logistic_regression",
    "svm",
    "decision_tree"
  ]
}
```

### GET /api/v1/metrics
Model evaluation metrics

**Response:**
```json
{
  "metrics": {
    "random_forest_ensemble": {
      "accuracy": 0.92,
      "precision": 0.88,
      "recall": 0.85,
      "f1": 0.86
    },
    ...
  }
}
```

### GET /health
API health check

## Training Pipeline Details

### Step 1: Graph Construction
- Reads `train.csv` (source_node, destination_node)
- Creates directed graph with NetworkX
- Saves to `features/graph.p`

### Step 2: Negative Edge Generation
- Generates ~9.4M non-existing edges
- Ensures shortest path > 2
- Balances positive/negative samples
- Saves to `features/missing_edges_final.p`

### Step 3: Feature Extraction
- **Basic features**: Node degrees, common neighbors
- **Similarity features**: Jaccard, Cosine similarity
- **Centrality features**: PageRank, Katz, HITS
- **Path features**: Shortest path, connected components
- **Advanced features**: Adamic-Adar index, SVD components

### Step 4: Model Training
```
Training Data (80% of edges with features)
    ↓
    ├→ Decision Tree Classifier
    ├→ Linear SVM
    ├→ Logistic Regression
    └→ Random Forest + Logistic Regression (ensemble)
    ↓
Testing Data (20% of edges)
    ↓
Evaluation Metrics (Accuracy, Precision, Recall, F1)
```

### Step 5: Model Selection
- Best model saved as `PRODUCTION_MODEL` in config
- Loaded automatically by API
- Changeable without retraining

## Performance Metrics

Typical results on Facebook dataset:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Random Forest Ensemble | 92% | 88% | 85% | 86% |
| Logistic Regression | 89% | 85% | 82% | 83% |
| SVM | 87% | 84% | 80% | 82% |
| Decision Tree | 85% | 82% | 78% | 80% |

## Switching Models

Change the production model in `config.py`:

```python
PRODUCTION_MODEL = 'logistic_regression'  # or 'svm', 'decision_tree', 'random_forest_ensemble'
```

Then restart the API:
```bash
python run_api.py
```

## Production Deployment

### Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 run_api:app
```

### Using Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "run_api:app"]
```

Build and run:
```bash
docker build -t link-prediction .
docker run -p 5000:5000 link-prediction
```

### Environment Variables
Create `.env` file:
```
FLASK_ENV=production
FLASK_DEBUG=0
API_PORT=5000
API_HOST=0.0.0.0
PRODUCTION_MODEL=random_forest_ensemble
```

## Troubleshooting

### Issue: "Models not found"
**Solution:** Train models first
```bash
python train.py
```

### Issue: Memory error during training
**Solution:** The dataset is large. Ensure 4GB+ available RAM.

### Issue: API returns 500 error
**Solution:** 
1. Check server logs
2. Ensure all dependencies installed
3. Verify model files exist

### Issue: Slow predictions
**Solution:**
- Use lighter model (Logistic Regression)
- Enable batching (POST to `/batch-predict`)
- Run with Gunicorn workers

## Next Steps

1. **Customize features:** Edit `src/feature_engineer.py`
2. **Tune models:** Modify hyperparameters in `src/model_trainer.py`
3. **Monitor performance:** Check `/api/v1/metrics`
4. **Scale up:** Use distributed training for larger datasets
5. **Integrate:** Call API from your applications

## Directory Structure After Training

```
task_management/
├── index.html                    # Todo app (separate)
├── config.py
├── train.py
├── run_api.py
├── test_api.py
├── requirements.txt
├── README.md
├── GETTING_STARTED.md           # This file
├── .env.example
├── train.csv                    # Input data
├── data/
│   └── train_woheader.csv
├── models/                      # Generated after training
│   ├── random_forest.p
│   ├── onehot_encoder.p
│   ├── logistic_regression.p
│   ├── decision_tree.p
│   ├── svm.p
│   └── logistic_regression_baseline.p
├── features/                    # Generated after training
│   ├── graph.p
│   ├── missing_edges_final.p
│   ├── page_rank.p
│   ├── katz.p
│   ├── hits.p
│   ├── wcc.p
│   └── svd_components.p
└── src/
    ├── __init__.py
    ├── api.py
    ├── data_processor.py
    ├── feature_engineer.py
    └── model_trainer.py
```

---

**Happy predicting!** 🚀
