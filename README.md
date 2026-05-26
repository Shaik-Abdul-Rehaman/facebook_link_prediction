# Link Prediction - Facebook Social Network

A production-ready machine learning project for predicting missing links in Facebook social networks using graph features and ensemble methods.

## Project Structure

```
.
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── train.py                  # Training pipeline
├── run_api.py               # Flask API server
├── test_api.py              # API test suite
├── README.md                # This file
├── data/                    # Data directory (created automatically)
├── models/                  # Trained models (created after training)
├── features/                # Intermediate feature files
└── src/
    ├── __init__.py
    ├── api.py              # Flask API endpoints
    ├── data_processor.py   # Data loading and preprocessing
    ├── feature_engineer.py # Feature engineering
    └── model_trainer.py    # Model training and evaluation
```

## Installation

1. **Clone and setup:**
```bash
cd task_management
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Prepare data:**
```bash
# Place train.csv in the root directory
# This should contain columns: source_node, destination_node
```

## Training the Model

Train all models (Decision Tree, SVM, Logistic Regression, Random Forest Ensemble):

```bash
python train.py
```

This will:
1. Load and build the graph from `train.csv`
2. Generate negative edges (non-existing links)
3. Compute network centrality measures (PageRank, Katz, HITS)
4. Extract SVD features from the adjacency matrix
5. Engineer 30+ features for each edge
6. Train 4 different models
7. Evaluate and save all models

**Note:** First run takes 1-2 hours depending on data size and hardware.

## Running the API Server

```bash
python run_api.py
```

The API will start on `http://localhost:5000`

## API Endpoints

### 1. Health Check
```
GET /health
```
Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "model": "random_forest_ensemble",
  "version": "1.0.0"
}
```

### 2. Single/Batch Prediction
```
POST /api/v1/predict
```
Predict link existence for given edges.

**Request:**
```json
{
  "edges": [
    {"source_node": 123, "destination_node": 456},
    {"source_node": 789, "destination_node": 101}
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
  ],
  "timestamp": "2024-01-15T10:30:00"
}
```

### 3. Batch Prediction (Optimized)
```
POST /api/v1/batch-predict
```
Optimized endpoint for large batches (up to 1000 edges).

**Request/Response:** Same as `/predict` but optimized for performance.

### 4. Model Information
```
GET /api/v1/model-info
```
Get details about the current model and available features.

**Response:**
```json
{
  "model": "random_forest_ensemble",
  "version": "1.0.0",
  "features": ["num_followers_s", "jaccard_followers", ...],
  "max_batch_size": 1000,
  "available_models": ["random_forest_ensemble", "logistic_regression", "svm", "decision_tree"]
}
```

### 5. Model Metrics
```
GET /api/v1/metrics
```
Retrieve evaluation metrics for all trained models.

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
    "logistic_regression": {...}
  }
}
```

## Testing the API

```bash
python test_api.py
```

This runs a comprehensive test suite including:
- Health check
- Model info retrieval
- Single edge prediction
- Batch prediction
- Metrics retrieval
- Large batch (1000 edges) performance test

## Features Engineered

The model uses 30+ features across multiple categories:

### Basic Node Features
- `num_followers_s/d`: Number of followers for each node
- `num_followees_s/d`: Number of people followed by each node
- `inter_followers`: Common followers between nodes
- `inter_followees`: Common people followed by both nodes

### Similarity Features
- `jaccard_followers/followees`: Jaccard similarity index
- `cosine_followers/followees`: Cosine similarity

### Centrality Features
- `pagerank_s/d`: PageRank centrality
- `katz_s/d`: Katz centrality
- `hits_hub_s/d`: HITS hub scores
- `hits_auth_s/d`: HITS authority scores

### Path Features
- `shortest_path`: Shortest path length between nodes
- `same_wcc`: Nodes in same weakly connected component
- `adar_index`: Adamic-Adar similarity index
- `follows_back`: Whether target follows source back

### SVD Features
- `svd_u_s_1` to `svd_u_s_6`: SVD components for source node
- `svd_u_d_1` to `svd_u_d_6`: SVD components for destination node

## Models Included

1. **Random Forest Ensemble (Recommended)**
   - Random Forest + One-Hot Encoding + Logistic Regression
   - Best overall performance
   - Best for production use

2. **Logistic Regression**
   - Fast and interpretable
   - Good baseline model

3. **Support Vector Machine (SVM)**
   - Linear SVM with strong generalization

4. **Decision Tree**
   - Interpretable, good for feature importance

## Configuration

Edit `config.py` to customize:
- Model selection: `PRODUCTION_MODEL`
- Batch prediction limit: `BATCH_PREDICT_LIMIT`
- Feature columns: `FEATURE_COLS`
- Data paths and directories

## Deployment

### Using Gunicorn (Production)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 run_api:app
```

### Using Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "run_api:app"]
```

## Performance Notes

- **Training time:** 1-2 hours (first run with full graph)
- **API response time:** ~100-200ms for single prediction
- **Batch processing:** ~50-100ms per edge for batch operations
- **Max batch size:** 1000 edges
- **Memory:** ~2-4GB for full model + graph data

## Troubleshooting

1. **Models not found error:**
   ```bash
   python train.py
   ```
   Train the models first before running the API.

2. **Out of memory:**
   - Reduce batch size
   - Use lighter model (Logistic Regression instead of Random Forest)
   - Process in smaller chunks

3. **API connection refused:**
   - Ensure API server is running
   - Check port 5000 is not in use
   - Try different port in `config.py`

## API Examples

### Using cURL

```bash
# Health check
curl http://localhost:5000/health

# Predict single edge
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"edges": [{"source_node": 1, "destination_node": 2}]}'

# Get model info
curl http://localhost:5000/api/v1/model-info
```

### Using Python

```python
import requests

url = "http://localhost:5000/api/v1/predict"
payload = {
    "edges": [
        {"source_node": 1, "destination_node": 2},
        {"source_node": 3, "destination_node": 4}
    ]
}
response = requests.post(url, json=payload)
print(response.json())
```

## Model Selection

To use a different model, modify `config.py`:

```python
PRODUCTION_MODEL = 'logistic_regression'  # or 'svm', 'decision_tree'
```

Or set environment variable:
```bash
export PRODUCTION_MODEL=svm
python run_api.py
```

## License

Internal use only.

## Support

For issues or questions, contact the data science team.
