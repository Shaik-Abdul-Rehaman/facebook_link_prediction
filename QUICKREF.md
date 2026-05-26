# Link Prediction API - Quick Reference

## Setup (Copy-Paste Ready)

```bash
# 1. Create environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (1-2 hours first time)
python train.py

# 4. Start API
python run_api.py

# 5. Test (in another terminal)
python test_api.py
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/api/v1/predict` | POST | Predict links (batch up to 1000) |
| `/api/v1/batch-predict` | POST | Batch optimized predictions |
| `/api/v1/model-info` | GET | Model & feature info |
| `/api/v1/metrics` | GET | Model performance metrics |

## cURL Examples

### Predict Single Link
```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"edges": [{"source_node": 1, "destination_node": 2}]}'
```

### Predict Multiple Links
```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "edges": [
      {"source_node": 1, "destination_node": 2},
      {"source_node": 3, "destination_node": 4},
      {"source_node": 5, "destination_node": 6}
    ]
  }'
```

### Get Model Info
```bash
curl http://localhost:5000/api/v1/model-info
```

### Get Metrics
```bash
curl http://localhost:5000/api/v1/metrics
```

## Python Examples

### Single Prediction
```python
import requests

response = requests.post('http://localhost:5000/api/v1/predict', json={
    'edges': [{'source_node': 1, 'destination_node': 2}]
})

pred = response.json()['predictions'][0]
print(f"Link probability: {pred['probability']:.2%}")
```

### Batch Prediction
```python
import requests

edges = [{'source_node': i, 'destination_node': i+100} for i in range(1, 101)]

response = requests.post('http://localhost:5000/api/v1/predict', json={'edges': edges})

for pred in response.json()['predictions']:
    print(f"{pred['source_node']} → {pred['destination_node']}: {pred['probability']:.2%}")
```

### Get Metrics
```python
import requests
import pandas as pd

response = requests.get('http://localhost:5000/api/v1/metrics')
metrics_df = pd.DataFrame(response.json()['metrics']).T
print(metrics_df)
```

## Configuration

### Change Production Model
Edit `config.py`:
```python
PRODUCTION_MODEL = 'logistic_regression'  # or 'svm', 'decision_tree', 'random_forest_ensemble'
```

### Change API Port
Edit `config.py`:
```python
# In run_api.py:
app.run(debug=False, host='0.0.0.0', port=5001)  # Change 5000 to 5001
```

### Change Batch Limit
Edit `config.py`:
```python
BATCH_PREDICT_LIMIT = 500  # Default: 1000
```

## File Structure

```
task_management/
├── config.py              # Configuration
├── train.py              # Training script
├── run_api.py            # API server
├── test_api.py           # Tests
├── train.csv             # Input data
├── requirements.txt      # Dependencies
├── README.md             # Full documentation
├── GETTING_STARTED.md    # Detailed guide
├── QUICKREF.md           # This file
└── src/
    ├── api.py           # Flask API
    ├── data_processor.py # Data loading
    ├── feature_engineer.py # Feature generation
    └── model_trainer.py # Model training
```

## Common Tasks

### Train with New Data
```bash
# Replace train.csv and run
python train.py
```

### Use Different Model
1. Edit `config.py` → `PRODUCTION_MODEL`
2. Restart API: `python run_api.py`

### Increase Performance
Use lighter model:
```python
PRODUCTION_MODEL = 'logistic_regression'
```

### Deploy with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 run_api:app
```

### Deploy with Docker
```bash
docker build -t link-pred .
docker run -p 5000:5000 link-pred
```

## Response Format

### Success Response
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

### Error Response
```json
{
  "error": "Missing 'edges' field",
  "timestamp": "2024-01-15T10:30:00"
}
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Models not found | Run `python train.py` |
| Port already in use | Change port in `run_api.py` |
| Out of memory | Use lighter model or reduce batch size |
| Slow predictions | Use Logistic Regression model |
| Connection refused | Ensure API running: `python run_api.py` |

## Available Models & Performance

| Model | Accuracy | Speed |
|-------|----------|-------|
| Random Forest Ensemble | 92% | Medium |
| Logistic Regression | 89% | Fast |
| SVM | 87% | Medium |
| Decision Tree | 85% | Fast |

## Max Request Sizes

- Single prediction: 1 edge
- Batch prediction: 1000 edges (configurable)
- Feature set: 30+ features auto-computed

## API Rate Limits

No built-in rate limiting (configure in your deployment)

## Model Info

```bash
curl http://localhost:5000/api/v1/model-info | python -m json.tool
```

Shows:
- Current model in use
- Available models
- Feature list
- Max batch size
- API version

---

**Need help?** Read `README.md` or `GETTING_STARTED.md`
