# Installation & Setup Guide

## Complete Installation (Copy-Paste Ready)

### Step 1: Verify Setup
```bash
cd C:/Users/reham/OneDrive/Desktop/bb2/task_management
python verify_setup.py
```

This checks:
- All source files exist
- Documentation complete
- Configuration files in place
- Python dependencies available

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs (on first run):
- Flask (API framework)
- Pandas (data processing)
- NumPy (numerical computing)
- scikit-learn (machine learning)
- NetworkX (graph processing)
- SciPy (scientific computing)
- XGBoost (ensemble methods)

### Step 3: Verify Data
```bash
# Check if train.csv exists
ls -l train.csv

# Or on Windows:
dir train.csv
```

File should be in root directory: `C:/Users/reham/OneDrive/Desktop/bb2/task_management/train.csv`

### Step 4: Train Models
```bash
python train.py
```

This will:
- Build graph from train.csv
- Generate negative edges (~9.4M)
- Compute network features
- Extract SVD components
- Train 4 different models
- Save models to `models/` directory
- **Time: 1-2 hours (first run)**

Once complete, you'll see:
```
============================================================
TRAINING COMPLETE
============================================================

Next steps:
1. Start the API server: python run_api.py
2. Test the API: python test_api.py
3. Or use curl to make predictions
```

### Step 5: Start API Server
```bash
python run_api.py
```

Output:
```
============================================================
STARTING FLASK API SERVER
============================================================

API Documentation:
  - Health Check: GET http://localhost:5000/health
  - Predict: POST http://localhost:5000/api/v1/predict
  - Batch Predict: POST http://localhost:5000/api/v1/batch-predict
  - Model Info: GET http://localhost:5000/api/v1/model-info
  - Metrics: GET http://localhost:5000/api/v1/metrics

Server running on http://localhost:5000
============================================================
```

### Step 6: Test API (New Terminal)
```bash
python test_api.py
```

Output will show:
- ✓ PASS: Health Check
- ✓ PASS: Model Info
- ✓ PASS: Single Prediction
- ✓ PASS: Batch Prediction
- ✓ PASS: Metrics
- ✓ PASS: Large Batch (1000 edges)

---

## Alternative Installation Methods

### Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Follow steps 3-6 above
```

### Using Conda

```bash
# Create conda environment
conda create -n link-prediction python=3.9

# Activate
conda activate link-prediction

# Install dependencies
pip install -r requirements.txt

# Follow steps 3-6 above
```

### Using Docker

```bash
# Build image
docker build -t link-prediction .

# Run container (will train on first run)
docker run -it -p 5000:5000 \
  -v $(pwd)/train.csv:/app/train.csv \
  link-prediction

# Wait for training to complete
# API will be available at http://localhost:5000
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# Check logs
docker-compose logs -f api

# Stop services
docker-compose down
```

---

## Verification at Each Step

### After Step 1: Verify Setup
```bash
python verify_setup.py
```
Expected: All checks pass ✓

### After Step 2: Check Dependencies
```bash
python -c "import pandas; import numpy; import sklearn; import networkx; print('✓ All dependencies OK')"
```
Expected: ✓ All dependencies OK

### After Step 3: Check Data
```bash
python -c "import pandas as pd; df = pd.read_csv('train.csv'); print(f'✓ Data loaded: {df.shape[0]} edges')"
```
Expected: ✓ Data loaded: N edges

### After Step 4: Check Models
```bash
ls -la models/
```
Expected: 6 model files exist:
- logistic_regression.p
- onehot_encoder.p
- random_forest.p
- decision_tree.p
- svm.p
- logistic_regression_baseline.p

### After Step 5: Check API Server
```bash
curl http://localhost:5000/health
```
Expected: JSON response with status "healthy"

### After Step 6: Check Tests
```bash
python test_api.py
```
Expected: All 6 tests pass ✓

---

## Troubleshooting Installation

### Issue: "No module named 'pandas'"
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

### Issue: "train.csv not found"
```bash
# Solution: Verify file location
ls -l train.csv

# Should output file info, not "file not found"
```

### Issue: "Permission denied" when running script
```bash
# On Windows (PowerShell): use python explicitly
python train.py

# On Mac/Linux: make executable
chmod +x train.py
```

### Issue: Port 5000 already in use
```bash
# Find and stop process using port 5000
# Then start API on different port (edit run_api.py)

# Or use Gunicorn on different port:
gunicorn -w 4 -b 0.0.0.0:5001 run_api:app
```

### Issue: "out of memory" error during training
```bash
# Solution: Ensure 4GB+ RAM available
# Check available memory:
# Windows: Task Manager → Performance
# Mac: Activity Monitor
# Linux: free -h
```

### Issue: "ModuleNotFoundError: No module named 'networkx'"
```bash
# Solution: Reinstall requirements
pip install --upgrade -r requirements.txt

# Or install specific package:
pip install networkx scipy scikit-learn
```

---

## System Requirements

### Minimum
- Python 3.8+
- 4 GB RAM
- 5 GB disk space
- 2-4 hours for training

### Recommended
- Python 3.9+
- 8 GB RAM (for faster training)
- 10 GB disk space (for models + features)
- SSD for faster I/O

### Operating Systems
- ✓ Windows 10/11
- ✓ macOS 10.14+
- ✓ Ubuntu 18.04+
- ✓ Any Linux distribution

---

## Quick Verification Checklist

After installation, verify:

- [ ] Python 3.8+ installed: `python --version`
- [ ] Dependencies installed: `python verify_setup.py`
- [ ] train.csv exists: `ls train.csv`
- [ ] Models trained: `ls models/*.p`
- [ ] API starts: `python run_api.py`
- [ ] API responds: `curl http://localhost:5000/health`
- [ ] Tests pass: `python test_api.py`

---

## Getting Help

1. **Check Documentation**
   - README.md - Full docs
   - GETTING_STARTED.md - Step-by-step guide
   - QUICKREF.md - Quick reference

2. **Run Verification**
   ```bash
   python verify_setup.py
   ```

3. **Check Logs**
   - During training: Check console output
   - API server: Check console output
   - Tests: Review test output

4. **Common Issues**
   - See Troubleshooting section above
   - See GETTING_STARTED.md troubleshooting section

---

## Next Steps After Installation

1. **Learn the API**
   - Read [QUICKREF.md](QUICKREF.md)
   - Try examples with cURL

2. **Integrate into Your App**
   - Use REST API endpoints
   - See Python examples in [QUICKREF.md](QUICKREF.md)

3. **Customize**
   - Modify features in src/feature_engineer.py
   - Change models in src/model_trainer.py
   - Adjust settings in config.py

4. **Deploy**
   - Use Docker for containerization
   - Deploy with Gunicorn for production
   - See README.md deployment section

---

**Installation Time:** 
- Minimal setup: 5 minutes
- Full setup (including training): 1-2 hours

**Support:** See README.md for help

**Status:** ✅ Ready to use
