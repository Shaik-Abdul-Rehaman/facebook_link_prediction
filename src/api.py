from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import traceback
import os
from datetime import datetime
from config import *
from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer

app = Flask(__name__)
CORS(app)

# Global objects
model_trainer = ModelTrainer()
production_model = None
feature_engineer = None
graph = None

@app.before_request
def load_model():
    global production_model, feature_engineer, graph
    if production_model is None:
        try:
            production_model = model_trainer.load_production_model(PRODUCTION_MODEL)
            print(f"Loaded production model: {PRODUCTION_MODEL}")

            # Load necessary data for feature engineering
            if os.path.exists(GRAPH_FILE):
                graph = pickle.load(open(GRAPH_FILE, 'rb'))
                pagerank = pickle.load(open(PAGERANK_FILE, 'rb'))
                katz = pickle.load(open(KATZ_FILE, 'rb'))
                hits = pickle.load(open(HITS_FILE, 'rb'))
                wcc = pickle.load(open(WCC_FILE, 'rb'))
                svd_data = pickle.load(open(SVD_COMPONENTS_FILE, 'rb'))

                feature_engineer = FeatureEngineer(
                    graph=graph,
                    pagerank=pagerank,
                    katz=katz,
                    hits=hits,
                    wcc=wcc,
                    adj_dict=svd_data['adj_dict'],
                    svd_u=svd_data['svd_u'],
                    svd_v=svd_data['svd_v']
                )
        except Exception as e:
            print(f"Error loading model: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'model': PRODUCTION_MODEL,
        'version': API_VERSION
    }), 200

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'edges' not in data:
            return jsonify({'error': 'Missing "edges" field'}), 400

        edges = data['edges']
        if not isinstance(edges, list) or len(edges) == 0:
            return jsonify({'error': 'edges must be a non-empty list'}), 400

        if len(edges) > BATCH_PREDICT_LIMIT:
            return jsonify({
                'error': f'Too many edges. Maximum is {BATCH_PREDICT_LIMIT}'
            }), 400

        # Validate and prepare edge data
        edge_df = pd.DataFrame(edges)
        if 'source_node' not in edge_df.columns or 'destination_node' not in edge_df.columns:
            return jsonify({
                'error': 'Each edge must have "source_node" and "destination_node"'
            }), 400

        edge_df = edge_df[['source_node', 'destination_node']]

        # Engineer features
        if feature_engineer is None:
            return jsonify({'error': 'Feature engineer not initialized'}), 500

        features = feature_engineer.engineer_features(edge_df)

        # Make predictions
        predictions, probabilities = model_trainer.predict(production_model, features)

        # Format response
        results = []
        for idx, row in edge_df.iterrows():
            results.append({
                'source_node': int(row['source_node']),
                'destination_node': int(row['destination_node']),
                'prediction': int(predictions[idx]),
                'probability': float(probabilities[idx]),
                'link_exists': bool(predictions[idx])
            })

        return jsonify({
            'status': 'success',
            'model': PRODUCTION_MODEL,
            'predictions': results,
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        print(f"Error in prediction: {traceback.format_exc()}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/api/v1/batch-predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()

        if not data or 'edges' not in data:
            return jsonify({'error': 'Missing "edges" field'}), 400

        edges = data['edges']
        if not isinstance(edges, list) or len(edges) == 0:
            return jsonify({'error': 'edges must be a non-empty list'}), 400

        if len(edges) > BATCH_PREDICT_LIMIT:
            return jsonify({
                'error': f'Too many edges. Maximum is {BATCH_PREDICT_LIMIT}'
            }), 400

        edge_df = pd.DataFrame(edges)
        if 'source_node' not in edge_df.columns or 'destination_node' not in edge_df.columns:
            return jsonify({
                'error': 'Each edge must have "source_node" and "destination_node"'
            }), 400

        features = feature_engineer.engineer_features(edge_df)
        predictions, probabilities = model_trainer.predict(production_model, features)

        results = []
        for idx in range(len(edge_df)):
            results.append({
                'source_node': int(edge_df.iloc[idx]['source_node']),
                'destination_node': int(edge_df.iloc[idx]['destination_node']),
                'prediction': int(predictions[idx]),
                'probability': float(probabilities[idx])
            })

        return jsonify({
            'status': 'success',
            'total': len(results),
            'predictions': results
        }), 200

    except Exception as e:
        print(f"Error in batch prediction: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model': PRODUCTION_MODEL,
        'version': API_VERSION,
        'features': FEATURE_COLS,
        'max_batch_size': BATCH_PREDICT_LIMIT,
        'available_models': ['random_forest_ensemble', 'logistic_regression', 'svm', 'decision_tree']
    }), 200

@app.route('/api/v1/metrics', methods=['GET'])
def get_metrics():
    if hasattr(model_trainer, 'metrics') and model_trainer.metrics:
        return jsonify({
            'metrics': model_trainer.metrics,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    return jsonify({'error': 'Metrics not available'}), 404

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
