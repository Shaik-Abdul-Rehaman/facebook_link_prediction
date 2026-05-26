import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
FEATURES_DIR = BASE_DIR / 'features'

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
FEATURES_DIR.mkdir(exist_ok=True)

# Dataset paths
TRAIN_CSV = BASE_DIR / 'train.csv'
GRAPH_CSV = DATA_DIR / 'train_woheader.csv'

# Pickle files for intermediate data
GRAPH_FILE = FEATURES_DIR / 'graph.p'
MISSING_EDGES_FILE = FEATURES_DIR / 'missing_edges_final.p'
PAGERANK_FILE = FEATURES_DIR / 'page_rank.p'
KATZ_FILE = FEATURES_DIR / 'katz.p'
HITS_FILE = FEATURES_DIR / 'hits.p'
WCC_FILE = FEATURES_DIR / 'wcc.p'
SVD_COMPONENTS_FILE = FEATURES_DIR / 'svd_components.p'

# Feature CSV files
FEATURES_TRAIN_CSV = FEATURES_DIR / 'features_train.csv'
FEATURES_TEST_CSV = FEATURES_DIR / 'features_test.csv'

# Model files
RFC_MODEL = MODELS_DIR / 'random_forest.p'
RFC_ENCODER = MODELS_DIR / 'onehot_encoder.p'
LR_MODEL = MODELS_DIR / 'logistic_regression.p'
DTC_MODEL = MODELS_DIR / 'decision_tree.p'
SVM_MODEL = MODELS_DIR / 'svm.p'
LR_BASELINE_MODEL = MODELS_DIR / 'logistic_regression_baseline.p'

# Feature sets
FEATURE_COLS = [
    'num_followers_s', 'num_followers_d', 'num_followees_s', 'num_followees_d',
    'inter_followers', 'inter_followees', 'jaccard_followers', 'jaccard_followees',
    'cosine_followers', 'cosine_followees', 'pagerank_s', 'pagerank_d',
    'katz_s', 'katz_d', 'hits_hub_s', 'hits_hub_d', 'hits_auth_s', 'hits_auth_d',
    'shortest_path', 'same_wcc', 'adar_index', 'follows_back',
    'svd_u_s_1', 'svd_u_s_2', 'svd_u_s_3', 'svd_u_s_4', 'svd_u_s_5', 'svd_u_s_6',
    'svd_u_d_1', 'svd_u_d_2', 'svd_u_d_3', 'svd_u_d_4', 'svd_u_d_5', 'svd_u_d_6'
]

# API settings
BATCH_PREDICT_LIMIT = 1000
MAX_NODES = 1862220
API_VERSION = '1.0.0'

# Model selection (can be changed)
PRODUCTION_MODEL = 'random_forest_ensemble'  # Options: random_forest_ensemble, logistic_regression, svm, decision_tree
