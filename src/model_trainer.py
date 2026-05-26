import pandas as pd
import numpy as np
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from config import *

class ModelTrainer:
    def __init__(self):
        self.dtc = None
        self.svm = None
        self.lr = None
        self.rfc = None
        self.rfc_encoder = None
        self.proposed = None
        self.models = {}
        self.metrics = {}

    def train_decision_tree(self, X_train, y_train):
        print("Training Decision Tree Classifier...")
        self.dtc = DecisionTreeClassifier(
            criterion='gini', max_depth=5, max_features='sqrt',
            min_samples_leaf=52, min_samples_split=120, random_state=25
        )
        self.dtc.fit(X_train, y_train)
        self.models['decision_tree'] = self.dtc
        print("Decision Tree trained.")
        return self.dtc

    def train_svm(self, X_train, y_train):
        print("Training Linear SVM...")
        self.svm = LinearSVC(random_state=25, max_iter=10000)
        self.svm.fit(X_train, y_train)
        self.models['svm'] = self.svm
        print("SVM trained.")
        return self.svm

    def train_logistic_regression(self, X_train, y_train):
        print("Training Logistic Regression...")
        self.lr = LogisticRegression(random_state=25, max_iter=1000)
        self.lr.fit(X_train, y_train)
        self.models['logistic_regression'] = self.lr
        print("Logistic Regression trained.")
        return self.lr

    def train_random_forest_ensemble(self, X_train, y_train):
        print("Training Random Forest...")
        self.rfc = RandomForestClassifier(
            bootstrap=True, criterion='gini', max_depth=5,
            max_features='sqrt', min_samples_leaf=52,
            min_samples_split=120, n_estimators=10,
            n_jobs=-1, random_state=25, verbose=0
        )
        self.rfc.fit(X_train, y_train)
        self.models['random_forest'] = self.rfc

        print("Training OneHotEncoder on Random Forest leaf indices...")
        rf_leaves = self.rfc.apply(X_train)
        self.rfc_encoder = OneHotEncoder(sparse_output=False)
        self.rfc_encoder.fit(rf_leaves)

        print("Training Logistic Regression on encoded features...")
        rf_encoded = self.rfc_encoder.transform(rf_leaves)
        self.proposed = LogisticRegression(solver='lbfgs', max_iter=3000)
        self.proposed.fit(rf_encoded, y_train)
        self.models['random_forest_ensemble'] = (self.rfc, self.rfc_encoder, self.proposed)

        print("Random Forest Ensemble trained.")
        return self.rfc, self.rfc_encoder, self.proposed

    def evaluate_model(self, model_name, model, X_test, y_test):
        print(f"\nEvaluating {model_name}...")

        if model_name == 'random_forest_ensemble':
            rfc, encoder, lr = model
            predictions = lr.predict(encoder.transform(rfc.apply(X_test)))
        else:
            predictions = model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1': f1_score(y_test, predictions, zero_division=0)
        }

        self.metrics[model_name] = metrics
        print(f"{model_name} Results:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")

        return metrics

    def train_all_models(self, X_train, y_train, X_test, y_test):
        print("=" * 50)
        print("TRAINING ALL MODELS")
        print("=" * 50)

        # Train models
        self.train_decision_tree(X_train, y_train)
        self.train_svm(X_train, y_train)
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest_ensemble(X_train, y_train)

        # Evaluate all models
        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)

        for model_name, model in self.models.items():
            self.evaluate_model(model_name, model, X_test, y_test)

        # Save metrics summary
        metrics_df = pd.DataFrame(self.metrics).T
        print("\n" + "=" * 50)
        print("METRICS SUMMARY")
        print("=" * 50)
        print(metrics_df)

        return self.models, self.metrics

    def save_models(self):
        print("\nSaving models...")
        MODELS_DIR.mkdir(exist_ok=True)

        if self.dtc:
            pickle.dump(self.dtc, open(DTC_MODEL, 'wb'))
            print(f"Saved Decision Tree to {DTC_MODEL}")

        if self.svm:
            pickle.dump(self.svm, open(SVM_MODEL, 'wb'))
            print(f"Saved SVM to {SVM_MODEL}")

        if self.lr:
            pickle.dump(self.lr, open(LR_BASELINE_MODEL, 'wb'))
            print(f"Saved Logistic Regression to {LR_BASELINE_MODEL}")

        if self.rfc and self.rfc_encoder and self.proposed:
            pickle.dump(self.rfc, open(RFC_MODEL, 'wb'))
            pickle.dump(self.rfc_encoder, open(RFC_ENCODER, 'wb'))
            pickle.dump(self.proposed, open(LR_MODEL, 'wb'))
            print(f"Saved Random Forest Ensemble to {RFC_MODEL}, {RFC_ENCODER}, {LR_MODEL}")

    def load_production_model(self, model_name=PRODUCTION_MODEL):
        print(f"Loading production model: {model_name}")

        if model_name == 'random_forest_ensemble':
            rfc = pickle.load(open(RFC_MODEL, 'rb'))
            encoder = pickle.load(open(RFC_ENCODER, 'rb'))
            lr = pickle.load(open(LR_MODEL, 'rb'))
            return ('random_forest_ensemble', (rfc, encoder, lr))

        elif model_name == 'logistic_regression':
            lr = pickle.load(open(LR_BASELINE_MODEL, 'rb'))
            return ('logistic_regression', lr)

        elif model_name == 'svm':
            svm = pickle.load(open(SVM_MODEL, 'rb'))
            return ('svm', svm)

        elif model_name == 'decision_tree':
            dtc = pickle.load(open(DTC_MODEL, 'rb'))
            return ('decision_tree', dtc)

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def predict(self, model, features):
        model_name, model_obj = model

        if model_name == 'random_forest_ensemble':
            rfc, encoder, lr = model_obj
            rf_leaves = rfc.apply(features)
            encoded = encoder.transform(rf_leaves)
            predictions = lr.predict(encoded)
            probabilities = lr.predict_proba(encoded)[:, 1]
        else:
            predictions = model_obj.predict(features)
            if hasattr(model_obj, 'predict_proba'):
                probabilities = model_obj.predict_proba(features)[:, 1]
            else:
                probabilities = predictions

        return predictions, probabilities
