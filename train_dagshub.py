import os
import mlflow
import dagshub
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from skopt import BayesSearchCV

# 1. Konfigurasi DagsHub & MLflow
REPO_OWNER = "ufff9"
REPO_NAME = "Chronic-Kidney-Disease-CKD"

dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
mlflow.set_experiment("CKD_Stage_Detection")

def train_ckd():
    # 2. Load Dataset
    train_df = pd.read_csv("Training_CKD_dataset.csv")
    test_df = pd.read_csv("Testing_CKD_dataset.csv")

    # Preprocessing: rapikan nama kolom
    train_df.columns = train_df.columns.str.lower().str.replace(' ', '_')
    test_df.columns = test_df.columns.str.lower().str.replace(' ', '_')

    # 3. Split fitur & target
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    # Encoding fitur kategorikal
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)

    # Samakan kolom train & test
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # 🔥 FIX: simpan nama fitur
    feature_names = X_train.columns.tolist()
    feat_path = "feature_names.pkl"

    # Encoding label
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # 4. Pipeline (ANTI DATA LEAKAGE ✅)
    pipeline = ImbPipeline([
        ('scaler', MinMaxScaler()),
        ('smoteenn', SMOTEENN(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Parameter Bayesian Optimization
    param_space = {
        'classifier__n_estimators': (100, 500),
        'classifier__max_depth': (5, 50),
        'classifier__min_samples_split': (2, 10)
    }

    # 5. Training + MLflow
    with mlflow.start_run(run_name="RF_Bayesian_Optimization"):

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("optimization", "BayesSearchCV")

        opt = BayesSearchCV(
            estimator=pipeline,
            search_spaces=param_space,
            n_iter=10,
            cv=StratifiedKFold(3),
            n_jobs=-1,
            random_state=42
        )

        print("Memulai training...")
        opt.fit(X_train, y_train_encoded)

        # 6. Evaluasi
        y_pred = opt.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test_encoded, y_pred),
            "precision": precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0),
            "best_cv_score": opt.best_score_
        }

        mlflow.log_metrics(metrics)
        mlflow.log_params(opt.best_params_)

        # 7. Simpan artifact
        joblib.dump(opt.best_estimator_, 'ckd_pipeline_model.pkl')
        joblib.dump(le, 'label_encoder.pkl')
        joblib.dump(feature_names, feat_path)

        mlflow.log_artifact('ckd_pipeline_model.pkl')
        mlflow.log_artifact('label_encoder.pkl')
        mlflow.log_artifact(feat_path)

        print(f"Training selesai. Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    train_ckd()