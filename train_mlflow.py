import os
import mlflow
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE # 1. Import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from skopt import BayesSearchCV

# 1. Konfigurasi MLflow Lokal "mlflow ui"
mlflow.set_tracking_uri("file:///C:/MLflow_Projects/CKD_Stage_Detection/mlruns")
mlflow.set_experiment("CKD_Stage_Detection_Local")

def train_ckd():
    # 2. Load Dataset
    try:
        train_df = pd.read_csv("Training_CKD_dataset.csv")
        test_df = pd.read_csv("Testing_CKD_dataset.csv")
    except FileNotFoundError:
        print("Error: File dataset tidak ditemukan. Pastikan file CSV ada di folder yang sama.")
        return

    # Preprocessing: Lowercase kolom
    train_df.columns = train_df.columns.str.lower().str.replace(' ', '_')
    test_df.columns = test_df.columns.str.lower().str.replace(' ', '_')

    # 3. Pemisahan Fitur dan Target
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    # Simpan nama fitur asli untuk keperluan deployment
    # (Pipeline akan memfilter ini secara otomatis nantinya karena RFE ada di dalam pipeline)
    feature_names = list(pd.get_dummies(X_train, drop_first=True).columns)

    # Encoding Kategorikal
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # 4. Definisi Pipeline dengan RFE
    # Menggunakan estimator RandomForest untuk RFE. 
    # (Catatan: Jika proses terlalu lama, ganti estimator RFE dengan DecisionTreeClassifier)
    rfe_estimator = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    pipeline = ImbPipeline([
        ('scaler', MinMaxScaler()),
        ('smoteenn', SMOTEENN(random_state=42)),
        ('rfe', RFE(estimator=rfe_estimator, step=1)), # 2. Tambahkan RFE ke dalam pipeline
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Jumlah fitur dinamis untuk batas atas BayesSearchCV
    n_features = X_train.shape[1]

    # 5. Konfigurasi Param Space (termasuk tuning jumlah fitur yang akan dipilih RFE)
    param_space = {
        'rfe__n_features_to_select': (1, n_features), # Mencari jumlah fitur terbaik dari 1 hingga total fitur
        'classifier__n_estimators': (100, 500),
        'classifier__max_depth': (5, 50),
        'classifier__min_samples_split': (2, 10)
    }

    # 6. Eksekusi dengan MLflow Tracking Lokal
    with mlflow.start_run(run_name="RF_RFE_Local_Optimization"):
        mlflow.log_param("model_type", "RandomForest_with_RFE")
        mlflow.log_param("optimization", "BayesSearchCV")
        mlflow.log_param("total_initial_features", n_features)

        opt = BayesSearchCV(
            estimator=pipeline,
            search_spaces=param_space,
            n_iter=10,
            cv=StratifiedKFold(3),
            n_jobs=-1,
            random_state=42
        )

        print("Memulai proses training lokal dengan RFE...")
        opt.fit(X_train, y_train_encoded)

        # 7. Evaluasi
        y_pred = opt.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test_encoded, y_pred),
            "precision": precision_score(y_test_encoded, y_pred, average='weighted'),
            "recall": recall_score(y_test_encoded, y_pred, average='weighted'),
            "f1_score": f1_score(y_test_encoded, y_pred, average='weighted'),
            "best_cv_score": opt.best_score_
        }

        mlflow.log_metrics(metrics)
        
        # Log parameter terbaik hasil optimasi (termasuk jumlah fitur terbaik dari RFE)
        best_params = {f"best_{k}": v for k, v in opt.best_params_.items()}
        mlflow.log_params(best_params)

        # Mengambil informasi fitur mana saja yang terpilih oleh RFE
        best_pipeline = opt.best_estimator_
        rfe_step = best_pipeline.named_steps['rfe']
        selected_features_mask = rfe_step.support_
        selected_features = [feature_names[i] for i, selected in enumerate(selected_features_mask) if selected]
        
        print(f"\nFitur awal: {n_features}")
        print(f"Fitur terpilih oleh RFE: {len(selected_features)}")
        # Menyimpan nama fitur yang lolos seleksi sebagai text artifact (opsional tapi sangat berguna)
        with open("selected_features.txt", "w") as f:
            f.write("\n".join(selected_features))
        mlflow.log_artifact("selected_features.txt")

        # 8. Simpan Artifact Secara Lokal
        model_path = 'ckd_pipeline_model.pkl'
        le_path = 'label_encoder.pkl'
        feat_path = 'feature_names.pkl'

        # Kita tetap menyimpan seluruh pipeline. Saat inferensi/prediksi, Anda cukup memasukkan
        # data dengan kolom asli, dan pipeline akan melakukan SMOTEENN -> RFE otomatis.
        joblib.dump(best_pipeline, model_path)
        joblib.dump(le, le_path)
        joblib.dump(feature_names, feat_path)
        
        # Log ke MLflow
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(le_path)
        mlflow.log_artifact(feat_path)

        print(f"Training Selesai!")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Jumlah Fitur yang Terpilih: {best_params['best_rfe__n_features_to_select']}")

if __name__ == "__main__":
    train_ckd()