import os
import time
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

def run_benchmark():
    # File path
    data_path = os.path.expanduser("~/ml-benchmark/creditcard.csv")
    if not os.path.exists(data_path):
        data_path = "creditcard.csv"
        
    print(f"Loading dataset from {data_path}...")
    start_load = time.time()

    # Dataset không có header nên đọc header=None
    df = pd.read_csv(data_path, header=None)

    # Gán tên cột chuẩn cho Credit Card Fraud Dataset
    df.columns = [
        "Time",
        "V1", "V2", "V3", "V4", "V5",
        "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15",
        "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25",
        "V26", "V27", "V28",
        "Amount",
        "Class"
    ]
    data_loading_time = time.time() - start_load
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns in {data_loading_time:.4f}s")
    
    # Feature / Target split
    X = df.drop(columns=["Class"])
    y = df["Class"]
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model training
    print("Training LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    start_train = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_train
    print(f"Training completed in {training_time:.4f}s")
    
    # Evaluation on test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc_roc = float(roc_auc_score(y_test, y_pred_proba))
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    best_iter = int(model.best_iteration_) if hasattr(model, 'best_iteration_') and model.best_iteration_ else 100
    
    # Inference Latency (1 row)
    sample_1 = X_test.iloc[:1]
    _ = model.predict_proba(sample_1) # Warmup
    
    start_lat = time.time()
    for _ in range(100):
        _ = model.predict_proba(sample_1)
    latency_ms = ((time.time() - start_lat) / 100.0) * 1000.0
    
    # Inference Throughput (1000 rows)
    sample_1000 = X_test.iloc[:1000]
    start_tp = time.time()
    for _ in range(10):
        _ = model.predict_proba(sample_1000)
    elapsed_tp = (time.time() - start_tp) / 10.0
    throughput_rows_per_sec = 1000.0 / elapsed_tp if elapsed_tp > 0 else 0.0
    
    results = {
        "data_loading_time_seconds": round(data_loading_time, 4),
        "training_time_seconds": round(training_time, 4),
        "best_iteration": best_iter,
        "auc_roc": round(auc_roc, 6),
        "accuracy": round(acc, 6),
        "f1_score": round(f1, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "inference_latency_ms_per_row": round(latency_ms, 4),
        "inference_throughput_rows_per_sec": round(throughput_rows_per_sec, 2)
    }
    
    print("\n--- BENCHMARK RESULTS ---")
    print(json.dumps(results, indent=2))
    
    output_path = os.path.expanduser("~/ml-benchmark/benchmark_result.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {output_path}")

if __name__ == "__main__":
    run_benchmark()
