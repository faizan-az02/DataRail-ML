## Project Overview

This repository contains a machine learning training pipeline that runs on an EC2 instance and reads/writes data and models via mounted storage backed by S3. The solution focuses on supervised learning for tabular data, implementing an end‑to‑end workflow from raw data ingestion and feature engineering through model training, evaluation, and artifact logging. The primary objective is to predict an individual's income bracket based on demographic and employment attributes, a pattern commonly used for credit risk, marketing segmentation, and policy analytics use cases.

---

## Data description

The project uses a structured dataset of anonymized adult records with a binary target column `income`. Each record contains demographic and socioeconomic features such as:

- **Age and education profile** (e.g., age, education level, education years)
- **Employment and occupation signals** (e.g., work class, occupation, hours per week)
- **Marital and relationship status**
- **Geographic indicators** (e.g., native country)

The target variable `income` indicates whether an individual's annual income is **greater than or equal to USD 50K or below that threshold**. This is treated as a standard **binary classification** problem, and the pipeline reports model accuracy and stores the best‑performing model and metrics for later inspection.

---

## Infrastructure

- **EC2 instance type**: `t3.micro`
- **S3 bucket name**: `mlops-bucket-uci-dataset`
- **IAM role name**: `EC2-S3-Access`

---

## Execution steps

```bash
# 1) SSH into EC2
ssh -i EC2.pem ubuntu@public_ip

# 2)Attach IAM role and mount data/model directories
sudo mkdir -p /mnt/ml-data/datasets
sudo mkdir -p /mnt/ml-data/models
sudo mkdir -p /mnt/ml-data/logs
sudo chown -R $USER:$USER /mnt/ml-data

# 3) Bootstrap Python / ML environment
chmod +x setup_ml_env.sh
./setup_ml_env.sh

# 4) Activate virtualenv (must match setup_ml_env.sh)
source ~/ml-venv/bin/activate

# 5) Copy data into expected paths (from S3 or elsewhere)
aws s3 cp s3://mlops-bucket-uci-dataset/raw.csv /mnt/ml-data/datasets/raw.csv
aws s3 cp s3://mlops-bucket-uci-dataset/processed.csv /mnt/ml-data/datasets/processed.csv

# 6) Run training pipeline
cd /mnt/ml-data/scripts
python train_pipeline.py
```

---

## Training Run Output

RandomForest Accuracy: 0.85
LogisticRegression Accuracy: 0.85

Best model saved: /mnt/ml-data/models/LogisticRegression_model.pkl
Metrics logged in: /mnt/ml-data/logs/metrics.log

Timestamp: 2026-02-09 12:30:21