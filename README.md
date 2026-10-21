# Credit Risk ML System

An end-to-end, production-grade Credit Risk Assessment and Underwriting System powered by Machine Learning and Explainable AI (SHAP). The system predicts loan default probabilities, tunes cost-sensitive decision thresholds, explains risk drivers, and delivers real-time decisions via a FastAPI microservice and an underwriting desk interface.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Machine Learning Workflow](#machine-learning-workflow)
  - [1. Data Cleaning & Outlier Handling](#1-data-cleaning--outlier-handling)
  - [2. Preprocessing & Class Imbalance](#2-preprocessing--class-imbalance)
  - [3. Model Benchmarking & Hyperparameter Tuning](#3-model-benchmarking--hyperparameter-tuning)
  - [4. Threshold Optimization](#4-threshold-optimization)
  - [5. Model Explainability with SHAP](#5-model-explainability-with-shap)
- [API Reference](#api-reference)
- [Interactive Underwriting Desk](#interactive-underwriting-desk)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Deployment](#deployment)
- [License](#license)

---

## Overview

In consumer lending, approving a loan to a borrower who defaults (False Negative) is significantly more costly than declining a creditworthy applicant (False Positive). Traditional credit scorecards often act as opaque black boxes or lack probabilistic calibration. 

This project bridges that gap by implementing a complete machine learning lifecycle:
1. **Model Development**: Benchmarking Logistic Regression against a tuned XGBoost Classifier on 32,000+ loan records.
2. **Threshold Tuning**: Moving beyond the arbitrary 0.5 threshold to optimize the F1-score and penalize costly credit defaults.
3. **Interpretability**: Using SHAP (SHapley Additive exPlanations) for global feature importance and individual applicant-level decision explanations.
4. **Production Deployment**: A low-latency FastAPI backend serving an interactive financial ledger interface.

---

## Key Features

- **Robust Data Pipeline**: Scikit-Learn `ColumnTransformer` and `Pipeline` for leakage-free numeric imputation and categorical one-hot encoding.
- **Class Imbalance Mitigation**: Weighted loss functions (`scale_pos_weight`) to handle low default-rate skewness.
- **Calibrated Probabilities**: Threshold optimization via Precision-Recall curve analysis to strike the optimal risk-reward balance.
- **Explainable AI (XAI)**: SHAP TreeExplainer integration to understand both macro lending trends and individual application denials.
- **Type-Safe API**: Pydantic validation schemas with comprehensive error handling.
- **Interactive Underwriting Dashboard**: Modern web UI with automatic loan-to-income ratio calculations, animated SVG probability gauges, and instant risk verdicts.
- **Cloud-Ready**: Preconfigured for instant deployment on Render.

---

## System Architecture

```
┌────────────────────────────────────────────────────────┐
│             Underwriting Client Interface             │
│        (HTML5 / CSS3 / Vanilla JS Ledger UI)           │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP POST /predict (JSON)
                            ▼
┌────────────────────────────────────────────────────────┐
│                   FastAPI Microservice                 │
│  - Pydantic Schema Validation (LoanApplication)        │
│  - Lifespan Model & Threshold Cache In-Memory          │
└───────────────────────────┬────────────────────────────┘
                            │ Feature Vector
                            ▼
┌────────────────────────────────────────────────────────┐
│                 Scikit-Learn Pipeline                  │
│  - SimpleImputer (Median for Numerical)                │
│  - OneHotEncoder (Categoricals)                        │
└───────────────────────────┬────────────────────────────┘
                            │ Transformed Matrix
                            ▼
┌────────────────────────────────────────────────────────┐
│            XGBoost Classifier + SHAP Engine            │
│  - Calibrated Default Probability Output               │
│  - Optimal Cut-Off Decision Rule (Threshold ~ 0.695)   │
└────────────────────────────────────────────────────────┘
```

---

## Machine Learning Workflow

The data science pipeline is thoroughly documented in [`Credit_Risk.ipynb`](Credit_Risk.ipynb):

### 1. Data Cleaning & Outlier Handling
- Filtered erroneous domain entries (e.g., borrower age > 100 years, employment length > 60 years).
- Evaluated missing data patterns in applicant employment length and interest rates.

### 2. Preprocessing & Class Imbalance
- Segmented features into numeric (`person_age`, `person_income`, `loan_amnt`, `loan_int_rate`, etc.) and categorical (`person_home_ownership`, `loan_intent`, `loan_grade`, `cb_person_default_on_file`).
- Implemented `ColumnTransformer` with median imputation and one-hot encoding.
- Handled class skewness by computing the negative-to-positive class weight ratio:
  $$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}}$$

### 3. Model Benchmarking & Hyperparameter Tuning
- **Baseline**: Logistic Regression with standard scaling.
- **Champion**: XGBoost Classifier tuned with `RandomizedSearchCV` across estimators, max depth, learning rate, and subsampling ratios using Stratified 5-Fold Cross-Validation.

### 4. Threshold Optimization
Rather than using a generic 0.5 decision boundary:
- Precision-Recall curves were computed across all probability thresholds.
- Optimal threshold ($\approx 0.695$) was selected to maximize the F1-score and reduce costly defaults.

### 5. Model Explainability with SHAP
- **Global Explanations**: Summary bee-swarm plots highlight top default drivers: loan-to-income ratio, interest rate, and historical default indicators.
- **Local Explanations**: Waterfall plots deconstruct single applicant scores into additive contributions, providing actionable justification for loan decisions.

---

## API Reference

### Health / Schema Check
- **GET** `/openapi.json` or `/docs` (Interactive Swagger UI)

### Predict Credit Risk
- **POST** `/predict`

#### Request Body
```json
{
  "person_age": 30,
  "person_income": 600000.0,
  "person_home_ownership": "RENT",
  "person_emp_length": 5.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 100000.0,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.17,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 6
}
```

#### Response Body
```json
{
  "default_probability": 0.124581,
  "default_prediction": 0,
  "threshold": 0.695198,
  "Result": "Low Risk"
}
```

---

## Interactive Underwriting Desk

The built-in web client (`/`) offers an intuitive underwriting desk:
- **Instant Ratio Calculation**: Loan-to-income percentage updates dynamically as loan amount or annual income changes.
- **Service Status Indicator**: Live pinging of the prediction engine.
- **Risk Gauge**: Radial gauge dynamically sweeps to the exact default percentage with a marker indicating the decision threshold.
- **Stamp Verdict**: Animated visual stamp indicating `LOW RISK` (Approved) or `HIGH RISK` (Requires Review).

---

## Repository Structure

```
.
├── Credit_Risk.ipynb           # End-to-end ML notebook (EDA, training, SHAP, evaluation)
├── credit_risk_dataset.csv     # 32,581 records loan applicant dataset
├── credit_risk_model.pkl       # Trained Scikit-Learn / XGBoost pipeline artifact
├── best_threshold.pkl          # Pickled optimal decision boundary
├── main.py                     # FastAPI backend application & API endpoints
├── render.yaml                 # Infrastructure-as-code for Render deployment
├── requirements.txt            # Production dependencies
├── runtime.txt                 # Python runtime version
├── static/                     # Frontend underwriting interface
│   ├── index.html              # HTML5 structure
│   ├── style.css               # Typography & ledger-style styling
│   └── script.js               # Reactive DOM updates & fetch logic
└── README.md                   # System documentation
```

---

## Getting Started

### Prerequisites
- Python 3.10+ (Recommended: Python 3.11)
- `git`

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MallaManoj/Credit-Risk-ML-System.git
   cd Credit-Risk-ML-System
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Start the FastAPI application with Uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser at:
- **Web Interface**: `http://localhost:8000/`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`

---

## Deployment

The repository includes a ready-to-use [`render.yaml`](render.yaml) specification for Render:
1. Connect your GitHub repository to [Render](https://render.com).
2. Create a new **Web Service** from the repository.
3. Render automatically picks up `requirements.txt` and `runtime.txt`, starting the server with:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
