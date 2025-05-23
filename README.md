# 🛡️ Credit Card Fraud Detection API

A production-style machine learning pipeline that detects fraudulent credit card transactions using anomaly detection. This project demonstrates best practices in ML preprocessing, model training, evaluation, deployment via FastAPI, and real-time inference.

## 🔍 Overview

* ✅ Anomaly detection using Isolation Forest
* ✅ Real-world dataset (Credit Card Fraud from OpenML)
* ✅ Scalable and modular ML pipeline (data → model → API)
* ✅ REST API for fraud prediction using FastAPI
* ✅ Feature scaling and model persistence with `joblib`
* ✅ Project structured for extensibility and deployment

---

## 📁 Project Structure

```
fraud_detection_project/
├── app/                    # FastAPI API for inference
│   └── app.py
├── src/                    # Training and preprocessing scripts
│   ├── preprocess.py
│   ├── train.py
│   ├── utils.py
│   └── __init__.py
├── models/                 # Saved model and scaler
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository and install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train.py
```

This script:

* Loads the dataset
* Drops unnecessary columns like `Time`
* Trains an Isolation Forest on 29 features
* Saves the model and scaler to `models/`

### 3. Run the FastAPI server

```bash
uvicorn app.app:app --reload
```

### 4. Test the API

Go to:

```
http://127.0.0.1:8000/docs
```

Use the `/predict` endpoint with a JSON like:

```json
{
  "data": [
    -1.359807134, -0.072781173, 2.536346738, 1.378155224, -0.33832077,
    0.462387777, 0.239598554, 0.098697901, 0.36378697, 0.090794172,
    -0.551599533, -0.617800856, -0.991389847, -0.311169354, 1.468176972,
    -0.470400525, 0.207971242, 0.02579058, 0.40399296, 0.251412098,
    -0.018306778, 0.277837576, -0.11047391, 0.066928075, 0.128539358,
    -0.189114844, 0.133558377, -0.021053053, 0.01472417
  ]
}
```

---

## 📊 Model Details

* **Algorithm**: Isolation Forest (unsupervised anomaly detection)
* **Metric Highlights**: Precision, Recall, F1-Score, ROC AUC
* **Features Used**: V1–V28 + Amount (29 total)

---

## 🛠️ Tech Stack

* Python 3.10+
* scikit-learn
* pandas, numpy
* FastAPI
* Uvicorn
* Joblib

---

## 📌 Notes

* Make sure you have internet access when fetching the dataset (via `fetch_openml`).
* You can easily extend this project with:

  * Streamlit dashboards
  * SHAP explainability
  * Cloud deployment (e.g., Render, Azure, AWS Lambda)

---

## 👨‍💻 Author

**Mohammad Mousavi**
Data Scientist | ML/AI Engineer
[LinkedIn](https://www.linkedin.com/in/mohammad-mousavi-895763113/) | [GitHub](https://github.com/kmousavi91)


