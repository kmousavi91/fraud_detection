# Fraud Detection API

This project includes a production-style machine learning pipeline for detecting fraudulent credit card transactions. It includes:

- Data preprocessing
- Model training (Isolation Forest)
- Model saving/loading
- A REST API for real-time fraud prediction using FastAPI

## To Run

1. Train the model:

```bash
python src/train.py
```

2. Start the API:

```bash
uvicorn app.app:app --reload
```

3. Test it via Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
