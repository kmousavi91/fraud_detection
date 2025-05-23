import joblib

def save_model(model, scaler, model_path="models/model.joblib", scaler_path="models/scaler.joblib"):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

def load_model(model_path="models/model.joblib", scaler_path="models/scaler.joblib"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler
