import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

def load_data():
    df = fetch_openml(name="creditcard", version=1, as_frame=True).frame
    df.dropna(inplace=True)
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])
    return df

def preprocess_data(df):
    X = df.drop(columns=['Class'])
    y = df['Class'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features used for training:", X_train.shape[1])
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
