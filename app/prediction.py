from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model(df):
    X = df.drop(columns=['customer_id', 'redeemed'])
    y = df['redeemed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/default_model.pkl")

    return model
