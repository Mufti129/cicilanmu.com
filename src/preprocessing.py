import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.copy()

    # Feature Engineering
    df['ltv'] = df['loan_amount'] / df['collateral_value']
    df['risk_flag'] = df['days_late'].apply(lambda x: 1 if x > 7 else 0)

    # Encoding
    df = pd.get_dummies(df, columns=['branch', 'job_type'], drop_first=True)

    return df
