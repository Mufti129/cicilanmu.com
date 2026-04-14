import pandas as pd

def rfm_segmentation(df):
    rfm = df.groupby('customer_id').agg({
        'days_late': 'mean',
        'loan_count': 'sum',
        'loan_amount': 'mean'
    }).reset_index()

    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

    rfm['segment'] = pd.qcut(rfm['monetary'], 3, labels=['Low', 'Mid', 'High'])

    return rfm
