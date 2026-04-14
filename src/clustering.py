from sklearn.cluster import KMeans

def run_clustering(df):
    features = df[['loan_amount', 'monthly_income', 'loan_count']]

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(features)

    return df
