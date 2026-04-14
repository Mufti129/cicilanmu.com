import streamlit as st
import pandas as pd
import joblib

from src.preprocessing import load_data, preprocess
from src.clustering import run_clustering
from src.segmentation import rfm_segmentation

st.set_page_config(layout="wide")
st.title("📊 Pawnshop Analytics Dashboard")

df = load_data('data/pawn_data.csv')
df_clean = preprocess(df)

menu = st.sidebar.selectbox("Menu", [
    "Overview",
    "Prediksi Gagal Bayar",
    "Clustering",
    "Segmentasi",
    "Perilaku Konsumen"
])

# OVERVIEW
if menu == "Overview":
    st.metric("Total Nasabah", len(df))
    st.metric("Total Loan", int(df['loan_amount'].sum()))
    st.metric("Default Rate (%)", round((1 - df['redeemed'].mean()) * 100, 2))

# PREDICTION
elif menu == "Prediksi Gagal Bayar":
    st.subheader("Prediksi Default")

    model = joblib.load('models/default_model.pkl')

    input_data = df_clean.drop(columns=['customer_id', 'redeemed']).iloc[0:1]
    pred = model.predict(input_data)[0]

    if pred == 0:
        st.error("⚠️ Risiko Tinggi")
    else:
        st.success("✅ Aman")

# CLUSTERING
elif menu == "Clustering":
    df_cluster = run_clustering(df_clean)
    st.bar_chart(df_cluster['cluster'].value_counts())

# SEGMENTASI
elif menu == "Segmentasi":
    rfm = rfm_segmentation(df)
    st.bar_chart(rfm['segment'].value_counts())

# BEHAVIOR
elif menu == "Perilaku Konsumen":
    st.bar_chart(df.groupby('job_type')['redeemed'].mean())
