import streamlit as st
import pandas as pd
import joblib

from preprocessing import load_data, preprocess
from clustering import run_clustering
from segmentation import rfm_segmentation

st.set_page_config(layout="wide")
st.title("Cicilanmu.com Analytics Dashboard")

df = load_data('data/pawn_data.csv')
df_clean = preprocess(df)

menu = st.sidebar.selectbox("Menu", [
    "Overview",
    "Prediksi Gagal Bayar",
    "Clustering Pelanggan",
    "Segmentasi Pelanggan",
    "Perilaku Konsumen"
])

# ======================
# OVERVIEW
# ======================
if menu == "Overview":
    st.header("Overview")

    st.markdown("""
    Dashboard ini dibuat untuk menganalisis bisnis gadai menggunakan pendekatan data analytics.

    **Metodologi yang digunakan:**
    - Machine Learning (Prediksi Risiko)
    - Clustering (Segmentasi Perilaku)
    - RFM Analysis (Segmentasi Nilai Customer)
    - Exploratory Data Analysis (EDA)
    """)

    st.metric("Total Nasabah", len(df))
    st.metric("Total Loan", int(df['loan_amount'].sum()))
    st.metric("Default Rate (%)", round((1 - df['redeemed'].mean()) * 100, 2))

# ======================
# PREDIKSI
# ======================
elif menu == "Prediksi Gagal Bayar":
    st.header("Prediksi Gagal Bayar")

    st.markdown("""
    ### Metodologi: Random Forest Classification

    Model ini digunakan untuk memprediksi apakah nasabah akan gagal bayar.

    **Cara kerja:**
    - Menggunakan banyak decision tree
    - Menggabungkan hasil prediksi (ensemble)

    **Kenapa dipilih:**
    - Mampu menangkap pola kompleks
    - Cocok untuk data keuangan
    - Stabil terhadap outlier

    **Fitur utama:**
    - Loan to Value (LTV)
    - Income
    - Days Late
    """)

    model = joblib.load('models/default_model.pkl')

    input_data = df_clean.drop(columns=['customer_id', 'redeemed']).iloc[0:1]
    pred = model.predict(input_data)[0]

    if pred == 0:
        st.error("⚠️ Risiko Tinggi Gagal Bayar")
    else:
        st.success("✅ Nasabah Aman")

# ======================
# CLUSTERING
# ======================
elif menu == "Clustering Pelanggan":
    st.header("Clustering Pelanggan")

    st.markdown("""
    ### Metodologi: K-Means Clustering

    Digunakan untuk mengelompokkan pelanggan berdasarkan kemiripan.

    **Variabel:**
    - Loan amount
    - Monthly income
    - Loan frequency

    **Tujuan:**
    - Mengetahui kelompok pelanggan
    - Membantu strategi bisnis (targeting)
    """)

    df_cluster = run_clustering(df_clean)

    st.bar_chart(df_cluster['cluster'].value_counts())

    st.markdown("""
    **Interpretasi:**
    - Cluster 0 → High Value
    - Cluster 1 → Medium
    - Cluster 2 → Risky
    """)

# ======================
# SEGMENTASI
# ======================
elif menu == "Segmentasi Pelanggan":
    st.header("Segmentasi Pelanggan")

    st.markdown("""
    ### Metodologi: RFM Analysis

    Segmentasi berdasarkan:
    - Recency (terakhir transaksi)
    - Frequency (jumlah transaksi)
    - Monetary (nilai transaksi)

    **Tujuan:**
    - Identifikasi pelanggan bernilai tinggi
    - Optimasi strategi marketing
    """)

    rfm = rfm_segmentation(df)

    st.bar_chart(rfm['segment'].value_counts())

# ======================
# PERILAKU
# ======================
elif menu == "Perilaku Konsumen":
    st.header("Perilaku Konsumen")

    st.markdown("""
    ### Metodologi: Exploratory Data Analysis (EDA)

    Digunakan untuk:
    - Mencari pola data
    - Mengidentifikasi hubungan antar variabel
    - Menemukan insight bisnis

    **Contoh analisis:**
    - Default rate berdasarkan pekerjaan
    - Distribusi pinjaman
    """)

    st.bar_chart(df.groupby('job_type')['redeemed'].mean())
