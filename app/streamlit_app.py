import streamlit as st
import pandas as pd
import joblib
import os

from preprocessing import load_data, preprocess
from clustering import run_clustering
from segmentation import rfm_segmentation
from prediction import train_model

st.set_page_config(layout="wide")
st.title("📊 Pawnshop Analytics Dashboard")

# ======================
# LOAD DATA
# ======================
df = load_data('data/pawn_data.csv')
df_clean = preprocess(df)

# ======================
# MODEL
# ======================
model_path = '../models/default_model.pkl'

if not os.path.exists(model_path):
    st.warning("Model belum ada, sedang training...")
    train_model(df_clean)
    st.success("Model berhasil dibuat!")

model = joblib.load(model_path)

# ======================
# MENU
# ======================
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
    st.header("📌 Overview")

    total_customer = len(df)
    total_loan = df['loan_amount'].sum()
    default_rate = (1 - df['redeemed'].mean()) * 100

    st.metric("Total Nasabah", total_customer)
    st.metric("Total Loan", int(total_loan))
    st.metric("Default Rate (%)", round(default_rate, 2))

    with st.expander("📘 Metodologi"):
        st.markdown("""
        - Machine Learning (Random Forest)
        - Clustering (K-Means)
        - RFM Analysis
        - Exploratory Data Analysis
        """)

    # 🔥 AUTO INSIGHT
    st.subheader("💡 Insight Otomatis")

    if default_rate > 20:
        st.warning("Default rate cukup tinggi → perlu pengetatan analisis kredit")
    else:
        st.success("Default rate masih dalam batas aman")

    if total_loan / total_customer > 3000000:
        st.info("Rata-rata pinjaman tinggi → potensi profit besar tapi risiko meningkat")

# ======================
# PREDIKSI
# ======================
elif menu == "Prediksi Gagal Bayar":
    st.header("🤖 Prediksi Gagal Bayar")

    with st.expander("📘 Metodologi Random Forest"):
       # st.markdown("""
        #Digunakan untuk klasifikasi risiko gagal bayar.
        #""")
        st.markdown("""
Metode yang digunakan dalam analisis ini adalah Random Forest Classification, yaitu algoritma machine learning berbasis ensemble yang menggabungkan banyak decision tree untuk menghasilkan prediksi yang lebih akurat dan stabil. Model ini bekerja dengan cara membangun sejumlah pohon keputusan dari subset data yang berbeda, kemudian menggabungkan hasil prediksi melalui mekanisme voting.
Dalam konteks bisnis gadai, metode ini digunakan untuk memprediksi kemungkinan nasabah mengalami gagal bayar (default) berdasarkan berbagai faktor seperti nilai pinjaman, nilai jaminan, tingkat keterlambatan pembayaran, serta kondisi ekonomi nasabah seperti pendapatan. Pemilihan Random Forest didasarkan pada kemampuannya dalam menangkap pola hubungan yang kompleks dan non-linear antar variabel, serta ketahanannya terhadap noise dan outlier pada data 
Dengan adanya model ini, perusahaan dapat melakukan mitigasi risiko secara proaktif, misalnya dengan memperketat persetujuan pinjaman untuk nasabah berisiko tinggi atau menyesuaikan strategi penagihan. Hasil dari model ini diharapkan dapat membantu meningkatkan kualitas portofolio pinjaman dan mengurangi potensi kerugian akibat gagal bayar.""")

    input_data = df_clean.drop(columns=['customer_id', 'redeemed']).iloc[0:1]
    pred = model.predict(input_data)[0]

    if pred == 0:
        st.error("⚠️ Risiko Tinggi")
    else:
        st.success("✅ Aman")

    # 🔥 AUTO INSIGHT
    st.subheader("💡 Insight Otomatis")

    high_ltv = df[df['loan_amount']/df['collateral_value'] > 0.8]

    if len(high_ltv) > 0:
        st.warning(f"{len(high_ltv)} nasabah memiliki LTV tinggi → berisiko gagal bayar")

    late_users = df[df['days_late'] > 7]

    st.info(f"{len(late_users)} nasabah sering terlambat → kandidat default")

# ======================
# CLUSTERING
# ======================
elif menu == "Clustering Pelanggan":
    st.header("👥 Clustering")

    with st.expander("📘 Metodologi K-Means"):
        #st.markdown("Segmentasi berdasarkan perilaku pinjaman")
        st.markdown("""Analisis clustering dalam dashboard ini menggunakan metode K-Means Clustering, yaitu teknik unsupervised learning yang bertujuan untuk mengelompokkan pelanggan ke dalam beberapa segmen berdasarkan kemiripan karakteristik mereka. Algoritma ini bekerja dengan membagi data ke dalam sejumlah cluster yang telah ditentukan sebelumnya, di mana setiap data akan masuk ke cluster dengan jarak terdekat terhadap centroid.
Dalam implementasinya pada bisnis gadai, variabel yang digunakan meliputi jumlah pinjaman, pendapatan bulanan, dan frekuensi transaksi. Dengan pendekatan ini, perusahaan dapat mengidentifikasi kelompok pelanggan seperti nasabah bernilai tinggi, nasabah dengan aktivitas rendah, maupun nasabah dengan potensi risiko tinggi.
Hasil clustering ini sangat berguna dalam mendukung pengambilan keputusan strategis, seperti penentuan target pemasaran, pemberian limit pinjaman, hingga penyesuaian layanan berdasarkan profil masing-masing segmen pelanggan. Dengan memahami karakteristik tiap cluster, perusahaan dapat meningkatkan efisiensi operasional dan optimalisasi revenue.
""")
    df_cluster = run_clustering(df_clean)

    cluster_counts = df_cluster['cluster'].value_counts()
    st.bar_chart(cluster_counts)

    # 🔥 AUTO INSIGHT
    st.subheader("💡 Insight Otomatis")

    biggest_cluster = cluster_counts.idxmax()

    st.info(f"Cluster terbesar adalah Cluster {biggest_cluster}")

    avg_income = df.groupby('customer_id')['monthly_income'].mean().mean()

    if avg_income < 5000000:
        st.warning("Mayoritas nasabah memiliki income rendah → risiko meningkat")

# ======================
# SEGMENTASI
# ======================
elif menu == "Segmentasi Pelanggan":
    st.header("📈 Segmentasi")

    with st.expander("📘 Metodologi RFM"):
        st.markdown("Segmentasi berdasarkan nilai customer")

    rfm = rfm_segmentation(df)
    segment_counts = rfm['segment'].value_counts()

    st.bar_chart(segment_counts)

    # 🔥 AUTO INSIGHT
    st.subheader("💡 Insight Otomatis")

    if 'High' in segment_counts:
        st.success(f"{segment_counts['High']} pelanggan high value → fokus retensi")

    if 'Low' in segment_counts:
        st.warning(f"{segment_counts['Low']} pelanggan low value → perlu engagement")

# ======================
# PERILAKU
# ======================
elif menu == "Perilaku Konsumen":
    st.header("📊 Perilaku Konsumen")

    with st.expander("📘 Metodologi EDA"):
        st.markdown("Analisis pola perilaku nasabah")

    job_default = df.groupby('job_type')['redeemed'].mean()
    st.bar_chart(job_default)

    # 🔥 AUTO INSIGHT
    st.subheader("💡 Insight Otomatis")

    risky_job = job_default.idxmin()

    st.warning(f"Pekerjaan dengan risiko tertinggi: {risky_job}")

    avg_loan = df.groupby('job_type')['loan_amount'].mean().idxmax()

    st.info(f"Pekerjaan dengan pinjaman terbesar: {avg_loan}")
