import streamlit as st
import pandas as pd
import joblib
import os

from preprocessing import load_data, preprocess
from clustering import run_clustering
from segmentation import rfm_segmentation
from prediction import train_model

st.set_page_config(layout="wide")
st.title("cicilanmu.com Analytics Dashboard")

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
    st.header("Overview")

    total_customer = len(df)
    total_loan = df['loan_amount'].sum()
    default_rate = (1 - df['redeemed'].mean()) * 100

    st.metric("Total Nasabah", total_customer)
    st.metric("Total Loan", int(total_loan))
    st.metric("Default Rate (%)", round(default_rate, 2))

    with st.expander("Metodologi"):
        st.markdown("""
        - Machine Learning (Random Forest)
        - Clustering (K-Means)
        - RFM Analysis
        - Exploratory Data Analysis
        """)

    # 🔥 AUTO INSIGHT
    st.subheader("Insight")

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
    st.header("Prediksi Gagal Bayar")

    with st.expander("Metodologi Random Forest"):
       # st.markdown("""
        #Digunakan untuk klasifikasi risiko gagal bayar.
        #""")
        st.markdown("""
Metode yang digunakan dalam analisis ini adalah Random Forest Classification, yaitu algoritma machine learning berbasis ensemble yang menggabungkan banyak decision tree untuk menghasilkan prediksi yang lebih akurat dan stabil. Model ini bekerja dengan cara membangun sejumlah pohon keputusan dari subset data yang berbeda, kemudian menggabungkan hasil prediksi melalui mekanisme voting.
Dalam konteks bisnis gadai, metode ini digunakan untuk memprediksi kemungkinan nasabah mengalami gagal bayar (default) berdasarkan berbagai faktor seperti nilai pinjaman, nilai jaminan, tingkat keterlambatan pembayaran, serta kondisi ekonomi nasabah seperti pendapatan. Pemilihan Random Forest didasarkan pada kemampuannya dalam menangkap pola hubungan yang kompleks dan non-linear antar variabel, serta ketahanannya terhadap noise dan outlier pada data 
Dengan adanya model ini, perusahaan dapat melakukan mitigasi risiko secara proaktif, misalnya dengan memperketat persetujuan pinjaman untuk nasabah berisiko tinggi atau menyesuaikan strategi penagihan. Hasil dari model ini diharapkan dapat membantu meningkatkan kualitas portofolio pinjaman dan mengurangi potensi kerugian akibat gagal bayar.""")

    #####
    # ======================
    # PILIH CUSTOMER
    # ======================
    st.subheader("Pilih Data Nasabah")

    selected_id = st.selectbox(
        "Pilih Customer ID",
        df['customer_id']
    )

    selected_data = df[df['customer_id'] == selected_id]

    st.write("Data Nasabah:")
    st.dataframe(selected_data, use_container_width=True)

    # ======================
    # PREPROCESS DATA
    # ======================
    selected_clean = preprocess(selected_data)

    # Samakan kolom dengan training
    model_features = df_clean.drop(columns=['customer_id', 'redeemed']).columns
    selected_clean = selected_clean.reindex(columns=model_features, fill_value=0)

    # ======================
    # PREDIKSI
    # ======================
    pred = model.predict(selected_clean)[0]
    prob = model.predict_proba(selected_clean)[0][pred]

    st.subheader("Hasil Prediksi")

    if pred == 0:
        st.error(f"Risiko Tinggi Gagal Bayar (Confidence: {round(prob*100,2)}%)")
    else:
        st.success(f"Nasabah Aman (Confidence: {round(prob*100,2)}%)")

    # ======================
    # 🔥 INSIGHT OTOMATIS
    # ======================
    st.subheader("Insight")

    loan = selected_data['loan_amount'].values[0]
    collateral = selected_data['collateral_value'].values[0]
    income = selected_data['monthly_income'].values[0]
    late = selected_data['days_late'].values[0]

    ltv = loan / collateral

    if ltv > 0.8:
        st.warning("LTV tinggi → risiko meningkat")

    if income < 4000000:
        st.warning("Pendapatan rendah → kemampuan bayar terbatas")

    if late > 7:
        st.error("Riwayat keterlambatan tinggi → indikator utama gagal bayar")

    if pred == 1:
        st.success("Nasabah layak dipertimbangkan untuk pinjaman lanjutan")
    else:
        st.warning("Perlu analisis tambahan sebelum persetujuan pinjaman")
    #####
    input_data = df_clean.drop(columns=['customer_id', 'redeemed']).iloc[0:1]
    pred = model.predict(input_data)[0]

    if pred == 0:
        st.error("Risiko Tinggi")
    else:
        st.success("Aman")

    # 🔥 AUTO INSIGHT
  #  st.subheader("💡 Insight Otomatis")

   # high_ltv = df[df['loan_amount']/df['collateral_value'] > 0.8]

    #if len(high_ltv) > 0:
     #st.warning(f"{len(high_ltv)} nasabah memiliki LTV tinggi → berisiko gagal bayar")

    #late_users = df[df['days_late'] > 7]

   # st.info(f"{len(late_users)} nasabah sering terlambat → kandidat default")
#####+list data
    # ======================
# 🔥 LIST NASABAH BERISIKO
# ======================
    st.subheader("Daftar Nasabah Berpotensi Gagal Bayar")
    
    # Ambil semua data untuk prediksi
    all_features = df_clean.drop(columns=['customer_id', 'redeemed'])
    
    # Prediksi semua nasabah
    all_preds = model.predict(all_features)
    all_probs = model.predict_proba(all_features)
    
    # Buat dataframe hasil
    result_df = df.copy()
    result_df['prediction'] = all_preds
    result_df['risk_probability'] = all_probs.max(axis=1)
    
    # Filter yang berisiko (0 = gagal bayar)
    risk_df = result_df[result_df['prediction'] == 0]
    
    if len(risk_df) > 0:
        st.warning(f"Terdapat {len(risk_df)} nasabah berisiko tinggi")
    
        # Sort dari paling berisiko
        risk_df = risk_df.sort_values(by='risk_probability', ascending=False)
    
        st.dataframe(
            risk_df[['customer_id', 'job_type', 'monthly_income', 'loan_amount', 'days_late', 'risk_probability']],
            use_container_width=True
        )
    
    else:
        st.success("Tidak ada nasabah berisiko tinggi")
    #####
# ======================
# CLUSTERING
# ======================
#elif menu == "Clustering Pelanggan":
 #   st.header("Clustering")

  #  with st.expander("Metodologi K-Means"):
   #     #st.markdown("Segmentasi berdasarkan perilaku pinjaman")
    #    st.markdown("""Analisis clustering dalam dashboard ini menggunakan metode K-Means Clustering, yaitu teknik unsupervised learning yang bertujuan untuk mengelompokkan pelanggan ke dalam beberapa segmen berdasarkan kemiripan karakteristik mereka. Algoritma ini bekerja dengan membagi data ke dalam sejumlah cluster yang telah ditentukan sebelumnya, di mana setiap data akan masuk ke cluster dengan jarak terdekat terhadap centroid.
#Dalam implementasinya pada bisnis gadai, variabel yang digunakan meliputi jumlah pinjaman, pendapatan bulanan, dan frekuensi transaksi. Dengan pendekatan ini, perusahaan dapat mengidentifikasi kelompok pelanggan seperti nasabah bernilai tinggi, nasabah dengan aktivitas rendah, maupun nasabah dengan potensi risiko tinggi.
#Hasil clustering ini sangat berguna dalam mendukung pengambilan keputusan strategis, seperti penentuan target pemasaran, pemberian limit pinjaman, hingga penyesuaian layanan berdasarkan profil masing-masing segmen pelanggan. Dengan memahami karakteristik tiap cluster, perusahaan dapat meningkatkan efisiensi operasional dan optimalisasi revenue.
#""")
 #   df_cluster = run_clustering(df_clean)

  #  cluster_counts = df_cluster['cluster'].value_counts()
   # st.bar_chart(cluster_counts)

    # 🔥 AUTO INSIGHT
   # st.subheader("Insight")

    #biggest_cluster = cluster_counts.idxmax()

    #st.info(f"Cluster terbesar adalah Cluster {biggest_cluster}")

    #avg_income = df.groupby('customer_id')['monthly_income'].mean().mean()

    #if avg_income < 5000000:
     #   st.warning("Mayoritas nasabah memiliki income rendah → risiko meningkat")
elif menu == "Clustering Pelanggan":
    st.header("Clustering Pelanggan")

    # ======================
    # METODOLOGI (HIDE)
    # ======================
    with st.expander("Metodologi K-Means"):
        st.markdown("""
        K-Means Clustering digunakan untuk mengelompokkan nasabah berdasarkan kemiripan karakteristik seperti jumlah pinjaman, pendapatan, dan frekuensi transaksi. Hasilnya adalah beberapa kelompok (cluster) dengan profil yang berbeda.
        """)

    # ======================
    # JALANKAN CLUSTERING
    # ======================
    df_cluster = run_clustering(df_clean)

    cluster_counts = df_cluster['cluster'].value_counts()
    st.subheader("Distribusi Cluster")
    st.bar_chart(cluster_counts)

    # ======================
    # 🔥 PENJELASAN HASIL CLUSTER
    # ======================
    st.subheader("📖 Penjelasan Hasil Cluster")

    st.markdown("""
    Grafik di atas menunjukkan jumlah nasabah dalam setiap cluster yang terbentuk dari proses K-Means.

    Setiap cluster merepresentasikan kelompok nasabah dengan karakteristik yang mirip. Perbedaan antar cluster biasanya dipengaruhi oleh:
    - Besaran pinjaman
    - Tingkat pendapatan
    - Frekuensi transaksi

    Dengan memahami cluster ini, perusahaan dapat membedakan antara nasabah bernilai tinggi, nasabah biasa, dan nasabah yang berpotensi berisiko.
    """)

    # ======================
    # 🔥 AUTO INTERPRETASI
    # ======================
    st.subheader("💡 Insight Otomatis")

    biggest_cluster = cluster_counts.idxmax()
    smallest_cluster = cluster_counts.idxmin()

    st.info(f"Cluster terbesar adalah Cluster {biggest_cluster} → mayoritas nasabah berada di kelompok ini")
    st.warning(f"Cluster terkecil adalah Cluster {smallest_cluster} → segmen ini lebih spesifik")

    # ======================
    # 🔥 PROFIL TIAP CLUSTER
    # ======================
    st.subheader("📊 Profil Rata-rata per Cluster")

    profile = df_cluster.groupby('cluster')[['loan_amount', 'monthly_income', 'loan_count']].mean()
    st.dataframe(profile, use_container_width=True)

    st.markdown("""
    Interpretasi:
    - Nilai loan_amount tinggi → nasabah dengan pinjaman besar
    - Income tinggi → kemampuan bayar lebih baik
    - Loan_count tinggi → nasabah aktif
    """)

    # ======================
    # 🔥 LIST DATA PER CLUSTER
    # ======================
    st.subheader("📋 Detail Nasabah per Cluster")

    selected_cluster = st.selectbox(
        "Pilih Cluster",
        sorted(df_cluster['cluster'].unique())
    )

    cluster_data = df_cluster[df_cluster['cluster'] == selected_cluster]

    # Gabungkan dengan data asli
    merged = df.merge(cluster_data[['customer_id', 'cluster']], on='customer_id')

    st.write(f"Menampilkan data untuk Cluster {selected_cluster}")

    st.dataframe(
        merged[['customer_id', 'job_type', 'monthly_income', 'loan_amount', 'loan_count', 'cluster']]
        .sort_values(by='loan_amount', ascending=False),
        use_container_width=True
    )

    # ======================
    # 🔥 INSIGHT PER CLUSTER
    # ======================
    st.subheader("🧠 Insight Cluster")

    avg_loan = cluster_data['loan_amount'].mean()
    avg_income = cluster_data['monthly_income'].mean()

    if avg_loan > df['loan_amount'].mean():
        st.warning("Cluster ini memiliki rata-rata pinjaman tinggi → potensi risiko & profit besar")

    if avg_income > df['monthly_income'].mean():
        st.success("Cluster ini memiliki income tinggi → relatif lebih aman")

    if avg_income < df['monthly_income'].mean():
        st.warning("Cluster ini memiliki income rendah → perlu perhatian")
# ======================
# SEGMENTASI
# ======================
elif menu == "Segmentasi Pelanggan":
    st.header("Segmentasi")

    with st.expander("Metodologi RFM"):
        #st.markdown("Segmentasi berdasarkan nilai customer")
        st.markdown("""
Segmentasi pelanggan dilakukan menggunakan metode RFM Analysis, yaitu pendekatan yang mengelompokkan pelanggan berdasarkan tiga dimensi utama: Recency (seberapa baru pelanggan melakukan transaksi), Frequency (seberapa sering pelanggan bertransaksi), dan Monetary (berapa besar nilai transaksi yang dihasilkan).

Metode ini umum digunakan dalam analisis perilaku pelanggan karena mampu memberikan gambaran yang jelas mengenai nilai dan kontribusi masing-masing pelanggan terhadap bisnis. Dalam konteks gadai, RFM membantu mengidentifikasi pelanggan yang aktif dan bernilai tinggi, serta membedakannya dengan pelanggan yang kurang aktif atau berisiko churn.

Dengan segmentasi ini, perusahaan dapat merancang strategi yang lebih tepat sasaran, seperti memberikan penawaran khusus kepada pelanggan high value, meningkatkan engagement pada pelanggan mid value, serta melakukan pendekatan retensi terhadap pelanggan low value. Implementasi RFM diharapkan dapat meningkatkan loyalitas pelanggan sekaligus memaksimalkan profitabilitas.
""")
    rfm = rfm_segmentation(df)
    segment_counts = rfm['segment'].value_counts()

    st.bar_chart(segment_counts)

    # 🔥 AUTO INSIGHT
    st.subheader("Insight")

    if 'High' in segment_counts:
        st.success(f"{segment_counts['High']} pelanggan high value → fokus retensi")

    if 'Low' in segment_counts:
        st.warning(f"{segment_counts['Low']} pelanggan low value → perlu engagement")
    # ======================
    # 🔥 LIST DATA PER SEGMENT
    # ======================
    st.subheader("Detail Pelanggan per Segmen")

    selected_segment = st.selectbox(
        "Pilih Segmen",
        ["High", "Mid", "Low"]
    )

    # Filter data
    segment_data = rfm[rfm['segment'] == selected_segment]

    # Gabungkan dengan data asli biar lebih lengkap
    merged = df.merge(segment_data, on="customer_id")

    st.write(f"Menampilkan data untuk segmen: **{selected_segment}**")

    st.dataframe(
        merged[['customer_id', 'job_type', 'monthly_income', 'loan_amount', 'frequency', 'monetary']]
        .sort_values(by='monetary', ascending=False),
        use_container_width=True
    )

    # ======================
    # 🔥 INSIGHT PER SEGMENT
    # ======================
    st.subheader("Insight Segmen")

    if selected_segment == "High":
        st.success("Pelanggan ini memiliki kontribusi tinggi → cocok untuk program loyalitas & peningkatan limit pinjaman")

    elif selected_segment == "Mid":
        st.info("Pelanggan ini memiliki potensi berkembang → bisa ditingkatkan dengan promo atau penawaran khusus")

    elif selected_segment == "Low":
        st.warning("Pelanggan ini memiliki kontribusi rendah → perlu strategi aktivasi atau edukasi")
# ======================
# PERILAKU
# ======================
elif menu == "Perilaku Konsumen":
    st.header("Perilaku Konsumen")

    with st.expander("Metodologi EDA"):
        #st.markdown("Analisis pola perilaku nasabah")
        st.markdown("""
Analisis perilaku konsumen dalam dashboard ini dilakukan باستخدام pendekatan Exploratory Data Analysis (EDA), yaitu proses eksplorasi data untuk memahami pola, tren, dan hubungan antar variabel yang terdapat dalam dataset. EDA merupakan langkah awal yang sangat penting dalam data analytics karena membantu mengidentifikasi insight yang relevan sebelum dilakukan analisis lanjutan.

Dalam studi ini, EDA digunakan untuk menganalisis hubungan antara karakteristik nasabah, seperti jenis pekerjaan, jumlah pinjaman, dan tingkat keterlambatan pembayaran terhadap risiko gagal bayar. Melalui visualisasi data seperti distribusi dan perbandingan antar kelompok, perusahaan dapat memperoleh pemahaman yang lebih dalam mengenai faktor-faktor yang mempengaruhi perilaku nasabah.

Hasil dari analisis ini dapat digunakan sebagai dasar dalam pengambilan keputusan bisnis, seperti penyesuaian kebijakan kredit, pengelolaan risiko, serta pengembangan strategi pemasaran yang lebih efektif dan berbasis data.
""")
    ###
    # Hitung default rate per job
    job_default = df.groupby('job_type')['redeemed'].mean()
    
    st.bar_chart(job_default)
    
    # ======================
    # 🔥 PENJELASAN SEDERHANA
    # ======================
    st.subheader("Penjelasan Grafik")
    
    st.markdown("""
    Grafik di atas menunjukkan tingkat keberhasilan pelunasan pinjaman (redeemed) berdasarkan jenis pekerjaan nasabah.
    
    Semakin tinggi nilai pada grafik, berarti semakin besar kemungkinan nasabah dari kelompok pekerjaan tersebut berhasil melunasi pinjaman.
    Sebaliknya, jika nilainya rendah, maka kelompok tersebut memiliki risiko gagal bayar yang lebih tinggi.
    """)
    
    # ======================
    # 🔥 AUTO INTERPRETASI
    # ======================
    st.subheader("Insight")
    
    # Cari yang terbaik & terburuk
    best_job = job_default.idxmax()
    worst_job = job_default.idxmin()
    
    st.success(f"Nasabah dengan pekerjaan **{best_job}** memiliki tingkat pelunasan terbaik (risiko paling rendah).")
    
    st.warning(f"Nasabah dengan pekerjaan **{worst_job}** memiliki risiko gagal bayar lebih tinggi dibandingkan lainnya.")
    
    # Tambahan interpretasi sederhana
    for job, value in job_default.items():
        if value >= 0.8:
            st.info(f"{job} → Sangat stabil dalam pembayaran")
        elif value >= 0.6:
            st.info(f"{job} → Cukup stabil")
        else:
            st.info(f"{job} → Perlu perhatian (risiko lebih tinggi)")
