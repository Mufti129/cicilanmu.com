# Pawnshop Data Analytics Project

## Overview
Project ini bertujuan untuk menganalisis bisnis gadai menggunakan data analytics dan machine learning.

## Features
- Prediksi gagal bayar (Random Forest)
- Clustering pelanggan (K-Means)
- Segmentasi pelanggan (RFM)
- Dashboard interaktif (Streamlit)

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train model
python train_model.py

### 3. Run dashboard
streamlit run app/streamlit_app.py

## Methodology

Project ini menggunakan beberapa metode analisis:

### 1. Classification (Random Forest)
Digunakan untuk memprediksi risiko gagal bayar nasabah berdasarkan fitur seperti:
- Loan to Value (LTV)
- Income
- Days Late

### 2. Clustering (K-Means)
Digunakan untuk mengelompokkan pelanggan berdasarkan perilaku pinjaman:
- Loan amount
- Income
- Loan frequency

### 3. RFM Analysis
Segmentasi pelanggan berdasarkan:
- Recency
- Frequency
- Monetary

### 4. Exploratory Data Analysis (EDA)
Digunakan untuk menemukan pola dan insight dari data.
