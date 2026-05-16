import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Konfigurasi Tema & Layout
st.set_page_config(page_title="CKD Clinical Decision Support", layout="wide", page_icon="🧪")

# --- LOAD MODEL DENGAN ERROR HANDLING --- streamlit run app.py
@st.cache_resource
def load_artifacts():
    # Pastikan path file benar
    model = joblib.load('ckd_pipeline_model.pkl')
    le = joblib.load('label_encoder.pkl')
    # Jika feature_names.pkl tidak ada, kita gunakan list manual sesuai kodingan training
    try:
        features = joblib.load('feature_names.pkl')
    except:
        # Fallback: List fitur berdasarkan dataset asli Anda
        features = ['age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
                    'serum_creatinine', 'blood_urea_nitrogen', 'egfr', 'urine_albumin', 
                    'urine_protein', 'albumin_creatinine_ratio', 'urine_specific_gravity', 
                    'hemoglobin', 'hba1c', 'sodium', 'potassium', 'cholesterol', 'wbc_count']
    return model, le, features

try:
    model, le, features = load_artifacts()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    .result-card { padding: 20px; border-radius: 10px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 Sistem Deteksi Stadium Gagal Ginjal (CKD)")
st.info("Masukkan data klinis pasien di bawah ini untuk mendapatkan estimasi stadium penyakit.")

tab1, tab2, tab3, tab4 = st.tabs(["👤 Profil & Vital", "🩸 Panel Darah", "🧪 Analisis Urin", "📋 Riwayat & Gaya Hidup"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Usia (Tahun)", 1, 100, 45)
        gender = st.selectbox("Jenis Kelamin", options=[0, 1], format_func=lambda x: "Pria" if x==1 else "Wanita")
    with col2:
        bmi = st.number_input("BMI (Indeks Massa Tubuh)", 10, 50, 24)
        heart_rate = st.number_input("Detak Jantung (BPM)", 40, 150, 75)
    with col3:
        sys_bp = st.number_input("Tekanan Darah Sistolik", 80, 200, 120)
        dia_bp = st.number_input("Tekanan Darah Diastolik", 50, 130, 80)

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        egfr = st.number_input("eGFR", 0, 200, 90)
        sc = st.number_input("Serum Creatinine", 0.0, 20.0, 1.0)
        bun = st.number_input("Blood Urea Nitrogen", 1, 150, 20)
    with col2:
        hb = st.number_input("Hemoglobin", 5.0, 20.0, 14.0)
        sodium = st.number_input("Sodium", 100, 160, 140)
        potassium = st.number_input("Potassium", 2.0, 10.0, 4.0)
    with col3:
        hba1c = st.number_input("HbA1c (%)", 4.0, 15.0, 5.5)
        chol = st.number_input("Cholesterol", 100, 400, 190)
        wbc = st.number_input("WBC Count", 2000, 20000, 7000)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        u_alb = st.number_input("Urine Albumin", 0, 500, 10)
        u_prot = st.number_input("Urine Protein", 0, 500, 10)
    with col2:
        acr = st.number_input("Albumin Creatinine Ratio", 0, 1000, 20)
        usg = st.number_input("Urine Specific Gravity", 1.000, 1.040, 1.015, format="%.3f")

with tab4:
    # Menggunakan dummy input untuk fitur tambahan agar sesuai dengan jumlah fitur saat training
    diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
    ht = st.selectbox("Hipertensi", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")

# Tombol Prediksi
if st.button("PROSES DIAGNOSA"):
    # Buat dictionary dengan semua kolom yang diharapkan model
    input_data = {f: 0 for f in features}
    
    # Mapping input UI ke key fitur
    mapping = {
        'age': age, 'gender': gender, 'bmi': bmi, 'systolic_bp': sys_bp, 'diastolic_bp': dia_bp,
        'heart_rate': heart_rate, 'serum_creatinine': sc, 'blood_urea_nitrogen': bun, 'egfr': egfr,
        'urine_albumin': u_alb, 'urine_protein': u_prot, 'albumin_creatinine_ratio': acr,
        'urine_specific_gravity': usg, 'hemoglobin': hb, 'hba1c': hba1c, 'sodium': sodium,
        'potassium': potassium, 'cholesterol': chol, 'wbc_count': wbc
    }
    
    input_data.update(mapping)
    df_input = pd.DataFrame([input_data])[features]

    # Prediksi
    pred = model.predict(df_input)
    proba = model.predict_proba(df_input)
    res_text = le.inverse_transform(pred)[0]

    st.markdown("---")
    st.subheader("Hasil Analisis Klinis")
    
    # Logic warna hasil
    if "Healthy" in res_text:
        st.balloons()
        st.success(f"### STATUS: {res_text}")
    else:
        st.error(f"### STATUS: {res_text}")
    
    conf_score = np.max(proba) * 100
    st.write(f"**Tingkat Kepercayaan Model:** {conf_score:.2f}%")
    st.progress(int(conf_score))