import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ================================
# HARUS PALING ATAS
# ================================
st.set_page_config(
    page_title="Prediksi Diabetes",
    page_icon="💉",
    layout="centered"
)

# ================================
# Load model
# ================================
@st.cache_resource
def load_model():
    with open("XGB_diabetes_medical.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

st.title("💉 Aplikasi Prediksi Diabetes")
st.write("Masukkan data pasien untuk memprediksi risiko diabetes menggunakan model XGBoost.")

def preprocess_input(df):
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    
    for col in zero_as_missing:
        df[col] = df[col].replace(0, np.nan)

    median_values = {
        "Glucose": 117,
        "BloodPressure": 72,
        "SkinThickness": 29,
        "Insulin": 125,
        "BMI": 32.3
    }

    df.fillna(median_values, inplace=True)
    return df

def explain_prediction(df):
    explanations = []

    if df["Glucose"].values[0] > 140:
        explanations.append("Glukosa darah tinggi")
    if df["BMI"].values[0] > 30:
        explanations.append("BMI menunjukkan indikasi obesitas")
    if df["Age"].values[0] > 45:
        explanations.append("Usia termasuk faktor risiko")
    if df["DiabetesPedigreeFunction"].values[0] > 0.8:
        explanations.append("Riwayat keluarga diabetes kuat")
    if df["BloodPressure"].values[0] > 85:
        explanations.append("Tekanan darah terlalu tinggi")
    if df["BloodPressure"].values[0] < 60:
        explanations.append("Tekanan darah terlalu rendah")

    if not explanations:
        explanations.append("Tidak ada faktor risiko dominan")

    return explanations

def risk_level(proba):
    if proba < 0.3:
        return "Rendah", "🟢"
    elif proba < 0.6:
        return "Sedang", "🟡"
    else:
        return "Tinggi", "🔴"

data_pasien, rentang_normal, tips_kesehatan = st.tabs(['📋  Data Pasien', '🔍 Rentang Normal Indikator Fisiologis', '🩺 Tips Kesehatan'])
with data_pasien:
    st.header("Masukkan Data Pasien")
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, step=1)
        glucose = st.number_input("Glukosa (mg/dL)", min_value=50, max_value=300, value=70)
        blood_pressure = st.number_input("Tekanan Darah diastolik (mmHg)", min_value=30, max_value=250, value=85)
        skin_thickness = 29
        
    with col2:
        insulin = 125
        bmi = st.number_input("BMI (kg/m²)", min_value=1.0, max_value=60.0, format="%.2f", 
                            help= "BMI dihitung dari berat badan (kg) dibagi tinggi badan kuadrat (m²). Contoh: 70 / (1.70²) = 24.22")
        dpf = 0.3725
        age = st.number_input("Usia (tahun)", min_value=15, max_value=120, step=1)

    # Warning input ekstrem
    if glucose < 70 or glucose > 200:
        st.warning("⚠️ Nilai glukosa berada di luar rentang normal.")
    if blood_pressure < 70 or blood_pressure > 200:
        st.warning("⚠️ Tekanan darah anda berada di luar rentang normal.")


    # ================================
    # Prediksi
    # ================================
    if st.button("Prediksi"):
        with st.spinner("🔍 Sedang memproses prediksi..."):
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, dpf, age]])

            input_df = pd.DataFrame(input_data, columns=[
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ])

            # preprocessing
            input_df = preprocess_input(input_df)

            # prediksi
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]

            # risk level
            level, icon = risk_level(proba)

            # ================================
            # Hasil Prediksi
            # ================================
            st.subheader("Hasil Prediksi")
            st.write(f"{icon} **Risiko Diabetes: {level}**")
            st.progress(int(proba * 100))
            st.caption(f"Probabilitas: {proba:.2%}")

            if prediction == 1:
                st.error("⚠️ Pasien berpotensi diabetes")
            else:
                st.success("✅ Pasien berpotensi tidak memiliki diabetes")

            # explainability
            st.markdown("### Faktor yang perlu diperhatikan:")
            explanations = explain_prediction(input_df)
            for e in explanations:
                st.write(f"- {e}")

            st.info("⚠️ Hasil ini bukan diagnosis medis. Harap konsultasikan ke tenaga kesehatan/klinik terdekat.")

with rentang_normal:
    st.info('Tekanan darah diastolik normal: 60–80 mmHg')
    st.info('kadar gula darah yang normal: 70-100 mg/dL (puasa 8 jam) dan di bawah 140 mg/dL dua jam setelah makan')
    st.info('cara menghitung Body Mass Index (BMI): Berat Badan (kg) / (Tinggi Badan (m) ) ²')
with tips_kesehatan:
    st.markdown(
    "<h3 style='text-align: center;'>tips kesehatan untuk meminimalisir risiko diabetes</h3>",
    unsafe_allow_html=True)
    column1, column2 = st.columns(2)
    with column1:
        st.info('- Atur Pola Makan Seimbang')
        st.info('- olahraga rutin setidaknya 30 menit/hari')
    with column2:
        st.info('- Hindari Rokok & Alkohol')
        st.info('- Tidur cukup & kelola stres')
# ================================
# Footer
# ================================
st.markdown("---")
st.caption("Dibuat dengan ❤️ menggunakan Streamlit dan XGBoost")