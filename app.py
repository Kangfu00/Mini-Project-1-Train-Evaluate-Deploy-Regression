import streamlit as st
import joblib
import pandas as pd

# โหลดโมเดล
try:
    model = joblib.load('diamond_model.pkl')
except:
    st.error("ไม่พบไฟล์โมเดล (diamond_model.pkl)")
    st.stop()

st.title("💎 Diamond Price Predictor")
st.write("ทำนายราคาเพชร (Multiple Linear Regression)")
st.write("---")

st.sidebar.header("ระบุคุณสมบัติเพชร")
# Input ต่างๆ
carat = st.sidebar.number_input("น้ำหนักกะรัต (Carat)", 0.1, 5.0, 0.5, 0.01)
depth = st.sidebar.number_input("ความลึก (Depth %)", 40.0, 80.0, 61.5, 0.1)
cut = st.sidebar.selectbox("การเจียระไน (Cut)", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.sidebar.selectbox("สี (Color)", ['J (แย่สุด)', 'I', 'H', 'G', 'F', 'E', 'D (ดีสุด)'])
clarity = st.sidebar.selectbox("ความสะอาด (Clarity)", ['I1 (แย่สุด)', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF (ดีสุด)'])

# Mapping ค่า
cut_val = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}[cut]
color_val = {'J (แย่สุด)': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D (ดีสุด)': 7}[color]
clarity_val = {'I1 (แย่สุด)': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF (ดีสุด)': 8}[clarity]

if st.button("💰 คำนวณราคา"):
    input_data = pd.DataFrame([[carat, cut_val, color_val, clarity_val, depth]],
                              columns=['carat', 'cut_score', 'color_score', 'clarity_score', 'depth'])
    
    # -------------------------------------------------------
    # จุดที่แก้: ใช้ค่าทำนายตรงๆ เลย ไม่ต้องมี np.exp() แล้ว
    # -------------------------------------------------------
    price = model.predict(input_data)[0]
    
    # ดักจับค่าติดลบ (ถ้ามี)
    if price < 0:
        st.error(f"⚠️ สเปกต่ำเกินไป (AI ประเมินได้ ${price:.2f}) แนะนำให้ปรับสเปกเพิ่มขึ้น")
    else:
        st.success(f"ราคาประเมิน: ${price:,.2f} USD")
        st.info(f"คิดเป็นเงินไทยประมาณ: {price * 35:,.0f} บาท")