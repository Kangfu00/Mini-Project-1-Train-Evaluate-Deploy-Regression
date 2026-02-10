import pandas as pd
import numpy as np  # <--- อย่าลืมบรรทัดนี้
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # <--- กลับมาใช้ตัวเดิมตามโจทย์
from sklearn.metrics import r2_score
import joblib

# 1. โหลดข้อมูล
print("กำลังโหลดข้อมูล...")
df = pd.read_csv('diamonds.csv')

# 2. จัดการข้อมูล (Mapping)
cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

df['cut_score'] = df['cut'].map(cut_map)
df['color_score'] = df['color'].map(color_map)
df['clarity_score'] = df['clarity'].map(clarity_map)

# 3. Train Model
feature_cols = ['carat', 'cut_score', 'color_score', 'clarity_score', 'depth']
X = df[feature_cols]

# *** ไฮไลท์สำคัญอยู่ตรงนี้ ***
# แปลงราคาเป็น Log ก่อนเทรน (เพื่อให้เป็นเส้นตรงและไม่ติดลบ)
y = np.log(df['price']) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("กำลังเทรนโมเดล (Multiple Linear Regression)...")
model = LinearRegression()
model.fit(X_train, y_train)

# 4. วัดผล (ต้องแปลงกลับเป็นค่าจริงก่อนวัด)
y_pred_log = model.predict(X_test)
y_pred_real = np.exp(y_pred_log) # แปลงจาก Log กลับเป็นราคาจริง
y_test_real = np.exp(y_test)

score = r2_score(y_test_real, y_pred_real)
print(f"✅ Model Accuracy (R2 Score): {score:.4f}") 
# ค่า R2 น่าจะอยู่ที่ประมาณ 0.90 - 0.93 ซึ่งถือว่าผ่านฉลุยสำหรับ Linear Regression

# 5. Save Model
joblib.dump(model, 'diamond_model.pkl')
print("💾 บันทึกโมเดลเรียบร้อย: diamond_model.pkl")