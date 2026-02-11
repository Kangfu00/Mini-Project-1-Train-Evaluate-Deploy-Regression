import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

print("⏳ กำลังโหลดข้อมูล...")
df = pd.read_csv('diamonds.csv')

# Mapping
cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

df['cut_score'] = df['cut'].map(cut_map)
df['color_score'] = df['color'].map(color_map)
df['clarity_score'] = df['clarity'].map(clarity_map)

# Features & Target (ไม่ต้องมี Log แล้ว!)
feature_cols = ['carat', 'cut_score', 'color_score', 'clarity_score', 'depth']
X = df[feature_cols]
y = df['price'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 กำลังเทรนโมเดล (Multiple Linear Regression)...")
model = LinearRegression()
model.fit(X_train, y_train)

# วัดผลแบบตรงไปตรงมา
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("-" * 30)
print(f"✅ Model Accuracy (R2 Score): {score:.4f}")  # ควรได้ประมาณ 0.90
print(f"✅ Mean Absolute Error: ${mae:.2f}")
print("-" * 30)

joblib.dump(model, 'diamond_model.pkl')
print("💾 บันทึกโมเดลเรียบร้อย: diamond_model.pkl")