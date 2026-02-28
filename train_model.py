import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import joblib

# 1️⃣ Load dataset
df = pd.read_csv("measures_v2.csv")

# 2️⃣ Drop unwanted columns
df = df.drop(['profile_id','torque'], axis=1)

# 3️⃣ Separate input (X) and output (y)
X = df.drop('pm', axis=1)   # pm = rotor temperature (target)
y = df['pm']

# 4️⃣ Scale the input data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 5️⃣ Split into training & testing data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# 6️⃣ Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 7️⃣ Save model & scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained successfully!")