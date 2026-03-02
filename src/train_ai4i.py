import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("data/ai4i.csv")

df = df.drop(columns=["UDI", "Product ID"])
df["Type"] = df["Type"].map({"L":0,"M":1,"H":2})

X = df[[
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]]

y = df["Machine failure"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=300, max_depth=15)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

joblib.dump(model, "models/failure_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")