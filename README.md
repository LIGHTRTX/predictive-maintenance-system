# Predictive Machine Failure Monitoring System

A machine learning based predictive monitoring system that estimates **machine failure probability and operational risk** using sensor data and stores real-time logs for analysis.

The system combines **machine learning predictions with electrical load analysis and signal metrics** to produce a unified machine health risk score.

---

# Overview

Industrial machines often fail due to hidden operational patterns such as overheating, overload conditions, or abnormal vibration.

This project builds a **predictive monitoring pipeline** that:

1. Predicts machine failure probability using a trained ML model
2. Calculates overload conditions from current measurements
3. Incorporates signal RMS values as a vibration indicator
4. Computes a unified risk score
5. Stores real-time machine logs in a database

---

# Dataset

Dataset used: **AI4I Predictive Maintenance Dataset**

Features used:

* Air temperature
* Process temperature
* Rotational speed
* Torque
* Tool wear

Target:

* Machine failure

---

# Machine Learning Model

Model: **Random Forest Classifier**

Training steps:

* Feature selection
* Data scaling using StandardScaler
* Train-test split
* Model training with 300 trees

Performance is evaluated using classification accuracy.

The trained model and scaler are exported for inference.

Saved artifacts:

```
models/failure_model.pkl
models/scaler.pkl
```

---

# Risk Assessment Pipeline

The system combines multiple indicators to estimate machine health.

## Failure Probability

Obtained from the trained machine learning model.

## Overload Ratio

Measures electrical overload conditions.

Formula:

```
overload_ratio = current / rated_current
```

## RMS Signal

Represents vibration intensity.

Higher RMS values may indicate abnormal machine behavior.

---

# Unified Risk Score

A weighted risk score combines all indicators:

```
risk_score =
0.4 × failure_probability
+ 0.3 × overload_ratio
+ 0.3 × rms_signal
```

---

# Risk Classification

| Risk Score | Status   | Color  |
| ---------- | -------- | ------ |
| < 0.4      | Normal   | Green  |
| 0.4 – 0.7  | Warning  | Orange |
| > 0.7      | Critical | Red    |

---

# Database Logging

The system stores machine telemetry and risk scores using **SQLite**.

Database table:

```
logs
```

Stored fields:

* Timestamp
* Air temperature
* Process temperature
* RPM
* Torque
* Current
* Failure probability
* Overload ratio
* RMS vibration
* Risk score
* Machine status

This allows long-term monitoring and historical analysis.

---

# Project Structure

```
project/
│
├── data/
│   └── ai4i.csv
│
├── models/
│   ├── failure_model.pkl
│   └── scaler.pkl
│
├── database/
│   └── machine_data.db
│
├── train_model.py
├── risk_engine.py
├── database.py
└── README.md
```

---

# How to Run

Install dependencies:

```
pip install pandas numpy scikit-learn joblib
```

Train model:

```
python train_model.py
```

Run monitoring pipeline:

```
python main.py
```

---

# Applications

* Industrial predictive maintenance
* Manufacturing equipment monitoring
* Early failure detection
* Smart factory systems

---

# Future Improvements

* Real-time sensor streaming
* IoT integration
* Dashboard visualization
* Deep learning based anomaly detection
* Cloud-based monitoring systems

---

# Technologies Used

* Python
* Scikit-learn
* NumPy
* Pandas
* SQLite
* Joblib
