# 🦠 EpiCast — Epidemic Intelligence Platform

## 🚀 Overview

EpiCast is an AI-powered epidemic prediction platform that forecasts disease spread using historical epidemiological data and machine learning models. It helps identify outbreak risks, detect hotspots, and simulate transmission dynamics.

---

## 🎯 Problem Statement

Predicting infectious disease spread is critical for public health preparedness. Traditional methods lack real-time adaptability and predictive intelligence.

---

## 💡 Solution

EpiCast combines:

* 📊 Data-driven ML forecasting
* 🧠 Epidemiological modeling (SEIR)
* 🔥 Hotspot detection using growth + Rt
* 🌍 Interactive visualization dashboard

---

## ✨ Features

* Global outbreak visualization
* ML-based case prediction (Random Forest + Gradient Boosting)
* SEIR transmission modeling
* Real-time hotspot detection
* Risk classification (Low → Critical)
* Interactive dashboards (Plotly + Streamlit)

---

## 🛠 Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **ML Models:** Random Forest, Gradient Boosting
* **Visualization:** Plotly
* **Data Processing:** Pandas, NumPy
* **Scientific Computing:** SciPy

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/your-username/epicast.git
cd epicast
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔄 Technical Workflow

1. **Data Collection**

   * COVID-19 global dataset (Johns Hopkins)

2. **Preprocessing**

   * Cleaning, aggregation, smoothing

3. **Feature Engineering**

   * Growth rate
   * Acceleration
   * Rt estimation
   * Moving averages

4. **Model Training**

   * Random Forest Regressor
   * Gradient Boosting Regressor

5. **Forecasting**

   * Future case prediction with uncertainty bounds

6. **Epidemiological Modeling**

   * SEIR simulation for transmission dynamics

7. **Visualization**

   * Global map
   * Time-series trends
   * Risk dashboards

---

## 📊 Expected Outcomes

* Accurate outbreak prediction
* Early hotspot detection
* Public health decision support

---

## 📸 Demo

(Add screenshots in /assets)

---

## 🚀 Future Scope

* Real-time API integration
* Mobility + climate data integration
* Deep learning models (LSTM)
* Mobile app deployment

---

## 👥 Team

* Bhumisti Das
