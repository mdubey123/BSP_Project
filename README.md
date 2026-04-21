# 🏭 BSP Procurement Intelligence Platform

An AI-powered procurement analytics and decision-support system built for **Bhilai Steel Plant (BSP)**.
This platform enables data-driven insights, vendor evaluation, and saving prediction using machine learning.

---

## 🚀 Project Overview

The BSP Procurement Intelligence Platform is designed to:

* Analyze procurement data
* Identify cost-saving opportunities
* Evaluate vendor performance
* Predict expected savings using ML
* Generate reports for decision-making

It provides a **complete end-to-end pipeline** from raw data → insights → prediction → reporting.

---

## ✨ Key Features

### 🔐 Secure Login System

* Role-based authentication (Admin, Analyst, Viewer)
* Session tracking & audit logging

---

### 📊 Analytics Dashboard

* Upload Excel datasets (multi-sheet supported)
* Automatic preprocessing & feature engineering
* KPI generation:

  * Total Saving
  * Average Saving
  * Total Records
  * Loss Cases

---

### 📈 Data Visualization

* Saving distribution (histogram)
* Vendor performance charts
* Monthly trend analysis

---

### 🏭 Vendor Intelligence

* Top vendors ranked by total savings
* Profitability classification:

  * ✅ Profitable
  * ❌ Loss-making

---

### 🤖 AI Prediction Engine

* Predict expected saving using:

  * PR Value
  * Negotiation Value

* Model: **Linear Regression**

* Output includes:

  * Predicted saving
  * Savings %
  * Status (Profit/Loss)

---

### 📄 Report Generation

* Export:

  * 📄 PDF reports
  * 📊 Excel reports
* Includes KPI summary + vendor analysis

---

### 🔍 Data Quality Check

* Missing value analysis
* Duplicate detection
* Statistical insights

---

### 📋 Audit Logging

* Tracks all user actions:

  * Login
  * Data upload
  * Predictions
  * Exports

---

## 🧠 Machine Learning Approach

* Algorithm: **Linear Regression**
* Features used:

  * `PR_VALUE`
  * `NEGOTIATION_VAL`
* Target:

  * `saving = PR_VALUE - NEGOTIATION_VAL`

This ensures alignment between:

* input fields (UI)
* model training
* prediction output

---

## 🛠️ Tech Stack

* **Python**
* **Pandas** (Data processing)
* **Scikit-learn** (ML model)
* **Matplotlib & Seaborn** (Visualization)
* **Gradio** (Frontend UI)
* **ReportLab** (PDF generation)

---

## 📁 Project Structure

```
BSP_Project/
│
├── assets/                # Images (BSP photos, logo)
├── data/                  # Dataset folder
│
├── src/                   # Backend logic
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── analysis.py
│   └── model.py
│
├── gradio_app.py          # Main application (UI + integration)
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/BSP-Procurement-Intelligence.git
cd BSP-Procurement-Intelligence
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run Application

```bash
python gradio_app.py
```

---

## 🔑 Login Credentials

```
Admin:
User ID: bsp_admin
Password: admin@2026

Analyst:
User ID: bsp_analyst
Password: analyst@2026

Viewer:
User ID: bsp_viewer
Password: view@2026
```

---

## 📊 How to Use

1. Login to the platform
2. Go to **Analytics tab**
3. Upload dataset (.xlsx)
4. View KPIs and charts
5. Explore:

   * Vendor Intelligence
   * Prediction Engine
   * Reports
6. Generate PDF/Excel reports

---

## 📌 Use Cases

* Procurement cost optimization
* Vendor performance analysis
* Loss detection in contracts
* Decision support for management

---

## 🎯 Future Enhancements

* Advanced ML models (Random Forest, XGBoost)
* Real-time database integration
* Cloud deployment (AWS/Azure)
* Role-based dashboards
* Predictive anomaly detection

---

## 👩‍💻 Author

**Manya Dubey**
Computer Science Student

---

## 📜 License

This project is for **academic and demonstration purposes only**.

---

## ⭐ Acknowledgement

Inspired by real-world procurement workflows at
**Bhilai Steel Plant (SAIL), India**
