# 🏭 BSP Procurement Intelligence Platform

An AI-powered analytics and decision-support system built for **Bhilai Steel Plant (BSP)** to analyze procurement data, evaluate vendor performance, and predict savings using machine learning.

---

## 🚀 Overview

This project provides a complete end-to-end solution for procurement analytics:

* 📊 Data Analysis & Visualization
* 🏭 Vendor Performance Evaluation
* 🤖 Machine Learning-based Prediction
* 📄 Automated Report Generation
* 🎯 Interactive Dashboard (Gradio UI)

---

## ✨ Features

### 🔐 Secure Login System

* Username & password-based access
* Clean enterprise-style UI

### 📂 Dynamic Data Upload

* Upload Excel datasets (.xlsx)
* Automatically processes multiple sheets

### 📊 Analytics Dashboard

* Saving distribution charts
* Key procurement insights
* KPI summary

### 🏭 Vendor Intelligence

* Top vendors based on savings
* Data-driven vendor evaluation

### 🤖 AI Prediction Engine

* Predict procurement savings using ML model
* Inputs:

  * PR Value
  * Negotiation Value

### 📄 Report Generation

* Generate downloadable PDF reports
* Summary of procurement insights

---

## 🧠 Tech Stack

* Python
* Pandas
* Scikit-learn
* Matplotlib & Seaborn
* Gradio
* ReportLab

---

## 📁 Project Structure

BSP_Project/
│
├── assets/
├── data/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── analysis.py
│   └── model.py
│
├── gradio_app.py
├── requirements.txt
└── README.md

---

## ⚙️ Installation

1. Clone the repository
   git clone https://github.com/your-username/BSP-Procurement-Intelligence.git

2. Go to project folder
   cd BSP-Procurement-Intelligence

3. Install dependencies
   pip install -r requirements.txt

4. Run the app
   python gradio_app.py

---

## 🔑 Login Credentials

Username: bsp_admin
Password: BSP_2026_AI

---

## 📊 How to Use

1. Login to the system
2. Go to Analytics tab
3. Upload dataset (.xlsx)
4. View charts and insights
5. Use:

   * Vendor Intelligence
   * Prediction Engine
   * Reports

---

## 🧪 Machine Learning Model

* Algorithm: Linear Regression
* Inputs:

  * PR_VALUE
  * NEGOTIATION_VAL
* Output:

  * Predicted Saving

---

## 📌 Use Case

This system helps:

* Identify cost-saving opportunities
* Evaluate vendor efficiency
* Detect loss-making procurements
* Support data-driven decision-making

---

## 🎯 Future Improvements

* Advanced ML models
* Real-time database integration
* Role-based authentication
* Cloud deployment

---

## 👩‍💻 Author

Manya Dubey
Computer Science Student

---

## 📜 License

This project is for academic and demonstration purposes.

---

## ⭐ Acknowledgement

Inspired by procurement processes at Bhilai Steel Plant (BSP).
