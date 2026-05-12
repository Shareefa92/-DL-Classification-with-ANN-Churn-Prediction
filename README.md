# 🏦 Customer Churn Prediction using ANN

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-85.85%25-brightgreen?style=flat)

> A deep learning model that predicts whether a bank customer will churn, built with an Artificial Neural Network (ANN) and deployed as an interactive web app using Streamlit.

---

## 🎯 Business Problem

Banks lose significant revenue when customers leave. This model identifies **at-risk customers before they churn**, enabling the retention team to take proactive action and reduce customer loss.

---

## 📊 Dataset

| Feature | Details |
|---------|---------|
| Source | Churn_Modelling.csv |
| Records | 10,000 customers |
| Features | 10 (demographics + account info) |
| Target | `Exited` — 1 (churned) / 0 (stayed) |

**Key Features:** Credit Score, Geography, Gender, Age, Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, Estimated Salary

---

## 🧠 Model Architecture

```
Input Layer  →  Dense(64, ReLU)  →  Dropout(0.3)
             →  Dense(32, ReLU)  →  Dropout(0.3)
             →  Dense(1, Sigmoid)
```

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Epochs | 5 |
| Batch Size | 32 |
| **Test Accuracy** | **85.85%** |

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Deep Learning:** TensorFlow / Keras
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit

---

## ⚙️ Pipeline

```
Raw Data → Clean & Encode → Scale Features → Train ANN → Evaluate → Deploy
```

1. Drop irrelevant columns (RowNumber, CustomerId, Surname)
2. Label encode Gender
3. One-hot encode Geography
4. Standard scale all features
5. Train/Test split (80/20)
6. Build & train ANN
7. Evaluate on test set → **85.85% accuracy**

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/SharifahAldalbahi/-DL-Classification-with-ANN-Churn-Prediction.git
cd -DL-Classification-with-ANN-Churn-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📦 Requirements

```
tensorflow==2.15.0
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## 💼 Business Impact

| Benefit | Description |
|---------|-------------|
| 🎯 Early Detection | Identify churners before they leave |
| 💰 Cost Reduction | Retention is cheaper than acquisition |
| 📈 Revenue Protection | Keep high-value customers engaged |
| ⚡ Real-time Prediction | Instant results via Streamlit app |

---

## 👩‍💻 Author

**Sharifah Al-Dalbahi**  
M.Sc. Information Systems | AI & Data Science  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/sharifah-aldalbahi)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/SharifahAldalbahi)
