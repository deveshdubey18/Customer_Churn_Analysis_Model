# Customer Churn Analysis Model

![Python](https://img.shields.io/badge/Python-3.13%2B-blue?logo=python)
![Scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple?logo=pandas)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Customer_Churn-green)

A modular **Customer Churn Analysis and Prediction Model** built using Python and Scikit-learn. The project predicts customer churn using **Random Forest** and performs **K-Means clustering** for customer segmentation.

---

## 📌 Project Overview

Customer churn can significantly affect business revenue and growth. This project uses machine learning to predict whether a customer is likely to churn and groups customers into different segments.

### Main Components

* **Random Forest Classification** — predicts customer churn.
* **K-Means Clustering** — segments customers into 3 groups.
* **SMOTE** — handles class imbalance.
* **PCA** — reduces feature dimensionality.

---

## 🎯 Objectives

* Preprocess customer data.
* Handle missing values, outliers, and categorical features.
* Balance the dataset using SMOTE.
* Reduce dimensionality using PCA.
* Predict customer churn using Random Forest.
* Segment customers using K-Means.
* Evaluate model performance.

---

## 🗂️ Project Structure

```text
Customer_Churn_Analysis_Model/
│
├── data/
│   └── raw/
│       └── churn_dataset.csv
│
├── models/
│   └── model.pkl
│
├── src/
│   └── customerchurnanalysismodel/
│       ├── data_ingestion.py
│       ├── data_preprocessing.py
│       ├── model_building.py
│       └── model_cluster.py
│
├── main.py
├── pyproject.toml
└── README.md
```

---

## ⚙️ Machine Learning Workflow

```text
Customer Dataset
      ↓
Data Ingestion
      ↓
Data Preprocessing
      ↓
SMOTE
      ↓
PCA
      ↓
 ┌───────────────┬───────────────┐
 ↓               ↓
Random Forest    K-Means
 ↓               ↓
Churn Prediction Customer Segments
```

---

## 🧩 Modules

### Data Ingestion

Loads the customer churn dataset using Pandas.

### Data Preprocessing

Cleans and transforms the data using imputation, encoding, scaling, winsorization, SMOTE, and PCA.

### Model Building

Trains a **Random Forest Classifier**, evaluates it using a classification report, and saves the model as `model.pkl`.

### Model Clustering

Uses **K-Means** with 3 clusters to segment customers.

---

## 📊 Dataset

The dataset contains:

* **51,047 records**
* **58 columns**
* Target: `Churn`

| Value | Meaning  |
| ----- | -------- |
| `0`   | No Churn |
| `1`   | Churn    |

---

## 🧠 Model Performance

### Random Forest

| Metric    | Class 0 | Class 1 |
| --------- | ------: | ------: |
| Precision |    0.72 |    0.36 |
| Recall    |    0.83 |    0.23 |
| F1-Score  |    0.77 |    0.28 |

**Test Accuracy: 65.54%**

| Metric      |      Score |
| ----------- | ---------: |
| Accuracy    | **65.54%** |
| Macro F1    |   **0.53** |
| Weighted F1 |   **0.63** |

---

## 🔍 K-Means Clustering

**Number of Clusters: 3**

### Training Distribution

| Cluster | Customers |
| ------- | --------: |
| 0       |    22,355 |
| 1       |    18,732 |
| 2       |     9,893 |

### Testing Distribution

| Cluster | Customers |
| ------- | --------: |
| 0       |     6,743 |
| 1       |     5,393 |
| 2       |     3,179 |

**Iterations:** 51
**Inertia:** 259667.0938
**Random State:** 42

---

## 🛠️ Tech Stack

* **Python 3.13+**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Imbalanced-learn**
* **SciPy**
* **Matplotlib**
* **Seaborn**
* **MLflow**
* **FLAML**

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/deveshdubey18/Customer_Churn_Analysis_Model.git
cd Customer_Churn_Analysis_Model
```

### Install Dependencies

Using `uv`:

```bash
uv sync
```

---

## ▶️ Run

```bash
python main.py
```

The trained Random Forest model will be saved to:

```text
models/model.pkl
```

---

## 📈 Future Improvements

* Improve churn-class recall.
* Perform hyperparameter tuning.
* Compare additional ML algorithms.
* Add feature importance analysis.
* Optimize the number of K-Means clusters.
* Add a Streamlit dashboard.
* Improve MLflow experiment tracking.

---

## 👨‍💻 Author

**Devesh Dubey**

GitHub: [@deveshdubey18](https://github.com/deveshdubey18)

---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended for **educational and portfolio purposes**.
