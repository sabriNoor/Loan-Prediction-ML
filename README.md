# 🏦 Loan Approval Prediction API

A machine learning project that predicts loan approval status based on applicant information.
The system includes data preprocessing, model training, evaluation, and deployment using FastAPI and Docker.

---

## 🚀 Features

* Data preprocessing (handling missing values, encoding categorical features)
* Multiple models comparison:

  * Logistic Regression
  * Random Forest (best model)
  * Support Vector Machine
* Model evaluation with accuracy, precision, recall, F1-score
* REST API for real-time predictions using FastAPI
* Dockerized for easy deployment

---

## 🧠 Problem Statement

Given applicant details (income, credit history, etc.), predict whether a loan will be approved.

---

## 📊 Dataset

* **Name:** Loan Prediction Dataset
* **Size:** 614 samples, 13 features
* **Target:** `Loan_Status` (Y/N)
* **Type:** Supervised classification

---

## ⚙️ Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* FastAPI
* Docker

---

## 🧹 Preprocessing

* Missing values handled using:

  * Mode (categorical)
  * Median (numerical)
* Encoding:

  * Binary encoding for categorical variables
  * One-hot encoding for `Property_Area`
* Feature alignment ensured between training and inference

---

## 🤖 Models

| Model               | Accuracy         |
| ------------------- | ---------------- |
| Logistic Regression | ~0.82            |
| Random Forest       | **~0.85 (Best)** |
| SVM                 | ~0.81            |

---

## 🧪 Evaluation

* Metric: Accuracy + F1-score
* Observed class imbalance (~69% approved)
* Applied `class_weight='balanced'` to reduce bias

---

## 🧱 Project Structure

```
loan-ml-project/
│
├── data/
│   └── train.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── columns.pkl
│
├── src/
│   ├── train.py
│   ├── preprocess.py
│   ├── model.py
│   ├── predict.py
│   └── api.py
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python src/train.py
uvicorn src.api:app --reload
```

---

## 🌐 API Usage

Open Swagger UI:

```
http://localhost:8000/docs
```

### Example Request

```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "1",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 2000,
  "LoanAmount": 150,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": "Urban"
}
```

### Response

```json
{
  "prediction": 1
}
```

---

## 🐳 Docker

### Build

```bash
docker build -t loan-ml-api .
```

### Run

```bash
docker run -p 8000:8000 loan-ml-api
```

---

## 🧠 Key Design Decisions

* Used structured JSON input instead of raw arrays for better usability
* Ensured consistent preprocessing between training and inference
* Stored model artifacts (model, scaler, columns) for reproducibility
* Selected Random Forest as best model based on evaluation metrics

---

## ⚠️ Limitations

* Small dataset → limited generalization
* Some imbalance remains in predictions
* Feature engineering could be improved

---

## 🚀 Future Improvements

* Use sklearn Pipeline for cleaner preprocessing
* Add model monitoring and logging
* Deploy to cloud (AWS / GCP)
* Improve performance with hyperparameter tuning

---

## 👨‍💻 Author

Nour Sabri

---
