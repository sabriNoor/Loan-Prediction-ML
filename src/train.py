import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from preprocess import preprocess
from model import LoanModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "train.csv")

df = pd.read_csv(data_path)
print(df.head())
print(df.info())
print(df.isnull().sum())

# String columns only (object + pandas StringDtype)
str_cols = df.select_dtypes(include=["object", "string"]).columns
str_cols = str_cols.drop("Loan_ID", errors="ignore")  # remove ID column from analysis

for col in str_cols:
    print(f"\n=== {col} ===")
    print("Values:", df[col].dropna().unique())          # distinct string values
    print("Counts:")
    print(df[col].value_counts(dropna=False))            # include NaN count

y = df['Loan_Status'].map({'Y': 1, 'N': 0})
X = df.drop('Loan_Status', axis=1)

X = preprocess(X)
print(X.isnull().sum())

# -------------------
# Train/test split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------
# Model
# -------------------
model = LoanModel()
model.train(X_train, y_train)

# -------------------
# Predict
# -------------------
y_pred = model.predict(X_test)

# -------------------
# Evaluate
# -------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


