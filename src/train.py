import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

from preprocess import preprocess
from model import LogisticModel, RandomForestModel, SVMModel

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
print(y.value_counts(normalize=True))

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
# Feature scaling
# -------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -------------------
# Models
# -------------------
models = {
    "logreg": LogisticModel(),
    "rf": RandomForestModel(),
    "svm": SVMModel()
}
best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    # Train and predict
    model.train(X_train, y_train)
    pred = model.predict(X_test)

    # Evaluation
    print(name)
    print(accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))

    # Update best model
    acc = accuracy_score(y_test, pred)
    if acc > best_score:
        best_score = acc
        best_model = model
        best_name = name

print(f"\nBest model: {best_name} with accuracy: {best_score}")

# create folder if not exists
models_path = os.path.join(BASE_DIR, "models")

os.makedirs(models_path, exist_ok=True)

# save scaler
joblib.dump(scaler, os.path.join(models_path, "scaler.pkl"))

# save best model
joblib.dump(best_model.model, os.path.join(models_path, "best_model.pkl"))

print(f"Saved best model: {best_name}")