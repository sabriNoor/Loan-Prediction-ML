import os
import pandas as pd

from preprocess import preprocess

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



