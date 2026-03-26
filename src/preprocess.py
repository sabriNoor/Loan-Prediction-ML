import pandas as pd

def preprocess(df):
    df = df.copy()

    # -------------------
    # 1. Missing values
    # -------------------

    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median())

    # -------------------
    # 2. Drop ID
    # -------------------
    df = df.drop('Loan_ID', axis=1)

    # -------------------
    # 3. Encoding
    # -------------------

    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})

    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)

    df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)

    return df