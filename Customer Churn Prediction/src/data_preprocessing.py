import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)
    # Drop unnecessary columns
    df.drop(['customerID'], axis=1, inplace=True)
    # Convert target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
