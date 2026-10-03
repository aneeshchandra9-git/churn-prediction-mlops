import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, filepath):
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def clean_data(self, df):
        print("Cleaning data...")
        df = df.drop(['CustomerID'], axis=1, errors='ignore')
        df = df.dropna()
        print(f"✓ Data cleaned: {df.shape}")
        return df
    
    def transform_data(self, df, fit=True):
        print("Transforming data...")
        df = df.copy()
        
        if 'Churn' in df.columns:
            X = df.drop('Churn', axis=1)
            y = df['Churn'].map({'Yes': 1, 'No': 0})
        else:
            X = df
            y = None
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                if col in self.label_encoders:
                    X[col] = self.label_encoders[col].transform(X[col])
        
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        if fit:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        print(f"✓ Data transformed: {X.shape}")
        
        if y is not None:
            return X, y
        else:
            return X
    
    def split_data(self, X, y, test_size=0.2):
        print("Splitting data...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"✓ Train set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def run_pipeline(self, filepath):
        print("="*50)
        print("Starting Data Pipeline")
        print("="*50)
        
        df = self.load_data(filepath)
        df = self.clean_data(df)
        X, y = self.transform_data(df, fit=True)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        print("="*50)
        print("✓ Pipeline Complete!")
        print("="*50)
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    pipeline = DataPipeline()
    X_train, X_test, y_train, y_test = pipeline.run_pipeline('data/raw/churn_data.csv')
    print("\nReady to train model!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")