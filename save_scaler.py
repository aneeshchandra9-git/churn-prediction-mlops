from src.data_pipeline import DataPipeline
import joblib

pipeline = DataPipeline()
df = pipeline.load_data('data/raw/churn_data.csv')
df = pipeline.clean_data(df)
X, y = pipeline.transform_data(df, fit=True)

joblib.dump(pipeline.scaler, 'models/scaler.pkl')
print("✓ Scaler saved to models/scaler.pkl")

print("\nColumn order the scaler expects:")
print(list(X.columns))