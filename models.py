# Define all models with their hyperparameters

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Dictionary of models to train
MODELS = {
    "Logistic Regression": LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    ),
    
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ),
    
    "XGBoost": XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        use_label_encoder=False
    ),
    
    "SVM": SVC(
        kernel='rbf',
        C=1.0,
        probability=True,
        random_state=42
    )
}

# Print available models
if __name__ == "__main__":
    print("Available models:")
    for model_name in MODELS.keys():
        print(f"  - {model_name}")
