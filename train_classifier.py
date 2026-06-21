import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

def main():
    input_path = "data/final_engineered_features.csv"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist. Please run feature_pipeline.py first.")
        return
        
    print(f"Loading engineered features from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Separate features and target
    # Exclude user_id (identifier) and is_churned (target)
    feature_cols = [col for col in df.columns if col not in ['user_id', 'is_churned']]
    X = df[feature_cols]
    y = df['is_churned']
    
    print(f"Features: {list(X.columns)}")
    print(f"Target count: \n{y.value_counts()}")
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    
    print("\n" + "="*40)
    print("Random Forest Classifier Evaluation:")
    print("="*40)
    print(classification_report(y_test, rf_preds))
    print(f"RF Test Accuracy: {rf_acc:.4f}")
    
    # Train XGBoost Classifier
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    
    print("\n" + "="*40)
    print("XGBoost Classifier Evaluation:")
    print("="*40)
    print(classification_report(y_test, xgb_preds))
    print(f"XGBoost Test Accuracy: {xgb_acc:.4f}")
    
    # Select the model with the higher accuracy
    if xgb_acc >= rf_acc:
        best_model = xgb_model
        model_name = "XGBoost Classifier"
    else:
        best_model = rf_model
        model_name = "Random Forest Classifier"
        
    print(f"\nSelecting {model_name} as the final classifier.")
    
    # Export trained classifier model
    os.makedirs("models", exist_ok=True)
    model_output_path = "models/churn_classifier.pkl"
    with open(model_output_path, 'wb') as handle:
        pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Successfully saved the final classifier model to '{model_output_path}'")

if __name__ == "__main__":
    main()
