import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

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
    
    # Train Scikit-Learn Random Forest Classifier
    print("\nTraining Scikit-Learn Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    
    print("\n" + "="*40)
    print("Random Forest Classifier Evaluation:")
    print("="*40)
    print(classification_report(y_test, rf_preds))
    print(f"RF Test Accuracy: {rf_acc:.4f}")
    
    # Rank feature importances to verify the impact of the sentiment tracking scores
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n" + "="*40)
    print("Ranked Feature Importances:")
    print("="*40)
    for idx, row in feature_importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")
        
    # Export trained classifier model
    os.makedirs("models", exist_ok=True)
    model_output_path = "models/churn_classifier.pkl"
    with open(model_output_path, 'wb') as handle:
        pickle.dump(rf_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"\nSuccessfully saved the Random Forest classifier model to '{model_output_path}'")

if __name__ == "__main__":
    main()
