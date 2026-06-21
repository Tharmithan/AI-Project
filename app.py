import os
import pickle
import numpy as np
import pandas as pd
import pymysql
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load environment
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "retail_analytics")

# Global dictionary to store models
ml_assets = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load all models into memory
    print("Loading sentiment analysis model and tokenizer...")
    tokenizer_path = "models/tokenizer.pickle"
    sentiment_path = "models/sentiment_model.h5"
    classifier_path = "models/churn_classifier.pkl"
    
    if not (os.path.exists(tokenizer_path) and os.path.exists(sentiment_path) and os.path.exists(classifier_path)):
        print("WARNING: Model files not found! Please run train_nlp.py and train_classifier.py first.")
    else:
        with open(tokenizer_path, "rb") as handle:
            ml_assets["tokenizer"] = pickle.load(handle)
        ml_assets["sentiment_model"] = load_model(sentiment_path)
        with open(classifier_path, "rb") as handle:
            ml_assets["churn_classifier"] = pickle.load(handle)
        print("All models loaded successfully!")
    yield
    # Shutdown: clean up
    ml_assets.clear()
    print("Cleaned up loaded models.")

app = FastAPI(
    title="Multimodal Churn Predictor API",
    description="Real-time churn risk classification using customer activity logs and support ticket sentiment analysis.",
    version="1.0.0",
    lifespan=lifespan
)

class ChurnRequest(BaseModel):
    user_id: str

class ChurnResponse(BaseModel):
    user_id: str
    churn_risk: float
    status: str
    trigger: str

@app.post("/predict", response_model=ChurnResponse)
async def predict_churn(payload: ChurnRequest):
    user_id = payload.user_id
    
    if "churn_classifier" not in ml_assets:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not loaded on server startup. Ensure training steps are completed."
        )
        
    # Connect to MySQL and fetch user details
    connection = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    
    # Query details
    query = """
    SELECT u.user_id, u.subscription_type,
           a.login_count_30d, a.total_spend, a.days_since_last_login,
           t.ticket_text
    FROM users u
    LEFT JOIN activity_logs a ON u.user_id = a.user_id
    LEFT JOIN support_tickets t ON u.user_id = t.user_id
    WHERE u.user_id = %s
    """
    
    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(query, (user_id,))
            rows = cursor.fetchall()
    finally:
        connection.close()
        
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID '{user_id}' not found."
        )
        
    # Aggregate data
    # Standard profile metrics (same for all rows of this user)
    base_row = rows[0]
    login_count_30d = float(base_row.get("login_count_30d") or 0.0)
    total_spend = float(base_row.get("total_spend") or 0.0)
    days_since_last_login = float(base_row.get("days_since_last_login") or 0.0)
    subscription_type = base_row.get("subscription_type") or "Free"
    
    # Collect support tickets text and predict sentiment scores
    tickets = [r["ticket_text"] for r in rows if r.get("ticket_text")]
    
    if tickets:
        tokenizer = ml_assets["tokenizer"]
        sentiment_model = ml_assets["sentiment_model"]
        
        # Preprocess and predict
        sequences = tokenizer.texts_to_sequences(tickets)
        padded = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
        sentiment_preds = sentiment_model.predict(padded)
        
        # Calculate mean sentiment score
        support_sentiment_score = float(np.mean(sentiment_preds))
    else:
        support_sentiment_score = 0.5
        
    # Construct tabular feature vector matching the encoder columns alphabetical order:
    # login_count_30d, total_spend, days_since_last_login, support_sentiment_score,
    # subscription_type_Basic, subscription_type_Free, subscription_type_Premium
    sub_basic = 1 if subscription_type == "Basic" else 0
    sub_free = 1 if subscription_type == "Free" else 0
    sub_premium = 1 if subscription_type == "Premium" else 0
    
    # Feature columns array
    features = np.array([[
        login_count_30d,
        total_spend,
        days_since_last_login,
        support_sentiment_score,
        sub_basic,
        sub_free,
        sub_premium
    ]])
    
    # Run classification model
    classifier = ml_assets["churn_classifier"]
    
    # Predict probability of churn (class 1)
    # Most classifiers support predict_proba
    if hasattr(classifier, "predict_proba"):
        churn_risk = float(classifier.predict_proba(features)[0][1])
    else:
        churn_risk = float(classifier.predict(features)[0])
        
    # Determine risk category
    if churn_risk >= 0.70:
        risk_status = "High Risk"
    elif churn_risk >= 0.35:
        risk_status = "Medium Risk"
    else:
        risk_status = "Low Risk"
        
    # Determine primary trigger reason
    if support_sentiment_score < 0.35:
        trigger = "Low sentiment score on recent support tickets."
    elif days_since_last_login > 15:
        trigger = "High inactivity period (days since last login exceeds 15 days)."
    elif login_count_30d < 3:
        trigger = "Very low login frequency in the last 30 days."
    elif total_spend < 15.0:
        trigger = "Low overall customer spend metrics."
    else:
        trigger = "Combination of moderate activity decline and general customer inquiries."
        
    return ChurnResponse(
        user_id=user_id,
        churn_risk=round(churn_risk, 4),
        status=risk_status,
        trigger=trigger
    )
