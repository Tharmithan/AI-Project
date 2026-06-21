import os
import pickle
import numpy as np
import pandas as pd
import pymysql
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

def main():
    print("Loading models and tokenizer...")
    # Load NLP model and tokenizer
    with open("models/tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    model = load_model("models/sentiment_model.h5")
    
    # Connect to MySQL database
    connection = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    
    # Query to join all database metrics
    query = """
    SELECT u.user_id, u.subscription_type, u.is_churned,
           a.login_count_30d, a.total_spend, a.days_since_last_login,
           t.ticket_text
    FROM users u
    LEFT JOIN activity_logs a ON u.user_id = a.user_id
    LEFT JOIN support_tickets t ON u.user_id = t.user_id
    """
    
    try:
        df = pd.read_sql(query, connection)
    finally:
        connection.close()
        
    print(f"Loaded {len(df)} database records for merging.")
    
    # Preprocess and predict support ticket sentiment
    # For rows where ticket_text is null or empty, use a default sentiment score of 0.5
    sentiment_scores = []
    
    # Find unique texts to predict efficiently
    unique_texts = df['ticket_text'].dropna().unique()
    text_to_sentiment = {}
    
    if len(unique_texts) > 0:
        max_length = 50
        sequences = tokenizer.texts_to_sequences(unique_texts)
        padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        
        predictions = model.predict(padded)
        
        for text, score in zip(unique_texts, predictions):
            text_to_sentiment[text] = float(score[0])
            
    # Map predictions back to main dataframe
    def get_sentiment(text):
        if pd.isna(text) or not text:
            return 0.5 # default neutral sentiment
        return text_to_sentiment.get(text, 0.5)
        
    df['support_sentiment_score'] = df['ticket_text'].apply(get_sentiment)
    
    # Drop ticket_text since it is unstructured and we have engineered support_sentiment_score
    df = df.drop(columns=['ticket_text'])
    
    # If a user has multiple rows (multiple tickets), aggregate by user_id
    # Take the mean for numeric features, and first/mode for subscription_type and is_churned
    aggregation_rules = {
        'subscription_type': 'first',
        'is_churned': 'first',
        'login_count_30d': 'mean',
        'total_spend': 'mean',
        'days_since_last_login': 'mean',
        'support_sentiment_score': 'mean'
    }
    
    df_grouped = df.groupby('user_id').agg(aggregation_rules).reset_index()
    
    # Apply One-Hot Encoding to categorical column 'subscription_type'
    # Specify prefix and ensure columns are consistent: 'subscription_type_Free', 'subscription_type_Basic', 'subscription_type_Premium'
    df_encoded = pd.get_dummies(df_grouped, columns=['subscription_type'], dtype=int)
    
    # Ensure all possible subscription types exist in the final columns
    for sub_type in ['Free', 'Basic', 'Premium']:
        col_name = f'subscription_type_{sub_type}'
        if col_name not in df_encoded.columns:
            df_encoded[col_name] = 0
            
    # Ensure data/ directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save the consolidated tabular matrix
    output_path = "data/final_engineered_features.csv"
    df_encoded.to_csv(output_path, index=False)
    print(f"Consolidated features saved successfully to '{output_path}'.")
    print(df_encoded.head())

if __name__ == "__main__":
    main()
