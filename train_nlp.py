import os
import pickle
import numpy as np
import pandas as pd
import pymysql
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

# Load environment
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "retail_analytics")

def main():
    # Fetch tickets joined with churn label
    connection = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    
    query = """
    SELECT t.ticket_text, u.is_churned
    FROM support_tickets t
    JOIN users u ON t.user_id = u.user_id
    """
    
    try:
        df = pd.read_sql(query, connection)
    finally:
        connection.close()
        
    print(f"Fetched {len(df)} records for NLP training.")
    
    # NLP parameters
    vocab_size = 1000
    max_length = 50
    embedding_dim = 16
    
    # Initialize & fit Tokenizer
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['ticket_text'])
    
    # Convert and pad sequences
    sequences = tokenizer.texts_to_sequences(df['ticket_text'])
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    
    labels = np.array(df['is_churned'], dtype=np.float32)
    
    # Construct sequential model
    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print("Training TensorFlow NLP Model...")
    model.fit(padded, labels, epochs=8, batch_size=32, validation_split=0.2)
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Save model and tokenizer
    model.save("models/sentiment_model.h5")
    with open("models/tokenizer.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("NLP model saved to 'models/sentiment_model.h5'")
    print("Tokenizer saved to 'models/tokenizer.pickle'")

if __name__ == "__main__":
    main()
