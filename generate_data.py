import os
import random
from datetime import datetime, timedelta
import pymysql
from dotenv import load_dotenv
from faker import Faker

# load .env variables
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "retail_analytics")

def create_db():
    connection = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME};")
        connection.commit()
        print(f"Database '{DB_NAME}' created or verified.")
    finally:
        connection.close()

def create_tables():
    connection = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    try:
        with connection.cursor() as cursor:
            # Drop tables if they exist to start fresh
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
            cursor.execute("DROP TABLE IF EXISTS support_tickets;")
            cursor.execute("DROP TABLE IF EXISTS activity_logs;")
            cursor.execute("DROP TABLE IF EXISTS users;")
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
            
            # Create users table
            cursor.execute("""
            CREATE TABLE users (
                user_id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) NOT NULL,
                subscription_type VARCHAR(50) NOT NULL,
                signup_date DATE NOT NULL,
                is_churned TINYINT(1) DEFAULT 0
            );
            """)
            
            # Create activity_logs table
            cursor.execute("""
            CREATE TABLE activity_logs (
                user_id VARCHAR(50) PRIMARY KEY,
                login_count_30d INT NOT NULL,
                total_spend DECIMAL(10, 2) NOT NULL,
                days_since_last_login INT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
            """)
            
            # Create support_tickets table
            cursor.execute("""
            CREATE TABLE support_tickets (
                ticket_id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(50) NOT NULL,
                ticket_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
            """)
        connection.commit()
        print("Database schemas created successfully.")
    finally:
        connection.close()

def generate_data(num_users=1000):
    fake = Faker()
    # Ensure reproducibility
    Faker.seed(42)
    random.seed(42)
    
    users = []
    activity_logs = []
    support_tickets = []
    
    # Text templates for sentiment correlation
    churn_angry_templates = [
        "I am extremely angry with your service! The app is super buggy and slow, it crashes constantly. Cancel my account immediately.",
        "Terrible support. My billing has been messed up twice and nobody responds to my issues. I want a full refund.",
        "Worst customer service experience ever. I am frustrated, disappointed, and leaving your platform.",
        "This product is useless and doesn't work as advertised. The support team is unhelpful. Close my account.",
        "I hate the new update. It has broken my workflow entirely. So annoying and painful to use.",
        "Very bad service. I'm paying premium prices for basic features that are broken. Refund me now.",
        "Disaster. Completely unusable interface, slow load times, and useless customer assistance.",
        "Nothing works properly. I've sent three emails with no reply. Cancel my subscription."
    ]
    
    retained_happy_templates = [
        "Love the app! Everything runs smoothly and the UI is beautiful. Thanks!",
        "Quick question: does the team plan to add auto-export features? Overall, great experience.",
        "Very satisfied with the quick support response. Problem solved in five minutes.",
        "Excellent product and customer service. Highly recommend it to my colleagues.",
        "How can I upgrade my billing profile to a yearly subscription? Thanks for the help.",
        "The interface works great. Standard query, is there dark mode support?",
        "No issues so far. The tool has been a lifesaver for our team's daily organization.",
        "Thanks for the support. The application is very helpful and user-friendly."
    ]
    
    for i in range(1, num_users + 1):
        user_id = f"USR_{i:05d}"
        name = fake.name()
        email = fake.unique.email()
        subscription_type = random.choice(["Free", "Basic", "Premium"])
        
        # signup date between 30 and 365 days ago
        signup_date = datetime.now().date() - timedelta(days=random.randint(30, 365))
        
        # ~20% churn rate
        is_churned = 1 if random.random() < 0.20 else 0
        
        if is_churned == 1:
            # Lower activity values
            login_count_30d = random.randint(0, 4)
            total_spend = round(random.uniform(5.00, 30.00), 2)
            days_since_last_login = random.randint(15, 30)
            
            # Angry ticket text
            ticket_text = random.choice(churn_angry_templates)
        else:
            # Higher activity values
            login_count_30d = random.randint(8, 30)
            total_spend = round(random.uniform(40.00, 250.00), 2)
            days_since_last_login = random.randint(0, 7)
            
            # Happy/neutral ticket text
            ticket_text = random.choice(retained_happy_templates)
            
        users.append((user_id, name, email, subscription_type, signup_date, is_churned))
        activity_logs.append((user_id, login_count_30d, total_spend, days_since_last_login))
        support_tickets.append((user_id, ticket_text))
        
    # Write to database
    connection = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    try:
        with connection.cursor() as cursor:
            # Batch insert users
            cursor.executemany(
                "INSERT INTO users (user_id, name, email, subscription_type, signup_date, is_churned) VALUES (%s, %s, %s, %s, %s, %s)",
                users
            )
            # Batch insert activity logs
            cursor.executemany(
                "INSERT INTO activity_logs (user_id, login_count_30d, total_spend, days_since_last_login) VALUES (%s, %s, %s, %s)",
                activity_logs
            )
            # Batch insert support tickets
            cursor.executemany(
                "INSERT INTO support_tickets (user_id, ticket_text) VALUES (%s, %s)",
                support_tickets
            )
        connection.commit()
        print(f"Successfully generated and inserted {num_users} records.")
    finally:
        connection.close()

if __name__ == "__main__":
    create_db()
    create_tables()
    generate_data()
