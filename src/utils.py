import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import os

def init_database(db_path):
    """Initialize SQLite database with required tables"""
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    
    # Convert relative path to absolute path if necessary
    db_path = os.path.abspath(db_path)
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create attention logs table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS attention_logs
                 (timestamp TEXT, face_score REAL, posture_score REAL, 
                  audio_score REAL, total_score REAL)''')
    
    conn.commit()
    return conn

def log_attention_scores(conn, face_score, posture_score, audio_score, total_score):
    """Log attention scores to database"""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO attention_logs VALUES (?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), face_score, posture_score, audio_score, total_score))
    conn.commit()

def get_attention_history(conn, hours=1):
    """Get attention history for the specified time period"""
    query = """
        SELECT timestamp, face_score, posture_score, audio_score, total_score
        FROM attention_logs
        WHERE timestamp >= ?
    """
    
    start_time = (datetime.now() - timedelta(hours=hours)).isoformat()
    
    df = pd.read_sql_query(query, conn, params=(start_time,))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def plot_attention_trends(df):
    """Create attention trends plot"""
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['timestamp'], df['face_score'], label='Face Score')
    plt.plot(df['timestamp'], df['posture_score'], label='Posture Score')
    plt.plot(df['timestamp'], df['audio_score'], label='Audio Score')
    plt.plot(df['timestamp'], df['total_score'], label='Total Score', linewidth=2)
    
    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.title('Attention Scores Over Time')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()

def calculate_attention_statistics(df):
    """Calculate attention statistics"""
    stats = {
        'average_attention': df['total_score'].mean(),
        'min_attention': df['total_score'].min(),
        'max_attention': df['total_score'].max(),
        'attention_std': df['total_score'].std(),
        'low_attention_periods': (df['total_score'] < 0.5).sum(),
    }
    
    return stats

def format_time_duration(seconds):
    """Format time duration in a human-readable format"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"
