import os
import sys
from setup_models import setup_models

# Ensure model files are present
if not setup_models():
    st.error("Failed to set up required model files. Please check the console for errors.")
    sys.exit(1)

# Set TensorFlow environment variables to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TF logging

import librosa
import sounddevice as sd
import sqlite3
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time
import atexit

from face_analyzer import FaceAnalyzer
from posture_detector import PostureDetector
from audio_analyzer import AudioAnalyzer
import utils

# Configure Streamlit
st.set_page_config(
    page_title="Student Attention Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'attention_scores' not in st.session_state:
    st.session_state.attention_scores = []
if 'running' not in st.session_state:
    st.session_state.running = False
if 'analyzers_initialized' not in st.session_state:
    st.session_state.analyzers_initialized = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'update_interval' not in st.session_state:
    st.session_state.update_interval = 5
if 'force_stop' not in st.session_state:
    st.session_state.force_stop = False

def cleanup_resources():
    """Clean up all resources properly"""
    try:
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        
        if hasattr(st.session_state, 'audio_analyzer'):
            st.session_state.audio_analyzer.stop_recording()
        
        cv2.destroyAllWindows()
        st.session_state.running = False
        st.session_state.force_stop = False
        st.session_state.analyzers_initialized = False
        
    except Exception as e:
        st.error(f"Error during cleanup: {str(e)}")

def force_quit():
    """Force quit the application"""
    try:
        st.session_state.force_stop = True
        st.session_state.running = False
        cleanup_resources()
        st.error("Application forcefully stopped. Please refresh the page to restart.")
        
        # Force close any remaining OpenCV windows
        cv2.destroyAllWindows()
        
        # Kill the current process
        pid = os.getpid()
        if os.name == 'nt':  # Windows
            os.system(f'taskkill /F /PID {pid}')
        else:  # Linux/Mac
            os.system(f'kill -9 {pid}')
            
    except Exception as e:
        st.error(f"Error during force quit: {str(e)}")
        sys.exit(1)

# Register cleanup function to run at exit
atexit.register(cleanup_resources)

def initialize_analyzers():
    """Initialize analyzers"""
    if not st.session_state.analyzers_initialized:
        st.session_state.face_analyzer = FaceAnalyzer()
        st.session_state.posture_detector = PostureDetector()
        st.session_state.audio_analyzer = AudioAnalyzer()
        st.session_state.analyzers_initialized = True

def process_frame(frame):
    """Process a single frame and return scores"""
    face_score, processed_frame = st.session_state.face_analyzer.analyze_attention(frame)
    posture_score, processed_frame = st.session_state.posture_detector.detect_posture(processed_frame)
    audio_score = st.session_state.audio_analyzer.analyze_attention()
    total_score = (0.4 * face_score) + (0.3 * posture_score) + (0.3 * audio_score)
    return face_score, posture_score, audio_score, total_score, processed_frame

def update_ui(video_placeholder, metrics_placeholder, chart_placeholder, frame_data):
    """Update UI components"""
    face_score, posture_score, audio_score, total_score, processed_frame = frame_data
    
    # Update video feed
    try:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying video: {str(e)}")
    
    # Update metrics
    metrics_placeholder.write(f"""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; padding: 10px; background-color: rgba(0,0,0,0.1); border-radius: 5px;">
            <div style="text-align: center; padding: 5px;">
                <div style="font-weight: bold;">Face Score</div>
                <div style="font-size: 1.2em;">{face_score:.2f}</div>
            </div>
            <div style="text-align: center; padding: 5px;">
                <div style="font-weight: bold;">Posture Score</div>
                <div style="font-size: 1.2em;">{posture_score:.2f}</div>
            </div>
            <div style="text-align: center; padding: 5px;">
                <div style="font-weight: bold;">Audio Score</div>
                <div style="font-size: 1.2em;">{audio_score:.2f}</div>
            </div>
            <div style="text-align: center; padding: 5px; background-color: rgba(0,0,0,0.05); border-radius: 3px;">
                <div style="font-weight: bold;">Total Score</div>
                <div style="font-size: 1.4em; color: #1f77b4;">{total_score:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Update graph only every few frames
    if st.session_state.frame_count % st.session_state.update_interval == 0:
        with chart_placeholder.container():
            recent_scores = st.session_state.attention_scores[-100:]
            df = pd.DataFrame({
                'Time': range(len(recent_scores)),
                'Attention Score': recent_scores
            })
            st.line_chart(df.set_index('Time'))

def draw_debug_info(frame, landmarks, attention_score, ear, pitch, yaw, visibility):
    """Draw debug information on frame"""
    # Draw face mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    
    # Create face landmarks protobuf message
    landmark_list = mp_face_mesh.FaceMesh().process(frame).multi_face_landmarks[0] if landmarks else None
    
    if landmark_list:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmark_list,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
        )
    
    # Draw metrics
    h, w = frame.shape[:2]
    cv2.putText(frame, f"Attention: {attention_score:.2f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Head Pose (P,Y): ({pitch:.1f}, {yaw:.1f})", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Face Visibility: {visibility:.2f}", (10, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def main():
    st.title("Student Attention Monitoring System")
    
    # Initialize analyzers
    initialize_analyzers()
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # Add force quit button in sidebar with warning color
    if st.sidebar.button("⚠️ Force Quit", help="Use this to forcefully stop the application", 
                        type="primary", key="force_quit"):
        force_quit()
    
    start_stop = st.sidebar.button(
        "Stop Monitoring" if st.session_state.running else "Start Monitoring"
    )
    
    # Main content layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Feed")
        video_placeholder = st.empty()
        
    with col2:
        st.subheader("Attention Metrics")
        metrics_placeholder = st.empty()
        
    # Graph section
    st.subheader("Attention Trends")
    chart_placeholder = st.empty()
    
    # Initialize database
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(os.path.dirname(current_dir), 'data', 'attention_logs.db')
    conn = utils.init_database(db_path)
    
    # Handle start/stop
    if start_stop:
        if not st.session_state.running:
            st.session_state.cap = cv2.VideoCapture(0)
            if not st.session_state.cap.isOpened():
                st.error("Failed to open camera!")
                return
            st.session_state.audio_analyzer.start_recording()
            st.session_state.frame_count = 0
        else:
            cleanup_resources()
        st.session_state.running = not st.session_state.running
    
    # Main processing loop
    try:
        while st.session_state.running and st.session_state.cap is not None:
            if st.session_state.force_stop:
                break
                
            ret, frame = st.session_state.cap.read()
            if ret:
                # Process frame
                frame_data = process_frame(frame)
                
                # Update UI
                update_ui(video_placeholder, metrics_placeholder, chart_placeholder, frame_data)
                
                # Store score
                _, _, _, total_score, _ = frame_data
                st.session_state.attention_scores.append(total_score)
                
                # Log to database (less frequently)
                if st.session_state.frame_count % 10 == 0:
                    face_score, posture_score, audio_score, total_score, _ = frame_data
                    utils.log_attention_scores(conn, face_score, posture_score, audio_score, total_score)
                
                st.session_state.frame_count += 1
                time.sleep(0.033)
            else:
                st.error("Failed to capture video frame")
                break
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        cleanup_resources()
    
    finally:
        cleanup_resources()
        if conn:
            conn.close()

if __name__ == "__main__":
    main()
