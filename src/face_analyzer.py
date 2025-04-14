import cv2
import numpy as np
import dlib
import mediapipe as mp
from typing import Tuple, List, Dict
import os
import time
from collections import deque

class FaceAnalyzer:
    def __init__(self):
        """Initialize the FaceAnalyzer with required components"""
        # Initialize face detection and landmark components
        self.detector = dlib.get_frontal_face_detector()
        
        # Load facial landmark predictor
        current_dir = os.path.dirname(os.path.abspath(__file__))
        predictor_path = os.path.join(os.path.dirname(current_dir), 'models', 'shape_predictor_68_face_landmarks.dat')
        
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(
                "shape_predictor_68_face_landmarks.dat not found. Please download it from: "
                "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
        
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Initialize history buffers
        self.history_size = 30
        self.eye_aspect_ratios = deque(maxlen=self.history_size)
        self.mouth_aspect_ratios = deque(maxlen=self.history_size)
        self.head_poses = deque(maxlen=self.history_size)
        self.attention_scores = deque(maxlen=self.history_size)
        
        # Define facial indices
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        self.MOUTH = list(range(48, 68))
        
        # Define thresholds
        self.EYE_AR_THRESH = 0.2
        self.MOUTH_AR_THRESH = 0.5
        self.HEAD_POSE_THRESH = 30.0
        self.HAND_DISTANCE_THRESH = 150.0
        
        print("Face Analyzer initialized successfully!")

    def analyze_attention(self, frame: np.ndarray) -> Tuple[float, np.ndarray]:
        """Analyze attention from video frame"""
        if frame is None or frame.size == 0:
            return 0.0, frame
            
        attention_score = 0.0
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray)
            
            if len(faces) > 0:
                face = faces[0]
                shape = self.predictor(gray, face)
                landmarks = self._shape_to_np(shape)
                
                # Calculate metrics
                ear = self._get_eye_aspect_ratio(landmarks)
                mar = self._get_mouth_aspect_ratio(landmarks)
                pose = self._get_head_pose(landmarks)
                hand_distance = self._get_hand_distance(frame, landmarks)
                
                # Update history
                self.eye_aspect_ratios.append(ear)
                self.mouth_aspect_ratios.append(mar)
                self.head_poses.append(pose)
                
                # Calculate attention score
                attention_score = self._calculate_attention_score(ear, mar, pose, hand_distance)
                self.attention_scores.append(attention_score)
                
                # Draw visualizations
                frame = self._draw_debug_info(frame, landmarks, ear, mar, pose, hand_distance, attention_score)
                
            else:
                cv2.putText(frame, "No Face Detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        except Exception as e:
            print(f"Error in analyze_attention: {str(e)}")
            
        return attention_score, frame

    def _shape_to_np(self, shape) -> np.ndarray:
        """Convert dlib shape to numpy array"""
        coords = np.zeros((68, 2), dtype=int)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def _get_eye_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """Calculate eye aspect ratio"""
        try:
            # Calculate for both eyes and take average
            left_ear = self._eye_ratio(landmarks[self.LEFT_EYE])
            right_ear = self._eye_ratio(landmarks[self.RIGHT_EYE])
            return (left_ear + right_ear) / 2
        except:
            return 1.0

    def _eye_ratio(self, eye_landmarks: np.ndarray) -> float:
        """Calculate aspect ratio for one eye"""
        try:
            # Vertical distances
            A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            # Horizontal distance
            C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            return (A + B) / (2.0 * C) if C > 0 else 1.0
        except:
            return 1.0

    def _get_mouth_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """Calculate mouth aspect ratio"""
        try:
            mouth = landmarks[self.MOUTH]
            # Vertical distances
            A = np.linalg.norm(mouth[2] - mouth[10])  # 50-58
            B = np.linalg.norm(mouth[4] - mouth[8])   # 52-56
            # Horizontal distance
            C = np.linalg.norm(mouth[0] - mouth[6])   # 48-54
            return (A + B) / (2.0 * C) if C > 0 else 0.0
        except:
            return 0.0

    def _get_head_pose(self, landmarks: np.ndarray) -> float:
        """Estimate head pose angle"""
        try:
            # Simplified head pose estimation using facial landmarks
            left_eye = np.mean(landmarks[self.LEFT_EYE], axis=0)
            right_eye = np.mean(landmarks[self.RIGHT_EYE], axis=0)
            
            # Calculate angle
            eye_angle = np.degrees(np.arctan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0]
            ))
            return abs(eye_angle)
        except:
            return 0.0

    def _get_hand_distance(self, frame: np.ndarray, landmarks: np.ndarray) -> float:
        """Calculate minimum distance between hands and face"""
        try:
            # Get face center
            face_center = np.mean(landmarks, axis=0)
            
            # Process hands
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            min_distance = float('inf')
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get wrist position
                    wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                    wrist_px = np.array([wrist.x * frame.shape[1], wrist.y * frame.shape[0]])
                    
                    # Calculate distance
                    distance = np.linalg.norm(wrist_px - face_center)
                    min_distance = min(min_distance, distance)
                    
                    # Draw hand landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Draw line to face if close
                    if distance < self.HAND_DISTANCE_THRESH:
                        cv2.line(frame, 
                                (int(wrist_px[0]), int(wrist_px[1])),
                                (int(face_center[0]), int(face_center[1])),
                                (0, 0, 255), 2)
            
            return min_distance
        except:
            return float('inf')

    def _calculate_attention_score(self, ear: float, mar: float, pose: float, hand_distance: float) -> float:
        """Calculate overall attention score"""
        try:
            # Initialize base score
            score = 1.0
            
            # Eye aspect ratio penalty
            if ear < self.EYE_AR_THRESH:
                score *= 0.7
            
            # Mouth aspect ratio penalty (yawning)
            if mar > self.MOUTH_AR_THRESH:
                score *= 0.8
            
            # Head pose penalty
            if pose > self.HEAD_POSE_THRESH:
                score *= 0.9
            
            # Hand distance penalty
            if hand_distance < self.HAND_DISTANCE_THRESH:
                score *= (hand_distance / self.HAND_DISTANCE_THRESH)
            
            # Smooth score using history
            if len(self.attention_scores) > 0:
                score = 0.7 * score + 0.3 * np.mean(self.attention_scores)
            
            return max(0.0, min(1.0, score))
        except:
            return 0.0

    def _draw_debug_info(self, frame: np.ndarray, landmarks: np.ndarray, ear: float, 
                        mar: float, pose: float, hand_distance: float, 
                        attention_score: float) -> np.ndarray:
        """Draw debug information on frame"""
        try:
            # Draw facial landmarks
            for i in range(68):
                cv2.circle(frame, (landmarks[i][0], landmarks[i][1]), 1, (0, 255, 0), -1)
            
            # Draw eyes
            for eye in [self.LEFT_EYE, self.RIGHT_EYE]:
                pts = landmarks[eye].astype(np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 255), 1)
            
            # Draw mouth
            mouth_pts = landmarks[self.MOUTH].astype(np.int32)
            cv2.polylines(frame, [mouth_pts], True, (0, 0, 255), 1)
            
            # Add metrics text
            h, w = frame.shape[:2]
            metrics = [
                f"Attention: {attention_score:.2f}",
                f"EAR: {ear:.2f}",
                f"MAR: {mar:.2f}",
                f"Pose: {pose:.1f}°",
                f"Hand Dist: {min(hand_distance, 999):.0f}px"
            ]
            
            for i, text in enumerate(metrics):
                y = 30 + (i * 30)
                cv2.putText(frame, text, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 255, 0) if i == 0 else (255, 255, 0), 2)
            
            # Add warning messages
            warnings = []
            if ear < self.EYE_AR_THRESH:
                warnings.append("Eyes Closed!")
            if mar > self.MOUTH_AR_THRESH:
                warnings.append("Yawning Detected!")
            if pose > self.HEAD_POSE_THRESH:
                warnings.append("Head Tilted!")
            if hand_distance < self.HAND_DISTANCE_THRESH:
                warnings.append("Hand Near Face!")
            
            for i, warning in enumerate(warnings):
                y = h - 30 - (i * 30)
                cv2.putText(frame, warning, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return frame
        except:
            return frame

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'hands'):
            self.hands.close()
