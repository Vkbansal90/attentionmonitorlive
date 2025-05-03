import cv2
import numpy as np
import dlib
from typing import Tuple, List, Dict, Any
import os
import mediapipe as mp
from fer import FER
from deepface import DeepFace
import torch
from transformers import Wav2vecForCTC, Wav2VecTokenizer
import whisper
import time
from collections import deque

class EnhancedAnalyzer:
    def __init__(self):
        """Initialize the EnhancedAnalyzer with advanced components"""
        # Initialize face detection and landmark models
        self.setup_face_models()
        
        # Initialize emotion detection models
        self.setup_emotion_models()
        
        # Initialize pose detection
        self.setup_pose_model()
        
        # Initialize audio models
        self.setup_audio_models()
        
        # Initialize history tracking
        self.setup_history()
        
        # Initialize parameters and thresholds
        self.setup_parameters()
        
        print("Enhanced Analyzer initialized successfully!")
    
    def setup_face_models(self):
        """Initialize face detection and landmark models"""
        # Dlib face detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        predictor_path = os.path.join(os.path.dirname(current_dir), 'models', 'shape_predictor_68_face_landmarks.dat')
        
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(
                "shape_predictor_68_face_landmarks.dat not found. Please download it from: "
                "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
        
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # MediaPipe Face Mesh for precise landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
    
    def setup_emotion_models(self):
        """Initialize emotion detection models"""
        self.fer_detector = FER(mtcnn=True)
        self.last_emotion_time = time.time()
        self.emotion_interval = 1.0  # Process emotions every second
    
    def setup_pose_model(self):
        """Initialize pose detection model"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
    
    def setup_audio_models(self):
        """Initialize audio processing models"""
        self.whisper_model = whisper.load_model("base")
        self.audio_buffer = []
        self.audio_sample_rate = 16000
    
    def setup_history(self):
        """Initialize history tracking"""
        self.history_size = 30
        self.attention_scores = deque(maxlen=self.history_size)
        
        
        
        self.eye_aspect_ratios = deque(maxlen=self.history_size)
        self.mouth_aspect_ratios = deque(maxlen=self.history_size)
        self.head_poses = deque(maxlen=self.history_size)
        self.gaze_scores = deque(maxlen=self.history_size)
        self.posture_scores = deque(maxlen=self.history_size)
    
    def setup_parameters(self):
        """Setup analysis parameters and weights"""
        # Component weights
        self.weights = {
            'face': 0.3,      # Basic face metrics (eye, mouth)
            'gaze': 0.2,      # Gaze direction
            'pose': 0.2,      # Body posture
            'hands': 0.15,    # Hand position
            'head': 0.15      # Head position
        }
        
        # Thresholds
        self.thresholds = {
            'EYE_AR_THRESH': 0.2,
            'MOUTH_AR_THRESH': 0.5,
            'HEAD_POSE_THRESH': 30.0,
            'HAND_DISTANCE_THRESH': 150.0,
            'GAZE_DEVIATION_THRESH': 0.3,
            'POSTURE_ANGLE_THRESH': 20.0
        }
    
    def analyze_face(self, frame, landmarks) -> Dict[str, float]:
        """Analyze facial features and attention indicators"""
        try:
            # Calculate eye aspect ratio and head pose
            ear = self.calculate_eye_aspect_ratio(landmarks)
            pitch, yaw = self.calculate_head_pose(landmarks)
            
            # Calculate face visibility
            visibility = self.detect_face_occlusion(landmarks)
            
            return {
                'ear': ear,
                'pitch': pitch,
                'yaw': yaw,
                'visibility': visibility
            }
        except Exception as e:
            print(f"Error in face analysis: {str(e)}")
            return {'ear': 1.0, 'pitch': 0.0, 'yaw': 0.0, 'visibility': 1.0}
    
    def analyze_emotions(self, frame) -> Dict[str, float]:
        """Analyze emotional state"""
        try:
            current_time = time.time()
            if current_time - self.last_emotion_time >= self.emotion_interval:
                # FER analysis
                fer_result = self.fer_detector.detect_emotions(frame)
                if fer_result:
                    emotions = fer_result[0]['emotions']
                    
                    # DeepFace analysis for verification
                    deepface_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    
                    # Combine results with weights
                    combined_emotions = {}
                    for emotion in emotions:
                        fer_score = emotions[emotion]
                        deepface_score = deepface_result[0]['emotion'][emotion] if emotion in deepface_result[0]['emotion'] else 0
                        combined_emotions[emotion] = 0.6 * fer_score + 0.4 * deepface_score
                    
                    self.last_emotion_time = current_time
                    return combined_emotions
                
        except Exception as e:
            print(f"Error in emotion analysis: {str(e)}")
        
        return {'neutral': 1.0}
    
    def analyze_pose(self, frame) -> Dict[str, float]:
        """Analyze body posture"""
        try:
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                # Extract key points for posture analysis
                landmarks = results.pose_landmarks.landmark
                
                # Calculate shoulder alignment
                left_shoulder = np.array([landmarks[11].x, landmarks[11].y])
                right_shoulder = np.array([landmarks[12].x, landmarks[12].y])
                shoulder_angle = np.abs(np.arctan2(right_shoulder[1] - left_shoulder[1],
                                                 right_shoulder[0] - left_shoulder[0]))
                
                # Calculate spine alignment
                nose = np.array([landmarks[0].x, landmarks[0].y])
                mid_hip = np.array([(landmarks[23].x + landmarks[24].x)/2,
                                  (landmarks[23].y + landmarks[24].y)/2])
                spine_angle = np.abs(np.arctan2(mid_hip[1] - nose[1],
                                              mid_hip[0] - nose[0]))
                
                return {
                    'shoulder_alignment': 1.0 - min(1.0, shoulder_angle / np.pi),
                    'spine_alignment': 1.0 - min(1.0, spine_angle / (np.pi/2))
                }
                
        except Exception as e:
            print(f"Error in pose analysis: {str(e)}")
        
        return {'shoulder_alignment': 1.0, 'spine_alignment': 1.0}
    
    def analyze_gaze(self, frame, face_landmarks) -> Dict[str, float]:
        """Analyze gaze direction using MediaPipe iris detection"""
        try:
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Get iris landmarks
                left_iris = np.mean([[landmarks[p].x, landmarks[p].y] 
                                   for p in range(468, 472)], axis=0)
                right_iris = np.mean([[landmarks[p].x, landmarks[p].y] 
                                    for p in range(473, 477)], axis=0)
                
                # Calculate gaze deviation from center
                left_eye_center = np.mean([[landmarks[p].x, landmarks[p].y] 
                                         for p in range(362, 386)], axis=0)
                right_eye_center = np.mean([[landmarks[p].x, landmarks[p].y] 
                                          for p in range(33, 157)], axis=0)
                
                gaze_deviation = np.mean([
                    np.linalg.norm(left_iris - left_eye_center),
                    np.linalg.norm(right_iris - right_eye_center)
                ])
                
                return {
                    'gaze_focus': 1.0 - min(1.0, gaze_deviation * 10)
                }
                
        except Exception as e:
            print(f"Error in gaze analysis: {str(e)}")
        
        return {'gaze_focus': 1.0}
    
    def process_audio(self, audio_chunk):
        """Process audio for attention indicators"""
        try:
            self.audio_buffer.extend(audio_chunk)
            if len(self.audio_buffer) >= self.audio_sample_rate * 2:  # Process 2 seconds of audio
                audio_data = np.array(self.audio_buffer[-self.audio_sample_rate*2:])
                
                # Use Whisper for speech detection
                result = self.whisper_model.transcribe(audio_data)
                speech_detected = len(result['text'].strip()) > 0
                
                # Clear buffer
                self.audio_buffer = []
                
                return {'speech_activity': 1.0 if speech_detected else 0.0}
                
        except Exception as e:
            print(f"Error in audio analysis: {str(e)}")
        
        return {'speech_activity': 0.0}
    
    def calculate_attention_score(self, 
                                face_metrics: Dict[str, float],
                                emotion_metrics: Dict[str, float],
                                pose_metrics: Dict[str, float],
                                gaze_metrics: Dict[str, float],
                                audio_metrics: Dict[str, float]) -> float:
        """Calculate overall attention score"""
        try:
            # Face attention score
            face_score = (0.4 * (1.0 if face_metrics['ear'] > 0.2 else 0.5) +
                         0.3 * (1.0 - min(1.0, max(abs(face_metrics['pitch']),
                                                  abs(face_metrics['yaw'])) / 90.0)) +
                         0.3 * face_metrics['visibility'])
            
            # Emotion attention score
            emotion_score = (emotion_metrics.get('neutral', 0) * 0.4 +
                           emotion_metrics.get('happy', 0) * 0.3 +
                           (1.0 - emotion_metrics.get('angry', 0)) * 0.15 +
                           (1.0 - emotion_metrics.get('sad', 0)) * 0.15)
            
            # Pose attention score
            pose_score = (pose_metrics['shoulder_alignment'] * 0.5 +
                         pose_metrics['spine_alignment'] * 0.5)
            
            # Gaze attention score
            gaze_score = gaze_metrics['gaze_focus']
            
            # Audio attention score
            audio_score = audio_metrics['speech_activity']
            
            # Calculate weighted average
            attention_score = (
                self.weights['face'] * face_score +
                self.weights['emotion'] * emotion_score +
                self.weights['pose'] * pose_score +
                self.weights['gaze'] * gaze_score +
                self.weights['audio'] * audio_score
            ) * 100  # Convert to percentage
            
            # Smooth score using history
            self.attention_scores.append(attention_score)
            if len(self.attention_scores) > self.history_size:
                self.attention_scores.pop(0)
            
            return np.mean(self.attention_scores)
            
        except Exception as e:
            print(f"Error calculating attention score: {str(e)}")
            return 70.0  # Default to moderate attention
    
    def analyze_attention(self, frame: np.ndarray) -> Tuple[float, np.ndarray]:
        """Enhanced attention analysis"""
        if frame is None or frame.size == 0:
            return 0.0, frame
        
        attention_score = 0.0
        metrics = {}
        
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with Face Mesh
            face_mesh_results = self.face_mesh.process(rgb_frame)
            
            if face_mesh_results.multi_face_landmarks:
                face_landmarks = face_mesh_results.multi_face_landmarks[0]
                
                # Basic face metrics
                metrics['face'] = self._analyze_face_metrics(frame, face_landmarks)
                
                # Emotion analysis
                current_time = time.time()
                if current_time - self.last_emotion_time >= self.emotion_interval:
                    # FER analysis
                    fer_result = self.fer_detector.detect_emotions(frame)
                    if fer_result:
                        emotions = fer_result[0]['emotions']
                        
                        # DeepFace analysis for verification
                        try:
                            deepface_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                            if isinstance(deepface_result, list):
                                deepface_result = deepface_result[0]
                            
                            # Combine results with weights
                            combined_emotions = {}
                            for emotion in emotions:
                                fer_score = emotions[emotion]
                                deepface_score = deepface_result['emotion'].get(emotion, 0) / 100  # DeepFace returns percentages
                                combined_emotions[emotion] = 0.6 * fer_score + 0.4 * deepface_score
                            
                            metrics['emotions'] = combined_emotions
                            self.last_emotion_time = current_time
                        except:
                            # If DeepFace fails, use FER results only
                            metrics['emotions'] = emotions
                            self.last_emotion_time = current_time
                
                # Gaze analysis
                metrics['gaze'] = self._analyze_gaze(frame, face_landmarks)
                
                # Pose analysis
                metrics['pose'] = self._analyze_pose(frame)
                
                # Hand analysis
                metrics['hands'] = self._analyze_hands(frame, face_landmarks)
                
                # Calculate attention score
                attention_score = self._calculate_attention_score(metrics)
                self.attention_scores.append(attention_score)
                
                # Draw debug information
                frame = self._draw_debug_info(frame, face_landmarks, metrics, attention_score)
            else:
                cv2.putText(frame, "No Face Detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        except Exception as e:
            print(f"Error in enhanced analysis: {str(e)}")
        
        return attention_score, frame
    
    def _analyze_face_metrics(self, frame: np.ndarray, face_landmarks) -> Dict[str, float]:
        """Analyze basic face metrics"""
        try:
            # Convert landmarks to numpy array
            points = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] 
                             for lm in face_landmarks.landmark])
            
            # Calculate eye aspect ratios
            left_ear = self._calculate_ear(points, [33, 160, 158, 133, 153, 144])
            right_ear = self._calculate_ear(points, [362, 385, 387, 263, 373, 380])
            avg_ear = (left_ear + right_ear) / 2
            
            # Calculate mouth aspect ratio
            mar = self._calculate_mar(points, [78, 81, 13, 311, 308, 402, 14, 178])
            
            # Calculate head pose
            pitch, yaw = self._calculate_head_pose(points)
            
            # Update histories
            self.eye_aspect_ratios.append(avg_ear)
            self.mouth_aspect_ratios.append(mar)
            self.head_poses.append(max(abs(pitch), abs(yaw)))
            
            return {
                'ear': avg_ear,
                'mar': mar,
                'pitch': pitch,
                'yaw': yaw
            }
        except:
            return {'ear': 1.0, 'mar': 0.0, 'pitch': 0.0, 'yaw': 0.0}
    
    def _analyze_gaze(self, frame, face_landmarks) -> Dict[str, float]:
        """Analyze gaze direction"""
        try:
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Get iris landmarks
                left_iris = np.mean([[landmarks[p].x, landmarks[p].y] 
                                   for p in range(468, 472)], axis=0)
                right_iris = np.mean([[landmarks[p].x, landmarks[p].y] 
                                    for p in range(473, 477)], axis=0)
                
                # Calculate gaze deviation from center
                left_eye_center = np.mean([[landmarks[p].x, landmarks[p].y] 
                                         for p in range(362, 386)], axis=0)
                right_eye_center = np.mean([[landmarks[p].x, landmarks[p].y] 
                                          for p in range(33, 157)], axis=0)
                
                gaze_deviation = np.mean([
                    np.linalg.norm(left_iris - left_eye_center),
                    np.linalg.norm(right_iris - right_eye_center)
                ])
                
                return {
                    'gaze_focus': 1.0 - min(1.0, gaze_deviation * 10)
                }
                
        except Exception as e:
            print(f"Error in gaze analysis: {str(e)}")
        
        return {'gaze_focus': 1.0}
    
    def _analyze_pose(self, frame) -> Dict[str, float]:
        """Analyze body posture"""
        try:
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                # Extract key points for posture analysis
                landmarks = results.pose_landmarks.landmark
                
                # Calculate shoulder alignment
                left_shoulder = np.array([landmarks[11].x, landmarks[11].y])
                right_shoulder = np.array([landmarks[12].x, landmarks[12].y])
                shoulder_angle = np.abs(np.arctan2(right_shoulder[1] - left_shoulder[1],
                                                 right_shoulder[0] - left_shoulder[0]))
                
                # Calculate spine alignment
                nose = np.array([landmarks[0].x, landmarks[0].y])
                mid_hip = np.array([(landmarks[23].x + landmarks[24].x)/2,
                                  (landmarks[23].y + landmarks[24].y)/2])
                spine_angle = np.abs(np.arctan2(mid_hip[1] - nose[1],
                                              mid_hip[0] - nose[0]))
                
                return {
                    'shoulder_alignment': 1.0 - min(1.0, shoulder_angle / np.pi),
                    'spine_alignment': 1.0 - min(1.0, spine_angle / (np.pi/2))
                }
                
        except Exception as e:
            print(f"Error in pose analysis: {str(e)}")
        
        return {'shoulder_alignment': 1.0, 'spine_alignment': 1.0}
    
    def _analyze_hands(self, frame: np.ndarray, face_landmarks) -> Dict[str, float]:
        """Analyze hand position relative to face"""
        try:
            # Get face center
            face_points = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] 
                                  for lm in face_landmarks.landmark])
            face_center = np.mean(face_points, axis=0)
            
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
                    
                    # Draw hand landmarks and connections
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Draw line to face if close
                    if distance < self.thresholds['HAND_DISTANCE_THRESH']:
                        cv2.line(frame, 
                                (int(wrist_px[0]), int(wrist_px[1])),
                                (int(face_center[0]), int(face_center[1])),
                                (0, 0, 255), 2)
            
            hand_score = min(1.0, min_distance / self.thresholds['HAND_DISTANCE_THRESH'])
            return {'hand_score': hand_score, 'min_distance': min_distance}
        except:
            return {'hand_score': 1.0, 'min_distance': float('inf')}
    
    def _calculate_attention_score(self, metrics: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall attention score"""
        try:
            # Face metrics score
            face_score = 1.0
            if 'face' in metrics:
                face = metrics['face']
                if face['ear'] < self.thresholds['EYE_AR_THRESH']:
                    face_score *= 0.7
                if face['mar'] > self.thresholds['MOUTH_AR_THRESH']:
                    face_score *= 0.8
                if max(abs(face['pitch']), abs(face['yaw'])) > self.thresholds['HEAD_POSE_THRESH']:
                    face_score *= 0.9
            
            # Gaze score
            gaze_score = metrics.get('gaze', {}).get('gaze_focus', 1.0)
            
            # Pose score
            pose_score = metrics.get('pose', {}).get('shoulder_alignment', 1.0)
            
            # Hand score
            hand_score = metrics.get('hands', {}).get('hand_score', 1.0)
            
            # Calculate weighted score
            attention_score = (
                self.weights['face'] * face_score +
                self.weights['gaze'] * gaze_score +
                self.weights['pose'] * pose_score +
                self.weights['hands'] * hand_score
            )
            
            # Smooth score using history
            if len(self.attention_scores) > 0:
                attention_score = 0.7 * attention_score + 0.3 * np.mean(self.attention_scores)
            
            return max(0.0, min(1.0, attention_score))
        except:
            return 0.0
    
    def _calculate_ear(self, points: np.ndarray, indices: List[int]) -> float:
        """Calculate eye aspect ratio"""
        try:
            p = points[indices]
            # Vertical distances
            A = np.linalg.norm(p[1] - p[5])
            B = np.linalg.norm(p[2] - p[4])
            # Horizontal distance
            C = np.linalg.norm(p[0] - p[3])
            return (A + B) / (2.0 * C) if C > 0 else 1.0
        except:
            return 1.0
    
    def _calculate_mar(self, points: np.ndarray, indices: List[int]) -> float:
        """Calculate mouth aspect ratio"""
        try:
            p = points[indices]
            # Vertical distances
            A = np.linalg.norm(p[1] - p[5])
            B = np.linalg.norm(p[2] - p[4])
            # Horizontal distance
            C = np.linalg.norm(p[0] - p[3])
            return (A + B) / (2.0 * C) if C > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_head_pose(self, points: np.ndarray) -> Tuple[float, float]:
        """Calculate head pose angles"""
        try:
            # Use eye positions to estimate head rotation
            left_eye = np.mean(points[33:37], axis=0)
            right_eye = np.mean(points[362:366], axis=0)
            
            # Calculate angles
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            
            # Estimate pitch and yaw
            yaw = np.degrees(np.arctan2(dy, dx))
            pitch = np.degrees(np.arctan2(np.linalg.norm([dx, dy]), 100))
            
            return pitch, yaw
        except:
            return 0.0, 0.0
    
    def _draw_debug_info(self, frame: np.ndarray, face_landmarks, metrics: Dict[str, Dict[str, float]], 
                        attention_score: float) -> np.ndarray:
        """Draw debug information on frame"""
        try:
            h, w = frame.shape[:2]
            
            # Draw face mesh
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_TESSELATION,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1)
            )
            
            # Draw metrics
            metrics_text = [
                f"Attention: {attention_score:.2f}",
                f"EAR: {metrics['face']['ear']:.2f}",
                f"MAR: {metrics['face']['mar']:.2f}",
                f"Gaze: {metrics['gaze']['gaze_focus']:.2f}",
                f"Posture: {metrics['pose']['shoulder_alignment']:.2f}",
                f"Hand Dist: {metrics['hands']['min_distance']:.0f}px"
            ]
            
            # Add emotion information if available
            if 'emotions' in metrics:
                # Get the dominant emotion
                dominant_emotion = max(metrics['emotions'].items(), key=lambda x: x[1])
                metrics_text.append(f"Emotion: {dominant_emotion[0]} ({dominant_emotion[1]:.2f})")
                
                # Add top 3 emotions with their scores
                sorted_emotions = sorted(metrics['emotions'].items(), key=lambda x: x[1], reverse=True)[:3]
                for emotion, score in sorted_emotions:
                    if score > 0.1:  # Only show emotions with significant scores
                        metrics_text.append(f"  {emotion}: {score:.2f}")
            
            for i, text in enumerate(metrics_text):
                y = 30 + (i * 30)
                # Use different colors for different types of information
                if i == 0:  # Attention score
                    color = (0, 255, 0)
                elif text.startswith("Emotion:"):  # Main emotion
                    color = (255, 165, 0)  # Orange
                elif text.startswith("  "):  # Detailed emotions
                    color = (255, 192, 203)  # Pink
                else:  # Other metrics
                    color = (255, 255, 0)
                cv2.putText(frame, text, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw warnings
            warnings = []
            if metrics['face']['ear'] < self.thresholds['EYE_AR_THRESH']:
                warnings.append("Eyes Closed!")
            if metrics['face']['mar'] > self.thresholds['MOUTH_AR_THRESH']:
                warnings.append("Yawning Detected!")
            if max(abs(metrics['face']['pitch']), abs(metrics['face']['yaw'])) > self.thresholds['HEAD_POSE_THRESH']:
                warnings.append("Head Tilted!")
            if metrics['hands']['min_distance'] < self.thresholds['HAND_DISTANCE_THRESH']:
                warnings.append("Hand Near Face!")
            if metrics['gaze']['gaze_focus'] < 0.5:
                warnings.append("Look at Screen!")
            if metrics['pose']['shoulder_alignment'] < 0.5:
                warnings.append("Fix Posture!")
            
            # Add emotion-based warnings
            if 'emotions' in metrics:
                if metrics['emotions'].get('angry', 0) > 0.5:
                    warnings.append("High Anger Detected!")
                if metrics['emotions'].get('sad', 0) > 0.5:
                    warnings.append("Sadness Detected!")
                if metrics['emotions'].get('disgust', 0) > 0.5:
                    warnings.append("Disgust Detected!")
            
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
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        if hasattr(self, 'pose'):
            self.pose.close() 