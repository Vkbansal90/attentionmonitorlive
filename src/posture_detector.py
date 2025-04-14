import cv2
import numpy as np
import mediapipe as mp

class PostureDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Define ideal angles and thresholds with better calibration
        self.IDEAL_NECK_ANGLE = 75  # Slightly forward tilt is natural
        self.NECK_ANGLE_THRESHOLD = 20
        self.SHOULDER_ANGLE_THRESHOLD = 15
        
        # Store initial calibration
        self.calibrated = False
        self.calibration_frames = []
        self.calibration_count = 30  # Number of frames for calibration
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
                 np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle
    
    def calibrate_posture(self, landmarks, frame_shape):
        """Calibrate ideal posture angles based on initial frames"""
        if len(self.calibration_frames) < self.calibration_count:
            nose = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x * frame_shape[1],
                   landmarks[self.mp_pose.PoseLandmark.NOSE.value].y * frame_shape[0]]
            
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame_shape[1],
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame_shape[0]]
            
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame_shape[1],
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame_shape[0]]
            
            neck_angle = self.calculate_angle(
                [nose[0], 0],
                nose,
                [(left_shoulder[0] + right_shoulder[0])/2,
                 (left_shoulder[1] + right_shoulder[1])/2]
            )
            
            self.calibration_frames.append(neck_angle)
            return False
            
        if not self.calibrated:
            # Set ideal neck angle as average of calibration frames
            self.IDEAL_NECK_ANGLE = np.mean(self.calibration_frames)
            self.calibrated = True
        return True
    
    def detect_posture(self, frame):
        """Detect posture and calculate attention score"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        attention_score = 1.0
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Calibrate if needed
            if not self.calibrated:
                if not self.calibrate_posture(landmarks, frame.shape):
                    attention_score = 1.0  # During calibration, return perfect score
                    cv2.putText(frame, "Calibrating posture...", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    return attention_score, frame
            
            # Get relevant landmarks
            nose = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x * frame.shape[1],
                   landmarks[self.mp_pose.PoseLandmark.NOSE.value].y * frame.shape[0]]
            
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]]
            
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]]
            
            # Calculate neck angle (vertical alignment)
            neck_angle = self.calculate_angle(
                [nose[0], 0],
                nose,
                [(left_shoulder[0] + right_shoulder[0])/2,
                 (left_shoulder[1] + right_shoulder[1])/2]
            )
            
            # Calculate shoulder alignment
            shoulder_angle = self.calculate_angle(
                [0, left_shoulder[1]],
                left_shoulder,
                right_shoulder
            )
            
            # Calculate scores with smoother penalties
            neck_deviation = abs(neck_angle - self.IDEAL_NECK_ANGLE)
            if neck_deviation > self.NECK_ANGLE_THRESHOLD:
                penalty = (neck_deviation - self.NECK_ANGLE_THRESHOLD) / (90 - self.NECK_ANGLE_THRESHOLD)
                attention_score *= max(0.5, 1 - penalty)
            
            if shoulder_angle > self.SHOULDER_ANGLE_THRESHOLD:
                penalty = (shoulder_angle - self.SHOULDER_ANGLE_THRESHOLD) / (45 - self.SHOULDER_ANGLE_THRESHOLD)
                attention_score *= max(0.7, 1 - penalty)
            
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # Add posture information
            cv2.putText(frame, f"Neck Angle: {neck_angle:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Shoulder Angle: {shoulder_angle:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Posture Score: {attention_score:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return attention_score, frame
    
    def __del__(self):
        """Cleanup"""
        self.pose.close()
