import numpy as np
import librosa
import sounddevice as sd
from scipy.signal import find_peaks
import threading
import queue

class AudioAnalyzer:
    def __init__(self):
        # Audio parameters
        self.sample_rate = 16000
        self.frame_duration = 1  # seconds
        self.buffer_size = int(self.sample_rate * self.frame_duration)
        
        # Analysis parameters
        self.yawn_freq_range = (50, 300)  # Hz
        self.noise_threshold = 0.1
        self.yawn_threshold = 0.3
        
        # Initialize audio buffer
        self.audio_buffer = queue.Queue()
        self.is_recording = False
        self.recording_thread = None
        
    def start_recording(self):
        """Start recording audio in a separate thread"""
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()
        
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
            
    def _record_audio(self):
        """Record audio continuously"""
        def callback(indata, frames, time, status):
            if status:
                print(f"Audio recording error: {status}")
            self.audio_buffer.put(indata.copy())
        
        with sd.InputStream(channels=1,
                          samplerate=self.sample_rate,
                          blocksize=self.buffer_size,
                          callback=callback):
            while self.is_recording:
                sd.sleep(100)
                
    def detect_yawn(self, audio_frame):
        """Detect yawning in audio frame"""
        # Convert to mono if necessary
        if len(audio_frame.shape) > 1:
            audio_frame = np.mean(audio_frame, axis=1)
            
        # Calculate spectrogram
        D = librosa.stft(audio_frame)
        mag = np.abs(D)
        
        # Get frequencies
        freqs = librosa.fft_frequencies(sr=self.sample_rate)
        
        # Find peaks in yawn frequency range
        mask = (freqs >= self.yawn_freq_range[0]) & (freqs <= self.yawn_freq_range[1])
        yawn_magnitudes = np.mean(mag[mask], axis=1)
        peaks, _ = find_peaks(yawn_magnitudes, height=self.yawn_threshold)
        
        return len(peaks) > 0
    
    def calculate_noise_level(self, audio_frame):
        """Calculate background noise level"""
        # Convert to mono if necessary
        if len(audio_frame.shape) > 1:
            audio_frame = np.mean(audio_frame, axis=1)
            
        # Calculate RMS energy
        noise_level = np.sqrt(np.mean(audio_frame**2))
        
        return noise_level
    
    def analyze_attention(self):
        """Analyze attention based on audio features"""
        if self.audio_buffer.empty():
            return 1.0  # Default score when no audio
            
        audio_frame = self.audio_buffer.get()
        attention_score = 1.0
        
        # Check for yawning
        if self.detect_yawn(audio_frame):
            attention_score *= 0.7
            
        # Check noise level
        noise_level = self.calculate_noise_level(audio_frame)
        if noise_level > self.noise_threshold:
            attention_score *= (1 - min(noise_level, 0.5))
            
        return attention_score
    
    def __del__(self):
        """Cleanup"""
        self.stop_recording()
