import cv2
import os
import numpy as np
import pyaudio
import wave
import threading
import datetime
from tkinter import *
from tkinter import ttk
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from transformers import pipeline
import pickle
import warnings
from mtcnn import MTCNN

warnings.filterwarnings("ignore")

class FaceRecognizer:
    def __init__(self):
        # Create Output directory
        self.output_folder = "Output"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        # Initialize face detector using MTCNN
        self.face_detector = MTCNN()
        
        # Initialize OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize face recognition model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        self.model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
        self.model.eval()  # Set to evaluation mode
        
        # Initialize recognition related variables
        self.known_face_embeddings = []
        self.known_face_names = []
        
        # Recognition thresholds (lower values = stricter)
        self.strict_threshold = 0.65    # Higher threshold for basic recognition
        self.high_threshold = 0.80     # Higher threshold for "Very likely"
        
        # Initialize recording states
        self.is_recording_video = False
        self.video_writer = None
        self.current_label = "unnamed"
        
        # Create GUI controls
        self.setup_gui()
        
        # Load existing training data
        self.load_training_data()
        
    def load_training_data(self, data_folder="training_data"):
        """Load and process training images using Hugging Face models"""
        if not os.path.exists(data_folder):
            print(f"Creating {data_folder} directory...")
            os.makedirs(data_folder)
            return
        
        print("Loading training data...")
        embeddings = []
        names = []
        
        # Process each person's directory
        for person_name in os.listdir(data_folder):
            person_dir = os.path.join(data_folder, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            print(f"\nProcessing images for {person_name}...")
            
            # Process each image
            for image_file in os.listdir(person_dir):
                if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                image_path = os.path.join(person_dir, image_file)
                print(f"  Processing {image_file}...")
                
                try:
                    # Load and process image
                    img = cv2.imread(image_path)
                    detections = self.face_detector.detect_faces(img)
                    
                    if not detections:
                        print(f"  No faces found in {image_file}")
                        continue
                    
                    # Get largest face
                    largest_face = max(detections, key=lambda x: x['box'][2] * x['box'][3])
                    box = largest_face['box']
                    x, y, w, h = box  # box is already a list: [x, y, width, height]
                    
                    # Extract and process face
                    face_img = img[y:y+h, x:x+w]
                    embedding = self.get_face_embedding(face_img)
                    
                    embeddings.append(embedding)
                    names.append(person_name)
                    print(f"  Successfully processed {image_file}")
                    
                except Exception as e:
                    print(f"  Error processing {image_file}: {str(e)}")
        
        # Update recognition variables
        if embeddings:
            self.known_face_embeddings = embeddings
            self.known_face_names = names
            print(f"\nProcessed {len(names)} face embeddings")
        else:
            print("\nNo valid face embeddings found in training data")

    def get_face_embedding(self, face_image):
        """Extract face embedding using Hugging Face model"""
        # Preprocess image
        inputs = self.feature_extractor(face_image, return_tensors="pt")
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.logits.numpy()
        
        return embeddings[0]  # Return the embedding vector

    def process_frame(self, frame):
        """Process a single frame for face detection and recognition"""
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with stricter parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,        # Smaller scale factor for more accurate detection
            minNeighbors=8,         # More neighbors required (was 5)
            minSize=(50, 50),       # Larger minimum face size (was 30, 30)
            maxSize=(300, 300)      # Maximum face size to avoid false positives
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            
            # Get face embedding
            embedding = self.get_face_embedding(face_img)
            
            # Find the best match
            if self.known_face_embeddings:
                # Calculate similarities with known faces
                similarities = [np.dot(embedding, known_emb) / 
                              (np.linalg.norm(embedding) * np.linalg.norm(known_emb))
                              for known_emb in self.known_face_embeddings]
                
                best_match_idx = np.argmax(similarities)
                confidence = similarities[best_match_idx]
                
                # Stricter confidence thresholds
                if confidence > self.strict_threshold:
                    name = self.known_face_names[best_match_idx]
                    
                    if confidence > self.high_threshold:
                        display_name = f"Very likely {name}"
                        color = (0, 255, 0)  # Green
                    else:
                        display_name = f"Possible {name}"  # Changed from "Likely"
                        color = (0, 165, 255)  # Orange
                else:
                    display_name = "Unknown"
                    color = (0, 0, 255)  # Red
                
                label_text = f"{display_name} ({confidence:.2f})"
            else:
                label_text = "Unknown"
                color = (0, 0, 255)
            
            # Draw face rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-35), (x+w, y), color, cv2.FILLED)
            cv2.putText(frame, label_text, (x+6, y-6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame

    def run(self):
        try:
            # Initialize video capture
            video = cv2.VideoCapture(0)
            if not video.isOpened():
                raise RuntimeError("Error: Could not access webcam")
            
            print("Camera initialized. Press 'q' to quit.")
            
            while True:
                ret, frame = video.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Store current frame for screenshot feature
                self.current_frame = frame.copy()
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Record video if active
                if self.is_recording_video and self.video_writer:
                    # Add recording indicator
                    cv2.circle(processed_frame, (30, 30), 10, (0, 0, 255), -1)  # Red circle when recording
                    self.video_writer.write(processed_frame)
                
                cv2.imshow('Face Recognition', processed_frame)
                self.control_window.update()  # Update GUI
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        
        finally:
            video.release()
            cv2.destroyAllWindows()
            if self.is_recording_video:
                self.toggle_video_recording()
            self.control_window.destroy()

    def setup_gui(self):
        """Setup GUI controls in the main window"""
        self.control_window = Tk()
        self.control_window.title("Face Recognition Controls")
        
        # Create frame for controls
        control_frame = ttk.Frame(self.control_window)
        control_frame.pack(pady=5, padx=5)
        
        # Label entry
        ttk.Label(control_frame, text="Label:").pack(side=LEFT, padx=5)
        self.label_entry = ttk.Entry(control_frame, width=20)
        self.label_entry.insert(0, "unnamed")
        self.label_entry.pack(side=LEFT, padx=5)
        
        # Buttons
        self.screenshot_btn = ttk.Button(control_frame, text="Capture Image", command=self.take_screenshot)
        self.screenshot_btn.pack(side=LEFT, padx=5)
        
        self.video_btn = ttk.Button(control_frame, text="Start Recording", command=self.toggle_video_recording)
        self.video_btn.pack(side=LEFT, padx=5)
        
        # Update GUI periodically
        self.control_window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def take_screenshot(self):
        """Capture and save labeled screenshot to Output folder"""
        if hasattr(self, 'current_frame'):
            label = self.label_entry.get().strip() or "unnamed"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_folder, f"{label}_image_{timestamp}.jpg")
            cv2.imwrite(filename, self.current_frame)
            print(f"Screenshot saved as {filename}")

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio recording"""
        if self.is_recording_audio:
            self.audio_frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    def stop_audio_recording(self):
        """Stop and save audio recording to Output folder"""
        self.is_recording_audio = False
        self.audio_btn.config(text="Start Audio Recording")
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio.terminate()
            
            # Save audio file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_folder, f"audio_{timestamp}.wav")
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(b''.join(self.audio_frames))
            print(f"Audio saved as {filename}")

    def on_closing(self):
        """Handle window closing"""
        if self.is_recording_video:
            self.toggle_video_recording()
        self.control_window.destroy()

def main():
    recognizer = FaceRecognizer()
    recognizer.run()

if __name__ == "__main__":
    main()

