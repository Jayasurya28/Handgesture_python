import cv2
import numpy as np
from typing import Optional
import time
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import torchvision.transforms as transforms
import os

class GestureDetector:
    def __init__(self):
        """Initialize the gesture detector with Hugging Face model."""
        print("Initializing gesture detector...")
        
        # Load fine-tuned model and processor
        model_path = "./fine_tuned_model"
        if os.path.exists(model_path):
            print("Loading fine-tuned model...")
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            self.model = AutoModelForImageClassification.from_pretrained(model_path)
        else:
            print("Fine-tuned model not found, loading base model...")
            model_name = "dima806/smart_tv_hand_gestures_image_detection"
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode
        
        # Setup image transforms to match mobile camera characteristics
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust for webcam lighting
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Define gesture mapping based on actual model labels
        self.gesture_mapping = {
            'Thumbs Up': "thumbs_up",
            'Up': "thumbs_up",
            'Thumbs Down': "thumbs_down",
            'Down': "thumbs_down",
            'Left Swipe': "left_swipe",
            'Right Swipe': "right_swipe",
            'Swipe': None,  # Ignore generic swipe to prevent false detections
            'Stop': "stop",
            'Stop Gesture': "stop"
        }
        
        # Add state tracking to prevent rapid-fire detections
        self.last_gesture = None
        self.last_detection_time = 0
        self.min_gesture_interval = 0.5  # Reduced to 0.5 seconds for better responsiveness
        
        print("Model initialized successfully!")
        print(f"Available gestures: {list(filter(None, set(self.gesture_mapping.values())))}")
        
    def detect_gesture(self, frame_data: dict) -> Optional[str]:
        """
        Detect gestures from frame data using Hugging Face model.
        
        Args:
            frame_data: Dictionary containing frame information
            
        Returns:
            str: Detected gesture or None if no gesture detected
        """
        try:
            current_time = time.time()
            
            # Check if enough time has passed since last detection
            if current_time - self.last_detection_time < self.min_gesture_interval:
                return None
            
            # Convert frame data to numpy array
            frame = np.array(frame_data['frame'], dtype=np.uint8)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Enhance image quality to match mobile camera characteristics
            enhanced_frame = self._enhance_frame(rgb_frame)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(enhanced_frame)
            
            # Apply transforms
            input_tensor = self.transform(pil_image)
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Get top prediction
                confidence, predicted_idx = torch.max(predictions, dim=1)
                confidence = confidence.item()
                predicted_label = self.model.config.id2label[predicted_idx.item()]
                
                # Print prediction details
                print(f"\nPredicted gesture: {predicted_label}")
                print(f"Confidence: {confidence:.3f}")
                
                # Get top 3 predictions for debugging
                top3_values, top3_indices = torch.topk(predictions, 3)
                print("\nTop 3 predictions:")
                for value, idx in zip(top3_values[0], top3_indices[0]):
                    label = self.model.config.id2label[idx.item()]
                    print(f"{label}: {value.item():.3f}")
                
                # Map the prediction to our gesture system
                if confidence > 0.4:  # Lowered threshold since we're enhancing the image
                    mapped_gesture = self.gesture_mapping.get(predicted_label)
                    if mapped_gesture:
                        # Check if this is a different gesture than last time
                        if mapped_gesture != self.last_gesture:
                            print(f"Mapped to: {mapped_gesture}")
                            self.last_gesture = mapped_gesture
                            self.last_detection_time = current_time
                            return mapped_gesture
                    
        except Exception as e:
            print(f"Error in gesture detection: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return None
        
    def _enhance_frame(self, frame):
        """Enhance webcam frame to match mobile camera image characteristics."""
        try:
            # Increase contrast
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            # Sharpen the image
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
        except Exception as e:
            print(f"Error enhancing frame: {e}")
            return frame
        
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'cap'):
            self.cap.release() 