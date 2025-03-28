import cv2
import os
import time
from datetime import datetime

def capture_training_images():
    # Define gestures
    gestures = ['thumbs_up', 'thumbs_down', 'left_swipe', 'right_swipe', 'stop']
    
    # Create directory for training data
    base_dir = "training_data"
    os.makedirs(base_dir, exist_ok=True)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    try:
        for gesture in gestures:
            # Create directory for this gesture
            gesture_dir = os.path.join(base_dir, gesture)
            os.makedirs(gesture_dir, exist_ok=True)
            
            print(f"\nCapturing images for {gesture}")
            print("Press 'c' to capture an image")
            print("Press 'n' to move to next gesture")
            print("Press 'q' to quit")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Show current frame
                cv2.imshow('Capture Training Data', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{gesture}_{timestamp}.jpg"
                    filepath = os.path.join(gesture_dir, filename)
                    
                    # Save the image
                    cv2.imwrite(filepath, frame)
                    print(f"Saved: {filepath}")
                    
                    # Wait a bit to avoid duplicate captures
                    time.sleep(0.5)
                    
                elif key == ord('n'):
                    print(f"Moving to next gesture...")
                    break
                    
                elif key == ord('q'):
                    print("Quitting...")
                    return
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting Training Data Collection")
    print("================================")
    print("Instructions:")
    print("1. Make sure you have good lighting")
    print("2. Keep your hand clearly visible")
    print("3. Vary your hand position slightly for each capture")
    print("4. Try to capture at least 50 images per gesture")
    print("5. Press 'c' to capture, 'n' for next gesture, 'q' to quit")
    print("\nStarting capture in 3 seconds...")
    time.sleep(3)
    
    capture_training_images() 