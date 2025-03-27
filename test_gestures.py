import cv2
import mediapipe as mp
import numpy as np
from app.gesture_detector import GestureDetector

def test_gesture_detection():
    # Initialize gesture detector
    detector = GestureDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    print("Gesture Recognition Test")
    print("=======================")
    print("Available gestures:")
    print("1. Thumbs Up - Increase Volume")
    print("2. Thumbs Down - Decrease Volume")
    print("3. Left Swipe - Rewind 10s")
    print("4. Right Swipe - Forward 10s")
    print("5. Stop - Pause/Play")
    print("\nPress 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Detect gesture
        gesture = detector.detect_gesture({'frame': frame})
        
        # Display the frame
        cv2.imshow('Gesture Recognition Test', frame)
        
        # Display detected gesture
        if gesture:
            print(f"\rDetected Gesture: {gesture}", end='')
        else:
            print("\rNo gesture detected", end='')
            
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_gesture_detection() 