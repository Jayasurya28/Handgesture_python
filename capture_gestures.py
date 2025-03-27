import cv2
import os
from datetime import datetime

def capture_gesture_images():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Create gestures directory if it doesn't exist
    gestures_dir = "static/gestures"
    os.makedirs(gestures_dir, exist_ok=True)
    
    print("Gesture Image Capture")
    print("=====================")
    print("Available gestures:")
    print("1. thumbs_up")
    print("2. thumbs_down")
    print("3. left_swipe")
    print("4. right_swipe")
    print("5. stop")
    print("\nPress 's' to save image")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Display the frame
        cv2.imshow('Gesture Image Capture', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Get gesture type from user
            print("\nEnter gesture type (1-5):")
            gesture_type = input().strip()
            
            # Map number to gesture name
            gesture_map = {
                '1': 'thumbs_up',
                '2': 'thumbs_down',
                '3': 'left_swipe',
                '4': 'right_swipe',
                '5': 'stop'
            }
            
            if gesture_type in gesture_map:
                gesture_name = gesture_map[gesture_type]
                gesture_dir = os.path.join(gestures_dir, gesture_name)
                os.makedirs(gesture_dir, exist_ok=True)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}.jpg"
                filepath = os.path.join(gesture_dir, filename)
                
                # Save the image
                cv2.imwrite(filepath, frame)
                print(f"Saved image to {filepath}")
            else:
                print("Invalid gesture type")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_gesture_images() 