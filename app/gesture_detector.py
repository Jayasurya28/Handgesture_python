import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple

class GestureDetector:
    def __init__(self):
        """Initialize the gesture detector with MediaPipe."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,  # Reduced from 0.7 to make it more sensitive
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Define gesture thresholds
        self.SWIPE_THRESHOLD = 0.15  # Reduced from 0.3 to make swipe detection more sensitive
        self.THUMB_ANGLE_THRESHOLD = 20  # Reduced from 30 to make it more sensitive
        self.THUMB_DOWN_THRESHOLD = 0.05  # Reduced from 0.1 to make it more sensitive
        
    def detect_gesture(self, frame_data: dict) -> Optional[str]:
        """
        Detect gestures from frame data.
        
        Args:
            frame_data: Dictionary containing frame information
            
        Returns:
            str: Detected gesture or None if no gesture detected
        """
        # Convert frame data to numpy array
        frame = np.array(frame_data['frame'], dtype=np.uint8)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None
            
        # Get the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Detect gestures
        gesture = self._analyze_gesture(hand_landmarks)
        return gesture
        
    def _analyze_gesture(self, hand_landmarks) -> Optional[str]:
        """
        Analyze hand landmarks to determine the gesture.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            str: Detected gesture or None
        """
        # Get thumb tip and IP joint
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        
        # Get index finger tip and MCP joint
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        # Calculate thumb angle and vertical difference
        thumb_angle = self._calculate_angle(thumb_tip, thumb_ip)
        thumb_vertical_diff = thumb_tip.y - thumb_ip.y
        
        # Detect thumbs up/down with improved logic
        if abs(thumb_angle) > self.THUMB_ANGLE_THRESHOLD:
            if thumb_vertical_diff < -self.THUMB_DOWN_THRESHOLD:  # Thumb is pointing up
                return "thumbs_up"
            elif thumb_vertical_diff > self.THUMB_DOWN_THRESHOLD:  # Thumb is pointing down
                return "thumbs_down"
                
        # Detect swipe gestures with improved logic
        horizontal_diff = index_tip.x - index_mcp.x
        if abs(horizontal_diff) > self.SWIPE_THRESHOLD:
            if horizontal_diff < 0:  # Moving left
                return "left_swipe"
            else:  # Moving right
                return "right_swipe"
                
        # Detect stop gesture (all fingers extended)
        if self._is_stop_gesture(hand_landmarks):
            return "stop"
            
        return None
        
    def _calculate_angle(self, point1, point2) -> float:
        """Calculate angle between two points."""
        return np.degrees(np.arctan2(point2.y - point1.y, point2.x - point1.x))
        
    def _is_swipe(self, tip, mcp) -> bool:
        """Check if a swipe gesture is detected."""
        return abs(tip.x - mcp.x) > self.SWIPE_THRESHOLD
        
    def _is_stop_gesture(self, hand_landmarks) -> bool:
        """Check if all fingers are extended (stop gesture)."""
        # Get finger tips and MCP joints
        fingers = [
            (self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_MCP),
            (self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
            (self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_MCP),
            (self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_MCP)
        ]
        
        # Check if all fingers are extended
        for tip, mcp in fingers:
            tip_point = hand_landmarks.landmark[tip]
            mcp_point = hand_landmarks.landmark[mcp]
            if tip_point.y > mcp_point.y:  # Finger is not extended
                return False
        return True 