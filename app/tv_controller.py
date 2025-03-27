import pyautogui
import time
from typing import Optional

class TVController:
    def __init__(self):
        """Initialize the TV controller."""
        # Set pyautogui safety settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.5
        
        # Define key mappings for different TV controls
        self.key_mappings = {
            "thumbs_up": "volumeup",
            "thumbs_down": "volumedown",
            "left_swipe": "left",
            "right_swipe": "right",
            "stop": "space"
        }
        
        # Initialize last command time to prevent rapid-fire commands
        self.last_command_time = 0
        self.COMMAND_COOLDOWN = 1.0  # seconds
        
    def execute_command(self, gesture: str) -> bool:
        """
        Execute TV control command based on detected gesture.
        
        Args:
            gesture: Detected gesture string
            
        Returns:
            bool: True if command was executed, False otherwise
        """
        current_time = time.time()
        
        # Check if enough time has passed since last command
        if current_time - self.last_command_time < self.COMMAND_COOLDOWN:
            return False
            
        # Get the corresponding key for the gesture
        key = self.key_mappings.get(gesture)
        if not key:
            return False
            
        try:
            # Execute the command
            if gesture in ["left_swipe", "right_swipe"]:
                # For seek commands, press the key multiple times
                for _ in range(5):  # 5 key presses = 10 seconds
                    pyautogui.press(key)
                    time.sleep(0.1)
            else:
                pyautogui.press(key)
                
            # Update last command time
            self.last_command_time = current_time
            return True
            
        except Exception as e:
            print(f"Error executing command: {e}")
            return False
            
    def get_available_commands(self) -> dict:
        """Get a dictionary of available commands and their descriptions."""
        return {
            "thumbs_up": "Increase volume",
            "thumbs_down": "Decrease volume",
            "left_swipe": "Rewind 10 seconds",
            "right_swipe": "Forward 10 seconds",
            "stop": "Pause/Play content"
        } 