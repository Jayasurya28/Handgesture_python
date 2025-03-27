# Smart TV Gesture Control System

A real-time gesture recognition system for controlling smart TVs using hand gestures. This system uses computer vision and machine learning to detect specific hand gestures and translate them into TV control commands.

## Features

- Real-time gesture recognition using webcam
- Five supported gestures:
  - Thumbs Up: Increase volume
  - Thumbs Down: Decrease volume
  - Left Swipe: Rewind 10 seconds
  - Right Swipe: Forward 10 seconds
  - Stop: Pause content
- FastAPI backend for efficient processing
- Web interface for monitoring and control

## Setup

1. Install Python 3.10
2. Create a virtual environment:
   ```bash
   python -m venv env
   .\env\Scripts\activate  # Windows
   source env/bin/activate  # Linux/Mac
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── gesture_detector.py  # Gesture recognition logic
│   ├── tv_controller.py     # TV control functionality
│   └── utils/
│       └── helpers.py       # Utility functions
├── static/
│   └── js/
│       └── main.js         # Frontend JavaScript
├── templates/
│   └── index.html          # Web interface
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## API Endpoints

- `GET /`: Web interface
- `POST /api/gesture`: Process gesture detection
- `GET /api/status`: Get system status

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License