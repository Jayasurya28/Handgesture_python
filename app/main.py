from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import json
import base64
import numpy as np
import cv2
import shutil
from datetime import datetime
from .gesture_detector import GestureDetector
from .tv_controller import TVController

app = FastAPI(title="Smart TV Gesture Control")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize gesture detector and TV controller
gesture_detector = GestureDetector()
tv_controller = TVController()

# Create directories if they don't exist
GESTURES_DIR = Path("static/gestures")
VIDEOS_DIR = Path("static/videos")
GESTURES_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface."""
    # Get list of uploaded gestures
    gestures = []
    for gesture_dir in GESTURES_DIR.iterdir():
        if gesture_dir.is_dir():
            gesture_images = []
            for img_path in gesture_dir.glob("*.jpg"):
                gesture_images.append({
                    "name": img_path.name,
                    "path": f"/static/gestures/{gesture_dir.name}/{img_path.name}",
                    "uploaded_at": datetime.fromtimestamp(img_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
            gestures.append({
                "name": gesture_dir.name,
                "images": gesture_images
            })
    
    # Get list of uploaded videos
    videos = []
    for video_path in VIDEOS_DIR.glob("*.mp4"):
        videos.append({
            "name": video_path.name,
            "path": f"/static/videos/{video_path.name}",
            "uploaded_at": datetime.fromtimestamp(video_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Smart TV Gesture Control",
            "gestures": gestures,
            "videos": videos
        }
    )

@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file."""
    try:
        # Check file extension
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            return JSONResponse({
                "status": "error",
                "message": "Only MP4, AVI, and MOV files are allowed"
            }, status_code=400)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = VIDEOS_DIR / filename
        
        # Save the file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return JSONResponse({
            "status": "success",
            "message": "Video uploaded successfully",
            "file_path": f"/static/videos/{filename}"
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.delete("/api/delete-video/{filename}")
async def delete_video(filename: str):
    """Delete a video file."""
    try:
        file_path = VIDEOS_DIR / filename
        if file_path.exists():
            file_path.unlink()
            return JSONResponse({
                "status": "success",
                "message": "Video deleted successfully"
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": "File not found"
            }, status_code=404)
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/api/upload-gesture")
async def upload_gesture(
    file: UploadFile = File(...),
    gesture_name: str = Form(...),
    description: str = Form(None)
):
    """Upload a gesture image."""
    try:
        # Create gesture directory if it doesn't exist
        gesture_dir = GESTURES_DIR / gesture_name
        gesture_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = gesture_dir / filename
        
        # Save the file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return JSONResponse({
            "status": "success",
            "message": "Gesture image uploaded successfully",
            "file_path": f"/static/gestures/{gesture_name}/{filename}"
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.delete("/api/delete-gesture/{gesture_name}/{filename}")
async def delete_gesture(gesture_name: str, filename: str):
    """Delete a gesture image."""
    try:
        file_path = GESTURES_DIR / gesture_name / filename
        if file_path.exists():
            file_path.unlink()
            return JSONResponse({
                "status": "success",
                "message": "Gesture image deleted successfully"
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": "File not found"
            }, status_code=404)
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time gesture detection."""
    await websocket.accept()
    print("WebSocket connection established")
    
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            # Convert base64 image to numpy array
            img_data = base64.b64decode(frame_data['frame'].split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("Error: Could not decode frame")
                continue
                
            # Process gesture
            gesture = gesture_detector.detect_gesture({'frame': frame})
            
            if gesture:
                # Execute TV control command
                tv_controller.execute_command(gesture)
                
                # Send response back to client
                await websocket.send_json({
                    "gesture": gesture,
                    "status": "success"
                })
            else:
                await websocket.send_json({
                    "gesture": None,
                    "status": "no_gesture_detected"
                })
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        await websocket.close()

@app.get("/api/status")
async def get_status():
    """Get the current status of the system."""
    return {
        "status": "running",
        "gesture_detector": "active",
        "tv_controller": "active"
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 