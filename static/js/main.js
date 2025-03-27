// WebSocket connection
let ws = null;
let video = null;
let canvas = null;
let stream = null;
let videoPlayer = null;

// Initialize the application
async function init() {
    try {
        // Get video elements
        video = document.getElementById('videoElement');
        videoPlayer = document.getElementById('videoPlayer');
        
        // Create canvas for frame capture
        canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 480;
        
        // Get webcam stream
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480,
                facingMode: 'user'
            }
        });
        
        // Set video source
        video.srcObject = stream;
        
        // Initialize WebSocket connection
        connectWebSocket();
        
        // Start frame capture loop
        startFrameCapture();
        
    } catch (error) {
        console.error('Error initializing:', error);
        updateStatus('Error: ' + error.message, false);
    }
}

// Connect to WebSocket server
function connectWebSocket() {
    // Determine the WebSocket protocol based on the current page protocol
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    console.log('Connecting to WebSocket:', wsUrl);
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        updateStatus('Connected', true);
    };
    
    ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        updateStatus('Disconnected', false);
        // Attempt to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus('Connection Error', false);
    };
    
    ws.onmessage = (event) => {
        try {
            const response = JSON.parse(event.data);
            updateGestureDisplay(response.gesture);
            handleGesture(response.gesture);
        } catch (error) {
            console.error('Error processing message:', error);
        }
    };
}

// Handle detected gestures
function handleGesture(gesture) {
    if (!gesture || !videoPlayer) {
        console.log('No gesture detected or video player not initialized');
        return;
    }
    
    console.log('Handling gesture:', gesture);
    console.log('Current volume:', videoPlayer.volume);
    
    // Add debouncing to prevent rapid-fire gestures
    if (window.lastGestureTime && Date.now() - window.lastGestureTime < 500) {
        console.log('Gesture ignored - too soon after last gesture');
        return;
    }
    window.lastGestureTime = Date.now();
    
    switch(gesture) {
        case 'thumbs_up':
            videoPlayer.volume = Math.min(1, videoPlayer.volume + 0.1);
            console.log('New volume after thumbs up:', videoPlayer.volume);
            break;
        case 'thumbs_down':
            videoPlayer.volume = Math.max(0, videoPlayer.volume - 0.1);
            console.log('New volume after thumbs down:', videoPlayer.volume);
            // Force volume update
            videoPlayer.muted = false;
            break;
        case 'left_swipe':
            videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime - 10);
            console.log('Rewinding 10 seconds');
            break;
        case 'right_swipe':
            videoPlayer.currentTime = Math.min(videoPlayer.duration, videoPlayer.currentTime + 10);
            console.log('Forwarding 10 seconds');
            break;
        case 'stop':
            if (videoPlayer.paused) {
                videoPlayer.play();
                console.log('Playing video');
            } else {
                videoPlayer.pause();
                console.log('Pausing video');
            }
            break;
    }
}

// Start frame capture loop
function startFrameCapture() {
    setInterval(() => {
        if (video && canvas && ws && ws.readyState === WebSocket.OPEN) {
            try {
                // Draw current video frame to canvas
                canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
                
                // Convert canvas to base64
                const frameData = {
                    frame: canvas.toDataURL('image/jpeg', 0.8)
                };
                
                // Send frame data to server
                ws.send(JSON.stringify(frameData));
            } catch (error) {
                console.error('Error capturing frame:', error);
            }
        }
    }, 100); // Capture every 100ms
}

// Update gesture display
function updateGestureDisplay(gesture) {
    const gestureText = document.getElementById('gestureText');
    console.log('Updating gesture display:', gesture);
    if (gesture) {
        gestureText.textContent = gesture.replace('_', ' ').toUpperCase();
        gestureText.style.color = '#28a745';
        // Add visual feedback for volume changes and seek operations
        if (gesture === 'thumbs_up' || gesture === 'thumbs_down') {
            const volumeIndicator = document.createElement('div');
            volumeIndicator.className = 'volume-indicator';
            volumeIndicator.textContent = `Volume: ${Math.round(videoPlayer.volume * 100)}%`;
            document.querySelector('.gesture-info').appendChild(volumeIndicator);
            setTimeout(() => volumeIndicator.remove(), 1000);
        } else if (gesture === 'left_swipe' || gesture === 'right_swipe') {
            const seekIndicator = document.createElement('div');
            seekIndicator.className = 'seek-indicator';
            seekIndicator.textContent = gesture === 'left_swipe' ? 'Rewinding 10s' : 'Forwarding 10s';
            document.querySelector('.gesture-info').appendChild(seekIndicator);
            setTimeout(() => seekIndicator.remove(), 1000);
        }
    } else {
        gestureText.textContent = 'None';
        gestureText.style.color = '#6c757d';
    }
}

// Update status display
function updateStatus(message, isActive) {
    const statusText = document.getElementById('statusText');
    const statusIndicator = document.querySelector('.status-indicator');
    
    statusText.textContent = message;
    statusIndicator.className = `status-indicator ${isActive ? 'status-active' : 'status-inactive'}`;
}

// Clean up resources when page is unloaded
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    if (ws) {
        ws.close();
    }
});

// Initialize when the page loads
window.addEventListener('load', init); 