# Real-Time Fraud Detection System - Connection Guide

This guide explains how to connect and use the backend fraud detection server with the frontend dashboard.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                         │
│                   aereal.tsx Dashboard                      │
│              ws://localhost:8765 connection                 │
└────────────────────────┬────────────────────────────────────┘
                         │ WebSocket
                         │ (bidirectional)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           Backend WebSocket Server (Python)                 │
│         server/websocket_dataset.py (Port 8765)            │
│                                                             │
│  ┌──────────────┐       ┌─────────────────┐               │
│  │   Dataset    │──────▶│  Fraud Model    │               │
│  │   Streamer   │       │  (Autoencoder)  │               │
│  └──────────────┘       └─────────────────┘               │
│                                                             │
│  Sends: Transaction data + Fraud detection results         │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Start the Backend Server

**Option A: Using the start script (Recommended)**
```bash
cd backend
python start_server.py
```

**Option B: With custom options**
```bash
# Custom port and speed
python start_server.py --port 8765 --speed 2.0

# Disable fraud detection (raw data only)
python start_server.py --no-detection

# Custom dataset
python start_server.py --dataset path/to/your/dataset.csv
```

**Option C: Direct execution**
```bash
cd backend/server
python websocket_dataset.py
```

### 2. Start the Frontend

```bash
# In another terminal, from the project root
npm run dev
```

### 3. Open the Dashboard

Navigate to the frontend in your browser (typically `http://localhost:5173`) and go to the **Real-Time Detection** page.

## Features

### Backend Server Features

✅ **Real-time Fraud Detection**
- Loads the trained Autoencoder model
- Processes transactions in real-time
- Calculates reconstruction errors
- Determines fraud probability and risk levels

✅ **WebSocket Communication**
- Bidirectional communication with clients
- Supports multiple simultaneous connections
- Command-based control system

✅ **Statistics Tracking**
- Total transactions processed
- Fraud detection count
- Average processing time
- Throughput (transactions per second)
- Error tracking

### Frontend Dashboard Features

✅ **Live Monitoring**
- Real-time transaction stream
- Live fraud detection results
- Interactive charts and visualizations

✅ **Stream Controls**
- Start/Stop streaming
- Adjust stream speed
- Reset statistics

✅ **Filtering & Search**
- Filter by risk level
- Show fraud only
- Search transactions

✅ **Model Information**
- View model architecture
- See detection threshold
- Monitor model performance

## WebSocket Commands

The backend server supports the following commands:

### Client → Server Commands

```javascript
// Test connection
ws.send(JSON.stringify({ command: "ping" }))

// Get model information
ws.send(JSON.stringify({ command: "get_model_info" }))

// Get statistics
ws.send(JSON.stringify({ command: "get_stats" }))

// Start streaming
ws.send(JSON.stringify({ command: "start_stream" }))

// Stop streaming
ws.send(JSON.stringify({ command: "stop_stream" }))

// Get server status
ws.send(JSON.stringify({ command: "get_status" }))

// Change stream speed
ws.send(JSON.stringify({ 
  command: "set_speed", 
  speed: 2.0  // records per second
}))
```

### Server → Client Messages

**Fraud Detection Result:**
```json
{
  "transaction_id": "TXN_000001",
  "transaction_data": {
    "amt": 123.45,
    "category": "gas_transport",
    "gender": "M",
    ...
  },
  "reconstruction_error": 0.000523,
  "threshold": 0.000450,
  "is_fraud": true,
  "fraud_probability": 0.856,
  "risk_level": "High",
  "processing_time_ms": 2.34,
  "timestamp": "2026-02-10T12:34:56.789"
}
```

**Model Information:**
```json
{
  "model_info": {
    "input_dim": 45,
    "architecture": "128-64-16",
    "threshold": 0.00045,
    "device": "cpu",
    "expected_features": 45,
    "feature_names": ["amt", "category_gas_transport", ...]
  }
}
```

**Statistics:**
```json
{
  "stats": {
    "total_processed": 1250,
    "fraud_detected": 45,
    "preprocessing_errors": 0,
    "dataset_loops": 2,
    "avg_processing_time": 2.45,
    "throughput_tps": 0.95,
    "fraud_rate": 3.6,
    "success_rate": 100.0
  }
}
```

## Configuration

### Backend Configuration

Edit `backend/server/websocket_dataset.py` or use command-line arguments:

```python
HOST = "localhost"
PORT = 8765
STREAM_SPEED = 1.0  # Records per second
ENABLE_DETECTION = True  # Enable/disable fraud detection
```

### Frontend Configuration

Edit `src/pages/aereal.tsx`:

```typescript
// WebSocket connection URL
const ws = new WebSocket('ws://localhost:8765');

// Max records to display
const [maxRecords, setMaxRecords] = useState(50);

// Stream speed (adjustable in UI)
const [streamSpeed, setStreamSpeed] = useState(1);
```

## Troubleshooting

### Connection Issues

**Problem:** "WebSocket connection failed"
- ✅ Check if backend server is running
- ✅ Verify port 8765 is not in use
- ✅ Check firewall settings

**Problem:** "Model not loading"
- ✅ Ensure `saved_models/autoencoder.pth` exists
- ✅ Verify `saved_models/scaler.pkl` exists
- ✅ Check `saved_models/threshold.json` exists
- ✅ Run model training first if files are missing

**Problem:** "Dataset not found"
- ✅ Check `backend/dataset/fraudTrain.csv` exists
- ✅ Use `--dataset` flag to specify custom path

### Performance Issues

**Slow Processing:**
- Lower the stream speed: `--speed 0.5`
- Check CPU/memory usage
- Verify model is loaded correctly

**Frontend Not Updating:**
- Check browser console for errors
- Verify WebSocket connection is active
- Refresh the page to reconnect

## File Structure

```
backend/
├── start_server.py              # Main server startup script
├── server/
│   └── websocket_dataset.py     # WebSocket server with fraud detection
├── AEmodel/
│   ├── model.py                 # Autoencoder model definition
│   ├── preprocessor.py          # Data preprocessing
│   └── ...
├── dataset/
│   ├── fraudTrain.csv           # Training dataset
│   └── fraudTest.csv            # Test dataset
└── saved_models/
    ├── autoencoder.pth          # Trained model weights
    ├── scaler.pkl               # Preprocessing scaler
    └── threshold.json           # Fraud detection threshold

src/
└── pages/
    ├── aereal.tsx               # Real-time detection dashboard
    └── streaming.tsx            # Alternative streaming view
```

## Development Notes

### Adding New Features

**Backend:**
1. Add new commands to `handler()` method in `websocket_dataset.py`
2. Implement command logic
3. Send response to client

**Frontend:**
1. Add command sender in `sendCommand()` function
2. Handle response in `handleWebSocketMessage()`
3. Update UI state accordingly

### Testing

**Test Backend Server:**
```python
# Simple test client
import asyncio
import websockets
import json

async def test():
    async with websockets.connect('ws://localhost:8765') as ws:
        # Test ping
        await ws.send(json.dumps({"command": "ping"}))
        response = await ws.recv()
        print(response)
        
        # Get model info
        await ws.send(json.dumps({"command": "get_model_info"}))
        response = await ws.recv()
        print(response)

asyncio.run(test())
```

## Support

For issues or questions:
1. Check the console logs (both backend and frontend)
2. Verify all dependencies are installed
3. Ensure model is trained and saved
4. Check dataset path and format

## Next Steps

- ✅ Backend and frontend are now connected
- ✅ Real-time fraud detection is working
- ✅ Dashboard displays live results
- 🔄 Test with your own data
- 🔄 Adjust detection threshold if needed
- 🔄 Customize dashboard visualizations
