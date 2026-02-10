#!/usr/bin/env python3
"""
Start the Fraud Detection WebSocket Server

This script starts the WebSocket server that:
1. Streams transactions from the dataset
2. Performs real-time fraud detection using the Autoencoder model
3. Broadcasts results to all connected clients (including the frontend)

Usage:
    python start_server.py [--port 8765] [--speed 1.0] [--no-detection]

Options:
    --port PORT         WebSocket server port (default: 8765)
    --speed SPEED       Stream speed in records per second (default: 1.0)
    --no-detection      Disable fraud detection (stream raw data only)
    --dataset PATH      Path to dataset CSV file
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import asyncio
from server.websocket_dataset import DatasetStreamServer, main

def parse_args():
    parser = argparse.ArgumentParser(
        description='Start Fraud Detection WebSocket Server',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=8765,
        help='WebSocket server port (default: 8765)'
    )
    
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Stream speed in records per second (default: 1.0)'
    )
    
    parser.add_argument(
        '--no-detection',
        action='store_true',
        help='Disable fraud detection (stream raw data only)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to dataset CSV file'
    )
    
    return parser.parse_args()

async def start_server(port=8765, speed=1.0, enable_detection=True, dataset_path=None):
    """Start the WebSocket server with specified configuration"""
    
    # Determine dataset path
    if dataset_path is None:
        dataset_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "dataset", 
            "fraudTrain.csv"
        )
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("   Please specify the correct path using --dataset")
        return
    
    # Create server
    server = DatasetStreamServer(dataset_path, speed, enable_detection)
    
    print(f"\n{'='*60}")
    print("FRAUD DETECTION WEBSOCKET SERVER")
    print("="*60)
    print(f"Host: localhost")
    print(f"Port: {port}")
    print(f"WebSocket URL: ws://localhost:{port}")
    print(f"Stream Speed: {speed} records/second")
    print(f"Dataset: {dataset_path}")
    print(f"Fraud Detection: {'✅ Enabled' if server.enable_detection else '❌ Disabled'}")
    print("="*60)
    print("\n📡 Server starting...")
    print("Commands available to clients:")
    print("  - ping: Test connection")
    print("  - get_model_info: Get fraud detection model information")
    print("  - get_stats: Get detection statistics") 
    print("  - start_stream: Begin streaming dataset with fraud detection")
    print("  - stop_stream: Stop streaming")
    print("  - get_status: Get server status")
    print("  - set_speed: Change streaming speed")
    print("\n⌨️ Press Ctrl+C to stop the server")
    print("="*60)
    
    try:
        # Import websockets here to catch import errors
        import websockets
        
        # Start WebSocket server
        async with websockets.serve(server.handler, "localhost", port):
            await asyncio.Future()  # Run forever
            
    except KeyboardInterrupt:
        print("\n\n🛑 Server shutting down...")
    except ImportError:
        print("\n❌ Error: 'websockets' package not found!")
        print("   Install it using: pip install websockets")
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    args = parse_args()
    
    print("\n🚀 Starting Fraud Detection Server...")
    print(f"   Configuration:")
    print(f"   - Port: {args.port}")
    print(f"   - Speed: {args.speed} records/sec")
    print(f"   - Detection: {'Disabled' if args.no_detection else 'Enabled'}")
    if args.dataset:
        print(f"   - Dataset: {args.dataset}")
    
    asyncio.run(start_server(
        port=args.port,
        speed=args.speed,
        enable_detection=not args.no_detection,
        dataset_path=args.dataset
    ))
