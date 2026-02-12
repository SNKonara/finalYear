#!/usr/bin/env python3
"""
Start the Unified Fraud Detection WebSocket Server

This script starts the WebSocket server that:
1. Streams transactions from the dataset
2. Performs real-time fraud detection using both Autoencoder and LSTM models
3. Broadcasts results to all connected clients (including the frontend)
4. Supports model switching on-the-fly

Usage:
    python start_server.py

The server runs on port 8765 and supports both Autoencoder and LSTM models.
Clients can switch between models using the 'set_model' command.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio

if __name__ == "__main__":
    print("\n🚀 Starting Unified Fraud Detection Server...")
    print("   Loading both Autoencoder and LSTM models...")
    print("   Server will run on ws://localhost:8765")
    
    # Import and run the unified server
    from server.websocket_unified import main
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Server shutting down...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
