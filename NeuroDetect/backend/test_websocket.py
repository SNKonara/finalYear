#!/usr/bin/env python3
"""
Test WebSocket Server Connection

Simple script to test the WebSocket server and verify it's sending fraud detection results.
"""

import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8765"
    
    print("🔌 Connecting to WebSocket server...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected successfully!\n")
            
            # Test 1: Ping
            print("Test 1: Ping")
            await websocket.send(json.dumps({"command": "ping"}))
            response = await websocket.recv()
            data = json.loads(response)
            print(f"  Response: {data}\n")
            
            # Test 2: Get model info
            print("Test 2: Get Model Info")
            await websocket.send(json.dumps({"command": "get_model_info"}))
            response = await websocket.recv()
            data = json.loads(response)
            print(f"  Detection enabled: {data.get('detection_enabled')}")
            if data.get('model_info'):
                print(f"  Model info: {data['model_info']}\n")
            
            # Test 3: Get status
            print("Test 3: Get Status")
            await websocket.send(json.dumps({"command": "get_status"}))
            response = await websocket.recv()
            data = json.loads(response)
            print(f"  Status: {json.dumps(data, indent=2)}\n")
            
            # Test 4: Start streaming and receive 5 records
            print("Test 4: Start Streaming (receiving 5 records)")
            await websocket.send(json.dumps({"command": "start_stream"}))
            
            # Receive and display 5 records
            for i in range(5):
                response = await websocket.recv()
                data = json.loads(response)
                
                # Skip status messages
                if 'status' in data or 'response' in data:
                    continue
                
                print(f"\n  📦 Record {i+1}:")
                print(f"     Transaction ID: {data.get('transaction_id', 'N/A')}")
                print(f"     Amount: ${data.get('amount', 0):.2f}")
                print(f"     Category: {data.get('category', 'N/A')}")
                
                # Check for fraud detection results
                if 'reconstruction_error' in data:
                    print(f"     🤖 Fraud Detection:")
                    print(f"        Reconstruction Error: {data['reconstruction_error']:.6f}")
                    print(f"        Threshold: {data.get('threshold', 0):.6f}")
                    print(f"        Is Fraud: {data.get('is_fraud', False)}")
                    print(f"        Fraud Probability: {data.get('fraud_probability', 0):.3f}")
                    print(f"        Risk Level: {data.get('risk_level', 'N/A')}")
                    print(f"        Processing Time: {data.get('processing_time_ms', 0):.2f} ms")
                else:
                    print(f"     ⚠️ No fraud detection results found!")
            
            # Stop streaming
            print("\n\nTest 5: Stop Streaming")
            await websocket.send(json.dumps({"command": "stop_stream"}))
            response = await websocket.recv()
            data = json.loads(response)
            print(f"  Response: {data}\n")
            
            # Get final stats
            print("Test 6: Get Final Statistics")
            await websocket.send(json.dumps({"command": "get_stats"}))
            response = await websocket.recv()
            data = json.loads(response)
            if 'stats' in data:
                print(f"  Statistics: {json.dumps(data['stats'], indent=2)}\n")
            
            print("✅ All tests completed successfully!")
            
    except websockets.exceptions.ConnectionRefusedError:
        print("❌ Connection refused! Make sure the server is running:")
        print("   python start_server.py")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("="*60)
    print("WEBSOCKET SERVER TEST")
    print("="*60)
    print()
    asyncio.run(test_websocket())
