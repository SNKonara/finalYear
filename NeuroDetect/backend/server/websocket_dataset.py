# websocket_dataset_server.py
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import re
import os
from datetime import datetime, timedelta
import time

class DatasetStreamServer:
    def __init__(self, dataset_path, stream_speed=1.0):
        """
        Server that streams your fraud detection dataset
        
        Args:
            dataset_path: Path to your CSV dataset
            stream_speed: Records per second (default: 1 record/sec)
        """
        self.dataset = self.load_dataset(dataset_path)
        self.stream_speed = stream_speed
        # Track streaming per-client and background tasks so stop commands are handled
        self.streaming_clients = {}
        self.stream_tasks = {}
        self.client_totals = {}
        self.clients = set()
        # Detector subprocess (real_time.py)
        self.detector_proc = None
        print(f"✅ Loaded dataset with {len(self.dataset)} records")
    
    def load_dataset(self, dataset_path):
        """Load and prepare your dataset"""
        print(f"📂 Loading dataset from: {dataset_path}")
        
        # Load your dataset
        df = pd.read_csv(dataset_path)
        
        # Clean column names (remove spaces, lowercase)
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

        # Try to standardize an amount column to 'amount'
        amount_cols = [c for c in df.columns if re.search(r'(^|_)amount$|(^|_)amt$|transaction_amt|transaction_amount', c)]
        if amount_cols:
            col = amount_cols[0]
            try:
                df['amount'] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            except Exception:
                df['amount'] = 0.0
        elif 'amount' not in df.columns:
            df['amount'] = 0.0
        
        # Add transaction_id 
        if 'transaction_id' not in df.columns and 'trans_num' in df.columns:
            df['transaction_id'] = df['trans_num']
        elif 'transaction_id' not in df.columns:
            df['transaction_id'] = [f'TXN_{i:06d}' for i in range(len(df))]
        
        # Add category if not present
        if 'category' not in df.columns and 'category' in df.columns.str.lower():
            category_col = [c for c in df.columns if 'category' in c.lower()][0]
            df['category'] = df[category_col]
        elif 'category' not in df.columns:
            df['category'] = "Unknown"
        
        # Add gender if not present
        if 'gender' not in df.columns:
            df['gender'] = "Unknown"
        
        # Convert to list of dictionaries
        records = df.to_dict('records')
        
        print(f"📊 Dataset columns: {df.columns.tolist()}")
        print(f"📝 First record sample: {records[0] if records else 'No data'}")
        
        return records
    
    def prepare_record(self, record, index):
        """Prepare a record for streaming"""
        # Ensure all values are JSON serializable
        prepared = {}
        for key, value in record.items():
            if isinstance(value, (np.integer, np.int64)):
                prepared[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                prepared[key] = float(value)
            elif isinstance(value, np.ndarray):
                prepared[key] = value.tolist()
            elif pd.isna(value):
                prepared[key] = None
            else:
                prepared[key] = value
        
        # Add metadata
        prepared['stream_index'] = index
        prepared['stream_timestamp'] = datetime.now().isoformat()
        prepared['is_fraud'] = bool(record.get('is_fraud', 0)) if 'is_fraud' in record else False

        # Ensure amount is present and numeric
        try:
            amt = prepared.get('amount', 0) if prepared.get('amount', 0) is not None else 0
            prepared['amount'] = float(amt)
        except Exception:
            prepared['amount'] = 0.0

        return prepared
    
    async def stream_to_client(self, websocket):
        """Stream dataset records to a connected client"""
        print("🚀 Starting data stream to client")

        index = 0
        total = len(self.dataset)

        # initialize running total for this client session
        if websocket not in self.client_totals:
            self.client_totals[websocket] = 0.0

        # stream while this client is marked as streaming
        while self.streaming_clients.get(websocket, False):
            try:
                # Get and prepare record
                record = self.dataset[index]
                prepared_record = self.prepare_record(record, index)

                # update running total for this client
                try:
                    self.client_totals[websocket] += float(prepared_record.get('amount', 0) or 0)
                except Exception:
                    pass

                # include running total in payload
                payload = dict(prepared_record)
                payload['stream_total_amount'] = self.client_totals.get(websocket, 0.0)

                # Send record
                await websocket.send(json.dumps(payload))

                print(
                    f"📤 Sent record {index + 1}/{total}: "
                    f"{prepared_record.get('transaction_id', 'Unknown')}"
                )

                # Move to next record
                index += 1

                # Always restart at end
                if index >= total:
                    print("🔄 End of dataset reached — restarting from beginning")
                    index = 0

                # Control streaming speed
                await asyncio.sleep(1.0 / self.stream_speed)

            except websockets.exceptions.ConnectionClosed:
                print("⚠️ Client disconnected during stream")
                break

            except Exception as e:
                print(f"❌ Error sending record: {e}")
                index += 1  # skip bad record

        print("🛑 Data stream stopped for client")
        # If the websocket disconnected, clear its running total
        if websocket not in self.clients:
            self.client_totals.pop(websocket, None)
    
    async def handler(self, websocket):
        """Handle client commands (control channel)"""
        client_id = id(websocket)
        self.clients.add(websocket)

        print(f"✅ Client connected (ID: {client_id})")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    command = data.get("command", "").lower()

                    if command == "ping":
                        await websocket.send(json.dumps({
                            "response": "pong",
                            "timestamp": datetime.now().isoformat()
                        }))

                    elif command == "start_stream":
                        if not self.streaming_clients.get(websocket, False):
                            print(f"▶️ Start stream for client {client_id}")
                            # mark this websocket as streaming and create a background task
                            self.streaming_clients[websocket] = True
                            task = asyncio.create_task(self.stream_to_client(websocket))
                            self.stream_tasks[websocket] = task
                            await websocket.send(json.dumps({
                                "status": "streaming_started",
                                "streaming": True,
                                "stream_speed": self.stream_speed
                            }))
                        else:
                            await websocket.send(json.dumps({
                                "status": "already_streaming",
                                "streaming": True
                            }))

                    elif command == "stop_stream":
                        print(f"⏹️ Stop stream for client {client_id}")
                        # mark this websocket as not streaming and cancel background task if present
                        self.streaming_clients[websocket] = False
                        task = self.stream_tasks.pop(websocket, None)
                        if task and not task.done():
                            try:
                                task.cancel()
                            except Exception:
                                pass
                        # include the final total for the session
                        await websocket.send(json.dumps({
                            "status": "stopped",
                            "streaming": False,
                            "stream_total_amount": self.client_totals.get(websocket, 0.0)
                        }))
                        # reset client total for next session
                        self.client_totals[websocket] = 0.0

                    elif command == "get_status":
                        await websocket.send(json.dumps({
                            "streaming": self.streaming_clients.get(websocket, False),
                            "total_records": len(self.dataset),
                            "stream_speed": self.stream_speed,
                            "active_clients": len(self.clients),
                            "stream_total_amount": self.client_totals.get(websocket, 0.0)
                        }))

                    elif command == "set_speed":
                        speed = data.get("speed")
                        if isinstance(speed, (int, float)) and speed > 0:
                            self.stream_speed = speed
                            print(f"⚙️ Stream speed changed to {speed} records/sec")
                            await websocket.send(json.dumps({
                                "status": "speed_updated",
                                "speed": self.stream_speed,
                                "stream_total_amount": self.client_totals.get(websocket, 0.0)
                            }))
                        else:
                            await websocket.send(json.dumps({
                                "error": "Invalid speed value"
                            }))

                    elif command == "start_detector":
                        model = (data.get("model") or "autoencoder").lower()
                        # Only autoencoder supported by real_time.py currently
                        if model != 'autoencoder':
                            await websocket.send(json.dumps({
                                "error": f"Model '{model}' not supported by detector"
                            }))
                        else:
                            if self.detector_proc and self.detector_proc.returncode is None:
                                await websocket.send(json.dumps({
                                    "status": "detector_already_running"
                                }))
                            else:
                                # Spawn the detector subprocess
                                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'real_time.py'))
                                try:
                                    self.detector_proc = await asyncio.create_subprocess_exec(
                                        'python', script_path,
                                        stdout=asyncio.subprocess.PIPE,
                                        stderr=asyncio.subprocess.PIPE
                                    )
                                    await websocket.send(json.dumps({
                                        "status": "detector_started",
                                        "model": model,
                                        "pid": self.detector_proc.pid
                                    }))
                                except Exception as e:
                                    await websocket.send(json.dumps({
                                        "error": f"Failed to start detector: {e}"
                                    }))

                    elif command == "stop_detector":
                        if self.detector_proc and self.detector_proc.returncode is None:
                            try:
                                self.detector_proc.terminate()
                            except Exception:
                                try:
                                    self.detector_proc.kill()
                                except Exception:
                                    pass
                            await websocket.send(json.dumps({
                                "status": "detector_stopped"
                            }))
                            self.detector_proc = None
                        else:
                            await websocket.send(json.dumps({
                                "status": "no_detector_running"
                            }))

                    else:
                        await websocket.send(json.dumps({
                            "error": f"Unknown command: {command}"
                        }))

                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "error": "Invalid JSON"
                    }))

        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 Client {client_id} disconnected")

        finally:
            # Ensure this client is cleaned up
            try:
                self.streaming_clients[websocket] = False
            except Exception:
                pass
            task = self.stream_tasks.pop(websocket, None)
            if task and not task.done():
                try:
                    task.cancel()
                except Exception:
                    pass
            # remove running total and client record
            self.client_totals.pop(websocket, None)
            self.clients.discard(websocket)
            print(f"👤 Client removed (Active clients: {len(self.clients)})")

async def main():
    # Configuration
    DATASET_PATH = "C:/NoGit/backend/dataset/fraudTrain.csv"  # Change to your dataset path
    HOST = "localhost"
    PORT = 8765
    STREAM_SPEED = 1.0  # Records per second
    
    # Create and start server
    server = DatasetStreamServer(DATASET_PATH, STREAM_SPEED)
    
    print(f"\n{'='*60}")
    print("FRAUD DETECTION DATASET STREAM SERVER")
    print("="*60)
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"WebSocket URL: ws://{HOST}:{PORT}")
    print(f"Stream Speed: {STREAM_SPEED} records/second")
    print(f"Dataset: {DATASET_PATH}")
    print("="*60)
    print("\n📡 Server starting...")
    print("Commands available to clients:")
    print("  - ping: Test connection")
    print("  - start_stream: Begin streaming dataset")
    print("  - stop_stream: Stop streaming")
    print("  - get_status: Get server status")
    print("  - set_speed: Change streaming speed")
    print("\n⌨️ Press Ctrl+C to stop the server")
    print("="*60)
    
    try:
        # Start WebSocket server
        async with websockets.serve(server.handler, HOST, PORT):
            await asyncio.Future()  # Run forever
            
    except KeyboardInterrupt:
        print("\n\n🛑 Server shutting down...")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    asyncio.run(main())