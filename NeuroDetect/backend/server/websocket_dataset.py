# websocket_dataset_server.py
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import re
import os
import sys
from datetime import datetime, timedelta
import time
import torch

# Add parent directory to path to import model components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AEmodel.preprocessor import DataPreprocessor
from AEmodel.model import FraudAutoencoder
from database.mongodb import get_mongodb_instance

class DatasetStreamServer:
    def __init__(self, dataset_path, stream_speed=1.0, enable_detection=True):
        """
        Server that streams your fraud detection dataset
        
        Args:
            dataset_path: Path to your CSV dataset
            stream_speed: Records per second (default: 1 record/sec)
            enable_detection: Whether to enable fraud detection (default: True)
        """
        self.dataset = self.load_dataset(dataset_path)
        self.stream_speed = stream_speed
        self.enable_detection = enable_detection
        
        # Track streaming per-client and background tasks so stop commands are handled
        self.streaming_clients = {}
        self.stream_tasks = {}
        self.client_totals = {}
        self.clients = set()
        
        # Fraud detection components
        self.model = None
        self.preprocessor = None
        self.threshold = None
        self.device = None
        self.model_info = {}
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'fraud_detected': 0,
            'preprocessing_errors': 0,
            'dataset_loops': 0,
            'avg_processing_time': 0,
            'start_time': time.time(),
            'risk_distribution': {'Low': 0, 'Medium': 0, 'High': 0},
            'prediction_history': []  # Store recent predictions for analysis
        }
        
        # MongoDB connection
        self.db = None
        
        print(f"✅ Loaded dataset with {len(self.dataset)} records")
        
        # Connect to MongoDB
        self.connect_database()
        
        # Load fraud detection model if enabled
        if self.enable_detection:
            self.load_fraud_detection_model()
    
    def connect_database(self):
        """Initialize MongoDB connection"""
        try:
            print("\n🗄️  Connecting to MongoDB...")
            self.db = get_mongodb_instance()
            if self.db.connected:
                print("✅ MongoDB connected successfully")
            else:
                print("⚠️ MongoDB connection failed - results will not be saved to database")
        except Exception as e:
            print(f"⚠️ MongoDB connection error: {e}")
            print("   Continuing without database - results will not be saved")
            self.db = None
    
    def load_fraud_detection_model(self):
        """Load the fraud detection model and components"""
        print("\n🤖 Loading Fraud Detection Model...")
        
        try:
            # Path: backend/server/websocket_dataset.py -> backend/server -> backend -> NeuroDetect -> finalYear -> saved_models
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'saved_models')
            
            # Load model checkpoint
            checkpoint_path = os.path.join(model_dir, 'autoencoder.pth')
            checkpoint = torch.load(checkpoint_path)
            input_dim = checkpoint['input_dim']
            
            # Initialize model
            self.model = FraudAutoencoder(
                input_dim=input_dim,
                hidden_dim1=128,
                hidden_dim2=64,
                latent_dim=16,
                dropout_rate=0.000287
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load preprocessor
            self.preprocessor = DataPreprocessor()
            preprocessor_path = os.path.join(model_dir, 'scaler.pkl')
            self.preprocessor.load_preprocessor(preprocessor_path)
            
            # Load threshold
            threshold_path = os.path.join(model_dir, 'threshold.json')
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)
                self.threshold = threshold_data['threshold']
            
            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            
            # Load features.json for detailed feature information
            features_path = os.path.join(model_dir, 'features.json')
            feature_info = {}
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    feature_info = json.load(f)
            
            # Store model info
            self.model_info = {
                'input_dim': input_dim,
                'architecture': '128-64-16',
                'threshold': float(self.threshold),
                'device': str(self.device),
                'expected_features': input_dim,
                'feature_names': checkpoint.get('feature_names', feature_info.get('feature_names', [])),
                'num_features': feature_info.get('num_features', input_dim),
                'top_categories': feature_info.get('top_categories', []),
                'category_columns': feature_info.get('category_columns', [])
            }
            
            print(f"✅ Fraud Detection Model Loaded Successfully")
            print(f"   Input dimensions: {input_dim}")
            print(f"   Threshold: {self.threshold:.6f}")
            print(f"   Device: {self.device}")
            print(f"   Features loaded: {len(self.model_info['feature_names'])}")
            
        except Exception as e:
            print(f"❌ Failed to load fraud detection model: {e}")
            print("   Detection will be disabled")
            self.enable_detection = False
            import traceback
            traceback.print_exc()
    
    def detect_fraud(self, transaction_data):
        """Run fraud detection on a transaction"""
        if not self.enable_detection or self.model is None:
            return None
        
        start_time = time.time()
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([transaction_data])
            
            # Preprocess
            X_processed, _, _ = self.preprocessor.preprocess(df, is_training=False)
            
            # Verify dimensions
            if X_processed.shape[1] != self.model_info['expected_features']:
                self.stats['preprocessing_errors'] += 1
                return None
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_processed).to(self.device)
            
            # Get reconstruction error
            with torch.no_grad():
                error_value = self.model.get_reconstruction_error(X_tensor)[0]
            
            # Determine if fraud
            is_fraud = error_value > self.threshold
            fraud_prob = min(error_value / self.threshold, 1.0)
            
            # Determine risk level
            if fraud_prob < 0.3:
                risk = "Low"
            elif fraud_prob < 0.7:
                risk = "Medium"
            else:
                risk = "High"
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Update statistics
            self.stats['total_processed'] += 1
            if is_fraud:
                self.stats['fraud_detected'] += 1
            
            # Update risk distribution
            self.stats['risk_distribution'][risk] = self.stats['risk_distribution'].get(risk, 0) + 1
            
            # Update average processing time
            if self.stats['total_processed'] > 0:
                total_time = (self.stats['avg_processing_time'] * (self.stats['total_processed'] - 1)) + processing_time
                self.stats['avg_processing_time'] = total_time / self.stats['total_processed']
            
            # Create result
            result = {
                'reconstruction_error': float(error_value),
                'threshold': float(self.threshold),
                'is_fraud': bool(is_fraud),
                'fraud_probability': float(fraud_prob),
                'risk_level': risk,
                'processing_time_ms': round(processing_time, 2),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store prediction in history (keep last 100)
            self.stats['prediction_history'].append({
                'error': float(error_value),
                'is_fraud': bool(is_fraud),
                'risk': risk,
                'timestamp': result['timestamp']
            })
            if len(self.stats['prediction_history']) > 100:
                self.stats['prediction_history'].pop(0)
            
            return result
            
        except Exception as e:
            self.stats['preprocessing_errors'] += 1
            print(f"⚠️ Detection error: {e}")
            return None

        
        # Load fraud detection model if enabled
        if self.enable_detection:
            self.load_fraud_detection_model()
    
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
                
                # Perform fraud detection if enabled
                if self.enable_detection:
                    detection_result = self.detect_fraud(prepared_record)
                    if detection_result:
                        # Merge detection results into payload
                        payload.update(detection_result)
                        payload['transaction_data'] = dict(prepared_record)
                        payload['transaction_id'] = prepared_record.get('transaction_id', f'TXN_{index:06d}')
                        
                        # Save to MongoDB immediately
                        if self.db and self.db.connected:
                            try:
                                # Add inserted_at timestamp
                                db_payload = dict(payload)
                                db_payload['inserted_at'] = datetime.now().isoformat()
                                
                                # Save to database
                                result_id = self.db.insert_fraud_result(db_payload)
                                
                                # Add database ID to payload for frontend
                                if result_id:
                                    payload['db_id'] = result_id
                                    payload['inserted_at'] = db_payload['inserted_at']
                                
                                # Save immediate alert for fraud transactions
                                if db_payload['is_fraud']:
                                    self.db.insert_immediate_alert(db_payload)
                                    
                            except Exception as e:
                                print(f"   ⚠️ MongoDB save error: {e}")
                        
                        # Debug: Print detection result every 10 records
                        if index % 10 == 0:
                            print(f"   🔍 Detection: Error={detection_result['reconstruction_error']:.6f}, "
                                  f"Fraud={detection_result['is_fraud']}, Risk={detection_result['risk_level']}")
                    else:
                        # If detection failed, mark it
                        payload['detection_error'] = True
                        print(f"   ⚠️ Detection failed for record {index + 1}")

                # Send record to frontend
                await websocket.send(json.dumps(payload))

                print(
                    f"📤 Sent record {index + 1}/{total}: "
                    f"{prepared_record.get('transaction_id', 'Unknown')}"
                    + (f" - {'🚨 FRAUD' if payload.get('is_fraud') else '✅ Normal'}" if self.enable_detection else "")
                )

                # Move to next record
                index += 1

                # Always restart at end
                if index >= total:
                    print("🔄 End of dataset reached — restarting from beginning")
                    self.stats['dataset_loops'] += 1
                    index = 0

                # Control streaming speed
                await asyncio.sleep(1.0 / self.stream_speed)

            except websockets.exceptions.ConnectionClosed:
                print("⚠️ Client disconnected during stream")
                break

            except Exception as e:
                print(f"❌ Error sending record: {e}")
                import traceback
                traceback.print_exc()
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
                    
                    elif command == "get_model_info":
                        await websocket.send(json.dumps({
                            "model_info": self.model_info if self.enable_detection else {},
                            "detection_enabled": self.enable_detection
                        }))
                    
                    elif command == "get_stats":
                        # Calculate additional stats
                        elapsed = time.time() - self.stats['start_time']
                        throughput = self.stats['total_processed'] / elapsed if elapsed > 0 else 0
                        fraud_rate = (self.stats['fraud_detected'] / self.stats['total_processed'] * 100) if self.stats['total_processed'] > 0 else 0
                        success_rate = ((self.stats['total_processed'] - self.stats['preprocessing_errors']) / self.stats['total_processed'] * 100) if self.stats['total_processed'] > 0 else 100
                        
                        await websocket.send(json.dumps({
                            "stats": {
                                'total_processed': self.stats['total_processed'],
                                'fraud_detected': self.stats['fraud_detected'],
                                'preprocessing_errors': self.stats['preprocessing_errors'],
                                'dataset_loops': self.stats['dataset_loops'],
                                'avg_processing_time': round(self.stats['avg_processing_time'], 2),
                                'throughput_tps': round(throughput, 2),
                                'fraud_rate': round(fraud_rate, 2),
                                'success_rate': round(success_rate, 2),
                                'risk_distribution': self.stats['risk_distribution'],
                                'prediction_history': self.stats['prediction_history'][-20:]  # Send last 20 predictions
                            }
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
                        elapsed = time.time() - self.stats['start_time']
                        throughput = self.stats['total_processed'] / elapsed if elapsed > 0 else 0
                        fraud_rate = (self.stats['fraud_detected'] / self.stats['total_processed'] * 100) if self.stats['total_processed'] > 0 else 0
                        success_rate = ((self.stats['total_processed'] - self.stats['preprocessing_errors']) / self.stats['total_processed'] * 100) if self.stats['total_processed'] > 0 else 100
                        
                        await websocket.send(json.dumps({
                            "streaming": self.streaming_clients.get(websocket, False),
                            "total_records": len(self.dataset),
                            "stream_speed": self.stream_speed,
                            "active_clients": len(self.clients),
                            "stream_total_amount": self.client_totals.get(websocket, 0.0),
                            "detection_enabled": self.enable_detection,
                            "stats": {
                                'total_processed': self.stats['total_processed'],
                                'fraud_detected': self.stats['fraud_detected'],
                                'preprocessing_errors': self.stats['preprocessing_errors'],
                                'dataset_loops': self.stats['dataset_loops'],
                                'avg_processing_time': round(self.stats['avg_processing_time'], 2),
                                'throughput_tps': round(throughput, 2),
                                'fraud_rate': round(fraud_rate, 2),
                                'success_rate': round(success_rate, 2)
                            }
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
    DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "fraudTrain.csv")
    HOST = "localhost"
    PORT = 8765
    STREAM_SPEED = 1.0  # Records per second
    ENABLE_DETECTION = True  # Enable fraud detection
    
    # Create and start server
    server = DatasetStreamServer(DATASET_PATH, STREAM_SPEED, ENABLE_DETECTION)
    
    print(f"\n{'='*60}")
    print("FRAUD DETECTION WEBSOCKET SERVER")
    print("="*60)
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"WebSocket URL: ws://{HOST}:{PORT}")
    print(f"Stream Speed: {STREAM_SPEED} records/second")
    print(f"Dataset: {DATASET_PATH}")
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
        # Start WebSocket server
        async with websockets.serve(server.handler, HOST, PORT):
            await asyncio.Future()  # Run forever
            
    except KeyboardInterrupt:
        print("\n\n🛑 Server shutting down...")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    asyncio.run(main())