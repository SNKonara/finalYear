"""
Real-Time LSTM Fraud Detection with WebSocket
Similar to autoencoder real-time detection but uses LSTM with sequence processing
"""
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import torch
import joblib
import time
import os
import sys
from datetime import datetime
import threading
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AEmodel.preprocessor import DataPreprocessor
from LSTMmodel.model import EnhancedFraudLSTM
from LSTMmodel.save_load import load_model
from database.mongodb import get_mongodb_instance

class LSTMFraudDetector:
    def __init__(self, websocket_url="ws://localhost:8765", sequence_length=10):
        """
        Real-time LSTM fraud detector with WebSocket integration
        
        Args:
            websocket_url: WebSocket server URL
            sequence_length: Number of transactions in sequence (default: 10)
        """
        self.websocket_url = websocket_url
        self.sequence_length = sequence_length
        self.connected = False
        self.processing = False
        self.results = []
        self.stats = {
            'total_processed': 0,
            'fraud_detected': 0,
            'avg_processing_time': 0,
            'start_time': None,
            'dataset_loops': 0,
            'preprocessing_errors': 0
        }
        
        # Transaction sequence buffer
        self.transaction_buffer = deque(maxlen=sequence_length)
        
        # Load model components
        self.model = None
        self.scaler = None
        self.optimal_threshold = 0.5
        self.preprocessor = None
        self.device = None
        self.expected_features = None
        self.feature_names = None
        
        # MongoDB connection
        self.db = None
        
        print("Initializing LSTM Fraud Detection Engine...")
        self.load_model_components()
        self.connect_database()
    
    def connect_database(self):
        """Initialize MongoDB connection"""
        try:
            print("\n📊 Connecting to database...")
            self.db = get_mongodb_instance()
            if self.db.connected:
                print("✅ Database connected successfully")
            else:
                print("⚠️ Database connection failed - results will not be saved")
        except Exception as e:
            print(f"⚠️ Database connection error: {e}")
            print("   Continuing without database - results will not be saved")
            self.db = None
    
    def load_model_components(self):
        """Load trained LSTM model and components"""
        print("Loading LSTM model components...")
        
        try:
            # Load LSTM model using save_load module
            model_path = 'saved_models/enhanced_lstm_fraud_model.pth'
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file not found: {model_path}\n"
                    "Please train the LSTM model first using: python scripts/runLSTM.py"
                )
            
            # Load model
            self.model, self.scaler, self.feature_names, model_config, results = load_model(
                model_path
            )
            
            self.expected_features = model_config['input_dim']
            self.optimal_threshold = results.get('optimal_threshold', 0.5)
            
            # Initialize preprocessor for data preprocessing
            self.preprocessor = DataPreprocessor()
            # Note: We'll use the scaler from LSTM model, not from AE model
            self.preprocessor.scaler = self.scaler
            
            # Load preprocessing parameters (categories, etc.)
            try:
                self.preprocessor.load_preprocessor('saved_models/scaler.pkl')
            except:
                print("⚠️ Could not load AE preprocessor parameters, using defaults")
            
            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ LSTM Model loaded successfully")
            print(f"   Sequence Length: {self.sequence_length}")
            print(f"   Input dimensions: {self.expected_features}")
            print(f"   Architecture: Enhanced LSTM with Attention")
            print(f"   Optimal Threshold: {self.optimal_threshold:.4f}")
            print(f"   Device: {self.device}")
            print(f"   Features: {len(self.feature_names)}")
            
            if results:
                print(f"\n   Model Performance:")
                print(f"   ├─ F1 Score: {results.get('f1', 'N/A'):.4f}")
                print(f"   ├─ Fraud F1: {results.get('fraud_f1', 'N/A'):.4f}")
                print(f"   ├─ Fraud Recall: {results.get('fraud_recall', 'N/A'):.4f}")
                print(f"   └─ ROC AUC: {results.get('roc_auc', 'N/A'):.4f}")
            
        except FileNotFoundError as e:
            print(f"❌ ERROR: {e}")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def preprocess_transaction(self, transaction_data):
        """Preprocess a single transaction"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([transaction_data])
            
            # Debug: Print incoming columns on first transaction
            if self.stats['total_processed'] == 1:
                print(f"\n🔍 DEBUG - Incoming data columns:")
                print(f"   {df.columns.tolist()}")
                print(f"\n🔍 DEBUG - Sample data:")
                print(df.head())
            
            # Preprocess using AEmodel preprocessor
            X_processed, _, _ = self.preprocessor.preprocess(df, is_training=False)
            
            # Verify dimensions
            if X_processed.shape[1] != self.expected_features:
                error_msg = f"Feature mismatch: got {X_processed.shape[1]}, expected {self.expected_features}"
                print(f"\n❌ {error_msg}")
                
                if self.feature_names:
                    print(f"   Expected features: {self.feature_names}")
                
                print(f"   Processed features shape: {X_processed.shape}")
                return None
            
            return X_processed[0]  # Return single transaction features
            
        except Exception as e:
            self.stats['preprocessing_errors'] += 1
            print(f"⚠️ Preprocessing error: {e}")
            print(f"   Transaction data keys: {list(transaction_data.keys())}")
            
            # Print more debug info on first error
            if self.stats['preprocessing_errors'] == 1:
                import traceback
                print("\n🔍 Full traceback:")
                traceback.print_exc()
            
            return None
    
    def detect_fraud(self, transaction_data, X_sequence):
        """Run LSTM fraud detection on a sequence"""
        start_time = time.time()
        
        try:
            # Convert sequence to tensor (batch_size=1, seq_len, features)
            X_tensor = torch.FloatTensor(X_sequence).unsqueeze(0).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                output, attention_weights = self.model(X_tensor)
                fraud_prob = output.item()
            
            # Determine if fraud using optimal threshold
            is_fraud = fraud_prob > self.optimal_threshold
            
            # Calculate confidence score
            confidence = abs(fraud_prob - self.optimal_threshold) / self.optimal_threshold
            confidence = min(confidence, 1.0)
            
            # Determine risk level (High means flagged fraud)
            if is_fraud:
                risk = "High"
            elif fraud_prob < self.optimal_threshold * 0.5:
                risk = "Low"
            elif fraud_prob < self.optimal_threshold:
                risk = "Medium-Low"
            else:
                risk = "Medium-High"
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Create result
            result = {
                'transaction_id': transaction_data.get('transaction_id', 'Unknown'),
                'transaction_data': transaction_data,
                'fraud_score': float(fraud_prob),
                'optimal_threshold': float(self.optimal_threshold),
                'is_fraud': bool(is_fraud),
                'confidence': float(confidence),
                'risk_level': risk,
                'sequence_length': self.sequence_length,
                'processing_time_ms': round(processing_time, 2),
                'timestamp': datetime.now().isoformat(),
                'model_type': 'LSTM'
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def connect_to_server(self):
        """Connect to WebSocket server"""
        print(f"Connecting to WebSocket server: {self.websocket_url}")
        
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            self.connected = True
            
            print("✅ Connected to server")
            
            # Send initial ping
            await self.websocket.send(json.dumps({"command": "ping"}))
            response = await self.websocket.recv()
            print(f"Server response: {json.loads(response)['response']}")
            
            # Get server status
            await self.websocket.send(json.dumps({"command": "get_status"}))
            status = await self.websocket.recv()
            print(f"📊 Server status: {json.loads(status)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.connected = False
            return False
    
    async def start_detection(self):
        """Start real-time LSTM fraud detection"""
        if not self.connected:
            print("❌ Not connected to server")
            return
        
        print("\n🚀 Starting real-time LSTM fraud detection...")
        print(f"   Sequence Length: {self.sequence_length} transactions")
        print(f"   Optimal Threshold: {self.optimal_threshold:.4f}")
        print("   (Press Ctrl+C to stop)")
        print("-" * 60)
        
        # Tell server to start streaming
        await self.websocket.send(json.dumps({"command": "start_stream"}))
        
        self.processing = True
        self.stats['start_time'] = time.time()
        
        try:
            async for message in self.websocket:
                if not self.processing:
                    break
                
                try:
                    data = json.loads(message)
                    
                    # Check for end of dataset message
                    if data.get('message') == 'END_OF_DATASET':
                        self.stats['dataset_loops'] += 1
                        print(f"\n🔄 Dataset completed (Loop #{self.stats['dataset_loops']})")
                        print(f"   Total records in dataset: {data.get('total_records', 0)}")
                        print(f"   Automatically restarting stream...")
                        
                        # Restart the stream
                        await self.websocket.send(json.dumps({"action": "restart"}))
                        continue
                    
                    # Process transaction
                    self.stats['total_processed'] += 1
                    transaction_id = data.get('transaction_id', f"TXN_{self.stats['total_processed']:06d}")
                    
                    print(f"\n📥 Processing transaction {self.stats['total_processed']}:")
                    print(f"   ID: {transaction_id}")
                    if 'amt' in data:
                        print(f"   Amount: ${data.get('amt', 0):.2f}")
                    print(f"   Category: {data.get('category', 'Unknown')}")
                    
                    # Preprocess
                    X_processed = self.preprocess_transaction(data)
                    
                    if X_processed is None:
                        print("   ⚠️ Skipped due to preprocessing error")
                        
                        # Stop after too many errors
                        if self.stats['preprocessing_errors'] >= 5:
                            print(f"\n❌ Too many preprocessing errors ({self.stats['preprocessing_errors']}). Stopping.")
                            print("   This usually means the data format doesn't match what the model expects.")
                            break
                        continue
                    
                    # Add to sequence buffer
                    self.transaction_buffer.append(X_processed)
                    
                    # Check if we have enough transactions for a sequence
                    if len(self.transaction_buffer) < self.sequence_length:
                        print(f"   ℹ️ Building sequence... ({len(self.transaction_buffer)}/{self.sequence_length})")
                        continue
                    
                    # Create sequence array
                    X_sequence = np.array(list(self.transaction_buffer))
                    
                    # Detect fraud
                    result = self.detect_fraud(data, X_sequence)
                    
                    if result:
                        self.results.append(result)
                        
                        # Update stats
                        if result['is_fraud']:
                            self.stats['fraud_detected'] += 1
                        
                        # Update average processing time
                        total_time = sum(r['processing_time_ms'] for r in self.results)
                        self.stats['avg_processing_time'] = total_time / len(self.results)
                        
                        # Display result
                        status = "🚨 FRAUD DETECTED" if result['is_fraud'] else "✅ Normal"
                        print(f"   Result: {status}")
                        print(f"   Score: {result['fraud_score']:.4f}")
                        print(f"   Confidence: {result['confidence']:.3f}")
                        print(f"   Risk Level: {result['risk_level']}")
                        print(f"   Processing Time: {result['processing_time_ms']:.2f} ms")
                        
                        # Save result to database
                        if self.db and self.db.connected:
                            self.db.insert_fraud_result(result)
                            
                            # Save immediate alert for high-risk fraud
                            if result['is_fraud']:
                                self.save_immediate_alert(result)
                    
                    # Display statistics periodically
                    if self.stats['total_processed'] % 10 == 0:
                        self.display_statistics()
                        
                except json.JSONDecodeError:
                    print(f"⚠️ Invalid JSON received: {message[:100]}...")
                except Exception as e:
                    print(f"⚠️ Error processing message: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except websockets.exceptions.ConnectionClosed:
            print("❌ Connection to server closed")
        except Exception as e:
            print(f"❌ Error in detection loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.processing = False
    
    def save_immediate_alert(self, result):
        """Save immediate fraud alert to database"""
        try:
            if self.db and self.db.connected:
                alert_id = self.db.insert_immediate_alert(result)
                if alert_id:
                    print(f"🚨 Fraud alert saved to database (ID: {alert_id})")
        except Exception as e:
            print(f"⚠️ Error saving fraud alert: {e}")
    
    def display_statistics(self):
        """Display current statistics"""
        print(f"\n📊 LSTM Statistics (after {self.stats['total_processed']} transactions):")
        print(f"   Fraud detected: {self.stats['fraud_detected']} ({self.stats['fraud_detected']/max(self.stats['total_processed'], 1)*100:.1f}%)")
        print(f"   Preprocessing errors: {self.stats['preprocessing_errors']}")
        print(f"   Avg processing time: {self.stats['avg_processing_time']:.2f} ms")
        print(f"   Sequence length: {self.sequence_length}")
        print(f"   Buffer size: {len(self.transaction_buffer)}")
        print(f"   Dataset loops completed: {self.stats['dataset_loops']}")
        
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            tps = self.stats['total_processed'] / elapsed if elapsed > 0 else 0
            print(f"   Transactions/sec: {tps:.2f}")
    
    async def stop_detection(self):
        """Stop fraud detection"""
        self.processing = False
        
        if self.connected:
            try:
                await self.websocket.send(json.dumps({"command": "stop_stream"}))
            except:
                pass
        
        print("\n🛑 LSTM fraud detection stopped")
        self.display_final_statistics()
    
    def display_final_statistics(self):
        """Display final statistics"""
        print("\n" + "="*60)
        print("FINAL LSTM DETECTION STATISTICS")
        print("="*60)
        
        print(f"\n📈 Summary:")
        print(f"   Model Type: Enhanced LSTM with Attention")
        print(f"   Sequence Length: {self.sequence_length}")
        print(f"   Total transactions processed: {self.stats['total_processed']}")
        print(f"   Fraud detected: {self.stats['fraud_detected']}")
        print(f"   Preprocessing errors: {self.stats['preprocessing_errors']}")
        print(f"   Dataset loops completed: {self.stats['dataset_loops']}")
        
        if self.stats['total_processed'] > 0:
            fraud_rate = (self.stats['fraud_detected'] / self.stats['total_processed']) * 100
            success_rate = ((self.stats['total_processed'] - self.stats['preprocessing_errors']) / self.stats['total_processed']) * 100
            print(f"   Fraud rate: {fraud_rate:.2f}%")
            print(f"   Success rate: {success_rate:.2f}%")
        
        print(f"   Average processing time: {self.stats['avg_processing_time']:.2f} ms")
        
        if self.stats['start_time']:
            total_time = time.time() - self.stats['start_time']
            print(f"   Total processing time: {total_time:.2f} seconds")
            if total_time > 0:
                print(f"   Throughput: {self.stats['total_processed'] / total_time:.2f} transactions/sec")
    
    def save_results(self):
        """Save statistics and summary to database"""
        print("\n💾 Saving LSTM session statistics to database...")
        
        if not self.db or not self.db.connected:
            print("⚠️ Database not connected - statistics not saved")
            return
        
        try:
            # Prepare session statistics
            session_stats = {
                'session_type': 'lstm_websocket_detection',
                'model_type': 'Enhanced_LSTM',
                'sequence_length': self.sequence_length,
                'total_processed': self.stats['total_processed'],
                'fraud_detected': self.stats['fraud_detected'],
                'preprocessing_errors': self.stats['preprocessing_errors'],
                'dataset_loops': self.stats['dataset_loops'],
                'avg_processing_time_ms': self.stats['avg_processing_time'],
                'fraud_rate': (self.stats['fraud_detected'] / max(self.stats['total_processed'], 1)) * 100,
                'optimal_threshold': self.optimal_threshold,
                'session_start': datetime.fromtimestamp(self.stats['start_time']).isoformat() if self.stats['start_time'] else None,
                'session_end': datetime.now().isoformat(),
                'total_duration_seconds': time.time() - self.stats['start_time'] if self.stats['start_time'] else 0,
                'throughput_tps': self.stats['total_processed'] / (time.time() - self.stats['start_time']) if self.stats['start_time'] and (time.time() - self.stats['start_time']) > 0 else 0
            }
            
            # Save to database
            stats_id = self.db.save_statistics(session_stats)
            if stats_id:
                print(f"✅ LSTM session statistics saved to database (ID: {stats_id})")
                
                # Display database summary
                summary = self.db.get_statistics_summary()
                if summary:
                    print(f"\n📊 Database Summary:")
                    print(f"   Total transactions in DB: {summary.get('total_transactions', 0)}")
                    print(f"   Total fraud detected in DB: {summary.get('fraud_detected', 0)}")
                    print(f"   Overall fraud rate: {summary.get('fraud_rate', 0):.2f}%")
            else:
                print("⚠️ Failed to save statistics to database")
                
        except Exception as e:
            print(f"❌ Error saving statistics: {e}")
    
    async def run(self):
        """Main run method"""
        print("\n" + "="*60)
        print("LSTM WEBSOCKET FRAUD DETECTION SYSTEM")
        print("="*60)
        
        # Connect to server
        if not await self.connect_to_server():
            return
        
        try:
            # Start detection
            await self.start_detection()
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
        finally:
            # Stop detection
            await self.stop_detection()
            
            # Save results
            self.save_results()
            
            # Close connections
            if self.connected:
                try:
                    await self.websocket.close()
                    print("🔌 WebSocket connection closed")
                except:
                    pass
            
            # Close database connection
            if self.db and self.db.connected:
                self.db.close()
            
            print("\n👋 LSTM detection completed!")

async def main():
    """Main entry point"""
    # Configuration
    WEBSOCKET_URL = "ws://localhost:8765"
    SEQUENCE_LENGTH = 10  # Number of transactions in each sequence
    
    # Create and run LSTM detector
    detector = LSTMFraudDetector(
        websocket_url=WEBSOCKET_URL,
        sequence_length=SEQUENCE_LENGTH
    )
    await detector.run()

if __name__ == "__main__":
    asyncio.run(main())
