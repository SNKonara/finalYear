# realtime_websocket_detector.py
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AEmodel.preprocessor import DataPreprocessor
from AEmodel.model import FraudAutoencoder

class WebSocketFraudDetector:
    def __init__(self, websocket_url="ws://localhost:8765"):
        """
        Real-time fraud detector with WebSocket integration
        
        Args:
            websocket_url: WebSocket server URL
        """
        self.websocket_url = websocket_url
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
        
        # Load model components
        self.model = None
        self.scaler = None
        self.threshold = None
        self.preprocessor = None
        self.device = None
        self.expected_features = None
        self.feature_names = None
        
        print("Initializing Fraud Detection Engine...")
        self.load_model_components()
    
    def load_model_components(self):
        """Load trained model and components"""
        print("Loading model components...")
        
        try:
            # Load model checkpoint
            checkpoint = torch.load('saved_models/autoencoder.pth')
            input_dim = checkpoint['input_dim']
            self.expected_features = input_dim
            
            # Try to get feature names if available
            if 'feature_names' in checkpoint:
                self.feature_names = checkpoint['feature_names']
                print(f"   Expected features ({len(self.feature_names)}): {self.feature_names}")
            
            # Initialize model using the proper class from model.py
            self.model = FraudAutoencoder(
                input_dim=input_dim,
                hidden_dim1=128,
                hidden_dim2=64,
                latent_dim=16,
                dropout_rate=0.000287
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Initialize and load preprocessor (includes scaler and parameters)
            self.preprocessor = DataPreprocessor()
            self.preprocessor.load_preprocessor('saved_models/scaler.pkl')
            self.scaler = self.preprocessor.scaler
            
            print(f"   Scaler expects {self.scaler.n_features_in_} features")
            
            # Load threshold
            with open('saved_models/threshold.json', 'r') as f:
                threshold_data = json.load(f)
                self.threshold = threshold_data['threshold']
            
            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            
            print(f"✅ Model loaded successfully")
            print(f"   Input dimensions: {input_dim}")
            print(f"   Architecture: 128-64-16 (encoder)")
            print(f"   Threshold: {self.threshold:.6f}")
            print(f"   Device: {self.device}")
            
        except FileNotFoundError as e:
            print(f"❌ ERROR: {e}")
            print("   Please train the model first using 01_train_model.py")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
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
            
            # Preprocess
            X_processed, _, _ = self.preprocessor.preprocess(df, is_training=False)
            
            # Verify dimensions
            if X_processed.shape[1] != self.expected_features:
                error_msg = f"Feature mismatch: got {X_processed.shape[1]}, expected {self.expected_features}"
                print(f"\n❌ {error_msg}")
                
                # Try to debug the issue
                if self.feature_names:
                    print(f"   Expected features: {self.feature_names}")
                
                print(f"   Processed features shape: {X_processed.shape}")
                return None
            
            return X_processed
            
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
    
    def detect_fraud(self, transaction_data, X_processed):
        """Run fraud detection on a transaction"""
        start_time = time.time()
        
        try:
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_processed).to(self.device)
            
            # Get reconstruction error using model's built-in method
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
            
            # Create result
            result = {
                'transaction_id': transaction_data.get('transaction_id', 'Unknown'),
                'transaction_data': transaction_data,
                'reconstruction_error': float(error_value),
                'threshold': float(self.threshold),
                'is_fraud': bool(is_fraud),
                'fraud_probability': float(fraud_prob),
                'risk_level': risk,
                'processing_time_ms': round(processing_time, 2),
                'timestamp': datetime.now().isoformat()
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
        """Start real-time fraud detection"""
        if not self.connected:
            print("❌ Not connected to server")
            return
        
        print("\n🚀 Starting real-time fraud detection...")
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
                        
                        # Automatically restart the stream
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
                            print("   Please check your preprocessor and model training data.")
                            break
                        continue
                    
                    # Detect fraud
                    result = self.detect_fraud(data, X_processed)
                    
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
                        print(f"   Error: {result['reconstruction_error']:.6f}")
                        print(f"   Probability: {result['fraud_probability']:.3f}")
                        print(f"   Risk Level: {result['risk_level']}")
                        print(f"   Processing Time: {result['processing_time_ms']:.2f} ms")
                        
                        # Save result immediately (optional)
                        if result['is_fraud']:
                            self.save_immediate_result(result)
                    
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
    
    def save_immediate_result(self, result):
        """Save immediate fraud result for quick review"""
        os.makedirs('results/immediate', exist_ok=True)
        
        filename = f"results/immediate/fraud_{result['transaction_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"💾 Immediate result saved: {filename}")
    
    def display_statistics(self):
        """Display current statistics"""
        print(f"\n📊 Statistics (after {self.stats['total_processed']} transactions):")
        print(f"   Fraud detected: {self.stats['fraud_detected']} ({self.stats['fraud_detected']/max(self.stats['total_processed'], 1)*100:.1f}%)")
        print(f"   Preprocessing errors: {self.stats['preprocessing_errors']}")
        print(f"   Avg processing time: {self.stats['avg_processing_time']:.2f} ms")
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
        
        print("\n🛑 Fraud detection stopped")
        self.display_final_statistics()
    
    def display_final_statistics(self):
        """Display final statistics"""
        print("\n" + "="*60)
        print("FINAL DETECTION STATISTICS")
        print("="*60)
        
        print(f"\n📈 Summary:")
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
        """Save all results to files"""
        if not self.results:
            print("\n⚠️ No results to save")
            return
            
        print("\n💾 Saving results...")
        os.makedirs('results', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as JSON
        json_path = f'results/websocket_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ JSON results saved: {json_path}")
        
        # Save as CSV
        if self.results:
            # Extract summary data
            summary_data = []
            for result in self.results:
                summary = {
                    'transaction_id': result['transaction_id'],
                    'amount': result['transaction_data'].get('amt', 0),
                    'category': result['transaction_data'].get('category', 'Unknown'),
                    'reconstruction_error': result['reconstruction_error'],
                    'is_fraud': result['is_fraud'],
                    'fraud_probability': result['fraud_probability'],
                    'risk_level': result['risk_level'],
                    'processing_time_ms': result['processing_time_ms'],
                    'timestamp': result['timestamp']
                }
                summary_data.append(summary)
            
            df = pd.DataFrame(summary_data)
            csv_path = f'results/websocket_results_{timestamp}.csv'
            df.to_csv(csv_path, index=False)
            print(f"✓ CSV results saved: {csv_path}")
    
    async def run(self):
        """Main run method"""
        print("\n" + "="*60)
        print("WEBSOCKET FRAUD DETECTION SYSTEM")
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
            
            # Close connection
            if self.connected:
                try:
                    await self.websocket.close()
                    print("🔌 Connection closed")
                except:
                    pass
            
            print("\n👋 Detection completed!")

async def main():
    """Main entry point"""
    # Configuration
    WEBSOCKET_URL = "ws://localhost:8765"  # Change if your server is different
    
    # Create and run detector
    detector = WebSocketFraudDetector(WEBSOCKET_URL)
    await detector.run()

if __name__ == "__main__":
    asyncio.run(main())