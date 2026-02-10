"""
Test MongoDB Connection and Operations
"""
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env.mongoDB'))

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.mongodb import MongoDB

def test_mongodb_connection():
    """Test MongoDB connection and basic operations"""
    
    print("="*60)
    print("MongoDB Connection Test")
    print("="*60)
    
    # Display connection info
    mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    mongodb_db = os.getenv('MONGODB_DATABASE', 'neurodetect')
    
    # Mask password in URI for display
    display_uri = mongodb_uri
    if '@' in mongodb_uri and '://' in mongodb_uri:
        parts = mongodb_uri.split('://')
        if len(parts) == 2 and '@' in parts[1]:
            credentials, host = parts[1].split('@', 1)
            if ':' in credentials:
                user, _ = credentials.split(':', 1)
                display_uri = f"{parts[0]}://{user}:****@{host}"
    
    print(f"\n📍 Connection Details:")
    print(f"   URI: {display_uri}")
    print(f"   Database: {mongodb_db}")
    
    # Test 1: Connection
    print("\n1️⃣ Testing connection...")
    db = MongoDB()
    
    if not db.connect():
        print("❌ Connection test failed")
        print("\nPlease check:")
        print("  1. Your MongoDB connection string is correct in .env.mongoDB")
        print("  2. For MongoDB Atlas: Network access allows your IP")
        print("  3. For MongoDB Atlas: Database user credentials are correct")
        print("  4. Your internet connection is stable (for cloud MongoDB)")
        return False
    
    print("✅ Connection successful")
    
    # Test 2: Insert test fraud result
    print("\n2️⃣ Testing fraud result insertion...")
    test_result = {
        'transaction_id': 'TEST_001',
        'transaction_data': {
            'amt': 150.00,
            'category': 'grocery_pos'
        },
        'reconstruction_error': 0.025,
        'threshold': 0.020,
        'is_fraud': True,
        'fraud_probability': 0.85,
        'risk_level': 'High',
        'processing_time_ms': 5.2,
        'timestamp': '2026-02-08T10:30:00'
    }
    
    result_id = db.insert_fraud_result(test_result)
    if result_id:
        print(f"✅ Test result inserted (ID: {result_id})")
    else:
        print("❌ Failed to insert test result")
        db.close()
        return False
    
    # Test 3: Insert immediate alert
    print("\n3️⃣ Testing immediate alert insertion...")
    alert_id = db.insert_immediate_alert(test_result)
    if alert_id:
        print(f"✅ Test alert inserted (ID: {alert_id})")
    else:
        print("❌ Failed to insert test alert")
    
    # Test 4: Get fraud count
    print("\n4️⃣ Testing fraud count query...")
    fraud_count = db.get_fraud_count()
    print(f"✅ Total frauds in database: {fraud_count}")
    
    # Test 5: Get recent frauds
    print("\n5️⃣ Testing recent frauds query...")
    recent = db.get_recent_frauds(limit=5)
    print(f"✅ Retrieved {len(recent)} recent fraud records")
    
    # Test 6: Get statistics summary
    print("\n6️⃣ Testing statistics summary...")
    summary = db.get_statistics_summary()
    if summary:
        print("✅ Statistics retrieved:")
        print(f"   Total transactions: {summary.get('total_transactions', 0)}")
        print(f"   Fraud detected: {summary.get('fraud_detected', 0)}")
        print(f"   Fraud rate: {summary.get('fraud_rate', 0):.2f}%")
    else:
        print("⚠️ No statistics available")
    
    # Test 7: Save test statistics
    print("\n7️⃣ Testing statistics saving...")
    test_stats = {
        'session_type': 'test_session',
        'total_processed': 100,
        'fraud_detected': 15,
        'fraud_rate': 15.0,
        'avg_processing_time_ms': 5.5
    }
    stats_id = db.save_statistics(test_stats)
    if stats_id:
        print(f"✅ Test statistics saved (ID: {stats_id})")
    else:
        print("❌ Failed to save test statistics")
    
    # Cleanup
    print("\n8️⃣ Closing connection...")
    db.close()
    print("✅ Connection closed")
    
    print("\n" + "="*60)
    print("✅ All tests passed successfully!")
    print("="*60)
    print("\nYour MongoDB setup is working correctly.")
    print("You can now run real_time.py to detect fraud with MongoDB storage.")
    
    return True

if __name__ == "__main__":
    try:
        success = test_mongodb_connection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
