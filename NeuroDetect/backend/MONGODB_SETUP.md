# MongoDB Setup and Migration Guide

## Overview
Your NeuroDetect application now saves fraud detection results to MongoDB instead of local files.

## What Changed

### 1. Dependencies Added
- `pymongo>=4.0.0` - MongoDB Python driver
- `python-dotenv>=0.19.0` - Environment variable management

### 2. New Files Created
- `backend/database/mongodb.py` - MongoDB connection and operations
- `backend/database/__init__.py` - Database module initialization
- `backend/.env.example` - Configuration template
- `backend/scripts/test_mongodb.py` - MongoDB testing script

### 3. Updated Files
- `backend/requirement.txt` - Added MongoDB dependencies
- `backend/scripts/real_time.py` - Now saves to MongoDB instead of files

## MongoDB Setup

### Option 1: Local MongoDB (Recommended for Development)

1. **Install MongoDB Community Edition**
   - Windows: https://www.mongodb.com/try/download/community
   - Download and run the installer
   - During installation, install as a Windows Service
   - Default connection: `mongodb://localhost:27017/`

2. **Verify MongoDB is Running**
   ```powershell
   # Check if MongoDB service is running
   Get-Service MongoDB
   
   # Or start it manually
   net start MongoDB
   ```

3. **Create Environment Configuration**
   ```powershell
   cd backend
   cp .env.example .env
   ```
   
   The default `.env` settings work for local MongoDB:
   ```
   MONGODB_URI=mongodb://localhost:27017/
   MONGODB_DATABASE=neurodetect
   ```

### Option 2: MongoDB Atlas (Cloud)

1. **Create Free Account**
   - Go to https://www.mongodb.com/cloud/atlas
   - Create a free cluster

2. **Get Connection String**
   - Click "Connect" on your cluster
   - Choose "Connect your application"
   - Copy the connection string

3. **Update .env File**
   ```
   MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/
   MONGODB_DATABASE=neurodetect
   ```
   Replace `<username>`, `<password>`, and `<cluster>` with your values

## Installation Steps

1. **Install Dependencies**
   ```powershell
   cd backend
   pip install -r requirement.txt
   ```

2. **Test MongoDB Connection**
   ```powershell
   python scripts/test_mongodb.py
   ```
   
   This will:
   - Test database connection
   - Insert test data
   - Query data
   - Verify all operations work

3. **Run Fraud Detection**
   ```powershell
   # Start the data stream server (in one terminal)
   python server/websocket_dataset.py
   
   # Start fraud detection (in another terminal)
   python scripts/real_time.py
   ```

## Database Collections

### 1. `fraud_results`
Stores all fraud detection results:
- transaction_id
- transaction_data
- reconstruction_error
- threshold
- is_fraud
- fraud_probability
- risk_level
- processing_time_ms
- timestamp
- inserted_at

### 2. `immediate_alerts`
Stores high-priority fraud alerts:
- Same fields as fraud_results
- Only for transactions flagged as fraud
- Used for immediate notification

### 3. `statistics`
Stores session statistics:
- session_type
- total_processed
- fraud_detected
- fraud_rate
- avg_processing_time_ms
- session_start
- session_end
- total_duration_seconds
- throughput_tps

### 4. `transactions`
Reserved for storing raw transaction data

## Viewing Data

### Using MongoDB Compass (GUI)
1. Download: https://www.mongodb.com/products/compass
2. Connect to `mongodb://localhost:27017/`
3. Browse the `neurodetect` database

### Using MongoDB Shell
```bash
# Connect to MongoDB
mongosh

# Switch to neurodetect database
use neurodetect

# View collections
show collections

# Query fraud results
db.fraud_results.find().pretty()

# Get fraud count
db.fraud_results.countDocuments({is_fraud: true})

# Get recent frauds
db.fraud_results.find({is_fraud: true}).sort({timestamp: -1}).limit(10)

# Get statistics
db.statistics.find().pretty()
```

## Querying Examples

### Python Code
```python
from database.mongodb import get_mongodb_instance

# Get database instance
db = get_mongodb_instance()

# Get fraud count
total_frauds = db.get_fraud_count()
print(f"Total frauds: {total_frauds}")

# Get recent frauds
recent = db.get_recent_frauds(limit=10)
for fraud in recent:
    print(f"Transaction: {fraud['transaction_id']}, Error: {fraud['reconstruction_error']}")

# Get summary statistics
summary = db.get_statistics_summary()
print(f"Fraud rate: {summary['fraud_rate']:.2f}%")
```

## Migration from Local Files

Your old results are still in the `results/` folder. To migrate them to MongoDB:

1. The data structure is compatible - fraud results have the same format
2. You can manually import JSON files using a script if needed
3. Going forward, all new results save to MongoDB automatically

## Benefits of MongoDB

✅ **Better Performance**: Indexed queries are much faster than file scans  
✅ **Scalability**: Can handle millions of transactions  
✅ **Real-time Queries**: Dashboard can query latest frauds instantly  
✅ **Analytics**: Built-in aggregation for statistics  
✅ **Concurrent Access**: Multiple services can read/write simultaneously  
✅ **Backup**: Easy to backup entire database  
✅ **Cloud Ready**: Works with MongoDB Atlas for cloud deployment  

## Troubleshooting

### "Connection failed" Error
- Ensure MongoDB is installed and running
- Check if service is active: `Get-Service MongoDB`
- Try: `net start MongoDB`

### "Authentication failed" Error
- Check your MONGODB_URI in `.env`
- For Atlas, verify username/password
- Ensure IP address is whitelisted in Atlas

### "Database not connected" Warning
- The system will continue but won't save results
- Check MongoDB logs for errors
- Verify network connectivity

### Performance Issues
- Indexes are created automatically
- For large datasets, consider sharding
- Monitor with MongoDB Atlas metrics

## Next Steps

1. ✅ Install MongoDB
2. ✅ Run test_mongodb.py to verify setup
3. ✅ Run your fraud detection with MongoDB storage
4. 📊 Build a dashboard to visualize MongoDB data
5. 🔔 Add real-time alerts from immediate_alerts collection

## Support

- MongoDB Docs: https://docs.mongodb.com/
- PyMongo Docs: https://pymongo.readthedocs.io/
- Atlas Tutorial: https://docs.atlas.mongodb.com/getting-started/
