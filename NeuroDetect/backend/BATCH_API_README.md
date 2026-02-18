# Batch Processing API - NeuroDetect

## Overview
The Batch Processing API allows you to upload CSV files containing transaction data and run fraud detection using trained Autoencoder or LSTM models. Results are saved to MongoDB and can be downloaded as CSV or PDF reports.

## Features
- ✅ Upload CSV files with transaction data
- ✅ Select model (Autoencoder or LSTM)
- ✅ Custom threshold support
- ✅ Real-time processing progress
- ✅ Fraud statistics and analytics
- ✅ MongoDB integration for result storage
- ✅ PDF report generation with statistics and top fraud cases
- ✅ CSV and JSON export
- ✅ Batch processing history

## Installation

### 1. Install Dependencies
```bash
cd backend
pip install -r requirement.txt
```

Required packages:
- fastapi>=0.109.0
- uvicorn[standard]>=0.27.0
- python-multipart>=0.0.6
- reportlab>=4.0.0
- torch, pandas, numpy, scikit-learn
- pymongo, python-dotenv

### 2. Verify Model Files
Ensure these files exist in `saved_models/`:
- `autoencoder.pth` - Trained autoencoder model
- `enhanced_lstm_fraud_model.pth` - Trained LSTM model
- `enhanced_lstm_fraud_model.json` - LSTM config
- `features.json` - Feature names and categories
- `threshold.json` - Autoencoder threshold
- `scaler.pkl` - Feature scaler

## Running the Server

### Windows
```bash
cd backend
start_batch_server.bat
```

### Linux/Mac
```bash
cd backend
chmod +x start_batch_server.sh
./start_batch_server.sh
```

### Manual Start
```bash
cd backend
conda activate fraud-detection
python server/batch_api.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### 1. Health Check
```http
GET http://localhost:8000/
```
Response:
```json
{
  "service": "NeuroDetect Batch API",
  "version": "1.0.0",
  "status": "running",
  "models_loaded": ["autoencoder", "lstm"]
}
```

### 2. Get Model Information
```http
GET http://localhost:8000/models
```
Returns configuration and metadata for loaded models.

### 3. Process Batch File
```http
POST http://localhost:8000/batch/process
Content-Type: multipart/form-data

file: <CSV file>
model_type: autoencoder | lstm
threshold: <optional custom threshold>
```

Response:
```json
{
  "success": true,
  "batch_id": "batch_autoencoder_20260218_143025",
  "statistics": {
    "total": 1000,
    "fraud_count": 45,
    "legitimate_count": 955,
    "fraud_percentage": 4.5,
    "avg_fraud_score": 0.123,
    "max_fraud_score": 0.987,
    "threshold": 0.0004117581993341446
  },
  "files": {
    "json": "results/batch/batch_autoencoder_20260218_143025_results.json",
    "csv": "results/batch/batch_autoencoder_20260218_143025_results.csv",
    "pdf": "results/batch/batch_autoencoder_20260218_143025_report.pdf"
  },
  "mongodb_saved": true,
  "preview": [...]
}
```

### 4. Download Report
```http
GET http://localhost:8000/batch/download/{batch_id}?format=pdf|csv|json
```
Downloads the specified report format.

### 5. Get Batch History
```http
GET http://localhost:8000/batch/history
```
Returns list of all processed batches with statistics.

## CSV File Format

Required columns:
- `trans_date_trans_time` - Transaction timestamp
- `amt` - Transaction amount
- `lat`, `long` - Transaction location
- `city_pop` - City population
- `merch_lat`, `merch_long` - Merchant location
- `category` - Transaction category
- `gender` - Customer gender (M/F)

Example:
```csv
trans_date_trans_time,amt,lat,long,city_pop,merch_lat,merch_long,category,gender
2020-01-01 00:00:18,2.86,33.9659,-80.9355,333497,33.986391,-81.200714,grocery_pos,F
2020-01-01 00:00:44,29.84,40.3207,-110.4360,302,40.495810,-112.096880,gas_transport,M
```

## Frontend Integration

The React frontend (`batch_upload.tsx`) connects to the API:

```typescript
// Upload and process
const formData = new FormData();
formData.append('file', file);
formData.append('model_type', selectedModel);
formData.append('threshold', threshold.toString());

const response = await fetch('http://localhost:8000/batch/process', {
  method: 'POST',
  body: formData,
});

// Download PDF report
const response = await fetch(`http://localhost:8000/batch/download/${batchId}?format=pdf`);
const blob = await response.blob();
// ... download blob
```

## MongoDB Integration

Results are automatically saved to MongoDB if configured. The system creates two collections:

### `batch_results` Collection
Stores batch metadata:
- batch_id
- model_type
- timestamp
- total_transactions
- fraud_detected
- fraud_percentage

### `fraud_results` Collection
Stores individual transaction results:
- batch_id
- transaction details
- fraud_score
- prediction
- timestamp

## PDF Report

The generated PDF includes:
1. **Header** - Report metadata (date, batch ID, model)
2. **Statistics Summary** - Fraud counts, percentages, scores
3. **Top 10 Fraud Cases** - Highest risk transactions with details
4. **Professional styling** - Color-coded tables and branding

## File Structure

```
backend/
├── server/
│   ├── batch_api.py              # Main FastAPI server
│   └── websocket_unified.py      # Real-time streaming server
├── AEmodel/
│   ├── model.py                  # Autoencoder model
│   └── preprocessor.py           # Data preprocessing
├── LSTMmodel/
│   ├── model.py                  # LSTM model
│   └── preprocessor.py           # LSTM preprocessing
├── database/
│   └── mongodb.py                # MongoDB operations
├── start_batch_server.bat        # Windows startup script
└── start_batch_server.sh         # Linux/Mac startup script

results/
└── batch/
    ├── batch_*_results.json      # JSON results
    ├── batch_*_results.csv       # CSV results
    └── batch_*_report.pdf        # PDF reports
```

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Models Not Loading
- Verify all model files exist in `saved_models/`
- Check file paths in batch_api.py
- Ensure PyTorch is installed with CUDA support if using GPU

### MongoDB Connection Failed
- Check `.env.mongoDB` configuration
- System still works without MongoDB (saves to files only)
- Warning message: "MongoDB disabled - results will be saved to local files only"

### PDF Generation Errors
- Ensure reportlab is installed: `pip install reportlab`
- Check write permissions in `results/batch/` directory

## Performance

- **Autoencoder**: ~100 transactions/second
- **LSTM**: ~50 transactions/second (due to sequence processing)
- **PDF Generation**: ~2-3 seconds for typical report
- **MongoDB Save**: ~500 records/second

## Logs

Logs are output to console with timestamps:
```
2026-02-18 14:30:25 - __main__ - INFO - Using device: cuda
2026-02-18 14:30:26 - __main__ - INFO - ✓ Autoencoder loaded - 24 features, threshold: 0.000412
2026-02-18 14:30:27 - __main__ - INFO - ✓ LSTM loaded - 24 features, threshold: 0.868516
2026-02-18 14:30:45 - __main__ - INFO - Processing 1000 transactions with autoencoder
2026-02-18 14:30:47 - __main__ - INFO - ✓ Saved 1000 results to MongoDB
2026-02-18 14:30:49 - __main__ - INFO - ✓ Generated PDF report
```

## Security Notes

- API currently has CORS enabled for all origins (development)
- For production, restrict CORS to specific domains
- Add authentication middleware for protected endpoints
- Validate and sanitize CSV input
- Implement rate limiting for batch processing

## Future Enhancements

- [ ] Real-time progress updates via WebSocket
- [ ] Batch processing queue for large files
- [ ] Email notifications on completion
- [ ] Advanced analytics dashboard
- [ ] Comparison between models
- [ ] A/B testing capabilities
- [ ] Custom model training from UI
