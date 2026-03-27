# Quick Start Guide - Batch Processing

## 1. Start the Backend Server

Open a terminal in the backend directory:

```bash
cd backend
conda activate fraud-detection
python server/batch_api.py
```

Server will start on `http://localhost:8000`

## 2. Start the Frontend

Open another terminal:

```bash
cd NeuroDetect
npm run dev
```

Frontend will start on `http://localhost:5173`

## 3. Navigate to Workplace

- Open browser to `http://localhost:5173`
- Click "Workplace" in the sidebar

## 4. Upload and Process

1. **Select Model**: Choose Autoencoder or LSTM
2. **Upload CSV**: Drag & drop or click to browse
3. **Review Preview**: Check first 5 rows
4. **Click Process**: Wait for results
5. **View Results**: See fraud statistics and predictions
6. **Download Reports**: 
   - Click "Export CSV" for raw data
   - Click "Download PDF Report" for formatted report

## CSV Format Example

```csv
trans_date_trans_time,amt,lat,long,city_pop,merch_lat,merch_long,category,gender
2020-01-01 00:00:18,2.86,33.9659,-80.9355,333497,33.986391,-81.200714,grocery_pos,F
2020-01-01 00:00:44,29.84,40.3207,-110.4360,302,40.495810,-112.096880,gas_transport,M
```

## Viewing Results

Results are saved in:
- **JSON**: `results/batch/batch_*_results.json`
- **CSV**: `results/batch/batch_*_results.csv`
- **PDF**: `results/batch/batch_*_report.pdf`

## MongoDB (Optional)

If MongoDB is configured in `.env.mongoDB`, results are automatically saved to database.

## Troubleshooting

**Backend not starting?**
- Check if port 8000 is free
- Verify conda environment is activated
- Ensure all dependencies are installed

**Frontend API error?**
- Verify backend is running on port 8000
- Check browser console for CORS errors
- Ensure model files exist in `saved_models/`

**No results showing?**
- Check browser network tab for API response
- Verify CSV format matches requirements
- Look at backend terminal for error logs

## Test Files

Use the provided test datasets:
- `backend/dataset/fraudTest.csv`
- `backend/dataset/fraudTrain.csv`

For quick testing, extract first 100 rows:
```bash
head -n 101 backend/dataset/fraudTest.csv > test_batch.csv
```
