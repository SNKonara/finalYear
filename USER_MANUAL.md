# NeuroDetect User Manual

## Overview
NeuroDetect is a full-stack fraud detection and investigation platform combining a React/TypeScript frontend, FastAPI backend, and advanced machine learning models. It supports role-based access, real-time and batch fraud detection, and escalation workflows for analysts and admins.

---

## 1. Project Structure

- **NeuroDetect/**: Main application folder
  - `src/`: Frontend React code (components, pages, auth, assets)
  - `backend/`: Python FastAPI backend, ML models, scripts
  - `public/`: Static assets for frontend
  - `saved_models/`: Trained model files (PyTorch, JSON)
  - `snn_models/`: SNN model artifacts (referenced only)
  - `results/`: Output results, logs, and this manual
- **documents/**: User and admin guides (including this manual)
- **testing/**: Test documentation and results

---

## 2. Prerequisites

- **Node.js** (v18+ recommended)
- **Python** (3.10+ recommended)
- **Conda** (recommended for Python env management)
- **MongoDB** (local or remote instance)
- (Optional) **Twilio** account for SMS notifications

---

## 3. Setup Instructions

### 3.1. Frontend (React)
1. Open terminal in `NeuroDetect/`
2. Install dependencies:
   ```sh
   npm install
   ```
3. Start the development server:
   ```sh
   npm run dev
   ```
   - App runs at [http://localhost:5173](http://localhost:5173) by default.

### 3.2. Backend (FastAPI, ML Models)
1. Open terminal in `NeuroDetect/backend/`
2. (Recommended) Create and activate a conda environment:
   ```sh
   conda create -n fraud-detection python=3.10
   conda activate fraud-detection
   ```
3. Install Python dependencies:
   ```sh
   pip install -r requirement.txt
   ```
4. Start all backend servers:
   ```sh
   python start_all_servers.py
   ```
   - This launches API, batch, and websocket servers.

### 3.3. Database (MongoDB)
- Ensure MongoDB is running locally or update connection string in backend config.
- See `documents/MONGODB_SETUP.md` for details.

### 3.4. Environment Variables
- Configure `.env` files for sensitive settings (Twilio, DB URI, etc.) as needed.
- See `documents/ADMIN_GUIDE.md` for environment variable details.

---

## 4. Usage Guide

### 4.1. User Roles
- **Admin**: Manage users, roles, thresholds, and view all investigations.
- **Analyst**: Investigate fraud alerts, escalate to senior analyst.
- **Senior Analyst**: Handle escalated cases, receive notifications.
- **Viewer**: Read-only access to results and dashboards.

### 4.2. Authentication
- Login with email and password.
- Admins can add users and assign roles/phone numbers.

### 4.3. Fraud Detection
- **Real-time**: Websocket-based streaming detection.
- **Batch**: Upload CSV for batch analysis (see BATCH_API_README.md).
- Results appear in the dashboard and can be exported.

### 4.4. Escalation Workflow
- Analysts can escalate suspicious cases to senior analysts.
- Senior analysts receive notifications (SMS if enabled).
- Admins can monitor all escalations and override thresholds.

### 4.5. Dark/Light Mode
- Toggle theme in the UI; all pages support both modes.


---

## 5. Troubleshooting

- **Frontend build errors**: Ensure Node.js version is correct, run `npm install` again.
- **Backend import errors**: Check Python environment, reinstall requirements.
- **Database connection issues**: Verify MongoDB URI and server status.
- **Model loading errors**: Ensure model files exist in `saved_models/` and paths are correct.
- **SMS not sent**: Check Twilio credentials and environment variables.

---

## 9. Test User Credentials

Use the following test credentials to log in:

- **Admin**
  - Email: admin@neurodetect.ai
  - Password: admin123
- **Analyst**
  - Email: analyst@neurodetect.ai
  - Password: analyst123
- **Senior Analyst**
  - Email: ckulasinghe@neurodetect.ai
  - Password: chamodi123
- **Viewer**
  - Email: viewer@neurodetect.ai
  - Password: viewer123






