# NeuroDetect

NeuroDetect is an AI-powered fraud detection and investigation platform built for real-time monitoring, batch analysis, and collaborative case handling.

It combines:
- A React + TypeScript frontend for dashboards and investigation workflows
- A FastAPI backend for APIs, model orchestration, and role-based operations
- ML models (Autoencoder, LSTM, SNN) for anomaly and fraud detection
- Dedicated model code folders including backend/AEmodel and backend/LSTMmodel
- MongoDB for persistence of users, alerts, and investigation data

## Project Overview

The system is designed to support multiple user roles and end-to-end fraud operations:
- Admins manage users, roles, thresholds, and global settings
- Analysts investigate alerts and escalate complex cases
- Senior analysts handle escalations and advanced review
- Viewers access read-only dashboards and reports

Core capabilities:
- Real-time transaction scoring via websocket pipeline
- Batch fraud analysis from uploaded datasets
- Investigation queue and escalation workflow
- Optional notification integration (for example, SMS)
- Configurable thresholds and model-backed decision support

## Repository Structure

```text
finalYear/
├─ NeuroDetect/                # Main application (frontend + backend)
│  ├─ src/                     # React frontend source
│  ├─ backend/                 # FastAPI services, ML integration, scripts
│  │  ├─ AEmodel/              # Autoencoder model implementation
│  │  ├─ LSTMmodel/            # LSTM model implementation
│  │  └─ SNNmodel/             # Spiking neural network model implementation
│  ├─ public/                  # Frontend static assets
│  ├─ saved_models/            # Trained model artifacts
│  ├─ snn_models/              # SNN model artifacts
│  └─ results/                 # Runtime/generated app results
├─ documents/                  # Guides and operational documentation
├─ testing/                    # Testing documentation and results
├─ results/                    # Root-level generated reports/manuals
└─ USER_MANUAL.md              # End-user operational manual
```

## Tech Stack

- Frontend: React, TypeScript, Vite, Tailwind CSS
- Backend: Python, FastAPI
- Database: MongoDB
- ML: PyTorch-based model artifacts and inference workflows

## Quick Start

### 1) Frontend Setup

```bash
cd NeuroDetect
npm install
npm run dev
```

Frontend runs at `http://localhost:5173` by default.

### 2) Backend Setup

```bash
cd NeuroDetect/backend
conda create -n fraud-detection python=3.10
conda activate fraud-detection
pip install -r requirement.txt
python start_all_servers.py
```


### 3) Database

Run MongoDB locally (or point to a remote cluster) and configure connection settings in backend environment configuration.

## Running the Platform

1. Start MongoDB
2. Start backend services
3. Start frontend dev server
4. Open the UI and log in with configured user credentials

## Documentation

- User manual: `USER_MANUAL.md`
- Operational guides: `documents/`
- Testing docs and outcomes: `testing/`


