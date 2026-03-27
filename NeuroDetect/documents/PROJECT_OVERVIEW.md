# NeuroDetect – Project Overview

## 1. Project Title
**NeuroDetect: Multi-Model Fraud Detection Platform for Real-Time and Batch Transaction Monitoring**

## 2. Problem Statement
Financial transaction streams contain both legitimate and fraudulent activity, and fraud patterns evolve over time. Traditional single-model systems can struggle to balance sensitivity, interpretability, and operational usability. This project addresses that gap by building a practical fraud detection platform that combines multiple deep learning approaches, supports both real-time and offline workflows, and provides analyst-friendly outputs for investigation.

## 3. Project Objective
The objective of NeuroDetect is to design and implement an end-to-end fraud analytics system that:
- Detects suspicious transactions using multiple ML/DL models.
- Supports both real-time streaming inference and batch CSV analysis.
- Enables model comparison (Autoencoder, LSTM, SNN) under a unified interface.
- Stores results for auditability and investigation.
- Provides role-based dashboards for different operational users.

## 4. System Overview
NeuroDetect is implemented as a full-stack architecture with:
- **Frontend dashboard** for monitoring, model-specific views, alerts, reports, and administration.
- **Backend inference services** for WebSocket real-time detection and FastAPI batch processing.
- **Model layer** containing three fraud-detection approaches (AE, LSTM, SNN).
- **Persistence layer** using MongoDB plus local result artifacts (JSON/CSV/PDF).

## 5. Core Features Implemented
### 5.1 Real-Time Fraud Detection
- WebSocket server streams transactions and performs online scoring.
- Clients can control stream operations (start/stop/speed) and request model/status metadata.
- Dynamic model selection is supported for operational switching across AE/LSTM/SNN.

### 5.2 Batch Fraud Processing
- CSV upload endpoint for offline scoring.
- Model selection and optional threshold handling during batch execution.
- Automatic generation of JSON/CSV outputs and PDF reports.
- Batch history and report download APIs for post-analysis workflows.

### 5.3 Alerting and Explainability Support
- High-risk result tracking and alert endpoints.
- SNN pipeline includes transaction-level explainability artifacts (top contributing factors and threshold margin context).

### 5.4 Authentication and Access Control
- Role-based frontend routing with separate analyst/admin/viewer access paths.
- Backend auth endpoints for login/session handling.

### 5.5 Data Persistence and Auditing
- MongoDB integration for fraud results, immediate alerts, and statistics.
- Indexed collections for operational queries and reporting.

## 6. Algorithms and Model Architectures
### 6.1 Autoencoder (Unsupervised Anomaly Detection)
- Input: engineered tabular transaction features.
- Encoder-decoder architecture with latent bottleneck representation.
- Fraud signal derived from reconstruction error against an optimized threshold.
- Strength: anomaly-focused detection behavior.

### 6.2 Enhanced LSTM (Supervised Sequence Classification)
- Bidirectional LSTM with attention mechanism.
- Batch normalization + dropout + dense classification head.
- Uses sequence windows of transactions for temporal pattern learning.
- Strength: strong supervised predictive performance on sequential behavior.

### 6.3 Spiking Neural Network (SNN)
- Fully connected spiking architecture with LIF neurons and temporal accumulation.
- Uses snnTorch and focal-loss-based training for imbalance handling.
- Includes customer-centric threshold policy metadata and explainability context.
- Strength: temporal/event-style representation and robust recall-oriented behavior.

## 7. Frameworks and Technologies Used
### 7.1 Frontend
- React 19
- TypeScript
- Vite
- React Router
- Tailwind CSS
- Lucide React icons

### 7.2 Backend and APIs
- Python
- FastAPI + Uvicorn (REST batch services)
- WebSockets (real-time communication)

### 7.3 Machine Learning / Data Science
- PyTorch
- snnTorch
- scikit-learn
- pandas
- numpy
- joblib
- imbalanced-learn

### 7.4 Data, Storage, and Reporting
- MongoDB (via pymongo)
- JSON/CSV result artifacts
- PDF report generation (reportlab)
- dotenv-based environment configuration

## 8. Implementation Status Summary
Implemented and integrated:
- Multi-model backend loading and inference (AE, LSTM, SNN).
- Real-time WebSocket detection workflow.
- Batch API with file upload, model choice, reporting, and history.
- Role-based frontend pages for model dashboards, alerts, reports, and admin functions.
- MongoDB-backed result logging and retrieval support.

## 9. Project Outcome
NeuroDetect demonstrates a production-style fraud detection workflow that combines multiple neural approaches in one operational platform. The system supports both live monitoring and retrospective batch analysis, making it suitable for analyst workflows, model benchmarking, and final-year research demonstration.

## 10. Future Enhancement Opportunities
- Unified experiment tracking and model registry.
- Automated threshold calibration per segment/customer cohort.
- Drift monitoring and periodic retraining pipelines.
- Stronger security hardening (strict CORS, API auth middleware, rate limits).
- Extended explainability dashboards for all model families.

---

## 11. Chapter 6 Technical Decisions & Implementation Status (Q&A)

### 11.1 Backend Technical Decisions (6.2 + 6.3)

**Q: Why FastAPI instead of Flask or Django?**  
**Answer:** FastAPI was selected because the project requires high-performance API endpoints, typed request handling, and async-friendly design that aligns with real-time streaming requirements. Flask would require more manual structure for typing/validation, and Django would introduce heavier framework overhead for this service-oriented ML pipeline.

**Q: Was it specifically for async + WebSocket?**  
**Answer:** **Yes (partly).** FastAPI is used for async REST APIs, while the real-time channel is implemented using Python `asyncio` + `websockets` in a unified streaming server. The async-first stack is a key design choice.

**Q: Are Pydantic models used for request validation?**  
**Answer:** **Yes (implemented).** Pydantic `BaseModel` is used for request payloads such as login and role updates.

**Q: Is dependency injection used?**  
**Answer:** **Partially / minimal.** FastAPI-style advanced dependency injection (`Depends`) is not heavily used. Authentication and authorization checks are mainly implemented through helper functions and header parsing.

**Q: Is JWT used, or session-based auth?**  
**Answer:** **Session-based token authentication (implemented), not JWT.** A random token is issued at login, token hashes are stored in MongoDB, and bearer token validation is done per request.

**Q: Are passwords hashed? Which library?**  
**Answer:** **Yes (implemented).** Passwords are hashed with `hashlib.pbkdf2_hmac` (SHA-256, salted, iterative PBKDF2). **bcrypt/passlib are not currently used.**

**Q: Is CORS configured?**  
**Answer:** **Yes (implemented).** `CORSMiddleware` is configured. Current setup allows all origins (`*`) for development convenience.

**Q: Is Uvicorn running async workers or default config?**  
**Answer:** **Default/single-process config (implemented).** Uvicorn is started with host/port/reload/log settings; explicit multi-worker configuration is not set.

---

### 11.2 WebSocket Streaming

**Q: Is WebSocket fully async?**  
**Answer:** **Yes (implemented).** The streaming server uses `asyncio`, async handlers, async send loops, and async sleep-based pacing.

**Q: Do you handle model warm-up before streaming?**  
**Answer:** **Partially.** Models are preloaded during server startup/init, but there is no dedicated explicit warm-up inference pass.

**Q: Is streaming simulated via timestamp replay?**  
**Answer:** **No (not implemented).** Streaming is dataset row replay with configurable records/sec pacing, not true timestamp delta replay.

**Q: Can user dynamically switch models during active stream?**  
**Answer:** **Partially.** `set_model` exists and updates the selected model per client, but the stream loop captures model at start, so immediate effect generally requires restarting stream.

---

### 11.3 Frontend Architecture

**Q: React Context / Zustand / Redux?**  
**Answer:** **React Context is implemented.** Zustand and Redux are **not implemented** in current codebase.

**Q: How is authentication state managed?**  
**Answer:** Auth state is managed through `AuthContext` + browser `localStorage` (user + bearer token), with backend validation via `/auth/me` and session lifecycle endpoints.

**Q: Is routing protected with role-based guards?**  
**Answer:** **Yes (implemented).** Route-level guard logic checks user role (admin/analyst/viewer) before rendering protected pages.

**Q: Is WebSocket managed through a custom hook?**  
**Answer:** **No (not implemented).** WebSocket lifecycle is managed directly inside page components using `useEffect`, refs, and handler functions.

**Q: Are dashboards component-based per model?**  
**Answer:** **Yes (implemented).** Separate model-specific pages exist for Autoencoder, LSTM, and SNN dashboards.

---

### 11.4 MongoDB & Persistence Layer

**Q: MongoDB local or Atlas?**  
**Answer:** **Configurable (implemented).** Connection URI is loaded from environment; if absent, MongoDB is disabled and the system falls back to local files.

**Q: Are indexes defined?**  
**Answer:** **Yes (implemented).** Indexes exist for key collections/fields such as `transaction_id`, `timestamp`, and fraud flags; auth indexes are also defined.  
**Note:** A dedicated `model_type` index is **not explicitly defined yet**.

**Q: Do you store raw transactions or only predictions?**  
**Answer:** **Both (implemented).** Stored documents include original transaction attributes plus model outputs (score/prediction/risk/explainability where applicable).

**Q: Are batch artifacts in DB or filesystem?**  
**Answer:** **Both (implemented).** Batch summaries and records are persisted in MongoDB; JSON/CSV/PDF artifacts are saved to `results/batch/`.

**Q: Is report metadata persisted?**  
**Answer:** **Yes (implemented).** Batch metadata/statistics and processing logs are stored (e.g., summary collections and log entries).

---

### 11.5 Model Deployment Strategy

**Q: Are models preloaded on startup or lazy-loaded?**  
**Answer:** **Preloaded on startup (implemented).** Batch API startup and WebSocket server initialization both load model assets before serving requests. Lazy loading is **not implemented**.

**Q: Are you using `.pt` files?**  
**Answer:** Main deployment artifacts are PyTorch `.pth` checkpoints (**implemented**). `.pt` is **not the current primary format**.

**Q: Are thresholds config-based or dynamic per request?**  
**Answer:** **Both.** Default thresholds come from saved configs/metadata (**implemented**), and batch endpoint supports optional threshold override (**implemented**).

**Q: Is SNN explainability computed during inference or post-processing?**  
**Answer:** In batch workflow, explainability payload is built **post-score computation** and attached to result rows (**implemented**). Real-time SNN explainability is comparatively lighter and less extensive than batch explainability.

---

### 11.6 Model-Specific Academic Justification

**Q: Why Huber loss for AE?**  
**Answer:** Huber loss offers robustness to outliers while retaining stable gradients, making reconstruction training less sensitive to noisy/extreme transaction values.

**Q: Why sequence length = 10 for LSTM?**  
**Answer:** Sequence length 10 is a practical trade-off between temporal context and latency/memory cost for real-time scoring.

**Q: Why Bidirectional LSTM?**  
**Answer:** Bidirectional modeling improves context capture across sequence windows, helping the classifier learn richer temporal representations.

**Q: Why focal loss (`gamma=2`) for SNN?**  
**Answer:** Fraud datasets are imbalanced; focal loss prioritizes hard minority-class samples and reduces dominance of easy negatives.

**Q: Why 20 time steps for SNN?**  
**Answer:** 20 steps provide enough temporal integration for spiking dynamics while keeping compute manageable for deployment.

**Q: Why 24 input features?**  
**Answer:** The 24-feature set is a curated, shared engineered representation used consistently across AE, LSTM, and SNN for fair comparison and unified serving.

---

### 11.7 Development Environment

- **Python version:** 3.9.25 (Conda environment)
- **Node version:** v22.19.0
- **Operating system:** Windows
- **GPU used:** NVIDIA GeForce RTX 4060 Laptop GPU (used in SNN training metadata)

---

### 11.8 Not Yet Implemented / Partial (Explicit)

- Dedicated timestamp-delta replay streaming (currently fixed-rate dataset replay).
- Full FastAPI dependency injection architecture (`Depends`) across auth/business logic.
- JWT-based auth flow (current design uses server-side session tokens).
- Explicit Uvicorn multi-worker production tuning in startup script.
- Dedicated global MongoDB index on `model_type` for all analytical queries.
- Unified frontend WebSocket custom hook abstraction (logic currently duplicated across pages).

---

## 12. Additional Non-Functional Q&A (Performance, Security, Reliability)

### 12.1 Performance Requirements

**Q: Have you measured average inference latency per transaction?**  
**Answer:** **Yes (approximate, implemented and measured from saved run artifact).**
- Source artifact: `results/websocket_results_20260206_151211.json`
- Including first warm-up outlier: average latency ≈ **8.92 ms**
- Steady-state (excluding first outlier): average latency ≈ **5.68 ms**, median ≈ **5.99 ms**, p95 ≈ **9.01 ms**

**Q: Have you measured transactions per second during streaming?**  
**Answer:** **Yes (observed).** In the sampled run, observed throughput was ≈ **1.01 TPS** (aligned with default stream pacing of ~1 record/sec).

**Q: Is system stable at 20–50 TPS?**  
**Answer:** **Not yet formally validated.** No dedicated load/stress benchmark report is currently implemented for 20–50 TPS sustained operation.

---

### 12.2 Security & Reliability

**Q: Does system automatically expire tokens?**  
**Answer:** **Not yet implemented.** Session validity is tracked via active/inactive flags, but explicit TTL-based token expiry is not enforced.

**Q: Is logout invalidation implemented?**  
**Answer:** **Yes (implemented).** Logout marks matching session tokens inactive (revoked).

**Q: Is rate limiting implemented?**  
**Answer:** **No (not yet implemented).**

**Q: Is input validation applied to CSV uploads?**  
**Answer:** **Partially implemented.** Model type and parsing flow are validated; however, strict schema/file-size/type constraints and comprehensive sanitization are not fully enforced yet.

---

### 12.3 Data & Logging

**Q: Are all predictions logged in MongoDB during real-time streaming?**  
**Answer:** **Conditionally yes.** Predictions are logged when MongoDB is connected and detection succeeds; if DB is unavailable or detection fails, complete logging is not guaranteed.

**Q: Are audit logs immutable or editable?**  
**Answer:** **Not immutable by strict policy yet.** Logs are stored in MongoDB, but write-once/append-only controls are not formally enforced.

**Q: Is explainability available only for SNN or all models?**  
**Answer:** **Detailed explainability is implemented for SNN only.** AE and LSTM currently provide score/risk outputs without equivalent factor-level explanation payloads.

---

### 12.4 Non-Functional Clarification

**Q: Does frontend support responsive design (mobile)?**  
**Answer:** **Yes (implemented).** Responsive breakpoints are defined in multiple page CSS files (e.g., max-width media queries).

**Q: Is system designed for single-node deployment only?**  
**Answer:** **Currently yes (single-node oriented).** Distributed service orchestration and horizontal scaling setup are not yet implemented.

**Q: Is GPU required for inference or only for training?**  
**Answer:** **GPU is optional for inference and primarily beneficial for training.** Runtime supports CPU fallback when CUDA is unavailable.
