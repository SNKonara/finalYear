# Always-On Streaming (24/7) Setup Guide

This guide explains how to run NeuroDetect streaming continuously, even when no user has the website open.

## Goal

Current behavior is client-driven:
- Frontend connects to WebSocket.
- Frontend sends `start_stream`.
- Backend streams only while client session is active.

Target behavior is server-driven:
- Backend processes transactions continuously in the background.
- Results are saved/broadcast independent of browser sessions.
- Frontend becomes a viewer/consumer, not the stream owner.

## Recommended Architecture

1. Add a background processing loop in `backend/server/websocket_unified.py`.
2. Start that loop when backend starts (not from frontend command).
3. Persist all outputs to MongoDB (`fraud_results`, alerts collection).
4. Keep WebSocket for live viewing only.
5. If no clients are connected, processing still continues.

## Minimal Backend Changes (Later Implementation)

Use this as your implementation checklist:

1. Create an always-on task
- Add `self.always_on_enabled = True`
- Add `self.always_on_task = None`
- Add `self.last_processed_payload = None`

2. Add a background loop method
- Example: `async def run_always_on_stream(self):`
- Loop through dataset continuously.
- Run detection using selected default model (or all models if needed).
- Save to MongoDB.
- Broadcast to connected clients if present.
- Sleep based on configured stream speed.

3. Start task at server startup
- In `main()`, after creating server instance:
- `server.always_on_task = asyncio.create_task(server.run_always_on_stream())`

4. Keep existing commands for UI compatibility
- `start_stream` and `stop_stream` can be retained as UI controls.
- They should control only client delivery mode, not core processing.

5. Add health/status fields
- Include `always_on_enabled`, `always_on_running`, `last_processed_at` in `get_status`.

## Data Safety and Reliability

For production-like behavior, add:
- try/except around each processing cycle
- automatic retries for transient DB failures
- periodic heartbeat logs
- graceful task cancellation on shutdown

Optional but useful:
- keep `last_processed_index` in MongoDB to resume from previous point after restart
- write periodic checkpoints every N records

## Windows 24/7 Deployment Options

## Option A: NSSM service (recommended on Windows)

1. Install NSSM.
2. Create service:

```powershell
nssm install NeuroDetectWebSocket
```

3. Configure:
- Path: `C:\Users\sachi\miniconda3\envs\fraud-detection\python.exe`
- Startup directory: `C:\finalYear\NeuroDetect\backend`
- Arguments: `server\websocket_unified.py`

4. Start service:

```powershell
nssm start NeuroDetectWebSocket
```

5. Enable auto-start:

```powershell
sc config NeuroDetectWebSocket start= auto
```

## Option B: Task Scheduler

Use when service setup is not available.

- Trigger: At startup
- Run whether user is logged on or not
- Program: `C:\Users\sachi\miniconda3\envs\fraud-detection\python.exe`
- Arguments: `C:\finalYear\NeuroDetect\backend\server\websocket_unified.py`
- Start in: `C:\finalYear\NeuroDetect\backend`
- Enable restart on failure

## Option C: Docker with restart policy

Use if you containerize backend later.

```bash
docker run --restart unless-stopped ...
```

## Suggested Environment Variables

Add these env vars for clean control:

- `ALWAYS_ON_STREAM=true`
- `ALWAYS_ON_MODEL=snn`
- `ALWAYS_ON_SPEED=1.0`
- `ALWAYS_ON_SAVE_TO_DB=true`

## Monitoring Checklist

After enabling always-on mode, verify:

1. Backend process is running after system reboot.
2. `get_status` shows always-on task active.
3. New records are being inserted into MongoDB when no browser is open.
4. Frontend can reconnect and receive recent/live updates.
5. Service auto-recovers after crashes.

## Operational Notes

- This is not complex for your current project; it is a medium-sized backend upgrade.
- Your current code already has most required pieces (dataset loop, model inference, MongoDB write, WebSocket send).
- Main change is moving control from browser session to server lifecycle.

## Rollout Plan (Safe)

1. Implement always-on mode behind env flag (`ALWAYS_ON_STREAM`).
2. Test locally with website closed.
3. Validate MongoDB growth and performance.
4. Deploy as Windows service.
5. Keep old client-driven mode as fallback until stable.

## Related Files

- `backend/server/websocket_unified.py`
- `backend/start_server.py`
- `backend/WEBSOCKET_GUIDE.md`
- `backend/MONGODB_SETUP.md`
