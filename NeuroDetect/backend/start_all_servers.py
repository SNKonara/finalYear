#!/usr/bin/env python3
"""
Start both NeuroDetect backend services with one command:
- FastAPI auth/batch API on port 8000
- Unified WebSocket server on port 8765
"""

from __future__ import annotations

import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def _free_port(port: int) -> None:
    """Kill any process listening on *port* so the server can bind cleanly."""
    try:
        import psutil
        for conn in psutil.net_connections(kind='tcp'):
            if conn.laddr.port == port and conn.status == 'LISTEN':
                try:
                    psutil.Process(conn.pid).kill()
                    print(f'  Freed port {port} (killed PID {conn.pid})')
                    time.sleep(0.5)
                except Exception:
                    pass
    except ImportError:
        # psutil not available — try a quick socket probe first
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return  # port already free
        # Fall back to netstat on Windows
        if sys.platform == 'win32':
            import subprocess as _sp
            try:
                out = _sp.check_output(
                    ['netstat', '-ano'], text=True, stderr=_sp.DEVNULL
                )
                for line in out.splitlines():
                    if f':{port} ' in line and 'LISTENING' in line:
                        pid = int(line.strip().split()[-1])
                        _sp.call(['taskkill', '/F', '/PID', str(pid)],
                                 stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
                        print(f'  Freed port {port} (killed PID {pid})')
                        time.sleep(0.5)
                        break
            except Exception:
                pass


def _stream_name(process: subprocess.Popen[str], fallback: str) -> str:
    if process.args and isinstance(process.args, list) and len(process.args) > 2:
        return Path(process.args[2]).name
    return fallback


def main() -> int:
    backend_dir = Path(__file__).resolve().parent
    python_exe = sys.executable

    commands = [
        [python_exe, str(backend_dir / 'server' / 'batch_api.py')],
        [python_exe, str(backend_dir / 'server' / 'websocket_unified.py')],
    ]

    processes: list[subprocess.Popen[str]] = []

    print('\nStarting NeuroDetect backend stack...')
    print('- API server: http://localhost:8000')
    print('- WebSocket server: ws://localhost:8765')

    # Release ports before binding so restarts never fail with EADDRINUSE
    _free_port(8000)
    _free_port(8765)
    time.sleep(0.5)

    try:
        for command in commands:
            process = subprocess.Popen(command, cwd=str(backend_dir))
            processes.append(process)
            script_name = Path(command[-1]).name
            print(f'  Started {script_name} (PID: {process.pid})')

        print('\nBoth services started. Press Ctrl+C to stop all.\n')

        while True:
            for process in processes:
                code = process.poll()
                if code is not None:
                    name = _stream_name(process, 'service')
                    print(f'\n{name} exited unexpectedly with code {code}. Shutting down all services...')
                    raise RuntimeError('A backend service stopped unexpectedly')
            time.sleep(1)

    except KeyboardInterrupt:
        print('\nStopping backend services...')
    except Exception as exc:
        print(f'Error: {exc}')
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()

        # Give processes a moment to exit cleanly, then force kill if needed.
        time.sleep(1.5)
        for process in processes:
            if process.poll() is None:
                process.kill()

        print('All backend services stopped.')

    return 0


if __name__ == '__main__':
    # Ensure Ctrl+C reaches child processes on Windows and POSIX.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    raise SystemExit(main())
