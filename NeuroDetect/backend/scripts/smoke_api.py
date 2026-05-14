"""
Simple smoke tests for critical NeuroDetect API routes.

Usage:
    python backend/scripts/smoke_api.py --base-url http://127.0.0.1:8000
"""

import argparse
import http.client
import json
import mimetypes
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
SAMPLE_BATCH_CSV_PATH = BACKEND_DIR / 'dataset' / 'fraudTest_sample20.csv'


def http_json(method: str, url: str, payload: Optional[dict] = None, headers: Optional[dict] = None):
    body = None
    req_headers = {'Accept': 'application/json'}
    if headers:
        req_headers.update(headers)

    if payload is not None:
        body = json.dumps(payload).encode('utf-8')
        req_headers['Content-Type'] = 'application/json'

    request = Request(url=url, data=body, method=method.upper(), headers=req_headers)
    with urlopen(request, timeout=20) as response:
        raw = response.read().decode('utf-8')
        return response.status, json.loads(raw) if raw else {}


def post_multipart(url: str, fields: Dict[str, str], files: Dict[str, Tuple[str, bytes, str]]):
    parsed = urlparse(url)
    boundary = f'----NeuroDetectBoundary{uuid.uuid4().hex}'
    boundary_bytes = boundary.encode('ascii')

    body = bytearray()

    for key, value in fields.items():
        body.extend(b'--' + boundary_bytes + b'\r\n')
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode('utf-8'))
        body.extend(str(value).encode('utf-8'))
        body.extend(b'\r\n')

    for field_name, (filename, file_bytes, content_type) in files.items():
        body.extend(b'--' + boundary_bytes + b'\r\n')
        body.extend(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode('utf-8')
        )
        body.extend(f'Content-Type: {content_type}\r\n\r\n'.encode('utf-8'))
        body.extend(file_bytes)
        body.extend(b'\r\n')

    body.extend(b'--' + boundary_bytes + b'--\r\n')

    conn_cls = http.client.HTTPSConnection if parsed.scheme == 'https' else http.client.HTTPConnection
    conn = conn_cls(parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80), timeout=30)
    try:
        path = parsed.path or '/'
        if parsed.query:
            path = f"{path}?{parsed.query}"

        conn.request(
            'POST',
            path,
            body=bytes(body),
            headers={
                'Content-Type': f'multipart/form-data; boundary={boundary}',
                'Accept': 'application/json',
                'Content-Length': str(len(body)),
            },
        )
        response = conn.getresponse()
        raw = response.read().decode('utf-8')
        data = json.loads(raw) if raw else {}
        return response.status, data
    finally:
        conn.close()


def build_sample_csv() -> Tuple[str, bytes]:
    if SAMPLE_BATCH_CSV_PATH.exists():
        return SAMPLE_BATCH_CSV_PATH.name, SAMPLE_BATCH_CSV_PATH.read_bytes()

    csv_text = (
        'trans_date_trans_time,cc_num,merchant,category,amt,first,last,gender,street,city,state,zip,lat,long,city_pop,job,dob,trans_num,unix_time,merch_lat,merch_long,is_fraud\n'
        '2024-01-01 12:00:00,4111111111111111,fraud_Test Merchant,gas_transport,100.00,Test,User,M,123 Main St,New York,NY,10001,40.7128,-74.0060,8419600,Engineer,1990-01-01,txn_smoke_001,1704110400,40.7580,-73.9855,0\n'
        '2024-01-01 12:05:00,4111111111111111,fraud_Test Merchant,shopping_pos,250.50,Test,User,F,123 Main St,New York,NY,10001,40.7128,-74.0060,8419600,Engineer,1990-01-01,txn_smoke_002,1704110700,40.7306,-73.9352,0\n'
    )
    return 'sample.csv', csv_text.encode('utf-8')


def run_model_smoke(base_url: str, model_type: str) -> List[str]:
    failures: List[str] = []

    print(f'2) Checking /batch/threshold for {model_type} ...')
    try:
        status, payload = http_json(
            'POST',
            f"{base_url}/batch/threshold",
            payload={
                'model_type': model_type,
                'threshold': 0.5,
                'persist': False,
                'reason': 'smoke test',
            },
        )
        if status != 200 or not payload.get('success'):
            failures.append(f"/batch/threshold ({model_type}) failed: status={status}, payload={payload}")
        else:
            print(f"   OK: {status}, threshold={payload.get('threshold')}")
    except Exception as error:
        failures.append(f"/batch/threshold ({model_type}) failed: {error}")

    print(f'3) Checking /batch/process for {model_type} ...')
    try:
        sample_name, sample_bytes = build_sample_csv()
        content_type = mimetypes.guess_type(sample_name)[0] or 'text/csv'
        status, payload = post_multipart(
            f"{base_url}/batch/process",
            fields={'model_type': model_type, 'threshold': '0.5'},
            files={'file': (sample_name, sample_bytes, content_type)},
        )
        if status != 200 or not payload.get('success'):
            failures.append(f"/batch/process ({model_type}) failed: status={status}, payload={payload}")
        else:
            stats = payload.get('statistics', {})
            print(f"   OK: {status}, total={stats.get('total')}, fraud_count={stats.get('fraud_count')}")
    except Exception as error:
        failures.append(f"/batch/process ({model_type}) failed: {error}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description='Run smoke tests for key NeuroDetect API endpoints.')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000', help='FastAPI base URL')
    parser.add_argument('--model-type', default='lstm', choices=['autoencoder', 'lstm', 'snn'])
    parser.add_argument('--all-models', action='store_true', help='Run batch smoke tests for autoencoder, lstm, and snn')
    args = parser.parse_args()

    base_url = args.base_url.rstrip('/')
    failures: List[str] = []
    model_types = ['autoencoder', 'lstm', 'snn'] if args.all_models else [args.model_type]

    print('1) Checking /investigations/alerts ...')
    try:
        status, payload = http_json('GET', f"{base_url}/investigations/alerts?hours=24&status=open&limit=5")
        if status != 200:
            failures.append(f"/investigations/alerts returned {status}")
        else:
            print(f"   OK: {status}, alerts={len(payload.get('alerts', []))}")
    except Exception as error:
        failures.append(f"/investigations/alerts failed: {error}")

    for model_type in model_types:
        failures.extend(run_model_smoke(base_url, model_type))

    if failures:
        print('\nSmoke tests failed:')
        for item in failures:
            print(f" - {item}")
        return 1

    print('\nAll smoke tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
