"""Capture raw Qwen3 completions for one local OpenHands rollout."""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class TraceServer(ThreadingHTTPServer):
    def __init__(self, address, handler, upstream, trace_path, run_id, instance_id, policy_revision):
        super().__init__(address, handler)
        self.upstream = upstream.rstrip('/')
        self.trace_path = trace_path
        self.run_id = run_id
        self.instance_id = instance_id
        self.policy_revision = policy_revision
        self.trace_lock = threading.Lock()
        self.sequence = 0

    def record(self, entry):
        with self.trace_lock:
            self.sequence += 1
            entry['sequence'] = self.sequence
            with self.trace_path.open('a', encoding='utf-8') as trace:
                trace.write(json.dumps(entry, ensure_ascii=False) + '\n')


class TraceHandler(BaseHTTPRequestHandler):
    server: TraceServer

    def do_GET(self):
        self.forward(None)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        if length > 16 * 1024 * 1024:
            self.send_error(413)
            return
        body = self.rfile.read(length)
        self.forward(body)

    def forward(self, body):
        started_ns = time.time_ns()
        request_json = None
        if body is not None and self.path.rstrip('/') == '/v1/chat/completions':
            request_json = json.loads(body)
            request_json['logprobs'] = True
            request_json['return_token_ids'] = True
            request_json['return_tokens_as_token_ids'] = True
            body = json.dumps(request_json).encode('utf-8')

        request = urllib.request.Request(
            self.server.upstream + self.path,
            data=body,
            headers={
                'Content-Type': self.headers.get('Content-Type', 'application/json'),
                'Authorization': self.headers.get('Authorization', 'Bearer EMPTY'),
            },
            method=self.command,
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status = response.status
                content_type = response.headers.get('Content-Type', 'application/json')
                payload = response.read()
        except urllib.error.HTTPError as error:
            status = error.code
            content_type = error.headers.get('Content-Type', 'application/json')
            payload = error.read()
        except Exception as error:
            status = 502
            content_type = 'application/json'
            payload = json.dumps({'error': str(error)}).encode('utf-8')

        if request_json is not None:
            try:
                response_json = json.loads(payload)
            except json.JSONDecodeError:
                response_json = {'raw_response': payload.decode('utf-8', errors='replace')}
            self.server.record({
                'run_id': self.server.run_id,
                'instance_id': self.server.instance_id,
                'policy_revision': self.server.policy_revision,
                'started_ns': started_ns,
                'ended_ns': time.time_ns(),
                'status': status,
                'request': request_json,
                'response': response_json,
            })

        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--listen-host', default='127.0.0.1')
    parser.add_argument('--listen-port', type=int, default=18001)
    parser.add_argument('--upstream', default='http://127.0.0.1:18000')
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--instance-id', required=True)
    parser.add_argument('--policy-revision', required=True,
                        help='Immutable model/adapter checkpoint identifier shared by a rollout group')
    args = parser.parse_args()
    if args.trace.exists() and args.trace.stat().st_size:
        parser.error('Trace path already contains data; use a fresh path for each rollout')
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    server = TraceServer(
        (args.listen_host, args.listen_port),
        TraceHandler,
        args.upstream,
        args.trace,
        args.run_id,
        args.instance_id,
        args.policy_revision,
    )
    print(f'Tracing {args.run_id} on {args.listen_host}:{args.listen_port}', flush=True)
    server.serve_forever(poll_interval=0.5)


if __name__ == '__main__':
    main()
