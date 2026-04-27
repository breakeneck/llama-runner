#!/usr/bin/env python3
"""Llama Runner - Web interface for managing llama-server instances."""

import os
import re
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

# ── Load configuration from .env ────────────────────────────────────────
load_dotenv()

HOST = os.getenv('HOST', '192.168.88.29')
PORT = int(os.getenv('PORT', '7878'))
MODEL_DIR = os.path.expanduser(
    os.getenv('MODEL_DIR', '~/.lmstudio/models')
)
LLAMA_SERVER_CMD = os.getenv(
    'LLAMA_SERVER_CMD',
    '/home/server/llama-cpp-turboquant/build/bin/llama-server'
)

# ── Process registry ───────────────────────────────────────────────────
running_processes: dict[str, dict] = {}
"""key=model_path, value={'pid': int, 'port': int, 'proc': Popen, **params}"""

app = Flask(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────

_SPLIT_RE = re.compile(r'-(\d{5})-of-(\d{5})\.gguf$', re.IGNORECASE)


def _scan_models() -> list[dict]:
    """Recursively find .gguf files, skip .mmproj paths, and group tensor-split models."""
    models = []
    base = Path(MODEL_DIR)
    if not base.is_dir():
        return models

    # Collect all .gguf files that are NOT inside .mmproj directories
    all_ggufs = []
    for gguf in sorted(base.rglob('*.gguf')):
        rel = gguf.relative_to(base)
        # Skip anything under a directory named *.mmproj or with .mmproj in path
        if any('.mmproj' in str(parent) for parent in gguf.parents):
            continue
        all_ggufs.append(gguf)

    # Group tensor-split files: treat the whole split set as one model
    seen_split_dirs: set[str] = set()
    split_groups: dict[str, list[Path]] = {}  # dir -> sorted split files

    for gguf in all_ggufs:
        m = _SPLIT_RE.search(gguf.name)
        if m:
            d = str(gguf.parent)
            split_groups.setdefault(d, []).append(gguf)
            seen_split_dirs.add((d, m.group(2)))  # dir + total parts

    for gguf in all_ggufs:
        m = _SPLIT_RE.search(gguf.name)
        if m:
            d = str(gguf.parent)
            key = (d, m.group(2))
            if key in seen_split_dirs and gguf != sorted(split_groups[d])[0]:
                continue  # skip non-first parts of split models
            display_name = f"{gguf.parent.name}-{_SPLIT_RE.sub('', gguf.stem)}"
            models.append({
                'name': display_name,
                'path': str(gguf),
                'split_parts': len(split_groups[d]),
            })
        else:
            rel = gguf.relative_to(base)
            parts = rel.parts  # ('author', 'model-folder', 'file.gguf')
            if len(parts) >= 2:
                author = parts[0]
                filename = Path(parts[-1]).stem  # file without .gguf
                display_name = f"{author}-{filename}"
            else:
                display_name = gguf.name
            models.append({
                'name': display_name,
                'path': str(gguf),
            })
    return models


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _clean_dead():
    """Remove entries whose processes are no longer running."""
    dead = [p for p, v in running_processes.items() if not _is_alive(v['pid'])]
    for p in dead:
        del running_processes[p]


# ── Routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/models')
def api_models():
    """Return the full list of discovered .gguf models."""
    return jsonify(_scan_models())


@app.route('/api/status')
def api_status():
    """Return currently running model information."""
    _clean_dead()
    info = []
    for path, v in running_processes.items():
        info.append({
            'path': path,
            'pid': v['pid'],
            'port': v['port'],
            'ctx_size': v.get('ctx_size', 200000),
            'temp': v.get('temp', 0.2),
            'cache_type_k': v.get('cache_type_k', 'q8_0'),
            'cache_type_v': v.get('cache_type_v', 'q8_0'),
            'n_gpu_layers': v.get('n_gpu_layers', 999),
        })
    return jsonify({'running': info})


@app.route('/api/models/run', methods=['POST'])
def api_run_model():
    """Start llama-server for a given model."""
    data = request.get_json(force=True)
    path = data['path']

    # Validate model file exists
    if not os.path.isfile(path):
        return jsonify({'ok': False, 'error': 'Model file not found'}), 400

    # Check already running
    _clean_dead()
    if path in running_processes:
        return jsonify({'ok': False, 'error': 'Already running'}), 409

    port = int(data.get('port', 12345))
    ctx_size = int(data.get('ctx_size', 200000))
    temp = float(data.get('temp', 0.2))
    cache_k = data.get('cache_type_k', 'q8_0') or 'q8_0'
    cache_v = data.get('cache_type_v', 'q8_0') or 'q8_0'
    n_gpu_layers = int(data.get('n_gpu_layers', 999))

    cmd = [
        LLAMA_SERVER_CMD,
        '-m', path,
        '--flash-attn', 'on',
        '--host', '0.0.0.0',
        '--port', str(port),
        '--temp', str(temp),
        '--cache-type-k', cache_k,
        '--cache-type-v', cache_v,
        '--ctx-size', str(ctx_size),
        '--n-gpu-layers', str(n_gpu_layers),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500

    running_processes[path] = {
        'pid': proc.pid,
        'proc': proc,
        'port': port,
        'ctx_size': ctx_size,
        'temp': temp,
        'cache_type_k': cache_k,
        'cache_type_v': cache_v,
        'n_gpu_layers': n_gpu_layers,
    }

    return jsonify({'ok': True, 'pid': proc.pid, 'port': port})


@app.route('/api/models/stop', methods=['POST'])
def api_stop_model():
    """Stop a running llama-server."""
    data = request.get_json(force=True)
    path = data['path']

    entry = running_processes.pop(path, None)
    if not entry:
        return jsonify({'ok': False, 'error': 'Not found or already stopped'})

    proc = entry['proc']
    try:
        # Kill the entire process group
        os.killpg(os.getpgid(proc.pid), 9)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    return jsonify({'ok': True})


@app.route('/api/vram')
def api_vram():
    """Parse nvidia-smi output and return GPU memory info."""
    gpus = []
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 6:
                continue
            try:
                gpus.append({
                    'name': parts[1],
                    'used': round(int(parts[2]) / 1024, 1),   # MB → GB
                    'total': round(int(parts[3]) / 1024, 1),
                    'temp': parts[4],
                    'power': parts[5],
                })
            except (ValueError, IndexError):
                continue
    except Exception:
        pass

    return jsonify({'gpus': gpus})


# ── Entrypoint ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"🦙 Llama Runner starting on {HOST}:{PORT}")
    print(f"   Model directory: {MODEL_DIR}")
    app.run(host=HOST, port=PORT, debug=False)
