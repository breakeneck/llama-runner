#!/usr/bin/env python3
"""Llama Runner - Web interface for managing llama-server instances."""

import json
import os
import re
import signal
import subprocess
from pathlib import Path

# ── Log directory: persistent per-model logs ─────────────────────────────
LOG_DIR = Path(__file__).parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)


def _log_path_for_model(model_path: str) -> Path:
    """Return a persistent log path like logs/<modelname>.log for a model."""
    stem = Path(model_path).stem  # e.g., 'Qwen2.5-7B-Instruct-Q4_K_M'
    return LOG_DIR / f'{stem}.log'

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
MODEL_PORT = int(os.getenv('MODEL_PORT', '12345'))

# ── Process registry ───────────────────────────────────────────────────
running_processes: dict[str, dict] = {}
"""key=model_path, value={'pid': int, 'port': int, 'proc': Popen, **params}"""

app = Flask(__name__)

# ── Per-model parameter persistence ─────────────────────────────────────
_PARAMS_FILE = Path(__file__).parent / 'model_params.json'

_DEFAULT_PARAMS = {
    'ctx_size': 100,          # shorthand: 100 means 100k
    'temp': 0.2,
    'cache_type_k': 'q8_0',
    'cache_type_v': 'q8_0',
    'n_gpu_layers': 999,
}

# Which params have enable/disable toggles (port excluded — always on)
_TOGGLED_PARAMS = {'ctx_size', 'temp', 'cache_type_k', 'cache_type_v', 'n_gpu_layers'}


def _load_params() -> dict:
    """Load saved per-model parameters from disk."""
    if _PARAMS_FILE.exists():
        try:
            return json.loads(_PARAMS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_params(params: dict):
    """Persist per-model parameters to disk."""
    _PARAMS_FILE.write_text(json.dumps(params, indent=2))


def get_model_params(path: str) -> dict:
    """Get saved params for a model path (falls back to defaults)."""
    all_params = _load_params()
    saved = all_params.get(path, {})
    result = {}
    for key, default in _DEFAULT_PARAMS.items():
        val = saved.get(key, default)
        if key == 'ctx_size':
            val = int(val)
        elif key == 'temp':
            val = float(val)
        elif key == 'n_gpu_layers':
            val = int(val)
        else:
            val = str(val)
        result[key] = val
        # enable flag defaults to True unless explicitly saved otherwise
        # cache_type_k and cache_type_v default to False (disabled)
        if key in _TOGGLED_PARAMS:
            cache_keys = {'cache_type_k', 'cache_type_v'}
            default_enabled = False if key in cache_keys else True
            result[f'{key}_enabled'] = bool(saved.get(f'{key}_enabled', default_enabled))
    return result


def save_model_params(path: str, params: dict):
    """Save (merge) params for a specific model path to disk."""
    all_params = _load_params()
    all_params[path] = {**all_params.get(path, {}), **params}
    _save_params(all_params)


# ── Helpers ─────────────────────────────────────────────────────────────

_SPLIT_RE = re.compile(r'-(\d{5})-of-(\d{5})\.gguf$', re.IGNORECASE)


def _scan_models() -> list[dict]:
    """Recursively find .gguf files, skip .mmproj paths, and group tensor-split models."""
    models = []
    base = Path(MODEL_DIR)
    if not base.is_dir():
        return models

    # Collect all .gguf files that are NOT inside directories ending with .mmproj
    all_ggufs = []
    for gguf in sorted(base.rglob('*.gguf')):
        rel = gguf.relative_to(base)
        # Skip mmproj files: directories named *.mmproj or files whose stem starts with 'mmproj'
        if any(parent.name.endswith('.mmproj') for parent in gguf.parents):
            continue
        if gguf.stem.startswith('mmproj'):
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
        mtime = int(gguf.stat().st_mtime * 1000)  # ms for JS Date
        if m:
            d = str(gguf.parent)
            key = (d, m.group(2))
            if key in seen_split_dirs and gguf != sorted(split_groups[d])[0]:
                continue  # skip non-first parts of split models
            display_name = f"{gguf.parent.name}-{_SPLIT_RE.sub('', gguf.stem)}"
            # Sum sizes of all split parts
            total_bytes = sum(f.stat().st_size for f in split_groups[d])
            size_gb = round(total_bytes / (1024 ** 3), 2)
            models.append({
                'name': display_name,
                'path': str(gguf),
                'mtime': mtime,
                'size_gb': size_gb,
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
            size_gb = round(gguf.stat().st_size / (1024 ** 3), 2)
            models.append({
                'name': display_name,
                'path': str(gguf),
                'mtime': mtime,
                'size_gb': size_gb,
            })
    # Sort by last_run timestamp (if saved), fallback to mtime, newest first
    all_params = _load_params()
    models.sort(key=lambda m: all_params.get(m.get('path', ''), {}).get('last_run', m.get('mtime', 0)), reverse=True)
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


def _stop_all_models():
    """Kill all running llama-server processes."""
    for path, entry in list(running_processes.items()):
        proc = entry['proc']
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            print(f"  Stopped model: {path} (pid {proc.pid})")
        except Exception as e:
            print(f"  Failed to stop {path}: {e}")
    running_processes.clear()


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM: stop all models and exit."""
    print("\\n🛑 Shutting down Llama Runner...")
    _stop_all_models()
    print("✓ All models stopped. Goodbye!")
    os._exit(0)


# ── Routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', MODEL_PORT=MODEL_PORT)


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

    # Port is global from env — ignore any per-model value
    port = MODEL_PORT

    cmd = [
        'stdbuf', '-oL', '-eL',  # force line-buffered stdout/stderr for real-time log reading
        LLAMA_SERVER_CMD,
        '-m', path,
        '--flash-attn', 'on',
        '--host', '0.0.0.0',
        '--port', str(port),
    ]

    # Build flags only for enabled options
    if data.get('ctx_size_enabled'):
        ctx_val = int(data.get('ctx_size', 100)) * 1_000   # shorthand: 10 → 10k
        cmd.extend(['--ctx-size', str(ctx_val)])

    if data.get('temp_enabled'):
        cmd.extend(['--temp', str(float(data.get('temp', 0.2)))])

    if data.get('cache_type_k_enabled'):
        cache_k = (data.get('cache_type_k') or 'q8_0')
        cmd.extend(['--cache-type-k', cache_k])

    if data.get('cache_type_v_enabled'):
        cache_v = (data.get('cache_type_v') or 'q8_0')
        cmd.extend(['--cache-type-v', cache_v])

    if data.get('n_gpu_layers_enabled'):
        cmd.extend(['--n-gpu-layers', str(int(data.get('n_gpu_layers', 999)))])

    # Use a persistent log file named after the model (overwritten each run)
    log_path = _log_path_for_model(path)
    log_file = open(log_path, 'w')

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
    except Exception as exc:
        log_file.close()
        return jsonify({'ok': False, 'error': str(exc)}), 500

    running_processes[path] = {
        'pid': proc.pid,
        'proc': proc,
        'port': port,
        'log_path': str(log_path),
        **{k: data[k] for k in ('ctx_size', 'temp', 'cache_type_k', 'cache_type_v', 'n_gpu_layers') if k in data},
    }

    # Record last_run timestamp for ordering
    save_model_params(path, {'last_run': int(__import__('time').time() * 1000)})

    return jsonify({'ok': True, 'pid': proc.pid, 'port': port})


@app.route('/api/model-log/<path:path>')
def api_model_log(path):
    """Return log for a model. Works for both running and previously-run models."""
    # First check running process log_path
    log_path = None
    entry = running_processes.get(path)
    if entry:
        log_path = entry.get('log_path')

    # Fall back to derived persistent log path
    if not log_path or not os.path.isfile(log_path):
        derived = _log_path_for_model(path)
        if derived.is_file():
            log_path = str(derived)

    if not log_path or not os.path.isfile(log_path):
        return jsonify({'lines': ['No log file available']}), 200

    try:
        with open(log_path, 'r', errors='replace') as f:
            lines = [l.rstrip('\n') for l in f.readlines()]
    except Exception:
        lines = []

    return jsonify({'lines': lines})


@app.route('/api/model-info/<path:path>')
def api_model_info(path):
    """Return runtime info (layers loaded) for a running model by parsing log."""
    # Reconstruct full path: the <path:> converter strips leading slashes
    full_path = '/' + path if not path.startswith('/') else path

    entry = running_processes.get(full_path)

    # Find log_path: try running process first, then derive from model path
    log_path = None
    if entry:
        log_path = entry.get('log_path')

    # Fall back to derived persistent log path (same as api_model_log)
    if not log_path or not os.path.isfile(log_path):
        derived = _log_path_for_model(full_path)
        if derived.is_file():
            log_path = str(derived)

    if not log_path or not os.path.isfile(log_path):
        return jsonify({'error': 'No log file available'}), 404

    layers_loaded = None
    layers_total = None
    vram_used_per_gpu = []
    try:
        with open(log_path, 'r', errors='replace') as f:
            lines = f.readlines()

        for line in lines:
            # Match "load_tensors: offloaded 65/65 layers to GPU"
            m = re.search(r'offloaded\s+(\d+)/(\d+)\s+layers?\s+to\s+GPU', line)
            if m:
                layers_loaded = int(m.group(1))
                layers_total = int(m.group(2))
            # VRAM per GPU summary like "GPU0: X.XX GiB"
            m2 = re.search(r'GPU\d+\:\s+([\d.]+)\s+GiB', line)
            if m2:
                vram_used_per_gpu.append(float(m2.group(1)))

    except Exception:
        pass

    return jsonify({
        'layers_loaded': layers_loaded,
        'layers_total': layers_total,
        'vram_per_gpu_gib': vram_used_per_gpu,
    })


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


@app.route('/api/models/params/<path:path>', methods=['GET'])
def api_get_model_params(path):
    """Return saved parameters for a specific model path."""
    params = get_model_params(path)
    return jsonify(params)


@app.route('/api/models/params', methods=['POST'])
def api_save_model_params():
    """Save (merge) parameters for a model path to disk."""
    data = request.get_json(force=True)
    path = data['path']
    params = {k: v for k, v in data.items() if k != 'path'}
    save_model_params(path, params)
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
                    'used': round(int(parts[2]) / 1024, 2),   # MB → GB
                    'total': round(int(parts[3]) / 1024, 2),
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
    # Register signal handlers for graceful shutdown (Ctrl+C / kill)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print(f"🦙 Llama Runner starting on {HOST}:{PORT}")
    print(f"   Model directory: {MODEL_DIR}")
    app.run(host=HOST, port=PORT, debug=False)
