#!/usr/bin/env python3
"""Llama Runner - Web interface for managing llama-server instances."""

import json
import os
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path

# ── Directories ─────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)
OLLAMA_DIR = Path(__file__).parent / 'ollama'
OLLAMA_DIR.mkdir(exist_ok=True)
RESULTS_JSON = Path(__file__).parent / 'results.json'
RESULTS_DIR = Path(__file__).parent / 'results'


def _log_path_for_model(model_path: str) -> Path:
    """Return a persistent log path like logs/<modelname>.log for a model."""
    stem = Path(model_path).stem  # e.g., 'Qwen2.5-7B-Instruct-Q4_K_M'
    return LOG_DIR / f'{stem}.log'


from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory

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

# ── Persistence files ──────────────────────────────────────────────────
_PARAMS_FILE = Path(__file__).parent / 'model_params.json'
_RUN_HISTORY_FILE = Path(__file__).parent / 'run_history.json'
_TEMPLATES_FILE = Path(__file__).parent / 'settings_templates.json'

_DEFAULT_PARAMS = {
    'ctx_size': 100,          # shorthand: 100 means 100k
    'temp': 0.2,
    'top_p': 0.95,
    'top_k': 20,
    'min_p': 0.0,
    'presence_penalty': 0.0,
    'repeat_penalty': 1.0,
    'cache_type_k': 'q8_0',
    'cache_type_v': 'q8_0',
    'n_gpu_layers': 999,
    'speculative_decoding': False,
}

# Which params have enable/disable toggles
_TOGGLED_PARAMS = {
    'ctx_size', 'temp', 'top_p', 'top_k', 'min_p',
    'presence_penalty', 'repeat_penalty',
    'cache_type_k', 'cache_type_v', 'n_gpu_layers', 'speculative_decoding'
}

# Params affected by templates (inference settings)
_INFERENCE_PARAMS = {'temp', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repeat_penalty'}

# Default settings templates (include _enabled flags for each param)
_DEFAULT_TEMPLATES = {
    'Thinking (General)': {
        'temp': 1.0, 'temp_enabled': True,
        'top_p': 0.95, 'top_p_enabled': True,
        'top_k': 20, 'top_k_enabled': True,
        'min_p': 0.0, 'min_p_enabled': False,
        'presence_penalty': 0.0, 'presence_penalty_enabled': False,
        'repeat_penalty': 1.0, 'repeat_penalty_enabled': True,
    },
    'Thinking (Coding)': {
        'temp': 0.6, 'temp_enabled': True,
        'top_p': 0.95, 'top_p_enabled': True,
        'top_k': 20, 'top_k_enabled': True,
        'min_p': 0.0, 'min_p_enabled': False,
        'presence_penalty': 0.0, 'presence_penalty_enabled': False,
        'repeat_penalty': 1.0, 'repeat_penalty_enabled': True,
    },
    'Instruct': {
        'temp': 0.7, 'temp_enabled': True,
        'top_p': 0.80, 'top_p_enabled': True,
        'top_k': 20, 'top_k_enabled': True,
        'min_p': 0.0, 'min_p_enabled': False,
        'presence_penalty': 1.5, 'presence_penalty_enabled': True,
        'repeat_penalty': 1.0, 'repeat_penalty_enabled': True,
    },
}

# KV-cache quantization options
_KV_QUANT_OPTIONS = ['q8_0', 'q4_0', 'turbo4', 'tbqx3', 'turbo3']


# ── Load/Save helpers ──────────────────────────────────────────────────

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


def _load_run_history() -> dict:
    """Load run history from disk. Returns {path: {status, timestamp, error, ...}}."""
    if _RUN_HISTORY_FILE.exists():
        try:
            return json.loads(_RUN_HISTORY_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_run_history(history: dict):
    """Persist run history to disk."""
    _RUN_HISTORY_FILE.write_text(json.dumps(history, indent=2))


def _load_templates() -> dict:
    """Load settings templates, merging with defaults."""
    saved = {}
    if _TEMPLATES_FILE.exists():
        try:
            saved = json.loads(_TEMPLATES_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    # Merge: defaults first, then saved overrides
    return {**_DEFAULT_TEMPLATES, **saved}


def _save_templates(templates: dict):
    """Persist settings templates to disk."""
    _TEMPLATES_FILE.write_text(json.dumps(templates, indent=2))


def update_run_history(path: str, status: str, error: str = None):
    """Record a run attempt in history. Only count as 'running' if actually successful."""
    history = _load_run_history()
    entry = {
        'status': status,          # 'running', 'error', 'stopped', 'starting'
        'timestamp': int(time.time() * 1000),
    }
    if error:
        entry['error'] = error
    history[path] = entry

    # Also update last_run in params for ordering (only for successful runs)
    if status == 'running':
        save_model_params(path, {'last_run': entry['timestamp']})

    _save_run_history(history)


def get_model_run_state(path: str) -> dict:
    """Get the last known run state for a model from persistent history."""
    history = _load_run_history()
    return history.get(path, {})


def get_model_params(path: str) -> dict:
    """Get saved params for a model path (falls back to defaults)."""
    all_params = _load_params()
    saved = all_params.get(path, {})
    result = {}
    for key, default in _DEFAULT_PARAMS.items():
        val = saved.get(key, default)
        if val is None:
            val = default
        if key == 'ctx_size':
            val = int(val)
        elif key in ('temp', 'top_p', 'min_p', 'presence_penalty', 'repeat_penalty'):
            val = float(val)
        elif key in ('top_k', 'n_gpu_layers'):
            val = int(val)
        elif key == 'speculative_decoding':
            val = bool(val)
        else:
            val = str(val)
        result[key] = val
        # enable flag defaults
        if key in _TOGGLED_PARAMS:
            disabled_by_default = {'cache_type_k', 'cache_type_v', 'speculative_decoding'}
            default_enabled = False if key in disabled_by_default else True
            result[f'{key}_enabled'] = bool(saved.get(f'{key}_enabled', default_enabled))
    # Also include last_run and template name if saved
    if 'last_run' in saved:
        result['last_run'] = saved['last_run']
    if 'template' in saved:
        result['template'] = saved['template']
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

    all_params = _load_params()

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
                'last_run': all_params.get(str(gguf), {}).get('last_run', 0),
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
                'last_run': all_params.get(str(gguf), {}).get('last_run', 0),
            })
    # Sort by last_run timestamp (if saved), fallback to mtime, newest first
    models.sort(key=lambda m: m.get('last_run', 0) or m.get('mtime', 0), reverse=True)
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
    print("\n🛑 Shutting down Llama Runner...")
    _stop_all_models()
    print("✓ All models stopped. Goodbye!")
    os._exit(0)


# ── Ollama helpers ─────────────────────────────────────────────────────

def _generate_modelfile(model_path: str, params: dict) -> str:
    """Generate an Ollama Modelfile from model params."""
    lines = [f'FROM {model_path}', '']

    if params.get('ctx_size_enabled', True):
        ctx_val = int(params.get('ctx_size', 100)) * 1000
        lines.append(f'PARAMETER num_ctx {ctx_val}')
    if params.get('temp_enabled', True):
        lines.append(f'PARAMETER temperature {params.get("temp", 0.2)}')
    if params.get('top_p_enabled', True):
        lines.append(f'PARAMETER top_p {params.get("top_p", 0.95)}')
    if params.get('top_k_enabled', True):
        lines.append(f'PARAMETER top_k {int(params.get("top_k", 20))}')
    if params.get('min_p_enabled', True):
        lines.append(f'PARAMETER min_p {params.get("min_p", 0.0)}')
    if params.get('repeat_penalty_enabled', True):
        lines.append(f'PARAMETER repeat_penalty {params.get("repeat_penalty", 1.0)}')
    if params.get('presence_penalty_enabled', True):
        lines.append(f'PARAMETER presence_penalty {params.get("presence_penalty", 0.0)}')

    return '\n'.join(lines)


def _save_ollama_modelfile(model_path: str, params: dict) -> str:
    """Save an Ollama Modelfile to the ollama/ directory."""
    stem = Path(model_path).stem
    modelfile_path = OLLAMA_DIR / f'{stem}.modelfile'
    content = _generate_modelfile(model_path, params)
    modelfile_path.write_text(content)
    return str(modelfile_path)


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
    history = _load_run_history()
    info = []
    for path, v in running_processes.items():
        # Don't report as running if model has error status in history
        hist_state = history.get(path, {})
        if hist_state.get('status') == 'error':
            continue
        info.append({
            'path': path,
            'pid': v['pid'],
            'port': v['port'],
            'ctx_size': v.get('ctx_size', 100),
            'temp': v.get('temp', 0.2),
            'top_p': v.get('top_p', 0.95),
            'top_k': v.get('top_k', 20),
            'min_p': v.get('min_p', 0.0),
            'presence_penalty': v.get('presence_penalty', 0.0),
            'repeat_penalty': v.get('repeat_penalty', 1.0),
            'cache_type_k': v.get('cache_type_k', 'q8_0'),
            'cache_type_v': v.get('cache_type_v', 'q8_0'),
            'n_gpu_layers': v.get('n_gpu_layers', 999),
            'speculative_decoding': v.get('speculative_decoding', False),
        })
    return jsonify({'running': info})


def _build_run_command(data: dict) -> list[str]:
    """Build the llama-server command list from request data."""
    path = data['path']
    port = MODEL_PORT

    cmd = [
        'stdbuf', '-oL', '-eL',
        LLAMA_SERVER_CMD,
        '-m', path,
        '--flash-attn', 'on',
        '--host', '0.0.0.0',
        '--port', str(port),
    ]

    if data.get('ctx_size_enabled'):
        ctx_val = int(data.get('ctx_size', 100)) * 1_000
        cmd.extend(['--ctx-size', str(ctx_val)])

    if data.get('temp_enabled'):
        cmd.extend(['--temp', str(float(data.get('temp', 0.2)))])

    if data.get('top_p_enabled'):
        cmd.extend(['--top-p', str(float(data.get('top_p', 0.95)))])

    if data.get('top_k_enabled'):
        cmd.extend(['--top-k', str(int(data.get('top_k', 20)))])

    if data.get('min_p_enabled'):
        cmd.extend(['--min-p', str(float(data.get('min_p', 0.0)))])

    if data.get('presence_penalty_enabled'):
        cmd.extend(['--presence-penalty', str(float(data.get('presence_penalty', 0.0)))])

    if data.get('repeat_penalty_enabled'):
        cmd.extend(['--repeat-penalty', str(float(data.get('repeat_penalty', 1.0)))])

    if data.get('cache_type_k_enabled'):
        cache_k = data.get('cache_type_k') or 'q8_0'
        cmd.extend(['--cache-type-k', cache_k])

    if data.get('cache_type_v_enabled'):
        cache_v = data.get('cache_type_v') or 'q8_0'
        cmd.extend(['--cache-type-v', cache_v])

    if data.get('n_gpu_layers_enabled'):
        cmd.extend(['--n-gpu-layers', str(int(data.get('n_gpu_layers', 999)))])

    if data.get('speculative_decoding_enabled'):
        cmd.extend(['--spec-type', 'ngram-mod', '--spec-ngram-size-n', '24', '--draft-min', '12', '--draft-max', '48'])

    return cmd


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

    cmd = _build_run_command(data)

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
        **{k: data[k] for k in (
            'ctx_size', 'temp', 'top_p', 'top_k', 'min_p',
            'presence_penalty', 'repeat_penalty',
            'cache_type_k', 'cache_type_v', 'n_gpu_layers', 'speculative_decoding'
        ) if k in data},
    }

    # Clear any previous error state and record as starting
    update_run_history(path, 'starting')

    return jsonify({'ok': True, 'pid': proc.pid, 'port': port})


@app.route('/api/models/run-command', methods=['POST'])
def api_run_command():
    """Return the shell command that would be used to run a model."""
    data = request.get_json(force=True)
    path = data.get('path', '')
    if not path or not os.path.isfile(path):
        return jsonify({'ok': False, 'error': 'Model file not found'}), 400
    cmd = _build_run_command(data)
    return jsonify({'ok': True, 'command': shlex.join(cmd)})


@app.route('/api/models/run-success', methods=['POST'])
def api_run_success():
    """Called by frontend when model confirms it's actually running (server listening detected)."""
    data = request.get_json(force=True)
    path = data.get('path', '')
    if path:
        update_run_history(path, 'running')
    return jsonify({'ok': True})


@app.route('/api/models/run-error', methods=['POST'])
def api_run_error():
    """Called by frontend when model fails to start (error detected in logs).
    Also kills the process and removes it from running_processes."""
    data = request.get_json(force=True)
    path = data.get('path', '')
    error = data.get('error', 'Unknown error')
    if path:
        update_run_history(path, 'error', error)
        # Kill the process if it's still in running_processes
        entry = running_processes.pop(path, None)
        if entry:
            proc = entry['proc']
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    return jsonify({'ok': True})


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

    # Record as stopped in history
    update_run_history(path, 'stopped')

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


@app.route('/api/all-params')
def api_all_params():
    """Return saved parameters for all models (for bulk pre-loading)."""
    all_params = _load_params()
    result = {}
    for path, saved in all_params.items():
        if path.startswith('_'):
            continue  # skip internal keys like _preferences
        result[path] = get_model_params(path)
    return jsonify(result)


@app.route('/api/preferences')
def api_get_preferences():
    """Get user preferences (sort field, sort direction)."""
    all_params = _load_params()
    prefs = all_params.get('_preferences', {})
    return jsonify({
        'sortField': prefs.get('sortField', 'status'),
        'sortDir': prefs.get('sortDir', -1),
    })


@app.route('/api/preferences', methods=['POST'])
def api_save_preferences():
    """Save user preferences (sort field, sort direction)."""
    data = request.get_json(force=True)
    all_params = _load_params()
    all_params['_preferences'] = {
        'sortField': data.get('sortField', 'status'),
        'sortDir': data.get('sortDir', -1),
    }
    _save_params(all_params)
    return jsonify({'ok': True})


@app.route('/api/run-states')
def api_run_states():
    """Return persistent run states for all models."""
    history = _load_run_history()
    # Cross-reference with currently running processes to clear stale states
    _clean_dead()
    changed = False
    for path in list(history.keys()):
        state = history[path]
        if path not in running_processes:
            if state['status'] in ('running', 'starting'):
                # Process died without proper stop - mark as stopped
                history[path]['status'] = 'stopped'
                history[path]['timestamp'] = int(time.time() * 1000)
                changed = True
        else:
            # Process is in running_processes but has error status - kill it
            if state.get('status') == 'error':
                entry = running_processes.pop(path, None)
                if entry:
                    proc = entry['proc']
                    try:
                        os.killpg(os.getpgid(proc.pid), 9)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
    if changed:
        _save_run_history(history)
    return jsonify(history)


@app.route('/api/last-running')
def api_last_running():
    """Return the last running model path (if still valid)."""
    history = _load_run_history()
    last_running = None
    last_time = 0
    for path, state in history.items():
        if state.get('status') == 'running':
            # Check if it's actually still running
            entry = running_processes.get(path)
            if entry and _is_alive(entry['pid']):
                if state.get('timestamp', 0) > last_time:
                    last_running = path
                    last_time = state.get('timestamp', 0)
    return jsonify({'path': last_running})


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


# ── Templates API ──────────────────────────────────────────────────────

@app.route('/api/templates')
def api_get_templates():
    """Get all settings templates."""
    return jsonify(_load_templates())


@app.route('/api/templates', methods=['POST'])
def api_save_templates():
    """Save settings templates."""
    data = request.get_json(force=True)
    _save_templates(data)
    return jsonify({'ok': True})


@app.route('/api/templates/apply', methods=['POST'])
def api_apply_template():
    """Apply a template to a model's params."""
    data = request.get_json(force=True)
    path = data.get('path', '')
    template_name = data.get('template', '')
    templates = _load_templates()
    template = templates.get(template_name)
    if not template:
        return jsonify({'ok': False, 'error': 'Template not found'}), 404
    # Apply template params to model (only inference params)
    save_model_params(path, {**template, 'template': template_name})
    return jsonify({'ok': True, 'params': get_model_params(path)})


# ── Ollama API ─────────────────────────────────────────────────────────

@app.route('/api/ollama/modelfile', methods=['POST'])
def api_ollama_modelfile():
    """Generate and save an Ollama Modelfile for a model."""
    data = request.get_json(force=True)
    path = data.get('path', '')
    params = data.get('params', {})
    if not path:
        return jsonify({'ok': False, 'error': 'No model path provided'}), 400
    # Use saved params if no params provided
    if not params:
        params = get_model_params(path)
    modelfile_path = _save_ollama_modelfile(path, params)
    content = _generate_modelfile(path, params)
    return jsonify({'ok': True, 'modelfile_path': modelfile_path, 'content': content})


@app.route('/api/ollama/modelfile/<path:path>', methods=['GET'])
def api_ollama_get_modelfile(path):
    """Get the saved Ollama Modelfile for a model."""
    stem = Path(path).stem
    modelfile_path = OLLAMA_DIR / f'{stem}.modelfile'
    if not modelfile_path.is_file():
        # Generate on the fly from saved params
        params = get_model_params(path)
        content = _generate_modelfile(path, params)
        return jsonify({'ok': True, 'content': content, 'saved': False})
    content = modelfile_path.read_text()
    return jsonify({'ok': True, 'content': content, 'saved': True})


@app.route('/api/ollama/run', methods=['POST'])
def api_ollama_run():
    """Create a model in ollama from modelfile (ollama serve must be running separately)."""
    data = request.get_json(force=True)
    path = data.get('path', '')
    if not path:
        return jsonify({'ok': False, 'error': 'No model path provided'}), 400
    params = data.get('params', {})
    if not params:
        params = get_model_params(path)
    # Save modelfile
    modelfile_path = _save_ollama_modelfile(path, params)
    # Create ollama model name from file stem
    stem = Path(path).stem
    ollama_name = f'llama-runner:{stem.lower()}'
    # Run ollama create
    try:
        result = subprocess.run(
            ['ollama', 'create', ollama_name, '-f', modelfile_path],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            return jsonify({'ok': False, 'error': f'ollama create failed: {result.stderr}'}), 500
    except FileNotFoundError:
        return jsonify({'ok': False, 'error': 'ollama not found. Install ollama first.'}), 500
    except subprocess.TimeoutExpired:
        return jsonify({'ok': False, 'error': 'ollama create timed out'}), 500
    return jsonify({'ok': True, 'ollama_name': ollama_name, 'modelfile_path': modelfile_path})


@app.route('/api/ollama/list')
def api_ollama_list():
    """List models available in ollama."""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True, text=True, timeout=10
        )
        models = []
        for line in result.stdout.strip().split('\n')[1:]:  # skip header
            if not line.strip():
                continue
            parts = line.split()
            if parts:
                models.append({'name': parts[0]})
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'models': [], 'error': str(e)})


# ── Results API ────────────────────────────────────────────────────────

@app.route('/api/results')
def api_results():
    """Return benchmark results from results.json."""
    if not RESULTS_JSON.exists():
        return jsonify({'results': [], 'metadata': {}})
    try:
        data = json.loads(RESULTS_JSON.read_text())
    except (json.JSONDecodeError, OSError):
        return jsonify({'results': [], 'metadata': {}})
    return jsonify(data)


@app.route('/api/results/<kvcache>')
def api_results_kvcache(kvcache):
    """Return benchmark results for a specific KV-cache quantization."""
    path = Path(__file__).parent / f'results_{kvcache}.json'
    if not path.exists():
        return jsonify({'results': [], 'metadata': {}})
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return jsonify({'results': [], 'metadata': {}})
    return jsonify(data)


@app.route('/results/<path:path>')
def serve_result_file(path):
    """Serve individual result files from the results directory."""
    if not RESULTS_DIR.is_dir():
        return jsonify({'error': 'Results directory not found'}), 404
    try:
        return send_from_directory(RESULTS_DIR, path)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404


@app.route('/api/run-python/<path:path>')
def api_run_python(path):
    """Run a Python result file and return its output (streaming-like with timeout)."""
    file_path = RESULTS_DIR / path
    # Security: ensure the file is inside RESULTS_DIR
    try:
        resolved = file_path.resolve()
        resolved_results = RESULTS_DIR.resolve()
        if not str(resolved).startswith(str(resolved_results)):
            return jsonify({'error': 'Invalid path'}), 403
    except (OSError, ValueError):
        return jsonify({'error': 'Invalid path'}), 403

    if not file_path.is_file():
        return jsonify({'error': 'File not found'}), 404

    import select
    try:
        proc = subprocess.Popen(
            ['python3', str(file_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout_lines = []
        stderr_lines = []
        max_output = 5000  # max chars
        timeout = 30       # seconds
        start = time.time()

        while proc.poll() is None and (time.time() - start) < timeout:
            ready, _, _ = select.select([proc.stdout, proc.stderr], [], [], 0.5)
            for stream in ready:
                line = stream.readline()
                if not line:
                    continue
                if len(''.join(stdout_lines) + ''.join(stderr_lines)) > max_output:
                    break
                if stream is proc.stdout:
                    stdout_lines.append(line)
                else:
                    stderr_lines.append(line)
            if len(''.join(stdout_lines) + ''.join(stderr_lines)) > max_output:
                break

        still_running = proc.poll() is None
        if still_running:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass

        # Drain remaining output
        try:
            remaining_out, remaining_err = proc.communicate(timeout=3)
        except Exception:
            remaining_out = remaining_err = ''

        stdout = ''.join(stdout_lines) + (remaining_out or '')
        stderr = ''.join(stderr_lines) + (remaining_err or '')

        # Trim if too large
        if len(stdout) > max_output:
            stdout = stdout[:max_output] + '\n... (output truncated)'
        if len(stderr) > max_output:
            stderr = stderr[:max_output] + '\n... (stderr truncated)'

        return jsonify({
            'ok': True,
            'stdout': stdout,
            'stderr': stderr,
            'returncode': proc.returncode if proc.returncode is not None else -1,
            'timed_out': still_running,
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


# ── Entrypoint ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Register signal handlers for graceful shutdown (Ctrl+C / kill)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print(f"🦙 Llama Runner starting on {HOST}:{PORT}")
    print(f"   Model directory: {MODEL_DIR}")
    app.run(host=HOST, port=PORT, debug=False)
