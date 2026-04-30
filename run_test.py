#!/usr/bin/env python3
"""Model benchmarking script for llama-runner.

Tests all discovered models with multiple tasks and temperature configurations,
measuring tokens/sec, total tokens, and total time. Results are saved to
results.json for resumability and displayed in a summary table.
"""

import argparse
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── Configuration ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / 'results'
TASKS_DIR = SCRIPT_DIR / 'tasks'
RESULTS_JSON = SCRIPT_DIR / 'results.json'

# Load .env
load_dotenv(SCRIPT_DIR / '.env')

HOST = os.getenv('HOST', '192.168.88.29')
PORT = int(os.getenv('PORT', '7878'))
MODEL_PORT = int(os.getenv('MODEL_PORT', '12345'))

LLAMA_RUNNER_URL = f'http://{HOST}:{PORT}'
LLAMA_SERVER_URL = f'http://{HOST}:{MODEL_PORT}'

CTX_SIZE = 60  # 60k context

# Temperature configurations (params sent per-request via OpenAI API)
TEMP_CONFIGS = {
    0.2: {
        # temperature only, no additional params
    },
    0.6: {
        'top_p': 0.95,
        'top_k': 20,
        'min_p': 0.0,
        'presence_penalty': 0.0,
        'repeat_penalty': 1.0,
    },
    0.7: {
        'top_p': 0.80,
        'top_k': 20,
        'min_p': 0.0,
        'presence_penalty': 1.5,
        'repeat_penalty': 1.0,
    },
    1.0: {
        'top_p': 0.95,
        'top_k': 20,
        'min_p': 0.0,
        'presence_penalty': 0.0,
        'repeat_penalty': 1.0,
    },
}

MAX_TOKENS = 32768  # Max tokens to generate per request
MODEL_LOAD_TIMEOUT = 600  # seconds to wait for model to load
COMPLETION_TIMEOUT = 600  # seconds per completion request
COOLDOWN_AFTER_STOP = 5  # seconds to wait after stopping a model


# ── Helper Functions ─────────────────────────────────────────────────────

def format_duration(seconds):
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def sanitize_model_name(name):
    """Sanitize model name for use as directory name."""
    return re.sub(r'[/\\:*?"<>|]', '-', name)


def result_key(model_name, task_num, temp):
    """Generate a unique key for a test result."""
    return f"{model_name}|{task_num}|{temp}"


# ── API Functions ────────────────────────────────────────────────────────

def check_llama_runner():
    """Check if llama-runner is accessible."""
    try:
        resp = requests.get(f'{LLAMA_RUNNER_URL}/api/models', timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def get_models():
    """Get list of models from llama-runner API."""
    resp = requests.get(f'{LLAMA_RUNNER_URL}/api/models', timeout=10)
    return resp.json()


def get_running_model():
    """Get the currently running model path, or None."""
    try:
        resp = requests.get(f'{LLAMA_RUNNER_URL}/api/status', timeout=5)
        data = resp.json()
        running = data.get('running', [])
        if running:
            return running[0]['path']
    except Exception:
        pass
    return None


def start_model(model_path):
    """Start a model via llama-runner API with ctx_size=60k."""
    data = {
        'path': model_path,
        'ctx_size': CTX_SIZE,
        'ctx_size_enabled': True,
        # Disable server-level sampling params — we'll set per-request
        'temp_enabled': False,
        'top_p_enabled': False,
        'top_k_enabled': False,
        'min_p_enabled': False,
        'presence_penalty_enabled': False,
        'repeat_penalty_enabled': False,
        # KV cache quantization
        'cache_type_k': 'q8_0',
        'cache_type_k_enabled': True,
        'cache_type_v': 'q8_0',
        'cache_type_v_enabled': True,
        # GPU layers
        'n_gpu_layers': 999,
        'n_gpu_layers_enabled': True,
    }
    try:
        resp = requests.post(
            f'{LLAMA_RUNNER_URL}/api/models/run',
            json=data,
            timeout=30,
        )
        return resp.json()
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def stop_model(model_path):
    """Stop a running model via llama-runner API."""
    data = {'path': model_path}
    try:
        resp = requests.post(
            f'{LLAMA_RUNNER_URL}/api/models/stop',
            json=data,
            timeout=10,
        )
        return resp.json()
    except Exception:
        return {'ok': False}


def wait_for_model_ready(max_wait=MODEL_LOAD_TIMEOUT):
    """Wait for the model to be ready by polling the health endpoint."""
    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = requests.get(f'{LLAMA_SERVER_URL}/health', timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('status') == 'ok':
                    return True
        except Exception:
            pass
        elapsed = time.time() - start
        print(f"\r⏳ Waiting for model... {format_duration(elapsed)}", end='', flush=True)
        time.sleep(3)
    print()
    return False


def send_completion(prompt, temp, temp_config):
    """Send a completion request and measure performance.

    Returns (result_dict, elapsed_time). result_dict is None on failure.
    """
    messages = [{'role': 'user', 'content': prompt}]
    params = {
        'messages': messages,
        'stream': False,
        'max_tokens': MAX_TOKENS,
        'temperature': temp,
        **temp_config,
    }

    start_time = time.time()
    try:
        resp = requests.post(
            f'{LLAMA_SERVER_URL}/v1/chat/completions',
            json=params,
            timeout=COMPLETION_TIMEOUT,
        )
        elapsed = time.time() - start_time

        if resp.status_code != 200:
            print(f"  ⚠️ API returned status {resp.status_code}: {resp.text[:200]}")
            return None, elapsed

        data = resp.json()

        # Extract usage info
        usage = data.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)

        # Check for llama-server detailed timings
        timings = data.get('timings') or data.get('stats')
        if timings and isinstance(timings, dict):
            predicted_ms = timings.get('predicted_ms', 0)
            predicted_n = timings.get('predicted_n', 0)
            if predicted_ms > 0 and predicted_n > 0:
                gen_time = predicted_ms / 1000.0
                tokens_per_sec = round(predicted_n / gen_time, 2)
            else:
                tokens_per_sec = round(
                    completion_tokens / elapsed, 2
                ) if elapsed > 0 else 0
        else:
            tokens_per_sec = round(
                completion_tokens / elapsed, 2
            ) if elapsed > 0 else 0

        # Extract the response content
        content = ''
        choices = data.get('choices', [])
        if choices:
            content = choices[0].get('message', {}).get('content', '')

        return {
            'content': content,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
            'total_time': round(elapsed, 2),
            'tokens_per_sec': tokens_per_sec,
        }, elapsed

    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        return None, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ⚠️ Request error: {e}")
        return None, elapsed


# ── Task Functions ───────────────────────────────────────────────────────

def get_tasks():
    """Get list of tasks from the tasks directory."""
    tasks = []
    if not TASKS_DIR.is_dir():
        return tasks
    for f in sorted(TASKS_DIR.iterdir()):
        if f.is_file() and f.suffix == '.txt':
            # Parse name like "1-svg.txt"
            parts = f.stem.split('-', 1)
            if len(parts) == 2:
                task_num = parts[0]
                task_format = parts[1]
                prompt = f.read_text().strip()
                tasks.append({
                    'num': task_num,
                    'format': task_format,
                    'prompt': prompt,
                    'filename': f.name,
                })
    return tasks


# ── Results Functions ────────────────────────────────────────────────────

def load_results():
    """Load existing results from results.json.

    Returns empty structure if file doesn't exist or is blank.
    """
    if RESULTS_JSON.exists():
        try:
            content = RESULTS_JSON.read_text().strip()
            if content:
                return json.loads(content)
        except (json.JSONDecodeError, OSError):
            pass
    return {'results': [], 'metadata': {}}


def save_results(results_data):
    """Save results to results.json."""
    RESULTS_JSON.write_text(json.dumps(results_data, indent=2))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Model benchmarking for llama-runner'
    )
    parser.add_argument(
        '--temperature', type=str, default=None,
        help='Comma-separated list of temperatures to test (e.g., "0.2,0.6")',
    )
    args = parser.parse_args()

    # ── Determine which temperatures to test ─────────────────────────────
    if args.temperature:
        try:
            selected_temps = [float(t.strip()) for t in args.temperature.split(',')]
        except ValueError:
            print("❌ Invalid temperature format. Use comma-separated values like '0.2,0.6'")
            sys.exit(1)
    else:
        selected_temps = list(TEMP_CONFIGS.keys())

    # Validate temperatures
    for t in selected_temps:
        if t not in TEMP_CONFIGS:
            print(f"❌ Unknown temperature: {t}. Available: {list(TEMP_CONFIGS.keys())}")
            sys.exit(1)

    temp_configs = {t: TEMP_CONFIGS[t] for t in selected_temps}

    # ── Check llama-runner connection ────────────────────────────────────
    print("🔍 Checking llama-runner connection...")
    if not check_llama_runner():
        print(f"❌ Cannot connect to llama-runner at {LLAMA_RUNNER_URL}")
        print("   Make sure llama-runner is running (./run.sh)")
        sys.exit(1)
    print(f"✅ Connected to llama-runner at {LLAMA_RUNNER_URL}")

    # ── Get models ───────────────────────────────────────────────────────
    models = get_models()
    if not models:
        print("❌ No models found!")
        sys.exit(1)
    print(f"📋 Found {len(models)} models")

    # ── Get tasks ────────────────────────────────────────────────────────
    tasks = get_tasks()
    if not tasks:
        print("❌ No tasks found in ./tasks/")
        sys.exit(1)
    print(f"📋 Found {len(tasks)} tasks: {[t['filename'] for t in tasks]}")

    # ── Load existing results (resume support) ───────────────────────────
    results_data = load_results()
    existing_results = results_data.get('results', [])
    completed_keys = {
        result_key(r['model'], r['task'], r['temperature'])
        for r in existing_results
    }

    # ── Calculate totals ─────────────────────────────────────────────────
    total_iterations = len(models) * len(tasks) * len(temp_configs)
    remaining_iterations = sum(
        1
        for model in models
        for task in tasks
        for temp in temp_configs
        if result_key(model['name'], task['num'], temp) not in completed_keys
    )
    completed_count = total_iterations - remaining_iterations

    print(f"\n📊 Benchmark Configuration:")
    print(f"   Models:       {len(models)}")
    print(f"   Tasks:        {len(tasks)}")
    print(f"   Temperatures: {list(temp_configs.keys())}")
    print(f"   Total iterations:    {total_iterations}")
    print(f"   Already completed:   {completed_count}")
    print(f"   Remaining:           {remaining_iterations}")
    print()

    if remaining_iterations == 0:
        print("✅ All iterations already completed!")
        return

    # ── Create results directory ─────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Track progress ───────────────────────────────────────────────────
    start_time = time.time()
    current_model_path = None
    interrupted = False

    def handle_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
        print(f"\n⛔ Interrupted! Saving results...")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        for model_idx, model in enumerate(models):
            if interrupted:
                break

            model_name = model['name']
            model_path = model['path']
            safe_name = sanitize_model_name(model_name)

            # Check if this model has any remaining work
            model_remaining = [
                (task, temp)
                for task in tasks
                for temp in temp_configs
                if result_key(model_name, task['num'], temp) not in completed_keys
            ]

            if not model_remaining:
                print(f"⏭️  Skipping {model_name} (all iterations completed)")
                continue

            print(f"\n{'=' * 60}")
            print(f"🚀 Model {model_idx + 1}/{len(models)}: {model_name}")
            print(f"   Remaining iterations for this model: {len(model_remaining)}")
            print(f"{'=' * 60}")

            # ── Stop any currently running model ─────────────────────────
            running_path = get_running_model()
            if running_path:
                print(f"⏹️  Stopping currently running model...")
                stop_model(running_path)
                time.sleep(COOLDOWN_AFTER_STOP)

            # ── Start new model ──────────────────────────────────────────
            print(f"🔄 Loading model...")
            start_result = start_model(model_path)

            if not start_result.get('ok'):
                error = start_result.get('error', 'Unknown error')
                print(f"❌ Failed to start model: {error}")

                # Record failure for all remaining iterations
                for task, temp in model_remaining:
                    key = result_key(model_name, task['num'], temp)
                    if key not in completed_keys:
                        result_entry = {
                            'model': model_name,
                            'model_path': model_path,
                            'task': task['num'],
                            'task_format': task['format'],
                            'temperature': temp,
                            'temp_config': {'temperature': temp, **temp_configs[temp]},
                            'status': 'model_failed',
                            'error': error,
                            'timestamp': time.time(),
                        }
                        existing_results.append(result_entry)
                        completed_keys.add(key)

                results_data['results'] = existing_results
                save_results(results_data)
                completed_count += len(model_remaining)
                continue

            current_model_path = model_path

            # ── Wait for model to be ready ───────────────────────────────
            print()
            if not wait_for_model_ready():
                print(f"\n❌ Model failed to become ready within timeout")

                # Record failure for all remaining iterations
                for task, temp in model_remaining:
                    key = result_key(model_name, task['num'], temp)
                    if key not in completed_keys:
                        result_entry = {
                            'model': model_name,
                            'model_path': model_path,
                            'task': task['num'],
                            'task_format': task['format'],
                            'temperature': temp,
                            'temp_config': {'temperature': temp, **temp_configs[temp]},
                            'status': 'timeout',
                            'error': 'Model failed to become ready',
                            'timestamp': time.time(),
                        }
                        existing_results.append(result_entry)
                        completed_keys.add(key)

                results_data['results'] = existing_results
                save_results(results_data)
                completed_count += len(model_remaining)

                # Try to stop the failed model
                stop_model(model_path)
                current_model_path = None
                time.sleep(COOLDOWN_AFTER_STOP)
                continue

            print(f"\n✅ Model ready!")

            # ── Create model results directory ────────────────────────────
            model_results_dir = RESULTS_DIR / safe_name
            model_results_dir.mkdir(exist_ok=True, parents=True)

            # ── Run tasks ────────────────────────────────────────────────
            for task_idx, (task, temp) in enumerate(model_remaining):
                if interrupted:
                    break

                temp_str = str(temp)
                temp_config = temp_configs[temp]
                key = result_key(model_name, task['num'], temp)

                if key in completed_keys:
                    continue

                print(f"\n📝 [{task_idx + 1}/{len(model_remaining)}] "
                      f"Task {task['num']}-{task['format']} | Temp {temp_str}")

                # ── Send completion request ───────────────────────────────
                result, elapsed = send_completion(task['prompt'], temp, temp_config)

                if result is None:
                    print(f"❌ Completion failed after {format_duration(elapsed)}")
                    result_entry = {
                        'model': model_name,
                        'model_path': model_path,
                        'task': task['num'],
                        'task_format': task['format'],
                        'temperature': temp,
                        'temp_config': {'temperature': temp, **temp_config},
                        'status': 'failed',
                        'total_time': round(elapsed, 2),
                        'timestamp': time.time(),
                    }
                else:
                    # ── Save result file ──────────────────────────────────
                    task_dir = model_results_dir / f"{task['num']}-{task['format']}"
                    task_dir.mkdir(exist_ok=True)
                    result_file = task_dir / f"{temp_str}.{task['format']}"
                    result_file.write_text(result['content'])

                    result_entry = {
                        'model': model_name,
                        'model_path': model_path,
                        'task': task['num'],
                        'task_format': task['format'],
                        'temperature': temp,
                        'temp_config': {'temperature': temp, **temp_config},
                        'status': 'success',
                        'prompt_tokens': result['prompt_tokens'],
                        'completion_tokens': result['completion_tokens'],
                        'total_tokens': result['total_tokens'],
                        'total_time': result['total_time'],
                        'tokens_per_sec': result['tokens_per_sec'],
                        'result_file': str(result_file.relative_to(SCRIPT_DIR)),
                        'timestamp': time.time(),
                    }

                    print(f"✅ {result['completion_tokens']} tokens | "
                          f"{result['tokens_per_sec']} tok/s | "
                          f"{format_duration(result['total_time'])}")

                # ── Save result to JSON ───────────────────────────────────
                existing_results.append(result_entry)
                results_data['results'] = existing_results
                save_results(results_data)
                completed_keys.add(key)
                completed_count += 1

                # ── Show progress ─────────────────────────────────────────
                elapsed_total = time.time() - start_time
                iterations_done = completed_count - (total_iterations - remaining_iterations)
                rate = iterations_done / elapsed_total if elapsed_total > 0 else 0
                remaining = total_iterations - completed_count
                eta = remaining / rate if rate > 0 else 0

                # Count fully completed models
                models_done = sum(
                    1 for m in models
                    if all(
                        result_key(m['name'], t['num'], tmp) in completed_keys
                        for t in tasks
                        for tmp in temp_configs
                    )
                )

                print(f"📊 Models: {models_done}/{len(models)} | "
                      f"Iterations: {completed_count}/{total_iterations} | "
                      f"Elapsed: {format_duration(elapsed_total)} | "
                      f"ETA: ~{format_duration(eta)}")

            # ── Stop the model after processing all tasks ────────────────
            if current_model_path:
                print(f"\n⏹️  Unloading model {model_name}...")
                stop_model(current_model_path)
                current_model_path = None
                time.sleep(COOLDOWN_AFTER_STOP)

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop current model if still running
        if current_model_path:
            try:
                stop_model(current_model_path)
            except Exception:
                pass

        # Save results
        results_data['results'] = existing_results
        save_results(results_data)

        if interrupted:
            print(f"\n⛔ Test interrupted. Results saved to {RESULTS_JSON}")
            print(f"   Run again to continue from where you left off.")
        else:
            print(f"\n✅ Benchmark complete! Results saved to {RESULTS_JSON}")
            print(f"   Run ./results.sh to see the summary table.")


if __name__ == '__main__':
    main()
