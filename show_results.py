#!/usr/bin/env python3
"""Display benchmark results from results.json in formatted tables."""

import json
import sys
from pathlib import Path

RESULTS_JSON = Path(__file__).parent / 'results.json'

# ANSI color codes
BOLD = '\033[1m'
DIM = '\033[2m'
GREEN = '\033[32m'
RED = '\033[31m'
YELLOW = '\033[33m'
CYAN = '\033[36m'
RESET = '\033[0m'


def colorize(text, color):
    """Apply ANSI color to text."""
    return f"{color}{text}{RESET}"


def print_table(headers, rows, title=None):
    """Print a formatted ASCII table."""
    if not rows:
        return

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    def format_row(cells):
        parts = []
        for i, cell in enumerate(cells):
            s = str(cell).ljust(col_widths[i])
            parts.append(s)
        return '| ' + ' | '.join(parts) + ' |'

    def separator():
        return '+-' + '-+-'.join('-' * w for w in col_widths) + '-+'

    if title:
        print(f"\n{colorize(title, BOLD)}")

    print(separator())
    print(format_row(headers))
    print(separator())
    for row in rows:
        print(format_row(row))
    print(separator())


def main():
    if not RESULTS_JSON.exists():
        print("❌ No results.json found. Run ./test.sh first.")
        sys.exit(1)

    try:
        data = json.loads(RESULTS_JSON.read_text())
    except json.JSONDecodeError:
        print("❌ results.json is corrupted.")
        sys.exit(1)

    results = data.get('results', [])
    if not results:
        print("❌ No results found in results.json.")
        sys.exit(1)

    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') != 'success']

    # ── Detailed Results Table ────────────────────────────────────────────
    headers = ['Model', 'Task', 'Temp', 'Tok/s', 'Comp Tokens', 'Total Time', 'Tool', 'Status']

    rows = []
    for r in results:
        model = r.get('model', '')
        if len(model) > 45:
            model = model[:42] + '...'

        task = f"{r.get('task', '')}-{r.get('task_format', '')}"
        temp = str(r.get('temperature', ''))

        if r.get('status') == 'success':
            tok_sec = f"{r.get('tokens_per_sec', 0):.2f}"
            comp_tokens = str(r.get('completion_tokens', 0))
            total_time = f"{r.get('total_time', 0):.1f}s"
            tool = '🔧' if r.get('used_tool_call') else '📝'
            status = colorize('✅', GREEN)
        else:
            tok_sec = '-'
            comp_tokens = '-'
            total_time = f"{r.get('total_time', 0):.1f}s" if r.get('total_time') else '-'
            tool = '-'
            status = colorize('❌', RED)

        rows.append([model, task, temp, tok_sec, comp_tokens, total_time, tool, status])

    print_table(headers, rows, title="📊 Detailed Results")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{colorize('📊 Summary', BOLD)}")
    print(f"   Total iterations:  {len(results)}")
    print(f"   Successful:        {colorize(str(len(successful)), GREEN)}")
    print(f"   Failed:            {colorize(str(len(failed)), RED)}")

    if successful:
        avg_tok_sec = sum(r['tokens_per_sec'] for r in successful) / len(successful)
        avg_time = sum(r['total_time'] for r in successful) / len(successful)
        avg_comp_tokens = sum(r['completion_tokens'] for r in successful) / len(successful)
        avg_total_tokens = sum(r['total_tokens'] for r in successful) / len(successful)

        print(f"\n   {colorize('Averages (successful only):', CYAN)}")
        print(f"   Tokens/sec:        {avg_tok_sec:.2f}")
        print(f"   Completion tokens: {avg_comp_tokens:.0f}")
        print(f"   Total tokens:      {avg_total_tokens:.0f}")
        print(f"   Total time:        {avg_time:.1f}s")

    # ── Per-Model Averages ────────────────────────────────────────────────
    if successful:
        model_stats = {}
        for r in successful:
            model = r.get('model', '')
            if model not in model_stats:
                model_stats[model] = {
                    'tok_sec': [],
                    'time': [],
                    'comp_tokens': [],
                    'total_tokens': [],
                }
            model_stats[model]['tok_sec'].append(r['tokens_per_sec'])
            model_stats[model]['time'].append(r['total_time'])
            model_stats[model]['comp_tokens'].append(r['completion_tokens'])
            model_stats[model]['total_tokens'].append(r['total_tokens'])

        model_headers = ['Model', 'Avg Tok/s', 'Avg Time', 'Avg Comp Tokens', 'Iters']
        model_rows = []
        for model, stats in sorted(model_stats.items()):
            name = model if len(model) <= 45 else model[:42] + '...'
            avg_ts = f"{sum(stats['tok_sec']) / len(stats['tok_sec']):.2f}"
            avg_t = f"{sum(stats['time']) / len(stats['time']):.1f}s"
            avg_ct = f"{sum(stats['comp_tokens']) / len(stats['comp_tokens']):.0f}"
            count = str(len(stats['tok_sec']))
            model_rows.append([name, avg_ts, avg_t, avg_ct, count])

        print_table(model_headers, model_rows, title="📊 Per-Model Averages")

    # ── Per-Temperature Averages ──────────────────────────────────────────
    if successful:
        temp_stats = {}
        for r in successful:
            temp = r.get('temperature', 0)
            if temp not in temp_stats:
                temp_stats[temp] = {
                    'tok_sec': [],
                    'time': [],
                    'comp_tokens': [],
                }
            temp_stats[temp]['tok_sec'].append(r['tokens_per_sec'])
            temp_stats[temp]['time'].append(r['total_time'])
            temp_stats[temp]['comp_tokens'].append(r['completion_tokens'])

        temp_headers = ['Temperature', 'Avg Tok/s', 'Avg Time', 'Avg Comp Tokens', 'Iters']
        temp_rows = []
        for temp in sorted(temp_stats.keys()):
            stats = temp_stats[temp]
            avg_ts = f"{sum(stats['tok_sec']) / len(stats['tok_sec']):.2f}"
            avg_t = f"{sum(stats['time']) / len(stats['time']):.1f}s"
            avg_ct = f"{sum(stats['comp_tokens']) / len(stats['comp_tokens']):.0f}"
            count = str(len(stats['tok_sec']))
            temp_rows.append([str(temp), avg_ts, avg_t, avg_ct, count])

        print_table(temp_headers, temp_rows, title="📊 Per-Temperature Averages")

    # ── Failed iterations ─────────────────────────────────────────────────
    if failed:
        fail_headers = ['Model', 'Task', 'Temp', 'Status', 'Error']
        fail_rows = []
        for r in failed:
            model = r.get('model', '')
            if len(model) > 35:
                model = model[:32] + '...'
            task = f"{r.get('task', '')}-{r.get('task_format', '')}"
            temp = str(r.get('temperature', ''))
            status = r.get('status', 'unknown')
            error = r.get('error', '')
            if len(error) > 40:
                error = error[:37] + '...'
            fail_rows.append([model, task, temp, status, error])

        print_table(fail_headers, fail_rows, title="❌ Failed Iterations")


if __name__ == '__main__':
    main()
