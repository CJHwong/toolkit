# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
Claude Stats Report - Analyze your local Claude Code usage data.

COMPATIBILITY:
    Tested with Claude Code v2.1.x (January 2026).
    Should work with any version that uses ~/.claude/ directory structure.

USAGE:
    # Run directly with uv (no install needed):
    uv run claude_stats_report.py

    # Or from GitHub:
    uv run https://raw.githubusercontent.com/CJHwong/toolkit/refs/heads/main/python/claude_stats_report.py

ANALYZES:
    - ~/.claude/projects/**/*.jsonl  (conversation sessions)
    - ~/.claude/debug/*.txt          (process logs)
    - ~/.claude/history.jsonl        (command history)

OUTPUT:
    - Active engagement time (unique vs parallel)
    - Process runtime statistics
    - Content generation volume (chat + code lines)
"""
import json
import os
import glob
from datetime import datetime, timedelta

# --- Configuration ---
CLAUDE_DIR = os.path.expanduser("~/.claude")
HISTORY_FILE = os.path.join(CLAUDE_DIR, "history.jsonl")
STATS_CACHE_FILE = os.path.join(CLAUDE_DIR, "stats-cache.json")
DEBUG_DIR = os.path.join(CLAUDE_DIR, "debug")
PROJECTS_DIR = os.path.join(CLAUDE_DIR, "projects")

# --- Helper Functions ---

def format_duration(ms):
    """Formats milliseconds into hours and days."""
    seconds = ms / 1000
    hours = seconds / 3600
    days = hours / 24
    return f"{int(hours)}h {int((seconds % 3600) // 60)}m ({days:.1f} days)"

def parse_iso_date(date_str):
    """Parses ISO 8601 date strings robustly."""
    try:
        # Handle "Z" suffix
        if date_str.endswith('Z'):
            date_str = date_str[:-1] + '+00:00'
        # Python 3.11+ handles timezone offsets natively
        # For older versions, strip timezone and parse naive datetime
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            # Fallback: strip timezone offset and parse as naive
            if '+' in date_str:
                date_str = date_str.rsplit('+', 1)[0]
            elif date_str.count('-') > 2:  # Has negative offset like -05:00
                date_str = date_str.rsplit('-', 1)[0]
            return datetime.fromisoformat(date_str)
    except (ValueError, AttributeError):
        return None

# --- Analysis Functions ---

def analyze_history_sessions(gap_minutes=7.5):
    """
    Groups commands from history.jsonl into sessions.
    gap_minutes: The detailed threshold we found that matches official stats.
    """
    commands = []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'timestamp' in entry:
                        commands.append({
                            'ts': entry['timestamp'],
                            'project': entry.get('project', 'unknown')
                        })
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    except FileNotFoundError:
        print(f"Warning: {HISTORY_FILE} not found.")
        return None

    if not commands:
        return {'count': 0, 'active_duration_ms': 0}

    commands.sort(key=lambda x: x['ts'])
    
    gap_ms = gap_minutes * 60 * 1000
    sessions = []
    
    # Initialize first session
    current_session = {
        'start': commands[0]['ts'],
        'end': commands[0]['ts'],
        'cmd_count': 1,
        'project': commands[0]['project']
    }
    
    for cmd in commands[1:]:
        time_diff = cmd['ts'] - current_session['end']
        
        # Simple heuristic: if same project and within gap, same session
        # If project changes, forces new session regardless of time
        is_same_project = (cmd['project'] == current_session['project'])
        
        if is_same_project and time_diff <= gap_ms:
            current_session['end'] = cmd['ts']
            current_session['cmd_count'] += 1
        else:
            sessions.append(current_session)
            current_session = {
                'start': cmd['ts'],
                'end': cmd['ts'],
                'cmd_count': 1,
                'project': cmd['project']
            }
    
    sessions.append(current_session)
    
    # Calculate Active Time
    # Active time = (End - Start) + buffer per session (e.g. 2 mins reading time)
    buffer_per_session_ms = 2 * 60 * 1000
    total_active_ms = sum((s['end'] - s['start']) + buffer_per_session_ms for s in sessions)
    
    return {
        'count': len(sessions),
        'active_duration_ms': total_active_ms,
        'avg_cmds_per_session': len(commands) / len(sessions) if sessions else 0,
        'projects': list(set(s['project'] for s in sessions))
    }

def merge_time_ranges(ranges):
    """Merges overlapping time ranges. Returns list of non-overlapping ranges."""
    if not ranges:
        return []
    # Sort by start time
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:  # Overlapping
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged

def analyze_debug_logs():
    """Calculates process uptime from debug logs, handling overlapping sessions."""
    log_files = glob.glob(os.path.join(DEBUG_DIR, "*.txt"))
    time_ranges = []  # List of (start_ts, end_ts) tuples
    valid_logs = 0

    for log_path in log_files:
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read start time (first line)
                first = f.readline()
                if not first: continue

                # Read end time (last line)
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 4096))
                lines = f.readlines()
                last = lines[-1] if lines else first

                # Extract timestamps
                t1 = parse_iso_date(first.split(' ')[0])
                t2 = parse_iso_date(last.split(' ')[0])

                if t1 and t2 and t2 >= t1:
                    time_ranges.append((t1, t2))
                    valid_logs += 1
        except (IOError, OSError, IndexError):
            continue

    # Calculate total accumulated (sum of all ranges, with overlaps counted multiple times)
    total_accumulated_ms = sum((end - start).total_seconds() * 1000 for start, end in time_ranges)

    # Merge overlapping ranges to get unique time
    merged_ranges = merge_time_ranges(time_ranges)
    unique_uptime_ms = sum((end - start).total_seconds() * 1000 for start, end in merged_ranges)

    # Parallel time = time spent with multiple sessions running
    parallel_ms = total_accumulated_ms - unique_uptime_ms

    return {
        'count': valid_logs,
        'total_accumulated_ms': total_accumulated_ms,
        'unique_uptime_ms': unique_uptime_ms,
        'parallel_ms': parallel_ms
    }

def analyze_session_engagement(gap_minutes=5):
    """
    Calculates actual engagement time from session .jsonl files.
    Uses real conversation timestamps instead of just slash commands.
    """
    time_ranges = []  # (start, end) for each session
    total_turns = 0
    files_analyzed = 0

    for root, dirs, files in os.walk(PROJECTS_DIR):
        for file in files:
            if file.endswith(".jsonl") and (len(file) > 20 or file.startswith("agent-")):
                file_path = os.path.join(root, file)
                timestamps = []
                turn_count = 0
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                entry = json.loads(line)
                                # Get timestamp from any message type
                                ts_str = entry.get('timestamp')
                                if ts_str:
                                    ts = parse_iso_date(ts_str)
                                    if ts:
                                        timestamps.append(ts)
                                # Count conversation turns (user + assistant)
                                if entry.get('type') in ('human', 'user', 'assistant'):
                                    turn_count += 1
                            except (json.JSONDecodeError, KeyError, TypeError):
                                continue

                    if timestamps:
                        files_analyzed += 1
                        total_turns += turn_count
                        timestamps.sort()
                        # Group into sub-sessions within this file based on gaps
                        gap_td = timedelta(minutes=gap_minutes)
                        session_start = timestamps[0]
                        session_end = timestamps[0]

                        for ts in timestamps[1:]:
                            if ts - session_end > gap_td:
                                # Save previous session, start new one
                                time_ranges.append((session_start, session_end))
                                session_start = ts
                            session_end = ts
                        time_ranges.append((session_start, session_end))

                except (IOError, OSError):
                    continue

    if not time_ranges:
        return None

    # Add buffer per session (1 min for reading/thinking)
    buffer_per_session = timedelta(minutes=1)
    total_with_buffer_ms = sum(
        ((end - start) + buffer_per_session).total_seconds() * 1000
        for start, end in time_ranges
    )

    # Merge overlapping to get unique engagement time
    merged = merge_time_ranges(time_ranges)
    unique_engagement_ms = sum(
        ((end - start) + buffer_per_session).total_seconds() * 1000
        for start, end in merged
    )

    return {
        'session_count': len(time_ranges),
        'total_with_buffer_ms': total_with_buffer_ms,
        'unique_engagement_ms': unique_engagement_ms,
        'parallel_ms': total_with_buffer_ms - unique_engagement_ms,
        'total_turns': total_turns,
        'files_analyzed': files_analyzed
    }

def analyze_content_volume():
    """Counts lines of code and chat from preserved .jsonl files."""
    stats = {'chat_lines': 0, 'code_lines': 0, 'files_scanned': 0}
    
    for root, dirs, files in os.walk(PROJECTS_DIR):
        for file in files:
            # Look for session files (UUIDs or agent-*)
            if file.endswith(".jsonl") and (len(file) > 20 or file.startswith("agent-")):
                stats['files_scanned'] += 1
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                entry = json.loads(line)
                                # We care about assistant output
                                if entry.get('type') != 'assistant':
                                    continue
                                    
                                msg = entry.get('message', entry)
                                content = msg.get('content', [])
                                
                                if isinstance(content, list):
                                    for block in content:
                                        b_type = block.get('type')
                                        if b_type == 'text':
                                            stats['chat_lines'] += block.get('text', '').count('\n')
                                        elif b_type == 'tool_use':
                                            # If tool is file edit or bash, count as code
                                            tool = block.get('name')
                                            inp = block.get('input', {})
                                            if tool in ['write_file', 'replace', 'Bash', 'run_shell_command']:
                                                # Heuristic: count lines in the largest string arg
                                                vals = [str(v) for v in inp.values() if isinstance(v, str)]
                                                if vals:
                                                    stats['code_lines'] += max(v.count('\n') for v in vals)
                            except (json.JSONDecodeError, KeyError, TypeError):
                                continue
                except (IOError, OSError):
                    continue
    return stats

# --- Main Execution ---

def main():
    print("="*60)
    print("🔎 CLAUDE USAGE FORENSICS REPORT")
    print("="*60)
    print("Analyzing your local data... this may take a moment.\n")

    # 1. Session Engagement (from actual conversation timestamps)
    engagement_stats = analyze_session_engagement(gap_minutes=5)

    # 2. Command History (slash commands only - less accurate)
    hist_stats = analyze_history_sessions(gap_minutes=5)

    # 3. Process Uptime
    uptime_stats = analyze_debug_logs()

    # 4. Content Volume
    content_stats = analyze_content_volume()

    # --- Report Generation ---

    if engagement_stats:
        print(f"🧠 ACTIVE ENGAGEMENT (From {engagement_stats['files_analyzed']} session files)")
        print(f"   Unique Work Time:     {format_duration(engagement_stats['unique_engagement_ms'])}")
        print(f"   Parallel Work Time:   {format_duration(engagement_stats['parallel_ms'])}")
        print(f"   Total Accumulated:    {format_duration(engagement_stats['total_with_buffer_ms'])}")
        print(f"   Conversation Turns:   {engagement_stats['total_turns']:,}")
        print(f"   Session Count:        {engagement_stats['session_count']:,}")
        if engagement_stats['session_count'] > 0:
            avg_session = engagement_stats['unique_engagement_ms'] / engagement_stats['session_count']
            print(f"   Avg Session Length:   {format_duration(avg_session)}")

    print("-" * 40)

    if hist_stats and hist_stats['count'] > 0:
        print(f"⌨️  COMMAND HISTORY (Slash commands only - partial data)")
        print(f"   Commands Logged:      {int(hist_stats['avg_cmds_per_session'] * hist_stats['count']):,}")
        print(f"   Unique Projects:      {len(hist_stats['projects'])}")

    print("-" * 40)

    if uptime_stats['count'] > 0:
        print(f"⚡ PROCESS RUNTIME (From {uptime_stats['count']} debug logs)")
        print(f"   Unique Uptime:        {format_duration(uptime_stats['unique_uptime_ms'])}")
        print(f"   Parallel Sessions:    {format_duration(uptime_stats['parallel_ms'])}")
        print(f"   Total Accumulated:    {format_duration(uptime_stats['total_accumulated_ms'])}")

    print("-" * 40)

    if content_stats['files_scanned'] > 0:
        print(f"📝 CONTENT GENERATION (From {content_stats['files_scanned']} preserved session files)")
        print(f"   Total Volume:         {content_stats['chat_lines'] + content_stats['code_lines']:,} lines")
        print(f"   ├── Chat/Analysis:    {content_stats['chat_lines']:,} lines")
        print(f"   └── Code/Actions:     {content_stats['code_lines']:,} lines")

    print("="*60)
    print("Report generated successfully.")

if __name__ == "__main__":
    main()
