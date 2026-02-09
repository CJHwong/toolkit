# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "click>=8.0",
#     "ollama>=0.6",
# ]
# ///
"""
Claude Code Insights - Generate a comprehensive HTML usage report.

Re-implementation of the Claude Code /insights command (v2.1.37) as a standalone script.
Reads session logs and cached facets from ~/.claude/ to produce a self-contained
HTML report with charts, statistics, and placeholder sections for LLM-generated
insights.

Stage 3 (facet extraction) uses a local Ollama instance to extract structured
facets from uncached sessions.  Stage 5 (insight generation) uses Ollama to
generate narrative insights from aggregated statistics.

USAGE:
    uv run python/cc_insights.py
    uv run python/cc_insights.py -v --max-extract 0      # cache + placeholders only
    uv run python/cc_insights.py -v --max-extract 5       # extract 5 uncached facets
    uv run python/cc_insights.py --model qwen3:30b        # use a different model
    uv run python/cc_insights.py -o ~/Desktop/insights.html
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLAUDE_DIR = Path.home() / ".claude"
PROJECTS_DIR = CLAUDE_DIR / "projects"
FACETS_DIR = CLAUDE_DIR / "usage-data" / "facets"
DEFAULT_OUTPUT = CLAUDE_DIR / "usage-data" / "report.html"

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# Types included in the UUID map (matching original iy() filter)
IY_TYPES = frozenset({"user", "assistant", "attachment", "system", "progress"})

LANGUAGE_MAP: dict[str, str] = {
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".py": "Python",
    ".rb": "Ruby",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".md": "Markdown",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".sh": "Shell",
    ".css": "CSS",
    ".html": "HTML",
}

ERROR_CATEGORIES: list[tuple[str, list[str]]] = [
    ("Command Failed", ["exit code"]),
    ("User Rejected", ["rejected", "doesn't want"]),
    ("Edit Failed", ["string to replace not found", "no changes"]),
    ("File Changed", ["modified since read"]),
    ("File Too Large", ["exceeds maximum", "too large"]),
    ("File Not Found", ["file not found", "does not exist"]),
]

DISPLAY_NAMES: dict[str, str] = {
    # Goals
    "debug_investigate": "Debug/Investigate",
    "implement_feature": "Implement Feature",
    "fix_bug": "Fix Bug",
    "write_script_tool": "Write Script/Tool",
    "refactor_code": "Refactor Code",
    "configure_system": "Configure System",
    "create_pr_commit": "Create PR/Commit",
    "analyze_data": "Analyze Data",
    "understand_codebase": "Understand Codebase",
    "write_tests": "Write Tests",
    "write_docs": "Write Docs",
    "deploy_infra": "Deploy/Infra",
    "warmup_minimal": "Cache Warmup",
    # Success types
    "fast_accurate_search": "Fast/Accurate Search",
    "correct_code_edits": "Correct Code Edits",
    "good_explanations": "Good Explanations",
    "proactive_help": "Proactive Help",
    "multi_file_changes": "Multi-file Changes",
    "handled_complexity": "Multi-file Changes",
    "good_debugging": "Good Debugging",
    # Friction types
    "misunderstood_request": "Misunderstood Request",
    "wrong_approach": "Wrong Approach",
    "buggy_code": "Buggy Code",
    "user_rejected_action": "User Rejected Action",
    "claude_got_blocked": "Claude Got Blocked",
    "user_stopped_early": "User Stopped Early",
    "wrong_file_or_location": "Wrong File/Location",
    "excessive_changes": "Excessive Changes",
    "slow_or_verbose": "Slow/Verbose",
    "tool_failed": "Tool Failed",
    "user_unclear": "User Unclear",
    "external_issue": "External Issue",
    # Satisfaction
    "frustrated": "Frustrated",
    "dissatisfied": "Dissatisfied",
    "likely_satisfied": "Likely Satisfied",
    "satisfied": "Satisfied",
    "happy": "Happy",
    "unsure": "Unsure",
    "neutral": "Neutral",
    "delighted": "Delighted",
    # Session types
    "single_task": "Single Task",
    "multi_task": "Multi Task",
    "iterative_refinement": "Iterative Refinement",
    "exploration": "Exploration",
    "quick_question": "Quick Question",
    # Outcomes
    "fully_achieved": "Fully Achieved",
    "mostly_achieved": "Mostly Achieved",
    "partially_achieved": "Partially Achieved",
    "not_achieved": "Not Achieved",
    "unclear_from_transcript": "Unclear",
    # Helpfulness
    "unhelpful": "Unhelpful",
    "slightly_helpful": "Slightly Helpful",
    "moderately_helpful": "Moderately Helpful",
    "very_helpful": "Very Helpful",
    "essential": "Essential",
}

FACET_PROMPT = """\
Analyze this Claude Code session and extract structured facets.

RESPOND WITH ONLY A VALID JSON OBJECT with these fields:
- underlying_goal (string)
- goal_categories (object: category → count)
- outcome: fully_achieved|mostly_achieved|partially_achieved|not_achieved|unclear_from_transcript
- user_satisfaction_counts (object: level → count)
- claude_helpfulness: unhelpful|slightly_helpful|moderately_helpful|very_helpful|essential
- session_type: single_task|multi_task|iterative_refinement|exploration|quick_question
- friction_counts (object: type → count)
- friction_detail (string)
- primary_success: fast_accurate_search|correct_code_edits|good_explanations|proactive_help|multi_file_changes|handled_complexity|good_debugging|none
- brief_summary (string, 1-2 sentences)

CRITICAL GUIDELINES:
1. goal_categories: Count ONLY what the USER explicitly asked for.
2. user_satisfaction_counts: Base ONLY on explicit user signals.
3. friction_counts: Be specific about what went wrong.
4. If very short or just warmup, use warmup_minimal for goal_category.

SESSION:
"""

# -- Stage 5 insight prompts ------------------------------------------------

INSIGHT_PROMPT_PROJECT_AREAS = """\
You are analyzing aggregated Claude Code usage statistics.
Identify the main project areas or domains this user works on, based on their
session summaries, goal categories, and tool usage.

RESPOND WITH ONLY A VALID JSON OBJECT matching this schema:
{"areas": [{"name": "<area name>", "session_count": <int>, "description": "<1-2 sentences>"}]}

Return 3-6 areas. session_count should be your best estimate of how many sessions
relate to each area. Areas should be meaningful groupings (e.g., "Backend API Development",
"Data Pipeline Work"), not generic labels.

USAGE DATA:
"""

INSIGHT_PROMPT_INTERACTION_STYLE = """\
You are analyzing aggregated Claude Code usage statistics.
Describe this user's interaction style — how they work with Claude Code.
Consider session types, message patterns, tool usage, response times, and outcomes.

RESPOND WITH ONLY A VALID JSON OBJECT matching this schema:
{"narrative": "<2-3 paragraph markdown narrative>", "key_pattern": "<one sentence summary of their dominant pattern>"}

Be specific and data-driven. Reference actual numbers. Use **bold** for emphasis.

USAGE DATA:
"""

INSIGHT_PROMPT_WHAT_WORKS = """\
You are analyzing aggregated Claude Code usage statistics.
Identify what works well for this user — their most effective workflows,
impressive patterns, and strengths in how they use Claude Code.

RESPOND WITH ONLY A VALID JSON OBJECT matching this schema:
{"intro": "<1-2 sentence overview>", "impressive_workflows": [{"title": "<short title>", "description": "<2-3 sentences>"}]}

Return 3-5 impressive workflows. Be specific about what makes them effective.
Reference actual tool usage, success rates, and patterns from the data.

USAGE DATA:
"""

INSIGHT_PROMPT_FRICTION = """\
You are analyzing aggregated Claude Code usage statistics.
Analyze friction points — what causes problems, errors, or frustration.
Consider tool errors, friction counts, user rejections, and failed outcomes.

RESPOND WITH ONLY A VALID JSON OBJECT matching this schema:
{"intro": "<1-2 sentence overview of friction landscape>", "categories": [{"category": "<category name>", "description": "<what goes wrong>", "examples": ["<specific example 1>", "<specific example 2>"]}]}

Return 2-4 friction categories. Be honest and specific. Reference actual error
counts and friction types from the data. Suggest root causes where possible.

USAGE DATA:
"""

INSIGHT_PROMPT_SUGGESTIONS = """\
You are analyzing aggregated Claude Code usage statistics.
Generate actionable suggestions to improve this user's Claude Code experience.
Consider their friction points, unused features, and workflow patterns.

RESPOND WITH ONLY A VALID JSON OBJECT matching this schema:
{"claude_md_additions": [{"addition": "<what to add>", "why": "<reason>", "prompt_scaffold": "<example text for CLAUDE.md>"}], "features_to_try": [{"feature": "<feature name>", "one_liner": "<brief description>", "why_for_you": "<personalized reason>", "example_code": "<example usage>"}], "usage_patterns": [{"title": "<pattern name>", "suggestion": "<what to do>", "detail": "<how it helps>", "copyable_prompt": "<example prompt to try>"}]}

Return 2-3 items per category. Make suggestions specific to this user's data,
not generic advice. Reference their actual usage patterns and gaps.

USAGE DATA:
"""

INSIGHT_PROMPT_HORIZON = """\
You are analyzing aggregated Claude Code usage statistics.
Suggest forward-looking opportunities — new workflows, techniques, or
capabilities this user hasn't fully explored yet.

RESPOND WITH ONLY A VALID JSON OBJECT matching this schema:
{"intro": "<1-2 sentence overview>", "opportunities": [{"title": "<opportunity>", "whats_possible": "<what they could achieve>", "how_to_try": "<concrete steps>", "copyable_prompt": "<example prompt to start>"}]}

Return 3-4 opportunities. Make them ambitious but realistic based on the user's
current skill level and usage patterns. Reference actual data.

USAGE DATA:
"""

INSIGHT_PROMPT_FUN_ENDING = """\
You are analyzing aggregated Claude Code usage statistics.
Create a fun, memorable closing statement for this user's insights report.
Highlight an interesting or surprising stat, a notable achievement, or a
quirky pattern from their usage.

RESPOND WITH ONLY A VALID JSON OBJECT matching this schema:
{"headline": "<catchy one-liner>", "detail": "<2-3 sentences expanding on the headline with real stats>"}

Be warm, genuine, and specific. Reference actual numbers from the data.
Avoid generic platitudes — find something genuinely interesting or remarkable.

USAGE DATA:
"""

INSIGHT_PROMPT_AT_A_GLANCE = """\
You are analyzing aggregated Claude Code usage statistics and insight results
from other analysis sections. Synthesize everything into a brief executive summary.

RESPOND WITH ONLY A VALID JSON OBJECT matching this schema:
{"whats_working": "<2-3 sentences on strengths>", "whats_hindering": "<2-3 sentences on friction>", "quick_wins": "<2-3 sentences on easy improvements>", "ambitious_workflows": "<2-3 sentences on bigger opportunities>"}

Be concise, specific, and data-driven. Use **bold** for key metrics.
This is the first thing the user sees — make it count.

USAGE DATA:
"""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SessionMetadata:
    session_id: str
    project_path: str
    start_time: str  # ISO 8601
    end_time: str
    duration_minutes: float
    user_message_count: int
    assistant_message_count: int
    first_prompt: str
    tool_counts: dict[str, int] = field(default_factory=dict)
    languages: dict[str, int] = field(default_factory=dict)
    git_commits: int = 0
    git_pushes: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    user_response_times: list[float] = field(default_factory=list)
    interruptions: int = 0
    tool_errors: int = 0
    tool_error_categories: dict[str, int] = field(default_factory=dict)
    lines_added: int = 0
    lines_removed: int = 0
    files_modified: int = 0
    message_hours: list[int] = field(default_factory=list)
    user_message_timestamps: list[str] = field(default_factory=list)
    uses_task_agent: bool = False
    uses_mcp: bool = False
    uses_web_search: bool = False
    uses_web_fetch: bool = False


@dataclass
class SessionFacet:
    session_id: str
    underlying_goal: str = ""
    goal_categories: dict[str, int] = field(default_factory=dict)
    outcome: str = "unclear_from_transcript"
    user_satisfaction_counts: dict[str, int] = field(default_factory=dict)
    claude_helpfulness: str = "moderately_helpful"
    session_type: str = "single_task"
    friction_counts: dict[str, int] = field(default_factory=dict)
    friction_detail: str = ""
    primary_success: str = "none"
    brief_summary: str = ""


@dataclass
class AggregatedStats:
    total_sessions: int = 0
    sessions_with_facets: int = 0
    date_range_start: str = ""
    date_range_end: str = ""
    total_messages: int = 0
    total_duration_hours: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    languages: dict[str, int] = field(default_factory=dict)
    git_commits: int = 0
    git_pushes: int = 0
    projects: dict[str, int] = field(default_factory=dict)
    goal_categories: dict[str, int] = field(default_factory=dict)
    outcomes: dict[str, int] = field(default_factory=dict)
    satisfaction: dict[str, int] = field(default_factory=dict)
    helpfulness: dict[str, int] = field(default_factory=dict)
    session_types: dict[str, int] = field(default_factory=dict)
    friction: dict[str, int] = field(default_factory=dict)
    success: dict[str, int] = field(default_factory=dict)
    session_summaries: list[dict] = field(default_factory=list)
    total_interruptions: int = 0
    total_tool_errors: int = 0
    tool_error_categories: dict[str, int] = field(default_factory=dict)
    user_response_times: list[float] = field(default_factory=list)
    median_response_time: float = 0.0
    avg_response_time: float = 0.0
    sessions_using_task_agent: int = 0
    sessions_using_mcp: int = 0
    sessions_using_web_search: int = 0
    sessions_using_web_fetch: int = 0
    total_lines_added: int = 0
    total_lines_removed: int = 0
    total_files_modified: int = 0
    days_active: int = 0
    messages_per_day: float = 0.0
    message_hours: list[int] = field(default_factory=list)
    multi_clauding: dict[str, int] = field(
        default_factory=lambda: {
            "overlap_events": 0,
            "sessions_involved": 0,
            "user_messages_during": 0,
        }
    )


@dataclass
class InsightResults:
    at_a_glance: dict = field(default_factory=dict)
    project_areas: dict = field(default_factory=dict)
    interaction_style: dict = field(default_factory=dict)
    what_works: dict = field(default_factory=dict)
    friction_analysis: dict = field(default_factory=dict)
    suggestions: dict = field(default_factory=dict)
    on_the_horizon: dict = field(default_factory=dict)
    fun_ending: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except (ValueError, AttributeError):
        return None


def _extract_text(content) -> str:
    """Extract plain text from a message content field (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return ""


def _get_content_blocks(content) -> list[dict]:
    """Return content blocks as a list of dicts."""
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


# ---------------------------------------------------------------------------
# Stage 1: Load & Filter Sessions
# ---------------------------------------------------------------------------


def _walk_chain(uuid_map: dict[str, dict], leaf: dict) -> list[dict]:
    """Walk from a leaf node up through parentUuid to build the conversation chain.

    Matches the original ``wmT`` function.  Returns entries in root→leaf order.
    """
    chain: list[dict] = []
    seen: set[str] = set()
    current = leaf
    while current:
        uuid = current.get("uuid", "")
        if uuid in seen:
            break  # cycle detected
        seen.add(uuid)
        chain.append(current)
        parent_uuid = current.get("parentUuid")
        current = uuid_map.get(parent_uuid) if parent_uuid else None
    chain.reverse()
    return chain


def _count_user_messages(chain: list[dict]) -> int:
    """Count user messages with text content (matching original ``cjA``).

    The original checks ``G.type==="text" && "text" in G`` — both the block
    type AND the existence of a ``text`` key.
    """
    count = 0
    for m in chain:
        if m.get("type") != "user":
            continue
        msg = m.get("message")
        if not msg:
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            count += 1
        elif isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and "text" in block
                ):
                    count += 1
                    break
    return count


def _parse_file_sessions(
    jsonl_path: Path,
) -> list[tuple[str, list[dict], datetime, datetime]]:
    """Parse a single JSONL file and return session objects (one per conv leaf).

    Matches the original ``EU8`` + ``G$T`` pipeline:
      1. Parse JSONL, build UUID map with ``iy`` filter
      2. Find structural leaves → conversational leaves (K set)
      3. For each conv leaf: walk chain, add dangling children
      4. Return ``(session_id, chain, created, modified)`` per leaf
    """
    # Parse JSONL
    entries: list[dict] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    entries.append(json.loads(raw_line))
                except json.JSONDecodeError:
                    continue
    except (OSError, IOError):
        return []

    # Build UUID map with iy filter (G$T)
    uuid_map: dict[str, dict] = {}
    for entry in entries:
        if entry.get("type") not in IY_TYPES:
            continue
        uuid = entry.get("uuid")
        if not uuid:
            continue
        uuid_map[uuid] = entry

    if not uuid_map:
        return []

    # Find structural leaves: entries whose UUID is not any entry's parentUuid
    parent_uuids: set[str] = set()
    for e in uuid_map.values():
        pu = e.get("parentUuid")
        if pu is not None:
            parent_uuids.add(pu)

    leaves = [e for e in uuid_map.values() if e["uuid"] not in parent_uuids]

    # Refine to conversational leaves (K set in G$T):
    # walk UP from each structural leaf to the first user/assistant ancestor.
    conv_leaf_uuids: set[str] = set()
    for leaf in leaves:
        current = leaf
        visited: set[str] = set()
        while current:
            uid = current.get("uuid", "")
            if uid in visited:
                break
            visited.add(uid)
            if current.get("type") in ("user", "assistant"):
                conv_leaf_uuids.add(uid)
                break
            pu = current.get("parentUuid")
            current = uuid_map.get(pu) if pu else None

    # Build session objects — one per conversational leaf (EU8)
    results: list[tuple[str, list[dict], datetime, datetime]] = []

    for leaf_uuid in conv_leaf_uuids:
        leaf_entry = uuid_map[leaf_uuid]

        # Walk chain from leaf to root (wmT)
        chain = _walk_chain(uuid_map, leaf_entry)
        if not chain:
            continue

        # Dangling children: entries whose parentUuid == leaf but that are NOT
        # themselves conversational leaves.  Sorted by timestamp, appended.
        dangling = [
            e
            for e in uuid_map.values()
            if e.get("parentUuid") == leaf_uuid and e.get("uuid") not in conv_leaf_uuids
        ]
        dangling.sort(key=lambda e: e.get("timestamp", ""))
        chain.extend(dangling)

        root = chain[0]
        created = parse_timestamp(root.get("timestamp"))
        modified = parse_timestamp(leaf_entry.get("timestamp"))
        session_id = root.get("sessionId", "")

        if created and modified and session_id:
            results.append((session_id, chain, created, modified))

    return results


def _is_synthetic(chain: list[dict]) -> bool:
    """Check if a session is synthetic.

    Matches the original ``NN8`` check: look at the first 5 entries in the
    chain (any type), and only check **string** content (not array blocks).
    """
    for entry in chain[:5]:
        if entry.get("type") != "user":
            continue
        msg = entry.get("message")
        if not msg:
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            if (
                "RESPOND WITH ONLY A VALID JSON OBJECT" in content
                or "record_facets" in content
            ):
                return True
    return False


def load_and_filter_sessions(
    verbose: bool = False,
) -> dict[str, tuple[list[dict], datetime, datetime]]:
    """Walk ``~/.claude/projects/``, process per-file, filter, deduplicate.

    Matches the original pipeline: ``hU8`` → ``LU8`` → ``foB`` → ``EU8``.
    Returns ``{session_id: (chain, created, modified)}``.
    """
    all_sessions: list[tuple[str, list[dict], datetime, datetime]] = []
    file_count = 0
    skipped_non_uuid = 0

    if not PROJECTS_DIR.is_dir():
        if verbose:
            click.echo(f"  Projects directory not found: {PROJECTS_DIR}")
        return {}

    # Iterate project directories (one level deep, matching hU8)
    for project_dir in sorted(PROJECTS_DIR.iterdir()):
        if not project_dir.is_dir():
            continue

        # Iterate JSONL files in this directory (one level, matching foB)
        for item in project_dir.iterdir():
            if not item.is_file() or not item.name.endswith(".jsonl"):
                continue

            # UUID filename validation (foB → GU)
            basename = item.stem
            if not UUID_RE.match(basename):
                skipped_non_uuid += 1
                continue

            file_count += 1
            sessions = _parse_file_sessions(item)
            all_sessions.extend(sessions)

    if verbose:
        click.echo(f"  Scanned {file_count} files, skipped {skipped_non_uuid} non-UUID")
        click.echo(f"  Raw session objects: {len(all_sessions)}")

    # Filter: synthetic check + AN8 (valid dates)
    filtered: list[tuple[str, list[dict], datetime, datetime]] = []
    for sid, chain, created, modified in all_sessions:
        if _is_synthetic(chain):
            continue
        filtered.append((sid, chain, created, modified))

    # Dedup: _N8 — keep best per session_id
    # (max user_message_count, tiebreak duration_minutes)
    best: dict[str, tuple[list[dict], datetime, datetime, int, int]] = {}
    for sid, chain, created, modified in filtered:
        uc = _count_user_messages(chain)
        dur = round((modified - created).total_seconds() / 60)

        existing = best.get(sid)
        if (
            not existing
            or uc > existing[3]
            or (uc == existing[3] and dur > existing[4])
        ):
            best[sid] = (chain, created, modified, uc, dur)

    # Filter: uc >= 2 and dur >= 1 (matching NN8's H check)
    result: dict[str, tuple[list[dict], datetime, datetime]] = {}
    for sid, (chain, created, modified, uc, dur) in best.items():
        if uc < 2 or dur < 1:
            continue
        result[sid] = (chain, created, modified)

    if verbose:
        click.echo(f"  After dedup: {len(best)} unique sessions")
        click.echo(f"  After filtering (uc>=2, dur>=1): {len(result)} sessions")

    return result


# ---------------------------------------------------------------------------
# Stage 2: Extract Session Metadata
# ---------------------------------------------------------------------------


def _categorize_error(text: str) -> str:
    lower = text.lower()
    for category, substrings in ERROR_CATEGORIES:
        if any(s in lower for s in substrings):
            return category
    return "Other"


def _file_extension_language(path_str: str) -> str | None:
    try:
        ext = Path(path_str).suffix.lower()
        return LANGUAGE_MAP.get(ext)
    except (ValueError, TypeError):
        return None


def extract_session_metadata(
    session_id: str,
    messages: list[dict],
    created: datetime,
    modified: datetime,
    verbose: bool = False,
) -> SessionMetadata:
    """Extract metadata from a session chain.

    ``messages`` is the full chain (all types including system/progress)
    produced by ``_parse_file_sessions``.  ``created`` and ``modified`` are
    the root and leaf timestamps from the session object, matching the
    original's ``T.created`` / ``T.modified``.

    Non-user/assistant entries are naturally skipped by the type checks.
    """
    duration_minutes = round((modified - created).total_seconds() / 60)

    # Project path from first message with cwd
    project_path = ""
    for m in messages:
        if m.get("cwd"):
            project_path = m["cwd"]
            break

    user_count = 0
    assistant_count = 0
    first_prompt = ""
    tool_counts: Counter = Counter()
    languages: Counter = Counter()
    git_commits = 0
    git_pushes = 0
    input_tokens = 0
    output_tokens = 0
    response_times: list[float] = []
    interruptions = 0
    tool_errors = 0
    error_cats: Counter = Counter()
    lines_added = 0
    lines_removed = 0
    modified_files: set[str] = set()
    message_hours: list[int] = []
    user_timestamps: list[str] = []
    uses_task = False
    uses_mcp = False
    uses_web_search = False
    uses_web_fetch = False

    last_assistant_ts: datetime | None = None

    # Iterate in chain order (root→leaf→dangling).  The chain is already
    # chronological so no timestamp-sort is needed.
    for m in messages:
        msg_type = m.get("type")
        msg_body = m.get("message") or {}
        content = msg_body.get("content", "")
        raw_ts = m.get("timestamp", "")
        ts = parse_timestamp(raw_ts) if raw_ts else None

        if msg_type == "assistant":
            assistant_count += 1
            if ts:
                last_assistant_ts = ts

            # Token usage — only input_tokens + output_tokens (matching RN8)
            usage = msg_body.get("usage") or {}
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)

            # Scan tool_use blocks in assistant messages
            for block in _get_content_blocks(content):
                if block.get("type") != "tool_use":
                    continue
                tool_name = block.get("name", "")
                tool_counts[tool_name] += 1
                tool_input = block.get("input") or {}

                # Feature flags
                if tool_name == "Task":
                    uses_task = True
                if tool_name.startswith("mcp__"):
                    uses_mcp = True
                if tool_name == "WebSearch":
                    uses_web_search = True
                if tool_name == "WebFetch":
                    uses_web_fetch = True

                # Language detection from file paths
                for key in ("file_path", "path", "filePath"):
                    fpath = tool_input.get(key)
                    if fpath and isinstance(fpath, str):
                        lang = _file_extension_language(fpath)
                        if lang:
                            languages[lang] += 1

                # Git commits/pushes from Bash commands
                if tool_name == "Bash":
                    cmd = tool_input.get("command", "")
                    if isinstance(cmd, str):
                        if "git commit" in cmd:
                            git_commits += 1
                        if "git push" in cmd:
                            git_pushes += 1

                # Lines added/removed — split('\n').length matching original
                if tool_name == "Edit":
                    old = tool_input.get("old_string", "")
                    new = tool_input.get("new_string", "")
                    if isinstance(old, str) and old:
                        lines_removed += len(old.split("\n"))
                    if isinstance(new, str) and new:
                        lines_added += len(new.split("\n"))
                    fpath = tool_input.get("file_path", "")
                    if fpath:
                        modified_files.add(fpath)

                if tool_name == "Write":
                    content_str = tool_input.get("content", "")
                    if isinstance(content_str, str) and content_str:
                        lines_added += len(content_str.split("\n"))
                    fpath = tool_input.get("file_path", "")
                    if fpath:
                        modified_files.add(fpath)

        elif msg_type == "user":
            if not m.get("message"):
                continue

            # Determine if this user message has actual text content.
            # Original checks: G.type==="text" && "text" in G
            has_text = False
            if isinstance(content, str) and content.strip():
                has_text = True
            elif isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                        and "text" in block
                    ):
                        has_text = True
                        break

            # Scan tool_result blocks for errors (all user messages)
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result" and block.get("is_error"):
                        tool_errors += 1
                        err_text = block.get("content", "")
                        if isinstance(err_text, list):
                            err_text = " ".join(
                                b.get("text", "")
                                for b in err_text
                                if isinstance(b, dict)
                            )
                        error_cats[_categorize_error(str(err_text))] += 1

            if has_text:
                user_count += 1

                # Response time
                if last_assistant_ts and ts:
                    delta = (ts - last_assistant_ts).total_seconds()
                    if 2 < delta < 3600:
                        response_times.append(delta)

                # Message hours + timestamps
                if ts:
                    message_hours.append(ts.hour)
                    user_timestamps.append(raw_ts)

                # First prompt
                text = _extract_text(content)
                if user_count == 1 and not first_prompt:
                    first_prompt = text[:500]

                # Check interruptions
                if isinstance(content, str):
                    if "[Request interrupted by user" in content:
                        interruptions += 1
                elif isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                            and "[Request interrupted by user" in block.get("text", "")
                        ):
                            interruptions += 1
                            break

    return SessionMetadata(
        session_id=session_id,
        project_path=project_path,
        start_time=created.isoformat(),
        end_time=modified.isoformat(),
        duration_minutes=duration_minutes,
        user_message_count=user_count,
        assistant_message_count=assistant_count,
        first_prompt=first_prompt,
        tool_counts=dict(tool_counts),
        languages=dict(languages),
        git_commits=git_commits,
        git_pushes=git_pushes,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        user_response_times=response_times,
        interruptions=interruptions,
        tool_errors=tool_errors,
        tool_error_categories=dict(error_cats),
        lines_added=lines_added,
        lines_removed=lines_removed,
        files_modified=len(modified_files),
        message_hours=message_hours,
        user_message_timestamps=user_timestamps,
        uses_task_agent=uses_task,
        uses_mcp=uses_mcp,
        uses_web_search=uses_web_search,
        uses_web_fetch=uses_web_fetch,
    )


# ---------------------------------------------------------------------------
# Stage 3: Extract Facets
# ---------------------------------------------------------------------------


def _build_transcript(
    chain: list[dict],
    metadata: SessionMetadata,
) -> str:
    """Build a text transcript from a session chain for LLM facet extraction.

    Matches the original ``BN8`` function.  Truncates to 30,000 chars.
    """
    lines: list[str] = [
        f"Session: {metadata.session_id[:8]}",
        f"Date: {metadata.start_time}",
        f"Project: {metadata.project_path}",
        f"Duration: {metadata.duration_minutes} min",
        "",
    ]
    for entry in chain:
        msg_type = entry.get("type")
        msg_body = entry.get("message") or {}
        content = msg_body.get("content", "")

        if msg_type == "user":
            text = _extract_text(content)
            if text.strip():
                lines.append(f"[User]: {text[:500]}")

        elif msg_type == "assistant":
            if isinstance(content, str):
                lines.append(f"[Assistant]: {content[:300]}")
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        lines.append(f"[Assistant]: {block.get('text', '')[:300]}")
                    elif block.get("type") == "tool_use":
                        lines.append(f"[Tool: {block.get('name', '?')}]")

    transcript = "\n".join(lines)
    return transcript[:30_000]


def _check_ollama(model: str) -> tuple[bool, str]:
    """Pre-flight check that Ollama is reachable and the model is available."""
    try:
        import ollama

        models = ollama.list()
        names = {m.model for m in models.models}
        if model not in names:
            return False, f"Model '{model}' not found. Available: {', '.join(sorted(names))}"
        return True, ""
    except ImportError:
        return False, "ollama package not installed (pip install ollama)"
    except Exception as e:
        return False, f"Ollama not reachable: {e}"


def _extract_facet_llm(transcript: str, model: str) -> dict | None:
    """Send a transcript to Ollama and parse the JSON facet response."""
    try:
        import ollama

        resp = ollama.generate(
            model=model,
            prompt=FACET_PROMPT + transcript,
            format="json",
        )
        data = json.loads(resp.response)
        if "underlying_goal" in data:
            return data
        return None
    except Exception:
        return None


def _facet_from_data(sid: str, data: dict) -> SessionFacet:
    """Create a SessionFacet from a raw dict (cache file or LLM response)."""
    return SessionFacet(
        session_id=sid,
        underlying_goal=data.get("underlying_goal", ""),
        goal_categories=data.get("goal_categories", {}),
        outcome=data.get("outcome", "unclear_from_transcript"),
        user_satisfaction_counts=data.get("user_satisfaction_counts", {}),
        claude_helpfulness=data.get("claude_helpfulness", "moderately_helpful"),
        session_type=data.get("session_type", "single_task"),
        friction_counts=data.get("friction_counts", {}),
        friction_detail=data.get("friction_detail", ""),
        primary_success=data.get("primary_success", "none"),
        brief_summary=data.get("brief_summary", ""),
    )


def _placeholder_facet(sid: str) -> SessionFacet:
    """Neutral placeholder for sessions without cached or LLM-extracted facets."""
    return SessionFacet(
        session_id=sid,
        underlying_goal="(uncached — LLM extraction not available)",
        goal_categories={"implement_feature": 1},
        outcome="unclear_from_transcript",
        user_satisfaction_counts={"likely_satisfied": 1},
        claude_helpfulness="moderately_helpful",
        session_type="single_task",
        friction_counts={},
        friction_detail="",
        primary_success="none",
        brief_summary=f"Session {sid[:8]}... (no facet data)",
    )


def extract_facets(
    metadata_map: dict[str, SessionMetadata],
    sessions: dict[str, tuple[list[dict], datetime, datetime]],
    *,
    model: str = "gpt-oss:20b-cloud",
    max_extract: int = 50,
    verbose: bool = False,
) -> dict[str, SessionFacet]:
    """Read cached facets; extract uncached via Ollama LLM; placeholder the rest."""
    facets: dict[str, SessionFacet] = {}
    cache_hits = 0
    uncached_sids: list[str] = []

    # 1. Load all cached facets
    for sid in metadata_map:
        cache_path = FACETS_DIR / f"{sid}.json"
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                facets[sid] = _facet_from_data(sid, data)
                cache_hits += 1
            except (json.JSONDecodeError, OSError):
                uncached_sids.append(sid)
        else:
            uncached_sids.append(sid)

    if verbose:
        click.echo(f"  Facet cache hits: {cache_hits}, uncached: {len(uncached_sids)}")

    # 2. LLM extraction for uncached sessions
    if uncached_sids and max_extract > 0:
        ok, reason = _check_ollama(model)
        if not ok:
            click.echo(f"  Ollama unavailable: {reason}")
            click.echo("  Falling back to placeholders for uncached sessions")
            for sid in uncached_sids:
                facets[sid] = _placeholder_facet(sid)
        else:
            to_extract = uncached_sids[:max_extract]
            remaining = uncached_sids[max_extract:]
            n = len(to_extract)
            extracted = 0
            failed = 0

            click.echo(f"  Extracting facets via {model} ({n} sessions)...")
            FACETS_DIR.mkdir(parents=True, exist_ok=True)

            for i, sid in enumerate(to_extract, 1):
                session_data = sessions.get(sid)
                if not session_data:
                    facets[sid] = _placeholder_facet(sid)
                    failed += 1
                    continue

                chain, _created, _modified = session_data
                meta = metadata_map[sid]
                transcript = _build_transcript(chain, meta)
                data = _extract_facet_llm(transcript, model)

                if data:
                    facets[sid] = _facet_from_data(sid, data)
                    # Write to cache for future runs
                    cache_path = FACETS_DIR / f"{sid}.json"
                    try:
                        cache_path.write_text(
                            json.dumps(data, indent=2), encoding="utf-8"
                        )
                    except OSError:
                        pass
                    extracted += 1
                    if verbose:
                        click.echo(f"    [{i}/{n}] {sid[:8]}... ok")
                else:
                    facets[sid] = _placeholder_facet(sid)
                    failed += 1
                    if verbose:
                        click.echo(f"    [{i}/{n}] {sid[:8]}... failed")

            click.echo(f"  Extracted: {extracted}, failed: {failed}")

            # Placeholder for sessions beyond max_extract
            for sid in remaining:
                facets[sid] = _placeholder_facet(sid)
    else:
        # max_extract == 0 or no uncached sessions
        for sid in uncached_sids:
            facets[sid] = _placeholder_facet(sid)

    # Post-filter: exclude sessions whose sole goal_category is warmup_minimal
    excluded = set()
    for sid, facet in facets.items():
        cats = facet.goal_categories
        if cats and list(cats.keys()) == ["warmup_minimal"]:
            excluded.add(sid)

    for sid in excluded:
        del facets[sid]

    if verbose and excluded:
        click.echo(f"  Excluded {len(excluded)} warmup-only sessions")

    return facets


# ---------------------------------------------------------------------------
# Stage 4: Aggregate Statistics
# ---------------------------------------------------------------------------


def aggregate_statistics(
    metadata_map: dict[str, SessionMetadata],
    facets: dict[str, SessionFacet],
    verbose: bool = False,
) -> AggregatedStats:
    """Combine all session metadata and facets into aggregate stats."""
    stats = AggregatedStats()

    # Only include sessions that survived facet post-filtering
    active_sids = set(facets.keys())
    active_metadata = {s: m for s, m in metadata_map.items() if s in active_sids}

    stats.total_sessions = len(active_metadata)
    stats.sessions_with_facets = len(facets)

    # Date range
    start_times = []
    for m in active_metadata.values():
        ts = parse_timestamp(m.start_time)
        if ts:
            start_times.append(ts)
    if start_times:
        start_times.sort()
        stats.date_range_start = start_times[0].strftime("%Y-%m-%d")
        stats.date_range_end = start_times[-1].strftime("%Y-%m-%d")

    # Days active
    unique_dates = {ts.date() for ts in start_times}
    stats.days_active = len(unique_dates)

    # Sum metadata
    tc: Counter = Counter()
    lc: Counter = Counter()
    ec: Counter = Counter()
    pc: Counter = Counter()
    all_response_times: list[float] = []
    all_hours: list[int] = []

    for m in active_metadata.values():
        stats.total_messages += m.user_message_count
        stats.total_duration_hours += m.duration_minutes / 60.0
        stats.total_input_tokens += m.input_tokens
        stats.total_output_tokens += m.output_tokens
        stats.git_commits += m.git_commits
        stats.git_pushes += m.git_pushes
        stats.total_interruptions += m.interruptions
        stats.total_tool_errors += m.tool_errors
        stats.total_lines_added += m.lines_added
        stats.total_lines_removed += m.lines_removed
        stats.total_files_modified += m.files_modified

        tc.update(m.tool_counts)
        lc.update(m.languages)
        ec.update(m.tool_error_categories)
        if m.project_path:
            pc[m.project_path] += 1

        all_response_times.extend(m.user_response_times)
        all_hours.extend(m.message_hours)

        if m.uses_task_agent:
            stats.sessions_using_task_agent += 1
        if m.uses_mcp:
            stats.sessions_using_mcp += 1
        if m.uses_web_search:
            stats.sessions_using_web_search += 1
        if m.uses_web_fetch:
            stats.sessions_using_web_fetch += 1

    stats.tool_counts = dict(tc)
    stats.languages = dict(lc)
    stats.tool_error_categories = dict(ec)
    stats.projects = dict(pc)
    stats.user_response_times = all_response_times
    stats.message_hours = all_hours

    if all_response_times:
        stats.median_response_time = round(statistics.median(all_response_times), 1)
        stats.avg_response_time = round(statistics.mean(all_response_times), 1)

    if stats.days_active > 0:
        stats.messages_per_day = round(stats.total_messages / stats.days_active, 1)

    # Aggregate facet distributions
    gc: Counter = Counter()
    oc: Counter = Counter()
    sc: Counter = Counter()
    hc: Counter = Counter()
    stc: Counter = Counter()
    fc: Counter = Counter()
    suc: Counter = Counter()

    summaries: list[dict] = []

    for sid, facet in facets.items():
        gc.update(facet.goal_categories)
        oc[facet.outcome] += 1
        sc.update(facet.user_satisfaction_counts)
        hc[facet.claude_helpfulness] += 1
        stc[facet.session_type] += 1
        fc.update(facet.friction_counts)
        if facet.primary_success and facet.primary_success != "none":
            suc[facet.primary_success] += 1

        meta = active_metadata.get(sid)
        summaries.append(
            {
                "session_id": sid,
                "brief_summary": facet.brief_summary,
                "outcome": facet.outcome,
                "helpfulness": facet.claude_helpfulness,
                "start_time": meta.start_time if meta else "",
            }
        )

    stats.goal_categories = dict(gc)
    stats.outcomes = dict(oc)
    stats.satisfaction = dict(sc)
    stats.helpfulness = dict(hc)
    stats.session_types = dict(stc)
    stats.friction = dict(fc)
    stats.success = dict(suc)

    # Session summaries: most recent first, cap at 50
    summaries.sort(key=lambda s: s.get("start_time", ""), reverse=True)
    stats.session_summaries = summaries[:50]

    # Multi-clauding detection
    stats.multi_clauding = _detect_multi_clauding(active_metadata)

    return stats


def _detect_multi_clauding(
    metadata_map: dict[str, SessionMetadata],
) -> dict[str, int]:
    """Detect cross-session overlaps within 30-minute windows.

    Matches the original Claude Code algorithm:
      - For each message i, find j (different session, within 30min)
      - Then find k (same session as i, within 30min of i, k > j)
      - If found: add sorted session pair "sid1:sid2" to overlap_events set
      - Track involved messages in user_messages_during set
    """
    # Collect (epoch_ms, session_id)
    all_msgs: list[tuple[float, str]] = []
    for sid, m in metadata_map.items():
        for ts_str in m.user_message_timestamps:
            ts = parse_timestamp(ts_str)
            if ts:
                all_msgs.append((ts.timestamp() * 1000, sid))

    if not all_msgs:
        return {"overlap_events": 0, "sessions_involved": 0, "user_messages_during": 0}

    all_msgs.sort(key=lambda x: x[0])
    window_ms = 30 * 60 * 1000
    n = len(all_msgs)

    overlap_pairs: set[str] = set()  # "sid1:sid2" (sorted)
    involved_messages: set[str] = set()  # "ts:sid"

    for i in range(n):
        ts_i, sid_i = all_msgs[i]
        for j in range(i + 1, n):
            ts_j, sid_j = all_msgs[j]
            if ts_j - ts_i > window_ms:
                break
            if sid_j != sid_i:
                # Found different session within window; look for same-session
                for k in range(j + 1, n):
                    ts_k, sid_k = all_msgs[k]
                    if ts_k - ts_i > window_ms:
                        break
                    if sid_k == sid_i:
                        pair_key = ":".join(sorted([sid_i, sid_j]))
                        overlap_pairs.add(pair_key)
                        involved_messages.add(f"{ts_i}:{sid_i}")
                        involved_messages.add(f"{ts_j}:{sid_j}")
                        involved_messages.add(f"{ts_k}:{sid_k}")
                        break

    # Extract unique sessions from pairs
    sessions_involved: set[str] = set()
    for pair in overlap_pairs:
        parts = pair.split(":")
        sessions_involved.update(parts)

    return {
        "overlap_events": len(overlap_pairs),
        "sessions_involved": len(sessions_involved),
        "user_messages_during": len(involved_messages),
    }


# ---------------------------------------------------------------------------
# Stage 5: Generate Insights
# ---------------------------------------------------------------------------


def _build_stats_context(stats: AggregatedStats) -> str:
    """Build a concise text summary of AggregatedStats for LLM prompts."""
    lines: list[str] = []

    # Overview
    lines.append(f"Sessions: {stats.total_sessions} ({stats.sessions_with_facets} with facets)")
    lines.append(f"Messages: {stats.total_messages}")
    lines.append(f"Duration: {stats.total_duration_hours:.1f} hours")
    lines.append(f"Date range: {stats.date_range_start} to {stats.date_range_end}")
    lines.append(f"Days active: {stats.days_active}, Msgs/day: {stats.messages_per_day:.1f}")

    # Top tools
    top_tools = sorted(stats.tool_counts.items(), key=lambda x: -x[1])[:8]
    if top_tools:
        lines.append("Top tools: " + ", ".join(f"{k} ({v})" for k, v in top_tools))

    # Top goal categories
    top_goals = sorted(stats.goal_categories.items(), key=lambda x: -x[1])[:8]
    if top_goals:
        lines.append("Top goals: " + ", ".join(
            f"{DISPLAY_NAMES.get(k, k)} ({v})" for k, v in top_goals
        ))

    # Distributions
    if stats.outcomes:
        lines.append("Outcomes: " + ", ".join(
            f"{DISPLAY_NAMES.get(k, k)}: {v}" for k, v in stats.outcomes.items()
        ))
    if stats.satisfaction:
        lines.append("Satisfaction: " + ", ".join(
            f"{DISPLAY_NAMES.get(k, k)}: {v}" for k, v in stats.satisfaction.items()
        ))
    if stats.helpfulness:
        lines.append("Helpfulness: " + ", ".join(
            f"{DISPLAY_NAMES.get(k, k)}: {v}" for k, v in stats.helpfulness.items()
        ))
    if stats.session_types:
        lines.append("Session types: " + ", ".join(
            f"{DISPLAY_NAMES.get(k, k)}: {v}" for k, v in stats.session_types.items()
        ))

    # Friction & success
    top_friction = sorted(stats.friction.items(), key=lambda x: -x[1])[:6]
    if top_friction:
        lines.append("Top friction: " + ", ".join(
            f"{DISPLAY_NAMES.get(k, k)} ({v})" for k, v in top_friction
        ))
    top_success = sorted(stats.success.items(), key=lambda x: -x[1])[:6]
    if top_success:
        lines.append("Top success: " + ", ".join(
            f"{DISPLAY_NAMES.get(k, k)} ({v})" for k, v in top_success
        ))

    # Git & code
    lines.append(f"Git: {stats.git_commits} commits, {stats.git_pushes} pushes")
    lines.append(
        f"Code: +{stats.total_lines_added} -{stats.total_lines_removed} lines, "
        f"{stats.total_files_modified} files modified"
    )
    lines.append(f"Tool errors: {stats.total_tool_errors}")
    if stats.tool_error_categories:
        lines.append("Error categories: " + ", ".join(
            f"{k} ({v})" for k, v in sorted(
                stats.tool_error_categories.items(), key=lambda x: -x[1]
            )[:5]
        ))

    # Feature usage
    features = []
    if stats.sessions_using_task_agent:
        features.append(f"Task agent: {stats.sessions_using_task_agent} sessions")
    if stats.sessions_using_mcp:
        features.append(f"MCP: {stats.sessions_using_mcp} sessions")
    if stats.sessions_using_web_search:
        features.append(f"Web search: {stats.sessions_using_web_search} sessions")
    if stats.sessions_using_web_fetch:
        features.append(f"Web fetch: {stats.sessions_using_web_fetch} sessions")
    if features:
        lines.append("Features: " + ", ".join(features))

    # Multi-clauding
    mc = stats.multi_clauding
    if mc["overlap_events"] > 0:
        lines.append(
            f"Multi-clauding: {mc['overlap_events']} overlap events, "
            f"{mc['sessions_involved']} sessions, "
            f"{mc['user_messages_during']} msgs during overlaps"
        )

    # Top session summaries
    top_summaries = stats.session_summaries[:5]
    if top_summaries:
        lines.append("Top session summaries:")
        for s in top_summaries:
            summary = s.get("brief_summary", "")
            outcome = s.get("outcome", "")
            if summary:
                lines.append(f"  - {summary} (outcome: {outcome})")

    return "\n".join(lines)


def _generate_insight_section(prompt: str, context: str, model: str) -> dict | None:
    """Send an insight prompt to Ollama and parse the JSON response."""
    try:
        import ollama

        resp = ollama.generate(
            model=model,
            prompt=prompt + context,
            format="json",
        )
        return json.loads(resp.response)
    except Exception:
        return None


def _mock_insights(stats: AggregatedStats) -> InsightResults:
    """Return placeholder insights (fallback when LLM unavailable)."""
    return InsightResults(
        at_a_glance={
            "whats_working": (
                "Your usage data shows productive patterns across "
                f"{stats.total_sessions} sessions. LLM-generated insights "
                "would appear here analyzing your specific workflow strengths."
            ),
            "whats_hindering": (
                "Friction analysis would identify specific pain points from "
                "your session data. This section requires LLM analysis."
            ),
            "quick_wins": (
                "Feature recommendations based on your tool usage patterns "
                "would appear here."
            ),
            "ambitious_workflows": (
                "Forward-looking workflow suggestions based on your usage "
                "patterns would appear here."
            ),
        },
        project_areas={
            "areas": [
                {
                    "name": "(LLM analysis required)",
                    "session_count": stats.total_sessions,
                    "description": (
                        "Project area detection requires LLM analysis of session "
                        "summaries. Run with Ollama available for this section."
                    ),
                },
            ],
        },
        interaction_style={
            "narrative": (
                f"Across **{stats.total_sessions}** sessions with "
                f"**{stats.total_messages}** messages, your interaction data "
                "has been collected. LLM analysis would provide a detailed "
                "narrative of your interaction style here."
            ),
            "key_pattern": (
                "Interaction style analysis requires LLM processing of session transcripts."
            ),
        },
        what_works={
            "intro": (
                "This section identifies impressive workflows from your sessions."
            ),
            "impressive_workflows": [
                {
                    "title": "(LLM analysis required)",
                    "description": (
                        "Workflow analysis requires LLM processing. "
                        "The real data shows you've used tools like "
                        + ", ".join(
                            f"{k} ({v}x)"
                            for k, v in sorted(
                                stats.tool_counts.items(), key=lambda x: -x[1]
                            )[:5]
                        )
                        + "."
                    ),
                },
            ],
        },
        friction_analysis={
            "intro": (
                f"Across your sessions, {stats.total_tool_errors} tool errors "
                "were detected. Detailed friction analysis requires LLM processing."
            ),
            "categories": [
                {
                    "category": "(LLM analysis required)",
                    "description": "Friction categorization requires LLM analysis.",
                    "examples": [
                        "Example friction points would be extracted from session data.",
                    ],
                },
            ],
        },
        suggestions={
            "claude_md_additions": [
                {
                    "addition": "(LLM analysis required)",
                    "why": "CLAUDE.md suggestions require LLM analysis of session patterns.",
                    "prompt_scaffold": "",
                },
            ],
            "features_to_try": [
                {
                    "feature": "(LLM analysis required)",
                    "one_liner": "Feature recommendations require LLM analysis.",
                    "why_for_you": "",
                    "example_code": "",
                },
            ],
            "usage_patterns": [
                {
                    "title": "(LLM analysis required)",
                    "suggestion": "Usage pattern suggestions require LLM analysis.",
                    "detail": "",
                    "copyable_prompt": "",
                },
            ],
        },
        on_the_horizon={
            "intro": "Future opportunity analysis requires LLM processing.",
            "opportunities": [
                {
                    "title": "(LLM analysis required)",
                    "whats_possible": "Opportunity analysis requires LLM processing.",
                    "how_to_try": "",
                    "copyable_prompt": "",
                },
            ],
        },
        fun_ending={
            "headline": f"You've had {stats.total_sessions} conversations with Claude Code",
            "detail": (
                f"That's {stats.total_messages} messages, "
                f"{stats.git_commits} commits, and "
                f"{stats.days_active} active days. "
                "A memorable moment analysis requires LLM processing."
            ),
        },
    )


def generate_insights(
    stats: AggregatedStats,
    *,
    model: str = "gpt-oss:20b-cloud",
    verbose: bool = False,
) -> InsightResults:
    """Generate insights via Ollama LLM; falls back to mock data if unavailable."""
    mock = _mock_insights(stats)
    context = _build_stats_context(stats)

    ok, reason = _check_ollama(model)
    if not ok:
        click.echo(f"  Ollama unavailable: {reason}")
        click.echo("  Falling back to placeholder insights")
        return mock

    sections = [
        ("project_areas", INSIGHT_PROMPT_PROJECT_AREAS),
        ("interaction_style", INSIGHT_PROMPT_INTERACTION_STYLE),
        ("what_works", INSIGHT_PROMPT_WHAT_WORKS),
        ("friction_analysis", INSIGHT_PROMPT_FRICTION),
        ("suggestions", INSIGHT_PROMPT_SUGGESTIONS),
        ("on_the_horizon", INSIGHT_PROMPT_HORIZON),
        ("fun_ending", INSIGHT_PROMPT_FUN_ENDING),
    ]

    results: dict[str, dict] = {}
    for i, (name, prompt) in enumerate(sections, 1):
        data = _generate_insight_section(prompt, context, model)
        if data is not None:
            results[name] = data
            click.echo(f"  [{i}/8] {name}... ok")
        else:
            results[name] = getattr(mock, name)
            click.echo(f"  [{i}/8] {name}... failed (using fallback)")

    # at_a_glance uses the other 7 results as additional context
    other_summary = json.dumps(
        {k: v for k, v in results.items()},
        indent=None,
        default=str,
    )
    aag_context = context + "\n\nPRIOR ANALYSIS RESULTS:\n" + other_summary
    aag_data = _generate_insight_section(INSIGHT_PROMPT_AT_A_GLANCE, aag_context, model)
    if aag_data is not None:
        results["at_a_glance"] = aag_data
        click.echo("  [8/8] at_a_glance... ok")
    else:
        results["at_a_glance"] = mock.at_a_glance
        click.echo("  [8/8] at_a_glance... failed (using fallback)")

    return InsightResults(
        at_a_glance=results["at_a_glance"],
        project_areas=results["project_areas"],
        interaction_style=results["interaction_style"],
        what_works=results["what_works"],
        friction_analysis=results["friction_analysis"],
        suggestions=results["suggestions"],
        on_the_horizon=results["on_the_horizon"],
        fun_ending=results["fun_ending"],
    )


# ---------------------------------------------------------------------------
# Stage 6: Build HTML Report
# ---------------------------------------------------------------------------


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def markdown_lite(s: str) -> str:
    """Escape HTML and convert **bold** to <strong>."""
    s = html_escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return s


def markdown_to_html(text: str) -> str:
    """Convert simple markdown text to HTML paragraphs."""
    paragraphs = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = markdown_lite(line)
        if line.startswith("- "):
            line = "&bull; " + line[2:]
        paragraphs.append(f"<p>{line}</p>")
    return "\n".join(paragraphs)


def render_bar_chart(
    data: dict[str, int],
    color: str,
    max_bars: int = 8,
    ordered_keys: list[str] | None = None,
) -> str:
    """Render horizontal bar chart rows."""
    if not data:
        return '<div class="empty">No data</div>'

    if ordered_keys:
        items = [(k, data.get(k, 0)) for k in ordered_keys if data.get(k, 0) > 0]
    else:
        items = sorted(data.items(), key=lambda x: -x[1])

    items = items[:max_bars]
    if not items:
        return '<div class="empty">No data</div>'

    max_val = max(v for _, v in items) if items else 1
    rows = []
    for key, val in items:
        label = DISPLAY_NAMES.get(key, key)
        width = (val / max_val) * 100 if max_val > 0 else 0
        rows.append(
            f'<div class="bar-row">\n'
            f'        <div class="bar-label">{html_escape(label)}</div>\n'
            f'        <div class="bar-track"><div class="bar-fill" '
            f'style="width:{width}%;background:{color}"></div></div>\n'
            f'        <div class="bar-value">{val}</div>\n'
            f"      </div>"
        )
    return "\n".join(rows)


def render_response_time_histogram(times: list[float]) -> str:
    """Bucket response times into standard ranges."""
    buckets = [
        ("2-10s", 2, 10),
        ("10-30s", 10, 30),
        ("30s-1m", 30, 60),
        ("1-2m", 60, 120),
        ("2-5m", 120, 300),
        ("5-15m", 300, 900),
        (">15m", 900, float("inf")),
    ]
    counts: dict[str, int] = {}
    for label, lo, hi in buckets:
        counts[label] = sum(1 for t in times if lo <= t < hi)

    max_val = max(counts.values()) if counts else 1
    rows = []
    for label, _, _ in buckets:
        val = counts[label]
        width = (val / max_val) * 100 if max_val > 0 else 0
        rows.append(
            f'<div class="bar-row">\n'
            f'        <div class="bar-label">{label}</div>\n'
            f'        <div class="bar-track"><div class="bar-fill" '
            f'style="width:{width}%;background:#6366f1"></div></div>\n'
            f'        <div class="bar-value">{val}</div>\n'
            f"      </div>"
        )
    return "\n".join(rows)


def render_time_of_day_chart(hours: list[int]) -> str:
    """Bucket hours into Morning/Afternoon/Evening/Night."""
    periods = [
        ("Morning (6-12)", range(6, 12)),
        ("Afternoon (12-18)", range(12, 18)),
        ("Evening (18-24)", range(18, 24)),
        ("Night (0-6)", range(0, 6)),
    ]
    counter = Counter(hours)
    period_counts = {}
    for label, hr_range in periods:
        period_counts[label] = sum(counter.get(h, 0) for h in hr_range)

    max_val = max(period_counts.values()) if period_counts else 1
    rows = []
    for label, _ in periods:
        val = period_counts[label]
        width = (val / max_val) * 100 if max_val > 0 else 0
        rows.append(
            f'      <div class="bar-row">\n'
            f'        <div class="bar-label">{label}</div>\n'
            f'        <div class="bar-track"><div class="bar-fill" '
            f'style="width:{width}%;background:#8b5cf6"></div></div>\n'
            f'        <div class="bar-value">{val}</div>\n'
            f"      </div>"
        )
    return "\n".join(rows)


def _build_raw_hour_counts_json(hours: list[int]) -> str:
    """Build JSON object of hour -> count for the timezone JS."""
    counter = Counter(hours)
    items = sorted(counter.items())
    pairs = [f'"{h}":{c}' for h, c in items]
    return "{" + ",".join(pairs) + "}"


def build_html_report(
    stats: AggregatedStats,
    insights: InsightResults,
    verbose: bool = False,
) -> str:
    """Build the self-contained HTML report."""
    aag = insights.at_a_glance
    pa = insights.project_areas
    ist = insights.interaction_style
    ww = insights.what_works
    fa = insights.friction_analysis
    sug = insights.suggestions
    oth = insights.on_the_horizon
    fun = insights.fun_ending

    # Stats bar — matches original: Messages, Lines, Files, Days, Msgs/Day
    stats_bar = (
        f'<div class="stat"><div class="stat-value">'
        f"{stats.total_messages:,}</div>"
        f'<div class="stat-label">Messages</div></div>\n'
        f'      <div class="stat"><div class="stat-value">'
        f"+{stats.total_lines_added:,}/-{stats.total_lines_removed:,}</div>"
        f'<div class="stat-label">Lines</div></div>\n'
        f'      <div class="stat"><div class="stat-value">'
        f"{stats.total_files_modified}</div>"
        f'<div class="stat-label">Files</div></div>\n'
        f'      <div class="stat"><div class="stat-value">'
        f"{stats.days_active}</div>"
        f'<div class="stat-label">Days</div></div>\n'
        f'      <div class="stat"><div class="stat-value">'
        f"{stats.messages_per_day}</div>"
        f'<div class="stat-label">Msgs/Day</div></div>'
    )

    # At a glance section
    at_a_glance_html = ""
    if aag:
        sections = []
        pairs = [
            ("What's working:", aag.get("whats_working", ""), "#section-wins"),
            (
                "What's hindering you:",
                aag.get("whats_hindering", ""),
                "#section-friction",
            ),
            ("Quick wins to try:", aag.get("quick_wins", ""), "#section-features"),
            (
                "Ambitious workflows:",
                aag.get("ambitious_workflows", ""),
                "#section-horizon",
            ),
        ]
        for title, text, link in pairs:
            sections.append(
                f'<div class="glance-section"><strong>{title}</strong> '
                f"{markdown_lite(text)} "
                f'<a href="{link}" class="see-more">See more &rarr;</a></div>'
            )
        at_a_glance_html = (
            '<div class="at-a-glance">\n'
            '      <div class="glance-title">At a Glance</div>\n'
            '      <div class="glance-sections">\n        '
            + "\n        ".join(sections)
            + "\n      </div>\n    </div>"
        )

    # Project areas
    project_areas_html = ""
    areas = pa.get("areas", [])
    if areas:
        cards = []
        for area in areas:
            cards.append(
                f'<div class="project-area">\n'
                f'          <div class="area-header">\n'
                f'            <span class="area-name">{html_escape(area.get("name", ""))}</span>\n'
                f'            <span class="area-count">~{area.get("session_count", 0)} sessions</span>\n'
                f"          </div>\n"
                f'          <div class="area-desc">{html_escape(area.get("description", ""))}</div>\n'
                f"        </div>"
            )
        project_areas_html = (
            '<div class="project-areas">\n      '
            + "\n      ".join(cards)
            + "\n    </div>"
        )

    # Interaction style
    narrative_html = ""
    if ist:
        narrative_text = markdown_to_html(ist.get("narrative", ""))
        key_pattern = markdown_lite(ist.get("key_pattern", ""))
        narrative_html = (
            f'<div class="narrative">\n      {narrative_text}\n'
            f'      <div class="key-insight"><strong>Key pattern:</strong> '
            f"{key_pattern}</div>\n    </div>"
        )

    # What works (impressive things)
    big_wins_html = ""
    ww_data = ww.get("impressive_workflows", [])
    if ww_data:
        intro = html_escape(ww.get("intro", ""))
        cards = []
        for w in ww_data:
            cards.append(
                f'<div class="big-win">\n'
                f'          <div class="big-win-title">{html_escape(w.get("title", ""))}</div>\n'
                f'          <div class="big-win-desc">{html_escape(w.get("description", ""))}</div>\n'
                f"        </div>"
            )
        big_wins_html = (
            f'<p class="section-intro">{intro}</p>\n'
            '    <div class="big-wins">\n      '
            + "\n      ".join(cards)
            + "\n    </div>"
        )

    # Friction analysis
    friction_html = ""
    fa_cats = fa.get("categories", [])
    if fa_cats:
        intro = html_escape(fa.get("intro", ""))
        cards = []
        for cat in fa_cats:
            examples = cat.get("examples", [])
            li_items = "".join(f"<li>{html_escape(e)}</li>" for e in examples)
            cards.append(
                f'<div class="friction-category">\n'
                f'          <div class="friction-title">{html_escape(cat.get("category", ""))}</div>\n'
                f'          <div class="friction-desc">{html_escape(cat.get("description", ""))}</div>\n'
                f'          <ul class="friction-examples">{li_items}</ul>\n'
                f"        </div>"
            )
        friction_html = (
            f'<p class="section-intro">{intro}</p>\n'
            '    <div class="friction-categories">\n      '
            + "\n      ".join(cards)
            + "\n    </div>"
        )

    # Suggestions: CLAUDE.md additions
    claude_md_html = ""
    cmd_additions = sug.get("claude_md_additions", [])
    if cmd_additions:
        items = []
        for i, item in enumerate(cmd_additions):
            code = html_escape(item.get("addition", ""))
            why = html_escape(item.get("why", ""))
            scaffold = html_escape(item.get("prompt_scaffold", ""))
            data_text = html_escape(scaffold if scaffold else code)
            items.append(
                f'<div class="claude-md-item">\n'
                f'          <input type="checkbox" id="cmd-{i}" class="cmd-checkbox" checked '
                f'data-text="{data_text}">\n'
                f'          <label for="cmd-{i}">\n'
                f'            <code class="cmd-code">{code}</code>\n'
                f'            <button class="copy-btn" onclick="copyCmdItem({i})">Copy</button>\n'
                f"          </label>\n"
                f'          <div class="cmd-why">{why}</div>\n'
                f"        </div>"
            )
        claude_md_html = (
            '<div class="claude-md-section">\n'
            "      <h3>Suggested CLAUDE.md Additions</h3>\n"
            '      <p style="font-size: 12px; color: #64748b; margin-bottom: 12px;">'
            "Just copy this into Claude Code to add it to your CLAUDE.md.</p>\n"
            '      <div class="claude-md-actions">\n'
            '        <button class="copy-all-btn" onclick="copyAllCheckedClaudeMd()">'
            "Copy All Checked</button>\n"
            "      </div>\n      " + "\n      ".join(items) + "\n    </div>"
        )

    # Features to try
    features_html = ""
    features = sug.get("features_to_try", [])
    if features:
        cards = []
        for feat in features:
            example_code = feat.get("example_code", "")
            code_section = ""
            if example_code:
                code_section = (
                    f'<div class="feature-code">\n'
                    f"              <code>{html_escape(example_code)}</code>\n"
                    f'              <button class="copy-btn" onclick="copyText(this)">Copy</button>\n'
                    f"            </div>"
                )
            cards.append(
                f'<div class="feature-card">\n'
                f'          <div class="feature-title">{html_escape(feat.get("feature", ""))}</div>\n'
                f'          <div class="feature-oneliner">{html_escape(feat.get("one_liner", ""))}</div>\n'
                f'          <div class="feature-why"><strong>Why for you:</strong> '
                f"{html_escape(feat.get('why_for_you', ''))}</div>\n"
                f"          {code_section}\n"
                f"        </div>"
            )
        features_html = (
            '<div class="features-section">\n      '
            + "\n      ".join(cards)
            + "\n    </div>"
        )

    # Usage patterns
    patterns_html = ""
    patterns = sug.get("usage_patterns", [])
    if patterns:
        cards = []
        for pat in patterns:
            prompt = pat.get("copyable_prompt", "")
            prompt_section = ""
            if prompt:
                prompt_section = (
                    f'<div class="copyable-prompt-section">\n'
                    f'            <div class="prompt-label">Paste into Claude Code:</div>\n'
                    f'            <div class="copyable-prompt-row">\n'
                    f'              <code class="copyable-prompt">{html_escape(prompt)}</code>\n'
                    f'              <button class="copy-btn" onclick="copyText(this)">Copy</button>\n'
                    f"            </div>\n"
                    f"          </div>"
                )
            cards.append(
                f'<div class="pattern-card">\n'
                f'          <div class="pattern-title">{html_escape(pat.get("title", ""))}</div>\n'
                f'          <div class="pattern-summary">{html_escape(pat.get("suggestion", ""))}</div>\n'
                f'          <div class="pattern-detail">{html_escape(pat.get("detail", ""))}</div>\n'
                f"          {prompt_section}\n"
                f"        </div>"
            )
        patterns_html = (
            '<div class="patterns-section">\n      '
            + "\n      ".join(cards)
            + "\n    </div>"
        )

    # Horizon
    horizon_html = ""
    opps = oth.get("opportunities", [])
    if opps:
        intro = html_escape(oth.get("intro", ""))
        cards = []
        for opp in opps:
            prompt = opp.get("copyable_prompt", "")
            prompt_section = ""
            if prompt:
                prompt_section = (
                    f'<div class="pattern-prompt"><div class="prompt-label">'
                    f"Paste into Claude Code:</div>"
                    f"<code>{html_escape(prompt)}</code>"
                    f'<button class="copy-btn" onclick="copyText(this)">Copy</button></div>'
                )
            cards.append(
                f'<div class="horizon-card">\n'
                f'          <div class="horizon-title">{html_escape(opp.get("title", ""))}</div>\n'
                f'          <div class="horizon-possible">{html_escape(opp.get("whats_possible", ""))}</div>\n'
                f'          <div class="horizon-tip"><strong>Getting started:</strong> '
                f"{html_escape(opp.get('how_to_try', ''))}</div>\n"
                f"          {prompt_section}\n"
                f"        </div>"
            )
        horizon_html = (
            f'<p class="section-intro">{intro}</p>\n'
            '    <div class="horizon-section">\n      '
            + "\n      ".join(cards)
            + "\n    </div>"
        )

    # Fun ending
    fun_html = ""
    if fun:
        fun_html = (
            f'<div class="fun-ending">\n'
            f'      <div class="fun-headline">{markdown_lite(fun.get("headline", ""))}</div>\n'
            f'      <div class="fun-detail">{html_escape(fun.get("detail", ""))}</div>\n'
            f"    </div>"
        )

    # Charts
    tool_chart = render_bar_chart(stats.tool_counts, "#0891b2", max_bars=6)
    goal_chart = render_bar_chart(stats.goal_categories, "#2563eb", max_bars=6)
    lang_chart = render_bar_chart(stats.languages, "#10b981", max_bars=6)
    session_type_chart = render_bar_chart(stats.session_types, "#8b5cf6")
    outcome_chart = render_bar_chart(stats.outcomes, "#8b5cf6")
    satisfaction_chart = render_bar_chart(stats.satisfaction, "#eab308")
    helpfulness_chart = render_bar_chart(stats.helpfulness, "#16a34a")
    friction_chart = render_bar_chart(stats.friction, "#dc2626", max_bars=6)
    success_chart = render_bar_chart(stats.success, "#16a34a", max_bars=6)
    error_chart = render_bar_chart(stats.tool_error_categories, "#dc2626", max_bars=6)
    response_time_chart = render_response_time_histogram(stats.user_response_times)
    time_of_day_chart = render_time_of_day_chart(stats.message_hours)
    raw_hours_json = _build_raw_hour_counts_json(stats.message_hours)

    # Multi-clauding section
    mc = stats.multi_clauding
    total_user_msgs = sum(
        1
        for _ in stats.message_hours  # rough proxy — same length as message_hours
    )
    mc_pct = (
        round(mc["user_messages_during"] / total_user_msgs * 100)
        if total_user_msgs > 0
        else 0
    )
    multi_clauding_html = ""
    if mc["overlap_events"] > 0:
        multi_clauding_html = (
            f'<div class="chart-card" style="margin: 24px 0;">\n'
            f'      <div class="chart-title">Multi-Clauding (Parallel Sessions)</div>\n'
            f'      <div style="display: flex; gap: 24px; margin: 12px 0;">\n'
            f'          <div style="text-align: center;">\n'
            f'            <div style="font-size: 24px; font-weight: 700; color: #7c3aed;">'
            f"{mc['overlap_events']}</div>\n"
            f'            <div style="font-size: 11px; color: #64748b; text-transform: uppercase;">'
            f"Overlap Events</div>\n"
            f"          </div>\n"
            f'          <div style="text-align: center;">\n'
            f'            <div style="font-size: 24px; font-weight: 700; color: #7c3aed;">'
            f"{mc['sessions_involved']}</div>\n"
            f'            <div style="font-size: 11px; color: #64748b; text-transform: uppercase;">'
            f"Sessions Involved</div>\n"
            f"          </div>\n"
            f'          <div style="text-align: center;">\n'
            f'            <div style="font-size: 24px; font-weight: 700; color: #7c3aed;">'
            f"{mc_pct}%</div>\n"
            f'            <div style="font-size: 11px; color: #64748b; text-transform: uppercase;">'
            f"Of Messages</div>\n"
            f"          </div>\n"
            f"        </div>\n"
            f'        <p style="font-size: 13px; color: #475569; margin-top: 12px;">\n'
            f"          You run multiple Claude Code sessions simultaneously. Multi-clauding is detected when sessions\n"
            f"          overlap in time, suggesting parallel workflows.\n"
            f"        </p>\n"
            f"    </div>"
        )

    # Assemble HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Claude Code Insights</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background: #f8fafc; color: #334155; line-height: 1.65; padding: 48px 24px; }}
    .container {{ max-width: 800px; margin: 0 auto; }}
    h1 {{ font-size: 32px; font-weight: 700; color: #0f172a; margin-bottom: 8px; }}
    h2 {{ font-size: 20px; font-weight: 600; color: #0f172a; margin-top: 48px; margin-bottom: 16px; }}
    .subtitle {{ color: #64748b; font-size: 15px; margin-bottom: 32px; }}
    .nav-toc {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 24px 0 32px 0; padding: 16px; background: white; border-radius: 8px; border: 1px solid #e2e8f0; }}
    .nav-toc a {{ font-size: 12px; color: #64748b; text-decoration: none; padding: 6px 12px; border-radius: 6px; background: #f1f5f9; transition: all 0.15s; }}
    .nav-toc a:hover {{ background: #e2e8f0; color: #334155; }}
    .stats-row {{ display: flex; gap: 24px; margin-bottom: 40px; padding: 20px 0; border-top: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; flex-wrap: wrap; }}
    .stat {{ text-align: center; }}
    .stat-value {{ font-size: 24px; font-weight: 700; color: #0f172a; }}
    .stat-label {{ font-size: 11px; color: #64748b; text-transform: uppercase; }}
    .at-a-glance {{ background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 1px solid #f59e0b; border-radius: 12px; padding: 20px 24px; margin-bottom: 32px; }}
    .glance-title {{ font-size: 16px; font-weight: 700; color: #92400e; margin-bottom: 16px; }}
    .glance-sections {{ display: flex; flex-direction: column; gap: 12px; }}
    .glance-section {{ font-size: 14px; color: #78350f; line-height: 1.6; }}
    .glance-section strong {{ color: #92400e; }}
    .see-more {{ color: #b45309; text-decoration: none; font-size: 13px; white-space: nowrap; }}
    .see-more:hover {{ text-decoration: underline; }}
    .project-areas {{ display: flex; flex-direction: column; gap: 12px; margin-bottom: 32px; }}
    .project-area {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }}
    .area-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
    .area-name {{ font-weight: 600; font-size: 15px; color: #0f172a; }}
    .area-count {{ font-size: 12px; color: #64748b; background: #f1f5f9; padding: 2px 8px; border-radius: 4px; }}
    .area-desc {{ font-size: 14px; color: #475569; line-height: 1.5; }}
    .narrative {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
    .narrative p {{ margin-bottom: 12px; font-size: 14px; color: #475569; line-height: 1.7; }}
    .key-insight {{ background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px 16px; margin-top: 12px; font-size: 14px; color: #166534; }}
    .section-intro {{ font-size: 14px; color: #64748b; margin-bottom: 16px; }}
    .big-wins {{ display: flex; flex-direction: column; gap: 12px; margin-bottom: 24px; }}
    .big-win {{ background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 16px; }}
    .big-win-title {{ font-weight: 600; font-size: 15px; color: #166534; margin-bottom: 8px; }}
    .big-win-desc {{ font-size: 14px; color: #15803d; line-height: 1.5; }}
    .friction-categories {{ display: flex; flex-direction: column; gap: 16px; margin-bottom: 24px; }}
    .friction-category {{ background: #fef2f2; border: 1px solid #fca5a5; border-radius: 8px; padding: 16px; }}
    .friction-title {{ font-weight: 600; font-size: 15px; color: #991b1b; margin-bottom: 6px; }}
    .friction-desc {{ font-size: 13px; color: #7f1d1d; margin-bottom: 10px; }}
    .friction-examples {{ margin: 0 0 0 20px; font-size: 13px; color: #334155; }}
    .friction-examples li {{ margin-bottom: 4px; }}
    .claude-md-section {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 16px; margin-bottom: 20px; }}
    .claude-md-section h3 {{ font-size: 14px; font-weight: 600; color: #1e40af; margin: 0 0 12px 0; }}
    .claude-md-actions {{ margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #dbeafe; }}
    .copy-all-btn {{ background: #2563eb; color: white; border: none; border-radius: 4px; padding: 6px 12px; font-size: 12px; cursor: pointer; font-weight: 500; transition: all 0.2s; }}
    .copy-all-btn:hover {{ background: #1d4ed8; }}
    .copy-all-btn.copied {{ background: #16a34a; }}
    .claude-md-item {{ display: flex; flex-wrap: wrap; align-items: flex-start; gap: 8px; padding: 10px 0; border-bottom: 1px solid #dbeafe; }}
    .claude-md-item:last-child {{ border-bottom: none; }}
    .cmd-checkbox {{ margin-top: 2px; }}
    .cmd-code {{ background: white; padding: 8px 12px; border-radius: 4px; font-size: 12px; color: #1e40af; border: 1px solid #bfdbfe; font-family: monospace; display: block; white-space: pre-wrap; word-break: break-word; flex: 1; }}
    .cmd-why {{ font-size: 12px; color: #64748b; width: 100%; padding-left: 24px; margin-top: 4px; }}
    .features-section, .patterns-section {{ display: flex; flex-direction: column; gap: 12px; margin: 16px 0; }}
    .feature-card {{ background: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 16px; }}
    .pattern-card {{ background: #f0f9ff; border: 1px solid #7dd3fc; border-radius: 8px; padding: 16px; }}
    .feature-title, .pattern-title {{ font-weight: 600; font-size: 15px; color: #0f172a; margin-bottom: 6px; }}
    .feature-oneliner {{ font-size: 14px; color: #475569; margin-bottom: 8px; }}
    .pattern-summary {{ font-size: 14px; color: #475569; margin-bottom: 8px; }}
    .feature-why, .pattern-detail {{ font-size: 13px; color: #334155; line-height: 1.5; }}
    .feature-code {{ background: #f8fafc; padding: 12px; border-radius: 6px; margin-top: 12px; border: 1px solid #e2e8f0; display: flex; align-items: flex-start; gap: 8px; }}
    .feature-code code {{ flex: 1; font-family: monospace; font-size: 12px; color: #334155; white-space: pre-wrap; }}
    .copyable-prompt-section {{ margin-top: 12px; padding-top: 12px; border-top: 1px solid #e2e8f0; }}
    .copyable-prompt-row {{ display: flex; align-items: flex-start; gap: 8px; }}
    .copyable-prompt {{ flex: 1; background: #f8fafc; padding: 10px 12px; border-radius: 4px; font-family: monospace; font-size: 12px; color: #334155; border: 1px solid #e2e8f0; white-space: pre-wrap; line-height: 1.5; }}
    .pattern-prompt {{ background: #f8fafc; padding: 12px; border-radius: 6px; margin-top: 12px; border: 1px solid #e2e8f0; }}
    .pattern-prompt code {{ font-family: monospace; font-size: 12px; color: #334155; display: block; white-space: pre-wrap; margin-bottom: 8px; }}
    .prompt-label {{ font-size: 11px; font-weight: 600; text-transform: uppercase; color: #64748b; margin-bottom: 6px; }}
    .copy-btn {{ background: #e2e8f0; border: none; border-radius: 4px; padding: 4px 8px; font-size: 11px; cursor: pointer; color: #475569; flex-shrink: 0; }}
    .copy-btn:hover {{ background: #cbd5e1; }}
    .charts-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 24px 0; }}
    .chart-card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }}
    .chart-title {{ font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 12px; }}
    .bar-row {{ display: flex; align-items: center; margin-bottom: 6px; }}
    .bar-label {{ width: 100px; font-size: 11px; color: #475569; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .bar-track {{ flex: 1; height: 6px; background: #f1f5f9; border-radius: 3px; margin: 0 8px; }}
    .bar-fill {{ height: 100%; border-radius: 3px; }}
    .bar-value {{ width: 28px; font-size: 11px; font-weight: 500; color: #64748b; text-align: right; }}
    .empty {{ color: #94a3b8; font-size: 13px; }}
    .horizon-section {{ display: flex; flex-direction: column; gap: 16px; }}
    .horizon-card {{ background: linear-gradient(135deg, #faf5ff 0%, #f5f3ff 100%); border: 1px solid #c4b5fd; border-radius: 8px; padding: 16px; }}
    .horizon-title {{ font-weight: 600; font-size: 15px; color: #5b21b6; margin-bottom: 8px; }}
    .horizon-possible {{ font-size: 14px; color: #334155; margin-bottom: 10px; line-height: 1.5; }}
    .horizon-tip {{ font-size: 13px; color: #6b21a8; background: rgba(255,255,255,0.6); padding: 8px 12px; border-radius: 4px; }}
    .fun-ending {{ background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 1px solid #fbbf24; border-radius: 12px; padding: 24px; margin-top: 40px; text-align: center; }}
    .fun-headline {{ font-size: 18px; font-weight: 600; color: #78350f; margin-bottom: 8px; }}
    .fun-detail {{ font-size: 14px; color: #92400e; }}
    .collapsible-section {{ margin-top: 16px; }}
    .collapsible-header {{ display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 12px 0; border-bottom: 1px solid #e2e8f0; }}
    .collapsible-header h3 {{ margin: 0; font-size: 14px; font-weight: 600; color: #475569; }}
    .collapsible-arrow {{ font-size: 12px; color: #94a3b8; transition: transform 0.2s; }}
    .collapsible-content {{ display: none; padding-top: 16px; }}
    .collapsible-content.open {{ display: block; }}
    .collapsible-header.open .collapsible-arrow {{ transform: rotate(90deg); }}
    @media (max-width: 640px) {{ .charts-row {{ grid-template-columns: 1fr; }} .stats-row {{ justify-content: center; }} }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Claude Code Insights</h1>
    <p class="subtitle">{stats.total_messages:,} messages across {stats.total_sessions} sessions | {stats.date_range_start} to {stats.date_range_end}</p>

    {at_a_glance_html}

    <nav class="nav-toc">
      <a href="#section-work">What You Work On</a>
      <a href="#section-usage">How You Use CC</a>
      <a href="#section-wins">Impressive Things</a>
      <a href="#section-friction">Where Things Go Wrong</a>
      <a href="#section-features">Features to Try</a>
      <a href="#section-patterns">New Usage Patterns</a>
      <a href="#section-horizon">On the Horizon</a>
      <a href="#section-stats">Stats Dashboard</a>
    </nav>

    <div class="stats-row">
      {stats_bar}
    </div>

    <h2 id="section-work">What You Work On</h2>
    {project_areas_html}

    <div class="charts-row">
      <div class="chart-card">
        <div class="chart-title">What You Wanted</div>
        {goal_chart}
      </div>
      <div class="chart-card">
        <div class="chart-title">Top Tools Used</div>
        {tool_chart}
      </div>
    </div>

    <div class="charts-row">
      <div class="chart-card">
        <div class="chart-title">Languages</div>
        {lang_chart}
      </div>
      <div class="chart-card">
        <div class="chart-title">Session Types</div>
        {session_type_chart}
      </div>
    </div>

    <h2 id="section-usage">How You Use Claude Code</h2>
    {narrative_html}

    <!-- Response Time Distribution -->
    <div class="chart-card" style="margin: 24px 0;">
      <div class="chart-title">User Response Time Distribution</div>
      {response_time_chart}
      <div style="font-size: 12px; color: #64748b; margin-top: 8px;">
        Median: {stats.median_response_time}s &bull; Average: {stats.avg_response_time}s
      </div>
    </div>

    {multi_clauding_html}

    <!-- Time of Day & Tool Errors -->
    <div class="charts-row">
      <div class="chart-card">
        <div class="chart-title" style="display: flex; align-items: center; gap: 12px;">
          User Messages by Time of Day
          <select id="timezone-select" style="font-size: 12px; padding: 4px 8px; border-radius: 4px; border: 1px solid #e2e8f0;">
            <option value="0">PT (UTC-8)</option>
            <option value="3">ET (UTC-5)</option>
            <option value="8">London (UTC)</option>
            <option value="9">CET (UTC+1)</option>
            <option value="17">Tokyo (UTC+9)</option>
            <option value="custom">Custom offset...</option>
          </select>
          <input type="number" id="custom-offset" placeholder="UTC offset" style="display: none; width: 80px; font-size: 12px; padding: 4px; border-radius: 4px; border: 1px solid #e2e8f0;">
        </div>
        <div id="hour-histogram">
      {time_of_day_chart}</div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Tool Errors Encountered</div>
        {error_chart}
      </div>
    </div>

    <h2 id="section-wins">Impressive Things You Did</h2>
    {big_wins_html}

    <div class="charts-row">
      <div class="chart-card">
        <div class="chart-title">What Helped Most (Claude's Capabilities)</div>
        {success_chart}
      </div>
      <div class="chart-card">
        <div class="chart-title">Outcomes</div>
        {outcome_chart}
      </div>
    </div>

    <h2 id="section-friction">Where Things Go Wrong</h2>
    {friction_html}

    <div class="charts-row">
      <div class="chart-card">
        <div class="chart-title">Primary Friction Types</div>
        {friction_chart}
      </div>
      <div class="chart-card">
        <div class="chart-title">Inferred Satisfaction (model-estimated)</div>
        {satisfaction_chart}
      </div>
    </div>

    <div class="charts-row">
      <div class="chart-card">
        <div class="chart-title">Claude Helpfulness</div>
        {helpfulness_chart}
      </div>
    </div>

    <h2 id="section-features">Existing CC Features to Try</h2>
    {claude_md_html}
    {features_html}

    <h2 id="section-patterns">New Ways to Use Claude Code</h2>
    {patterns_html}

    <h2 id="section-horizon">On the Horizon</h2>
    {horizon_html}

    {fun_html}

  </div>
  <script>
    function toggleCollapsible(header) {{
      header.classList.toggle('open');
      const content = header.nextElementSibling;
      content.classList.toggle('open');
    }}
    function copyText(btn) {{
      const code = btn.previousElementSibling;
      navigator.clipboard.writeText(code.textContent).then(() => {{
        btn.textContent = 'Copied!';
        setTimeout(() => {{ btn.textContent = 'Copy'; }}, 2000);
      }});
    }}
    function copyCmdItem(idx) {{
      const checkbox = document.getElementById('cmd-' + idx);
      if (checkbox) {{
        const text = checkbox.dataset.text;
        navigator.clipboard.writeText(text).then(() => {{
          const btn = checkbox.nextElementSibling.querySelector('.copy-btn');
          if (btn) {{ btn.textContent = 'Copied!'; setTimeout(() => {{ btn.textContent = 'Copy'; }}, 2000); }}
        }});
      }}
    }}
    function copyAllCheckedClaudeMd() {{
      const checkboxes = document.querySelectorAll('.cmd-checkbox:checked');
      const texts = [];
      checkboxes.forEach(cb => {{
        if (cb.dataset.text) {{ texts.push(cb.dataset.text); }}
      }});
      const combined = texts.join('\\n');
      const btn = document.querySelector('.copy-all-btn');
      if (btn) {{
        navigator.clipboard.writeText(combined).then(() => {{
          btn.textContent = 'Copied ' + texts.length + ' items!';
          btn.classList.add('copied');
          setTimeout(() => {{ btn.textContent = 'Copy All Checked'; btn.classList.remove('copied'); }}, 2000);
        }});
      }}
    }}
    // Timezone selector for time of day chart
    const rawHourCounts = {raw_hours_json};
    function updateHourHistogram(offsetFromPT) {{
      const periods = [
        {{ label: "Morning (6-12)", range: [6,7,8,9,10,11] }},
        {{ label: "Afternoon (12-18)", range: [12,13,14,15,16,17] }},
        {{ label: "Evening (18-24)", range: [18,19,20,21,22,23] }},
        {{ label: "Night (0-6)", range: [0,1,2,3,4,5] }}
      ];
      const adjustedCounts = {{}};
      for (const [hour, count] of Object.entries(rawHourCounts)) {{
        const newHour = (parseInt(hour) + offsetFromPT + 24) % 24;
        adjustedCounts[newHour] = (adjustedCounts[newHour] || 0) + count;
      }}
      const periodCounts = periods.map(p => ({{
        label: p.label,
        count: p.range.reduce((sum, h) => sum + (adjustedCounts[h] || 0), 0)
      }}));
      const maxCount = Math.max(...periodCounts.map(p => p.count)) || 1;
      const container = document.getElementById('hour-histogram');
      container.textContent = '';
      periodCounts.forEach(p => {{
        const row = document.createElement('div');
        row.className = 'bar-row';
        const label = document.createElement('div');
        label.className = 'bar-label';
        label.textContent = p.label;
        const track = document.createElement('div');
        track.className = 'bar-track';
        const fill = document.createElement('div');
        fill.className = 'bar-fill';
        fill.style.width = (p.count / maxCount) * 100 + '%';
        fill.style.background = '#8b5cf6';
        track.appendChild(fill);
        const value = document.createElement('div');
        value.className = 'bar-value';
        value.textContent = p.count;
        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(value);
        container.appendChild(row);
      }});
    }}
    document.getElementById('timezone-select').addEventListener('change', function() {{
      const customInput = document.getElementById('custom-offset');
      if (this.value === 'custom') {{
        customInput.style.display = 'inline-block';
        customInput.focus();
      }} else {{
        customInput.style.display = 'none';
        updateHourHistogram(parseInt(this.value));
      }}
    }});
    document.getElementById('custom-offset').addEventListener('change', function() {{
      const offset = parseInt(this.value) + 8;
      updateHourHistogram(offset);
    }});
  </script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help=f"Output path (default: {DEFAULT_OUTPUT})",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option(
    "--model",
    default="gpt-oss:20b-cloud",
    show_default=True,
    help="Ollama model for facet extraction and insight generation",
)
@click.option(
    "--max-extract",
    default=50,
    type=int,
    help="Max facets to extract via LLM per run (0=skip)",
)
def main(output: Path | None, verbose: bool, model: str, max_extract: int):
    """Generate a Claude Code usage insights HTML report."""
    output_path = output or DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stage 1: Load & Filter
    click.echo("Stage 1: Loading session logs...")
    sessions = load_and_filter_sessions(verbose)
    click.echo(f"  Found {len(sessions)} valid sessions")

    if not sessions:
        raise click.ClickException(
            f"No valid sessions found. Check that {PROJECTS_DIR} contains .jsonl files."
        )

    # Stage 2: Extract Metadata
    click.echo("Stage 2: Extracting metadata...")
    metadata_map: dict[str, SessionMetadata] = {}
    for sid, (messages, created, modified) in sessions.items():
        metadata_map[sid] = extract_session_metadata(
            sid,
            messages,
            created,
            modified,
            verbose,
        )
    click.echo(f"  Processed {len(metadata_map)} sessions")

    # Stage 3: Extract Facets
    click.echo("Stage 3: Loading facets...")
    facets = extract_facets(
        metadata_map,
        sessions,
        model=model,
        max_extract=max_extract,
        verbose=verbose,
    )
    click.echo(f"  {len(facets)} sessions with facets")

    # Stage 4: Aggregate Statistics
    click.echo("Stage 4: Aggregating statistics...")
    stats = aggregate_statistics(metadata_map, facets, verbose)
    if verbose:
        click.echo(f"  Total messages: {stats.total_messages}")
        click.echo(f"  Total duration: {stats.total_duration_hours:.1f}h")
        click.echo(f"  Git commits: {stats.git_commits}")
        click.echo(f"  Days active: {stats.days_active}")
        click.echo(f"  Multi-clauding events: {stats.multi_clauding['overlap_events']}")

    # Stage 5: Generate Insights
    click.echo(f"Stage 5: Generating insights via Ollama ({model})...")
    insights = generate_insights(stats, model=model, verbose=verbose)

    # Stage 6: Build HTML Report
    click.echo("Stage 6: Building HTML report...")
    html = build_html_report(stats, insights, verbose)
    output_path.write_text(html, encoding="utf-8")
    click.echo(f"\nReport written to: file://{output_path}")
    click.echo("Open in browser to view.")


if __name__ == "__main__":
    main()
