#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Rewrite Claude Code's assistant messages through codex exec.

Installs a MessageDisplay hook. The hook buffers each streamed batch to disk,
then sends the finished message to `codex exec` and shows the rewrite instead.
Claude Code keeps the original text, so ctrl+o still reads it.

CREDIT:
    Zach Ahn's vomit had the idea first: hook Claude Code's MessageDisplay event,
    buffer the streamed batches, and hand the finished message to a model that
    edits the prose.

        https://github.com/zachahn/vomit

    That project is Go and talks to a local OpenAI-compatible server, so it stays
    on the machine and costs nothing per message. Read it if you want the local
    version, which is the better default for most people. It is GNU GPLv3.

    This file is a separate implementation against `codex exec`, written in
    Python, with its own buffering code, prompts, language gate, and codex
    trimming. It shares no code with vomit. It does share the idea, and the idea
    is Zach's.

USAGE:
    # Run directly from GitHub (no clone needed). Works in bash and zsh:
    URL=https://raw.githubusercontent.com/CJHwong/toolkit/main/python/ccrewrite.py

    uv run "$URL" check                     # one real round trip, with timings
    uv run "$URL" rewrite "你的文字"          # rewrite one string, print it, exit
    pbpaste | uv run "$URL" rewrite --time   # or read stdin, so newlines survive
    uv run "$URL" rewrite --timeout 600 < long.md   # past the 100s default
    uv run "$URL" install --lang zh         # rewrite Chinese, leave the rest alone
    uv run "$URL" install --lang all        # rewrite every message
    uv run "$URL" install --lang zh --scope project
    uv run "$URL" install --lang zh --model gpt-5.6-luna --effort low
    uv run "$URL" uninstall
    uv run "$URL" hook                      # what the hook itself runs

WHAT IT COSTS:
    Every assistant message becomes one codex request, so the text leaves the
    machine. codex is not a local model.

    A hook cannot rewrite a partial message, so the live stream stays blank
    until the message ends. Measured on gpt-5.6-luna on an M-series Mac: about 16
    seconds for a 179-character Chinese paragraph. Effort barely moves this. The
    same text took 16.3s at high, 15.9s at medium, 13.3s at low, because the
    input tokens and the round trip dominate, not the reasoning. `--model` and
    `--effort` take any value codex accepts. Run `check` for your own numbers.

    Each message also costs about 8300 input tokens, down from 20138 stock. That
    floor is codex's shell and apply_patch tool schemas, which no config option
    removes. See isolated_env, TRIM_CONFIG, and disable_flags for the rest.

    Any failure prints the original text under a `[edit] kept as written` line, so a
    broken hook never eats a message. Under that line comes what codex printed,
    in a code block, minus the prompt codex echoes back. Both ends of that output
    matter: codex opens with a banner and ends with the error, a version-manager
    shim does the opposite, so it is never cut to fit.

    A rewrite costs about 15 seconds plus 9ms per character, measured on
    gpt-5.6-luna over 1000, 3000 and 6000-character Chinese inputs. The model
    emits roughly as many characters as it reads, so output length sets the
    cost, not reasoning. CODEX_TIMEOUT of 100s therefore gives out near 9300
    characters. The hook keeps that default on purpose, because nobody waits
    minutes to read a message. It skips anything over skip_over(), 7065
    characters at the default budget, rather than spending the whole 100s to
    fail. `rewrite --timeout` raises the budget for a document, where waiting
    is the point.

INSTALL DETAIL:
    `install` copies this file to ~/.claude/hooks/ccrewrite.py and points the
    hook at that copy. No message waits on a network fetch of the script.
"""

import argparse
import functools
import json
import re
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EVENT = "MessageDisplay"
MARK = "[edit]"
HOOK_NAME = "ccrewrite.py"
SPOOL_PREFIX = "ccrewrite"

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_EFFORT = "medium"
# The reasoning-effort enum the API accepts. A bad value fails the request with
# a 400, so reject it at argument parsing instead of on the next message.
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
CODEX_TIMEOUT = 100.0
HOOK_TIMEOUT = 120

# A straggler batch lands in milliseconds when it lands at all. Two seconds
# is already far past that, and waiting longer only delays a lost message.
STRAGGLER_WAIT = 2.0
# Only a session that died mid-message leaves a spool behind. Half an hour
# outlives any real message and keeps the temp directory small.
STALE_AFTER = 1800.0
# Below this, a round trip costs more than the edit is worth. 240 characters
# is roughly three sentences of English.
SKIP_UNDER = 240
# A Han character carries far more than a Latin one, so 200 characters of
# Chinese is a long paragraph, not a one-line acknowledgement.
SKIP_UNDER_HAN = 72
# Above this, the rewrite cannot finish inside CODEX_TIMEOUT, so attempting it
# spends the whole budget and shows the message late and unedited. A rewrite
# costs a fixed setup plus a per-character rate, because the model emits about
# as many characters as it reads. Measured on gpt-5.6-luna: 1000 characters in
# 24.0s, 3000 in 46.5s, 6000 in 69.8s. Keep a margin, so a slow request still
# lands rather than dying at the ceiling.
SETUP_SECONDS = 15.0
SECONDS_PER_CHAR = 0.0092
TIMEOUT_MARGIN = 0.8


def skip_over(budget: float = CODEX_TIMEOUT) -> int:
    """The longest message a rewrite can finish inside the budget."""
    return int((budget * TIMEOUT_MARGIN - SETUP_SECONDS) / SECONDS_PER_CHAR)

# Shown only on a message that was actually rewritten. A pass-through already
# prints the original text, so pointing at ctrl+o there would be noise.
ORIGINAL_HINT = "ctrl+o for the original"
REASON_LIMIT = 160

# A failure is rare and worth reading, so it shows what the tool printed rather
# than a fragment of it. Deep enough for a stack trace, short enough to scroll.
DETAIL_LINES = 16

# Rewrite Chinese only. English messages then cost nothing and appear the moment
# the stream ends, which keeps the blank window off the messages that do not
# need it. Han script is the practical test: distinguishing Traditional from
# Simplified needs a conversion table, and this agent writes Taiwanese Chinese.
# Which messages earn a rewrite. `install --lang` is required, because the two
# modes cost very differently: English is 0.1s and free under "zh", 8-13s and
# ~8300 tokens under "all". install writes the choice into the hook command, so
# DEFAULT_LANG only backstops a hook run by hand.
LANGS = ("zh", "all")
DEFAULT_LANG = "zh"

HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
KANA = re.compile(r"[\u3040-\u30ff]")
HANGUL = re.compile(r"[\uac00-\ud7af]")
LATIN = re.compile(r"[A-Za-z]")

# A message is Chinese when Han carries most of it, not when Han appears in it.
# One quoted character must not reroute an English message: "Delete 們 after a
# plural noun" is English about Chinese, and rewriting it as Chinese destroys it.
HAN_SHARE = 0.25


def is_chinese(text: str) -> bool:
    han = len(HAN.findall(text))
    if han == 0:
        return False
    # Japanese and Korean borrow these same blocks. Kana or hangul settles it.
    if KANA.search(text) or HANGUL.search(text):
        return False
    return han / (han + len(LATIN.findall(text))) >= HAN_SHARE

PROMPT_EN = """You are an editor. The <stdin> block holds one message written by a coding agent.

Rewrite it in clear, conversational, first-person prose. Remove these traits:

- Objects doing action verbs. Never "the buffer carries", "the config names".
  Only people and agents act. An API may do stereotypical things: create, read,
  queue, run, call.
- Pseudo-epiphanies and roundabout reasoning.
- Self-praise.
- Em dashes, which add a distracting beat.
- Weird subject and verb pairs, and subjects that should be objects.

Keep the intent and every detail. Keep code, paths, commands, and error strings
verbatim.

This is a text task, not a coding task. Do not run commands, read files, or call
any tool. Reply with the edited prose and nothing else."""

# The Chinese rules follow the primary source:
#   余光中〈怎樣改進英式中文？──論中文的常態與變態〉
#   《明報月刊》1987 年 10 月號
# The essay is not published free online by its rights holder. Every copy in
# circulation is an unofficial reproduction, so the citation names the journal
# rather than linking to one.
PROMPT_ZH = """You are an editor. The <stdin> block holds one message written by a coding agent
in Chinese. It reads like translated English. Rewrite it as natural Taiwanese
Mandarin (台灣正體), applying Yu Kwang-chung's rules against 英式中文:

- Cut empty verbs. 進行研究 -> 研究. 作出貢獻 -> 貢獻很大.
- Cut 被. Write the active sentence. 被你這句話嚇倒 -> 你這句話嚇不倒我.
- Cut decorative suffixes. 知名度很高 -> 很有名. 可讀性頗高 -> 好看. 難度很高 -> 很難.
- At most one 的 per phrase. Use 而, or restructure.
- Delete 們 after a plural noun, 一個 before a person, 之一, 有關, 關於, and
  由於...所以. Let parataxis carry the logic.
- Put a modifier after the noun, never in a long chain before 的.
- Use a comma instead of 及. Use a verb instead of an abstract noun:
  造成九十八人死亡 -> 死了九十八人.
- Remove self-praise and pseudo-epiphanies such as 從中浮現的是.

Keep every sentence under 25 characters, one idea each. Write flowing
paragraphs. Never put one sentence per line, and never end a line with two
spaces. Keep the intent and every detail. Keep code, paths, commands, error
strings, and English technical terms verbatim.

This is a text task, not a coding task. Do not run commands, read files, or call
any tool. Reply with the edited prose and nothing else."""


# --- hook protocol -----------------------------------------------------------


def respond(text: str) -> None:
    payload = {"hookSpecificOutput": {"hookEventName": EVENT, "displayContent": text}}
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")


def respond_untouched() -> None:
    json.dump({}, sys.stdout)
    sys.stdout.write("\n")



def status(how: str) -> str:
    return f"*{MARK} {how}*\n\n"


def keep_original(raw: str, why: str, output: str = "") -> str:
    return status(f"kept as written, {why}") + fenced(output) + raw.rstrip()


def fenced(output: str) -> str:
    """A code block holding what the tool printed, or nothing when it said nothing."""
    return f"```\n{output}\n```\n\n" if output else ""


def parse_batch(stream) -> dict:
    """The batch payload. Raise when the json is unreadable or belongs elsewhere."""
    payload = json.load(stream)
    event = payload.get("hook_event_name", "")
    if event and event != EVENT:
        raise ValueError(f"hook event is {event}, not {EVENT}")
    return payload


# --- delta buffer ------------------------------------------------------------
#
# Claude Code runs a separate process per streamed batch, so the parts of one
# message can only meet on disk.


def spool_root() -> Path:
    return Path(tempfile.gettempdir()) / f"{SPOOL_PREFIX}-{os.getuid()}"


def slug(message_id: str) -> str:
    kept = [c for c in message_id if c.isascii() and (c.isalnum() or c in "-_")]
    name = "".join(kept)[:128]
    if not name:
        raise ValueError(f"unusable message id: {message_id!r}")
    return name


def batch_dir(message_id: str) -> Path:
    return spool_root() / slug(message_id)


def store_batch(message_id: str, index: int, delta: str) -> None:
    if not message_id:
        raise ValueError("hook json has no message_id")
    target = batch_dir(message_id)
    target.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=target, prefix=".part-")
    with os.fdopen(handle, "w", encoding="utf-8") as part:
        part.write(delta)
    os.replace(temp, target / f"{index:06d}.txt")


def join_batches(target: Path, upto: int) -> tuple[str, int]:
    names = sorted(p for p in target.iterdir() if p.suffix == ".txt")
    body = "".join(p.read_text(encoding="utf-8") for p in names)
    return body, upto + 1 - len(names)


def assemble(message_id: str, upto: int, wait: float) -> tuple[str, bool]:
    """Join every part. Poll until the last straggler lands or the wait runs out."""
    target = batch_dir(message_id)
    deadline = time.monotonic() + wait
    while True:
        body, missing = join_batches(target, upto)
        if missing <= 0 or time.monotonic() >= deadline:
            return body, missing <= 0
        time.sleep(0.01)


def forget(message_id: str) -> None:
    shutil.rmtree(batch_dir(message_id), ignore_errors=True)


def leftovers() -> list[Path]:
    """Every directory a run can leave behind: batch spools, and the scratch dir
    of a codex that outlived the kill and beat TemporaryDirectory to its files.
    A stale scratch holds a symlink to the real auth.json, so it must not sit
    in the temp directory until the operating system decides to sweep it."""
    root = spool_root()
    spools = [entry for entry in root.iterdir() if entry.is_dir()] if root.is_dir() else []
    scratch = Path(tempfile.gettempdir()).glob(f"{SPOOL_PREFIX}-run-*")
    return spools + [entry for entry in scratch if entry.is_dir()]


def purge_stale(older_than: float) -> None:
    cutoff = time.time() - older_than
    for entry in leftovers():
        try:
            expired = entry.stat().st_mtime < cutoff
        except OSError:
            continue  # another run swept it first
        if expired:
            shutil.rmtree(entry, ignore_errors=True)


# --- codex -------------------------------------------------------------------


def real_codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME")
    if configured:
        return Path(configured)
    return Path.home() / ".codex"


# A version manager (mise, asdf, pyenv, rbenv) puts a shim on PATH in place of
# the real binary. A shim reads its own config and trust state from $HOME, and
# isolated_env replaces $HOME, so launching through one dies with a trust error
# before codex ever starts. Resolve past the shim to the binary behind it.
SHIM_DIR = f"{os.sep}shims{os.sep}"


@functools.cache
def codex_binary() -> str | None:
    """The real codex executable, or None. Never a version manager's shim."""
    found = shutil.which("codex")
    if found is None or SHIM_DIR not in found:
        return found
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory or SHIM_DIR in f"{directory}{os.sep}":
            continue
        candidate = Path(directory) / "codex"
        if os.access(candidate, os.X_OK):
            return str(candidate)
    return found


def isolated_env(scratch: Path) -> dict:
    """Keep codex's own context out of the request.

    A stock `codex exec` loads the user's config, plugins, and whatever skills it
    discovers, which cost 20138 input tokens on a "say ok" probe. Pointing
    CODEX_HOME and HOME at empty directories cuts that to 11411. Credentials
    still resolve, because auth.json is symlinked into the scratch CODEX_HOME.
    """
    codex_home = scratch / "codex-home"
    codex_home.mkdir()
    (scratch / "home").mkdir()

    auth = real_codex_home() / "auth.json"
    if auth.exists():
        (codex_home / "auth.json").symlink_to(auth)

    env = os.environ.copy()
    env["CODEX_HOME"] = str(codex_home)
    env["HOME"] = str(scratch / "home")
    return env


# Prose does not need an agent. Every one of these strips something codex adds
# to the request that a rewrite cannot use. Measured with a "say ok" probe:
# 20138 input tokens stock, 11411 with the scratch environment, 8322 with all
# of this. What is left is codex's shell and apply_patch schemas, which no
# config option removes, because that is what codex is.
TRIM_CONFIG = [
    "include_permissions_instructions=false",
    "include_apps_instructions=false",
    "include_collaboration_mode_instructions=false",
    "include_environment_context=false",
    "tools.web_search=false",
]

# Disabling this one makes codex respond a fail-closed error item on every turn.
KEEP_FEATURES = {"code_mode_host"}


def disable_flags(env: dict) -> list[str]:
    """Ask codex which features it has, then turn off every one it will allow.

    `codex features list` costs about 20ms, so this runs per message rather than
    baking a list into the source. An unknown feature name is a hard error
    ("Error: Unknown feature flag: ..."), so a hardcoded list would break the
    hook on the next codex upgrade that renames or retires a flag.

    This asks under the same isolated environment the rewrite runs in. A feature
    reads as enabled in one environment and disabled in another, so asking with
    the caller's own environment silently misses some.
    """
    try:
        listed = subprocess.run(
            [codex_binary(), "features", "list"], env=env, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if listed.returncode != 0:
        return []

    flags = []
    for line in listed.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[-1] == "true" and fields[0] not in KEEP_FEATURES:
            flags += ["--disable", fields[0]]
    return flags


def codex_argv(
    workspace: Path, answer: Path, env: dict, prompt: str, model: str, effort: str
) -> list[str]:
    argv = [
        codex_binary(),
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-rules",
        "-C",
        str(workspace),
        "-s",
        "read-only",
        "-m",
        model,
        "-c",
        f"model_reasoning_effort={effort}",
    ]
    for setting in TRIM_CONFIG:
        argv += ["-c", setting]
    return argv + disable_flags(env) + ["-o", str(answer), prompt]


class CodexFailed(RuntimeError):
    """A rewrite that did not happen, plus whatever codex printed before it stopped."""

    def __init__(self, why: str, output: str = "", echo: str = "") -> None:
        super().__init__(why)
        self.output = detail(output, echo)


def rewrite(
    raw: str,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    budget: float = CODEX_TIMEOUT,
) -> str:
    """Send one message to codex. Raise RuntimeError with a printable reason."""
    if codex_binary() is None:
        raise CodexFailed("codex is not on PATH")

    prompt = PROMPT_ZH if is_chinese(raw) else PROMPT_EN
    sent = f"{prompt}\n{raw}"  # codex echoes both onto stderr, so detail can strip them

    with tempfile.TemporaryDirectory(prefix=f"{SPOOL_PREFIX}-run-") as scratch:
        workspace = Path(scratch) / "workspace"
        workspace.mkdir()
        answer = Path(scratch) / "answer.txt"
        env = isolated_env(Path(scratch))
        try:
            done = subprocess.run(
                codex_argv(workspace, answer, env, prompt, model, effort),
                env=env,
                input=raw,
                capture_output=True,
                text=True,
                timeout=budget,
            )
        except subprocess.TimeoutExpired as err:
            raise CodexFailed(
                f"codex took longer than {budget:.0f}s", decoded(err.stderr), sent
            ) from None

        if done.returncode != 0:
            raise CodexFailed(f"codex exited {done.returncode}", done.stderr, sent)
        if not answer.exists():
            raise CodexFailed("codex wrote no answer", done.stderr, sent)

        said = answer.read_text(encoding="utf-8").strip()
        if not said:
            raise CodexFailed("codex said nothing", done.stderr, sent)
        return said


def oneline(text: str) -> str:
    """A python exception squeezed onto the status line."""
    line = " ".join(text.split())
    return line[:REASON_LIMIT] if line else "no reason given"


def decoded(stream) -> str:
    """subprocess.run decodes stdout and stderr under text=True, but a
    TimeoutExpired hands back the raw buffer instead. Decode it here."""
    if stream is None:
        return ""
    return stream if isinstance(stream, str) else stream.decode("utf-8", "replace")


def detail(text: str, echo: str = "") -> str:
    """What the tool printed, kept whole. Never cut a line in half: which end
    holds the diagnosis depends on the tool. codex opens with a banner and ends
    with the error, mise opens with the error and ends with its version. Cutting
    to a character count turned both into noise.

    codex also echoes the prompt and the message back onto stderr, which buries
    the error under our own input. `echo` carries what this script sent, so those
    lines come back out. What is left is what codex had to say for itself."""
    sent = {line.strip() for line in echo.splitlines() if line.strip()}
    kept = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and line.strip() not in sent
    ]
    return "\n".join(kept[-DETAIL_LINES:])


# --- hook command ------------------------------------------------------------


def run_hook(args: argparse.Namespace) -> int:
    """Never raise. A hook that crashes hides the message it was given."""
    try:
        payload = parse_batch(sys.stdin)
    except Exception:
        # Unreadable json carries no delta to echo back, so silence is the only
        # answer that does not wipe the text off the screen.
        respond_untouched()
        return 0

    delta = payload.get("delta", "")
    try:
        store_batch(payload.get("message_id", ""), int(payload.get("index", 0)), delta)
    except Exception as err:
        # The delta survived the json, so say what broke and show it anyway.
        respond(status(f"kept as written, {oneline(str(err))}") + delta)
        return 0

    if not payload.get("final"):
        respond(opening_notice(payload, args.lang))
        return 0

    try:
        respond(final_display(payload, args))
    except Exception as err:
        respond(status(f"kept as written, {oneline(str(err))}"))
    finally:
        forget(payload["message_id"])
        purge_stale(STALE_AFTER)
    return 0


def opening_notice(payload: dict, lang: str) -> str:
    """Warn about the wait, but only on a message that will actually pay for it.

    Each batch runs in its own process, so this reads the first batch only. A
    message that opens in Chinese gets the status; anything else stays silent,
    because a stale "waiting" line above an untouched English message is noise.
    """
    if payload.get("index") != 0:
        return ""
    if lang == "zh" and not is_chinese(payload.get("delta", "")):
        return ""
    return f"*{MARK} waiting for the whole message*"


def final_display(payload: dict, args: argparse.Namespace) -> str:
    raw, whole = assemble(payload["message_id"], int(payload.get("index", 0)), STRAGGLER_WAIT)
    if not raw.strip():
        return status("nothing to rewrite")
    if not whole:
        return keep_original(raw, "part of the message never arrived")
    # Language first. A message this hook will never rewrite must show no status
    # line at all, whatever its length. Only a candidate earns an explanation.
    if args.lang == "zh" and not is_chinese(raw):
        return raw.rstrip()
    limit = SKIP_UNDER_HAN if is_chinese(raw) else SKIP_UNDER
    if len(raw.strip()) <= limit:
        return keep_original(raw, "left a short message alone")
    ceiling = skip_over()
    if len(raw.strip()) > ceiling:
        return keep_original(raw, f"too long to rewrite, {len(raw.strip())} over {ceiling}")

    began = time.monotonic()
    try:
        said = rewrite(raw, args.model, args.effort)
    except CodexFailed as err:
        return keep_original(raw, str(err), err.output)
    return status(f"rewrote in {time.monotonic() - began:.1f}s, {ORIGINAL_HINT}") + said


# --- install -----------------------------------------------------------------


def settings_path(scope: str) -> Path:
    if scope == "project":
        return Path.cwd() / ".claude" / "settings.json"
    return claude_dir() / "settings.json"


def claude_dir() -> Path:
    configured = os.environ.get("CLAUDE_CONFIG_DIR")
    if configured:
        return Path(configured)
    return Path.home() / ".claude"


def installed_script() -> Path:
    return claude_dir() / "hooks" / HOOK_NAME


def all_settings_paths() -> list[Path]:
    """Every settings file Claude Code reads hooks from, in no particular order."""
    return [
        claude_dir() / "settings.json",
        claude_dir() / "settings.local.json",
        Path.cwd() / ".claude" / "settings.json",
        Path.cwd() / ".claude" / "settings.local.json",
    ]


def rival_installs(chosen: Path) -> list[Path]:
    """Find this hook in settings files other than the one being written.

    Claude Code merges hooks across these files rather than letting the nearest
    one win, so a copy left in a second file fires on every message alongside
    this one. Stripping entries from the chosen file cannot see those.
    """
    found = []
    for path in all_settings_paths():
        if path == chosen or not path.exists():
            continue
        try:
            entries = load_settings(path).get("hooks", {}).get(EVENT, [])
        except (OSError, json.JSONDecodeError):
            continue
        if any(is_ours(entry) for entry in entries):
            found.append(path)
    return found


def load_settings(path: Path) -> dict:
    if not path.exists():
        return {}
    body = path.read_text(encoding="utf-8").strip()
    if not body:
        return {}
    return json.loads(body)


def save_settings(path: Path, settings: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def is_ours(entry: dict) -> bool:
    hooks = entry.get("hooks", []) if isinstance(entry, dict) else []
    return any(HOOK_NAME in str(h.get("command", "")) for h in hooks)


def strip_ours(settings: dict) -> list:
    matchers = settings.get("hooks", {}).get(EVENT, [])
    return [entry for entry in matchers if not is_ours(entry)]


def copy_self(target: Path) -> None:
    source = Path(__file__).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.resolve() == source:
        return
    shutil.copyfile(source, target)
    target.chmod(0o755)


def cmd_install(args: argparse.Namespace) -> int:
    script = installed_script()
    copy_self(script)

    path = settings_path(args.scope)
    settings = load_settings(path)
    matchers = strip_ours(settings)
    matchers.append(
        {
            "hooks": [
                {
                    "type": "command",
                    "command": (
                        f"uv run {script} hook --lang {args.lang}"
                        f" --model {args.model} --effort {args.effort}"
                    ),
                    "timeout": HOOK_TIMEOUT,
                }
            ]
        }
    )
    settings.setdefault("hooks", {})[EVENT] = matchers
    save_settings(path, settings)

    print(f"copied  {script}")
    print(f"hooked  {EVENT} in {path}")
    print(f"model   {args.model} at {args.effort} effort")
    print(f"scope   {'Chinese messages only' if args.lang == 'zh' else 'every message'}")
    rivals = rival_installs(path)
    if rivals:
        print("\nWARNING: this hook is also installed in:")
        for rival in rivals:
            print(f"  {rival}")
        scope = "project" if rivals[0].parent.name == ".claude" and rivals[0].parent.parent == Path.cwd() else "home"
        print("Claude Code merges hooks across these files, so each copy rewrites")
        print(f"every message again. Remove the other one with:")
        print(f"  uv run {script} uninstall --scope {scope}")

    print("\nActive on the next message. ctrl+o shows what Claude actually wrote.")
    return 0


def cmd_uninstall(args: argparse.Namespace) -> int:
    path = settings_path(args.scope)
    settings = load_settings(path)
    matchers = strip_ours(settings)

    if matchers:
        settings["hooks"][EVENT] = matchers
    else:
        settings.get("hooks", {}).pop(EVENT, None)
    if settings.get("hooks") == {}:
        settings.pop("hooks")
    save_settings(path, settings)

    print(f"removed {EVENT} hook from {path}")
    print(f"left    {installed_script()} in place, remove it by hand if you want it gone")
    return 0


# --- check -------------------------------------------------------------------

SAMPLE = (
    "The refactor lands cleanly. The buffer now carries each delta to disk, and "
    "the store names the parts in index order, which means reassembly no longer "
    "depends on process lifetime, a subtle but important shift. What emerges is a "
    "design where the filesystem itself becomes the coordination primitive, and I "
    "think that's the right call here. The purge_stale quietly reaps anything older than "
    "an hour, so the temp directory never grows without bound. I'm quite happy with "
    "how clean this turned out."
)


SAMPLE_ZH = (
    "關於這次重構的部分，我們進行了詳細的研究。由於原本的緩衝區的設計具有很高的"
    "複雜度，所以每個 delta 都被寫入到磁碟裡面。開發者們作出了重大的貢獻，這是我們"
    "最重要的改進之一。從中浮現的是一個以檔案系統作為協調原語的設計。"
)


def cmd_check(args: argparse.Namespace) -> int:
    if codex_binary() is None:
        print("codex is not on PATH. Install it first.", file=sys.stderr)
        return 1

    print(f"codex   {codex_binary()}")
    print(f"model   {args.model} at {args.effort} effort")

    failed = 0
    for label, sample in (("English", SAMPLE), ("Chinese", SAMPLE_ZH)):
        routed = "Chinese" if is_chinese(sample) else "English"
        print(f"\n--- {label} sample, {len(sample)} characters, routed to {routed} prompt ---")
        began = time.monotonic()
        try:
            said = rewrite(sample, args.model, args.effort)
        except RuntimeError as err:
            print(f"FAILED after {time.monotonic() - began:.1f}s: {err}", file=sys.stderr)
            failed += 1
            continue
        print(said)
        print(f"[{time.monotonic() - began:.1f}s]")

    if failed:
        return 1
    print(f"\nUnder --lang zh only the Chinese path runs, on messages longer than")
    print(f"{SKIP_UNDER_HAN} characters. Under --lang all both run.")
    return 0


def cmd_rewrite(args: argparse.Namespace) -> int:
    """Rewrite text given on the command line or on stdin, and print the result."""
    raw = " ".join(args.text) if args.text else sys.stdin.read()
    raw = raw.strip()
    if not raw:
        print("nothing to rewrite", file=sys.stderr)
        return 1

    began = time.monotonic()
    try:
        said = rewrite(raw, args.model, args.effort, args.timeout)
    except CodexFailed as err:
        print(err, file=sys.stderr)
        if err.output:
            print(err.output, file=sys.stderr)
        return 1
    print(said)
    if args.time:
        print(f"[{time.monotonic() - began:.1f}s]", file=sys.stderr)
    return 0


# --- entry point -------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ccrewrite.py",
        description="Rewrite Claude Code's messages through codex exec.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    install = sub.add_parser("install", help="write the MessageDisplay hook")
    install.add_argument(
        "--scope",
        choices=["home", "project"],
        default="home",
        help="home writes ~/.claude/settings.json, project writes ./.claude/settings.json",
    )
    install.add_argument(
        "--lang",
        choices=LANGS,
        required=True,
        help="zh rewrites Chinese only and leaves the rest verbatim, all rewrites everything",
    )
    install.set_defaults(run=cmd_install)

    remove = sub.add_parser("uninstall", help="remove the MessageDisplay hook")
    remove.add_argument("--scope", choices=["home", "project"], default="home")
    remove.set_defaults(run=cmd_uninstall)

    check = sub.add_parser("check", help="run one real rewrite and time it")
    check.set_defaults(run=cmd_check)

    prose = sub.add_parser("rewrite", help="rewrite text from an argument or stdin")
    prose.add_argument("text", nargs="*", help="the text to rewrite. Omit it to read stdin.")
    prose.add_argument("--time", action="store_true", help="print the elapsed time on stderr")
    prose.add_argument(
        "--timeout",
        type=float,
        default=CODEX_TIMEOUT,
        metavar="SECONDS",
        help=(
            f"how long to wait for codex, default {CODEX_TIMEOUT:.0f}. A rewrite costs"
            " about 15s plus 9ms per character, so raise this for a long document."
        ),
    )
    prose.set_defaults(run=cmd_rewrite)

    listen = sub.add_parser("hook", help="read a MessageDisplay hook payload on stdin")
    listen.add_argument("--lang", choices=LANGS, default=DEFAULT_LANG)
    listen.set_defaults(run=run_hook)

    for knobs in (install, check, prose, listen):
        knobs.add_argument("--model", default=DEFAULT_MODEL, help="any model codex exec accepts")
        knobs.add_argument("--effort", choices=EFFORTS, default=DEFAULT_EFFORT)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
