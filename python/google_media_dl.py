# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "playwright>=1.50.0",
#     "click>=8.0",
#     "httpx>=0.25.0",
#     "yt-dlp>=2024.0",
# ]
# ///
"""
Video Capture - yt-dlp primary, Playwright fallback for view-only Google Drive.

Downloads videos from any URL using yt-dlp. For view-only Google Drive files
(where yt-dlp fails), falls back to Playwright stream capture.

USAGE:
    # Run directly from GitHub (no clone needed):
    URL=https://raw.githubusercontent.com/CJHwong/toolkit/main/python/google_media_dl.py

    # YouTube / YouTube Music (uses yt-dlp)
    uv run $URL 'https://www.youtube.com/watch?v=VIDEO_ID'
    uv run $URL -a 'https://music.youtube.com/watch?v=VIDEO_ID'

    # Google Drive (yt-dlp first, Playwright fallback for view-only)
    uv run $URL 'https://drive.google.com/file/d/FILE_ID/view'

    # Audio only
    uv run $URL -a 'https://drive.google.com/file/d/FILE_ID/view'

    # With explicit output filename
    uv run $URL 'https://drive.google.com/file/d/FILE_ID/view' output.mp4

    # Visible browser for Playwright fallback debugging
    uv run $URL --no-headless -v 'https://drive.google.com/file/d/FILE_ID/view'

    # Use Firefox cookies instead of Zen
    uv run $URL -b firefox 'https://www.youtube.com/watch?v=VIDEO_ID'

    # Or run locally:
    uv run google_media_dl.py [OPTIONS] URL [OUTPUT]

OPTIONS:
    -a, --audio-only            Download audio only (m4a)
    -b, --browser TEXT          Browser to extract cookies from (default: zen)
    --headless/--no-headless    Run browser headless (default: headless, Playwright only)
    -t, --timeout INT           Timeout in seconds for stream capture (default: 30, Playwright only)
    -v, --verbose               Verbose output
    --version                   Show version
    -h, --help                  Show help

HOW IT WORKS:
    1. Tries yt-dlp with browser cookies — works for YouTube, most sites,
       and some Google Drive files that allow downloads.
    2. If yt-dlp fails on a Google Drive URL, falls back to Playwright:
       a. Launches Firefox and injects browser cookies
       b. Navigates to the file and triggers video playback
       c. Intercepts videoplayback CDN URLs (self-authenticating with signed tokens)
       d. Downloads streams via httpx with sequential range requests
    3. If yt-dlp fails on a non-Google Drive URL, reports the error and exits.

REQUIREMENTS:
    - A browser with an active Google session (for authenticated content)
    - yt-dlp: brew install yt-dlp
    - ffmpeg (for muxing video+audio): brew install ffmpeg
    - Playwright Firefox (Google Drive fallback only): uv run playwright install firefox
      (Not needed for YouTube / YouTube Music — those use yt-dlp directly)

NOTES:
    - Zen browser is auto-detected via profiles.ini.
    - First run of Playwright fallback may need: playwright install firefox
    - Google Drive uses DASH (adaptive streaming) — video and audio are separate
      streams that get muxed together with ffmpeg after download.
"""
import configparser
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import click
import httpx
from playwright.sync_api import sync_playwright

# Google Drive video stream itag -> resolution mapping (common ones)
ITAG_MAP: dict[int, dict[str, str | int]] = {
    18: {"res": "360p", "height": 360, "mime": "video/mp4"},
    22: {"res": "720p", "height": 720, "mime": "video/mp4"},
    37: {"res": "1080p", "height": 1080, "mime": "video/mp4"},
    59: {"res": "480p", "height": 480, "mime": "video/mp4"},
    78: {"res": "480p", "height": 480, "mime": "video/mp4"},
    84: {"res": "720p", "height": 720, "mime": "video/mp4"},
    85: {"res": "1080p", "height": 1080, "mime": "video/mp4"},
    # Adaptive streams (video only)
    134: {"res": "360p", "height": 360, "mime": "video/mp4"},
    135: {"res": "480p", "height": 480, "mime": "video/mp4"},
    136: {"res": "720p", "height": 720, "mime": "video/mp4"},
    137: {"res": "1080p", "height": 1080, "mime": "video/mp4"},
    298: {"res": "720p60", "height": 720, "mime": "video/mp4"},
    299: {"res": "1080p60", "height": 1080, "mime": "video/mp4"},
}


def is_google_drive_url(url: str) -> bool:
    """Check if a URL points to Google Drive."""
    return "drive.google.com" in url or "docs.google.com" in url


def try_ytdlp(
    url: str,
    output: str | None,
    audio_only: bool,
    browser: str,
    verbose: bool,
) -> tuple[bool, str]:
    """Try downloading with yt-dlp. Returns (success, stderr_output)."""
    if browser == "zen":
        profile_path = resolve_zen_profile()
        cookie_arg = f"firefox:{profile_path}"
    else:
        cookie_arg = browser

    cmd = ["yt-dlp", "--cookies-from-browser", cookie_arg]

    if audio_only:
        cmd += ["-x", "--audio-format", "m4a", "--audio-quality", "0"]

    cmd += [
        "--no-part", "--concurrent-fragments", "8", "--buffer-size", "16K",
        "--remote-components", "ejs:github",
    ]

    if is_google_drive_url(url):
        cmd += ["--http-chunk-size", "10M"]

    if output:
        cmd += ["-o", output]

    cmd.append(url)

    if verbose:
        click.echo(f"[INFO] Running: {' '.join(cmd)}")

    # yt-dlp writes all output (progress, info, errors) to stderr.
    # Capture it for analysis while streaming to terminal in verbose mode.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    stderr_chunks: list[bytes] = []
    fd = proc.stderr.fileno()
    while True:
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        stderr_chunks.append(chunk)
        if verbose:
            sys.stderr.buffer.write(chunk)
            sys.stderr.buffer.flush()

    proc.wait()
    stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
    return proc.returncode == 0, stderr



def log(message: str, verbose: bool = True, level: str = "INFO"):
    """Log message if verbose or if error."""
    if verbose or level == "ERROR":
        click.echo(f"[{level}] {message}", err=(level == "ERROR"))


def extract_file_id(url: str) -> str | None:
    """Extract Google Drive file ID from various URL formats."""
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return None


def resolve_zen_profile() -> str:
    """Resolve Zen browser's default Firefox-compatible profile path."""
    system = platform.system()
    if system == "Darwin":
        zen_base = Path.home() / "Library" / "Application Support" / "zen"
    elif system == "Linux":
        zen_base = Path.home() / ".zen"
    else:
        raise click.UsageError(f"Zen browser is not supported on {system}")

    profiles_ini = zen_base / "profiles.ini"
    if not profiles_ini.exists():
        raise click.UsageError(f"Zen profiles.ini not found at {profiles_ini}")

    config = configparser.ConfigParser()
    config.read(profiles_ini)

    for section in config.sections():
        if section.startswith("Install"):
            default_profile = config.get(section, "Default", fallback=None)
            if default_profile:
                profile_path = zen_base / default_profile
                if not profile_path.is_dir():
                    raise click.UsageError(
                        f"Zen profile directory not found: {profile_path}"
                    )
                return str(profile_path)

    raise click.UsageError(
        f"Could not detect default Zen profile from {profiles_ini}"
    )


def extract_cookies(browser: str, verbose: bool = False) -> list[dict]:
    """Extract Google cookies from browser using yt-dlp's cookie extractor."""
    from yt_dlp.cookies import extract_cookies_from_browser

    if browser == "zen":
        profile_path = resolve_zen_profile()
        log(f"Zen profile resolved to: {profile_path}", verbose)
        cookie_jar = extract_cookies_from_browser("firefox", profile=profile_path)
    else:
        cookie_jar = extract_cookies_from_browser(browser)

    pw_cookies = []
    for cookie in cookie_jar:
        domain = cookie.domain
        if not domain:
            continue
        if not any(
            d in domain for d in [".google.com", "google.com", ".googleapis.com"]
        ):
            continue
        pw_cookies.append({
            "name": cookie.name,
            "value": cookie.value,
            "domain": domain,
            "path": cookie.path or "/",
            "secure": bool(cookie.secure),
            "httpOnly": bool(cookie.has_nonstandard_attr("HttpOnly")),
        })

    log(f"Extracted {len(pw_cookies)} Google cookies", verbose)
    if not pw_cookies:
        raise click.UsageError(
            "No Google cookies found. Make sure you're logged into Google "
            f"in your {browser} browser."
        )
    return pw_cookies


def parse_stream_info(url: str) -> dict | None:
    """Parse a videoplayback URL to extract stream metadata."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    itag_str = params.get("itag", [None])[0]
    if itag_str is None:
        return None

    itag = int(itag_str)
    mime = params.get("mime", [params.get("type", ["unknown"])[0]])[0]
    clen_str = params.get("clen", [None])[0]
    clen = int(clen_str) if clen_str else None
    info = ITAG_MAP.get(itag, {"res": "unknown", "height": 0, "mime": mime})

    return {
        "url": url,
        "itag": itag,
        "height": info["height"],
        "res": info["res"],
        "mime": info.get("mime", mime),
        "clen": clen,
    }


def pick_best_stream(
    streams: list[dict], verbose: bool = False
) -> dict | None:
    """Pick the highest resolution stream available."""
    if not streams:
        return None

    candidates = sorted(streams, key=lambda s: s["height"], reverse=True)
    selected = candidates[0]
    log(
        f"Selected stream: itag={selected['itag']} "
        f"res={selected['res']} mime={selected['mime']}",
        verbose,
    )
    return selected


def capture_streams(
    url: str,
    cookies: list[dict],
    headless: bool = True,
    timeout: int = 30,
    verbose: bool = False,
) -> tuple[list[dict], str | None, list[dict]]:
    """Launch Playwright, navigate to Drive URL, and capture videoplayback URLs.

    Returns (streams, title, browser_cookies).
    """
    streams: list[dict] = []
    seen_itags: set[int] = set()
    title: str | None = None

    def on_response(response):
        nonlocal title
        url_str = response.url
        if "videoplayback?" not in url_str:
            return
        info = parse_stream_info(url_str)
        if info and info["itag"] not in seen_itags:
            seen_itags.add(info["itag"])
            streams.append(info)
            log(
                f"Captured stream: itag={info['itag']} "
                f"res={info['res']} mime={info['mime']}",
                verbose,
            )

    with sync_playwright() as pw:
        browser = pw.firefox.launch(headless=headless)
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:137.0) "
                "Gecko/20100101 Firefox/137.0"
            ),
        )

        log(f"Injecting {len(cookies)} cookies...", verbose)
        context.add_cookies(cookies)

        # Mute audio so playback doesn't produce sound
        page = context.new_page()
        page.evaluate("() => { Object.defineProperty(HTMLMediaElement.prototype, 'muted', { get: () => true, set: () => {} }) }")
        page.on("response", on_response)

        log(f"Navigating to {url}...", verbose)
        page.goto(url, wait_until="domcontentloaded")

        # Extract title
        try:
            title_el = page.wait_for_selector(
                "[data-tooltip-unhoverable='true'], .uc-name-size a",
                timeout=10000,
            )
            if title_el:
                title = title_el.text_content()
        except Exception:
            pass

        if not title:
            try:
                title = page.title()
                title = re.sub(r"\s*-\s*Google Drive\s*$", "", title).strip()
            except Exception:
                pass

        log(f"Page title: {title or '(unknown)'}", verbose)

        # Click play
        log("Waiting for video player...", verbose)
        try:
            play_selectors = [
                "[aria-label='Play']",
                "[data-tooltip='Play']",
                "button.ytp-large-play-button",
                ".ndfHFb-c4YZDc-cYSp0e-DARUcf",
                "video",
            ]
            play_clicked = False
            for selector in play_selectors:
                try:
                    el = page.wait_for_selector(selector, timeout=5000)
                    if el:
                        el.click()
                        play_clicked = True
                        log(f"Clicked play button: {selector}", verbose)
                        break
                except Exception:
                    continue

            if not play_clicked:
                log("No play button found, clicking page center...", verbose)
                page.mouse.click(640, 360)

        except Exception as e:
            log(f"Play button interaction failed: {e}", verbose)

        # Wait for streams
        log(f"Waiting up to {timeout}s for video streams...", verbose)
        deadline = timeout * 1000
        elapsed = 0
        while elapsed < deadline and not streams:
            page.wait_for_timeout(500)
            elapsed += 500

        if not streams:
            log("No streams captured, attempting direct video play...", verbose)
            try:
                page.evaluate("document.querySelector('video')?.play()")
                page.wait_for_timeout(5000)
            except Exception:
                pass

        if streams:
            log("Waiting for additional quality streams...", verbose)
            page.wait_for_timeout(3000)

        log(f"Captured {len(streams)} unique stream(s) total", verbose)

        browser_cookies = context.cookies()
        browser.close()

    return streams, title, browser_cookies


def build_cookie_header(browser_cookies: list[dict]) -> str:
    """Build a Cookie header string from Playwright browser cookies."""
    parts = []
    for c in browser_cookies:
        domain = c.get("domain", "")
        if any(d in domain for d in [".google.com", "google.com", ".googleapis.com"]):
            parts.append(f"{c['name']}={c['value']}")
    return "; ".join(parts)


def strip_range_params(url: str) -> str:
    """Strip range/chunk params from a videoplayback URL.

    Removes range, rn, rbuf and other player-specific params that are NOT
    part of the signed params (sparams). This gives us a clean base URL
    that we can use with HTTP Range headers for downloading.
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    for key in ("range", "rn", "rbuf", "ump", "srfvp", "cpn", "c", "cver", "alr"):
        params.pop(key, None)
    cleaned_parts = []
    for key, values in params.items():
        for val in values:
            cleaned_parts.append(f"{key}={val}")
    cleaned_query = "&".join(cleaned_parts)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{cleaned_query}"


def download_stream(
    stream: dict,
    output_path: Path,
    browser_cookies: list[dict] | None = None,
    verbose: bool = False,
):
    """Download a video stream using HTTP Range headers in 2 MB chunks.

    The range query parameter wraps data in protobuf metadata. HTTP Range
    headers return raw media data, producing valid MP4/M4A files.
    """
    clean_url = strip_range_params(stream["url"])
    total_bytes = stream.get("clen")
    log(f"Downloading to {output_path}...", verbose)

    headers = {
        "Referer": "https://drive.google.com/",
        "Origin": "https://drive.google.com",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:137.0) "
            "Gecko/20100101 Firefox/137.0"
        ),
    }
    if browser_cookies:
        headers["Cookie"] = build_cookie_header(browser_cookies)

    if total_bytes:
        total_mb = total_bytes / (1024 * 1024)
        log(f"File size: {total_mb:.1f} MB", verbose)

    chunk_size = 2 * 1024 * 1024  # 2 MB per request
    downloaded = 0

    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0),
    ) as client:
        with open(output_path, "wb") as f:
            while True:
                start = downloaded
                end = start + chunk_size - 1
                if total_bytes:
                    end = min(end, total_bytes - 1)

                chunk_headers = {**headers, "Range": f"bytes={start}-{end}"}
                resp = client.get(clean_url, headers=chunk_headers)

                if resp.status_code not in (200, 206):
                    resp.raise_for_status()

                data = resp.content
                if not data:
                    break

                f.write(data)
                downloaded += len(data)

                if total_bytes and sys.stdout.isatty():
                    pct = downloaded / total_bytes * 100
                    dl_mb = downloaded / (1024 * 1024)
                    click.echo(
                        f"\r  {dl_mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)",
                        nl=False,
                    )

                # Done conditions
                if resp.status_code == 200:
                    break  # Server sent full file
                if total_bytes and downloaded >= total_bytes:
                    break
                if len(data) < (end - start + 1):
                    break

    if sys.stdout.isatty():
        click.echo()

    final_mb = output_path.stat().st_size / (1024 * 1024)
    log(f"Downloaded {final_mb:.1f} MB to {output_path}", verbose=True)


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip(". ")
    return name or "video"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("url")
@click.argument("output", required=False, default=None)
@click.option(
    "-a", "--audio-only", is_flag=True,
    help="Download audio only (m4a)",
)
@click.option(
    "-b", "--browser", default="zen",
    help="Browser to extract cookies from (default: zen)",
)
@click.option(
    "--headless/--no-headless", default=True,
    help="Run browser headless (default: headless)",
)
@click.option(
    "-t", "--timeout", default=30, type=int,
    help="Timeout in seconds for stream capture (default: 30)",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.version_option(version="0.1.0")
def cli(
    url: str,
    output: str | None,
    audio_only: bool,
    browser: str,
    headless: bool,
    timeout: int,
    verbose: bool,
):
    """Download videos via yt-dlp, with Playwright fallback for view-only Google Drive.

    URL is any video URL (YouTube, Google Drive, etc.).
    OUTPUT is the optional output filename (default: yt-dlp picks title-based name).

    \b
    Examples:
        google_media_dl.py 'https://www.youtube.com/watch?v=VIDEO_ID'
        google_media_dl.py -a 'https://music.youtube.com/watch?v=VIDEO_ID'
        google_media_dl.py 'https://drive.google.com/file/d/FILE_ID/view'
        google_media_dl.py -a URL                    # audio only
        google_media_dl.py --no-headless -v URL       # visible browser (Playwright fallback)
    """
    # Step 1: Try yt-dlp first (works for YouTube, most sites, some GDrive)
    click.echo("Trying yt-dlp...")
    success, stderr = try_ytdlp(url, output, audio_only, browser, verbose)
    if success:
        return

    # Step 2: yt-dlp failed — fallback only for Google Drive
    if not is_google_drive_url(url):
        if not verbose and stderr:
            click.echo(stderr, err=True)
        raise click.ClickException("yt-dlp failed.")

    click.echo("yt-dlp failed. Falling back to Playwright stream capture...")

    file_id = extract_file_id(url)
    if not file_id:
        raise click.UsageError(
            "Could not extract file ID from URL. "
            "Expected format: https://drive.google.com/file/d/FILE_ID/view"
        )

    canonical_url = f"https://drive.google.com/file/d/{file_id}/view"
    log(f"File ID: {file_id}", verbose)

    # Extract cookies for Playwright
    click.echo(f"Extracting cookies from {browser}...")
    cookies = extract_cookies(browser, verbose)

    # Capture streams via Playwright
    click.echo("Launching browser to capture video streams...")
    try:
        streams, title, browser_cookies = capture_streams(
            canonical_url, cookies, headless=headless,
            timeout=timeout, verbose=verbose,
        )
    except Exception as e:
        if "Executable doesn't exist" in str(e):
            raise click.ClickException(
                "Playwright Firefox browser is not installed.\n"
                "  Run: uv run playwright install firefox"
            ) from e
        raise

    if not streams:
        raise click.ClickException(
            "No video streams captured. Possible causes:\n"
            "  - The file may not be a video\n"
            "  - Your Google session may lack access to this file\n"
            "  - The video player didn't start (try --no-headless to debug)\n"
            "  - Try increasing --timeout"
        )

    # Separate video and audio streams
    video_streams = [s for s in streams if "video" in s.get("mime", "")]
    audio_streams = [s for s in streams if "audio" in s.get("mime", "")]

    log(f"Video streams: {len(video_streams)}, Audio streams: {len(audio_streams)}", verbose)
    for s in streams:
        log(f"  itag={s['itag']} res={s['res']} mime={s['mime']} clen={s.get('clen')}", verbose)

    # Audio-only mode
    if audio_only:
        if not audio_streams:
            raise click.ClickException("No audio stream captured.")

        selected_audio = audio_streams[0]
        clen_mb = (selected_audio.get("clen") or 0) / (1024 * 1024)
        click.echo(f"Audio: itag {selected_audio['itag']} ({clen_mb:.1f} MB)")

        if output is None:
            base_name = sanitize_filename(title) if title else f"gdrive_{file_id}"
            output = base_name + ".m4a"

        output_path = Path(output)
        click.echo("Downloading audio...")
        download_stream(selected_audio, output_path, browser_cookies, verbose)
        click.echo(f"Saved: {output_path}")
        return

    # Video mode — pick best video stream
    selected_video = pick_best_stream(video_streams, verbose)
    if not selected_video:
        raise click.ClickException("No video stream found in captured URLs.")

    selected_audio = audio_streams[0] if audio_streams else None
    click.echo(
        f"Video: {selected_video['res']} (itag {selected_video['itag']})"
        + (f" | Audio: itag {selected_audio['itag']}" if selected_audio else " | No audio")
    )

    # Determine output filename
    if output is None:
        base_name = sanitize_filename(title) if title else f"gdrive_{file_id}"
        output = base_name + ".mp4"

    output_path = Path(output)

    # Download video (and audio if available), then mux
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        if selected_audio:
            video_tmp = tmp / "video.mp4"
            audio_tmp = tmp / "audio.m4a"

            click.echo(f"Downloading video ({selected_video['res']})...")
            download_stream(selected_video, video_tmp, browser_cookies, verbose)

            click.echo("Downloading audio...")
            download_stream(selected_audio, audio_tmp, browser_cookies, verbose)

            click.echo("Muxing video + audio...")
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(video_tmp),
                "-i", str(audio_tmp),
                "-c", "copy",
                "-movflags", "+faststart",
                str(output_path),
            ]
            log(f"Running: {' '.join(ffmpeg_cmd)}", verbose)
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                log(f"ffmpeg stderr: {result.stderr}", verbose)
                raise click.ClickException(
                    f"ffmpeg muxing failed (exit {result.returncode}). "
                    "Is ffmpeg installed? (brew install ffmpeg)"
                )
        else:
            click.echo(f"Downloading video ({selected_video['res']})...")
            download_stream(selected_video, output_path, browser_cookies, verbose)

    click.echo(f"Saved: {output_path}")


if __name__ == "__main__":
    cli()
