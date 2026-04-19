// Auto Dub — main app. Thin React shell over window.pywebview.api.
const { useState, useEffect, useRef, useMemo, useCallback } = React;

const DEFAULT_FORMATS = 'MP4 · MOV · MKV · WAV · M4A';

// ─── Empty state ──────────────────────────────────────────────────
function EmptyState({ onPick, recents, onOpenRecent, ready }) {
  const [over, setOver] = useState(false);
  return (
    <div className="empty">
      <div style={{ width: '100%', maxWidth: 720 }}>
        <div
          className={`dropzone ${over ? 'over' : ''}`}
          onDragOver={(e)=>{e.preventDefault(); setOver(true);}}
          onDragLeave={()=>setOver(false)}
          onDrop={(e)=>{
            e.preventDefault(); setOver(false);
            const f = e.dataTransfer.files && e.dataTransfer.files[0];
            if (f && f.path) onOpenRecent(f.path);
            else onPick();
          }}
          onClick={() => ready && onPick()}
        >
          <div className="dz-icon">
            <Icon name="film" size={28} stroke={1.4} style={{ color: 'var(--accent)' }}/>
          </div>
          <div>
            <div className="dz-title">Drop a video to dub</div>
            <div className="dz-sub" style={{ marginTop: 8 }}>
              Auto Dub transcribes the original audio, optionally translates it, and re-voices each segment using the speaker's own voice.
            </div>
          </div>
          <div className="row" style={{ gap: 10 }}>
            <button className="btn primary lg" onClick={(e)=>{e.stopPropagation(); ready && onPick();}} disabled={!ready}>
              <Icon name="file" size={14}/> Choose video…
            </button>
          </div>
          <div className="dz-formats">{DEFAULT_FORMATS}</div>
        </div>

        {recents.length > 0 && (
          <div className="recents">
            <div className="recents-title">Recent projects</div>
            {recents.map((r, i) => (
              <div key={i} className="recent-row" onClick={() => onOpenRecent(r.path)}>
                <div className="recent-thumb"/>
                <div className="grow">
                  <div className="recent-name">{r.name}</div>
                  <div className="recent-meta">{r.meta || r.path}</div>
                </div>
                <Icon name="chevr" size={14} style={{ color: 'var(--text-faint)' }}/>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── RunBar ──────────────────────────────────────────────────────
const STAGES = [
  { key: 'tts', label: 'Generate voice (TTS)' },
  { key: 'mux', label: 'Combine & mux' },
];

function RunBar({ state, cancelling, onRun, onCancel, onToggleLogs, logsOpen, progress, stage, etaSec, transcriptReady, outputUrl, onReveal }) {
  const running = state === 'running';
  const done = state === 'done';
  const stageIdx = STAGES.findIndex(s => s.key === stage);
  const label = cancelling ? 'Cancelling…'
              : running ? (STAGES[stageIdx >= 0 ? stageIdx : 0]?.label || 'Rendering')
              : done ? 'Complete'
              : !transcriptReady ? 'Preparing transcript…'
              : 'Ready to dub';

  return (
    <div className={`runbar ${running ? 'running' : ''}`}>
      <button className={`btn ghost sm`} onClick={onToggleLogs}>
        <Icon name="logs" size={12}/>
        Logs
        <Icon name={logsOpen ? 'chevd' : 'chevu'} size={12} style={{ marginLeft: 2, color: 'var(--text-faint)' }}/>
      </button>
      <div style={{ width: 0.5, height: 28, background: 'var(--line)' }}/>
      <div style={{ minWidth: 160 }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)' }}>{label}</div>
        <div className="mono" style={{ fontSize: 10.5, color: 'var(--text-faint)', marginTop: 2 }}>
          {running ? (
            <>Stage {Math.max(1, stageIdx+1)}/{STAGES.length} · ETA {etaSec > 0 ? fmtShort(etaSec) : '—'}</>
          ) : done ? (
            'Dub saved — use Reveal in Finder to open'
          ) : (
            <>{STAGES.length} stages · first run may download the TTS model</>
          )}
        </div>
      </div>
      <div className={`progress ${running && progress < 0.02 ? 'indet' : ''}`}>
        <div className="progress-fill" style={{ width: done ? '100%' : `${Math.max(0, Math.min(1, progress))*100}%`, background: done ? 'var(--success)' : 'var(--accent)' }}/>
      </div>
      <div className="mono" style={{ fontSize: 11, color: 'var(--text-faint)', minWidth: 44, textAlign: 'right' }}>
        {done ? '100%' : `${Math.round(progress*100)}%`}
      </div>
      {running ? (
        <button className="btn" onClick={onCancel} disabled={cancelling}>
          {cancelling ? (
            <>
              <span className="tsc-spin" style={{ width: 10, height: 10, borderWidth: 1.2 }}/>
              Cancelling…
            </>
          ) : (
            <>
              <Icon name="x" size={12}/> Cancel
            </>
          )}
        </button>
      ) : done ? (
        <>
          <button className="btn" onClick={onReveal}>
            <Icon name="save" size={12}/> Reveal in Finder
          </button>
          <button className="btn primary" onClick={onRun}>
            <Icon name="refresh" size={12}/> Re-render
          </button>
        </>
      ) : (
        <button className="btn primary lg" onClick={onRun} disabled={!transcriptReady}>
          <Icon name="sparkle" size={13}/> Dub video
        </button>
      )}
    </div>
  );
}

// ─── Logs panel ──────────────────────────────────────────────────
function Logs({ entries, open }) {
  const ref = useRef();
  useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [entries]);
  return (
    <div className={`logs-wrap ${open ? 'open' : ''}`}>
      <div className="logs" ref={ref}>
        {entries.filter(Boolean).map((e, i) => (
          <div key={i}>
            <span className="time">{e.t}</span>{'  '}
            <span className={e.level || e.lvl}>{e.msg}</span>
          </div>
        ))}
        {entries.length === 0 && <div className="faint">No output yet. Press <b>Dub video</b> to begin.</div>}
      </div>
    </div>
  );
}

// ─── Bridge helpers ──────────────────────────────────────────────
function usePywebviewReady() {
  const [ready, setReady] = useState(!!(window.pywebview && window.pywebview.api));
  useEffect(() => {
    if (ready) return;
    const onReady = () => setReady(true);
    window.addEventListener('pywebviewready', onReady);
    // Fallback poll (pywebview sometimes fires ready before listeners mount)
    const id = setInterval(() => {
      if (window.pywebview && window.pywebview.api) { setReady(true); clearInterval(id); }
    }, 50);
    return () => { window.removeEventListener('pywebviewready', onReady); clearInterval(id); };
  }, [ready]);
  return ready;
}

const api = () => window.pywebview && window.pywebview.api;

// ─── App ─────────────────────────────────────────────────────────
function App() {
  const ready = usePywebviewReady();

  // File / recents
  const [file, setFile] = useState(null);
  const [recents, setRecents] = useState([]);

  // Configuration
  const [sourceLang, setSourceLang] = useState('chinese');
  const [targetLang, setTargetLang] = useState('english');
  const [translate, setTranslate] = useState(true);
  const [voiceMode, setVoiceMode] = useState('auto');
  const [modelSize, setModelSize] = useState('quality');
  const [refVoiceName, setRefVoiceName] = useState(null);
  // Reference transcript is per voice mode — auto-extracted clip and
  // uploaded clip are different audios and need different transcripts.
  const [refTranscriptAuto, setRefTranscriptAuto] = useState('');
  const [refTranscriptUpload, setRefTranscriptUpload] = useState('');
  const refTranscript = voiceMode === 'upload' ? refTranscriptUpload : refTranscriptAuto;
  const setRefTranscript = voiceMode === 'upload' ? setRefTranscriptUpload : setRefTranscriptAuto;
  const [refClip, setRefClip] = useState(null);
  const [refTranscribing, setRefTranscribing] = useState(false);
  const [showOriginal, setShowOriginal] = useState(true);

  // Transcript / segments
  const [transcriptState, setTranscriptState] = useState('idle'); // idle | awaiting | transcribing | translating | ready | error
  const [segments, setSegments] = useState([]);
  const [originalMap, setOriginalMap] = useState({});
  const [activeSegIdx, setActiveSegIdx] = useState(null);
  const [outputName, setOutputName] = useState('');

  // Player
  const videoRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [audioTrack, setAudioTrack] = useState('dubbed');

  // Run state
  const [runState, setRunState] = useState('idle');
  const [cancelling, setCancelling] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState(null);
  const [logs, setLogs] = useState([]);
  const [logsOpen, setLogsOpen] = useState(false);
  const [etaSec, setEtaSec] = useState(0);
  const [output, setOutput] = useState(null); // {path, url, name}
  const [error, setError] = useState(null);
  const [envIssues, setEnvIssues] = useState(null);  // {missing:[tool], hints:{tool:cmd}} when deps missing

  const runStart = useRef(0);
  const segDoneTimes = useRef([]);  // performance.now() at each segment completion (bounded)
  const cacheProbeSeq = useRef(0);   // ignore stale probe results after config changes
  // Values the event handler closure needs, mirrored through a ref so the
  // handler can install once instead of re-binding on every keystroke.
  const handlerDeps = useRef({ transcriptState: 'idle', refTranscriptAuto: '' });
  handlerDeps.current = { transcriptState, refTranscriptAuto };

  // Load recents when bridge is ready
  useEffect(() => {
    if (!ready) return;
    api().get_recents().then(setRecents).catch(() => {});
  }, [ready]);

  // Check external tool availability on startup, and again when `translate`
  // flips (claude is only required when translating).
  useEffect(() => {
    if (!ready) return;
    api().check_environment(translate).then(env => {
      if (env && !env.ok) setEnvIssues({ missing: env.missing, hints: env.hints });
      else setEnvIssues(null);
    }).catch(() => {});
  }, [ready, translate]);

  // Event bus from Python. Installed once; latest-deps read through
  // handlerDeps.current so user typing in the ref transcript doesn't
  // reinstall a handler 60× per second.
  useEffect(() => {
    const SEG_TIMES_CAP = 10;
    const ETA_WINDOW = 5;

    window.__autoDubEvent = (evt) => {
      if (!evt || !evt.type) return;
      switch (evt.type) {
        case 'log':
          setLogs(l => [...l, evt]);
          break;
        case 'stage':
          setStage(evt.key);
          if (evt.key === 'transcribe') setTranscriptState('transcribing');
          else if (evt.key === 'translate') setTranscriptState('translating');
          else if (evt.key === 'tts') setProgress(p => Math.max(p, 0.04));
          else if (evt.key === 'mux') setProgress(0.92);
          else if (evt.key === 'done') setProgress(1);
          break;
        case 'segment':
          if (evt.status === 'done') {
            const times = segDoneTimes.current;
            times.push(performance.now());
            if (times.length > SEG_TIMES_CAP) times.splice(0, times.length - SEG_TIMES_CAP);
          }
          setSegments(segs => {
            const next = segs.map(s => s.i === evt.idx ? { ...s, status: evt.status } : s);
            if (evt.status === 'done') {
              const total = next.length || 1;
              const done = next.filter(s => s.status === 'done').length;
              // 0.05 baseline (extract+transcribe done pre-Dub), 0.85 span for
              // per-segment TTS, 0.10 reserved for combine+mux.
              setProgress(0.05 + (done / total) * 0.85);
              // Rolling window ETA — wall-clock across all segments skews
              // early on (model load counted against the first segment).
              const times = segDoneTimes.current;
              if (times.length >= 2) {
                const recent = times.slice(-ETA_WINDOW);
                const perSegMs = (recent[recent.length - 1] - recent[0]) / (recent.length - 1);
                setEtaSec(Math.max(0, Math.round(perSegMs * (total - done) / 1000)));
              } else {
                setEtaSec(0);
              }
            }
            return next;
          });
          break;
        case 'transcript_ready':
          setSegments((evt.segments || []).map(s => ({ ...s, status: 'ready', edited: false })));
          setOriginalMap(evt.original || {});
          setOutputName(evt.output_name || '');
          setRefClip(evt.ref || null);
          // Populate the auto-mode slot from the ref clip's SRT text.
          // Upload slot is whatever the user typed — never overwritten here.
          if (evt.ref && evt.ref.text && !handlerDeps.current.refTranscriptAuto) {
            setRefTranscriptAuto(evt.ref.text);
          }
          setTranscriptState('ready');
          break;
        case 'segment_ready':
          setSegments(segs => segs.map(s => s.i === evt.idx ? { ...s, status: 'done' } : s));
          break;
        case 'ref_transcribed':
          if (evt.mode === 'upload') setRefTranscriptUpload(evt.text || '');
          else setRefTranscriptAuto(evt.text || '');
          setRefTranscribing(false);
          break;
        case 'done':
          setOutput({ path: evt.output, url: evt.url, name: evt.name });
          setRunState('done');
          setProgress(1);
          break;
        case 'cancelling':
          setCancelling(true);
          break;
        case 'cancelled':
          setRunState('idle');
          setProgress(0);
          setRefTranscribing(false);
          setCancelling(false);
          if (handlerDeps.current.transcriptState === 'transcribing' ||
              handlerDeps.current.transcriptState === 'translating') {
            setTranscriptState('awaiting');
          }
          break;
        case 'error':
          setError(evt.msg || 'Pipeline failed');
          if (evt.phase === 'transcribe') setTranscriptState('awaiting');
          setRefTranscribing(false);
          setCancelling(false);
          setRunState(prev => prev === 'running' ? 'idle' : prev);
          break;
      }
    };
    return () => { delete window.__autoDubEvent; };
  }, []);

  // Probe on-disk cache whenever the video or relevant config changes.
  // A cache hit is handled via the `transcript_ready` event (flipping state
  // to 'ready' and populating segments); a miss resets to 'awaiting'.
  useEffect(() => {
    if (!ready || !file) return;
    if (runState === 'running') return;
    if (transcriptState === 'transcribing' || transcriptState === 'translating') return;
    const seq = ++cacheProbeSeq.current;
    (async () => {
      try {
        const result = await api().probe_cache({
          language: sourceLang,
          target_lang: translate ? targetLang : null,
          use_ref_upload: voiceMode === 'upload',
          small: modelSize === 'fast',
        });
        // Drop stale results if the user flipped config again meanwhile
        if (seq !== cacheProbeSeq.current) return;
        if (!result || !result.cached) {
          setTranscriptState('awaiting');
          setSegments([]);
          // Dub from a prior config is no longer relevant — drop the
          // Dubbed track toggle and Re-render state until the user
          // produces a dub for the new config.
          setOutput(null);
          setRunState('idle');
          setProgress(0);
        }
      } catch { /* probe failure is non-fatal — user can still click Transcribe */ }
    })();
  }, [ready, file, sourceLang, targetLang, translate, voiceMode, modelSize]);

  // Active segment tracking. onTimeUpdate fires ~4×/sec; guard the
  // setActiveSegIdx so React doesn't reconcile the full transcript when
  // the playhead is still inside the same segment.
  useEffect(() => {
    const idx = segments.findIndex(s => currentTime >= s.start && currentTime < s.end);
    if (idx !== -1 && idx !== activeSegIdx) setActiveSegIdx(idx);
  }, [currentTime, segments, activeSegIdx]);

  // Video URL — swap when audio track toggles or file/output changes
  const videoUrl = useMemo(() => {
    if (!file) return null;
    if (output && audioTrack === 'dubbed') return output.url;
    return file.url;
  }, [file, output, audioTrack]);

  // When the src changes, restore the previous playhead
  const lastUrlRef = useRef(null);
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !videoUrl) return;
    if (lastUrlRef.current === videoUrl) return;
    lastUrlRef.current = videoUrl;
    const t = v.currentTime;
    const wasPlaying = !v.paused;
    const onLoaded = () => {
      if (t && isFinite(t) && t > 0.1) v.currentTime = t;
      if (wasPlaying) v.play().catch(() => {});
    };
    v.addEventListener('loadedmetadata', onLoaded, { once: true });
  }, [videoUrl]);

  // ── Actions ──
  // JS bridge exceptions arrive as `Error: <msg>`; strip the prefix so the
  // banner matches the friendlier Python-side error strings.
  const reportError = useCallback((e) => {
    const raw = String(e || '');
    setError(raw.replace(/^Error:\s*/, '') || 'Unknown error');
  }, []);

  const applyOpenedFile = useCallback((meta) => {
    if (!meta) return;
    setFile(meta);
    setTranscriptState('awaiting');
    setSegments([]);
    setRefClip(null);
    setRefVoiceName(null);
    setRefTranscriptAuto('');
    setRefTranscriptUpload('');
    setOutput(null);
    setRunState('idle');
    setProgress(0);
    setLogs([]);
  }, []);

  const openPath = useCallback(async (path) => {
    setError(null);
    try {
      applyOpenedFile(await api().open_file(path));
      setRecents(await api().get_recents());
    } catch (e) {
      reportError(e);
    }
  }, [applyOpenedFile, reportError]);

  const pickFile = useCallback(async () => {
    setError(null);
    try {
      const meta = await api().pick_file();
      if (!meta) return;
      applyOpenedFile(meta);
      setRecents(await api().get_recents());
    } catch (e) {
      reportError(e);
    }
  }, [applyOpenedFile, reportError]);

  const pickRefAudio = useCallback(async () => {
    try {
      const result = await api().pick_ref_audio();
      if (result) setRefVoiceName(result.name);
    } catch (e) {
      reportError(e);
    }
  }, []);

  const whisperRef = useCallback(async () => {
    setError(null);
    setRefTranscribing(true);
    try {
      await api().transcribe_ref_audio(sourceLang, voiceMode);
    } catch (e) {
      reportError(e);
      setRefTranscribing(false);
    }
  }, [sourceLang, voiceMode]);

  // When voice mode flips back to auto, tell Python
  useEffect(() => {
    if (!ready) return;
    if (voiceMode === 'auto') {
      api().set_ref_mode_auto().catch(() => {});
      setRefVoiceName(null);
    }
  }, [voiceMode, ready]);

  // Persist ref transcript edits to Python, per mode. Each slot syncs to
  // its own Python counterpart so switching voice modes doesn't leak text
  // across the two clips.
  useEffect(() => {
    if (!ready) return;
    api().set_ref_text(refTranscriptAuto, 'auto').catch(() => {});
  }, [refTranscriptAuto, ready]);
  useEffect(() => {
    if (!ready) return;
    api().set_ref_text(refTranscriptUpload, 'upload').catch(() => {});
  }, [refTranscriptUpload, ready]);

  const startTranscribe = useCallback(async () => {
    setError(null);
    setLogs([]);
    setTranscriptState('transcribing');
    try {
      await api().transcribe({
        language: sourceLang,
        target_lang: translate ? targetLang : null,
        small: modelSize === 'fast',
        use_ref_upload: voiceMode === 'upload',
      });
    } catch (e) {
      reportError(e);
      setTranscriptState('awaiting');
    }
  }, [sourceLang, targetLang, translate, modelSize, voiceMode]);

  const editSegmentText = useCallback(async (idx, text) => {
    setSegments(segs => segs.map(s => s.i === idx ? { ...s, text, edited: true, status: 'edited' } : s));
    try { await api().edit_segment(idx, { text }); } catch (e) { reportError(e); }
  }, []);

  const editSegmentTime = useCallback(async (idx, field, value) => {
    setSegments(segs => segs.map(s => {
      if (s.i !== idx) return s;
      let start = s.start, end = s.end;
      if (field === 'start') start = Math.max(0, Math.min(value, end - 0.1));
      if (field === 'end')   end   = Math.max(start + 0.1, value);
      return { ...s, start, end, edited: true, status: s.status === 'done' ? 'edited' : s.status };
    }));
    try { await api().edit_segment(idx, { [field]: value }); } catch (e) { reportError(e); }
  }, []);

  const regenSegment = useCallback(async (idx) => {
    setSegments(segs => segs.map(s => s.i === idx ? { ...s, status: 'gen' } : s));
    try { await api().regenerate_segment(idx); }
    catch (e) {
      reportError(e);
      setSegments(segs => segs.map(s => s.i === idx ? { ...s, status: 'ready' } : s));
    }
  }, []);

  const seekTo = useCallback((t) => {
    const v = videoRef.current;
    if (v) v.currentTime = t;
    setCurrentTime(t);
  }, []);

  // Ref-mirror segments so per-segment handlers can be reference-stable.
  // Without this, Segment memoization is defeated by a new handler
  // closure on every App render.
  const segmentsRef = useRef(segments);
  segmentsRef.current = segments;

  const previewSegment = useCallback((idx) => {
    const s = segmentsRef.current.find(seg => seg.i === idx);
    if (!s) return;
    seekTo(s.start);
    videoRef.current?.play().catch(() => {});
  }, [seekTo]);

  const jumpToSeg = useCallback((idx) => {
    setActiveSegIdx(idx);
    const s = segmentsRef.current.find(seg => seg.i === idx);
    if (s) seekTo(s.start);
  }, [seekTo]);

  const runDub = useCallback(async () => {
    setError(null);
    setLogs([]);
    setOutput(null);
    setRunState('running');
    setCancelling(false);
    setProgress(0.02);
    setLogsOpen(true);
    runStart.current = performance.now();
    segDoneTimes.current = [];
    setSegments(segs => segs.map(s => ({ ...s, status: s.edited ? 'edited' : 'ready' })));
    try {
      await api().run_dub({ small: modelSize === 'fast' });
    } catch (e) {
      reportError(e);
      setRunState('idle');
      setProgress(0);
    }
  }, [modelSize]);

  const cancelRun = useCallback(async () => {
    try { await api().cancel(); } catch {}
  }, []);

  const reveal = useCallback(async () => {
    if (!output) return;
    try { await api().reveal_in_finder(output.path); } catch {}
  }, [output]);

  // ── Render ──
  const envBanner = envIssues && (
    <div className="banner-warn">
      <Icon name="info" size={14} style={{ marginTop: 1, flexShrink: 0 }}/>
      <div>
        <b>Missing tools on PATH:</b>{' '}
        {envIssues.missing.map((t, i) => (
          <React.Fragment key={t}>
            {i > 0 && ', '}
            <code>{t}</code>
            {envIssues.hints[t] && <> — install: <code>{envIssues.hints[t]}</code></>}
          </React.Fragment>
        ))}
      </div>
    </div>
  );

  return (
    <div className="desktop">
      <div className="win">
        {envBanner}
        {!file ? (
          <EmptyState
            onPick={pickFile}
            onOpenRecent={openPath}
            recents={recents}
            ready={ready}
          />
        ) : (
          <>
            {error && (
              <div className="banner-error">
                <Icon name="info" size={14}/>
                <span>{error}</span>
                <button onClick={() => setError(null)}>Dismiss</button>
              </div>
            )}
            <div className="app">
              <div className="main">
                <Player
                  file={file}
                  videoUrl={videoUrl}
                  videoRef={videoRef}
                  playing={playing}
                  setPlaying={setPlaying}
                  currentTime={currentTime}
                  setCurrentTime={setCurrentTime}
                  duration={file.duration}
                  segments={segments}
                  audioTrack={audioTrack}
                  setAudioTrack={setAudioTrack}
                  hasDubbed={!!output}
                />
                <TranscriptHead
                  segments={segments}
                  sourceLang={sourceLang}
                  targetLang={targetLang}
                  translate={translate}
                  showOriginal={showOriginal}
                  setShowOriginal={setShowOriginal}
                  transcriptState={transcriptState}
                />
                <div className="transcript">
                  {transcriptState === 'awaiting' ? (
                    <div className="tsc-preflight">
                      <div className="tsc-preflight-icon">
                        <Icon name="caption" size={22} stroke={1.5} style={{ color: 'var(--accent)' }}/>
                      </div>
                      <div className="tsc-preflight-title">Ready to transcribe</div>
                      <div className="tsc-preflight-sub">
                        Confirm your source language
                        {translate ? ' and target language' : ''} on the right, then start transcription.
                      </div>
                      <div className="row" style={{ gap: 10, marginTop: 18 }}>
                        <div className="tsc-preflight-lang">
                          <span className="lang-flag" style={{ width: 22, height: 22, fontSize: 11 }}>
                            {LANGUAGES.find(l=>l.code===sourceLang)?.short || 'A'}
                          </span>
                          <span>{LANGUAGES.find(l=>l.code===sourceLang)?.label}</span>
                        </div>
                        {translate && (
                          <>
                            <Icon name="arrow" size={14} style={{ color: 'var(--text-faint)' }}/>
                            <div className="tsc-preflight-lang">
                              <span className="lang-flag" style={{ width: 22, height: 22, fontSize: 11 }}>
                                {LANGUAGES.find(l=>l.code===targetLang)?.short || 'En'}
                              </span>
                              <span>{LANGUAGES.find(l=>l.code===targetLang)?.label}</span>
                            </div>
                          </>
                        )}
                      </div>
                      <button className="btn primary lg" style={{ marginTop: 22 }} onClick={startTranscribe}>
                        <Icon name="sparkle" size={13}/> Transcribe{translate ? ' & translate' : ''}
                      </button>
                    </div>
                  ) : transcriptState === 'transcribing' || transcriptState === 'translating' ? (
                    <div className="tsc-skeleton">
                      {[...Array(6)].map((_, i) => (
                        <div key={i} className="tsc-skel-row" style={{ animationDelay: `${i * 0.12}s` }}>
                          <div className="tsc-skel-time"/>
                          <div className="tsc-skel-dot"/>
                          <div className="tsc-skel-lines">
                            <div className="tsc-skel-line" style={{ width: `${60 + (i*7)%30}%` }}/>
                            <div className="tsc-skel-line" style={{ width: `${35 + (i*11)%40}%` }}/>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    segments.map((s, i) => (
                      <Segment
                        key={s.i}
                        seg={s}
                        active={activeSegIdx === i}
                        onClick={jumpToSeg}
                        onEdit={editSegmentText}
                        onEditTime={editSegmentTime}
                        onRegen={regenSegment}
                        onPlay={previewSegment}
                        showOriginal={showOriginal}
                        translate={translate}
                        originalText={originalMap[s.i]}
                      />
                    ))
                  )}
                </div>
              </div>
              <SidePanel
                sourceLang={sourceLang} setSourceLang={setSourceLang}
                targetLang={targetLang} setTargetLang={setTargetLang}
                translate={translate} setTranslate={setTranslate}
                voiceMode={voiceMode} setVoiceMode={setVoiceMode}
                refVoiceName={refVoiceName}
                onUploadRef={pickRefAudio}
                refClip={refClip}
                refTranscript={refTranscript}
                setRefTranscript={setRefTranscript}
                onWhisperRef={whisperRef}
                refTranscribing={refTranscribing}
                canWhisperRef={
                  (voiceMode === 'upload' && !!refVoiceName) ||
                  (voiceMode === 'auto' && !!(refClip && refClip.url))
                }
                modelSize={modelSize} setModelSize={setModelSize}
                outputName={outputName}
                disabled={
                  runState === 'running' ||
                  transcriptState === 'transcribing' ||
                  transcriptState === 'translating' ||
                  refTranscribing
                }
              />
            </div>
            <Logs entries={logs} open={logsOpen}/>
            <RunBar
              state={runState}
              cancelling={cancelling}
              onRun={runDub}
              onCancel={cancelRun}
              onToggleLogs={() => setLogsOpen(o => !o)}
              logsOpen={logsOpen}
              progress={progress}
              stage={stage}
              etaSec={etaSec}
              transcriptReady={transcriptState === 'ready'}
              outputUrl={output?.url}
              onReveal={reveal}
            />
          </>
        )}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
