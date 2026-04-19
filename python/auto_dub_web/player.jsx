// Video player, transcript header, and per-segment row.
// Wired to a real <video> element that reads from the embedded HTTP server.

function fmt(t) {
  if (!isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60).toString().padStart(2, '0');
  const s = Math.floor(t % 60).toString().padStart(2, '0');
  const cs = Math.floor((t % 1) * 100).toString().padStart(2, '0');
  return `${m}:${s}.${cs}`;
}
function fmtShort(t) {
  if (!isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60).toString().padStart(2, '0');
  const s = Math.floor(t % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

function Player({
  file, videoUrl, videoRef,
  playing, setPlaying,
  currentTime, setCurrentTime,
  duration, segments,
  audioTrack, setAudioTrack, hasDubbed,
}) {
  const activeSeg = segments.find(s => currentTime >= s.start && currentTime < s.end);

  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play().catch(() => {});
    else v.pause();
  };

  const nudge = (delta) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = Math.max(0, Math.min(duration || v.duration || 0, v.currentTime + delta));
  };

  const onScrub = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const p = (e.clientX - r.left) / r.width;
    const v = videoRef.current;
    const dur = duration || (v ? v.duration : 0);
    const target = Math.max(0, Math.min(dur, p * dur));
    if (v) v.currentTime = target;
    setCurrentTime(target);
  };

  return (
    <div className="player-wrap">
      <div className={`player ${videoUrl ? 'has-video' : ''}`}>
        {/* Static placeholder — hidden by .has-video when the video loads */}
        <div className="player-placeholder">
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14, position: 'relative', zIndex: 1 }}>
            <div style={{ width: 80, height: 80, borderRadius: '50%', background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.12)', display: 'grid', placeItems: 'center' }}>
              <Icon name={playing ? 'pause' : 'play'} size={32} stroke={1.4} style={{ color: 'rgba(255,255,255,0.85)' }}/>
            </div>
            <div className="mono" style={{ fontSize: 10, letterSpacing: '0.16em' }}>VIDEO PREVIEW</div>
          </div>
        </div>

        {videoUrl && (
          <video
            ref={videoRef}
            className="player-video"
            src={videoUrl}
            onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onEnded={() => setPlaying(false)}
            preload="metadata"
            playsInline
          />
        )}

        <div className="player-overlay-title">
          <span>{file.name}</span>
          {hasDubbed && <span className="badge"><span className="dot"/> {audioTrack === 'dubbed' ? 'Dubbed' : 'Original'}</span>}
        </div>

        {activeSeg && (
          <div style={{
            position: 'absolute', left: 0, right: 0, bottom: 92,
            display: 'flex', justifyContent: 'center', pointerEvents: 'none',
            zIndex: 2,
          }}>
            <div style={{
              background: 'rgba(0,0,0,0.72)', backdropFilter: 'blur(8px)',
              color: 'white', padding: '6px 14px', borderRadius: 6,
              fontSize: 16, fontWeight: 500, letterSpacing: '-0.005em',
              maxWidth: '70%', textAlign: 'center',
            }}>
              {activeSeg.text}
            </div>
          </div>
        )}

        <div className="pctl">
          <div className="scrubber" onClick={onScrub}>
            <div className="scrubber-buffered" style={{ width: '100%' }}/>
            <div className="scrubber-played" style={{ width: `${duration ? (currentTime/duration)*100 : 0}%` }}/>
            <div className="scrubber-marks">
              {segments.map(s => (
                <div key={s.i} className="scrubber-mark" style={{ left: `${duration ? (s.start/duration)*100 : 0}%` }}/>
              ))}
            </div>
          </div>
          <div className="pctl-row">
            <button className="pctl-btn" onClick={() => nudge(-5)}><Icon name="back" size={14}/></button>
            <button className="pctl-btn big" onClick={togglePlay}>
              <Icon name={playing ? 'pause' : 'play'} size={18}/>
            </button>
            <button className="pctl-btn" onClick={() => nudge(5)}><Icon name="forward" size={14}/></button>
            <div style={{ minWidth: 76, whiteSpace: 'nowrap' }}>{fmt(currentTime)}</div>
            <div style={{ opacity: 0.55, whiteSpace: 'nowrap' }}>/ {fmt(duration || 0)}</div>
            <div style={{ flex: 1 }}/>
            {hasDubbed && (
              <div className="track-sel">
                <button className={audioTrack==='original'?'on':''} onClick={()=>setAudioTrack('original')}>Original</button>
                <button className={audioTrack==='dubbed'?'on':''} onClick={()=>setAudioTrack('dubbed')}>Dubbed</button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function TranscriptHead({ segments, sourceLang, targetLang, translate, showOriginal, setShowOriginal, transcriptState }) {
  const total = segments.length;
  const done = segments.filter(s => s.status === 'done').length;
  const edited = segments.filter(s => s.edited).length;
  const busy = transcriptState === 'transcribing' || transcriptState === 'translating';
  const label = transcriptState === 'transcribing' ? 'Transcribing with Whisper…'
              : transcriptState === 'translating'  ? `Translating to ${targetLang}…`
              : null;
  return (
    <div className="transcript-head">
      <div>
        <div className="tsc-title">Transcript & Voice Track</div>
        <div className="tsc-meta">
          {busy ? (
            <span className="row" style={{ gap: 6, color: 'var(--accent)' }}>
              <span className="tsc-spin"/>
              {label}
            </span>
          ) : (
            <>
              {total} segments
              {done > 0 && ` · ${done} rendered`}
              {edited > 0 && ` · ${edited} edited (re-render to update voice)`}
              {translate && ` · ${sourceLang} → ${targetLang}`}
            </>
          )}
        </div>
      </div>
      <div className="grow"/>
      {translate && !busy && (
        <div className="row" style={{ gap: 8 }}>
          <span className="frow-label" style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-muted)', textTransform: 'none', letterSpacing: 0 }}>Show original</span>
          <div className={`switch ${showOriginal ? 'on' : ''}`} onClick={() => setShowOriginal(!showOriginal)}/>
        </div>
      )}
    </div>
  );
}

function parseTime(str) {
  // Accepts MM:SS.CS, MM:SS, SS.S, or raw seconds
  const s = (str || '').trim();
  if (!s) return null;
  const mmss = s.match(/^(\d{1,2}):(\d{1,2})(?:\.(\d{1,3}))?$/);
  if (mmss) {
    const m = parseInt(mmss[1], 10);
    const sec = parseInt(mmss[2], 10);
    const cs = mmss[3] ? parseFloat('0.' + mmss[3]) : 0;
    return m * 60 + sec + cs;
  }
  const num = parseFloat(s);
  return isNaN(num) ? null : num;
}

// Handlers receive `seg.i` as first arg so App can pass stable callbacks
// and memoization isn't defeated by a new closure on every render.
function _Segment({ seg, active, onClick, onEdit, onEditTime, onRegen, onPlay, showOriginal, translate, originalText }) {
  const dur = seg.end - seg.start;
  const status = seg.status || 'ready';
  const statusIcon = {
    ready:  null,
    done:   <Icon name="check" size={11} stroke={2.2}/>,
    gen:    <svg width="10" height="10" viewBox="0 0 10 10"><circle cx="5" cy="5" r="3" fill="none" stroke="white" strokeWidth="1.4" strokeDasharray="12 4"/></svg>,
    err:    <Icon name="x" size={10} stroke={2.4}/>,
    edited: <Icon name="edit" size={9} stroke={2.2}/>,
  }[status];
  const statusLabel = {
    ready:  'Not yet rendered',
    done:   'Voice rendered · up to date',
    gen:    'Generating voice…',
    err:    'Render failed — check logs',
    edited: 'Text edited · voice out of date (regenerate to update)',
  }[status];

  const commitTime = (field, text) => {
    const parsed = parseTime(text);
    if (parsed == null) return;
    // Blur fires even with no change; skip no-ops so we don't flag "edited".
    if (Math.abs(parsed - (seg[field] || 0)) < 0.005) return;
    onEditTime(seg.i, field, parsed);
  };

  const commitText = (el) => {
    const newText = (el.innerText || '').trim();
    if (newText === (seg.text || '').trim()) return;
    onEdit(seg.i, newText);
  };

  return (
    <div className={`seg ${active ? 'active' : ''}`} onClick={() => onClick(seg.i)}>
      <div className="seg-time" onClick={(e) => e.stopPropagation()}>
        <span
          className="seg-time-field"
          contentEditable
          suppressContentEditableWarning
          spellCheck={false}
          title="Start time · MM:SS.CS"
          onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); e.currentTarget.blur(); } }}
          onBlur={(e) => commitTime('start', e.currentTarget.innerText)}
        >{fmtShort(seg.start)}</span>
        <span className="seg-time-sep">→</span>
        <span
          className="seg-time-field"
          contentEditable
          suppressContentEditableWarning
          spellCheck={false}
          title="End time · MM:SS.CS"
          onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); e.currentTarget.blur(); } }}
          onBlur={(e) => commitTime('end', e.currentTarget.innerText)}
        >{fmtShort(seg.end)}</span>
        <span className="dur">{dur.toFixed(1)}s</span>
      </div>
      <div className={`seg-status ${status}`} title={statusLabel}>{statusIcon}</div>
      <div>
        <div
          className="seg-text"
          contentEditable
          suppressContentEditableWarning
          onBlur={(e) => commitText(e.currentTarget)}
          spellCheck={false}
        >{seg.text}</div>
        {translate && showOriginal && originalText && (
          <div className="seg-text translated" style={{ marginTop: 4, fontStyle: 'italic', color: 'var(--text-faint)', fontSize: 12.5 }}>
            {originalText}
          </div>
        )}
      </div>
      <div className="seg-actions">
        <button className="seg-action" title="Play from here" onClick={(e)=>{e.stopPropagation(); e.currentTarget.blur(); onPlay(seg.i);}}>
          <Icon name="play-s" size={12}/>
        </button>
        <button className="seg-action" title="Regenerate voice for this segment" onClick={(e)=>{e.stopPropagation(); e.currentTarget.blur(); onRegen(seg.i);}}>
          <Icon name="refresh" size={12}/>
        </button>
      </div>
    </div>
  );
}
// onTimeUpdate fires ~4×/sec during playback. Without memoization this
// reconciles every row in the transcript per tick; with it, only rows
// whose props actually change (status, active, etc.) re-render.
const Segment = React.memo(_Segment);

window.Player = Player;
window.TranscriptHead = TranscriptHead;
window.Segment = Segment;
window.fmt = fmt;
window.fmtShort = fmtShort;
