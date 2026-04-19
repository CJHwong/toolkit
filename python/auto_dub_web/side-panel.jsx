// Right-side job configuration panel.
// Voice-clip upload and ref-text changes are persisted to Python via
// window.pywebview.api so they're available when transcription starts.

function SidePanel(props) {
  const {
    sourceLang, setSourceLang, targetLang, setTargetLang, translate, setTranslate,
    voiceMode, setVoiceMode, refVoiceName, onUploadRef,
    refClip, refTranscript, setRefTranscript,
    onWhisperRef, refTranscribing, canWhisperRef,
    modelSize, setModelSize,
    outputName, disabled,
  } = props;

  // contentEditable divs don't resync with their children after mount —
  // sync imperatively whenever the underlying state changes (e.g. Whisper
  // just returned fresh reference-transcript text).
  const refTextEl = React.useRef(null);
  React.useEffect(() => {
    const el = refTextEl.current;
    if (el && el.innerText !== (refTranscript || '')) {
      el.innerText = refTranscript || '';
    }
  }, [refTranscript]);

  const refClipRange = refClip && refClip.end > refClip.start
    ? `~${(refClip.end - refClip.start).toFixed(1)}s clip from ${fmtShort(refClip.start)}`
    : 'Auto-extracted from transcript';

  return (
    <div className="side">
      <div className="side-head">
        <div className="side-title">Project</div>
        <div className="side-sub">Configure transcription, voice, and output</div>
      </div>

      <div className="side-body">
        {/* Languages */}
        <div className="group">
          <div className="group-title">Languages</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div className="lang-pair">
              <div className="lang-flag">{LANGUAGES.find(l=>l.code===sourceLang)?.short || 'A'}</div>
              <div className="grow col" style={{ gap: 2 }}>
                <div className="frow-hint" style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>Source</div>
                <select className="select" value={sourceLang} onChange={(e)=>setSourceLang(e.target.value)} disabled={disabled} style={{ height: 26, width: '100%' }}>
                  {LANGUAGES.map(l => <option key={l.code} value={l.code}>{l.label}</option>)}
                </select>
              </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'center', color: 'var(--text-faint)', opacity: translate ? 1 : 0.25 }}>
              <Icon name="chevd" size={14}/>
            </div>
            <div className="lang-pair" style={{ opacity: translate ? 1 : 0.45 }}>
              <div className="lang-flag">{translate ? (LANGUAGES.find(l=>l.code===targetLang)?.short || 'En') : '—'}</div>
              <div className="grow col" style={{ gap: 2 }}>
                <div className="frow-hint" style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>Target</div>
                <select
                  className="select"
                  value={targetLang}
                  disabled={!translate || disabled}
                  onChange={(e)=>setTargetLang(e.target.value)}
                  style={{ height: 26, width: '100%' }}
                >
                  {LANGUAGES.filter(l=>l.code!=='auto').map(l => <option key={l.code} value={l.code}>{l.label}</option>)}
                </select>
              </div>
            </div>
          </div>
          <div className="frow" style={{ marginTop: 8 }}>
            <div>
              <div className="frow-label">Translate</div>
              <div className="frow-hint">Use Claude to translate the transcript before dubbing</div>
            </div>
            <div className={`switch ${translate ? 'on' : ''} ${disabled ? 'disabled' : ''}`} onClick={() => !disabled && setTranslate(!translate)}/>
          </div>
        </div>

        {/* Voice */}
        <div className="group">
          <div className="group-title">Reference voice</div>
          <div className="seg" style={{ width: '100%', display: 'flex', marginBottom: 10 }}>
            <button className={voiceMode==='auto' ? 'on' : ''} onClick={()=>!disabled && setVoiceMode('auto')} disabled={disabled} style={{ flex: 1 }}>Auto-extract</button>
            <button className={voiceMode==='upload' ? 'on' : ''} onClick={()=>!disabled && setVoiceMode('upload')} disabled={disabled} style={{ flex: 1 }}>Upload clip</button>
          </div>
          {voiceMode === 'auto' ? (
            <div className="voice-card">
              <div className="voice-wave">
                {[5,10,18,12,22,16,26,14,20,8,14,22,10,16,6].map((h, i) => (
                  <span key={i} style={{ height: h }}/>
                ))}
              </div>
              <div className="grow">
                <div style={{ fontSize: 12.5, fontWeight: 500 }}>Original speaker</div>
                <div className="frow-hint">{refClipRange}</div>
              </div>
            </div>
          ) : (
            <div className="voice-card" style={{ cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.6 : 1 }} onClick={() => !disabled && onUploadRef && onUploadRef()}>
              <div className="voice-wave" style={{ background: 'var(--bg-sunken)' }}>
                <Icon name="upload" size={14} style={{ margin: 'auto', color: 'var(--text-faint)' }}/>
              </div>
              <div className="grow">
                <div style={{ fontSize: 12.5, fontWeight: 500 }}>{refVoiceName || 'Choose reference clip'}</div>
                <div className="frow-hint">.wav, .mp3 · 6–15s · clean speech</div>
              </div>
              <Icon name="chevr" size={14} style={{ color: 'var(--text-faint)' }}/>
            </div>
          )}

          <div style={{ marginTop: 12 }}>
            <div className="frow-label" style={{ fontSize: 11.5, marginBottom: 6 }}>
              Reference transcript
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 6 }}>
              <button
                className="btn sm ghost"
                style={{ gap: 5, padding: '3px 8px', fontSize: 11, whiteSpace: 'nowrap' }}
                disabled={!canWhisperRef || refTranscribing || disabled}
                onClick={() => onWhisperRef && onWhisperRef()}
                title={canWhisperRef
                  ? 'Run Whisper on the reference clip'
                  : 'Select or auto-extract a clip first'}
              >
                {refTranscribing ? (
                  <>
                    <span className="tsc-spin" style={{ width: 10, height: 10, borderWidth: 1.2 }}/>
                    Transcribing…
                  </>
                ) : (
                  <>
                    <Icon name="sparkle" size={11}/>
                    Transcribe with Whisper
                  </>
                )}
              </button>
            </div>
            <div className="ref-transcript-wrap">
              <div
                ref={refTextEl}
                className="ref-transcript"
                contentEditable={!disabled}
                suppressContentEditableWarning
                onBlur={(e) => setRefTranscript(e.currentTarget.innerText)}
                spellCheck={false}
                data-empty={!refTranscript}
                data-placeholder={voiceMode === 'auto' ? 'Auto-detecting from clip…' : 'Type what the speaker says in the clip'}
              >{refTranscript}</div>
            </div>
            <div className="frow-hint" style={{ marginTop: 6, lineHeight: 1.45 }}>
              Must match the clip word-for-word. Fix names, numbers, and punctuation before dubbing.
            </div>
          </div>
        </div>

        {/* Model */}
        <div className="group">
          <div className="group-title">Qwen3-TTS Model</div>
          <div className="seg" style={{ width: '100%', display: 'flex' }}>
            <button className={modelSize==='fast' ? 'on' : ''} onClick={()=>!disabled && setModelSize('fast')} disabled={disabled} style={{ flex: 1 }}>
              0.6B · Fast
            </button>
            <button className={modelSize==='quality' ? 'on' : ''} onClick={()=>!disabled && setModelSize('quality')} disabled={disabled} style={{ flex: 1 }}>
              1.7B · Quality
            </button>
          </div>
          <div className="frow-hint" style={{ marginTop: 8 }}>
            {modelSize === 'fast'
              ? 'Lighter, ~2× faster. Great for iteration.'
              : 'Higher fidelity, especially across languages.'}
          </div>
        </div>

        {/* Output */}
        <div className="group">
          <div className="group-title">Output</div>
          <div className="frow">
            <div>
              <div className="frow-label">Filename</div>
              <div className="frow-hint mono" style={{ fontSize: 11, marginTop: 3 }}>{outputName || '—'}</div>
            </div>
          </div>
          <div className="frow">
            <div style={{ minWidth: 0, flex: 1 }}>
              <div className="frow-label">Keep background music</div>
              <div className="frow-hint">Isolate vocals so music & SFX survive the dub</div>
            </div>
            <span className="pill-coming" style={{ whiteSpace: 'nowrap', flexShrink: 0 }}>Coming soon</span>
          </div>
        </div>
      </div>
    </div>
  );
}

window.SidePanel = SidePanel;
