"""
voice_input.py — Aura AI | Voice Input Component (v8.0)
========================================================
v8.0 changes (Browser STT only — Whisper AI tab unchanged):
  • Browser STT bridge rebuilt with triple-layer data persistence:
      The old single querySelector bridge lost the transcript on Streamlit
      reruns (React resets hidden inputs before Python reads them).
      v8.0 fixes this with three layers:
        A — window.__bsttWrite__() injected into the parent frame writes
            into stable per-question session_state keys (_bstt_last_tx_{q},
            _bstt_last_audio_{q}) that are NEVER reset by reruns.
        B — Legacy querySelector on hidden st.text_input (original approach)
            kept as fallback when Method A is unavailable.
        C — sessionStorage as last-resort, belt-and-suspenders backup.
      Python reads with priority: stable key → bridge input → last known value.
  • New helpers: _inject_bstt_bridge_script(), _build_browser_stt_html()
    now accepts stable_tx_key / stable_audio_key parameters.
  • bridge_status indicator shows which method(s) succeeded (A/B/C).

v7.0 carried forward (unchanged):
  • Live Filler + WPM HUD (Yoodli-style real-time coaching overlay)
  • whisper_post_hud() post-submission summary card in Whisper tab
  • _FILLER_WORDS_JS mirrors FILLER_WORDS in answer_evaluator.py
  • Whisper AI tab: st.audio_input → Whisper → text (completely unchanged)

Session state keys written:
  last_audio_bytes        bytes — raw audio for UnifiedVoicePipeline
  last_audio_source       str   — "whisper_mic" | "browser_stt"
  transcribed_text        str   — cached transcript
  last_audio_id           int   — dedup guard for Whisper tab
  _bstt_last_tx_{q}       str   — stable per-question Browser STT transcript
  _bstt_last_audio_{q}    str   — stable per-question base64 audio
"""

from __future__ import annotations

import base64
import streamlit as st
import streamlit.components.v1 as components
from typing import Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE FILLER + WPM HUD  (Yoodli-style real-time coaching overlay)
# ══════════════════════════════════════════════════════════════════════════════

# Filler words mirror the list in answer_evaluator.py — keep in sync.
_FILLER_WORDS_JS = (
    '"um","uh","like","basically","actually","you know","right","so",'
    '"just","kind of","sort of","i mean","literally","honestly",'
    '"obviously","clearly","simply","really","very","quite",'
    '"pretty much","i guess","stuff","things"'
)

def _build_live_hud_html(hud_id: str) -> str:
    """
    Returns a self-contained HTML+JS overlay that:
      • reads window._auraLiveTranscript (updated by the sibling STT widget)
      • updates every 3 seconds while recording is active
      • shows: live filler count, filler ratio badge, WPM bar, pause coaching tip
      • resets when a new recording starts

    The sibling widget (_build_browser_stt_html) must call:
        window._auraLiveTranscript = currentText;
        window._auraRecording      = true / false;
        window._auraStartTime      = Date.now();  // on recording start
    to feed data into this HUD.
    """
    return f"""
<style>
  *{{box-sizing:border-box;font-family:'DM Sans',Inter,sans-serif;margin:0;padding:0;}}
  body{{background:transparent;}}

  #hud_{hud_id}{{
    background:rgba(10,16,40,.95);
    border:1px solid rgba(124,107,255,.3);
    border-radius:12px;
    padding:10px 14px;
    display:flex;
    flex-direction:column;
    gap:8px;
  }}

  .hud-title{{
    font-size:.65rem;font-weight:700;letter-spacing:.08em;
    color:#5a7098;text-transform:uppercase;margin-bottom:2px;
  }}

  /* Three metric columns */
  .hud-row{{display:flex;gap:8px;align-items:stretch;}}

  .hud-card{{
    flex:1;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);
    border-radius:8px;padding:8px 10px;display:flex;flex-direction:column;gap:4px;
    min-width:0;
  }}
  .hud-card-label{{font-size:.58rem;color:#5a7098;font-weight:600;letter-spacing:.06em;text-transform:uppercase;}}
  .hud-card-value{{font-size:1.35rem;font-weight:800;color:#e2ecf9;line-height:1;}}
  .hud-card-sub{{font-size:.6rem;color:#7c6bff;}}

  /* Filler badge colour logic */
  .badge-ok  {{color:#22d87a;}}
  .badge-warn{{color:#fbbf24;}}
  .badge-bad {{color:#ff5c5c;animation:pulse-hud .9s infinite;}}
  @keyframes pulse-hud{{0%,100%{{opacity:1;}}50%{{opacity:.55;}}}}

  /* WPM bar */
  .wpm-bar-wrap{{
    width:100%;height:7px;border-radius:4px;
    background:rgba(255,255,255,.07);overflow:hidden;
  }}
  .wpm-bar-fill{{
    height:100%;border-radius:4px;transition:width .6s ease;
    background:linear-gradient(90deg,#7c6bff,#3b8bff);
  }}
  .wpm-bar-fill.fast{{background:linear-gradient(90deg,#f97316,#ef4444);}}
  .wpm-bar-fill.slow{{background:linear-gradient(90deg,#7c6bff,#818cf8);}}

  /* Coaching tip */
  .hud-tip{{
    font-size:.67rem;color:#8fc4e0;background:rgba(124,107,255,.08);
    border-left:2px solid rgba(124,107,255,.5);
    border-radius:0 6px 6px 0;padding:5px 8px;min-height:22px;
    transition:all .4s;
  }}

  /* Live dot */
  .live-dot{{
    width:6px;height:6px;border-radius:50%;
    background:#ff5c5c;display:inline-block;margin-right:4px;
    animation:pulse-hud .9s infinite;
  }}
  .idle-dot{{background:#3b4060;animation:none;}}
</style>

<div id="hud_{hud_id}">
  <div class="hud-title">
    <span id="liveDot_{hud_id}" class="live-dot idle-dot"></span>
    Live Speaking Coach
  </div>

  <div class="hud-row">
    <!-- Filler count -->
    <div class="hud-card">
      <div class="hud-card-label">Fillers</div>
      <div id="fillerVal_{hud_id}" class="hud-card-value badge-ok">0</div>
      <div id="fillerSub_{hud_id}" class="hud-card-sub">of 0 words</div>
    </div>

    <!-- WPM -->
    <div class="hud-card">
      <div class="hud-card-label">WPM</div>
      <div id="wpmVal_{hud_id}" class="hud-card-value">—</div>
      <div class="wpm-bar-wrap" style="margin-top:4px;">
        <div id="wpmBar_{hud_id}" class="wpm-bar-fill" style="width:0%;"></div>
      </div>
    </div>

    <!-- Duration -->
    <div class="hud-card">
      <div class="hud-card-label">Duration</div>
      <div id="durVal_{hud_id}" class="hud-card-value">0s</div>
      <div id="durSub_{hud_id}" class="hud-card-sub">aim: 60–150s</div>
    </div>
  </div>

  <!-- Coaching tip row -->
  <div id="hudTip_{hud_id}" class="hud-tip">
    💡 Start recording — coaching tips appear here as you speak
  </div>
</div>

<script>
(function(){{
  const FILLERS = [{_FILLER_WORDS_JS}];
  const HUD_INTERVAL_MS = 3000;

  // ── helpers ───────────────────────────────────────────────────────────────
  function countFillers(text) {{
    if (!text) return 0;
    let t = ' ' + text.toLowerCase() + ' ';
    let n = 0;
    FILLERS.forEach(f => {{
      const re = new RegExp('\\\\s' + f.replace(/[.*+?^${{}}()|[\\]\\\\]/g,'\\\\$&') + '\\\\s', 'gi');
      const m  = t.match(re);
      if (m) n += m.length;
    }});
    return n;
  }}

  function wordCount(text) {{
    if (!text || !text.trim()) return 0;
    return text.trim().split(/\\s+/).length;
  }}

  function elapsedSeconds() {{
    const t0 = window._auraStartTime || 0;
    return t0 ? ((Date.now() - t0) / 1000) : 0;
  }}

  function pickTip(wpm, fillerRatio, elapsed) {{
    if (elapsed < 5) return '💡 Speak your answer — coaching updates every 3s';
    if (fillerRatio > 0.12) return '⚠️ High filler rate — try pausing silently instead of saying "um" or "uh"';
    if (fillerRatio > 0.07) return '🟡 A few fillers detected — slow down slightly and breathe between thoughts';
    if (wpm > 180)          return '🐇 You\'re speaking fast — slow to ~140 WPM for clarity';
    if (wpm < 90 && wpm > 0)return '🐢 Speaking slowly — a bit more pace will signal confidence';
    if (elapsed > 150)      return '⏱ You\'re running long — consider wrapping up with your key result';
    if (elapsed > 60 && wpm >= 90 && fillerRatio <= 0.07)
                            return '✅ Great pace and clarity — keep going!';
    return '💡 Pause well: a 1s breath beats "um" every time';
  }}

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const fillerVal = document.getElementById('fillerVal_{hud_id}');
  const fillerSub = document.getElementById('fillerSub_{hud_id}');
  const wpmVal    = document.getElementById('wpmVal_{hud_id}');
  const wpmBar    = document.getElementById('wpmBar_{hud_id}');
  const durVal    = document.getElementById('durVal_{hud_id}');
  const durSub    = document.getElementById('durSub_{hud_id}');
  const hudTip    = document.getElementById('hudTip_{hud_id}');
  const liveDot   = document.getElementById('liveDot_{hud_id}');

  // ── update loop ───────────────────────────────────────────────────────────
  function update() {{
    const recording = !!window._auraRecording;
    const text      = window._auraLiveTranscript || '';
    const elapsed   = elapsedSeconds();

    // Live dot
    liveDot.className = recording ? 'live-dot' : 'live-dot idle-dot';

    if (!recording && !text) return;  // idle with no data — skip render

    const words  = wordCount(text);
    const wpm    = elapsed > 3 ? Math.round((words / elapsed) * 60) : 0;
    const fc     = countFillers(text);
    const ratio  = words > 0 ? fc / words : 0;

    // ── Filler count ──────────────────────────────────────────────────────
    fillerVal.textContent = fc;
    fillerSub.textContent = 'of ' + words + ' words';
    fillerVal.className   = 'hud-card-value ' +
      (ratio > 0.12 ? 'badge-bad' : ratio > 0.07 ? 'badge-warn' : 'badge-ok');

    // ── WPM ───────────────────────────────────────────────────────────────
    if (wpm > 0) {{
      wpmVal.textContent = wpm;
      // bar: 80 WPM → 30%, 140 → 60% (ideal), 200 → 100%
      const pct = Math.min(100, Math.max(0, ((wpm - 60) / 160) * 100));
      wpmBar.style.width = pct + '%';
      wpmBar.className   = 'wpm-bar-fill' + (wpm > 180 ? ' fast' : wpm < 90 ? ' slow' : '');
    }} else {{
      wpmVal.textContent = '—';
    }}

    // ── Duration ──────────────────────────────────────────────────────────
    if (elapsed > 0) {{
      durVal.textContent = Math.round(elapsed) + 's';
      durSub.textContent = elapsed < 60 ? 'keep going…'
                         : elapsed < 150 ? '✅ ideal range'
                         : '⏱ wrapping up?';
    }}

    // ── Coaching tip ──────────────────────────────────────────────────────
    hudTip.textContent = pickTip(wpm, ratio, elapsed);
  }}

  setInterval(update, HUD_INTERVAL_MS);
  update();  // immediate first paint
}})();
</script>
"""


def whisper_post_hud(answer_text: str) -> None:
    """
    Post-submission HUD for the Whisper tab.
    Shown after transcription completes — gives filler + WPM summary
    based on the final transcript (no live update needed for Whisper
    since it transcribes after recording, not during).
    Called from whisper_audio_input() after a successful transcription.
    """
    if not answer_text or answer_text.startswith("["):
        return

    words = answer_text.strip().split()
    n = len(words)
    if n < 3:
        return

    text_lower = " " + answer_text.lower() + " "
    from answer_evaluator import FILLER_WORDS as _EV_FILLERS
    fc = sum(
        len(__import__("re").findall(
            r"\s" + __import__("re").escape(f) + r"\s", text_lower
        ))
        for f in _EV_FILLERS
    )
    ratio = fc / n if n else 0

    # Colour thresholds
    filler_color = "#22d87a" if ratio <= 0.07 else ("#fbbf24" if ratio <= 0.12 else "#ff5c5c")
    filler_label = "✅ Low" if ratio <= 0.07 else ("⚠️ Moderate" if ratio <= 0.12 else "❌ High")

    tip = (
        "Great — minimal filler words detected." if ratio <= 0.07
        else "Try pausing silently instead of using filler words."
        if ratio <= 0.12
        else "High filler rate — practice pausing and breathing between thoughts."
    )

    st.markdown(
        f"""
        <div style="background:rgba(10,16,40,.9);border:1px solid rgba(124,107,255,.25);
                    border-radius:10px;padding:10px 14px;margin-top:8px;display:flex;
                    gap:16px;align-items:center;flex-wrap:wrap;">
          <div style="text-align:center;min-width:60px;">
            <div style="font-size:.58rem;color:#5a7098;text-transform:uppercase;
                        letter-spacing:.06em;font-weight:600;">Fillers</div>
            <div style="font-size:1.3rem;font-weight:800;color:{filler_color};">{fc}</div>
            <div style="font-size:.6rem;color:#5a7098;">{filler_label}</div>
          </div>
          <div style="text-align:center;min-width:60px;">
            <div style="font-size:.58rem;color:#5a7098;text-transform:uppercase;
                        letter-spacing:.06em;font-weight:600;">Words</div>
            <div style="font-size:1.3rem;font-weight:800;color:#e2ecf9;">{n}</div>
            <div style="font-size:.6rem;color:#5a7098;">total</div>
          </div>
          <div style="font-size:.67rem;color:#8fc4e0;background:rgba(124,107,255,.08);
                      border-left:2px solid rgba(124,107,255,.5);border-radius:0 6px 6px 0;
                      padding:5px 8px;flex:1;min-width:120px;">
            💡 {tip}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC PANEL
# ══════════════════════════════════════════════════════════════════════════════

def render_mic_diagnostic(stt) -> None:
    """Show mic status and package availability."""
    from speech_to_text import SF_OK, PYDUB_OK, LIBROSA_OK, SCIPY_OK, TORCH_OK, TRANSFORMERS_OK

    with st.expander("🔧 Microphone & STT Diagnostics", expanded=False):
        st.markdown(
            '<div style="color:#ffffff;font-size:.82rem;font-weight:600;margin-bottom:.4rem;">Package Status</div>',
            unsafe_allow_html=True,
        )
        checks = [
            ("transformers", TRANSFORMERS_OK, "pip install transformers"),
            ("torch",        TORCH_OK,        "pip install torch"),
            ("soundfile",    SF_OK,           "pip install soundfile"),
            ("pydub",        PYDUB_OK,        "pip install pydub  +  install ffmpeg"),
            ("librosa",      LIBROSA_OK,      "pip install librosa"),
            ("scipy",        SCIPY_OK,        "pip install scipy"),
        ]
        for name, ok, fix in checks:
            icon  = "✅" if ok else "❌"
            color = "#6ee7b7" if ok else "#fca5a5"
            note  = (
                ""
                if ok
                else (
                    f'&nbsp;→&nbsp;<code style="color:#fcd34d;background:rgba(255,255,255,.07);'
                    f'padding:1px 5px;border-radius:3px;">{fix}</code>'
                )
            )
            st.markdown(
                f'<div style="font-size:.8rem;color:{color};margin-bottom:3px;">'
                f'{icon} <strong style="color:{color};">{name}</strong>{note}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr style="border-color:rgba(255,255,255,.08);margin:.5rem 0;">', unsafe_allow_html=True)
        st.markdown(
            '<div style="color:#ffffff;font-size:.82rem;font-weight:600;margin-bottom:.3rem;">Whisper Model</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div style="color:#8fc4e0;font-size:.78rem;">{stt.status}</div>', unsafe_allow_html=True)

        st.markdown('<hr style="border-color:rgba(255,255,255,.08);margin:.5rem 0;">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:.78rem;color:#b0cce8;line-height:1.75;">'
            '<strong style="color:#c8e0f4;">Common fixes:</strong><br>'
            '• Microphone needs <strong>HTTPS</strong> or <strong>localhost</strong><br>'
            '• Chrome/Edge required for Browser STT tab<br>'
            '• If audio records but transcription is empty: install <code style="color:#fcd34d;">pydub</code> + <code style="color:#fcd34d;">ffmpeg</code><br>'
            '• Allow mic in browser: check address bar lock icon<br>'
            '• Test mic: open <code style="color:#fcd34d;">chrome://settings/content/microphone</code>'
            '</div>',
            unsafe_allow_html=True,
        )

        if st.button("🧪 Test with synthetic audio", key="diag_test"):
            import numpy as np, io, wave
            sr = 16000; dur = 1
            tone = (np.sin(2 * np.pi * 440 * np.arange(sr * dur) / sr) * 32767).astype(np.int16)
            buf  = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                wf.writeframes(tone.tobytes())
            result = stt.transcribe(buf.getvalue())
            st.info(f"Synthetic audio test result: `{result}`")


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 1 — Whisper via st.audio_input
# ══════════════════════════════════════════════════════════════════════════════

def _build_whisper_mic_html(component_key: str) -> str:
    """
    Decorative mic UI for the Whisper tab — mirrors the Browser STT widget
    (same mic button, spinning ring, pulse rings, waveform bars, timer).

    Clicking the mic or "START RECORDING" button calls triggerWhisper(), which
    walks into the parent Streamlit frame via window.parent.document and clicks
    the real st.audio_input <button> by its aria-label / data-testid.
    The native st.audio_input still owns all actual recording and byte capture.
    """
    return f"""
<style>
  *{{box-sizing:border-box;font-family:'Inter',sans-serif;}}
  body{{margin:0;padding:0;background:transparent;overflow:hidden;}}
  #wrap{{background:rgba(5,12,32,0.96);border-radius:12px;padding:14px 14px 10px;}}

  #status{{
    font-size:.68rem;padding:4px 12px;border-radius:20px;margin-bottom:8px;
    background:rgba(255,255,255,.05);color:rgba(90,112,152,.8);
    text-align:center;transition:all .3s;
    font-family:'JetBrains Mono',monospace;letter-spacing:1px;
  }}
  #status.listening{{background:rgba(244,63,94,.12);color:#f43f5e;animation:sPulse 1.2s infinite;}}
  #status.done{{background:rgba(124,107,255,.1);color:#a78bfa;}}
  @keyframes sPulse{{0%,100%{{opacity:1}}50%{{opacity:.5}}}}

  .mic-row{{display:flex;align-items:center;gap:16px;margin-bottom:10px;}}

  .mic-outer{{position:relative;width:72px;height:72px;flex-shrink:0;}}
  .mic-ring{{
    position:absolute;inset:-10px;border-radius:50%;border:1.5px solid transparent;
    background:conic-gradient(from 0deg,transparent,#7c6bff 40%,#00ff88 70%,transparent) border-box;
    -webkit-mask:linear-gradient(#fff 0 0) padding-box,linear-gradient(#fff 0 0);
    -webkit-mask-composite:destination-out;mask-composite:exclude;
    animation:micSpin 5s linear infinite;opacity:.55;
  }}
  @keyframes micSpin{{from{{transform:rotate(0)}}to{{transform:rotate(360deg)}}}}

  .pulse-ring{{
    position:absolute;top:50%;left:50%;border-radius:50%;
    border:1.5px solid rgba(244,63,94,.5);
    transform:translate(-50%,-50%);opacity:0;pointer-events:none;
  }}
  .pr1,.pr2,.pr3{{width:72px;height:72px;}}
  .pr2{{animation-delay:.5s;}}
  .pr3{{animation-delay:1s;}}
  .recording .pulse-ring{{animation:ringExpand 1.8s ease-out infinite;}}
  @keyframes ringExpand{{
    from{{width:72px;height:72px;opacity:.7;}}
    to{{width:130px;height:130px;opacity:0;}}
  }}

  #micBtn{{
    width:72px;height:72px;border-radius:50%;
    background:rgba(124,107,255,.07);
    border:1.5px solid rgba(124,107,255,.35);
    display:flex;align-items:center;justify-content:center;
    cursor:pointer;font-size:26px;transition:all .25s;
    position:relative;z-index:1;
    box-shadow:0 0 20px rgba(124,107,255,.1);
  }}
  #micBtn:hover{{
    background:rgba(124,107,255,.14);
    border-color:rgba(124,107,255,.7);
    box-shadow:0 0 28px rgba(124,107,255,.25);
  }}
  #micBtn.rec{{
    background:rgba(244,63,94,.12);
    border-color:rgba(244,63,94,.55);
    animation:recPulse 1.4s ease-in-out infinite;
  }}
  @keyframes recPulse{{
    0%,100%{{box-shadow:0 0 0 0 rgba(244,63,94,.45);}}
    50%{{box-shadow:0 0 0 14px rgba(244,63,94,0);}}
  }}

  #waveform{{
    display:flex;align-items:center;gap:3px;height:44px;flex:1;
    background:rgba(2,7,20,.7);
    border:1px solid rgba(124,107,255,.08);
    border-radius:10px;padding:6px 10px;
  }}
  .wb{{width:3px;border-radius:2px;background:rgba(124,107,255,.3);height:3px;flex-shrink:0;}}
  .wb.live{{
    background:rgba(124,107,255,.85);
    animation:wbAnim var(--d,.55s) ease-in-out infinite alternate;
  }}
  @keyframes wbAnim{{from{{height:3px}}to{{height:var(--h,28px)}}}}

  .rec-meta{{
    display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;
  }}
  #hint{{
    font-family:'JetBrains Mono',monospace;font-size:10px;
    color:rgba(124,107,255,.5);letter-spacing:1.5px;
  }}
  #eq-timer{{
    font-family:'JetBrains Mono',monospace;font-size:11px;
    color:#f43f5e;letter-spacing:2px;display:none;
  }}

  .btn-row{{display:flex;gap:6px;}}
  #recBtnFull{{
    flex:1;border:1px solid rgba(124,107,255,.3);border-radius:10px;padding:9px;
    background:rgba(124,107,255,.06);color:#a78bfa;
    font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;
    letter-spacing:1.5px;cursor:pointer;transition:all .25s;
  }}
  #recBtnFull:hover{{background:rgba(124,107,255,.12);border-color:rgba(124,107,255,.6);}}
  #recBtnFull.rec{{
    background:rgba(244,63,94,.1);border-color:rgba(244,63,94,.5);color:#f43f5e;
  }}
  #whisperBadge{{
    font-size:.62rem;text-align:center;margin-top:5px;
    color:rgba(124,107,255,.55);font-family:'JetBrains Mono',monospace;min-height:14px;
  }}
</style>

<div id="wrap">
  <div id="status">CLICK MIC TO START RECORDING</div>

  <div class="mic-row">
    <div class="mic-outer" id="micOuter">
      <div class="mic-ring"></div>
      <div class="pulse-ring pr1"></div>
      <div class="pulse-ring pr2"></div>
      <div class="pulse-ring pr3"></div>
      <div id="micBtn" onclick="triggerWhisper()">🎤</div>
    </div>
    <div id="waveform"></div>
  </div>

  <div class="rec-meta">
    <span id="hint">Click mic · speak · click again to stop</span>
    <span id="eq-timer">00:00</span>
  </div>

  <div class="btn-row">
    <button id="recBtnFull" onclick="triggerWhisper()">▶ START RECORDING</button>
  </div>
  <div id="whisperBadge">⚡ powered by Whisper AI — transcription runs after you stop</div>
</div>

<script>
(function(){{
  // ── Build waveform bars ───────────────────────────────────────────────────
  var wf = document.getElementById('waveform');
  var wbEls = [];
  for (var i = 0; i < 40; i++) {{
    var b = document.createElement('div');
    b.className = 'wb';
    b.style.setProperty('--h', (4 + Math.random()*32) + 'px');
    b.style.setProperty('--d', (0.28 + Math.random()*.55) + 's');
    wf.appendChild(b);
    wbEls.push(b);
  }}

  var running = false, eqSec = 0, eqInterval = null;

  function setRunning(on) {{
    running = on;
    document.getElementById('micBtn').className        = on ? 'rec' : '';
    document.getElementById('micOuter').className      = on ? 'mic-outer recording' : 'mic-outer';
    document.getElementById('recBtnFull').className    = on ? 'rec' : '';
    document.getElementById('recBtnFull').textContent  = on ? '■ STOP RECORDING' : '▶ START RECORDING';
    var st = document.getElementById('status');
    st.className   = on ? 'listening' : 'done';
    st.textContent = on ? 'RECORDING — WHISPER TRANSCRIBES ON STOP'
                        : 'PROCESSING WITH WHISPER AI…';
    document.getElementById('hint').textContent = on
      ? 'Speak now · click again to stop'
      : 'Transcription in progress…';
    wbEls.forEach(function(b) {{ b.className = 'wb' + (on ? ' live' : ''); }});
    if (on) {{
      eqSec = 0;
      document.getElementById('eq-timer').style.display = 'inline';
      clearInterval(eqInterval);
      eqInterval = setInterval(function() {{
        eqSec++;
        var m = Math.floor(eqSec/60), s = eqSec%60;
        document.getElementById('eq-timer').textContent =
          (m<10?'0':'')+m+':'+(s<10?'0':'')+s;
        wbEls.forEach(function(b) {{
          b.style.setProperty('--h', (4+Math.random()*36)+'px');
        }});
      }}, 180);
    }} else {{
      clearInterval(eqInterval);
      document.getElementById('eq-timer').style.display = 'none';
    }}
  }}

  function triggerWhisper() {{
    // Locate Streamlit's native st.audio_input button in the parent frame
    // and click it — try multiple selectors for resilience across ST versions.
    var pd = window.parent.document;
    var btn = pd.querySelector('[data-testid="stAudioInput"] button')
           || pd.querySelector('button[aria-label="Record"]')
           || pd.querySelector('button[title="Record"]');
    if (btn) {{
      btn.click();
      setRunning(!running);
    }} else {{
      document.getElementById('hint').textContent =
        'Use the Streamlit mic widget below ↓';
    }}
  }}

  window.triggerWhisper = triggerWhisper;
}})();
</script>
"""


def whisper_audio_input(stt, q_number: int) -> str:
    """
    Streamlit native audio recorder → Whisper transcription.
    Raw bytes are stored in st.session_state["last_audio_bytes"] so
    submit_answer() can pass them to the voice nervousness pipeline.
    Returns the transcribed text (or empty string).
    """
    stt_ok = getattr(stt, "ready", False)

    if not stt_ok:
        st.markdown(
            f'<div style="font-size:.8rem;color:#fca5a5;">❌ Whisper not available: {stt.error}<br>'
            f'Run: <code style="color:#fcd34d;">pip install transformers torch soundfile</code></div>',
            unsafe_allow_html=True,
        )

    # ── Styled mic widget (mirrors Browser STT design) ───────────────────────
    components.html(
        _build_whisper_mic_html(f"whisper_{q_number}"),
        height=190,
    )

    # ── Native Streamlit recorder (hidden; JS clicks it via parent DOM) ──────
    # Suppress the native widget UI — it shows its own timer and mic icon
    # which duplicates our custom widget above. Button is still clickable
    # by JS via window.parent.document.querySelector.
    st.markdown("""<style>
[data-testid="stAudioInput"] {
    position:absolute!important;width:1px!important;height:1px!important;
    overflow:hidden!important;opacity:0!important;pointer-events:none!important;
}
</style>""", unsafe_allow_html=True)
    audio_val = st.audio_input(
        "Record answer",
        key=f"whisper_mic_{q_number}",
        label_visibility="collapsed",
    )

    if audio_val is None:
        return st.session_state.get("transcribed_text", "")

    audio_id = audio_val.size
    if st.session_state.get("last_audio_id") == audio_id:
        return st.session_state.get("transcribed_text", "")

    raw = audio_val.read()

    # Store bytes for voice nervousness pipeline in submit_answer()
    st.session_state["last_audio_bytes"]  = raw
    st.session_state["last_audio_source"] = "whisper_mic"

    with st.spinner("🔄 Transcribing with Whisper…"):
        text = stt.transcribe(raw) if stt_ok else "[Whisper not available]"

    st.session_state["last_audio_id"]    = audio_id
    st.session_state["transcribed_text"] = text

    if text.startswith("["):
        st.warning(f"⚠ {text}")
    else:
        st.success(f"✅ Transcribed ({len(text.split())} words)")
        whisper_post_hud(text)

    return text


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 2 — Browser STT + MediaRecorder (triple-bridge — v8.0)
#
#  ROOT CAUSE of the original bug:
#    The old bridge wrote transcript/audio into hidden st.text_input fields
#    located via window.parent.document.querySelector('input[placeholder="..."]').
#    On any Streamlit rerun the React tree re-renders, resetting those inputs
#    before Python can read them — so the transcript vanished on submit.
#
#  FIX (three-layer approach):
#    1. PRIMARY  — Stable per-question session_state keys (_bstt_last_tx_{q},
#       _bstt_last_audio_{q}) that are NEVER reset by Streamlit reruns.
#       JS writes to these via window.__bsttWrite__() injected into parent.
#    2. SECONDARY — Legacy querySelector bridge (original approach) kept as
#       fallback. Still works when React hasn't yet unmounted the input.
#    3. TERTIARY — sessionStorage as last-resort, read back on next render.
#
#  Python reads with priority: stable key → bridge input → last known value.
# ══════════════════════════════════════════════════════════════════════════════

def _build_browser_stt_html(component_key: str, tx_placeholder: str,
                             audio_placeholder: str,
                             stable_tx_key: str, stable_audio_key: str) -> str:
    """
    Returns the HTML+JS string for the dual-track browser capture widget.

    Track 1 — Web Speech API  : streams live transcript into the UI
    Track 2 — MediaRecorder   : records raw audio bytes for nervousness model

    Data bridge (v8.0 — triple-layer):
      Method A — window.__bsttWrite__(key, value) injected into parent frame
                 writes directly into a stable session_state key (survives reruns)
      Method B — legacy querySelector on hidden st.text_input (original approach)
      Method C — sessionStorage backup

    tx_placeholder    — bridge input placeholder for the transcript text
    audio_placeholder — bridge input placeholder for the base64-encoded audio
    stable_tx_key     — session_state key that survives reruns (transcript)
    stable_audio_key  — session_state key that survives reruns (audio b64)
    """
    return f"""
<style>
  *{{box-sizing:border-box;font-family:'Inter',sans-serif;}}
  body{{margin:0;padding:0;background:transparent;overflow:hidden;}}

  /* ── Root wrap ── */
  #wrap{{
    background:rgba(5,12,32,0.96);
    border-radius:0 0 12px 12px;
    padding:14px 14px 10px;
  }}

  /* ── Mic button + radial waveform container ── */
  .mic-outer{{
    position:relative;
    width:120px;height:120px;
    flex-shrink:0;
    display:flex;align-items:center;justify-content:center;
  }}
  #vizCanvas{{
    position:absolute;
    top:0;left:0;
    width:120px;height:120px;
    pointer-events:none;
    border-radius:50%;
  }}

  /* Mic button itself — centered inside 120×120 canvas wrapper */
  #micBtn{{
    width:72px;height:72px;border-radius:50%;
    background:rgba(0,229,255,.07);
    border:1.5px solid rgba(0,229,255,.35);
    display:flex;align-items:center;justify-content:center;
    cursor:pointer;font-size:26px;
    transition:all .25s;position:relative;z-index:1;
    box-shadow:0 0 20px rgba(0,229,255,.1);
  }}
  #micBtn:hover{{
    background:rgba(0,229,255,.14);
    border-color:rgba(0,229,255,.7);
    box-shadow:0 0 28px rgba(0,229,255,.25);
  }}
  #micBtn.rec{{
    background:rgba(244,63,94,.12);
    border-color:rgba(244,63,94,.55);
    box-shadow:0 0 0 0 rgba(244,63,94,.4);
  }}

  /* ── Top row: mic + waveform ── */
  .mic-row{{display:flex;align-items:center;gap:16px;margin-bottom:10px;}}

  /* ── Waveform bars ── */
  #waveform{{
    display:flex;align-items:center;gap:3px;
    height:44px;flex:1;
    background:rgba(2,7,20,.7);
    border:1px solid rgba(0,229,255,.08);
    border-radius:10px;padding:6px 10px;
  }}
  .wb{{
    width:3px;border-radius:2px;
    background:rgba(0,229,255,.3);
    height:3px;flex-shrink:0;
    transition:height .08s ease;
  }}
  .wb.live{{
    background:rgba(0,229,255,.85);
    animation:wbAnim var(--d,.55s) ease-in-out infinite alternate;
  }}
  @keyframes wbAnim{{from{{height:3px}}to{{height:var(--h,28px)}}}}

  /* Timer + status */
  .rec-meta{{
    display:flex;align-items:center;justify-content:space-between;
    margin-bottom:8px;
  }}
  #hint{{
    font-family:'JetBrains Mono','Share Tech Mono',monospace;
    font-size:10px;color:rgba(0,229,255,.5);letter-spacing:1.5px;
  }}
  #eq-timer{{
    font-family:'JetBrains Mono','Share Tech Mono',monospace;
    font-size:11px;color:#f43f5e;letter-spacing:2px;display:none;
  }}

  /* ── Transcript box ── */
  #tb{{
    background:rgba(2,6,18,.85);
    border:1px solid rgba(0,229,255,.09);
    border-radius:10px;
    padding:10px 12px;
    min-height:72px;max-height:110px;
    overflow-y:auto;
    font-size:.78rem;color:#94b0d8;
    white-space:pre-wrap;
    margin-bottom:8px;
    line-height:1.65;
    position:relative;
  }}
  .tb-cursor{{
    display:inline-block;width:2px;height:.9em;
    background:#00e5ff;vertical-align:middle;margin-left:2px;
    animation:tbBlink 1s step-end infinite;
  }}
  @keyframes tbBlink{{0%,100%{{opacity:1}}50%{{opacity:0}}}}

  /* ── Status bar ── */
  #status{{
    font-size:.68rem;padding:4px 12px;border-radius:20px;margin-bottom:8px;
    background:rgba(255,255,255,.05);color:rgba(90,112,152,.8);
    text-align:center;transition:all .3s;
    font-family:'JetBrains Mono',monospace;letter-spacing:1px;
  }}
  #status.listening{{background:rgba(244,63,94,.12);color:#f43f5e;animation:statusPulse 1.2s infinite;}}
  #status.done{{background:rgba(0,255,136,.1);color:#00ff88;}}
  #status.processing{{background:rgba(124,107,255,.12);color:#a78bfa;}}
  @keyframes statusPulse{{0%,100%{{opacity:1}}50%{{opacity:.5}}}}

  /* ── Bottom row ── */
  .btn-row{{display:flex;gap:6px;}}
  #recBtnFull{{
    flex:1;border:1px solid rgba(0,229,255,.3);border-radius:10px;
    padding:9px;
    background:rgba(0,229,255,.06);
    color:#00e5ff;
    font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;
    letter-spacing:1.5px;cursor:pointer;transition:all .25s;
  }}
  #recBtnFull:hover{{background:rgba(0,229,255,.12);border-color:rgba(0,229,255,.6);}}
  #recBtnFull.rec{{
    background:rgba(244,63,94,.1);border-color:rgba(244,63,94,.5);
    color:#f43f5e;
  }}
  #clearBtn{{
    background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);
    color:#5a7098;padding:9px 14px;border-radius:9px;
    font-size:11px;cursor:pointer;transition:all .2s;
    font-family:'JetBrains Mono',monospace;
  }}
  #clearBtn:hover{{background:rgba(255,255,255,.1);color:#94b0d8;}}

  #nerv_badge{{font-size:.65rem;text-align:center;margin-top:5px;color:#00ff88;min-height:14px;font-family:'JetBrains Mono',monospace;}}
  #bridge_status{{font-size:.55rem;text-align:center;margin-top:3px;min-height:12px;font-family:'JetBrains Mono',monospace;color:rgba(0,255,136,.4);letter-spacing:1px;}}
</style>

<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<div id="wrap">
  <!-- Status -->
  <div id="status">CLICK MIC TO START RECORDING</div>

  <!-- Mic + waveform row -->
  <div class="mic-row">
    <div class="mic-outer" id="micOuter">
      <canvas id="vizCanvas" width="120" height="120"></canvas>
      <div id="micBtn" onclick="toggleRec()">🎤</div>
    </div>
    <div id="waveform">
      <!-- bars injected by JS -->
    </div>
  </div>

  <!-- Timer + hint -->
  <div class="rec-meta">
    <span id="hint">Click mic · speak · click again to stop</span>
    <span id="eq-timer">00:00</span>
  </div>

  <!-- Transcript -->
  <div id="tb"><span style="color:rgba(90,112,152,.5);font-style:italic;">Transcript appears here…</span></div>

  <!-- Buttons -->
  <div class="btn-row">
    <button id="recBtnFull" onclick="toggleRec()">▶ START RECORDING</button>
    <button id="clearBtn" onclick="clearAll()">✕ CLEAR</button>
  </div>
  <div id="nerv_badge"></div>
  <div id="bridge_status"></div>
</div>

<script>
(function(){{

  /* ── Build waveform bars ── */
  var wf = document.getElementById('waveform');
  var WB_COUNT = 40;
  var wbEls = [];
  for (var wi = 0; wi < WB_COUNT; wi++) {{
    var b = document.createElement('div');
    b.className = 'wb';
    b.style.setProperty('--h', (4 + Math.random()*32) + 'px');
    b.style.setProperty('--d', (0.28 + Math.random()*.55) + 's');
    wf.appendChild(b);
    wbEls.push(b);
  }}
  setInterval(function() {{
    if (running) wbEls.forEach(function(b) {{
      b.style.setProperty('--h', (4 + Math.random()*36) + 'px');
    }});
  }}, 180);

  /* ── Radial canvas visualizer ── */
  var vizCanvas  = document.getElementById('vizCanvas');
  var vizCtx     = vizCanvas.getContext('2d');
  var VIZ_W      = 120, VIZ_H = 120, VIZ_CX = 60, VIZ_CY = 60;
  var VIZ_INNER  = 36;
  var VIZ_OUTER  = 56;
  var VIZ_BARS   = 48;
  var vizPhase   = 0;

  function drawViz() {{
    requestAnimationFrame(drawViz);
    vizCtx.clearRect(0, 0, VIZ_W, VIZ_H);

    var isRec = running && analyser;

    if (isRec) {{
      var buf = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteFrequencyData(buf);
      var step = Math.floor(buf.length / VIZ_BARS);

      for (var i = 0; i < VIZ_BARS; i++) {{
        var amp   = buf[i * step] / 255;
        var angle = (i / VIZ_BARS) * Math.PI * 2 - Math.PI / 2;
        var barH  = VIZ_INNER + amp * (VIZ_OUTER - VIZ_INNER);

        var x0 = VIZ_CX + Math.cos(angle) * VIZ_INNER;
        var y0 = VIZ_CY + Math.sin(angle) * VIZ_INNER;
        var x1 = VIZ_CX + Math.cos(angle) * barH;
        var y1 = VIZ_CY + Math.sin(angle) * barH;

        var grad = vizCtx.createLinearGradient(x0, y0, x1, y1);
        grad.addColorStop(0, 'rgba(0,229,255,' + (0.4 + amp * 0.5) + ')');
        grad.addColorStop(1, 'rgba(0,255,180,' + (0.6 + amp * 0.4) + ')');

        vizCtx.beginPath();
        vizCtx.moveTo(x0, y0);
        vizCtx.lineTo(x1, y1);
        vizCtx.strokeStyle = grad;
        vizCtx.lineWidth   = amp > 0.05 ? 2.5 : 1.5;
        vizCtx.lineCap     = 'round';
        vizCtx.stroke();
      }}
    }} else {{
      vizPhase += 0.025;
      var breath = 0.35 + 0.18 * Math.sin(vizPhase);
      for (var j = 0; j < VIZ_BARS; j++) {{
        var idleAmp  = breath * (0.7 + 0.3 * Math.sin(vizPhase * 1.7 + j * 0.4));
        var idleAng  = (j / VIZ_BARS) * Math.PI * 2 - Math.PI / 2;
        var idleBarH = VIZ_INNER + idleAmp * (VIZ_OUTER - VIZ_INNER) * 0.55;

        var ix0 = VIZ_CX + Math.cos(idleAng) * VIZ_INNER;
        var iy0 = VIZ_CY + Math.sin(idleAng) * VIZ_INNER;
        var ix1 = VIZ_CX + Math.cos(idleAng) * idleBarH;
        var iy1 = VIZ_CY + Math.sin(idleAng) * idleBarH;

        vizCtx.beginPath();
        vizCtx.moveTo(ix0, iy0);
        vizCtx.lineTo(ix1, iy1);
        vizCtx.strokeStyle = 'rgba(124,107,255,' + (0.25 + idleAmp * 0.35) + ')';
        vizCtx.lineWidth   = 1.5;
        vizCtx.lineCap     = 'round';
        vizCtx.stroke();
      }}
    }}
  }}
  drawViz();

  /* ── Equalizer / analyser state (kept for HUD + nervousness feed compatibility) ── */
  var eqTmr = document.getElementById('eq-timer');

  function eqSetRecording(on) {{
    if (on) {{
      eqTmr.style.display = 'inline';
      eqSec = 0;
      clearInterval(eqInterval);
      eqInterval = setInterval(function() {{
        eqSec++;
        var m = Math.floor(eqSec/60), s = eqSec%60;
        eqTmr.textContent = (m<10?'0':'')+m+':'+(s<10?'0':'')+s;
      }}, 1000);
    }} else {{
      eqTmr.style.display = 'none';
      clearInterval(eqInterval);
    }}
  }}
  window._auraEqSetState = eqSetRecording;

  var eqSec = 0, eqInterval = null;

  /* ── App state ── */
  var recognition     = null;
  var mediaRecorder   = null;
  var audioChunks     = [];
  var finalTranscript = '';
  var running         = false;
  var micStream       = null;
  var audioCtx        = null;
  var analyser        = null;

  var SpeechRec        = window.SpeechRecognition || window.webkitSpeechRecognition;
  var hasMediaRecorder = !!window.MediaRecorder;

  if (!SpeechRec) {{
    setStatus('BROWSER STT REQUIRES CHROME OR EDGE', '');
    document.getElementById('recBtnFull').disabled = true;
    document.getElementById('micBtn').style.opacity = '.4';
    document.getElementById('micBtn').style.cursor = 'not-allowed';
  }}

  /* ── Stable key names (must match Python) ── */
  var STABLE_TX_KEY    = '{stable_tx_key}';
  var STABLE_AUDIO_KEY = '{stable_audio_key}';
  var TX_PLACEHOLDER   = '{tx_placeholder}';
  var AUDIO_PH         = '{audio_placeholder}';

  /* ── Helpers ── */
  function setStatus(msg, cls) {{
    var el = document.getElementById('status');
    el.textContent = msg;
    el.className = cls || '';
  }}

  function setBridgeStatus(msg) {{
    var el = document.getElementById('bridge_status');
    if (el) el.textContent = msg;
  }}

  function setRecordingUI(on) {{
    var outer = document.getElementById('micOuter');
    var btn   = document.getElementById('micBtn');
    var full  = document.getElementById('recBtnFull');

    if (on) {{
      outer.classList.add('recording');
      btn.classList.add('rec');
      btn.textContent = '⏹';
      full.classList.add('rec');
      full.textContent = '⏹ STOP RECORDING';
      wbEls.forEach(function(b) {{ b.classList.add('live'); }});
    }} else {{
      outer.classList.remove('recording');
      btn.classList.remove('rec');
      btn.textContent = '🎤';
      full.classList.remove('rec');
      full.textContent = '▶ START RECORDING';
      wbEls.forEach(function(b) {{ b.classList.remove('live'); b.style.height = '3px'; }});
    }}
  }}

  function setTranscript(text) {{
    var tb = document.getElementById('tb');
    if (!text) {{
      tb.innerHTML = '<span style="color:rgba(90,112,152,.5);font-style:italic;">Transcript appears here…</span>';
      return;
    }}
    tb.innerHTML = '<span style="color:#c4ddf4;">' + text + '</span><span class="tb-cursor"></span>';
  }}

  /* ══ BRIDGE: Three-method approach (v8.0) ══
   *
   * Method A: window.__bsttWrite__(key, value) — injected into parent frame
   *   by _inject_bstt_bridge_script(). Writes into a stable session_state key
   *   that Streamlit never resets. Most reliable.
   *
   * Method B: querySelector on hidden st.text_input (legacy fallback).
   *   Same as v7.0. Works when React hasn't yet unmounted the input.
   *
   * Method C: sessionStorage as last-resort signal.
   */
  function trySetParentInput(placeholder, value) {{
    try {{
      var el = window.parent.document.querySelector('input[placeholder="' + placeholder + '"]');
      if (!el) return false;
      var nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
      nativeSetter.call(el, value);
      el.dispatchEvent(new Event('input', {{ bubbles: true }}));
      return true;
    }} catch(e) {{ return false; }}
  }}

  function persistData(transcript, audio_b64) {{
    var ok_a = false, ok_b = false, ok_c = false;

    /* Method A — direct parent bridge (stable session_state key) */
    try {{
      if (window.parent && typeof window.parent.__bsttWrite__ === 'function') {{
        window.parent.__bsttWrite__(STABLE_TX_KEY, transcript);
        if (audio_b64) window.parent.__bsttWrite__(STABLE_AUDIO_KEY, audio_b64);
        ok_a = true;
      }}
    }} catch(e) {{}}

    /* Method B — querySelector on hidden bridge inputs */
    ok_b = trySetParentInput(TX_PLACEHOLDER, transcript);
    if (audio_b64) trySetParentInput(AUDIO_PH, audio_b64);

    /* Method C — sessionStorage */
    try {{
      sessionStorage.setItem('__bstt_tx__', transcript);
      if (audio_b64) sessionStorage.setItem('__bstt_audio__', audio_b64);
      ok_c = true;
    }} catch(e) {{}}

    var methods = (ok_a ? 'A' : '') + (ok_b ? 'B' : '') + (ok_c ? 'C' : '');
    setBridgeStatus(methods ? '✓ SAVED via ' + methods : '⚠ BRIDGE FAILED — type answer manually');
  }}

  /* ── Toggle ── */
  async function toggleRec() {{
    if (!SpeechRec) return;
    if (!running) {{ await startCapture(); }}
    else          {{ stopCapture(); }}
  }}

  async function startCapture() {{
    finalTranscript = '';
    audioChunks     = [];
    setTranscript('');
    document.getElementById('nerv_badge').innerText = '';
    setBridgeStatus('');
    document.getElementById('hint').innerText = 'Recording…';
    running = true;
    setRecordingUI(true);
    setStatus('● RECORDING', 'listening');
    eqSetRecording(true);

    window._auraLiveTranscript = '';
    window._auraStartTime      = Date.now();
    window._auraRecording      = true;

    /* Track 1: MediaRecorder */
    var stream;
    try {{
      stream    = await navigator.mediaDevices.getUserMedia({{ audio: true }});
      micStream = stream;
    }} catch(e) {{
      setStatus('MIC ACCESS DENIED: ' + e.message, '');
      running = false; setRecordingUI(false); return;
    }}

    try {{
      audioCtx  = new (window.AudioContext || window.webkitAudioContext)();
      var src   = audioCtx.createMediaStreamSource(stream);
      analyser  = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      src.connect(analyser);
    }} catch(e) {{ console.warn('AudioContext:', e); }}

    if (hasMediaRecorder) {{
      var mimeType =
        MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' :
        MediaRecorder.isTypeSupported('audio/webm')             ? 'audio/webm' : 'audio/ogg';
      mediaRecorder = new MediaRecorder(stream, {{ mimeType: mimeType }});
      mediaRecorder.ondataavailable = function(e) {{
        if (e.data && e.data.size > 0) audioChunks.push(e.data);
      }};
      mediaRecorder.onstop = function() {{ finalize(); }};
      mediaRecorder.start(250);
    }}

    /* Track 2: Web Speech API */
    recognition = new SpeechRec();
    recognition.continuous     = true;
    recognition.interimResults = true;
    recognition.lang           = 'en-US';

    recognition.onresult = function(e) {{
      var interim = '';
      var gotFinal = false;
      for (var i = e.resultIndex; i < e.results.length; i++) {{
        if (e.results[i].isFinal) {{
          finalTranscript += e.results[i][0].transcript + ' ';
          gotFinal = true;
        }} else {{
          interim += e.results[i][0].transcript;
        }}
      }}
      var live = finalTranscript + interim;
      setTranscript(live);
      window._auraLiveTranscript = live;

      /* v9.0 — AUTO-PUSH: write finalTranscript to stable key on every
         final result so Python always has the latest text even if the user
         closes the tab or Streamlit reruns before stopCapture() fires.
         We push only when there is new finalized text to avoid flooding. */
      if (gotFinal && finalTranscript.trim()) {{
        try {{
          if (window.parent && typeof window.parent.__bsttWrite__ === 'function') {{
            window.parent.__bsttWrite__(STABLE_TX_KEY, finalTranscript.trim());
          }}
        }} catch(e2) {{}}
        /* sessionStorage mirror so Python can read it via Method C too */
        try {{ sessionStorage.setItem('__bstt_tx__', finalTranscript.trim()); }} catch(e2) {{}}
      }}
    }};

    recognition.onerror = function(e) {{
      var msgs = {{
        'not-allowed':   'MIC PERMISSION DENIED',
        'no-speech':     'NO SPEECH DETECTED',
        'audio-capture': 'NO MICROPHONE FOUND',
        'network':       'NETWORK ERROR',
        'aborted':       'Recording stopped.',
      }};
      setStatus(msgs[e.error] || e.error, '');
      if (e.error !== 'aborted') {{ running = false; setRecordingUI(false); }}
    }};

    recognition.onend = function() {{
      if (running) recognition.start();
    }};

    recognition.start();
  }}

  function stopCapture() {{
    running = false;
    setRecordingUI(false);
    setStatus('PROCESSING AUDIO…', 'processing');
    document.getElementById('hint').innerText = 'Sending to voice model…';
    window._auraRecording = false;
    eqSetRecording(false);

    if (recognition)  {{ recognition.onend = null; recognition.stop(); }}
    if (micStream)    micStream.getTracks().forEach(function(t) {{ t.stop(); }});
    if (audioCtx)     {{ audioCtx.close(); analyser = null; }}

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {{
      mediaRecorder.stop();
    }} else {{
      finalize();
    }}
  }}

  function finalize() {{
    var transcript = finalTranscript.trim();
    setTranscript(transcript || '(no speech detected)');

    if (!transcript && audioChunks.length === 0) {{
      setStatus('NOTHING RECORDED — TRY AGAIN', '');
      document.getElementById('hint').innerText = 'Speak closer to mic';
      return;
    }}

    if (audioChunks.length > 0) {{
      var mimeType = mediaRecorder ? mediaRecorder.mimeType : 'audio/webm';
      var blob     = new Blob(audioChunks, {{ type: mimeType }});
      var reader   = new FileReader();
      reader.onloadend = function() {{
        var b64 = reader.result.split(',')[1];
        persistData(transcript, b64);
        setStatus('✓ ANSWER RECORDED', 'done');
        document.getElementById('hint').innerText = 'Transcript saved ✓';
        document.getElementById('nerv_badge').innerText = '🎙 Audio captured for nervousness analysis';
      }};
      reader.readAsDataURL(blob);
    }} else {{
      persistData(transcript, '');
      setStatus('✓ ANSWER RECORDED', 'done');
      document.getElementById('hint').innerText = 'Transcript saved ✓';
    }}
  }}

  function clearAll() {{
    finalTranscript = '';
    audioChunks     = [];
    setTranscript('');
    document.getElementById('hint').innerText = 'Click mic · speak · click again to stop';
    document.getElementById('nerv_badge').innerText = '';
    setBridgeStatus('');
    setStatus('CLICK MIC TO START RECORDING', '');
    /* Clear all three bridge layers */
    persistData('', '');
    try {{ sessionStorage.removeItem('__bstt_tx__'); sessionStorage.removeItem('__bstt_audio__'); }} catch(e) {{}}
  }}

  window.toggleRec = toggleRec;
  window.clearAll  = clearAll;

}})();
</script>
"""


def _inject_bstt_bridge_script(stable_tx_key: str, stable_audio_key: str) -> None:
    """
    Injects a <script> into the parent Streamlit frame that registers
    window.__bsttWrite__(key, value).

    When called from the child iframe JS, this function simulates a React
    synthetic input event on a hidden text_input stamped with data-bstt-key,
    so Streamlit picks up the value on the next rerun and stores it in a
    stable session_state key that is never reset between reruns.
    """
    components.html(
        f"""
<script>
(function(){{
  var parent = window.parent;
  if (!parent) return;
  if (typeof parent.__bsttWrite__ === 'function') return;  /* idempotent */

  parent.__bsttWrite__ = function(key, value) {{
    /* Find the hidden input stamped with data-bstt-key and fire a React event */
    try {{
      var inputs = parent.document.querySelectorAll('input[type="text"]');
      for (var i = 0; i < inputs.length; i++) {{
        if (inputs[i].getAttribute('data-bstt-key') === key) {{
          var setter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype, 'value').set;
          setter.call(inputs[i], value);
          inputs[i].dispatchEvent(new Event('input', {{bubbles: true}}));
          return;
        }}
      }}
    }} catch(e) {{}}
    /* Last resort: sessionStorage signal */
    try {{ parent.sessionStorage.setItem('__bstt__' + key, value); }} catch(e) {{}}
  }};
}})();
</script>
""",
        height=0,
    )


def browser_stt_with_audio(q_number: int) -> Tuple[str, Optional[bytes]]:
    """
    Dual-track browser voice capture (v8.0 — triple-bridge).
    Returns (transcript_text, raw_audio_bytes).

    Stable keys (never reset by Streamlit reruns):
        _bstt_last_tx_{q_number}     — transcript text
        _bstt_last_audio_{q_number}  — base64 audio (decoded to bytes on read)

    Bridge inputs (secondary path, may reset on rerun):
        _bstt_tx_{q_number}          — hidden st.text_input for transcript
        _bstt_audio_{q_number}       — hidden st.text_input for audio b64

    Python reads with priority: stable key → bridge input → last known value.
    """
    component_key     = f"browser_stt_{q_number}"
    tx_placeholder    = f"__bstt_tx_{q_number}__"
    audio_placeholder = f"__bstt_audio_{q_number}__"

    # Stable session_state keys — survive reruns, never reset by Streamlit
    stable_tx_key    = f"_bstt_last_tx_{q_number}"
    stable_audio_key = f"_bstt_last_audio_{q_number}"

    st.markdown(
        '<div style="font-size:.74rem;color:#8fc4e0;margin-bottom:.4rem;">'
        '🌐 Uses Chrome/Edge built-in speech recognition. '
        'Transcript and audio are captured automatically when you stop recording.</div>',
        unsafe_allow_html=True,
    )

    # ── Inject the parent-frame bridge script (Method A) ────────────────────
    _inject_bstt_bridge_script(stable_tx_key, stable_audio_key)

    # ── Hidden bridge text_inputs (Method B — legacy querySelector path) ────
    # Stamped with data-bstt-key via JS so __bsttWrite__ can find them.
    tx_bridge = st.text_input(
        "bstt_transcript_bridge",
        key=f"_bstt_tx_{q_number}",
        label_visibility="collapsed",
        placeholder=tx_placeholder,
    )
    audio_bridge = st.text_input(
        "bstt_audio_bridge",
        key=f"_bstt_audio_{q_number}",
        label_visibility="collapsed",
        placeholder=audio_placeholder,
    )

    # Stamp data-bstt-key attributes so __bsttWrite__ can locate the inputs
    components.html(
        f"""
<script>
(function(){{
  var PARENT = window.parent.document;
  function stamp(ph, key) {{
    var el = PARENT.querySelector('input[placeholder="' + ph + '"]');
    if (el) el.setAttribute('data-bstt-key', key);
  }}
  setTimeout(function() {{
    stamp('{tx_placeholder}',    '{stable_tx_key}');
    stamp('{audio_placeholder}', '{stable_audio_key}');
  }}, 200);
}})();
</script>
""",
        height=0,
    )

    # ── Render the recording widget ─────────────────────────────────────────
    components.html(
        _build_browser_stt_html(
            component_key,
            tx_placeholder,
            audio_placeholder,
            stable_tx_key,
            stable_audio_key,
        ),
        height=260,
    )

    # ── Live Filler + WPM HUD ──────────────────────────────────────────────
    components.html(
        _build_live_hud_html(f"bstt_{q_number}"),
        height=148,
    )

    # ── READ TRANSCRIPT — three-tier priority ────────────────────────────────
    # P1: stable session_state key (set by __bsttWrite__ via Method A)
    # P2: bridge text_input (set by querySelector via Method B)
    # P3: previously stored value from last successful recording
    transcript: str = ""
    audio_bytes: Optional[bytes] = None

    stable_tx = (st.session_state.get(stable_tx_key) or "").strip()
    bridge_tx = (tx_bridge or "").strip()

    # Sync bridge → stable key whenever bridge has a newer value
    if bridge_tx and bridge_tx != st.session_state.get(stable_tx_key, ""):
        st.session_state[stable_tx_key] = bridge_tx
        stable_tx = bridge_tx

    transcript = stable_tx or bridge_tx

    # v9.0 — AUTO-SYNC: if we got a transcript from any source, always write
    # it into the stable key so score_from_transcript() can read it reliably.
    # Also sync the shared transcribed_text → stable key as a fallback for
    # cases where the JS bridge fired before Python saw the stable key write.
    if not transcript:
        shared_tx = (st.session_state.get("transcribed_text") or "").strip()
        if shared_tx:
            transcript = shared_tx
            st.session_state[stable_tx_key] = shared_tx   # backfill stable key

    if transcript:
        # Always keep stable key up to date — this is the key score_from_transcript reads
        st.session_state[stable_tx_key]      = transcript
        st.session_state["transcribed_text"] = transcript
        st.session_state["last_audio_source"] = "browser_stt"

    # ── READ AUDIO — same three-tier logic ──────────────────────────────────
    stable_audio_b64 = (st.session_state.get(stable_audio_key) or "").strip()
    bridge_audio_b64 = (audio_bridge or "").strip()

    if bridge_audio_b64 and bridge_audio_b64 != stable_audio_b64:
        st.session_state[stable_audio_key] = bridge_audio_b64
        stable_audio_b64 = bridge_audio_b64

    audio_b64 = stable_audio_b64 or bridge_audio_b64

    if audio_b64:
        try:
            audio_bytes = base64.b64decode(audio_b64)
            st.session_state["last_audio_bytes"]      = audio_bytes
            st.session_state["last_audio_source"]     = "browser_stt"
            # v9.0: stable backup — survives the submit rerun that clears
            # last_audio_bytes (app.py line ~4178 sets it to None on submit).
            st.session_state["_bstt_last_audio_bytes"] = audio_bytes
        except Exception:
            audio_bytes = None  # malformed base64 — skip silently

    if transcript:
        st.success(f"✅ Transcript ready ({len(transcript.split())} words)"
                   + (" · 🎙 Audio captured for nervousness analysis" if audio_bytes else ""))
    else:
        # Fall back to last known value for this question
        transcript = st.session_state.get(stable_tx_key, "") \
                  or st.session_state.get("transcribed_text", "")

    return transcript, audio_bytes


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED VOICE INPUT PANEL — 3 tabs (Upload removed)
# ══════════════════════════════════════════════════════════════════════════════

def voice_input_panel(stt, q_number: int = 0, *, key_suffix: Optional[int] = None) -> str:
    """
    Full voice input panel with 3 tabs:
      Tab 1 — Whisper AI      : st.audio_input → Whisper → text
                                raw bytes stored for voice model
      Tab 2 — Browser STT     : Web Speech API (live transcript) +
                                MediaRecorder (audio bytes for voice model)
                                NO copy-paste required — auto-sends on stop
      Tab 3 — Type            : plain text area fallback

    Args:
        stt        : SpeechToText instance
        q_number   : integer key used to namespace Streamlit widget keys
        key_suffix : alias for q_number — pass either one; key_suffix takes
                     precedence when both are supplied (supports legacy callers
                     that pass ``voice_input_panel(stt, key_suffix=9900 + idx)``)

    Returns the best available answer text string.
    All tabs also populate st.session_state["last_audio_bytes"] so
    submit_answer() can pass audio to the voice nervousness pipeline.
    """
    # ── Resolve q_number from key_suffix alias ────────────────────────────────
    if key_suffix is not None:
        q_number = int(key_suffix)
    tab_w, tab_b, tab_t = st.tabs([
        "🎙 Whisper AI",
        "🌐 Browser STT",
        "⌨️ Type",
    ])

    answer = ""

    # ── Tab 1: Whisper ────────────────────────────────────────────────────────
    with tab_w:
        text_w = whisper_audio_input(stt, q_number)
        if text_w and not text_w.startswith("["):
            edited = st.text_area(
                "Edit transcription if needed:",
                value=text_w,
                height=110,
                key=f"whisper_edit_{q_number}",
            )
            answer = edited
        elif text_w and text_w.startswith("["):
            st.markdown(
                f'<div style="font-size:.78rem;color:#fcd34d;">{text_w}</div>',
                unsafe_allow_html=True,
            )
    # ── Tab 2: Browser STT (dual-track) ───────────────────────────────────────
    with tab_b:
        transcript, audio_bytes = browser_stt_with_audio(q_number)

        if transcript:
            # Show editable transcript — user can fix any STT errors before submitting
            edited_b = st.text_area(
                "Edit transcript if needed:",
                value=transcript,
                height=110,
                key=f"browser_edit_{q_number}",
            )
            answer = edited_b

            # audio_bytes is now populated from the audio bridge when MediaRecorder
            # captured audio. Store it so submit_answer() passes it to the voice
            # nervousness pipeline. (Already stored in session_state by
            # browser_stt_with_audio, but re-set here for safety in case the
            # user edits the text after stopping.)
            if audio_bytes:
                st.session_state["last_audio_bytes"]  = audio_bytes
                st.session_state["last_audio_source"] = "browser_stt"

    # ── Tab 3: Type ───────────────────────────────────────────────────────────
    with tab_t:
        typed = st.text_area(
            "Type your answer:",
            height=130,
            key=f"typed_{q_number}",
            placeholder="Write a detailed answer here…",
        )
        if typed.strip():
            answer = typed.strip()
            # No audio for typed answers — clear stale bytes to avoid
            # running voice model on the previous question's audio
            st.session_state.pop("last_audio_bytes",  None)
            st.session_state.pop("last_audio_source", None)

    return answer