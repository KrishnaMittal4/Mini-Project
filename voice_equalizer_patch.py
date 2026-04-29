"""
voice_equalizer_patch.py
========================
Drop-in replacement for the audio_waveform() function in app.py (v9.1).

CHANGES
-------
Replace the existing audio_waveform() block (lines 937-962) with the two
functions below:

  • audio_waveform(recording, emotion, nervousness)  — backwards-compatible
    drop-in; adds emotion label + nervousness tint to the existing call sites.
  • render_eq_widget(recording, emotion, nervousness) — the full standalone
    widget (equalizer + status row + rec-time counter).

HOW TO INTEGRATE
----------------
1.  Delete lines 937-962 in app.py (the old audio_waveform function).
2.  Paste this entire file's content in their place.
3.  In voice_input_panel (voice_input.py) swap any call like:
        audio_waveform(recording=True)
    for:
        audio_waveform(recording=True,
                       emotion=st.session_state.get("live_voice_emotion","Neutral"),
                       nervousness=st.session_state.get("live_voice_nerv", 0.2))

    Old callers that pass only `recording` keep working unchanged.

DESIGN NOTES
------------
• 12 bars with staggered animation-duration (0.61 – 0.97 s) so motion is
  never uniform.  Heights are randomised per-bar at initialisation, then
  driven by Math.sin() with a per-bar phase offset so adjacent bars differ.
• When recording=False all bars collapse to 3 px (flat / frozen look).
• Bar colour blends green→violet through cyan depending on nervousness:
    low  (<0.35) → #00ff88  matrix-green
    mid  (<0.65) → #00d4ff  neon-cyan
    high (≥0.65) → #7f5af0  electric-violet
  matching the existing nerv_css() colour scheme in app.py.
• A pulsing REC dot + elapsed-time counter (MM:SS) appear while recording.
• The emotion label is shown right-aligned in Share Tech Mono, matching the
  sidebar Live Signals block.
• The component uses components.html() exactly like the original, so no new
  dependencies are needed.
"""

import streamlit.components.v1 as components


# ── colour helpers (mirror nerv_css / emo_css in app.py) ─────────────────────
def _eq_color(nervousness: float) -> str:
    if nervousness < 0.35:
        return "#00ff88"   # matrix-green
    if nervousness < 0.65:
        return "#00d4ff"   # neon-cyan
    return "#7f5af0"       # electric-violet


def _eq_gradient(nervousness: float) -> str:
    """CSS linear-gradient string for the bar fill."""
    if nervousness < 0.35:
        return "linear-gradient(to top, #00ff88, #00d4ff)"
    if nervousness < 0.65:
        return "linear-gradient(to top, #00d4ff, #7f5af0)"
    return "linear-gradient(to top, #7f5af0, #ff3366)"


# ── main widget ───────────────────────────────────────────────────────────────
def render_eq_widget(
    recording: bool = False,
    emotion: str = "Neutral",
    nervousness: float = 0.2,
    num_bars: int = 12,
    height_px: int = 72,
) -> None:
    """
    Render the animated equalizer widget inside a Streamlit components.html
    iframe.  Call this wherever you previously called audio_waveform().

    Parameters
    ----------
    recording   : True while the mic / pipeline is actively capturing audio.
    emotion     : Current voice emotion label (e.g. "Calm", "Nervous").
    nervousness : Float 0-1 from NervousnessPipeline / live_voice_nerv.
    num_bars    : Number of equalizer bars (default 12).
    height_px   : Iframe height in pixels (default 72).
    """
    col        = _eq_color(nervousness)
    grad       = _eq_gradient(nervousness)
    is_rec     = "true" if recording else "false"
    nerv_pct   = int(nervousness * 100)

    # Per-bar animation durations spread between 0.61 s and 0.97 s
    durations = [
        round(0.61 + (i / max(num_bars - 1, 1)) * 0.36, 2)
        for i in range(num_bars)
    ]
    # JS array literal
    dur_js = "[" + ",".join(str(d) for d in durations) + "]"

    # Status label
    if not recording:
        status_text  = "STANDBY"
        status_color = "rgba(122,184,216,0.5)"
    elif nervousness < 0.35:
        status_text  = "RECORDING — CALM"
        status_color = "#00ff88"
    elif nervousness < 0.65:
        status_text  = "RECORDING — MODERATE"
        status_color = "#00d4ff"
    else:
        status_text  = "RECORDING — HIGH NERV."
        status_color = "#ff3366"

    html = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@700&display=swap');

  @keyframes rec-pulse {{
    0%,100% {{ opacity:1; box-shadow:0 0 0 0 {col}66; }}
    50%      {{ opacity:.5; box-shadow:0 0 0 5px transparent; }}
  }}
  @keyframes rec-ring {{
    0%   {{ box-shadow:0 0 0 0 {col}55; }}
    70%  {{ box-shadow:0 0 0 7px transparent; }}
    100% {{ box-shadow:0 0 0 0 transparent; }}
  }}

  @media (prefers-reduced-motion: reduce) {{
    * {{ animation:none !important; transition:none !important; }}
  }}

  #eq-root {{
    display:flex; flex-direction:column; gap:6px;
    background:rgba(5,10,22,0.88);
    border:0.5px solid {col}28;
    border-radius:10px;
    padding:10px 14px 8px;
    position:relative;
    overflow:hidden;
  }}
  /* Scanline shimmer on top edge */
  #eq-root::before {{
    content:'';
    position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,{col},transparent);
    opacity:0.4;
  }}
  #eq-bars {{
    display:flex;
    align-items:flex-end;
    gap:4px;
    height:36px;
  }}
  .eq-bar {{
    flex:1;
    border-radius:2px;
    background:{grad};
    min-width:6px;
    transition:height 0.05s ease;
    will-change:height;
  }}
  #eq-status-row {{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:8px;
  }}
  #eq-left {{
    display:flex;
    align-items:center;
    gap:6px;
  }}
  #rec-dot {{
    width:7px; height:7px; border-radius:50%;
    background:{col};
    flex-shrink:0;
    display:none;  /* shown by JS when recording */
    animation: rec-pulse 1s ease-in-out infinite,
               rec-ring  1.4s ease-out infinite;
  }}
  #eq-status {{
    font-family:'Share Tech Mono', monospace;
    font-size:10px;
    color:{status_color};
    letter-spacing:1px;
  }}
  #eq-timer {{
    font-family:'Share Tech Mono', monospace;
    font-size:10px;
    color:rgba(122,184,216,0.6);
    letter-spacing:1px;
    display:none;
  }}
  #eq-right {{
    display:flex;
    align-items:center;
    gap:8px;
  }}
  #eq-nerv {{
    font-family:'Share Tech Mono', monospace;
    font-size:10px;
    color:{col};
    opacity:0.8;
  }}
  #eq-emotion {{
    font-family:'Share Tech Mono', monospace;
    font-size:10px;
    color:{col};
    font-weight:700;
    letter-spacing:1px;
    text-transform:uppercase;
  }}
</style>

<div id="eq-root">
  <!-- Equalizer bars -->
  <div id="eq-bars">
    {''.join(f'<div class="eq-bar" id="bar{i}" style="height:3px"></div>' for i in range(num_bars))}
  </div>

  <!-- Status row -->
  <div id="eq-status-row">
    <div id="eq-left">
      <div id="rec-dot"></div>
      <span id="eq-status">{status_text}</span>
      <span id="eq-timer">00:00</span>
    </div>
    <div id="eq-right">
      <span id="eq-nerv">NERV {nerv_pct}%</span>
      <span id="eq-emotion">{emotion}</span>
    </div>
  </div>
</div>

<script>
(function() {{
  const RECORDING  = {is_rec};
  const NUM_BARS   = {num_bars};
  const DURATIONS  = {dur_js};       // s per bar
  const BAR_H_MAX  = 34;             // px max height when recording
  const BAR_H_MIN  = 3;              // px flat height when standby

  const bars = Array.from({{length: NUM_BARS}}, (_, i) => document.getElementById('bar' + i));
  const recDot   = document.getElementById('rec-dot');
  const timerEl  = document.getElementById('eq-timer');

  // Unique phase per bar so adjacent bars never move in sync
  const phases   = bars.map((_, i) => Math.random() * Math.PI * 2);
  // Amplitude weight — middle bars are taller for a dome shape
  const weights  = bars.map((_, i) => {{
    const centre = (NUM_BARS - 1) / 2;
    const dist   = Math.abs(i - centre) / centre;  // 0 at centre, 1 at edge
    return 0.45 + (1 - dist) * 0.55;               // 0.45 – 1.0
  }});

  let startTime  = null;
  let rafId      = null;

  function pad(n) {{ return String(n).padStart(2, '0'); }}

  function tick(timestamp) {{
    if (!startTime) startTime = timestamp;
    const elapsed = (timestamp - startTime) / 1000;  // seconds

    if (RECORDING) {{
      // Animate bars
      bars.forEach((bar, i) => {{
        const freq = 1 / DURATIONS[i];
        const raw  = Math.sin(elapsed * 2 * Math.PI * freq + phases[i]);
        // raw ∈ [-1, 1] → [0.1, 1]
        const norm = (raw + 1) / 2 * 0.9 + 0.1;
        const h    = Math.round(norm * weights[i] * BAR_H_MAX);
        bar.style.height = Math.max(h, 2) + 'px';
      }});

      // Update timer
      const mins = Math.floor(elapsed / 60);
      const secs = Math.floor(elapsed % 60);
      timerEl.textContent = pad(mins) + ':' + pad(secs);
    }} else {{
      // Frozen / standby — all bars collapse to min height
      bars.forEach(bar => {{ bar.style.height = BAR_H_MIN + 'px'; }});
    }}

    rafId = requestAnimationFrame(tick);
  }}

  // Show/hide rec dot & timer
  if (RECORDING) {{
    recDot.style.display  = 'block';
    timerEl.style.display = 'inline';
  }}

  rafId = requestAnimationFrame(tick);

  // Clean up when iframe is removed (Streamlit re-renders)
  window.addEventListener('beforeunload', () => {{ cancelAnimationFrame(rafId); }});
}})();
</script>
"""

    components.html(html, height=height_px)


# ── backwards-compatible drop-in ─────────────────────────────────────────────
def audio_waveform(
    recording: bool = False,
    emotion: str = "Neutral",
    nervousness: float = 0.2,
) -> None:
    """
    Backwards-compatible wrapper around render_eq_widget().
    Existing callers that pass only `recording=True/False` work unchanged.

    Usage (existing voice_input.py call sites):
        audio_waveform(recording=True)

    Enhanced usage (passes live state from session):
        audio_waveform(
            recording    = True,
            emotion      = st.session_state.get("live_voice_emotion", "Neutral"),
            nervousness  = st.session_state.get("live_voice_nerv", 0.2),
        )
    """
    render_eq_widget(
        recording=recording,
        emotion=emotion,
        nervousness=nervousness,
    )
