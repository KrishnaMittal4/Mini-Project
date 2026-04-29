"""
speech_to_text.py — Aura AI | Robust Whisper ASR Module (v5.3)
===============================================================
v5.3 — ACCENT-ADAPTIVE MODEL SELECTION + AUTO-DOWNGRADE
────────────────────────────────────────────────────────
Research basis:
  • Radford et al. (2022) report whisper-base WER degrades +8–12% on
    non-native/accented English vs. native English benchmarks.
    whisper-small closes ~50% of that gap at only ~2× inference time
    on CPU; whisper-medium closes ~80% at ~6× inference time.
  • DEFAULT_MODEL is now "openai/whisper-small".
  • WHISPER_MODEL_PRIORITY defines a three-tier auto-downgrade chain:
      small → base → tiny
    If small fails to load (OOM or missing weights), _load() silently
    retries with the next smaller model before giving up.
  • Users can override at init: SpeechToText("openai/whisper-medium")
    for highest accuracy when GPU or fast CPU is available.
  • Environment variable AURA_WHISPER_MODEL overrides the default at
    import time — useful for deployment without code changes:
      export AURA_WHISPER_MODEL=openai/whisper-medium

DOWNGRADE CHAIN RATIONALE:
  Model       | Size  | Relative CPU latency | Accented WER gain
  ------------|-------|----------------------|-------------------
  whisper-tiny|  39 MB| 1×  (baseline)       | —
  whisper-base|  74 MB| 2×                   | −3–4 pp vs tiny
  whisper-small| 244 MB| 4–5×                 | −5–8 pp vs base ← DEFAULT
  whisper-medium|769 MB| 12–15×               | −2–3 pp vs small

Fixes carried forward from v5.2:
  • SF_OK, PYDUB_OK, LIBROSA_OK, SCIPY_OK, TORCH_OK, TRANSFORMERS_OK
    are guaranteed to exist at module level (voice_input.py imports them)
  • Silent audio threshold 5e-3 (raised from 1e-4)
  • librosa.beat.beat_track tuple unpack guard for librosa >= 0.10

Research basis:
  • Radford et al. "Robust Speech Recognition via Large-Scale Weak
    Supervision", OpenAI 2022 — Whisper multilingual ASR
  • OpenAI Whisper model card (2023) — per-model WER on Common Voice
    accented English subsets
"""

from __future__ import annotations

import io
import logging
import os
import warnings
from typing import List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("STT")

# ── Model selection ───────────────────────────────────────────────────────────
# Default upgraded from whisper-base → whisper-small (v5.3).
# Override via env var, e.g.:  export AURA_WHISPER_MODEL=openai/whisper-medium
DEFAULT_MODEL = os.environ.get("AURA_WHISPER_MODEL", "openai/whisper-small")

# Auto-downgrade chain: if the preferred model fails to load (OOM, missing
# weights), _load() retries with each successive smaller model in order.
# The chain is only walked on load failure, not on inference error.
WHISPER_MODEL_PRIORITY: List[str] = [
    "openai/whisper-small",   # default — best balance of accuracy vs speed
    "openai/whisper-base",    # fallback — 74 MB, still usable on low-RAM machines
    "openai/whisper-tiny",    # last resort — 39 MB, always loads
]

# Per-model approximate CPU latency multipliers (relative to tiny = 1×).
# Surfaced in SpeechToText.status so the UI can show users what to expect.
_MODEL_LATENCY: dict = {
    "openai/whisper-tiny":   "~1× (fastest)",
    "openai/whisper-base":   "~2×",
    "openai/whisper-small":  "~4–5× (default)",
    "openai/whisper-medium": "~12–15× (highest accuracy)",
    "openai/whisper-large":  "~20–25× (research grade)",
    "openai/whisper-large-v2": "~20–25× (research grade)",
    "openai/whisper-large-v3": "~20–25× (research grade)",
}

# ── Optional deps — flags ALWAYS defined at module level ─────────────────────
# voice_input.py imports these directly:
#   from speech_to_text import SF_OK, PYDUB_OK, LIBROSA_OK, SCIPY_OK, TORCH_OK

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import soundfile as sf
    SF_OK = True
except ImportError:
    SF_OK = False

try:
    from pydub import AudioSegment
    PYDUB_OK = True
except ImportError:
    PYDUB_OK = False

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

try:
    import scipy.io.wavfile as scipy_wav
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_OK = True
except ImportError:
    TRANSFORMERS_OK = False

TARGET_SR = 16_000   # Whisper requires 16 kHz


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO DECODER — tries multiple libraries in order
# ══════════════════════════════════════════════════════════════════════════════

def decode_audio(raw_bytes: bytes) -> Tuple[Optional[np.ndarray], int]:
    """
    Decode any audio format browsers may produce (WAV / WebM / OGG / MP4).
    Returns (float32 numpy array, sample_rate) or (None, 0) on total failure.
    """
    audio: Optional[np.ndarray] = None
    sr = 0

    # 1. soundfile (fastest, handles WAV/FLAC/OGG natively)
    if SF_OK and audio is None:
        try:
            audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
            log.info(f"soundfile decoded: {len(audio)} samples @ {sr} Hz")
        except Exception as e:
            log.debug(f"soundfile failed: {e}")
            audio = None

    # 2. pydub (handles WebM / OGG / MP3 / MP4 — needs ffmpeg)
    if PYDUB_OK and audio is None:
        try:
            seg   = AudioSegment.from_file(io.BytesIO(raw_bytes))
            seg   = seg.set_channels(1).set_frame_rate(TARGET_SR)
            arr   = np.array(seg.get_array_of_samples(), dtype=np.float32)
            audio = arr / (2 ** (8 * seg.sample_width - 1))
            sr    = TARGET_SR
            log.info(f"pydub decoded: {len(audio)} samples")
        except Exception as e:
            log.debug(f"pydub failed: {e}")
            audio = None

    # 3. scipy wavfile (WAV only fallback)
    if SCIPY_OK and audio is None:
        try:
            sr, arr = scipy_wav.read(io.BytesIO(raw_bytes))
            audio   = arr.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            max_val = np.iinfo(arr.dtype).max if np.issubdtype(arr.dtype, np.integer) else 1.0
            if max_val > 1.0:
                audio /= max_val
            log.info(f"scipy decoded: {len(audio)} samples @ {sr} Hz")
        except Exception as e:
            log.debug(f"scipy failed: {e}")
            audio = None

    # 4. librosa (most permissive — uses ffmpeg internally if available)
    if LIBROSA_OK and audio is None:
        try:
            audio, sr = librosa.load(io.BytesIO(raw_bytes), sr=None, mono=True)
            log.info(f"librosa decoded: {len(audio)} samples @ {sr} Hz")
        except Exception as e:
            log.debug(f"librosa failed: {e}")
            audio = None

    if audio is None:
        log.error("All decoders failed. Install pydub+ffmpeg for WebM/OGG support.")
        return None, 0

    # Normalise to mono float32
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    # Resample to 16 kHz if needed
    if sr != TARGET_SR:
        if LIBROSA_OK:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        elif SCIPY_OK:
            from scipy.signal import resample as sp_resample
            target_len = int(len(audio) * TARGET_SR / sr)
            audio      = sp_resample(audio, target_len).astype(np.float32)
        sr = TARGET_SR

    return audio, sr


def is_silent(audio: np.ndarray, threshold: float = 5e-3) -> bool:
    """
    Returns True if audio is effectively silent.
    Threshold raised from 1e-4 to 5e-3 to skip near-silent recordings
    that would waste Whisper inference time.
    """
    return float(np.abs(audio).mean()) < threshold


# ══════════════════════════════════════════════════════════════════════════════
#  SPEECH-TO-TEXT CLASS
# ══════════════════════════════════════════════════════════════════════════════

class SpeechToText:
    """
    Whisper-based ASR with robust multi-format audio decoding and
    accent-adaptive model selection (v5.3).

    Model selection (in priority order):
      1. Explicit model_name passed to __init__
      2. AURA_WHISPER_MODEL environment variable
      3. DEFAULT_MODEL (openai/whisper-small)

    If the chosen model fails to load (OOM, missing weights), _load()
    automatically retries with each model in WHISPER_MODEL_PRIORITY until
    one succeeds.  The finally-loaded model name is stored in self.model_name
    so callers always know which model is actually running.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name    = model_name   # may be updated by downgrade chain
        self._pipe         = None
        self._ready        = False
        self._error_msg    = ""
        self._downgraded   = False        # True if a smaller model was used
        self._load()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _try_load_model(self, model_name: str) -> bool:
        """
        Attempt to load a single Whisper model via HuggingFace pipeline.
        Returns True on success, False on any failure.
        Populates self._pipe and self.model_name on success.
        """
        try:
            device = 0 if (TORCH_OK and torch.cuda.is_available()) else -1
            pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=device,
                chunk_length_s=30,
                stride_length_s=5,
                return_timestamps=False,
            )
            self._pipe      = pipe
            self.model_name = model_name
            latency_hint    = _MODEL_LATENCY.get(model_name, "unknown latency")
            log.info(
                f"Whisper ready: {model_name}  "
                f"device={'cuda' if device == 0 else 'cpu'}  "
                f"latency={latency_hint}"
            )
            return True
        except Exception as exc:
            log.warning(f"Failed to load {model_name}: {exc}")
            return False

    def _load(self) -> None:
        """
        Load Whisper with automatic downgrade fallback.

        Strategy:
          1. Try self.model_name (the caller's preferred model).
          2. If that fails, walk WHISPER_MODEL_PRIORITY from the top.
             Skip any model already tried (self.model_name).
          3. If every model in the chain fails, set error state.

        This means:
          • SpeechToText()                    → tries small, falls back to base/tiny
          • SpeechToText("openai/whisper-medium") → tries medium, falls back to small/base/tiny
          • SpeechToText("openai/whisper-tiny")   → tries tiny only (already at bottom)
        """
        if not TRANSFORMERS_OK:
            self._error_msg = (
                "transformers not installed. "
                "Run: pip install transformers"
            )
            log.warning(self._error_msg)
            return
        if not TORCH_OK:
            self._error_msg = (
                "torch not installed. "
                "Run: pip install torch"
            )
            log.warning(self._error_msg)
            return

        # Build the ordered list of models to try: preferred first, then chain
        already_tried: set = set()
        models_to_try: List[str] = [self.model_name]
        already_tried.add(self.model_name)

        for m in WHISPER_MODEL_PRIORITY:
            if m not in already_tried:
                models_to_try.append(m)
                already_tried.add(m)

        preferred = self.model_name
        for model_name in models_to_try:
            if self._try_load_model(model_name):
                self._ready     = True
                self._error_msg = ""
                if model_name != preferred:
                    self._downgraded = True
                    log.warning(
                        f"Whisper downgraded: {preferred} → {model_name}. "
                        f"Consider increasing system RAM for better accuracy."
                    )
                return

        # All models failed
        self._error_msg = (
            f"All Whisper models failed to load "
            f"(tried: {', '.join(models_to_try)}). "
            f"Run: pip install transformers torch"
        )
        log.error(self._error_msg)

    # ── Public API ────────────────────────────────────────────────────────────

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        bytes → transcribed text string, or a descriptive error string.

        Error strings are always wrapped in square brackets so callers
        can detect them: text.startswith("[") and text.endswith("]").
        """
        if not audio_bytes or len(audio_bytes) < 100:
            return ""
        if not self._ready:
            return f"[STT unavailable: {self._error_msg}]"

        audio, sr = decode_audio(audio_bytes)
        if audio is None:
            return "[Could not decode audio — try WAV format or install pydub+ffmpeg]"
        if is_silent(audio):
            return "[Silent recording — speak clearly into the microphone]"
        if len(audio) < sr * 0.5:
            return "[Recording too short — hold record while speaking]"

        try:
            result = self._pipe({"sampling_rate": sr, "raw": audio})
            text   = result.get("text", "").strip()
            log.info(f"Transcribed ({len(audio)/sr:.1f}s): {text[:80]}")
            return text if text else "[No speech detected]"
        except Exception as exc:
            log.error(f"Whisper inference error: {exc}")
            return f"[Transcription error: {exc}]"

    def transcribe_file(self, path: str) -> str:
        """Convenience wrapper: read a file and transcribe its bytes."""
        try:
            with open(path, "rb") as f:
                return self.transcribe(f.read())
        except Exception as exc:
            return f"[File error: {exc}]"

    def switch_model(self, model_name: str) -> bool:
        """
        Hot-swap to a different Whisper model at runtime without
        restarting the app.  Called from the Streamlit Settings page
        when the user changes the model selector.

        Returns True if the switch succeeded, False if it failed
        (original model remains active on failure).

        Usage in app.py / Settings:
            if st.selectbox("Whisper model", [...]) != stt.model_name:
                ok = stt.switch_model(new_model)
                st.success(...) if ok else st.error(...)
        """
        old_pipe        = self._pipe
        old_model_name  = self.model_name
        old_ready       = self._ready

        self._ready      = False
        self._downgraded = False

        # Walk the downgrade chain (mirrors _load() behaviour):
        # try the requested model first, then fall back through
        # WHISPER_MODEL_PRIORITY until one succeeds.
        already_tried: set = {model_name}
        models_to_try: List[str] = [model_name]
        for m in WHISPER_MODEL_PRIORITY:
            if m not in already_tried:
                models_to_try.append(m)
                already_tried.add(m)

        for candidate in models_to_try:
            if self._try_load_model(candidate):
                self._ready     = True
                self._error_msg = ""
                if candidate != model_name:
                    self._downgraded = True
                    log.warning(
                        f"switch_model: {model_name} failed; downgraded to {candidate}."
                    )
                else:
                    log.info(f"Whisper model switched: {old_model_name} → {candidate}")
                return True

        # All candidates failed — restore previous working state
        self._pipe      = old_pipe
        self.model_name = old_model_name
        self._ready     = old_ready
        self._error_msg = (
            f"Failed to load {model_name} (and all fallbacks); "
            f"still using {old_model_name}"
        )
        log.error(self._error_msg)
        return False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str:
        return self._error_msg

    @property
    def downgraded(self) -> bool:
        """True when a smaller fallback model is running instead of the preferred one."""
        return self._downgraded

    @property
    def latency_hint(self) -> str:
        """Human-readable CPU latency estimate for the active model."""
        return _MODEL_LATENCY.get(self.model_name, "unknown")

    @property
    def status(self) -> str:
        """
        One-line status string for display in the Streamlit sidebar / Settings.
        Examples:
            ✅ Whisper ready  (openai/whisper-small  •  ~4–5× latency)
            ⚠ Whisper downgraded  (whisper-medium → whisper-small  •  low RAM)
            ✗ STT unavailable: transformers not installed
        """
        if not self._ready:
            return f"✗ STT unavailable: {self._error_msg or 'Not loaded'}"
        base = f"✅ Whisper ready  ({self.model_name}  •  {self.latency_hint})"
        if self._downgraded:
            base += "  ⚠ downgraded (low RAM — see logs)"
        return base