"""
app.py — Aura AI | Multimodal Interview Coach (v9.1)
=====================================================
v8.0 upgrades:
  • SBERT relevance source badge in render_eval_results()
  • RL sequencer recommendation widget in Live Interview right panel
  • RL session report section in Final Report (action distribution chart,
    Q-table heatmap, step history table)
  • SBERT status in Model Setup + Settings
  • Session state: rl_report, rl_hint keys added
Design: Matrix-green / neon-cyan / electric-violet on deep black
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()

import json
import re
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")
sys.path.append(".")

# ── Continuous webcam capture strategy (v9.1) ────────────────────────────────
#
# PRIMARY — streamlit-webrtc (true 30fps WebRTC stream):
#   Requires: pip install "streamlit-webrtc==0.47.1" "aiortc==1.6.0" av
#   streamlit-webrtc ≥0.48 raises React error #62 with Streamlit ≥1.30 due
#   to breaking changes in how Streamlit registers React component contexts.
#   Pinning to 0.47.1 + aiortc 1.6.0 fixes this across Streamlit 1.28-1.35.
#
# FALLBACK — auto-rerun camera loop (no extra packages):
#   If streamlit-webrtc is not installed or raises a runtime error, the app
#   falls back to st.camera_input + st.rerun(). Each captured photo triggers
#   an immediate rerun that shows the camera again — creating a pseudo-
#   continuous loop at ~1-3fps (limited by Streamlit rerun latency).
#   This is enough for reliable emotion tracking without any extra installs.
#
# HOW TO CHOOSE:
#   • For best accuracy (30fps EMA, SEBR): install pinned webrtc packages
#   • For zero-dependency usage: skip install, fallback activates automatically
# ─────────────────────────────────────────────────────────────────────────────
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_OK = True
except (ImportError, Exception):
    WEBRTC_OK = False
    VideoProcessorBase = object     # dummy base so class def below compiles
_RTC_CONFIG = RTCConfiguration({"iceServers": [
    {"urls": ["stun:stun.l.google.com:19302"]},
    {"urls": ["stun:stun1.l.google.com:19302"]},
]}) if WEBRTC_OK else None

from backend_engine        import InterviewEngine, DISC_KEYWORDS, ScoreAggregator
from dataset_loader        import bytes_to_bgr, pil_to_bgr, EMOTION_LABELS, EMOTION_COLORS
from speech_to_text        import SpeechToText
from answer_evaluator      import AnswerEvaluator
from voice_input           import voice_input_panel
from unified_voice_pipeline import UnifiedVoicePipeline as NervousnessPipeline
from resume_rephraser       import page_resume, RESUME_DEFAULTS
from weekly_prep_plan       import page_weekly_plan, WEEKLY_PLAN_DEFAULTS   # v10.1

# ── Placement Test ────────────────────────────────────────────────────────────
try:
    from placement_test_mode import page_placement_test, PLACEMENT_DEFAULTS as _PT_DEFAULTS
    _PT_OK = True
except ImportError:
    _PT_OK = False
    _PT_DEFAULTS = {}
    def page_placement_test(): st.error("placement_test_mode.py not found.")

# ── Company Question Upload ───────────────────────────────────────────────────
try:
    from company_question_upload import (
        page_company_questions, COMPANY_UPLOAD_DEFAULTS as _CQ_DEFAULTS,
        page_placement_setup, PLACEMENT_SETUP_DEFAULTS as _PS_DEFAULTS,
    )
    _CQ_OK = True
except ImportError:
    _CQ_OK = False
    _CQ_DEFAULTS = {}
    _PS_DEFAULTS = {}
    def page_company_questions(): st.error("company_question_upload.py not found.")
    def page_placement_setup(): st.error("company_question_upload.py not found.")


# ── Feature 8 + 10: JD Engine + Company Profiles ─────────────────────────────
try:
    from jd_question_engine import (
        render_company_selector,
        JD_ENGINE_DEFAULTS,
    )
    JD_ENGINE_OK = True
except ImportError:
    JD_ENGINE_OK = False
    JD_ENGINE_DEFAULTS = {}
    def render_company_selector(): pass
from finish_interview       import _build_pdf
from hr_round               import page_hr_round
from live_coach             import generate_coaching_tip, render_coach_card, render_coach_settings
from follow_up_engine       import (
    render_follow_up_ui, generate_follow_up,
    FOLLOW_UP_DEFAULTS, FollowUpQuestion,
)

# ── v10.0: AI Avatar Interviewer ──────────────────────────────────────────────
try:
    from avatar_interviewer import render_avatar_interviewer
    AVATAR_OK = True
except ImportError:
    AVATAR_OK = False

# ── v8.0: SBERT availability flag (for UI status display) ────────────────────
try:
    from answer_evaluator import SBERT_AVAILABLE, SBERT_MODEL
except ImportError:
    SBERT_AVAILABLE = False
    SBERT_MODEL     = "all-MiniLM-L6-v2"

# ── v8.0: RL sequencer availability flag ─────────────────────────────────────
try:
    from adaptive_sequencer import RLAdaptiveSequencer
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONTINUOUS VIDEO PROCESSOR (v9.1)
# ══════════════════════════════════════════════════════════════════════════════

# ── Thread-safe shared state (WebRTC path) ────────────────────────────────────
_frame_lock  = threading.Lock()
_frame_state: Dict = {
    "annotated_bgr": None,
    "result":         {},
    "ready":          False,
    "frame_count":    0,
}


class AuraVideoProcessor(VideoProcessorBase):
    """
    WebRTC video processor. recv() is called ~30fps on a background thread.
    Results are written to _frame_state under _frame_lock so the Streamlit
    UI thread can read them safely on each re-render cycle.

    React error #62 fix: pin packages to compatible versions —
        pip install "streamlit-webrtc==0.47.1" "aiortc==1.6.0" av
    streamlit-webrtc ≥0.48 conflicts with Streamlit ≥1.30 React internals.
    """

    def recv(self, frame):
        """
        Called ~30fps by the WebRTC worker thread.
        av is imported lazily here so it never affects app startup.
        """
        import av as _av   # lazy — only runs when WebRTC stream is active
        img_bgr = frame.to_ndarray(format="bgr24")
        try:
            annotated_bgr, result = engine.analyse_webcam_frame(img_bgr)
        except Exception:
            annotated_bgr, result = img_bgr, {}
        with _frame_lock:
            _frame_state["annotated_bgr"] = annotated_bgr
            _frame_state["result"]        = result
            _frame_state["ready"]         = True
            _frame_state["frame_count"]  += 1
        return _av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")


def _read_frame_state() -> tuple:
    """Thread-safe read. Returns (annotated_bgr | None, result_dict, frame_count)."""
    with _frame_lock:
        return (
            _frame_state["annotated_bgr"],
            dict(_frame_state["result"]),
            _frame_state["frame_count"],
        )


def _reset_frame_state() -> None:
    """Clear stale WebRTC state at session start."""
    with _frame_lock:
        _frame_state["annotated_bgr"] = None
        _frame_state["result"]        = {}
        _frame_state["ready"]         = False
        _frame_state["frame_count"]   = 0


# ── Auto-rerun camera loop (fallback path, no extra packages) ─────────────────
# When WEBRTC_OK is False, the UI renders st.camera_input. On each new photo
# it processes the frame then immediately calls st.rerun(), which renders the
# camera input again. This creates a pseudo-continuous loop at ~1-3fps —
# slower than true WebRTC but fully reliable with zero extra dependencies.

def _run_camera_loop(qn: int) -> Optional[Dict]:
    """
    Auto-rerun camera capture loop (fallback when streamlit-webrtc unavailable).

    Shows st.camera_input. When a new photo is captured:
      1. Runs full FER + MediaPipe pipeline on the frame.
      2. Updates all session state signals.
      3. Calls st.rerun() — which re-renders the camera widget immediately,
         creating the next capture opportunity automatically.

    Returns the latest result dict, or None if no frame yet.
    """
    cam_img = st.camera_input(
        "", key=f"cam_loop_{qn}", label_visibility="collapsed"
    )
    if cam_img is None:
        return None

    # Deduplicate: skip if same image bytes as last processed frame
    img_bytes = cam_img.getvalue()
    img_hash  = hash(img_bytes)
    if img_hash == st.session_state.get("_cam_last_hash"):
        return st.session_state.get("_cam_last_result")

    ann_bytes, result = process_camera_frame(cam_img)

    if ann_bytes and result:
        st.image(ann_bytes, channels="BGR", use_container_width=True)

        dom     = result.get("dominant", "Neutral")
        nerv    = result.get("smoothed_nervousness", result.get("nervousness", 0.2))
        posture = result.get("posture", {})
        ear_val = result.get("ear", posture.get("ear", 0.28))

        st.session_state.live_emotion     = dom
        st.session_state.live_nervousness = nerv
        hist = st.session_state.emotion_history
        hist.append(dom)
        if len(hist) > 80:
            hist.pop(0)
        st.session_state.emotion_history = hist
        if posture:
            st.session_state.live_posture = posture

        mm = engine.get_multimodal_confidence()
        st.session_state.live_confidence   = mm.get("confidence_score", 3.5)
        st.session_state["_cam_last_hash"] = img_hash
        st.session_state["_cam_last_result"] = result

        # Immediately trigger next capture cycle
        st.rerun()

    return result


st.set_page_config(
    page_title="AURA AI · Interview Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CACHED SINGLETONS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_engine() -> InterviewEngine:
    return InterviewEngine("question_pool.json")

@st.cache_resource
def get_stt() -> SpeechToText:
    return SpeechToText("openai/whisper-base")

@st.cache_resource
def get_evaluator() -> AnswerEvaluator:
    return AnswerEvaluator()

@st.cache_resource
def get_nervousness_pipeline() -> NervousnessPipeline:
    return NervousnessPipeline()

engine        = get_engine()
stt           = get_stt()
evaluator     = get_evaluator()
nerv_pipeline = get_nervousness_pipeline()

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
DEFAULTS: Dict = {
    "page"                : "Dashboard",
    "question"            : None,
    "q_number"            : 0,
    "transcript"          : "",
    "transcribed_text"    : "",
    "last_audio_id"       : None,
    "last_audio_bytes"    : None,   # v9.2: raw WAV bytes from last recording
    "last_audio_source"   : None,   # v9.2: "whisper_mic" | "file_upload" | None
    "candidate_name"      : "Candidate",
    "target_role"         : "Software Engineer",
    "target_company"      : "No specific company",
    "difficulty"          : "Medium",   # Easy | Medium | Hard | All (RL Adaptive)
    "num_questions"       : 5,
    "last_score"          : None,
    "last_feedback"       : "",
    "last_star"           : {},
    "last_keywords"       : [],
    "last_eval"           : {},
    "session_answers"     : [],
    "submitted"           : False,
    "q_start_time"        : None,
    "q_is_follow_up"      : False,   # v9.2: True when current Q is a follow-up sub-part
    "webcam_enabled"      : True,
    "fer_ready"           : False,
    "voice_ready"         : False,
    "pipeline_metrics"    : {},
    "voice_metrics"       : {},
    "live_emotion"        : "Neutral",
    "live_nervousness"    : 0.2,
    "live_emotion_dist"   : {},
    "emotion_history"     : [],
    "live_voice_emotion"  : "Neutral",
    "live_voice_nerv"     : 0.2,
    "live_posture"        : {},
    "live_confidence"     : 3.5,
    "stt_ready"           : False,
    "nervousness_ready"   : False,
    "nervousness_metrics" : {},
    # Resume rephraser integration
    "custom_resume_questions" : [],   # full question dicts from resume_rephraser
    "resume_mode_active"      : False,# True when live interview is running resume Qs
    "resume_q_index"          : 0,    # current index into custom_resume_questions
    # Live coach
    "last_coaching_tip"       : "",
    "last_improvement"        : "",   # v13.0 improvement suggestion
    "coach_auto_speak"        : False,
    "coach_voice_rate"        : 1.0,
    "coach_voice_pitch"       : 1.0,
    # v8.0 RL sequencer
    "rl_report"               : {},    # saved after each session finish
    "rl_hint"                 : {},    # current RL recommendation
    "show_rl_hint"            : True,  # kept for backward compat — always True now
    # v9.0 WebRTC frame counter (prevents duplicate emotion_history entries)
    "_last_frame_count"        : 0,
    "_cam_last_hash"           : None,   # v9.1 auto-rerun dedup
    "_cam_last_result"         : None,
    # v8.0 SBERT
    "sbert_status"            : "unknown",
    # Navigation history for Back button
    "page_history"            : [],
    # v10.0 AI Avatar Interviewer
    "avatar_enabled"          : True,   # show ARIA-7 avatar above question card
    "avatar_auto_speak"       : True,   # auto-read question aloud on load
    # DISC radar spin — bumped after each answer eval to trigger spin animation
    "_disc_spin_version"      : 0,
    # ── Blind scoring (Google-style) ──────────────────────────────────────
    "blind_mode"              : False,  # True = hide scores during interview
    "blind_scores"            : {},     # {q_number: eval_dict} sealed during session
    "blind_revealed"          : False,  # True once Final Report has merged them

}
DEFAULTS.update(RESUME_DEFAULTS)
DEFAULTS.update(FOLLOW_UP_DEFAULTS)
DEFAULTS.update(JD_ENGINE_DEFAULTS)
DEFAULTS.update(WEEKLY_PLAN_DEFAULTS)          # v10.1 — Weekly Prep Plan
if _PT_OK:
    DEFAULTS.update(_PT_DEFAULTS)              # Placement Test session state
if _CQ_OK:
    DEFAULTS.update(_CQ_DEFAULTS)              # Company Question Upload session state
    DEFAULTS.update(_PS_DEFAULTS)              # Placement Setup page session state
CALIBRATION_DEFAULTS = {
    "calibration_done":     False,
    "calibration_baseline": 0.2,
    "calibration_skipped":  False,
}
DEFAULTS.update(CALIBRATION_DEFAULTS)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state.stt_ready and stt.ready:
    st.session_state.stt_ready = True
if not st.session_state.nervousness_ready and nerv_pipeline.ready:
    st.session_state.nervousness_ready = True
    st.session_state.nervousness_metrics = nerv_pipeline.get_metrics()


# ══════════════════════════════════════════════════════════════════════════════
#  NAV — single source of truth, uses st.rerun() safely
# ══════════════════════════════════════════════════════════════════════════════
# Linear interview flow — determines wipe direction
_PAGE_ORDER = [
    "Dashboard",
    "Model Setup",
    "Start Interview",
    "Live Interview",
    "Final Report",
]

def _nav_direction(from_page: str, to_page: str) -> str:
    """Return 'fwd' if moving forward in the interview flow, else 'bwd'."""
    try:
        return "fwd" if _PAGE_ORDER.index(to_page) > _PAGE_ORDER.index(from_page) else "bwd"
    except ValueError:
        return "fwd"  # pages outside linear flow (Settings, HR Practice, etc.) default fwd


def nav(page: str) -> None:
    history = st.session_state.get("page_history", [])
    current = st.session_state.page
    if current != page:
        st.session_state["_nav_dir"] = _nav_direction(current, page)
        history.append(current)
        if len(history) > 20:
            history = history[-20:]
        st.session_state.page_history = history
    st.session_state.page = page

def nav_to(page: str):
    """Return an on_click callback that navigates to page."""
    def _cb():
        history = st.session_state.get("page_history", [])
        current = st.session_state.page
        if current != page:
            st.session_state["_nav_dir"] = _nav_direction(current, page)
            history.append(current)
            if len(history) > 20:
                history = history[-20:]
            st.session_state.page_history = history
        st.session_state.page = page
    return _cb



def render_top_navbar() -> None:
    """
    Renders the Aura Focus top navbar: Back button (left) + Aura AI logo +
    wizard progress bar (center) + nav links (right).
    Sidebar has been removed; navigation is fully top-bar driven.
    """
    p = st.session_state.page
    history = st.session_state.get("page_history", [])
    has_back = len(history) > 0

    # Map page → wizard step
    step = 1
    if p in ("Live Interview",):
        step = 2
    elif p in ("Final Report",):
        step = 3

    steps_html = ""
    labels = ["Setup", "Interview", "Review"]
    for i, lbl in enumerate(labels, 1):
        if i < step:
            dot = f'<span style="width:10px;height:10px;border-radius:50%;background:#6366f1;display:inline-block;margin-right:5px;"></span>'
            txt = f'<span style="color:#a5b4fc;font-weight:500;">{lbl}</span>'
        elif i == step:
            dot = f'<span style="width:10px;height:10px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#8b5cf6);display:inline-block;margin-right:5px;box-shadow:0 0 0 3px rgba(99,102,241,.25);"></span>'
            txt = f'<span style="color:#f1f5f9;font-weight:700;">{lbl}</span>'
        else:
            dot = f'<span style="width:10px;height:10px;border-radius:50%;background:rgba(255,255,255,.12);display:inline-block;margin-right:5px;"></span>'
            txt = f'<span style="color:#475569;">{lbl}</span>'

        steps_html += f'<span style="display:inline-flex;align-items:center;font-size:.82rem;font-family:Inter,sans-serif;">{dot}{txt}</span>'
        if i < 3:
            line_col = "rgba(99,102,241,.4)" if i < step else "rgba(255,255,255,.1)"
            steps_html += f'<span style="display:inline-block;width:50px;height:1px;background:{line_col};margin:0 8px;vertical-align:middle;"></span>'

    # Role-specific accent for navbar bottom border
    _role_accent, _role_glow, _ = get_role_theme(
        st.session_state.get("target_role", "Software Engineer") or "Software Engineer"
    )

    st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
  padding:.65rem 1.4rem;background:rgba(15,23,42,.96);
  backdrop-filter:blur(20px);border-bottom:2px solid {_role_accent};
  box-shadow:0 2px 20px {_role_glow};
  position:sticky;top:0;z-index:999;margin-bottom:1.2rem;gap:1rem;">

  <!-- Left: Logo -->
  <div style="display:flex;align-items:center;gap:.9rem;flex-shrink:0;">
    <div style="display:flex;align-items:center;gap:.6rem;">
      <div style="width:36px;height:36px;border-radius:50%;border:2px solid rgba(99,102,241,.7);
        display:flex;align-items:center;justify-content:center;
        box-shadow:0 0 14px rgba(99,102,241,.35);flex-shrink:0;">
        <div style="width:14px;height:14px;border-radius:50%;background:transparent;
          border:2.5px solid #a5b4fc;"></div>
      </div>
      <span style="font-size:1.3rem;font-family:Inter,sans-serif;font-weight:800;letter-spacing:-.02em;line-height:1;">
        <span style="color:#a5b4fc;">Aura</span><span style="color:#f1f5f9;"> AI</span>
      </span>
    </div>
  </div>

  <!-- Centre: Wizard bar -->
  <div style="display:flex;align-items:center;flex-shrink:0;">{steps_html}</div>

  <!-- Right: spacer to keep wizard centred -->
  <div style="width:80px;"></div>
</div>
""", unsafe_allow_html=True)

    # Back button — floated into the navbar via fixed positioning so it sits
    # inside the top bar without consuming any page space below it.
    if has_back:
        st.markdown("""
<style>
/* Float the Back button into the navbar — zero extra page space consumed */
.back-btn-navbar {
    position: fixed;
    top: 0;
    right: 1.4rem;
    z-index: 10000;
    height: 56px;          /* match navbar height */
    display: flex;
    align-items: center;
}
.back-btn-navbar > div[data-testid="stButton"] > button {
    background: rgba(99,102,241,.12) !important;
    border: 1px solid rgba(99,102,241,.35) !important;
    color: #a5b4fc !important;
    font-family: Inter, sans-serif !important;
    font-size: .78rem !important;
    font-weight: 600 !important;
    letter-spacing: .02em !important;
    text-transform: none !important;
    padding: .38rem 1rem !important;
    width: auto !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    transform: none !important;
    animation: none !important;
    line-height: 1.4 !important;
}
.back-btn-navbar > div[data-testid="stButton"] > button:hover {
    background: rgba(99,102,241,.25) !important;
    border-color: rgba(99,102,241,.6) !important;
}
</style>
""", unsafe_allow_html=True)
        st.markdown('<div class="back-btn-navbar">', unsafe_allow_html=True)
        if st.button("← Back", key="_back_btn_"):
            history = st.session_state.get("page_history", [])
            if history:
                prev = history.pop()
                st.session_state.page_history = history
                st.session_state["_nav_dir"] = "bwd"
                st.session_state.page = prev
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def _sync_session_for_report() -> None:
    """
    Bridge: maps session_answers (app.py /5 scale) into the keys that
    finish_interview.py / _build_pdf expects before rendering the report.
    Safe to call multiple times — always overwrites.

    v9.2: follow-up entries (is_follow_up=True) are excluded from the main
    questions/answers/scores lists that feed the Q-by-Q PDF loop.
    They are handled separately as sub-entries via follow_up_records.
    """
    answers = st.session_state.get("session_answers", [])
    # v9.2: only original questions go into the PDF Q-by-Q loop
    primary = [a for a in answers if not a.get("is_follow_up", False)]

    st.session_state["interview_questions"] = [a.get("question", "") for a in primary]
    st.session_state["interview_answers"]   = [a.get("answer",   "") for a in primary]
    # Convert score /5 -> /10 for finish_interview.py
    st.session_state["answer_scores"] = [
        {"score": a.get("score", 0) * 2, "feedback": a.get("feedback", "")}
        for a in primary
    ]
    st.session_state["emotion_history"] = [
        {"dominant": a.get("emotion", "Neutral"), "nervousness": a.get("nervousness", 0.2)}
        for a in primary
    ]
    if not st.session_state.get("job_role"):
        st.session_state["job_role"] = st.session_state.get("target_role", "")
    import time as _t
    if "interview_start_time" in st.session_state and "session_duration_s" not in st.session_state:
        st.session_state["session_duration_s"] = _t.time() - st.session_state["interview_start_time"]


# ══════════════════════════════════════════════════════════════════════════════
#  CYBERSECURITY × AI DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
def inject_css() -> None:
    # Inject Google Fonts separately (no style tag — avoids render-as-text issue)
    st.markdown(
        '<link href="https://fonts.googleapis.com/css2?'
        'family=Inter:wght@300;400;500;600;700'
        '&family=Orbitron:wght@500;700'
        '&family=Share+Tech+Mono'
        '&family=JetBrains+Mono:wght@400;500'
        '&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    
    css = """<div style="display:none" hidden><style>
/* ══ AURA FOCUS THEME ══════════════════════════════════════════ */
:root{--mut:#8fc4e0;--mut-dim:#6a96b8;}
html,body,[class*="css"]{font-family:Inter,sans-serif!important;font-size:15px;color:#e2e8f0;}
.stApp{
  background-image:
    url("https://raw.githubusercontent.com/KrishnaMittal4/Mini-Project/a6874c2754bd4d48f11d5284fefcc15bcb643ea1/46add291-41f4-4cbe-b6a1-ecc107a9a180.png"),
    linear-gradient(160deg,#000512 0%,#000f28 40%,#00081c 100%);
  background-size:cover;
  background-position:center;
  background-attachment:fixed;
  min-height:100vh;
}
/* Subtle dark overlay — low opacity so the photocreatic bg image shows through */
.stApp::before{
  content:'';
  position:fixed;top:0;left:0;width:100%;height:100%;
  background:linear-gradient(160deg,rgba(0,5,18,0.62) 0%,rgba(0,15,40,0.57) 40%,rgba(0,8,28,0.65) 100%);
  z-index:0;pointer-events:none;
}
/* Raise all Streamlit content above the overlay */
.stApp>*{position:relative;z-index:1;}
[data-testid="stSidebar"]{z-index:100!important;}
#MainMenu,footer,header{visibility:hidden;}
h1,h2,h3,h4,h5{font-family:Inter,sans-serif!important;color:#f8fafc!important;letter-spacing:-.01em;font-weight:600;text-transform:none!important;}

/* ── GLASS CARDS ── */
.card{background:rgba(0,12,30,0.55);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border:1px solid rgba(0,240,255,0.15);border-radius:16px;padding:1.4rem;margin-bottom:1rem;position:relative;overflow:hidden;transition:border-color .2s,box-shadow .2s;box-shadow:0 0 0 1px rgba(0,0,0,0.3),0 20px 50px rgba(0,0,0,0.6);}
.card:hover{border-color:rgba(0,240,255,0.4);box-shadow:0 0 0 1px rgba(0,0,0,0.3),0 25px 60px rgba(0,0,0,0.7),0 0 30px rgba(0,240,255,0.07);}
.card::before{display:none!important;}
.card::after{display:none!important;}
.card-sm{padding:.85rem;border-radius:12px;}
.mchip{background:rgba(255,255,255,.06);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,.1);border-radius:12px;padding:.9rem .65rem;text-align:center;position:relative;}
.mchip::before{display:none!important;}
.mchip-v{font-family:Inter,sans-serif;font-size:1.7rem;font-weight:700;color:#a5b4fc;line-height:1.1;}
.mchip-l{font-size:.62rem;color:#b4cde4;text-transform:uppercase;letter-spacing:.1em;font-weight:500;margin-top:4px;}

/* ── SCORE BADGES (inline pills) ── */
.badge{padding:3px 12px;border-radius:20px;font-size:.72rem;font-weight:700;display:inline-block;font-family:Inter,sans-serif;text-transform:none;letter-spacing:0;}
.b-ex{background:rgba(16,185,129,.2);color:#34d399;border:1px solid rgba(16,185,129,.45);}
.b-gd{background:rgba(99,102,241,.2);color:#c4b5fd;border:1px solid rgba(99,102,241,.45);}
.b-av{background:rgba(245,158,11,.2);color:#fcd34d;border:1px solid rgba(245,158,11,.45);}
.b-po{background:rgba(239,68,68,.2);color:#fca5a5;border:1px solid rgba(239,68,68,.45);}

/* ── ALERT BANNERS ── */
.b-warn{background:rgba(245,158,11,.15);border:1px solid rgba(245,158,11,.5);border-left:4px solid #f59e0b;padding:.7rem 1.1rem;border-radius:8px;color:#fcd34d;margin:.5rem 0;font-size:.85rem;font-family:Inter,sans-serif;font-weight:500;}
.b-info{background:rgba(99,102,241,.15);border:1px solid rgba(99,102,241,.4);border-left:4px solid #6366f1;padding:.7rem 1.1rem;border-radius:8px;color:#c4b5fd;margin:.5rem 0;font-size:.85rem;font-family:Inter,sans-serif;font-weight:500;}
.b-ok{background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.4);border-left:4px solid #10b981;padding:.7rem 1.1rem;border-radius:8px;color:#6ee7b7;margin:.5rem 0;font-size:.85rem;font-family:Inter,sans-serif;font-weight:500;}
.b-err{background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.4);border-left:4px solid #ef4444;padding:.7rem 1.1rem;border-radius:8px;color:#fca5a5;margin:.5rem 0;font-size:.85rem;font-family:Inter,sans-serif;font-weight:500;}

/* ── BUTTONS ── */
.stButton>button{font-family:Orbitron,monospace!important;font-size:.82rem!important;font-weight:700!important;letter-spacing:.08em!important;text-transform:uppercase!important;color:#000820!important;background:linear-gradient(135deg,#00F0FF 0%,#0070E0 100%)!important;border:none!important;border-radius:10px!important;padding:.85rem 1.5rem!important;width:100%!important;cursor:pointer!important;transition:all .25s!important;box-shadow:0 0 22px rgba(0,240,255,0.35),0 4px 14px rgba(0,0,0,0.4)!important;animation:btn-glow-pulse 2.8s ease-in-out infinite!important;}
.stButton>button:hover{background:linear-gradient(135deg,#20F5FF 0%,#0088FF 100%)!important;box-shadow:0 0 38px rgba(0,240,255,0.6),0 6px 20px rgba(0,0,0,0.5)!important;transform:translateY(-2px)!important;animation-play-state:paused!important;}
.stButton>button:active{transform:translateY(0)!important;}
.stDownloadButton>button{background:rgba(255,255,255,.08)!important;border:1px solid rgba(255,255,255,.15)!important;color:#cbd5e1!important;font-family:Inter,sans-serif!important;border-radius:10px!important;box-shadow:none!important;}

/* ── SIDEBAR — fully hidden ── */
[data-testid="stSidebar"]{display:none!important;}
[data-testid="collapsedControl"]{display:none!important;}
section[data-testid="stSidebar"]{display:none!important;}

.sb-section-lbl{font-size:.68rem;letter-spacing:.1em;color:#818cf8;font-family:Inter,sans-serif;text-transform:uppercase;font-weight:600;padding:.7rem .3rem .3rem;display:block;border-bottom:1px solid rgba(129,140,248,.15);margin-bottom:.3rem;}
.sb-nav-btn>button{background:transparent!important;border:1px solid transparent!important;border-radius:8px!important;color:#b4cde4!important;font-family:Inter,sans-serif!important;font-size:.875rem!important;font-weight:500!important;letter-spacing:0!important;text-transform:none!important;text-align:left!important;padding:12px 16px!important;width:100%!important;box-shadow:none!important;transition:all .2s!important;}
.sb-nav-btn>button:hover{background:rgba(99,102,241,.1)!important;color:#e2e8f0!important;transform:none!important;box-shadow:none!important;}
.sb-nav-active>button{background:rgba(99,102,241,.12)!important;color:#a5b4fc!important;border:1px solid rgba(99,102,241,.3)!important;box-shadow:none!important;transform:none!important;font-weight:600!important;font-size:.875rem!important;padding:12px 16px!important;}
.sb-col-btn>button{background:transparent!important;border:1px solid rgba(255,255,255,.1)!important;border-radius:6px!important;color:#64748b!important;font-size:.7rem!important;padding:2px 8px!important;width:auto!important;min-width:0!important;box-shadow:none!important;transform:none!important;}
.sb-col-btn>button:hover{border-color:rgba(99,102,241,.5)!important;color:#a5b4fc!important;}

/* ── PROGRESS ── */
.stProgress>div>div{background:linear-gradient(90deg,#6366f1,#8b5cf6)!important;border-radius:4px!important;}
.stProgress>div{background:rgba(255,255,255,.08)!important;border-radius:4px!important;}

/* ── FORM INPUTS ── */
.stTextInput>div>div>input,.stTextArea>div>div>textarea{background:rgba(0,8,22,0.75)!important;border:1px solid rgba(0,240,255,0.15)!important;border-radius:10px!important;color:#f1f5f9!important;font-family:Inter,sans-serif!important;font-size:.95rem!important;}
.stTextInput>div>div>input:focus,.stTextArea>div>div>textarea:focus{border-color:rgba(0,240,255,0.55)!important;box-shadow:0 0 0 3px rgba(0,240,255,0.12)!important;}

/* ── ANIMATED FORM LABELS (v2) ── */
/* Hide Streamlit native labels — custom animated ones rendered via st.markdown() */
.stTextInput    label,
.stTextInput    [data-testid="stWidgetLabel"],
.stTextArea     label,
.stTextArea     [data-testid="stWidgetLabel"],
.stSelectbox    label,
.stSelectbox    [data-testid="stWidgetLabel"],
.stSlider       label,
.stSlider       [data-testid="stWidgetLabel"],
.stSelectSlider label,
div[data-testid="stSelectSlider"] label,
div[data-testid="stSelectSlider"] [data-testid="stWidgetLabel"] {
  display:none!important;
}
/* Keep toggle labels visible */
.stToggle label,
.stToggle label p,
.stToggle [data-testid="stWidgetLabel"],
.stToggle [data-testid="stWidgetLabel"] p,
div[data-testid="stToggle"] label,
div[data-testid="stToggle"] label p,
div[data-testid="stToggle"] [data-testid="stWidgetLabel"],
div[data-testid="stToggle"] [data-testid="stWidgetLabel"] p {
  display:block!important;
  color:#c7e8ff!important;
  font-family:'Share Tech Mono',monospace!important;
  font-size:.82rem!important;
  letter-spacing:.08em!important;
  opacity:1!important;
}
/* Base animated label */
.aura-lbl {
  display:flex;align-items:center;gap:7px;
  font-family:'Share Tech Mono',monospace;
  font-size:10px;letter-spacing:0.20em;
  margin-bottom:6px;margin-top:8px;position:relative;width:fit-content;
  text-shadow:0 1px 8px rgba(0,0,0,0.9),0 0 2px rgba(0,0,0,1);
  filter:drop-shadow(0 0 1px rgba(0,0,0,0.8));
}
@keyframes aura-lbl-in {
  from{opacity:0;transform:translateX(-8px);}
  to{opacity:1;transform:translateX(0);}
}
@keyframes aura-ul-grow {
  from{width:0;} to{width:100%;}
}
@keyframes aura-dot-pulse {
  0%,100%{opacity:1;transform:scale(1);}
  50%{opacity:0.3;transform:scale(0.55);}
}
@keyframes aura-bracket-in {
  from{opacity:0;transform:translateY(-4px);}
  to{opacity:1;transform:none;}
}
@keyframes aura-chip-in {
  from{opacity:0;transform:scale(0.75);}
  to{opacity:1;transform:scale(1);}
}
/* GREEN — Candidate Name */
.aura-lbl-green{
  color:#00ff88;
  text-shadow:0 0 10px rgba(0,255,136,0.55);
  animation:aura-lbl-in 0.45s ease both;
}
.aura-lbl-green::after{
  content:'';position:absolute;bottom:-3px;left:0;
  height:1px;width:0;background:#00ff88;
  box-shadow:0 0 6px #00ff88;
  animation:aura-ul-grow 0.55s cubic-bezier(.4,0,.2,1) 0.35s both;
}
.aura-lbl-green .aura-dot{
  width:5px;height:5px;border-radius:50%;
  background:#00ff88;box-shadow:0 0 7px #00ff88;
  flex-shrink:0;animation:aura-dot-pulse 2s ease-in-out infinite;
}
/* CYAN — Target Role / Target Company */
.aura-lbl-cyan{
  color:#00d4ff;
  text-shadow:0 0 8px rgba(0,212,255,0.5);
  padding:0 10px;
  animation:aura-lbl-in 0.45s ease 0.05s both;
}
.aura-lbl-cyan::before{
  content:'[';position:absolute;left:0;
  color:#00d4ff;opacity:0;
  animation:aura-bracket-in 0.3s ease 0.45s both;
}
.aura-lbl-cyan::after{
  content:']';position:absolute;right:0;
  color:#00d4ff;opacity:0;
  animation:aura-bracket-in 0.3s ease 0.6s both;
}
.aura-lbl-cyan .aura-opt{
  font-size:8px;color:rgba(0,212,255,0.4);letter-spacing:0.1em;
}
/* VIOLET — Difficulty Level */
.aura-lbl-violet{
  color:#c990ff;
  text-shadow:0 0 8px rgba(180,100,255,0.4);
  animation:aura-lbl-in 0.45s ease 0.10s both;
}
.aura-lbl-violet::after{
  content:'';position:absolute;bottom:-3px;left:0;
  height:1px;width:0;background:rgba(180,100,255,0.6);
  animation:aura-ul-grow 0.5s ease 0.4s both;
}
.aura-lbl-violet .aura-chip{
  display:inline-flex;align-items:center;
  border-radius:2px;padding:1px 7px;
  font-size:8px;letter-spacing:0.10em;
  animation:aura-chip-in 0.35s ease 0.75s both;
}
/* AMBER — Number of Questions */
.aura-lbl-amber{
  color:#ffb400;
  text-shadow:0 0 8px rgba(255,180,0,0.4);
  animation:aura-lbl-in 0.45s ease 0.15s both;
}
.aura-lbl-amber::after{
  content:'';position:absolute;bottom:-3px;left:0;
  height:1px;width:0;background:rgba(255,180,0,0.5);
  animation:aura-ul-grow 0.5s ease 0.4s both;
}
.aura-lbl-amber .aura-badge{
  display:inline-flex;align-items:center;
  border-left:2px solid #ffb400;border-radius:0;
  padding:0 6px;font-size:9px;letter-spacing:0.08em;
  color:#ffb400;animation:aura-chip-in 0.35s ease 0.75s both;
}

.stSelectbox>div>div,.stSlider>div{color:#f1f5f9!important;}
.stSelectbox [data-baseweb="select"] *{background:rgba(0,8,22,0.85)!important;color:#f1f5f9!important;}
.stSelectbox [data-baseweb="select"] [data-baseweb="base-input"]{background:rgba(0,8,22,0.75)!important;color:#f1f5f9!important;border:1px solid rgba(0,240,255,0.15)!important;border-radius:10px!important;}
.stSelectbox [data-baseweb="select"] svg{fill:#ff4444!important;}
.stSelectbox [data-baseweb="select"] [role="option"]{background:rgba(0,15,40,.95)!important;color:#e2e8f0!important;}
.stSelectbox [data-baseweb="select"] [role="option"]:hover{background:rgba(0,240,255,.1)!important;color:#00F0FF!important;}
[data-baseweb="select"]>div{background:rgba(0,8,22,0.75)!important;border:1px solid rgba(0,240,255,0.15)!important;border-radius:10px!important;}
[data-baseweb="select"] input{background:transparent!important;color:#f1f5f9!important;}
[data-baseweb="select"] span{color:#f1f5f9!important;}
[data-baseweb="menu"]{background:rgba(0,15,40,.97)!important;border:1px solid rgba(0,240,255,.2)!important;border-radius:10px!important;backdrop-filter:blur(12px)!important;}
[data-baseweb="menu"] li{color:#e2e8f0!important;}
[data-baseweb="menu"] li:hover,[data-baseweb="menu"] li[aria-selected="true"]{background:rgba(0,240,255,.12)!important;color:#00F0FF!important;}
.stSlider [data-baseweb="slider"] [data-testid="stSliderThumb"]{background:#00F0FF!important;box-shadow:0 0 10px #00F0FF!important;}
.stSlider [data-baseweb="slider"] div[role="slider"]{background:#00F0FF!important;}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child{background:rgba(0,240,255,0.15)!important;}
.streamlit-expanderHeader,.st-expander summary,[data-testid="stExpander"] summary,[data-testid="stExpanderToggleIcon"]{background:rgba(0,12,30,.7)!important;border:1px solid rgba(0,240,255,.18)!important;border-radius:10px!important;color:#ffffff!important;font-family:Inter,sans-serif!important;font-size:.82rem!important;font-weight:600!important;}
[data-testid="stExpander"] summary p,[data-testid="stExpander"] summary span{color:#ffffff!important;font-weight:600!important;}
[data-testid="stExpander"] summary svg{fill:#ffffff!important;}

/* ── QUESTION CARD ── */
.qcard{background:rgba(10,20,45,0.72);backdrop-filter:blur(20px);border:1px solid rgba(99,102,241,0.28);border-left:4px solid var(--qcard-accent,#6366f1);border-radius:14px;padding:1.4rem 1.6rem;margin-bottom:1rem;position:relative;overflow:hidden;box-shadow:0 0 32px rgba(99,102,241,0.08),0 8px 32px rgba(0,0,0,0.5);}
/* Scan-line sweep on top edge */
.qcard::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent 0%,var(--qcard-accent,#6366f1) 40%,rgba(0,240,255,0.9) 60%,transparent 100%);animation:qcard-scan 3.5s ease-in-out infinite;transform:scaleX(0);transform-origin:left;}
@keyframes qcard-scan{0%,100%{transform:scaleX(0);opacity:0;}10%{transform:scaleX(0);opacity:1;}60%{transform:scaleX(1);opacity:1;}80%,100%{transform:scaleX(1);opacity:0;}}
/* Corner bracket decoration */
.qcard::after{content:'';position:absolute;bottom:12px;right:12px;width:18px;height:18px;border-right:2px solid rgba(99,102,241,0.25);border-bottom:2px solid rgba(99,102,241,0.25);border-radius:0 0 4px 0;}

/* ── DIFFICULTY BADGE PULSE ── */
@keyframes diff-badge-breathe{0%,100%{box-shadow:0 0 0 0 rgba(0,240,255,0.0);}50%{box-shadow:0 0 12px rgba(0,240,255,0.35),0 0 0 3px rgba(0,240,255,0.08);}}
.diff-badge-animated{animation:diff-badge-breathe 2.4s ease-in-out infinite;}

/* ── LIVE EMOTION PULSE DOT ── */
@keyframes emo-live-dot{0%,100%{box-shadow:0 0 0 0 rgba(16,185,129,0.5);transform:scale(1);}50%{box-shadow:0 0 0 5px rgba(16,185,129,0);transform:scale(1.15);}}
.emo-live-dot{width:7px;height:7px;border-radius:50%;background:#10b981;display:inline-block;margin-right:5px;vertical-align:middle;animation:emo-live-dot 1.6s ease-in-out infinite;}

/* ── PROGRESS BAR SHIMMER ── */
.stProgress>div>div{background:linear-gradient(90deg,#6366f1,#8b5cf6,#00F0FF)!important;border-radius:4px!important;position:relative;overflow:hidden;}
.stProgress>div>div::after{content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent 0%,rgba(255,255,255,0.3) 50%,transparent 100%);animation:prog-shimmer 1.8s ease-in-out infinite;}
@keyframes prog-shimmer{0%{transform:translateX(-100%);}100%{transform:translateX(400%);}}

/* ── SUBMIT BUTTON — enhanced sweep ── */
@keyframes btn-sweep{0%{left:-100%;}55%,100%{left:200%;}}
.stButton>button::after{content:'';position:absolute;top:0;left:-100%;width:50%;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.12),transparent);transform:skewX(-18deg);animation:btn-sweep 3.2s ease-in-out infinite;pointer-events:none;}
.stButton>button{position:relative;overflow:hidden;}

/* ── VOICE INPUT PANEL GLOW BORDER ── */
@keyframes voice-panel-glow{0%,100%{border-color:rgba(0,240,255,0.12);}50%{border-color:rgba(0,240,255,0.3);}}
.voice-panel-live{animation:voice-panel-glow 2s ease-in-out infinite;}

/* ── WAVEFORM BAR ANIMATION BASE ── */
@keyframes wbar-bounce{from{transform:scaleY(0.2);opacity:0.3;}to{transform:scaleY(1);opacity:1;}}

/* ── HINT KEYWORD PILL HOVER ── */
.kw-tag{display:inline-block;background:rgba(0,255,136,.07);color:#00ff88;padding:3px 10px;border-radius:5px;font-size:.7rem;margin:2px;font-family:'Share Tech Mono',monospace;border:1px solid rgba(0,255,136,.2);transition:all .2s;}
.kw-tag:hover{background:rgba(0,255,136,.18);border-color:rgba(0,255,136,.5);transform:translateY(-1px);box-shadow:0 0 8px rgba(0,255,136,0.2);}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,.05)!important;border-radius:10px!important;padding:4px!important;border:1px solid rgba(255,255,255,.08)!important;gap:2px!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#b4cde4!important;font-family:Inter,sans-serif!important;font-size:.8rem!important;font-weight:500!important;border-radius:8px!important;padding:8px 16px!important;text-transform:none!important;}
.stTabs [aria-selected="true"]{background:rgba(99,102,241,.15)!important;color:#a5b4fc!important;border-bottom:none!important;}

/* ── MISC COMPONENTS ── */
.sec-lbl{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#a5b4fc;margin-bottom:.4rem;font-family:Inter,sans-serif;}
.cbl{font-size:.68rem;color:#b4cde4;display:flex;justify-content:space-between;margin-bottom:3px;font-family:Inter,sans-serif;font-weight:500;}
.cbbg{background:rgba(255,255,255,.14);border-radius:4px;height:6px;overflow:hidden;}
.cbfill{height:100%;border-radius:4px;transition:width .5s cubic-bezier(.4,0,.2,1);}
.txbox{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:.8rem;max-height:180px;overflow-y:auto;color:#a5b4fc;font-size:.78rem;line-height:1.7;white-space:pre-wrap;font-family:Inter,sans-serif;}
.emo-pill{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:20px;font-size:.72rem;font-weight:500;border:1px solid;margin:2px;font-family:Inter,sans-serif;}

/* ── WEBCAM (CLEAN — no HUD brackets or scanlines) ── */
.hud-webcam{position:relative;border:1px solid rgba(99,102,241,.3);border-radius:12px;overflow:hidden;box-shadow:0 4px 24px rgba(99,102,241,.1);}
.hud-webcam::after{display:none!important;}
.hud-tl,.hud-tr,.hud-bl,.hud-br{display:none!important;}

/* ── NERVOUSNESS BAR ── */
.nerv-bar-track{background:rgba(255,255,255,.08);border-radius:4px;height:6px;overflow:hidden;margin:5px 0;}
.nerv-bar-fill{height:100%;border-radius:4px;transition:width .5s;}

/* ── FOLLOW-UP ALERT (clean glass style) ── */
.protocol-alert{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);border-radius:12px;padding:1rem 1.2rem;margin-top:.75rem;animation:none!important;}

/* ── WIZARD PROGRESS BAR ── */
.wizard-bar{display:flex;align-items:center;justify-content:center;gap:0;margin-bottom:2rem;padding:0 1rem;}
.wizard-step{display:flex;flex-direction:column;align-items:center;gap:4px;position:relative;}
.wizard-dot{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.72rem;font-weight:700;font-family:Inter,sans-serif;transition:all .3s;}
.wizard-dot.done{background:#6366f1;color:#fff;box-shadow:0 0 0 4px rgba(99,102,241,.2);}
.wizard-dot.active{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;box-shadow:0 0 0 4px rgba(99,102,241,.25);}
.wizard-dot.pending{background:rgba(255,255,255,.08);color:#64748b;border:1px solid rgba(255,255,255,.1);}
.wizard-label{font-size:.68rem;font-weight:500;font-family:Inter,sans-serif;white-space:nowrap;}
.wizard-label.done,.wizard-label.active{color:#a5b4fc;}
.wizard-label.pending{color:#475569;}
.wizard-line{width:60px;height:1px;background:rgba(255,255,255,.1);margin:0 4px;margin-bottom:16px;}
.wizard-line.done{background:rgba(99,102,241,.5);}

/* ── ENCRYPTION BADGE → FOCUS BADGE ── */
.encryption-badge{padding:4px 14px;background:rgba(99,102,241,.12);border:1px solid rgba(99,102,241,.3);border-radius:20px;font-size:.72rem;color:#a5b4fc;font-family:Inter,sans-serif;letter-spacing:0;font-weight:500;}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(99,102,241,.3);border-radius:4px;}

/* ── BUTTON GLOW PULSE ── */
@keyframes btn-glow-pulse{0%,100%{box-shadow:0 0 22px rgba(0,240,255,0.35),0 4px 14px rgba(0,0,0,0.4);}50%{box-shadow:0 0 38px rgba(0,240,255,0.6),0 0 18px rgba(0,240,255,0.3),0 4px 14px rgba(0,0,0,0.4);}}

/* ── TIMER DOT ── */
@keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.4;transform:scale(.8);}}
.pdot{width:7px;height:7px;border-radius:50%;display:inline-block;animation:pulse-dot 1.2s infinite;}
.timer-dot{width:6px;height:6px;border-radius:50%;display:inline-block;animation:pulse-dot 1.2s infinite;}

[data-testid="stDataFrame"]{border-radius:10px!important;}
hr{border-color:rgba(255,255,255,.06)!important;}
.grad-g{color:#34d399;}
.grad-v{color:#a5b4fc;}
/* Hide Streamlit deprecation + warning banners */
[data-testid="stDeprecationWarning"]{display:none!important;}
div[data-testid="stAlert"] > div[data-baseweb="notification"][kind="warning"]{display:none!important;}
.stAlert[data-baseweb="notification"]{display:none!important;}
div.stException{display:none!important;}

/* ── HIDDEN NAV BUTTONS (checklist) — zero footprint ── */
[data-cl-hidden-nav]{height:0!important;overflow:hidden!important;
  line-height:0!important;padding:0!important;margin:0!important;}
[data-cl-hidden-nav] *{height:0!important;min-height:0!important;
  padding:0!important;margin:0!important;border:none!important;
  visibility:hidden!important;pointer-events:none!important;}

/* ── PAGE WIPE TRANSITION — directional (Idea 15 extended) ── */
@keyframes aura-wipe-fwd{
  from{opacity:0;transform:translateX(42px);}
  to  {opacity:1;transform:translateX(0);}
}
@keyframes aura-wipe-bwd{
  from{opacity:0;transform:translateX(-42px);}
  to  {opacity:1;transform:translateX(0);}
}
.aura-page-enter-fwd{
  animation:aura-wipe-fwd .32s cubic-bezier(.22,1,.36,1) both;
}
.aura-page-enter-bwd{
  animation:aura-wipe-bwd .32s cubic-bezier(.22,1,.36,1) both;
}

/* ── V3 INTERVIEW SCREEN — qcard scan animation ── */
@keyframes qcard-scan{
  0%{transform:scaleX(0);opacity:0;}
  20%{transform:scaleX(1);opacity:1;}
  80%{transform:scaleX(1);opacity:1;}
  100%{transform:scaleX(0);opacity:0;transform-origin:right;}
}
/* v3 live-emotion dot */
@keyframes emo-live-dot{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.4;transform:scale(.7);}}

/* ── QUESTION FLIP CARD ── */
.qflip-scene{perspective:1200px;width:100%;margin-bottom:1rem;}
.qflip-card{
  position:relative;width:100%;
  transform-style:preserve-3d;
  transition:transform .55s cubic-bezier(.4,0,.2,1);
}
.qflip-card.is-flipped{transform:rotateY(180deg);}
.qflip-front,.qflip-back{
  width:100%;backface-visibility:hidden;-webkit-backface-visibility:hidden;
  border-radius:14px;
}
.qflip-front{
  background:rgba(99,102,241,.08);
  border:1px solid rgba(99,102,241,.25);
  border-left:4px solid #6366f1;
  padding:1.4rem 1.6rem;
  display:flex;align-items:center;gap:12px;
  color:#a5b4fc;font-family:Share\ Tech\ Mono,monospace;font-size:.9rem;
}
.qflip-back{
  position:absolute;top:0;left:0;
  transform:rotateY(180deg);
  background:rgba(255,255,255,.06);
  backdrop-filter:blur(12px);
  border:1px solid rgba(255,255,255,.1);
  border-left:4px solid var(--qcard-accent,#6366f1);
  padding:1.4rem 1.6rem;
}
.qflip-spinner{
  width:16px;height:16px;border-radius:50%;
  border:2px solid rgba(99,102,241,.2);
  border-top-color:#a5b4fc;
  animation:qflip-spin .8s linear infinite;
  flex-shrink:0;
}
@keyframes qflip-spin{to{transform:rotate(360deg);}}

/* ── SESSION TIMELINE SCRUBBER ── */
.aura-timeline{
  background:rgba(10,25,47,.92);
  border:1px solid rgba(0,212,255,.1);
  border-radius:7px;padding:.6rem .75rem;margin-bottom:.3rem;
}
.aura-tl-track{
  position:relative;height:4px;
  background:rgba(255,255,255,.06);
  border-radius:2px;margin:.45rem 0 .55rem;
}
.aura-tl-fill{
  position:absolute;top:0;left:0;height:100%;
  background:linear-gradient(90deg,#00ff88,#00d4ff);
  border-radius:2px;transition:width .5s cubic-bezier(.4,0,.2,1);
}
.aura-tl-dots{
  position:absolute;top:50%;transform:translateY(-50%);
  width:100%;display:flex;justify-content:space-between;
  pointer-events:none;
}
.aura-tl-dot{
  width:11px;height:11px;border-radius:50%;
  border:2px solid #060d15;
  flex-shrink:0;transition:background .35s,box-shadow .35s;
}
.aura-tl-dot.done-ex {background:#00ff88;box-shadow:0 0 6px #00ff8888;animation:dot-pop .44s cubic-bezier(.34,1.56,.64,1) both;}
.aura-tl-dot.done-gd {background:#a5b4fc;box-shadow:0 0 6px #a5b4fc88;animation:dot-pop .44s cubic-bezier(.34,1.56,.64,1) both;}
.aura-tl-dot.done-av {background:#fbbf24;box-shadow:0 0 6px #fbbf2488;animation:dot-pop .44s cubic-bezier(.34,1.56,.64,1) both;}
.aura-tl-dot.done-po {background:#ff3366;box-shadow:0 0 6px #ff336688;animation:dot-pop .44s cubic-bezier(.34,1.56,.64,1) both;}
.aura-tl-dot.current {background:#00d4ff;animation:dot-glow-ring 1.8s ease-in-out infinite;}
.aura-tl-dot.pending {background:rgba(255,255,255,.1);border-color:rgba(255,255,255,.15);}
.aura-tl-labels{display:flex;justify-content:space-between;}
.aura-tl-label{font-size:.42rem;color:#c7e8ff;font-family:Share\ Tech\ Mono,monospace;text-align:center;}
/* ── TIMELINE DOT: pop+bounce for completed, glow-ring pulse for active ── */
@keyframes dot-pop{
  0%  {transform:scale(0);  opacity:0;}
  55% {transform:scale(1.35);opacity:1;}
  75% {transform:scale(0.88);}
  100%{transform:scale(1);  opacity:1;}
}
@keyframes dot-glow-ring{
  0%  {box-shadow:0 0 0 0   rgba(0,212,255,.0),  0 0 6px rgba(0,212,255,.5);}
  50% {box-shadow:0 0 0 5px rgba(0,212,255,.22), 0 0 14px rgba(0,212,255,.7);}
  100%{box-shadow:0 0 0 0   rgba(0,212,255,.0),  0 0 6px rgba(0,212,255,.5);}
}
.sb-mini-map{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;justify-items:center;padding:.45rem .3rem;}
.sb-mini-dot{width:9px;height:9px;border-radius:50%;border:1.5px solid #060d15;flex-shrink:0;transition:background .4s,box-shadow .4s;}
.sb-mini-dot.done-ex{background:#00ff88;box-shadow:0 0 5px #00ff8888;animation:dot-pop .42s cubic-bezier(.34,1.56,.64,1) both;}
.sb-mini-dot.done-gd{background:#a5b4fc;box-shadow:0 0 5px #a5b4fc88;animation:dot-pop .42s cubic-bezier(.34,1.56,.64,1) both;}
.sb-mini-dot.done-av{background:#fbbf24;box-shadow:0 0 5px #fbbf2488;animation:dot-pop .42s cubic-bezier(.34,1.56,.64,1) both;}
.sb-mini-dot.done-po{background:#ff3366;box-shadow:0 0 5px #ff336688;animation:dot-pop .42s cubic-bezier(.34,1.56,.64,1) both;}
.sb-mini-dot.current{background:#00d4ff;animation:dot-glow-ring 1.8s ease-in-out infinite;}
.sb-mini-dot.pending{background:rgba(255,255,255,.08);border-color:rgba(255,255,255,.12);}

/* ══ SESSION INIT PAGE — dark panel overlay ══════════════════════════════════
   Activated when JS stamps data-page="start" on <body>.
   Covers the main block-container with a solid dark box so the starfield
   background is fully hidden behind it. All widgets render on top of this.
   =========================================================================== */
body[data-page="start"] [data-testid="stMainBlockContainer"]{
  background:#07101f!important;
  border:1.5px solid rgba(0,220,255,0.35)!important;
  border-radius:20px!important;
  box-shadow:0 0 80px rgba(0,0,0,0.98),0 0 0 1px #000,inset 0 1px 0 rgba(255,255,255,0.04)!important;
  padding:2rem 2.5rem 3rem!important;
  margin-top:0.5rem!important;
  max-width:600px!important;
  margin-left:auto!important;
  margin-right:auto!important;
}
/* Inputs */
body[data-page="start"] input[type=text],
body[data-page="start"] textarea{
  background:#040b18!important;
  border:1px solid rgba(0,212,255,0.5)!important;
  color:#e8f4ff!important;border-radius:10px!important;
}
/* Selectboxes */
body[data-page="start"] [data-baseweb="select"]>div{
  background:#040b18!important;
  border:1px solid rgba(0,212,255,0.5)!important;
  border-radius:10px!important;
}
body[data-page="start"] [data-baseweb="select"] span,
body[data-page="start"] [data-baseweb="select"] input{
  color:#e8f4ff!important;background:transparent!important;
}
/* Slider thumbs */
body[data-page="start"] [data-testid="stSlider"] [role="slider"]{
  background:#00d4ff!important;border:2px solid #00d4ff!important;
  box-shadow:0 0 14px rgba(0,212,255,0.85)!important;
}
body[data-page="start"] [data-testid="stSelectSlider"] [role="slider"]{
  background:#a855f7!important;border:2px solid #a855f7!important;
  box-shadow:0 0 14px rgba(168,85,247,0.85)!important;
}
/* Toggle labels */
body[data-page="start"] [data-testid="stToggle"] label p{
  color:#c7e8ff!important;font-family:'Share Tech Mono',monospace!important;
}
/* Aura animated labels — strong glow + black backing */
body[data-page="start"] .aura-lbl-green{
  color:#00ff88!important;
  text-shadow:0 0 16px rgba(0,255,136,1),0 0 4px #000,0 2px 12px #000!important;
  font-size:11px!important;
}
body[data-page="start"] .aura-lbl-cyan{
  color:#00d4ff!important;
  text-shadow:0 0 16px rgba(0,212,255,1),0 0 4px #000,0 2px 12px #000!important;
  font-size:11px!important;
}
body[data-page="start"] .aura-lbl-violet{
  color:#d4a0ff!important;
  text-shadow:0 0 16px rgba(180,100,255,1),0 0 4px #000,0 2px 12px #000!important;
  font-size:11px!important;
}
body[data-page="start"] .aura-lbl-amber{
  color:#ffb400!important;
  text-shadow:0 0 16px rgba(255,180,0,1),0 0 4px #000,0 2px 12px #000!important;
  font-size:11px!important;
}

/* ── MARQUEE ANIMATIONS (Dashboard hero) ── */
@keyframes mqAuraL{from{transform:translateX(0)}to{transform:translateX(-50%)}}
@keyframes mqAuraR{from{transform:translateX(-50%)}to{transform:translateX(0)}}

/* ── LIVE INTERVIEW PAGE OVERRIDES ── */
/* Make textarea dark glass */
[data-testid="stTabs"] textarea,
.stTextArea textarea{
  background:rgba(2,6,18,0.85)!important;
  border:1px solid rgba(0,240,255,0.12)!important;
  border-radius:10px!important;
  color:#e2e8f0!important;
  font-family:Inter,sans-serif!important;
  font-size:.9rem!important;
  resize:vertical!important;
}
[data-testid="stTabs"] textarea:focus,
.stTextArea textarea:focus{
  border-color:rgba(0,240,255,0.35)!important;
  box-shadow:0 0 0 3px rgba(0,240,255,0.06)!important;
}
/* Tabs styled as pill selector */
.stTabs [data-baseweb="tab-list"]{
  background:rgba(2,6,18,0.75)!important;
  border:1px solid rgba(0,240,255,0.1)!important;
  border-radius:12px!important;
  padding:5px!important;
  gap:4px!important;
}
.stTabs [data-baseweb="tab"]{
  border-radius:9px!important;
  font-size:.78rem!important;
  padding:7px 16px!important;
  font-family:Inter,sans-serif!important;
  letter-spacing:.01em!important;
}
.stTabs [aria-selected="true"]{
  background:rgba(0,240,255,0.1)!important;
  color:#00F0FF!important;
  border:1px solid rgba(0,240,255,0.25)!important;
}
/* Expander dark style */
[data-testid="stExpander"] summary{
  background:rgba(2,6,18,0.6)!important;
  border:1px solid rgba(255,255,255,0.07)!important;
  border-radius:10px!important;
}
/* st.audio_input dark */
[data-testid="stAudioInput"]>div{
  background:rgba(2,6,18,0.8)!important;
  border:1px solid rgba(0,240,255,0.12)!important;
  border-radius:12px!important;
}
</style></div>"""
    st.markdown(css, unsafe_allow_html=True)

def badge(s: float):
    if s >= 4.2: return "b-ex", "EXCELLENT"
    if s >= 3.5: return "b-gd", "GOOD"
    if s >= 2.5: return "b-av", "AVERAGE"
    return "b-po", "CRITICAL"


# ── Feature 12: Role-Specific Colour Theming ─────────────────────────────────
_ROLE_THEME_MAP = [
    # (keywords,              accent_hex,  glow_rgba,                  label        )
    (["software","engineer","swe","backend","frontend","fullstack","web","mobile","ios","android","dev"],
     "#00F0FF", "rgba(0,240,255,.25)", "Software Engineer"),
    (["data scientist","data science","machine learning","ml","ai","nlp","deep learning","analytics","analyst"],
     "#f97316", "rgba(249,115,22,.25)",  "Data Science / ML"),
    (["finance","financial","banking","investment","quant","trader","trading","economist","economist"],
     "#FFD700", "rgba(255,215,0,.22)",   "Finance"),
    (["hr","human resources","talent","recruiter","people ops","people operations"],
     "#c084fc", "rgba(192,132,252,.22)", "HR / People"),
    (["product","product manager","pm","product owner","ux","designer","design"],
     "#34d399", "rgba(52,211,153,.22)",  "Product / Design"),
    (["devops","cloud","infrastructure","sre","platform","aws","gcp","azure","kubernetes","docker"],
     "#38bdf8", "rgba(56,189,248,.22)",  "DevOps / Cloud"),
    (["security","cybersecurity","pentest","soc","infosec"],
     "#f43f5e", "rgba(244,63,94,.22)",   "Security"),
    (["marketing","growth","seo","content","brand","social media"],
     "#fb923c", "rgba(251,146,60,.22)",  "Marketing"),
    (["legal","compliance","lawyer","attorney","paralegal"],
     "#a78bfa", "rgba(167,139,250,.22)", "Legal"),
]
_ROLE_DEFAULT_ACCENT = "#6366f1"
_ROLE_DEFAULT_GLOW   = "rgba(99,102,241,.22)"

def get_role_theme(role: str) -> tuple:
    """Return (accent_hex, glow_rgba, label) for the given role string."""
    r = role.lower()
    for keywords, accent, glow, label in _ROLE_THEME_MAP:
        if any(kw in r for kw in keywords):
            return accent, glow, label
    return _ROLE_DEFAULT_ACCENT, _ROLE_DEFAULT_GLOW, role or "General"

def mchip(lbl: str, val, maxv: float = 5.0, col=None) -> None:
    disp = f"{val:.1f}" if isinstance(val, float) else str(val)
    sub  = f"/{maxv:.0f}" if isinstance(val, float) else ""
    html = (f'<div class="mchip"><div class="mchip-v">{disp}'
            f'<span style="font-size:.65rem;color:#c7e8ff;font-family:Rajdhani,sans-serif;">{sub}</span></div>'
            f'<div class="mchip-l">{lbl}</div></div>')
    (col or st).markdown(html, unsafe_allow_html=True)

def coach_bar(lbl: str, pct: float, color: str = "#00FFD1") -> None:
    pct = max(0.0, min(100.0, pct))
    st.markdown(
        f'<div style="margin:.2rem 0;">'
        f'<div class="cbl"><span>{lbl}</span><span style="color:{color};">{pct:.0f}%</span></div>'
        f'<div class="cbbg"><div class="cbfill" style="width:{pct}%;background:{color};'
        f'box-shadow:0 0 6px {color}55;"></div></div>'
        f'</div>', unsafe_allow_html=True)

def dcl(fig, h: int = 260):
    fig.update_layout(
        height=h, margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#5a7a9a", family="Rajdhani"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="rgba(0,255,136,.04)", zerolinecolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="rgba(0,255,136,.04)", zerolinecolor="rgba(0,0,0,0)")
    return fig

def emo_css(emotion: str) -> str:
    return {
        "Happy":   "#00FFD1", "Neutral": "#4a9eff", "Surprise": "#00D4FF",
        "Calm":    "#00D4FF", "Sad":     "#4a9eff", "Fear":     "#FFD700",
        "Angry":   "#FF3366", "Disgust": "#a855f7",
    }.get(emotion, "#4a9eff")

def nerv_css(n: float) -> str:
    return "#00FFD1" if n < 0.35 else ("#FFD700" if n < 0.65 else "#FF3366")

def conf_css(s: float) -> str:
    return "#00FFD1" if s >= 4.0 else ("#FFD700" if s >= 3.0 else "#FF3366")

def process_camera_frame(img_file):
    if img_file is None: return None, {}
    try:
        frame_bgr = bytes_to_bgr(img_file.getvalue())
        if frame_bgr is None: return None, {}
        annotated, result = engine.analyse_webcam_frame(frame_bgr)
        _, buf = cv2.imencode(".png", annotated)
        return buf.tobytes(), result
    except Exception:
        return None, {}


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED WIDGETS
# ══════════════════════════════════════════════════════════════════════════════
# render_emotion_bar removed in v10.0 — facial nervousness has no role


def render_posture_widget(posture: dict) -> None:
    if not posture or not posture.get("detected", False):
        return
    raw    = posture.get("raw_scores", {})
    conf   = posture.get("confidence_score", 3.5)
    ear    = posture.get("ear", 0.28)
    alerts = posture.get("alerts", [])
    cc     = conf_css(conf)
    mh = ""
    for lbl, key in [("SHOULDER","shoulder_alignment"),("EYE","eye_contact"),
                     ("HEAD","head_tilt"),("LEAN","body_lean"),("HANDS","hand_movement")]:
        v  = raw.get(key, 3.5)
        vc = conf_css(v)
        fw = int(v / 5 * 100)
        mh += (
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'background:rgba(0,255,136,.03);border:1px solid rgba(0,255,136,.07);'
            f'border-radius:4px;padding:4px 3px;">'
            f'<div style="font-family:Orbitron,monospace;font-size:.88rem;font-weight:700;color:{vc};">{v:.1f}</div>'
            f'<div style="font-size:.48rem;color:#c7e8ff;text-transform:uppercase;letter-spacing:.06em;'
            f'font-family:Share Tech Mono,monospace;margin:2px 0;">{lbl}</div>'
            f'<div style="width:100%;height:2px;background:rgba(255,255,255,.04);border-radius:2px;overflow:hidden;">'
            f'<div style="width:{fw}%;height:100%;background:{vc};border-radius:2px;"></div></div>'
            f'</div>'
        )
    if ear > 0.25:   eye_txt, eye_c = "OPEN",    "#00FFD1"
    elif ear > 0.15: eye_txt, eye_c = "PARTIAL",  "#FFD700"
    else:            eye_txt, eye_c = "BLINK",    "#4a9eff"
    ah = "".join(
        f'<div style="font-size:.65rem;color:{nerv_css(.7)};margin-top:2px;font-family:Share Tech Mono,monospace;">▸ {a}</div>'
        for a in alerts[:3])
    st.markdown(f"""
<div style="background:rgba(5,10,20,.8);border:1px solid rgba(0,255,136,.1);
  border-radius:6px;padding:.6rem .8rem;margin:.25rem 0;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.35rem;">
    <span class="sec-lbl" style="margin:0;">POSTURE ANALYSIS</span>
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="font-size:.7rem;font-weight:700;color:{eye_c};font-family:Share Tech Mono,monospace;">EYE:{eye_txt} EAR:{ear:.2f}</span>
      <span style="font-family:Orbitron,monospace;font-size:.95rem;font-weight:700;color:{cc};">{conf:.1f}/5</span>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:3px;margin-bottom:.3rem;">{mh}</div>
  {ah}
</div>""", unsafe_allow_html=True)


def render_confidence_widget(conf_data: dict) -> None:
    cs = conf_data.get("confidence_score", 3.5)
    ps = conf_data.get("posture_score", 3.5)
    fs = conf_data.get("facial_score", 3.5)
    vs = conf_data.get("voice_score", 3.5)
    cc = conf_css(cs)
    deg = int(cs / 5 * 360)
    st.markdown(f"""
<div style="background:rgba(5,10,20,.85);border:1px solid rgba(0,212,255,.2);
  border-radius:10px;padding:.8rem;margin:.25rem 0;">
  <div class="sec-lbl" style="text-align:center;margin-bottom:.5rem;color:#00d4ff;">
    ◈ MULTIMODAL CONFIDENCE INDEX
  </div>
  <div style="display:flex;align-items:center;gap:.8rem;">
    <div style="width:68px;height:68px;border-radius:50%;flex-shrink:0;
      background:conic-gradient({cc} {deg}deg,rgba(255,255,255,.04) {deg}deg);
      display:flex;align-items:center;justify-content:center;
      box-shadow:0 0 20px {cc}44,0 0 40px {cc}22;">
      <div style="width:54px;height:54px;border-radius:50%;background:#050A14;
        display:flex;align-items:center;justify-content:center;
        font-family:Orbitron,monospace;font-size:.95rem;font-weight:700;color:{cc};">{cs:.1f}</div>
    </div>
    <div style="flex:1;">
      <div style="font-size:.58rem;color:#c7e8ff;margin-bottom:4px;font-family:Share Tech Mono,monospace;">
        0.4×POSTURE + 0.3×FACIAL + 0.3×VOICE
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:3px;">
        <div style="text-align:center;background:rgba(0,255,136,.04);border:1px solid rgba(0,255,136,.08);border-radius:4px;padding:4px;">
          <div style="font-family:Orbitron,monospace;font-size:.9rem;font-weight:700;color:{conf_css(ps)};">{ps:.1f}</div>
          <div style="font-size:.5rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;">POSTURE</div>
        </div>
        <div style="text-align:center;background:rgba(0,212,255,.04);border:1px solid rgba(0,212,255,.08);border-radius:4px;padding:4px;">
          <div style="font-family:Orbitron,monospace;font-size:.9rem;font-weight:700;color:{emo_css(st.session_state.live_emotion)};">{fs:.1f}</div>
          <div style="font-size:.5rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;">FACIAL</div>
        </div>
        <div style="text-align:center;background:rgba(124,58,255,.04);border:1px solid rgba(124,58,255,.08);border-radius:4px;padding:4px;">
          <div style="font-family:Orbitron,monospace;font-size:.9rem;font-weight:700;color:{emo_css(st.session_state.live_voice_emotion)};">{vs:.1f}</div>
          <div style="font-size:.5rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;">VOICE</div>
        </div>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


def audio_waveform(recording: bool = False) -> None:
    col   = "#00FFD1" if recording else "#0d2040"
    pulse = "true"    if recording else "false"
    components.html(f"""
<canvas id="wv" width="500" height="44"
  style="width:100%;border-radius:4px;background:rgba(5,10,20,.8);display:block;
  border:1px solid rgba(0,255,136,.08);"></canvas>
<script>(function(){{
  const c=document.getElementById('wv'),ctx=c.getContext('2d');
  const W=c.width,H=c.height,bars=55,pulse={pulse};
  let t=0;
  function draw(){{
    ctx.clearRect(0,0,W,H);ctx.fillStyle='{col}';
    const bw=(W/bars)*.45,gap=(W/bars)*.55;
    for(let i=0;i<bars;i++){{
      let h=pulse?(Math.abs(Math.sin(t*2.5+i*.4))*.75+Math.random()*.25)*H*.8
              :Math.abs(Math.sin(i*.35+t*.1))*H*.1+1.5;
      const x=i*(bw+gap)+gap/2;
      ctx.shadowColor='{col}';
      ctx.shadowBlur=pulse?6:0;
      ctx.beginPath();ctx.roundRect(x,(H-h)/2,bw,h,1);ctx.fill();
    }}
    t+=0.04;requestAnimationFrame(draw);
  }}
  draw();
}})();</script>""", height=48)


def inject_eval_border_pulse(active: bool = True) -> None:
    """
    v12.2 — Injects a breathing violet border glow onto the question card
    while the AI is evaluating.  Call with active=True just before the answer
    widget; the animation is automatically removed when render_eval_results()
    fires on the next rerun.
    """
    if not active:
        return
    components.html("""
<script>
(function(){
  var PARENT = window.parent.document;
  if (PARENT.getElementById('aura-eval-pulse-style')) return;
  var s = PARENT.createElement('style');
  s.id = 'aura-eval-pulse-style';
  s.textContent =
    '@keyframes aura-border-breathe{' +
    '0%,100%{box-shadow:0 0 0 2px rgba(124,107,255,.18),0 0 14px rgba(124,107,255,.10);}' +
    '50%{box-shadow:0 0 0 2px rgba(124,107,255,.55),0 0 28px rgba(124,107,255,.28);}}' +
    '.aura-evaluating{border-radius:10px!important;' +
    'animation:aura-border-breathe 1.8s ease-in-out infinite!important;}';
  PARENT.head.appendChild(s);
  var target = PARENT.querySelector('[data-testid="stVerticalBlock"]');
  if (target) target.classList.add('aura-evaluating');
})();
</script>
""", height=0)


def _stop_eval_border_pulse() -> None:
    """Remove the border pulse animation after evaluation results are shown."""
    components.html("""
<script>
(function(){
  var PARENT = window.parent.document;
  PARENT.querySelectorAll('.aura-evaluating').forEach(function(el){
    el.classList.remove('aura-evaluating');
  });
  var s = PARENT.getElementById('aura-eval-pulse-style');
  if (s) s.remove();
})();
</script>
""", height=0)


def render_eval_results(ev: Dict) -> None:
    """
    v12.2 — Animated evaluation results card.

    Three real-computation animations:
      1. Border glow pulse   — breathes on the question card while AI processes
                               (injected via inject_eval_border_pulse before submit)
      2. Dimension bars      — Relevance / Grammar / Depth / Keywords / STAR / Pace
                               fill in with staggered eased reveals on score card
      3. Keyword scan beam   — sliding highlight sweeps the answer text and lights
                               up matched keyword tokens as the beam crosses them
    """
    if not ev: return
    import json as _json_eval
    import re as _re_eval

    fs  = ev.get("final_score", 0)
    sim = ev.get("similarity_score", 0)
    grm = ev.get("grammar_score", 0)
    sc  = ev.get("score", 1.0)
    wc  = ev.get("word_count", 0)
    bc, bt = badge(sc)
    icon = "◈" if fs >= 70 else ("▸" if fs >= 45 else "▲")

    # Answer text for scan beam — stored when candidate types / speaks
    _answer_for_scan = (
        st.session_state.get("last_answer_text", "")
        or st.session_state.get("_pending_answer", "")
        or ""
    )[:800]

    # ── v8.0: Relevance source badge ─────────────────────────────────────────
    rel_source = ev.get("relevance_source", "")
    _src_map = {
        "api_groq":      ("🧠 Groq LLM",  "#00FFD1",  "rgba(0,255,136,.08)"),
        "api_groq_kw":   ("🧠 Groq+KW",   "#00FFD1",  "rgba(0,255,136,.08)"),
        "sbert":         ("⚡ SBERT",      "#00D4FF",  "rgba(0,212,255,.08)"),
        "tfidf_fallback":("📐 TF-IDF",     "#FFD700",  "rgba(255,170,0,.08)"),
        "none":          ("— no scorer",  "#5a7a9a",  "rgba(0,0,0,0)"),
    }
    _src_label, _src_col, _src_bg = _src_map.get(
        rel_source, ("", "#5a7a9a", "rgba(0,0,0,0)"))
    _src_badge_html = (
        f'<span style="background:{_src_bg};color:{_src_col};border:1px solid {_src_col}44;'
        f'border-radius:3px;padding:1px 8px;font-size:.58rem;font-weight:700;'
        f'font-family:Share Tech Mono,monospace;margin-left:6px;"'
        f' title="Relevance scored by: {rel_source}">{_src_label}</span>'
        if _src_label else ""
    )

    # ── Card colours ─────────────────────────────────────────────────────────
    if fs >= 70:
        _rc_bg, _rc_bdr, _rc_txt = "rgba(16,185,129,.15)", "rgba(16,185,129,.5)", "#6ee7b7"
        _rc_left = "#10b981"
    elif fs >= 45:
        _rc_bg, _rc_bdr, _rc_txt = "rgba(245,158,11,.15)", "rgba(245,158,11,.5)", "#fcd34d"
        _rc_left = "#f59e0b"
    else:
        _rc_bg, _rc_bdr, _rc_txt = "rgba(239,68,68,.15)", "rgba(239,68,68,.45)", "#fca5a5"
        _rc_left = "#ef4444"

    _badge_map = {
        "b-ex": ("#34d399","rgba(16,185,129,.2)","rgba(16,185,129,.5)"),
        "b-gd": ("#c4b5fd","rgba(99,102,241,.2)","rgba(99,102,241,.5)"),
        "b-av": ("#fcd34d","rgba(245,158,11,.2)","rgba(245,158,11,.5)"),
        "b-po": ("#fca5a5","rgba(239,68,68,.2)","rgba(239,68,68,.5)"),
    }
    _btc, _btbg, _btbd = _badge_map.get(bc, ("#94a3b8","transparent","rgba(255,255,255,.1)"))

    # ── Dimension bars ────────────────────────────────────────────────────────
    q_type_k = ev.get("question_type", "").lower()
    _star_applicable = q_type_k in ("behavioural", "behavioral", "hr", "")
    star     = ev.get("star_scores", {})
    t_eff    = ev.get("time_efficiency", 0.0)
    t_label  = ev.get("time_label", "")
    t_show   = t_eff > 0 and t_label not in ("No timing data", "N/A", "")
    depth_sc = ev.get("depth_score", 0.0)
    kw_hits  = ev.get("keyword_hits", [])

    dims = [
        ("Relevance", min(sim, 100.0),           "#00FFD1"),
        ("Grammar",   min(grm, 100.0),           "#c4b5fd"),
        ("Depth",     min(depth_sc * 100, 100.0),"#00D4FF"),
        ("Keywords",  min(len(kw_hits) / max(1, max(len(kw_hits), 5)) * 100, 100.0), "#fcd34d"),
    ]
    if _star_applicable and star:
        _star_pct = sum(1 for v in star.values() if v) / max(len(star), 1) * 100
        dims.append(("STAR", _star_pct, "#f9a8d4"))
    if t_show:
        dims.append(("Pace", min(t_eff, 100.0), "#86efac"))

    # ── Feedback lines ────────────────────────────────────────────────────────
    _feedback_raw = ev.get('feedback', '')
    _feedback_sentences = [s.strip() for s in _feedback_raw.replace('•','|').replace('▸','|').split('|') if s.strip()]
    if not _feedback_sentences:
        _feedback_sentences = [s.strip() for s in _re_eval.split(r'(?<=[.!?])\s+', _feedback_raw) if s.strip()]
    if not _feedback_sentences:
        _feedback_sentences = [_feedback_raw]

    # ── Serialise for JS ──────────────────────────────────────────────────────
    _fb_json   = _json_eval.dumps(_feedback_sentences)
    _dims_json = _json_eval.dumps(dims)
    _kw_json   = _json_eval.dumps(kw_hits[:12])
    _ans_json  = _json_eval.dumps(_answer_for_scan)
    _n_dims    = len(dims)
    _n_fb      = len(_feedback_sentences)
    _scan_h    = 72 if _answer_for_scan else 0
    _height    = max(160, 130 + _n_dims * 22 + _scan_h + _n_fb * 46)

    components.html(f"""<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Inter:wght@400;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:transparent;font-family:Inter,sans-serif;}}
/* ── Score card ── */
#eval-card{{
  background:{_rc_bg};border:1px solid {_rc_bdr};border-left:4px solid {_rc_left};
  border-radius:10px;padding:.85rem 1.1rem 1rem;margin-bottom:.6rem;overflow:hidden;
}}
#eval-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem;}}
#score-val{{font-size:1rem;font-weight:700;font-family:Orbitron,monospace;color:{_rc_txt};
  opacity:0;transform:translateY(-6px);transition:opacity .4s,transform .4s;}}
#score-val.show{{opacity:1;transform:translateY(0);}}
.badge-pill{{background:{_btbg};color:{_btc};border:1px solid {_btbd};border-radius:20px;
  padding:2px 10px;font-size:.7rem;font-weight:700;font-family:Inter,sans-serif;margin-left:8px;}}
#words-lbl{{font-size:.65rem;color:#b4cde4;font-family:'Share Tech Mono',monospace;opacity:0;transition:opacity .5s;}}
#words-lbl.show{{opacity:1;}}
/* ── Dimension bars ── */
#dim-bars{{margin:.15rem 0 .5rem;display:flex;flex-direction:column;gap:5px;}}
.dim-row{{display:flex;align-items:center;gap:8px;}}
.dim-label{{font-size:.58rem;color:#94b0d8;font-family:'Share Tech Mono',monospace;
  font-weight:700;letter-spacing:.05em;width:58px;flex-shrink:0;text-align:right;}}
.dim-track{{flex:1;height:5px;background:rgba(255,255,255,.06);border-radius:3px;overflow:hidden;}}
.dim-fill{{height:100%;border-radius:3px;width:0%;transition:width 1s cubic-bezier(.25,.46,.45,.94);}}
.dim-pct{{font-size:.58rem;font-family:'Share Tech Mono',monospace;color:#b4cde4;
  min-width:30px;text-align:right;opacity:0;transition:opacity .5s;}}
.dim-pct.show{{opacity:1;}}
/* ── Keyword scan beam ── */
#scan-wrap{{
  background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);
  border-radius:8px;padding:.6rem .8rem;margin-bottom:.55rem;
  position:relative;overflow:hidden;
  font-size:.8rem;line-height:1.75;color:#c7e8ff;font-family:Inter,sans-serif;
}}
#scan-beam{{
  position:absolute;top:0;left:-10%;height:100%;width:10%;
  background:linear-gradient(90deg,transparent 0%,rgba(0,255,209,.16) 40%,rgba(0,255,209,.32) 60%,transparent 100%);
  pointer-events:none;display:none;
}}
.kw-token{{display:inline;border-radius:3px;padding:1px 3px;
  transition:background .25s,color .25s,box-shadow .25s;}}
.kw-token.lit{{background:rgba(0,255,209,.22);color:#00FFD1;box-shadow:0 0 6px rgba(0,255,209,.3);}}
/* ── Feedback lines ── */
#fb-lines{{margin-top:.35rem;}}
.fb-line{{
  display:flex;align-items:flex-start;gap:8px;
  font-size:.82rem;color:#e2e8f0;font-family:Inter,sans-serif;line-height:1.6;
  opacity:0;transform:translateX(-8px);transition:opacity .45s ease,transform .45s ease;
  margin-bottom:.3rem;
}}
.fb-line.show{{opacity:1;transform:translateX(0);}}
.fb-bullet{{flex-shrink:0;width:6px;height:6px;border-radius:50%;background:{_rc_left};margin-top:7px;}}
.cursor-blink{{display:inline-block;width:2px;height:.9em;background:{_rc_txt};
  vertical-align:middle;margin-left:2px;animation:cur .65s step-end infinite;}}
@keyframes cur{{0%,100%{{opacity:1}}50%{{opacity:0}}}}
</style>
</head><body>

<div id="eval-card">
  <div id="eval-header">
    <span id="score-val">
      {icon}&nbsp;{fs:.0f}%
      <span class="badge-pill">{bt}</span>
      {_src_badge_html}
    </span>
    <span id="words-lbl">{wc} WORDS</span>
  </div>
  <div id="dim-bars"></div>
</div>

{'<div id="scan-wrap"><div id="scan-beam"></div><span id="scan-text"></span></div>' if _answer_for_scan else ""}

<div id="fb-lines"></div>

<script>
(function(){{
  var dims   = {_dims_json};
  var kws    = {_kw_json};
  var ansRaw = {_ans_json};
  var lines  = {_fb_json};

  var scoreEl = document.getElementById('score-val');
  var wordsEl = document.getElementById('words-lbl');
  var dimBar  = document.getElementById('dim-bars');
  var fbDiv   = document.getElementById('fb-lines');
  var scanWrap= document.getElementById('scan-wrap');
  var beam    = document.getElementById('scan-beam');
  var scanTxt = document.getElementById('scan-text');

  /* 1 — Score header fade-in */
  setTimeout(function(){{ scoreEl.classList.add('show'); }}, 80);
  setTimeout(function(){{ wordsEl.classList.add('show'); }}, 260);

  /* 2 — Dimension bars */
  dims.forEach(function(d, i){{
    var label = d[0], pct = Math.round(d[1]), color = d[2];
    var row = document.createElement('div');
    row.className = 'dim-row';
    row.innerHTML =
      '<div class="dim-label">' + label + '</div>' +
      '<div class="dim-track"><div class="dim-fill" id="df' + i + '" style="background:' + color + '"></div></div>' +
      '<div class="dim-pct" id="dp' + i + '">' + pct + '%</div>';
    dimBar.appendChild(row);
    setTimeout(function(){{
      document.getElementById('df' + i).style.width = pct + '%';
      document.getElementById('dp' + i).classList.add('show');
    }}, 380 + i * 110);
  }});

  /* 3 — Keyword scan beam */
  if (scanTxt && kws.length > 0 && ansRaw) {{
    var escapedKws = kws.map(function(k){{ return k.toLowerCase(); }});
    var tokens = ansRaw.split(/(\s+)/);
    var html = ''; var idx = 0;
    tokens.forEach(function(tok){{
      if (/^\s+$/.test(tok)) {{ html += tok; return; }}
      var clean = tok.replace(/[^a-z0-9]/gi, '').toLowerCase();
      var hit = escapedKws.some(function(k){{ return clean === k || clean.includes(k); }});
      if (hit) {{ html += '<span class="kw-token" id="kwt' + idx + '">' + tok + '</span>'; }}
      else     {{ html += tok; }}
      idx++;
    }});
    scanTxt.innerHTML = html;

    setTimeout(function(){{
      var wrap  = document.getElementById('scan-wrap');
      var wrapW = wrap.offsetWidth;
      beam.style.width  = (wrapW * 0.1) + 'px';
      beam.style.display= 'block';
      var duration = Math.max(1400, ansRaw.length * 5);
      var start = null; var litSet = new Set();
      var matchedIds = [];
      wrap.querySelectorAll('.kw-token').forEach(function(el){{ matchedIds.push(el.id); }});
      function step(ts){{
        if (!start) start = ts;
        var prog   = Math.min((ts - start) / duration, 1);
        var leftPx = -0.1 * wrapW + prog * (wrapW + 0.1 * wrapW);
        beam.style.left = leftPx + 'px';
        matchedIds.forEach(function(id){{
          if (litSet.has(id)) return;
          var el = document.getElementById(id); if (!el) return;
          if (el.offsetLeft >= leftPx - 4 && el.offsetLeft <= leftPx + wrapW * 0.12) {{
            el.classList.add('lit'); litSet.add(id);
          }}
        }});
        if (prog < 1) requestAnimationFrame(step);
        else beam.style.display = 'none';
      }}
      requestAnimationFrame(step);
    }}, 500);
  }}

  /* 4 — Streaming feedback lines */
  var FB_OFFSET = 700 + dims.length * 110;
  lines.forEach(function(txt, i){{
    setTimeout(function(){{
      var row  = document.createElement('div'); row.className = 'fb-line';
      var dot  = document.createElement('div'); dot.className = 'fb-bullet';
      var span = document.createElement('span'); span.textContent = txt;
      if (i === lines.length - 1) {{
        var cur = document.createElement('span'); cur.className = 'cursor-blink';
        span.appendChild(cur);
        setTimeout(function(){{ cur.style.display = 'none'; }}, 2800);
      }}
      row.appendChild(dot); row.appendChild(span);
      fbDiv.appendChild(row);
      requestAnimationFrame(function(){{
        requestAnimationFrame(function(){{ row.classList.add('show'); }});
      }});
    }}, FB_OFFSET + i * 320);
  }});
}})();
</script>
</body></html>""", height=_height)

    # Stop the border pulse now that results are visible
    _stop_eval_border_pulse()

    # ── STAR strip (Behavioural / HR only) ───────────────────────────────────
    if star and _star_applicable:
        parts = ""
        for comp in ["Situation","Task","Action","Result"]:
            hit = star.get(comp, False)
            ic  = "#00FFD1" if hit else "rgba(255,255,255,.05)"
            tc  = "#050A14"  if hit else "#5a7a9a"
            parts += (f'<div style="flex:1;text-align:center;padding:5px 0;border:1px solid {ic};'
                      f'background:{ic if hit else "transparent"};border-radius:3px;color:{tc};'
                      f'font-size:.62rem;font-weight:700;font-family:Share Tech Mono,monospace;">{comp[0]}<br>'
                      f'<span style="font-size:.48rem;">{comp.upper()}</span></div>')
        st.markdown(f'<div style="display:flex;gap:3px;margin:.45rem 0;">{parts}</div>', unsafe_allow_html=True)

    # ── Grammar errors expander ───────────────────────────────────────────────
    errs = ev.get("grammar_errors", [])
    if errs:
        with st.expander(f"▸ {len(errs)} Grammar Issues"):
            for i, err in enumerate(errs[:5]):
                st.markdown(f"**Issue {i+1}:** {err['message']}")
                st.markdown(f"<span style='color:#c7e8ff;font-size:.72rem;'>Context: …{err['context']}…</span>", unsafe_allow_html=True)
                if err.get("replacements"):
                    st.markdown(f"💡 `{', '.join(err['replacements'])}`")
                if i < len(errs)-1: st.divider()


# ══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION WIDGET (v8.1)
# ══════════════════════════════════════════════════════════════════════════════
def render_calibration_widget() -> None:
    """
    v8.1 — Warmup step shown BEFORE question 1.

    Replaces the pangram prompt with a natural conversational question so the
    candidate enters the right headspace while the system quietly measures
    their personal voice baseline.  The candidate never sees the word
    "calibration" — they just answer a warmup question.

    Three explicit states handled:
      1. Voice pipeline not ready  → show a friendly "checking mic" message
         and offer Skip so the candidate is never blocked.
      2. Audio recorded            → run calibrate_voice_baseline(), confirm
         warmup complete, rerun into Q1.
      3. Skip pressed              → set calibration_skipped=True, fall back
         to absolute nervousness scores (original behaviour), rerun into Q1.
    """
    if (st.session_state.get("calibration_done") or
            st.session_state.get("calibration_skipped")):
        return

    role = st.session_state.get("target_role", "the role")

    # ── Voice pipeline not ready — don't block the candidate ─────────────────
    if not engine.is_voice_pipeline_ready():
        st.markdown(f"""
<div style="background:rgba(99,102,241,.07);border:1px solid rgba(99,102,241,.2);
  border-radius:14px;padding:1.1rem 1.4rem;margin-bottom:1rem;">
  <div style="font-size:.95rem;font-weight:700;color:#dde8ff;margin-bottom:.35rem;">
    Almost ready — just getting your mic set up
  </div>
  <div style="font-size:.8rem;color:#94b0d8;line-height:1.6;">
    The voice analysis model is still loading. You can skip ahead and your
    interview will start immediately — voice feedback will be available once
    the model finishes in the background.
  </div>
</div>""", unsafe_allow_html=True)
        if st.button("Continue to interview →", key="cal_skip_nomodel",
                     use_container_width=True):
            st.session_state["calibration_skipped"] = True
            st.rerun()
        return

    # ── Main warmup card ──────────────────────────────────────────────────────
    st.markdown(f"""
<div style="background:rgba(0,212,200,.05);border:1px solid rgba(0,212,200,.2);
  border-radius:14px;padding:1.4rem 1.6rem;margin-bottom:1.1rem;">

  <!-- Step label -->
  <div style="display:inline-block;font-size:.6rem;font-family:'Share Tech Mono',monospace;
    color:#00d4c8;letter-spacing:.18em;background:rgba(0,212,200,.1);
    border:1px solid rgba(0,212,200,.25);border-radius:20px;
    padding:3px 12px;margin-bottom:.85rem;">
    WARMUP — before we begin
  </div>

  <!-- Headline -->
  <div style="font-size:1.1rem;font-weight:700;color:#f1f5f9;margin-bottom:.5rem;
    font-family:Inter,sans-serif;line-height:1.4;">
    Let's start with a quick introduction
  </div>

  <!-- Human explanation — no technical jargon -->
  <div style="font-size:.82rem;color:#94b0d8;line-height:1.75;margin-bottom:1.1rem;
    font-family:Inter,sans-serif;">
    Before your first question, take 20–30 seconds to introduce yourself
    naturally. This helps settle nerves and lets the system adjust to
    <em>your</em> voice so feedback during the interview is more accurate.
  </div>

  <!-- The warmup question — styled like the real interview question card -->
  <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);
    border-left:3px solid #00d4c8;border-radius:10px;
    padding:1rem 1.2rem;margin-bottom:.2rem;">
    <div style="font-size:.6rem;font-weight:700;color:#00d4c8;
      font-family:'Share Tech Mono',monospace;letter-spacing:.1em;
      text-transform:uppercase;margin-bottom:.5rem;">
      Warmup question
    </div>
    <div style="font-size:1.25rem;color:#f1f5f9;font-weight:600;
      font-family:Inter,sans-serif;line-height:1.55;">
      Tell me briefly about your background and what draws you to
      the <span style="color:#00d4c8;">{role}</span> role today.
    </div>
  </div>

</div>""", unsafe_allow_html=True)

    # ── Audio recorder ────────────────────────────────────────────────────────
    audio_val = st.audio_input(
        "Record your introduction",
        key="calibration_audio",
        label_visibility="collapsed",
    )

    col_cal, col_skip = st.columns([3, 1])

    with col_cal:
        if audio_val is not None:
            raw = audio_val.read()
            if raw and len(raw) > 1024:
                with st.spinner("Getting ready…"):
                    result = engine.calibrate_voice_baseline(raw)

                calibrated = result.get("calibrated", False)
                st.session_state.update({
                    "calibration_done":     True,
                    "calibration_baseline": result.get("baseline", 0.2),
                })

                if calibrated:
                    st.success("✅ Great — your voice is calibrated. Starting your interview now.")
                else:
                    # Pipeline accepted the audio but couldn't calibrate — not
                    # a blocker; absolute scores will be used transparently.
                    st.info("✓ Warmup recorded. Starting your interview now.")

                st.rerun()
            else:
                # Audio widget returned something but it's too short / empty
                st.warning("Recording seems too short — try speaking for at least 15 seconds.")

    with col_skip:
        st.markdown("<div style='padding-top:.3rem;'></div>", unsafe_allow_html=True)
        if st.button("Skip warmup", key="cal_skip",
                     use_container_width=True):
            st.session_state["calibration_skipped"] = True
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard() -> None:
    name    = st.session_state.candidate_name
    answers = st.session_state.session_answers
    has_data = len(answers) > 0

    primary_answers = [a for a in answers if not a.get("is_follow_up", False)]
    _avg_answers    = answers

    if has_data:
        avg_score   = round(sum(a["score"] for a in _avg_answers) / len(_avg_answers), 2)
        avg_emo     = round(sum(a.get("nervousness", 0.2) for a in _avg_answers) / len(_avg_answers), 2)
        avg_voice   = avg_emo
        avg_depth   = round(sum(a.get("depth_score", a.get("score", 2.5)) for a in _avg_answers) / len(_avg_answers), 2)
        avg_fluency = round(sum(a.get("fluency", a.get("fluency_score", 3.5)) for a in _avg_answers) / len(_avg_answers), 2)
        sc = {
            "final":     avg_score,
            "knowledge": avg_score,
            "emotion":   round((1 - avg_emo)   * 5, 2),
            "voice":     round((1 - avg_voice)  * 5, 2),
            "depth":     avg_depth,
            "fluency":   avg_fluency,
        }
    else:
        sc = None

    bc, bt = badge(sc["final"]) if sc else ("b-av", "No sessions yet")

    # ══════════════════════════════════════════════════════════════════════════
    #  ANIMATED DASHBOARD — full redesign
    # ══════════════════════════════════════════════════════════════════════════

    import datetime as _dt
    _today_str  = _dt.date.today().isoformat()
    _last_str   = st.session_state.get("last_practice_date", "")
    _streak     = st.session_state.get("practice_streak", 0)
    if _last_str != _today_str:
        _yesterday = (_dt.date.today() - _dt.timedelta(days=1)).isoformat()
        _streak = _streak + 1 if (_last_str == _yesterday or _streak == 0) else 1
        st.session_state["practice_streak"]    = _streak
        st.session_state["last_practice_date"] = _today_str
    _flame = "🔥" if _streak >= 3 else "⚡"

    components.html(f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=JetBrains+Mono:wght@400;500&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Syne',sans-serif;background:transparent;color:#fff;overflow-x:hidden;}}

.scan-line{{position:fixed;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,rgba(0,229,255,.7),rgba(0,229,255,1),rgba(0,229,255,.7),transparent);box-shadow:0 0 14px 4px rgba(0,200,255,.4);animation:scan 5s linear infinite;pointer-events:none;z-index:9999;}}
@keyframes scan{{0%{{top:-2px;opacity:0}}5%{{opacity:1}}95%{{opacity:1}}100%{{top:100vh;opacity:0}}}}

.crt{{position:fixed;inset:0;pointer-events:none;z-index:9998;background:repeating-linear-gradient(to bottom,transparent,transparent 3px,rgba(0,180,255,.012) 3px,rgba(0,180,255,.012) 4px);}}

.glass{{background:rgba(8,14,35,.82);backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);border:1px solid rgba(0,229,255,.18);border-radius:20px;padding:2rem 2.2rem;margin-bottom:1.4rem;position:relative;overflow:hidden;box-shadow:0 8px 36px rgba(0,0,0,.55),inset 0 1px 0 rgba(0,229,255,.08);}}
.glass::before{{content:'';position:absolute;top:0;left:15%;right:15%;height:1px;background:linear-gradient(90deg,transparent,rgba(0,229,255,.55),transparent);}}

/* Staggered fade-up */
.g1{{animation:fu .55s ease both .05s;}}
.g2{{animation:fu .55s ease both .18s;}}
.g3{{animation:fu .55s ease both .30s;}}
@keyframes fu{{from{{opacity:0;transform:translateY(16px)}}to{{opacity:1;transform:translateY(0)}}}}

/* Hero */
.hero-panel{{text-align:center;}}
.hero-badge{{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:.63rem;color:#00e5ff;border:1px solid rgba(0,229,255,.3);padding:5px 20px;border-radius:30px;background:rgba(0,229,255,.05);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:1.4rem;animation:badge-pulse 3s ease-in-out infinite;}}
@keyframes badge-pulse{{0%,100%{{box-shadow:0 0 0 0 rgba(0,229,255,0)}}50%{{box-shadow:0 0 18px 4px rgba(0,229,255,.18)}}}}
.hero-title{{font-family:'Orbitron',monospace;font-size:clamp(1.45rem,3vw,2.5rem);font-weight:900;line-height:1.2;color:#fff;margin-bottom:.9rem;text-shadow:0 0 30px rgba(0,229,255,.25),0 2px 10px rgba(0,0,0,.7);animation:glitch 9s ease-in-out infinite;}}
.hero-title span{{color:#00e5ff;}}
@keyframes glitch{{0%,94%,100%{{text-shadow:0 0 30px rgba(0,229,255,.25),0 2px 10px rgba(0,0,0,.7)}}95%{{text-shadow:-2px 0 rgba(255,0,80,.5),2px 0 rgba(0,229,255,.5)}}97%{{text-shadow:2px 0 rgba(255,0,80,.4),-1px 0 rgba(0,229,255,.4)}}}}
.hero-desc{{font-size:.9rem;color:rgba(200,225,255,.72);max-width:660px;margin:0 auto;line-height:1.8;}}
.hero-desc strong{{color:#a5b4fc;}}

/* Marquee */
.mq-wrap{{overflow:hidden;padding:9px 0;margin-top:1.4rem;border-top:1px solid rgba(0,229,255,.07);border-bottom:1px solid rgba(0,229,255,.07);background:rgba(0,0,0,.2);}}
.mq-row{{display:flex;gap:56px;white-space:nowrap;margin-bottom:3px;}}
.mq-row:last-child{{margin-bottom:0;}}
.mq-row span{{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.18em;flex-shrink:0;}}
.mq-sep{{opacity:.28;margin:0 10px;}}
.r1 span{{color:#00ff88;animation:mqL 22s linear infinite;}}
.r2 span{{color:#00d4ff;animation:mqR 28s linear infinite;}}
.r3 span{{color:#bf00ff;animation:mqL 18s linear infinite;}}
@keyframes mqL{{from{{transform:translateX(0)}}to{{transform:translateX(-50%)}}}}
@keyframes mqR{{from{{transform:translateX(-50%)}}to{{transform:translateX(0)}}}}

/* Stats */
.stats-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:1.1rem;}}
.stat-card{{background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.06);border-radius:14px;padding:1.4rem 1rem;text-align:center;transition:all .3s;cursor:default;position:relative;overflow:hidden;}}
.stat-card::after{{content:'';position:absolute;inset:0;background:radial-gradient(circle at 50% 0%,rgba(0,229,255,.07),transparent 65%);opacity:0;transition:opacity .3s;}}
.stat-card:hover{{border-color:rgba(0,229,255,.38);transform:translateY(-5px);}}
.stat-card:hover::after{{opacity:1;}}
.stat-value{{font-family:'Orbitron',monospace;font-size:1.85rem;font-weight:700;color:#00e5ff;margin-bottom:.35rem;animation:cglow 3s ease-in-out infinite;}}
@keyframes cglow{{0%,100%{{text-shadow:0 0 8px rgba(0,229,255,.3)}}50%{{text-shadow:0 0 22px rgba(0,229,255,.7),0 0 40px rgba(0,180,255,.3)}}}}
.stat-label{{font-size:.7rem;color:rgba(255,255,255,.45);text-transform:uppercase;letter-spacing:.08em;font-family:'JetBrains Mono',monospace;}}

/* Features */
.feat-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:1.1rem;}}
.feat-card{{background:rgba(0,0,0,.22);border:1px solid rgba(255,255,255,.06);border-radius:14px;padding:1.5rem 1.4rem;transition:all .35s;position:relative;overflow:hidden;}}
.feat-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,rgba(0,229,255,.45),transparent);transform:scaleX(0);transition:transform .35s;}}
.feat-card:hover{{border-color:rgba(0,229,255,.3);background:rgba(0,229,255,.04);transform:translateY(-4px);}}
.feat-card:hover::before{{transform:scaleX(1);}}
.feat-icon{{font-size:1.7rem;margin-bottom:.85rem;}}
.feat-title{{font-family:'Orbitron',monospace;font-size:.8rem;font-weight:700;color:#fff;margin-bottom:.55rem;letter-spacing:.04em;}}
.feat-text{{font-size:.78rem;color:rgba(255,255,255,.52);line-height:1.65;}}
.feat-text strong{{color:#a5b4fc;}}

/* Streak */
.streak-wrap{{display:flex;align-items:center;justify-content:center;gap:.5rem;margin-top:1.4rem;}}
.streak-pill{{display:inline-flex;align-items:center;gap:.45rem;background:rgba(0,255,209,.07);border:1px solid rgba(0,255,209,.25);border-radius:30px;padding:5px 16px 5px 12px;animation:spulse 2.5s ease-in-out infinite;}}
@keyframes spulse{{0%,100%{{box-shadow:0 0 0 0 rgba(0,255,209,0)}}50%{{box-shadow:0 0 14px 3px rgba(0,255,209,.15)}}}}
.streak-num{{font-family:'Orbitron',monospace;font-size:.85rem;font-weight:700;color:#00FFD1;}}
.streak-lbl{{font-size:.65rem;color:rgba(255,255,255,.38);font-family:'JetBrains Mono',monospace;}}

@media(max-width:768px){{.stats-grid{{grid-template-columns:repeat(2,1fr)}}.feat-grid{{grid-template-columns:1fr}}.hero-title{{font-size:1.3rem;}}}}
</style>
</head>
<body>
<div class="scan-line"></div>
<div class="crt"></div>

<div class="glass hero-panel g1">
  <div class="hero-badge">⬡ AI-Based Multi-Modal · Interview Intelligence System</div>
  <div class="hero-title">Master Your Next Job Interview<br><span>with Expert Preparation</span></div>
  <div class="hero-desc">The most comprehensive AI mock interview platform trusted by job seekers worldwide. Practice with real questions from <strong>Google, Amazon, Microsoft, JPMorgan</strong> and many more.</div>
  <div class="mq-wrap">
    <div class="mq-row r1"><span>EMOTION DETECTION <span class="mq-sep">◆</span> SBERT SCORING <span class="mq-sep">◆</span> RL SEQUENCER <span class="mq-sep">◆</span> WEBRTC 30FPS <span class="mq-sep">◆</span> VOICE ANALYSIS <span class="mq-sep">◆</span> DISC KEYWORDS <span class="mq-sep">◆</span> POSTURE SCORING <span class="mq-sep">◆</span> FOLLOW-UP ENGINE</span><span>EMOTION DETECTION <span class="mq-sep">◆</span> SBERT SCORING <span class="mq-sep">◆</span> RL SEQUENCER <span class="mq-sep">◆</span> WEBRTC 30FPS <span class="mq-sep">◆</span> VOICE ANALYSIS <span class="mq-sep">◆</span> DISC KEYWORDS <span class="mq-sep">◆</span> POSTURE SCORING <span class="mq-sep">◆</span> FOLLOW-UP ENGINE</span></div>
    <div class="mq-row r2"><span>AI AVATAR ARIA-7 <span class="mq-sep">◈</span> GRAMMAR CHECK <span class="mq-sep">◈</span> RESUME PARSER <span class="mq-sep">◈</span> JD MATCHING <span class="mq-sep">◈</span> HR ROUND COACH <span class="mq-sep">◈</span> CONFIDENCE INDEX <span class="mq-sep">◈</span> LIVE COACHING <span class="mq-sep">◈</span> PDF REPORT</span><span>AI AVATAR ARIA-7 <span class="mq-sep">◈</span> GRAMMAR CHECK <span class="mq-sep">◈</span> RESUME PARSER <span class="mq-sep">◈</span> JD MATCHING <span class="mq-sep">◈</span> HR ROUND COACH <span class="mq-sep">◈</span> CONFIDENCE INDEX <span class="mq-sep">◈</span> LIVE COACHING <span class="mq-sep">◈</span> PDF REPORT</span></div>
    <div class="mq-row r3"><span>MULTIMODAL FUSION <span class="mq-sep">⬡</span> NERVOUSNESS SCORE <span class="mq-sep">⬡</span> EYE CONTACT <span class="mq-sep">⬡</span> STAR FRAMEWORK <span class="mq-sep">⬡</span> WHISPER STT <span class="mq-sep">⬡</span> ADAPTIVE SCORING <span class="mq-sep">⬡</span> LIVE SIGNALS</span><span>MULTIMODAL FUSION <span class="mq-sep">⬡</span> NERVOUSNESS SCORE <span class="mq-sep">⬡</span> EYE CONTACT <span class="mq-sep">⬡</span> STAR FRAMEWORK <span class="mq-sep">⬡</span> WHISPER STT <span class="mq-sep">⬡</span> ADAPTIVE SCORING <span class="mq-sep">⬡</span> LIVE SIGNALS</span></div>
  </div>
</div>

<div class="glass g2">
  <div class="stats-grid">
    <div class="stat-card"><div class="stat-value" id="c1">—</div><div class="stat-label">Real Questions</div></div>
    <div class="stat-card"><div class="stat-value" id="c2">0</div><div class="stat-label">Top Companies</div></div>
    <div class="stat-card"><div class="stat-value" id="c3">0</div><div class="stat-label">AI Layers</div></div>
    <div class="stat-card"><div class="stat-value" id="c4" style="font-size:1.55rem;">—</div><div class="stat-label">Live Analysis</div></div>
  </div>
</div>

<div class="glass g3">
  <div class="feat-grid">
    <div class="feat-card">
      <div class="feat-icon">🎯</div>
      <div class="feat-title">Build Confidence</div>
      <div class="feat-text">Candidates who practice are <strong>3× more likely to succeed</strong>. Beat nervousness and structure answers with the <strong>STAR method</strong>.</div>
    </div>
    <div class="feat-card">
      <div class="feat-icon">📊</div>
      <div class="feat-title">Instant AI Feedback</div>
      <div class="feat-text"><strong>STAR scoring, OCEAN personality, DISC profiling</strong>, emotion tracking, nervousness detection — like a personal <strong>interview coach</strong>.</div>
    </div>
    <div class="feat-card">
      <div class="feat-icon">🏢</div>
      <div class="feat-title">Real Company Questions</div>
      <div class="feat-text"><strong>Real questions</strong> from 65+ top companies. All major career paths with <strong>industry-specific questions</strong> for your exact role.</div>
    </div>
  </div>
  <div class="streak-wrap">
    <div class="streak-pill">
      <span style="font-size:1rem;">{_flame}</span>
      <span class="streak-num">{_streak}</span>
      <span class="streak-lbl">day streak</span>
    </div>
  </div>
</div>

<script>
function animCount(id, target, suffix, dur){{
  var el=document.getElementById(id); if(!el) return;
  var s=0, step=target/(dur/16);
  var iv=setInterval(function(){{
    s=Math.min(s+step,target);
    el.textContent=(s>=target?target:Math.floor(s))+suffix;
    if(s>=target) clearInterval(iv);
  }},16);
}}
setTimeout(function(){{
  document.getElementById('c1').textContent='10M+';
  animCount('c2',65,'+',1100);
  animCount('c3',9,'',800);
  setTimeout(function(){{document.getElementById('c4').textContent='Live';}},500);
}},350);
</script>
</body></html>""", height=800)

    # ══════════════════════════════════════════════════════════════════════════
    #  ACTION BUTTONS
    # ══════════════════════════════════════════════════════════════════════════
    col_left, col_right = st.columns(2)
    with col_left:
        st.button("▶  START INTERVIEW",
                  key="db_start_e" if not has_data else "db_start_d",
                  on_click=nav_to("Start Interview"),
                  use_container_width=True)
    with col_right:
        st.button("▶  HR PRACTICE",
                  key="db_hr_e" if not has_data else "db_hr_d",
                  on_click=nav_to("HR Practice"),
                  use_container_width=True)

    st.button("📅  WEEKLY PREP PLAN  — sequence every module into a structured 7-day loop",
              key="db_weekly_plan",
              on_click=nav_to("Weekly Plan"),
              use_container_width=True)

    st.button("🎯  PLACEMENT TEST  — 10 MCQ Aptitude · 5 Technical · 5 HR with AI evaluation & PDF report",
              key="db_placement_test",
              on_click=nav_to("Placement Setup"),
              use_container_width=True)

    if not has_data:
        return

    # ══════════════════════════════════════════════════════════════════════════
    #  SESSION DATA — shown only after at least one interview
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    cs = st.columns(6)
    for col, lbl, k in zip(cs,
        ["Score","Emotion","Voice","Knowledge","Depth","Fluency"],
        ["final","emotion","voice","knowledge","depth","fluency"]):
        mchip(lbl, sc.get(k, 3.5), col=col)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown('<div class="card hex-bg">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top:0;font-size:.85rem;">Readiness gauge</h4>', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=sc["final"] * 20,
            delta={"reference": 70},
            title={"text":"Readiness %","font":{"color":"#5a7a9a","family":"Orbitron","size":11}},
            gauge={"axis":{"range":[0,100],"tickfont":{"color":"#5a7a9a","size":8}},
                   "bar":{"color":"#00FFD1","thickness":.26},
                   "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                   "threshold":{"line":{"color":"#00D4FF","width":2},"thickness":.85,"value":70},
                   "steps":[{"range":[0,40],"color":"rgba(255,51,102,.08)"},
                             {"range":[40,70],"color":"rgba(255,170,0,.05)"},
                             {"range":[70,100],"color":"rgba(0,255,136,.07)"}]},
            number={"suffix":"%","font":{"color":"#00FFD1","family":"Orbitron","size":26}},
        ))
        dcl(fig, 230); st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown(f'<div style="text-align:center;margin-top:.3rem;"><span class="badge {bc}">{sc["final"]:.2f}/5 · {bt}</span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Start new interview", key="db_s",
                  on_click=nav_to("Start Interview"))
        st.button("View full report", key="db_r",
                  on_click=nav_to("Final Report"))

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top:0;font-size:.85rem;">Performance radar</h4>', unsafe_allow_html=True)
        cats = ["Emotion","Voice","Knowledge","Depth","Fluency"]
        vals = [sc.get(k.lower(), 3.5) for k in cats]
        fig2 = go.Figure(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself",
            fillcolor="rgba(0,255,136,.05)", line=dict(color="#00FFD1", width=2),
            marker=dict(color="#00FFD1", size=5),
        ))
        fig2.update_layout(polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,5], gridcolor="rgba(0,255,136,.08)",
                            tickfont=dict(color="#5a7a9a", size=8),
                            linecolor="rgba(0,255,136,.1)"),
            angularaxis=dict(gridcolor="rgba(0,255,136,.06)",
                             tickfont=dict(color="#4a9eff", size=11, family="Orbitron")),
        ), showlegend=False)
        dcl(fig2, 280); st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top:0;font-size:.85rem;">Session summary</h4>', unsafe_allow_html=True)
        hist_rows = []
        for i, a in enumerate(answers):
            bc_a, bt_a = badge(a.get("score", 0))
            hist_rows.append({
                "Q": f"Q{a.get('number', i+1)}",
                "Type": a.get("type", "—"),
                "Score": f"{a.get('score', 0):.1f}/5",
                "Grade": bt_a,
                "Nervousness": f"{int(a.get('nervousness', 0.2)*100)}%",
                "Time": f"{a.get('time_s', 0)//60}m {a.get('time_s', 0)%60:02d}s",
            })
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Feature 2: Confidence Heatmap Timeline ────────────────────────────
        if len(answers) >= 2:
            st.markdown('<div class="card" style="padding:1rem 1.2rem;">', unsafe_allow_html=True)
            st.markdown('<h4 style="margin:0 0 .6rem;font-size:.82rem;">Confidence timeline</h4>', unsafe_allow_html=True)

            _hm_cells = ""
            for _hi, _a in enumerate(answers):
                _cs  = _a.get("confidence_score", _a.get("score", 3.5))
                _pct = min(100, int(_cs / 5 * 100))
                # deep blue → cyan → green gradient
                if _pct < 40:
                    _cell_col = f"rgba(30,60,180,{0.4 + _pct/100:.2f})"
                    _txt_col  = "#c7e8ff"
                elif _pct < 70:
                    _cell_col = f"rgba(0,180,220,{0.35 + _pct/140:.2f})"
                    _txt_col  = "#00d4ff"
                else:
                    _cell_col = f"rgba(0,{int(180 + (_pct-70)*2.5)},140,{0.45 + (_pct-70)/200:.2f})"
                    _txt_col  = "#00FFD1"
                _nerv_pct_hm = int(_a.get("nervousness", 0.2) * 100)
                _hm_cells += f"""<div title="Q{_a.get('number',_hi+1)}: {_cs:.1f}/5 · Nerv {_nerv_pct_hm}%"
  style="flex:1;min-width:0;border-radius:6px;background:{_cell_col};
  border:1px solid rgba(255,255,255,.08);padding:.35rem .2rem;
  display:flex;flex-direction:column;align-items:center;gap:2px;cursor:default;">
  <div style="font-size:.6rem;color:{_txt_col};font-family:Orbitron,monospace;
    font-weight:700;line-height:1;">{_cs:.1f}</div>
  <div style="font-size:.5rem;color:rgba(255,255,255,.45);font-family:Inter,sans-serif;">
    Q{_a.get('number', _hi+1)}</div>
</div>"""

            st.markdown(f"""
<div style="display:flex;gap:4px;align-items:stretch;margin-bottom:.4rem;">
  {_hm_cells}
</div>
<div style="display:flex;justify-content:space-between;align-items:center;margin-top:.2rem;">
  <span style="font-size:.58rem;color:#334155;font-family:Inter,sans-serif;">Low confidence</span>
  <div style="flex:1;height:4px;margin:0 .5rem;border-radius:2px;
    background:linear-gradient(90deg,rgba(30,60,180,.6),rgba(0,180,220,.7),rgba(0,255,200,.8));"></div>
  <span style="font-size:.58rem;color:#334155;font-family:Inter,sans-serif;">High confidence</span>
</div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL SETUP
# ══════════════════════════════════════════════════════════════════════════════
def page_setup() -> None:
    st.markdown("""
<div style="margin-bottom:1.2rem;">
  <div style="font-size:.62rem;font-family:Share Tech Mono,monospace;color:#00d4ff;letter-spacing:.2em;margin-bottom:.2rem;">
    // NEURAL PIPELINE INITIALIZATION
  </div>
  <h2 style="margin:0;font-size:1.6rem;">MODEL SETUP</h2>
</div>""", unsafe_allow_html=True)

    steps = [
        ("⬡","RAF-DB","In-wild\nfaces"),
        ("⬡","HOG+LBP","2020-dim\nvectors"),
        ("⬡","Face MLP","512→256\n→128→7"),
        ("⬡","CREMA-D","91 actors\n7,442 clips"),
        ("⬡","TESS","2,800 clips\nstudio"),
        ("⬡","108-dim","MFCC+Delta\n+Prosodic"),
        ("⬡","MediaPipe","Pose+Mesh\n+Hands"),
        ("⬡","Whisper","OpenAI ASR\nbase model"),
    ]
    st.markdown('<div style="display:grid;grid-template-columns:repeat(8,1fr);gap:.4rem;margin:1rem 0;">', unsafe_allow_html=True)
    for icon, title, desc in steps:
        st.markdown(f"""
<div class="card card-sm" style="text-align:center;padding:.6rem .35rem;">
  <div style="font-size:1.2rem;color:#00ff88;">{icon}</div>
  <div style="font-size:.65rem;font-weight:700;color:#e0f0ff;margin:.2rem 0;
    font-family:Orbitron,monospace;">{title}</div>
  <div style="font-size:.55rem;color:#c7e8ff;white-space:pre-line;font-family:Share Tech Mono,monospace;">{desc}</div>
</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    stt_ok    = st.session_state.stt_ready
    fer_ready = st.session_state.fer_ready
    nerv_ready= st.session_state.nervousness_ready

    if stt_ok:
        st.markdown('<div class="b-ok">◈ WHISPER ASR ONLINE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="b-warn">▲ WHISPER OFFLINE — <code>pip install transformers torch soundfile</code></div>', unsafe_allow_html=True)

    # v9.0: WebRTC continuous stream status
    if WEBRTC_OK:
        st.markdown('<div class="b-ok">◈ WEBRTC STREAM ONLINE — continuous 30fps emotion analysis active</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="b-warn">▲ WEBRTC OFFLINE — snapshot mode only. '
                    'Run: <code>pip install streamlit-webrtc aiortc</code></div>',
                    unsafe_allow_html=True)
    # v8.0: SBERT and RL status banners
    if SBERT_AVAILABLE:
        st.markdown(f'<div class="b-ok">◈ SBERT ONLINE — {SBERT_MODEL} (offline semantic relevance · Layer-2 fallback)</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="b-info">◑ SBERT OFFLINE — run: <code>pip install sentence-transformers</code> '
                    '(adds semantic relevance scoring without Groq)</div>', unsafe_allow_html=True)
    if RL_AVAILABLE:
        st.markdown('<div class="b-ok">◈ RL SEQUENCER ONLINE — Q-learning adaptive difficulty active</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="b-warn">▲ RL SEQUENCER OFFLINE — ensure adaptive_sequencer.py is in same directory</div>',
                    unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top:0;font-size:.85rem;">◈ FACIAL EMOTION PIPELINE (RAF-DB)</h4>', unsafe_allow_html=True)
        if fer_ready:
            m = st.session_state.pipeline_metrics
            st.markdown('<div class="b-ok">◈ FER MODEL ONLINE</div>', unsafe_allow_html=True)
            if m and "train_accuracy" in m:
                ca, cb2, cc = st.columns(3)
                ca.metric("Train", f"{m.get('train_accuracy','?')}%")
                cb2.metric("Val",  f"{m.get('val_accuracy','?')}%")
                cc.metric("Test",  f"{m.get('test_accuracy',m.get('val_accuracy','?'))}%")
        else:
            st.markdown('<div class="b-warn">▲ MODEL NOT TRAINED</div>', unsafe_allow_html=True)
        max_s     = st.slider("Max FER samples", 2000, 28000, 10000, 1000, key="setup_fer_max")
        force_fer = st.checkbox("Force retrain FER", key="setup_fer_force")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top:0;font-size:.85rem;">⬡ DATASET SELECTION MATRIX</h4>', unsafe_allow_html=True)
        st.markdown("""
<div style="font-size:.75rem;color:#b0cce8;line-height:2;font-family:Share Tech Mono,monospace;">
  <span style="color:#00ff88;">✓ CREMA-D</span> — 91 actors, 7,442 clips, multi-ethnic, 6 emotions<br>
  <span style="color:#00ff88;">✓ TESS</span>    — 2 speakers, 2,800 clips, studio-quality, 7 emotions<br>
  <span style="color:#ff3366;">✗ RAVDESS</span> — DROPPED: only 24 actors, theatrical (55% real-world acc.)<br>
  <span style="color:#ff3366;">✗ SAVEE</span>   — DROPPED: only 4 male actors, 480 clips — too small<br>
  <hr>
  <span style="color:#ffaa00;">▸ COMBINED: ~10,242 clips | 93 speakers | 108-dim features</span>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
<div style="background:rgba(0,255,136,.04);border:1px solid rgba(0,255,136,.2);
  border-radius:10px;padding:.8rem 1rem;margin:.8rem 0;">
  <span style="font-size:.65rem;font-weight:700;color:#00ff88;
    text-transform:uppercase;letter-spacing:.15em;font-family:Share Tech Mono,monospace;">
    ⬡ ENHANCED NERVOUSNESS DETECTION ENGINE v7.0
  </span><br>
  <span style="font-size:.73rem;color:#b0cce8;font-family:Rajdhani,sans-serif;">
    Unified corpus: CREMA-D (91 actors) + TESS — 10,242 samples, 93 speakers, 108-dim features, 5-fold CV.
  </span>
</div>""", unsafe_allow_html=True)

    nc1, nc2 = st.columns(2)
    with nc1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top:0;font-size:.85rem;">▸ UNIFIED VOICE MODEL (CREMA-D + TESS)</h4>', unsafe_allow_html=True)
        if nerv_ready:
            nm = st.session_state.nervousness_metrics
            st.markdown('<div class="b-ok">◈ VOICE MODEL ONLINE</div>', unsafe_allow_html=True)
            if nm and "test_accuracy" in nm:
                r1,r2,r3,r4 = st.columns(4)
                r1.metric("Train",  f"{nm.get('train_accuracy','?')}%")
                r2.metric("Val",    f"{nm.get('val_accuracy','?')}%")
                r3.metric("Test",   f"{nm.get('test_accuracy','?')}%")
                r4.metric("5-Fold", f"{nm.get('cv_mean_accuracy','?')}%")
                r1b, r2b = st.columns(2)
                r1b.metric("Nerv Acc", f"{nm.get('nervousness_binary_accuracy','?')}%")
                r2b.metric("Nerv F1",  f"{nm.get('nervousness_binary_f1','?')}")
                st.markdown(
                    f'<div style="font-size:.65rem;color:#c7e8ff;margin-top:.3rem;font-family:Share Tech Mono,monospace;">' +
                    f'SRC: {nm.get("source","—")} | {nm.get("n_total","?")} SAMPLES | {nm.get("feature_size","?")} FEATURES</div>',
                    unsafe_allow_html=True)
        else:
            st.markdown('<div class="b-warn">▲ NOT TRAINED — CLICK TRAIN BUTTON BELOW</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with nc2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top:0;font-size:.85rem;">⚙ TRAINING PARAMETERS</h4>', unsafe_allow_html=True)
        max_per_ds = st.slider("Max samples per dataset", 500, 2000, 1500, 100, key="setup_nerv_max")
        force_nerv = st.checkbox("Force retrain voice model", key="setup_nerv_force")
        st.markdown(
            '<div style="font-size:.7rem;color:#c7e8ff;line-height:1.8;margin-top:.5rem;font-family:Share Tech Mono,monospace;">' +
            'DOWNLOADS VIA KAGGLEHUB:<br>• CREMA-D (91 actors, 7,442 clips)<br>• TESS (2,800 studio clips)<br>' +
            '<span style="color:#ff3366;">(RAVDESS & SAVEE excluded)</span></div>',
            unsafe_allow_html=True)
        if st.button("▶  TRAIN VOICE MODEL (CREMA-D + TESS)", key="train_nerv_only"):
            _run_nervousness_training(max_per_ds, force_nerv)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    all_ready = fer_ready and nerv_ready
    if all_ready:
        st.markdown('<div class="b-ok">◈ ALL NEURAL PIPELINES ONLINE — FULL MULTIMODAL ANALYSIS ACTIVE</div>', unsafe_allow_html=True)
        st.button("▶  LAUNCH INTERVIEW", key="setup_go",
                  on_click=nav_to("Start Interview"))
        return

    if st.button("⬡  INITIALIZE ALL MODELS (FER + VOICE + LEGACY)", key="setup_all"):
        prog  = st.progress(0, "Initializing…")
        log_a = st.empty()
        logs  = []
        def cb(msg: str) -> None:
            logs.append(msg)
            log_a.markdown('<div class="txbox">' + "<br>".join(logs[-14:]) + "</div>", unsafe_allow_html=True)
            if   "FER pipeline" in msg:  prog.progress(35, "◈ FER ONLINE")
            elif "Training FER" in msg:  prog.progress(25, "▸ FER TRAINING…")
            elif "Extracting"   in msg:  prog.progress(15, "⬡ EXTRACTING FEATURES…")
            elif "All pipelines" in msg: prog.progress(50, "◈ LEGACY COMPLETE")
        results = engine.setup_all_pipelines(
            force_retrain=force_fer, max_fer_samples=max_s,
            max_ravdess_samples=0, progress_callback=cb)
        fer_m   = results.get("fer", {})
        voice_m = results.get("voice", {})
        if "error" not in fer_m:
            st.session_state.fer_ready        = True
            st.session_state.pipeline_metrics = fer_m
        if "error" not in voice_m:
            st.session_state.voice_ready   = True
            st.session_state.voice_metrics = voice_m
        _run_nervousness_training(max_per_ds, force_nerv, cb=cb)
        if st.session_state.fer_ready or st.session_state.nervousness_ready:
            st.success("◈ ALL MODELS ONLINE")
            st.balloons(); time.sleep(1)
            nav("Start Interview"); st.rerun()
        else:
            st.error("▲ SETUP FAILED — CHECK TERMINAL LOGS")


def _run_nervousness_training(max_per_ds: int = 1500, force: bool = False, cb=None) -> None:
    prog2  = st.progress(0, "INITIALIZING NERVOUSNESS TRAINING…")
    log_a2 = st.empty()
    logs2: list = []
    def nerv_cb(msg: str) -> None:
        logs2.append(msg)
        log_a2.markdown('<div class="txbox">' + "<br>".join(logs2[-14:]) + "</div>", unsafe_allow_html=True)
        if   "Downloading" in msg: prog2.progress(10, "⬇ DOWNLOADING DATASETS…")
        elif "Loading"     in msg: prog2.progress(30, "◈ LOADING AUDIO…")
        elif "Total"       in msg: prog2.progress(50, "▸ DATA READY…")
        elif "Training"    in msg: prog2.progress(60, "⬡ MLP TRAINING…")
        elif "5-fold"      in msg: prog2.progress(75, "⬡ CROSS-VALIDATING…")
        elif "binary"      in msg: prog2.progress(90, "◈ NERVOUSNESS EVAL…")
        elif "Done"        in msg: prog2.progress(100,"◈ COMPLETE")
        if cb: cb(msg)
    nm = nerv_pipeline.setup(force_retrain=force, max_per_dataset=max_per_ds, progress_cb=nerv_cb)
    if nm and nm.get("test_accuracy") is not None:
        st.session_state.nervousness_ready   = True
        st.session_state.nervousness_metrics = nm
        st.success(f"◈ VOICE MODEL ONLINE — Test={nm.get('test_accuracy','?')}%  "
                   f"CV={nm.get('cv_mean_accuracy','?')}%  "
                   f"NervAcc={nm.get('nervousness_binary_accuracy','?')}%")
    elif nm and nm.get("source") == "random_init":
        st.warning("▲ NO DATASETS AVAILABLE — RANDOM INIT. INSTALL KAGGLEHUB.")
    else:
        st.error("▲ VOICE TRAINING FAILED — CHECK TERMINAL LOGS")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: START INTERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def page_start() -> None:
    # ══════════════════════════════════════════════════════════════════════════
    #  SESSION PARAMETERS — pure Streamlit widgets, styled with CSS to match
    #  the cyberpunk design. No iframes, no hidden widgets, fully functional.
    # ══════════════════════════════════════════════════════════════════════════
    custom_qs = st.session_state.get("custom_resume_questions", [])
    roles     = engine.qbank.roles
    cur_role  = st.session_state.get("target_role", roles[0] if roles else "Software Engineer")
    cur_name  = (st.session_state.get("resume_parsed", {}).get("name", "")
                 or st.session_state.get("candidate_name", "Candidate"))
    cur_diff  = st.session_state.get("difficulty", "Medium")
    cur_nq    = st.session_state.get("num_questions", 5)
    cur_co    = st.session_state.get("target_company", "No specific company")
    has_custom = bool(custom_qs)
    btn_label  = "START RESUME INTERVIEW" if has_custom else "INITIALIZE SESSION"

    # ── Model init guard ─────────────────────────────────────────────────────
    if not st.session_state.fer_ready or not st.session_state.nervousness_ready:
        _init_key = "_auto_init_triggered"
        if not st.session_state.get(_init_key):
            st.session_state[_init_key] = True
            prog = st.progress(0, "Preparing environment…")
            def _cb(msg):
                if "FER"       in msg: prog.progress(30, "Calibrating facial analysis…")
                elif "Extract" in msg: prog.progress(15, "Processing datasets…")
                elif "All pip" in msg: prog.progress(50, "Almost ready…")
            res = engine.setup_all_pipelines(force_retrain=False,
                      max_fer_samples=10000, max_ravdess_samples=0,
                      progress_callback=_cb)
            if "error" not in res.get("fer",  {}):
                st.session_state.fer_ready        = True
                st.session_state.pipeline_metrics = res["fer"]
            if "error" not in res.get("voice", {}):
                st.session_state.voice_ready  = True
                st.session_state.voice_metrics = res["voice"]
            prog.progress(70, "Preparing voice analysis…")
            _run_nervousness_training(1500, False, cb=_cb)
            prog.progress(100, "✅ Ready!")
            time.sleep(0.6)
            st.rerun()
        else:
            st.info("Setting up your session…")
        return

    # ══════════════════════════════════════════════════════════════════════════
    #  RESUME UPLOAD POPUP  — shown once after models are ready, before the
    #  Initialize Session card.  Candidate can upload their resume for
    #  personalised questions, or skip to the generic session form.
    # ══════════════════════════════════════════════════════════════════════════
    if not st.session_state.get("_resume_popup_shown", False):
        # ── Popup CSS ────────────────────────────────────────────────────────
        st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500&display=swap');
[data-testid="stMainBlockContainer"] {
  background: transparent !important;
  display: flex !important; flex-direction: column !important;
  align-items: center !important; justify-content: center !important;
  min-height: 100vh !important; padding: 2rem 1rem !important;
}
@keyframes rp-card-in { from{opacity:0;transform:translateY(24px);} to{opacity:1;transform:translateY(0);} }
@keyframes rp-scan { 0%{top:-2px;opacity:0;} 5%{opacity:1;} 95%{opacity:1;} 100%{top:100vh;opacity:0;} }
body::after {
  content:''; position:fixed; left:0; right:0; top:0; height:2px;
  background:linear-gradient(90deg,transparent,rgba(0,229,255,.75),rgba(0,229,255,1),rgba(0,229,255,.75),transparent);
  animation:rp-scan 5s linear infinite; pointer-events:none; z-index:99999;
}
.rp-card {
  width:100%; max-width:580px; margin:0 auto;
  background:rgba(8,12,28,0.90); backdrop-filter:blur(28px);
  border:1px solid rgba(0,229,255,0.18); border-radius:24px;
  padding:2.6rem 2.4rem 2.2rem;
  box-shadow:0 0 0 1px rgba(0,0,0,1),0 40px 100px rgba(0,0,0,.88),
             inset 0 1px 0 rgba(0,229,255,.08);
  position:relative; animation:rp-card-in .55s ease both;
}
.rp-card::before {
  content:''; position:absolute; top:0; left:15%; right:15%; height:2px;
  background:linear-gradient(90deg,transparent,rgba(0,229,255,.65),transparent);
  border-radius:2px;
}
.rp-corner-tl,.rp-corner-br { position:absolute; width:22px; height:22px; pointer-events:none; }
.rp-corner-tl { top:-1px; left:-1px; border-top:2.5px solid #00d4ff; border-left:2.5px solid #00d4ff; border-radius:5px 0 0 0; }
.rp-corner-br { bottom:-1px; right:-1px; border-bottom:2.5px solid #00d4ff; border-right:2.5px solid #00d4ff; border-radius:0 0 5px 0; }
.rp-badge {
  display:block; width:fit-content; margin:0 auto 1.4rem;
  background:rgba(0,212,255,.08); border:1px solid rgba(0,212,255,.28);
  border-radius:20px; padding:4px 14px;
  font-family:'JetBrains Mono',monospace; font-size:10px;
  letter-spacing:.18em; color:#00d4ff;
}
.rp-title {
  font-family:'Orbitron',monospace; font-size:clamp(1.1rem,2.6vw,1.5rem);
  font-weight:800; letter-spacing:.12em; color:#f1f5f9;
  text-align:center; text-transform:uppercase; margin-bottom:.5rem;
}
.rp-subtitle {
  text-align:center; font-family:'Inter',sans-serif;
  font-size:.82rem; color:rgba(148,185,220,.65); margin-bottom:1.6rem;
}
.rp-divider { height:1px; background:linear-gradient(90deg,transparent,rgba(0,212,255,.2),transparent); margin:1.2rem 0; }
.rp-benefit-row { display:flex; gap:10px; align-items:flex-start; margin-bottom:.75rem; }
.rp-dot { width:6px; height:6px; border-radius:50%; background:#00d4ff; margin-top:5px; flex-shrink:0; box-shadow:0 0 6px rgba(0,212,255,.6); }
.rp-benefit-text { font-family:'Inter',sans-serif; font-size:.8rem; color:rgba(148,185,220,.85); line-height:1.5; }
.rp-benefit-text b { color:#f1f5f9; }
[data-testid="stFileUploader"] {
  border:1.5px dashed rgba(0,229,255,0.35) !important;
  border-radius:12px !important; background:rgba(0,30,60,0.25) !important; padding:1rem !important;
}
[data-testid="stFileUploader"]:hover { border-color:rgba(0,229,255,0.65) !important; }
</style>
""", unsafe_allow_html=True)

        # ── Static card HTML (no dynamic content inside — avoids re-render) ──
        st.markdown("""
<div class="rp-card">
  <div class="rp-corner-tl"></div>
  <div class="rp-corner-br"></div>
  <div class="rp-badge">&#9670; RESUME INTELLIGENCE</div>
  <div class="rp-title">Upload Your Resume</div>
  <div class="rp-subtitle">Get personalised interview questions tailored to your actual experience</div>
  <div class="rp-divider"></div>
  <div class="rp-benefit-row">
    <div class="rp-dot"></div>
    <div class="rp-benefit-text"><b>Skill-gap questions</b> — targets gaps between your resume and the job description</div>
  </div>
  <div class="rp-benefit-row">
    <div class="rp-dot"></div>
    <div class="rp-benefit-text"><b>Project-specific probing</b> — questions built from your own projects and experience</div>
  </div>
  <div class="rp-benefit-row">
    <div class="rp-dot"></div>
    <div class="rp-benefit-text"><b>ATS-optimised rephrasing</b> — rewrite your resume bullets before practising</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── File uploader (Streamlit widget — must be outside pure HTML) ─────
        # Import extract helpers from resume_rephraser directly (no wrapper needed)
        try:
            from resume_rephraser import (
                _extract_text_from_pdf  as _rr_pdf,
                _extract_text_from_docx as _rr_docx,
                PYPDF_OK  as _RR_PYPDF_OK,
                DOCX_OK   as _RR_DOCX_OK,
            )
            _rr_import_ok = True
        except ImportError:
            _rr_import_ok = False
            _RR_PYPDF_OK  = False
            _RR_DOCX_OK   = False

        _accepted = []
        if _RR_PYPDF_OK: _accepted.append("pdf")
        if _RR_DOCX_OK:  _accepted.append("docx")
        _accepted.append("txt")

        uploaded = st.file_uploader(
            "Upload resume (PDF, DOCX or TXT)",
            type=_accepted,
            key="rp_file_upload",
            label_visibility="collapsed",
            help="Processed locally — never stored on any server.",
        )

        # ── Process uploaded file immediately ─────────────────────────────────
        _raw_text = ""
        if uploaded is not None:
            _file_bytes = uploaded.read()
            _ext = uploaded.name.rsplit(".", 1)[-1].lower()

            if _rr_import_ok:
                if _ext == "pdf":
                    _raw_text = _rr_pdf(_file_bytes)
                elif _ext == "docx":
                    _raw_text = _rr_docx(_file_bytes)
                else:
                    _raw_text = _file_bytes.decode("utf-8", errors="ignore")
            else:
                # fallback if import failed
                _raw_text = _file_bytes.decode("utf-8", errors="ignore")

            if _raw_text:
                # Store ONLY resume_raw_text — do NOT touch resume_parsed here.
                # page_resume reads resume_raw_text as its pre-filled fallback
                # and runs the proper Groq parse pipeline from there.
                st.session_state["resume_raw_text"] = _raw_text
                st.markdown(
                    f'<div style="margin:.4rem 0 .6rem;padding:.6rem 1rem;'
                    f'background:rgba(0,255,136,.07);border:1px solid rgba(0,255,136,.25);'
                    f'border-radius:10px;font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:10px;letter-spacing:.12em;color:#00ff88;">'
                    f'&#10003;&nbsp;{uploaded.name} — ready ({len(_raw_text):,} chars)</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Could not extract text from this file. Try pasting the text in Resume Rephraser instead.")

        # Show if a resume was already loaded in a previous visit to this popup
        elif st.session_state.get("resume_raw_text"):
            st.markdown(
                '<div style="margin:.4rem 0 .6rem;padding:.5rem .9rem;'
                'background:rgba(0,212,255,.06);border:1px solid rgba(0,212,255,.22);'
                'border-radius:10px;font-family:\'JetBrains Mono\',monospace;'
                'font-size:10px;letter-spacing:.12em;color:#00d4ff;">'
                '&#9670;&nbsp;Resume already loaded — click Continue to proceed</div>',
                unsafe_allow_html=True,
            )

        # ── Action buttons ────────────────────────────────────────────────────
        _has_resume_text = bool(
            _raw_text or
            st.session_state.get("resume_raw_text") or
            st.session_state.get("resume_parsed") or
            st.session_state.get("custom_resume_questions")
        )

        ca, cb = st.columns([1, 1])
        with ca:
            _continue_label = "▶  Open Resume Rephraser" if _has_resume_text else "▶  Continue to Session"
            if st.button(_continue_label, key="rp_continue", use_container_width=True, type="primary"):
                st.session_state["_resume_popup_shown"] = True
                if _has_resume_text:
                    # Route to Resume Rephraser so the full parse + question
                    # generation pipeline runs properly there
                    st.session_state["_resume_popup_route_rephraser"] = True
                st.rerun()

        with cb:
            if st.button("Skip — use generic questions", key="rp_skip", use_container_width=True):
                st.session_state["_resume_popup_shown"] = True
                st.session_state["_resume_popup_route_rephraser"] = False
                st.rerun()

        return   # Don't render the Initialize Session form yet

    # ── After popup: route to Resume Rephraser if resume was provided ─────────
    if st.session_state.pop("_resume_popup_route_rephraser", False):
        st.session_state["page"] = "Resume Rephraser"
        st.rerun()
        return

    # ── Inject page-specific CSS to style Streamlit widgets like the overlay ──
    st.markdown("""
<style>
/* ── Full-page dark background for Start Interview ── */
[data-testid="stMainBlockContainer"] {
  background: transparent !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
  min-height: 100vh !important;
  padding: 2rem 1rem !important;
}

/* ── Animated scan line ── */
@keyframes si-scan {
  0%   { top: -2px; opacity: 0; }
  5%   { opacity: 1; }
  95%  { opacity: 1; }
  100% { top: 100vh; opacity: 0; }
}
body::after {
  content: '';
  position: fixed; left: 0; right: 0; top: 0; height: 2px;
  background: linear-gradient(90deg, transparent, rgba(0,229,255,.75), rgba(0,229,255,1), rgba(0,229,255,.75), transparent);
  box-shadow: 0 0 14px 4px rgba(0,200,255,.4);
  animation: si-scan 5s linear infinite;
  pointer-events: none; z-index: 99999;
}

/* ── Animated background grid ── */
body::before {
  content: '';
  position: fixed; inset: 0;
  background-image:
    linear-gradient(rgba(0,212,255,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,.03) 1px, transparent 1px);
  background-size: 44px 44px;
  animation: siGridMove 24s linear infinite;
  pointer-events: none; z-index: 0;
}
@keyframes siGridMove { from{background-position:0 0;} to{background-position:44px 44px;} }

/* ═══════════════════════════════════════════════
   SI-CARD — Animated glass panel matching reference
═══════════════════════════════════════════════ */
.si-card {
  width: 100%;
  max-width: 620px;
  margin: 0 auto;
  background: rgba(8, 12, 28, 0.82);
  backdrop-filter: blur(28px);
  -webkit-backdrop-filter: blur(28px);
  border: 1px solid rgba(0, 229, 255, 0.18);
  border-radius: 24px;
  padding: 2.8rem 2.6rem 2.4rem;
  box-shadow:
    0 0 0 1px rgba(0,0,0,1),
    0 40px 100px rgba(0,0,0,.88),
    0 0 80px rgba(0,80,220,.10),
    inset 0 1px 0 rgba(0,229,255,.08),
    inset 0 0 60px rgba(0,30,80,.22);
  position: relative;
  overflow: visible;
  animation: si-card-in .6s ease both;
}
@keyframes si-card-in {
  from { opacity:0; transform:translateY(20px); }
  to   { opacity:1; transform:translateY(0); }
}

/* Top glow bar */
.si-card::before {
  content: '';
  position: absolute;
  top: 0; left: 15%; right: 15%; height: 2px;
  background: linear-gradient(90deg, transparent, rgba(0,229,255,.65), transparent);
  border-radius: 2px;
}

/* Corner bracket TL */
.si-corner-tl, .si-corner-br {
  position: absolute;
  width: 24px; height: 24px;
  pointer-events: none;
}
.si-corner-tl {
  top: -1px; left: -1px;
  border-top: 2.5px solid #00d4ff;
  border-left: 2.5px solid #00d4ff;
  border-radius: 5px 0 0 0;
  animation: corner-pulse 3s ease-in-out infinite;
}
.si-corner-br {
  bottom: -1px; right: -1px;
  border-bottom: 2.5px solid #00d4ff;
  border-right: 2.5px solid #00d4ff;
  border-radius: 0 0 5px 0;
  animation: corner-pulse 3s ease-in-out infinite .5s;
}
@keyframes corner-pulse {
  0%,100% { border-color: #00d4ff; box-shadow: none; }
  50%     { border-color: #00e5ff; box-shadow: 0 0 12px rgba(0,229,255,.5); }
}

/* ── Hexagon icon at top-centre ── */
.si-hex-wrap {
  display: flex;
  justify-content: center;
  margin-bottom: 1.4rem;
}
.si-hex {
  width: 64px; height: 64px;
  border-radius: 14px;
  background: rgba(0,212,255,.07);
  border: 1.5px solid rgba(0,212,255,.45);
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 0 24px rgba(0,212,255,.18), inset 0 0 14px rgba(0,212,255,.06);
  clip-path: polygon(50% 0%, 93% 25%, 93% 75%, 50% 100%, 7% 75%, 7% 25%);
  animation: siHexPulse 3s ease-in-out infinite;
}
@keyframes siHexPulse {
  0%,100% { box-shadow: 0 0 24px rgba(0,212,255,.18), inset 0 0 14px rgba(0,212,255,.06); }
  50%     { box-shadow: 0 0 44px rgba(0,212,255,.38), inset 0 0 22px rgba(0,212,255,.12); }
}
.si-hex svg { width: 28px; height: 28px; }

/* ── Main title block ── */
.si-title-block { text-align: center; margin-bottom: 1.6rem; }
.si-main-title {
  font-family: 'Orbitron', monospace;
  font-size: clamp(1.3rem, 3vw, 1.75rem);
  font-weight: 800;
  letter-spacing: .14em;
  color: #f1f5f9;
  text-transform: uppercase;
  line-height: 1.15;
}
.si-main-sub {
  font-family: 'Inter', sans-serif;
  font-size: .82rem;
  color: rgba(148,185,220,.65);
  margin-top: .45rem;
  letter-spacing: .02em;
}

/* ── Field labels ── */
.si-lbl {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: .22em;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 7px;
  text-transform: uppercase;
}
.si-lbl-green  { color: #00ff88; text-shadow: 0 0 10px rgba(0,255,136,.45); }
.si-lbl-cyan   { color: #00d4ff; text-shadow: 0 0 10px rgba(0,212,255,.45); }
.si-lbl-teal   { color: #00e5cc; text-shadow: 0 0 10px rgba(0,229,204,.45); }
.si-lbl-violet { color: #c084fc; text-shadow: 0 0 10px rgba(192,132,252,.45); }
.si-lbl-amber  { color: #fbbf24; text-shadow: 0 0 10px rgba(251,191,36,.45); }
.si-dot {
  width: 5px; height: 5px; border-radius: 50%;
  background: #00ff88; box-shadow: 0 0 6px #00ff88;
  animation: siDotPulse 2s ease-in-out infinite;
  flex-shrink: 0; display: inline-block;
}
@keyframes siDotPulse {
  0%,100% { opacity:1; transform:scale(1); }
  50%     { opacity:.2; transform:scale(.5); }
}

/* ── Streamlit text-input overrides (inside si-card) ── */
.si-card [data-testid="stTextInputRootElement"] input,
.si-card [data-testid="stTextInput"] input {
  background: rgba(2,6,20,.95) !important;
  border: 1px solid rgba(0,212,255,.28) !important;
  border-radius: 12px !important;
  color: #e8f4ff !important;
  font-family: 'Inter', sans-serif !important;
  font-size: .92rem !important;
  padding: .72rem 1.1rem !important;
  transition: border-color .2s, box-shadow .2s !important;
}
.si-card [data-testid="stTextInput"] input:focus {
  border-color: rgba(0,212,255,.65) !important;
  box-shadow: 0 0 0 3px rgba(0,212,255,.1) !important;
}

/* ── Streamlit selectbox overrides ── */
.si-card [data-testid="stSelectbox"] > div > div {
  background: rgba(2,6,20,.95) !important;
  border: 1px solid rgba(0,212,255,.28) !important;
  border-radius: 12px !important;
  color: #e8f4ff !important;
}

/* ── Slider thumb overrides ── */
.si-card [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
  background: #bc13fe !important;
  box-shadow: 0 0 14px rgba(188,19,254,.7) !important;
}
.si-card [data-testid="stSelectSlider"] [data-baseweb="slider"] div[role="slider"] {
  background: #c084fc !important;
  box-shadow: 0 0 14px rgba(192,132,252,.7) !important;
}

/* ── Divider ── */
.si-divider { height: 1px; background: rgba(0,212,255,.09); margin: 1.1rem 0 1.25rem; }

/* ── Status chips row ── */
.si-chips {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 1.1rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(255,255,255,.06);
}
.si-chip {
  display: inline-flex; align-items: center; gap: 5px;
  background: rgba(0,255,136,.07);
  border: 1px solid rgba(0,255,136,.22);
  border-radius: 20px;
  padding: 4px 11px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 9.5px;
  letter-spacing: .1em;
  color: rgba(160,220,196,.85);
}
.si-chip-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: #00ff88; box-shadow: 0 0 6px #00ff88;
  animation: siDotPulse 2s ease-in-out infinite;
  display: inline-block; flex-shrink: 0;
}
.si-chip-cyan   { background: rgba(0,212,255,.07); border-color: rgba(0,212,255,.28); color: rgba(160,210,255,.85); }
.si-chip-cyan   .si-chip-dot { background: #00d4ff; box-shadow: 0 0 6px #00d4ff; }
.si-chip-violet { background: rgba(192,132,252,.07); border-color: rgba(192,132,252,.28); color: rgba(210,180,255,.85); }
.si-chip-violet .si-chip-dot { background: #c084fc; box-shadow: 0 0 6px #c084fc; }

/* ── Launch button ── */
.si-card [data-testid="stButton"] > button {
  width: 100% !important;
  background: linear-gradient(90deg, rgba(0,60,160,.85) 0%, rgba(80,20,180,.85) 100%) !important;
  border: 1.5px solid rgba(0,212,255,.5) !important;
  border-radius: 12px !important;
  color: #00d4ff !important;
  font-family: 'Orbitron', monospace !important;
  font-size: .82rem !important;
  font-weight: 700 !important;
  letter-spacing: .18em !important;
  text-transform: uppercase !important;
  padding: 1rem 1.5rem !important;
  box-shadow: 0 0 24px rgba(0,100,255,.25), inset 0 1px 0 rgba(255,255,255,.05) !important;
  transition: box-shadow .25s, border-color .25s, transform .2s !important;
  cursor: pointer !important;
  position: relative; overflow: hidden !important;
}
.si-card [data-testid="stButton"] > button:hover {
  box-shadow: 0 0 44px rgba(0,180,255,.5), inset 0 1px 0 rgba(255,255,255,.08) !important;
  border-color: rgba(0,229,255,.85) !important;
  transform: translateY(-2px) !important;
  color: #fff !important;
}
/* Shimmer sweep on launch button */
.si-card [data-testid="stButton"] > button::after {
  content: '';
  position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.18), transparent);
  transition: .55s;
}
.si-card [data-testid="stButton"] > button:hover::after { left: 100%; }
</style>
""", unsafe_allow_html=True)

    # ── Layout: centered card ─────────────────────────────────────────────────
    _, col, _ = st.columns([1, 2.5, 1])
    with col:
        st.markdown("""
<div class="si-card">
  <div class="si-corner-tl"></div>
  <div class="si-corner-br"></div>
""", unsafe_allow_html=True)

        # ── Hex icon + title ─────────────────────────────────────────────────
        st.markdown("""
<div class="si-hex-wrap">
  <div class="si-hex">
    <svg viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
      <polygon points="14,2 25,8 25,20 14,26 3,20 3,8"
               stroke="#00d4ff" stroke-width="1.6" fill="rgba(0,212,255,0.08)"/>
      <polygon points="14,7 20,10.5 20,17.5 14,21 8,17.5 8,10.5"
               stroke="#00d4ff" stroke-width="1" fill="rgba(0,212,255,0.05)" opacity="0.6"/>
    </svg>
  </div>
</div>
<div class="si-title-block">
  <div class="si-main-title">Initialize Session</div>
  <div class="si-main-sub">Configure your interview parameters</div>
</div>
""", unsafe_allow_html=True)

        # Candidate Name
        st.markdown('<div class="si-lbl si-lbl-green"><span class="si-dot"></span>&#42;&nbsp;CANDIDATE NAME</div>', unsafe_allow_html=True)
        name = st.text_input("name", value=cur_name, key="si_nm", label_visibility="collapsed")

        # Target Role
        st.markdown('<div class="si-lbl si-lbl-cyan" style="margin-top:.9rem;">[ TARGET ROLE ]</div>', unsafe_allow_html=True)
        role_idx = roles.index(cur_role) if cur_role in roles else 0
        role = st.selectbox("role", roles, index=role_idx, key="si_rl", label_visibility="collapsed")

        # Target Company
        st.markdown('<div class="si-lbl si-lbl-teal" style="margin-top:.9rem;">&#9670;&nbsp;TARGET COMPANY <span style="opacity:.5;font-size:9px;">(optional)</span></div>', unsafe_allow_html=True)
        company_opts = ["No specific company","Google","Meta","Amazon","Microsoft","Apple",
                        "Netflix","Uber","Airbnb","Stripe","Flipkart","Infosys","TCS","Wipro","Other"]
        co_idx = company_opts.index(cur_co) if cur_co in company_opts else 0
        company = st.selectbox("company", company_opts, index=co_idx, key="si_co", label_visibility="collapsed")

        # Difficulty
        diff_opts = ["Easy", "Medium", "Hard", "All (RL Adaptive)"]
        diff_cur  = cur_diff if cur_diff in diff_opts else "Medium"
        st.markdown('<div class="si-lbl si-lbl-violet" style="margin-top:.9rem;">DIFFICULTY LEVEL</div>', unsafe_allow_html=True)
        diff = st.select_slider("diff", diff_opts, value=diff_cur, key="si_df", label_visibility="collapsed")

        # Number of Questions
        default_nq = min(15, max(3, len(custom_qs))) if custom_qs else cur_nq
        st.markdown('<div class="si-lbl si-lbl-amber" style="margin-top:.9rem;">NUMBER OF QUESTIONS</div>', unsafe_allow_html=True)
        nq = st.slider("nq", 3, 15, default_nq, key="si_nq", label_visibility="collapsed")

        # ── Resume section ────────────────────────────────────────────────────
        _has_resume_qs   = bool(st.session_state.get("custom_resume_questions"))
        _has_resume_text = bool(st.session_state.get("resume_raw_text") or
                                st.session_state.get("resume_parsed"))
        _resume_tick     = " ✓" if (_has_resume_qs or _has_resume_text) else ""
        st.markdown(
            f'<div class="si-lbl si-lbl-teal" style="margin-top:.9rem;">'
            f'&#9670;&nbsp;RESUME{_resume_tick}</div>',
            unsafe_allow_html=True,
        )
        _rr_col1, _rr_col2 = st.columns([1, 1])
        with _rr_col1:
            if st.button(
                "📄  Resume Rephraser",
                key="si_resume_btn",
                use_container_width=True,
                help="Upload and rephrase your resume, then auto-generate interview questions from it",
            ):
                st.session_state["page"] = "Resume Rephraser"
                st.rerun()
        with _rr_col2:
            if _has_resume_qs:
                _qs_count = len(st.session_state["custom_resume_questions"])
                st.markdown(
                    f'<div style="padding:.45rem .7rem;background:rgba(0,255,136,.08);'
                    f'border:1px solid rgba(0,255,136,.28);border-radius:9px;'
                    f'font-family:\'JetBrains Mono\',monospace;font-size:10px;'
                    f'letter-spacing:.1em;color:#00ff88;text-align:center;">'
                    f'&#10003;&nbsp;{_qs_count} resume Qs ready</div>',
                    unsafe_allow_html=True,
                )
            elif _has_resume_text:
                st.markdown(
                    '<div style="padding:.45rem .7rem;background:rgba(0,212,255,.07);'
                    'border:1px solid rgba(0,212,255,.22);border-radius:9px;'
                    'font-family:\'JetBrains Mono\',monospace;font-size:10px;'
                    'letter-spacing:.1em;color:#00d4ff;text-align:center;">'
                    '&#9670;&nbsp;Resume loaded</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="padding:.45rem .7rem;background:rgba(90,112,152,.06);'
                    'border:1px dashed rgba(90,112,152,.3);border-radius:9px;'
                    'font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                    'letter-spacing:.1em;color:rgba(148,185,220,.4);text-align:center;">'
                    'No resume loaded</div>',
                    unsafe_allow_html=True,
                )

        # Divider + Toggles
        st.markdown('<div class="si-divider"></div>', unsafe_allow_html=True)
        tc1, tc2 = st.columns(2)
        with tc1:
            webcam_on = st.toggle("Enable Webcam", value=True, key="si_cam")
        with tc2:
            st.toggle("Question Timer", value=True, key="si_tmr")

        # Blind mode is OFF for regular interviews — placement test enables it automatically
        st.session_state["blind_mode"] = False


        # Submit button
        st.markdown('<div style="margin-top:1.2rem;">', unsafe_allow_html=True)

        def start_session():
            _resume_qs  = st.session_state.get("custom_resume_questions", [])
            _use_resume = bool(_resume_qs)
            _nq = st.session_state.get("si_nq", nq)
            _role = st.session_state.get("si_rl", role)
            _diff = st.session_state.get("si_df", diff)
            _name = st.session_state.get("si_nm", name)
            _company = st.session_state.get("si_co", company)
            _cam = st.session_state.get("si_cam", True)
            # Sync into JD engine state keys so jd_question_engine picks it up
            if JD_ENGINE_OK:
                st.session_state["selected_company"]     = _company
                st.session_state["jd_selected_company"]  = _company
            st.session_state.update({
                "candidate_name":    _name,
                "target_role":       _role,
                "target_company":    _company,
                "difficulty":        _diff,
                "num_questions":     _nq,
                "webcam_enabled":    _cam,
                "transcript": "", "transcribed_text": "",
                "last_audio_id": None, "last_audio_bytes": None, "last_audio_source": None,
                "q_number": 0, "last_score": None, "last_feedback": "",
                "last_star": {}, "last_keywords": [], "last_eval": {},
                "session_answers": [], "submitted": False,
                "live_emotion": "Neutral", "live_nervousness": 0.2,
                "live_emotion_dist": {}, "emotion_history": [],
                "live_voice_emotion": "Neutral", "live_voice_nerv": 0.2,
                "live_posture": {}, "live_confidence": 3.5,
                "resume_mode_active": _use_resume, "resume_q_index": 0,
                "_pending_answer": "",
                "page": "Live Interview",
                "calibration_done": False, "calibration_baseline": 0.2,
                "calibration_skipped": False,
                # Blind scoring — reset per session
                "blind_scores":   {},
                "blind_revealed": False,
            })
            _reset_frame_state()
            if _use_resume:
                engine.questions = [
                    {"question": q.get("question",""), "type": q.get("type","Technical"),
                     "difficulty": q.get("difficulty","Medium").lower(),
                     "keywords": q.get("ideal_keywords", q.get("keywords",[])),
                     "ideal_answer": q.get("ideal_answer","")}
                    for q in _resume_qs[:_nq]
                ]
                engine.current_index = 0
                st.session_state["_resume_q_dicts"] = _resume_qs[:_nq]
            else:
                _diff_norm = "all" if "all" in _diff.lower() else _diff.lower()
                engine.start_session(role=_role, difficulty=_diff_norm, num_questions=_nq)
                engine._resume_parsed = st.session_state.get("resume_parsed", {})
                st.session_state["_resume_q_dicts"] = []
            st.session_state.question     = engine.get_next_question()
            st.session_state.q_number     = 1
            st.session_state.q_start_time = time.time()

        st.button(f"▶  LAUNCH INTERVIEW SEQUENCE" if not has_custom else f"▶  {btn_label}", key="si_go", on_click=start_session,
                  use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Status chips — Image-2 style (Neural Engine Ready | Voice Pipeline Active | Camera Access Granted)
        fer_ok  = st.session_state.fer_ready
        nerv_ok = st.session_state.nervousness_ready
        cam_ok  = st.session_state.get("webcam_enabled", True)
        st.markdown(f"""
<div class="si-chips">
  <span class="si-chip">
    <span class="si-chip-dot"></span>Neural Engine Ready
  </span>
  <span class="si-chip si-chip-cyan">
    <span class="si-chip-dot"></span>Voice Pipeline Active
  </span>
  <span class="si-chip si-chip-violet">
    <span class="si-chip-dot"></span>Camera Access Granted
  </span>
</div>
""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # close .si-card



# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: LIVE INTERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def _get_current_q_dict(qn: int) -> dict:
    """
    Return the current question dict — from resume Q dicts when in resume mode,
    otherwise from the engine as normal.
    """
    resume_mode = st.session_state.get("resume_mode_active", False)
    if resume_mode:
        dicts = st.session_state.get("_resume_q_dicts", [])
        idx   = qn - 1   # qn is 1-based
        if 0 <= idx < len(dicts):
            d = dicts[idx]
            # Normalise keys to match what the engine returns
            return {
                "question":     d.get("question", ""),
                "type":         d.get("type", "Technical"),
                "difficulty":   d.get("difficulty", "Medium"),
                "keywords":     d.get("ideal_keywords", d.get("keywords", [])),
                "ideal_answer": d.get("ideal_answer", ""),
                "target":       d.get("target", ""),
                "source":       "resume",
            }
    return engine.get_current_question_dict() or {}


def page_live() -> None:
    if not st.session_state.question:
        if not engine.questions:
            engine.start_session(st.session_state.target_role,
                                  st.session_state.difficulty.lower(),
                                  st.session_state.num_questions)
        st.session_state.question     = engine.get_next_question()
        st.session_state.q_number     = 1
        st.session_state.q_start_time = time.time()

    total = len(engine.questions)
    qn    = st.session_state.q_number
    prog  = min(1.0, max(0.0, (qn - 1) / max(total, 1)))

    # v8.1: baseline calibration step — only shown before Q1, no-op after
    if qn == 1:
        render_calibration_widget()
        if not (st.session_state.get("calibration_done") or
                st.session_state.get("calibration_skipped")):
            st.stop()   # pause rendering until calibration is resolved

    # ── Resume mode indicator banner ─────────────────────────────────────────
    resume_mode = st.session_state.get("resume_mode_active", False)
    if resume_mode:
        q_dicts = st.session_state.get("_resume_q_dicts", [])
        cur_dict = _get_current_q_dict(qn)
        target_lbl = cur_dict.get("target", "")
        st.markdown(f"""
<div style="background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.2);
  border-radius:10px;padding:.5rem 1rem;margin-bottom:.75rem;
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.4rem;">
  <div style="font-size:.78rem;font-weight:500;color:#a5b4fc;font-family:Inter,sans-serif;">
    ✦ Resume mode — {len(q_dicts)} personalised questions
  </div>
  {f'<div style="font-size:.72rem;color:#b4cde4;font-family:Inter,sans-serif;">{target_lbl}</div>' if target_lbl else ''}
</div>""", unsafe_allow_html=True)

    # ── SCANNING LINE OVERLAY ─────────────────────────────────────────────────
    # A neon-blue horizontal scan line sweeps the full viewport continuously,
    # giving the live interview screen a real-time HUD / sci-fi monitor feel.
    st.markdown("""
<style>
@keyframes scanline-sweep {
    0%   { top: -4px; opacity: 0; }
    5%   { opacity: 1; }
    95%  { opacity: 1; }
    100% { top: 100vh; opacity: 0; }
}
@keyframes scanline-glow-pulse {
    0%, 100% { opacity: .85; }
    50%       { opacity: 1;   }
}
#aura-scanline {
    position: fixed;
    left: 0; right: 0;
    top: -4px;
    height: 3px;
    z-index: 99999;
    pointer-events: none;
    background: linear-gradient(
        90deg,
        transparent 0%,
        rgba(0,180,255,.18) 8%,
        rgba(0,210,255,.72) 30%,
        rgba(0,229,255,1)   50%,
        rgba(0,210,255,.72) 70%,
        rgba(0,180,255,.18) 92%,
        transparent 100%
    );
    box-shadow:
        0 0 6px  2px rgba(0,229,255,.55),
        0 0 18px 4px rgba(0,180,255,.30),
        0 0 40px 8px rgba(0,140,255,.15);
    border-radius: 2px;
    animation:
        scanline-sweep       5s linear infinite,
        scanline-glow-pulse  2s ease-in-out infinite;
}
/* Faint horizontal CRT scanline texture across full screen */
#aura-scanline-bg {
    position: fixed;
    inset: 0;
    z-index: 99998;
    pointer-events: none;
    background: repeating-linear-gradient(
        to bottom,
        transparent,
        transparent 3px,
        rgba(0,180,255,.018) 3px,
        rgba(0,180,255,.018) 4px
    );
}
</style>
<div id="aura-scanline-bg"></div>
<div id="aura-scanline"></div>
""", unsafe_allow_html=True)

    # ── REDESIGNED INTERVIEW SCREEN — AURA v3 ────────────────────────────────
    _diff_color_map = {"easy": "#00ff88", "medium": "#f59e0b", "hard": "#f43f5e"}
    _diff_c = _diff_color_map.get(st.session_state.difficulty.lower(), "#00e5ff")
    _prog_pct = int(prog * 100)
    _role_lbl = st.session_state.get("target_role", "Software Engineer")

    # ── REDESIGNED HUD TOPBAR ─────────────────────────────────────────────────
    _elapsed_s = time.time() - (st.session_state.q_start_time or time.time())
    _lim_s = {"easy": 120, "medium": 180, "hard": 240, "all": 180}.get(
        st.session_state.difficulty.lower().split()[0], 180)
    _arc_frac = min(1.0, _elapsed_s / max(_lim_s, 1))
    _arc_colour = ("#00ff88" if _arc_frac < 0.55 else "#f59e0b" if _arc_frac < 0.80 else "#f43f5e")
    _R = 18; _SW = 3; _CX = _R + _SW; _CIR = 2 * 3.14159265 * _R

    # Build progress pip dots HTML
    _pip_dots = ""
    for _pi in range(total):
        if _pi < qn - 1:
            _pip_dots += f'<div style="width:28px;height:3px;border-radius:2px;background:#00ff88;box-shadow:0 0 5px #00ff8866;"></div>'
        elif _pi == qn - 1:
            _pip_dots += f'<div style="width:28px;height:3px;border-radius:2px;background:{_diff_c};box-shadow:0 0 8px {_diff_c}88;animation:pipPulse 2s ease-in-out infinite;"></div>'
        else:
            _pip_dots += '<div style="width:28px;height:3px;border-radius:2px;background:rgba(0,229,255,.12);"></div>'

    if False:  # HUD topbar hidden
     components.html(f"""
<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:transparent;font-family:'Inter',sans-serif;}}
@keyframes pipPulse{{0%,100%{{opacity:1}}50%{{opacity:.5}}}}
@keyframes ldot{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.4;transform:scale(.7)}}}}
.hud{{
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 18px;
  background:rgba(4,9,26,.92);
  border:1px solid rgba(0,229,255,.12);
  border-radius:14px;
  backdrop-filter:blur(16px);
  position:relative;overflow:hidden;
}}
.hud::after{{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,{_diff_c},{_diff_c}88,rgba(0,229,255,.6),transparent);
}}
.hud-logo{{font-family:'Orbitron',monospace;font-size:11px;font-weight:700;color:#00e5ff;letter-spacing:3px;display:flex;align-items:center;gap:8px;}}
.live-dot{{width:7px;height:7px;border-radius:50%;background:#00ff88;box-shadow:0 0 8px #00ff88;animation:ldot 1.4s ease-in-out infinite;flex-shrink:0;}}
.hud-mid{{display:flex;align-items:center;gap:6px;}}
.hud-right{{display:flex;align-items:center;gap:10px;}}
.badge{{padding:4px 12px;border-radius:20px;font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:500;letter-spacing:1.5px;}}
.badge-role{{background:rgba(0,229,255,.07);color:#00e5ff;border:1px solid rgba(0,229,255,.18);}}
.badge-diff{{background:{_diff_c}18;color:{_diff_c};border:1px solid {_diff_c}44;}}
.timer-num{{font-family:'JetBrains Mono',monospace;font-size:11px;color:#f43f5e;letter-spacing:2px;}}
.pip-lbl{{font-family:'JetBrains Mono',monospace;font-size:10px;color:rgba(0,229,255,.5);margin-left:6px;letter-spacing:1px;}}
</style>
</head><body>
<div class="hud">
  <div class="hud-logo">
    <div class="live-dot"></div>
    AURA INTERVIEW
  </div>
  <div class="hud-mid">
    {_pip_dots}
    <span class="pip-lbl">Q {qn} / {total}</span>
  </div>
  <div class="hud-right">
    <div class="badge badge-role">{_role_lbl[:22]}</div>
    <div class="badge badge-diff">⬡ {st.session_state.difficulty.upper()}</div>
    <div class="timer-num" id="hudT">--:--</div>
  </div>
</div>
<script>
(function(){{
  var lim = {_lim_s * 1000};
  var el  = document.getElementById('hudT');
  var elapsed = 0;       // ms already consumed (accumulated before pauses)
  var lastTick = null;   // timestamp of last animation frame when running

  function fmt(rem) {{
    rem = Math.max(0, rem);
    var m = Math.floor(rem / 60000), sec = Math.floor((rem % 60000) / 1000);
    return (m < 10 ? '0' : '') + m + ':' + (sec < 10 ? '0' : '') + sec;
  }}

  function tick(ts) {{
    var p = window.parent;
    var running = !!p._auraTimerRunning;
    var paused  = !!p._auraTimerPaused;

    if (running && !paused) {{
      // Accumulate elapsed only while actively running & not paused
      if (lastTick !== null) elapsed += (ts - lastTick);
      lastTick = ts;
      el.textContent = fmt(lim - elapsed);
    }} else {{
      // Not yet started or paused — show '--:--' until first start
      if (!running) {{
        el.textContent = '--:--';
      }} else {{
        // paused — freeze display
      }}
      lastTick = null;   // reset so we don't double-count on resume
    }}

    if (elapsed < lim || !running) {{
      requestAnimationFrame(tick);
    }}
  }}

  requestAnimationFrame(tick);
}})();
</script>
</body></html>""", height=56)

    center, right = st.columns([3, 1])

    # ── RIGHT PANEL — REDESIGNED AVATAR CARD + WEBCAM (backend-only)
    with right:
        # ── v10.0 / v3.0: Redesigned AI Avatar Interviewer card ──────────────
        if AVATAR_OK and st.session_state.get("avatar_enabled", True):
            _q_for_avatar   = st.session_state.question or ""
            _auto_speak     = st.session_state.get("avatar_auto_speak", True)
            _last_spoken_qn = st.session_state.get("_avatar_last_spoken_qn", -1)
            _should_auto    = _auto_speak and (qn != _last_spoken_qn)
            if _should_auto:
                st.session_state["_avatar_last_spoken_qn"] = qn
            _qtype_av = st.session_state.get("_pending_qtype", "Technical") or "Technical"
            # ── Hologram flicker overlay injected ABOVE the avatar iframe ─────
            components.html("""<!DOCTYPE html><html><head>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:transparent;overflow:hidden;}
#holo-shell{
  position:relative;width:100%;
  border-radius:14px;
  border:1px solid rgba(0,229,255,.18);
  overflow:hidden;
  pointer-events:none;
}
#holo-scan{
  position:absolute;left:0;right:0;height:3px;
  background:linear-gradient(90deg,transparent,rgba(0,229,255,.55),rgba(0,229,255,.8),rgba(0,229,255,.55),transparent);
  pointer-events:none;z-index:10;
  animation:scanMove 4s ease-in-out infinite;
}
@keyframes scanMove{0%{top:-3px;opacity:0}5%{opacity:1}90%{opacity:.7}100%{top:100%;opacity:0}}
#holo-lines{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:9;}
.hl{position:absolute;left:0;right:0;height:1px;background:rgba(0,229,255,.055);}
#holo-vignette{
  position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:8;
  background:radial-gradient(ellipse at center,transparent 55%,rgba(0,12,30,.55) 100%);
  animation:holoFlick 7s ease-in-out infinite;
}
@keyframes holoFlick{
  0%,100%{opacity:1}
  91%{opacity:1}92%{opacity:.15}93%{opacity:.9}94.5%{opacity:.05}96%{opacity:1}
  97%{opacity:.7}97.5%{opacity:1}
}
#holo-corner-tl,#holo-corner-tr,#holo-corner-bl,#holo-corner-br{
  position:absolute;width:16px;height:16px;pointer-events:none;z-index:11;
}
#holo-corner-tl{top:6px;left:6px;border-top:1.5px solid rgba(0,229,255,.7);border-left:1.5px solid rgba(0,229,255,.7);}
#holo-corner-tr{top:6px;right:6px;border-top:1.5px solid rgba(0,229,255,.7);border-right:1.5px solid rgba(0,229,255,.7);}
#holo-corner-bl{bottom:6px;left:6px;border-bottom:1.5px solid rgba(0,229,255,.7);border-left:1.5px solid rgba(0,229,255,.7);}
#holo-corner-br{bottom:6px;right:6px;border-bottom:1.5px solid rgba(0,229,255,.7);border-right:1.5px solid rgba(0,229,255,.7);}
#holo-status{
  position:absolute;bottom:10px;left:50%;transform:translateX(-50%);
  font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2px;
  color:rgba(0,229,255,.55);pointer-events:none;z-index:12;
  animation:statusPulse 2.5s ease-in-out infinite;white-space:nowrap;
}
@keyframes statusPulse{0%,100%{opacity:.4}50%{opacity:.85}}
</style>
</head><body>
<div id="holo-shell" style="height:4px;">
  <div id="holo-scan"></div>
  <div id="holo-lines" id="holo-lines"></div>
  <div id="holo-vignette"></div>
  <div id="holo-corner-tl"></div>
  <div id="holo-corner-tr"></div>
  <div id="holo-corner-bl"></div>
  <div id="holo-corner-br"></div>
  <div id="holo-status">◈ AURA · HOLOGRAPHIC PROJECTION</div>
</div>
<script>
(function(){
  var lines=document.getElementById('holo-lines');
  for(var i=0;i<32;i++){
    var d=document.createElement('div');d.className='hl';
    d.style.top=(i*3.2)+'%';lines.appendChild(d);
  }
})();
</script>
</body></html>""", height=6)
            render_avatar_interviewer(
                question_text = _q_for_avatar,
                question_type = _qtype_av,
                q_number      = qn,
                total_qs      = total,
                auto_speak    = _should_auto,
                height        = 480,
            )

        # ── Webcam runs BACKEND-ONLY — no camera UI shown to the user ──────────
        # Emotion / nervousness / posture signals are captured silently via the
        # WebRTC background thread or the auto-rerun loop. Results are read from
        # _frame_state and written to session_state so the sidebar live signals
        # and score aggregator still work — camera permission popup is suppressed.
        if st.session_state.get("webcam_enabled", True):
            # ── BACKEND-ONLY webcam processing — no video UI shown to user ───
            # Camera runs silently: WebRTC thread or auto-rerun loop processes
            # frames and writes emotion/nervousness/posture to session_state.
            # The avatar is shown in the right column instead of a webcam feed.
            if WEBRTC_OK and st.session_state.fer_ready:
                # Run the WebRTC streamer hidden (display:none via css).
                # The AuraVideoProcessor.recv() thread still processes frames.
                components.html("""
<div style="display:none;">
<script>
// WebRTC runs in background — video element hidden from user
</script>
</div>""", height=0)
                # Pull latest result from the WebRTC background thread
                _, result, frame_count = _read_frame_state()
                if result:
                    dom     = result.get("dominant", "Neutral")
                    nerv    = result.get("smoothed_nervousness",
                                         result.get("nervousness", 0.2))
                    posture = result.get("posture", {})
                    st.session_state.live_emotion     = dom
                    st.session_state.live_nervousness = nerv
                    prev_count = st.session_state.get("_last_frame_count", 0)
                    if frame_count > prev_count:
                        hist = st.session_state.emotion_history
                        hist.append(dom)
                        if len(hist) > 80:
                            hist.pop(0)
                        st.session_state.emotion_history = hist
                        st.session_state["_last_frame_count"] = frame_count
                    if posture:
                        st.session_state.live_posture = posture
                    mm = engine.get_multimodal_confidence()
                    st.session_state.live_confidence = mm.get("confidence_score", 3.5)

            elif not WEBRTC_OK and st.session_state.fer_ready:
                # Auto-rerun loop: captures frames silently, no camera widget shown
                result = _run_camera_loop(qn)
                if result:
                    dom     = result.get("dominant", "Neutral")
                    nerv    = result.get("smoothed_nervousness",
                                         result.get("nervousness", 0.2))
                    posture = result.get("posture", {})
                    st.session_state.live_emotion     = dom
                    st.session_state.live_nervousness = nerv
                    if posture:
                        st.session_state.live_posture = posture
                    mm = engine.get_multimodal_confidence()
                    st.session_state.live_confidence = mm.get("confidence_score", 3.5)

        if st.session_state.nervousness_ready:
            vr = engine.get_live_voice_result()
            ve = vr.get("dominant", "Neutral")
            vn = vr.get("nervousness", 0.2)
            ec = emo_css(ve); nc = nerv_css(vn)
            st.session_state.live_voice_emotion = ve
            st.session_state.live_voice_nerv    = vn
            _nerv_pct = int(vn * 100)
            _nerv_lbl = "LOW" if vn < 0.35 else "MODERATE" if vn < 0.65 else "HIGH"
            _nerv_col = "#00FFD1" if vn < 0.35 else "#FFD700" if vn < 0.65 else "#FF3366"
            _bar_grad = f"linear-gradient(90deg,#00D4FF,{_nerv_col})"

            # Show voice nervousness only AFTER answer is submitted
            if st.session_state.get("submitted", False):
                st.markdown(f"""
<div style="background:rgba(0,12,30,0.55);border:1px solid rgba(0,240,255,0.15);
  backdrop-filter:blur(12px);border-radius:12px;padding:.85rem 1rem;margin:.3rem 0;">
  <div style="font-size:.65rem;font-weight:700;color:#00F0FF;font-family:Orbitron,monospace;
    letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">Voice Nervousness</div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
    <span style="font-size:.82rem;font-weight:500;color:#e2e8f0;font-family:Inter,sans-serif;">
      {_nerv_lbl}</span>
    <span style="font-size:1rem;font-weight:700;color:{_nerv_col};font-family:Orbitron,monospace;">
      {_nerv_pct}%</span>
  </div>
  <div style="background:rgba(255,255,255,.06);border-radius:3px;height:6px;overflow:hidden;margin-bottom:8px;">
    <div style="width:{_nerv_pct}%;height:100%;background:{_bar_grad};border-radius:3px;
      box-shadow:0 0 8px {_nerv_col}66;transition:width .5s;"></div>
  </div>
  <div style="font-size:.72rem;color:#b4cde4;font-family:Inter,sans-serif;">
    <span style="color:{ec};font-weight:700;">{ve}</span>
    <span style="font-size:.62rem;color:rgba(148,163,184,0.6);margin-left:6px;">Voice emotion</span>
  </div>
</div>""", unsafe_allow_html=True)

                # ── EMOTION PULSE RING ──────────────────────────────────────────
                # SVG rings that pulse color + speed based on dominant voice emotion
                # and nervousness level. Calm/Happy → slow cyan. Fear/Angry/High → fast red strobe.
                # Nervous thresholds mirror nerv_css() logic.
                _ring_color   = emo_css(ve)          # driven by voice emotion
                _ring_color2  = _nerv_col             # driven by nervousness level

                # Pulse speed: low nerv = slow (2.2s), moderate = 1.2s, high = 0.55s
                _pulse_dur    = 2.2 if vn < 0.35 else (1.2 if vn < 0.65 else 0.55)
                # Ring opacity peak: low nerv = soft, high = harsh strobe
                _ring_opacity = 0.45 if vn < 0.35 else (0.65 if vn < 0.65 else 0.95)
                # Scale factor on pulse — nervous = bigger expansion
                _ring_scale   = 1.18 if vn < 0.35 else (1.28 if vn < 0.65 else 1.45)

                components.html(f"""
<div style="display:flex;align-items:center;justify-content:center;
  padding:.5rem 0 .2rem;margin-top:.2rem;">
  <svg width="110" height="110" viewBox="0 0 110 110"
       xmlns="http://www.w3.org/2000/svg" style="overflow:visible;">
    <defs>
      <style>
        /* Ring 1 — primary emotion color, drives the main pulse */
        @keyframes pulse1 {{
          0%   {{ opacity:{_ring_opacity}; transform:scale(1); }}
          50%  {{ opacity:0.05; transform:scale({_ring_scale}); }}
          100% {{ opacity:{_ring_opacity}; transform:scale(1); }}
        }}
        /* Ring 2 — nervousness color, offset half a cycle */
        @keyframes pulse2 {{
          0%   {{ opacity:{_ring_opacity * 0.6:.2f}; transform:scale(1); }}
          50%  {{ opacity:0.02; transform:scale({_ring_scale * 1.1:.2f}); }}
          100% {{ opacity:{_ring_opacity * 0.6:.2f}; transform:scale(1); }}
        }}
        /* Ring 3 — outermost, very faint, slowest */
        @keyframes pulse3 {{
          0%   {{ opacity:{_ring_opacity * 0.35:.2f}; transform:scale(1); }}
          50%  {{ opacity:0.0; transform:scale({_ring_scale * 1.22:.2f}); }}
          100% {{ opacity:{_ring_opacity * 0.35:.2f}; transform:scale(1); }}
        }}
        /* Center icon subtle breathe */
        @keyframes centerBreath {{
          0%,100% {{ opacity:0.9; }}
          50%      {{ opacity:0.5; }}
        }}
        .r1 {{
          transform-origin: 55px 55px;
          animation: pulse1 {_pulse_dur:.2f}s ease-in-out infinite;
        }}
        .r2 {{
          transform-origin: 55px 55px;
          animation: pulse2 {_pulse_dur:.2f}s ease-in-out infinite;
          animation-delay: {_pulse_dur / 2:.2f}s;
        }}
        .r3 {{
          transform-origin: 55px 55px;
          animation: pulse3 {_pulse_dur * 1.4:.2f}s ease-in-out infinite;
          animation-delay: {_pulse_dur * 0.25:.2f}s;
        }}
        .cicon {{
          animation: centerBreath {_pulse_dur * 1.2:.2f}s ease-in-out infinite;
        }}
      </style>
    </defs>

    <!-- Outer ring 3 (faintest, largest) -->
    <circle class="r3" cx="55" cy="55" r="48"
      fill="none" stroke="{_ring_color2}" stroke-width="1.2"
      stroke-dasharray="6 4"/>

    <!-- Mid ring 2 (nervousness color) -->
    <circle class="r2" cx="55" cy="55" r="40"
      fill="none" stroke="{_ring_color2}" stroke-width="1.8"/>

    <!-- Inner ring 1 (emotion color — strongest) -->
    <circle class="r1" cx="55" cy="55" r="32"
      fill="none" stroke="{_ring_color}" stroke-width="2.5"/>

    <!-- Static base ring — always visible, no animation -->
    <circle cx="55" cy="55" r="24"
      fill="rgba(0,8,22,0.75)" stroke="{_ring_color}" stroke-width="1"
      stroke-opacity="0.4"/>

    <!-- Center icon: mic/voice waveform -->
    <g class="cicon" transform="translate(55,55)">
      <!-- Mic body -->
      <rect x="-5" y="-12" width="10" height="14" rx="5"
        fill="none" stroke="{_ring_color}" stroke-width="1.5"/>
      <!-- Mic stand arc -->
      <path d="M-9 0 Q-9 10 0 10 Q9 10 9 0"
        fill="none" stroke="{_ring_color}" stroke-width="1.5" stroke-linecap="round"/>
      <!-- Stand pole -->
      <line x1="0" y1="10" x2="0" y2="14"
        stroke="{_ring_color}" stroke-width="1.5" stroke-linecap="round"/>
      <!-- Base bar -->
      <line x1="-5" y1="14" x2="5" y2="14"
        stroke="{_ring_color}" stroke-width="1.5" stroke-linecap="round"/>
    </g>

    <!-- Emotion label arc text (static, small) -->
    <text x="55" y="100" text-anchor="middle"
      font-family="Share Tech Mono, monospace" font-size="7.5"
      fill="{_ring_color}" opacity="0.7" letter-spacing="1">
      {ve.upper()} · {_nerv_lbl}
    </text>
  </svg>
</div>""", height=125)

        # Voice modulation waveform removed per UI update

    # ── LEFT/CENTER PANEL — QUESTION + ANSWER + BUTTONS
    with center:
        # Use resume dict when in resume mode, engine dict otherwise
        q_dict = _get_current_q_dict(qn)
        qtype  = q_dict.get("type", "")
        kws    = q_dict.get("keywords", q_dict.get("ideal_keywords", []))

        # ── Feature 12: inject role-accent CSS var for qcard border ──────────
        _live_accent, _live_glow, _ = get_role_theme(
            st.session_state.get("target_role", "Software Engineer") or "Software Engineer"
        )
        st.markdown(
            f'<style>:root{{--qcard-accent:{_live_accent};}}',
            unsafe_allow_html=True,
        )
        qt_col = {"Technical":"#a5b4fc","Behavioural":"#34d399","HR":"#fbbf24",
                  "Project-Based":"#f0abfc","System Design":"#fbbf24",
                  "Situational":"#fb923c"}.get(qtype, "#94a3b8")
        dom    = st.session_state.live_emotion
        ec     = emo_css(dom)
        ve_c   = emo_css(st.session_state.live_voice_emotion)

        # ── Resume question target tag ────────────────────────────────────────
        target = q_dict.get("target", "")
        target_html = ""
        if resume_mode and target:
            target_html = (f'<div style="font-size:.75rem;color:#c084fc;'
                           f'font-family:Inter,sans-serif;margin-bottom:.5rem;font-weight:500;">'
                           f'From resume: {target}</div>')

        # Typewriter variables
        _q_text  = st.session_state.question or ""
        _q_chars = max(len(_q_text), 1)
        _diff_str = (q_dict.get("difficulty") or
                     st.session_state.get("difficulty", "Medium") or "Medium")
        _diff_col = ("#FF3366" if "hard" in _diff_str.lower()
                     else "#FFD700" if "medium" in _diff_str.lower()
                     else "#00ff88")
        _fu_tag = (' <span style="font-size:.6rem;color:#f0abfc;font-family:Share Tech Mono,'
                   'monospace;margin-left:4px;">FOLLOW-UP</span>'
                   if st.session_state.get("q_is_follow_up") else "")
        _rl_badge_html = ""
        if RL_AVAILABLE and "all" in st.session_state.get("difficulty", "").lower():
            _rl_badge_html = (
                f'<div id="tw-rl-{qn}" style="margin-top:.65rem;opacity:0;transform:translateY(5px);'
                f'font-family:Share Tech Mono,monospace;font-size:.62rem;color:#7f5af0;'
                f'display:flex;align-items:center;gap:6px;">'
                f'<span style="width:5px;height:5px;border-radius:50%;background:#7f5af0;'
                f'display:inline-block;"></span>'
                f'RL SEQUENCER — {_diff_str.upper()} difficulty &bull; adaptive selection active'
                f'</div>'
            )

        _fu_tag_safe = "FOLLOW-UP · " if st.session_state.get("q_is_follow_up") else ""
        import json as _json
        _q_text_js   = _json.dumps(_q_text)
        _target_safe = _json.dumps(target_html)

        # ── v4: Animated Question Card with STAR tracker + live emotion HUD ──
        # Upgraded design: gradient border sweep, STAR method chips, corner glow,
        # animated scan line, live emotion + voice pills, typewriter effect.
        _q_chars_per_line = 55
        _q_lines = max(1, -(-len(_q_text) // _q_chars_per_line))
        _q_card_height = max(260, 130 + _q_lines * 48)

        # Determine which STAR elements apply for this question type
        _is_behavioural = qtype.lower() in ("behavioural", "hr", "behavioral")
        _star_chips_html = ""
        if _is_behavioural:
            _star_items = [("S","SITUATION","#00d4c8"),("T","TASK","#3b8bff"),
                           ("A","ACTION","#a855f7"),("R","RESULT","#22c55e")]
            _star_chips_html = "".join([
                f'<div class="star-chip" style="border-color:{col}33;color:{col}88;">'
                f'<span class="star-letter" style="color:{col};">{lt}</span>{name}</div>'
                for lt,name,col in _star_items
            ])

        components.html(f"""<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:transparent;font-family:'Inter',sans-serif;overflow:hidden;}}

@keyframes borderSweep{{0%,100%{{background-position:0% 50%}}50%{{background-position:100% 50%}}}}
@keyframes glowPulse{{0%,100%{{box-shadow:0 0 16px rgba(0,212,200,.12),0 0 0 0 rgba(0,212,200,0)}}50%{{box-shadow:0 0 36px rgba(0,212,200,.28),0 4px 28px rgba(0,140,200,.15)}}}}
@keyframes scanMove{{0%{{top:-32px}}100%{{top:108%}}}}
@keyframes emoDot{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.35;transform:scale(.55)}}}}
@keyframes cursor{{0%,100%{{opacity:1}}50%{{opacity:0}}}}
@keyframes slideUp{{0%{{opacity:0;transform:translateY(14px)}}100%{{opacity:1;transform:translateY(0)}}}}
@keyframes starPop{{0%{{transform:scale(.7);opacity:0}}70%{{transform:scale(1.08)}}100%{{transform:scale(1);opacity:1}}}}
@keyframes cornerGlow{{0%,100%{{opacity:.4}}50%{{opacity:.9}}}}
@keyframes neonFlicker{{0%,94%,100%{{opacity:1}}95%{{opacity:.6}}97%{{opacity:1}}99%{{opacity:.75}}}}

/* Gradient border shell */
.card-border{{
  background:linear-gradient(135deg,{_live_accent},{_live_accent}aa,#3b8bff,#a855f7,{_live_accent});
  background-size:300% 300%;
  animation:borderSweep 6s ease infinite, glowPulse 3.5s ease-in-out infinite;
  border-radius:16px;
  padding:2px;
  width:100%;
}}
.card{{
  background:linear-gradient(145deg,#060e20 0%,#0a1830 60%,#070f1e 100%);
  border-radius:14px;
  position:relative;
  overflow:hidden;
  width:100%;
  animation:slideUp .45s cubic-bezier(.22,.68,0,1.2) both;
}}
/* Scan line shimmer */
.scan{{
  position:absolute;left:0;width:100%;height:28px;
  background:linear-gradient(transparent,rgba(0,220,200,.05),transparent);
  pointer-events:none;
  animation:scanMove 5s linear infinite;
  z-index:0;
}}
/* Corner accent glow */
.corner-tl{{
  position:absolute;top:0;left:0;width:70px;height:70px;
  background:radial-gradient(circle at top left,rgba(0,220,200,.14),transparent 65%);
  animation:cornerGlow 3s ease-in-out infinite;
}}
.corner-tr{{
  position:absolute;top:0;right:0;width:90px;height:90px;
  background:radial-gradient(circle at top right,rgba(59,139,255,.10),transparent 65%);
}}
/* Header row */
.header{{
  display:flex;align-items:center;gap:7px;flex-wrap:wrap;
  padding:11px 16px 9px;
  border-bottom:1px solid rgba(0,220,200,.1);
  background:rgba(0,220,200,.025);
  position:relative;z-index:1;
}}
/* Progress pip track */
.pip-track{{display:flex;gap:5px;align-items:center;margin-left:auto;}}
.pip{{width:22px;height:3px;border-radius:99px;background:rgba(255,255,255,.1);transition:all .3s;}}
.pip.done{{background:rgba(0,220,200,.7);box-shadow:0 0 5px rgba(0,220,200,.4);}}
.pip.active{{background:{_live_accent};box-shadow:0 0 8px {_live_accent}88;animation:neonFlicker 2s infinite;}}
/* Badges */
.qnum{{
  font-family:'JetBrains Mono',monospace;font-size:10.5px;
  color:rgba(0,220,200,.75);letter-spacing:1.5px;font-weight:500;
}}
.diff-pill{{
  padding:2px 10px;border-radius:20px;font-size:10px;font-weight:600;
  background:{_diff_col}15;color:{_diff_col};border:1px solid {_diff_col}35;
  font-family:'JetBrains Mono',monospace;letter-spacing:1px;
}}
.emo-pill{{
  display:inline-flex;align-items:center;gap:5px;
  padding:3px 10px;border-radius:20px;font-size:10px;
  border:1px solid {ec}35;background:{ec}0c;color:{ec};
}}
.emo-dot{{
  width:5px;height:5px;border-radius:50%;background:{ec};
  display:inline-block;animation:emoDot 1.6s infinite;flex-shrink:0;
}}
.voice-pill{{
  display:inline-flex;align-items:center;gap:5px;
  padding:3px 10px;border-radius:20px;font-size:10px;
  border:1px solid {ve_c}35;background:{ve_c}0c;color:{ve_c};
}}
.fu-badge{{
  display:inline-block;font-size:9px;color:#f0abfc;
  font-family:'JetBrains Mono',monospace;letter-spacing:1px;
  background:rgba(240,171,252,.08);border:1px solid rgba(240,171,252,.25);
  border-radius:10px;padding:2px 9px;
}}
/* Body */
.body{{padding:16px 18px 14px;position:relative;z-index:1;}}
.accent-bar{{
  width:3px;flex-shrink:0;border-radius:2px;margin-top:5px;min-height:56px;
  background:linear-gradient(180deg,{_live_accent},{_live_accent}55,transparent);
  box-shadow:0 0 8px {_live_accent}44;
}}
.qtext{{
  font-size:1.6rem;color:#d8eeff;font-family:'Inter',sans-serif;
  font-weight:500;line-height:1.74;flex:1;letter-spacing:-.01em;
}}
.cursor{{
  display:inline-block;width:2px;height:1.05em;
  background:{_live_accent};vertical-align:middle;margin-left:3px;
  animation:cursor .65s step-end infinite;
}}
/* STAR chips */
.star-row{{
  display:flex;gap:7px;margin-top:14px;flex-wrap:wrap;
}}
.star-chip{{
  display:inline-flex;align-items:center;gap:5px;
  padding:4px 12px;border-radius:6px;
  border:1px solid;
  font-size:9.5px;letter-spacing:.8px;font-family:'JetBrains Mono',monospace;
  animation:starPop .4s cubic-bezier(.22,.68,0,1.3) both;
  transition:all .2s;
}}
.star-chip:nth-child(1){{animation-delay:.1s;}}
.star-chip:nth-child(2){{animation-delay:.2s;}}
.star-chip:nth-child(3){{animation-delay:.3s;}}
.star-chip:nth-child(4){{animation-delay:.4s;}}
.star-letter{{font-weight:800;font-size:11px;font-family:'Syne',sans-serif;margin-right:1px;}}
</style>
</head>
<body>
<div class="card-border">
<div class="card">
  <div class="scan"></div>
  <div class="corner-tl"></div>
  <div class="corner-tr"></div>
  <div class="header">
    <span class="qnum">Q{qn}/{total} &middot; {qtype.upper() or "GENERAL"}</span>
    {'<span class="fu-badge">↳ FOLLOW-UP</span>' if st.session_state.get("q_is_follow_up") else ""}
    <span class="diff-pill">{_diff_str.upper()}</span>
    <span class="emo-pill"><span class="emo-dot"></span>{dom}</span>
    <span class="voice-pill">&#9836; {st.session_state.live_voice_emotion}</span>
    <div class="pip-track">
      {"".join(f'<div class="pip {"done" if i < qn-1 else "active" if i == qn-1 else ""}"></div>' for i in range(total))}
    </div>
  </div>
  <div class="body">
    <div id="target-html"></div>
    <div style="display:flex;gap:14px;align-items:flex-start;">
      <div class="accent-bar"></div>
      <div>
        <div id="qtext" class="qtext"><span class="cursor"></span></div>
        {"<div class='star-row'>" + _star_chips_html + "</div>" if _star_chips_html else ""}
      </div>
    </div>
  </div>
</div>
</div>
<script>
(function(){{
  var full   = {_q_text_js};
  var target = {_target_safe};
  var th = document.getElementById('target-html');
  if (th && target) th.innerHTML = target;

  var el  = document.getElementById('qtext');
  var cur = el.querySelector('.cursor');
  var i   = 0;
  function tick(){{
    if (i < full.length) {{
      cur.insertAdjacentText('beforebegin', full[i]);
      i++;
      setTimeout(tick, 16);
    }} else {{
      setTimeout(function(){{ cur.style.display='none'; }}, 2000);
    }}
  }}
  setTimeout(tick, 60);
}})();
</script>
</body>
</html>""", height=_q_card_height)

        # ── Ideal answer hint (resume mode only) ──────────────────────────────
        ideal = q_dict.get("ideal_answer", "")
        if resume_mode and ideal:
            with st.expander("◈ MODEL ANSWER HINT"):
                st.markdown(
                    f'<div style="font-size:.82rem;color:#c8dff2;line-height:1.65;'
                    f'background:rgba(0,212,255,.06);border-left:3px solid #00d4ff;'
                    f'padding:.65rem 1rem;border-radius:4px;">{ideal}</div>',
                    unsafe_allow_html=True,
                )

        # ── Voice input panel (pushed down with larger container) ────────────
        st.markdown('<div style="margin-top:0.3rem;"></div>', unsafe_allow_html=True)

        # ── BiLSTM nervousness model status banner (visible, not hidden) ──────
        if not st.session_state.get("nervousness_ready", False):
            st.markdown("""
<div style="background:rgba(251,191,36,.07);border:1px solid rgba(251,191,36,.3);
  border-radius:8px;padding:.5rem .9rem;margin-bottom:.5rem;
  display:flex;align-items:center;gap:.7rem;">
  <span style="font-size:1rem;">⚠️</span>
  <div style="font-size:.74rem;color:#fbbf24;font-family:Inter,sans-serif;line-height:1.4;">
    <b>Voice nervousness model not trained</b> — the nervousness section will be empty.
    Go to <b>Model Setup → Train Voice Model (CREMA-D + TESS)</b> to enable it.
  </div>
</div>""", unsafe_allow_html=True)

        # ── v4: Animated voice section header with live 4-metric HUD ────────
        # Hidden: show this HUD only after the answer is submitted
        if st.session_state.get("submitted", False):
         components.html(f"""
<style>
@keyframes vDotPulse{{0%,100%{{box-shadow:0 0 0 0 rgba(0,255,136,.45)}}50%{{box-shadow:0 0 0 7px rgba(0,255,136,0)}}}}
@keyframes waveBar{{0%,100%{{height:5px}}50%{{height:var(--h,18px)}}}}
@keyframes hud-in{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes neonBlink{{0%,94%,100%{{opacity:1}}95%{{opacity:.5}}98%{{opacity:.85}}}}
body{{margin:0;padding:0;background:transparent;font-family:'JetBrains Mono',monospace;}}

.vh-wrap{{
  border:1px solid rgba(0,220,200,.18);
  border-bottom:none;
  border-radius:14px 14px 0 0;
  overflow:hidden;
  background:rgba(4,12,30,.9);
}}
.vh-top{{
  display:flex;align-items:center;gap:9px;padding:9px 16px;
  background:rgba(0,220,200,.025);
  border-bottom:1px solid rgba(0,220,200,.1);
}}
.vdot{{
  width:6px;height:6px;border-radius:50%;background:#00ff88;
  box-shadow:0 0 6px #00ff88;animation:vDotPulse 1.8s infinite;flex-shrink:0;
}}
.vlbl{{
  font-size:9px;color:rgba(0,220,200,.55);letter-spacing:2.5px;text-transform:uppercase;
}}
.vrec-pill{{
  margin-left:auto;
  display:inline-flex;align-items:center;gap:5px;
  background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);
  border-radius:99px;padding:2px 10px;
  font-size:9px;color:#ff5c5c;letter-spacing:1px;
  animation:neonBlink 2.5s infinite;
}}
.vrec-dot{{
  width:5px;height:5px;border-radius:50%;background:#ff5c5c;
  animation:vDotPulse 1s infinite;
}}

/* 4-metric HUD row */
.hud-row{{
  display:flex;gap:0;border-top:1px solid rgba(255,255,255,.04);
}}
.hud-cell{{
  flex:1;padding:8px 12px;border-right:1px solid rgba(255,255,255,.04);
  display:flex;flex-direction:column;gap:3px;
  animation:hud-in .4s ease both;
}}
.hud-cell:last-child{{border-right:none;}}
.hud-cell:nth-child(1){{animation-delay:.05s;}}
.hud-cell:nth-child(2){{animation-delay:.12s;}}
.hud-cell:nth-child(3){{animation-delay:.19s;}}
.hud-cell:nth-child(4){{animation-delay:.26s;}}
.hc-label{{
  font-size:7.5px;letter-spacing:1.5px;color:rgba(255,255,255,.22);
  text-transform:uppercase;
}}
.hc-val{{
  font-size:18px;font-weight:700;line-height:1;color:#d4eeff;
  font-family:'JetBrains Mono',monospace;
}}
.hc-sub{{font-size:8px;color:rgba(0,220,200,.45);}}
.hc-bar{{
  height:3px;border-radius:99px;overflow:hidden;
  background:rgba(255,255,255,.06);margin-top:2px;
}}
.hc-fill{{height:100%;border-radius:99px;}}

/* Mini waveform inside HUD */
.mini-wave{{display:flex;align-items:center;gap:2px;height:18px;margin-top:3px;}}
.mwb{{
  width:3px;border-radius:2px;min-height:3px;
  background:rgba(0,220,200,.4);
  animation:waveBar 1s ease-in-out infinite;
}}
.mwb:nth-child(2n){{animation-delay:.12s;}}
.mwb:nth-child(3n){{animation-delay:.22s;}}
.mwb:nth-child(5n){{animation-delay:.07s;}}
</style>

<div class="vh-wrap">
  <div class="vh-top">
    <div class="vdot"></div>
    <span class="vlbl">Voice Input System</span>
    <div class="vrec-pill">
      <div class="vrec-dot"></div>LIVE
    </div>
  </div>
  <div class="hud-row">
    <div class="hud-cell">
      <div class="hc-label">Fillers</div>
      <div class="hc-val" id="hc-fillers" style="color:#22c55e;">—</div>
      <div class="hc-sub" id="hc-filler-sub">of 0 words</div>
    </div>
    <div class="hud-cell">
      <div class="hc-label">WPM</div>
      <div class="hc-val" id="hc-wpm">—</div>
      <div class="hc-bar"><div class="hc-fill" id="hc-wpm-bar"
        style="width:0%;background:linear-gradient(90deg,#00d4c8,#3b8bff);"></div></div>
    </div>
    <div class="hud-cell">
      <div class="hc-label">Duration</div>
      <div class="hc-val" id="hc-dur">0s</div>
      <div class="hc-sub" style="color:rgba(251,191,36,.5);">aim 60–150s</div>
    </div>
    <div class="hud-cell">
      <div class="hc-label">Waveform</div>
      <div class="mini-wave">
        <div class="mwb" style="--h:22px"></div><div class="mwb" style="--h:14px"></div>
        <div class="mwb" style="--h:28px"></div><div class="mwb" style="--h:10px"></div>
        <div class="mwb" style="--h:20px"></div><div class="mwb" style="--h:26px"></div>
        <div class="mwb" style="--h:12px"></div><div class="mwb" style="--h:18px"></div>
        <div class="mwb" style="--h:24px"></div><div class="mwb" style="--h:16px"></div>
      </div>
    </div>
  </div>
</div>

<script>
(function(){{
  var FILLERS = ["um","uh","like","basically","actually","you know","right","so",
    "just","kind of","sort of","i mean","literally","honestly","obviously","really","very"];
  var startTime = null;

  function countFillers(txt){{
    var lo = txt.toLowerCase();
    return FILLERS.reduce(function(n,f){{ return n + (lo.split(f).length-1); }},0);
  }}

  function update(){{
    var p = window.parent;
    var areas = (p ? p.document : document).querySelectorAll('textarea');
    var best='', bestLen=-1;
    areas.forEach(function(ta){{if(ta.value.length>bestLen){{best=ta.value;bestLen=ta.value.length;}}  }});

    var words = best.trim()==='' ? [] : best.trim().split(/\\s+/);
    var wc = words.length;
    var fc = countFillers(best);

    // Fillers
    var fEl  = document.getElementById('hc-fillers');
    var fsEl = document.getElementById('hc-filler-sub');
    if(fEl){{ fEl.textContent = fc; fEl.style.color = fc>3?'#ef4444':fc>1?'#fbbf24':'#22c55e'; }}
    if(fsEl) fsEl.textContent = 'of '+wc+' words';

    // WPM (rough: words / elapsed minutes)
    if(wc>5 && !startTime) startTime=Date.now();
    var wpmEl = document.getElementById('hc-wpm');
    var wpmBar = document.getElementById('hc-wpm-bar');
    if(startTime && wc>0){{
      var mins = (Date.now()-startTime)/60000;
      var wpm  = mins>0 ? Math.round(wc/mins) : 0;
      if(wpmEl) wpmEl.textContent = wpm>0 ? wpm : '—';
      if(wpmBar) wpmBar.style.width = Math.min(100, wpm/200*100)+'%';
    }}

    // Duration
    var durEl = document.getElementById('hc-dur');
    if(startTime && durEl){{
      var secs = Math.round((Date.now()-startTime)/1000);
      durEl.textContent = secs+'s';
      durEl.style.color = secs>=60&&secs<=150?'#22c55e':secs>150?'#ef4444':'#d4eeff';
    }}
  }}

  setInterval(update, 1200);
  update();
}})();
</script>
""", height=118)
        # Thin styled border wrapping the voice panel body
        st.markdown('<div style="border:1px solid rgba(0,229,255,.13);border-top:none;border-radius:0 0 14px 14px;padding:.5rem .4rem .3rem;margin-bottom:.4rem;background:rgba(7,15,36,.7);">', unsafe_allow_html=True)
        answer_text = voice_input_panel(stt, qn)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── LIVE WORD COUNT INDICATOR (hidden) ───────────────────────────────
        # Word range targets per difficulty level (kept for JS timer logic,
        # but the visual bar is hidden per UI preference).
        _diff_raw = (q_dict.get("difficulty") or
                     st.session_state.get("difficulty", "Medium") or "Medium")
        _diff_key = str(_diff_raw).lower()
        if "easy" in _diff_key:
            _wc_lo, _wc_hi, _diff_label = 60,  120, "Easy"
        elif "hard" in _diff_key:
            _wc_lo, _wc_hi, _diff_label = 150, 280, "Hard"
        else:  # medium / all / adaptive
            _wc_lo, _wc_hi, _diff_label = 100, 200, "Medium"

        if False:  # Word count bar hidden — remove this condition to re-enable
         components.html(f"""
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:transparent;font-family:Inter,sans-serif;}}
#wc-root{{
  margin:.3rem 0 .1rem;
  padding:.55rem 1rem;
  background:rgba(0,8,22,0.6);
  border:1px solid rgba(0,240,255,0.12);
  border-radius:10px;
  display:flex;align-items:center;gap:.8rem;
  min-height:40px;
  transition:border-color .4s;
}}
#wc-root.ok{{border-color:rgba(16,185,129,0.35);}}
#wc-root.short{{border-color:rgba(245,158,11,0.35);}}
#wc-root.over{{border-color:rgba(129,140,248,0.35);}}

/* Pulsing dot */
#wc-dot{{
  width:9px;height:9px;border-radius:50%;flex-shrink:0;
  background:#334155;
  transition:background .3s,box-shadow .3s;
}}
#wc-dot.live{{animation:dot-pulse-wc 1.4s ease-in-out infinite;}}
@keyframes dot-pulse-wc{{
  0%,100%{{box-shadow:0 0 0 0 rgba(16,185,129,0.5);transform:scale(1);}}
  50%{{box-shadow:0 0 0 5px rgba(16,185,129,0);transform:scale(1.2);}}
}}

/* Count number */
#wc-count{{
  font-size:1.05rem;font-weight:700;color:#334155;
  font-family:'Orbitron',monospace;
  transition:color .3s;min-width:2.4rem;
}}

/* Waveform mini bars */
#wc-wave{{display:flex;align-items:flex-end;gap:2px;height:18px;flex-shrink:0;}}
.wb{{width:3px;border-radius:2px 2px 0 0;background:#1e293b;transition:height .2s,background .3s;}}

/* Progress bar + labels */
.wc-mid{{flex:1;display:flex;flex-direction:column;gap:3px;}}
.wc-labels{{display:flex;justify-content:space-between;align-items:baseline;}}
.wc-words-lbl{{font-size:.68rem;color:#475569;}}
.wc-range-lbl{{font-size:.65rem;color:#334155;font-family:'Share Tech Mono',monospace;transition:color .3s;}}
.wc-track{{height:4px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden;position:relative;}}
.wc-fill{{height:100%;width:0%;border-radius:2px;transition:width .35s cubic-bezier(.4,0,.2,1),background .3s;background:#334155;}}
.wc-fill::after{{content:'';position:absolute;right:0;top:0;height:100%;width:24px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.25));animation:bar-tip-glow 1.2s ease-in-out infinite;}}
@keyframes bar-tip-glow{{0%,100%{{opacity:0;}}50%{{opacity:1;}}}}

/* Status chip */
#wc-status{{
  font-size:.68rem;font-weight:700;color:#475569;
  font-family:'Share Tech Mono',monospace;white-space:nowrap;
  min-width:5.5rem;text-align:right;
  padding:2px 8px;border-radius:6px;
  border:1px solid transparent;
  transition:all .3s;
}}
</style>
</head>
<body>
<div id="wc-root">
  <div id="wc-dot"></div>
  <span id="wc-count">0</span>
  <span style="font-size:.7rem;color:#475569;">words</span>

  <!-- Mini waveform bars -->
  <div id="wc-wave">
    <div class="wb" id="wb0"></div><div class="wb" id="wb1"></div><div class="wb" id="wb2"></div>
    <div class="wb" id="wb3"></div><div class="wb" id="wb4"></div><div class="wb" id="wb5"></div>
    <div class="wb" id="wb6"></div><div class="wb" id="wb7"></div>
  </div>

  <div class="wc-mid">
    <div class="wc-labels">
      <span class="wc-words-lbl" id="wc-hint">· target {_wc_lo}–{_wc_hi} for {_diff_label}</span>
      <span class="wc-range-lbl" id="wc-status">Start typing</span>
    </div>
    <div class="wc-track"><div class="wc-fill" id="wc-fill"></div></div>
  </div>
</div>

<script>
(function(){{
  const LO  = {_wc_lo};
  const HI  = {_wc_hi};
  const CAP = HI * 1.6;
  const root   = document.getElementById('wc-root');
  const dot    = document.getElementById('wc-dot');
  const count  = document.getElementById('wc-count');
  const fill   = document.getElementById('wc-fill');
  const status = document.getElementById('wc-status');
  const bars   = [0,1,2,3,4,5,6,7].map(i=>document.getElementById('wb'+i));

  const COLORS = {{
    empty: {{text:'#334155',bar:'#334155',dot:'#334155',lbl:'Start typing',cls:''}},
    short: {{text:'#f59e0b',bar:'#f59e0b',dot:'#f59e0b',lbl:null,cls:'short'}},
    ok:    {{text:'#10b981',bar:'linear-gradient(90deg,#10b981,#00d4ff)',dot:'#10b981',lbl:'✓ In range',cls:'ok'}},
    over:  {{text:'#a5b4fc',bar:'linear-gradient(90deg,#818cf8,#a5b4fc)',dot:'#818cf8',lbl:null,cls:'over'}},
  }};

  // Animate waveform bars based on word count
  function animateBars(n){{
    const intensity = Math.min(n / HI, 1);
    bars.forEach(function(b,i){{
      const rnd = 0.15 + Math.random() * 0.85 * intensity;
      const h   = Math.max(2, Math.round(rnd * 16));
      b.style.height = h + 'px';
      const col = n === 0 ? '#1e293b' : (n < LO ? '#f59e0b' : n <= HI ? '#10b981' : '#818cf8');
      b.style.background = col;
    }});
  }}

  function applyState(n){{
    let c;
    if (n === 0)      c = COLORS.empty;
    else if (n < LO)  c = COLORS.short;
    else if (n <= HI) c = COLORS.ok;
    else              c = COLORS.over;

    count.textContent = n;
    count.style.color = c.text;
    fill.style.background = c.bar;
    dot.style.background  = c.dot;
    status.textContent = c.lbl || (n < LO ? (LO - n) + ' short' : (n - HI) + ' over');
    status.style.color = c.text;
    status.style.borderColor = c.text + '33';
    status.style.background  = c.text + '11';

    root.className = c.cls;
    if (n > 0) {{ dot.classList.add('live'); dot.style.animationName='dot-pulse-wc'; }}
    else dot.classList.remove('live');

    fill.style.width = Math.min(100, n / CAP * 100) + '%';
    animateBars(n);
  }}

  function countWords(txt){{
    return txt.trim() === '' ? 0 : txt.trim().split(/\s+/).length;
  }}

  function probe(){{
    const areas = window.parent ? window.parent.document.querySelectorAll('textarea') : document.querySelectorAll('textarea');
    if (!areas.length) return;
    let best = null, bestLen = -1;
    areas.forEach(function(ta){{
      if(ta.value.length > bestLen){{ best=ta; bestLen=ta.value.length; }}
    }});
    if(best){{
      applyState(countWords(best.value));
      if(!best._wcBound){{
        best._wcBound = true;
        best.addEventListener('input', function(){{ applyState(countWords(best.value)); }});
      }}
    }}
  }}

  // Idle bar animation when empty
  setInterval(function(){{ if(count.textContent === '0') animateBars(0); }}, 600);

  let polls = 0;
  const iv = setInterval(function(){{ probe(); if(++polls>60) clearInterval(iv); }}, 100);
  setInterval(probe, 1500);
}})();
</script>
</body>
</html>
""", height=56)
        # ─────────────────────────────────────────────────────────────────────

        # Persist answer to session_state so the submit callback (which runs
        # on a fresh Streamlit re-run) can still read it — fixes closure bug.
        if answer_text.strip():
            # answer_text is non-empty: this is either a fresh Whisper/typed answer,
            # or the Browser STT stable key was recovered by browser_stt_with_audio()
            # on the submit rerun. Persist everything so the callback can read it.
            st.session_state["_pending_answer"]  = answer_text
            st.session_state["_pending_q_text"]  = st.session_state.question
            st.session_state["_pending_qn"]      = qn
            st.session_state["_pending_qtype"]   = qtype
            st.session_state["_pending_q_dict"]  = q_dict
            st.session_state["_pending_kws"]     = kws
        else:
            # answer_text is "": the bridge text_input was reset by Streamlit.
            # Pull from the STABLE Browser STT key (never reset by Streamlit):
            #   priority: _bstt_last_tx_{qn}  →  transcribed_text_browser_{qn}
            # The old guard "elif not _pending_answer" wrongly blocked this path
            # whenever a previous question's answer was still in _pending_answer.
            _bstt_fallback = (
                st.session_state.get(f"_bstt_last_tx_{qn}", "")
                or st.session_state.get(f"transcribed_text_browser_{qn}", "")
            )
            if _bstt_fallback.strip():
                st.session_state["_pending_answer"] = _bstt_fallback
                st.session_state["_pending_q_text"] = st.session_state.question
                st.session_state["_pending_qn"]     = qn
                st.session_state["_pending_qtype"]  = qtype
                st.session_state["_pending_q_dict"] = q_dict
                st.session_state["_pending_kws"]    = kws
        _pending    = st.session_state.get("_pending_answer", "")
        _pending_qn = st.session_state.get("_pending_qn", qn)

        # ── SMART BUTTON ROW ─────────────────────────────────────────────────
        # Before submit: show SUBMIT ANSWER only
        # After submit: show NEXT QUESTION or FINISH SESSION (not both)
        # Finish shows when this was the last question in the session.
        _submitted      = st.session_state.get("submitted", False)
        _num_qs         = st.session_state.get("num_questions", 5)
        _orig_answered  = len([a for a in st.session_state.session_answers
                                if not a.get("is_follow_up", False)])
        _is_last_q      = (_orig_answered >= _num_qs) or (qn >= total and _submitted)

        # Follow-up pending label
        _fu_pending_btn = (
            "all" in st.session_state.get("difficulty","").lower()
            and hasattr(engine, "is_follow_up_pending")
            and engine.is_follow_up_pending()
            and _submitted
        )

        if not _submitted:
            # ── Before submit: single full-width Submit button ───────────────
            def submit_answer():
                _ans    = st.session_state.get("_pending_answer", "")
                # Multi-tier fallback using the LOCAL closure `qn` (NOT
                # st.session_state.q_number which can differ with key_suffix).
                # Priority: stable _bstt_last_tx key → compat key → shared key.
                if not _ans.strip():
                    _ans = st.session_state.get(f"_bstt_last_tx_{qn}", "")
                if not _ans.strip():
                    _ans = st.session_state.get(f"transcribed_text_browser_{qn}", "")
                if not _ans.strip():
                    _ans = st.session_state.get("transcribed_text", "")
                # Recover audio bytes if submit rerun cleared last_audio_bytes.
                # Try both the stable backup key AND the typed/whisper path.
                if not st.session_state.get("last_audio_bytes"):
                    for _backup_key in ("_bstt_last_audio_bytes",):
                        _saved = st.session_state.get(_backup_key)
                        if _saved:
                            st.session_state["last_audio_bytes"]  = _saved
                            st.session_state["last_audio_source"] = "browser_stt"
                            break
                # Guard: nothing to submit — set a flag so UI can warn user
                if not _ans.strip():
                    st.session_state["_submit_empty_warning"] = True
                    return
                _q_text = st.session_state.get("_pending_q_text", st.session_state.question)
                _qn     = st.session_state.get("_pending_qn", st.session_state.q_number)
                _qtype  = st.session_state.get("_pending_qtype", "")
                _q_dict = st.session_state.get("_pending_q_dict", {})
                _kws    = st.session_state.get("_pending_kws", [])
                if _ans.strip():
                    elapsed2 = max(1, int(time.time() - (st.session_state.q_start_time or time.time())))
                    _q_difficulty = (_q_dict.get("difficulty") or
                                     st.session_state.get("difficulty", "medium"))
                    if "all" in str(_q_difficulty).lower():
                        _q_difficulty = "medium"
                    ev    = evaluator.evaluate(
                        _ans,
                        ideal_answer=_q_dict.get("ideal_answer", ""),
                        question_keywords=_kws,
                        answer_duration_seconds=float(elapsed2),
                        question_text=_q_text,
                        question_type=_qtype or "Technical",
                        difficulty=_q_difficulty,
                    )
                    res   = engine.evaluate_answer(_q_text, _ans)
                    score = ev.get("score", res.get("score", 1.0))
                    st.session_state.update({
                        "last_score":    score,
                        "last_feedback": ev.get("feedback", res.get("feedback","")),
                        "last_star":     ev.get("star_scores", {}),
                        "last_keywords": ev.get("keyword_hits", []),
                        "last_eval":        ev,
                        "submitted":        True,
                        "transcribed_text": "",
                        "_pending_answer":  "",
                        "last_answer_text": _ans,   # v12.2: used by scan beam
                    })
                    em_summary = engine.get_emotion_summary()
                    _audio_raw = st.session_state.get("last_audio_bytes")
                    _audio_src = st.session_state.get("last_audio_source", "")
                    # v9.0: recover stable audio backup (Browser STT sets this
                    # before the rerun clears last_audio_bytes)
                    if not _audio_raw:
                        _audio_raw = st.session_state.get("_bstt_last_audio_bytes")

                    if st.session_state.get("nervousness_ready"):
                        try:
                            if _audio_src == "browser_stt":
                                # Browser STT path: transcript is the reliable signal.
                                # Audio bytes are opportunistic — included when available
                                # but the text score alone is sufficient.
                                _voice_result = nerv_pipeline.score_from_transcript(
                                    _ans, _audio_raw
                                )
                            elif _audio_raw and len(_audio_raw) > 1024:
                                # Whisper / file path: full acoustic pipeline
                                _voice_result = nerv_pipeline.predict_from_bytes(_audio_raw)
                                # Also compute text score and store for reference
                                _txt = nerv_pipeline._text_analyzer.analyze(_ans)
                                _voice_result["text_nervousness"] = _txt["nervousness"]
                                _voice_result["method"] = "audio_primary"
                            else:
                                # No audio at all (typed answer) — text only
                                _voice_result = nerv_pipeline.score_from_transcript(_ans)

                            _v_conf = _voice_result.get("confidence", 50.0) / 100 * 5
                            engine._live_voice_scores.append(_v_conf)
                            engine.voice_quality.record(_voice_result)
                        except Exception:
                            pass
                    st.session_state["last_audio_bytes"]  = None
                    st.session_state["last_audio_source"] = None
                    st.session_state.pop("_bstt_last_audio_bytes", None)  # v9.0 clear backup
                    vs_summary = engine.get_voice_session_summary()
                    mm_conf    = engine.get_multimodal_confidence()
                    st.session_state.session_answers.append({
                        "number": _qn, "question": _q_text,
                        "answer": _ans, "score": score, "time_s": elapsed2,
                        "type": _qtype, "feedback": ev.get("feedback",""),
                        "difficulty": _q_difficulty,
                        "star": ev.get("star_scores",{}),
                        "final_score":       ev.get("final_score",0),
                        "similarity_score":  ev.get("similarity_score",0),
                        "grammar_score":     ev.get("grammar_score",0),
                        "emotion":           em_summary.get("dominant","Neutral"),
                        "nervousness":       vs_summary.get("nervousness",0.2),   # v10.0: 100% voice
                        "facial_nervousness":em_summary.get("nervousness", 0.2),  # kept for reference only
                        "voice_emotion":     vs_summary.get("dominant","Neutral"),
                        "voice_nervousness": vs_summary.get("nervousness",0.2),
                        "confidence_score":  mm_conf.get("confidence_score",3.5),
                        "posture_score":     mm_conf.get("posture_score",3.5),
                        "facial_score":      mm_conf.get("facial_score",3.5),
                        "voice_score":       mm_conf.get("voice_score",3.5),
                        "eye_score":         mm_conf.get("eye_score",3.5),
                        "depth_score":       ev.get("depth_score", round(score, 2)),
                        "fluency":           ev.get("fluency_score", 3.5),
                        "fluency_score":     ev.get("fluency_score", 3.5),
                        "disc_traits":       ev.get("disc_traits", {}),
                        "disc_dominant":     ev.get("disc_dominant", "None"),
                        "word_count":        ev.get("word_count", 0),
                        "filler_count":      ev.get("filler_count", 0),
                        "filler_ratio":      ev.get("filler_ratio", 0.0),
                        "hiring_signal":     ev.get("hiring_signal", 2.5),
                        "star_count":        ev.get("star_count", 0),
                        "vocab_diversity":   ev.get("vocab_diversity", 0.0),
                        "relevance_source":  ev.get("relevance_source", "none"),
                        "question_type":     ev.get("question_type", _qtype.lower()),
                        "keyword_hits":      ev.get("keyword_hits", []),
                        "keywords":          _kws or [],
                        "resume_target":     _q_dict.get("target", ""),
                        "source":            _q_dict.get("source", "pool"),
                        "time_score":        ev.get("time_score", 0.0),
                        "time_label":        ev.get("time_label", "N/A"),
                        "time_efficiency":   ev.get("time_efficiency", 0.0),
                        "time_verdict":      ev.get("time_verdict", "N/A"),
                        "time_ideal_window": ev.get("time_ideal_window", "—"),
                        "rl_next_action":    ev.get("rl_next_action", "static"),
                        "rl_epsilon":        (engine.sequencer.current_epsilon
                                              if hasattr(engine,"sequencer") and engine.sequencer else None),
                        "ocean_scores":      ev.get("ocean_scores", {}),
                        "hiring_breakdown":  ev.get("hiring_breakdown", {}),
                        "is_follow_up":      False,
                        "q_is_follow_up":    False,
                        "parent_q_number":   None,
                        # v8.1: delta-nervousness (relative to personal baseline)
                        "nervousness_delta":       vs_summary.get("nervousness_delta",
                                                   vs_summary.get("nervousness", 0.2)),
                        "calibration_baseline":    st.session_state.get("calibration_baseline", 0.2),
                        "baseline_calibrated":     st.session_state.get("calibration_done", False),
                    })
                    # DISC radar spin — increment version so the chart knows to animate
                    st.session_state["_disc_spin_version"] = st.session_state.get("_disc_spin_version", 0) + 1
                    try:
                        import os as _os_coach
                        _ct = generate_coaching_tip(
                            ev=ev,
                            score=score, answer=_ans, question=_q_text,
                            q_type=_qtype,
                            groq_api_key=_os_coach.environ.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", "")))
                        st.session_state["last_coaching_tip"] = _ct
                    except Exception:
                        st.session_state["last_coaching_tip"] = ""
                    try:
                        _imp = evaluator.generate_improvement_suggestion(
                            answer=_ans, question=_q_text,
                            eval_result=ev, q_type=_qtype)
                        st.session_state["last_improvement"] = _imp
                    except Exception:
                        st.session_state["last_improvement"] = ""
                    st.session_state.transcript += (
                        f"Q{_qn}: {_q_text}\n"
                        f"A: {_ans}\n"
                        f"Score: {score:.1f}/5 | "
                        f"Emotion: {em_summary.get('dominant','Neutral')}\n\n"
                    )

            st.markdown("""<style>
/* ── Animated gradient-border submit button ── */
.v3-submit-wrap{
  position:relative;
  border-radius:14px;
  background:linear-gradient(135deg,#00d4c8,#3b8bff,#a855f7,#00d4c8);
  background-size:300% 300%;
  animation:submitBorderSweep 4s ease infinite;
  padding:2px;
  margin-top:.9rem;
  box-shadow:0 0 28px rgba(0,212,200,.18), 0 4px 24px rgba(59,139,255,.12);
}
.v3-submit-wrap>button{
  font-family:'Orbitron',monospace!important;font-size:.8rem!important;font-weight:700!important;
  letter-spacing:.18em!important;text-transform:uppercase!important;
  color:#00e5ff!important;
  background:linear-gradient(135deg,rgba(4,12,30,.97),rgba(6,16,38,.97))!important;
  border:none!important;
  border-radius:12px!important;
  padding:.95rem 1.5rem!important;
  width:100%!important;
  transition:all .25s!important;
  position:relative!important;
}
.v3-submit-wrap>button::before{
  content:'◆  ';
  font-size:.6rem;
}
.v3-submit-wrap>button::after{
  content:'  ◆';
  font-size:.6rem;
}
.v3-submit-wrap>button:hover{
  color:#ffffff!important;
  background:linear-gradient(135deg,rgba(0,212,200,.15),rgba(59,139,255,.12))!important;
  letter-spacing:.22em!important;
}
@keyframes submitBorderSweep{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
</style>
<div class="v3-submit-wrap">""", unsafe_allow_html=True)
            # Show warning if last submit attempt had no answer captured
            if st.session_state.pop("_submit_empty_warning", False):
                st.warning(
                    "⚠ No answer captured yet. "
                    "Please record or type your answer first, "
                    "then click **Submit Answer**.",
                    icon=None,
                )
            st.button("SUBMIT ANSWER", key=f"sub_{qn}", on_click=submit_answer,
                      use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.submitted and st.session_state.last_score is not None:

            # ── BLIND MODE: seal score, show lock card instead of feedback ────
            if st.session_state.get("blind_mode"):
                _qn_key = str(st.session_state.get("_pending_qn", st.session_state.q_number))
                _blind  = st.session_state.get("blind_scores", {})
                if _qn_key not in _blind:
                    _blind[_qn_key] = st.session_state.get("last_eval", {})
                    st.session_state["blind_scores"] = _blind
                st.markdown(f"""
<div style="
  background:rgba(192,132,252,.07);
  border:1px solid rgba(192,132,252,.3);
  border-left:3px solid #c084fc;
  border-radius:10px;
  padding:1rem 1.25rem;
  margin:.5rem 0;
  display:flex;align-items:center;gap:.85rem;
">
  <div style="width:36px;height:36px;border-radius:50%;
    background:rgba(192,132,252,.12);border:1.5px solid rgba(192,132,252,.4);
    display:flex;align-items:center;justify-content:center;flex-shrink:0;">
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <rect x="3" y="7" width="10" height="7" rx="2"
        stroke="#c084fc" stroke-width="1.4" fill="none"/>
      <path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2"
        stroke="#c084fc" stroke-width="1.4" stroke-linecap="round" fill="none"/>
    </svg>
  </div>
  <div>
    <div style="font-size:.78rem;font-weight:600;color:#c084fc;
      font-family:Share Tech Mono,monospace;letter-spacing:.08em;margin-bottom:2px;">
      ANSWER SEALED &mdash; Q{_qn_key}</div>
    <div style="font-size:.74rem;color:rgba(192,132,252,.65);font-family:Inter,sans-serif;">
      Score computed and locked. Revealed in Final Report with all answers.</div>
  </div>
</div>""", unsafe_allow_html=True)

            else:
                # ── NORMAL MODE: existing keyword burst + eval + coach drawer ─
                _kw_hits = st.session_state.last_eval.get("keyword_hits", [])
                if _kw_hits:
                    _kw_pills = "".join([
                        f'<span style="display:inline-block;background:rgba(0,255,136,.12);'
                        f'color:#00ff88;border:1px solid rgba(0,255,136,.35);'
                        f'border-radius:20px;padding:3px 12px;font-size:.7rem;'
                        f'font-family:Share Tech Mono,monospace;font-weight:700;'
                        f'margin:2px;letter-spacing:.04em;'
                        f'animation:kw-burst .55s cubic-bezier(.22,.68,0,1.2) {i*0.07:.2f}s both;">'
                        f'{kw}</span>'
                        for i, kw in enumerate(_kw_hits[:8])
                    ])
                    st.markdown(f"""
<style>
@keyframes kw-burst {{
  from {{ opacity:0; transform:scale(.6) translateY(6px); }}
  to   {{ opacity:1; transform:scale(1)  translateY(0);   }}
}}
</style>
<div style="background:rgba(0,255,136,.04);border:1px solid rgba(0,255,136,.15);
  border-radius:10px;padding:.5rem .75rem;margin:.4rem 0 .2rem;
  display:flex;align-items:center;flex-wrap:wrap;gap:2px;">
  <span style="font-size:.6rem;color:#4ade80;font-family:Share Tech Mono,monospace;
    font-weight:700;letter-spacing:.1em;margin-right:.35rem;white-space:nowrap;">
    ✓ KEYWORDS HIT</span>
  {_kw_pills}
</div>""", unsafe_allow_html=True)

                render_eval_results(st.session_state.last_eval)

                # ── FLOATING COACH DRAWER ────────────────────────────────────
                _tip         = st.session_state.get("last_coaching_tip", "")
                _coach_score = st.session_state.last_score or 0
                _coach_fs    = st.session_state.last_eval.get("final_score", 0)
                _drawer_key  = f"coach_drawer_open_{st.session_state.q_number}"
                if _drawer_key not in st.session_state:
                    st.session_state[_drawer_key] = True   # auto-open after submit

                _drawer_open = st.session_state.get(_drawer_key, True)
                _tog_icon    = "◈ Coach  ▾" if not _drawer_open else "◈ Coach  ▴"

                st.markdown("""<style>
.coach-drawer-toggle > button {
  background: linear-gradient(135deg,rgba(124,107,255,.18),rgba(59,139,255,.12)) !important;
  border: 1px solid rgba(124,107,255,.45) !important;
  border-radius: 20px !important;
  color: #a78bfa !important;
  font-family: Share Tech Mono, monospace !important;
  font-size: .72rem !important;
  font-weight: 700 !important;
  letter-spacing: .1em !important;
  text-transform: none !important;
  padding: .3rem 1.1rem !important;
  width: auto !important;
  box-shadow: 0 0 18px rgba(124,107,255,.2) !important;
  transition: all .2s !important;
}
.coach-drawer-toggle > button:hover {
  box-shadow: 0 0 28px rgba(124,107,255,.45) !important;
  border-color: rgba(124,107,255,.7) !important;
}
@keyframes drawer-slide-in {
  from { opacity:0; transform:translateX(32px) scaleY(.96); }
  to   { opacity:1; transform:translateX(0)    scaleY(1);   }
}
.coach-drawer-panel {
  animation: drawer-slide-in .28s cubic-bezier(.22,.68,0,1.2) both;
}
</style>""", unsafe_allow_html=True)

                _tog_col, _ = st.columns([2, 5])
                with _tog_col:
                    st.markdown('<div class="coach-drawer-toggle">', unsafe_allow_html=True)
                    if st.button(_tog_icon, key=f"coach_tog_{st.session_state.q_number}",
                                 use_container_width=False):
                        st.session_state[_drawer_key] = not _drawer_open
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

                if st.session_state.get(_drawer_key, True):
                    _sc_col  = ("#10b981" if _coach_score >= 4.0
                                else "#f59e0b" if _coach_score >= 2.5 else "#ef4444")
                    _tip_body = _tip if _tip else "Keep practising — your answer has been recorded."
                    st.markdown(f"""
<div class="coach-drawer-panel" style="
  background:linear-gradient(160deg,rgba(18,12,48,.97),rgba(10,20,50,.97));
  border:1px solid rgba(124,107,255,.4);
  border-left:3px solid #7c6bff;
  border-radius:14px;
  padding:1.1rem 1.25rem 1rem;
  margin:.45rem 0 .6rem;
  box-shadow:0 8px 40px rgba(124,107,255,.18),0 2px 12px rgba(0,0,0,.5);
">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.75rem;">
    <div style="display:flex;align-items:center;gap:.55rem;">
      <div style="width:30px;height:30px;border-radius:8px;flex-shrink:0;
        background:linear-gradient(135deg,#7c6bff,#3b8bff);
        display:flex;align-items:center;justify-content:center;
        font-size:.9rem;box-shadow:0 0 12px rgba(124,107,255,.4);">◈</div>
      <div>
        <div style="font-size:.72rem;font-weight:700;color:#a78bfa;
          font-family:Share Tech Mono,monospace;letter-spacing:.1em;line-height:1.2;">
          LIVE COACH FEEDBACK</div>
        <div style="font-size:.58rem;color:#6b7db3;font-family:Share Tech Mono,monospace;">
          Personalised coaching · Q{st.session_state.q_number}</div>
      </div>
    </div>
    <div style="background:rgba(0,0,0,.3);border:1px solid {_sc_col}55;border-radius:20px;
      padding:3px 12px;display:flex;align-items:baseline;gap:3px;">
      <span style="font-family:Orbitron,monospace;font-size:1rem;font-weight:700;
        color:{_sc_col};">{_coach_score:.1f}</span>
      <span style="font-size:.6rem;color:#475569;font-family:Inter,sans-serif;">/5</span>
    </div>
  </div>
  <div style="height:1px;background:linear-gradient(90deg,rgba(124,107,255,.35),transparent);
    margin-bottom:.75rem;"></div>
  <div style="font-size:.84rem;color:#c8d8f0;font-family:Inter,sans-serif;
    line-height:1.75;">{_tip_body}</div>
  <div style="margin-top:.85rem;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
      <span style="font-size:.58rem;color:#6b7db3;font-family:Share Tech Mono,monospace;
        letter-spacing:.08em;">OVERALL SCORE</span>
      <span style="font-size:.65rem;font-weight:700;color:{_sc_col};
        font-family:Share Tech Mono,monospace;">{_coach_fs:.0f}%</span>
    </div>
    <div style="height:3px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden;">
      <div style="width:{min(_coach_fs,100):.0f}%;height:100%;
        background:linear-gradient(90deg,#7c6bff,{_sc_col});border-radius:2px;
        box-shadow:0 0 8px {_sc_col}66;"></div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

                # v13.0: improvement suggestion card
                _improvement = st.session_state.get("last_improvement", "")
                if _improvement:
                    st.markdown(f"""
<div style="background:rgba(0,212,255,.05);border:1px solid rgba(0,212,255,.2);
  border-left:3px solid #00d4ff;border-radius:6px;padding:.75rem 1rem;margin:.4rem 0;">
  <div style="font-size:.58rem;font-weight:700;color:#00d4ff;text-transform:uppercase;
    letter-spacing:.12em;font-family:Share Tech Mono,monospace;margin-bottom:.35rem;">
    ◈ IMPROVEMENT SUGGESTION
  </div>
  <div style="font-size:.82rem;color:#b0cce8;font-family:Rajdhani,sans-serif;
    line-height:1.65;">{_improvement}</div>
</div>""", unsafe_allow_html=True)

            # ── Follow-up engine (wired in) ───────────────────────────────────
            # Reads pending_follow_up / follow_up_records from session_state.
            # Auto-triggers when score <= fu_score_threshold (default 3.2).
            # Full probe strategy, scoring, and PDF export are handled inside.
            import os as _os_fu
            _fu_ans  = st.session_state.get("_pending_answer", "") or ""
            _fu_eval = st.session_state.get("last_eval", {})
            _fu_qd   = st.session_state.get("_pending_q_dict", q_dict or {})
            _fu_idx  = max(0, st.session_state.get("q_number", 1) - 1)
            if _fu_ans.strip() and _fu_eval:
                render_follow_up_ui(
                    evaluator       = evaluator,
                    question_dict   = _fu_qd,
                    original_answer = _fu_ans,
                    evaluation      = _fu_eval,
                    q_index         = _fu_idx,
                    api_key         = _os_fu.environ.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", "")),
                    stt             = stt,
                )

        # ── Analytics below buttons — ANSWER ANALYSIS ONLY ──
        st.markdown('<hr style="border-color:rgba(255,255,255,.06);margin:.8rem 0;">', unsafe_allow_html=True)
        ans_text = answer_text or ""
        q_d      = q_dict or {}
        start_t  = st.session_state.get("q_start_time", time.time())
        duration = max(1.0, time.time() - start_t)

        last_ev = st.session_state.get("last_eval", {})
        if st.session_state.get("submitted") and last_ev:
            nlp_stats = {
                "score":      last_ev.get("final_score",   0),
                "star":       last_ev.get("star_scores",   {}),
                "hits":       last_ev.get("keyword_hits",  []),
                "word_count": last_ev.get("word_count",    0),
                "keywords":   last_ev.get("keyword_hits",  []),
            }
        else:
            nlp_stats = engine.aura_engine.analyze_answer_quality(ans_text, q_d)
        con_stats = engine.aura_engine.analyze_consistency(ans_text, duration)

        # ── Knowledge Depth — only shown AFTER answer is submitted ────────────
        if st.session_state.get("submitted") and last_ev:
            import json as _json_kd
            _kd_pct   = int(nlp_stats["score"])
            _kd_col   = "#00FFD1" if _kd_pct >= 70 else ("#FFD700" if _kd_pct >= 40 else "#FF6B6B")
            _kd_label = "STRONG" if _kd_pct >= 70 else ("MODERATE" if _kd_pct >= 40 else "WEAK")
            _cur_qtype = (q_dict or {}).get("type", "").lower()
            _star_applicable_rp = _cur_qtype in ("behavioural", "behavioral", "hr", "")
            _star = nlp_stats.get("star", {}) if _star_applicable_rp else {}
            _star_json = _json_kd.dumps(_star)
            _hits = nlp_stats.get("hits", [])[:6]
            _hits_json = _json_kd.dumps(_hits)
            components.html(f"""<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:transparent;font-family:Inter,sans-serif;overflow:hidden;}}
#kd-card{{
  background:rgba(5,12,32,0.96);
  border:1px solid rgba(0,229,255,.15);
  border-radius:12px;
  padding:14px 16px 12px;
  opacity:0;transform:translateY(10px);
  animation:cardIn .5s ease forwards;
}}
@keyframes cardIn{{to{{opacity:1;transform:translateY(0);}}}}
.kd-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;}}
.kd-title{{font-size:.68rem;color:rgba(184,212,238,.7);font-family:'JetBrains Mono',monospace;
  letter-spacing:.12em;text-transform:uppercase;}}
.kd-score{{font-size:1.3rem;font-weight:900;font-family:'Orbitron',monospace;color:{_kd_col};
  opacity:0;animation:numIn .6s ease .7s forwards;}}
@keyframes numIn{{from{{opacity:0;transform:scale(.7)}}to{{opacity:1;transform:scale(1)}}}}
.kd-label{{font-size:.58rem;color:{_kd_col};font-family:'JetBrains Mono',monospace;
  letter-spacing:2px;opacity:.7;text-align:right;margin-top:1px;}}
.track{{height:6px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;margin-bottom:10px;}}
.fill{{height:100%;width:0%;border-radius:3px;background:{_kd_col};
  box-shadow:0 0 10px {_kd_col}66;
  animation:fillBar 1.1s cubic-bezier(.4,0,.2,1) .4s forwards;}}
@keyframes fillBar{{to{{width:{_kd_pct}%;}}}}
.star-row{{display:flex;gap:4px;margin-bottom:9px;}}
.star-cell{{flex:1;text-align:center;padding:4px 2px;border-radius:4px;
  font-size:.58rem;font-weight:700;font-family:'JetBrains Mono',monospace;
  letter-spacing:.5px;opacity:0;}}
.star-cell.hit{{background:rgba(0,255,209,.12);border:1px solid #00FFD1;color:#00FFD1;
  animation:cellIn .35s ease forwards;}}
.star-cell.miss{{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
  color:rgba(90,112,152,.6);animation:cellIn .35s ease forwards;}}
@keyframes cellIn{{to{{opacity:1;}}}}
.kw-row{{display:flex;flex-wrap:wrap;gap:5px;}}
.kw-tag{{font-size:.6rem;font-family:'JetBrains Mono',monospace;
  padding:2px 8px;border-radius:10px;letter-spacing:.5px;
  background:rgba(196,181,253,.1);border:1px solid rgba(196,181,253,.25);
  color:#c4b5fd;opacity:0;animation:tagIn .3s ease forwards;}}
.kw-none{{font-size:.6rem;color:rgba(90,112,152,.5);font-family:'JetBrains Mono',monospace;}}
.divider{{height:1px;background:rgba(0,229,255,.07);margin:9px 0;}}
</style>
</head><body>
<div id="kd-card">
  <div class="kd-header">
    <div>
      <div class="kd-title">◈ Knowledge Depth</div>
    </div>
    <div style="text-align:right;">
      <div class="kd-score" id="scoreNum">0%</div>
      <div class="kd-label">{_kd_label}</div>
    </div>
  </div>
  <div class="track"><div class="fill"></div></div>
  <div class="star-row" id="starRow"></div>
  <div class="divider"></div>
  <div class="kw-row" id="kwRow"></div>
</div>
<script>
(function(){{
  var star  = {_star_json};
  var hits  = {_hits_json};
  var score = {_kd_pct};
  var scoreEl = document.getElementById('scoreNum');
  var starRow = document.getElementById('starRow');
  var kwRow   = document.getElementById('kwRow');
  var LABELS  = {{'Situation':'S · SIT','Task':'T · TASK','Action':'A · ACT','Result':'R · RES'}};

  /* animate score counter */
  var start = null;
  function countUp(ts){{
    if(!start) start=ts;
    var prog = Math.min((ts-start)/900, 1);
    var val  = Math.round(prog * score);
    scoreEl.textContent = val + '%';
    if(prog < 1) requestAnimationFrame(countUp);
  }}
  setTimeout(function(){{requestAnimationFrame(countUp);}}, 700);

  /* STAR cells */
  if(Object.keys(star).length > 0){{
    Object.keys(star).forEach(function(comp, i){{
      var hit = star[comp];
      var cell = document.createElement('div');
      cell.className = 'star-cell ' + (hit ? 'hit' : 'miss');
      cell.textContent = LABELS[comp] || comp;
      cell.style.animationDelay = (0.5 + i * 0.12) + 's';
      starRow.appendChild(cell);
    }});
  }} else {{
    starRow.style.display = 'none';
  }}

  /* keyword tags */
  if(hits.length > 0){{
    hits.forEach(function(kw, i){{
      var tag = document.createElement('div');
      tag.className = 'kw-tag';
      tag.textContent = kw;
      tag.style.animationDelay = (0.8 + i * 0.1) + 's';
      kwRow.appendChild(tag);
    }});
  }} else {{
    var none = document.createElement('div');
    none.className = 'kw-none';
    none.textContent = 'No keyword matches';
    kwRow.appendChild(none);
  }}
}})();
</script>
</body></html>""", height=185 if _star else 150)

        eh = st.session_state.emotion_history[-20:]
        if len(eh) > 2:
            st.markdown('<div class="sec-lbl" style="margin-top:.8rem;">▸ EMOTION TIMELINE</div>', unsafe_allow_html=True)
            ue = list(dict.fromkeys(eh))
            ei = [ue.index(e) for e in eh]
            fig_em = go.Figure(go.Scatter(
                y=ei, mode="lines+markers",
                line=dict(color="#00FFD1", width=2),
                marker=dict(size=5, color=[emo_css(e) for e in eh]),
                text=eh, hoverinfo="text",
                fill="tozeroy", fillcolor="rgba(0,255,136,.04)",
            ))
            fig_em.update_layout(
                height=65, margin=dict(l=0,r=0,t=5,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                yaxis=dict(tickmode="array", tickvals=list(range(len(ue))), ticktext=ue,
                           tickfont=dict(color="#5a7a9a", size=7),
                           gridcolor="rgba(0,255,136,.04)"),
                xaxis=dict(visible=False),
            )
            st.plotly_chart(fig_em, use_container_width=True, config={"displayModeBar":False})

        # ── NEXT QUESTION / FINISH SESSION — always at bottom after all analysis ──
        if st.session_state.get("submitted", False):
            _num_qs_now        = st.session_state.get("num_questions", 5)
            _orig_answered_now = len([a for a in st.session_state.session_answers
                                      if not a.get("is_follow_up", False)])
            _show_finish = _orig_answered_now >= _num_qs_now or qn >= total

            st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)

            if _show_finish:
                def finish_session():
                    _sync_session_for_report()
                    if hasattr(engine, "get_rl_report"):
                        st.session_state["rl_report"] = engine.get_rl_report()
                    st.session_state.page = "Final Report"
                st.markdown("""<style>
.v3-finish-wrap{
  border-radius:14px;
  background:linear-gradient(135deg,#f59e0b,#f97316,#fbbf24,#f59e0b);
  background-size:300% 300%;
  animation:finBorderSweep 3.5s ease infinite;
  padding:2px;margin-top:.6rem;
  box-shadow:0 0 22px rgba(245,158,11,.2);
}
.v3-finish-wrap>button{
  color:#fbbf24!important;background:rgba(10,6,0,.95)!important;
  border:none!important;border-radius:12px!important;
  font-family:'Orbitron',monospace!important;font-size:.78rem!important;font-weight:700!important;
  letter-spacing:.14em!important;width:100%!important;padding:.9rem!important;
  transition:all .2s!important;
}
.v3-finish-wrap>button:hover{color:#fff!important;background:rgba(245,158,11,.15)!important;}
@keyframes finBorderSweep{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
</style><div class="v3-finish-wrap">""", unsafe_allow_html=True)
                st.button("★  FINISH & VIEW REPORT", key="fin_btn",
                          on_click=finish_session, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                def next_question():
                    _sess_diff = st.session_state.get("difficulty", "medium").lower()
                    _rl_mode   = "all" in _sess_diff
                    _is_followup = (
                        _rl_mode
                        and hasattr(engine, "is_follow_up_pending")
                        and engine.is_follow_up_pending()
                    )
                    if _is_followup:
                        st.session_state.update({
                            "question":       engine.get_next_question(),
                            "q_number":       qn,
                            "q_is_follow_up": True,
                            "last_score": None, "last_feedback": "", "last_star": {},
                            "last_keywords": [], "last_eval": {}, "submitted": False,
                            "q_start_time": time.time(), "transcribed_text": "",
                            "_pending_answer": "", "last_coaching_tip": "",
                            "last_improvement": "",
                            "pending_follow_up": st.session_state.get("last_eval", {}),
                            "last_audio_bytes": None, "last_audio_source": None,
                        })
                        for _k in [f"_bstt_last_tx_{qn}", f"transcribed_text_browser_{qn}",
                                   "_bstt_last_audio_bytes"]:
                            st.session_state.pop(_k, None)
                    else:
                        st.session_state.update({
                            "question":       engine.get_next_question(),
                            "q_number":       qn + 1,
                            "q_is_follow_up": False,
                            "last_score": None, "last_feedback": "", "last_star": {},
                            "last_keywords": [], "last_eval": {}, "submitted": False,
                            "q_start_time": time.time(), "transcribed_text": "",
                            "_pending_answer": "", "last_coaching_tip": "",
                            "last_improvement": "", "pending_follow_up": None,
                            "last_audio_bytes": None, "last_audio_source": None,
                        })
                        for _k in [f"_bstt_last_tx_{qn}", f"transcribed_text_browser_{qn}",
                                   "_bstt_last_audio_bytes"]:
                            st.session_state.pop(_k, None)

                _fu_lbl = (
                    "all" in st.session_state.get("difficulty", "").lower()
                    and hasattr(engine, "is_follow_up_pending")
                    and engine.is_follow_up_pending()
                )
                _btn_label = "▶ Answer Follow-up" if _fu_lbl else "▶ Next Question"
                st.markdown("""<style>
.v3-next-wrap{
  border-radius:14px;
  background:linear-gradient(135deg,#00ff88,#00d4c8,#3b8bff,#00ff88);
  background-size:300% 300%;
  animation:nextBorderSweep 4s ease infinite;
  padding:2px;margin-top:.6rem;
  box-shadow:0 0 20px rgba(0,255,136,.14);
}
.v3-next-wrap>button{
  color:#00ff88!important;background:rgba(0,10,22,.95)!important;
  border:none!important;border-radius:12px!important;
  font-family:'Orbitron',monospace!important;font-size:.78rem!important;font-weight:700!important;
  letter-spacing:.12em!important;width:100%!important;padding:.9rem!important;
  transition:all .2s!important;
}
.v3-next-wrap>button:hover{color:#ffffff!important;background:rgba(0,255,136,.1)!important;letter-spacing:.16em!important;}
@keyframes nextBorderSweep{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
</style><div class="v3-next-wrap">""", unsafe_allow_html=True)
                st.button(_btn_label, key=f"nxt_{qn}", on_click=next_question,
                          use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)




# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  COMPETENCY MAPPING (v13.1)
# ══════════════════════════════════════════════════════════════════════════════
# Maps DISC dominant trait + STAR completion rate + hiring signal into 6
# standard interview competency scores on a 0-100 scale.
#
# Competency framework (Bartram 2005, SHL Great Eight; Spencer & Spencer 1993,
# Competency at Work):
#   Leadership         — initiative, influence, directing others
#   Communication      — clarity, listening, persuasion
#   Problem-solving    — analysis, creativity, decision quality
#   Adaptability       — resilience, flexibility, learning agility
#   Collaboration      — teamwork, empathy, relationship building
#   Delivery           — results orientation, planning, follow-through
#
# Scoring formula:
#   Each competency starts at a base derived from hiring_signal (0-100).
#   DISC dominant trait adds a profile-specific boost (+0 to +20 pts) because
#   the DISC profile is empirically correlated with competency expression
#   (Marston 1928; DISC validation: Bess & Harvey 2002, J Appl Psych).
#   STAR completion rate adds up to +15 pts — structured storytelling is a
#   validated behavioural predictor (Schmidt & Hunter 1998, meta-analysis).
#   Final score is clipped to [0, 100] and classified:
#     ≥70 → green  (Strong)
#     50-69 → amber (Developing)
#     <50  → red   (Needs work)
# ─────────────────────────────────────────────────────────────────────────────

_COMPETENCY_DISC_BOOSTS: dict = {
    # disc_dominant → {competency: boost_points}
    "Dominance": {
        "Leadership": 20, "Communication": 5,  "Problem-solving": 10,
        "Adaptability": 5, "Collaboration": 0,  "Delivery": 15,
    },
    "Influence": {
        "Leadership": 10, "Communication": 20, "Problem-solving": 5,
        "Adaptability": 10, "Collaboration": 15, "Delivery": 5,
    },
    "Steadiness": {
        "Leadership": 5,  "Communication": 10, "Problem-solving": 5,
        "Adaptability": 5, "Collaboration": 20, "Delivery": 15,
    },
    "Conscientiousness": {
        "Leadership": 5,  "Communication": 5,  "Problem-solving": 20,
        "Adaptability": 5, "Collaboration": 10, "Delivery": 15,
    },
}

_COMPETENCY_NAMES = [
    "Leadership", "Communication", "Problem-solving",
    "Adaptability", "Collaboration", "Delivery",
]


def _compute_competency_grid(
    disc_dominant:   str,
    star_rate:       float,   # 0-1 (avg STAR completeness across answers)
    hiring_signal:   float,   # 1-5 scale
    knowledge_score: float,   # 1-5 scale
    nervousness:     float,   # 0-1 (lower = calmer = better)
) -> list:
    """
    Compute 6 competency scores and traffic-light ratings.

    Returns list of dicts:
      [{"name": str, "score": int (0-100), "level": "Strong"|"Developing"|"Needs work",
        "color": "#hex", "rationale": str}, ...]

    Sorted weakest-first so the practice question generator can use index 0.
    """
    base = round((hiring_signal - 1) / 4.0 * 100)   # map 1-5 → 0-100
    star_bonus = round(star_rate * 15)                # 0-15 pts from STAR rate
    calm_bonus = round((1 - nervousness) * 10)        # 0-10 pts from calmness

    boosts = _COMPETENCY_DISC_BOOSTS.get(disc_dominant,
                                          {c: 0 for c in _COMPETENCY_NAMES})

    # Competency-specific knowledge weighting
    kn_pct = round((knowledge_score - 1) / 4.0 * 100)
    kn_weights = {
        "Leadership":      0.20, "Communication":  0.15, "Problem-solving": 0.35,
        "Adaptability":    0.20, "Collaboration":  0.10, "Delivery":        0.30,
    }

    results = []
    for comp in _COMPETENCY_NAMES:
        boost   = boosts.get(comp, 0)
        kn_part = round(kn_pct * kn_weights[comp])
        score   = min(100, max(0, base + boost + star_bonus + calm_bonus + kn_part))

        if score >= 70:
            level  = "Strong"
            color  = "#22d87a"
            bg     = "rgba(34,216,122,.08)"
            border = "rgba(34,216,122,.25)"
        elif score >= 50:
            level  = "Developing"
            color  = "#ffb840"
            bg     = "rgba(255,184,64,.08)"
            border = "rgba(255,184,64,.25)"
        else:
            level  = "Needs work"
            color  = "#ff5c5c"
            bg     = "rgba(255,92,92,.08)"
            border = "rgba(255,92,92,.25)"

        # One-line rationale for the hiring manager
        rationale_map = {
            "Leadership": (
                f"DISC {disc_dominant} profile + "
                f"{'strong' if star_bonus >= 10 else 'partial'} structured storytelling"
            ),
            "Communication": (
                f"Hiring signal {hiring_signal:.1f}/5 + "
                f"calmness score {round((1-nervousness)*100)}%"
            ),
            "Problem-solving": (
                f"Knowledge depth {knowledge_score:.1f}/5 + "
                f"DISC {disc_dominant} analytical pattern"
            ),
            "Adaptability": (
                f"Nervousness trend {round(nervousness*100)}% + "
                f"DISC {disc_dominant} flexibility indicator"
            ),
            "Collaboration": (
                f"DISC {disc_dominant} interpersonal style + "
                f"STAR completion {round(star_rate*100)}%"
            ),
            "Delivery": (
                f"STAR completion {round(star_rate*100)}% + "
                f"knowledge score {knowledge_score:.1f}/5"
            ),
        }
        results.append({
            "name":      comp,
            "score":     score,
            "level":     level,
            "color":     color,
            "bg":        bg,
            "border":    border,
            "rationale": rationale_map.get(comp, ""),
        })

    # Sort weakest first — index 0 = lowest competency (used by practice Qs)
    results.sort(key=lambda x: x["score"])
    return results


def _generate_practice_questions(
    weakest_competency: str,
    role:               str,
    groq_api_key:       str,
) -> list:
    """
    Use Groq to generate 3 targeted practice questions for the weakest competency.

    Returns list of 3 question strings, or [] on failure.
    Called once per report render — result cached in st.session_state to
    avoid repeated API calls on re-renders.
    """
    if not groq_api_key:
        return []

    cache_key = f"_practice_qs_{weakest_competency}_{role}"
    if st.session_state.get(cache_key):
        return st.session_state[cache_key]

    comp_context = {
        "Leadership":      "motivating teams, making decisions under pressure, taking ownership",
        "Communication":   "presenting ideas clearly, active listening, giving/receiving feedback",
        "Problem-solving": "debugging complex issues, trade-off analysis, root cause analysis",
        "Adaptability":    "handling change, learning new skills quickly, recovering from setbacks",
        "Collaboration":   "cross-functional teamwork, conflict resolution, building relationships",
        "Delivery":        "meeting deadlines, prioritisation, managing quality under constraints",
    }.get(weakest_competency, weakest_competency.lower())

    prompt = (
        f"You are an expert interview coach.\n\n"
        f"Role: {role}\n"
        f"Weakest competency: {weakest_competency} ({comp_context})\n\n"
        f"Generate exactly 3 targeted interview practice questions that specifically "
        f"address this competency gap for this role. Each question must:\n"
        f"  1. Be a realistic interview question a hiring manager would ask\n"
        f"  2. Require a structured STAR-format answer\n"
        f"  3. Be specific to the {weakest_competency} competency\n"
        f"  4. Be progressively more challenging (easy to medium to hard)\n\n"
        f"Return ONLY a JSON array of exactly 3 strings — the questions themselves.\n"
        f"No preamble, no numbering, no explanation. Example format:\n"
        f'["Question 1", "Question 2", "Question 3"]'
    )

    try:
        import os as _os, json as _json
        from groq import Groq as _Groq
        client = _Groq(api_key=groq_api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.5,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = raw.strip("```json").strip("```").strip()
        qs  = _json.loads(raw)
        if isinstance(qs, list) and len(qs) >= 3:
            result = [str(q) for q in qs[:3]]
            st.session_state[cache_key] = result
            return result
    except Exception as _e:
        pass  # silently fall back to static questions

    # Static fallback — always provides something useful
    fallback = {
        "Leadership":      [
            "Tell me about a time you stepped up to lead a team through a difficult situation.",
            "Describe a decision you made that was unpopular but you believed was right.",
            "Tell me about a time you had to influence people without having formal authority.",
        ],
        "Communication":   [
            "Tell me about a time you had to explain a complex technical concept to a non-technical audience.",
            "Describe a situation where miscommunication caused a problem and how you resolved it.",
            "Tell me about a presentation or proposal you delivered that changed someone's mind.",
        ],
        "Problem-solving": [
            "Tell me about the most complex problem you have had to solve at work.",
            "Describe a time you identified a flaw in a system or process and fixed it.",
            "Tell me about a situation where you had to make a decision with incomplete information.",
        ],
        "Adaptability":    [
            "Tell me about a time you had to rapidly learn a new technology or skill.",
            "Describe a situation where your original plan failed and how you adapted.",
            "Tell me about the biggest change you have had to manage in your career.",
        ],
        "Collaboration":   [
            "Tell me about a time you resolved a conflict within your team.",
            "Describe a project where you had to work closely with someone whose style was very different from yours.",
            "Tell me about a cross-functional project and your specific contribution to the team's success.",
        ],
        "Delivery":        [
            "Tell me about a time you delivered a project under significant time pressure.",
            "Describe a situation where you had to juggle multiple high-priority tasks simultaneously.",
            "Tell me about a time you missed a deadline and what you did about it.",
        ],
    }
    result = fallback.get(weakest_competency, fallback["Problem-solving"])
    st.session_state[cache_key] = result
    return result


def page_report() -> None:
    # ── Blind score reveal (runs once on first report visit) ──────────────────
    if (st.session_state.get("blind_mode")
            and not st.session_state.get("blind_revealed")
            and st.session_state.get("blind_scores")):

        blind_scores = st.session_state["blind_scores"]
        answers      = st.session_state.get("session_answers", [])

        for ans in answers:
            key = str(ans.get("number", ""))
            if key in blind_scores:
                sealed_ev = blind_scores[key]
                ans["score"]            = sealed_ev.get("score",            ans["score"])
                ans["feedback"]         = sealed_ev.get("feedback",         ans.get("feedback", ""))
                ans["star"]             = sealed_ev.get("star_scores",       ans.get("star", {}))
                ans["keyword_hits"]     = sealed_ev.get("keyword_hits",      [])
                ans["depth_score"]      = sealed_ev.get("depth_score",       ans.get("depth_score", 0))
                ans["fluency_score"]    = sealed_ev.get("fluency_score",     ans.get("fluency_score", 3.5))
                ans["hiring_signal"]    = sealed_ev.get("hiring_signal",     ans.get("hiring_signal", 2.5))
                ans["ocean_scores"]     = sealed_ev.get("ocean_scores",      {})
                ans["hiring_breakdown"] = sealed_ev.get("hiring_breakdown",  {})

        st.session_state["session_answers"] = answers
        st.session_state["blind_revealed"]  = True

        st.markdown("""
<div style="
  background:rgba(192,132,252,.08);
  border:1px solid rgba(192,132,252,.35);
  border-radius:12px;padding:1rem 1.4rem;
  margin-bottom:1.2rem;
  display:flex;align-items:center;gap:1rem;
">
  <div style="font-size:1.4rem;flex-shrink:0;">&#128275;</div>
  <div>
    <div style="font-size:.82rem;font-weight:600;color:#c084fc;
      font-family:Share Tech Mono,monospace;letter-spacing:.08em;">
      BLIND SESSION UNLOCKED</div>
    <div style="font-size:.78rem;color:rgba(200,216,240,.75);
      font-family:Inter,sans-serif;margin-top:3px;">
      All scores were sealed during the interview to remove recency bias.
      Results are now revealed for the first time.</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style="text-align:center;margin-bottom:2rem;">
  <div style="font-size:.78rem;font-weight:500;color:#818cf8;letter-spacing:.06em;
    text-transform:uppercase;margin-bottom:.5rem;">Session Complete</div>
  <h2 style="margin:0;font-size:1.8rem;font-weight:700;color:#f8fafc;">
    Your Interview Review
  </h2>
  <p style="color:#b4cde4;font-size:.92rem;margin:.4rem 0 0;">
    Full performance analysis — download your PDF report below.
  </p>
</div>""", unsafe_allow_html=True)

    answers = st.session_state.session_answers
    if not answers:
        # Empty state — no session data yet
        st.markdown("""
<div style="background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);
  border-radius:16px;padding:3rem;text-align:center;margin:2rem 0;">
  <div style="font-size:2.5rem;margin-bottom:1rem;">📋</div>
  <div style="font-size:1.1rem;font-weight:600;color:#f1f5f9;margin-bottom:.6rem;">
    No session data yet</div>
  <div style="color:#b4cde4;font-size:.88rem;max-width:380px;margin:0 auto 1.5rem;line-height:1.7;">
    Complete an interview first. Your full review will appear here.
  </div>
</div>""", unsafe_allow_html=True)
        st.button("Start your first interview", key="rp_empty_start",
                  on_click=nav_to("Start Interview"))
        return

    if answers:
        report = engine.final_report()
        sc     = report.get("final_scores", {}) if report else {}
        mm     = report.get("multimodal", {})    if report else {}
        em_fb  = report.get("emotion_feedback", {}) if report else {}
        conf   = report.get("confidence_score", 3.5) if report else 3.5
        if not sc:
            kn = sum(a["score"] for a in answers) / len(answers)
            # v10.0: nervousness = 100% voice — "nervousness" key already stores voice value
            avg_nerv  = sum(a.get("nervousness", 0.2) for a in answers) / len(answers)
            avg_vnerv = avg_nerv   # same source — voice only
            # Build real scores from session data — invert nervousness to get calm score
            sc = {
                "final":     round(kn, 2),
                "knowledge": round(kn, 2),
                "emotion":   round((1 - avg_nerv)  * 5, 2),   # voice nervousness → calm 1-5
                "voice":     round((1 - avg_vnerv) * 5, 2),   # voice nervousness → calm 1-5
                "depth":     round(sum(a.get("depth_score", a.get("score", 2.5)) for a in answers) / len(answers), 2),
                "fluency":   round(sum(a.get("fluency", a.get("fluency_score", 3.5)) for a in answers) / len(answers), 2),
            }

    bc, bt = badge(sc.get("final", 3.5))

    # ── Feature 9: Interview Replay Summary Cards ─────────────────────────────
    _primary_ans = [a for a in answers if not a.get("is_follow_up", False)]
    if _primary_ans:
        # Build one card per question as a scrollable horizontal strip
        _replay_cards = ""
        for _ri, _ra in enumerate(_primary_ans):
            _rs      = _ra.get("score", 0)
            _rfs     = _ra.get("final_score", round(_rs/5*100))
            _rbc, _rbt = badge(_rs)
            _badge_colours = {
                "b-ex": ("#34d399","rgba(16,185,129,.2)","rgba(16,185,129,.45)"),
                "b-gd": ("#c4b5fd","rgba(99,102,241,.2)","rgba(99,102,241,.45)"),
                "b-av": ("#fcd34d","rgba(245,158,11,.2)","rgba(245,158,11,.45)"),
                "b-po": ("#fca5a5","rgba(239,68,68,.2)","rgba(239,68,68,.45)"),
            }
            _btc, _btbg, _btbd = _badge_colours.get(_rbc, ("#94a3b8","transparent","rgba(255,255,255,.1)"))
            _rem   = emo_css(_ra.get("emotion","Neutral"))
            _rvm   = emo_css(_ra.get("voice_emotion","Neutral"))
            _rnerv = int(_ra.get("nervousness", 0.2) * 100)
            _rnc   = nerv_css(_ra.get("nervousness", 0.2))
            _rwc   = _ra.get("word_count", 0)
            _rqt   = _ra.get("type", "")
            _rqshort = (_ra.get("question","")[:55] + "…") if len(_ra.get("question","")) > 55 else _ra.get("question","")
            _r_border_top = {
                "b-ex": "#10b981", "b-gd": "#6366f1",
                "b-av": "#f59e0b", "b-po": "#ef4444"
            }.get(_rbc, "#6366f1")
            _replay_cards += f"""
<div style="flex:0 0 200px;background:rgba(0,8,22,.72);border:1px solid rgba(255,255,255,.08);
  border-top:2px solid {_r_border_top};border-radius:12px;padding:.85rem .9rem;
  display:flex;flex-direction:column;gap:.4rem;">
  <!-- Q label + type -->
  <div style="display:flex;align-items:center;justify-content:space-between;gap:.3rem;">
    <span style="font-size:.62rem;font-weight:700;color:#a5b4fc;
      font-family:Orbitron,monospace;">Q{_ra.get('number',_ri+1)}</span>
    <span style="font-size:.58rem;color:#475569;font-family:Inter,sans-serif;">{_rqt}</span>
  </div>
  <!-- Question preview -->
  <div style="font-size:.7rem;color:#94a3b8;font-family:Inter,sans-serif;
    line-height:1.5;flex:1;min-height:2.5rem;">{_rqshort}</div>
  <!-- Score badge -->
  <div style="background:{_btbg};border:1px solid {_btbd};border-radius:20px;
    padding:2px 10px;text-align:center;margin:.1rem 0;">
    <span style="font-size:.68rem;font-weight:700;color:{_btc};
      font-family:Inter,sans-serif;">{_rbt} · {_rs:.1f}/5</span>
  </div>
  <!-- Stats row -->
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:3px;">
    <div style="text-align:center;background:rgba(255,255,255,.03);border-radius:4px;padding:3px 2px;">
      <div style="font-size:.65rem;font-weight:700;color:{_rem};
        font-family:Share Tech Mono,monospace;">😐</div>
      <div style="font-size:.48rem;color:#334155;font-family:Inter,sans-serif;">Face</div>
    </div>
    <div style="text-align:center;background:rgba(255,255,255,.03);border-radius:4px;padding:3px 2px;">
      <div style="font-size:.62rem;font-weight:700;color:{_rnc};
        font-family:Share Tech Mono,monospace;">{_rnerv}%</div>
      <div style="font-size:.48rem;color:#334155;font-family:Inter,sans-serif;">Nerv</div>
    </div>
    <div style="text-align:center;background:rgba(255,255,255,.03);border-radius:4px;padding:3px 2px;">
      <div style="font-size:.62rem;font-weight:700;color:#c7e8ff;
        font-family:Share Tech Mono,monospace;">{_rwc}</div>
      <div style="font-size:.48rem;color:#334155;font-family:Inter,sans-serif;">Words</div>
    </div>
  </div>
</div>"""

        st.markdown(f"""
<div style="margin-bottom:1.2rem;">
  <div style="font-size:.62rem;font-weight:700;color:#818cf8;font-family:Share Tech Mono,monospace;
    letter-spacing:.12em;text-transform:uppercase;margin-bottom:.6rem;">◈ Session Replay</div>
  <div style="display:flex;gap:.6rem;overflow-x:auto;padding-bottom:.4rem;
    scrollbar-width:thin;scrollbar-color:rgba(99,102,241,.3) transparent;">
    {_replay_cards}
  </div>
</div>""", unsafe_allow_html=True)

    # v8.0: posture + confidence removed; knowledge_depth added as explicit metric
    _depth_sc = round(
        sum(a.get("depth_score", a.get("score", 2.5)) for a in answers) / max(1, len(answers)), 2
    ) if answers else sc.get("knowledge", 3.5)
    cs_cols = st.columns(5)
    for col, (lbl, val) in zip(cs_cols, [
        ("FINAL",     sc.get("final",     3.5)),
        ("KNOWLEDGE", sc.get("knowledge", 3.5)),
        ("DEPTH",     _depth_sc),
        ("EMOTION",   sc.get("emotion",   3.5)),
        ("VOICE",     sc.get("voice",     3.5)),
    ]):
        mchip(lbl, val, col=col)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if answers:
            st.markdown('<h4 style="margin-top:0;font-size:.85rem;">ANSWER REVIEW</h4>', unsafe_allow_html=True)
            for a in answers:
                bc2, bt2 = badge(a["score"])
                star_ok  = [k for k,v in a.get("star",{}).items() if v]
                star_ms  = [k for k,v in a.get("star",{}).items() if not v]
                mm2, ss2 = divmod(a.get("time_s",0), 60)
                em_c     = emo_css(a.get("emotion","Neutral"))
                ve_c     = emo_css(a.get("voice_emotion","Neutral"))
                cs_v     = conf_css(a.get("confidence_score",3.5))
                # v10.0: only voice nervousness used
                v_nerv   = a.get("nervousness", a.get("voice_nervousness", 0.2))
                vn_c     = nerv_css(v_nerv)
                miss_h   = (f'&nbsp;<span style="font-size:.68rem;color:#ffaa00;font-family:Share Tech Mono,monospace;">MISSING: {", ".join(star_ms)}</span>'
                             if star_ms else "")
                nlp_pct  = a.get("final_score", round(a["score"]/5*100))
                sim_pct  = a.get("similarity_score", 0)
                grm_pct  = a.get("grammar_score", 0)
                # Time efficiency colour coding
                _t_eff   = a.get("time_efficiency", 0.0)
                _t_label = a.get("time_label", "")
                if _t_eff >= 80:   _tc_col, _tc_bg = "#22d87a", "rgba(34,216,122,.08)"
                elif _t_eff >= 60: _tc_col, _tc_bg = "#ffb840", "rgba(255,184,64,.08)"
                elif _t_eff >= 40: _tc_col, _tc_bg = "#ff9040", "rgba(255,144,64,.08)"
                else:              _tc_col, _tc_bg = "#ff5c5c", "rgba(255,92,92,.08)"
                _t_show  = _t_eff > 0 and _t_label not in ("N/A", "No timing data", "")
                st.markdown(f"""
<div class="card card-sm" style="margin:.3rem 0;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.3rem;">
    <span style="font-size:.62rem;font-weight:700;color:#c7e8ff;text-transform:uppercase;
      font-family:Share Tech Mono,monospace;">Q{a['number']} · {a.get('type','')}</span>
    <div style="display:flex;gap:.25rem;align-items:center;flex-wrap:wrap;">
      <span class="emo-pill" style="color:{em_c};border-color:{em_c}30;background:{em_c}10;font-size:.58rem;">{a.get('emotion','Neutral')}</span>
      <span class="emo-pill" style="color:{ve_c};border-color:{ve_c}30;background:{ve_c}10;font-size:.58rem;">▸{a.get('voice_emotion','Neutral')}</span>
      <span style="font-size:.62rem;color:#00d4ff;font-family:Share Tech Mono,monospace;">D:{a.get('depth_score',0):.1f}</span>
      <span class="badge {bc2}">{a['score']:.1f}·{bt2}</span>
      <span style="font-size:.62rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;">⏱{mm2:02d}:{ss2:02d}</span>
      <span style="font-size:.6rem;font-family:Share Tech Mono,monospace;padding:1px 6px;border-radius:3px;
        background:{_tc_bg};color:{_tc_col};border:1px solid {_tc_col}55;">{a.get('time_label','')}</span>
    </div>
  </div>
  <div style="font-size:.85rem;color:#e0f0ff;font-weight:600;margin-bottom:.2rem;font-family:Orbitron,monospace;">{a['question']}</div>
  <div style="font-size:.76rem;color:#c7e8ff;margin-bottom:.25rem;">{a['answer'][:200]}{'…' if len(a['answer'])>200 else ''}</div>
  <div style="display:flex;gap:8px;font-size:.62rem;color:#c7e8ff;margin-bottom:.18rem;font-family:Share Tech Mono,monospace;">
    <span>NLP:<b style="color:#00d4ff;">{nlp_pct:.0f}%</b></span>
    <span>REL:<b style="color:#a855f7;">{sim_pct:.0f}%</b></span>
    <span>GRM:<b style="color:#00ff88;">{grm_pct:.0f}%</b></span>
    {"<span>PACE:<b style='color:"+_tc_col+";'>"+str(round(_t_eff))+"% · "+_t_label+"</b></span>" if _t_show else ""}
  </div>
  <span style="font-size:.68rem;color:#00ff88;font-family:Share Tech Mono,monospace;">✓ {', '.join(star_ok) or 'NO STAR'}</span>{miss_h}
  <div style="display:flex;gap:10px;font-size:.62rem;color:#c7e8ff;margin-top:.18rem;font-family:Share Tech Mono,monospace;">
    <span>VOICE NERV:<b style="color:{vn_c};">{int(v_nerv*100)}%</b></span>
  </div>
  <div style="font-size:.68rem;color:#c7e8ff;margin-top:.12rem;">{a.get('feedback','')}</div>
</div>""", unsafe_allow_html=True)

            if len(answers) >= 2:
                st.markdown("<hr><h4 style='font-size:.85rem;'>SCORE TRAJECTORY</h4>", unsafe_allow_html=True)
                _t_effs = [a.get("time_efficiency", 0.0) for a in answers]
                _has_time = any(e > 0 for e in _t_effs)
                tdf_cols = {
                    "Q":      [f"Q{a['number']}" for a in answers],
                    "Score":  [a["score"] for a in answers],
                    "Depth":  [a.get("depth_score", round(a["score"], 2)) for a in answers],
                    "NLP %":  [a.get("final_score", a["score"]/5*100) for a in answers],
                }
                _clrs = ["#00FFD1","#00D4FF","#7c3aff"]
                if _has_time:
                    tdf_cols["Time Efficiency %"] = _t_effs
                    _clrs.append("#ffb840")
                tdf = pd.DataFrame(tdf_cols)
                _y_cols = ["Score","Depth","NLP %"] + (["Time Efficiency %"] if _has_time else [])
                fig_t = px.line(tdf, x="Q", y=_y_cols, markers=True, color_discrete_sequence=_clrs)
                fig_t.update_traces(line_width=2.5, marker_size=7)
                fig_t.add_hline(y=3.5, line_dash="dot", line_color="rgba(255,170,0,.4)")
                dcl(fig_t, 200); st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar":False})

                # Time efficiency summary stats
                if _has_time:
                    _valid_te = [e for e in _t_effs if e > 0]
                    _avg_te   = round(sum(_valid_te) / len(_valid_te), 1)
                    _best_q   = answers[_t_effs.index(max(_t_effs))]["number"]
                    _worst_q  = answers[_t_effs.index(min(e for e in _t_effs if e > 0))]["number"]
                    _avg_ts   = round(sum(a.get("time_s", 0) for a in answers) / len(answers))
                    _am, _as2 = divmod(_avg_ts, 60)
                    st.markdown(
                        f'<div style="display:flex;gap:.6rem;flex-wrap:wrap;margin-top:.4rem;">' +
                        f'<div style="background:rgba(255,184,64,.06);border:1px solid rgba(255,184,64,.2);' +
                        f'border-radius:8px;padding:.5rem .8rem;font-size:.7rem;font-family:Share Tech Mono,monospace;">' +
                        f'<div style="color:#ffb840;font-weight:700;">AVG TIME EFFICIENCY</div>' +
                        f'<div style="color:#dde8ff;font-size:.95rem;font-weight:800;">{_avg_te:.0f}%</div></div>' +
                        f'<div style="background:rgba(0,255,136,.06);border:1px solid rgba(0,255,136,.2);' +
                        f'border-radius:8px;padding:.5rem .8rem;font-size:.7rem;font-family:Share Tech Mono,monospace;">' +
                        f'<div style="color:#00ff88;font-weight:700;">BEST PACED</div>' +
                        f'<div style="color:#dde8ff;font-size:.95rem;font-weight:800;">Q{_best_q}</div></div>' +
                        f'<div style="background:rgba(255,92,92,.06);border:1px solid rgba(255,92,92,.2);' +
                        f'border-radius:8px;padding:.5rem .8rem;font-size:.7rem;font-family:Share Tech Mono,monospace;">' +
                        f'<div style="color:#ff5c5c;font-weight:700;">NEEDS PACING</div>' +
                        f'<div style="color:#dde8ff;font-size:.95rem;font-weight:800;">Q{_worst_q}</div></div>' +
                        f'<div style="background:rgba(0,212,255,.06);border:1px solid rgba(0,212,255,.2);' +
                        f'border-radius:8px;padding:.5rem .8rem;font-size:.7rem;font-family:Share Tech Mono,monospace;">' +
                        f'<div style="color:#00d4ff;font-weight:700;">AVG ANSWER TIME</div>' +
                        f'<div style="color:#dde8ff;font-size:.95rem;font-weight:800;">{_am}m {_as2:02d}s</div></div>' +
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown('<h4>NO ANSWERS RECORDED — COMPLETE AN INTERVIEW FIRST</h4>', unsafe_allow_html=True)

        if em_fb:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h4 style="margin-top:0;font-size:.85rem;">▸ MULTIMODAL EMOTION FEEDBACK</h4>', unsafe_allow_html=True)
            nerv_c = nerv_css(em_fb.get("nervousness_score", 20) / 100)
            st.markdown(f'<div style="font-size:.95rem;font-weight:700;color:{nerv_c};font-family:Orbitron,monospace;">{em_fb.get("nervousness_level","?")} NERVOUSNESS · {em_fb.get("nervousness_score","?")}%</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="b-info">{em_fb.get("overall_feedback","")}</div>', unsafe_allow_html=True)
            c_s, c_i = st.columns(2)
            with c_s:
                st.markdown("<b style='color:#00ff88'>✓ STRENGTHS</b>", unsafe_allow_html=True)
                for s in em_fb.get("strengths",[]): st.markdown(f'<div style="font-size:.78rem;color:#c7e8ff;">▸ {s}</div>', unsafe_allow_html=True)
            with c_i:
                st.markdown("<b style='color:#ffaa00'>▲ IMPROVEMENTS</b>", unsafe_allow_html=True)
                for i in em_fb.get("improvements",[]): st.markdown(f'<div style="font-size:.78rem;color:#c7e8ff;">▸ {i}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        fc = conf_css(sc.get("final",3.5))
        st.markdown(f"""
<div class="card" style="text-align:center;">
  <h3 style="margin-top:0;font-size:.8rem;color:#c7e8ff;">FINAL GRADE</h3>
  <div style="font-size:3.8rem;font-family:Orbitron,monospace;font-weight:900;
    color:{fc};text-shadow:0 0 30px {fc}66;line-height:1.05;">{sc.get('final',3.5):.2f}</div>
  <div style="color:#c7e8ff;font-size:.72rem;font-family:Share Tech Mono,monospace;">/5.0</div>
  <span class="badge {bc}" style="font-size:.85rem;">{bt}</span>
  <hr>
  <div class="sec-lbl">KNOWLEDGE DEPTH</div>
  <div style="font-size:1.9rem;font-weight:900;font-family:Orbitron,monospace;color:{conf_css(_depth_sc)};
    text-shadow:0 0 20px {conf_css(_depth_sc)}55;">{_depth_sc:.2f}</div>
  <div style="font-size:.6rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;">AVG DEPTH ACROSS ALL ANSWERS</div>
</div>""", unsafe_allow_html=True)

        # v8.0: SIGNAL BREAKDOWN now shows knowledge depth components
        if answers:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h4 style="margin-top:0;font-size:.85rem;">⬡ SIGNAL BREAKDOWN</h4>', unsafe_allow_html=True)
            _avg_nlp  = round(sum(a.get("final_score", a["score"]/5*100) for a in answers) / len(answers), 1)
            _avg_rel  = round(sum(a.get("similarity_score", 0) for a in answers) / len(answers), 1)
            _avg_grm  = round(sum(a.get("grammar_score", 0) for a in answers) / len(answers), 1)
            _avg_star = round(sum(len([k for k,v in a.get("star",{}).items() if v]) / 4 * 100 for a in answers) / len(answers), 1)
            _avg_time = round(sum(a.get("time_efficiency", 0) for a in answers) / len(answers), 1)
            coach_bar("NLP QUALITY",   _avg_nlp,  "#00D4FF")
            coach_bar("RELEVANCE",     _avg_rel,  "#a855f7")
            coach_bar("DEPTH",         _depth_sc / 5 * 100, "#00FFD1")
            coach_bar("GRAMMAR",       _avg_grm,  "#FFD700")
            if _avg_time > 0:
                coach_bar("PACING",    _avg_time, "#ff7f50")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── DISC Profile — real data from session_answers ────────────────────
        # answer_evaluator.py computes disc_traits per answer and stores them in
        # session_answers[n]["disc_traits"] as {"Dominance":n,"Influence":n,...}
        _disc_raw = {"Dominance": 0.0, "Influence": 0.0,
                     "Steadiness": 0.0, "Conscientiousness": 0.0}
        _disc_n = 0
        for _a in answers:
            _dt = _a.get("disc_traits", {})
            if _dt and any(_dt.values()):
                for k in _disc_raw:
                    _disc_raw[k] += float(_dt.get(k, 0))
                _disc_n += 1

        if _disc_n > 0:
            disc = {k: round(v / _disc_n, 1) for k, v in _disc_raw.items()}
            _disc_source = f"Based on {_disc_n} scored answer{'s' if _disc_n>1 else ''}"
        else:
            # Genuine fallback only when no disc_traits data exists at all
            disc = {"Dominance": 0, "Influence": 0, "Steadiness": 0, "Conscientiousness": 0}
            _disc_source = "Complete an interview to see your DISC profile"

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top:0;font-size:.85rem;">DISC profile</h4>', unsafe_allow_html=True)
        if _disc_n > 0:
            _disc_max  = max(disc.values()) if disc.values() else 5
            _r_axis_max = max(10, _disc_max * 1.2)
            _disc_labels = list(disc.keys())
            _disc_vals   = list(disc.values())

            # ── Build spin animation via Plotly frames ────────────────────────
            # We generate 24 intermediate frames that rotate the angular axis
            # from 0° → 360°, then snap back to the settled state.
            # Plotly polar `rotation` (angularaxis.rotation) spins the whole chart.
            _N_FRAMES   = 24
            _DURATION_MS = 25   # ms per frame → ~600 ms total spin
            _polar_base = dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, range=[0, _r_axis_max],
                    gridcolor="rgba(0,212,255,.06)",
                    tickfont=dict(color="#5a7a9a", size=7),
                    linecolor="rgba(0,212,255,.08)",
                ),
                angularaxis=dict(
                    gridcolor="rgba(0,212,255,.05)",
                    tickfont=dict(color="#4a9eff", size=9, family="Orbitron"),
                    direction="clockwise",
                ),
            )

            _trace = go.Scatterpolar(
                r=_disc_vals + [_disc_vals[0]],
                theta=_disc_labels + [_disc_labels[0]],
                fill="toself",
                fillcolor="rgba(0,212,255,.06)",
                line=dict(color="#00D4FF", width=2),
                marker=dict(size=6, color="#00D4FF"),
            )

            # Frames: rotate angularaxis 0° → 360° over _N_FRAMES steps
            _frames = []
            for _fi in range(_N_FRAMES):
                _rot = int(_fi * 360 / _N_FRAMES)
                _polar_frame = dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(
                        visible=True, range=[0, _r_axis_max],
                        gridcolor="rgba(0,212,255,.06)",
                        tickfont=dict(color="#5a7a9a", size=7),
                    ),
                    angularaxis=dict(
                        rotation=_rot,
                        gridcolor="rgba(0,212,255,.05)",
                        tickfont=dict(color="#4a9eff", size=9, family="Orbitron"),
                        direction="clockwise",
                    ),
                )
                _frames.append(go.Frame(
                    layout=dict(polar=_polar_frame),
                    name=str(_fi),
                ))

            fig_d = go.Figure(data=[_trace], frames=_frames)
            fig_d.update_layout(
                polar=_polar_base,
                showlegend=False,
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    visible=False,   # hidden — triggered via JS below
                    buttons=[dict(
                        label="spin",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=_DURATION_MS, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=False,
                            mode="immediate",
                        )],
                    )],
                )],
            )
            dcl(fig_d, 220)

            # Unique element id keyed to spin version — React re-mounts this div
            # on every new answer, which fires the JS animation trigger fresh.
            _spin_ver = st.session_state.get("_disc_spin_version", 0)
            _chart_key = f"disc_radar_v{_spin_ver}"
            st.plotly_chart(fig_d, use_container_width=True,
                            config={"displayModeBar": False},
                            key=_chart_key)

            # JS: click Plotly's hidden animate button ~80 ms after mount so
            # the chart is fully rendered before we trigger the spin.
            # We gate on _spin_ver > 0 so the radar only spins after a real eval,
            # not on first render of an empty chart.
            if _spin_ver > 0:
                components.html(f"""<script>
(function(){{
  var ver = {_spin_ver};
  function triggerSpin() {{
    var PARENT = window.parent.document;
    // Find Plotly chart containers — look for the one containing the DISC polar trace
    var charts = PARENT.querySelectorAll('.js-plotly-plot');
    for (var i = 0; i < charts.length; i++) {{
      var chart = charts[i];
      if (!chart._fullLayout || !chart._fullLayout.polar) continue;
      // Confirm it's the DISC radar (4 angular categories)
      var polar = chart._fullLayout.polar;
      if (!polar.angularaxis) continue;
      // Stamp with version to avoid double-firing on same chart
      if (chart._discSpinVer === ver) continue;
      chart._discSpinVer = ver;
      // Trigger Plotly.animate for the 360° spin frames
      Plotly.animate(chart,
        null,   // null = play all frames
        {{
          frame: {{ duration: {_DURATION_MS}, redraw: true }},
          transition: {{ duration: 0 }},
          mode: 'immediate',
        }}
      );
      break;
    }}
  }}
  // Retry loop: chart may not be mounted yet when script runs
  var attempts = 0;
  var tid = setInterval(function() {{
    attempts++;
    var PARENT = window.parent.document;
    var charts = PARENT.querySelectorAll('.js-plotly-plot');
    var found = false;
    for (var i = 0; i < charts.length; i++) {{
      if (charts[i]._fullLayout && charts[i]._fullLayout.polar) {{ found = true; break; }}
    }}
    if (found || attempts > 20) {{
      clearInterval(tid);
      if (found) triggerSpin();
    }}
  }}, 80);
}})();
</script>""", height=0)

            # Dominant trait callout
            _dom_trait = max(disc, key=disc.get)
            _dom_desc = {
                "Dominance":         "Results-driven — you emphasise outcomes and direct action.",
                "Influence":         "People-focused — you communicate with energy and enthusiasm.",
                "Steadiness":        "Process-oriented — you value reliability and methodical work.",
                "Conscientiousness": "Detail-oriented — you prioritise accuracy and quality.",
            }.get(_dom_trait, "")
            st.markdown(
                f'<div style="font-size:.72rem;color:#00d4ff;font-family:Share Tech Mono,monospace;'
                f'margin-top:.3rem;"><b>Dominant: {_dom_trait}</b> — {_dom_desc}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="text-align:center;padding:1.5rem;color:#c7e8ff;'
                f'font-family:Share Tech Mono,monospace;font-size:.75rem;">{_disc_source}</div>',
                unsafe_allow_html=True
            )
        st.markdown(
            f'<div style="font-size:.58rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;'
            f'margin-top:.2rem;">{_disc_source}</div>',
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)



    # ══════════════════════════════════════════════════════════════════════════
    #  BEHAVIOURAL COMPETENCY GRID (v13.1)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<div style="margin-bottom:.8rem;">
  <div style="font-size:.62rem;font-family:Share Tech Mono,monospace;color:#00ff88;
    letter-spacing:.2em;margin-bottom:.2rem;">// BEHAVIOURAL COMPETENCY FRAMEWORK</div>
  <h3 style="margin:0;font-size:1.1rem;">COMPETENCY MAPPING</h3>
  <div style="font-size:.7rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;margin-top:.2rem;">
    Bartram 2005 SHL Great Eight · Spencer &amp; Spencer 1993 · DISC × STAR × Hiring Signal
  </div>
</div>""", unsafe_allow_html=True)

    # ── Gather inputs for competency grid ─────────────────────────────────────
    _disc_dominant   = max(_disc_raw, key=_disc_raw.get) if _disc_n > 0 else "Conscientiousness"
    _avg_hiring      = (sum(a.get("hiring_signal", 2.5) for a in answers)
                       / max(1, len(answers)))
    _avg_star_rate   = (sum(sum(a.get("star", {}).values()) / 4.0
                           for a in answers) / max(1, len(answers)))
    _avg_nerv        = (sum(a.get("nervousness", 0.2) for a in answers)
                       / max(1, len(answers)))
    _know_sc         = sc.get("knowledge", 3.5)
    _role_str        = st.session_state.get("target_role", "Software Engineer")

    _competencies    = _compute_competency_grid(
        disc_dominant   = _disc_dominant,
        star_rate       = _avg_star_rate,
        hiring_signal   = _avg_hiring,
        knowledge_score = _know_sc,
        nervousness     = _avg_nerv,
    )

    # ── Traffic-light grid (3 columns × 2 rows) ───────────────────────────────
    _cg_cols = st.columns(3)
    for _ci, _comp in enumerate(_competencies):
        with _cg_cols[_ci % 3]:
            # Bar width as percentage of max (100)
            _bar_w = _comp["score"]
            st.markdown(f"""
<div style="background:var(--color-background-secondary);border:1px solid {_comp['border']};
  border-left:3px solid {_comp['color']};border-radius:6px;
  padding:.65rem .85rem;margin:.25rem 0;min-height:88px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.3rem;">
    <span style="font-size:.72rem;font-weight:700;color:{_comp['color']};
      font-family:Share Tech Mono,monospace;letter-spacing:.04em;">{_comp['name'].upper()}</span>
    <span style="font-size:.68rem;font-weight:700;color:{_comp['color']};
      font-family:Orbitron,monospace;">{_comp['score']}</span>
  </div>
  <div style="height:4px;background:rgba(255,255,255,.06);border-radius:2px;margin-bottom:.4rem;">
    <div style="height:100%;width:{_bar_w}%;background:{_comp['color']};
      border-radius:2px;transition:width .4s;"></div>
  </div>
  <div style="font-size:.58rem;color:#3a6070;font-family:Share Tech Mono,monospace;
    line-height:1.4;">
    <span style="color:{_comp['color']};font-weight:700;">{_comp['level']}</span>
    &nbsp;·&nbsp;{_comp['rationale']}
  </div>
</div>""", unsafe_allow_html=True)

    # ── Legend ────────────────────────────────────────────────────────────────
    st.markdown("""
<div style="display:flex;gap:1.2rem;margin:.5rem 0 .2rem;flex-wrap:wrap;">
  <span style="font-size:.6rem;color:#22d87a;font-family:Share Tech Mono,monospace;">
    ● STRONG ≥70</span>
  <span style="font-size:.6rem;color:#ffb840;font-family:Share Tech Mono,monospace;">
    ● DEVELOPING 50-69</span>
  <span style="font-size:.6rem;color:#ff5c5c;font-family:Share Tech Mono,monospace;">
    ● NEEDS WORK &lt;50</span>
  <span style="font-size:.6rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;margin-left:auto;">
    Inputs: DISC profile · STAR completion · Hiring signal · Knowledge depth · Calmness
  </span>
</div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  PRACTICE QUESTIONS FOR WEAKEST COMPETENCY (v13.1)
    # ══════════════════════════════════════════════════════════════════════════
    _weakest     = _competencies[0]   # sorted weakest-first in _compute_competency_grid
    _weakest_name = _weakest["name"]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
<div style="margin-bottom:.8rem;">
  <div style="font-size:.62rem;font-family:Share Tech Mono,monospace;color:{_weakest['color']};
    letter-spacing:.2em;margin-bottom:.2rem;">// TARGETED PRACTICE</div>
  <h3 style="margin:0;font-size:1.1rem;">PRACTICE QUESTIONS</h3>
  <div style="font-size:.7rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;margin-top:.2rem;">
    Your <span style="color:{_weakest['color']};font-weight:700;">{_weakest_name}</span>
    score was low ({_weakest['score']}/100) — here are 3 questions to practise:
  </div>
</div>""", unsafe_allow_html=True)

    import os as _os_pq
    _groq_key = _os_pq.environ.get(
        "GROQ_API_KEY",
        os.environ.get("GROQ_API_KEY", "")
    )

    with st.spinner(f"Generating {_weakest_name} practice questions…"):
        _pqs = _generate_practice_questions(
            weakest_competency = _weakest_name,
            role               = _role_str,
            groq_api_key       = _groq_key,
        )

    _diff_labels = ["Starter", "Intermediate", "Advanced"]
    _diff_colors = ["#22d87a",  "#ffb840",       "#ff5c5c"]

    for _pi, _pq in enumerate(_pqs):
        _dlabel = _diff_labels[_pi] if _pi < len(_diff_labels) else ""
        _dcolor = _diff_colors[_pi] if _pi < len(_diff_colors) else "#5a7a9a"
        st.markdown(f"""
<div style="background:var(--color-background-secondary);
  border:1px solid rgba(255,255,255,.06);border-left:3px solid {_dcolor};
  border-radius:6px;padding:.7rem 1rem;margin:.35rem 0;">
  <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.3rem;">
    <span style="font-size:.58rem;padding:1px 7px;border-radius:3px;font-weight:700;
      color:{_dcolor};border:1px solid {_dcolor}40;
      font-family:Share Tech Mono,monospace;letter-spacing:.06em;">
      {_dlabel.upper()}
    </span>
    <span style="font-size:.58rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;">
      {_weakest_name.upper()} · {_role_str.upper()}
    </span>
  </div>
  <div style="font-size:.82rem;color:#b0cce8;font-family:Rajdhani,sans-serif;
    line-height:1.65;">{_pq}</div>
</div>""", unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-size:.6rem;color:#c7e8ff;font-family:Share Tech Mono,'
        f'monospace;margin:.4rem 0 1rem;">◈ Questions tailored to {_role_str} · '
        f'{_weakest_name} competency · Groq LLaMA-3.3-70B</div>',
        unsafe_allow_html=True,
    )

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.button("⬡  DASHBOARD", key="rp_db", on_click=nav_to("Dashboard"))
    with rc2:
        st.button("▶  NEW SESSION", key="rp_ni", on_click=nav_to("Start Interview"))
    with rc3:
        _sync_session_for_report()
        from finish_interview import _collect_session_data
        _pdf_data  = _collect_session_data(st.session_state)
        _pdf_bytes = _build_pdf(_pdf_data)
        _candidate = st.session_state.get("candidate_name", "Candidate").replace(" ", "_")
        st.download_button(
            "◈ DOWNLOAD PDF REPORT",
            data=_pdf_bytes,
            file_name=f"aura_report_{_candidate}.pdf",
            mime="application/pdf",
        )

    # ── v8.0: RL Sequencer Session Report ─────────────────────────────────────
    _rl = st.session_state.get("rl_report", {})
    # Pull live from engine if not already saved to session state
    if not _rl and hasattr(engine, "get_rl_report"):
        _rl = engine.get_rl_report()
        st.session_state["rl_report"] = _rl

    if _rl and _rl.get("available", False) and _rl.get("steps_recorded", 0) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
<div style="margin-bottom:.8rem;">
  <div style="font-size:.62rem;font-family:Share Tech Mono,monospace;color:#7c3aff;
    letter-spacing:.2em;margin-bottom:.2rem;">// RL ADAPTIVE SEQUENCER REPORT</div>
  <h3 style="margin:0;font-size:1.1rem;">🎮 AGENT LEARNING TRACE</h3>
  <div style="font-size:.7rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;margin-top:.2rem;">
    Research: Patel et al. (2023, Springer AI Review) — RL adaptive question sequencing
  </div>
</div>""", unsafe_allow_html=True)

        _rl_c1, _rl_c2 = st.columns([2, 1])

        with _rl_c1:
            # ── Action Distribution Chart ─────────────────────────────────────
            _act_dist = _rl.get("action_distribution", {})
            if _act_dist:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h4 style="margin-top:0;font-size:.82rem;">⬡ ACTION DISTRIBUTION</h4>',
                            unsafe_allow_html=True)
                _act_labels = list(_act_dist.keys())
                _act_vals   = list(_act_dist.values())
                _act_colors = []
                for lbl in _act_labels:
                    if "technical"   in lbl: _act_colors.append("#00FFD1")
                    elif "behaviour" in lbl: _act_colors.append("#00D4FF")
                    else:                    _act_colors.append("#FFD700")
                _fig_rl = go.Figure(go.Bar(
                    x=_act_labels, y=_act_vals,
                    marker_color=_act_colors,
                    marker_line=dict(width=0),
                    text=_act_vals, textposition="outside",
                    textfont=dict(color="#b0cce8", size=11),
                ))
                _fig_rl.update_layout(
                    height=200, margin=dict(l=0,r=0,t=10,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#5a7a9a", family="Share Tech Mono"),
                    yaxis=dict(gridcolor="rgba(0,255,136,.04)", tickfont=dict(color="#5a7a9a",size=9)),
                    xaxis=dict(tickfont=dict(color="#b0cce8", size=9)),
                )
                st.plotly_chart(_fig_rl, use_container_width=True, config={"displayModeBar":False})
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Q-Value Heatmap ───────────────────────────────────────────────
            _hmap = _rl.get("heatmap", {})
            if _hmap:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h4 style="margin-top:0;font-size:.82rem;">◈ Q-VALUE HEATMAP</h4>',
                            unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-size:.65rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;margin-bottom:.5rem;">'
                    'Agent Q-values for STRUGGLING vs PERFORMING states — higher = agent prefers this action</div>',
                    unsafe_allow_html=True)
                _h_labels = _hmap.get("action_labels", [])
                _h_data   = [
                    _hmap.get("q_values_struggling", []),
                    _hmap.get("q_values_performing", []),
                ]
                _fig_hmap = go.Figure(go.Heatmap(
                    z=_h_data,
                    x=_h_labels,
                    y=["Struggling", "Performing"],
                    colorscale=[[0,"#0a1128"],[0.5,"#1e3a6e"],[1,"#7c3aff"]],
                    text=[[f"{v:.3f}" for v in row] for row in _h_data],
                    texttemplate="%{text}",
                    showscale=False,
                ))
                _fig_hmap.update_layout(
                    height=120, margin=dict(l=0,r=0,t=5,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#b0cce8", size=9, family="Share Tech Mono"),
                    xaxis=dict(tickfont=dict(color="#b0cce8", size=8)),
                    yaxis=dict(tickfont=dict(color="#b0cce8", size=9)),
                )
                st.plotly_chart(_fig_hmap, use_container_width=True, config={"displayModeBar":False})
                # Recommended next action labels
                _rec_str  = f'<div style="display:flex;gap:.6rem;font-size:.65rem;font-family:Share Tech Mono,monospace;margin-top:.2rem;">'
                _rec_str += f'<span style="color:#ff5c5c;">■ Struggling → <b style="color:#00d4ff;">{_hmap.get("recommended_struggling","")}</b></span>'
                _rec_str += f'<span style="color:#22d87a;">■ Performing → <b style="color:#00ff88;">{_hmap.get("recommended_performing","")}</b></span>'
                _rec_str += '</div>'
                st.markdown(_rec_str, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with _rl_c2:
            # ── RL KPI cards ──────────────────────────────────────────────────
            _steps   = _rl.get("steps_recorded", 0)
            _eps_f   = _rl.get("final_epsilon", 0)
            _best_s  = _rl.get("best_rewarded_step", {})
            _worst_s = _rl.get("worst_rewarded_step", {})
            _pref    = _rl.get("preferred_next_question", "—")
            _life    = _rl.get("session_count_lifetime", 1)

            for _lbl, _val, _col in [
                ("STEPS RECORDED",   str(_steps),      "#00D4FF"),
                ("FINAL ε (EXPLORE)",f"{_eps_f:.4f}",  "#7c3aff"),
                ("SESSIONS TRAINED", str(_life),       "#00FFD1"),
            ]:
                st.markdown(f"""
<div style="background:rgba(10,25,47,.9);border:1px solid {_col}33;border-radius:8px;
  padding:.65rem .8rem;margin-bottom:.5rem;text-align:center;">
  <div style="font-size:.52rem;color:#c7e8ff;text-transform:uppercase;
    font-family:Share Tech Mono,monospace;letter-spacing:.1em;">{_lbl}</div>
  <div style="font-family:Orbitron,monospace;font-size:1.2rem;font-weight:700;
    color:{_col};margin-top:.2rem;">{_val}</div>
</div>""", unsafe_allow_html=True)

            if _best_s:
                st.markdown(f"""
<div style="background:rgba(0,255,136,.05);border:1px solid rgba(0,255,136,.2);
  border-radius:8px;padding:.65rem .8rem;margin-bottom:.5rem;">
  <div style="font-size:.52rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;
    text-transform:uppercase;letter-spacing:.1em;">BEST STEP</div>
  <div style="font-size:.78rem;color:#00ff88;font-family:Share Tech Mono,monospace;margin:.2rem 0;">
    Q{_best_s.get('q_number','?')} · {_best_s.get('action','')}</div>
  <div style="font-size:.68rem;color:#c7e8ff;">
    Score {_best_s.get('score',0):.1f} · Reward +{_best_s.get('reward',0):.2f}</div>
</div>""", unsafe_allow_html=True)

            if _pref:
                st.markdown(f"""
<div style="background:rgba(124,58,255,.06);border:1px solid rgba(124,58,255,.25);
  border-radius:8px;padding:.65rem .8rem;margin-bottom:.5rem;">
  <div style="font-size:.52rem;color:#7c3aff;font-family:Share Tech Mono,monospace;
    text-transform:uppercase;letter-spacing:.1em;">AGENT RECOMMENDS NEXT</div>
  <div style="font-size:.9rem;font-weight:700;color:#b4a9ff;
    font-family:Share Tech Mono,monospace;margin-top:.25rem;">{_pref.upper()}</div>
  <div style="font-size:.58rem;color:#c7e8ff;margin-top:.15rem;">
    Based on learned Q-values from this session</div>
</div>""", unsafe_allow_html=True)

            # ── Step-by-step history table ─────────────────────────────────────
            _steps_data = _rl.get("all_steps", [])
            if _steps_data:
                st.markdown('<div style="font-size:.6rem;font-weight:700;color:#c7e8ff;'
                            'text-transform:uppercase;font-family:Share Tech Mono,monospace;'
                            'margin:.6rem 0 .3rem;">STEP HISTORY</div>', unsafe_allow_html=True)
                import pandas as _pd2
                _df_rl = _pd2.DataFrame([{
                    "Q":      f"Q{s['q']}",
                    "Action": s["action"],
                    "Score":  f"{s['score']:.1f}",
                    "Nerv":   f"{s['nerv']:.2f}",
                    "Reward": f"{s['reward']:+.2f}",
                } for s in _steps_data])
                st.dataframe(_df_rl, use_container_width=True, hide_index=True,
                             height=min(180, 35 + len(_steps_data) * 35))




# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
def page_settings() -> None:
    st.markdown("""
<div style="margin-bottom:1.2rem;">
  <div style="font-size:.62rem;font-family:Share Tech Mono,monospace;color:#00d4ff;letter-spacing:.2em;margin-bottom:.2rem;">
    // SYSTEM CONFIGURATION
  </div>
  <h2 style="margin:0;font-size:1.6rem;">SETTINGS</h2>
</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_coach_settings()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top:0;font-size:.85rem;">INTERVIEW OPTIONS</h4>', unsafe_allow_html=True)
        st.toggle("Enable Webcam",       value=True, key="s_cam")
        st.toggle("Whisper ASR",         value=True, key="s_stt")
        st.toggle("AI Coaching Panel",   value=True, key="s_cch")
        st.toggle("Question Timer",      value=True, key="s_tmr")
        st.toggle("Adaptive Difficulty", value=True, key="s_adp")

        st.markdown('<hr style="border-color:rgba(0,212,255,0.08);margin:.6rem 0;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:.65rem;font-family:Share Tech Mono,monospace;color:rgba(0,212,255,0.5);letter-spacing:.15em;margin-bottom:.4rem;">AI AVATAR INTERVIEWER</div>', unsafe_allow_html=True)

        _av_on = st.toggle(
            "Show ARIA-7 Avatar",
            value=st.session_state.get("avatar_enabled", True),
            key="_s_avatar_enabled",
            help="Show the animated AI interviewer face above each question",
        )
        st.session_state["avatar_enabled"] = _av_on

        if _av_on:
            _av_speak = st.toggle(
                "Auto-read questions aloud",
                value=st.session_state.get("avatar_auto_speak", True),
                key="_s_avatar_speak",
                help="ARIA-7 speaks each question via browser Text-to-Speech when it loads",
            )
            st.session_state["avatar_auto_speak"] = _av_speak
            if AVATAR_OK:
                st.markdown('<div class="b-ok">◈ ARIA-7 AVATAR READY — Web Speech API</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="b-warn">▲ avatar_interviewer.py not found — copy it to your project root</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top:0;font-size:.85rem;">SYSTEM STATUS</h4>', unsafe_allow_html=True)
        for label, flag in [("FER FACIAL MODEL", "fer_ready"),
                             ("UNIFIED VOICE (CREMA-D+TESS)", "nervousness_ready"),
                             ("WHISPER ASR", "stt_ready")]:
            cls = "b-ok" if st.session_state[flag] else "b-warn"
            ico = "◈" if st.session_state[flag] else "▲"
            st.markdown(f'<div class="{cls}">{ico} {label}</div>', unsafe_allow_html=True)
        # v9.0: WebRTC status
        _wrtc_cls = "b-ok" if WEBRTC_OK else "b-warn"
        _wrtc_ico = "◈"    if WEBRTC_OK else "▲"
        _wrtc_msg = "WEBRTC STREAM ACTIVE — 30fps (pin: webrtc==0.47.1 aiortc==1.6.0)" if WEBRTC_OK else "WEBRTC OFFLINE — auto-rerun loop active (pip install 'streamlit-webrtc==0.47.1' 'aiortc==1.6.0' for 30fps)"
        st.markdown(f'<div class="{_wrtc_cls}">{_wrtc_ico} {_wrtc_msg}</div>', unsafe_allow_html=True)
        # v8.0: SBERT and RL status
        _sbert_cls = "b-ok" if SBERT_AVAILABLE else "b-warn"
        _sbert_ico = "◈"    if SBERT_AVAILABLE else "▲"
        _sbert_msg = f"SBERT ({SBERT_MODEL})" if SBERT_AVAILABLE else "SBERT OFFLINE — pip install sentence-transformers"
        st.markdown(f'<div class="{_sbert_cls}">{_sbert_ico} {_sbert_msg}</div>', unsafe_allow_html=True)
        _rl_cls = "b-ok" if RL_AVAILABLE else "b-warn"
        _rl_ico = "◈"    if RL_AVAILABLE else "▲"
        _rl_msg = "RL SEQUENCER ACTIVE" if RL_AVAILABLE else "RL SEQUENCER OFFLINE — check adaptive_sequencer.py"
        st.markdown(f'<div class="{_rl_cls}">{_rl_ico} {_rl_msg}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("⬡  RETRAIN / REINITIALISE MODELS", key="s_setup",
                  on_click=nav_to("Model Setup"))
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top:0;font-size:.85rem;">◈ ABOUT — AURA AI v9.0</h4>', unsafe_allow_html=True)
        _sbert_line = (f'SBERT ({SBERT_MODEL}) — offline semantic relevance Layer-2'
                       if SBERT_AVAILABLE else
                       'SBERT — NOT INSTALLED (pip install sentence-transformers)')
        _rl_line    = ('RL SEQUENCER — Q-learning adaptive question selection (Patel et al. 2023)'
                       if RL_AVAILABLE else
                       'RL SEQUENCER — NOT INSTALLED (check adaptive_sequencer.py)')
        _webrtc_line = ('WEBRTC STREAM — continuous 30fps · AuraVideoProcessor · thread-safe'
                        if WEBRTC_OK else
                        'WEBRTC — NOT INSTALLED · pip install streamlit-webrtc aiortc')
        st.markdown(f"""
<div style="font-size:.78rem;color:#c7e8ff;line-height:2.1;font-family:Share Tech Mono,monospace;">
<b style="color:#00ff88;">WEBCAM:</b>         {_webrtc_line}<br>
<b style="color:#00ff88;">AVATAR:</b>         ARIA-7 SVG avatar · Web Speech API TTS · CSS lip-sync · eye-blink (v10.0)<br>
<b style="color:#00ff88;">FACIAL EMOTION:</b> HOG+LBP · RAF-DB+RAVDESS · 7 classes · 2020-dim<br>
<b style="color:#00ff88;">FACIAL NERV:</b>    5-signal: Emotion(35%) + AU(25%) + Blink-SEBR(18%) + Asymmetry(12%) + Gaze(10%)<br>
<b style="color:#00ff88;">NERV FUSION:</b>    Voice×1.00 — 100% voice nervousness, facial excluded (v10.0)<br>
<b style="color:#00ff88;">VOICE EMOTION:</b>  108-dim · CREMA-D+TESS (10,242 samples) · CNN+BiLSTM · 5-fold CV<br>
<b style="color:#00ff88;">EAR:</b>            Eye Aspect Ratio · blink-rate SEBR 300-frame window (Barbato 1995)<br>
<b style="color:#00ff88;">SPEECH-TO-TEXT:</b> OpenAI Whisper-base (Radford et al., 2022)<br>
<b style="color:#00ff88;">CONFIDENCE:</b>     0.25×Eye + 0.25×Fluency + 0.35×Voice + 0.15×Facial<br>
<b style="color:#00ff88;">NLP SCORING:</b>    STAR + Keywords + Groq LLM → SBERT → TF-IDF fallback chain<br>
<b style="color:#00ff88;">SBERT:</b>          {_sbert_line}<br>
<b style="color:#00ff88;">RL SEQUENCER:</b>   {_rl_line}<br>
<b style="color:#00ff88;">DISC:</b>           Keyword density profiling — 4 traits<br>
<b style="color:#00ff88;">GRAMMAR:</b>        LanguageTool API (en-US)<br>
<b style="color:#00ff88;">EMA SMOOTHING:</b>  α=0.22 emotion · α=0.10 nervousness (AffWild2-calibrated)
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-top:0;font-size:.85rem;">▲ CLEAR SESSION DATA</h4>', unsafe_allow_html=True)
        def clear_session():
            for k in ["session_answers","transcript","question","q_number","last_score",
                      "last_feedback","submitted","live_emotion","live_nervousness",
                      "emotion_history","live_voice_emotion","live_voice_nerv",
                      "live_posture","live_confidence","transcribed_text","last_audio_id",
                      "last_audio_bytes","last_audio_source",
                      "_last_frame_count","_cam_last_hash","_cam_last_result"]:
                st.session_state[k] = DEFAULTS.get(k)
            _reset_frame_state()   # v9.1 — clear WebRTC shared buffer
            st.success("◈ SESSION DATA CLEARED")

        st.button("▲  PURGE SESSION DATA", key="s_clr", on_click=clear_session)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar() -> None:
    with st.sidebar:

        # ── Collapse state ────────────────────────────────────────────────────
        if "sb_collapsed" not in st.session_state:
            st.session_state.sb_collapsed = False
        collapsed = st.session_state.sb_collapsed
        cur_page  = st.session_state.page

        # ── Model status ──────────────────────────────────────────────────────
        fer_ok  = st.session_state.fer_ready
        nerv_ok = st.session_state.nervousness_ready
        stt_ok  = st.session_state.stt_ready

        # ── HEADER: candidate chip + brand + collapse toggle ──────────────────
        name      = st.session_state.candidate_name or "Candidate"
        role_disp = st.session_state.target_role    or "No role selected"
        diff_disp = st.session_state.get("difficulty", "Medium")
        # v9.2: in "all" mode show the RL-selected difficulty for current question
        if "all" in diff_disp.lower():
            _rl_current = (engine.get_rl_next_hint() if hasattr(engine, "get_rl_next_hint") else {})
            _rl_d = _rl_current.get("difficulty", "")
            diff_disp = f"RL→{_rl_d.capitalize()}" if _rl_d else "RL Adaptive"
        else:
            diff_disp = diff_disp.capitalize()
        initial   = name[0].upper()
        diff_col  = {"easy":"#00FFD1","medium":"#FFD700","hard":"#FF3366"}.get(
                     diff_disp.lower().replace("rl→",""), "#7c6bff")

        hdr_col, toggle_col = st.columns([5, 1])
        with hdr_col:
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:.65rem;padding:.6rem 0 .4rem;">
  <div style="width:34px;height:34px;border-radius:10px;flex-shrink:0;
    background:linear-gradient(135deg,#6366f1,#8b5cf6);
    display:flex;align-items:center;justify-content:center;
    font-size:.82rem;font-weight:700;color:#fff;font-family:Inter,sans-serif;
    box-shadow:0 4px 12px rgba(99,102,241,.4);">
    {initial}</div>
  {"" if collapsed else f'<div style="min-width:0;"><div style="font-family:Inter,sans-serif;font-size:.95rem;font-weight:700;color:#f1f5f9;letter-spacing:-.01em;">Aura <span style="color:#a5b4fc;">AI</span></div><div style="font-size:.68rem;color:#64748b;font-family:Inter,sans-serif;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:130px;">{name} · <span style="color:{diff_col};">{diff_disp}</span></div></div>'}
</div>""", unsafe_allow_html=True)

        with toggle_col:
            st.markdown('<div class="sb-col-btn">', unsafe_allow_html=True)
            if st.button("◂" if not collapsed else "▸", key="sb_col_btn"):
                st.session_state.sb_collapsed = not collapsed
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:1px;background:rgba(255,255,255,.07);margin-bottom:.4rem;"></div>', unsafe_allow_html=True)

        # ── STATUS PILLS (hidden when collapsed) ──────────────────────────────
        if not collapsed:
            def _pill(lbl, ok):
                c  = "#00FFD1" if ok else "#FF3366"
                bg = "rgba(0,255,136,.07)" if ok else "rgba(255,51,102,.07)"
                dot = "●" if ok else "○"
                return (f'<div style="display:flex;align-items:center;gap:5px;padding:3px 7px;'
                        f'background:{bg};border-radius:3px;margin-bottom:2px;">'
                        f'<span style="font-size:.55rem;color:{c};">{dot}</span>'
                        f'<span style="font-size:.6rem;color:{c};font-family:Share Tech Mono,monospace;">{lbl}</span></div>')

            st.markdown(
                '<div style="padding:0 0 .35rem;">' +
                _pill("FER model",   fer_ok)  +
                _pill("Voice model", nerv_ok) +
                _pill("Whisper ASR", stt_ok)  +
                '</div>',
                unsafe_allow_html=True
            )
            st.markdown('<div style="height:1px;background:rgba(255,255,255,.06);margin-bottom:.3rem;"></div>', unsafe_allow_html=True)

        # ── NAVIGATION: st.button per page (fixes st.radio session-state bug) ─
        # Root cause of the original navigation bug:
        #   st.radio stored its value in st.session_state["sb_nav"].
        #   Any other button that set st.session_state.page was overridden on
        #   the next rerun by the stale radio value → buttons appeared to do nothing.
        # Fix: one st.button per page with key sb_btn_{page}. No shared state.

        NAV_GROUPS = {
            "Interview": ["Dashboard", "Start Interview", "Live Interview", "Final Report", "HR Practice", "Weekly Plan", "Placement Setup", "Placement Test"],
            "Tools":     ["Resume Rephraser", "Company Questions", "Dataset Analysis"],
            "System":    ["Settings"],
        }

        for group, pages in NAV_GROUPS.items():
            if not collapsed:
                st.markdown(f'<span class="sb-section-lbl">{group}</span>',
                            unsafe_allow_html=True)
            for page in pages:
                is_active  = (cur_page == page)
                css_class  = "sb-nav-active" if is_active else "sb-nav-btn"
                # Collapsed: show only the first letter; expanded: full name
                if collapsed:
                    btn_label = page[0]
                elif page == "Model Setup":
                    fd = "●" if fer_ok  else "○"
                    nd = "●" if nerv_ok else "○"
                    sd = "●" if stt_ok  else "○"
                    btn_label = f"{page}  {fd}{nd}{sd}"
                else:
                    btn_label = page

                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                if st.button(btn_label, key=f"sb_btn_{page}"):
                    st.session_state.page = page
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            if not collapsed:
                st.markdown('<div style="height:3px;"></div>', unsafe_allow_html=True)

        st.markdown('<div style="height:1px;background:rgba(0,255,136,.06);margin:.3rem 0;"></div>', unsafe_allow_html=True)

        # ── LIVE SIGNALS (only on Live Interview page) ────────────────────────
        if cur_page == "Live Interview":
            dom  = st.session_state.live_emotion
            nerv = st.session_state.live_nervousness
            vdom = st.session_state.live_voice_emotion
            conf = st.session_state.live_confidence
            ec   = emo_css(dom); vc = emo_css(vdom)
            nc   = nerv_css(nerv); cc = conf_css(conf)
            np_  = int(nerv * 100)

            if not collapsed:
                st.markdown('<span class="sb-section-lbl">Live signals</span>',
                            unsafe_allow_html=True)
                st.markdown(f"""
<div style="background:rgba(10,25,47,.92);border:1px solid rgba(0,255,136,.1);
  border-radius:7px;padding:.55rem .7rem;margin-bottom:.3rem;">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:6px;">
    <div>
      <div style="font-size:.5rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;margin-bottom:2px;">Face</div>
      <div style="font-size:.85rem;font-weight:700;font-family:Orbitron,monospace;color:{ec};">{dom}</div>
    </div>
    <div>
      <div style="font-size:.5rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;margin-bottom:2px;">Voice</div>
      <div style="font-size:.85rem;font-weight:700;font-family:Orbitron,monospace;color:{vc};">{vdom}</div>
    </div>
  </div>
  <div style="margin-bottom:5px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
      <span style="font-size:.48rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;">Nervousness</span>
      <span style="font-size:.65rem;font-weight:700;color:{nc};font-family:Share Tech Mono,monospace;">{np_}%</span>
    </div>
    <div style="background:rgba(255,255,255,.05);border-radius:3px;height:5px;overflow:hidden;">
      <div style="width:{np_}%;height:100%;background:{nc};border-radius:3px;transition:width .4s;"></div>
    </div>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;
    border-top:1px solid rgba(255,255,255,.04);padding-top:5px;">
    <span style="font-size:.48rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;">Confidence</span>
    <span style="font-size:.9rem;font-weight:700;font-family:Orbitron,monospace;
      color:{cc};">{conf:.1f}<span style="font-size:.5rem;color:#c7e8ff;">/5</span></span>
  </div>
</div>""", unsafe_allow_html=True)
            else:
                # Collapsed: 5-dot 2+3 grid mini-map of session progress
                total_qs  = st.session_state.get("num_questions", 5)
                n_mini    = min(total_qs, 5)   # show up to 5 dots (spec: one per question)
                n_orig    = len([a for a in st.session_state.session_answers
                                 if not a.get("is_follow_up", False)])
                orig_ans  = [a for a in st.session_state.session_answers
                              if not a.get("is_follow_up", False)]

                def _mini_dot_class(idx):
                    if idx < len(orig_ans):
                        s = orig_ans[idx]["score"]
                        if s >= 4.2: return "done-ex"
                        if s >= 3.5: return "done-gd"
                        if s >= 2.5: return "done-av"
                        return "done-po"
                    if idx == n_orig:
                        return "current"
                    return "pending"

                # 2+3 grid layout: first row 2 dots, second row 3 dots
                row1 = "".join(
                    f'<div class="sb-mini-dot {_mini_dot_class(i)}" title="Q{i+1}"></div>'
                    for i in range(min(2, n_mini))
                )
                row2 = "".join(
                    f'<div class="sb-mini-dot {_mini_dot_class(i)}" title="Q{i+1}"></div>'
                    for i in range(2, n_mini)
                )
                st.markdown(
                    f'<div style="padding:.35rem .1rem;">'
                    f'  <div style="display:flex;justify-content:center;gap:5px;margin-bottom:5px;">{row1}</div>'
                    f'  <div style="display:flex;justify-content:center;gap:5px;">{row2}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # ── SESSION PROGRESS — TIMELINE SCRUBBER (Idea 17) ──────────────────
        n_done = len(st.session_state.session_answers)
        # v9.2: count only original answers (not follow-ups) for progress
        n_original = len([a for a in st.session_state.session_answers
                          if not a.get("is_follow_up", False)])
        if n_done > 0 and not collapsed:
            avg      = sum(a["score"] for a in st.session_state.session_answers) / n_done
            ac       = conf_css(avg)
            total_qs = st.session_state.get("num_questions", 5)
            prog_pct = int(n_original / max(total_qs, 1) * 100)
            _fu_label = " (follow-up)" if st.session_state.get("q_is_follow_up") else ""

            # Build dot HTML for each question slot
            def _score_dot_class(idx):
                """Return CSS class for dot at position idx (0-based)."""
                # collect original answers in order
                orig = [a for a in st.session_state.session_answers
                        if not a.get("is_follow_up", False)]
                if idx < len(orig):
                    s = orig[idx]["score"]
                    if s >= 4.2: return "done-ex"
                    if s >= 3.5: return "done-gd"
                    if s >= 2.5: return "done-av"
                    return "done-po"
                if idx == n_original:  # current in-progress slot
                    return "current"
                return "pending"

            dots_html = "".join(
                f'<div class="aura-tl-dot {_score_dot_class(i)}" '
                f'title="Q{i+1}"></div>'
                for i in range(total_qs)
            )
            labels_html = "".join(
                f'<span class="aura-tl-label">Q{i+1}</span>'
                for i in range(total_qs)
            )
            # Top arrow row: Q1 → Q2 → Q3 → ... (matches screenshot header)
            arrow_parts = []
            for i in range(total_qs):
                cls = _score_dot_class(i)
                if cls in ("done-ex","done-gd","done-av","done-po"):
                    col = "#00ff88" if cls=="done-ex" else ("#a5b4fc" if cls=="done-gd" else ("#fbbf24" if cls=="done-av" else "#ff3366"))
                elif cls == "current":
                    col = "#00d4ff"
                else:
                    col = "rgba(255,255,255,.25)"
                arrow_parts.append(f'<span style="font-size:.4rem;color:{col};font-family:Share Tech Mono,monospace;">Q{i+1}</span>')
                if i < total_qs - 1:
                    arrow_parts.append('<span style="font-size:.35rem;color:rgba(255,255,255,.2);margin:0 2px;">→</span>')
            arrow_row = "".join(arrow_parts)

            st.markdown('<span class="sb-section-lbl">Session</span>',
                        unsafe_allow_html=True)
            st.markdown(f"""
<div class="aura-timeline">
  <div style="display:flex;justify-content:center;align-items:center;
    gap:1px;margin-bottom:5px;flex-wrap:nowrap;">{arrow_row}</div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
    <div>
      <div style="font-size:.42rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;margin-bottom:1px;">Avg score</div>
      <div style="font-size:1.05rem;font-family:Orbitron,monospace;font-weight:800;color:{ac};">
        {avg:.2f}<span style="font-size:.48rem;color:#c7e8ff;">/5</span></div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:.42rem;color:#c7e8ff;font-family:Share Tech Mono,monospace;margin-bottom:1px;">Progress</div>
      <div style="font-size:.78rem;font-weight:700;color:#00d4ff;
        font-family:Share Tech Mono,monospace;">{n_original}/{total_qs}{_fu_label}</div>
    </div>
  </div>
  <div class="aura-tl-track">
    <div class="aura-tl-fill" style="width:{prog_pct}%;"></div>
    <div class="aura-tl-dots">{dots_html}</div>
  </div>
  <div class="aura-tl-labels">{labels_html}</div>
  <div style="margin-top:5px;font-size:.44rem;color:#00d4ff;
    font-family:Share\ Tech\ Mono,monospace;letter-spacing:.08em;
    text-align:center;opacity:.8;">
    Question {n_original + 1} of {total_qs} in progress…
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    inject_css()
    # Suppress Streamlit deprecation warnings from frontend
    import warnings
    warnings.filterwarnings("ignore")

    # Clear data-page attribute on every rerun (page_start sets it itself)
    if st.session_state.page != "Start Interview":
        components.html("""<script>
window.parent.document.body.removeAttribute('data-page');
</script>""", height=0)

    # ── PAGE WIPE TRANSITION — directional (Idea 15 extended) ───────────────
    # nav() / nav_to() / Back button all write _nav_dir = "fwd" | "bwd" into
    # session state before rerun.  We pass it into the iframe as a literal so
    # the script can pick the correct CSS class synchronously.
    _nav_dir = st.session_state.get("_nav_dir", "fwd")
    components.html(f"""<script>
(function(){{
  var PARENT = window.parent.document;
  var main = PARENT.querySelector('[data-testid="stMainBlockContainer"]')
          || PARENT.querySelector('.main .block-container')
          || PARENT.querySelector('section.main');
  if (!main) return;
  var enterCls = "{_nav_dir}" === "bwd" ? "aura-page-enter-bwd" : "aura-page-enter-fwd";
  main.classList.remove("aura-page-enter-fwd", "aura-page-enter-bwd");
  void main.offsetWidth;
  main.classList.add(enterCls);
  main.addEventListener("animationend", function h(){{
    main.classList.remove("aura-page-enter-fwd", "aura-page-enter-bwd");
    main.removeEventListener("animationend", h);
  }});
}})();
</script>""", height=0)

    render_top_navbar()
    p = st.session_state.page
    if   p == "Dashboard":        page_dashboard()
    elif p == "Resume Rephraser": page_resume(engine=engine)
    elif p == "Model Setup":      page_setup()
    elif p == "Start Interview":  page_start()
    elif p == "Live Interview":   page_live()
    elif p == "Final Report":     page_report()
    elif p == "Settings":          page_settings()
    elif p == "HR Practice":       page_hr_round()
    elif p == "Weekly Plan":       page_weekly_plan(engine=engine)   # v10.1
    elif p == "Placement Setup":   page_placement_setup()            # Pre-test setup
    elif p == "Placement Test":    page_placement_test()              # Placement Test
    elif p == "Company Questions": page_company_questions()           # Company Question Upload
    else:                         page_dashboard()

main()