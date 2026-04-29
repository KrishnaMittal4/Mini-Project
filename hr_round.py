"""
hr_round.py — Aura AI | HR Round Practice Mode (v1.0)
======================================================
Full HR/Managerial interview practice engine based on
"Crafting Your Competitive Edge" .
Features:
  • 21 structured HR/Behavioural questions (Part 1 + Part 2)
  • Per-question framework hints (Key + Focus)
  • AI-powered answer evaluation via Groq LLM
  • STAR / SOAR / MOLI method guidance
  • Session score tracking with per-question breakdown
  • Professional ReportLab PDF report download
"""

from __future__ import annotations

import io
import os
from dotenv import load_dotenv
load_dotenv()
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

log = logging.getLogger("HR_ROUND")

# ── Voice input (reuse existing pipeline, no nervousness wiring) ──────────────
try:
    from voice_input    import voice_input_panel as _vip
    from speech_to_text import SpeechToText as _STT
    VOICE_OK = True
except ImportError:
    VOICE_OK = False
    _vip = None
    _STT = None

# ── ReportLab (PDF generation) ────────────────────────────────────────────────
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        HRFlowable, PageBreak, Paragraph,
        SimpleDocTemplate, Spacer, Table, TableStyle,
    )
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

# ══════════════════════════════════════════════════════════════════════════════
#  HR QUESTION BANK  ( Crafting Your Competitive Edge)
# ══════════════════════════════════════════════════════════════════════════════

HR_QUESTIONS: List[Dict] = [
    # ── PART 1: The Frequent Golden Key ──────────────────────────────────────
    {
        "id": 1,
        "part": "Part 1: The Frequent Golden Key",
        "question": "Tell me about yourself.",
        "key": "Elevator Pitch – Tug of War: pull the string to win",
        "focus": "Education → Key skills/strength → Projects → Career goal/mission statement",
        "method": "Elevator Pitch",
        "why": "Sets the tone for the entire interview; interviewers judge clarity and confidence.",
        "tip": "Keep it 90 seconds. End with a forward-looking mission statement.",
    },
    {
        "id": 2,
        "part": "Part 1: The Frequent Golden Key",
        "question": "What do you know about our organization?",
        "key": "Research company overview – recent developments/news/values and mission",
        "focus": "4W Formula: Who they are – what they do – why they are different – what are their principles and values",
        "method": "4W Formula",
        "why": "Tests preparation and genuine interest in the company.",
        "tip": "Mention a specific recent news item or product to stand out.",
    },
    {
        "id": 3,
        "part": "Part 1: The Frequent Golden Key",
        "question": "Why did you choose Computer Science?",
        "key": "Explain interest in technology, problem-solving, and innovation",
        "focus": "Logical thinking | Problem solving | Team collaboration | Adaptability | Creative thinking",
        "method": "Passion + Skills alignment",
        "why": "Checks authentic motivation and fit with technical roles.",
        "tip": "Tie your choice back to a specific experience or moment of curiosity.",
    },
    {
        "id": 4,
        "part": "Part 1: The Frequent Golden Key",
        "question": "What are your strengths?",
        "key": "Mention 2–3 strengths with concrete examples",
        "focus": "Logical thinking | Problem solving | Team collaboration | Adaptability | Creative thinking",
        "method": "Strength + Example",
        "why": "Evaluates self-awareness and ability to articulate value.",
        "tip": "Use real project examples. Never just list adjectives.",
    },
    {
        "id": 5,
        "part": "Part 1: The Frequent Golden Key",
        "question": "What is your weakness?",
        "key": "Mention a genuine but improvable weakness and show how you are improving it",
        "focus": "Weakness → Awareness → Action → Improvement",
        "method": "WAAI Framework",
        "why": "Tests honesty and growth mindset.",
        "tip": "Never say 'I'm a perfectionist.' Choose a real weakness that does not directly impact core job duties, then explain your corrective action.",
    },
    {
        "id": 6,
        "part": "Part 1: The Frequent Golden Key",
        "question": "Why do you want to join our company?",
        "key": "Show research about the organization, its culture, and growth opportunities",
        "focus": "Company fit + Role alignment + Learning opportunity + Contribution",
        "method": "CRLC Framework",
        "why": "Assesses motivation beyond salary and genuine cultural alignment.",
        "tip": "Personalise — generic answers are immediately spotted by experienced interviewers.",
    },
    {
        "id": 7,
        "part": "Part 1: The Frequent Golden Key",
        "question": "Explain your final year project.",
        "key": "Structure: Objective → Technology → Role → Result",
        "focus": "COER: Context → Objective → Approach → Result",
        "method": "COER Method",
        "why": "Tests technical communication and ownership of work.",
        "tip": "Mention your specific contribution clearly — not just what the team did.",
    },
    {
        "id": 8,
        "part": "Part 1: The Frequent Golden Key",
        "question": "Where do you see yourself in five years?",
        "key": "Show growth mindset and long-term commitment to learning and alignment with the company",
        "focus": "Career progression + Expertise building + Value creation",
        "method": "Career Vision",
        "why": "Checks ambition, retention likelihood, and goal clarity.",
        "tip": "Link your personal growth trajectory to value you will create for the company.",
    },
    {
        "id": 9,
        "part": "Part 1: The Frequent Golden Key",
        "question": "Why should we hire you?",
        "key": "Combine skills + attitude + learning ability",
        "focus": "Skills match + Right attitude + Value addition",
        "method": "SAV Framework",
        "why": "The core pitch — your final chance to consolidate a strong impression.",
        "tip": "Summarise your top 2–3 differentiators confidently. This is your sales pitch — own it.",
    },
    {
        "id": 10,
        "part": "Part 1: The Frequent Golden Key",
        "question": "What motivates you?",
        "key": "Purpose + Growth + Achievement",
        "focus": "Solving complex problems | Learning new technologies | Building useful solutions",
        "method": "PGA Framework",
        "why": "Reveals drive, passion, and cultural fit.",
        "tip": "Align your motivators with what the role and company offer.",
    },
    {
        "id": 11,
        "part": "Part 1: The Frequent Golden Key",
        "question": "Do you have any questions for us?",
        "key": "Ask for learning intent and interest",
        "focus": "Show curiosity about contribution, not just benefits",
        "method": "Curiosity Signals",
        "why": "Tests engagement level and whether you are genuinely interested.",
        "tip": "Prepare 2–3 thoughtful questions about team culture, growth paths, or current challenges.",
    },
    # ── PART 2: Situation-Based HR/Managerial Questions ───────────────────────
    {
        "id": 12,
        "part": "Part 2: Situation-Based HR/Managerial Questions",
        "question": "Tell me about a time when you faced a difficult problem.",
        "key": "Recruiters test problem-solving ability",
        "focus": "Situation → Challenge/Opportunity → Action → Result",
        "method": "SOAR Method",
        "why": "Tests analytical thinking and resilience under challenges.",
        "tip": "Highlight what you learned from the experience, not just the outcome.",
    },
    {
        "id": 13,
        "part": "Part 2: Situation-Based HR/Managerial Questions",
        "question": "Describe a situation where you had a conflict with a team member.",
        "key": "Tests communication and emotional intelligence",
        "focus": "Disagreement → Communication → Understanding → Outcome",
        "method": "DCUO Method",
        "why": "Interviewers assess how you handle people, not just problems.",
        "tip": "Never blame the other person. Focus on maturity, communication, and the positive outcome.",
    },
    {
        "id": 14,
        "part": "Part 2: Situation-Based HR/Managerial Questions",
        "question": "Tell me about a time when you worked under pressure or tight deadlines.",
        "key": "Shows time management and resilience",
        "focus": "High pressure + smart planning + focused execution + successful delivery",
        "method": "SOAR Method",
        "why": "Reveals composure, prioritisation skills, and ability to deliver under constraints.",
        "tip": "Highlight calmness, prioritisation, and delivery under pressure — that's what recruiters truly test.",
    },
    {
        "id": 15,
        "part": "Part 2: Situation-Based HR/Managerial Questions",
        "question": "Describe a situation where you took initiative.",
        "key": "Recruiters want proactive employees",
        "focus": "Identified a gap → Took ownership → Implemented solution → Achieved impact",
        "method": "SOAR Method",
        "why": "Tests proactiveness and ownership mindset.",
        "tip": "Emphasise proactiveness and ownership — companies hire people who act, not just react.",
    },
    {
        "id": 16,
        "part": "Part 2: Situation-Based HR/Managerial Questions",
        "question": "Tell me about a failure or mistake you made.",
        "key": "Evaluates accountability and learning mindset",
        "focus": "Mistake → Ownership → Learning → Improvement",
        "method": "MOLI Method",
        "why": "Checks honesty, maturity, and the ability to grow from setbacks.",
        "tip": "Never justify or blame others. Focus on accountability and growth — that's what impresses recruiters.",
    },
    {
        "id": 17,
        "part": "Part 2: Situation-Based HR/Managerial Questions",
        "question": "Describe a time when you helped a teammate.",
        "key": "Shows collaboration and leadership potential",
        "focus": "Situation → Support → Action → Result",
        "method": "SOAR Method",
        "why": "Companies value team players over solo performers.",
        "tip": "Demonstrate team spirit, empathy, and collaborative instinct.",
    },
    {
        "id": 18,
        "part": "Part 2: Situation-Based HR/Managerial Questions",
        "question": "Tell me about a time you had to learn something quickly.",
        "key": "Important for roles where continuous learning is required",
        "focus": "Situation → Learning need → Action → Result",
        "method": "SOAR Method",
        "why": "Recruiters assess adaptability and learning agility in dynamic roles.",
        "tip": "Highlight the speed, method of learning, and how you applied it practically.",
    },
    {
        "id": 19,
        "part": "Part 2: Situation-Based HR/Managerial Questions",
        "question": "Describe a situation where you had multiple tasks to manage.",
        "key": "Tests prioritisation and productivity",
        "focus": "Situation → Multiple tasks → Prioritisation → Execution → Result",
        "method": "SOAR Method",
        "why": "Key skills recruiters look for in high-performance roles.",
        "tip": "Emphasise prioritisation, organisation, and time management — not just that you were busy.",
    },
    {
        "id": 20,
        "part": "Part 2: Situation-Based HR/Managerial Questions",
        "question": "Tell me about a time when your idea improved a project or process.",
        "key": "Tests innovation and critical thinking",
        "focus": "Problem → Innovative idea → Action → Measurable impact",
        "method": "SOAR Method",
        "why": "Companies need people who actively improve things, not just execute.",
        "tip": "Always mention the measurable impact (time saved, efficiency improved, quality enhanced) — that makes the answer stand out.",
    },
    {
        "id": 21,
        "part": "Part 2: Situation-Based HR/Managerial Questions",
        "question": "Describe a situation where you had to adapt to change.",
        "key": "Evaluates flexibility and adaptability",
        "focus": "Disruption → Mindset shift → Smart action → Achievement",
        "method": "SOAR Method",
        "why": "Recruiters assess flexibility, resilience, and positive mindset in dynamic environments.",
        "tip": "Emphasise your willingness to embrace change — not just that you survived it.",
    },
]

# ══════════════════════════════════════════════════════════════════════════════
#  GROQ AI EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

_GROQ_KEY = os.environ.get(
    "GROQ_API_KEY",
    os.environ.get("GROQ_API_KEY", "")
)

def _evaluate_with_groq(question: Dict, answer: str) -> Dict:
    """
    Call Groq LLM to evaluate an HR answer.
    Returns dict with keys: score (1-10), verdict, strengths, improvements,
    ideal_structure, method_used.
    """
    prompt = f"""You are an expert HR interview coach evaluating a campus placement candidate.

QUESTION: {question['question']}
FRAMEWORK: {question['focus']}
METHOD: {question['method']}
CANDIDATE'S ANSWER: {answer}

Evaluate this answer strictly as an HR expert would in a campus placement interview.
Respond ONLY with a valid JSON object — no preamble, no markdown:
{{
  "score": <integer 1-10>,
  "verdict": "<one phrase: Excellent / Good / Average / Needs Improvement / Poor>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "improvements": ["<improvement 1>", "<improvement 2>"],
  "ideal_structure": "<1-2 sentence ideal answer structure using the {question['method']}>"
}}

Scoring guide:
9-10: Follows {question['method']}, specific examples, confident and structured
7-8: Good structure, minor gaps in specifics or confidence
5-6: Partially follows framework, missing key elements
3-4: Vague or generic, no framework observed
1-2: Off-topic or very poor answer
"""
    try:
        from groq import Groq
        client = Groq(api_key=_GROQ_KEY)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        log.warning(f"Groq eval failed: {exc}")
        return _heuristic_eval(answer)


def _heuristic_eval(answer: str) -> Dict:
    """Simple offline fallback when Groq is unavailable."""
    words = len(answer.split())
    if words < 10:
        score, verdict = 2, "Poor"
    elif words < 30:
        score, verdict = 4, "Needs Improvement"
    elif words < 60:
        score, verdict = 6, "Average"
    elif words < 100:
        score, verdict = 7, "Good"
    else:
        score, verdict = 8, "Good"
    return {
        "score": score,
        "verdict": verdict,
        "strengths": ["Answer provided"],
        "improvements": ["Add specific examples", "Follow the suggested framework"],
        "ideal_structure": "Follow the recommended method: use specific examples and structured flow.",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _init_hr_state():
    defaults = {
        "hr_mode":           False,
        "hr_q_index":        0,
        "hr_answers":        [],        # List[dict] — answer + eval per question
        "hr_started":        False,
        "hr_finished":       False,
        "hr_start_time":     None,
        "hr_selected_ids":   list(range(1, 22)),   # all 21 by default
        "hr_show_hint":      False,
        "hr_filter":         "All Questions (21)",
        "hr_candidate":      "",
        "hr_pending_eval":   None,
        "hr_stt":            None,   # SpeechToText instance (lazy init)
        "hr_show_mc":        False,  # Model Comparison sub-page toggle
        "hr_mc_results":     None,   # cached benchmark results
        "hr_mc_dataset":     None,   # 40 random questions loaded for live scoring
        "hr_mc_live_result": None,   # latest live answer scoring result
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Lazy-initialise Whisper STT (once per session) ────────────────────────
    # Reuse the app-level STT object if it already exists in session state,
    # so we don't load Whisper twice. Fall back to creating a dedicated one.
    if st.session_state.hr_stt is None and VOICE_OK:
        # Try to reuse the stt already loaded by the main app
        if "stt" in st.session_state and getattr(st.session_state.stt, "ready", False):
            st.session_state.hr_stt = st.session_state.stt
        else:
            try:
                st.session_state.hr_stt = _STT()
            except Exception as exc:
                log.warning(f"HR STT init failed: {exc}")
                st.session_state.hr_stt = None


def _active_questions() -> List[Dict]:
    filt = st.session_state.get("hr_filter", "All Questions (21)")
    if filt == "Part 1 Only (Q1–11)":
        return [q for q in HR_QUESTIONS if q["part"].startswith("Part 1")]
    elif filt == "Part 2 Only (Q12–21)":
        return [q for q in HR_QUESTIONS if q["part"].startswith("Part 2")]
    elif filt == "Quick Practice (Top 10)":
        top_ids = [1, 4, 5, 6, 9, 12, 13, 14, 16, 21]
        return [q for q in HR_QUESTIONS if q["id"] in top_ids]
    return HR_QUESTIONS


def _reset_hr_session():
    st.session_state.hr_q_index      = 0
    st.session_state.hr_answers      = []
    st.session_state.hr_started      = True
    st.session_state.hr_finished     = False
    st.session_state.hr_start_time   = time.time()
    st.session_state.hr_show_hint    = False
    st.session_state.hr_pending_eval = None
    # Clear any cached voice answers from a previous session
    for k in list(st.session_state.keys()):
        if k.startswith("hr_cached_ans_"):
            del st.session_state[k]


# ══════════════════════════════════════════════════════════════════════════════
#  CSS (matches Aura AI dark theme)
# ══════════════════════════════════════════════════════════════════════════════

def _inject_hr_css():
    st.markdown("""
<style>
/* ── HR Card ── */
.hr-card {
  background: rgba(10,21,53,0.92);
  border: 1px solid rgba(0,212,255,0.18);
  border-radius: 12px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1rem;
}
.hr-part-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: .62rem;
  color: #5a7a9a;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-bottom: .4rem;
}
.hr-question-text {
  font-family: Inter, sans-serif;
  font-size: 1.22rem;
  font-weight: 700;
  color: #e8f0fe;
  line-height: 1.45;
  margin-bottom: .8rem;
}
.hr-method-badge {
  display: inline-block;
  background: rgba(0,255,209,0.10);
  border: 1px solid rgba(0,255,209,0.25);
  border-radius: 20px;
  padding: 2px 10px;
  font-size: .65rem;
  color: #00FFD1;
  font-family: 'Share Tech Mono', monospace;
  letter-spacing: .8px;
}
.hr-hint-box {
  background: rgba(99,102,241,0.10);
  border-left: 3px solid #6366f1;
  border-radius: 0 8px 8px 0;
  padding: .7rem 1rem;
  margin: .8rem 0;
}
.hr-hint-key {
  font-size: .78rem;
  color: #a5b4fc;
  font-weight: 600;
  margin-bottom: .25rem;
}
.hr-hint-focus {
  font-size: .75rem;
  color: #94a3b8;
  line-height: 1.4;
}
.hr-tip {
  font-size: .74rem;
  color: #fbbf24;
  background: rgba(251,191,36,0.06);
  border-radius: 6px;
  padding: .4rem .7rem;
  margin-top: .6rem;
}
/* Score badge */
.hr-score-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: Orbitron, monospace;
  font-size: 1.5rem;
  font-weight: 800;
  padding: .35rem .9rem;
  border-radius: 8px;
}
/* Progress bar */
.hr-prog-wrap {
  background: rgba(255,255,255,.06);
  border-radius: 4px;
  height: 6px;
  overflow: hidden;
  margin: .4rem 0 .8rem;
}
.hr-prog-fill {
  height: 100%;
  border-radius: 4px;
  background: linear-gradient(90deg,#00ff88,#00d4ff);
  transition: width .5s ease;
}
/* Eval box */
.hr-eval-box {
  background: rgba(10,21,53,0.95);
  border: 1px solid rgba(0,212,255,0.15);
  border-radius: 10px;
  padding: 1rem 1.2rem;
  margin-top: .6rem;
}
.hr-eval-verdict {
  font-size: .85rem;
  font-family: 'Share Tech Mono', monospace;
  letter-spacing: 1px;
  text-transform: uppercase;
  font-weight: 700;
  margin-bottom: .4rem;
}
.hr-strength { color: #22d87a; }
.hr-improve  { color: #ffb840; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER: SETUP SCREEN
# ══════════════════════════════════════════════════════════════════════════════

def _render_setup():
    st.markdown("""
<div style="text-align:center;padding:1.4rem 0 .6rem;">
  <div style="font-family:Orbitron,monospace;font-size:1.6rem;font-weight:800;
    background:linear-gradient(90deg,#00FFD1,#a5b4fc);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;letter-spacing:.04em;">
    HR ROUND PRACTICE
  </div>
  <div style="font-size:.82rem;color:#5a7a9a;font-family:Inter,sans-serif;margin-top:.3rem;">
    Based on <em>Crafting Your Competitive Edge</em>
  </div>
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="hr-card">', unsafe_allow_html=True)
        st.markdown('<div class="hr-part-label">Session Configuration</div>', unsafe_allow_html=True)

        cand = st.text_input("Your Name (optional)", value=st.session_state.hr_candidate,
                             placeholder="e.g. Rahul Sharma", key="hr_cand_input")
        st.session_state.hr_candidate = cand

        filt = st.selectbox(
            "Question Set",
            ["All Questions (21)", "Part 1 Only (Q1–11)",
             "Part 2 Only (Q12–21)", "Quick Practice (Top 10)"],
            key="hr_filter_sel",
        )
        st.session_state.hr_filter = filt

        qs = _active_questions()
        st.markdown(f'<div style="font-size:.72rem;color:#5a7a9a;margin-top:.3rem;">'
                    f'📋 {len(qs)} questions selected</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🎯  Top 21 HR Interview Questions", key="hr_start_btn",
                     use_container_width=True):
            _reset_hr_session()
            st.rerun()

    with col2:
        # Question bank overview
        st.markdown('<div class="hr-card">', unsafe_allow_html=True)
        st.markdown('<div class="hr-part-label">What\'s Covered</div>', unsafe_allow_html=True)
        st.markdown("""
<div style="font-size:.78rem;color:#94a3b8;line-height:1.8;">
  <div style="color:#00FFD1;font-weight:600;margin-bottom:.3rem;">Part 1 – Golden Key Questions (Q1–11)</div>
  Tell me about yourself · Company research · Strengths &amp; weaknesses ·
  Career goals · Why hire you · Motivation
  <div style="color:#a5b4fc;font-weight:600;margin:.6rem 0 .3rem;">Part 2 – Situational / Behavioural (Q12–21)</div>
  Difficult problem · Conflict · Pressure · Initiative · Failure · Teamwork ·
  Fast learning · Multi-tasking · Innovation · Adaptability
  <div style="color:#fbbf24;margin-top:.6rem;font-size:.7rem;">
  🧠 AI-powered evaluation using SOAR / STAR / MOLI frameworks
  </div>
</div>
""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:.4rem;"></div>', unsafe_allow_html=True)
        if st.button("📊  250000+ HR Interview Practice Questions", key="hr_mc_open_btn",
                     use_container_width=True):
            st.session_state.hr_show_mc = True
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER: INDIVIDUAL QUESTION
# ══════════════════════════════════════════════════════════════════════════════

def _render_question(q: Dict, idx: int, total: int):
    pct = int(idx / total * 100)
    # Progress bar
    st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
  font-size:.68rem;color:#5a7a9a;font-family:Share Tech Mono,monospace;margin-bottom:.2rem;">
  <span>Question {idx+1} of {total}</span>
  <span>{pct}% complete</span>
</div>
<div class="hr-prog-wrap"><div class="hr-prog-fill" style="width:{pct}%;"></div></div>
""", unsafe_allow_html=True)

    # Question card
    st.markdown(f"""
<div class="hr-card">
  <div class="hr-part-label">{q['part']} &nbsp;·&nbsp; Q{q['id']}</div>
  <div class="hr-question-text">{q['question']}</div>
  <span class="hr-method-badge">📐 {q['method']}</span>
</div>
""", unsafe_allow_html=True)

    # Hint toggle
    hint_col, _ = st.columns([1, 3])
    with hint_col:
        hint_lbl = "🙈 Hide Hint" if st.session_state.hr_show_hint else "💡 Show Hint"
        if st.button(hint_lbl, key=f"hr_hint_btn_{idx}"):
            st.session_state.hr_show_hint = not st.session_state.hr_show_hint
            st.rerun()

    if st.session_state.hr_show_hint:
        st.markdown(f"""
<div class="hr-hint-box">
  <div class="hr-hint-key">🔑 KEY: {q['key']}</div>
  <div class="hr-hint-focus">🎯 FOCUS: {q['focus']}</div>
  <div class="hr-hint-focus" style="margin-top:.4rem;">❓ WHY ASKED: {q['why']}</div>
</div>
<div class="hr-tip">💬 PRO TIP: {q['tip']}</div>
""", unsafe_allow_html=True)

    # ── Voice / text answer input ─────────────────────────────────────────────
    # Cache key: preserve the last answer across reruns for this question
    ans_cache_key = f"hr_cached_ans_{q['id']}"

    stt = st.session_state.get("hr_stt")

    if VOICE_OK and stt is not None:
        st.markdown(
            '<div style="font-size:.72rem;color:#5a7a9a;'
            'font-family:Share Tech Mono,monospace;margin:.4rem 0 .2rem;">'
            '🎙 ANSWER — speak or type below</div>',
            unsafe_allow_html=True,
        )
        # voice_input_panel returns the latest answer (voice or typed)
        voice_ans = _vip(stt, q["id"])
        # Persist non-empty result so the Submit button sees it after rerun
        if voice_ans and voice_ans.strip() and not voice_ans.startswith("["):
            st.session_state[ans_cache_key] = voice_ans.strip()
        ans = st.session_state.get(ans_cache_key, "")

    else:
        # Fallback when voice modules are unavailable
        if not VOICE_OK:
            st.caption("⚠️ Voice input unavailable — `voice_input` or `speech_to_text` not found.")
        ans = st.text_area(
            "Your Answer",
            placeholder=f"Structure using the {q['method']}...",
            height=160,
            key=f"hr_ans_{idx}",
            label_visibility="collapsed",
        )
        if ans:
            st.session_state[ans_cache_key] = ans

    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])

    # Show what answer is currently captured (helps with voice mode)
    cached = st.session_state.get(ans_cache_key, "")
    if cached:
        word_count = len(cached.split())
        st.markdown(
            f'<div style="font-size:.68rem;color:#22d87a;'
            f'font-family:Share Tech Mono,monospace;margin:.35rem 0 .2rem;">'
            f'✅ Answer captured — {word_count} word{"s" if word_count != 1 else ""} ready to submit</div>',
            unsafe_allow_html=True,
        )
    with btn_col1:
        submit = st.button("✅  Submit & Evaluate", key=f"hr_submit_{idx}",
                           use_container_width=True)
    with btn_col2:
        skip = st.button("⏭  Skip", key=f"hr_skip_{idx}", use_container_width=True)
    with btn_col3:
        if idx > 0:
            if st.button("⏮  Back", key=f"hr_back_{idx}", use_container_width=True):
                st.session_state.hr_q_index = max(0, idx - 1)
                st.session_state.hr_show_hint = False
                st.session_state.hr_pending_eval = None
                st.rerun()

    if submit:
        final_ans = st.session_state.get(ans_cache_key, "").strip()
        if not final_ans:
            st.warning("⚠️ Please record or type your answer before submitting.")
        else:
            with st.spinner("🧠 AI evaluating your answer…"):
                eval_result = _evaluate_with_groq(q, final_ans)
            record = {
                "q_id":      q["id"],
                "question":  q["question"],
                "part":      q["part"],
                "method":    q["method"],
                "answer":    final_ans,
                "eval":      eval_result,
            }
            # Replace existing answer if re-submitting same question
            existing = [i for i, a in enumerate(st.session_state.hr_answers)
                        if a["q_id"] == q["id"]]
            if existing:
                st.session_state.hr_answers[existing[0]] = record
            else:
                st.session_state.hr_answers.append(record)
            st.session_state.hr_pending_eval = record
            st.rerun()

    if skip:
        record = {
            "q_id":      q["id"],
            "question":  q["question"],
            "part":      q["part"],
            "method":    q["method"],
            "answer":    "[Skipped]",
            "eval":      {"score": 0, "verdict": "Skipped",
                          "strengths": [], "improvements": [],
                          "ideal_structure": ""},
        }
        existing = [i for i, a in enumerate(st.session_state.hr_answers)
                    if a["q_id"] == q["id"]]
        if existing:
            st.session_state.hr_answers[existing[0]] = record
        else:
            st.session_state.hr_answers.append(record)
        st.session_state.hr_q_index += 1
        st.session_state.hr_show_hint = False
        st.session_state.hr_pending_eval = None
        # Clear answer cache for this question
        st.session_state.pop(ans_cache_key, None)
        st.rerun()


def _render_eval_result(record: Dict, total: int, current_idx: int):
    """Show the AI evaluation result and a 'Next Question' button."""
    ev = record["eval"]
    score = ev.get("score", 0)
    verdict = ev.get("verdict", "—")

    score_col = ("#22d87a" if score >= 8 else
                 "#fbbf24" if score >= 6 else
                 "#ff5c5c")

    st.markdown(f"""
<div class="hr-eval-box">
  <div style="display:flex;align-items:center;gap:1rem;margin-bottom:.7rem;">
    <div class="hr-score-badge" style="color:{score_col};
      background:rgba(0,0,0,.3);border:1px solid {score_col}44;">
      {score}<span style="font-size:.7rem;color:#5a7a9a;font-family:Inter;">/10</span>
    </div>
    <div>
      <div class="hr-eval-verdict" style="color:{score_col};">{verdict}</div>
      <div style="font-size:.7rem;color:#5a7a9a;font-family:Share Tech Mono,monospace;">
        {record['method']} evaluation
      </div>
    </div>
  </div>
""", unsafe_allow_html=True)

    strengths = ev.get("strengths", [])
    improvements = ev.get("improvements", [])
    ideal = ev.get("ideal_structure", "")

    if strengths:
        st.markdown('<div style="margin-bottom:.4rem;">'
                    '<span class="hr-strength" style="font-size:.72rem;font-weight:600;">✅ STRENGTHS</span></div>',
                    unsafe_allow_html=True)
        for s in strengths:
            st.markdown(f'<div style="font-size:.78rem;color:#94a3b8;padding-left:.8rem;margin-bottom:.2rem;">• {s}</div>',
                        unsafe_allow_html=True)

    if improvements:
        st.markdown('<div style="margin:.5rem 0 .3rem;">'
                    '<span class="hr-improve" style="font-size:.72rem;font-weight:600;">🔧 IMPROVEMENTS</span></div>',
                    unsafe_allow_html=True)
        for imp in improvements:
            st.markdown(f'<div style="font-size:.78rem;color:#94a3b8;padding-left:.8rem;margin-bottom:.2rem;">• {imp}</div>',
                        unsafe_allow_html=True)

    if ideal:
        st.markdown(f'<div style="margin-top:.6rem;padding:.5rem .7rem;'
                    f'background:rgba(99,102,241,.08);border-radius:6px;'
                    f'font-size:.75rem;color:#a5b4fc;">'
                    f'<b>Ideal Structure:</b> {ideal}</div>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    is_last = (current_idx + 1 >= total)
    btn_lbl = "🏁  Finish & See Report" if is_last else "➡️  Next Question"
    if st.button(btn_lbl, key=f"hr_next_{current_idx}", use_container_width=True):
        # Clear cached answer for the question we just finished
        st.session_state.pop(f"hr_cached_ans_{record['q_id']}", None)
        if is_last:
            st.session_state.hr_finished = True
        else:
            st.session_state.hr_q_index += 1
        st.session_state.hr_show_hint = False
        st.session_state.hr_pending_eval = None
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER: FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _score_colour(s: float) -> str:
    if s >= 8:  return "#22d87a"
    if s >= 6:  return "#fbbf24"
    if s >= 4:  return "#ff8c42"
    return "#ff5c5c"


def _render_final_report():
    answers = st.session_state.hr_answers
    if not answers:
        st.warning("No answers recorded. Complete at least one question first.")
        return

    answered   = [a for a in answers if a["answer"] != "[Skipped]"]
    skipped    = len(answers) - len(answered)
    scores     = [a["eval"]["score"] for a in answered]
    avg_score  = round(sum(scores) / len(scores), 2) if scores else 0
    max_score  = max(scores) if scores else 0
    min_score  = min(scores) if scores else 0
    duration_s = int(time.time() - (st.session_state.hr_start_time or time.time()))
    duration_m = duration_s // 60
    candidate  = st.session_state.hr_candidate or "Candidate"

    overall_verdict = ("Excellent 🏆" if avg_score >= 8 else
                       "Good 👍" if avg_score >= 6 else
                       "Average 📈" if avg_score >= 4 else
                       "Needs Work 💪")
    score_col = _score_colour(avg_score)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="text-align:center;padding:1rem 0 .4rem;">
  <div style="font-family:Orbitron,monospace;font-size:1.5rem;font-weight:800;
    background:linear-gradient(90deg,#00FFD1,#a5b4fc);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    HR PRACTICE REPORT
  </div>
  <div style="font-size:.78rem;color:#5a7a9a;margin-top:.3rem;">
    {candidate} &nbsp;·&nbsp; {datetime.now().strftime("%d %b %Y, %I:%M %p")}
  </div>
</div>
""", unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, sub, col_hex in [
        (c1, "Avg Score",  f"{avg_score}/10",       overall_verdict, score_col),
        (c2, "Answered",   f"{len(answered)}/{len(answers)}", "questions",       "#00d4ff"),
        (c3, "Best Score", f"{max_score}/10",        "single answer",   "#22d87a"),
        (c4, "Duration",   f"{duration_m}m",         "practice time",   "#a5b4fc"),
    ]:
        with col:
            st.markdown(f"""
<div class="hr-card" style="text-align:center;padding:.8rem .5rem;">
  <div style="font-size:.58rem;color:#5a7a9a;font-family:Share Tech Mono,monospace;
    text-transform:uppercase;letter-spacing:1px;margin-bottom:.3rem;">{label}</div>
  <div style="font-family:Orbitron,monospace;font-size:1.35rem;font-weight:800;
    color:{col_hex};">{val}</div>
  <div style="font-size:.65rem;color:#5a7a9a;margin-top:.2rem;">{sub}</div>
</div>""", unsafe_allow_html=True)

    # ── Per-question breakdown ────────────────────────────────────────────────
    st.markdown('<div style="font-family:Inter,sans-serif;font-size:1rem;'
                'font-weight:700;color:#e8f0fe;margin:1rem 0 .5rem;">'
                '📋 Question-by-Question Breakdown</div>', unsafe_allow_html=True)

    for rec in answers:
        ev = rec["eval"]
        s  = ev.get("score", 0)
        sc = _score_colour(s)
        verdict = ev.get("verdict", "Skipped")
        st.markdown(f"""
<div class="hr-card" style="padding:1rem 1.2rem;margin-bottom:.6rem;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div style="flex:1;">
      <div class="hr-part-label" style="margin-bottom:.2rem;">Q{rec['q_id']} · {rec['method']}</div>
      <div style="font-size:.88rem;font-weight:600;color:#e8f0fe;margin-bottom:.4rem;">
        {rec['question']}
      </div>
      <div style="font-size:.75rem;color:#94a3b8;font-style:italic;margin-bottom:.4rem;">
        "{rec['answer'][:200]}{"…" if len(rec['answer']) > 200 else ""}"
      </div>
    </div>
    <div style="text-align:center;margin-left:1rem;min-width:60px;">
      <div style="font-family:Orbitron,monospace;font-size:1.3rem;font-weight:800;color:{sc};">{s}</div>
      <div style="font-size:.55rem;color:#5a7a9a;font-family:Share Tech Mono,monospace;">/{10 if s>0 else "—"}</div>
      <div style="font-size:.6rem;color:{sc};margin-top:.2rem;">{verdict}</div>
    </div>
  </div>
""", unsafe_allow_html=True)
        # Improvements inline
        imps = ev.get("improvements", [])
        if imps and s > 0:
            st.markdown('<div style="font-size:.7rem;color:#ffb840;margin-top:.2rem;">'
                        + " &nbsp;|&nbsp; ".join(f"🔧 {i}" for i in imps[:2])
                        + '</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── PDF Download button ───────────────────────────────────────────────────
    st.markdown('<div style="height:.8rem;"></div>', unsafe_allow_html=True)
    pdf_bytes = _build_hr_pdf({
        "candidate":   candidate,
        "timestamp":   datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "avg_score":   avg_score,
        "max_score":   max_score,
        "min_score":   min_score,
        "n_answered":  len(answered),
        "n_total":     len(answers),
        "n_skipped":   skipped,
        "duration_m":  duration_m,
        "verdict":     overall_verdict,
        "answers":     answers,
        "filter":      st.session_state.hr_filter,
    })

    col_dl, col_restart = st.columns(2)
    with col_dl:
        st.download_button(
            label="⬇️  Download PDF Report",
            data=pdf_bytes,
            file_name=f"HR_Practice_{candidate.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with col_restart:
        if st.button("🔄  Practice Again", key="hr_restart_btn", use_container_width=True):
            for k in ["hr_q_index","hr_answers","hr_started","hr_finished",
                      "hr_start_time","hr_show_hint","hr_pending_eval","hr_mc_run"]:
                st.session_state[k] = ([] if k=="hr_answers" else
                                       False if k in ("hr_started","hr_finished","hr_show_hint","hr_mc_run") else
                                       None)
            st.rerun()

    # ── Model Comparison — Q22 ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<div style="background:rgba(99,102,241,.07);border:1px solid rgba(99,102,241,.25);
  border-radius:12px;padding:1rem 1.3rem;margin-bottom:.6rem;">
  <div style="font-family:Orbitron,monospace;font-size:.8rem;color:#a5b4fc;
    letter-spacing:.08em;margin-bottom:.3rem;">◈ MODEL COMPARISON — HOW DO THE 5 AI SCORERS RATE YOUR ANSWERS?</div>
  <div style="font-size:.78rem;color:#94a3b8;font-family:Inter,sans-serif;line-height:1.5;">
    Run all 5 scoring models (Keyword Match · TF-IDF · BM25 · SBERT · Aura AI) on every
    answer you submitted. See which model rates you highest and where Aura AI adds value
    over the simpler baselines.
  </div>
</div>""", unsafe_allow_html=True)

    if st.button("🔬  Run Model Comparison on My HR Answers", key="hr_mc_run_btn",
                 use_container_width=True):
        st.session_state["hr_mc_run"] = True

    if st.session_state.get("hr_mc_run"):
        _hr_mc_answers = [a for a in answers if a["answer"] != "[Skipped]"]
        if not _hr_mc_answers:
            st.warning("No answered questions to compare.")
        else:
            # Import scoring functions
            try:
                from model_comparison import (
                    score_keyword_match as _mc_kw,
                    score_tfidf as _mc_tf,
                    score_bm25 as _mc_bm,
                    score_sbert as _mc_sb,
                    score_aura as _mc_aura,
                )
                _MC_OK = True
            except ImportError:
                _MC_OK = False
                st.warning("⚠️ model_comparison.py not found — cannot run scorer comparison.")

            if _MC_OK:
                # Build ideal answer proxies from the HR question focus/key hints
                _GROQ_KEY_HR = os.environ.get("GROQ_API_KEY",
                    os.environ.get("GROQ_API_KEY", ""))
                import plotly.graph_objects as _go_hr

                _SCORERS = ["Keyword Match", "TF-IDF", "BM25", "SBERT", "Aura AI"]
                _SCORER_COLS = {
                    "Aura AI":       "#a5b4fc",
                    "SBERT":         "#34d399",
                    "BM25":          "#fcd34d",
                    "TF-IDF":        "#fb923c",
                    "Keyword Match": "#94a3b8",
                }

                # Build a reference ideal per question from the HR question bank
                _q_map = {q["id"]: q for q in HR_QUESTIONS}
                _mc_rows = []
                with st.spinner("Running 5 models across your HR answers…"):
                    for rec in _hr_mc_answers:
                        q_meta = _q_map.get(rec["q_id"], {})
                        # Use the question focus as a proxy ideal answer
                        _ideal_proxy = (
                            f"{q_meta.get('key','')}. "
                            f"{q_meta.get('focus','')}. "
                            f"{q_meta.get('tip','')}"
                        ).strip()
                        _kws = [w.strip() for w in q_meta.get("focus","").replace("→","·").split("·")
                                if len(w.strip()) > 3][:6]
                        ans = rec["answer"]
                        _row_scores = {
                            "Keyword Match": _mc_kw(ans, _ideal_proxy, _kws),
                            "TF-IDF":        _mc_tf(ans, _ideal_proxy),
                            "BM25":          _mc_bm(ans, _ideal_proxy),
                            "SBERT":         _mc_sb(ans, _ideal_proxy),
                            "Aura AI":       _mc_aura(ans, _ideal_proxy, _kws,
                                                      category="HR",
                                                      groq_api_key=_GROQ_KEY_HR),
                        }
                        _mc_rows.append({
                            "q_id":    rec["q_id"],
                            "label":   f"Q{rec['q_id']}",
                            "question": rec["question"][:55] + ("…" if len(rec["question"])>55 else ""),
                            "scores":  _row_scores,
                            "ai_score": rec["eval"].get("score", 0) * 10,  # /10 → /100
                        })

                # Summary table
                st.markdown("""
<div style="font-family:Share Tech Mono,monospace;font-size:.65rem;color:#a5b4fc;
  letter-spacing:.08em;margin:.6rem 0 .4rem;">◈ SCORES PER QUESTION × MODEL</div>""",
                    unsafe_allow_html=True)
                hdr_c = st.columns([3] + [1]*5 + [1])
                hdr_c[0].markdown('<div style="font-size:.62rem;color:#a5b4fc;font-family:Share Tech Mono,monospace;">QUESTION</div>', unsafe_allow_html=True)
                for _i, _s in enumerate(_SCORERS):
                    hdr_c[_i+1].markdown(f'<div style="font-size:.58rem;color:{_SCORER_COLS[_s]};font-family:Share Tech Mono,monospace;">{_s[:8]}</div>', unsafe_allow_html=True)
                hdr_c[6].markdown('<div style="font-size:.58rem;color:#fbbf24;font-family:Share Tech Mono,monospace;">HR AI</div>', unsafe_allow_html=True)

                for _row in _mc_rows:
                    _rc = st.columns([3] + [1]*5 + [1])
                    _rc[0].markdown(
                        f'<div style="font-size:.7rem;color:#e2e8f0;font-family:Inter,sans-serif;'
                        f'padding:.3rem 0;">{_row["label"]}<br>'
                        f'<span style="font-size:.58rem;color:#475569;">{_row["question"]}</span></div>',
                        unsafe_allow_html=True)
                    for _i, _s in enumerate(_SCORERS):
                        _v = _row["scores"][_s]
                        _c = _SCORER_COLS[_s]
                        _bar = min(100, _v)
                        _rc[_i+1].markdown(
                            f'<div style="padding:.25rem 0;">'
                            f'<div style="font-family:Orbitron,monospace;font-size:.82rem;font-weight:700;color:{_c};">{_v:.0f}</div>'
                            f'<div style="background:rgba(255,255,255,.06);border-radius:2px;height:3px;overflow:hidden;margin-top:2px;">'
                            f'<div style="width:{_bar:.0f}%;height:100%;background:{_c};border-radius:2px;"></div></div>'
                            f'</div>', unsafe_allow_html=True)
                    _ai_v = _row["ai_score"]
                    _ai_c = "#22d87a" if _ai_v >= 70 else "#fbbf24" if _ai_v >= 50 else "#ff5c5c"
                    _rc[6].markdown(
                        f'<div style="font-family:Orbitron,monospace;font-size:.82rem;font-weight:700;'
                        f'color:{_ai_c};padding:.25rem 0;">{_ai_v:.0f}</div>',
                        unsafe_allow_html=True)

                # Grouped bar chart across all questions
                st.markdown("<br>", unsafe_allow_html=True)
                _fig_hr_mc = _go_hr.Figure()
                for _s in _SCORERS:
                    _fig_hr_mc.add_trace(_go_hr.Bar(
                        name=_s,
                        x=[r["label"] for r in _mc_rows],
                        y=[r["scores"][_s] for r in _mc_rows],
                        marker_color=_SCORER_COLS[_s],
                    ))
                # Also show HR AI score
                _fig_hr_mc.add_trace(_go_hr.Scatter(
                    name="HR AI Score (×10)",
                    x=[r["label"] for r in _mc_rows],
                    y=[r["ai_score"] for r in _mc_rows],
                    mode="lines+markers",
                    line=dict(color="#fbbf24", width=2, dash="dot"),
                    marker=dict(color="#fbbf24", size=7),
                ))
                _fig_hr_mc.update_layout(
                    barmode="group", height=360,
                    margin=dict(l=0, r=0, t=20, b=40),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(font=dict(color="#b4cde4", size=9),
                                bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.22),
                    xaxis=dict(tickfont=dict(color="#a5b4fc", size=9)),
                    yaxis=dict(tickfont=dict(color="#7ec8e8", size=9),
                               gridcolor="rgba(255,255,255,.04)", range=[0, 105]),
                    title=dict(text="All 5 Models vs HR AI Score",
                               font=dict(color="#a5b4fc", size=11, family="Share Tech Mono")),
                )
                st.plotly_chart(_fig_hr_mc, use_container_width=True,
                                config={"displayModeBar": False})

                # Where Aura AI adds the most value
                st.markdown("""
<div style="font-family:Share Tech Mono,monospace;font-size:.65rem;color:#34d399;
  letter-spacing:.08em;margin:.6rem 0 .4rem;">◈ WHERE AURA AI ADDS MOST VALUE OVER BASELINES</div>""",
                    unsafe_allow_html=True)
                _delta_cols = st.columns(len(_mc_rows))
                for _ci, _row in enumerate(_mc_rows):
                    _aura_v = _row["scores"]["Aura AI"]
                    _best_bl = max(_row["scores"][_s] for _s in ["Keyword Match","TF-IDF","BM25","SBERT"])
                    _delta = _aura_v - _best_bl
                    _dc = "#34d399" if _delta >= 0 else "#fca5a5"
                    _sign = "+" if _delta >= 0 else ""
                    _delta_cols[_ci].markdown(f"""
<div style="background:rgba(0,12,30,0.5);border:1px solid #a5b4fc22;
  border-radius:8px;padding:.6rem .4rem;text-align:center;">
  <div style="font-family:Orbitron,monospace;font-size:.95rem;font-weight:700;
    color:{_dc};">{_sign}{_delta:.0f}</div>
  <div style="font-size:.55rem;color:#a5b4fc;font-family:Share Tech Mono,monospace;
    margin:.15rem 0;">{_row["label"]}</div>
  <div style="font-size:.52rem;color:#475569;font-family:Inter,sans-serif;">
    Aura {_aura_v:.0f} vs {_best_bl:.0f}</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PDF REPORT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_hr_pdf(data: Dict) -> bytes:
    """Build and return a professional HR Practice Report PDF."""
    if not REPORTLAB_OK:
        return _plain_text_hr(data).encode("utf-8")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2*cm,
        title=f"HR Practice Report — {data['candidate']}",
        author="Aura AI",
    )
    W, H = A4

    # ── Colour palette ────────────────────────────────────────────────────────
    C_BG      = colors.HexColor("#040816")
    C_BG2     = colors.HexColor("#0a1535")
    C_ACCENT  = colors.HexColor("#00FFD1")
    C_PURPLE  = colors.HexColor("#a5b4fc")
    C_TEXT    = colors.HexColor("#e8f0fe")
    C_MUTED   = colors.HexColor("#5a7098")
    C_BORDER  = colors.HexColor("#1e3a5f")
    C_GREEN   = colors.HexColor("#22d87a")
    C_YELLOW  = colors.HexColor("#fbbf24")
    C_ORANGE  = colors.HexColor("#ff8c42")
    C_RED     = colors.HexColor("#ff5c5c")

    def sc(score) -> colors.HexColor:
        if score >= 8: return C_GREEN
        if score >= 6: return C_YELLOW
        if score >= 4: return C_ORANGE
        return C_RED

    # ── Styles ────────────────────────────────────────────────────────────────
    SS = getSampleStyleSheet()

    def S(name, **kw):
        return ParagraphStyle(name, parent=SS["Normal"], **kw)

    styles = {
        "title":    S("hr_title",   fontSize=20, textColor=C_TEXT,
                      fontName="Helvetica-Bold", alignment=TA_CENTER,
                      spaceBefore=4, spaceAfter=4),
        "subtitle": S("hr_sub",     fontSize=9,  textColor=C_MUTED,
                      alignment=TA_CENTER, spaceAfter=8),
        "h2":       S("hr_h2",      fontSize=12, textColor=C_ACCENT,
                      fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=4),
        "h3":       S("hr_h3",      fontSize=9.5,textColor=C_TEXT,
                      fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=2),
        "body":     S("hr_body",    fontSize=8.5,textColor=C_TEXT,
                      leading=13, spaceAfter=3),
        "muted":    S("hr_muted",   fontSize=7.5,textColor=C_MUTED,
                      leading=11, spaceAfter=2),
        "answer":   S("hr_ans",     fontSize=8,  textColor=colors.HexColor("#94b0d8"),
                      leading=12, leftIndent=6, spaceAfter=3),
        "label":    S("hr_lbl",     fontSize=6.5,textColor=C_MUTED,
                      fontName="Helvetica-Bold", spaceBefore=3),
    }

    story = []

    # ── Cover header ──────────────────────────────────────────────────────────
    hdr = Table([[Paragraph("AURA AI  ·  HR ROUND PRACTICE REPORT", styles["title"])]],
                colWidths=[W - 4*cm])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_BG),
        ("TOPPADDING",    (0,0),(-1,-1), 16),
        ("BOTTOMPADDING", (0,0),(-1,-1), 16),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
    ]))
    story.extend([hdr, Spacer(1, 4)])

    meta = f"{data['candidate']}  ·  {data['timestamp']}  ·  {data['filter']}"
    story.append(Paragraph(meta, styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_ACCENT, spaceAfter=12))

    # ── Summary metrics ───────────────────────────────────────────────────────
    story.append(Paragraph("Performance Summary", styles["h2"]))

    avg  = data["avg_score"]
    n_a  = data["n_answered"]
    n_t  = data["n_total"]
    dur  = data["duration_m"]
    mxs  = data["max_score"]
    sc_c = sc(avg)

    def metric_cell(lbl, val, sub, c):
        return [
            Paragraph(lbl, S(f"mc_lbl_{lbl}", fontSize=6.5, textColor=colors.white,
                             fontName="Helvetica-Bold")),
            Paragraph(f'<b>{val}</b>',
                      S(f"mc_val_{lbl}", fontSize=15, textColor=c,
                        fontName="Helvetica-Bold", spaceAfter=1)),
            Paragraph(sub, S(f"mc_sub_{lbl}", fontSize=7, textColor=C_MUTED,
                             alignment=TA_CENTER)),
        ]

    met = [[
        Table([metric_cell("Avg Score",  f"{avg}/10",   data["verdict"],       sc_c)],     colWidths=[3.8*cm]),
        Table([metric_cell("Answered",   f"{n_a}/{n_t}", "questions",          C_ACCENT)], colWidths=[3.8*cm]),
        Table([metric_cell("Best Score", f"{mxs}/10",   "single answer",       C_GREEN)],  colWidths=[3.8*cm]),
        Table([metric_cell("Duration",   f"{dur}m",     "practice session",    C_PURPLE)], colWidths=[3.8*cm]),
    ]]
    mt = Table(met, colWidths=[3.8*cm]*4)
    mt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_BG2),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("VALIGN",        (0,0),(-1,-1), "TOP"),
        ("GRID",          (0,0),(-1,-1), 0.4, C_BORDER),
    ]))
    story.extend([mt, Spacer(1, 12)])

    # ── Score summary table ───────────────────────────────────────────────────
    story.append(Paragraph("Question-by-Question Results", styles["h2"]))

    tbl_header = [
        Paragraph("Q#",       S("th", fontSize=7, textColor=C_ACCENT, fontName="Helvetica-Bold")),
        Paragraph("Question", S("th", fontSize=7, textColor=C_ACCENT, fontName="Helvetica-Bold")),
        Paragraph("Method",   S("th", fontSize=7, textColor=C_ACCENT, fontName="Helvetica-Bold")),
        Paragraph("Score",    S("th", fontSize=7, textColor=C_ACCENT, fontName="Helvetica-Bold")),
        Paragraph("Verdict",  S("th", fontSize=7, textColor=C_ACCENT, fontName="Helvetica-Bold")),
    ]
    tbl_rows = [tbl_header]
    for rec in data["answers"]:
        ev = rec["eval"]
        s  = ev.get("score", 0)
        v  = ev.get("verdict", "Skipped")
        sc_clr = sc(s) if s > 0 else C_MUTED
        tbl_rows.append([
            Paragraph(str(rec["q_id"]),    S(f"td_qid_{rec['q_id']}", fontSize=7, textColor=C_MUTED)),
            Paragraph(rec["question"][:60] + ("…" if len(rec["question"])>60 else ""),
                      S(f"td_q_{rec['q_id']}", fontSize=7, textColor=C_TEXT, leading=10)),
            Paragraph(rec["method"],       S(f"td_m_{rec['q_id']}", fontSize=6.5, textColor=C_MUTED)),
            Paragraph(f'<font color="{sc_clr.hexval() if hasattr(sc_clr,"hexval") else "#22d87a"}"><b>{s}/10</b></font>',
                      S(f"td_s_{rec['q_id']}", fontSize=8, fontName="Helvetica-Bold")),
            Paragraph(v,                   S(f"td_v_{rec['q_id']}", fontSize=7, textColor=sc_clr)),
        ])

    summary_tbl = Table(tbl_rows, colWidths=[1*cm, 7.5*cm, 3*cm, 1.5*cm, 2.5*cm])
    summary_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  C_BG2),
        ("BACKGROUND",    (0,1), (-1,-1), C_BG),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [C_BG, colors.HexColor("#060f26")]),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 5),
        ("RIGHTPADDING",  (0,0), (-1,-1), 5),
        ("GRID",          (0,0), (-1,-1), 0.3, C_BORDER),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]))
    story.extend([summary_tbl, Spacer(1, 14)])

    # ── Detailed Q&A ─────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Detailed Answer Review", styles["h2"]))

    for i, rec in enumerate(data["answers"]):
        ev  = rec["eval"]
        s   = ev.get("score", 0)
        sc_c2 = sc(s) if s > 0 else C_MUTED

        # Question header row
        q_hdr = Table([[
            Paragraph(f'Q{rec["q_id"]}', S(f"qn_{i}", fontSize=9, textColor=C_MUTED,
                                            fontName="Helvetica-Bold")),
            Paragraph(rec["question"],   S(f"qq_{i}", fontSize=9,  textColor=C_TEXT,
                                            fontName="Helvetica-Bold", leading=12)),
            Paragraph(f'{s}/10 · {ev.get("verdict","—")}',
                      S(f"qs_{i}", fontSize=8.5, textColor=sc_c2,
                        fontName="Helvetica-Bold", alignment=TA_RIGHT)),
        ]], colWidths=[1*cm, 11*cm, 3.5*cm])
        q_hdr.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), C_BG2),
            ("TOPPADDING",    (0,0),(-1,-1), 7),
            ("BOTTOMPADDING", (0,0),(-1,-1), 7),
            ("LEFTPADDING",   (0,0),(-1,-1), 8),
            ("RIGHTPADDING",  (0,0),(-1,-1), 8),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ]))
        story.append(q_hdr)

        # Answer
        ans_text = rec["answer"] if rec["answer"] != "[Skipped]" else "(Skipped)"
        story.append(Paragraph(f'Answer: {ans_text[:500]}', styles["answer"]))

        # Method + framework
        story.append(Paragraph(f'Method: {rec["method"]}', styles["label"]))

        # Strengths
        strengths = ev.get("strengths", [])
        if strengths:
            story.append(Paragraph("Strengths:", S(f"st_lbl_{i}", fontSize=7,
                         textColor=C_GREEN, fontName="Helvetica-Bold")))
            for st_item in strengths:
                story.append(Paragraph(f"• {st_item}", S(f"st_item_{i}_{id(st_item)}",
                                       fontSize=7.5, textColor=C_TEXT, leading=11,
                                       leftIndent=8)))

        # Improvements
        imps = ev.get("improvements", [])
        if imps:
            story.append(Paragraph("Improvements:", S(f"imp_lbl_{i}", fontSize=7,
                         textColor=C_YELLOW, fontName="Helvetica-Bold")))
            for imp in imps:
                story.append(Paragraph(f"• {imp}", S(f"imp_item_{i}_{id(imp)}",
                                       fontSize=7.5, textColor=C_TEXT, leading=11,
                                       leftIndent=8)))

        # Ideal structure
        ideal = ev.get("ideal_structure", "")
        if ideal:
            story.append(Paragraph(f"Ideal Structure: {ideal}",
                                   S(f"ideal_{i}", fontSize=7.5, textColor=C_PURPLE,
                                     leading=11, leftIndent=4, spaceAfter=2)))

        story.append(HRFlowable(width="100%", thickness=0.3, color=C_BORDER, spaceAfter=6))

    # ── Footer note ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    footer_txt = (
        "This report was generated by Aura AI HR Practice Mode. "
        "Question frameworks sourced from 'Crafting Your Competitive Edge' "
        "Focus on building clarity of thought and "
        "structured responses rather than memorising answers."
    )
    story.append(Paragraph(footer_txt, S("footer", fontSize=7, textColor=C_MUTED,
                                          alignment=TA_CENTER, leading=11)))

    doc.build(story)
    return buf.getvalue()


def _plain_text_hr(data: Dict) -> str:
    lines = [
        "AURA AI — HR PRACTICE REPORT",
        "=" * 50,
        f"Candidate : {data['candidate']}",
        f"Date      : {data['timestamp']}",
        f"Avg Score : {data['avg_score']}/10",
        f"Verdict   : {data['verdict']}",
        f"Answered  : {data['n_answered']}/{data['n_total']}",
        "", "QUESTION BREAKDOWN", "-" * 40,
    ]
    for rec in data["answers"]:
        ev = rec["eval"]
        lines.append(f"\nQ{rec['q_id']}: {rec['question']}")
        lines.append(f"Score  : {ev.get('score','—')}/10  |  {ev.get('verdict','—')}")
        lines.append(f"Answer : {rec['answer'][:300]}")
        for imp in ev.get("improvements", []):
            lines.append(f"  ➜ {imp}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  HR MODEL COMPARISON SUB-PAGE
# ══════════════════════════════════════════════════════════════════════════════

# Top companies and their HR question frequency data (curated)
_HR_COMPANY_DATA = [
    {"company": "Google",        "questions": 18500, "categories": "Behavioural, Leadership, Problem Solving, Adaptability",       "top_q": "Tell me about a time you failed and what you learned."},
    {"company": "Amazon",        "questions": 22400, "categories": "Leadership Principles, Behavioural, Situational",              "top_q": "Describe a time you delivered results under a tight deadline."},
    {"company": "Microsoft",     "questions": 16700, "categories": "Behavioural, Collaboration, Growth Mindset",                    "top_q": "How do you handle disagreement with your manager?"},
    {"company": "Meta",          "questions": 14300, "categories": "Behavioural, Leadership, Communication",                        "top_q": "Tell me about a time you influenced without authority."},
    {"company": "Apple",         "questions": 12800, "categories": "Behavioural, Innovation, Problem Solving",                     "top_q": "Describe a creative solution you developed at work."},
    {"company": "Goldman Sachs", "questions": 11200, "categories": "Situational, Leadership, Self-Awareness",                      "top_q": "Where do you see yourself in five years?"},
    {"company": "McKinsey",      "questions": 10900, "categories": "Behavioural, Problem Solving, Leadership",                     "top_q": "Walk me through how you approach a problem you've never seen."},
    {"company": "Deloitte",      "questions": 10400, "categories": "Behavioural, Collaboration, Career Goals",                     "top_q": "Why do you want to join our company?"},
    {"company": "TCS",           "questions": 19800, "categories": "Behavioural, Situational, HR Basics",                          "top_q": "Tell me about yourself."},
    {"company": "Infosys",       "questions": 17600, "categories": "Behavioural, Teamwork, Communication",                         "top_q": "Describe a situation where you worked under pressure."},
    {"company": "Wipro",         "questions": 15200, "categories": "HR Basics, Behavioural, Adaptability",                         "top_q": "What are your strengths and weaknesses?"},
    {"company": "Accenture",     "questions": 16900, "categories": "Behavioural, Leadership, Self-Awareness",                      "top_q": "Why should we hire you?"},
    {"company": "IBM",           "questions": 13100, "categories": "Behavioural, Problem Solving, Innovation",                     "top_q": "Tell me about a time you led a team through change."},
    {"company": "Cognizant",     "questions": 14700, "categories": "HR Basics, Behavioural, Teamwork",                             "top_q": "How do you motivate a team that is losing morale?"},
    {"company": "Capgemini",     "questions": 12300, "categories": "Behavioural, Communication, Career Goals",                     "top_q": "Describe a time you helped a teammate."},
    {"company": "JP Morgan",     "questions": 11800, "categories": "Situational, Leadership, Self-Awareness",                      "top_q": "How do you make decisions under uncertainty?"},
    {"company": "Adobe",         "questions": 9600,  "categories": "Behavioural, Innovation, Collaboration",                       "top_q": "Describe a time your idea improved a process."},
    {"company": "Salesforce",    "questions": 9200,  "categories": "Behavioural, Leadership, Communication",                       "top_q": "Tell me about a time you had to adapt quickly to change."},
    {"company": "Netflix",       "questions": 8700,  "categories": "Behavioural, Self-Awareness, Leadership",                      "top_q": "Describe your leadership style."},
    {"company": "Uber",          "questions": 8400,  "categories": "Behavioural, Problem Solving, Adaptability",                   "top_q": "Tell me about a time you had to learn something quickly."},
    {"company": "Flipkart",      "questions": 13600, "categories": "Behavioural, HR Basics, Teamwork",                             "top_q": "Where do you see yourself in five years?"},
    {"company": "Swiggy",        "questions": 8100,  "categories": "Behavioural, Adaptability, Problem Solving",                   "top_q": "How do you handle multiple tasks simultaneously?"},
    {"company": "Zomato",        "questions": 7800,  "categories": "Behavioural, Communication, Leadership",                       "top_q": "Tell me about a time you faced a difficult problem."},
    {"company": "Paytm",         "questions": 7200,  "categories": "Behavioural, HR Basics, Career Goals",                         "top_q": "Why did you choose this field?"},
    {"company": "HCL",           "questions": 11900, "categories": "HR Basics, Behavioural, Teamwork",                             "top_q": "Describe a conflict with a team member and how you resolved it."},
]

_CATEGORY_STATS = {
    "Behavioural":   {"count": 98000, "color": "#00d4ff"},
    "Leadership":    {"count": 41000, "color": "#a5b4fc"},
    "Problem Solving":{"count": 35000,"color": "#00ff88"},
    "Communication": {"count": 28000, "color": "#fbbf24"},
    "Adaptability":  {"count": 22000, "color": "#f472b6"},
    "Collaboration": {"count": 18000, "color": "#34d399"},
    "Self-Awareness":{"count": 15000, "color": "#fb923c"},
    "Career Goals":  {"count": 12000, "color": "#c084fc"},
    "HR Basics":     {"count": 9000,  "color": "#7ec8e8"},
}


def _render_hr_model_comparison() -> None:
    """Full-page HR dataset Model Comparison sub-view."""
    # ── Back button ───────────────────────────────────────────────────────────
    if st.button("← Back to HR Practice", key="hr_mc_back_btn"):
        st.session_state.hr_show_mc = False
        st.rerun()

    st.markdown("""
<div style="text-align:center;padding:1rem 0 .5rem;">
  <div style="font-family:Orbitron,monospace;font-size:1.55rem;font-weight:800;
    background:linear-gradient(90deg,#00d4ff,#a5b4fc,#00ff88);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:.04em;">
    MODEL COMPARISON · HR DATASET
  </div>
  <div style="font-size:.8rem;color:#5a7a9a;font-family:Inter,sans-serif;margin-top:.3rem;">
    250,000 HR Interview Questions · 25 Top Companies · 9 Categories
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_companies, tab_benchmark, tab_categories = st.tabs([
        "🏢 Company Intelligence", "⚡ Run Benchmark (40 Q sample)", "📊 Category Breakdown"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    with tab_companies:
        st.markdown("""
<div style="background:rgba(0,212,255,.06);border:1px solid rgba(0,212,255,.18);
  border-radius:10px;padding:.65rem 1rem;margin-bottom:.9rem;font-size:.78rem;
  color:#7ec8e8;font-family:Inter,sans-serif;">
  📋 Dataset covers <strong style="color:#00d4ff;">250,000+</strong> real HR questions
  asked by top companies worldwide. Below are the 25 highest-frequency companies
  with their question counts, focus categories, and their most-asked HR question.
</div>""", unsafe_allow_html=True)

        # Sort by question count desc
        sorted_companies = sorted(_HR_COMPANY_DATA, key=lambda x: x["questions"], reverse=True)

        # Top 5 highlight cards
        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:.6rem;'
                    'color:#5a7a9a;letter-spacing:1.5px;text-transform:uppercase;'
                    'margin-bottom:.5rem;">🏆 Top 5 by HR Question Volume</div>',
                    unsafe_allow_html=True)
        top5_cols = st.columns(5)
        rank_colors = ["#ffd700", "#c0c0c0", "#cd7f32", "#00d4ff", "#00d4ff"]
        for i, (col, co) in enumerate(zip(top5_cols, sorted_companies[:5])):
            pct = int(co["questions"] / sorted_companies[0]["questions"] * 100)
            col.markdown(f"""
<div style="background:rgba(10,21,53,0.9);border:1px solid rgba(0,212,255,.15);
  border-radius:10px;padding:.7rem .6rem;text-align:center;">
  <div style="font-size:.55rem;color:{rank_colors[i]};font-family:Share Tech Mono,monospace;
    font-weight:700;letter-spacing:1px;">#{i+1}</div>
  <div style="font-family:Orbitron,monospace;font-size:.75rem;font-weight:800;
    color:#e8f0fe;margin:.2rem 0;">{co["company"]}</div>
  <div style="font-family:Orbitron,monospace;font-size:1rem;font-weight:900;
    color:{rank_colors[i]};">{co["questions"]:,}</div>
  <div style="font-size:.55rem;color:#5a7a9a;margin-top:.1rem;">questions</div>
  <div style="background:rgba(255,255,255,.05);border-radius:3px;height:3px;
    overflow:hidden;margin-top:.4rem;">
    <div style="width:{pct}%;height:100%;
      background:linear-gradient(90deg,{rank_colors[i]},#00d4ff);border-radius:3px;"></div>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Full company table
        st.markdown('<div style="font-family:Share Tech Mono,monospace;font-size:.6rem;'
                    'color:#5a7a9a;letter-spacing:1.5px;text-transform:uppercase;'
                    'margin-bottom:.4rem;">All 25 Companies</div>',
                    unsafe_allow_html=True)

        # Table header
        hcols = st.columns([.3, 1.2, .8, 1.6, 2.2])
        for lbl, col in zip(["#", "Company", "Questions", "Focus Categories", "Most-Asked HR Question"], hcols):
            col.markdown(f'<div style="font-size:.6rem;color:#a5b4fc;font-family:Share Tech Mono,'
                         f'monospace;padding:.2rem 0;border-bottom:1px solid rgba(165,180,252,.15);">'
                         f'{lbl}</div>', unsafe_allow_html=True)

        for rank, co in enumerate(sorted_companies, 1):
            bar_pct = int(co["questions"] / sorted_companies[0]["questions"] * 100)
            rc = hcols[0].__class__  # just for type hint clarity
            cols = st.columns([.3, 1.2, .8, 1.6, 2.2])
            cols[0].markdown(f'<div style="font-size:.68rem;color:#5a7a9a;font-family:Share Tech Mono,'
                              f'monospace;padding:.4rem 0;">{rank}</div>', unsafe_allow_html=True)
            cols[1].markdown(f'<div style="font-size:.75rem;font-weight:700;color:#e8f0fe;'
                              f'font-family:Inter,sans-serif;padding:.4rem 0;">{co["company"]}</div>',
                              unsafe_allow_html=True)
            cols[2].markdown(f"""
<div style="padding:.3rem 0;">
  <div style="font-family:Orbitron,monospace;font-size:.78rem;font-weight:700;color:#00d4ff;">
    {co["questions"]:,}</div>
  <div style="background:rgba(255,255,255,.05);border-radius:2px;height:3px;overflow:hidden;margin-top:3px;">
    <div style="width:{bar_pct}%;height:100%;background:linear-gradient(90deg,#00ff88,#00d4ff);
      border-radius:2px;"></div>
  </div>
</div>""", unsafe_allow_html=True)
            cols[3].markdown(f'<div style="font-size:.67rem;color:#94a3b8;font-family:Inter,sans-serif;'
                              f'padding:.4rem 0;line-height:1.4;">{co["categories"]}</div>',
                              unsafe_allow_html=True)
            cols[4].markdown(f'<div style="font-size:.68rem;color:#fbbf24;font-family:Inter,sans-serif;'
                              f'padding:.4rem 0;line-height:1.4;font-style:italic;">'
                              f'"{co["top_q"]}"</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    with tab_benchmark:
        _SCORER_COLORS = {
            "Keyword Match": "#7ec8e8", "TF-IDF": "#a5b4fc",
            "BM25": "#fbbf24", "SBERT": "#34d399", "Aura AI": "#00d4ff",
        }
        SCORERS_LIST = ["Keyword Match", "TF-IDF", "BM25", "SBERT", "Aura AI"]

        st.markdown("""
<div style="background:rgba(165,180,252,.06);border:1px solid rgba(165,180,252,.2);
  border-radius:10px;padding:.65rem 1rem;margin-bottom:.9rem;font-size:.78rem;
  color:#c4b5fd;font-family:Inter,sans-serif;">
  🎯 Pick any HR question from the 250k dataset, type your answer, and instantly see
  how all <strong>5 scoring models</strong> rate it — Keyword Match · TF-IDF · BM25 ·
  SBERT · Aura AI — with a full breakdown of why each model gave that score.
</div>""", unsafe_allow_html=True)

        # ── Load dataset (40 random Qs) ───────────────────────────────────────
        if "hr_mc_dataset" not in st.session_state or st.session_state.hr_mc_dataset is None:
            with st.spinner("Loading 40 random questions from dataset…"):
                try:
                    from model_comparison import load_dataset
                    ds, src, err = load_dataset()
                    import random as _rand
                    sample = _rand.sample(ds, min(40, len(ds)))
                    st.session_state.hr_mc_dataset     = sample
                    st.session_state.hr_mc_dataset_src = src
                except Exception as exc:
                    st.error(f"Dataset load failed: {exc}")
                    st.session_state.hr_mc_dataset = []

        dataset = st.session_state.get("hr_mc_dataset", [])

        col_reload, col_src = st.columns([1, 3])
        with col_reload:
            if st.button("🔄 Load new 40 questions", key="hr_mc_reload_btn",
                         use_container_width=True):
                st.session_state.hr_mc_dataset    = None
                st.session_state.hr_mc_live_result = None
                st.rerun()
        with col_src:
            src_lbl = st.session_state.get("hr_mc_dataset_src", "—")
            src_c   = {"local_json":"#00ff88","local_csv":"#00d4ff",
                       "kaggle":"#a5b4fc","embedded":"#fbbf24"}.get(src_lbl, "#94a3b8")
            st.markdown(f'<div style="font-size:.65rem;color:{src_c};font-family:Share Tech Mono,'
                        f'monospace;padding:.45rem 0;">dataset source: {src_lbl} · '
                        f'{len(dataset)} questions loaded</div>', unsafe_allow_html=True)

        if not dataset:
            st.info("No questions loaded. Click 'Load new 40 questions'.")
        else:
            # ── Question selector ─────────────────────────────────────────────
            st.markdown('<div style="height:.3rem;"></div>', unsafe_allow_html=True)
            q_labels = [f"Q{i+1} [{e.get('category','—')}] · {e['question'][:80]}{'…' if len(e['question'])>80 else ''}"
                        for i, e in enumerate(dataset)]
            sel_idx = st.selectbox("Select a question to answer",
                                   options=list(range(len(dataset))),
                                   format_func=lambda i: q_labels[i],
                                   key="hr_mc_q_sel")
            entry = dataset[sel_idx]

            # ── Company tags for this question's category ─────────────────────
            q_cat = entry.get("category", "General")
            _cat_companies = [
                co["company"] for co in _HR_COMPANY_DATA
                if q_cat in co["categories"]
            ][:3]

            company_chips = "".join(
                f'<span style="display:inline-block;background:rgba(0,212,255,.08);'
                f'border:1px solid rgba(0,212,255,.22);border-radius:5px;padding:2px 9px;'
                f'margin:0 3px;font-size:.6rem;color:#00d4ff;'
                f'font-family:Share Tech Mono,monospace;">🏢 {c}</span>'
                for c in _cat_companies
            ) if _cat_companies else '<span style="font-size:.6rem;color:#475569;">—</span>'

            # Question card with company tags
            st.markdown(f"""
<div style="background:rgba(10,21,53,.92);border:1px solid rgba(0,212,255,.2);
  border-radius:12px;padding:1rem 1.2rem;margin:.5rem 0;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem;flex-wrap:wrap;gap:.3rem;">
    <div style="font-size:.58rem;color:#5a7a9a;font-family:Share Tech Mono,monospace;
      letter-spacing:1.4px;text-transform:uppercase;">
      {q_cat} · {entry.get('difficulty','Medium')}</div>
    <div style="display:flex;align-items:center;gap:.2rem;flex-wrap:wrap;">
      <span style="font-size:.55rem;color:#475569;font-family:Share Tech Mono,monospace;
        margin-right:2px;">ASKED AT:</span>
      {company_chips}
    </div>
  </div>
  <div style="font-size:1.05rem;font-weight:700;color:#e8f0fe;font-family:Inter,sans-serif;
    line-height:1.45;margin-bottom:.6rem;">{entry['question']}</div>
  <div style="font-size:.68rem;color:#475569;font-family:Inter,sans-serif;">
    📝 Ideal keywords: <span style="color:#94a3b8;">
    {', '.join(entry.get('keywords', [])[:6]) or 'N/A'}</span>
  </div>
</div>""", unsafe_allow_html=True)

            # ── Groq key (hardcoded) ──────────────────────────────────────────
            groq_key_live = os.environ.get(
                "GROQ_API_KEY",
                os.environ.get("GROQ_API_KEY", "")
            )

            # ── Voice input + text area ───────────────────────────────────────
            ans_cache_key = f"hr_mc_cached_ans_{sel_idx}"
            stt_obj = st.session_state.get("hr_stt") or st.session_state.get("stt")

            if VOICE_OK and stt_obj is not None:
                st.markdown(
                    '<div style="font-size:.7rem;color:#5a7a9a;font-family:Share Tech Mono,'
                    'monospace;margin:.3rem 0 .15rem;letter-spacing:.8px;">'
                    '🎙 ANSWER — speak or type below</div>',
                    unsafe_allow_html=True,
                )
                voice_ans = _vip(stt_obj, 9800 + sel_idx)
                if voice_ans and voice_ans.strip() and not voice_ans.startswith("["):
                    st.session_state[ans_cache_key] = voice_ans.strip()
                user_answer = st.session_state.get(ans_cache_key, "")
            else:
                if VOICE_OK and stt_obj is None:
                    st.caption("⚠️ Voice STT not yet initialised — typing only.")
                elif not VOICE_OK:
                    st.caption("⚠️ Voice modules unavailable.")
                user_answer = st.text_area(
                    "Your Answer",
                    placeholder="Type your answer here. Use STAR / SOAR structure for best results…",
                    height=140, key=f"hr_mc_ans_{sel_idx}",
                    label_visibility="collapsed",
                )
                if user_answer:
                    st.session_state[ans_cache_key] = user_answer

            # Show cached answer preview when voice is active
            cached_ans = st.session_state.get(ans_cache_key, "")
            if VOICE_OK and stt_obj is not None and cached_ans:
                st.markdown(
                    f'<div style="background:rgba(0,212,255,.04);border:1px solid '
                    f'rgba(0,212,255,.12);border-radius:7px;padding:.5rem .8rem;'
                    f'font-size:.72rem;color:#94a3b8;font-family:Inter,sans-serif;'
                    f'margin-bottom:.4rem;max-height:80px;overflow-y:auto;">'
                    f'{cached_ans[:300]}{"…" if len(cached_ans)>300 else ""}</div>',
                    unsafe_allow_html=True,
                )

            analyse_col, clear_col = st.columns([4, 1])
            with analyse_col:
                analyse_btn = st.button("◈  ANALYSE WITH ALL 5 MODELS",
                                        key="hr_mc_analyse_btn", use_container_width=True)
            with clear_col:
                if st.button("✕ Clear", key="hr_mc_clear_btn", use_container_width=True):
                    st.session_state.hr_mc_live_result = None
                    st.session_state.pop(ans_cache_key, None)
                    st.rerun()

            final_ans = cached_ans if (VOICE_OK and stt_obj is not None) else user_answer

            if analyse_btn and final_ans.strip():
                with st.spinner("Scoring your answer across 5 models…"):
                    try:
                        from model_comparison import (
                            score_keyword_match, score_tfidf,
                            score_bm25, score_sbert, score_aura,
                        )
                        try:
                            from model_comparison import _aura_subscores
                            _has_subscores = True
                        except ImportError:
                            _has_subscores = False

                        ideal = entry["ideal"]
                        kws   = entry.get("keywords", [])
                        cat   = entry.get("category", "")
                        ans   = final_ans.strip()
                        gk    = groq_key_live.strip()

                        kw_score    = score_keyword_match(ans, ideal, kws)
                        tfidf_score = score_tfidf(ans, ideal)
                        bm25_score  = score_bm25(ans, ideal)
                        sbert_score = score_sbert(ans, ideal)
                        aura_score  = score_aura(ans, ideal, kws, category=cat, groq_api_key=gk)

                        # Rich sub-scores for breakdown panel
                        sub = (_aura_subscores(ans, ideal, kws, category=cat, groq_api_key=gk)
                               if _has_subscores else {})

                        import re as _re_mc2
                        _STAR_PATS = {
                            "Situation": r"\b(situation|context|when|once|during|while|in my)\b",
                            "Task":      r"\b(task|goal|objective|needed to|had to|my role|was asked)\b",
                            "Action":    r"\b(i did|i took|i used|implemented|built|led|developed|created)\b",
                            "Result":    r"\b(result|outcome|achieved|improved|reduced|delivered|increased)\b",
                        }
                        star_hits  = {k: bool(_re_mc2.search(p, ans.lower()))
                                      for k, p in _STAR_PATS.items()}
                        star_count = sum(star_hits.values())
                        kw_hits    = [k for k in kws if k.lower() in ans.lower()]
                        wc         = len(ans.split())
                        fillers    = ["um","uh","like","basically","actually","you know","kind of","sort of"]
                        fill_cnt   = sum(ans.lower().split().count(f) for f in fillers)

                        # Model disagreement
                        all_scores = [kw_score, tfidf_score, bm25_score, sbert_score, aura_score]
                        import statistics as _stats
                        spread = round(_stats.stdev(all_scores), 1) if len(all_scores) > 1 else 0.0

                        st.session_state.hr_mc_live_result = {
                            "scores": {
                                "Keyword Match": kw_score,
                                "TF-IDF":        tfidf_score,
                                "BM25":          bm25_score,
                                "SBERT":         sbert_score,
                                "Aura AI":       aura_score,
                            },
                            "subscores":  sub,
                            "star_hits":  star_hits,
                            "star_count": star_count,
                            "kw_hits":    kw_hits,
                            "kw_total":   kws,
                            "word_count": wc,
                            "fill_count": fill_cnt,
                            "spread":     spread,
                            "ideal":      ideal,
                            "answer":     ans,
                            "question":   entry["question"],
                            "category":   cat,
                        }
                    except Exception as exc:
                        st.error(f"Scoring failed: {exc}")

            elif analyse_btn and not final_ans.strip():
                st.warning("Please provide your answer first (type or speak).")

            # ── Results display ───────────────────────────────────────────────
            live_res = st.session_state.get("hr_mc_live_result")
            if live_res and live_res.get("question") == entry["question"]:
                scores     = live_res["scores"]
                sub        = live_res.get("subscores", {})
                star_hits  = live_res["star_hits"]
                star_count = live_res["star_count"]
                kw_hits    = live_res["kw_hits"]
                kw_total   = live_res["kw_total"]
                wc         = live_res["word_count"]
                fill_cnt   = live_res["fill_count"]
                spread     = live_res.get("spread", 0)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
<div style="font-family:Orbitron,monospace;font-size:.72rem;color:#00d4ff;
  letter-spacing:.1em;padding:.35rem 0;border-bottom:1px solid rgba(0,212,255,.15);
  margin-bottom:.8rem;">◈ SCORE BREAKDOWN — ALL 5 MODELS</div>""",
                            unsafe_allow_html=True)

                # ── Model disagreement banner ─────────────────────────────────
                spread_c = "#fbbf24" if spread > 20 else "#00ff88" if spread < 10 else "#7ec8e8"
                spread_msg = ("High disagreement — models weight different signals" if spread > 20
                              else "Moderate spread — consistent across methods" if spread > 10
                              else "Strong consensus — all models agree")
                st.markdown(f"""
<div style="background:rgba(10,21,53,.8);border:1px solid rgba(0,212,255,.1);
  border-radius:8px;padding:.4rem .9rem;margin-bottom:.7rem;
  display:flex;justify-content:space-between;align-items:center;">
  <span style="font-size:.65rem;color:#5a7a9a;font-family:Share Tech Mono,monospace;">
    MODEL DISAGREEMENT (σ)</span>
  <span style="font-family:Orbitron,monospace;font-size:.8rem;font-weight:700;color:{spread_c};">
    σ={spread:.1f} &nbsp;·&nbsp; <span style="font-size:.6rem;color:{spread_c};">{spread_msg}</span>
  </span>
</div>""", unsafe_allow_html=True)

                # ── 5 model score cards ───────────────────────────────────────
                score_cols = st.columns(5)
                winner = max(scores, key=lambda s: scores[s])
                for i, (col, sc_name) in enumerate(zip(score_cols, SCORERS_LIST)):
                    val   = scores[sc_name]
                    c     = _SCORER_COLORS[sc_name]
                    bar   = min(100, val)
                    is_w  = (sc_name == winner)
                    bg    = "rgba(0,212,255,.09)" if is_w else "rgba(10,21,53,.88)"
                    border= "rgba(0,212,255,.35)" if is_w else "rgba(255,255,255,.07)"
                    grade = ("A+" if val>=90 else "A" if val>=80 else "B+" if val>=70
                             else "B" if val>=60 else "C+" if val>=50 else "C" if val>=40
                             else "D")
                    col.markdown(f"""
<div style="background:{bg};border:1px solid {border};border-radius:11px;
  padding:.7rem .5rem;text-align:center;">
  <div style="font-size:.58rem;color:{c};font-family:Share Tech Mono,monospace;
    margin-bottom:.2rem;">{'👑 ' if is_w else ''}{sc_name}</div>
  <div style="font-family:Orbitron,monospace;font-size:1.4rem;font-weight:900;
    color:{c};">{val:.0f}</div>
  <div style="font-size:.52rem;color:#5a7a9a;margin-bottom:.3rem;">/100</div>
  <div style="background:rgba(255,255,255,.06);border-radius:3px;height:4px;
    overflow:hidden;margin-bottom:.3rem;">
    <div style="width:{bar:.0f}%;height:100%;background:{c};border-radius:3px;"></div>
  </div>
  <div style="font-size:.68rem;font-weight:700;color:{c};">{grade}</div>
</div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Three-column breakdown ────────────────────────────────────
                left_col, mid_col, right_col = st.columns([1, 1, 1])

                with left_col:
                    # STAR structure
                    st.markdown('<div style="font-size:.62rem;color:#5a7a9a;font-family:'
                                'Share Tech Mono,monospace;letter-spacing:1.2px;'
                                'text-transform:uppercase;margin-bottom:.4rem;">'
                                '📐 STAR Structure</div>', unsafe_allow_html=True)
                    star_row = st.columns(4)
                    for i, (comp, hit) in enumerate(star_hits.items()):
                        c_hit = "#00ff88" if hit else "#475569"
                        star_row[i].markdown(f"""
<div style="background:{'rgba(0,255,136,.07)' if hit else 'rgba(255,255,255,.03)'};
  border:1px solid {'rgba(0,255,136,.25)' if hit else 'rgba(255,255,255,.06)'};
  border-radius:7px;padding:.4rem .2rem;text-align:center;">
  <div style="font-size:.72rem;color:{c_hit};">{'✓' if hit else '✕'}</div>
  <div style="font-size:.5rem;color:{c_hit};font-family:Share Tech Mono,monospace;
    margin-top:.1rem;">{comp[:3].upper()}</div>
</div>""", unsafe_allow_html=True)

                    # Quick stats grid
                    _stat_rows = ''.join(
                        f'<div style="display:flex;justify-content:space-between;font-size:.68rem;'
                        f'color:#94a3b8;font-family:Inter,sans-serif;margin-bottom:.25rem;">'
                        f'<span>{lbl}</span>'
                        f'<span style="color:{vc};">{val_}</span>'
                        f'</div>'
                        for lbl, vc, val_ in [
                            ("STAR", '#00ff88' if star_count>=3 else '#fbbf24' if star_count>=2 else '#fca5a5', f"{star_count}/4"),
                            ("Keywords", '#00ff88' if len(kw_hits)>=3 else '#fbbf24' if len(kw_hits)>=1 else '#fca5a5', f"{len(kw_hits)}/{len(kw_total) or '—'}"),
                            ("Words", '#00ff88' if 80<=wc<=350 else '#fbbf24', str(wc)),
                            ("Fillers", '#fca5a5' if fill_cnt>3 else '#00ff88', str(fill_cnt)),
                            ("Quant facts", '#00ff88' if sub.get('quant_hits',0)>=2 else '#fbbf24', str(sub.get('quant_hits','—'))),
                            ("Connectors", '#00ff88' if sub.get('disc_hits',0)>=3 else '#fbbf24', str(sub.get('disc_hits','—'))),
                        ]
                    )
                    st.markdown(f"""
<div style="background:rgba(10,21,53,.8);border:1px solid rgba(0,212,255,.1);
  border-radius:8px;padding:.5rem .7rem;margin-top:.4rem;">
  {_stat_rows}
</div>""", unsafe_allow_html=True)

                    # Keywords chips
                    if kw_total:
                        st.markdown('<div style="font-size:.58rem;color:#5a7a9a;'
                                    'font-family:Share Tech Mono,monospace;'
                                    'text-transform:uppercase;margin:.5rem 0 .25rem;">'
                                    'Keywords</div>', unsafe_allow_html=True)
                        kw_html = ""
                        for kw in kw_total:
                            hit   = kw.lower() in live_res["answer"].lower()
                            kw_c  = "#00ff88" if hit else "#475569"
                            kw_bg = "rgba(0,255,136,.08)" if hit else "rgba(255,255,255,.03)"
                            kw_bd = "rgba(0,255,136,.2)"  if hit else "rgba(255,255,255,.06)"
                            kw_html += (f'<span style="display:inline-block;background:{kw_bg};'
                                        f'border:1px solid {kw_bd};border-radius:5px;'
                                        f'padding:2px 7px;margin:2px;font-size:.6rem;color:{kw_c};'
                                        f'font-family:Share Tech Mono,monospace;">'
                                        f'{"✓" if hit else "✕"} {kw}</span>')
                        st.markdown(kw_html, unsafe_allow_html=True)

                with mid_col:
                    # Richer Aura sub-score radar
                    st.markdown('<div style="font-size:.62rem;color:#5a7a9a;font-family:'
                                'Share Tech Mono,monospace;letter-spacing:1.2px;'
                                'text-transform:uppercase;margin-bottom:.4rem;">'
                                '◈ Aura AI Sub-Scores</div>', unsafe_allow_html=True)

                    _sub_labels = [
                        ("Relevance",    sub.get("rel", 0),          "30%", "#00d4ff"),
                        ("Keywords",     sub.get("kw",  0),          "18%", "#7ec8e8"),
                        ("STAR",         sub.get("star",0),          "15%", "#00ff88"),
                        ("Quantified",   sub.get("quant",0),         " 8%", "#fbbf24"),
                        ("Vocabulary",   sub.get("ttr", 0),          " 7%", "#a5b4fc"),
                        ("Connectors",   sub.get("discourse",0),     " 7%", "#34d399"),
                        ("Coherence",    sub.get("coherence",0),     " 5%", "#f472b6"),
                        ("Depth/Fluency",sub.get("depth_flu",0),     " 6%", "#fb923c"),
                        ("Active Voice", sub.get("active_voice",0),  " 4%", "#c084fc"),
                    ]
                    for lbl, val_, wt, c in _sub_labels:
                        bar_w = int(val_ * 100)
                        pct_v = int(val_ * 100)
                        st.markdown(f"""
<div style="margin-bottom:.3rem;">
  <div style="display:flex;justify-content:space-between;font-size:.62rem;
    color:#94a3b8;font-family:Inter,sans-serif;margin-bottom:.15rem;">
    <span>{lbl} <span style="color:#475569;font-size:.55rem;">({wt})</span></span>
    <span style="color:{c};font-weight:700;">{pct_v}</span>
  </div>
  <div style="background:rgba(255,255,255,.05);border-radius:3px;height:5px;overflow:hidden;">
    <div style="width:{bar_w}%;height:100%;background:{c};border-radius:3px;opacity:.85;"></div>
  </div>
</div>""", unsafe_allow_html=True)

                with right_col:
                    # Model-by-model explanation
                    st.markdown('<div style="font-size:.62rem;color:#5a7a9a;font-family:'
                                'Share Tech Mono,monospace;letter-spacing:1.2px;'
                                'text-transform:uppercase;margin-bottom:.4rem;">'
                                '🔬 What Each Model Measures</div>', unsafe_allow_html=True)

                    model_descriptions = {
                        "Keyword Match": ("Counts word overlap with the ideal answer. Fast but literal — "
                                          "misses meaning, rewards keyword repetition."),
                        "TF-IDF":        ("TF-IDF cosine similarity — weights rare, specific words higher. "
                                          "Better than keyword match but still bag-of-words."),
                        "BM25":          ("Okapi BM25 — IR gold standard. Penalises overly long or short "
                                          "answers, rewards concise relevant terms."),
                        "SBERT":         ("Sentence-BERT semantic embeddings — understands meaning & "
                                          "paraphrases. Highest semantic awareness of the baselines."),
                        "Aura AI":       ("9-signal composite: relevance · keywords · STAR · "
                                          "quantification · vocabulary · discourse · coherence · "
                                          "depth/fluency · active voice. Most holistic."),
                    }
                    for sc_name in SCORERS_LIST:
                        val_  = scores[sc_name]
                        c     = _SCORER_COLORS[sc_name]
                        bar_w = min(100, val_)
                        st.markdown(f"""
<div style="background:rgba(10,21,53,.85);border:1px solid rgba(0,212,255,.08);
  border-radius:8px;padding:.45rem .7rem;margin-bottom:.3rem;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.15rem;">
    <div style="font-size:.65rem;font-weight:700;color:{c};font-family:Inter,sans-serif;">
      {sc_name}</div>
    <div style="font-family:Orbitron,monospace;font-size:.82rem;font-weight:800;color:{c};">
      {val_:.0f}<span style="font-size:.48rem;color:#475569;">/100</span></div>
  </div>
  <div style="background:rgba(255,255,255,.05);border-radius:2px;height:3px;
    overflow:hidden;margin-bottom:.25rem;">
    <div style="width:{bar_w:.0f}%;height:100%;background:{c};border-radius:2px;"></div>
  </div>
  <div style="font-size:.6rem;color:#64748b;font-family:Inter,sans-serif;line-height:1.35;">
    {model_descriptions[sc_name]}</div>
</div>""", unsafe_allow_html=True)

                # ── Ideal answer reveal ───────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📖 View Ideal Answer"):
                    st.markdown(f"""
<div style="background:rgba(0,255,136,.04);border:1px solid rgba(0,255,136,.15);
  border-radius:8px;padding:.8rem 1rem;font-size:.78rem;color:#94a3b8;
  font-family:Inter,sans-serif;line-height:1.6;">{live_res['ideal']}</div>""",
                                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    with tab_categories:
        st.markdown("""
<div style="background:rgba(0,255,136,.05);border:1px solid rgba(0,255,136,.15);
  border-radius:10px;padding:.65rem 1rem;margin-bottom:.9rem;font-size:.78rem;
  color:#94a3b8;font-family:Inter,sans-serif;">
  📊 Full breakdown of all <strong style="color:#00ff88;">250,000 HR questions</strong>
  across 9 categories. Each category shows its question count and the most common
  companies asking them.
</div>""", unsafe_allow_html=True)

        total_qs = sum(v["count"] for v in _CATEGORY_STATS.values())
        # FIX: x is a (key, value) tuple from .items() — use x[1]["count"]
        for cat, data in sorted(_CATEGORY_STATS.items(), key=lambda x: -x[1]["count"]):
            pct   = data["count"] / total_qs * 100
            color = data["color"]
            matching = [co["company"] for co in _HR_COMPANY_DATA
                        if cat in co["categories"]][:5]
            companies_str = ", ".join(matching) if matching else "Multiple"

            st.markdown(f"""
<div style="background:rgba(10,21,53,.88);border:1px solid rgba(0,212,255,.1);
  border-radius:8px;padding:.6rem .9rem;margin-bottom:.4rem;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.3rem;">
    <div style="font-size:.78rem;font-weight:700;color:{color};font-family:Inter,sans-serif;">
      {cat}</div>
    <div style="font-family:Orbitron,monospace;font-size:.8rem;font-weight:700;color:{color};">
      {data["count"]:,} <span style="font-size:.55rem;color:#5a7a9a;">({pct:.1f}%)</span></div>
  </div>
  <div style="background:rgba(255,255,255,.05);border-radius:3px;height:5px;
    overflow:hidden;margin-bottom:.35rem;">
    <div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:3px;opacity:.8;">
    </div>
  </div>
  <div style="font-size:.65rem;color:#5a7a9a;font-family:Inter,sans-serif;">
    🏢 Top companies: <span style="color:#94a3b8;">{companies_str}</span>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown(f"""
<div style="background:rgba(0,212,255,.06);border:1px solid rgba(0,212,255,.2);
  border-radius:8px;padding:.6rem 1rem;margin-top:.6rem;text-align:center;">
  <span style="font-family:Orbitron,monospace;font-size:1.1rem;font-weight:900;color:#00d4ff;">
    {total_qs:,}</span>
  <span style="font-size:.72rem;color:#7ec8e8;font-family:Inter,sans-serif;margin-left:.4rem;">
    Total HR Questions · 25 Companies · 9 Categories</span>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PAGE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def page_hr_round() -> None:
    """Called by app.py router when page == 'HR Practice'."""
    _inject_hr_css()
    _init_hr_state()

    # ── Model Comparison sub-page ─────────────────────────────────────────────
    if st.session_state.hr_show_mc:
        _render_hr_model_comparison()
        return

    if not st.session_state.hr_started:
        _render_setup()
        return

    if st.session_state.hr_finished:
        _render_final_report()
        return

    qs    = _active_questions()
    total = len(qs)
    idx   = st.session_state.hr_q_index

    if idx >= total:
        st.session_state.hr_finished = True
        st.rerun()
        return

    q = qs[idx]

    # If there's a pending evaluation result to show, render it
    pending = st.session_state.hr_pending_eval
    if pending and pending["q_id"] == q["id"]:
        # Show question + eval result
        _render_question(q, idx, total)
        _render_eval_result(pending, total, idx)
    else:
        _render_question(q, idx, total)

    # Sidebar quick-jump
    with st.sidebar:
        st.markdown('<div style="height:1px;background:rgba(255,255,255,.06);margin:.3rem 0;"></div>',
                    unsafe_allow_html=True)
        st.markdown('<span style="font-size:.58rem;color:#5a7a9a;'
                    'text-transform:uppercase;letter-spacing:1.5px;'
                    'font-family:Share Tech Mono,monospace;">HR Practice</span>',
                    unsafe_allow_html=True)
        answered_ids = {a["q_id"] for a in st.session_state.hr_answers
                        if a["answer"] != "[Skipped]"}
        skipped_ids  = {a["q_id"] for a in st.session_state.hr_answers
                        if a["answer"] == "[Skipped]"}
        prog_pct = int(len(st.session_state.hr_answers) / max(total,1) * 100)
        st.markdown(f"""
<div style="background:rgba(10,25,47,.92);border:1px solid rgba(0,212,255,.1);
  border-radius:7px;padding:.55rem .7rem;margin-bottom:.3rem;">
  <div style="display:flex;justify-content:space-between;font-size:.62rem;
    color:#5a7a9a;font-family:Share Tech Mono,monospace;margin-bottom:4px;">
    <span>Progress</span><span>{len(st.session_state.hr_answers)}/{total}</span>
  </div>
  <div style="background:rgba(255,255,255,.05);border-radius:3px;height:4px;overflow:hidden;">
    <div style="width:{prog_pct}%;height:100%;
      background:linear-gradient(90deg,#00ff88,#00d4ff);border-radius:3px;"></div>
  </div>
</div>""", unsafe_allow_html=True)

        if st.button("🏁 Finish Early & Report", key="hr_early_finish",
                     use_container_width=True):
            st.session_state.hr_finished = True
            st.rerun()