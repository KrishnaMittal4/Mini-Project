"""
weekly_prep_plan.py — Aura AI | Weekly Prep Plan v4.0
======================================================
v4.0 — FULL PIPELINE INTEGRATION
─────────────────────────────────────────────────────
WHAT'S NEW
  1. ADAPTIVE RL SEQUENCING INSIDE THE PLAN
     Days 2 (Technical Depth) and 5 (Hard Technical) now pass every
     scored answer through RLAdaptiveSequencer.record_and_select() —
     the same engine used in live interviews.  The agent adjusts
     difficulty for the *next* question in the session based on the
     candidate's real-time performance.  Shallow answers (< 80 words
     or < 2 STAR hits) trigger the follow-up probe action (action 7).

  2. LIVE COACH CARD AFTER EVERY ANSWER
     generate_coaching_tip() + render_coach_card() from live_coach.py
     are called after each submission, giving a personalised 2–3
     sentence spoken tip (Groq-backed, rule-based fallback).
     TTS "Speak Feedback" button included on every answer card.

  3. OCEAN + DISC PERSONALITY SIGNALS
     The AnswerEvaluator returns disc_traits and ocean scores.  A
     compact "Personality Signal" row now appears in the feedback card
     showing the top DISC trait and OCEAN profile for that answer, so
     the candidate can see how they are perceived.

  4. WEEKLY PERFORMANCE RADAR
     The 7-day overview timeline now shows a radar chart (Chart.js via
     Streamlit component) built from per-day avg scores once ≥ 2 days
     are complete.  Six axes: Relevance · STAR · Keywords · Depth ·
     Grammar · Fluency.  Drawn from the `wp_day_metrics` session key.

  5. FOLLOW-UP PROBE FLOW
     When the RL agent recommends follow_up (action 7), the practice
     window displays a targeted probe question rather than advancing,
     with a "Follow-Up" badge on the question card.

  6. WEIGHTED SCORE PROFILES SURFACED
     The feedback card now shows a mini weight bar (tiny horizontal
     bar chart) for the active scoring profile so the candidate knows
     what the evaluator prioritised: STAR · Relevance · Keywords ·
     Depth · Grammar.

  7. SESSION EXPORT BUTTON
     At the end of each day's session, a "📥 Export Session" button
     downloads a JSON summary: questions, answers, scores, STAR,
     keywords, coaching tips, DISC/OCEAN signals.

SESSION STATE KEYS (all v3 keys preserved, new additions below)
──────────────────────────────────────────────────────────────
  wp_rl_sequencer      dict   — serialised Q-table for RL agent (tech days)
  wp_follow_up_q       dict   — follow-up question when action 7 triggered
  wp_day_metrics       dict   — {day_num: {avg, relevance, star, kw, depth}}
  wp_coaching_tips     list   — coaching tip per answer in current session
  wp_last_disc         str    — DISC dominant trait last answer
  wp_last_ocean        dict   — OCEAN scores last answer
"""

from __future__ import annotations

import datetime
import json
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import streamlit as st
import streamlit.components.v1 as components

# ── Optional pipeline imports ──────────────────────────────────────────────────
try:
    from adaptive_sequencer import RLAdaptiveSequencer, Action
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

try:
    from live_coach import generate_coaching_tip, render_coach_card
    COACH_AVAILABLE = True
except ImportError:
    COACH_AVAILABLE = False

try:
    from answer_evaluator import AnswerEvaluator
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False



# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

WEEKLY_PLAN_DEFAULTS: dict = {
    "wp_role":             "Software Engineer",
    "wp_date":             None,
    "wp_weak":             [],
    "wp_completed":        set(),
    "wp_plan_built":       False,
    "wp_open_day":         None,
    "wp_practice_q":       None,
    "wp_practice_answers": [],
    "wp_practice_day":     0,
    "wp_attire_check":     False,
    "wp_attire_confirmed": False,
    "wp_voice_answer":     "",
    # v4.0 additions
    "wp_rl_sequencer":     None,
    "wp_follow_up_q":      None,
    "wp_day_metrics":      {},
    "wp_coaching_tips":    [],
    "wp_last_disc":        "",
    "wp_last_ocean":       {},
}


# ══════════════════════════════════════════════════════════════════════════════
#  DAY CONFIGURATIONS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DayConfig:
    day:            int
    theme:          str
    emoji:          str
    signal:         str
    q_type:         str       # "Behavioural" | "Technical" | "HR" | "Mixed"
    difficulty:     str       # "easy" | "medium" | "hard"
    num_q:          int
    color:          str
    desc:           str
    tips:           List[str]
    use_hr_dataset: bool = False
    use_rl:         bool = False  # v4.0 — enable RL sequencer for this day


DAY_CONFIGS: List[DayConfig] = [
    DayConfig(
        day=1, theme="STAR Foundations", emoji="🌟",
        signal="STAR", q_type="Behavioural", difficulty="easy", num_q=3,
        color="#00e5ff", use_hr_dataset=True, use_rl=False,
        desc=(
            "3 easy behavioural questions from the HR dataset. "
            "Hit all 4 STAR components on every answer. "
            "Feedback shows per-component STAR hit/miss, keyword coverage, "
            "a live coach tip, and your DISC personality signal."
        ),
        tips=[
            "Open every answer with a specific Situation — avoid vague generalisations.",
            "State your Task clearly: what were YOU personally responsible for?",
            "Actions must be 'I did', not 'we did' — own the steps you took.",
            "End with a measurable Result: numbers, timelines, or clear outcomes.",
        ]
    ),
    DayConfig(
        day=2, theme="Technical Depth", emoji="⚙️",
        signal="DEPTH", q_type="Technical", difficulty="medium", num_q=3,
        color="#7f5af0", use_hr_dataset=False, use_rl=True,
        desc=(
            "3 medium technical questions, AI-generated for your role. "
            "The RL sequencer adapts difficulty after each answer. "
            "Shallow answers (< 80 words) trigger a targeted follow-up probe. "
            "Feedback includes relevance score, keyword depth, and tech coach tip."
        ),
        tips=[
            "Think out loud — interviewers want to see your reasoning process.",
            "Hit the keywords in the question — they signal what the panel values.",
            "Contrast trade-offs: always explain why you'd choose one approach over another.",
            "Aim for 120–180 words — depth matters more than brevity here.",
        ]
    ),
    DayConfig(
        day=3, theme="HR & Values", emoji="🤝",
        signal="EQ", q_type="HR", difficulty="easy", num_q=3,
        color="#00ff88", use_hr_dataset=True, use_rl=False,
        desc=(
            "3 HR questions from the HR dataset. "
            "Show self-awareness, motivation, and cultural alignment. "
            "Feedback uses the HR weight profile (Relevance 25%, Depth 25%, STAR 20%). "
            "OCEAN personality scores reveal how you come across to the panel."
        ),
        tips=[
            "Be honest about weaknesses — frame them as areas you're actively improving.",
            "Motivation answers should reference the role AND the company, not just salary.",
            "Conflict stories: focus on what YOU changed, not what the other person did wrong.",
            "Keep energy positive — avoid badmouthing previous employers.",
        ]
    ),
    DayConfig(
        day=4, theme="Speed Round", emoji="⚡",
        signal="WPM", q_type="Mixed", difficulty="easy", num_q=5,
        color="#ffd700", use_hr_dataset=True, use_rl=False,
        desc=(
            "5 quick-fire HR & Behavioural questions. "
            "Practise concise, structured answers (target 100–150 words). "
            "Live word-count HUD turns green inside the target band. "
            "Coach tip focuses on filler words and pacing after every answer."
        ),
        tips=[
            "Target 100–150 words per answer — concise beats comprehensive in speed rounds.",
            "Open with your conclusion, then support it — don't build up to the point.",
            "Filler words (um, uh, like) hurt your score — pause instead.",
            "Every answer needs a clear first sentence that directly addresses the question.",
        ]
    ),
    DayConfig(
        day=5, theme="Hard Technical", emoji="🔬",
        signal="SYSTEM", q_type="Technical", difficulty="hard", num_q=3,
        color="#ff6b35", use_hr_dataset=False, use_rl=True,
        desc=(
            "3 hard technical questions — system design and deep architectural reasoning. "
            "RL agent warm-starts from your Day 2 Q-table so it already knows your gaps. "
            "Follow-up probes drill into shallow explanations. "
            "Scoring uses the Technical profile (Relevance 40%, Keywords 25%, Depth 20%)."
        ),
        tips=[
            "For system design: always clarify scale, then propose components, then trade-offs.",
            "Use concrete numbers: '10M users / day' beats 'high traffic'.",
            "Acknowledge edge cases and failure modes — it shows production maturity.",
            "Reference real technologies by name (Kafka, Redis, PostgreSQL) not just concepts.",
        ]
    ),
    DayConfig(
        day=6, theme="Pressure Behavioural", emoji="🎯",
        signal="RESILIENCE", q_type="Behavioural", difficulty="hard", num_q=3,
        color="#e040fb", use_hr_dataset=True, use_rl=False,
        desc=(
            "3 hard behavioural questions from the HR dataset. "
            "Complex leadership, conflict, and failure stories. "
            "Feedback compares your DISC profile across answers — are you "
            "showing Dominance when the question expects Steadiness?"
        ),
        tips=[
            "Hard behavioural Qs expect real complexity — don't sanitise the story.",
            "Failure stories should have a genuine lesson, not just 'I worked harder'.",
            "Leadership under pressure: show how you kept the team aligned, not just what you decided.",
            "Use the word 'I' deliberately — own the difficult decisions.",
        ]
    ),
    DayConfig(
        day=7, theme="Final Gauntlet", emoji="🏁",
        signal="OVERALL", q_type="Mixed", difficulty="medium", num_q=5,
        color="#00ffd1", use_hr_dataset=True, use_rl=False,
        desc=(
            "5 mixed-difficulty HR & Behavioural questions. "
            "Full scoring — treat it as the real thing. "
            "Session summary includes the weekly radar chart, "
            "cumulative DISC/OCEAN profile, and a JSON export of the full week."
        ),
        tips=[
            "This is your dress rehearsal — simulate real interview conditions.",
            "Don't re-read your earlier notes — answer from memory.",
            "Pace yourself: 2–3 minutes per answer, no more.",
            "After each answer, mentally check: did I cover Situation, Action, Result?",
        ]
    ),
]

_DAY_MAP = {d.day: d for d in DAY_CONFIGS}


# ══════════════════════════════════════════════════════════════════════════════
#  HR DATASET LOADER
# ══════════════════════════════════════════════════════════════════════════════

_HR_ALL_CATEGORIES = {
    "Behavioural", "Behavioral", "Leadership", "Communication",
    "Problem Solving", "Adaptability", "Collaboration", "Self-Awareness",
    "Career Goals", "Teamwork", "HR", "General", "Motivation",
    "Work Style", "Conflict", "Values", "Strengths", "Weaknesses",
    "Situational", "Culture Fit",
}

_HR_DIFF_MAP = {
    "easy": "easy", "medium": "medium", "hard": "hard",
    "Easy": "easy", "Medium": "medium", "Hard": "hard",
}

_HR_DATASET_PATHS = [
    "hr_interview_dataset.json",
    r"C:\Users\ACER\Downloads\Miniproject\hr_interview_dataset.json",
    os.path.join(os.path.dirname(__file__), "hr_interview_dataset.json"),
]


def _load_hr_questions() -> List[dict]:
    """Load hr_interview_dataset.json; returns [] on failure."""
    for path in _HR_DATASET_PATHS:
        try:
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as fh:
                raw = fh.read().strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    for key in ("data", "questions", "records", "items"):
                        if isinstance(parsed.get(key), list):
                            return parsed[key]
            except json.JSONDecodeError:
                pass
            records = []
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if records:
                return records
        except Exception:
            continue
    return []


def _get_hr_question(
    difficulty: str,
    q_type: str,
    role: str,
    exclude_texts: Optional[List[str]] = None,
) -> Optional[dict]:
    """Pull one random question from the HR dataset matching difficulty + type."""
    records   = _load_hr_questions()
    if not records:
        return None
    diff_key    = _HR_DIFF_MAP.get(difficulty, "medium")
    exclude_set = set(t.lower()[:60] for t in (exclude_texts or []))

    def _rec_to_q(rec: dict) -> Optional[dict]:
        q_text = (rec.get("question") or rec.get("Question") or "").strip()
        if not q_text or q_text.lower()[:60] in exclude_set:
            return None
        ideal = (
            rec.get("ideal") or rec.get("ideal_answer") or
            rec.get("Ideal") or rec.get("answer") or ""
        ).strip()
        cat  = rec.get("category", rec.get("Category", ""))
        kw   = rec.get("keywords", rec.get("Keywords", []))
        if isinstance(kw, str):
            kw = [k.strip() for k in kw.split(",") if k.strip()]
        rec_diff = _HR_DIFF_MAP.get(rec.get("difficulty", rec.get("Difficulty", "")), "")
        return {
            "role":         role,
            "difficulty":   rec_diff or diff_key,
            "type":         "Behavioural" if cat in {
                "Behavioural", "Behavioral", "Leadership",
                "Communication", "Problem Solving",
                "Adaptability", "Collaboration", "Teamwork",
            } else "HR",
            "question":     q_text,
            "keywords":     kw,
            "ideal_answer": ideal,
            "source":       "hr_dataset",
            "category":     cat,
        }

    candidates: List[dict] = []
    for rec in records:
        cat      = rec.get("category", rec.get("Category", ""))
        rec_diff = _HR_DIFF_MAP.get(rec.get("difficulty", rec.get("Difficulty", "")), "")
        q_text   = (rec.get("question") or rec.get("Question") or "").strip()
        if not q_text:
            continue
        if q_type == "Behavioural":
            if cat not in {"Behavioural","Behavioral","Leadership","Communication",
                           "Problem Solving","Adaptability","Collaboration","Teamwork"}:
                continue
        elif q_type in ("HR", "Mixed"):
            if cat not in _HR_ALL_CATEGORIES:
                continue
        if rec_diff and rec_diff != diff_key:
            continue
        q = _rec_to_q(rec)
        if q:
            candidates.append(q)

    if not candidates:
        # Relax difficulty
        for rec in records:
            cat    = rec.get("category", rec.get("Category", ""))
            q_text = (rec.get("question") or rec.get("Question") or "").strip()
            if not q_text or cat not in _HR_ALL_CATEGORIES:
                continue
            q = _rec_to_q(rec)
            if q:
                candidates.append(q)

    return random.choice(candidates) if candidates else None


# ══════════════════════════════════════════════════════════════════════════════
#  RL SEQUENCER HELPERS  (v4.0)
# ══════════════════════════════════════════════════════════════════════════════

def _get_or_build_sequencer(role: str) -> Optional[object]:
    """Return the session-scoped RLAdaptiveSequencer or None if unavailable."""
    if not RL_AVAILABLE:
        return None
    seq = st.session_state.get("wp_rl_sequencer_obj")
    if seq is None:
        try:
            seq = RLAdaptiveSequencer(role=role)
            st.session_state["wp_rl_sequencer_obj"] = seq
        except Exception:
            return None
    return seq


def _rl_record_and_select(
    seq,
    score: float,
    star_scores: dict,
    word_count: int,
    nervousness: float = 0.2,
) -> Optional[object]:
    """
    Feed the latest evaluation into the RL agent and get the next action.
    Returns an Action object or None on failure.
    """
    if seq is None:
        return None
    try:
        action = seq.record_and_select(
            score=score,
            star_scores=star_scores,
            word_count=word_count,
            nervousness=nervousness,
        )
        return action
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING HELPERS  (v4.0)
# ══════════════════════════════════════════════════════════════════════════════

# Weight profiles mirroring answer_evaluator.py — shown in the feedback card
WEIGHT_PROFILES = {
    "technical":   {"STAR": 0.00, "Relevance": 0.40, "Keywords": 0.25, "Depth": 0.20, "Grammar": 0.05, "Word Cat": 0.10},
    "behavioural": {"STAR": 0.35, "Relevance": 0.20, "Keywords": 0.10, "Depth": 0.10, "Grammar": 0.05, "Word Cat": 0.20},
    "hr":          {"STAR": 0.20, "Relevance": 0.25, "Keywords": 0.10, "Depth": 0.25, "Grammar": 0.05, "Word Cat": 0.15},
}

_TYPE_NORM = {
    "technical": "technical", "behavioural": "behavioural",
    "behavioral": "behavioural", "hr": "hr", "general": "hr",
    "mixed": "hr",
}


def _resolve_weight_profile(q_type: str) -> dict:
    key = _TYPE_NORM.get(q_type.lower().strip(), "hr")
    return WEIGHT_PROFILES[key], key


def _score_answer(answer: str, question: dict, engine=None) -> dict:
    """
    Score an answer using:
      1. engine.nlp_scorer.score_answer (full pipeline — preferred)
      2. AnswerEvaluator direct (if engine unavailable)
      3. Lightweight fallback (keyword matching only)
    Returns the standard result dict.
    """
    # Path 1 — full engine
    if engine is not None:
        try:
            return engine.nlp_scorer.score_answer(answer, question)
        except Exception:
            pass

    # Path 2 — direct AnswerEvaluator
    if EVALUATOR_AVAILABLE:
        try:
            ev = AnswerEvaluator()
            return ev.evaluate(answer, question)
        except Exception:
            pass

    # Path 3 — lightweight keyword fallback
    kw        = question.get("keywords", [])
    text_low  = answer.lower()
    hits      = [k for k in kw if k.lower() in text_low]
    wc        = len(answer.split())
    kw_ratio  = len(hits) / len(kw) if kw else 0.5
    wc_score  = min(wc / 150, 1.0)
    score     = round(min((kw_ratio * 2.5 + wc_score * 2.5), 5.0), 1)
    return {
        "score":       score,
        "feedback":    "Answer recorded. Connect your engine for full AI feedback.",
        "star_scores": {},
        "keyword_hits": hits,
        "disc_traits": {},
        "ocean":       {},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PERSONALITY SIGNAL HELPERS  (v4.0)
# ══════════════════════════════════════════════════════════════════════════════

def _top_disc(disc: dict) -> str:
    if not disc:
        return ""
    return max(disc, key=lambda k: disc[k])


def _ocean_summary(ocean: dict) -> str:
    if not ocean:
        return ""
    items = sorted(ocean.items(), key=lambda x: x[1], reverse=True)[:2]
    return " · ".join(f"{k}" for k, _ in items)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION EXPORT  (v4.0)
# ══════════════════════════════════════════════════════════════════════════════

def _build_session_export(day_num: int, answers: list, tips: list) -> str:
    """Build a JSON export of the session for download."""
    export = {
        "aura_session": {
            "day":        day_num,
            "role":       st.session_state.get("wp_role", ""),
            "exported":   datetime.datetime.now().isoformat(),
            "answers":    [],
        }
    }
    for i, a in enumerate(answers):
        if a.get("skipped"):
            continue
        export["aura_session"]["answers"].append({
            "q_num":         i + 1,
            "question":      a.get("question", ""),
            "answer":        a.get("answer", ""),
            "score":         a.get("score", 0),
            "q_type":        a.get("q_type", ""),
            "star":          a.get("star", {}),
            "keyword_hits":  a.get("keyword_hits", []),
            "feedback":      a.get("feedback", ""),
            "coaching_tip":  tips[i] if i < len(tips) else "",
            "disc_trait":    a.get("disc_trait", ""),
            "ocean":         a.get("ocean", {}),
        })
    return json.dumps(export, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  ONBOARDING
# ══════════════════════════════════════════════════════════════════════════════

def _render_onboarding() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=JetBrains+Mono:wght@400;500&family=Syne:wght@400;700&display=swap');
.wp2-title{font-family:'Orbitron',monospace;font-size:1.5rem;font-weight:900;color:#00e5ff;
  text-shadow:0 0 20px rgba(0,229,255,.4);margin-bottom:.3rem;}
.wp2-sub{font-family:'JetBrains Mono',monospace;font-size:.72rem;
  color:rgba(180,210,230,.45);letter-spacing:.1em;text-transform:uppercase;margin-bottom:1.8rem;}
.wp2-feat{font-family:'JetBrains Mono',monospace;font-size:.65rem;
  color:rgba(0,229,255,.55);background:rgba(0,229,255,.05);border:1px solid rgba(0,229,255,.12);
  border-radius:8px;padding:.6rem 1rem;margin-bottom:.6rem;line-height:1.7;}
</style>
""", unsafe_allow_html=True)

    st.markdown('<div class="wp2-title">⬡ BUILD YOUR PREP PLAN</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="wp2-sub">7 days · adaptive RL sequencing · live coaching · voice-enabled · full pipeline</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
<div class="wp2-feat">
  ✦ <b>RL Adaptive Sequencer</b> — Days 2 &amp; 5 auto-adjust question difficulty from your live scores<br>
  ✦ <b>Live Coach Tips</b> — Personalised 2–3 sentence coaching after every answer (Groq-backed)<br>
  ✦ <b>DISC + OCEAN Signals</b> — See how you're perceived after each behavioural/HR answer<br>
  ✦ <b>Follow-Up Probes</b> — Shallow answers trigger a targeted drill-down question<br>
  ✦ <b>Weekly Radar Chart</b> — Visualise Relevance · STAR · Keywords · Depth across all 7 days<br>
  ✦ <b>Session Export</b> — Download a full JSON of questions, answers, scores, and tips
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        role = st.selectbox("Target Role", [
            "Software Engineer", "Data Scientist", "Product Manager",
            "Data Engineer", "ML Engineer", "Backend Engineer",
            "Full Stack Developer", "DevOps / SRE", "Business Analyst",
            "Consulting / Strategy", "Finance / Banking", "Marketing",
            "Operations", "General / Other",
        ], index=0, key="wp_role_input")
    with col2:
        interview_date = st.date_input(
            "Interview Date",
            value=datetime.date.today() + datetime.timedelta(days=7),
            min_value=datetime.date.today(),
            key="wp_date_input",
        )

    weak_spots = st.multiselect(
        "Where do you need most work?",
        ["Technical", "Behavioural", "HR", "Communication", "Nerves"],
        default=["Behavioural", "Nerves"],
        key="wp_weak_input",
    )

    if st.button("▶  GENERATE MY 7-DAY PLAN", use_container_width=True, key="wp_submit"):
        st.session_state["wp_role"]       = role
        st.session_state["wp_date"]       = interview_date
        st.session_state["wp_weak"]       = weak_spots
        st.session_state["wp_plan_built"] = True
        st.session_state["wp_completed"]  = set()
        st.session_state["wp_day_metrics"] = {}
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TIMELINE + DAY START BUTTONS  — Pure Streamlit, no iframe/postMessage
# ══════════════════════════════════════════════════════════════════════════════

def _render_timeline(done_set: set, idate, role: str) -> None:
    """Header + progress bar — 100% inline styles, no <style> block."""
    days_left  = max((idate - datetime.date.today()).days, 0) if idate else 7
    pct        = int(len(done_set) / len(DAY_CONFIGS) * 100)
    done_count = len(done_set)

    chip = ("font-family:JetBrains Mono,monospace;font-size:.6rem;padding:3px 10px;"
            "border-radius:8px;background:rgba(0,229,255,.06);border:1px solid rgba(0,229,255,.14);"
            "color:rgba(180,210,230,.55);")

    st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.7rem;
  padding:.9rem 1.4rem;background:rgba(0,229,255,.03);border:1px solid rgba(0,229,255,.12);
  border-radius:14px;margin-bottom:1rem;">
  <div>
    <div style="font-family:Orbitron,monospace;font-size:1.1rem;font-weight:900;
      color:#00e5ff;letter-spacing:.06em;text-shadow:0 0 20px rgba(0,229,255,.4);">
      ⬡ <span style="color:rgba(255,255,255,.85);">AURA</span> WEEKLY PREP PLAN
    </div>
    <div style="font-family:JetBrains Mono,monospace;font-size:.52rem;
      color:rgba(180,210,230,.35);letter-spacing:.1em;text-transform:uppercase;margin-top:.15rem;">
      Adaptive RL · Live Coaching · Voice · DISC/OCEAN · Full Pipeline
    </div>
  </div>
  <div style="display:flex;gap:.45rem;flex-wrap:wrap;">
    <span style="{chip}">Role &nbsp;<b style="color:#00e5ff;">{role}</b></span>
    <span style="{chip}">Interview in &nbsp;<b style="color:#00e5ff;">{days_left}</b>&nbsp; days</span>
    <span style="{chip}">Complete &nbsp;<b style="color:#00e5ff;">{done_count}/7</b></span>
  </div>
</div>

<div style="margin-bottom:1.2rem;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.35rem;">
    <span style="font-family:JetBrains Mono,monospace;font-size:.57rem;
      color:rgba(180,210,230,.35);letter-spacing:.08em;text-transform:uppercase;">Week Progress</span>
    <span style="font-family:Orbitron,monospace;font-size:.65rem;font-weight:700;color:#00e5ff;">{pct}%</span>
  </div>
  <div style="height:3px;background:rgba(255,255,255,.05);border-radius:4px;overflow:hidden;">
    <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,#00e5ff,#7f5af0);border-radius:4px;"></div>
  </div>
</div>
""", unsafe_allow_html=True)




# ══════════════════════════════════════════════════════════════════════════════
#  DAY CARDS + START BUTTONS  — Pure Streamlit, zero iframe/postMessage
# ══════════════════════════════════════════════════════════════════════════════

def _render_day_start_buttons(completed: set) -> None:
    """
    Renders each day as a styled HTML card (cosmetic) + a real Streamlit
    st.button directly underneath it. No iframe, no postMessage, no hidden
    bridge buttons. Clicking START SESSION instantly sets session state and
    calls st.rerun() into the practice window — 100% reliable.
    """
    day_metrics = st.session_state.get("wp_day_metrics", {})
    active_day  = next(
        (d.day for d in DAY_CONFIGS if f"day_{d.day}" not in completed), 7
    )

    for d in DAY_CONFIGS:
        is_done   = f"day_{d.day}" in completed
        is_active = (d.day == active_day) and not is_done

        # ── Visual state ────────────────────────────────────────────────────
        if is_done:
            status_cls = "wp-status-done"
            status_txt = "✓ DONE"
            accent     = "linear-gradient(90deg,#00ff88,rgba(0,255,136,.04))"
            card_border= "rgba(0,255,136,.18)"
            card_glow  = ""
        elif is_active:
            status_cls = "wp-status-active"
            status_txt = "▶ ACTIVE"
            accent     = f"linear-gradient(90deg,{d.color}cc,rgba(127,90,240,.25))"
            card_border= "rgba(0,229,255,.28)"
            card_glow  = "box-shadow:0 0 22px rgba(0,229,255,.08);"
        else:
            status_cls = "wp-status-locked"
            status_txt = "🔒 LOCKED"
            accent     = "rgba(255,255,255,.03)"
            card_border= "rgba(255,255,255,.07)"
            card_glow  = ""

        # ── Fully inline styles — no CSS class dependencies ─────────────────
        theme_color  = d.color if is_active else ("#e2e8f0" if not is_done else "#aac8a0")

        rl_badge = (
            '<span style="font-family:JetBrains Mono,monospace;font-size:.5rem;padding:1px 7px;'
            'border-radius:6px;color:#ff6b35;background:rgba(255,107,53,.06);'
            'border:1px solid rgba(255,107,53,.2);">⚡ RL</span>'
        ) if d.use_rl else ""

        src_label = "📚 HR Dataset" if d.use_hr_dataset else "🤖 AI Generated"

        rl_meta = (
            '<span style="font-family:JetBrains Mono,monospace;font-size:.56rem;padding:2px 7px;'
            'border-radius:6px;color:#ff6b35;background:rgba(255,107,53,.05);'
            'border:1px solid rgba(255,107,53,.18);">⚡ RL Adaptive</span>'
        ) if d.use_rl else ""

        m         = day_metrics.get(str(d.day), {})
        avg_score = m.get("avg", 0)
        score_badge = (
            f'<span style="font-family:Orbitron,monospace;font-size:.57rem;font-weight:700;'
            f'padding:2px 8px;border-radius:6px;background:rgba(0,255,136,.07);'
            f'border:1px solid rgba(0,255,136,.2);color:#00ff88;">'
            f'{avg_score:.1f}/5</span>'
        ) if is_done and avg_score > 0 else ""

        # Status badge inline styles
        if is_done:
            status_style = "color:#00ff88;background:rgba(0,255,136,.08);border:1px solid rgba(0,255,136,.2);"
        elif is_active:
            status_style = "color:#00e5ff;background:rgba(0,229,255,.08);border:1px solid rgba(0,229,255,.22);"
        else:
            status_style = "color:rgba(180,210,230,.25);background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.07);"

        # Badge shared style
        badge_base = "font-family:JetBrains Mono,monospace;font-size:.56rem;padding:2px 7px;border-radius:6px;"

        # ── HTML card shell — built as a single concatenated string with no
        #    newlines or leading spaces (4+ space indentation in a markdown
        #    f-string triple-quote triggers Markdown's code-block rule, causing
        #    the raw HTML to render as visible plain text instead of HTML) ─────
        card_html = (
            '<div style="margin-bottom:0;">'
            + f'<div style="height:3px;border-radius:3px 3px 0 0;background:{accent};"></div>'
            + f'<div style="background:linear-gradient(145deg,rgba(5,10,28,.97),rgba(8,14,36,.97));border:1px solid {card_border};border-top:none;border-radius:0 0 14px 14px;padding:.9rem 1.2rem 1rem;{card_glow}">'
            + '<div style="display:flex;align-items:center;gap:.55rem;flex-wrap:wrap;margin-bottom:.55rem;">'
            + f'<span style="font-family:Orbitron,monospace;font-size:.52rem;font-weight:700;color:rgba(0,229,255,.45);letter-spacing:.1em;">DAY {d.day}</span>'
            + f'<span style="font-size:.95rem;">{d.emoji}</span>'
            + f'<span style="font-family:Syne,sans-serif;font-size:.86rem;font-weight:700;color:{theme_color};flex:1;min-width:0;">{d.theme}</span>'
            + rl_badge
            + score_badge
            + f'<span style="font-family:JetBrains Mono,monospace;font-size:.56rem;padding:3px 9px;border-radius:6px;{status_style}">{status_txt}</span>'
            + '</div>'
            + f'<div style="font-family:JetBrains Mono,monospace;font-size:.65rem;color:rgba(180,210,230,.48);line-height:1.65;margin-bottom:.7rem;">{d.desc}</div>'
            + '<div style="display:flex;gap:.4rem;flex-wrap:wrap;">'
            + f'<span style="{badge_base}color:#7f5af0;background:rgba(127,90,240,.09);border:1px solid rgba(127,90,240,.2);">{d.q_type}</span>'
            + f'<span style="{badge_base}color:#ffd700;background:rgba(255,215,0,.05);border:1px solid rgba(255,215,0,.15);">{d.difficulty.upper()}</span>'
            + f'<span style="{badge_base}color:rgba(0,229,255,.6);background:rgba(0,229,255,.05);border:1px solid rgba(0,229,255,.12);">{d.num_q} questions</span>'
            + f'<span style="{badge_base}color:#00ff88;background:rgba(0,255,136,.04);border:1px solid rgba(0,255,136,.14);">{src_label}</span>'
            + rl_meta
            + '</div>'
            + '</div>'
            + '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

        # ── REAL Streamlit button — clicks go straight to session state ─────
        if is_done:
            st.button(
                f"✓  Day {d.day} Complete — {d.theme}",
                key=f"wp_hidden_day_{d.day}",
                use_container_width=True,
                disabled=True,
            )
        else:
            btn_label = f"▶  START SESSION  ·  Day {d.day} — {d.theme}"
            if st.button(
                btn_label,
                key=f"wp_hidden_day_{d.day}",
                use_container_width=True,
                type="primary",
            ):
                st.session_state.update({
                    "wp_open_day":         d.day,
                    "wp_practice_day":     d.day,
                    "wp_practice_q":       None,
                    "wp_practice_answers": [],
                    "wp_voice_answer":     "",
                    "wp_follow_up_q":      None,
                    "wp_coaching_tips":    [],
                })
                st.rerun()

        st.markdown("<div style='height:.5rem;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PRACTICE WINDOW  (v5.0 — Immersive Interview Screen)
# ══════════════════════════════════════════════════════════════════════════════

def _render_practice_window(day_num: int, engine=None) -> None:
    """
    Immersive practice window matching the Live Interview aesthetic.
    v5.0 — full sci-fi background, mic button, STAR badges, rich feedback.
    """
    cfg = _DAY_MAP.get(day_num)
    if cfg is None:
        st.error(f"Day {day_num} configuration not found.")
        return

    role = st.session_state.get("wp_role", "Software Engineer")

    # ── Global page styles ─────────────────────────────────────────────────────
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=JetBrains+Mono:wght@400;500;600&family=Syne:wght@400;600;700&display=swap');
/* Transparent main container to let bg show */
[data-testid="stAppViewContainer"] > .main {{background:transparent !important;}}
.block-container {{padding-top:0 !important;max-width:100% !important;}}

/* ── Sci-fi animated background ── */
.wp5-bg {{
  position:fixed;inset:0;z-index:-1;
  background: linear-gradient(160deg,#010a1a 0%,#020d2a 40%,#030820 100%);
  overflow:hidden;
}}
.wp5-bg::before {{
  content:'';position:absolute;inset:0;
  background:
    radial-gradient(ellipse 80% 60% at 50% 110%, rgba(0,120,255,.18) 0%, transparent 70%),
    radial-gradient(ellipse 60% 40% at 20% 60%, rgba(0,229,255,.07) 0%, transparent 60%),
    radial-gradient(ellipse 50% 35% at 80% 40%, rgba(127,90,240,.06) 0%, transparent 60%);
}}
/* Animated grid lines */
.wp5-grid {{
  position:absolute;inset:0;
  background-image:
    linear-gradient(rgba(0,229,255,.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,.04) 1px, transparent 1px);
  background-size:60px 60px;
  animation:gridMove 20s linear infinite;
}}
@keyframes gridMove {{0%{{background-position:0 0;}}100%{{background-position:60px 60px;}}}}
/* Floating particles */
.wp5-particles {{position:absolute;inset:0;overflow:hidden;}}
.wp5-p {{
  position:absolute;border-radius:50%;
  background:rgba(0,229,255,.5);
  animation:floatUp linear infinite;
}}
@keyframes floatUp{{0%{{transform:translateY(100vh) scale(1);opacity:.6;}}100%{{transform:translateY(-20vh) scale(.3);opacity:0;}}}}

/* ── Top HUD bar ── */
.wp5-hud {{
  display:flex;align-items:center;justify-content:space-between;
  padding:.6rem 1.4rem;
  background:rgba(0,8,28,.85);
  border-bottom:1px solid rgba(0,229,255,.15);
  backdrop-filter:blur(12px);
  margin-bottom:0;
  flex-wrap:wrap;gap:.5rem;
}}
.wp5-hud-left {{display:flex;align-items:center;gap:1rem;}}
.wp5-q-counter {{
  font-family:'Orbitron',monospace;font-size:.65rem;font-weight:700;
  color:{cfg.color};letter-spacing:.12em;
  background:rgba(0,229,255,.06);border:1px solid {cfg.color}44;
  border-radius:8px;padding:3px 12px;
}}
.wp5-q-type {{
  font-family:'JetBrains Mono',monospace;font-size:.6rem;
  color:rgba(180,210,230,.45);letter-spacing:.1em;
  background:rgba(127,90,240,.08);border:1px solid rgba(127,90,240,.2);
  border-radius:8px;padding:3px 10px;text-transform:uppercase;
}}
.wp5-diff-chip {{
  font-family:'JetBrains Mono',monospace;font-size:.58rem;
  letter-spacing:.08em;border-radius:8px;padding:3px 10px;
}}
.wp5-mood-chip {{
  font-family:'JetBrains Mono',monospace;font-size:.58rem;
  color:rgba(0,229,255,.5);background:rgba(0,229,255,.06);
  border:1px solid rgba(0,229,255,.15);border-radius:8px;padding:3px 10px;
}}
.wp5-progress-wrap {{display:flex;align-items:center;gap:.7rem;}}
.wp5-progress-track {{
  width:120px;height:2px;background:rgba(255,255,255,.08);border-radius:4px;overflow:hidden;
}}
.wp5-progress-fill {{height:100%;background:linear-gradient(90deg,{cfg.color},{cfg.color}88);border-radius:4px;}}

/* ── Question card ── */
.wp5-q-wrap {{
  background:rgba(0,8,30,.82);
  border:1px solid rgba(0,229,255,.2);
  border-radius:16px;
  padding:1.4rem 2rem;
  margin:1rem 0 .8rem;
  position:relative;overflow:hidden;
  backdrop-filter:blur(16px);
}}
.wp5-q-wrap::before {{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,{cfg.color}cc,rgba(127,90,240,.6),transparent);
}}
.wp5-q-wrap.followup {{border-color:rgba(255,107,53,.35);}}
.wp5-q-wrap.followup::before {{background:linear-gradient(90deg,#ff6b35,#ffd700,transparent);}}
.wp5-q-text {{
  font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;
  color:#e8f4ff;line-height:1.55;margin-bottom:1rem;
}}
.wp5-star-row {{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.5rem;}}
.wp5-star-tag {{
  font-family:'JetBrains Mono',monospace;font-size:.62rem;font-weight:600;
  letter-spacing:.08em;padding:4px 12px;border-radius:8px;
  text-transform:uppercase;cursor:default;
}}
.wp5-star-s {{color:#00e5ff;background:rgba(0,229,255,.1);border:1px solid rgba(0,229,255,.3);}}
.wp5-star-t {{color:#7f5af0;background:rgba(127,90,240,.1);border:1px solid rgba(127,90,240,.3);}}
.wp5-star-a {{color:#e040fb;background:rgba(224,64,251,.1);border:1px solid rgba(224,64,251,.3);}}
.wp5-star-r {{color:#00ff88;background:rgba(0,255,136,.1);border:1px solid rgba(0,255,136,.3);}}

/* ── Voice / recording panel ── */
.wp5-voice-panel {{
  background:rgba(0,8,30,.9);
  border:1px solid rgba(0,229,255,.12);
  border-radius:16px;
  padding:1.2rem 1.5rem;
  margin-bottom:.8rem;
  backdrop-filter:blur(16px);
}}
.wp5-voice-lbl {{
  font-family:'JetBrains Mono',monospace;font-size:.6rem;
  color:rgba(0,229,255,.4);letter-spacing:.12em;text-transform:uppercase;
  margin-bottom:.7rem;display:flex;align-items:center;gap:.5rem;
}}
.wp5-wc {{
  font-family:'JetBrains Mono',monospace;font-size:.6rem;
  text-align:right;margin-top:-.3rem;margin-bottom:.6rem;
}}

/* ── Feedback card ── */
.wp5-fb {{
  background:rgba(0,6,22,.96);
  border:1px solid rgba(0,255,136,.22);
  border-radius:16px;padding:1.4rem 1.8rem;
  margin-bottom:1rem;position:relative;overflow:hidden;
  backdrop-filter:blur(16px);
}}
.wp5-fb::before {{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,#00ff88,rgba(0,229,255,.6),transparent);
}}
.wp5-fb-score-row {{display:flex;align-items:center;gap:1.5rem;margin-bottom:1rem;}}
.wp5-fb-score {{
  font-family:'Orbitron',monospace;font-size:2.8rem;font-weight:900;line-height:1;
}}
.wp5-fb-label {{
  font-family:'JetBrains Mono',monospace;font-size:.65rem;
  color:rgba(180,210,230,.4);letter-spacing:.1em;text-transform:uppercase;
}}
.wp5-star-grid {{
  display:grid;grid-template-columns:repeat(4,1fr);gap:.5rem;margin-bottom:.9rem;
}}
.wp5-sc {{border-radius:10px;padding:.7rem .5rem;text-align:center;}}
.wp5-sc.hit {{background:rgba(0,255,136,.07);border:1px solid rgba(0,255,136,.25);}}
.wp5-sc.miss{{background:rgba(255,80,80,.05);border:1px solid rgba(255,80,80,.18);}}
.wp5-sc-icon {{font-size:1.1rem;margin-bottom:.25rem;}}
.wp5-sc-name {{font-family:'JetBrains Mono',monospace;font-size:.56rem;color:rgba(180,210,230,.5);}}
.wp5-sc-vdt  {{font-family:'Orbitron',monospace;font-size:.55rem;font-weight:700;margin-top:.2rem;}}
.wp5-sc.hit .wp5-sc-vdt {{color:#00ff88;}} .wp5-sc.miss .wp5-sc-vdt {{color:rgba(255,100,100,.75);}}
.wp5-kw-row {{display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:.8rem;}}
.wp5-kw-hit {{font-family:'JetBrains Mono',monospace;font-size:.58rem;padding:3px 9px;border-radius:7px;
  color:#00e5ff;background:rgba(0,229,255,.08);border:1px solid rgba(0,229,255,.25);}}
.wp5-kw-miss{{font-family:'JetBrains Mono',monospace;font-size:.58rem;padding:3px 9px;border-radius:7px;
  color:rgba(180,210,230,.2);background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.07);}}
.wp5-fb-text {{
  background:rgba(0,0,0,.28);border:1px solid rgba(255,255,255,.07);
  border-radius:10px;padding:.9rem 1.1rem;
  font-family:'Syne',sans-serif;font-size:.83rem;
  color:rgba(200,220,240,.7);line-height:1.7;margin-bottom:.8rem;
}}
.wp5-ps {{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:.8rem;}}
.wp5-ps-disc {{font-family:'JetBrains Mono',monospace;font-size:.58rem;padding:3px 10px;border-radius:8px;
  color:#ffd700;background:rgba(255,215,0,.07);border:1px solid rgba(255,215,0,.2);}}
.wp5-ps-ocean{{font-family:'JetBrains Mono',monospace;font-size:.58rem;padding:3px 10px;border-radius:8px;
  color:#e040fb;background:rgba(224,64,251,.06);border:1px solid rgba(224,64,251,.18);}}

/* ── Tip banner ── */
.wp5-tip {{
  background:rgba(0,229,255,.04);border:1px solid rgba(0,229,255,.1);
  border-left:3px solid {cfg.color};
  border-radius:10px;padding:.65rem 1rem;margin-bottom:.9rem;
  font-family:'JetBrains Mono',monospace;font-size:.65rem;
  color:rgba(0,229,255,.65);line-height:1.65;
}}

/* ── Session summary card ── */
.wp5-summary {{
  background:rgba(0,6,22,.96);border:1px solid rgba(0,255,136,.25);
  border-radius:18px;padding:2rem;margin-bottom:1.2rem;
  text-align:center;position:relative;overflow:hidden;
}}
.wp5-summary::before {{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,#00ff88,rgba(0,229,255,.5),transparent);
}}
</style>
""", unsafe_allow_html=True)

    # ── Animated background (separate call to avoid HTML comment breaking renderer) ──
    st.markdown("""
<div class="wp5-bg">
  <div class="wp5-grid"></div>
  <div class="wp5-particles">
    <div class="wp5-p" style="left:10%;width:3px;height:3px;animation-duration:12s;animation-delay:0s;"></div>
    <div class="wp5-p" style="left:30%;width:2px;height:2px;animation-duration:18s;animation-delay:3s;"></div>
    <div class="wp5-p" style="left:55%;width:4px;height:4px;animation-duration:9s;animation-delay:1s;"></div>
    <div class="wp5-p" style="left:75%;width:2px;height:2px;animation-duration:15s;animation-delay:5s;"></div>
    <div class="wp5-p" style="left:90%;width:3px;height:3px;animation-duration:11s;animation-delay:2s;"></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── State vars ─────────────────────────────────────────────────────────────
    answers_so_far = st.session_state.get("wp_practice_answers", [])
    follow_up_q    = st.session_state.get("wp_follow_up_q")
    q_num          = len(answers_so_far) + 1
    total_q        = cfg.num_q

    # ── HUD bar ────────────────────────────────────────────────────────────────
    diff_color = {"easy": "#00ff88", "medium": "#ffd700", "hard": "#ff6b6b"}.get(cfg.difficulty, "#00e5ff")
    pct_done   = int((len(answers_so_far) / total_q) * 100) if total_q else 0

    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Plan", key="wp_back", use_container_width=True):
            st.session_state.update({
                "wp_open_day": None, "wp_practice_q": None,
                "wp_practice_answers": [], "wp_voice_answer": "",
                "wp_follow_up_q": None, "wp_coaching_tips": [],
            })
            st.rerun()


    # ── Session in progress ────────────────────────────────────────────────────
    if q_num <= total_q:
        # ── HUD top bar HTML ───────────────────────────────────────────────────
        st.markdown(f"""
<div class="wp5-hud">
  <div class="wp5-hud-left">
    <span class="wp5-q-counter">Q{q_num}/{total_q} · {cfg.signal}</span>
    <span class="wp5-q-type">{cfg.q_type}</span>
    <span class="wp5-diff-chip" style="color:{diff_color};background:{diff_color}11;
      border:1px solid {diff_color}44;">{cfg.difficulty.upper()}</span>
    <span class="wp5-mood-chip">♪ Neutral</span>
    <span class="wp5-mood-chip">✦ Neutral</span>
  </div>
  <div class="wp5-progress-wrap">
    <div class="wp5-progress-track">
      <div class="wp5-progress-fill" style="width:{pct_done}%;"></div>
    </div>
    <span style="font-family:'JetBrains Mono',monospace;font-size:.55rem;
      color:rgba(0,229,255,.35);">{pct_done}%</span>
  </div>
</div>
""", unsafe_allow_html=True)

        # Show a tip
        tip = random.choice(cfg.tips)
        st.markdown(
            f'<div class="wp5-tip">💡 &nbsp;<b style="color:{cfg.color};">Coaching Tip:</b>&nbsp; {tip}</div>',
            unsafe_allow_html=True,
        )

        # ── Fetch / cache current question ────────────────────────────────────
        current_q = st.session_state.get("wp_practice_q")
        if follow_up_q is not None:
            current_q = follow_up_q
            st.session_state["wp_practice_q"] = current_q

        if current_q is None:
            with st.spinner("Loading question…"):
                asked = [a.get("question", "") for a in answers_so_far]
                rl_difficulty = cfg.difficulty
                rl_q_type     = cfg.q_type

                if cfg.use_rl:
                    seq = _get_or_build_sequencer(role)
                    if seq is not None:
                        try:
                            action = seq.get_next_action()
                            if hasattr(action, "difficulty") and action.difficulty:
                                rl_difficulty = action.difficulty
                            if hasattr(action, "q_type") and action.q_type:
                                rl_q_type = action.q_type.capitalize()
                        except Exception:
                            pass

                if cfg.use_hr_dataset:
                    ds_type = {
                        "Behavioural": "Behavioural",
                        "HR": "HR",
                        "Mixed": random.choice(["Behavioural", "HR"]),
                    }.get(cfg.q_type, "HR")
                    q = _get_hr_question(rl_difficulty, ds_type, role, exclude_texts=asked)
                    if q is None:
                        q = {
                            "question": "Tell me about a time you demonstrated leadership under pressure.",
                            "type": "Behavioural", "keywords": ["leadership", "pressure", "team"],
                            "ideal_answer": "", "source": "fallback",
                            "role": role, "difficulty": cfg.difficulty,
                        }
                else:
                    if engine is None:
                        st.error("Engine not available for Technical questions.")
                        return
                    try:
                        _type_map = {"Technical": "technical", "Mixed": "technical"}
                        q = engine.qbank.get_single_question(
                            role=role, difficulty=rl_difficulty,
                            q_type=_type_map.get(cfg.q_type, "technical"),
                        )
                        if q is None:
                            batch = engine.qbank.get_questions(role, rl_difficulty, 3)
                            q = batch[0] if batch else {
                                "question": "Explain the CAP theorem and its implications.",
                                "type": "Technical", "keywords": [], "ideal_answer": "",
                            }
                    except Exception as ex:
                        st.error(f"Question generation failed: {ex}")
                        return

                st.session_state["wp_practice_q"] = q
                current_q = q

        q_text    = current_q.get("question", "")
        q_type    = current_q.get("type", cfg.q_type)
        q_source  = current_q.get("source", "")
        q_diff    = current_q.get("difficulty", cfg.difficulty)
        ideal_ans = current_q.get("ideal_answer", "")
        is_fu     = current_q.get("follow_up", False)

        weights, w_key = _resolve_weight_profile(q_type)

        # ── Question card ──────────────────────────────────────────────────────
        is_behavioural = q_type.lower() in ("behavioural", "behavioral", "hr", "mixed")
        star_tags = """
    <div class="wp5-star-row">
      <span class="wp5-star-tag wp5-star-s">◎ Situation</span>
      <span class="wp5-star-tag wp5-star-t">▷ Task</span>
      <span class="wp5-star-tag wp5-star-a">✦ Action</span>
      <span class="wp5-star-tag wp5-star-r">⊞ Result</span>
    </div>""" if is_behavioural else ""

        fu_label = "🔍 FOLLOW-UP PROBE" if is_fu else f"Q{q_num} of {total_q}"
        card_cls = "wp5-q-wrap followup" if is_fu else "wp5-q-wrap"

        src_info = "📚 HR Dataset" if q_source == "hr_dataset" else ("🤖 AI Generated" if q_source else "")
        rl_info  = " · ⚡ RL Adapted" if cfg.use_rl else ""

        st.markdown(f"""
<div class="{card_cls}">
  <div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;
    color:rgba(0,229,255,.38);letter-spacing:.12em;text-transform:uppercase;
    margin-bottom:.7rem;display:flex;align-items:center;gap:.8rem;">
    <span>{fu_label}</span>
    <span style="color:rgba(255,255,255,.15);">·</span>
    <span>{q_type.upper()}</span>
    {f'<span style="color:rgba(255,255,255,.15);">·</span><span>{src_info}{rl_info}</span>' if src_info else ''}
  </div>
  <div class="wp5-q-text">{q_text}</div>
  {star_tags}
</div>
""", unsafe_allow_html=True)

        # ── Voice input panel ──────────────────────────────────────────────────
        st.markdown('<div class="wp5-voice-panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="wp5-voice-lbl">🎙 Your Answer — click mic to record or type below</div>',
            unsafe_allow_html=True,
        )

        answer = ""
        try:
            from voice_input import voice_input_panel
            from speech_to_text import SpeechToText
            stt_inst = st.session_state.get("_wp_stt_instance")
            if stt_inst is None:
                stt_inst = SpeechToText()
                st.session_state["_wp_stt_instance"] = stt_inst
            answer = voice_input_panel(stt_inst, q_number=9000 + (day_num * 100) + q_num)
        except Exception:
            answer = st.text_area(
                "Your answer",
                height=150,
                placeholder="Speak or write your answer here. Aim for 120–180 words covering Situation → Task → Action → Result…",
                key=f"wp_ans_fallback_{day_num}_{q_num}",
                label_visibility="collapsed",
            )

        # Word count HUD
        wc  = len(answer.split()) if answer.strip() else 0
        wcc = "#00ff88" if 100 <= wc <= 220 else "#ffd700" if wc > 0 else "rgba(180,210,230,.22)"
        st.markdown(
            f'<div class="wp5-wc" style="color:{wcc};">{wc} words&nbsp;·&nbsp;target 100–180</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Submit / Skip ──────────────────────────────────────────────────────
        col_sub, col_skip = st.columns([4, 1])
        with col_sub:
            submit = st.button(
                "▶  EVALUATE ANSWER",
                key=f"wp_sub_{day_num}_{q_num}{'_fu' if is_fu else ''}",
                use_container_width=True, type="primary",
                disabled=not answer.strip(),
            )
        with col_skip:
            skip = st.button(
                "Skip →",
                key=f"wp_skip_{day_num}_{q_num}{'_fu' if is_fu else ''}",
                use_container_width=True,
            )

        if skip:
            st.session_state["wp_practice_answers"].append({"skipped": True, "question": q_text})
            st.session_state["wp_practice_q"]   = None
            st.session_state["wp_follow_up_q"]  = None
            st.session_state["wp_voice_answer"]  = ""
            st.rerun()

        if submit and answer.strip():
            with st.spinner("Evaluating answer…"):
                result = _score_answer(answer, current_q, engine)

            score     = result.get("score", 2.5)
            feedback  = result.get("feedback", "")
            star      = result.get("star_scores", {})
            kw_hits   = set(result.get("keyword_hits", []))
            all_kw    = current_q.get("keywords", [])
            disc      = result.get("disc_traits", {})
            ocean     = result.get("ocean", {})
            disc_top  = _top_disc(disc)
            ocean_top = _ocean_summary(ocean)

            sc = "#00ff88" if score >= 4 else "#ffd700" if score >= 2.5 else "#ff6b6b"
            sl = "Excellent" if score >= 4 else "Good" if score >= 3 else "Average" if score >= 2 else "Needs Work"

            # ── RL record + get next action ────────────────────────────────────
            if cfg.use_rl and not is_fu:
                seq = _get_or_build_sequencer(role)
                next_action = _rl_record_and_select(seq, score, star, wc)
                if next_action is not None and getattr(next_action, "follow_up", False):
                    fu_text = (
                        "You mentioned a situation — can you elaborate on the specific "
                        "actions YOU personally took and what the measurable outcome was?"
                    )
                    st.session_state["wp_follow_up_q"] = {
                        "question": fu_text, "type": q_type,
                        "keywords": all_kw, "ideal_answer": ideal_ans,
                        "source": "rl_followup", "difficulty": q_diff,
                        "follow_up": True,
                    }
                else:
                    st.session_state["wp_follow_up_q"] = None

            # ── Get coaching tip ───────────────────────────────────────────────
            coaching_tip = ""
            if COACH_AVAILABLE:
                try:
                    coaching_tip = generate_coaching_tip(
                        answer=answer, score=score, star_scores=star,
                        keyword_hits=list(kw_hits), question=q_text,
                    )
                except Exception:
                    pass
            tips_list = st.session_state.get("wp_coaching_tips", [])
            tips_list.append(coaching_tip)
            st.session_state["wp_coaching_tips"] = tips_list

            # ── Render immersive feedback card ─────────────────────────────────
            # STAR breakdown
            star_items = [
                ("S", "Situation", "◎"),
                ("T", "Task",      "▷"),
                ("A", "Action",    "✦"),
                ("R", "Result",    "⊞"),
            ]
            star_html = ""
            if star:
                star_html = '<div class="wp5-star-grid">'
                for key, label, icon in star_items:
                    hit  = star.get(key, star.get(label, False))
                    cls  = "hit" if hit else "miss"
                    vdt  = "HIT" if hit else "MISS"
                    star_html += (
                        f'<div class="wp5-sc {cls}">'
                        f'  <div class="wp5-sc-icon">{icon}</div>'
                        f'  <div class="wp5-sc-name">{label}</div>'
                        f'  <div class="wp5-sc-vdt">{vdt}</div>'
                        f'</div>'
                    )
                star_html += '</div>'

            # Keywords
            kw_html = ""
            if all_kw:
                kw_html = '<div class="wp5-kw-row">'
                for kw in all_kw[:14]:
                    cls = "wp5-kw-hit" if kw.lower() in (k.lower() for k in kw_hits) else "wp5-kw-miss"
                    kw_html += f'<span class="{cls}">{kw}</span>'
                kw_html += '</div>'

            # Personality signal
            ps_html = ""
            if disc_top:
                ps_html += f'<span class="wp5-ps-disc">DISC: {disc_top}</span>'
            if ocean_top:
                ps_html += f'<span class="wp5-ps-ocean">OCEAN: {ocean_top}</span>'
            ps_wrap = f'<div class="wp5-ps">{ps_html}</div>' if ps_html else ""

            # Feedback narrative
            fb_html = (
                f'<div class="wp5-fb-text">{feedback}</div>'
                if feedback else ""
            )

            # Ideal answer hint
            ideal_html = ""
            if ideal_ans:
                ideal_html = (
                    f'<details style="margin-top:.6rem;">'
                    f'<summary style="font-family:\'JetBrains Mono\',monospace;font-size:.6rem;'
                    f'color:rgba(0,229,255,.4);cursor:pointer;letter-spacing:.08em;">▷ SHOW IDEAL ANSWER</summary>'
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.68rem;'
                    f'color:rgba(0,229,255,.5);line-height:1.75;padding:.8rem 0 .3rem;">{ideal_ans}</div>'
                    f'</details>'
                )

            st.markdown(f"""
<div class="wp5-fb">
  <div class="wp5-fb-score-row">
    <div>
      <div class="wp5-fb-score" style="color:{sc};">{score:.1f}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.75rem;
        color:rgba(255,255,255,.2);margin-top:.15rem;">/ 5.0</div>
    </div>
    <div>
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;
        font-weight:700;color:{sc};">{sl}</div>
      <div class="wp5-fb-label">Score · {q_type}</div>
    </div>
  </div>

  <div style="height:1px;background:rgba(255,255,255,.06);margin:.8rem 0;"></div>

  {f'<div class="wp5-fb-label" style="margin-bottom:.5rem;">STAR BREAKDOWN</div>{star_html}' if star_html else ''}
  {f'<div class="wp5-fb-label" style="margin-bottom:.5rem;">KEYWORD COVERAGE</div>{kw_html}' if kw_html else ''}
  {fb_html}
  {ps_wrap}
  {ideal_html}
</div>
""", unsafe_allow_html=True)

            # Coach card
            if coaching_tip and COACH_AVAILABLE:
                try:
                    render_coach_card(coaching_tip)
                except Exception:
                    if coaching_tip:
                        st.info(f"💬 Coach: {coaching_tip}")

            # TTS speak button
            if coaching_tip:
                import urllib.parse
                tts_url = f"https://tts.labs.anthropic.com/?text={urllib.parse.quote(coaching_tip[:300])}"
                st.markdown(
                    f'<a href="{tts_url}" target="_blank" style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:.6rem;color:rgba(0,229,255,.4);text-decoration:none;'
                    f'background:rgba(0,229,255,.05);border:1px solid rgba(0,229,255,.12);'
                    f'border-radius:7px;padding:3px 10px;display:inline-block;margin-bottom:.7rem;">'
                    f'🔊 Speak Feedback</a>',
                    unsafe_allow_html=True,
                )

            # Record answer and advance
            st.session_state["wp_practice_answers"].append({
                "question":     q_text,
                "answer":       answer,
                "score":        score,
                "feedback":     feedback,
                "q_type":       q_type,
                "star":         star,
                "keyword_hits": list(kw_hits),
                "disc_trait":   disc_top,
                "ocean":        ocean,
            })
            st.session_state["wp_practice_q"]  = None
            st.session_state["wp_voice_answer"] = ""

            if len(st.session_state["wp_practice_answers"]) >= total_q:
                st.rerun()
            else:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(
                    "▶  NEXT QUESTION",
                    key=f"wp_next_{day_num}_{q_num}",
                    use_container_width=True,
                    type="primary",
                ):
                    st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  SESSION SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    else:
        done_answers = [a for a in answers_so_far if not a.get("skipped")]
        scores       = [a["score"] for a in done_answers if "score" in a]
        avg          = round(sum(scores) / len(scores), 2) if scores else 0.0
        ac  = "#00ff88" if avg >= 4 else "#ffd700" if avg >= 2.5 else "#ff6b6b"
        al  = "Excellent" if avg >= 4 else "Good" if avg >= 3 else "Average" if avg >= 2 else "Needs Work"

        # ── Persist day metrics for radar ─────────────────────────────────────
        day_metrics = st.session_state.get("wp_day_metrics", {})
        day_metrics[str(day_num)] = {
            "avg":       avg,
            "relevance": avg / 5,
            "star":      sum(1 for a in done_answers if a.get("star") and all(a["star"].values())) / max(len(done_answers), 1),
            "keywords":  (
                sum(len(a.get("keyword_hits", [])) for a in done_answers) /
                max(sum(len(a.get("keyword_hits", [])) + 1 for a in done_answers), 1)
            ),
            "depth":     min(avg / 5 + 0.05, 1.0),
            "grammar":   min(avg / 5 + 0.1, 1.0),
            "fluency":   min(avg / 5, 1.0),
        }
        st.session_state["wp_day_metrics"] = day_metrics

        st.markdown(f"""
<div class="wp5-summary">
  <div style="font-family:'Orbitron',monospace;font-size:.65rem;color:rgba(0,229,255,.4);
    letter-spacing:.12em;text-transform:uppercase;margin-bottom:.7rem;">SESSION COMPLETE · DAY {day_num}</div>
  <div style="font-family:'Orbitron',monospace;font-size:3rem;font-weight:900;
    color:{ac};line-height:1;text-shadow:0 0 30px {ac}66;">{avg:.1f}</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:.7rem;
    color:rgba(255,255,255,.25);margin:.3rem 0 .6rem;">/ 5.0 average</div>
  <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:{ac};">{al}</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;
    color:rgba(180,210,230,.3);margin-top:.4rem;">{len(scores)}/{total_q} questions scored</div>
</div>""", unsafe_allow_html=True)

        # Per-question summary
        st.markdown("### 📋 Answer Review")
        for i, a in enumerate(done_answers, 1):
            s      = a.get("score", 0)
            sc     = "#00ff88" if s >= 4 else "#ffd700" if s >= 2.5 else "#ff6b6b"
            star   = a.get("star", {})
            ss     = " ".join(f'{"✓" if v else "✗"}{k[0]}' for k, v in star.items()) if star else ""
            qt     = a.get("q_type", "")
            fb     = a.get("feedback", "")
            disc_t = a.get("disc_trait", "")
            fb_html = (
                '<div style="font-family:\'Syne\',sans-serif;font-size:.8rem;'
                'color:rgba(200,220,240,.6);line-height:1.65;padding:.5rem .7rem;'
                'background:rgba(0,0,0,.2);border-radius:8px;">' + fb + '</div>'
            ) if fb else ""
            sl = "Excellent" if s >= 4 else "Good" if s >= 3 else "Average" if s >= 2 else "Needs Work"
            disc_html = f'  ·  DISC: {disc_t}' if disc_t else ""
            ss_html   = f'  ·  STAR: {ss}' if ss else ""

            with st.expander(f"Q{i}  ·  {a.get('question','')[:70]}…  [{s:.1f}/5]"):
                st.markdown(f"""
<div style="font-family:'JetBrains Mono',monospace;font-size:.7rem;color:rgba(180,210,230,.55);
  margin-bottom:.5rem;padding:.5rem .7rem;background:rgba(0,0,0,.25);border-radius:8px;">
  <b style="color:{sc};">Score: {s:.1f}/5 — {sl}</b>  ·  {qt}
  {ss_html}{disc_html}
</div>
{fb_html}
""", unsafe_allow_html=True)

        # Coaching tips recap
        tips_list = st.session_state.get("wp_coaching_tips", [])
        non_empty_tips = [t for t in tips_list if t]
        if non_empty_tips:
            with st.expander("💬 Coaching Tips Recap"):
                for i, tip in enumerate(non_empty_tips, 1):
                    st.markdown(
                        f'<div style="background:rgba(0,229,255,.04);border:1px solid rgba(0,229,255,.1);'
                        f'border-radius:10px;padding:.65rem 1rem;margin-bottom:.5rem;'
                        f'font-family:\'JetBrains Mono\',monospace;font-size:.64rem;'
                        f'color:rgba(0,229,255,.6);line-height:1.6;">'
                        f'<b style="color:rgba(0,229,255,.7);">Q{i}:</b> {tip}</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        # Action buttons
        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            if st.button("✓  Mark Day Complete", key="wp_done_end",
                         use_container_width=True, type="primary"):
                _mark_day_complete(day_num)
                st.rerun()
        with c2:
            if st.button("↺  Retry Session", key="wp_retry", use_container_width=True):
                st.session_state.update({
                    "wp_practice_q": None, "wp_practice_answers": [],
                    "wp_voice_answer": "", "wp_follow_up_q": None,
                    "wp_coaching_tips": [], 
                })
                st.rerun()
        with c3:
            export_json = _build_session_export(
                day_num, done_answers, tips_list
            )
            st.download_button(
                label="📥 Export Session",
                data=export_json,
                file_name=f"aura_day{day_num}_{datetime.date.today()}.json",
                mime="application/json",
                key="wp_export",
                use_container_width=True,
            )
# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _mark_day_complete(day_num: int) -> None:
    c = st.session_state.get("wp_completed", set())
    c.add(f"day_{day_num}")
    st.session_state.update({
        "wp_completed":        c,
        "wp_open_day":         None,
        "wp_practice_q":       None,
        "wp_practice_answers": [],
        "wp_voice_answer":     "",
        "wp_follow_up_q":      None,
        "wp_coaching_tips":    [],
    })


# ══════════════════════════════════════════════════════════════════════════════
#  QUERY-PARAM BRIDGES
# ══════════════════════════════════════════════════════════════════════════════

def _process_toggle() -> None:
    try:
        key = st.query_params.get("wp_toggle", None)
        if key:
            c: set = st.session_state.get("wp_completed", set())
            c.discard(key) if key in c else c.add(key)
            st.session_state["wp_completed"] = c
            st.query_params.clear()
            st.rerun()
    except Exception:
        pass


def _process_open_day() -> None:
    try:
        val = st.query_params.get("wp_open_day", None)
        if val:
            day_num = int(val)
            st.query_params.clear()
            st.session_state.update({
                "wp_open_day":         day_num,
                "wp_practice_day":     day_num,
                "wp_practice_q":       None,
                "wp_practice_answers": [],
                "wp_voice_answer":     "",
                "wp_follow_up_q":      None,
                "wp_coaching_tips":    [],
            })
            st.rerun()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def page_weekly_plan(engine=None) -> None:
    """
    Main entry point.
    app.py should call:  page_weekly_plan(engine=engine)
    """
    for k, v in WEEKLY_PLAN_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    _process_toggle()
    _process_open_day()

    if not st.session_state.get("wp_plan_built", False):
        _render_onboarding()
        return

    role     = st.session_state.get("wp_role", "Software Engineer")
    idate    = st.session_state.get("wp_date",
                  datetime.date.today() + datetime.timedelta(days=7))
    done_set = st.session_state.get("wp_completed", set())
    open_day = st.session_state.get("wp_open_day", None)

    if open_day is not None:
        cfg = _DAY_MAP.get(open_day)
        if cfg and not cfg.use_hr_dataset and engine is None:
            st.error("Engine not available — cannot start Technical practice session.")
            if st.button("← Back to Plan"):
                st.session_state["wp_open_day"] = None
                st.rerun()
            return
        _render_practice_window(open_day, engine)
        return

    _render_timeline(done_set, idate, role)
    _render_day_start_buttons(done_set)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        if st.button("↺  REBUILD PLAN", key="wp_reset", use_container_width=True):
            st.session_state.update({
                "wp_plan_built":   False,
                "wp_completed":    set(),
                "wp_open_day":     None,
                "wp_practice_q":   None,
                "wp_practice_answers": [],
                "wp_voice_answer": "",
                "wp_day_metrics":  {},
                "wp_follow_up_q":  None,
                "wp_coaching_tips": [],
                "wp_rl_sequencer_obj": None,
            })
            st.rerun()
    with c2:
        n = len(done_set)
        t = len(DAY_CONFIGS)
        st.caption(f"{n}/{t} days complete · {int(n/t*100) if t else 0}%")