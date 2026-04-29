"""
company_question_upload.py  –  Aura AI | Company Question Bank Manager
=======================================================================
v4.0 CHANGES:
  * Popup modal on Placement Test click — upload all CSVs at once + Skip option
  * Smart auto-detection: system decides which CSV is aptitude/technical/HR
  * Fixed CSV parsing — handles Windows line endings, BOM, extra whitespace
  * Multi-file upload in popup (up to 3 files at once)
  * Backward-compatible with existing page_company_questions() page
"""

from __future__ import annotations

import io
import json
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import openpyxl  # noqa: F401
    _XLSX_OK = True
except ImportError:
    _XLSX_OK = False


# =============================================================================
# 0  CONSTANTS
# =============================================================================

_TOPICS_MCQ = [
    "Numbers", "Percentage", "Profit and Loss", "Average",
    "Ratio and Proportion", "Mixture and Alligation", "Time and Work",
    "Pipes and Cisterns", "Time Speed Distance", "Algebra",
    "Trigonometry", "Geometry", "Probability",
    "Permutation and Combination", "Age Problems",
    "Logical Reasoning", "Data Structures", "Algorithms",
    "Computer Networks", "Databases", "Operating Systems",
    "OOP", "Programming", "Design Patterns", "Custom",
]

_TOPICS_TECH = [
    "Data Structures", "Algorithms", "System Design", "OOP",
    "Databases / SQL", "Computer Networks", "Operating Systems",
    "Cloud / DevOps", "Web Development", "Python", "Java",
    "JavaScript", "C / C++", "Machine Learning", "Custom",
]

_TOPICS_HR = [
    "Introduction", "Strengths & Weaknesses", "Situational",
    "Teamwork", "Leadership", "Conflict Resolution",
    "Career Goals", "Company Fit", "Stress / Pressure",
    "Achievement", "Failure / Learning", "Custom",
]

_DIFFICULTIES = ["easy", "medium", "hard"]

_BANK_KEYS = {
    "aptitude":  "company_mcq_bank",
    "technical": "company_tech_bank",
    "hr":        "company_hr_bank",
}

_CSV_TEMPLATES = {
    "aptitude": pd.DataFrame([
        {
            "question":    "What is the LCM of 4 and 6?",
            "option_A":    "8",
            "option_B":    "12",
            "option_C":    "24",
            "option_D":    "6",
            "correct":     "B",
            "topic":       "Numbers",
            "difficulty":  "easy",
            "explanation": "LCM(4,6) = 12.",
        }
    ]),
    "technical": pd.DataFrame([
        {
            "question":   "Explain the difference between a stack and a queue.",
            "topic":      "Data Structures",
            "difficulty": "easy",
        }
    ]),
    "hr": pd.DataFrame([
        {
            "question":   "Tell me about a time you resolved a conflict in a team.",
            "topic":      "Teamwork",
            "difficulty": "medium",
        }
    ]),
}


# =============================================================================
# 1  SESSION STATE DEFAULTS
# =============================================================================

COMPANY_UPLOAD_DEFAULTS: Dict = {
    "company_mcq_bank":  [],
    "company_tech_bank": [],
    "company_hr_bank":   [],
    "_cq_tab":           "upload",
    "_cq_bank_type":     "aptitude",
    "_cq_builder_draft": {},
    "_cq_edit_idx":      None,
    # Popup state
    "_cq_popup_open":    False,
    "_cq_popup_done":    False,
}


def _init_defaults() -> None:
    for k, v in COMPANY_UPLOAD_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v if not isinstance(v, list) else list(v)


# =============================================================================
# 2  AUTO-DETECTION ENGINE
# =============================================================================

# MCQ signature: has option columns or exactly A/B/C/D columns
_MCQ_SIGNALS = {"option_a", "option_b", "option_c", "option_d", "a", "b", "c", "d",
                "correct", "correct_answer", "answer", "explanation", "opt_a", "opt_b",
                "choice_a", "choice_b"}

# HR signal keywords in questions
_HR_KEYWORDS = [
    "tell me about", "yourself", "strength", "weakness", "teamwork",
    "conflict", "challenge", "failure", "success", "leadership", "career",
    "why did you", "describe a time", "situational", "behavioral",
    "pressure", "stress", "goal", "motivation", "achievement",
]

# Technical signal keywords
_TECH_KEYWORDS = [
    "algorithm", "data structure", "complexity", "database", "sql",
    "network", "operating system", "thread", "process", "memory",
    "pointer", "recursion", "sorting", "binary", "linked list",
    "hash", "tree", "graph", "api", "rest", "http", "oop",
    "class", "inheritance", "polymorphism", "design pattern",
    "big o", "time complexity", "space complexity",
]


def _detect_file_type(df: pd.DataFrame, filename: str = "") -> str:
    """
    Auto-detect whether a CSV is aptitude MCQ, technical, or HR.
    Returns 'aptitude', 'technical', or 'hr'.
    """
    cols = set(df.columns.str.lower().str.strip().str.replace(" ", "_").tolist())
    fname = filename.lower()

    # 1. Filename hint (fastest signal)
    if any(x in fname for x in ["aptitude", "mcq", "quant", "apti"]):
        return "aptitude"
    if any(x in fname for x in ["technical", "tech", "coding", "programming"]):
        return "technical"
    if any(x in fname for x in ["hr", "behavioral", "behavioural", "soft"]):
        return "hr"

    # 2. Column structure — MCQ has option columns
    if cols & _MCQ_SIGNALS:
        return "aptitude"

    # 3. Analyse question text
    questions = " ".join(str(v) for v in df.get("question", pd.Series([])).fillna("").tolist()).lower()

    hr_score = sum(1 for kw in _HR_KEYWORDS if kw in questions)
    tech_score = sum(1 for kw in _TECH_KEYWORDS if kw in questions)

    if hr_score > tech_score:
        return "hr"
    elif tech_score > 0:
        return "technical"

    # 4. Topic column hints
    topics_text = " ".join(str(v) for v in df.get("topic", pd.Series([])).fillna("").tolist()).lower()
    if any(x in topics_text for x in ["teamwork", "leadership", "introduction", "strength"]):
        return "hr"
    if any(x in topics_text for x in ["data structure", "algorithm", "network", "database"]):
        return "technical"

    return "technical"  # safe default


# =============================================================================
# 3  ROBUST CSV / XLSX PARSER
# =============================================================================

def _read_dataframe(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Read a CSV or XLSX robustly, handling BOM, encodings, Windows endings."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith(".xlsx"):
        if not _XLSX_OK:
            return None, "Install openpyxl to read .xlsx files: pip install openpyxl"
        try:
            df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
            return df, None
        except Exception as exc:
            return None, f"Could not read Excel file: {exc}"

    # CSV — try multiple encodings and handle BOM
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            text = raw.decode(enc)
            # Normalise line endings
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            df = pd.read_csv(io.StringIO(text))
            return df, None
        except Exception:
            continue

    return None, "Could not decode the file. Please save as UTF-8 CSV."


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names: lowercase, strip, spaces→underscore."""
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    # Handle alternate column name spellings
    renames = {
        "option a": "option_a", "option b": "option_b",
        "option c": "option_c", "option d": "option_d",
        "opt_a": "option_a", "opt_b": "option_b",
        "opt_c": "option_c", "opt_d": "option_d",
        "choice_a": "option_a", "choice_b": "option_b",
        "choice_c": "option_c", "choice_d": "option_d",
        "a": "option_a", "b": "option_b",
        "c": "option_c", "d": "option_d",
        "correct_answer": "correct", "answer": "correct",
        "ans": "correct",
        "q": "question", "ques": "question",
        "expl": "explanation", "explain": "explanation",
        "level": "difficulty", "diff": "difficulty",
    }
    df.rename(columns=renames, inplace=True)
    return df


def _validate_mcq(row: Dict) -> Tuple[Optional[Dict], Optional[str]]:
    """Validate and clean one MCQ row."""
    q = str(row.get("question", "")).strip()
    if not q or q.lower() in ("nan", "none", ""):
        return None, "Missing question text."

    opts: Dict[str, str] = {}
    for ltr in "ABCD":
        v = str(row.get(f"option_{ltr.lower()}", row.get(f"option_{ltr}", ""))).strip()
        if not v or v.lower() in ("nan", "none"):
            return None, f"Missing option {ltr}."
        opts[ltr] = v

    correct = str(row.get("correct", "")).strip().upper()
    # Handle formats like "A.", "A)", "(A)", "Option A"
    for pattern in ["OPTION ", "OPT ", "(", ")", "."]:
        correct = correct.replace(pattern, "")
    correct = correct.strip()
    if len(correct) > 1:
        correct = correct[0]
    if correct not in "ABCD":
        return None, f"'correct' must be A/B/C/D, got '{correct}'."

    topic = str(row.get("topic", "Custom")).strip()
    if not topic or topic.lower() in ("nan", "none"):
        topic = "Custom"

    diff = str(row.get("difficulty", "easy")).strip().lower()
    if diff not in _DIFFICULTIES:
        diff = "easy"

    expl = str(row.get("explanation", "")).strip()
    if expl.lower() in ("nan", "none"):
        expl = ""

    return {
        "question":    q,
        "options":     opts,
        "correct":     correct,
        "topic":       topic,
        "difficulty":  diff,
        "explanation": expl,
        "q_format":    "mcq",
        "source":      "company",
    }, None


def _validate_text(row: Dict, q_type: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Validate and clean one text question row."""
    q = str(row.get("question", "")).strip()
    if not q or q.lower() in ("nan", "none", ""):
        return None, "Missing question text."

    topic = str(row.get("topic", "Custom")).strip()
    if not topic or topic.lower() in ("nan", "none"):
        topic = "Custom"

    diff = str(row.get("difficulty", "medium")).strip().lower()
    if diff not in _DIFFICULTIES:
        diff = "medium"

    return {
        "question":   q,
        "topic":      topic,
        "difficulty": diff,
        "q_format":   "text",
        "q_type":     q_type,
        "source":     "company",
    }, None


def _parse_file(uploaded_file, bank_type: str) -> Tuple[List[Dict], List[str]]:
    """
    Parse an uploaded file with robust error handling.
    Returns (valid_questions, error_messages).
    """
    df, read_err = _read_dataframe(uploaded_file)
    if df is None:
        return [], [read_err or "Could not read file."]

    if df.empty:
        return [], ["The file appears to be empty."]

    df = _normalise_columns(df)

    # Drop fully empty rows
    df = df.dropna(how="all")

    records = df.to_dict("records")
    valid, errors = [], []

    for i, row in enumerate(records, start=2):  # row 1 = header
        # Clean all string values
        row = {k: v for k, v in row.items()}
        if bank_type == "aptitude":
            q, err = _validate_mcq(row)
        else:
            q, err = _validate_text(row, bank_type)

        if err:
            errors.append(f"Row {i}: {err}")
        else:
            valid.append(q)

    return valid, errors


def _parse_file_auto(uploaded_file) -> Tuple[str, List[Dict], List[str]]:
    """
    Parse a file with auto-detection of its type.
    Returns (detected_type, valid_questions, error_messages).
    """
    df, read_err = _read_dataframe(uploaded_file)
    if df is None:
        return "unknown", [], [read_err or "Could not read file."]

    if df.empty:
        return "unknown", [], ["The file appears to be empty."]

    df = _normalise_columns(df)
    df = df.dropna(how="all")

    detected = _detect_file_type(df, uploaded_file.name)

    records = df.to_dict("records")
    valid, errors = [], []

    for i, row in enumerate(records, start=2):
        if detected == "aptitude":
            q, err = _validate_mcq(row)
        else:
            q, err = _validate_text(row, detected)

        if err:
            errors.append(f"Row {i}: {err}")
        else:
            valid.append(q)

    return detected, valid, errors


# =============================================================================
# 4  PLACEMENT TEST POPUP  (the NEW feature)
# =============================================================================

def show_placement_test_upload_popup() -> bool:
    """
    Show a sleek popup that lets users upload up to 3 CSV files (one per type)
    before starting the Placement Test.

    Returns True when the user clicks "Start Test" (with or without upload).
    Returns False while popup is still open.

    Usage in placement_test_mode.py _page_setup():
        from company_question_upload import show_placement_test_upload_popup
        ...
        if st.button("▶ Begin Placement Test", ...):
            st.session_state._cq_popup_open = True
            st.rerun()

        if st.session_state.get("_cq_popup_open"):
            if show_placement_test_upload_popup():
                # proceed with test start logic
    """
    _init_defaults()

    # ── Popup CSS & overlay ───────────────────────────────────────────────────
    st.markdown("""
<style>
/* Overlay */
.cq-popup-overlay {
  position: fixed; inset: 0; z-index: 9990;
  background: rgba(0,0,0,.75);
  backdrop-filter: blur(12px);
  display: flex; align-items: center; justify-content: center;
}
/* Modal box */
.cq-popup-box {
  background: linear-gradient(135deg, #070d2a 0%, #0d1340 100%);
  border: 1.5px solid rgba(99,102,241,.45);
  box-shadow: 0 0 80px rgba(99,102,241,.3), 0 0 20px rgba(0,212,255,.1);
  border-radius: 24px;
  padding: 40px 44px 36px;
  max-width: 640px; width: 95%;
  animation: cq-pop-in .3s cubic-bezier(.34,1.56,.64,1);
}
@keyframes cq-pop-in {
  from { opacity:0; transform:scale(.82) translateY(20px); }
  to   { opacity:1; transform:scale(1) translateY(0); }
}
.cq-popup-title {
  font-family: 'Orbitron', monospace;
  font-size: 1.4rem; font-weight: 700;
  color: #f1f5f9; margin: 0 0 6px;
  letter-spacing: .04em;
}
.cq-popup-sub {
  font-family: 'Share Tech Mono', monospace;
  font-size: 10px; letter-spacing: 2px;
  color: rgba(99,102,241,.85); margin: 0 0 24px;
}
.cq-popup-desc {
  color: rgba(255,255,255,.55); font-size: .88rem;
  line-height: 1.65; margin: 0 0 22px;
}
/* Type badge on detected files */
.cq-type-badge {
  display: inline-block; padding: 3px 10px;
  border-radius: 4px; font-family: 'Share Tech Mono', monospace;
  font-size: 9px; letter-spacing: 1.5px; margin-left: 8px;
}
.cq-badge-apt  { background: rgba(0,212,255,.12); color: #00d4ff; border: .5px solid #00d4ff; }
.cq-badge-tech { background: rgba(99,102,241,.15); color: #a5b4fc; border: .5px solid #6366f1; }
.cq-badge-hr   { background: rgba(0,255,136,.10); color: #00ff88; border: .5px solid #00ff88; }
.cq-badge-unk  { background: rgba(255,255,255,.06); color: rgba(255,255,255,.4); border: .5px solid rgba(255,255,255,.15); }

/* File result cards */
.cq-file-result {
  background: rgba(4,9,26,.8);
  border: .5px solid rgba(255,255,255,.08);
  border-radius: 12px; padding: 12px 16px;
  margin-bottom: 10px;
  display: flex; align-items: center; gap: 12px;
}
.cq-file-ok   { border-color: rgba(0,255,136,.3); }
.cq-file-warn { border-color: rgba(245,158,11,.3); }
.cq-file-err  { border-color: rgba(255,51,102,.3); }

/* Stats pill row */
.cq-stat-row { display: flex; gap: 10px; margin: 18px 0 6px; flex-wrap: wrap; }
.cq-stat-pill {
  padding: 6px 14px; border-radius: 20px;
  font-family: 'Share Tech Mono', monospace;
  font-size: 10px; letter-spacing: 1px;
}
.cq-stat-pill-apt  { background: rgba(0,212,255,.08); color: #00d4ff; border: .5px solid rgba(0,212,255,.3); }
.cq-stat-pill-tech { background: rgba(99,102,241,.1);  color: #a5b4fc; border: .5px solid rgba(99,102,241,.35); }
.cq-stat-pill-hr   { background: rgba(0,255,136,.08); color: #00ff88; border: .5px solid rgba(0,255,136,.3); }
</style>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Share+Tech+Mono&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

    st.markdown("""
<div style="background:rgba(4,9,26,.97);border:1.5px solid rgba(99,102,241,.4);
  border-radius:20px;padding:36px 40px 30px;margin:12px 0 20px;
  box-shadow:0 0 60px rgba(99,102,241,.2);">

  <p style="font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:3px;
    color:rgba(99,102,241,.8);margin:0 0 10px;">PLACEMENT TEST SETUP</p>

  <h2 style="font-family:'Orbitron',monospace;font-size:1.5rem;font-weight:700;
    color:#f1f5f9;margin:0 0 8px;letter-spacing:.03em;">
    Upload Company Questions
  </h2>

  <p style="color:rgba(255,255,255,.48);font-size:.88rem;line-height:1.65;margin:0 0 6px;">
    Upload up to 3 CSV files — the system will <strong style="color:#00d4ff">automatically detect</strong>
    which is Aptitude, Technical, or HR. You can also skip this step and use AI-generated questions.
  </p>
</div>
""", unsafe_allow_html=True)

    # ── Multi-file uploader ───────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload CSV/XLSX files (up to 3)",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="cq_popup_files",
        help="Upload aptitude MCQ, technical, and/or HR question files. The system auto-detects the type.",
    )

    # ── Process uploaded files ────────────────────────────────────────────────
    results = []
    if uploaded_files:
        for f in uploaded_files[:3]:
            detected, valid, errors = _parse_file_auto(f)
            results.append({
                "name": f.name,
                "detected": detected,
                "valid": valid,
                "errors": errors,
            })

        # Show results
        st.markdown("<div style='margin:14px 0 6px'>", unsafe_allow_html=True)
        for r in results:
            n_ok = len(r["valid"])
            n_err = len(r["errors"])
            det = r["detected"]

            type_label = {"aptitude": "APTITUDE MCQ", "technical": "TECHNICAL", "hr": "HR"}.get(det, "UNKNOWN")
            badge_cls  = {"aptitude": "cq-badge-apt", "technical": "cq-badge-tech", "hr": "cq-badge-hr"}.get(det, "cq-badge-unk")
            status_cls = "cq-file-ok" if n_ok > 0 else "cq-file-err"
            status_icon = "✅" if n_ok > 0 and n_err == 0 else ("⚠️" if n_ok > 0 else "❌")

            st.markdown(f"""
<div class="cq-file-result {status_cls}">
  <span style="font-size:1.4rem">{status_icon}</span>
  <div style="flex:1;min-width:0;">
    <div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap;">
      <span style="color:#f1f5f9;font-size:.88rem;font-weight:600">{r['name']}</span>
      <span class="cq-type-badge {badge_cls}">{type_label}</span>
    </div>
    <div style="color:rgba(255,255,255,.45);font-size:.78rem;margin-top:3px;">
      {n_ok} question(s) ready
      {f' · ⚠ {n_err} row(s) skipped' if n_err else ''}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

            if r["errors"] and n_ok > 0:
                with st.expander(f"⚠ {n_err} row error(s) in {r['name']}", expanded=False):
                    for e in r["errors"][:10]:
                        st.caption(f"• {e}")

        # Summary stats
        apt_n  = sum(len(r["valid"]) for r in results if r["detected"] == "aptitude")
        tech_n = sum(len(r["valid"]) for r in results if r["detected"] == "technical")
        hr_n   = sum(len(r["valid"]) for r in results if r["detected"] == "hr")
        total  = apt_n + tech_n + hr_n

        if total > 0:
            st.markdown(f"""
<div class="cq-stat-row">
  {"<span class='cq-stat-pill cq-stat-pill-apt'>🧠 " + str(apt_n) + " Aptitude</span>" if apt_n else ""}
  {"<span class='cq-stat-pill cq-stat-pill-tech'>⚙️ " + str(tech_n) + " Technical</span>" if tech_n else ""}
  {"<span class='cq-stat-pill cq-stat-pill-hr'>🤝 " + str(hr_n) + " HR</span>" if hr_n else ""}
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Action buttons ────────────────────────────────────────────────────────
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    has_valid = any(len(r["valid"]) > 0 for r in results) if results else False

    col_start, col_skip = st.columns([3, 1])
    with col_start:
        btn_label = "✅  Use These Questions + Start Test" if has_valid else "▶  Start Test"
        btn_type = "primary"
        if st.button(btn_label, type=btn_type, use_container_width=True, key="cq_popup_start"):
            # Save detected questions into session state banks
            if results:
                for r in results:
                    if not r["valid"]:
                        continue
                    det = r["detected"]
                    bank_key = _BANK_KEYS.get(det)
                    if bank_key:
                        existing = st.session_state.get(bank_key, [])
                        # Deduplicate by question text
                        existing_qs = {q["question"] for q in existing}
                        new_qs = [q for q in r["valid"] if q["question"] not in existing_qs]
                        st.session_state[bank_key] = existing + new_qs

            st.session_state._cq_popup_open = False
            st.session_state._cq_popup_done = True
            return True

    with col_skip:
        if st.button("Skip ⏭", use_container_width=True, key="cq_popup_skip"):
            st.session_state._cq_popup_open = False
            st.session_state._cq_popup_done = True
            return True

    # ── Info note ─────────────────────────────────────────────────────────────
    if not uploaded_files:
        st.markdown("""
<p style="color:rgba(255,255,255,.28);font-size:.78rem;text-align:center;margin-top:8px;">
  No upload needed — AI will generate all questions. Click Skip to proceed.
</p>""", unsafe_allow_html=True)

    return False  # Still in popup, not started yet


# =============================================================================
# 5  CSS (page-level) — CINEMATIC ANIMATED REDESIGN
# =============================================================================

def _inject_css() -> None:
    st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>

/* ═══════════════════════════════════════════════════════════════════════════
   GLOBAL ANIMATED DARK BACKGROUND
   ═══════════════════════════════════════════════════════════════════════════ */

/* Animated grid background injected via ::before on main container */
[data-testid="stMainBlockContainer"] {
  position: relative;
  background: #020818 !important;
}

/* Starfield canvas overlay */
#cq-starfield-canvas {
  position: fixed; inset: 0;
  pointer-events: none; z-index: 0;
}

/* Floating orbs */
.cq-orb {
  position: fixed; border-radius: 50%;
  pointer-events: none; z-index: 0;
  filter: blur(80px); opacity: .18;
}
.cq-orb-1 {
  width: 500px; height: 500px;
  background: radial-gradient(circle, #6366f1, transparent);
  top: -120px; left: -100px;
  animation: cq-orb-drift1 18s ease-in-out infinite alternate;
}
.cq-orb-2 {
  width: 400px; height: 400px;
  background: radial-gradient(circle, #00d4ff, transparent);
  bottom: 10%; right: -80px;
  animation: cq-orb-drift2 22s ease-in-out infinite alternate;
}
.cq-orb-3 {
  width: 300px; height: 300px;
  background: radial-gradient(circle, #00ff88, transparent);
  top: 50%; left: 40%;
  animation: cq-orb-drift3 26s ease-in-out infinite alternate;
}
@keyframes cq-orb-drift1 { 0%{transform:translate(0,0);} 100%{transform:translate(60px,80px);} }
@keyframes cq-orb-drift2 { 0%{transform:translate(0,0);} 100%{transform:translate(-50px,-60px);} }
@keyframes cq-orb-drift3 { 0%{transform:translate(0,0) scale(1);} 100%{transform:translate(40px,-50px) scale(1.3);} }

/* Grid lines */
.cq-grid-lines {
  position: fixed; inset: 0;
  background-image:
    linear-gradient(rgba(99,102,241,.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(99,102,241,.04) 1px, transparent 1px);
  background-size: 60px 60px;
  pointer-events: none; z-index: 0;
  animation: cq-grid-pan 60s linear infinite;
}
@keyframes cq-grid-pan { 0%{background-position:0 0;} 100%{background-position:60px 60px;} }

/* Scanline overlay */
.cq-scanlines {
  position: fixed; inset: 0;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 3px,
    rgba(0,0,0,.015) 3px, rgba(0,0,0,.015) 4px
  );
  pointer-events: none; z-index: 1;
}

/* Content z-index above background */
[data-testid="stMainBlockContainer"] > div { position: relative; z-index: 2; }

/* ═══════════════════════════════════════════════════════════════════════════
   HEADER
   ═══════════════════════════════════════════════════════════════════════════ */
.cq-hero {
  text-align: center;
  padding: 52px 0 36px;
  position: relative;
}
.cq-hero-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; letter-spacing: 4px;
  color: rgba(99,102,241,.75);
  margin: 0 0 16px;
  animation: cq-fade-up .6s ease both;
}
.cq-hero-title {
  font-family: 'Syne', sans-serif;
  font-size: clamp(2rem, 5vw, 3.2rem);
  font-weight: 800; color: #f1f5f9;
  margin: 0 0 14px; line-height: 1.1;
  letter-spacing: -.02em;
  animation: cq-fade-up .7s .08s ease both;
}
.cq-hero-title .cq-glow-word {
  background: linear-gradient(90deg, #00d4ff, #6366f1, #00ff88);
  background-size: 200%;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: cq-gradient-shift 4s linear infinite;
}
@keyframes cq-gradient-shift { 0%{background-position:0%} 100%{background-position:200%} }
.cq-hero-sub {
  font-family: 'DM Sans', sans-serif;
  font-size: .95rem; color: rgba(255,255,255,.42);
  margin: 0; animation: cq-fade-up .8s .16s ease both;
}
@keyframes cq-fade-up {
  from { opacity:0; transform:translateY(18px); }
  to   { opacity:1; transform:translateY(0); }
}

/* Animated stat pills */
.cq-stat-strip {
  display: flex; justify-content: center; gap: 12px;
  flex-wrap: wrap; margin: 24px 0 0;
  animation: cq-fade-up .9s .24s ease both;
}
.cq-stat {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 8px 18px; border-radius: 100px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; letter-spacing: .5px;
  transition: transform .2s, box-shadow .2s;
  position: relative; overflow: hidden;
}
.cq-stat::after {
  content: ''; position: absolute; inset: 0;
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,.06) 50%, transparent 100%);
  background-size: 200%;
  animation: cq-stat-shimmer 3s linear infinite;
}
@keyframes cq-stat-shimmer { 0%{background-position:200%} 100%{background-position:-200%} }
.cq-stat:hover { transform: translateY(-2px); }
.cq-stat-apt { background:rgba(0,212,255,.08); color:#00d4ff; border:.5px solid rgba(0,212,255,.35);
  box-shadow:0 0 20px rgba(0,212,255,.08); }
.cq-stat-tec { background:rgba(99,102,241,.1); color:#a5b4fc; border:.5px solid rgba(99,102,241,.4);
  box-shadow:0 0 20px rgba(99,102,241,.08); }
.cq-stat-hr  { background:rgba(0,255,136,.08); color:#00ff88; border:.5px solid rgba(0,255,136,.35);
  box-shadow:0 0 20px rgba(0,255,136,.08); }

/* ═══════════════════════════════════════════════════════════════════════════
   BANK TYPE SELECTOR — glowing pill toggle
   ═══════════════════════════════════════════════════════════════════════════ */
.cq-type-selector {
  display: flex; gap: 8px; justify-content: center;
  margin: 0 0 28px; flex-wrap: wrap;
}
.cq-type-btn {
  padding: 10px 24px; border-radius: 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px; letter-spacing: 1.5px;
  cursor: pointer; transition: all .22s;
  border: 1.5px solid rgba(255,255,255,.08);
  background: rgba(255,255,255,.03); color: rgba(255,255,255,.4);
  position: relative; overflow: hidden;
}
.cq-type-btn-apt-on {
  border-color: #00d4ff; color: #00d4ff;
  background: rgba(0,212,255,.1);
  box-shadow: 0 0 28px rgba(0,212,255,.25), inset 0 0 20px rgba(0,212,255,.06);
}
.cq-type-btn-tec-on {
  border-color: #6366f1; color: #a5b4fc;
  background: rgba(99,102,241,.15);
  box-shadow: 0 0 28px rgba(99,102,241,.3), inset 0 0 20px rgba(99,102,241,.07);
}
.cq-type-btn-hr-on {
  border-color: #00ff88; color: #00ff88;
  background: rgba(0,255,136,.1);
  box-shadow: 0 0 28px rgba(0,255,136,.25), inset 0 0 20px rgba(0,255,136,.06);
}

/* ═══════════════════════════════════════════════════════════════════════════
   UPLOAD ZONE — animated drag-drop
   ═══════════════════════════════════════════════════════════════════════════ */
.cq-dropzone-wrap {
  position: relative; border-radius: 20px;
  padding: 2px; margin: 8px 0 16px;
  background: linear-gradient(90deg, #6366f1, #00d4ff, #00ff88, #f59e0b, #6366f1);
  background-size: 400%;
  animation: cq-border-flow 5s linear infinite;
}
@keyframes cq-border-flow { 0%{background-position:0%} 100%{background-position:400%} }
.cq-dropzone-inner {
  background: rgba(4,9,30,.95);
  border-radius: 18px; padding: 40px 32px;
  text-align: center;
  transition: background .2s;
}
.cq-dropzone-inner:hover { background: rgba(8,14,42,.95); }
.cq-upload-icon {
  width: 64px; height: 64px; margin: 0 auto 18px;
  position: relative;
}
.cq-upload-icon svg { width: 100%; height: 100%; }
.cq-upload-ring {
  position: absolute; inset: -8px; border-radius: 50%;
  border: 2px solid transparent;
  border-top-color: #6366f1; border-right-color: #00d4ff;
  animation: cq-ring-spin 2.5s linear infinite;
}
@keyframes cq-ring-spin { to { transform: rotate(360deg); } }
.cq-upload-ring2 {
  position: absolute; inset: -16px; border-radius: 50%;
  border: 1px solid transparent;
  border-bottom-color: rgba(0,255,136,.4); border-left-color: rgba(99,102,241,.3);
  animation: cq-ring-spin 4s linear infinite reverse;
}
.cq-dz-title {
  font-family: 'Syne', sans-serif; font-size: 1.15rem;
  font-weight: 700; color: #f1f5f9; margin: 0 0 6px;
}
.cq-dz-sub {
  font-family: 'JetBrains Mono', monospace; font-size: 10px;
  letter-spacing: 2px; color: rgba(255,255,255,.3);
}

/* ═══════════════════════════════════════════════════════════════════════════
   RESULT CARDS — animated appear
   ═══════════════════════════════════════════════════════════════════════════ */
.cq-card {
  background: rgba(4,9,26,.92);
  border: .5px solid rgba(255,255,255,.08);
  border-radius: 16px; padding: 20px 22px; margin-bottom: 12px;
  transition: border-color .25s, transform .2s, box-shadow .25s;
  animation: cq-card-in .4s ease both;
  position: relative; overflow: hidden;
}
@keyframes cq-card-in {
  from { opacity:0; transform:translateY(12px) scale(.98); }
  to   { opacity:1; transform:translateY(0) scale(1); }
}
.cq-card::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  width: 3px; border-radius: 3px;
  background: linear-gradient(180deg, #6366f1, #00d4ff);
  opacity: 0; transition: opacity .2s;
}
.cq-card:hover { border-color: rgba(99,102,241,.35); transform: translateX(3px);
  box-shadow: 0 8px 32px rgba(0,0,0,.3); }
.cq-card:hover::before { opacity: 1; }

/* Question preview */
.cq-preview {
  background: rgba(8,14,40,.8);
  border: .5px solid rgba(99,102,241,.2);
  border-left: 3px solid #6366f1;
  border-radius: 0 14px 14px 0;
  padding: 16px 20px; margin: 8px 0;
  animation: cq-card-in .35s ease both;
}
.cq-preview-q { color:#f1f5f9; font-size:.95rem; font-weight:600; margin-bottom:8px;
  font-family:'DM Sans',sans-serif; }
.cq-preview-opt { color:rgba(255,255,255,.5); font-size:.82rem; line-height:1.9;
  font-family:'JetBrains Mono',monospace; }
.cq-preview-correct { color:#00ff88; font-weight:700; }

/* Index badge */
.cq-idx {
  display:inline-flex; align-items:center; justify-content:center;
  width:28px; height:28px; border-radius:8px;
  background: rgba(99,102,241,.18); border:.5px solid rgba(99,102,241,.5);
  font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700;
  color:#a5b4fc; flex-shrink:0;
}

/* ═══════════════════════════════════════════════════════════════════════════
   BANNERS
   ═══════════════════════════════════════════════════════════════════════════ */
.cq-banner-ok {
  background:rgba(0,255,136,.06); border:.5px solid rgba(0,255,136,.4);
  border-radius:12px; padding:14px 20px; color:#00ff88;
  font-family:'DM Sans',sans-serif; font-size:.9rem; margin:10px 0;
  animation: cq-banner-pop .4s cubic-bezier(.34,1.56,.64,1) both;
  display:flex; align-items:center; gap:10px;
}
.cq-banner-err {
  background:rgba(255,51,102,.06); border:.5px solid rgba(255,51,102,.4);
  border-radius:12px; padding:14px 20px; color:#ff3366;
  font-family:'DM Sans',sans-serif; font-size:.9rem; margin:10px 0;
  animation: cq-banner-pop .4s cubic-bezier(.34,1.56,.64,1) both;
  display:flex; align-items:center; gap:10px;
}
.cq-banner-warn {
  background:rgba(245,158,11,.06); border:.5px solid rgba(245,158,11,.4);
  border-radius:12px; padding:14px 20px; color:#f59e0b;
  font-family:'DM Sans',sans-serif; font-size:.9rem; margin:10px 0;
  animation: cq-banner-pop .4s cubic-bezier(.34,1.56,.64,1) both;
  display:flex; align-items:center; gap:10px;
}
@keyframes cq-banner-pop {
  from { opacity:0; transform:scale(.92) translateY(8px); }
  to   { opacity:1; transform:scale(1) translateY(0); }
}

/* ═══════════════════════════════════════════════════════════════════════════
   PILL TYPE BADGES
   ═══════════════════════════════════════════════════════════════════════════ */
.cq-pill {
  display:inline-block; padding:4px 12px; border-radius:6px;
  font-family:'JetBrains Mono',monospace; font-size:9px; letter-spacing:1.5px;
  margin-right:6px;
}
.cq-pill-apt-on { background:rgba(0,212,255,.12); color:#00d4ff; border:.5px solid #00d4ff; }
.cq-pill-tec-on { background:rgba(99,102,241,.15); color:#a5b4fc; border:.5px solid #6366f1; }
.cq-pill-hr-on  { background:rgba(0,255,136,.10);  color:#00ff88; border:.5px solid #00ff88; }

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION LABELS
   ═══════════════════════════════════════════════════════════════════════════ */
.cq-section-label {
  font-family:'JetBrains Mono',monospace; font-size:10px;
  letter-spacing:3px; color:rgba(99,102,241,.7);
  margin:24px 0 12px; display:block;
}

/* ═══════════════════════════════════════════════════════════════════════════
   ANIMATED SUCCESS STATE (post-upload glow)
   ═══════════════════════════════════════════════════════════════════════════ */
.cq-success-glow {
  position:relative; border-radius:16px;
  background:rgba(0,255,136,.05);
  border:.5px solid rgba(0,255,136,.35);
  padding:22px 26px; margin:14px 0;
  animation:cq-success-pulse 2.5s ease-in-out infinite;
}
@keyframes cq-success-pulse {
  0%,100%{box-shadow:0 0 0px rgba(0,255,136,0);}
  50%{box-shadow:0 0 40px rgba(0,255,136,.15);}
}

/* ═══════════════════════════════════════════════════════════════════════════
   LAUNCH BUTTON AREA
   ═══════════════════════════════════════════════════════════════════════════ */
.cq-launch-wrap {
  margin:28px 0 0; position:relative; text-align:center;
}
.cq-launch-label {
  font-family:'JetBrains Mono',monospace; font-size:10px;
  letter-spacing:3px; color:rgba(255,255,255,.3);
  margin-bottom:12px; display:block;
}

/* Streamlit button overrides */
div[data-testid="stButton"] button {
  font-family:'DM Sans',sans-serif !important; font-weight:600 !important;
}

/* Streamlit tab overrides */
.stTabs [data-baseweb="tab-list"] {
  background:rgba(255,255,255,.03) !important;
  border-radius:12px !important; padding:4px !important;
  border:.5px solid rgba(255,255,255,.08) !important;
  gap:4px !important;
}
.stTabs [data-baseweb="tab"] {
  font-family:'JetBrains Mono',monospace !important;
  font-size:11px !important; letter-spacing:1.5px !important;
  border-radius:8px !important; padding:8px 20px !important;
  color:rgba(255,255,255,.4) !important;
}
.stTabs [aria-selected="true"] {
  background:rgba(99,102,241,.18) !important; color:#a5b4fc !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top:20px !important; }

/* File uploader styling */
[data-testid="stFileUploader"] {
  background:transparent !important;
}
[data-testid="stFileUploader"] > div > div {
  background:rgba(4,9,30,.95) !important;
  border:.5px solid rgba(99,102,241,.3) !important;
  border-radius:16px !important;
  transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stFileUploader"] > div > div:hover {
  border-color:rgba(99,102,241,.7) !important;
  box-shadow:0 0 30px rgba(99,102,241,.15) !important;
}

</style>

<div class="cq-orb cq-orb-1"></div>
<div class="cq-orb cq-orb-2"></div>
<div class="cq-orb cq-orb-3"></div>
<div class="cq-grid-lines"></div>
<div class="cq-scanlines"></div>
""", unsafe_allow_html=True)


# =============================================================================
# 6  HEADER + STATS BAR
# =============================================================================

def _render_header() -> None:
    mcq_n  = len(st.session_state.company_mcq_bank)
    tech_n = len(st.session_state.company_tech_bank)
    hr_n   = len(st.session_state.company_hr_bank)

    st.markdown("""
<div style="padding:28px 0 18px;border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:24px;">
  <p style="font-family:'Share Tech Mono',monospace;font-size:10px;
    color:rgba(99,102,241,.7);letter-spacing:3px;margin-bottom:10px;">COMPANY PORTAL</p>
  <h1 style="font-size:2rem;font-weight:700;color:#f1f5f9;margin:0 0 6px;">
    Question Bank Manager</h1>
  <p style="color:rgba(255,255,255,.38);font-size:.88rem;margin:0;">
    Upload a CSV or build questions one-by-one. They will be used in the Placement Test automatically.</p>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="margin-bottom:18px;">
  <span class="cq-stat cq-stat-apt">🧠 {mcq_n} Aptitude MCQs</span>
  <span class="cq-stat cq-stat-tec">⚙️ {tech_n} Technical Qs</span>
  <span class="cq-stat cq-stat-hr">🤝 {hr_n} HR Qs</span>
</div>""", unsafe_allow_html=True)


# =============================================================================
# 7  BANK TYPE SELECTOR
# =============================================================================

def _bank_selector() -> str:
    bt = st.session_state._cq_bank_type

    col1, col2, col3, _ = st.columns([1, 1, 1, 3])
    with col1:
        if st.button("🧠 Aptitude MCQ",
                     type="primary" if bt == "aptitude" else "secondary",
                     use_container_width=True):
            st.session_state._cq_bank_type = "aptitude"
            st.rerun()
    with col2:
        if st.button("⚙️ Technical",
                     type="primary" if bt == "technical" else "secondary",
                     use_container_width=True):
            st.session_state._cq_bank_type = "technical"
            st.rerun()
    with col3:
        if st.button("🤝 HR",
                     type="primary" if bt == "hr" else "secondary",
                     use_container_width=True):
            st.session_state._cq_bank_type = "hr"
            st.rerun()
    return st.session_state._cq_bank_type


# =============================================================================
# 8  CSV DOWNLOAD TEMPLATE
# =============================================================================

def _download_template(bank_type: str) -> None:
    df = _CSV_TEMPLATES[bank_type]
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        label=f"⬇ Download {bank_type.title()} CSV template",
        data=csv_bytes,
        file_name=f"aura_ai_{bank_type}_template.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =============================================================================
# 9  TAB A — CSV UPLOAD (single bank, with auto-detect option)
# =============================================================================

def _tab_upload(bank_type: str) -> None:
    st.markdown("### 📂 Upload Questions via CSV / Excel")

    # Auto-detect toggle
    use_auto = st.checkbox(
        "🤖 Auto-detect file type (recommended when uploading multiple files at once)",
        value=True,
        key=f"cq_auto_detect_{bank_type}",
    )

    with st.expander("📋 Download CSV Template first", expanded=False):
        st.markdown(f"""
<div class="cq-card">
  <p style="color:rgba(255,255,255,.55);font-size:.85rem;margin:0 0 12px;">
    Download the template, fill in your questions, and re-upload below.
  </p>
</div>""", unsafe_allow_html=True)
        _download_template(bank_type)

        if bank_type == "aptitude":
            st.markdown("""
**Required columns:**
`question`, `option_A`, `option_B`, `option_C`, `option_D`, `correct` (A/B/C/D), `topic`, `difficulty` (easy/medium/hard), `explanation`
""")
        else:
            st.markdown("""
**Required columns:**
`question`, `topic`, `difficulty` (easy/medium/hard)
""")

    st.markdown('<div class="cq-dropzone">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        f"Drop your {bank_type} questions CSV / XLSX here",
        type=["csv", "xlsx"],
        key=f"cq_upload_{bank_type}",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded is None:
        st.markdown("""
<p style="text-align:center;color:rgba(255,255,255,.25);font-size:.8rem;margin-top:-8px;">
  Supports .csv and .xlsx files · Max 500 questions per upload
</p>""", unsafe_allow_html=True)
        return

    # Parse
    with st.spinner("Validating questions…"):
        if use_auto:
            detected, valid, errors = _parse_file_auto(uploaded)
            if detected != bank_type:
                st.markdown(
                    f'<div class="cq-banner-warn">🤖 Auto-detected as <strong>{detected.upper()}</strong> '
                    f'(not {bank_type}). Questions will be added to the <strong>{detected}</strong> bank.</div>',
                    unsafe_allow_html=True,
                )
                bank_type = detected
        else:
            valid, errors = _parse_file(uploaded, bank_type)

    if errors:
        with st.expander(f"⚠️ {len(errors)} row(s) had issues — click to review", expanded=True):
            for e in errors[:20]:
                st.markdown(f'<div class="cq-banner-warn">⚠ {e}</div>', unsafe_allow_html=True)
            if len(errors) > 20:
                st.caption(f"… and {len(errors)-20} more errors not shown.")

    if not valid:
        st.markdown('<div class="cq-banner-err">❌ No valid questions found. Fix the errors and re-upload.</div>',
                    unsafe_allow_html=True)
        return

    st.markdown(f'<div class="cq-banner-ok">✅ {len(valid)} valid question(s) ready to import.</div>',
                unsafe_allow_html=True)

    with st.expander(f"👁 Preview ({min(5, len(valid))} of {len(valid)})", expanded=True):
        for i, q in enumerate(valid[:5]):
            _render_question_preview(q, i)

    bank_key = _BANK_KEYS[bank_type]
    existing_n = len(st.session_state[bank_key])

    merge_mode = "replace"
    if existing_n > 0:
        st.markdown(f"""
<div class="cq-card" style="margin-top:10px;">
  <p style="color:#f1f5f9;font-size:.9rem;font-weight:600;margin:0 0 8px;">
    You already have {existing_n} questions in this bank.</p>
  <p style="color:rgba(255,255,255,.45);font-size:.82rem;margin:0;">
    Choose how to handle the import:</p>
</div>""", unsafe_allow_html=True)
        merge_mode = st.radio(
            "Import mode",
            ["Append to existing", "Replace all existing"],
            key=f"cq_merge_{bank_type}",
            label_visibility="collapsed",
        )

    col_imp, col_cancel = st.columns([2, 1])
    with col_imp:
        if st.button(f"✅ Import {len(valid)} Question(s)", type="primary",
                     use_container_width=True, key=f"cq_import_{bank_type}"):
            if existing_n > 0 and merge_mode == "Replace all existing":
                st.session_state[bank_key] = valid
            else:
                st.session_state[bank_key] = st.session_state[bank_key] + valid
            st.success(f"✅ Imported {len(valid)} questions into the {bank_type} bank!")
            st.rerun()
    with col_cancel:
        if st.button("Cancel", use_container_width=True, key=f"cq_cancel_{bank_type}"):
            st.rerun()


# =============================================================================
# 10  QUESTION PREVIEW CARD
# =============================================================================

def _render_question_preview(q: Dict, idx: int) -> None:
    qf    = q.get("q_format", "text")
    topic = q.get("topic", "")
    diff  = q.get("difficulty", "")
    diff_col = {"easy": "#00ff88", "medium": "#f59e0b", "hard": "#ff3366"}.get(diff, "#a5b4fc")

    if qf == "mcq":
        opts    = q.get("options", {})
        correct = q.get("correct", "")
        opts_html = "".join(
            f'<span class="{"cq-preview-correct" if k == correct else ""}">'
            f'{k}. {v}{" ✓" if k == correct else ""}</span><br>'
            for k, v in opts.items()
        )
        expl = q.get("explanation", "")
        st.markdown(f"""
<div class="cq-preview">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
    <span class="cq-idx">{idx+1}</span>
    <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:rgba(0,212,255,.8);">MCQ</span>
    <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:rgba(255,255,255,.3);">{topic}</span>
    <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:{diff_col};">{diff.upper()}</span>
  </div>
  <div class="cq-preview-q">{q['question']}</div>
  <div class="cq-preview-opt">{opts_html}</div>
  {'<div style="font-size:.78rem;color:rgba(255,255,255,.35);margin-top:6px;font-style:italic">💡 ' + expl + '</div>' if expl else ''}
</div>""", unsafe_allow_html=True)
    else:
        q_type    = q.get("q_type", "technical")
        type_col  = {"technical": "#a5b4fc", "hr": "#00ff88"}.get(q_type, "#a5b4fc")
        st.markdown(f"""
<div class="cq-preview">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
    <span class="cq-idx">{idx+1}</span>
    <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:{type_col};">{q_type.upper()}</span>
    <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:rgba(255,255,255,.3);">{topic}</span>
    <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:{diff_col};">{diff.upper()}</span>
  </div>
  <div class="cq-preview-q">{q['question']}</div>
</div>""", unsafe_allow_html=True)


# =============================================================================
# 11  TAB B — QUESTION BUILDER
# =============================================================================

def _tab_builder(bank_type: str) -> None:
    st.markdown("### 🛠️ Build a Question")

    topics = {
        "aptitude":  _TOPICS_MCQ,
        "technical": _TOPICS_TECH,
        "hr":        _TOPICS_HR,
    }[bank_type]

    is_mcq = (bank_type == "aptitude")

    with st.form(key=f"cq_builder_form_{bank_type}", clear_on_submit=True):
        st.markdown("#### Question Details")

        q_text = st.text_area(
            "Question *",
            placeholder="Enter your question here…",
            height=100,
            key=f"cq_b_q_{bank_type}",
        )

        col_t, col_d = st.columns(2)
        with col_t:
            topic = st.selectbox("Topic", topics, key=f"cq_b_topic_{bank_type}")
        with col_d:
            difficulty = st.selectbox("Difficulty", _DIFFICULTIES, key=f"cq_b_diff_{bank_type}")

        if is_mcq:
            st.markdown("#### Answer Options")
            col_a, col_b = st.columns(2)
            col_c, col_d2 = st.columns(2)
            with col_a:
                opt_a = st.text_input("Option A *", key=f"cq_b_oa_{bank_type}")
            with col_b:
                opt_b = st.text_input("Option B *", key=f"cq_b_ob_{bank_type}")
            with col_c:
                opt_c = st.text_input("Option C *", key=f"cq_b_oc_{bank_type}")
            with col_d2:
                opt_d = st.text_input("Option D *", key=f"cq_b_od_{bank_type}")

            correct = st.radio(
                "Correct Answer *",
                ["A", "B", "C", "D"],
                horizontal=True,
                key=f"cq_b_correct_{bank_type}",
            )
            explanation = st.text_input(
                "Explanation (shown after answer)",
                placeholder="Brief explanation of the correct answer…",
                key=f"cq_b_expl_{bank_type}",
            )
        else:
            opt_a = opt_b = opt_c = opt_d = correct = explanation = ""

        submitted = st.form_submit_button(
            "➕ Add Question to Bank",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        errors = []
        if not q_text.strip():
            errors.append("Question text is required.")
        if is_mcq:
            for lbl, val in [("A", opt_a), ("B", opt_b), ("C", opt_c), ("D", opt_d)]:
                if not val.strip():
                    errors.append(f"Option {lbl} is required.")

        if errors:
            for e in errors:
                st.markdown(f'<div class="cq-banner-err">❌ {e}</div>', unsafe_allow_html=True)
        else:
            if is_mcq:
                new_q = {
                    "question":    q_text.strip(),
                    "options":     {"A": opt_a.strip(), "B": opt_b.strip(),
                                    "C": opt_c.strip(), "D": opt_d.strip()},
                    "correct":     correct,
                    "topic":       topic,
                    "difficulty":  difficulty,
                    "explanation": explanation.strip(),
                    "q_format":    "mcq",
                    "source":      "company",
                }
            else:
                new_q = {
                    "question":   q_text.strip(),
                    "topic":      topic,
                    "difficulty": difficulty,
                    "q_format":   "text",
                    "q_type":     bank_type,
                    "source":     "company",
                }

            bank_key = _BANK_KEYS[bank_type]
            st.session_state[bank_key].append(new_q)
            n = len(st.session_state[bank_key])
            st.markdown(
                f'<div class="cq-banner-ok">✅ Question added! Bank now has <strong>{n}</strong> question(s).</div>',
                unsafe_allow_html=True,
            )

    bank_key = _BANK_KEYS[bank_type]
    n = len(st.session_state[bank_key])
    if n > 0:
        st.markdown(f"""
<div class="cq-card" style="margin-top:16px;">
  <p style="color:rgba(255,255,255,.5);font-size:.82rem;margin:0;">
    📚 <strong style="color:#f1f5f9;">{n}</strong> question(s) currently in the {bank_type} bank.
    Switch to the <strong style="color:#00d4ff;">Manage</strong> tab to review or delete them.
  </p>
</div>""", unsafe_allow_html=True)


# =============================================================================
# 12  TAB C — MANAGE / REVIEW BANK
# =============================================================================

def _tab_manage(bank_type: str) -> None:
    bank_key = _BANK_KEYS[bank_type]
    bank: List[Dict] = st.session_state[bank_key]

    st.markdown(f"### 📋 Manage {bank_type.title()} Bank  ({len(bank)} questions)")

    if not bank:
        st.markdown("""
<div class="cq-card" style="text-align:center;padding:40px;">
  <div style="font-size:2.5rem;margin-bottom:12px;">📭</div>
  <p style="color:rgba(255,255,255,.4);font-size:.9rem;">
    No questions yet. Use Upload or Builder to add some.</p>
</div>""", unsafe_allow_html=True)
        return

    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        search = st.text_input("🔍 Search", placeholder="Filter by keyword…",
                               key=f"cq_search_{bank_type}", label_visibility="collapsed")
    with col_f2:
        diff_filter = st.selectbox("Difficulty", ["All"] + _DIFFICULTIES,
                                   key=f"cq_diff_filter_{bank_type}", label_visibility="collapsed")
    with col_f3:
        df_export = _bank_to_dataframe(bank, bank_type)
        csv_bytes = df_export.to_csv(index=False).encode()
        st.download_button(
            "⬇ Export CSV",
            data=csv_bytes,
            file_name=f"company_{bank_type}_questions.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"cq_export_{bank_type}",
        )

    filtered = [
        (i, q) for i, q in enumerate(bank)
        if (not search or search.lower() in q.get("question", "").lower()
                       or search.lower() in q.get("topic", "").lower())
        and (diff_filter == "All" or q.get("difficulty") == diff_filter)
    ]

    st.markdown(f"<p style='color:rgba(255,255,255,.3);font-size:.78rem;margin:4px 0 12px;'>"
                f"Showing {len(filtered)} of {len(bank)}</p>", unsafe_allow_html=True)

    to_delete = []
    for display_pos, (orig_idx, q) in enumerate(filtered):
        col_q, col_del = st.columns([10, 1])
        with col_q:
            _render_question_preview(q, orig_idx)
        with col_del:
            st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
            if st.button("🗑", key=f"cq_del_{bank_type}_{orig_idx}",
                         help="Delete this question"):
                to_delete.append(orig_idx)

    if to_delete:
        for idx in sorted(to_delete, reverse=True):
            if 0 <= idx < len(st.session_state[bank_key]):
                st.session_state[bank_key].pop(idx)
        st.rerun()

    st.markdown("---")
    with st.expander("⚠️ Danger Zone"):
        st.markdown(f"""
<div class="cq-card">
  <p style="color:#ff3366;font-size:.88rem;margin:0 0 12px;">
    This will permanently delete all {len(bank)} questions from the {bank_type} bank.</p>
</div>""", unsafe_allow_html=True)
        if st.button(f"🗑 Clear entire {bank_type} bank",
                     type="secondary",
                     key=f"cq_clear_{bank_type}"):
            st.session_state[f"_cq_confirm_clear_{bank_type}"] = True

        if st.session_state.get(f"_cq_confirm_clear_{bank_type}", False):
            st.warning("Are you sure? This cannot be undone.")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, delete all", type="primary",
                             key=f"cq_clear_yes_{bank_type}"):
                    st.session_state[bank_key] = []
                    st.session_state[f"_cq_confirm_clear_{bank_type}"] = False
                    st.rerun()
            with col_no:
                if st.button("Cancel", key=f"cq_clear_no_{bank_type}"):
                    st.session_state[f"_cq_confirm_clear_{bank_type}"] = False
                    st.rerun()


# =============================================================================
# 13  EXPORT HELPER
# =============================================================================

def _bank_to_dataframe(bank: List[Dict], bank_type: str) -> pd.DataFrame:
    rows = []
    for q in bank:
        if bank_type == "aptitude":
            opts = q.get("options", {})
            rows.append({
                "question":    q.get("question", ""),
                "option_A":    opts.get("A", ""),
                "option_B":    opts.get("B", ""),
                "option_C":    opts.get("C", ""),
                "option_D":    opts.get("D", ""),
                "correct":     q.get("correct", ""),
                "topic":       q.get("topic", ""),
                "difficulty":  q.get("difficulty", ""),
                "explanation": q.get("explanation", ""),
            })
        else:
            rows.append({
                "question":   q.get("question", ""),
                "topic":      q.get("topic", ""),
                "difficulty": q.get("difficulty", ""),
            })
    return pd.DataFrame(rows)


# =============================================================================
# 14  JSON IMPORT / EXPORT
# =============================================================================

def _json_io_panel(bank_type: str) -> None:
    bank_key = _BANK_KEYS[bank_type]
    bank = st.session_state[bank_key]

    with st.expander("🔧 Advanced: JSON Import / Export", expanded=False):
        col_exp, col_imp = st.columns(2)
        with col_exp:
            st.markdown("**Export as JSON**")
            json_str = json.dumps(bank, indent=2, ensure_ascii=False)
            st.download_button(
                "⬇ Download JSON",
                data=json_str.encode(),
                file_name=f"company_{bank_type}_questions.json",
                mime="application/json",
                use_container_width=True,
                key=f"cq_json_exp_{bank_type}",
            )

        with col_imp:
            st.markdown("**Import from JSON**")
            json_file = st.file_uploader(
                "Upload JSON",
                type=["json"],
                key=f"cq_json_upload_{bank_type}",
                label_visibility="collapsed",
            )
            if json_file:
                try:
                    data = json.loads(json_file.read().decode())
                    if not isinstance(data, list):
                        st.error("JSON must be a list of question objects.")
                    else:
                        st.info(f"{len(data)} questions found.")
                        if st.button("Import JSON", key=f"cq_json_import_{bank_type}",
                                     use_container_width=True):
                            st.session_state[bank_key] = st.session_state[bank_key] + data
                            st.success(f"Imported {len(data)} questions!")
                            st.rerun()
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")


# =============================================================================
# 15  MAIN PAGE
# =============================================================================

def page_company_questions() -> None:
    """
    Main entry point. Call from app.py router:
        elif p == "Company Questions": page_company_questions()
    """
    _init_defaults()
    _inject_css()
    _render_header()

    bank_type = _bank_selector()
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    tab_upload, tab_build, tab_manage = st.tabs([
        "📂  CSV Upload",
        "🛠️  Question Builder",
        "📋  Manage Bank",
    ])

    with tab_upload:
        _tab_upload(bank_type)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        _json_io_panel(bank_type)

    with tab_build:
        _tab_builder(bank_type)

    with tab_manage:
        _tab_manage(bank_type)

    mcq_n  = len(st.session_state.company_mcq_bank)
    tech_n = len(st.session_state.company_tech_bank)
    hr_n   = len(st.session_state.company_hr_bank)

    if mcq_n + tech_n + hr_n > 0:
        st.markdown("---")
        st.markdown(f"""
<div class="cq-card" style="border-color:rgba(0,212,255,.25);">
  <p style="color:#00d4ff;font-family:'Share Tech Mono',monospace;font-size:10px;
    letter-spacing:2px;margin:0 0 8px;">READY FOR PLACEMENT TEST</p>
  <p style="color:rgba(255,255,255,.65);font-size:.88rem;margin:0;">
    Your company bank has
    <strong style="color:#f1f5f9;">{mcq_n} Aptitude</strong> ·
    <strong style="color:#f1f5f9;">{tech_n} Technical</strong> ·
    <strong style="color:#f1f5f9;">{hr_n} HR</strong> questions.
    These will be used automatically in the next Placement Test session.
  </p>
</div>""", unsafe_allow_html=True)


# =============================================================================
# 16  INTEGRATION HELPERS  (called from placement_test_mode.py)
# =============================================================================

def get_company_mcq_batch(n: int, difficulty: str = "easy") -> List[Dict]:
    """Return up to n MCQs from the company bank."""
    bank: List[Dict] = st.session_state.get("company_mcq_bank", [])
    if not bank:
        return []
    preferred = [q for q in bank if q.get("difficulty") == difficulty]
    pool = preferred if len(preferred) >= n else bank
    chosen = random.sample(pool, min(n, len(pool)))
    for q in chosen:
        q.setdefault("q_format", "mcq")
    return chosen


def get_company_text_batch(n: int, q_type: str = "technical") -> List[Dict]:
    """Return up to n text questions for a given round type."""
    key = "company_tech_bank" if q_type == "technical" else "company_hr_bank"
    bank: List[Dict] = st.session_state.get(key, [])
    if not bank:
        return []
    chosen = random.sample(bank, min(n, len(bank)))
    for q in chosen:
        q.setdefault("q_format", "text")
        q.setdefault("q_type", q_type)
    return chosen


def has_company_questions(q_type: str = "aptitude") -> bool:
    """Return True if there are company questions for the given type."""
    key_map = {
        "aptitude":  "company_mcq_bank",
        "technical": "company_tech_bank",
        "hr":        "company_hr_bank",
    }
    return len(st.session_state.get(key_map.get(q_type, "company_mcq_bank"), [])) > 0

# =============================================================================
# 17  PLACEMENT SETUP PAGE  (required by app.py)
# =============================================================================

PLACEMENT_SETUP_DEFAULTS: dict = {
    "_ps_apt_count":   10,
    "_ps_tech_count":  5,
    "_ps_hr_count":    5,
    "_ps_setup_done":  False,
}


def page_placement_setup() -> None:
    """
    Pre-test configuration page shown before the Placement Test starts.
    Lets the recruiter/candidate choose how many questions per round,
    and optionally upload a company question bank inline.

    Called from app.py router:
        elif p == "Placement Setup": page_placement_setup()
    """
    _init_defaults()
    _inject_css()

    st.markdown("""
<div style="padding:28px 0 18px;">
  <p style="font-family:'Share Tech Mono',monospace;font-size:10px;
    color:rgba(99,102,241,.7);letter-spacing:3px;margin-bottom:10px;">PLACEMENT TEST</p>
  <h1 style="font-size:2rem;font-weight:700;color:#f1f5f9;margin:0 0 6px;">
    Test Setup</h1>
  <p style="color:rgba(255,255,255,.38);font-size:.88rem;margin:0;">
    Configure your placement test rounds, then start when ready.</p>
</div>""", unsafe_allow_html=True)

    st.markdown("### Round Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""<div style="background:rgba(0,212,255,.06);border:.5px solid rgba(0,212,255,.25);
  border-radius:12px;padding:14px 16px;text-align:center;margin-bottom:8px;">
  <div style="font-size:22px;margin-bottom:4px;">🧠</div>
  <p style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#00d4ff;margin:0 0 8px;">APTITUDE MCQ</p>
</div>""", unsafe_allow_html=True)
        apt_count = st.slider(
            "Aptitude questions", 5, 20,
            int(st.session_state.get("_ps_apt_count", 10)),
            key="_ps_slider_apt",
            label_visibility="collapsed",
        )
        st.caption(f"{apt_count} MCQ questions · ~{apt_count} min")
        st.session_state["_ps_apt_count"] = apt_count

    with col2:
        st.markdown("""<div style="background:rgba(99,102,241,.08);border:.5px solid rgba(99,102,241,.25);
  border-radius:12px;padding:14px 16px;text-align:center;margin-bottom:8px;">
  <div style="font-size:22px;margin-bottom:4px;">⚙️</div>
  <p style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#a5b4fc;margin:0 0 8px;">TECHNICAL</p>
</div>""", unsafe_allow_html=True)
        tech_count = st.slider(
            "Technical questions", 3, 10,
            int(st.session_state.get("_ps_tech_count", 5)),
            key="_ps_slider_tech",
            label_visibility="collapsed",
        )
        st.caption(f"{tech_count} open-ended questions · ~{tech_count * 2} min")
        st.session_state["_ps_tech_count"] = tech_count

    with col3:
        st.markdown("""<div style="background:rgba(0,255,136,.06);border:.5px solid rgba(0,255,136,.25);
  border-radius:12px;padding:14px 16px;text-align:center;margin-bottom:8px;">
  <div style="font-size:22px;margin-bottom:4px;">🤝</div>
  <p style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#00ff88;margin:0 0 8px;">HR</p>
</div>""", unsafe_allow_html=True)
        hr_count = st.slider(
            "HR questions", 3, 10,
            int(st.session_state.get("_ps_hr_count", 5)),
            key="_ps_slider_hr",
            label_visibility="collapsed",
        )
        st.caption(f"{hr_count} behavioural questions · ~{hr_count * 1.5:.0f} min")
        st.session_state["_ps_hr_count"] = hr_count

    # ── Company question bank inline ──────────────────────────────────────────
    st.markdown("---")
    mcq_n  = len(st.session_state.get("company_mcq_bank", []))
    tech_n = len(st.session_state.get("company_tech_bank", []))
    hr_n   = len(st.session_state.get("company_hr_bank", []))
    total  = mcq_n + tech_n + hr_n

    if total > 0:
        st.markdown(f"""
<div style="background:rgba(0,255,136,.06);border:.5px solid rgba(0,255,136,.3);
  border-radius:10px;padding:12px 18px;margin-bottom:14px;
  font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:1.5px;color:#00ff88;">
  ✦ COMPANY BANK LOADED &nbsp;·&nbsp;
  {mcq_n} Aptitude &nbsp;·&nbsp; {tech_n} Technical &nbsp;·&nbsp; {hr_n} HR
</div>""", unsafe_allow_html=True)
    else:
        with st.expander("🏢 Upload Company Questions (optional)", expanded=False):
            try:
                page_company_questions()
            except Exception as _err:
                st.error(f"Company questions panel error: {_err}")

    # ── Start button ──────────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    total_min = apt_count + tech_count * 2 + hr_count * 1.5
    st.markdown(f"""
<div style="background:rgba(4,9,26,.85);border:.5px solid rgba(99,102,241,.25);
  border-radius:12px;padding:14px 20px;margin-bottom:16px;
  font-family:'Share Tech Mono',monospace;font-size:11px;color:rgba(255,255,255,.45);">
  📊 Total: {apt_count + tech_count + hr_count} questions &nbsp;·&nbsp;
  ⏱ ~{total_min:.0f} minutes estimated
</div>""", unsafe_allow_html=True)

    if st.button("▶  Proceed to Placement Test", type="primary",
                 use_container_width=True, key="_ps_proceed_btn"):
        st.session_state["_ps_setup_done"] = True
        st.session_state["page"] = "Placement Test"
        st.rerun()