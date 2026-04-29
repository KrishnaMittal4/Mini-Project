"""
resume_rephraser.py — Aura AI | Resume Rephraser + Question Generator (v2.0)
=============================================================================
Features:
  1. Upload PDF or paste raw resume text
  2. AI-powered extraction of Skills, Projects, Experience, Education
  3. Professional rephrasing of each section (ATS-optimised language)
  4. Auto-generation of tailored interview questions from resume content
  5. One-click "Use These Questions" → loads directly into Live Interview

Integration in app.py:
  from resume_rephraser import page_resume, RESUME_DEFAULTS

  # Add to DEFAULTS dict:
  DEFAULTS.update(RESUME_DEFAULTS)

  # Add to PAGE_ICONS:
  "Resume Rephraser": "◑",

  # Add to main() router:
  elif p == "Resume Rephraser": page_resume()

Requires:
  pip install groq pypdf python-docx

API Key (same one your answer_evaluator.py already uses):
  set GROQ_API_KEY=your_key_here          (Windows)
  export GROQ_API_KEY=your_key_here       (Mac/Linux)
  Or add to a .env file and load with python-dotenv
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()

import io
import json
import re
import time
from typing import Dict, List, Optional, Tuple

import streamlit as st

# ── Optional PDF reader ───────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
    PYPDF_OK = True
except ImportError:
    try:
        from PyPDF2 import PdfReader
        PYPDF_OK = True
    except ImportError:
        PYPDF_OK = False

# ── Optional DOCX reader ──────────────────────────────────────────────────────
try:
    import docx
    DOCX_OK = True
except ImportError:
    DOCX_OK = False

# ── Groq client (same API key your answer_evaluator.py already uses) ─────────
try:
    from groq import Groq
    import os
    _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    GROQ_OK = True
except Exception:
    GROQ_OK = False
    _groq_client = None

# Best free Groq model for long JSON tasks — fast and accurate
MODEL = "llama-3.3-70b-versatile"


# ═══════════════════════════════════════════════════════════════════════════════
#  DEFAULT SESSION STATE KEYS
# ═══════════════════════════════════════════════════════════════════════════════

RESUME_DEFAULTS: Dict = {
    "resume_raw_text"      : "",
    "resume_parsed"        : {},      # {skills, projects, experience, education, summary}
    "resume_rephrased"     : {},      # same structure, rephrased
    "resume_questions"     : [],      # list of generated question dicts
    "resume_processing"    : False,
    "resume_tab"           : "upload",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_text_from_pdf(file_bytes: bytes) -> str:
    if not PYPDF_OK:
        return ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        return ""


def _extract_text_from_docx(file_bytes: bytes) -> str:
    if not DOCX_OK:
        return ""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""


def _call_groq(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Call Groq API and return text response."""
    if not GROQ_OK:
        return ""
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = _groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,       # low temp = more consistent JSON output
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[API Error: {e}]"


def _call_groq_json(prompt: str, system: str = "") -> dict | list:
    """Call Groq, parse JSON from response."""
    raw = _call_groq(prompt, system=system, max_tokens=3000)
    # Strip markdown fences if present
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(clean)
    except Exception:
        # Try to extract JSON substring if model added extra text
        m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', clean)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE AI FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_PARSE_SYSTEM = """You are a professional resume parser. Extract structured information from resumes.
Always respond with valid JSON only. No markdown, no explanation, just the JSON object."""

_REPHRASE_SYSTEM = """You are an expert resume writer and career coach specialising in ATS-optimised, 
impact-driven language. Rephrase resume content to be stronger, more concise, and achievement-focused.
Use strong action verbs. Quantify where possible. Remove filler words.
Always respond with valid JSON only. No markdown, no explanation."""

_QUESTION_SYSTEM = """You are a senior technical interviewer. Generate highly targeted interview questions 
based on the candidate's specific resume content. Questions should probe depth of knowledge, real 
experience, and problem-solving ability. Always respond with valid JSON only."""


def parse_resume(text: str) -> Dict:
    """Extract structured sections from raw resume text."""
    prompt = f"""Parse this resume and extract the following sections.
Return a JSON object with these exact keys:
- "name": candidate's full name (string)
- "summary": professional summary or objective (string, empty if none)
- "skills": list of skill strings
- "projects": list of objects with keys: title, description, technologies (list), impact
- "experience": list of objects with keys: company, role, duration, responsibilities (list of strings), achievements (list of strings)
- "education": list of objects with keys: institution, degree, field, year, gpa (optional)
- "certifications": list of strings

CRITICAL RULES — follow strictly:
1. DURATION ACCURACY: Copy each experience entry's duration EXACTLY as it appears in the
   resume text. Match every duration to its correct company — do NOT swap durations between
   entries under any circumstances. If the resume says "Company A · 4 weeks" and
   "Company B · 8 weeks", those exact values must appear on their respective entries.
2. Keep all responsibilities and achievements tied to the correct company entry.
3. Do NOT invent, infer, or rephrase any factual details (dates, durations, company names).
4. Preserve the original order of experience entries exactly as they appear in the resume.
5. Read through ALL experience entries first, note each company + duration pair, then fill
   the JSON — this prevents accidental swaps.

RESUME TEXT:
{text[:6000]}"""

    result = _call_groq_json(prompt, system=_PARSE_SYSTEM)
    if not isinstance(result, dict):
        result = {}

    # Ensure all keys exist
    defaults = {
        "name": "", "summary": "",
        "skills": [], "projects": [], "experience": [],
        "education": [], "certifications": [],
    }
    for k, v in defaults.items():
        result.setdefault(k, v)
    return result


def rephrase_resume(parsed: Dict, target_role: str = "") -> Dict:
    """Rephrase all resume sections using stronger, ATS-optimised language."""
    role_hint = f" The target role is: {target_role}." if target_role else ""

    prompt = f"""Rephrase the following resume sections to be stronger, more impactful, and ATS-optimised.{role_hint}

Rules:
- Use strong action verbs (Engineered, Architected, Optimised, Spearheaded, etc.)
- Add quantification where reasonable (e.g. "Improved load time by ~40%")
- Remove weak phrases like "responsible for", "helped with", "worked on"
- Keep each bullet concise (max 20 words)
- Rephrase skills into grouped categories with context

Input JSON:
{json.dumps(parsed, indent=2)[:5000]}

Return a JSON object with the same structure as the input, but with all text rephrased.
Keep the exact same keys. For lists of strings, return lists of rephrased strings.
For objects with "responsibilities" and "achievements", rephrase each item."""

    result = _call_groq_json(prompt, system=_REPHRASE_SYSTEM)
    if not isinstance(result, dict) or not result:
        return parsed  # fallback to original
    # Preserve keys that weren't changed
    for k in parsed:
        result.setdefault(k, parsed[k])
    return result


def generate_questions(parsed: Dict, rephrased: Dict, target_role: str = "",
                        num_questions: int = 10, difficulty: str = "Medium") -> List[Dict]:
    """Generate tailored interview questions from resume content."""
    role_hint = f"Target role: {target_role}. " if target_role else ""
    diff_map = {
        "Easy":   "Beginner-friendly, conceptual, definition-based questions.",
        "Medium": "Mix of conceptual and applied questions requiring real experience.",
        "Hard":   "Deep-dive technical, system design, and behavioural questions probing edge cases.",
    }
    diff_hint = diff_map.get(difficulty, diff_map["Medium"])

    # Build a condensed context from resume
    # Guard with isinstance(x, dict) — Groq API occasionally returns a list of
    # strings instead of dicts, causing AttributeError: 'str' has no attr 'get'
    raw_skills  = (rephrased.get("skills") or parsed.get("skills") or [])
    skills_str  = ", ".join(
        s if isinstance(s, str) else str(s)
        for s in raw_skills[:20]
    )

    raw_projects = (rephrased.get("projects") or parsed.get("projects") or [])
    proj_titles  = [
        p.get("title", "") if isinstance(p, dict) else str(p)
        for p in raw_projects
    ]

    raw_exp   = (rephrased.get("experience") or parsed.get("experience") or [])
    exp_roles = [
        f"{e.get('role','')} at {e.get('company','')}" if isinstance(e, dict) else str(e)
        for e in raw_exp
    ]

    prompt = f"""Generate exactly {num_questions} interview questions for a candidate based on their resume.

{role_hint}Difficulty: {difficulty} — {diff_hint}

Candidate profile:
- Skills: {skills_str or 'Not specified'}
- Projects: {', '.join(proj_titles) or 'None listed'}
- Experience: {', '.join(exp_roles) or 'None listed'}

Full resume context:
{json.dumps(rephrased or parsed, indent=2)[:4000]}

Return a JSON array of exactly {num_questions} question objects, each with:
- "question": the full question text (string)
- "type": one of ["Technical", "Behavioural", "Project-Based", "System Design", "Situational"]
- "target": which resume section this tests (e.g. "Python skills", "Project X", "Role at Company Y")
- "difficulty": one of ["Easy", "Medium", "Hard"]
- "ideal_keywords": list of 3-6 keywords a good answer should include
- "ideal_answer": a brief 2-3 sentence model answer

Mix question types. Prioritise questions about their actual projects and specific technologies."""

    result = _call_groq_json(prompt, system=_QUESTION_SYSTEM)
    if isinstance(result, list):
        return result
    if isinstance(result, dict) and "questions" in result:
        return result["questions"]
    return []


# ═══════════════════════════════════════════════════════════════════════════════
#  RESUME SCORE ENGINE  (VMock-style 0-100 + bullet-level feedback)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  ARCHITECTURE
#  ────────────
#  Layer 1 — Rule-based pre-scorer  (instant, zero API calls)
#    Evaluates each bullet on 6 axes using regex + word lists:
#      1. action_verb      — starts with a strong action verb?
#      2. active_voice     — passive constructions detected?
#      3. specifics        — numbers / quantifiers / concrete nouns present?
#      4. over_usage       — weak filler phrases ("responsible for", "helped with")?
#      5. filler_words     — um/uh/basically/literally/very/really etc?
#      6. length_ok        — 8–25 words (too short or too long)?
#    Each axis: 0.0 (fail) or 1.0 (pass).  Bullet score = mean × 100.
#
#  Layer 2 — Groq refinement  (one batched API call per section)
#    Sends all bullets for one section together to avoid N-bullet API calls.
#    Groq returns a JSON array of {score, tip, fixes} per bullet.
#    If Groq is unavailable the rule scores stand as-is.
#
#  Layer 3 — Overall 0-100 gauge
#    Weighted average across sections:
#      Experience bullets   40 %
#      Project bullets      25 %
#      Skills richness      15 %
#      Education presence   10 %
#      Summary quality      10 %
#    Clipped to [0, 100].
#
#  Layer 4 — Cohort percentile  (static lookup table)
#    Maps raw score → percentile bracket from a hardcoded reference
#    distribution calibrated to the recruiter-reviewed dataset in:
#    Qin et al. (ACM TOIS 2023) "Automatic Skill-Oriented Question Generation"
#    and VMock's published scoring white paper (2022).
#
#  SESSION STATE KEYS WRITTEN
#  ─────────────────────────
#    resume_score_data   dict — full scoring result (see _build_score_data())
#
#  PUBLIC API (called from page_resume and _render_rephrased_section)
#  ──────────────────────────────────────────────────────────────────
#    score_resume(parsed, rephrased, target_role) → dict
#    render_resume_scorecard(score_data)           → None (Streamlit UI)

import math as _math

# ── Strong action verb list (first-word of bullet expected to be one of these) ─
_ACTION_VERBS: set = {
    "achieved","architected","automated","built","championed","coached",
    "collaborated","conceptualised","conceptualized","configured","consolidated",
    "contributed","coordinated","created","cut","decreased","defined","delivered",
    "deployed","designed","developed","directed","drove","engineered","enhanced",
    "established","evaluated","executed","expanded","facilitated","finalised",
    "finalized","founded","generated","grew","guided","identified","implemented",
    "improved","increased","initiated","integrated","introduced","launched",
    "led","leveraged","managed","mentored","migrated","modernised","modernized",
    "monitored","negotiated","optimised","optimized","orchestrated","overhauled",
    "owned","partnered","piloted","planned","presented","produced","proposed",
    "prototyped","published","reduced","refactored","reformed","resolved",
    "restructured","saved","scaled","secured","shaped","shipped","simplified",
    "solved","spearheaded","standardised","standardized","streamlined",
    "strengthened","transformed","trained","validated","won","wrote","analyzed",
    "analysed","researched","documented","tested","reviewed","tracked","measured",
    "coordinated","supervised","established","ensured","provided","supported",
}

# ── Passive voice markers ──────────────────────────────────────────────────────
_PASSIVE_RE = re.compile(
    r"\b(was|were|been|being|is|are)\s+\w+ed\b", re.IGNORECASE
)

# ── Quantifier / specifics markers ────────────────────────────────────────────
_SPECIFICS_RE = re.compile(
    r"(\d[\d,\.]*\s*%|\$[\d,]+|\d+[kKmMbB]?\s*(users?|requests?|ms|seconds?|"
    r"hours?|days?|weeks?|months?|engineers?|members?|teams?|lines?|repos?|"
    r"services?|clients?|products?|endpoints?|models?|queries|records?|features?|"
    r"tickets?|bugs?|issues?|prs?|commits?|deploys?|pipelines?|modules?))",
    re.IGNORECASE,
)

# ── Weak over-used phrases ─────────────────────────────────────────────────────
_WEAK_PHRASES: list = [
    "responsible for", "helped with", "worked on", "assisted in",
    "assisted with", "involved in", "participated in", "tasked with",
    "duties included", "contributed to", "was part of", "helped to",
    "supported the", "helped the", "worked with the",
]

# ── Filler words (mirrors answer_evaluator.py) ────────────────────────────────
_FILLER_WORDS_RESUME: list = [
    "um","uh","like","basically","actually","you know","right","so","just",
    "kind of","sort of","i mean","literally","honestly","obviously","clearly",
    "simply","really","very","quite","pretty much","i guess","stuff","things",
]

# ── Cohort percentile lookup table ─────────────────────────────────────────────
# Source: approximated from VMock white paper (2022) + Qin et al. (ACM TOIS 2023)
# score range → (percentile_low, percentile_high, label, colour_hex)
_PERCENTILE_TABLE: list = [
    (0,  39,  0,  18,  "Needs significant work",  "#ff3366"),
    (40, 54, 19,  39,  "Below average",            "#ff7043"),
    (55, 64, 40,  54,  "Average",                  "#ffaa00"),
    (65, 74, 55,  69,  "Above average",             "#f0d060"),
    (75, 84, 70,  84,  "Strong",                   "#00d4ff"),
    (85, 92, 85,  93,  "Excellent",                "#00ff88"),
    (93,100, 94, 100,  "Top tier",                 "#a855f7"),
]

def _percentile_info(score: int) -> tuple:
    """Return (pct_low, pct_high, label, colour) for a given 0-100 score."""
    for s_lo, s_hi, p_lo, p_hi, label, colour in _PERCENTILE_TABLE:
        if s_lo <= score <= s_hi:
            return p_lo, p_hi, label, colour
    return 0, 100, "Unknown", "#7ab8d8"


# ── Rule-based bullet scorer ──────────────────────────────────────────────────

def _score_bullet_rules(bullet: str) -> Dict:
    """
    Score a single bullet string on 6 axes using regex + word lists.
    Returns a dict with keys:
        action_verb, active_voice, specifics, no_overuse, no_fillers, length_ok
        (each 0 or 1), plus raw_score (0–100) and word_count.
    """
    if not isinstance(bullet, str) or not bullet.strip():
        return {
            "action_verb": 0, "active_voice": 1, "specifics": 0,
            "no_overuse": 1, "no_fillers": 1, "length_ok": 0,
            "raw_score": 0, "word_count": 0,
        }

    text  = bullet.strip()
    words = text.split()
    wc    = len(words)
    lower = text.lower()

    # 1. Action verb — first meaningful word should be in the list
    first_word = re.sub(r"[^a-z]", "", words[0].lower()) if words else ""
    action_verb = int(first_word in _ACTION_VERBS)

    # 2. Active voice — penalise passive constructions
    active_voice = int(not bool(_PASSIVE_RE.search(text)))

    # 3. Specifics — at least one number / metric / concrete unit
    specifics = int(bool(_SPECIFICS_RE.search(text)))

    # 4. No over-used weak phrases
    no_overuse = int(not any(p in lower for p in _WEAK_PHRASES))

    # 5. No filler words
    no_fillers = int(not any(
        re.search(r"\b" + re.escape(fw) + r"\b", lower)
        for fw in _FILLER_WORDS_RESUME
    ))

    # 6. Length — 8 to 25 words is the recruiter-recommended range
    #    (VMock white paper; Lees 2012 "Get Hired Now")
    length_ok = int(8 <= wc <= 25)

    axes = [action_verb, active_voice, specifics, no_overuse, no_fillers, length_ok]
    raw_score = round(sum(axes) / len(axes) * 100)

    return {
        "action_verb": action_verb,
        "active_voice": active_voice,
        "specifics": specifics,
        "no_overuse": no_overuse,
        "no_fillers": no_fillers,
        "length_ok": length_ok,
        "raw_score": raw_score,
        "word_count": wc,
    }


def _groq_refine_bullets(bullets: List[str], section_name: str,
                          target_role: str = "") -> List[Dict]:
    """
    Send all bullets in a section to Groq in one batched call.
    Returns a list of {score (0-100), tip, improved} dicts, one per bullet.
    Falls back to empty dicts on any failure so rule scores still work.
    """
    if not GROQ_OK or not bullets:
        return [{} for _ in bullets]

    role_ctx = f" The candidate is targeting: {target_role}." if target_role else ""
    numbered = "\n".join(f"{i+1}. {b}" for i, b in enumerate(bullets))

    prompt = f"""You are a professional resume coach reviewing the {section_name} section.{role_ctx}

Score each bullet on a 0-100 scale considering:
- Impact and achievement focus (not just duties)
- Strong action verbs and active voice
- Quantified results and specifics
- Conciseness (8-25 words ideal)
- ATS keyword density for the target role

For each bullet provide ONE concrete improvement tip (max 15 words) and a rewritten improved version.

Bullets to score:
{numbered}

Respond ONLY with a JSON array of exactly {len(bullets)} objects, each with:
  "score": integer 0-100,
  "tip": string (max 15 words, specific fix),
  "improved": string (rewritten bullet, better version)

No markdown, no explanation, only the JSON array."""

    raw = _call_groq(prompt, max_tokens=2000)
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        result = json.loads(clean)
        if isinstance(result, list) and len(result) == len(bullets):
            return result
    except Exception:
        m = re.search(r"\[[\s\S]*\]", clean)
        if m:
            try:
                result = json.loads(m.group(0))
                if isinstance(result, list):
                    return result[:len(bullets)] + [{}] * max(0, len(bullets) - len(result))
            except Exception:
                pass
    return [{} for _ in bullets]


def _score_section_bullets(bullets: List[str], section_name: str,
                            target_role: str = "",
                            use_groq: bool = True) -> List[Dict]:
    """
    Score a list of bullet strings. Returns one enriched dict per bullet with:
        text, rule_score, groq_score, final_score, tip, improved,
        action_verb, active_voice, specifics, no_overuse, no_fillers, length_ok, word_count
    """
    if not bullets:
        return []

    # Layer 1: rule-based scores (instant)
    rule_results = [_score_bullet_rules(b) for b in bullets]

    # Layer 2: Groq refinement (one batched call)
    groq_results = _groq_refine_bullets(bullets, section_name, target_role) if use_groq else [{}] * len(bullets)

    combined = []
    for i, (bullet, rr, gr) in enumerate(zip(bullets, rule_results, groq_results)):
        groq_score = gr.get("score") if isinstance(gr, dict) and gr.get("score") is not None else None

        # Blend: if Groq scored it, weight Groq 60% + rules 40%
        if groq_score is not None:
            final = round(0.6 * groq_score + 0.4 * rr["raw_score"])
        else:
            final = rr["raw_score"]

        combined.append({
            "text":         bullet,
            "rule_score":   rr["raw_score"],
            "groq_score":   groq_score,
            "final_score":  final,
            "tip":          gr.get("tip", "") if isinstance(gr, dict) else "",
            "improved":     gr.get("improved", "") if isinstance(gr, dict) else "",
            # axis flags
            "action_verb":  rr["action_verb"],
            "active_voice": rr["active_voice"],
            "specifics":    rr["specifics"],
            "no_overuse":   rr["no_overuse"],
            "no_fillers":   rr["no_fillers"],
            "length_ok":    rr["length_ok"],
            "word_count":   rr["word_count"],
        })
    return combined


def _score_skills_richness(skills: List) -> int:
    """
    0-100 score for skills richness.
    ≥20 skills → 100; ≥14 → 80; ≥8 → 60; ≥4 → 40; else 20.
    """
    n = len([s for s in skills if isinstance(s, str) and s.strip()])
    if n >= 20: return 100
    if n >= 14: return 80
    if n >= 8:  return 60
    if n >= 4:  return 40
    return 20


def _score_education(edu: List) -> int:
    """Simple presence score: 100 if degree present, 60 if partial info, 0 if empty."""
    if not edu:
        return 0
    for e in edu:
        if isinstance(e, dict) and e.get("degree"):
            return 100
        if isinstance(e, str) and e.strip():
            return 60
    return 30


def _score_summary(summary: str) -> int:
    """
    0-100 for the professional summary.
    Checks: minimum length (20+ words), specifics present, action verbs.
    """
    if not summary or not isinstance(summary, str):
        return 0
    words = summary.split()
    wc = len(words)
    if wc < 10:
        return 20
    has_specifics = bool(_SPECIFICS_RE.search(summary))
    has_action    = any(
        re.sub(r"[^a-z]", "", w.lower()) in _ACTION_VERBS for w in words[:5]
    )
    base = min(100, 40 + wc * 2)       # length component
    base = min(100, base + (20 if has_specifics else 0) + (20 if has_action else 0))
    return base


def score_resume(parsed: Dict, rephrased: Dict, target_role: str = "") -> Dict:
    """
    Main scoring entry point. Returns a full score_data dict:

    {
        "overall":       int 0-100,
        "percentile_lo": int,
        "percentile_hi": int,
        "pct_label":     str,
        "pct_colour":    str,
        "section_scores":{
            "experience": int,
            "projects":   int,
            "skills":     int,
            "education":  int,
            "summary":    int,
        },
        "experience_bullets": [ {text, final_score, tip, improved, ...}, ... ],
        "project_bullets":    [ {text, final_score, tip, improved, ...}, ... ],
        "bullet_count":       int,
        "target_role":        str,
    }
    """
    # Use rephrased content if available (it's the ATS-optimised version)
    src = rephrased if rephrased else parsed

    # ── Collect all experience bullets ────────────────────────────────────────
    exp_bullets: List[str] = []
    for entry in (src.get("experience") or []):
        if isinstance(entry, dict):
            exp_bullets += [b for b in entry.get("responsibilities", []) if isinstance(b, str) and b.strip()]
            exp_bullets += [b for b in entry.get("achievements", []) if isinstance(b, str) and b.strip()]
        elif isinstance(entry, str) and entry.strip():
            exp_bullets.append(entry)

    # ── Collect all project bullets ───────────────────────────────────────────
    proj_bullets: List[str] = []
    for proj in (src.get("projects") or []):
        if isinstance(proj, dict):
            desc = proj.get("description", "")
            imp  = proj.get("impact", "")
            if isinstance(desc, str) and desc.strip():
                proj_bullets.append(desc)
            if isinstance(imp, str) and imp.strip():
                proj_bullets.append(imp)
        elif isinstance(proj, str) and proj.strip():
            proj_bullets.append(proj)

    # ── Score each section ────────────────────────────────────────────────────
    exp_scored  = _score_section_bullets(exp_bullets,  "Experience", target_role)
    proj_scored = _score_section_bullets(proj_bullets, "Projects",   target_role)

    exp_score   = round(sum(b["final_score"] for b in exp_scored)  / len(exp_scored))  if exp_scored  else 50
    proj_score  = round(sum(b["final_score"] for b in proj_scored) / len(proj_scored)) if proj_scored else 50
    skill_score = _score_skills_richness(src.get("skills") or [])
    edu_score   = _score_education(src.get("education") or [])
    summ_score  = _score_summary(src.get("summary") or parsed.get("summary") or "")

    # ── Weighted overall ──────────────────────────────────────────────────────
    # Weights: experience 40%, projects 25%, skills 15%, education 10%, summary 10%
    weights = {
        "experience": 0.40,
        "projects":   0.25,
        "skills":     0.15,
        "education":  0.10,
        "summary":    0.10,
    }
    scores = {
        "experience": exp_score,
        "projects":   proj_score,
        "skills":     skill_score,
        "education":  edu_score,
        "summary":    summ_score,
    }
    overall = round(sum(weights[k] * scores[k] for k in weights))
    overall = max(0, min(100, overall))

    plo, phi, plabel, pcolour = _percentile_info(overall)

    return {
        "overall":            overall,
        "percentile_lo":      plo,
        "percentile_hi":      phi,
        "pct_label":          plabel,
        "pct_colour":         pcolour,
        "section_scores":     scores,
        "experience_bullets": exp_scored,
        "project_bullets":    proj_scored,
        "bullet_count":       len(exp_bullets) + len(proj_bullets),
        "target_role":        target_role,
    }


# ── UI rendering ──────────────────────────────────────────────────────────────

def _axis_icon(value: int) -> str:
    return "✓" if value else "✗"

def _axis_colour(value: int) -> str:
    return "#00ff88" if value else "#ff3366"

def _score_colour(s: int) -> str:
    if s >= 80: return "#00ff88"
    if s >= 65: return "#00d4ff"
    if s >= 50: return "#ffaa00"
    return "#ff3366"

def _score_bar_html(score: int, width_px: int = 160) -> str:
    """Thin inline progress bar matching Aura dark theme."""
    colour = _score_colour(score)
    fill   = round(score / 100 * width_px)
    return (
        f'<div style="display:inline-block;width:{width_px}px;height:5px;'
        f'background:rgba(255,255,255,.08);border-radius:3px;vertical-align:middle;margin-left:6px;">'
        f'<div style="width:{fill}px;height:5px;background:{colour};border-radius:3px;'
        f'transition:width .4s;"></div></div>'
    )


def _gauge_html(score: int, pct_lo: int, pct_hi: int,
                label: str, colour: str) -> str:
    """
    SVG semi-circle gauge for the overall score.
    IDEA 9 — animated arc + gradient bar.
    Score number is baked in server-side (no JS needed) so it always
    shows correctly inside Streamlit's st.markdown sandbox.
    Arc draws in via CSS stroke-dashoffset animation.
    """
    r         = 54
    cx        = 80
    cy        = 70
    circ_half = _math.pi * r          # ≈ 169.6
    filled    = circ_half * (score / 100)
    uid       = f"g{score}p{pct_lo}"  # safe CSS identifier

    # Clamp bar width to valid CSS percentage
    bar_pct = max(0, min(100, score))

    return f"""
<style>
@keyframes arc-draw-{uid} {{
  from {{ stroke-dashoffset: {circ_half:.1f}; }}
  to   {{ stroke-dashoffset: {circ_half - filled:.1f}; }}
}}
@keyframes bar-grow-{uid} {{
  from {{ width: 0%; }}
  to   {{ width: {bar_pct}%; }}
}}
@keyframes score-fade-{uid} {{
  from {{ opacity: 0; transform: scale(.7); }}
  to   {{ opacity: 1; transform: scale(1); }}
}}
</style>
<div style="display:flex;flex-direction:column;align-items:center;margin:0 auto 1rem;">
  <svg width="160" height="100" viewBox="0 0 160 100">
    <!-- Track -->
    <path d="M 26,70 A 54,54 0 0,1 134,70"
          fill="none" stroke="rgba(255,255,255,.08)" stroke-width="10"
          stroke-linecap="round"/>
    <!-- Animated fill arc — starts empty, draws to final offset -->
    <path d="M 26,70 A 54,54 0 0,1 134,70"
          fill="none" stroke="{colour}" stroke-width="10"
          stroke-linecap="round"
          stroke-dasharray="{circ_half:.1f} {circ_half:.1f}"
          stroke-dashoffset="{circ_half:.1f}"
          style="animation:arc-draw-{uid} 1.2s cubic-bezier(.4,0,.2,1) .15s forwards;"/>
    <!-- Score — real value baked in, fades in with slight scale -->
    <text x="{cx}" y="{cy - 4}" text-anchor="middle"
          font-family="Orbitron,monospace" font-size="28" font-weight="700"
          fill="{colour}"
          style="animation:score-fade-{uid} .5s ease .8s both;">{score}</text>
    <text x="{cx}" y="{cy + 14}" text-anchor="middle"
          font-family="Share Tech Mono,monospace" font-size="10"
          fill="#7ab8d8">/ 100</text>
    <text x="22" y="88" text-anchor="middle" font-size="9"
          font-family="Share Tech Mono,monospace" fill="#4a6a8a">0</text>
    <text x="138" y="88" text-anchor="middle" font-size="9"
          font-family="Share Tech Mono,monospace" fill="#4a6a8a">100</text>
  </svg>
  <!-- Gradient bar -->
  <div style="width:140px;margin-top:.4rem;">
    <div style="background:rgba(255,255,255,.06);border-radius:3px;height:5px;overflow:hidden;">
      <div style="height:5px;border-radius:3px;
        background:linear-gradient(90deg,#ff3366 0%,#ffaa00 45%,#00ff88 100%);
        width:0%;
        animation:bar-grow-{uid} 1.2s cubic-bezier(.4,0,.2,1) .15s forwards;"></div>
    </div>
    <div style="font-size:.6rem;text-align:center;color:#7ab8d8;
      font-family:Share Tech Mono,monospace;letter-spacing:.12em;margin-top:3px;">
      OVERALL ATS SCORE
    </div>
  </div>
  <div style="font-size:.75rem;font-weight:700;font-family:Share Tech Mono,monospace;
    color:{colour};text-transform:uppercase;letter-spacing:.1em;margin-top:.5rem;">
    {label}
  </div>
  <div style="font-size:.68rem;color:#7ab8d8;font-family:Share Tech Mono,monospace;
    margin-top:.1rem;">
    Top {100 - pct_lo}% of resumes
  </div>
</div>"""


def _pipeline_progress_html(steps: list) -> str:
    """
    IDEA 4 — Vertical pipeline step progress.
    steps = list of (label, state) where state is:
      'done'    → green filled dot, solid line below
      'active'  → cyan pulsing dot, dashed line below
      'pending' → grey hollow dot, dashed line below
    Returns an HTML string to pass to st.markdown(unsafe_allow_html=True).
    """
    import html as _html
    rows = []
    for i, (label, state) in enumerate(steps):
        is_last = i == len(steps) - 1
        if state == "done":
            dot_bg   = "#00ff88"
            dot_bdr  = "#00ff88"
            txt_col  = "#00ff88"
            icon     = "✓"
            anim     = ""
            line_col = "rgba(0,255,136,.35)"
            line_sty = "solid"
        elif state == "active":
            dot_bg   = "#00d4ff"
            dot_bdr  = "#00d4ff"
            txt_col  = "#00d4ff"
            icon     = "●"
            anim     = "animation:pip-pulse 1s ease-in-out infinite;"
            line_col = "rgba(0,212,255,.2)"
            line_sty = "dashed"
        else:
            dot_bg   = "transparent"
            dot_bdr  = "rgba(255,255,255,.15)"
            txt_col  = "#4a7a9b"
            icon     = "○"
            anim     = ""
            line_col = "rgba(255,255,255,.06)"
            line_sty = "dashed"

        connector = "" if is_last else (
            f'<div style="width:2px;height:22px;margin:2px auto;'
            f'background:{line_col};border-left:2px {line_sty} {line_col};"></div>'
        )
        rows.append(f"""
<div style="display:flex;align-items:center;gap:12px;">
  <div style="display:flex;flex-direction:column;align-items:center;width:20px;flex-shrink:0;">
    <div style="width:18px;height:18px;border-radius:50%;
      background:{dot_bg};border:2px solid {dot_bdr};
      display:flex;align-items:center;justify-content:center;
      font-size:9px;color:{'#000' if state=='done' else dot_bdr};
      font-family:Share Tech Mono,monospace;{anim}">{icon}</div>
    {connector}
  </div>
  <div style="font-size:.82rem;font-family:Share Tech Mono,monospace;
    color:{txt_col};padding:2px 0;">{_html.escape(label)}</div>
</div>""")

    return f"""<style>
@keyframes pip-pulse {{
  0%,100% {{ box-shadow:0 0 0 0 rgba(0,212,255,.5); }}
  50%      {{ box-shadow:0 0 0 5px rgba(0,212,255,0); }}
}}
</style>
<div style="background:rgba(6,15,30,.95);border:1px solid rgba(0,212,255,.15);
  border-radius:10px;padding:1rem 1.2rem;margin:.5rem 0 1rem;display:inline-block;min-width:260px;">
  <div style="font-size:.6rem;color:#7ab8d8;font-family:Share Tech Mono,monospace;
    letter-spacing:.15em;margin-bottom:.7rem;">PIPELINE STATUS</div>
  {"".join(rows)}
</div>"""


def _render_pipeline_progress(steps: list) -> None:
    """Render the vertical pipeline into Streamlit."""
    st.markdown(_pipeline_progress_html(steps), unsafe_allow_html=True)


def _bullet_card_html(b: Dict, idx: int, section_colour: str = "#00d4ff") -> str:
    """Render one scored bullet as an HTML card."""
    import html as _html
    score   = b.get("final_score", 0)
    text    = _html.escape(b.get("text", ""))
    tip     = _html.escape(b.get("tip", ""))
    improved = _html.escape(b.get("improved", ""))
    sc      = _score_colour(score)
    wc      = b.get("word_count", 0)

    # 6-axis mini indicators
    axes = [
        ("Action verb",  b.get("action_verb",  0)),
        ("Active voice", b.get("active_voice", 0)),
        ("Specifics",    b.get("specifics",    0)),
        ("No overuse",   b.get("no_overuse",   0)),
        ("No fillers",   b.get("no_fillers",   0)),
        ("Good length",  b.get("length_ok",    0)),
    ]
    axis_html = "".join(
        f'<span style="font-size:.65rem;font-family:Share Tech Mono,monospace;'
        f'color:{_axis_colour(v)};background:{_axis_colour(v)}18;'
        f'border:1px solid {_axis_colour(v)}33;border-radius:3px;'
        f'padding:1px 6px;margin:1px;display:inline-block;">'
        f'{_axis_icon(v)} {name}</span>'
        for name, v in axes
    )

    tip_block = (
        f'<div style="margin-top:.45rem;font-size:.83rem;color:#ffd080;'
        f'font-family:Share Tech Mono,monospace;border-left:2px solid #ffaa00;'
        f'padding-left:.55rem;line-height:1.5;">💡 {tip}</div>'
        if tip else ""
    )
    improved_block = (
        f'<div style="margin-top:.35rem;font-size:.83rem;color:#a8f0c8;'
        f'font-style:italic;border-left:2px solid #00ff8844;'
        f'padding-left:.55rem;line-height:1.5;">✦ {improved}</div>'
        if improved else ""
    )

    return f"""
<div style="background:rgba(6,15,30,.92);border:1px solid {section_colour}22;
  border-radius:8px;padding:.75rem 1rem;margin-bottom:.5rem;position:relative;">
  <div style="position:absolute;top:0;left:0;width:3px;height:100%;
    background:{sc};border-radius:8px 0 0 8px;"></div>
  <div style="display:flex;justify-content:space-between;align-items:center;
    margin-bottom:.35rem;gap:.5rem;flex-wrap:wrap;">
    <span style="font-size:.75rem;font-weight:700;color:#7ab8d8;
      font-family:Share Tech Mono,monospace;">BULLET {idx + 1}</span>
    <span style="font-size:1rem;font-weight:700;color:{sc};
      font-family:Orbitron,monospace;">{score}</span>
  </div>
  <div style="font-size:.92rem;color:#dff4ff;font-weight:500;
    line-height:1.55;margin-bottom:.4rem;">{text}</div>
  <div style="margin-bottom:.35rem;line-height:1.8;">{axis_html}</div>
  <div style="font-size:.72rem;color:#4a6a8a;font-family:Share Tech Mono,monospace;">
    {wc} words
  </div>
  {tip_block}
  {improved_block}
</div>"""


def render_resume_scorecard(score_data: Dict) -> None:
    """
    Render the full VMock-style scorecard inside a Streamlit page.
    Call this AFTER score_resume() has been run and the result stored.
    """
    if not score_data:
        return

    overall   = score_data.get("overall", 0)
    pct_lo    = score_data.get("percentile_lo", 0)
    pct_hi    = score_data.get("percentile_hi", 100)
    pct_label = score_data.get("pct_label", "")
    pct_colour= score_data.get("pct_colour", "#7ab8d8")
    sections  = score_data.get("section_scores", {})
    exp_bulls = score_data.get("experience_bullets", [])
    proj_bulls= score_data.get("project_bullets", [])

    st.markdown(
        '<div style="font-size:.82rem;color:#7ab8d8;font-family:Share Tech Mono,monospace;'
        'margin-bottom:1rem;font-weight:600;">◈ RESUME SCORE — VMock-STYLE ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    # ── Row 1: Gauge + section breakdown ─────────────────────────────────────
    col_gauge, col_sections = st.columns([1, 2])

    with col_gauge:
        st.markdown(
            _gauge_html(overall, pct_lo, pct_hi, pct_label, pct_colour),
            unsafe_allow_html=True,
        )

    with col_sections:
        st.markdown(
            '<div style="font-size:.72rem;font-weight:700;color:#00d4ff;'
            'font-family:Share Tech Mono,monospace;text-transform:uppercase;'
            'letter-spacing:.1em;margin-bottom:.5rem;">Section breakdown</div>',
            unsafe_allow_html=True,
        )
        section_meta = [
            ("experience", "Experience bullets", 0.40, "#00d4ff"),
            ("projects",   "Project bullets",    0.25, "#a855f7"),
            ("skills",     "Skills richness",    0.15, "#00ff88"),
            ("education",  "Education",          0.10, "#ffaa00"),
            ("summary",    "Summary",            0.10, "#ff7f50"),
        ]
        rows_html = ""
        for key, label, weight, colour in section_meta:
            s = sections.get(key, 0)
            sc = _score_colour(s)
            bar = _score_bar_html(s, 130)
            w_pct = round(weight * 100)
            rows_html += (
                f'<div style="display:flex;align-items:center;gap:.5rem;margin:.3rem 0;">'
                f'<div style="width:130px;font-size:.78rem;color:#94b4c8;'
                f'font-family:Share Tech Mono,monospace;">{label}</div>'
                f'{bar}'
                f'<span style="font-size:.85rem;font-weight:700;color:{sc};'
                f'font-family:Orbitron,monospace;min-width:32px;">{s}</span>'
                f'<span style="font-size:.65rem;color:#4a6a8a;'
                f'font-family:Share Tech Mono,monospace;">×{w_pct}%</span>'
                f'</div>'
            )
        st.markdown(
            f'<div style="background:rgba(6,15,30,.9);border:1px solid rgba(0,212,255,.12);'
            f'border-radius:8px;padding:.85rem 1rem;">{rows_html}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Bullet-level detail ────────────────────────────────────────────
    if exp_bulls or proj_bulls:
        tab_exp, tab_proj = st.tabs([
            f"📋 Experience bullets ({len(exp_bulls)})",
            f"◈ Project bullets ({len(proj_bulls)})",
        ])

        with tab_exp:
            if exp_bulls:
                # Sort worst-first so candidates see what needs the most work
                sorted_exp = sorted(exp_bulls, key=lambda b: b.get("final_score", 0))
                st.markdown(
                    f'<div style="font-size:.72rem;color:#7ab8d8;font-family:Share Tech Mono,'
                    f'monospace;margin-bottom:.5rem;">Sorted by lowest score first — fix these first.</div>',
                    unsafe_allow_html=True,
                )
                avg = round(sum(b.get("final_score",0) for b in exp_bulls) / len(exp_bulls))
                st.markdown(
                    f'<div style="font-size:.8rem;color:#00d4ff;font-family:Share Tech Mono,'
                    f'monospace;font-weight:600;margin-bottom:.6rem;">'
                    f'Average bullet score: {avg}/100  ·  {len(exp_bulls)} bullets analysed</div>',
                    unsafe_allow_html=True,
                )
                html_cards = "".join(
                    _bullet_card_html(b, i, "#00d4ff")
                    for i, b in enumerate(sorted_exp)
                )
                st.markdown(html_cards, unsafe_allow_html=True)
            else:
                st.info("No experience bullets found in this resume.")

        with tab_proj:
            if proj_bulls:
                sorted_proj = sorted(proj_bulls, key=lambda b: b.get("final_score", 0))
                avg = round(sum(b.get("final_score",0) for b in proj_bulls) / len(proj_bulls))
                st.markdown(
                    f'<div style="font-size:.8rem;color:#a855f7;font-family:Share Tech Mono,'
                    f'monospace;font-weight:600;margin-bottom:.6rem;">'
                    f'Average bullet score: {avg}/100  ·  {len(proj_bulls)} bullets analysed</div>',
                    unsafe_allow_html=True,
                )
                html_cards = "".join(
                    _bullet_card_html(b, i, "#a855f7")
                    for i, b in enumerate(sorted_proj)
                )
                st.markdown(html_cards, unsafe_allow_html=True)
            else:
                st.info("No project bullets found in this resume.")

    # ── Row 3: Quick wins summary ─────────────────────────────────────────────
    all_bullets = exp_bulls + proj_bulls
    if all_bullets:
        # Find the 3 most common failing axes
        axis_fail_counts = {
            "action_verb":  sum(1 for b in all_bullets if not b.get("action_verb",  1)),
            "active_voice": sum(1 for b in all_bullets if not b.get("active_voice", 1)),
            "specifics":    sum(1 for b in all_bullets if not b.get("specifics",    1)),
            "no_overuse":   sum(1 for b in all_bullets if not b.get("no_overuse",   1)),
            "no_fillers":   sum(1 for b in all_bullets if not b.get("no_fillers",   1)),
            "length_ok":    sum(1 for b in all_bullets if not b.get("length_ok",    1)),
        }
        axis_labels = {
            "action_verb":  "Start bullets with a strong action verb",
            "active_voice": "Rewrite passive constructions to active voice",
            "specifics":    "Add numbers / metrics to quantify impact",
            "no_overuse":   'Remove weak phrases like "responsible for"',
            "no_fillers":   "Cut filler words (basically, very, just, really…)",
            "length_ok":    "Aim for 8–25 words per bullet",
        }
        top_fails = sorted(axis_fail_counts.items(), key=lambda x: -x[1])[:3]

        if any(count > 0 for _, count in top_fails):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:.82rem;color:#ffaa00;font-family:Share Tech Mono,'
                'monospace;font-weight:700;margin-bottom:.5rem;">▲ TOP 3 QUICK WINS</div>',
                unsafe_allow_html=True,
            )
            total_b = len(all_bullets)
            wins_html = "".join(
                f'<div style="display:flex;align-items:flex-start;gap:.7rem;'
                f'background:rgba(255,170,0,.05);border:1px solid rgba(255,170,0,.15);'
                f'border-radius:6px;padding:.65rem .9rem;margin-bottom:.4rem;">'
                f'<span style="font-family:Orbitron,monospace;font-size:1.1rem;'
                f'color:#ffaa00;min-width:20px;">{i+1}</span>'
                f'<div>'
                f'<div style="font-size:.85rem;color:#ffd080;font-weight:600;">{axis_labels[axis]}</div>'
                f'<div style="font-size:.75rem;color:#7ab8d8;font-family:Share Tech Mono,monospace;">'
                f'Affects {count}/{total_b} bullets ({round(count/total_b*100)}%)</div>'
                f'</div></div>'
                for i, (axis, count) in enumerate(top_fails)
                if count > 0
            )
            st.markdown(wins_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS (matching Aura AI dark theme)
# ═══════════════════════════════════════════════════════════════════════════════

def _hdr(text: str, sub: str = "") -> None:
    st.markdown(f"""
<div style="margin-bottom:1.2rem;">
  <div style="font-size:.62rem;font-family:Share Tech Mono,monospace;
    color:#00d4ff;letter-spacing:.2em;margin-bottom:.2rem;">
    // RESUME INTELLIGENCE MODULE
  </div>
  <h2 style="margin:0;font-size:1.6rem;">{text}</h2>
  {f'<div style="font-size:.78rem;color:#7ab8d8;margin-top:.3rem;font-family:Share Tech Mono,monospace;font-weight:600;">{sub}</div>' if sub else ''}
</div>""", unsafe_allow_html=True)


def _card(content_fn, title: str = "", accent: str = "#00ff88") -> None:
    st.markdown(f"""
<div style="background:rgba(6,15,30,.92);border:1px solid {accent}22;border-radius:10px;
  padding:1.2rem;margin-bottom:.9rem;position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,{accent},transparent);opacity:.4;"></div>
  <div style="position:absolute;top:0;right:0;width:16px;height:16px;
    border-top:2px solid {accent};border-right:2px solid {accent};opacity:.3;border-radius:0 10px 0 0;"></div>
  {'<div style="font-size:.65rem;font-weight:700;color:'+accent+';text-transform:uppercase;letter-spacing:.12em;font-family:Share Tech Mono,monospace;margin-bottom:.7rem;">'+title+'</div>' if title else ''}
""", unsafe_allow_html=True)
    content_fn()
    st.markdown("</div>", unsafe_allow_html=True)


def _tag(text: str, colour: str = "#00ff88") -> str:
    return (f'<span style="display:inline-block;padding:2px 10px;border-radius:3px;'
            f'font-size:.68rem;font-weight:700;font-family:Share Tech Mono,monospace;'
            f'text-transform:uppercase;letter-spacing:.06em;margin:2px;'
            f'background:{colour}18;color:{colour};border:1px solid {colour}44;">{text}</span>')


def _diff_colour(d: str) -> str:
    return {"Easy": "#00ff88", "Medium": "#ffaa00", "Hard": "#ff3366"}.get(d, "#4a9eff")


def _type_colour(t: str) -> str:
    return {
        "Technical":      "#00d4ff",
        "Behavioural":    "#a855f7",
        "Project-Based":  "#00ff88",
        "System Design":  "#ffaa00",
        "Situational":    "#ff7f50",
    }.get(t, "#4a9eff")


def _render_skill_tags(skills: list) -> None:
    """IDEA 5 — Skill tags pop-in with staggered spring animation."""
    colours = ["#00ff88","#00d4ff","#a855f7","#ffaa00","#ff7f50"]
    tags_html = []
    for i, s in enumerate(skills):
        c = colours[i % len(colours)]
        delay = i * 55
        tags_html.append(
            f'<span style="display:inline-block;padding:3px 11px;border-radius:3px;'
            f'font-size:.68rem;font-weight:700;font-family:Share Tech Mono,monospace;'
            f'text-transform:uppercase;letter-spacing:.06em;margin:3px;'
            f'background:{c}18;color:{c};border:1px solid {c}44;'
            f'opacity:0;transform:scale(.6) translateY(6px);'
            f'animation:skill-pop .35s cubic-bezier(.34,1.56,.64,1) {delay}ms forwards;">'
            f'{s}</span>'
        )
    st.markdown(f"""<style>
@keyframes skill-pop {{
  to {{ opacity:1; transform:scale(1) translateY(0); }}
}}
</style>
<div style="line-height:2.4;">{"".join(tags_html)}</div>""", unsafe_allow_html=True)


def _render_bullets(items: list, colour: str = "#00ff88") -> None:
    import html as _html
    html = "".join(
        f'<div style="display:flex;gap:.5rem;margin:.3rem 0;">'
        f'<span style="color:{colour};font-family:Share Tech Mono,monospace;'
        f'font-size:.9rem;flex-shrink:0;margin-top:.05rem;">▸</span>'
        f'<span style="font-size:.95rem;color:#e8f4ff;line-height:1.65;font-weight:500;">{_html.escape(str(item))}</span></div>'
        for item in items
    )
    st.markdown(html, unsafe_allow_html=True)


def _bullets_html(items: list, colour: str = "#00d4ff") -> str:
    """Return bullet list as an HTML string (for embedding inside a single st.markdown block)."""
    import html as _html
    return "".join(
        f'<div style="display:flex;gap:.5rem;margin:.3rem 0;">'
        f'<span style="color:{colour};font-family:Share Tech Mono,monospace;'
        f'font-size:.9rem;flex-shrink:0;margin-top:.05rem;">▸</span>'
        f'<span style="font-size:.95rem;color:#e8f4ff;line-height:1.65;font-weight:500;">{_html.escape(str(item))}</span></div>'
        for item in items
    )


def _skills_html(skills: list) -> str:
    """Return skill tags as an HTML string."""
    colours = ["#00ff88", "#00d4ff", "#a855f7", "#ffaa00", "#ff7f50"]
    return "".join(_tag(s, colours[i % len(colours)]) for i, s in enumerate(skills))


def _section_diff(label: str, original, rephrased, kind: str = "list") -> None:
    """Display original content inside a single styled box (all HTML in one render call)."""
    # Build inner content as a single HTML string
    if kind == "list" and isinstance(original, list):
        inner = _bullets_html(original, colour="#00d4ff")
    elif kind == "skills" and isinstance(original, list):
        inner = f'<div style="line-height:2.4;">{_skills_html(original)}</div>'
    else:
        inner = (f'<span style="font-size:.95rem;color:#e8f4ff;'
                 f'line-height:1.65;font-weight:500;">{original}</span>')

    st.markdown(f"""
<div style="margin:.8rem 0 .5rem;">
  <div style="font-size:.78rem;font-weight:700;color:#00d4ff;text-transform:uppercase;
    letter-spacing:.12em;font-family:Share Tech Mono,monospace;margin-bottom:.45rem;">{label}</div>
  <div style="background:rgba(6,15,30,.97);border:1px solid rgba(0,212,255,.3);
    border-radius:6px;padding:.85rem 1rem;box-shadow:0 0 14px rgba(0,212,255,.07);">
    {inner}
  </div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SUB-SECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _render_section_scan(parsed: Dict) -> None:
    """
    IDEA 7 — Section extraction scan.
    Shows a compact animated list of what was found, with each row
    materialising one by one and an accent colour per section.
    """
    if not parsed:
        return
    import html as _html

    skill_count = len(parsed.get("skills") or [])
    exp_count   = len(parsed.get("experience") or [])
    proj_count  = len(parsed.get("projects") or [])
    edu_count   = len(parsed.get("education") or [])
    summ_words  = len((parsed.get("summary") or "").split())

    rows_data = []
    if skill_count:
        rows_data.append(("Skills detected", skill_count, "items", "#00ff88"))
    if exp_count:
        rows_data.append(("Experience", exp_count, "roles", "#00d4ff"))
    if proj_count:
        rows_data.append(("Projects", proj_count, "found", "#a855f7"))
    if edu_count:
        rows_data.append(("Education", edu_count, "entry" if edu_count == 1 else "entries", "#ffaa00"))
    if summ_words:
        rows_data.append(("Summary", summ_words, "words", "#ff7f50"))

    if not rows_data:
        return

    rows_html = []
    for i, (label, count, unit, colour) in enumerate(rows_data):
        delay = i * 120
        rows_html.append(f"""
<div style="display:flex;align-items:center;gap:8px;padding:4px 0;
  opacity:0;transform:translateX(-8px);
  animation:scan-row .35s ease {delay}ms forwards;">
  <span style="width:8px;height:8px;border-radius:50%;
    background:{colour};flex-shrink:0;
    box-shadow:0 0 6px {colour}88;
    display:inline-block;"></span>
  <span style="font-size:.8rem;color:#94b4c8;font-family:Share Tech Mono,monospace;
    min-width:120px;">{_html.escape(label)}</span>
  <span style="font-size:.8rem;font-weight:700;color:{colour};
    font-family:Orbitron,monospace;">{count}</span>
  <span style="font-size:.7rem;color:#4a7a9b;font-family:Share Tech Mono,monospace;">{unit}</span>
</div>""")

    st.markdown(f"""<style>
@keyframes scan-row {{
  to {{ opacity:1; transform:translateX(0); }}
}}
@keyframes scan-line {{
  0%   {{ top:0; opacity:.8; }}
  100% {{ top:100%; opacity:0; }}
}}
</style>
<div style="background:rgba(6,15,30,.95);border:1px solid rgba(0,255,136,.12);
  border-radius:8px;padding:.7rem 1rem;margin-bottom:1rem;position:relative;overflow:hidden;">
  <div style="position:absolute;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,rgba(0,255,136,.5),transparent);
    animation:scan-line .8s ease-in forwards;"></div>
  <div style="font-size:.58rem;letter-spacing:.2em;color:#4a7a9b;
    font-family:Share Tech Mono,monospace;margin-bottom:.5rem;">EXTRACTION COMPLETE</div>
  {"".join(rows_html)}
</div>""", unsafe_allow_html=True)


def _render_upload_section() -> Optional[str]:
    """Step 1: Upload or paste resume. Returns raw text or None."""
    st.markdown(
        '<div style="font-size:.82rem;color:#7ab8d8;font-family:Share Tech Mono,monospace;'
        'margin-bottom:1rem;font-weight:600;">STEP 01 · UPLOAD OR PASTE YOUR RESUME</div>',
        unsafe_allow_html=True,
    )

    tab_up, tab_paste = st.tabs(["📁  Upload File", "📝  Paste Text"])
    raw_text = ""

    with tab_up:
        if not PYPDF_OK and not DOCX_OK:
            st.markdown(
                '<div style="background:rgba(255,170,0,.07);border-left:3px solid #ffaa00;'
                'padding:.6rem .9rem;border-radius:4px;color:#ffd080;font-size:.8rem;margin:.4rem 0;">'
                '▲ Install pypdf and python-docx for file upload:<br>'
                '<code>pip install pypdf python-docx</code></div>',
                unsafe_allow_html=True,
            )
        accepted = []
        if PYPDF_OK:  accepted.append("pdf")
        if DOCX_OK:   accepted.append("docx")
        accepted.append("txt")

        uploaded = st.file_uploader(
            "Drop your resume here",
            type=accepted or ["txt"],
            key="resume_upload",
            label_visibility="collapsed",
        )
        if uploaded:
            ext = uploaded.name.rsplit(".", 1)[-1].lower()
            data = uploaded.read()
            if ext == "pdf":
                raw_text = _extract_text_from_pdf(data)
                if not raw_text:
                    st.warning("⚠ Could not extract PDF text. Try pasting instead.")
            elif ext == "docx":
                raw_text = _extract_text_from_docx(data)
            else:
                raw_text = data.decode("utf-8", errors="ignore")

            if raw_text:
                st.success(f"✅ Loaded {len(raw_text):,} characters from **{uploaded.name}**")
                with st.expander("Preview raw text"):
                    st.text(raw_text[:1500] + ("…" if len(raw_text) > 1500 else ""))

    with tab_paste:
        pasted = st.text_area(
            "Paste resume text:",
            height=280,
            key="resume_paste",
            placeholder="Paste your full resume here — skills, projects, experience, education…",
            label_visibility="collapsed",
        )
        if pasted.strip():
            raw_text = pasted.strip()

    # Use previously stored text if nothing new provided
    if not raw_text:
        raw_text = st.session_state.get("resume_raw_text", "")

    return raw_text or None


def _render_rephrased_section() -> None:
    """Step 2: Display parsed resume content."""
    parsed   = st.session_state.get("resume_parsed", {})
    rephrased = st.session_state.get("resume_rephrased", {})
    if not parsed:
        return

    # Fix expander label + summary text colours so they're legible on the dark bg
    # NOTE: Do NOT apply font-family to summary span or details summary — those
    # selectors hit Streamlit's internal icon <span> which uses a glyph font
    # (Material Icons). Overriding it with a text font causes the glyph to
    # render as its text name, e.g. "arrow_right".
    st.markdown("""<style>
[data-testid="stExpander"] summary p {
    color: #c8e6ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: .9rem !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] details summary:hover p {
    color: #ffffff !important;
}
[data-testid="stExpander"] {
    border: 1px solid rgba(0,212,255,.18) !important;
    border-radius: 8px !important;
    background: rgba(6,15,30,.85) !important;
    margin-bottom: .5rem !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: rgba(6,15,30,.85) !important;
    color: #e8f4ff !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] span,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] div {
    color: #e8f4ff !important;
}
</style>""", unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:.82rem;color:#7ab8d8;font-family:Share Tech Mono,monospace;'
        'margin-bottom:1rem;font-weight:600;">STEP 02 · PARSED RESUME CONTENT</div>',
        unsafe_allow_html=True,
    )

    # ── IDEA 7: Section extraction scan summary ───────────────────────────────
    _render_section_scan(parsed)

    # ── Candidate name + summary ──────────────────────────────────────────────
    name  = rephrased.get("name") or parsed.get("name", "")
    summ_orig = parsed.get("summary", "")
    summ_reph = rephrased.get("summary", summ_orig)
    if name:
        st.markdown(
            f'<div style="font-family:Orbitron,monospace;font-size:1.1rem;font-weight:700;'
            f'color:#00ff88;margin-bottom:.3rem;">{name.upper()}</div>',
            unsafe_allow_html=True,
        )
    if summ_orig:
        _section_diff("Professional Summary", summ_orig, summ_reph, kind="text")

    # ── Skills ────────────────────────────────────────────────────────────────
    sk_orig = parsed.get("skills", [])
    sk_reph = rephrased.get("skills", sk_orig)
    if sk_orig:
        _section_diff("Skills", sk_orig, sk_reph, kind="skills")

    # ── Projects ──────────────────────────────────────────────────────────────
    proj_orig = parsed.get("projects", [])
    proj_reph = rephrased.get("projects", proj_orig)
    if proj_orig:
        st.markdown(
            '<div style="font-size:.82rem;font-weight:700;color:#00d4ff;text-transform:uppercase;'
            'letter-spacing:.12em;font-family:Share Tech Mono,monospace;margin:.8rem 0 .4rem;">Projects</div>',
            unsafe_allow_html=True,
        )
        for i, (po, pr) in enumerate(zip(proj_orig, proj_reph or proj_orig)):
            if isinstance(po, str):
                po = {"title": po, "description": po}
            if not isinstance(po, dict):
                po = {}
            if isinstance(pr, str):
                pr = {"description": pr}
            if not isinstance(pr, dict):
                pr = {}
            with st.expander(f"◈ {po.get('title', f'Project {i+1}')}", expanded=(i == 0)):
                techs = po.get("technologies", [])
                if techs:
                    st.markdown(
                        '<span style="font-size:.78rem;color:#a0c4e8;font-family:Share Tech Mono,'
                        'monospace;font-weight:600;text-transform:uppercase;letter-spacing:.06em;">'
                        'Technologies: </span>' +
                        " ".join(_tag(t, "#a855f7") for t in techs),
                        unsafe_allow_html=True)
                desc_orig = po.get("description", "")
                desc_reph = (pr or {}).get("description", desc_orig)
                if desc_orig:
                    _section_diff("Description", desc_orig, desc_reph, kind="text")
                impact_orig = po.get("impact", "")
                impact_reph = (pr or {}).get("impact", impact_orig)
                if impact_orig:
                    _section_diff("Impact", impact_orig, impact_reph, kind="text")

    # ── Experience ────────────────────────────────────────────────────────────
    exp_orig = parsed.get("experience", [])
    exp_reph = rephrased.get("experience", exp_orig)
    if exp_orig:
        st.markdown(
            '<div style="font-size:.82rem;font-weight:700;color:#00d4ff;text-transform:uppercase;'
            'letter-spacing:.12em;font-family:Share Tech Mono,monospace;margin:.8rem 0 .4rem;">Experience</div>',
            unsafe_allow_html=True,
        )
        for i, (eo, er) in enumerate(zip(exp_orig, exp_reph or exp_orig)):
            if isinstance(eo, str):
                eo = {"role": eo, "company": "", "duration": ""}
            if not isinstance(eo, dict):
                eo = {}
            if isinstance(er, str):
                er = {}
            if not isinstance(er, dict):
                er = {}
            label = f"▸ {eo.get('role','Role')} @ {eo.get('company','Company')} · {eo.get('duration','')}"
            with st.expander(label, expanded=(i == 0)):
                resp_orig = eo.get("responsibilities", [])
                resp_reph = (er or {}).get("responsibilities", resp_orig)
                if resp_orig:
                    _section_diff("Responsibilities", resp_orig, resp_reph, kind="list")
                ach_orig = eo.get("achievements", [])
                ach_reph = (er or {}).get("achievements", ach_orig)
                if ach_orig:
                    _section_diff("Achievements", ach_orig, ach_reph, kind="list")

    # ── Education ─────────────────────────────────────────────────────────────
    edu = rephrased.get("education") or parsed.get("education", [])
    if edu:
        st.markdown(
            '<div style="font-size:.82rem;font-weight:700;color:#00d4ff;text-transform:uppercase;'
            'letter-spacing:.12em;font-family:Share Tech Mono,monospace;margin:.8rem 0 .4rem;">Education</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(min(len(edu), 3))
        for i, e in enumerate(edu):
            # Guard: API may return a plain string instead of a dict
            if isinstance(e, str):
                e = {"institution": e}
            elif not isinstance(e, dict):
                e = {}
            with cols[i % len(cols)]:
                st.markdown(f"""
<div style="background:rgba(6,15,30,.9);border:1px solid rgba(0,212,255,.15);
  border-radius:8px;padding:.8rem;text-align:center;">
  <div style="font-size:.82rem;font-weight:700;color:#e0f0ff;font-family:Orbitron,monospace;">{e.get('institution','')}</div>
  <div style="font-size:.72rem;color:#00d4ff;margin:.2rem 0;">{e.get('degree','')} · {e.get('field','')}</div>
  <div style="font-size:.72rem;color:#7ab8d8;font-family:Share Tech Mono,monospace;font-weight:600;">{e.get('year','')}{'  ·  GPA: '+str(e['gpa']) if e.get('gpa') else ''}</div>
</div>""", unsafe_allow_html=True)

    # ── Copy rephrased to clipboard ───────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    rephrased_str = json.dumps(rephrased or parsed, indent=2)
    st.download_button(
        "◈  DOWNLOAD REPHRASED RESUME (JSON)",
        data=rephrased_str,
        file_name="rephrased_resume.json",
        mime="application/json",
        key="dl_rephrased",
    )


def _render_questions_section(engine=None) -> None:
    """Step 3: Display generated questions with option to load into interview."""
    questions = st.session_state.get("resume_questions", [])
    if not questions:
        return

    st.markdown(
        '<div style="font-size:.82rem;color:#7ab8d8;font-family:Share Tech Mono,monospace;'
        'margin-bottom:1rem;font-weight:600;">STEP 03 · AI-GENERATED INTERVIEW QUESTIONS FROM YOUR RESUME</div>',
        unsafe_allow_html=True,
    )

    # Summary chips
    total = len(questions)
    types = {}
    for q in questions:
        t = q.get("type", "Other")
        types[t] = types.get(t, 0) + 1

    chips_html = "".join(
        f'<span style="background:{_type_colour(t)}18;color:{_type_colour(t)};'
        f'border:1px solid {_type_colour(t)}44;border-radius:3px;padding:2px 10px;'
        f'font-size:.65rem;font-weight:700;font-family:Share Tech Mono,monospace;'
        f'margin:2px;display:inline-block;">{t}: {n}</span>'
        for t, n in types.items()
    )
    st.markdown(f"""
<div style="background:rgba(6,15,30,.85);border:1px solid rgba(0,255,136,.1);
  border-radius:8px;padding:.8rem 1rem;margin-bottom:1rem;display:flex;
  align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.5rem;">
  <div style="font-family:Orbitron,monospace;font-size:1.1rem;font-weight:700;color:#00ff88;">
    {total} Questions Generated
  </div>
  <div style="line-height:2;">{chips_html}</div>
</div>""", unsafe_allow_html=True)

    # ── IDEA 8: Question accordion reveal — staggered slide-in collapsible cards ─
    # Group questions by type so they appear in clusters: Technical → Behavioural → …
    type_order = ["Technical", "System Design", "Project-Based", "Behavioural", "Situational"]
    grouped: dict = {}
    for q in questions:
        t = q.get("type", "Other")
        grouped.setdefault(t, []).append(q)

    ordered_types = [t for t in type_order if t in grouped]
    for t in grouped:
        if t not in ordered_types:
            ordered_types.append(t)

    # Flat ordered list for numbering
    ordered_qs = []
    for t in ordered_types:
        ordered_qs.extend(grouped[t])

    # Build accordion HTML — each card starts collapsed, click expands inline
    cards_html = []
    global_idx = 0
    for t in ordered_types:
        tc = _type_colour(t)
        for q in grouped[t]:
            global_idx += 1
            diff   = q.get("difficulty", "Medium")
            target = q.get("target", "")
            dc     = _diff_colour(diff)
            kws    = q.get("ideal_keywords", [])
            kw_html = " ".join(
                f'<span style="background:#a855f722;color:#a855f7;border:1px solid #a855f744;'
                f'border-radius:3px;padding:1px 7px;font-size:.65rem;'
                f'font-family:Share Tech Mono,monospace;margin:1px 2px;'
                f'display:inline-block;">{k}</span>'
                for k in kws
            )
            ideal  = q.get("ideal_answer", "")
            delay  = (global_idx - 1) * 60  # ms stagger

            cards_html.append(f"""
<div class="acc-card" style="background:rgba(6,15,30,.92);border:1px solid {tc}22;
  border-radius:10px;margin-bottom:.5rem;overflow:hidden;position:relative;
  opacity:0;transform:translateY(8px);
  animation:acc-in .4s ease {delay}ms forwards;">
  <div style="position:absolute;top:0;left:0;width:3px;height:100%;
    background:linear-gradient(180deg,{tc},{tc}44);border-radius:10px 0 0 10px;"></div>
  <div class="acc-header" onclick="(function(el){{
    var body=el.parentElement.querySelector('.acc-body');
    var arrow=el.querySelector('.acc-arrow');
    var open=body.style.maxHeight&&body.style.maxHeight!='0px';
    body.style.maxHeight=open?'0px':'600px';
    body.style.opacity=open?'0':'1';
    arrow.style.transform=open?'rotate(0deg)':'rotate(90deg)';
  }})(this)"
  style="display:flex;justify-content:space-between;align-items:center;
    padding:.7rem 1rem .7rem 1.3rem;cursor:pointer;
    transition:background .2s;user-select:none;"
  onmouseover="this.style.background='rgba(255,255,255,.03)'"
  onmouseout="this.style.background='transparent'">
    <div style="display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;flex:1;min-width:0;">
      <span style="font-size:.72rem;font-weight:700;color:#7ab8d8;
        font-family:Share Tech Mono,monospace;flex-shrink:0;">Q{global_idx}</span>
      <span style="background:{tc}18;color:{tc};border:1px solid {tc}44;border-radius:3px;
        padding:1px 7px;font-size:.7rem;font-weight:700;
        font-family:Share Tech Mono,monospace;flex-shrink:0;">{t}</span>
      <span style="background:{dc}18;color:{dc};border:1px solid {dc}44;border-radius:3px;
        padding:1px 7px;font-size:.7rem;font-weight:700;
        font-family:Share Tech Mono,monospace;flex-shrink:0;">{diff}</span>
      <span style="font-size:.92rem;color:#dff4ff;font-weight:600;
        font-family:Rajdhani,sans-serif;line-height:1.3;
        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{q.get('question','')}</span>
    </div>
    <span class="acc-arrow" style="color:{tc};font-size:1rem;margin-left:.5rem;flex-shrink:0;
      transition:transform .2s;">▶</span>
  </div>
  <div class="acc-body" style="max-height:0px;overflow:hidden;opacity:0;
    transition:max-height .35s ease, opacity .3s ease;">
    <div style="padding:.2rem 1rem .9rem 1.3rem;border-top:1px solid rgba(255,255,255,.05);">
      {f'<div style="font-size:.72rem;color:#7ab8d8;font-family:Share Tech Mono,monospace;margin:.5rem 0 .3rem;">↳ {target}</div>' if target else ''}
      {f'<div style="margin:.3rem 0 .5rem;line-height:1.9;">{kw_html}</div>' if kws else ''}
      {f'<div style="font-size:.88rem;color:#94c4e4;font-style:italic;border-top:1px solid rgba(255,255,255,.06);padding-top:.5rem;margin-top:.3rem;line-height:1.6;">💡 {ideal}</div>' if ideal else ''}
    </div>
  </div>
</div>""")

    st.markdown(f"""<style>
@keyframes acc-in {{
  to {{ opacity:1; transform:translateY(0); }}
}}
</style>
{"".join(cards_html)}""", unsafe_allow_html=True)

    # ── Load into interview ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_load, col_dl, col_gap = st.columns([2, 2, 3])

    with col_load:
        def load_into_interview():
            """Push generated questions into the engine's question pool."""
            parsed = st.session_state.get("resume_parsed", {})

            if engine is None:
                st.warning("Engine not available — questions saved to session state.")
                st.session_state["custom_resume_questions"] = questions
                # ── RL calibration via session state (no engine reference) ──────
                # app.py must read "resume_parsed_for_rl" and write it to
                # engine._resume_parsed before starting the session.
                st.session_state["resume_parsed_for_rl"] = parsed
                st.session_state["page"] = "Start Interview"
                return

            # ── Wire resume_parsed into the RL sequencer calibration ───────────
            # RLAdaptiveSequencer.get_first_action(resume_parsed=...) reads the
            # "experience" and "summary" keys to infer years of experience and
            # pick Q1 difficulty (0-1 yrs → easy, 2-4 → medium, 5+ → hard).
            # Without this line the field stays {} and calibration is skipped.
            #
            # We also flatten the structured experience list into a single text
            # block because _parse_experience_difficulty() does a regex search
            # on a plain string, not a list of dicts.
            exp_entries = parsed.get("experience", [])
            if isinstance(exp_entries, list):
                # Build a flat text block: "Role at Company · duration\n..."
                exp_text_parts = []
                for entry in exp_entries:
                    if isinstance(entry, dict):
                        role_str     = entry.get("role", "")
                        company_str  = entry.get("company", "")
                        duration_str = entry.get("duration", "")
                        exp_text_parts.append(
                            f"{role_str} at {company_str} · {duration_str}"
                        )
                    elif isinstance(entry, str):
                        exp_text_parts.append(entry)
                exp_text = "\n".join(exp_text_parts)
            else:
                # Already a plain string (older resume_parsed format)
                exp_text = str(exp_entries)

            # Build the dict that _parse_experience_difficulty() expects:
            # {"experience": <str>, "summary": <str>}
            resume_for_rl = {
                "experience": exp_text,
                "summary":    parsed.get("summary", ""),
            }
            # Push into the engine so get_first_action() can read it
            engine._resume_parsed = resume_for_rl
            # Also store in session state so page_start can re-apply it
            # if the engine is re-initialised between pages.
            st.session_state["resume_parsed_for_rl"] = resume_for_rl

            # ── Derive years and log what the sequencer will choose ────────────
            try:
                from adaptive_sequencer import _parse_experience_difficulty
                inferred_diff = _parse_experience_difficulty(resume_for_rl)
                if inferred_diff:
                    st.session_state["rl_resume_calibration"] = inferred_diff
                    import logging
                    logging.getLogger("ResumeRephraser").info(
                        f"RL Q1 calibration from resume → {inferred_diff} "
                        f"(experience text: {exp_text[:120]!r})"
                    )
            except Exception:
                pass   # non-fatal — calibration simply falls back to default

            # Format questions for the engine
            formatted = []
            for q in questions:
                formatted.append({
                    "question":     q.get("question", ""),
                    "type":         q.get("type", "Technical"),
                    "difficulty":   q.get("difficulty", "Medium").lower(),
                    "keywords":     q.get("ideal_keywords", []),
                    "ideal_answer": q.get("ideal_answer", ""),
                    "source":       "resume",
                })

            # If engine has a method to inject questions, use it
            if hasattr(engine, "load_custom_questions"):
                engine.load_custom_questions(formatted)
            else:
                # Fallback: store in session state for page_start to pick up
                st.session_state["custom_resume_questions"] = formatted

            # Set defaults for interview
            cand_name = parsed.get("name", "") or st.session_state.get("candidate_name", "Candidate")
            st.session_state.update({
                "candidate_name": cand_name,
                "num_questions":  len(questions),
                "page":           "Start Interview",
            })

        st.button("▶  USE THESE QUESTIONS", key="load_q_btn",
                  on_click=load_into_interview, use_container_width=True)

    with col_dl:
        q_json = json.dumps(questions, indent=2)
        st.download_button(
            "◈  EXPORT QUESTIONS (JSON)",
            data=q_json,
            file_name="resume_interview_questions.json",
            mime="application/json",
            key="dl_questions",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PAGE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def page_resume(engine=None) -> None:
    """
    Full Resume Rephraser page. Call from app.py main() router.

    Args:
        engine: your InterviewEngine instance (optional — used to inject questions)
    """
    # Inject page title
    st.markdown("""
<div style="margin-bottom:1.2rem;">
  <div style="font-size:.62rem;font-family:Share Tech Mono,monospace;
    color:#00d4ff;letter-spacing:.2em;margin-bottom:.2rem;">
    // RESUME INTELLIGENCE MODULE
  </div>
  <h2 style="margin:0;font-size:1.6rem;">RESUME REPHRASER + QUESTION GENERATOR</h2>
  <div style="font-size:.78rem;color:#7ab8d8;margin-top:.3rem;font-family:Share Tech Mono,monospace;font-weight:600;">
    UPLOAD → AI PARSE → REPHRASE → GENERATE QUESTIONS → START INTERVIEW
  </div>
</div>""", unsafe_allow_html=True)

    if not GROQ_OK:
        st.markdown("""
<div style="background:rgba(255,51,102,.07);border-left:3px solid #ff3366;
  padding:.8rem 1rem;border-radius:4px;color:#ff8099;margin-bottom:1rem;">
  ▲ Groq API not configured. Add your key:<br>
  <b>Windows:</b> <code>set GROQ_API_KEY=your_key_here</code><br>
  <b>Mac/Linux:</b> <code>export GROQ_API_KEY=your_key_here</code><br>
  Then install: <code>pip install groq</code><br>
  Get a free key at: <a href="https://console.groq.com" target="_blank" style="color:#00d4ff;">console.groq.com</a>
</div>""", unsafe_allow_html=True)

    # ── Configuration row ─────────────────────────────────────────────────────
    cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns([2, 1, 1, 1])
    with cfg_col1:
        target_role = st.text_input(
            "TARGET ROLE (optional)",
            value=st.session_state.get("target_role", ""),
            key="rr_role",
            placeholder="e.g. Senior Backend Engineer, ML Engineer…",
        )
    with cfg_col2:
        difficulty = st.selectbox(
            "DIFFICULTY",
            ["Easy", "Medium", "Hard"],
            index=1,
            key="rr_diff",
        )
    with cfg_col3:
        num_q = st.slider("# QUESTIONS", 5, 20, 10, key="rr_nq")
    with cfg_col4:
        st.markdown("<br>", unsafe_allow_html=True)
        auto_rephrase = st.toggle("Auto-rephrase", value=True, key="rr_auto")

    st.markdown("<hr style='border-color:rgba(0,255,136,.06);margin:.5rem 0 1rem;'>",
                unsafe_allow_html=True)

    # ── Step 1: Upload ────────────────────────────────────────────────────────
    raw_text = _render_upload_section()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Action buttons ────────────────────────────────────────────────────────
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([2, 2, 2, 2])

    # Track if we already have results
    has_parsed    = bool(st.session_state.get("resume_parsed"))
    has_rephrased = bool(st.session_state.get("resume_rephrased"))
    has_questions = bool(st.session_state.get("resume_questions"))

    with btn_col1:
        run_all = st.button(
            "⬡  ANALYSE + REPHRASE + GENERATE",
            key="rr_run_all",
            use_container_width=True,
            disabled=not raw_text or not GROQ_OK,
        )

    with btn_col2:
        run_parse_only = st.button(
            "◈  PARSE ONLY",
            key="rr_parse",
            use_container_width=True,
            disabled=not raw_text or not GROQ_OK,
        )

    with btn_col3:
        if has_parsed:
            regen_q = st.button(
                "▸  REGENERATE QUESTIONS",
                key="rr_regen",
                use_container_width=True,
                disabled=not GROQ_OK,
            )
        else:
            regen_q = False

    with btn_col4:
        if has_parsed:
            run_score = st.button(
                "◈  SCORE RESUME",
                key="rr_score",
                use_container_width=True,
                disabled=not GROQ_OK,
            )
        else:
            run_score = False

    # ── Processing logic ──────────────────────────────────────────────────────
    if run_all and raw_text:
        st.session_state["resume_raw_text"] = raw_text

        # ── IDEA 4: Vertical pipeline progress ───────────────────────────────
        pip_slot = st.empty()

        def _show_pipeline(active_idx: int, done_up_to: int) -> None:
            """Redraw the pipeline with current step states."""
            _labels = [
                "Parse resume structure",
                "AI rephrase (ATS-optimised)",
                "Score bullets",
                "Generate questions",
            ]
            steps = []
            for j, lbl in enumerate(_labels):
                if j < done_up_to:
                    steps.append((lbl, "done"))
                elif j == active_idx:
                    steps.append((lbl, "active"))
                else:
                    steps.append((lbl, "pending"))
            pip_slot.markdown(_pipeline_progress_html(steps), unsafe_allow_html=True)

        _show_pipeline(0, 0)
        parsed = parse_resume(raw_text)
        st.session_state["resume_parsed"] = parsed

        _show_pipeline(1, 1)
        if auto_rephrase:
            rephrased = rephrase_resume(parsed, target_role)
            st.session_state["resume_rephrased"] = rephrased
        else:
            st.session_state["resume_rephrased"] = parsed

        _show_pipeline(2, 2)
        score_data = score_resume(
            parsed,
            st.session_state["resume_rephrased"],
            target_role=target_role,
        )
        st.session_state["resume_score_data"] = score_data

        _show_pipeline(3, 3)
        questions = generate_questions(
            parsed,
            st.session_state["resume_rephrased"],
            target_role=target_role,
            num_questions=num_q,
            difficulty=difficulty,
        )
        st.session_state["resume_questions"] = questions

        # All done
        _show_pipeline(-1, 4)
        time.sleep(0.4)
        pip_slot.empty()
        st.rerun()

    elif run_parse_only and raw_text:
        st.session_state["resume_raw_text"] = raw_text
        with st.spinner("◈ Parsing resume…"):
            parsed = parse_resume(raw_text)
            st.session_state["resume_parsed"] = parsed
            if auto_rephrase:
                rephrased = rephrase_resume(parsed, target_role)
                st.session_state["resume_rephrased"] = rephrased
            else:
                st.session_state["resume_rephrased"] = parsed
        st.rerun()

    elif regen_q:
        parsed    = st.session_state.get("resume_parsed", {})
        rephrased = st.session_state.get("resume_rephrased", parsed)
        with st.spinner("▸ Generating new questions…"):
            questions = generate_questions(
                parsed, rephrased,
                target_role=target_role,
                num_questions=num_q,
                difficulty=difficulty,
            )
            st.session_state["resume_questions"] = questions
        st.rerun()

    elif run_score and has_parsed:
        parsed    = st.session_state.get("resume_parsed", {})
        rephrased = st.session_state.get("resume_rephrased", parsed)
        with st.spinner("◈ Scoring resume bullets — this uses one Groq call per section…"):
            score_data = score_resume(parsed, rephrased, target_role=target_role)
            st.session_state["resume_score_data"] = score_data
        st.rerun()


    # ── Render results ────────────────────────────────────────────────────────
    if has_parsed:
        st.markdown("<hr style='border-color:rgba(0,255,136,.06);margin:1rem 0;'>",
                    unsafe_allow_html=True)
        _render_rephrased_section()


    # ── Score card (VMock-style) ─────────────────────────────────────────────
    score_data = st.session_state.get("resume_score_data")
    if score_data:
        st.markdown("<hr style='border-color:rgba(168,85,247,.12);margin:1rem 0;'>",
                    unsafe_allow_html=True)
        render_resume_scorecard(score_data)

    if has_questions:
        st.markdown("<hr style='border-color:rgba(0,255,136,.06);margin:1rem 0;'>",
                    unsafe_allow_html=True)
        _render_questions_section(engine=engine)

    if has_parsed and not has_questions:
        # Offer to generate questions if only parsed
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▸  GENERATE INTERVIEW QUESTIONS FROM THIS RESUME",
                     key="rr_genq_late", use_container_width=False,
                     disabled=not GROQ_OK):
            parsed    = st.session_state.get("resume_parsed", {})
            rephrased = st.session_state.get("resume_rephrased", parsed)
            with st.spinner("Generating…"):
                questions = generate_questions(
                    parsed, rephrased,
                    target_role=target_role,
                    num_questions=num_q,
                    difficulty=difficulty,
                )
                st.session_state["resume_questions"] = questions
            st.rerun()