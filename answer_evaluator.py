"""
answer_evaluator.py — Aura AI | NLP Answer Evaluation Engine (v11.0)
=======================================================================
v11.0 — GROQ API AS PRIMARY RELEVANCE ENGINE (TF-IDF fully replaced)

What changed:
  TF-IDF is no longer the primary relevance checker anywhere.
  Every relevance scoring path now calls the Groq API first.

  Old fallback chain:
    Groq API -> TF-IDF -> keywords -> 0.0

  New fallback chain:
    Groq API (vs generated/static ideal) -> relevance_source = "api_groq"
    Groq API (vs question + keywords)    -> relevance_source = "api_groq_kw"
    TF-IDF   (Groq HTTP fails, emergency)-> relevance_source = "tfidf_fallback"
    0.0      (no key AND no sklearn)     -> relevance_source = "none"

  New method: _groq_relevance_score(answer, reference, context)
    Universal Groq-based scorer called by all relevance paths.

  sklearn/TF-IDF is now OPTIONAL. Works without it if GROQ_API_KEY is set.

v10.1 carried forward: Type-aware depth score zones (technical/behavioural/hr)
v10.0 carried forward: Question-type-aware weight profiles, STAR gated by type
v9.0 carried forward: Dynamic ideal answer generation (LRU cached)
v8.3 carried forward: Prompt injection defence, GROQ_API_KEY warning
v8.0 carried forward: STAR linear, fluency additive, WPM conditional, OCEAN

SCORING WEIGHT PROFILES:
  Technical:   star=0.00 word_cat=0.10 relevance=0.40 keyword=0.25 depth=0.20 grammar=0.05
  Behavioural: star=0.35 word_cat=0.20 relevance=0.20 keyword=0.10 depth=0.10 grammar=0.05
  HR:          star=0.20 word_cat=0.15 relevance=0.25 keyword=0.10 depth=0.25 grammar=0.05
"""

from __future__ import annotations

import functools
import json
import logging
import os
from dotenv import load_dotenv
load_dotenv()
import re
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("AnswerEvaluator")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import language_tool_python
    LT_AVAILABLE = True
except ImportError:
    LT_AVAILABLE = False

# ── SentenceTransformer — semantic keyword matching (v12.1) ───────────────────
# Replaces binary exact-match keyword hits with cosine-similarity matching.
# Falls back gracefully to exact-match if package not installed.
# Install: pip install sentence-transformers
try:
    from sentence_transformers import SentenceTransformer as _ST
    _SBERT_KW_MODEL = None   # lazy-loaded on first use

    def _get_sbert_kw_model():
        global _SBERT_KW_MODEL
        if _SBERT_KW_MODEL is None:
            try:
                _SBERT_KW_MODEL = _ST("paraphrase-MiniLM-L6-v2")
                log.info("SentenceTransformer keyword model loaded (paraphrase-MiniLM-L6-v2).")
            except Exception as exc:
                log.warning(f"SentenceTransformer load failed: {exc}. "
                            "Keyword scoring will fall back to exact-match.")
        return _SBERT_KW_MODEL

    SBERT_KW_AVAILABLE = True
except ImportError:
    SBERT_KW_AVAILABLE = False
    _SBERT_KW_MODEL    = None

    def _get_sbert_kw_model():
        return None

# Threshold above which a keyword is considered semantically matched.
# 0.45 catches synonyms, acronyms, paraphrases while rejecting unrelated terms.
# Lower → more permissive; raise to 0.55 for stricter matching.
_KW_SEMANTIC_THRESHOLD: float = 0.45


# ══════════════════════════════════════════════════════════════════════════════
#  GROQ CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_GROQ_MODEL = "llama-3.3-70b-versatile"


# ══════════════════════════════════════════════════════════════════════════════
#  QUESTION-TYPE-AWARE WEIGHT PROFILES  (v10.0 core fix)
# ══════════════════════════════════════════════════════════════════════════════

# Each profile must sum to exactly 1.00.
# Keys: star, word_cat, relevance, keyword, depth_flu, grammar
WEIGHT_PROFILES: Dict[str, Dict[str, float]] = {

    "technical": {
        # STAR disabled — technical explanations have no Situation/Result.
        # Relevance is primary: does the candidate actually know the topic?
        # Keywords are raised: technical terms, tools, frameworks signal depth.
        # Depth/fluency raised: longer, well-structured explanations score better.
        "star":      0.00,
        "word_cat":  0.10,
        "relevance": 0.40,
        "keyword":   0.25,
        "depth_flu": 0.20,
        "grammar":   0.05,
    },

    "behavioural": {
        # STAR is the primary framework for behavioural stories (Rasipuram 2017).
        # Word categories matter: quantifiers + perceptual verbs signal confidence.
        # Relevance stays on topic with the behavioural theme.
        # Keywords reduced: less critical than story structure for behavioural.
        "star":      0.35,
        "word_cat":  0.20,
        "relevance": 0.20,
        "keyword":   0.10,
        "depth_flu": 0.10,
        "grammar":   0.05,
    },

    "hr": {
        # HR answers (motivation, strengths, goals) need genuine self-reflection.
        # Relevance is primary: answer must actually address the question asked.
        # Depth/fluency raised: HR answers need substance, not one-liners.
        # Partial STAR still valued: structured examples strengthen HR answers.
        # Word categories matter: confident language signals positive attitude.
        "star":      0.20,
        "word_cat":  0.15,
        "relevance": 0.25,
        "keyword":   0.10,
        "depth_flu": 0.25,
        "grammar":   0.05,
    },
}

# Normalised type key lookup — handles capitalisation variants from question_pool.json
_TYPE_MAP: Dict[str, str] = {
    "technical":    "technical",
    "behavioural":  "behavioural",
    "behavioral":   "behavioural",   # American spelling
    "hr":           "hr",
    "soft":         "hr",            # alias sometimes used
    "general":      "hr",
}


def _resolve_type(raw_type: str) -> str:
    """
    Normalise question type string to one of: "technical" | "behavioural" | "hr".
    Defaults to "technical" for unknown values.
    """
    return _TYPE_MAP.get(raw_type.lower().strip(), "technical")


# ══════════════════════════════════════════════════════════════════════════════
#  STAR PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

STAR_PATTERNS: Dict[str, str] = {
    "Situation": (
        r"\b(situation|context|background|when|once|there was|faced|encountered|"
        r"during|at the time|previously|in my previous|in that project|"
        r"working at|while at|at my last|i was working|we were|at that point)\b"
    ),
    "Task": (
        r"\b(task|goal|objective|responsible|needed to|had to|assigned|my role|"
        r"challenge|was asked|my responsibility|my job was|i was tasked|"
        r"required to|expected to|set out to|aim was|purpose was)\b"
    ),
    "Action": (
        r"\b(i did|i took|i used|implemented|developed|created|decided|solved|"
        r"built|designed|led|coordinated|introduced|refactored|optimised|"
        r"automated|proposed|initiated|established|deployed|migrated|"
        r"collaborated with|worked with|reached out|presented to|"
        r"i approached|i focused|i prioritised|i identified)\b"
    ),
    "Result": (
        r"\b(result|outcome|achieved|improved|reduced|increased|success|impact|"
        r"as a result|completed|delivered|saved|cut|boosted|grew|gained|"
        r"received|won|promoted|recognised|measurable|percent|%|"
        r"within deadline|on time|under budget|positive feedback|"
        r"successfully|ultimately|in the end|this led to)\b"
    ),
}

# ── DISC keyword bank ─────────────────────────────────────────────────────────
DISC_KEYWORDS: Dict[str, List[str]] = {
    "Dominance"        : [
        "lead","decided","took charge","goal","direct","challenge","result",
        "win","fast","control","drove","pushed","owned","accountable",
        "assertive","competitive","decisive","bold","vision","strategic",
    ],
    "Influence"        : [
        "team","collaborate","communicate","inspire","enthusiasm","motivated",
        "people","fun","support","engaged","presented","networked","shared",
        "persuaded","encouraged","energised","positive","relationship","culture",
    ],
    "Steadiness"       : [
        "consistent","reliable","patient","support","stable","process",
        "listen","careful","methodical","thorough","steady","calm",
        "dependable","systematic","predictable","structured","organised",
    ],
    "Conscientiousness": [
        "accurate","detail","quality","process","data","systematic","standard",
        "precise","analysis","metrics","documented","reviewed","validated",
        "tested","verified","researched","planned","tracked","measured",
    ],
}

# ── Big Five OCEAN proxies ────────────────────────────────────────────────────
OCEAN_KEYWORDS: Dict[str, List[str]] = {
    "Openness"         : [
        "creative","innovative","novel","explored","experiment","curious",
        "learned","researched","ideated","brainstormed","new approach",
        "alternative","reimagined","rethought","discovery","creative solution",
    ],
    "Conscientiousness": [
        "organised","planned","deadline","systematic","structured","careful",
        "detail","accurate","documented","tracked","scheduled","prioritised",
        "reviewed","verified","tested","quality","standard","process",
    ],
    "Extraversion"     : [
        "team","presented","collaborated","led","networked","communicated",
        "shared","engaged","discussed","facilitated","mentored","coached",
        "convinced","negotiated","proactive","outreach",
    ],
    "Agreeableness"    : [
        "supported","helped","assisted","listened","understood","empathised",
        "compromise","cooperative","patient","flexible","accommodated",
        "considered","respected","valued","inclusive","collaborative",
    ],
    "Neuroticism_inv"  : [
        "calm","composed","confident","clear","focused","steady","resilient",
        "handled","managed","adapted","persevered","overcome","resolved",
        "constructive","rational","objective",
    ],
}

# ── Word categories (Naim et al. IEEE 2018) ───────────────────────────────────
QUANTIFIERS = [
    "all","every","each","entire","whole","best","most","always","never",
    "completely","absolutely","definitely","certainly","consistently",
    "thoroughly","precisely","exactly","specifically","particularly",
    "invariably","100%","fully","entirely","maximum",
]

PERCEPTUAL_WORDS = [
    "see","observe","notice","watch","look","view","hear","listen","feel",
    "sense","recognise","identify","understand","know","realise","discover",
    "learn","find","detect","measure","track","monitor","analyse","assess",
    "evaluate","review","investigate","examine","study","explore",
]

POSITIVE_SENTIMENT = [
    "great","excellent","outstanding","successful","effective","efficient",
    "improved","enhanced","optimised","solved","achieved","delivered",
    "exceeded","innovative","creative","strong","confident","clear","proud",
    "excited","motivated","passionate","dedicated","committed","thrilled",
    "positive","growth","impactful","rewarding","meaningful","valuable",
]

NEGATIVE_SENTIMENT = [
    "failed","difficult","struggled","problem","issue","mistake","error",
    "delay","miss","wrong","bad","poor","limited","confused","worried",
    "anxious","uncertain","unclear","incomplete","behind","frustrated",
    "challenging","blocker","obstacle","bottleneck","setback",
]

FILLER_WORDS = [
    "um","uh","like","basically","actually","you know","right","so",
    "just","kind of","sort of","i mean","literally","honestly",
    "obviously","clearly","simply","really","very","quite",
    "pretty much","i guess","stuff","things",
]


# ══════════════════════════════════════════════════════════════════════════════
#  TIME-AWARE SCORING  (v12.0)
# ══════════════════════════════════════════════════════════════════════════════
# Research basis:
#   Naim et al. (2018) IEEE Trans. Affective Computing — optimal behavioural
#   answer length ~90–150 s at 130 WPM; too short = insufficient depth,
#   too long = rambling/low confidence signal.
#   Rasipuram & Bhatt (2019) — technical answers benefit from longer responses
#   (up to 240 s) due to explanation complexity.
#
# Windows: (absolute_min_s, ideal_min_s, ideal_max_s, absolute_max_s)
# Outside the ideal window → linear interpolation toward 1.0.
# Outside the absolute window → floor at 1.0.

_TIME_WINDOWS: Dict[str, Dict[str, tuple]] = {
    "technical": {
        "easy":   (20,  50, 120, 200),
        "medium": (35,  80, 180, 300),
        "hard":   (50, 110, 240, 380),
    },
    "behavioural": {
        "easy":   (20,  50, 110, 180),
        "medium": (35,  75, 150, 240),
        "hard":   (45,  85, 180, 280),
    },
    "hr": {
        "easy":   (15,  40,  90, 150),
        "medium": (20,  50, 110, 180),
        "hard":   (25,  55, 120, 200),
    },
}

# Ideal window label strings for feedback messages
_TIME_IDEAL_LABELS: Dict[str, Dict[str, str]] = {
    "technical":   {"easy": "50–120 s", "medium": "80–180 s", "hard": "110–240 s"},
    "behavioural": {"easy": "50–110 s", "medium": "75–150 s", "hard":  "85–180 s"},
    "hr":          {"easy": "40–90 s",  "medium": "50–110 s", "hard":  "55–120 s"},
}


def compute_time_score(
    elapsed_s: float,
    q_type:    str = "technical",
    difficulty: str = "medium",
) -> Dict[str, object]:
    """
    Score answer timing on a 0–5 scale and return a rich metadata dict.

    Returns
    -------
    {
      "time_score"      : float   0–5
      "time_label"      : str     e.g. "Ideal pace" / "Too brief" / "Too long"
      "time_efficiency" : float   0–100  (percentage of ideal score)
      "time_verdict"    : str     one-word status for UI chip
      "time_feedback"   : str     one-sentence coaching note, or "" if ideal
      "elapsed_s"       : float
      "ideal_window"    : str     e.g. "80–180 s"
    }
    """
    qt   = _resolve_type(q_type)
    diff = difficulty.lower().strip()
    # v9.2: "all" mode passes the actual per-question difficulty from the RL
    # sequencer. If it somehow arrives as "all" here, default to medium.
    if diff not in ("easy", "medium", "hard"):
        diff = "medium"

    window     = _TIME_WINDOWS.get(qt, _TIME_WINDOWS["technical"]).get(diff)
    ideal_str  = _TIME_IDEAL_LABELS.get(qt, _TIME_IDEAL_LABELS["technical"]).get(diff, "—")
    abs_min, ideal_min, ideal_max, abs_max = window

    # If no timing data provided, skip scoring
    if elapsed_s <= 0:
        return {
            "time_score": 0.0, "time_label": "No timing data",
            "time_efficiency": 0.0, "time_verdict": "N/A",
            "time_feedback": "", "elapsed_s": 0.0,
            "ideal_window": ideal_str,
        }

    # Score computation
    if elapsed_s < abs_min:
        score = 1.0
    elif elapsed_s < ideal_min:
        # Ramp from 1.0 → 5.0 as elapsed moves from abs_min → ideal_min
        t     = (elapsed_s - abs_min) / max(1, ideal_min - abs_min)
        score = 1.0 + t * 4.0
    elif elapsed_s <= ideal_max:
        score = 5.0
    elif elapsed_s <= abs_max:
        # Ramp from 5.0 → 1.0 as elapsed moves from ideal_max → abs_max
        t     = (elapsed_s - ideal_max) / max(1, abs_max - ideal_max)
        score = 5.0 - t * 4.0
    else:
        score = 1.0

    score = round(float(max(1.0, min(5.0, score))), 2)
    eff   = round(score / 5.0 * 100.0, 1)

    # Label + verdict
    m, s = divmod(int(elapsed_s), 60)
    time_str = f"{m}m {s:02d}s" if m else f"{s}s"

    if score >= 4.5:
        label   = "Ideal pace"
        verdict = "Ideal"
        tip     = ""
    elif score >= 3.5:
        if elapsed_s < ideal_min:
            label   = "Slightly brief"
            verdict = "Brief"
            tip     = f"You answered in {time_str} — aim for at least {ideal_min}s to show full depth."
        else:
            label   = "Slightly long"
            verdict = "Long"
            tip     = f"Your answer ran {time_str} — try to wrap up by {ideal_max}s to stay concise."
    elif score >= 2.5:
        if elapsed_s < ideal_min:
            label   = "Too brief"
            verdict = "Brief"
            tip     = f"Only {time_str} — interviewers need more depth. Aim for {ideal_str}."
        else:
            label   = "Too long"
            verdict = "Long"
            tip     = f"Answer ran {time_str} — edit to the 3 most important points. Aim for {ideal_str}."
    else:
        if elapsed_s < abs_min:
            label   = "Much too brief"
            verdict = "Brief"
            tip     = f"Only {time_str} — this is too short to be evaluated properly. Aim for {ideal_str}."
        else:
            label   = "Much too long"
            verdict = "Long"
            tip     = f"Answer ran {time_str} — interviewers lose focus after {abs_max}s. Aim for {ideal_str}."

    return {
        "time_score":      score,
        "time_label":      label,
        "time_efficiency": eff,
        "time_verdict":    verdict,
        "time_feedback":   tip,
        "elapsed_s":       elapsed_s,
        "ideal_window":    ideal_str,
    }

_STAR_ORDER = ["Situation", "Task", "Action", "Result"]


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _sanitise(text: str, max_chars: int = 2000) -> str:
    """Strip control chars, backticks, and curly braces before prompt insertion."""
    text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', text)
    text = text.replace('`', "'").replace('{', '(').replace('}', ')')
    return text[:max_chars]


def _groq_call(api_key: str, messages: List[Dict],
               max_tokens: int = 512, temperature: float = 0.2,
               timeout: int = 30,
               retries: int = 2) -> Optional[str]:
    """
    Single Groq chat completion using the official groq package.
    Bypasses urllib/Cloudflare issues entirely.

    Returns the assistant message text, or None on unrecoverable failure.
    Retries up to `retries` times on transient errors.
    """
    try:
        from groq import Groq
    except ImportError:
        log.warning("groq package not installed. Run: pip install groq")
        return None

    import time

    client = Groq(api_key=api_key)

    for attempt in range(1, retries + 2):  # attempts = retries + 1
        try:
            response = client.chat.completions.create(
                model=_GROQ_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            exc_name = type(exc).__name__
            exc_msg  = str(exc)

            # Check for rate-limit or server errors by message content
            is_rate_limit  = "429" in exc_msg or "rate_limit" in exc_msg.lower()
            is_server_err  = any(c in exc_msg for c in ("500", "502", "503"))
            is_permanent   = any(c in exc_msg for c in ("401", "400", "403"))

            if is_permanent:
                log.warning(
                    f"Groq permanent error on attempt {attempt}: {exc_name}: {exc_msg}\n"
                    f"  401 = invalid/expired API key\n"
                    f"  403 = key lacks permission (check Cloudflare / key scope)\n"
                    f"  400 = bad request (check model name: {_GROQ_MODEL})"
                )
                return None

            elif is_rate_limit:
                wait = 2 ** attempt
                log.warning(
                    f"Groq rate limit on attempt {attempt}. "
                    f"Waiting {wait}s before retry."
                )
                if attempt <= retries:
                    time.sleep(wait)
                    continue

            elif is_server_err:
                log.warning(
                    f"Groq server error on attempt {attempt}: {exc_name}: {exc_msg}"
                )
                if attempt <= retries:
                    time.sleep(1.5 * attempt)
                    continue

            else:
                log.warning(
                    f"Groq error on attempt {attempt}: {exc_name}: {exc_msg}"
                )
                if attempt <= retries:
                    time.sleep(1.5 * attempt)
                    continue

            return None

    return None


def _star_order_bonus(answer_lower: str, star_scores: Dict[str, bool]) -> float:
    """Returns +0.5 if present STAR components appear in S→T→A→R order."""
    present = [c for c in _STAR_ORDER if star_scores.get(c, False)]
    if len(present) < 2:
        return 0.0
    positions = {}
    for component in present:
        m = re.search(STAR_PATTERNS[component], answer_lower, re.IGNORECASE)
        if m:
            positions[component] = m.start()
    ordered = sorted(positions.keys(), key=lambda c: positions.get(c, 0))
    return 0.5 if ordered == present else 0.0


def _compute_depth_score(wc: int, q_type: str = "technical") -> float:
    """
    Piecewise depth score — type-aware optimal zones (v10.1 FIX B).

    Technical (explanations are legitimately longer):
      0-49   → 0.0–1.0  (too brief)
      50-149 → 1.0–4.0  (building substance)
      150-350→ 4.0–5.0  (optimal zone)
      351-500→ 5.0–4.0  (slight over-length)
      500+   → floor 2.0 (genuinely rambling)

    Behavioural (concise stories — Naim 2018 mode ~175 words):
      0-39   → 0.0–1.0
      40-119 → 1.0–4.0
      120-200→ 4.0–5.0  (optimal zone)
      201-300→ 5.0–4.0
      300+   → floor 2.0

    HR (self-reflection needs substance but stays concise):
      0-39   → 0.0–1.0
      40-99  → 1.0–4.0
      100-175→ 4.0–5.0  (optimal zone)
      176-280→ 5.0–4.0
      280+   → floor 2.0
    """
    t = q_type.lower()

    if t == "technical":
        if wc < 50:
            depth = wc / 50.0
        elif wc < 150:
            depth = 1.0 + (wc - 50) / 100.0 * 3.0
        elif wc <= 350:
            depth = 4.0 + (wc - 150) / 200.0 * 1.0
        elif wc <= 500:
            depth = 5.0 - (wc - 350) / 150.0 * 1.0
        else:
            depth = max(2.0, 4.0 - (wc - 500) / 150.0 * 2.0)

    elif t == "hr":
        if wc < 40:
            depth = wc / 40.0
        elif wc < 100:
            depth = 1.0 + (wc - 40) / 60.0 * 3.0
        elif wc <= 175:
            depth = 4.0 + (wc - 100) / 75.0 * 1.0
        elif wc <= 280:
            depth = 5.0 - (wc - 175) / 105.0 * 1.0
        else:
            depth = max(2.0, 4.0 - (wc - 280) / 100.0 * 2.0)

    else:  # behavioural — unchanged from v8.0 research baseline
        if wc < 40:
            depth = wc / 40.0
        elif wc < 120:
            depth = 1.0 + (wc - 40) / 80.0 * 3.0
        elif wc <= 200:
            depth = 4.0 + (wc - 120) / 80.0 * 1.0
        elif wc <= 300:
            depth = 5.0 - (wc - 200) / 100.0 * 1.0
        else:
            depth = max(2.0, 4.0 - (wc - 300) / 100.0 * 2.0)

    return round(float(depth), 2)


def _compute_wpm_score(wc: int, duration_seconds: float) -> Optional[float]:
    """1-5 WPM score, or None if duration ≤5s. Optimal 120-160 WPM."""
    if duration_seconds <= 5:
        return None
    wpm = (wc / duration_seconds) * 60.0
    if 120 <= wpm <= 160:   score = 5.0
    elif 100 <= wpm < 120:  score = 3.0 + (wpm - 100) / 20.0 * 2.0
    elif wpm < 100:         score = max(1.0, 3.0 - (100 - wpm) / 20.0)
    elif 160 < wpm <= 200:  score = 5.0 - (wpm - 160) / 40.0 * 2.0
    else:                   score = max(1.0, 3.0 - (wpm - 200) / 30.0)
    return round(float(score), 2)


def _compute_fluency_score(filler_ratio: float) -> float:
    """0.5-5.0 additive fluency score. Formula: max(0.5, 5-ratio*25)."""
    return round(max(0.5, min(5.0, 5.0 - filler_ratio * 25.0)), 2)


def _compute_personality_nlp_score(ocean_scores: Dict[str, float]) -> float:
    """Weighted OCEAN → [1,5]. Kim et al. IEEE Access 2023 weights."""
    c = ocean_scores.get("Conscientiousness", 1.0)
    e = ocean_scores.get("Extraversion",      1.0)
    o = ocean_scores.get("Openness",          1.0)
    a = ocean_scores.get("Agreeableness",     1.0)
    return round(min(5.0, max(1.0, c*0.40 + e*0.35 + o*0.15 + a*0.10)), 2)


# ══════════════════════════════════════════════════════════════════════════════
#  IDEAL ANSWER CACHE  (module-level LRU)
# ══════════════════════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=256)
def _cached_generate_ideal_answer(question: str, role: str, difficulty: str,
                                   q_type: str, keywords_tuple: tuple,
                                   api_key: str) -> str:
    """
    LRU-cached ideal answer generation via Groq API.
    Cache key = (question, role, difficulty, q_type, keywords_tuple, api_key).
    Returns ideal answer text, or "" on failure.
    """
    difficulty_guidance = {
        "easy":   "Clear, accurate definition with 1-2 concrete examples. 80-100 words.",
        "medium": "Core concepts, practical usage, one key tradeoff. 100-120 words.",
        "hard":   "Deep technical answer: concepts, tradeoffs, edge cases, tools, "
                  "production considerations. 120-150 words. Senior level.",
    }
    type_guidance = {
        "technical":   "Focus on technical accuracy, correct terminology, tradeoffs, tools.",
        "behavioural": "Use STAR format: Situation, Task, Action, Result. Specific actions "
                       "and measurable outcomes.",
        "hr":          "Demonstrate self-awareness, career motivation, cultural fit, "
                       "professional goals. Concrete and specific.",
    }
    diff_hint = difficulty_guidance.get(difficulty.lower(), difficulty_guidance["medium"])
    type_hint = type_guidance.get(q_type.lower(), type_guidance["technical"])
    kw_str    = ", ".join(keywords_tuple) if keywords_tuple else "none specified"

    system_msg = (
        "You are a senior technical interviewer and hiring expert. "
        "Generate a model ideal answer a top candidate would give in a professional "
        "job interview. The answer must be factually accurate, concise, and directly "
        "address the question. Return ONLY the ideal answer text — no preamble, "
        "no labels, no markdown."
    )
    user_msg = (
        f"Question: {_sanitise(question, 500)}\n"
        f"Role: {_sanitise(role, 100)}\n"
        f"Difficulty: {difficulty}\n"
        f"Question type: {q_type}\n"
        f"Key concepts to cover: {kw_str}\n\n"
        f"Difficulty guidance: {diff_hint}\n"
        f"Type guidance: {type_hint}\n\n"
        "Write the ideal answer now:"
    )
    result = _groq_call(
        api_key,
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user",   "content": user_msg}],
        max_tokens=300, temperature=0.2,
    )
    if result:
        result = re.sub(r'^(ideal\s+answer\s*:|answer\s*:)\s*', '',
                        result, flags=re.IGNORECASE).strip()
        log.info(f"Generated ideal answer ({len(result.split())} words) "
                 f"[{role}][{difficulty}][{q_type}]: {question[:60]}")
        return result
    log.warning(f"Ideal answer generation failed: {question[:60]}")
    return ""


# ══════════════════════════════════════════════════════════════════════════════
#  SEMANTIC KEYWORD MATCHING  (v12.1)
# ══════════════════════════════════════════════════════════════════════════════

def _semantic_keyword_score(
    answer: str,
    kw_list: List[str],
    threshold: float = _KW_SEMANTIC_THRESHOLD,
) -> Tuple[float, List[str], List[Dict]]:
    """
    Replace binary exact-match keyword scoring with semantic cosine similarity.

    For each keyword, checks:
      1. Exact substring match (zero cost, fast path).
      2. SentenceTransformer cosine similarity between keyword and answer
         sentences — catches synonyms, acronyms, paraphrases.
      3. Falls back to exact-match only if SBERT unavailable.

    Returns
    -------
    kw_sc       : float  0–5  (proportion of semantically matched keywords × 5)
    kw_hits     : list[str]   keywords matched (exact OR semantic)
    kw_details  : list[dict]  per-keyword detail for UI display
                  {keyword, matched, match_type, score, best_sentence}
    """
    if not kw_list:
        return 0.0, [], []

    model = _get_sbert_kw_model()

    # ── Split answer into sentences for targeted matching ─────────────────────
    # Simple sentence splitter — avoids nltk dependency.
    sentences = [s.strip() for s in re.split(r'[.!?;]\s+', answer) if len(s.strip()) > 8]
    if not sentences:
        sentences = [answer]

    kw_hits: List[str]   = []
    kw_details: List[Dict] = []

    if model is not None:
        try:
            # Encode all sentences once (batch — efficient)
            sent_embeddings = model.encode(sentences, convert_to_numpy=True,
                                           show_progress_bar=False,
                                           normalize_embeddings=True)
        except Exception as exc:
            log.warning(f"SBERT encode failed: {exc}. Falling back to exact-match.")
            model = None

    answer_lower = answer.lower()

    for kw in kw_list:
        kw_lower = kw.lower()

        # ── Fast path: exact substring match ──────────────────────────────────
        if kw_lower in answer_lower:
            kw_hits.append(kw)
            kw_details.append({
                "keyword":      kw,
                "matched":      True,
                "match_type":   "exact",
                "score":        1.0,
                "best_sentence": "",
            })
            continue

        # ── Semantic path: cosine similarity via SBERT ────────────────────────
        if model is not None:
            try:
                kw_emb = model.encode([kw], convert_to_numpy=True,
                                      show_progress_bar=False,
                                      normalize_embeddings=True)
                # Cosine sim = dot product when both are L2-normalised
                sims = (sent_embeddings @ kw_emb.T).flatten()
                best_idx  = int(np.argmax(sims))
                best_score = float(sims[best_idx])

                if best_score >= threshold:
                    kw_hits.append(kw)
                    kw_details.append({
                        "keyword":       kw,
                        "matched":       True,
                        "match_type":    "semantic",
                        "score":         round(best_score, 3),
                        "best_sentence": sentences[best_idx][:120],
                    })
                else:
                    kw_details.append({
                        "keyword":       kw,
                        "matched":       False,
                        "match_type":    "semantic",
                        "score":         round(best_score, 3),
                        "best_sentence": "",
                    })
                continue
            except Exception as exc:
                log.warning(f"SBERT keyword sim failed for '{kw}': {exc}.")

        # ── Fallback: no match ────────────────────────────────────────────────
        kw_details.append({
            "keyword":      kw,
            "matched":      False,
            "match_type":   "exact_only",
            "score":        0.0,
            "best_sentence": "",
        })

    matched_count = len(kw_hits)
    kw_sc = round(min(5.0, matched_count / max(1, len(kw_list)) * 5.0), 3)
    return kw_sc, kw_hits, kw_details


# ══════════════════════════════════════════════════════════════════════════════
#  ANSWER EVALUATOR  v12.1
# ══════════════════════════════════════════════════════════════════════════════

class AnswerEvaluator:
    """
    Research-validated multimodal NLP scorer v10.0.

    Key upgrade from v9.0: Question-type-aware scoring weights.
      Technical   → STAR disabled; relevance + keywords primary
      Behavioural → STAR primary; word categories valued
      HR          → Relevance + depth primary; partial STAR valued

    Also carries: dynamic ideal answer generation (v9.0), all v8.x fixes.
    """

    def __init__(self, groq_api_key: str = "") -> None:
        self._groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
        if not self._groq_api_key:
            msg = (
                "[AnswerEvaluator] WARNING: GROQ_API_KEY not set. "
                "Dynamic ideal answer generation is DISABLED. "
                "Relevance will fall back to TF-IDF on static ideal_answer. "
                "Set GROQ_API_KEY env var or pass groq_api_key= to AnswerEvaluator()."
            )
            log.warning(msg)
            print(msg, file=sys.stderr)

        self._grammar_tool = None
        if LT_AVAILABLE:
            try:
                self._grammar_tool = language_tool_python.LanguageToolPublicAPI("en-US")
                log.info("LanguageTool grammar checker ready.")
            except Exception as exc:
                log.warning(f"LanguageTool unavailable: {exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def score_answer(self, answer: str, question_dict: Dict,
                     answer_duration_seconds: float = 0.0) -> Dict:
        """
        Primary entry point. Generates a dynamic ideal answer via Groq API
        (role/difficulty/type-aware), then runs the full type-aware evaluation.

        question_dict expected keys (all optional, graceful defaults):
          question     : str  — interview question text
          role         : str  — e.g. "Data Analyst"
          difficulty   : str  — "Easy" | "Medium" | "Hard"
          type         : str  — "Technical" | "Behavioural" | "HR"
          keywords     : list — expected domain keywords
          ideal_answer : str  — static fallback if API unavailable

        answer_duration_seconds : time taken to answer in seconds (enables
                                  time-aware scoring). Pass 0.0 to disable.
        """
        generated, ideal_source = self._get_ideal_answer(question_dict)
        q_type     = question_dict.get("type", "Technical")
        difficulty = question_dict.get("difficulty", "medium")

        return self.evaluate(
            answer=answer,
            ideal_answer=generated,
            question_keywords=question_dict.get("keywords", []),
            answer_duration_seconds=answer_duration_seconds,
            question_text=question_dict.get("question", ""),
            question_type=q_type,
            difficulty=difficulty,
            _ideal_answer_source=ideal_source,
        )

    def evaluate(self,
                 answer: str,
                 ideal_answer: str = "",
                 question_keywords: Optional[List[str]] = None,
                 answer_duration_seconds: float = 0.0,
                 question_text: str = "",
                 question_type: str = "Technical",
                 difficulty: str = "medium",
                 _ideal_answer_source: str = "none") -> Dict:
        """
        Full evaluation pipeline with question-type-aware weights.

        Args:
            answer                  : Candidate's answer (ASR or typed)
            ideal_answer            : Reference for relevance scoring (generated or static)
            question_keywords       : Domain keywords expected in the answer
            answer_duration_seconds : Duration for WPM (0.0 = exclude WPM)
            question_text           : Question text — enriches TF-IDF fallback reference
            question_type           : "Technical" | "Behavioural" | "HR"
                                      Controls which weight profile is applied.
            _ideal_answer_source    : Internal tag set by score_answer()
        """
        if not answer or len(answer.strip()) < 5:
            return self._empty_result(question_type)

        # ── Resolve question type → weight profile ────────────────────────────
        q_type_key = _resolve_type(question_type)
        W = WEIGHT_PROFILES[q_type_key]   # shorthand

        al    = answer.lower()
        words = al.split()
        wc    = len(words)

        # ── 1. STAR structural analysis ───────────────────────────────────────
        # Computed always (surfaced in report even when weight=0 for Technical).
        star_scores = {
            c: bool(re.search(p, al, re.IGNORECASE))
            for c, p in STAR_PATTERNS.items()
        }
        star_count  = sum(star_scores.values())
        star_base   = star_count / 4.0 * 5.0
        order_bonus = _star_order_bonus(al, star_scores)
        star_sc     = round(min(5.0, star_base + order_bonus), 3)

        # ── 2. Word category quality ──────────────────────────────────────────
        quant_count    = sum(1 for w in words if w in QUANTIFIERS)
        percep_count   = sum(1 for w in words if w in PERCEPTUAL_WORDS)
        quant_density  = min(1.0, quant_count  / max(1, wc) * 100 / 5.0)
        percep_density = min(1.0, percep_count / max(1, wc) * 100 / 4.0)
        word_cat_sc    = round(
            (quant_density * 0.55 + percep_density * 0.45) * 5.0, 3)

        # ── 3. Relevance score (v11.0 — Groq API for ALL paths) ──────────────
        # _groq_relevance_score() calls Groq in every case.
        # When ideal_answer exists: scores against it directly.
        # When only keywords exist: builds a context prompt from question+keywords.
        # Falls back to TF-IDF ONLY if the Groq HTTP call itself errors out.
        if ideal_answer:
            tfidf_sim, relevance_source = self._groq_relevance_score(
                answer       = answer,
                reference    = ideal_answer,
                context      = question_text,
                source_label = "api_groq",
            )
        elif question_keywords:
            # Build a rich context reference from question text + keywords
            kw_reference = (
                f"Question: {question_text}\n"
                f"Key concepts expected: {', '.join(question_keywords)}"
            )
            tfidf_sim, relevance_source = self._groq_relevance_score(
                answer       = answer,
                reference    = kw_reference,
                context      = "",
                source_label = "api_groq_kw",
            )
        else:
            tfidf_sim        = 0.0
            relevance_source = "none"

        # ── 4. Keyword relevance (v12.1 — semantic matching) ─────────────────
        # Replaces binary exact-match (v11.x) with SentenceTransformer cosine
        # similarity scoring.  Each keyword is matched against the full answer
        # using sentence-level embeddings, catching synonyms, acronyms, and
        # paraphrases that exact-match silently misses.
        #
        # Example: keyword="database", answer mentions "Postgres" or "RDBMS"
        #   Old (v11): kw_hits=[]  → kw_sc=0.0  (under-scores technical answers)
        #   New (v12): semantic sim ≥ 0.45 → kw_hits=["database"] → kw_sc=5.0
        #
        # Fallback chain (preserves all v11.1 bug fixes):
        #   SBERT available  → semantic cosine similarity per keyword
        #   SBERT unavailable → exact substring match (original behaviour)
        #
        # FIX v11.1 (carried forward):
        #   Empty kw_list → kw_sc=0.0, W["keyword"]=0, freed weight → relevance.
        #   Always copy W before mutation so WEIGHT_PROFILES is never changed.

        kw_list = [k.lower() for k in (question_keywords or [])]

        W = dict(W)   # always work on a local copy from here on

        if not kw_list:
            # No keywords defined — zero out the dimension and
            # redistribute its weight to relevance (v11.1 fix carried forward)
            kw_sc      = 0.0
            kw_hits    = []
            kw_details = []
            freed_kw        = W["keyword"]
            W["keyword"]    = 0.0
            W["relevance"] += freed_kw
        else:
            # v12.1: semantic matching replaces binary hit/miss
            kw_sc, kw_hits, kw_details = _semantic_keyword_score(
                answer  = answer,
                kw_list = kw_list,
                threshold = _KW_SEMANTIC_THRESHOLD,
            )
            log.info(
                f"Keyword scoring (semantic): {len(kw_hits)}/{len(kw_list)} matched "
                f"[exact: {sum(1 for d in kw_details if d['match_type']=='exact')}, "
                f"semantic: {sum(1 for d in kw_details if d['match_type']=='semantic' and d['matched'])}] "
                f"→ kw_sc={kw_sc:.2f}"
            )

        # v8.3 FIX 3 (updated for v11.0 source labels, carried forward):
        # When using keywords-only Groq path, keyword dimension and relevance
        # dimension measure similar things — halve keyword weight to avoid overlap.
        if relevance_source == "api_groq_kw" and kw_list:
            freed = W["keyword"] / 2.0
            W["keyword"]   -= freed
            W["relevance"] += freed

        # ── 5. Depth + fluency composite ──────────────────────────────────────
        # v10.1 FIX B: type-aware depth zones (technical allows up to 350 words)
        depth_sc      = _compute_depth_score(wc, q_type_key)
        filler_count  = sum(al.split().count(f) for f in FILLER_WORDS)
        filler_ratio  = filler_count / max(1, wc)
        fluency_sc    = _compute_fluency_score(filler_ratio)

        wpm_score_val = _compute_wpm_score(wc, answer_duration_seconds)
        has_wpm       = wpm_score_val is not None
        wpm_score     = wpm_score_val if has_wpm else 0.0

        if has_wpm:
            depth_fluency_sc = round(max(0.5, min(5.0,
                depth_sc * 0.40 + wpm_score_val * 0.35 + fluency_sc * 0.25)), 2)
        else:
            depth_fluency_sc = round(max(0.5, min(5.0,
                depth_sc * 0.65 + fluency_sc * 0.35)), 2)

        # ── 6. Grammar quality ────────────────────────────────────────────────
        grammar_score, grammar_errors = self._grammar_check(answer)

        # ── 7. Time-aware scoring (v12.0) ─────────────────────────────────────
        # Computes a 0–5 time score from elapsed seconds vs ideal window.
        # Applied as a small post-processing modifier (±0.25 max) so timing
        # never dominates NLP quality — it is a tiebreaker signal.
        time_data = compute_time_score(
            elapsed_s  = answer_duration_seconds,
            q_type     = q_type_key,
            difficulty = difficulty,
        )
        time_score_val = time_data["time_score"]

        # Modifier: time penalty/bonus — capped at ±0.25 to stay tiebreaker only
        if answer_duration_seconds > 0:
            if time_score_val >= 4.5:
                time_modifier = +0.10
            elif time_score_val >= 3.5:
                time_modifier = +0.05
            elif time_score_val >= 2.5:
                time_modifier =  0.00
            elif time_score_val >= 1.5:
                time_modifier = -0.10
            else:
                time_modifier = -0.20
        else:
            time_modifier = 0.0

        # ── Composite score — type-aware weights ──────────────────────────────
        raw_score = (
            star_sc                    * W["star"]      +
            word_cat_sc                * W["word_cat"]  +
            tfidf_sim * 5.0            * W["relevance"] +
            kw_sc                      * W["keyword"]   +
            depth_fluency_sc           * W["depth_flu"] +
            grammar_score / 100.0 * 5.0 * W["grammar"]
        )
        final_score = round(min(5.0, max(1.0, raw_score + time_modifier)), 2)

        # ── DISC profiling ────────────────────────────────────────────────────
        disc_traits = {
            tr: sum(1 for w in ws if w in al)
            for tr, ws in DISC_KEYWORDS.items()
        }
        disc_dominant = (max(disc_traits, key=disc_traits.get)
                         if any(disc_traits.values()) else "None")

        # ── Big Five OCEAN ────────────────────────────────────────────────────
        ocean_scores: Dict[str, float] = {}
        for trait, kws in OCEAN_KEYWORDS.items():
            hits = sum(1 for w in kws if w in al)
            ocean_scores[trait] = round(min(5.0, 1.0 + hits * 0.8), 2)

        conscientiousness = round(
            ocean_scores.get("Conscientiousness", 3.0) * 0.6 +
            min(5.0, disc_traits.get("Conscientiousness", 0) * 0.8 + 1.0) * 0.4,
            2)
        personality_nlp_sc = _compute_personality_nlp_score(ocean_scores)

        # ── Sentiment intensity ───────────────────────────────────────────────
        pos_count = sum(1 for w in POSITIVE_SENTIMENT if w in al)
        neg_count = sum(1 for w in NEGATIVE_SENTIMENT if w in al)
        sentiment_intensity = round(
            (pos_count - neg_count) / max(1, pos_count + neg_count + 1) * 3.0, 2)

        # ── Vocabulary diversity ──────────────────────────────────────────────
        ttr = round(len(set(words)) / max(1, wc), 3)

        # ── Hiring signal (anchored to final_score) ───────────────────────────
        sentiment_adj = (sentiment_intensity + 3.0) / 6.0
        fluency_adj   = fluency_sc / 5.0
        hiring_signal = round(min(5.0, max(1.0,
            final_score   * 0.70 +
            sentiment_adj * 5.0 * 0.15 +
            fluency_adj   * 5.0 * 0.15
        )), 2)

        # ── Feedback (type-aware) ─────────────────────────────────────────────
        fb: List[str] = []

        # STAR feedback — only for Behavioural/HR (weight > 0)
        if W["star"] > 0:
            missing_star = [k for k, v in star_scores.items() if not v]
            if missing_star:
                fb.append(
                    f"Add STAR component{'s' if len(missing_star) > 1 else ''}: "
                    f"{', '.join(missing_star)}.")
            if star_count >= 2 and order_bonus == 0.0:
                fb.append(
                    "Structure your answer in order: "
                    "Situation → Task → Action → Result.")

        # Word category feedback — only when weight is meaningful
        if W["word_cat"] >= 0.15:
            if quant_count / max(1, wc) * 100 < 0.30:
                fb.append(
                    "Use confident quantifiers — 'every', 'consistently', "
                    "'precisely' signal certainty.")
            if percep_count / max(1, wc) * 100 < 0.20:
                fb.append(
                    "Add analytical verbs — 'I identified', 'I measured', "
                    "'I observed' show structured thinking.")

        # Keyword coverage feedback — uses semantic match details when available
        if kw_list and len(kw_hits) / max(1, len(kw_list)) < 0.5:
            missing_kw = [d["keyword"] for d in kw_details if not d["matched"]][:3]
            if missing_kw:
                fb.append(f"Cover key concepts: {', '.join(missing_kw)}.")

        # Relevance feedback
        if tfidf_sim < 0.25 and ideal_answer:
            fb.append("Stay focused on the core question topic.")

        # Filler words
        if filler_ratio > 0.07:
            fb.append(
                f"Reduce filler words ({int(filler_count)} detected) — "
                "pause instead of saying 'um' or 'like'.")

        # Word count — type-aware thresholds (v10.1 FIX B)
        wc_min  = {"technical": 80,  "behavioural": 60,  "hr": 50 }.get(q_type_key, 60)
        wc_max  = {"technical": 500, "behavioural": 300, "hr": 280}.get(q_type_key, 300)
        wc_opt  = {"technical": "150-350", "behavioural": "120-200", "hr": "100-175"}.get(q_type_key, "120-200")
        if wc < wc_min:
            fb.append(
                f"Elaborate more — aim for {wc_opt} words to give enough context.")
        elif wc > wc_max:
            fb.append(
                f"Answer is {wc} words — aim to be more concise ({wc_opt} words optimal).")

        # Sentiment
        if sentiment_intensity < -0.5:
            fb.append(
                "Frame negatives positively — focus on what you learned "
                "and how you overcame the challenge.")

        # Vocabulary
        if ttr < 0.45:
            fb.append("Vary your vocabulary to show language depth.")

        # WPM
        if has_wpm and wpm_score < 3.0:
            wpm_actual = round((wc / answer_duration_seconds) * 60)
            if wpm_actual < 120:
                fb.append(
                    f"Speaking pace ({wpm_actual} WPM) is slow — aim for 120-160 WPM.")
            else:
                fb.append(
                    f"Speaking pace ({wpm_actual} WPM) is fast — slow down slightly.")

        # Time efficiency feedback — appended only when timing data exists
        if answer_duration_seconds > 0 and time_data.get("time_feedback"):
            fb.append(time_data["time_feedback"])

        if not fb:
            feedback = {
                5: "Exceptional answer — confident, structured, impactful.",
                4: "Strong answer. Add one more specific outcome or metric.",
                3: "Good structure. Strengthen with concrete examples.",
            }.get(round(hiring_signal),
                  "Well-structured answer. Keep practising.")
        else:
            feedback = " ".join(fb)

        # ── Output dict ───────────────────────────────────────────────────────
        return {
            # ── Core (backwards-compatible) ───────────────────────────────────
            "score":              final_score,
            "final_score":        round(final_score / 5.0 * 100, 1),
            "similarity_score":   round(tfidf_sim * 100, 1),
            "grammar_score":      round(grammar_score, 1),
            "grammar_errors":     grammar_errors,
            "star_scores":        star_scores,
            "disc_traits":        disc_traits,
            "keyword_hits":       kw_hits,
            "keyword_details":    kw_details,   # v12.1: per-keyword semantic scores
            "tfidf_sim":          round(tfidf_sim, 3),
            "word_count":         wc,
            "feedback":           feedback,
            "relevance_source":   relevance_source,
            "fluency_penalty":    round(fluency_sc / 5.0, 3),

            # ── v7.0 ─────────────────────────────────────────────────────────
            "word_cat_score":     round(word_cat_sc, 2),
            "quantifier_count":   quant_count,
            "perceptual_count":   percep_count,
            "depth_fluency_sc":   depth_fluency_sc,
            "wpm_score":          round(wpm_score, 2),
            "filler_count":       filler_count,
            "filler_ratio":       round(filler_ratio, 3),
            "ocean_scores":       ocean_scores,
            "disc_dominant":      disc_dominant,
            "conscientiousness":  conscientiousness,
            "sentiment_intensity":sentiment_intensity,
            "vocab_diversity":    ttr,
            "hiring_signal":      hiring_signal,
            "star_count":         star_count,

            # ── v8.0 ─────────────────────────────────────────────────────────
            "depth_score":        depth_sc,
            "fluency_score":      fluency_sc,
            "order_bonus":        order_bonus,
            "has_wpm":            has_wpm,
            "personality_nlp_sc": personality_nlp_sc,

            # ── v8.3 ─────────────────────────────────────────────────────────
            "star_weight_used":   W["star"],
            "kw_weight_used":     W["keyword"],   # 0.0 when no keywords defined
            "kw_defined":         bool(kw_list),  # UI can use this to hide the dim

            # ── v9.0 ─────────────────────────────────────────────────────────
            "generated_ideal_answer": ideal_answer,
            "ideal_answer_source":    _ideal_answer_source,

            # ── v10.0 ────────────────────────────────────────────────────────
            # question_type and weight_profile let the UI show exactly which
            # scoring mode was applied and why.
            "question_type":    q_type_key,
            "weight_profile":   dict(W),

            # ── v12.0 time-aware scoring ──────────────────────────────────────
            "time_score":       time_data["time_score"],
            "time_label":       time_data["time_label"],
            "time_efficiency":  time_data["time_efficiency"],
            "time_verdict":     time_data["time_verdict"],
            "time_feedback":    time_data["time_feedback"],
            "time_elapsed_s":   time_data["elapsed_s"],
            "time_ideal_window":time_data["ideal_window"],
            "time_modifier":    time_modifier,
        }

    def hiring_prediction(self, result: Dict) -> Dict:
        """Hiring likelihood from evaluate() result. Anchored to final_score."""
        hs = result.get("hiring_signal", 2.5)
        if hs >= 4.5:   label, tip = "Strong Hire",  "Excellent indicators across all dimensions."
        elif hs >= 3.5: label, tip = "Lean Hire",    "Good. Strengthen with confident quantifiers."
        elif hs >= 2.5: label, tip = "Neutral",      "Mixed signals. More structured examples needed."
        elif hs >= 1.5: label, tip = "Lean No-Hire", "Low confidence language. Practice STAR format."
        else:           label, tip = "No-Hire",      "Significant improvement in structure needed."
        return {
            "hiring_signal": hs,
            "hiring_label":  label,
            "hiring_tip":    tip,
            "top_factors": {
                "STAR completeness": result.get("star_count", 0),
                "Quantifiers used":  result.get("quantifier_count", 0),
                "Vocab diversity":   result.get("vocab_diversity", 0.0),
                "Filler ratio":      result.get("filler_ratio", 0.0),
                "Sentiment":         result.get("sentiment_intensity", 0.0),
                "OCEAN personality": result.get("personality_nlp_sc", 2.5),
            },
        }

    def generate_improvement_suggestion(self,
                                         answer:      str,
                                         question:    str,
                                         eval_result: Dict,
                                         q_type:      str = "Technical") -> str:
        """
        v13.0: Generate a one-line, actionable improvement suggestion for the
        candidate based on the weakest signal in their answer.

        Called AFTER scoring — does not affect the score, only the UI tip card.
        Skipped entirely if score >= 4.0 (already strong answer).

        Priority order for weakness detection:
          1. Missing required STAR component  (Behavioural/HR only)
          2. Low relevance  (similarity_score < 40)
          3. Shallow depth  (depth_score < 2.5 or word_count < 70)
          4. Vague Action   (Behavioural, star Action missing, score < 3.0)
          5. Generic fallback

        Format: "Your [X] was [problem] — try: '[rewrite hint]'"

        Returns empty string on failure or if no suggestion needed.
        """
        try:
            score = eval_result.get("score", 0)
            if score >= 4.0:
                return ""   # strong answer — no suggestion needed

            q_type_k = _resolve_type(q_type)

            # ── Detect weakest signal ──────────────────────────────────────────
            weakness = ""

            # 1. Missing required STAR component (Behavioural / HR only)
            if q_type_k in ("behavioural", "hr"):
                required = {
                    "behavioural": ["Situation", "Task", "Action", "Result"],
                    "hr":          ["Situation", "Action"],
                }.get(q_type_k, [])
                star = eval_result.get("star_scores", {})
                missing = [c for c in required if not star.get(c, False)]
                if missing:
                    weakness = f"missing {missing[0]} in your STAR structure"

            # 2. Low relevance
            if not weakness:
                rel = eval_result.get("similarity_score", 100)
                if rel < 40:
                    weakness = "answer relevance was low — you may have drifted off-topic"

            # 3. Shallow depth
            if not weakness:
                depth = eval_result.get("depth_score", 5.0)
                wc    = eval_result.get("word_count", 200)
                if depth < 2.5 or wc < 70:
                    weakness = "answer lacked depth and detail"

            # 4. Vague Action (Behavioural only)
            if not weakness and q_type_k == "behavioural":
                star = eval_result.get("star_scores", {})
                if not star.get("Action", True) and score < 3.0:
                    weakness = "Action component was vague — be more specific about what YOU did"

            # 5. Generic fallback
            if not weakness:
                weakness = "overall structure could be tighter"

            # ── Call Groq for the rewrite hint ─────────────────────────────────
            if not self._api_key:
                return f"Tip: Your {weakness}."

            prompt = (
                f"Question: {question[:200]}\n"
                f"Candidate answer (first 300 chars): {answer[:300]}\n"
                f"Identified weakness: {weakness}\n\n"
                f"Write exactly ONE sentence of improvement feedback in this format:\n"
                f"\"Your [X] was [problem] — try: '[concrete rewrite hint]'\"\n"
                f"Keep it under 25 words. Be specific and actionable. "
                f"Do not mention scores or percentages."
            )

            messages = [
                {"role": "system",
                 "content": "You are a concise interview coach. Return only the feedback sentence, nothing else."},
                {"role": "user", "content": prompt},
            ]
            raw = _groq_call(
                api_key=self._api_key,
                messages=messages,
                max_tokens=80,
                temperature=0.4,
            )
            suggestion = raw.strip().strip('"').strip("'")
            return suggestion if suggestion else f"Tip: Your {weakness}."

        except Exception as exc:
            log.debug(f"generate_improvement_suggestion failed: {exc}")
            return ""

    def clear_cache(self) -> None:
        """Clear in-process ideal answer LRU cache."""
        _cached_generate_ideal_answer.cache_clear()
        log.info("Ideal answer cache cleared.")

    def cache_info(self) -> str:
        info = _cached_generate_ideal_answer.cache_info()
        return (f"Cache — hits: {info.hits}, misses: {info.misses}, "
                f"size: {info.currsize}/{info.maxsize}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_ideal_answer(self, question_dict: Dict) -> Tuple[str, str]:
        """
        Returns (ideal_answer_text, source_tag).
        Source: "generated" | "static_fallback" | "none"
        """
        question   = question_dict.get("question", "").strip()
        role       = question_dict.get("role", "General").strip()
        difficulty = question_dict.get("difficulty", "Medium").strip()
        q_type     = question_dict.get("type", "Technical").strip()
        keywords   = tuple(sorted(question_dict.get("keywords", [])))
        static_ia  = question_dict.get("ideal_answer", "").strip()

        if self._groq_api_key and question:
            generated = _cached_generate_ideal_answer(
                question, role, difficulty, q_type, keywords, self._groq_api_key)
            if generated and len(generated.split()) >= 30:
                return generated, "generated"

        if static_ia:
            log.debug(f"Using static ideal_answer: {question[:60]}")
            return static_ia, "static_fallback"

        return "", "none"

    def _groq_relevance_score(self,
                               answer: str,
                               reference: str,
                               context: str = "",
                               source_label: str = "api_groq") -> Tuple[float, str]:
        """
        Universal Groq-based relevance scorer (v11.0).

        Replaces both _api_relevance() and _tfidf_similarity() as the
        primary relevance path. Called for every question type and every
        fallback scenario.

        Args:
            answer       : candidate's answer text
            reference    : ideal answer OR question+keywords context string
            context      : optional extra context (question text)
            source_label : tag written into relevance_source in the result dict
                           "api_groq"    — scored vs generated/static ideal answer
                           "api_groq_kw" — scored vs question+keywords context

        Returns:
            (score 0.0-1.0, source_label) on success
            (tfidf_fallback_score, "tfidf_fallback") if Groq HTTP call fails
            (0.0, "none") if no API key and sklearn unavailable

        Scoring rubric sent to LLM:
            1.0  — covers all key concepts of the reference
            0.75 — covers most concepts, minor gaps
            0.50 — covers about half the concepts
            0.25 — touches the topic but misses most key points
            0.00 — off-topic or no meaningful overlap

        Why this is better than TF-IDF:
            The LLM understands that "AUC" and "discrimination ability across
            thresholds" are the same concept. TF-IDF sees them as completely
            different tokens and returns near-zero similarity. The LLM scores
            this correctly at ~0.85.
        """
        if not answer or not reference:
            return 0.0, "none"

        # No API key — fall back to TF-IDF immediately
        if not self._groq_api_key:
            score = self._tfidf_similarity(answer, reference)
            return score, "tfidf_fallback"

        safe_ref    = _sanitise(reference, max_chars=2000)
        safe_answer = _sanitise(answer,    max_chars=2000)
        safe_ctx    = _sanitise(context,   max_chars=300) if context else ""

        context_line = f"\nAdditional context: {safe_ctx}" if safe_ctx else ""

        # ── BARS rubrics per question type (Sapia AI / SIOP 2024) ────────────
        # Behaviorally Anchored Rating Scales anchor each score level to
        # observable candidate behaviours rather than abstract quality labels.
        # This produces significantly more consistent and explainable scores.
        _bars_rubric = {
            "behavioural": (
                "\nBehaviorally Anchored Rating Scale for this answer:\n"
                "  1.0 — Candidate gave a specific situation with named stakeholders, "
                "their personal action described step-by-step, and a measurable outcome with quantified impact.\n"
                "  0.75 — Specific situation and clear personal action, but outcome is described "
                "generally without quantified impact.\n"
                "  0.50 — Situation described but personal contribution is vague (uses 'we' throughout); "
                "result mentioned without detail.\n"
                "  0.25 — Generic or theoretical response; no concrete personal example; "
                "could apply to anyone.\n"
                "  0.00 — No example given, off-topic, or complete absence of STAR structure.\n"
            ),
            "technical": (
                "\nBehaviorally Anchored Rating Scale for this answer:\n"
                "  1.0 — Demonstrates deep understanding: explains the mechanism, "
                "trade-offs, edge cases, and when NOT to use the approach.\n"
                "  0.75 — Correct explanation of the main concept with at least one "
                "trade-off or real-world application mentioned.\n"
                "  0.50 — Covers the surface-level definition accurately but misses "
                "depth, trade-offs, or practical context.\n"
                "  0.25 — Partially correct; contains errors or misconceptions mixed "
                "with correct information.\n"
                "  0.00 — Incorrect, off-topic, or no meaningful technical content.\n"
            ),
            "hr": (
                "\nBehaviorally Anchored Rating Scale for this answer:\n"
                "  1.0 — Clear, structured response showing self-awareness, "
                "genuine reflection, and alignment with professional values.\n"
                "  0.75 — Thoughtful answer with a concrete example or rationale; "
                "minor lack of depth.\n"
                "  0.50 — Reasonable answer but generic; could apply to any candidate "
                "without personal specificity.\n"
                "  0.25 — Superficial or rehearsed-sounding; no personal connection "
                "to the question.\n"
                "  0.00 — Off-topic, evasive, or no meaningful content.\n"
            ),
        }
        # Detect question type from context string (contains question text)
        _ctx_lower = (safe_ctx + safe_ref).lower()
        if any(w in _ctx_lower for w in ["tell me about a time","describe a situation",
                                          "give me an example","behavioural","behavioral"]):
            _bars = _bars_rubric["technical"]   # will be overridden below if type detected
        _bars = _bars_rubric.get(
            "behavioural" if any(w in _ctx_lower for w in
                ["tell me about a time","describe a situation","give me an example","past experience"])
            else "technical" if any(w in _ctx_lower for w in
                ["explain","implement","design","algorithm","complexity","database","system"])
            else "hr",
            _bars_rubric["technical"]
        )

        prompt = (
            "You are an expert interview answer evaluator.\n"
            "Score how well the candidate answer covers the key concepts "
            "in the reference. Return ONLY a JSON object with a single key "
            "'relevance' whose value is a float between 0.0 and 1.0.\n\n"
            "Scoring rubric:\n"
            "  1.0  — covers all key concepts of the reference\n"
            "  0.75 — covers most concepts, minor gaps\n"
            "  0.50 — covers about half the concepts\n"
            "  0.25 — touches the topic but misses most key points\n"
            "  0.00 — off-topic or no meaningful overlap\n\n"
            f"{_bars}"
            "IMPORTANT: Score based on CONCEPTS and MEANING, not exact wording.\n"
            "If the candidate uses different words that mean the same thing "
            "(e.g. 'AUC' vs 'discrimination ability', 'generalizes' vs "
            "'generalises', 'MSE' vs 'Mean Squared Error'), treat them as equivalent.\n"
            "FAIRNESS: Evaluate only the technical correctness and relevance of "
            "the content. Do NOT penalise non-native English patterns, informal "
            "phrasing, accent-correlated transcription errors, or speech disfluencies "
            "(um, uh, you know). Judge what was communicated, not how it was said.\n"
            f"{context_line}\n\n"
            f"Reference:\n{safe_ref}\n\n"
            f"Candidate answer:\n{safe_answer}\n\n"
            'Return ONLY valid JSON. Example: {"relevance": 0.75}'
        )

        raw = _groq_call(
            self._groq_api_key,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0.0,
        )

        if raw:
            try:
                raw   = raw.replace("```json", "").replace("```", "").strip()
                score = float(json.loads(raw)["relevance"])
                score = max(0.0, min(1.0, score))
                log.info(f"Groq relevance ({source_label}): {score:.2f}")
                return score, source_label
            except Exception as exc:
                log.warning(
                    f"Groq relevance response parse failed ({exc}). "
                    f"Raw response was: {repr(raw[:200])}. "
                    f"Using TF-IDF fallback."
                )
        else:
            log.warning(
                "Groq relevance call returned None (see error above). "
                "Using TF-IDF fallback. "
                "Common causes: invalid API key, no internet, rate limit."
            )

        # Emergency TF-IDF fallback — only if Groq HTTP call itself failed
        fallback = self._tfidf_similarity(answer, reference)
        log.warning(f"TF-IDF fallback score: {fallback:.2f}")
        return fallback, "tfidf_fallback"

    def _tfidf_similarity(self, answer: str, reference: str) -> float:
        """
        Emergency TF-IDF fallback. Only called when the Groq HTTP call itself
        fails (network error, rate limit). NOT used as a primary scorer.

        Uses sublinear_tf=True to reduce frequency dominance.
        Returns 0.0 if sklearn is not installed.
        """
        if not SKLEARN_AVAILABLE or not reference:
            return 0.0
        try:
            vect = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
            ).fit([reference, answer])
            vecs = vect.transform([reference, answer])
            return float(cosine_similarity(vecs[0], vecs[1])[0][0])
        except Exception:
            return 0.0

    def _grammar_check(self, text: str) -> Tuple[float, List[Dict]]:
        if not self._grammar_tool:
            return 85.0, []
        try:
            matches = self._grammar_tool.check(text)
            penalty = min(100.0, len(matches) * 4)
            errors  = []
            for m in matches[:6]:
                ctx = text[max(0, m.offset - 20): m.offset + m.errorLength + 20]
                errors.append({
                    "message":      m.message,
                    "context":      ctx,
                    "replacements": list(m.replacements[:3]),
                })
            return max(0.0, 100.0 - penalty), errors
        except Exception:
            return 80.0, []

    def _empty_result(self, question_type: str = "Technical") -> Dict:
        q_type_key = _resolve_type(question_type)
        W = WEIGHT_PROFILES[q_type_key]
        return {
            "score": 1.0, "final_score": 20.0,
            "similarity_score": 0.0, "grammar_score": 0.0,
            "grammar_errors": [], "star_scores": {},
            "disc_traits": {}, "keyword_hits": [],
            "keyword_details": [],   # v12.1
            "tfidf_sim": 0.0, "word_count": 0,
            "feedback": "No answer provided.",
            "fluency_penalty": 1.0,
            "word_cat_score": 0.0, "quantifier_count": 0,
            "perceptual_count": 0, "depth_fluency_sc": 0.0,
            "wpm_score": 0.0, "filler_count": 0, "filler_ratio": 0.0,
            "ocean_scores": {}, "disc_dominant": "None",
            "conscientiousness": 1.0, "sentiment_intensity": 0.0,
            "vocab_diversity": 0.0, "hiring_signal": 1.0, "star_count": 0,
            "depth_score": 0.0, "fluency_score": 0.5,
            "order_bonus": 0.0, "has_wpm": False,
            "personality_nlp_sc": 1.0,
            "relevance_source": "none",
            "star_weight_used": W["star"],
            "kw_weight_used":   W["keyword"],
            "kw_defined":       False,
            "generated_ideal_answer": "",
            "ideal_answer_source":    "none",
            "question_type":   q_type_key,
            "weight_profile":  dict(W),
            # v12.0 time keys — zeroed when no answer given
            "time_score":        0.0,
            "time_label":        "N/A",
            "time_efficiency":   0.0,
            "time_verdict":      "N/A",
            "time_feedback":     "",
            "time_elapsed_s":    0.0,
            "time_ideal_window": "—",
            "time_modifier":     0.0,
        }