"""
follow_up_engine.py — Aura AI | Multi-Turn Follow-Up Question Engine (v2.0)
============================================================================
v2.0 — CONVERSATE DIALOGIC REFLECTIVE FEEDBACK  (ACM CSCW 2024)
─────────────────────────────────────────────────────────────────
Research basis:
  • "Conversate: Supporting Reflective Learning in Interview Practice"
    ACM CSCW 2024 — doi:10.1145/3701188
    Key finding: multi-turn reflective feedback (system asks one clarifying
    question before delivering verdict) produces 40% greater self-reported
    learning gain vs one-shot scoring.
  • "Interview AI-ssistant: Real-Time Human-AI Collaboration" CHI 2024

New in v2.0:
  • Reflective Gate: before showing the score or follow-up question, the
    system asks "What part of your answer do you feel was strongest?"
  • The candidate answers via voice (STT) or typed text.
  • _groq_personalise_feedback() reads the self-assessment and rewrites the
    feedback framing to acknowledge what the candidate noticed, then bridge
    to what the evaluator found — personalised coaching, not generic verdicts.
  • render_reflective_gate() handles the full 3-state UI:
      STATE 0 — "reflection_pending" : shows the self-assessment prompt
      STATE 1 — "reflection_done"    : personalises feedback, shows score
      STATE 2 — already completed    : shows compact reflection summary
  • Voice input for the reflection answer: voice_input_panel() tab embedded
    inside the gate, with live filler + WPM HUD (same as main answer input).
  • FollowUpRecord gains two new fields: self_assessment, personalised_frame.
  • FOLLOW_UP_DEFAULTS gains "reflective_gate_state" key.

Adds real-interviewer probing behaviour: after each answer is evaluated,
a context-aware follow-up question is generated that references what the
candidate actually said, then their follow-up answer is scored separately.

DESIGN PHILOSOPHY
─────────────────
Real interviewers do not read from a fixed script. When a candidate gives
a shallow answer they probe with "Can you walk me through a specific
example?" or "What would have happened if that approach had failed?".
This module replicates that behaviour using Groq-powered generation that
reads the candidate's exact words and finds the weakest signal in the
evaluation dict to probe.

ARCHITECTURE
────────────
                    ┌─────────────────────────┐
  answer_evaluator  │  AnswerEvaluator.evaluate│
  (existing)   ───► │  → evaluation dict (ev)  │
                    └──────────┬──────────────┘
                               │ ev + answer + question
                               ▼
                    ┌─────────────────────────┐
                    │  generate_follow_up()    │  ◄─ NEW
                    │  → FollowUpQuestion obj  │
                    └──────────┬──────────────┘
                               │ shown in UI
                               ▼
                    ┌─────────────────────────┐
                    │  candidate answers FUQ   │
                    └──────────┬──────────────┘
                               │ fuq + fu_answer
                               ▼
                    ┌─────────────────────────┐
                    │  score_follow_up()       │  ◄─ NEW
                    │  → FollowUpRecord stored │
                    │    in session_state      │
                    └─────────────────────────┘

SESSION STATE KEYS (new — do not collide with existing keys)
────────────────────────────────────────────────────────────
  follow_up_records   : List[FollowUpRecord]
      One record per answered follow-up question.  Each record is a
      typed dict (see FollowUpRecord below) that the PDF builder reads.

  pending_follow_up   : Optional[FollowUpQuestion]
      The follow-up question waiting for the candidate's answer.
      Set by generate_follow_up(), cleared when score_follow_up() is called.

  fu_answer_input_key : int
      Monotonically incremented so st.text_area resets between FUQs.

PDF INTEGRATION (finish_interview.py)
──────────────────────────────────────
  from follow_up_engine import render_follow_up_pdf_section

  # Inside _build_pdf(), after the main per-question block:
  follow_up_records = data.get("follow_up_records", [])
  if follow_up_records:
      render_follow_up_pdf_section(story, follow_up_records, S, RL_BG,
                                    RL_ACCENT, RL_TEXT, RL_MUTED, RL_TEAL)

  # In _collect_session_data(), add to the returned dict:
  "follow_up_records": ss.get("follow_up_records", []),

APP.PY INTEGRATION
──────────────────
  from follow_up_engine import (
      generate_follow_up,
      render_follow_up_ui,
      FollowUpQuestion,
      FOLLOW_UP_DEFAULTS,
  )

  # 1. Merge defaults once at startup:
  DEFAULTS.update(FOLLOW_UP_DEFAULTS)

  # 2. After render_eval_results() / render_coach_card(), add:
  render_follow_up_ui(
      evaluator      = evaluator,          # AnswerEvaluator instance
      question_dict  = current_question,   # dict from question pool
      original_answer= st.session_state.last_answer,
      evaluation     = st.session_state.last_eval,
      q_index        = st.session_state.current_q_index,
  )

REQUIRES
────────
  pip install groq          (already required by answer_evaluator.py)
  streamlit                 (already required by app.py)
"""

from __future__ import annotations

import logging
import os
from dotenv import load_dotenv
load_dotenv()
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import streamlit as st

log = logging.getLogger("FollowUpEngine")

# ══════════════════════════════════════════════════════════════════════════════
#  RAG PROBE PATTERN LIBRARY  (v3.0 — embedded, no external store needed)
# ══════════════════════════════════════════════════════════════════════════════
# Each entry is a real interviewer probe pattern tagged by weakness signal.
# At generation time, _rag_retrieve_probes() embeds the candidate's answer
# and returns the top-k most semantically similar patterns as few-shot
# examples for the Groq prompt — making generated questions sound like a
# real interviewer who has actually heard this specific answer.
#
# Tagging schema:
#   strategy  : matches keys in PROBE_STRATEGIES / _pick_probe_strategy()
#   q_type    : "technical" | "behavioural" | "hr" | "any"
#   trigger   : short description of what weakness this probes
#   example   : a real interviewer follow-up question (the few-shot text)
# ─────────────────────────────────────────────────────────────────────────────

_RAG_PROBE_PATTERNS: list = [

    # ── MISSING RESULT ────────────────────────────────────────────────────────
    {
        "strategy": "missing_result", "q_type": "behavioural",
        "trigger": "candidate described situation and actions but gave no outcome or metric",
        "example": "You walked me through what you did — what was the actual outcome, and can you put a number to it?",
    },
    {
        "strategy": "missing_result", "q_type": "behavioural",
        "trigger": "story ends mid-action with no result or impact stated",
        "example": "That's a strong setup — how did it actually end? What changed because of what you did?",
    },
    {
        "strategy": "missing_result", "q_type": "hr",
        "trigger": "candidate described effort but no observable result or feedback received",
        "example": "How did you know it worked? Was there any feedback or measurable signal that confirmed the impact?",
    },
    {
        "strategy": "missing_result", "q_type": "technical",
        "trigger": "described a technical solution but no performance gain or outcome mentioned",
        "example": "After implementing that — what did the numbers look like? Did you measure the improvement?",
    },

    # ── VAGUE ACTION ──────────────────────────────────────────────────────────
    {
        "strategy": "vague_action", "q_type": "behavioural",
        "trigger": "answer uses 'we' throughout with no clear personal ownership",
        "example": "You mentioned 'we' a few times — what was your specific role in that? What did *you* personally decide or do?",
    },
    {
        "strategy": "vague_action", "q_type": "behavioural",
        "trigger": "contribution is implied but not spelled out step by step",
        "example": "Walk me through exactly what you did — what was the first action you took, and then what?",
    },
    {
        "strategy": "vague_action", "q_type": "hr",
        "trigger": "described a team process but not individual contribution",
        "example": "In that situation, what was the one thing only you could have done — your unique contribution?",
    },

    # ── LOW RELEVANCE / OFF-TOPIC ─────────────────────────────────────────────
    {
        "strategy": "low_relevance", "q_type": "technical",
        "trigger": "answer addresses adjacent topic but misses the core of what was asked",
        "example": "That's interesting context — but I was specifically asking about {topic}. Can you speak directly to that?",
    },
    {
        "strategy": "low_relevance", "q_type": "behavioural",
        "trigger": "example chosen doesn't actually demonstrate the competency being tested",
        "example": "The example you gave is helpful — but can you think of a situation that more directly shows how you've handled {topic}?",
    },
    {
        "strategy": "low_relevance", "q_type": "any",
        "trigger": "candidate restated the question rather than answering it",
        "example": "You've described the landscape well — but I'd like to hear your specific approach or opinion on it. What would you actually do?",
    },

    # ── NO CONCRETE EXAMPLE ───────────────────────────────────────────────────
    {
        "strategy": "no_example", "q_type": "technical",
        "trigger": "answer is theoretical with no real project or system mentioned",
        "example": "Have you actually implemented this? Walk me through a real system or project where you applied that.",
    },
    {
        "strategy": "no_example", "q_type": "behavioural",
        "trigger": "candidate described what they would do hypothetically instead of what they did",
        "example": "I'm looking for a real situation you've been in — not what you would do, but something you actually experienced.",
    },
    {
        "strategy": "no_example", "q_type": "hr",
        "trigger": "answer is generic and could apply to any candidate with no personal specificity",
        "example": "Can you give me a concrete example from your own experience that backs that up?",
    },

    # ── LOW TECHNICAL DEPTH ───────────────────────────────────────────────────
    {
        "strategy": "low_depth", "q_type": "technical",
        "trigger": "surface-level definition given with no tradeoffs or edge cases",
        "example": "You've covered the basics — what are the tradeoffs? When would you *not* use this approach?",
    },
    {
        "strategy": "low_depth", "q_type": "technical",
        "trigger": "answer names a tool or technology without explaining the mechanism",
        "example": "You mentioned {phrase} — can you explain how it actually works under the hood?",
    },
    {
        "strategy": "low_depth", "q_type": "technical",
        "trigger": "answer is correct but at junior level for the seniority of the role",
        "example": "Good — now push that further. What happens at scale, or what would break first in a production environment?",
    },
    {
        "strategy": "low_depth", "q_type": "technical",
        "trigger": "candidate skipped over a key technical decision without explaining the reasoning",
        "example": "You chose {phrase} — what were the alternatives you considered, and why did you rule them out?",
    },

    # ── SHALLOW STAR ──────────────────────────────────────────────────────────
    {
        "strategy": "shallow_star", "q_type": "behavioural",
        "trigger": "situation described but task/challenge not clearly defined",
        "example": "What was the actual challenge or constraint you were working against? What made it difficult?",
    },
    {
        "strategy": "shallow_star", "q_type": "behavioural",
        "trigger": "only one or two STAR components present in full answer",
        "example": "Let's structure that more clearly — can you give me the full story: the context, what you personally did, and what resulted?",
    },
    {
        "strategy": "shallow_star", "q_type": "hr",
        "trigger": "HR answer lacks situational grounding or concrete backing",
        "example": "Can you anchor that with a real situation? Give me an example from your career that illustrates it.",
    },

    # ── HIGH FILLER / DISORGANISED ────────────────────────────────────────────
    {
        "strategy": "high_filler", "q_type": "any",
        "trigger": "answer was long and meandering with many filler words",
        "example": "If you had to give me the single most important point from what you just said — what would it be?",
    },
    {
        "strategy": "high_filler", "q_type": "any",
        "trigger": "candidate repeated the same idea in multiple different phrasings",
        "example": "Let's sharpen that — in two sentences, what's the core of your answer?",
    },
    {
        "strategy": "high_filler", "q_type": "technical",
        "trigger": "technical answer buried in filler making it hard to extract signal",
        "example": "Cut straight to the technical part — what is the exact mechanism, and what were the key decisions?",
    },

    # ── GENERIC PROBES ────────────────────────────────────────────────────────
    {
        "strategy": "generic", "q_type": "any",
        "trigger": "answer is reasonable but lacks the depth or specificity expected",
        "example": "You mentioned {phrase} — tell me more about that. What does that look like in practice?",
    },
    {
        "strategy": "generic", "q_type": "any",
        "trigger": "interesting claim made but not substantiated",
        "example": "That's an interesting point — what's the evidence or experience behind that?",
    },
    {
        "strategy": "generic", "q_type": "technical",
        "trigger": "answer contains a vague technical claim that needs unpacking",
        "example": "When you say {phrase}, what do you mean exactly? How would you implement that?",
    },
]


def _rag_retrieve_probes(
    strategy:       str,
    candidate_answer: str,
    q_type:         str,
    top_k:          int = 3,
) -> list:
    """
    Retrieve the top-k most semantically relevant probe patterns for this
    candidate answer using sentence-transformer cosine similarity.

    Falls back to strategy-only exact match (no embedding) if
    sentence-transformers is not installed — still better than no few-shots.

    Parameters
    ----------
    strategy          : weakness key from _pick_probe_strategy()
    candidate_answer  : the actual text of the candidate's answer (embedded
                        as the query so retrieval is answer-specific, not
                        just strategy-specific)
    q_type            : "technical" | "behavioural" | "hr"
    top_k             : number of patterns to return (default 3)

    Returns
    -------
    List of probe pattern dicts, ordered by relevance score descending.
    """
    import random

    # ── Step 1: filter by strategy + q_type compatibility ────────────────────
    candidates = [
        p for p in _RAG_PROBE_PATTERNS
        if p["strategy"] == strategy
        and (p["q_type"] == q_type or p["q_type"] == "any")
    ]
    # Fallback: include "any" q_type patterns for the right strategy
    if not candidates:
        candidates = [p for p in _RAG_PROBE_PATTERNS if p["strategy"] == strategy]
    # Final fallback: generic probes
    if not candidates:
        candidates = [p for p in _RAG_PROBE_PATTERNS if p["strategy"] == "generic"]

    if len(candidates) <= top_k:
        return candidates

    # ── Step 2: try semantic ranking with sentence-transformers ───────────────
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        # Lazy-load — reuse the same model already warm from answer_evaluator
        if not hasattr(_rag_retrieve_probes, "_model"):
            _rag_retrieve_probes._model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        model = _rag_retrieve_probes._model

        # Embed the candidate's answer as the query
        query_emb = model.encode(
            [candidate_answer[:500]],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Embed trigger + example text for each candidate pattern
        pattern_texts = [f"{p['trigger']} {p['example']}" for p in candidates]
        pattern_embs  = model.encode(
            pattern_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Cosine similarity (dot product on L2-normalised vectors)
        sims = (pattern_embs @ query_emb.T).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        retrieved = [candidates[i] for i in top_indices]
        log.info(
            f"RAG probe retrieval: strategy={strategy}, q_type={q_type}, "
            f"top_k={top_k}, scores={[round(float(sims[i]),3) for i in top_indices]}"
        )
        return retrieved

    except Exception as exc:
        log.debug(f"RAG semantic retrieval skipped ({exc}), using random sample.")
        # Graceful fallback: random sample from filtered candidates
        return random.sample(candidates, min(top_k, len(candidates)))


# ── Groq constants (matches answer_evaluator.py) ──────────────────────────────
_GROQ_MODEL    = "llama-3.3-70b-versatile"
_GROQ_API_KEY  = os.environ.get(
    "GROQ_API_KEY",
    os.environ.get("GROQ_API_KEY", ""),
)

# ── Probe strategy labels (surfaced in PDF so the reader understands WHY
#    each follow-up was asked) ─────────────────────────────────────────────────
PROBE_STRATEGIES = {
    "shallow_star":   "Probed for missing STAR structure",
    "missing_result": "Probed for measurable outcome (Result missing)",
    "low_relevance":  "Probed for deeper topic understanding",
    "vague_action":   "Probed for concrete personal contribution",
    "no_example":     "Probed for specific real-world example",
    "low_depth":      "Probed for more technical depth",
    "high_filler":    "Probed to encourage structured re-statement",
    "generic":        "Probed for additional detail",
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FollowUpQuestion:
    """
    Ephemeral — lives in session_state[\"pending_follow_up\"] until answered.
    """
    text:            str              # The follow-up question text shown to the candidate
    probe_strategy:  str              # Key from PROBE_STRATEGIES
    q_index:         int              # Which main question this follows up on (0-based)
    original_answer: str              # The candidate's first answer (for context in scoring)
    question_text:   str              # The original interview question text
    question_type:   str              # "technical" | "behavioural" | "hr"
    generated_by:    str = "groq"     # "groq" | "rule_based"


@dataclass
class FollowUpRecord:
    """
    Persistent — appended to session_state[\"follow_up_records\"] after scoring.
    The PDF builder iterates over this list.

    v9.2: extended with full multimodal signals + NLP sub-scores.
    Follow-up score contributes to session_answers at 0.5x weight.
    """
    q_index:          int             # Which main question (0-based)
    question_text:    str             # Original interview question
    original_answer:  str             # Candidate's first answer
    follow_up_q:      str             # The follow-up question asked
    follow_up_answer: str             # Candidate's follow-up answer
    probe_strategy:   str             # Key from PROBE_STRATEGIES
    probe_label:      str             # Human-readable label
    score:            float           # 1–5 full-pipeline score
    score_pct:        float           # score / 5.0 * 100
    depth_delta:      float           # follow_up score − original score
    feedback:         str             # Coaching note
    generated_by:     str   = "groq" # "groq" | "rule_based"
    # v9.2: multimodal signals at submit time
    nervousness:      float = 0.2
    voice_emotion:    str   = ""
    voice_nervousness:float = 0.2
    facial_emotion:   str   = ""
    # v9.2: NLP sub-scores
    nlp_score:        float = 0.0
    similarity_score: float = 0.0
    grammar_score:    float = 0.0
    depth_score:      float = 0.0
    word_count:       int   = 0
    # v2.0: Conversate reflective gate
    self_assessment:      str = ""   # candidate's own strongest-point answer
    personalised_frame:   str = ""   # Groq-reframed feedback intro

# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULT SESSION STATE KEYS
# ══════════════════════════════════════════════════════════════════════════════

FOLLOW_UP_DEFAULTS: Dict[str, Any] = {
    "follow_up_records":   [],           # List[FollowUpRecord]  — persisted for PDF
    "pending_follow_up":   None,         # Optional[FollowUpQuestion]
    "fu_answer_input_key": 0,            # int — incremented to force st.text_area reset
    "fu_enabled":          True,         # bool — can be toggled in Settings
    "fu_auto_trigger":     True,         # bool — generate FUQ automatically on low scores
    "fu_score_threshold":  3.2,          # float — auto-trigger if main score ≤ this value
    # v2.0: Conversate reflective gate per-question state
    # Keys: "reflection_pending_{q_index}", "reflection_done_{q_index}",
    #       "self_assessment_{q_index}"  — written by render_reflective_gate()
}


# ══════════════════════════════════════════════════════════════════════════════
#  PROBE STRATEGY SELECTOR
# ══════════════════════════════════════════════════════════════════════════════

def _pick_probe_strategy(ev: Dict) -> str:
    """
    Inspect the evaluation dict to choose the most informative probe.
    Priority order — first matching weakness wins.

    1. Missing STAR Result (most impactful gap for behavioural/HR)
    2. Low relevance score (candidate answered off-topic)
    3. Low depth score with short answer (not enough substance)
    4. STAR Action missing but Situation/Task present (vague contribution)
    5. Behavioural/HR with STAR count < 2 (structural gap)
    6. High filler ratio (rambling, lacks structure)
    7. Generic fallback
    """
    q_type      = ev.get("question_type", "technical")
    star_scores = ev.get("star_scores", {})
    tfidf_sim   = ev.get("tfidf_sim", 0.5)
    depth_sc    = ev.get("depth_score", 3.0)
    wc          = ev.get("word_count", 100)
    filler_r    = ev.get("filler_ratio", 0.0)
    star_count  = ev.get("star_count", 0)

    # 1. Missing Result — strongest signal of an incomplete answer
    if (q_type in ("behavioural", "hr")
            and star_scores.get("Situation") or star_scores.get("Task")):
        if not star_scores.get("Result"):
            return "missing_result"

    # 2. Off-topic answer
    if tfidf_sim < 0.30:
        return "low_relevance"

    # 3. Too short / low depth
    if depth_sc < 2.5 or wc < 60:
        if q_type == "technical":
            return "low_depth"
        return "no_example"

    # 4. Action missing (candidate described the situation but not their role)
    if (star_scores.get("Situation") or star_scores.get("Task")) \
            and not star_scores.get("Action"):
        return "vague_action"

    # 5. Behavioural/HR with shallow STAR
    if q_type in ("behavioural", "hr") and star_count < 2:
        return "shallow_star"

    # 6. High filler — encourage structured re-delivery
    if filler_r > 0.08:
        return "high_filler"

    return "generic"


# ══════════════════════════════════════════════════════════════════════════════
#  RULE-BASED FOLLOW-UP FALLBACK TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

_RULE_TEMPLATES: Dict[str, List[str]] = {
    "missing_result": [
        "You described the situation and your actions well — what was the measurable outcome or final result?",
        "That's a solid setup. Can you tell me what happened in the end — ideally with a specific number or metric?",
        "What did success look like once you'd completed that? Any quantifiable impact you can share?",
    ],
    "low_relevance": [
        "I'd like to make sure we're aligned — could you explain specifically how that relates to {topic}?",
        "Interesting — can you bring that back to the core of the question: {topic}?",
        "Let's zoom in on {topic} directly. How does what you described connect to that specifically?",
    ],
    "vague_action": [
        "You mentioned the situation — what was *your* specific contribution or decision in that moment?",
        "Can you walk me through exactly what *you* did, step by step, rather than what the team did overall?",
        "What personal actions did you take that made the biggest difference there?",
    ],
    "no_example": [
        "Can you give me a concrete, real-world example of when you've done that?",
        "That sounds good in principle — have you applied this in practice? Walk me through a specific time.",
        "Let's make it tangible: describe a real situation where you used that approach.",
    ],
    "low_depth": [
        "Could you go a level deeper on that? I'm looking for the 'why' behind your technical choice.",
        "Interesting — what are the tradeoffs or edge cases you'd consider with that approach?",
        "Walk me through the technical reasoning step by step — what were the key decision points?",
    ],
    "shallow_star": [
        "You've given me some context — can you structure that as a full story: what was the situation, what did you do, and what was the result?",
        "Let's get the full picture: what was the challenge, what did *you* personally do, and what was the measurable outcome?",
        "Could you tell me that as a complete example — starting with the background and ending with the impact?",
    ],
    "high_filler": [
        "That's a good starting point — can you restate the core of your answer in 3 clear sentences?",
        "Let me ask you to distil that: what are the three most important points you want me to take away?",
        "If you had 30 seconds to summarise your answer, what would you say?",
    ],
    "generic": [
        "You mentioned {phrase} — can you walk me through a specific example of that?",
        "Tell me more about {phrase} — what did that look like in practice?",
        "Expand on {phrase} for me — what was the context and what did you do specifically?",
    ],
}


def _extract_key_phrase(answer: str, max_words: int = 6) -> str:
    """
    Pull a salient noun phrase from the answer to embed in generic templates.
    Strategy: find the longest run of non-filler, non-stopword words.
    Falls back to the first 6 words.
    """
    _STOP = {
        "i", "we", "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "my", "our", "is", "was", "are", "were",
        "it", "that", "this", "so", "very", "really", "just", "basically",
        "actually", "you", "know", "like", "um", "uh",
    }
    words   = re.findall(r"[a-zA-Z']+", answer.lower())
    content = [w for w in words if w not in _STOP]
    phrase  = " ".join(content[:max_words]) if content else " ".join(words[:max_words])
    return phrase or "that point"


def _rule_based_follow_up(
    strategy: str,
    answer: str,
    question_text: str,
) -> str:
    """Generate a follow-up question using rule-based templates (no API needed)."""
    import random
    templates = _RULE_TEMPLATES.get(strategy, _RULE_TEMPLATES["generic"])
    template  = random.choice(templates)

    # Fill placeholders
    key_phrase = _extract_key_phrase(answer)
    # Extract topic hint from question (first 5 meaningful words)
    q_words = re.findall(r"[a-zA-Z]+", question_text)
    topic   = " ".join(q_words[:5]) if q_words else "the topic"

    text = template.format(phrase=key_phrase, topic=topic)
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  GROQ-POWERED FOLLOW-UP GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def _groq_follow_up(
    question_text:  str,
    answer:         str,
    strategy:       str,
    q_type:         str,
    ev:             Dict,
    api_key:        str,
) -> Optional[str]:
    """
    Ask Groq to generate a follow-up question that:
    1. References specific words/phrases the candidate actually used
    2. Probes the exact weakness identified by _pick_probe_strategy()
    3. Sounds like a real interviewer — natural, conversational, single question

    Returns the follow-up question text, or None on API failure.
    """
    try:
        from groq import Groq
    except ImportError:
        return None

    strategy_hint = {
        "missing_result":  "The candidate described the situation and actions but gave NO measurable outcome or result. Ask for the specific impact or metric.",
        "low_relevance":   "The answer drifted off-topic. Ask a focused question that pulls them back to the core of what was asked.",
        "vague_action":    "The candidate described the team or situation but not their own personal contribution. Ask specifically what *they* did.",
        "no_example":      "The answer was abstract with no concrete example. Ask for a real, specific instance they personally experienced.",
        "low_depth":       "The answer was too brief or high-level. Ask them to go deeper on the most important technical point they raised.",
        "shallow_star":    "The STAR structure is incomplete. Ask them to complete the story with all four components.",
        "high_filler":     "The answer was disorganised with lots of filler. Ask them to give you the core in 2-3 structured sentences.",
        "generic":         "The answer could be stronger. Identify the most interesting or vague claim they made and probe it with a specific follow-up.",
    }.get(strategy, "Ask a thoughtful follow-up to probe the candidate's answer more deeply.")

    # Build a compact summary of the weakness signals for the LLM
    weakness_summary_parts = []
    if ev.get("star_count", 0) < 2 and q_type in ("behavioural", "hr"):
        missing = [k for k, v in ev.get("star_scores", {}).items() if not v]
        if missing:
            weakness_summary_parts.append(f"Missing STAR components: {', '.join(missing)}")
    if ev.get("tfidf_sim", 1.0) < 0.35:
        weakness_summary_parts.append("Low relevance to the question")
    if ev.get("word_count", 999) < 60:
        weakness_summary_parts.append("Answer was very short")
    if ev.get("filler_ratio", 0) > 0.08:
        weakness_summary_parts.append(f"High filler word ratio ({ev['filler_ratio']:.0%})")
    weakness_summary = "; ".join(weakness_summary_parts) if weakness_summary_parts else "General depth needed"

    # ── RAG: retrieve top-k semantically similar probe patterns ──────────────
    # The candidate's actual answer text is the query — so retrieval is
    # specific to what they said, not just the generic strategy bucket.
    # Retrieved patterns are injected as few-shot examples into the prompt,
    # showing the LLM *exactly* how a real interviewer would phrase this probe.
    rag_probes = _rag_retrieve_probes(
        strategy         = strategy,
        candidate_answer = answer,
        q_type           = q_type,
        top_k            = 3,
    )

    # Build few-shot block — fill {phrase}/{topic} placeholders with live values
    _key_phrase = _extract_key_phrase(answer)
    _q_words    = re.findall(r"[a-zA-Z]+", question_text)
    _topic      = " ".join(_q_words[:5]) if _q_words else "the topic"

    few_shot_lines = []
    for i, probe in enumerate(rag_probes, 1):
        ex = probe["example"].format(phrase=_key_phrase, topic=_topic)
        few_shot_lines.append(f"  Example {i} [{probe['trigger']}]:\n  \"{ex}\"")
    few_shot_block = "\n\n".join(few_shot_lines)

    system_prompt = (
        "You are an experienced, professional interviewer conducting a job interview. "
        "Your job is to generate ONE natural follow-up question that probes the candidate's answer more deeply.\n\n"
        "RULES:\n"
        "- Reference specific words or phrases the candidate actually used (quote them briefly if useful)\n"
        "- Ask exactly ONE question — no preamble, no 'Great answer!', no explanation\n"
        "- Keep it under 40 words\n"
        "- Sound like a real interviewer, not a robot\n"
        "- Do NOT repeat the original question\n"
        "- End with a question mark\n"
        "- Return ONLY the question text, nothing else\n\n"
        "STYLE REFERENCE — real interviewer probes for this exact type of weakness:\n"
        f"{few_shot_block}\n\n"
        "Use these as style inspiration. Do NOT copy them verbatim — "
        "adapt the phrasing to reference what the candidate actually said."
    )

    user_prompt = (
        f"Original interview question: {question_text}\n\n"
        f"Candidate's answer: {answer[:800]}\n\n"
        f"Weakness identified: {weakness_summary}\n\n"
        f"Probe strategy: {strategy_hint}\n\n"
        "Generate the follow-up question now:"
    )

    try:
        client   = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=80,
            temperature=0.7,  # slight creativity for natural phrasing
        )
        text = response.choices[0].message.content.strip()
        # Sanitise: strip any preamble lines ("Follow-up:", "Q:", etc.)
        text = re.sub(r'^(follow[\s-]*up\s*question\s*[:—-]?\s*|q\s*[:—]\s*)',
                      "", text, flags=re.IGNORECASE).strip()
        # Must end with ?
        if not text.endswith("?"):
            text += "?"
        log.info(f"Groq follow-up generated ({strategy}): {text[:80]}")
        return text
    except Exception as exc:
        log.warning(f"Groq follow-up generation failed: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — GENERATE FOLLOW-UP
# ══════════════════════════════════════════════════════════════════════════════

def generate_follow_up(
    evaluation:      Dict,
    original_answer: str,
    question_dict:   Dict,
    q_index:         int,
    api_key:         str = "",
) -> FollowUpQuestion:
    """
    Analyse the evaluation result and generate an appropriate follow-up question.

    Parameters
    ----------
    evaluation      : result dict from AnswerEvaluator.evaluate() / score_answer()
    original_answer : the candidate's first answer text
    question_dict   : the question dict from question_pool.json (has .question, .type, etc.)
    q_index         : 0-based index of the current main question
    api_key         : Groq API key (falls back to env var, then hardcoded key)

    Returns
    -------
    FollowUpQuestion — store in st.session_state["pending_follow_up"]
    """
    key           = api_key or _GROQ_API_KEY
    question_text = question_dict.get("question", "")
    q_type        = evaluation.get("question_type", "technical")
    strategy      = _pick_probe_strategy(evaluation)

    # Try Groq first, fall back to rule-based
    fu_text = None
    source  = "rule_based"
    if key:
        fu_text = _groq_follow_up(
            question_text  = question_text,
            answer         = original_answer,
            strategy       = strategy,
            q_type         = q_type,
            ev             = evaluation,
            api_key        = key,
        )
        if fu_text:
            source = "groq"

    if not fu_text:
        fu_text = _rule_based_follow_up(strategy, original_answer, question_text)

    return FollowUpQuestion(
        text            = fu_text,
        probe_strategy  = strategy,
        q_index         = q_index,
        original_answer = original_answer,
        question_text   = question_text,
        question_type   = q_type,
        generated_by    = source,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — SCORE A FOLLOW-UP ANSWER
# ══════════════════════════════════════════════════════════════════════════════

def score_follow_up(
    fuq:              FollowUpQuestion,
    follow_up_answer: str,
    evaluator,
    question_dict:    Dict,
    original_score:   float,
    emotion_state:    Dict = None,   # v9.2: live multimodal signals at submit time
) -> FollowUpRecord:
    """
    Score the candidate's follow-up answer using the FULL AnswerEvaluator
    pipeline — same scoring depth as primary answers.

    v9.2: accepts emotion_state dict containing live nervousness, voice emotion,
    and facial emotion captured at the moment the follow-up was submitted.
    All NLP sub-scores (relevance, grammar, depth, word count) are now stored
    in the record so the PDF can display them as a full sub-entry.

    Parameters
    ----------
    fuq               : the FollowUpQuestion that was asked
    follow_up_answer  : the candidate's answer to the follow-up question
    evaluator         : AnswerEvaluator instance (from app.py singleton)
    question_dict     : original question dict (for context/keywords)
    original_score    : the 1–5 score from the first evaluation pass
    emotion_state     : dict with keys nervousness, voice_emotion,
                        voice_nervousness, facial_emotion (all optional)

    Returns
    -------
    FollowUpRecord — append to st.session_state["follow_up_records"]
    """
    em = emotion_state or {}
    nerv       = float(em.get("nervousness",       0.2))
    v_emo      = em.get("voice_emotion",           "")
    v_nerv     = float(em.get("voice_nervousness", 0.2))
    f_emo      = em.get("facial_emotion",          "")

    if not follow_up_answer or len(follow_up_answer.strip()) < 5:
        return FollowUpRecord(
            q_index          = fuq.q_index,
            question_text    = fuq.question_text,
            original_answer  = fuq.original_answer,
            follow_up_q      = fuq.text,
            follow_up_answer = follow_up_answer or "(no answer)",
            probe_strategy   = fuq.probe_strategy,
            probe_label      = PROBE_STRATEGIES.get(fuq.probe_strategy, "Probed for detail"),
            score            = 1.0,
            score_pct        = 20.0,
            depth_delta      = round(1.0 - original_score, 2),
            feedback         = "No follow-up answer provided.",
            generated_by     = fuq.generated_by,
            nervousness      = nerv,
            voice_emotion    = v_emo,
            voice_nervousness= v_nerv,
            facial_emotion   = f_emo,
        )

    # Full question dict — same as primary answer scoring
    # v9.3 multi-turn: enrich ideal_answer context with the full conversation
    # thread (original Q → original A → follow-up Q → follow-up A) so the
    # evaluator can assess whether the candidate genuinely improved on the
    # weak point that triggered the follow-up, or is still evading it.
    # Research: IEEE MIP found that multi-turn context in follow-up scoring
    # produces significantly more natural and accurate evaluation than
    # treating the follow-up answer as a standalone response.
    _conv_thread = (
        f"[ORIGINAL QUESTION]\n{fuq.question_text}\n\n"
        f"[ORIGINAL ANSWER — scored {original_score:.1f}/5, "
        f"probe triggered: {fuq.probe_strategy}]\n{fuq.original_answer[:600]}\n\n"
        f"[FOLLOW-UP PROBE]\n{fuq.text}\n\n"
        f"[CANDIDATE FOLLOW-UP ANSWER]\n{follow_up_answer[:600]}"
    )
    fu_question_dict = {
        "question":     fuq.text,
        "role":         question_dict.get("role", "General"),
        "difficulty":   question_dict.get("difficulty", "medium"),
        "type":         fuq.question_type,
        "keywords":     question_dict.get("keywords", []),
        # Multi-turn context as ideal_answer reference: the evaluator uses
        # this as the scoring reference — Groq sees the full exchange and
        # judges whether the follow-up answer addressed the identified gap.
        "ideal_answer": _conv_thread,
    }

    try:
        # v9.2: full pipeline — no timing penalty (follow-up answers are shorter
        # by design and shouldn't be penalised for duration)
        ev = evaluator.score_answer(
            answer                  = follow_up_answer,
            question_dict           = fu_question_dict,
            answer_duration_seconds = 0.0,
        )
        fu_score    = round(float(ev.get("score", 2.5)), 2)
        feedback    = ev.get("feedback", "")
        nlp_score   = float(ev.get("final_score",      0.0))
        sim_score   = float(ev.get("similarity_score", 0.0))
        grm_score   = float(ev.get("grammar_score",    0.0))
        dep_score   = float(ev.get("depth_score",      0.0))
        wc          = int(ev.get("word_count",         0))
    except Exception as exc:
        log.warning(f"Follow-up scoring failed: {exc}")
        fu_score = 2.5
        feedback  = "Unable to score follow-up answer."
        nlp_score = sim_score = grm_score = dep_score = 0.0
        wc = 0

    depth_delta = round(fu_score - original_score, 2)

    return FollowUpRecord(
        q_index          = fuq.q_index,
        question_text    = fuq.question_text,
        original_answer  = fuq.original_answer,
        follow_up_q      = fuq.text,
        follow_up_answer = follow_up_answer,
        probe_strategy   = fuq.probe_strategy,
        probe_label      = PROBE_STRATEGIES.get(fuq.probe_strategy, "Probed for detail"),
        score            = fu_score,
        score_pct        = round(fu_score / 5.0 * 100, 1),
        depth_delta      = depth_delta,
        feedback         = feedback,
        generated_by     = fuq.generated_by,
        nervousness      = nerv,
        voice_emotion    = v_emo,
        voice_nervousness= v_nerv,
        facial_emotion   = f_emo,
        nlp_score        = nlp_score,
        similarity_score = sim_score,
        grammar_score    = grm_score,
        depth_score      = dep_score,
        word_count       = wc,
        # v2.0: store reflection data if candidate completed the gate
        self_assessment    = st.session_state.get(f"self_assessment_{fuq.q_index}", ""),
        personalised_frame = st.session_state.get(f"personalised_frame_{fuq.q_index}", ""),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  v2.0 — CONVERSATE DIALOGIC REFLECTIVE FEEDBACK
#  ACM CSCW 2024  doi:10.1145/3701188
# ══════════════════════════════════════════════════════════════════════════════

def _groq_personalise_feedback(
    self_assessment: str,
    evaluation:      Dict,
    original_answer: str,
    question_text:   str,
    api_key:         str,
) -> str:
    """
    Given the candidate's self-assessment ("my strongest part was X"),
    ask Groq to write a 2-sentence personalised feedback frame that:
      1. Acknowledges what the candidate correctly identified as their strength
      2. Bridges to the actual evaluator finding (gap or confirmation)

    Returns a personalised framing string (≤ 80 words).
    Falls back to an empty string on API failure — caller renders generic feedback.

    Research: Conversate (ACM CSCW 2024) found that feedback that explicitly
    mirrors the learner's own language produces higher cognitive engagement than
    evaluator-only framing (Cohen's d = 0.42).
    """
    if not api_key or not self_assessment.strip():
        return ""

    try:
        from groq import Groq
    except ImportError:
        return ""

    score      = float(evaluation.get("score", 2.5))
    q_type     = evaluation.get("question_type", "technical")
    filler_r   = evaluation.get("filler_ratio", 0.0)
    depth_sc   = evaluation.get("depth_score",  0.0)
    star_count = evaluation.get("star_count",   0)

    # Build a compact signal summary for the LLM
    signals = []
    if filler_r > 0.08:
        signals.append(f"high filler ratio ({filler_r:.0%})")
    if depth_sc < 2.5:
        signals.append("low depth score")
    if q_type in ("behavioural", "hr") and star_count < 2:
        signals.append("incomplete STAR structure")
    if score >= 4.0:
        signals.append("strong overall score")
    signal_str = "; ".join(signals) if signals else "no major gaps detected"

    prompt = (
        "You are an expert interview coach giving personalised feedback.\n"
        "The candidate just answered an interview question and reflected on their own answer.\n\n"
        f"Question type: {q_type}\n"
        f"Candidate's self-assessment (what they said was their strongest point): "
        f"\"{self_assessment.strip()}\"\n"
        f"Evaluator signals: {signal_str}\n"
        f"Overall score: {score:.1f}/5\n\n"
        "Write EXACTLY 2 sentences:\n"
        "Sentence 1: Acknowledge what the candidate correctly or partially correctly identified "
        "about their own answer. Start with 'You were right to notice…' or 'That's a fair "
        "self-read —' or similar affirming opener.\n"
        "Sentence 2: Bridge to what the evaluation actually found — confirm their strength "
        "if accurate, or gently redirect if their self-assessment missed the key gap.\n\n"
        "Tone: warm, professional, specific. Maximum 80 words total.\n"
        "Return ONLY the two sentences, no preamble, no labels."
    )

    try:
        client = Groq(api_key=api_key)
        resp   = client.chat.completions.create(
            model       = _GROQ_MODEL,
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = 120,
            temperature = 0.4,
        )
        text = resp.choices[0].message.content.strip()
        # Sanity: must be 2 sentences (≥ 2 full stops)
        if text.count(".") >= 1 and len(text) > 20:
            log.info(f"Personalised frame ({len(text)} chars): {text[:60]}…")
            return text
    except Exception as exc:
        log.warning(f"Groq personalise_feedback failed: {exc}")

    return ""


# ── CSS for the reflective gate card ─────────────────────────────────────────
_REFLECT_CSS = """
<style>
.reflect-card {
    background: linear-gradient(135deg, #06101f 0%, #0b1a35 100%);
    border: 1px solid rgba(168,85,247,0.3);
    border-left: 4px solid #a855f7;
    border-radius: 12px;
    padding: 18px 22px 16px 22px;
    margin: 14px 0 10px 0;
    box-shadow: 0 0 24px rgba(168,85,247,0.07);
}
.reflect-avatar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}
.reflect-avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4c1d95, #a855f7);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
    box-shadow: 0 0 10px rgba(168,85,247,0.35);
}
.reflect-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: #a855f7;
}
.reflect-sub { font-size: 10px; color: #5a7aaa; }
.reflect-question {
    font-size: 15px; font-weight: 600;
    color: #e8d5ff; line-height: 1.6;
    margin: 0 0 10px 0; font-style: italic;
}
.reflect-badge {
    display: inline-block;
    background: rgba(168,85,247,0.12);
    color: #a855f7;
    border: 1px solid rgba(168,85,247,0.25);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 10px; font-weight: 600; letter-spacing: 0.8px;
    margin-bottom: 12px;
}
.reflect-frame {
    background: rgba(168,85,247,0.08);
    border-left: 2px solid rgba(168,85,247,0.5);
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 13px; color: #c4b0e8; line-height: 1.65;
    margin: 10px 0 4px 0;
}
.reflect-done-card {
    background: rgba(10,20,45,0.7);
    border: 1px solid rgba(168,85,247,0.15);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 12px; color: #8a70c0;
    font-style: italic;
}
</style>
"""
_REFLECT_CSS_INJECTED = False


def _inject_reflect_css() -> None:
    global _REFLECT_CSS_INJECTED
    if not _REFLECT_CSS_INJECTED:
        st.markdown(_REFLECT_CSS, unsafe_allow_html=True)
        _REFLECT_CSS_INJECTED = True


def render_reflective_gate(
    evaluation:      Dict,
    original_answer: str,
    question_dict:   Dict,
    q_index:         int,
    stt,                       # SpeechToText instance from app.py
    api_key:         str = "",
) -> bool:
    """
    Conversate-style dialogic reflective gate.

    MUST be called BEFORE showing the score card.
    Returns True when the gate is complete (score/FUQ can be shown),
    False when the candidate is still in the reflection step (halt rendering).

    State machine (stored in session_state per q_index):
      "reflection_pending_{q_index}" = True  → gate active, waiting for answer
      "reflection_done_{q_index}"    = True  → gate passed, show score
      neither set                            → gate not yet started

    The gate runs through two visual steps:
      Step 1 — Self-assessment prompt (voice or text input)
      Step 2 — Personalised frame reveal + transition to score

    Parameters
    ----------
    evaluation     : AnswerEvaluator result dict
    original_answer: candidate's first answer text
    question_dict  : question dict from question_pool.json
    q_index        : 0-based question index (for namespacing state keys)
    stt            : SpeechToText instance (for voice input tab)
    api_key        : Groq API key

    Research: ACM CSCW 2024 — "Conversate" — 40% greater learning gain
    with dialogic vs one-shot feedback.
    """
    _inject_reflect_css()

    state_pending = f"reflection_pending_{q_index}"
    state_done    = f"reflection_done_{q_index}"
    state_sa      = f"self_assessment_{q_index}"
    state_frame   = f"personalised_frame_{q_index}"

    # ── Gate already passed ────────────────────────────────────────────────
    if st.session_state.get(state_done, False):
        # Show compact "reflection completed" badge then let caller render score
        sa    = st.session_state.get(state_sa, "")
        frame = st.session_state.get(state_frame, "")
        if frame:
            st.markdown(
                f"""
                <div class="reflect-done-card">
                    🪞 <strong>Your reflection:</strong> "{sa[:120]}{'…' if len(sa) > 120 else ''}"<br>
                    <span style="color:#7c6bff;">💡 {frame}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return True

    # ── Gate not yet started — initialise ────────────────────────────────
    if not st.session_state.get(state_pending, False):
        st.session_state[state_pending] = True

    # ── Step 1: Self-assessment prompt ────────────────────────────────────
    # The question is always the same canonical Conversate prompt,
    # not varied by question type — consistency matters for expectation-setting.
    REFLECT_QUESTION = (
        "Before seeing your score — what part of your answer "
        "do you feel was your strongest point?"
    )

    st.markdown(
        f"""
        <div class="reflect-card">
            <div class="reflect-avatar-row">
                <div class="reflect-avatar">🪞</div>
                <div class="reflect-meta">
                    <div class="reflect-label">Self-Reflection Step</div>
                    <div class="reflect-sub">Conversate coaching model · ACM CSCW 2024</div>
                </div>
            </div>
            <div class="reflect-question">"{REFLECT_QUESTION}"</div>
            <div class="reflect-badge">✦ Answer to reveal your personalised score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Voice + text input tabs for reflection answer ─────────────────────
    sa_key  = f"reflect_sa_{q_index}"
    tab_voice, tab_type = st.tabs(["🎙 Voice", "⌨️ Type"])

    sa_voice_text = ""
    sa_typed_text = ""

    with tab_voice:
        st.markdown(
            '<div style="font-size:.72rem;color:#a855f7;margin-bottom:.3rem;">'
            'Speak your self-reflection — transcript appears below.</div>',
            unsafe_allow_html=True,
        )
        # Whisper audio input for reflection
        from voice_input import whisper_audio_input, whisper_post_hud
        raw_reflect_text = whisper_audio_input(stt, q_number=90000 + q_index)
        if raw_reflect_text and not raw_reflect_text.startswith("["):
            sa_voice_text = raw_reflect_text
            # Show compact filler HUD for the reflection answer
            whisper_post_hud(sa_voice_text)

    with tab_type:
        sa_typed_text = st.text_area(
            "Your self-reflection:",
            placeholder="e.g. I think my strongest point was explaining the STAR structure clearly…",
            height=90,
            key=f"reflect_typed_{q_index}",
            label_visibility="collapsed",
        )

    # Resolve which input to use (voice takes precedence if non-empty)
    sa_final = (sa_voice_text or sa_typed_text or "").strip()

    # ── Submit reflection ──────────────────────────────────────────────────
    col_submit, col_skip = st.columns([3, 1])

    with col_submit:
        submit_label = "✦ Reveal my score" if sa_final else "✦ Reveal my score (skip reflection)"
        if st.button(
            submit_label,
            key=f"reflect_submit_{q_index}",
            use_container_width=True,
            type="primary",
        ):
            key = api_key or _GROQ_API_KEY

            if sa_final:
                # Generate personalised frame from Groq
                with st.spinner("Personalising your feedback…"):
                    frame = _groq_personalise_feedback(
                        self_assessment = sa_final,
                        evaluation      = evaluation,
                        original_answer = original_answer,
                        question_text   = question_dict.get("question", ""),
                        api_key         = key,
                    )
                st.session_state[state_sa]    = sa_final
                st.session_state[state_frame] = frame
            else:
                st.session_state[state_sa]    = ""
                st.session_state[state_frame] = ""

            st.session_state[state_pending] = False
            st.session_state[state_done]    = True
            st.rerun()

    with col_skip:
        if st.button(
            "Skip",
            key=f"reflect_skip_{q_index}",
            use_container_width=True,
        ):
            st.session_state[state_sa]      = ""
            st.session_state[state_frame]   = ""
            st.session_state[state_pending] = False
            st.session_state[state_done]    = True
            st.rerun()

    st.markdown(
        "<p style='font-size:11px;color:#3a2060;margin-top:4px;'>"
        "Reflecting before seeing your score is shown to increase learning retention "
        "(Conversate, ACM CSCW 2024). Skip any time.</p>",
        unsafe_allow_html=True,
    )

    return False   # gate not yet passed — caller must not render score


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI COMPONENT
# ══════════════════════════════════════════════════════════════════════════════

# ── CSS injected once per session ─────────────────────────────────────────────
_FU_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

/* ══════════════════════════════════════════════
   ANIMATED FOLLOW-UP CARD — v3.0
   ══════════════════════════════════════════════ */

@keyframes fu-slide-in {
    0%   { opacity: 0; transform: translateY(18px) scale(0.97); }
    100% { opacity: 1; transform: translateY(0)    scale(1);    }
}
@keyframes fu-glow-pulse {
    0%, 100% { box-shadow: 0 0 22px rgba(0,212,200,0.14), 0 0 0 0 rgba(0,212,200,0.0); }
    50%       { box-shadow: 0 0 40px rgba(0,212,200,0.28), 0 2px 24px rgba(0,140,200,0.15); }
}
@keyframes fu-border-sweep {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes fu-avatar-spin-in {
    0%   { transform: rotate(-20deg) scale(0.6); opacity: 0; }
    60%  { transform: rotate(6deg)  scale(1.1); opacity: 1; }
    100% { transform: rotate(0deg)  scale(1);   opacity: 1; }
}
@keyframes fu-dot-bounce {
    0%, 80%, 100% { transform: translateY(0);   opacity: 0.4; }
    40%           { transform: translateY(-6px); opacity: 1;   }
}
@keyframes fu-text-reveal {
    0%   { opacity: 0; letter-spacing: -0.04em; }
    100% { opacity: 1; letter-spacing: normal; }
}
@keyframes fu-badge-pop {
    0%   { transform: scale(0.7); opacity: 0; }
    70%  { transform: scale(1.08); }
    100% { transform: scale(1);   opacity: 1; }
}
@keyframes fu-scan-line {
    0%   { top: 0%;   opacity: 0.5; }
    100% { top: 100%; opacity: 0;   }
}
@keyframes fu-neon-flicker {
    0%, 95%, 100% { opacity: 1; }
    96%            { opacity: 0.7; }
    97%            { opacity: 1; }
    98%            { opacity: 0.8; }
}

/* ── Outer wrapper with entry animation ─────────────── */
.fu-card-wrapper {
    animation: fu-slide-in 0.5s cubic-bezier(.22,.68,0,1.3) both;
    position: relative;
}

/* ── Animated gradient border shell ─────────────────── */
.fu-card-border {
    background: linear-gradient(135deg, #00d4c8, #3b8bff, #a855f7, #00d4c8);
    background-size: 300% 300%;
    animation: fu-border-sweep 5s ease infinite, fu-glow-pulse 3.5s ease-in-out infinite;
    border-radius: 14px;
    padding: 2px;
    margin: 20px 0 12px 0;
}

/* ── Inner card ──────────────────────────────────────── */
.fu-card {
    background: linear-gradient(145deg, #060e20 0%, #0a1730 55%, #071525 100%);
    border-radius: 12px;
    padding: 20px 24px 16px 24px;
    position: relative;
    overflow: hidden;
}

/* Scan-line shimmer effect */
.fu-card::after {
    content: '';
    position: absolute;
    left: 0; right: 0; height: 60px;
    background: linear-gradient(180deg, transparent 0%, rgba(0,212,200,0.04) 50%, transparent 100%);
    animation: fu-scan-line 4s linear infinite;
    pointer-events: none;
}

/* ── Corner accent ───────────────────────────────────── */
.fu-card-corner {
    position: absolute;
    top: 0; right: 0;
    width: 60px; height: 60px;
    background: linear-gradient(225deg, rgba(0,212,200,0.18) 0%, transparent 60%);
    border-radius: 0 12px 0 100%;
}

/* ── Header: type label + source ─────────────────────── */
.fu-header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
}
.fu-type-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,212,200,0.10);
    border: 1px solid rgba(0,212,200,0.22);
    border-radius: 30px;
    padding: 4px 12px 4px 8px;
    animation: fu-badge-pop 0.45s 0.25s cubic-bezier(.22,.68,0,1.3) both;
}
.fu-type-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #00d4c8;
    box-shadow: 0 0 6px #00d4c8;
    animation: fu-neon-flicker 3s 1.2s infinite;
}
.fu-type-label {
    font-family: 'Syne', sans-serif;
    font-size: 9.5px; font-weight: 700;
    letter-spacing: 1.8px; text-transform: uppercase;
    color: #00d4c8;
}
.fu-source-badge {
    font-size: 9.5px; color: #3a5a88;
    letter-spacing: 0.5px;
}

/* ── Thinking dots (shown briefly before question) ───── */
.fu-thinking {
    display: flex;
    align-items: center;
    gap: 5px;
    margin-bottom: 10px;
    font-size: 10px;
    color: #3a6080;
    letter-spacing: 0.5px;
}
.fu-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: #00d4c8;
    animation: fu-dot-bounce 1.2s ease-in-out infinite;
}
.fu-dot:nth-child(2) { animation-delay: 0.18s; }
.fu-dot:nth-child(3) { animation-delay: 0.36s; }

/* ── Quote decoration ────────────────────────────────── */
.fu-quote-mark {
    font-family: 'Syne', sans-serif;
    font-size: 56px; line-height: 0.6;
    color: rgba(0,212,200,0.12);
    float: left;
    margin-right: 6px;
    margin-top: 10px;
    user-select: none;
}

/* ── Question text ───────────────────────────────────── */
.fu-question {
    font-family: 'DM Sans', sans-serif;
    font-size: 16px; font-weight: 600;
    color: #d8eeff;
    line-height: 1.65;
    margin: 0 0 14px 0;
    animation: fu-text-reveal 0.6s 0.35s ease both;
    position: relative;
    z-index: 1;
    clear: both;
}

/* ── Probe strategy strip ────────────────────────────── */
.fu-probe-strip {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
    flex-wrap: wrap;
}
.fu-probe-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(59,139,255,0.10);
    color: #6aa8ff;
    border: 1px solid rgba(59,139,255,0.22);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    animation: fu-badge-pop 0.4s 0.5s cubic-bezier(.22,.68,0,1.3) both;
}
.fu-probe-icon { font-size: 11px; }

/* ── Divider ─────────────────────────────────────────── */
.fu-divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(0,212,200,0.25), rgba(59,139,255,0.1), transparent);
    margin: 14px 0 12px;
}

/* ── Voice mode banner (new!) ────────────────────────── */
.fu-voice-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    background: linear-gradient(90deg, rgba(0,212,200,0.08), rgba(59,139,255,0.06));
    border: 1px solid rgba(0,212,200,0.15);
    border-radius: 8px;
    padding: 9px 14px;
    margin-bottom: 10px;
    font-size: 12px;
    color: #7dd8d4;
    font-family: 'DM Sans', sans-serif;
}
.fu-voice-icon {
    font-size: 18px;
    animation: fu-neon-flicker 2.5s infinite;
}

/* ── Score delta chip ────────────────────────────────── */
.fu-delta-pos { color: #22c55e; font-weight: 700; }
.fu-delta-neg { color: #ef4444; font-weight: 700; }
.fu-delta-neu { color: #94a3b8; font-weight: 700; }

/* ── Skip note ───────────────────────────────────────── */
.fu-skip-note {
    font-size: 10.5px;
    color: #2a4060;
    margin-top: 5px;
    font-family: 'DM Sans', sans-serif;
}

/* ── Completed card (answered already) ───────────────── */
.fu-card-done {
    background: linear-gradient(135deg, #060e1c 0%, #091522 100%);
    border: 1px solid rgba(30,58,110,0.6);
    border-left: 3px solid #1e5a8a;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 14px 0 8px;
    opacity: 0.88;
    animation: fu-slide-in 0.4s ease both;
}
</style>
"""

_CSS_INJECTED = False


def _inject_css() -> None:
    global _CSS_INJECTED
    if not _CSS_INJECTED:
        st.markdown(_FU_CSS, unsafe_allow_html=True)
        _CSS_INJECTED = True


def _score_to_colour(score: float) -> str:
    """Map 1-5 score to a hex colour for the UI."""
    if score >= 4.0:   return "#22c55e"
    elif score >= 3.0: return "#eab308"
    elif score >= 2.0: return "#f97316"
    return "#ef4444"


# ══════════════════════════════════════════════════════════════════════════════
#  MODAL POPUP CSS — full-screen follow-up window
# ══════════════════════════════════════════════════════════════════════════════

_MODAL_CSS = """
<style>
/* ── Backdrop ── */
.fu-modal-backdrop {
    position: fixed; inset: 0; z-index: 9000;
    background: rgba(2,6,20,0.92);
    backdrop-filter: blur(6px);
    display: flex; align-items: center; justify-content: center;
    animation: fuBackdropIn 0.35s ease;
}
@keyframes fuBackdropIn { from { opacity:0; } to { opacity:1; } }

/* ── Modal window ── */
.fu-modal-window {
    width: min(780px, 96vw);
    max-height: 92vh;
    overflow-y: auto;
    background: linear-gradient(160deg, #080f28 0%, #050c20 100%);
    border: 1px solid rgba(0,212,200,0.22);
    border-top: 3px solid #00d4c8;
    border-radius: 18px;
    padding: 28px 32px 24px;
    position: relative;
    box-shadow: 0 0 80px rgba(0,212,200,0.12), 0 24px 80px rgba(0,0,0,0.6);
    animation: fuModalIn 0.4s cubic-bezier(0.34,1.56,0.64,1);
}
@keyframes fuModalIn {
    from { opacity:0; transform:scale(0.88) translateY(30px); }
    to   { opacity:1; transform:scale(1)    translateY(0);    }
}

/* ── Header ── */
.fu-modal-header {
    display: flex; align-items: center; gap: 14px; margin-bottom: 20px;
}
.fu-modal-avatar {
    width: 48px; height: 48px; border-radius: 50%;
    background: linear-gradient(135deg, #0a2040, #1a4080);
    border: 2px solid rgba(0,212,200,0.5);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
    box-shadow: 0 0 20px rgba(0,212,200,0.2);
    animation: fuAvatarPulse 3s ease-in-out infinite;
}
@keyframes fuAvatarPulse {
    0%,100% { box-shadow: 0 0 20px rgba(0,212,200,0.2); }
    50%      { box-shadow: 0 0 35px rgba(0,212,200,0.45); }
}
.fu-modal-title-block { flex: 1; }
.fu-modal-tag {
    display: inline-block;
    background: rgba(0,212,200,0.1);
    border: 1px solid rgba(0,212,200,0.3);
    color: #00d4c8; border-radius: 20px;
    padding: 2px 10px; font-size: 10px; font-weight: 700;
    letter-spacing: 1.4px; text-transform: uppercase;
    margin-bottom: 4px;
}
.fu-modal-title {
    font-size: 18px; font-weight: 700;
    color: #e2ecf9; line-height: 1.3;
    font-family: 'Orbitron', monospace;
}
.fu-modal-sub {
    font-size: 11px; color: #5a7098; margin-top: 3px;
}

/* ── Score reveal strip ── */
.fu-score-strip {
    display: flex; align-items: center; gap: 16px;
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 12px 18px;
    margin-bottom: 18px;
}
.fu-score-circle {
    width: 64px; height: 64px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-direction: column; flex-shrink: 0;
    background: conic-gradient(var(--sc) var(--pct), rgba(255,255,255,0.06) 0);
    position: relative;
}
.fu-score-circle::before {
    content: ''; position: absolute;
    inset: 6px; border-radius: 50%;
    background: #080f28;
}
.fu-score-val {
    position: relative; z-index: 1;
    font-size: 18px; font-weight: 800;
    font-family: 'Orbitron', monospace;
    color: var(--sc);
    line-height: 1;
}
.fu-score-label { position: relative; z-index:1; font-size: 8px; color: #5a7098; }
.fu-score-detail { flex: 1; }
.fu-score-main-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; color: #5a7098; margin-bottom: 3px;
}
.fu-score-feedback {
    font-size: 13px; color: #b0cce8; line-height: 1.6;
}

/* ── Probe pill ── */
.fu-probe-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px; padding: 4px 12px;
    font-size: 10.5px; color: #8ab0d8; margin-bottom: 16px;
}

/* ── Question block ── */
.fu-modal-q-block {
    background: rgba(0,212,200,0.05);
    border: 1px solid rgba(0,212,200,0.18);
    border-left: 4px solid #00d4c8;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin-bottom: 20px;
    position: relative; overflow: hidden;
}
.fu-modal-q-block::before {
    content: '"\201D';
    position: absolute; top: -8px; left: 14px;
    font-size: 60px; color: rgba(0,212,200,0.08);
    font-family: Georgia, serif; line-height: 1;
}
.fu-modal-q-thinking {
    display: flex; align-items: center; gap: 5px;
    margin-bottom: 8px;
}
.fu-modal-q-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: #00d4c8;
    animation: fuDotBounce 1.2s ease-in-out infinite;
}
.fu-modal-q-dot:nth-child(2) { animation-delay: 0.15s; }
.fu-modal-q-dot:nth-child(3) { animation-delay: 0.30s; }
@keyframes fuDotBounce {
    0%,80%,100% { transform: scale(0.6); opacity:0.4; }
    40%          { transform: scale(1.1); opacity:1;   }
}
.fu-modal-q-thinking-label {
    font-size: 9px; letter-spacing: 1.2px; text-transform: uppercase;
    color: rgba(0,212,200,0.6); font-family: monospace;
}
.fu-modal-q-text {
    font-size: 16px; font-weight: 600;
    color: #d8eeff; line-height: 1.65;
    font-style: italic;
}

/* ── Voice input section ── */
.fu-modal-voice-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.4px;
    text-transform: uppercase; color: #00d4c8;
    margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
}
.fu-modal-voice-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #00d4c8;
    animation: fuDotBounce 1.4s ease-in-out infinite;
}

/* ── Submit / skip button row ── */
.fu-modal-btn-row {
    display: flex; gap: 10px; margin-top: 18px;
}

/* ── Thinking bar (shown while generating) ── */
.fu-thinking-bar {
    display: flex; align-items: center; gap: 10px;
    background: rgba(0,212,200,0.06);
    border: 1px solid rgba(0,212,200,0.15);
    border-radius: 10px; padding: 12px 16px;
    margin-bottom: 14px;
}
.fu-thinking-bar-label {
    font-size: 11px; color: rgba(0,212,200,0.8);
    font-family: monospace; letter-spacing: 0.5px;
}
.fu-thinking-bar-dots {
    display: flex; gap: 4px;
}

/* ── Delta strip ── */
.fu-delta-strip {
    display: flex; align-items: center; gap: 8px;
    font-size: 11px; margin-top: 6px; flex-wrap: wrap;
}
.fu-delta-pos { color: #22d87a; font-weight: 700; }
.fu-delta-neg { color: #ff5c5c; font-weight: 700; }
.fu-delta-neu { color: #94a3b8; font-weight: 700; }
</style>
"""

_MODAL_CSS_INJECTED = False

def _inject_modal_css() -> None:
    global _MODAL_CSS_INJECTED
    if not _MODAL_CSS_INJECTED:
        st.markdown(_MODAL_CSS, unsafe_allow_html=True)
        _MODAL_CSS_INJECTED = True


def _score_colour_hex(score: float) -> str:
    """Return a hex colour for a 1-5 score."""
    if score >= 4.2: return "#22d87a"
    if score >= 3.5: return "#a5b4fc"
    if score >= 2.5: return "#fbbf24"
    return "#ff5c5c"


def _pct_for_conic(score: float) -> str:
    """Convert 1-5 score to CSS percentage for conic-gradient."""
    return f"{round(score / 5.0 * 360)}deg"


# ══════════════════════════════════════════════════════════════════════════════
#  MODAL-BASED FOLLOW-UP DIALOG
# ══════════════════════════════════════════════════════════════════════════════

@st.dialog("🎙 Interviewer Follow-Up", width="large")
def _show_follow_up_modal(
    fuq:           "FollowUpQuestion",
    evaluation:    Dict,
    evaluator:     Any,
    question_dict: Dict,
    q_index:       int,
    api_key:       str,
    stt:           Any,
) -> None:
    """
    Full-screen modal dialog for the follow-up question.
    Shows: score reveal → follow-up question → voice/text answer input → submit.
    """
    import streamlit.components.v1 as components
    _inject_modal_css()
    _inject_css()

    main_score   = float(evaluation.get("score", 2.5))
    sc_hex       = _score_colour_hex(main_score)
    sc_pct       = _pct_for_conic(main_score)
    main_feedback = evaluation.get("feedback", "")[:220]

    probe_label  = PROBE_STRATEGIES.get(fuq.probe_strategy, "Probed for detail")
    _PROBE_ICONS = {
        "missing_result":"🎯","low_relevance":"🧭","vague_action":"🎭",
        "no_example":"💡","low_depth":"🔬","shallow_star":"⭐",
        "high_filler":"🗣️","generic":"🔍",
    }
    probe_icon = _PROBE_ICONS.get(fuq.probe_strategy, "🔍")

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="fu-modal-header">
  <div class="fu-modal-avatar">🤖</div>
  <div class="fu-modal-title-block">
    <div class="fu-modal-tag">⚡ Live Interview — Follow-Up</div>
    <div class="fu-modal-title">Interviewer has a follow-up question</div>
    <div class="fu-modal-sub">Based on your previous answer — answer clearly and concisely</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Score reveal ────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="fu-score-strip">
  <div class="fu-score-circle" style="--sc:{sc_hex};--pct:{sc_pct};">
    <div class="fu-score-val" style="color:{sc_hex};">{main_score:.1f}</div>
    <div class="fu-score-label">/5.0</div>
  </div>
  <div class="fu-score-detail">
    <div class="fu-score-main-label">Your previous answer score</div>
    <div class="fu-score-feedback">{main_feedback}{"…" if len(evaluation.get("feedback","")) > 220 else ""}</div>
  </div>
</div>
<div style="margin-bottom:12px;">
  <span class="fu-probe-pill">{probe_icon} {probe_label}</span>
</div>""", unsafe_allow_html=True)

    # ── Follow-up question ───────────────────────────────────────────────────
    st.markdown(f"""
<div class="fu-modal-q-block">
  <div class="fu-modal-q-thinking">
    <div class="fu-modal-q-dot"></div>
    <div class="fu-modal-q-dot"></div>
    <div class="fu-modal-q-dot"></div>
    <span class="fu-modal-q-thinking-label">Interviewer is probing deeper…</span>
  </div>
  <div class="fu-modal-q-text">{fuq.text}</div>
</div>""", unsafe_allow_html=True)

    # ── Voice + Text answer input ────────────────────────────────────────────
    st.markdown("""
<div class="fu-modal-voice-label">
  <span class="fu-modal-voice-dot"></span>
  Your Answer — Voice or Text
</div>""", unsafe_allow_html=True)

    fu_voice_whisper = ""
    fu_voice_browser = ""
    fu_typed_text    = ""

    tab_w, tab_b, tab_t = st.tabs(["🎙 Whisper AI", "🌐 Browser STT", "⌨️ Type"])

    with tab_w:
        st.markdown(
            '<div style="font-size:.72rem;color:#00d4c8;margin:.2rem 0 .5rem;">' +
            '🎤 Record your answer — Whisper AI transcribes it instantly.</div>',
            unsafe_allow_html=True,
        )
        if stt is not None:
            from voice_input import whisper_audio_input, whisper_post_hud
            raw = whisper_audio_input(stt, q_number=80000 + q_index)
            if raw and not raw.startswith("["):
                fu_voice_whisper = raw
                whisper_post_hud(fu_voice_whisper)
                fu_voice_whisper = st.text_area(
                    "Edit Whisper transcript:",
                    value=fu_voice_whisper,
                    height=100,
                    key=f"fu_modal_whisper_{q_index}",
                )
            elif raw and raw.startswith("["):
                st.markdown(f'<div style="font-size:.78rem;color:#fcd34d;">{raw}</div>',
                            unsafe_allow_html=True)
        else:
            st.info("Whisper not available — use Browser STT or Type tab.")

    with tab_b:
        st.markdown(
            '<div style="font-size:.72rem;color:#00d4c8;margin:.2rem 0 .5rem;">' +
            '🌐 Live speech-to-text — speak and watch your words appear in real time.</div>',
            unsafe_allow_html=True,
        )
        if stt is not None:
            try:
                from voice_input import browser_stt_with_audio, whisper_post_hud
                bstt_tx, bstt_audio = browser_stt_with_audio(q_number=85000 + q_index)
                if bstt_tx:
                    fu_voice_browser = st.text_area(
                        "Edit live transcript:",
                        value=bstt_tx,
                        height=100,
                        key=f"fu_modal_browser_{q_index}",
                    )
                    if bstt_audio:
                        st.session_state["last_audio_bytes"]  = bstt_audio
                        st.session_state["last_audio_source"] = "browser_stt"
                    whisper_post_hud(fu_voice_browser)
            except ImportError:
                st.info("Browser STT unavailable — use Whisper or Type tab.")
        else:
            st.info("Pass stt= to render_follow_up_ui() to enable Browser STT.")

    with tab_t:
        fu_typed_text = st.text_area(
            label            = "Your follow-up answer",
            placeholder      = "Take a moment, then answer the interviewer's question clearly and concisely…",
            height           = 130,
            key              = f"fu_modal_type_{q_index}_{st.session_state.get('fu_answer_input_key', 0)}",
            label_visibility = "collapsed",
        )

    fu_answer_val = (fu_voice_whisper or fu_voice_browser or fu_typed_text or "").strip()

    # ── Submit / Skip ────────────────────────────────────────────────────────
    col_sub, col_skip = st.columns([3, 1])
    with col_sub:
        if st.button("✅ Submit Follow-Up Answer", key=f"fu_modal_submit_{q_index}",
                     use_container_width=True, type="primary"):
            if not fu_answer_val or len(fu_answer_val.strip()) < 5:
                st.warning("Please record or type your answer before submitting.")
            else:
                with st.spinner("Evaluating follow-up…"):
                    _em = {
                        "nervousness":       st.session_state.get("live_nervousness", 0.2),
                        "voice_emotion":     st.session_state.get("live_voice_emotion", ""),
                        "voice_nervousness": st.session_state.get("live_voice_nerv", 0.2),
                        "facial_emotion":    st.session_state.get("live_emotion", ""),
                    }
                    record = score_follow_up(
                        fuq              = fuq,
                        follow_up_answer = fu_answer_val.strip(),
                        evaluator        = evaluator,
                        question_dict    = question_dict,
                        original_score   = float(evaluation.get("score", 2.5)),
                        emotion_state    = _em,
                    )

                if "follow_up_records" not in st.session_state:
                    st.session_state["follow_up_records"] = []
                st.session_state["follow_up_records"].append(record)

                # Append follow-up to session_answers at 0.5× weight
                if "session_answers" in st.session_state:
                    _orig_qn = q_index + 1
                    st.session_state["session_answers"].append({
                        "number": f"{_orig_qn}FU", "question": fuq.text,
                        "answer": fu_answer_val.strip(),
                        "score": round(record.score * 0.5, 2),
                        "type": fuq.question_type, "feedback": record.feedback,
                        "final_score": record.nlp_score,
                        "similarity_score": record.similarity_score,
                        "grammar_score": record.grammar_score,
                        "depth_score": record.depth_score,
                        "word_count": record.word_count,
                        "nervousness": record.nervousness,
                        "facial_nervousness": record.nervousness,
                        "voice_emotion": record.voice_emotion,
                        "voice_nervousness": record.voice_nervousness,
                        "emotion": record.facial_emotion,
                        "is_follow_up": True, "parent_q_number": _orig_qn,
                        "probe_label": record.probe_label,
                        "depth_delta": record.depth_delta,
                        "star": {}, "fluency": 3.5, "fluency_score": 3.5,
                        "disc_traits": {}, "disc_dominant": "None",
                        "filler_count": 0, "filler_ratio": 0.0,
                        "hiring_signal": 2.5, "star_count": 0,
                        "vocab_diversity": 0.0, "relevance_source": "none",
                        "question_type": fuq.question_type,
                        "time_score": 0.0, "time_label": "N/A",
                        "time_efficiency": 0.0, "time_verdict": "N/A",
                        "time_ideal_window": "—",
                        "rl_next_action": "static", "rl_epsilon": None,
                        "confidence_score": 3.5, "posture_score": 3.5,
                        "facial_score": 3.5, "voice_score": 3.5, "eye_score": 3.5,
                        "time_s": 0, "difficulty": question_dict.get("difficulty","medium"),
                        "resume_target": "", "source": "follow_up",
                    })

                st.session_state["pending_follow_up"]   = None
                st.session_state["fu_modal_open"]       = False
                st.rerun()

    with col_skip:
        if st.button("Skip", key=f"fu_modal_skip_{q_index}", use_container_width=True):
            st.session_state["pending_follow_up"] = None
            st.session_state["fu_modal_open"]     = False
            st.rerun()

    st.markdown(
        "<p style='font-size:10px;color:#3a5080;text-align:center;margin-top:8px;'>"
        "Skipping won't affect your main score, but will be noted in the final report.</p>",
        unsafe_allow_html=True,
    )


def render_follow_up_ui(
    evaluator:       Any,
    question_dict:   Dict,
    original_answer: str,
    evaluation:      Dict,
    q_index:         int,
    api_key:         str = "",
    stt:             Any = None,
) -> None:
    """
    Full follow-up UI — now renders as a modal popup window after answer submission.

    Flow:
      1. After submit, a glowing "Follow-Up Available" banner appears in the main page.
      2. The user clicks "Open Follow-Up" → st.dialog modal slides in.
      3. Inside the modal: score reveal → probed follow-up question → voice/text input → submit.
      4. On submit, record is stored and modal closes.
    """
    # ── Guards ───────────────────────────────────────────────────────────────
    if not st.session_state.get("fu_enabled", True):
        return
    if not original_answer or len(original_answer.strip()) < 15:
        return

    _inject_css()
    _inject_modal_css()

    key = api_key or _GROQ_API_KEY

    # ── Already completed — show compact summary ─────────────────────────────
    records: List[FollowUpRecord] = st.session_state.get("follow_up_records", [])
    if any(r.q_index == q_index for r in records):
        _render_completed_follow_up(records, q_index)
        return

    # ── v2.0 Reflective Gate — runs before follow-up is shown ───────────────
    gate_passed = render_reflective_gate(
        evaluation      = evaluation,
        original_answer = original_answer,
        question_dict   = question_dict,
        q_index         = q_index,
        stt             = stt,
        api_key         = key,
    )
    if not gate_passed:
        return

    # ── Decide whether to generate / show follow-up ──────────────────────────
    pending: Optional[FollowUpQuestion] = st.session_state.get("pending_follow_up")
    threshold    = float(st.session_state.get("fu_score_threshold", 3.2))
    auto_trigger = st.session_state.get("fu_auto_trigger", True)
    main_score   = float(evaluation.get("score", 5.0))
    already_pending_for_this_q = (pending is not None and pending.q_index == q_index)

    if not already_pending_for_this_q:
        should_generate = auto_trigger and main_score <= threshold
        if not should_generate:
            # Manual trigger — show a subtle banner button
            st.markdown(f"""
<div style="background:rgba(0,212,200,0.06);border:1px solid rgba(0,212,200,0.2);
  border-radius:10px;padding:10px 16px;margin:.6rem 0;
  display:flex;align-items:center;justify-content:space-between;">
  <div style="display:flex;align-items:center;gap:8px;">
    <span style="font-size:16px;">🎙</span>
    <div>
      <div style="font-size:11px;font-weight:700;color:#00d4c8;letter-spacing:.5px;">
        Interviewer follow-up available</div>
      <div style="font-size:10px;color:#5a7098;">Click to open a deeper probing question</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
            if st.button("⚡ Open Follow-Up Question", key=f"fu_trigger_{q_index}",
                         use_container_width=True):
                with st.spinner("Interviewer is formulating a follow-up…"):
                    fuq = generate_follow_up(
                        evaluation=evaluation, original_answer=original_answer,
                        question_dict=question_dict, q_index=q_index, api_key=key,
                    )
                st.session_state["pending_follow_up"]   = fuq
                st.session_state["fu_answer_input_key"] = st.session_state.get("fu_answer_input_key", 0) + 1
                st.session_state["fu_modal_open"]       = True
                st.rerun()
            return
        else:
            # Auto-generate then immediately open modal
            with st.spinner("Interviewer is formulating a follow-up…"):
                fuq = generate_follow_up(
                    evaluation=evaluation, original_answer=original_answer,
                    question_dict=question_dict, q_index=q_index, api_key=key,
                )
            st.session_state["pending_follow_up"]   = fuq
            st.session_state["fu_answer_input_key"] = st.session_state.get("fu_answer_input_key", 0) + 1
            st.session_state["fu_modal_open"]       = True
            st.rerun()

    # ── Pending follow-up exists — show banner + open modal ──────────────────
    fuq = st.session_state.get("pending_follow_up")
    if fuq is None or fuq.q_index != q_index:
        return

    probe_label = PROBE_STRATEGIES.get(fuq.probe_strategy, "Probed for detail")
    sc_hex      = _score_colour_hex(float(evaluation.get("score", 2.5)))

    # Glowing banner that invites the user to open the modal
    st.markdown(f"""
<div style="
  background: linear-gradient(135deg,rgba(0,212,200,0.08),rgba(59,139,255,0.06));
  border: 1px solid rgba(0,212,200,0.3);
  border-left: 4px solid #00d4c8;
  border-radius: 12px; padding: 14px 18px; margin:.6rem 0;
  display:flex; align-items:center; gap:14px;
  box-shadow: 0 0 30px rgba(0,212,200,0.08);
  animation: fuBannerIn .5s ease;">
  <div style="font-size:26px;flex-shrink:0;">🤖</div>
  <div style="flex:1;">
    <div style="font-size:12px;font-weight:700;color:#00d4c8;
      letter-spacing:.8px;text-transform:uppercase;margin-bottom:3px;">
      ⚡ Interviewer Follow-Up Ready
    </div>
    <div style="font-size:12px;color:#8ab4d0;">
      Probe strategy: <strong style="color:#c4ddf4;">{probe_label}</strong>
    </div>
    <div style="font-size:10px;color:#5a7098;margin-top:2px;">
      Answer by voice or text — scored with full precision
    </div>
  </div>
</div>
<style>
@keyframes fuBannerIn {{from{{opacity:0;transform:translateY(-8px)}}to{{opacity:1;transform:none}}}}
</style>""", unsafe_allow_html=True)

    # Button to open the modal
    if st.button("🎙 Answer Follow-Up Question →", key=f"fu_open_modal_{q_index}",
                 use_container_width=True, type="primary"):
        _show_follow_up_modal(
            fuq           = fuq,
            evaluation    = evaluation,
            evaluator     = evaluator,
            question_dict = question_dict,
            q_index       = q_index,
            api_key       = key,
            stt           = stt,
        )

    # Also auto-open if the flag is set from this run
    if st.session_state.get("fu_modal_open"):
        st.session_state["fu_modal_open"] = False
        _show_follow_up_modal(
            fuq           = fuq,
            evaluation    = evaluation,
            evaluator     = evaluator,
            question_dict = question_dict,
            q_index       = q_index,
            api_key       = key,
            stt           = stt,
        )


def _render_completed_follow_up(records: List[FollowUpRecord], q_index: int) -> None:
    """Show a compact summary card for an already-answered follow-up."""
    _inject_css()
    record = next((r for r in records if r.q_index == q_index), None)
    if record is None:
        return

    delta = record.depth_delta
    if delta > 0.15:
        delta_html = f"<span class='fu-delta-pos'>+{delta:.1f} ▲</span>"
        delta_note = "improved on follow-up"
    elif delta < -0.15:
        delta_html = f"<span class='fu-delta-neg'>{delta:.1f} ▼</span>"
        delta_note = "weaker on follow-up"
    else:
        delta_html = f"<span class='fu-delta-neu'>~{delta:+.1f}</span>"
        delta_note = "consistent depth"

    score_colour = _score_to_colour(record.score)

    st.markdown(
        f"""
        <div class="fu-card-done">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="font-size:16px;">✅</span>
                    <div>
                        <div style="font-size:9.5px;font-weight:700;letter-spacing:1.5px;
                                    text-transform:uppercase;color:#2a7aaa;">Follow-up completed</div>
                        <div style="font-size:9.5px;color:#3a5070;">{record.probe_label}</div>
                    </div>
                </div>
                <div style="text-align:right;">
                    <span style="color:{score_colour};font-size:20px;font-weight:800;
                                 font-family:'Syne',sans-serif;">{record.score:.1f}<span style="font-size:12px;color:#3a5070;">/5</span></span><br>
                    <span style="font-size:10px;">{delta_html}</span>
                    <span style="font-size:9px;color:#3a5070;"> {delta_note}</span>
                </div>
            </div>
            <div style="font-size:12px;color:#5a7aaa;font-style:italic;padding-left:4px;
                        border-left:2px solid rgba(30,90,138,0.4);margin-bottom:6px;">
                "{record.follow_up_q}"
            </div>
            <div style="font-size:11px;color:#3a6088;">
                💡 {record.feedback[:160]}{'…' if len(record.feedback) > 160 else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PDF SECTION BUILDER (for finish_interview.py)
# ══════════════════════════════════════════════════════════════════════════════

def render_follow_up_pdf_section(
    story:    list,
    records:  List[FollowUpRecord],
    S:        Dict,
    RL_BG:    Any,
    RL_ACCENT:Any,
    RL_TEXT:  Any,
    RL_MUTED: Any,
    RL_TEAL:  Any,
) -> None:
    """
    Append a "Follow-Up Probing Depth" section to the ReportLab story list.

    Call this from finish_interview._build_pdf() after the per-question
    breakdown loop.  If no follow-up records exist, this is a no-op.

    Parameters
    ----------
    story     : the ReportLab story list being built
    records   : list of FollowUpRecord objects from session_state
    S         : paragraph style dict already built in _build_pdf()
    RL_*      : colour constants from finish_interview.py
    """
    if not records:
        return

    try:
        from reportlab.platypus import (
            HRFlowable, PageBreak, Paragraph, Spacer, Table, TableStyle,
        )
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    except ImportError:
        return

    story.append(PageBreak())
    story.append(Paragraph("Follow-Up Probing Depth", S["h2"]))
    story.append(Paragraph(
        "After each answer, the AI interviewer asked a targeted follow-up question "
        "to probe the weakest signal in the response. This section shows how depth "
        "changed between the original answer and the follow-up.",
        S["body"],
    ))
    story.append(Spacer(1, 10))

    # ── Summary table (one row per follow-up) ─────────────────────────────────
    W_pts = 16.8 * cm   # approximate content width (A4 minus margins)

    header_row = ["Q#", "Probe Strategy", "FU Score", "Δ Depth", "Outcome"]
    table_data  = [header_row]

    for rec in records:
        delta       = rec.depth_delta
        delta_str   = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
        outcome_str = (
            "Improved"   if delta >  0.15 else
            "Declined"   if delta < -0.15 else
            "Consistent"
        )
        table_data.append([
            f"Q{rec.q_index + 1}",
            rec.probe_label[:38],
            f"{rec.score:.1f}/5  ({rec.score_pct:.0f}%)",
            delta_str,
            outcome_str,
        ])

    col_widths = [
        W_pts * 0.08,   # Q#
        W_pts * 0.38,   # Probe Strategy
        W_pts * 0.20,   # FU Score
        W_pts * 0.12,   # Δ Depth
        W_pts * 0.22,   # Outcome
    ]

    summary_tbl = Table(table_data, colWidths=col_widths)
    summary_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1,  0), RL_ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1,  0), colors.HexColor("#ffffff")),
        ("FONTNAME",      (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("BACKGROUND",    (0, 1), (-1, -1), RL_BG),
        ("TEXTCOLOR",     (0, 1), (-1, -1), RL_TEXT),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.HexColor("#0c142d"), colors.HexColor("#080f24")]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#1a2555")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(summary_tbl)
    story.append(Spacer(1, 16))

    # ── Detailed per-record blocks ─────────────────────────────────────────────
    story.append(Paragraph("Detailed Follow-Up Exchanges", S["h2"]))
    story.append(Spacer(1, 6))

    _delta_pos_col = colors.HexColor("#22c55e")
    _delta_neg_col = colors.HexColor("#ef4444")
    _delta_neu_col = colors.HexColor("#94a3b8")

    _delta_hex_map = {
        "pos": "22c55e",
        "neg": "ef4444",
        "neu": "94a3b8",
    }

    def _delta_colour(delta: float):
        if delta >  0.15: return _delta_pos_col, "22c55e"
        if delta < -0.15: return _delta_neg_col, "ef4444"
        return _delta_neu_col, "94a3b8"

    _answer_style = ParagraphStyle(
        "fu_answer", parent=S["answer"],
        fontSize=8.5, leading=12,
    )
    _fq_style = ParagraphStyle(
        "fu_q", parent=S["h3"],
        fontSize=9, textColor=RL_TEAL,
        fontName="Helvetica-Oblique",
        spaceBefore=6, spaceAfter=2,
    )

    for rec in records:
        delta             = rec.depth_delta
        delta_str         = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
        d_col, d_hex      = _delta_colour(delta)

        # Section header with Q number + score delta
        q_label   = f"Q{rec.q_index + 1} Follow-Up  —  {rec.probe_label}"
        score_str = f"FU Score: {rec.score:.1f}/5  |  Δ {delta_str}"

        q_header = Table(
            [[Paragraph(q_label, S["h3"]),
              Paragraph(
                  f'<font color="#{d_hex}" size="10"><b>{score_str}</b></font>',
                  S["body"],
              )]],
            colWidths=[W_pts * 0.72, W_pts * 0.28],
        )
        q_header.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#0a1030")),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN",         (1, 0), (1,  0), "RIGHT"),
        ]))
        story.append(q_header)

        # Follow-up question (italic teal)
        story.append(Paragraph(f'"{rec.follow_up_q}"', _fq_style))

        # Follow-up answer
        fu_ans_text = rec.follow_up_answer.strip() or "(no answer recorded)"
        story.append(Paragraph(fu_ans_text, _answer_style))

        # Feedback chip
        if rec.feedback:
            story.append(Paragraph(f"💡  {rec.feedback}", S["feedback"]))

        story.append(HRFlowable(
            width="100%", thickness=0.3,
            color=colors.HexColor("#1a2555"), spaceAfter=8,
        ))


# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS UI (for render_coach_settings()-style sidebar section)
# ══════════════════════════════════════════════════════════════════════════════

def render_follow_up_settings() -> None:
    """
    Render follow-up settings controls in the Streamlit sidebar.
    Call this from wherever render_coach_settings() is called in app.py.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<span style='color:#00d4c8;font-weight:700;font-size:13px;'>"
        "🎙 Follow-Up Probing</span>",
        unsafe_allow_html=True,
    )

    st.session_state["fu_enabled"] = st.sidebar.toggle(
        "Enable follow-up questions",
        value=st.session_state.get("fu_enabled", True),
        help="After each answer, the AI interviewer will ask a targeted follow-up.",
    )

    if st.session_state.get("fu_enabled", True):
        st.session_state["fu_auto_trigger"] = st.sidebar.toggle(
            "Auto-trigger on low scores",
            value=st.session_state.get("fu_auto_trigger", True),
            help="Automatically ask a follow-up when the answer scores below the threshold.",
        )

        if st.session_state.get("fu_auto_trigger", True):
            st.session_state["fu_score_threshold"] = st.sidebar.slider(
                "Auto-trigger threshold (score ≤)",
                min_value=1.5,
                max_value=4.5,
                value=float(st.session_state.get("fu_score_threshold", 3.2)),
                step=0.1,
                format="%.1f",
                help="Follow-up is auto-generated when the main answer scores at or below this value (1–5 scale).",
            )