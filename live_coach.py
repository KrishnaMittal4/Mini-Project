"""
live_coach.py — Aura AI | Live Spoken Coaching Feedback (v1.1)
v1.1 — GROQ RETRY + EXPONENTIAL BACKOFF
  • _groq_coaching_tip() now retries up to 3 attempts with exponential backoff
    on rate-limit (429) and server errors (5xx).
  • Permanent errors (401/403/400) abort immediately — no wasted retries.
  • Mirrors the retry strategy in answer_evaluator._groq_call().

==============================================================
Delivers a personalised, spoken coaching message after each interview
answer — like having a coach in the room, not just a score at the end.

HOW IT WORKS
────────────
1.  generate_coaching_tip(ev, q_type, score)
      Builds a 2–3 sentence coaching message from the evaluation dict
      produced by AnswerEvaluator.evaluate().  Uses Groq when available
      for a natural, conversational tip.  Falls back to a rich
      rule-based generator (no API key required) that covers every
      weakness signal the evaluator already detects.

2.  render_coach_card(coaching_tip, ev, score)
      Renders the coaching card in Streamlit: animated coach avatar,
      colour-coded score badge, the spoken coaching text, and a
      "Speak Feedback" button that fires window.speechSynthesis via
      a tiny injected HTML component.

3.  render_tts_button(text, key)
      Standalone TTS injector — can be called anywhere to attach a
      speak button to any text.

INTEGRATION (app.py)
─────────────────────
    from live_coach import generate_coaching_tip, render_coach_card

    # Inside submit_answer() callback, store the coaching tip:
    coaching_tip = generate_coaching_tip(ev, _qtype, score)
    st.session_state["last_coaching_tip"] = coaching_tip

    # After the submit button, where render_eval_results() is called:
    if st.session_state.submitted and st.session_state.last_score is not None:
        render_eval_results(st.session_state.last_eval)
        render_coach_card(
            st.session_state.get("last_coaching_tip", ""),
            st.session_state.last_eval,
            st.session_state.last_score,
        )

SESSION STATE KEYS USED
────────────────────────
    last_coaching_tip   str   — coaching message for the current answer
    coach_voice_rate    float — TTS speech rate (0.7 – 1.3), set in Settings
    coach_voice_pitch   float — TTS pitch (0.8 – 1.2), set in Settings
    coach_auto_speak    bool  — auto-speak on submit without clicking button
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components


# ══════════════════════════════════════════════════════════════════════════════
#  GROQ COACHING GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

_GROQ_MODEL = "llama-3.3-70b-versatile"


# ══════════════════════════════════════════════════════════════════════════════
#  RAG COACHING PLAYBOOK  (v2.0 — embedded, no external store needed)
# ══════════════════════════════════════════════════════════════════════════════
# A library of expert coaching interventions tagged by weakness signal.
# _rag_retrieve_coaching() embeds the dominant weakness description and
# returns the top-k most semantically similar interventions as grounding
# examples for the Groq prompt.
#
# This replaces the rule-based fallback for the Groq path AND enriches the
# fallback itself when Groq is unavailable — retrieved examples always
# provide more specific, coach-quality phrasing than generic templates.
#
# Schema per entry:
#   weakness   : short slug matching _detect_dominant_weakness() output
#   q_type     : "technical" | "behavioural" | "hr" | "any"
#   signal     : human-readable description of what triggered this (used
#                as the embedding text so retrieval is semantic, not exact)
#   praise     : one sentence acknowledging something positive
#   improvement: the key improvement the candidate needs to make
#   micro_tip  : one concrete action they can take on the very next answer
# ─────────────────────────────────────────────────────────────────────────────

_RAG_COACHING_PLAYBOOK: list = [

    # ── MISSING RESULT ────────────────────────────────────────────────────────
    {
        "weakness": "missing_result", "q_type": "behavioural",
        "signal": "candidate completed STAR story but omitted the final result or impact",
        "praise": "You gave a clear situation and walked me through your actions step by step.",
        "improvement": "The one thing missing was the outcome — always close with what actually changed: a number, a deadline met, or feedback you received.",
        "micro_tip": "On your next answer, make 'result' the last word you plan for — say it out loud before you start.",
    },
    {
        "weakness": "missing_result", "q_type": "behavioural",
        "signal": "answer ends mid-story with no measurable impact or quantified outcome",
        "praise": "Your example was specific and showed real ownership of the problem.",
        "improvement": "Finish the story with a concrete number or outcome — even a rough figure like '30% faster' or 'two weeks ahead of deadline' makes your impact tangible.",
        "micro_tip": "Ask yourself: 'What would my manager have noticed changed?' — that's your result.",
    },
    {
        "weakness": "missing_result", "q_type": "hr",
        "signal": "described personal effort or growth but gave no observable evidence or feedback received",
        "praise": "You showed genuine self-awareness about the situation you chose.",
        "improvement": "Anchor the reflection with evidence — what feedback did you receive, or what measurable change showed it worked?",
        "micro_tip": "Think of one thing someone else said or did that confirmed your growth — that's your result.",
    },

    # ── VAGUE ACTION / NO PERSONAL OWNERSHIP ──────────────────────────────────
    {
        "weakness": "vague_action", "q_type": "behavioural",
        "signal": "candidate used 'we' throughout with no clear individual contribution",
        "praise": "You described the team situation clearly and showed collaborative instincts.",
        "improvement": "Interviewers need to see YOUR specific contribution — replace 'we did' with 'I personally did' and name the exact decision or action you owned.",
        "micro_tip": "Before answering, ask yourself: 'What would not have happened without me?' — lead with that.",
    },
    {
        "weakness": "vague_action", "q_type": "behavioural",
        "signal": "actions described at high level without step-by-step personal ownership",
        "praise": "You identified a relevant situation and showed awareness of the challenge.",
        "improvement": "Go deeper on your actions — not 'I managed the project' but 'I made three decisions: I cut scope, moved the deadline, and escalated to the CTO on day two.'",
        "micro_tip": "Use the word 'I decided' at least once in your Action section — it signals ownership instantly.",
    },

    # ── LOW TECHNICAL DEPTH ───────────────────────────────────────────────────
    {
        "weakness": "low_depth", "q_type": "technical",
        "signal": "correct surface-level definition given but no tradeoffs edge cases or production context",
        "praise": "You got the core definition right and stayed on topic.",
        "improvement": "Senior interviewers expect tradeoffs — next time add one thing this approach is bad at, and one production scenario where you'd reach for something else.",
        "micro_tip": "End every technical answer with 'The main limitation is…' — it signals you've used it in the real world.",
    },
    {
        "weakness": "low_depth", "q_type": "technical",
        "signal": "named a tool or technology without explaining how it actually works internally",
        "praise": "You demonstrated familiarity with the right tools for this problem.",
        "improvement": "Naming the tool is a start — explain the mechanism. How does it actually work under the hood? What decision does it make that matters?",
        "micro_tip": "Practice explaining one technology you use daily in three sentences — mechanism, tradeoff, when-not-to-use.",
    },
    {
        "weakness": "low_depth", "q_type": "technical",
        "signal": "answer correct but at junior level for the seniority of the role being interviewed for",
        "praise": "Your answer was technically accurate — that foundation is solid.",
        "improvement": "Push to the next level: mention a failure mode, a scale consideration, or a time this approach broke in production. That's what separates senior candidates.",
        "micro_tip": "Ask yourself: 'What would break first at 10× the load?' — that question unlocks senior-level answers.",
    },

    # ── NO CONCRETE EXAMPLE ───────────────────────────────────────────────────
    {
        "weakness": "no_example", "q_type": "technical",
        "signal": "answer is theoretical with no real project system or codebase mentioned",
        "praise": "You showed solid conceptual understanding of the topic.",
        "improvement": "Theory needs grounding — mention a real project, even briefly. 'I used this in a Django API serving 50k requests a day' is ten times more convincing than a textbook definition.",
        "micro_tip": "Pick one project you know inside out and map every technical concept you explain back to it.",
    },
    {
        "weakness": "no_example", "q_type": "behavioural",
        "signal": "described what they would do hypothetically instead of a past real experience",
        "praise": "Your instincts about how to handle the situation were sound.",
        "improvement": "Behavioural questions need real past examples — 'I would' signals you haven't done it, which weakens your answer significantly.",
        "micro_tip": "Keep a mental bank of five strong stories — one per competency — that you can adapt to any behavioural question.",
    },
    {
        "weakness": "no_example", "q_type": "hr",
        "signal": "gave a generic answer that could apply to any candidate with no personal story",
        "praise": "Your answer covered the right themes and showed self-awareness.",
        "improvement": "Generic answers are forgettable — tie this to a real moment from your career that only you could tell. Specificity is what makes an answer stick.",
        "micro_tip": "Add one proper noun — a company name, a person, a product — to instantly make the answer feel real.",
    },

    # ── HIGH FILLER WORDS ─────────────────────────────────────────────────────
    {
        "weakness": "high_filler", "q_type": "any",
        "signal": "answer contained many filler words such as um uh like basically you know",
        "praise": "You had good content in there — the ideas were solid.",
        "improvement": "The filler words are diluting your message. A silent pause sounds far more confident than 'um' — interviewers actually respect the pause.",
        "micro_tip": "In your next answer, when you feel 'um' coming, close your mouth and breathe instead — one second of silence beats five filler words.",
    },
    {
        "weakness": "high_filler", "q_type": "any",
        "signal": "candidate repeated same idea multiple times using filler to fill gaps",
        "praise": "You clearly understood the question and had relevant things to say.",
        "improvement": "You repeated the core point two or three times — once is enough. Structure your answer as three distinct points and stop after the third.",
        "micro_tip": "Before you speak, plan three words: 'Point 1, Point 2, Result' — then stick to exactly those three beats.",
    },

    # ── ANSWER TOO BRIEF ──────────────────────────────────────────────────────
    {
        "weakness": "too_brief", "q_type": "technical",
        "signal": "technical answer was very short missing explanation and context",
        "praise": "You identified the right concept — the direction was correct.",
        "improvement": "The answer needed more depth — aim for around 150 words that cover the definition, how it works, and one tradeoff or real use case.",
        "micro_tip": "After every technical answer, ask yourself: 'Did I say WHY it matters and WHEN I'd use it?' — if not, add one more sentence.",
    },
    {
        "weakness": "too_brief", "q_type": "behavioural",
        "signal": "behavioural story was too short to give a complete STAR narrative",
        "praise": "You gave a relevant example — the instinct was right.",
        "improvement": "The story needed more flesh — a complete behavioural answer should have a clear Situation, your specific Task, your Actions in detail, and a concrete Result.",
        "micro_tip": "If your answer takes under 45 seconds to say out loud, it's almost always too short — add one more specific detail to each STAR component.",
    },

    # ── ANSWER TOO LONG ───────────────────────────────────────────────────────
    {
        "weakness": "too_long", "q_type": "any",
        "signal": "answer was excessively long and rambling losing the key message",
        "praise": "You clearly have deep experience and a lot to say on this topic.",
        "improvement": "The length is working against you — interviewers disengage after about 2 minutes. Edit your answer to the three most important points and stop there.",
        "micro_tip": "Practice the 'newspaper headline' test: what's the one sentence that captures your whole answer? Lead with that next time.",
    },
    {
        "weakness": "too_long", "q_type": "behavioural",
        "signal": "behavioural story included too much background and context before reaching the action",
        "praise": "You gave a rich, detailed example with real depth.",
        "improvement": "Cut the setup in half — interviewers need just enough context to follow the story. Get to your personal action within the first 30 seconds.",
        "micro_tip": "Limit your Situation to two sentences maximum — one line on context, one line on the challenge.",
    },

    # ── GRAMMAR / FLUENCY ─────────────────────────────────────────────────────
    {
        "weakness": "grammar", "q_type": "any",
        "signal": "answer contained multiple grammatical errors or incomplete sentences",
        "praise": "The ideas you were communicating were relevant and clear in intent.",
        "improvement": "Clean sentence structure signals confidence and preparation — interviewers often can't separate communication quality from competence, even if they try.",
        "micro_tip": "Record yourself answering one practice question today and listen back — you'll immediately hear what needs cleaning up.",
    },

    # ── LOW RELEVANCE ─────────────────────────────────────────────────────────
    {
        "weakness": "low_relevance", "q_type": "technical",
        "signal": "technical answer addressed adjacent topic but missed the core of what was asked",
        "praise": "You demonstrated broad technical knowledge across the area.",
        "improvement": "The answer drifted from what was asked — make sure your first sentence directly names the specific concept in the question before expanding.",
        "micro_tip": "Start every technical answer by repeating the core term from the question — it forces you to stay on target.",
    },
    {
        "weakness": "low_relevance", "q_type": "behavioural",
        "signal": "example chosen did not demonstrate the competency the question was testing",
        "praise": "You told a complete story with a clear structure.",
        "improvement": "The example didn't quite match what the question was testing — choose a story that more directly shows that specific competency in action.",
        "micro_tip": "Before picking your example, ask: 'Does this story PROVE I have [the skill]?' — if you're not sure, pick a different one.",
    },

    # ── SHALLOW STAR ──────────────────────────────────────────────────────────
    {
        "weakness": "shallow_star", "q_type": "behavioural",
        "signal": "STAR components present but described at surface level without specificity",
        "praise": "You followed the STAR structure — that shows preparation and discipline.",
        "improvement": "The framework was there but the detail was thin — push each component deeper. Name a specific person, deadline, tool, or number in at least two of the four parts.",
        "micro_tip": "If you said 'a client project', make it 'a fintech client migrating from Oracle to Postgres in Q3 2023' — proper nouns make stories real.",
    },
    {
        "weakness": "shallow_star", "q_type": "hr",
        "signal": "HR answer gave general themes without grounding in a specific past moment",
        "praise": "Your answer showed self-awareness and a thoughtful perspective.",
        "improvement": "Anchor it to a real moment — the most memorable HR answers always reference a specific situation that only you could describe.",
        "micro_tip": "Pick the single most formative career moment that relates to the question and build the answer around that one story.",
    },

    # ── KEYWORD GAP (TECHNICAL) ───────────────────────────────────────────────
    {
        "weakness": "missing_keywords", "q_type": "technical",
        "signal": "answer correct in concept but missing expected technical terminology that signals depth",
        "praise": "The reasoning behind your answer was sound — you understood the problem.",
        "improvement": "Use the domain-specific vocabulary — terms like these signal to the interviewer that you've worked with this in a real environment, not just read about it.",
        "micro_tip": "Write down the five most important technical terms for each topic area you're interviewing on — make sure each appears naturally in your answer.",
    },
]


def _detect_dominant_weakness(
    ev:          Dict,
    q_type:      str,
    score:       float,
) -> str:
    """
    Inspect the evaluation dict and return the single most impactful
    weakness slug. Priority order mirrors _pick_probe_strategy() in
    follow_up_engine.py for consistency across the coaching pipeline.

    Returns one of:
      missing_result | vague_action | low_depth | no_example |
      high_filler | too_brief | too_long | grammar |
      low_relevance | shallow_star | missing_keywords | generic
    """
    qt          = q_type.lower().strip()
    star_scores = ev.get("star_scores",  {})
    star_count  = ev.get("star_count",   0)
    tfidf_sim   = ev.get("tfidf_sim",    0.5)
    depth_sc    = ev.get("depth_score",  3.0)
    wc          = ev.get("word_count",   100)
    filler_r    = ev.get("filler_ratio", 0.0)
    grammar_sc  = ev.get("grammar_score", 80.0)
    kw_details  = ev.get("keyword_details", [])

    # 1. Missing Result — highest impact gap for behavioural / hr
    if qt in ("behavioural", "hr"):
        if not star_scores.get("Result") and (
            star_scores.get("Situation") or star_scores.get("Task")
        ):
            return "missing_result"

    # 2. Off-topic answer
    if tfidf_sim < 0.28:
        return "low_relevance"

    # 3. No personal contribution (vague action)
    if (star_scores.get("Situation") or star_scores.get("Task")) \
            and not star_scores.get("Action") \
            and qt in ("behavioural", "hr"):
        return "vague_action"

    # 4. Too brief to evaluate properly
    wc_min = {"technical": 60, "behavioural": 50, "hr": 40}.get(qt, 50)
    if wc < wc_min or depth_sc < 1.8:
        if qt == "technical":
            return "low_depth"
        return "no_example" if qt in ("behavioural", "hr") else "too_brief"

    # 5. High filler
    if filler_r > 0.08:
        return "high_filler"

    # 6. Too long
    wc_max = {"technical": 500, "behavioural": 300, "hr": 280}.get(qt, 300)
    if wc > wc_max:
        return "too_long"

    # 7. Shallow STAR (present but thin)
    if qt in ("behavioural", "hr") and star_count < 2:
        return "shallow_star"

    # 8. Low technical depth (correct but surface-level)
    if qt == "technical" and depth_sc < 3.0:
        return "low_depth"

    # 9. Missing technical keywords
    if kw_details:
        missed = [d for d in kw_details if not d.get("matched")]
        if len(missed) / max(1, len(kw_details)) >= 0.5:
            return "missing_keywords"

    # 10. Grammar
    if grammar_sc < 60:
        return "grammar"

    return "generic"


def _rag_retrieve_coaching(
    weakness:         str,
    q_type:           str,
    ev:               Dict,
    top_k:            int = 2,
) -> list:
    """
    Retrieve the top-k most semantically relevant coaching interventions
    for this candidate's dominant weakness using sentence-transformer
    cosine similarity.

    The query is built from the weakness slug + key eval signals so
    retrieval is specific to what this candidate actually did, not just
    the generic category.

    Falls back to strategy-filtered exact match if sentence-transformers
    is not installed — still returns relevant playbook entries.

    Parameters
    ----------
    weakness  : slug from _detect_dominant_weakness()
    q_type    : "technical" | "behavioural" | "hr"
    ev        : full AnswerEvaluator result dict (used to build query)
    top_k     : number of entries to return (default 2)

    Returns
    -------
    List of coaching playbook dicts ordered by relevance.
    """
    import random

    # ── Step 1: filter by weakness + q_type ──────────────────────────────────
    pool = [
        p for p in _RAG_COACHING_PLAYBOOK
        if p["weakness"] == weakness
        and (p["q_type"] == q_type or p["q_type"] == "any")
    ]
    # Relax q_type filter if pool is too small
    if not pool:
        pool = [p for p in _RAG_COACHING_PLAYBOOK if p["weakness"] == weakness]
    if not pool:
        pool = [p for p in _RAG_COACHING_PLAYBOOK if p["weakness"] == "no_example"]

    if len(pool) <= top_k:
        return pool

    # ── Step 2: build a rich query from the candidate's actual eval signals ───
    # Using real signal text (not just the slug) means retrieval surfaces the
    # entry whose 'signal' description most closely matches what happened.
    filler_r   = ev.get("filler_ratio",  0.0)
    wc         = ev.get("word_count",    0)
    star_count = ev.get("star_count",    0)
    depth_sc   = ev.get("depth_score",   0.0)
    tfidf_sim  = ev.get("tfidf_sim",     0.5)

    query_parts = [f"candidate weakness: {weakness}"]
    if filler_r > 0.06:
        query_parts.append(f"filler word ratio {filler_r:.0%}")
    if wc < 60:
        query_parts.append("very short answer")
    elif wc > 400:
        query_parts.append("excessively long answer")
    if star_count < 2 and q_type in ("behavioural", "hr"):
        query_parts.append("incomplete STAR structure")
    if depth_sc < 2.5:
        query_parts.append("low depth score shallow explanation")
    if tfidf_sim < 0.3:
        query_parts.append("low relevance off topic answer")
    query = " ".join(query_parts)

    # ── Step 3: semantic ranking ──────────────────────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        # Reuse model if already loaded (shared with answer_evaluator / follow_up_engine)
        if not hasattr(_rag_retrieve_coaching, "_model"):
            _rag_retrieve_coaching._model = SentenceTransformer(
                "paraphrase-MiniLM-L6-v2"
            )
        model = _rag_retrieve_coaching._model

        query_emb = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Embed each entry's 'signal' field — that's the semantic anchor
        pool_texts = [p["signal"] for p in pool]
        pool_embs  = model.encode(
            pool_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sims        = (pool_embs @ query_emb.T).flatten()
        top_indices = __import__("numpy").argsort(sims)[::-1][:top_k]
        retrieved   = [pool[i] for i in top_indices]
        import logging
        logging.getLogger("LiveCoach").info(
            f"RAG coaching retrieval: weakness={weakness}, q_type={q_type}, "
            f"scores={[round(float(sims[i]), 3) for i in top_indices]}"
        )
        return retrieved

    except Exception as exc:
        import logging
        logging.getLogger("LiveCoach").debug(
            f"RAG coaching semantic retrieval skipped ({exc}), using random sample."
        )
        return random.sample(pool, min(top_k, len(pool)))


def _groq_coaching_tip(
    answer:        str,
    question:      str,
    q_type:        str,
    score:         float,
    feedback:      str,
    star_scores:   Dict[str, bool],
    keyword_hits:  List[str],
    missing_kws:   List[str],
    filler_count:  int,
    word_count:    int,
    groq_api_key:  str,
    rag_examples:  list = None,   # RAG-retrieved coaching interventions
) -> Optional[str]:
    """
    Ask Groq to produce a 2–3 sentence spoken coaching tip.
    rag_examples — list of coaching playbook dicts from _rag_retrieve_coaching().
    When provided, they are injected as few-shot grounding so Groq produces
    coach-quality phrasing specific to this weakness pattern rather than
    generic LLM feedback.
    Returns None if the API call fails.
    """
    try:
        from groq import Groq
    except ImportError:
        return None

    if not groq_api_key:
        return None

    score_10     = round(score * 2, 1)
    star_missing = [k for k, v in star_scores.items() if not v]
    star_present = [k for k, v in star_scores.items() if v]
    # v9.2: STAR is not a valid framework for Technical questions — don't
    # tell the coach model to mention missing STAR components for them.
    _star_relevant = q_type.lower() in ("behavioural", "behavioral", "hr", "")
    star_missing_str = (', '.join(star_missing) if star_missing and _star_relevant
                        else 'N/A — technical question, STAR not required')
    star_present_str = (', '.join(star_present) if star_present else 'none')

    # ── RAG grounding block ───────────────────────────────────────────────────
    # Build a few-shot style block from the retrieved coaching playbook entries.
    # Each entry shows: what the weakness looks like + how an expert coach
    # would phrase the praise, core improvement, and micro-tip.
    # This steers Groq toward concrete, specific coaching phrasing and away
    # from generic "work on your STAR framework" boilerplate.
    rag_block = ""
    if rag_examples:
        lines = []
        for i, ex in enumerate(rag_examples, 1):
            lines.append(
                f"  Coaching example {i} (weakness: {ex['weakness']}, "
                f"signal: {ex['signal']}):\n"
                f"    Praise:      {ex['praise']}\n"
                f"    Improvement: {ex['improvement']}\n"
                f"    Micro-tip:   {ex['micro_tip']}"
            )
        rag_block = (
            "\n\nCOACHING STYLE REFERENCE — expert coach interventions for this weakness:\n"
            + "\n\n".join(lines)
            + "\n\nAdapt this style and specificity to what THIS candidate actually said. "
            "Do NOT copy these examples verbatim — personalise them to the answer above."
        )

    prompt = f"""You are a supportive but honest interview coach giving spoken feedback to a candidate immediately after they answered a question. Speak directly to them in second person ("You...").

Question asked: {question}
Question type: {q_type}
Candidate's answer (first 400 chars): {answer[:400]}
Score: {score_10}/10
NLP system feedback: {feedback}
STAR components present: {star_present_str}
STAR components missing: {star_missing_str}
Keywords covered: {', '.join(keyword_hits[:4]) if keyword_hits else 'none'}
Keywords missing: {', '.join(missing_kws[:3]) if missing_kws else 'none'}
Filler words detected: {filler_count}
Word count: {word_count}{rag_block}

Write EXACTLY 2–3 sentences of spoken coaching feedback. Rules:
- Start with one specific thing they did well (even if the score is low, find something)
- Then give the single most important improvement they should make RIGHT NOW
- End with one concrete, actionable micro-tip they can apply on the very next answer
- Sound like a supportive human coach, not a scoring system
- Do NOT mention numbers, scores, or percentages
- Do NOT use bullet points or headers — plain spoken sentences only
- Keep it under 60 words total"""

    import time
    client = Groq(api_key=groq_api_key)
    messages = [
        {
            "role":    "system",
            "content": (
                "You are a warm, direct interview coach giving real-time spoken "
                "feedback. You always find something to praise, then give one "
                "clear improvement. Plain sentences only — no lists, no scores."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    # Exponential backoff — mirrors answer_evaluator._groq_call() strategy.
    # Rate-limit (429): wait 2^attempt seconds then retry.
    # Server error (5xx): wait 1.5×attempt seconds then retry.
    # Auth/bad-request (401/400/403): permanent — no retry.
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model       = _GROQ_MODEL,
                messages    = messages,
                max_tokens  = 120,
                temperature = 0.72,
            )
            tip = resp.choices[0].message.content.strip()
            tip = re.sub(r"\*+", "", tip).strip()
            tip = re.sub(r"#+\s*", "", tip).strip()
            return tip if len(tip.split()) >= 10 else None

        except Exception as exc:
            msg = str(exc)
            is_rate_limit = "429" in msg or "rate_limit" in msg.lower()
            is_server_err = any(c in msg for c in ("500", "502", "503"))
            is_permanent  = any(c in msg for c in ("401", "400", "403"))

            if is_permanent:
                print(f"[LiveCoach] Groq permanent error (attempt {attempt}): {exc}")
                return None

            if attempt < max_attempts:
                wait = (2 ** attempt) if is_rate_limit else (1.5 * attempt)
                print(f"[LiveCoach] Groq error (attempt {attempt}/{max_attempts}), "
                      f"retrying in {wait:.1f}s: {exc}")
                time.sleep(wait)
            else:
                print(f"[LiveCoach] Groq tip failed after {max_attempts} attempts: {exc}")

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  RULE-BASED FALLBACK COACHING GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def _rule_based_coaching_tip(
    q_type:       str,
    score:        float,
    star_scores:  Dict[str, bool],
    keyword_hits: List[str],
    missing_kws:  List[str],
    filler_count: int,
    word_count:   int,
    feedback:     str,
    grammar_score: float,
) -> str:
    """
    Deterministic coaching tip built from the evaluator's signals.
    Used when Groq is unavailable. Always produces a warm, useful message.
    """
    qt = q_type.lower()
    star_missing = [k for k, v in star_scores.items() if not v]
    star_present = [k for k, v in star_scores.items() if v]
    n_star       = len(star_present)

    # ── Praise (always find something) ───────────────────────────────────────
    if score >= 4.0:
        praise = "That was a strong answer — you covered the core ideas confidently."
    elif n_star >= 3:
        praise = f"Good structure — you clearly included the {', '.join(star_present[:2])} components."
    elif keyword_hits:
        praise = f"You touched on the right concepts — mentioning {keyword_hits[0]} shows solid awareness."
    elif word_count >= 80:
        praise = "You gave a detailed response, which shows engagement with the question."
    elif score >= 2.5:
        praise = "You made a reasonable attempt and stayed on topic."
    else:
        praise = "You answered the question, which is a good starting point to build from."

    # ── Primary improvement (pick the most impactful gap) ────────────────────
    improvement = ""

    if qt in ("behavioural", "hr") and star_missing:
        if "Result" in star_missing:
            improvement = (
                "The biggest thing missing was a concrete result — "
                "always close your story with what actually happened: "
                "a number, a deadline met, or feedback you received."
            )
        elif "Action" in star_missing:
            improvement = (
                "Make sure to be specific about what YOU personally did — "
                "interviewers want to hear your individual actions, "
                "not just what the team did."
            )
        elif "Situation" in star_missing:
            improvement = (
                "Set the scene first — briefly explain the context "
                "before diving into what you did, so the interviewer "
                "understands why the situation mattered."
            )
        else:
            improvement = (
                f"Your answer was missing the {star_missing[0]} part of the STAR framework — "
                f"add that next time to give your answer a complete structure."
            )

    elif missing_kws and qt == "technical":
        kw_str = " and ".join(f'"{k}"' for k in missing_kws[:2])
        improvement = (
            f"To strengthen this answer, work in {kw_str} — "
            "those are the technical terms that signal real depth "
            "to a senior interviewer."
        )

    elif filler_count >= 5:
        improvement = (
            f"You used around {filler_count} filler words like 'um' or 'like' — "
            "try pausing silently instead. A confident pause reads much better "
            "than a filler word."
        )

    elif word_count < 60:
        opt = {"technical": "150 to 250", "behavioural": "120 to 180", "hr": "100 to 160"}.get(qt, "120 to 200")
        improvement = (
            f"Your answer was quite brief — aim for around {opt} words "
            "to give the interviewer enough context to evaluate your experience."
        )

    elif word_count > 350 and qt != "technical":
        improvement = (
            "The answer was on the longer side — practice editing it down to "
            "the three most important points. Concise answers are memorable."
        )

    elif grammar_score < 60:
        improvement = (
            "Watch your sentence structure when you speak — "
            "clean, complete sentences signal confidence and professionalism "
            "far more than the content alone."
        )

    else:
        # Generic improvement based on score band
        if score < 2.5:
            improvement = (
                "Focus on giving a specific example from your experience "
                "rather than a general answer — concrete stories are always "
                "more convincing than abstract statements."
            )
        elif score < 3.5:
            improvement = (
                "Add one measurable outcome to your answer — a percentage, "
                "a timeline, or a direct result — to make your impact tangible."
            )
        else:
            improvement = (
                "To push this from good to excellent, add a brief reflection "
                "on what you learned — it shows self-awareness and maturity."
            )

    # ── Micro-tip for the next question ──────────────────────────────────────
    if qt in ("behavioural", "hr"):
        micro = "On your next answer, say the word 'result' near the end to anchor your outcome clearly."
    elif qt == "technical":
        micro = "Next time, try mentioning one trade-off or limitation — it signals senior thinking."
    else:
        micro = "Keep that momentum — you're building a clearer pattern with each answer."

    return f"{praise} {improvement} {micro}"


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def generate_coaching_tip(
    ev:           Dict,
    q_type:       str   = "Technical",
    score:        float = 2.5,
    question:     str   = "",
    answer:       str   = "",
    groq_api_key: str   = "",
) -> str:
    """
    Generate a 2–3 sentence spoken coaching tip from an AnswerEvaluator result.

    Args:
        ev            : dict returned by AnswerEvaluator.evaluate()
        q_type        : "Technical" | "Behavioural" | "HR"
        score         : raw score on 0–5 scale
        question      : the original question text (for Groq context)
        answer        : the candidate's answer text (for Groq context)
        groq_api_key  : if provided, uses Groq for a natural coaching tip

    Returns:
        A plain-text coaching message (2–3 sentences, ~40–60 words).
    """
    star_scores   = ev.get("star_scores",  {})
    keyword_hits  = ev.get("keyword_hits", [])
    all_kws       = ev.get("keyword_hits", [])   # already-hit kws
    # Reconstruct missing kws from feedback string if not stored separately
    feedback      = ev.get("feedback", "")
    filler_count  = ev.get("filler_count",  0)
    word_count    = ev.get("word_count",    0)
    grammar_score = ev.get("grammar_score", 80.0)

    # Try to infer missing keywords from feedback text
    missing_kws: List[str] = []
    kw_match = re.search(r"Cover key concepts: ([^.]+)\.", feedback)
    if kw_match:
        missing_kws = [k.strip() for k in kw_match.group(1).split(",")]

    # ── RAG: detect dominant weakness + retrieve coaching interventions ───────
    # Run for BOTH the Groq path (as grounding) and the rule-based fallback
    # (as the primary source of praise/improvement/micro-tip text).
    qt_lower  = q_type.lower()
    weakness  = _detect_dominant_weakness(ev, qt_lower, score)
    rag_entries = _rag_retrieve_coaching(
        weakness = weakness,
        q_type   = qt_lower,
        ev       = ev,
        top_k    = 2,
    )

    # Try Groq first — pass RAG entries as grounding examples
    if groq_api_key and question and answer:
        tip = _groq_coaching_tip(
            answer        = answer,
            question      = question,
            q_type        = q_type,
            score         = score,
            feedback      = feedback,
            star_scores   = star_scores,
            keyword_hits  = keyword_hits,
            missing_kws   = missing_kws,
            filler_count  = filler_count,
            word_count    = word_count,
            groq_api_key  = groq_api_key,
            rag_examples  = rag_entries,
        )
        if tip:
            return tip

    # ── RAG-powered fallback (replaces pure rule-based) ───────────────────────
    # If Groq is unavailable, compose the tip directly from the best-matching
    # playbook entry — guaranteed coach-quality phrasing, no generic templates.
    if rag_entries:
        best = rag_entries[0]
        tip  = f"{best['praise']} {best['improvement']} {best['micro_tip']}"
        return tip

    # Final safety net — original rule-based (Groq down + no sentence-transformers)
    return _rule_based_coaching_tip(
        q_type        = q_type,
        score         = score,
        star_scores   = star_scores,
        keyword_hits  = keyword_hits,
        missing_kws   = missing_kws,
        filler_count  = filler_count,
        word_count    = word_count,
        feedback      = feedback,
        grammar_score = grammar_score,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TTS COMPONENT
# ══════════════════════════════════════════════════════════════════════════════

def render_tts_button(text: str, key: str = "tts_btn", auto_speak: bool = False) -> None:
    """
    Inject a browser Web Speech API TTS button via st.components.html.
    Uses window.speechSynthesis — zero extra packages, works in any browser.

    Args:
        text       : The text to speak.
        key        : Unique key for the component (avoid collisions across questions).
        auto_speak : If True, speech fires automatically on render (no button click).
    """
    if not text:
        return

    # Sanitise text for safe JS string insertion
    safe_text = (text
                 .replace("\\", "\\\\")
                 .replace('"',  '\\"')
                 .replace("\n", " ")
                 .replace("\r", ""))

    rate  = st.session_state.get("coach_voice_rate",  1.0)
    pitch = st.session_state.get("coach_voice_pitch", 1.0)
    auto  = "true" if auto_speak else "false"

    html = f"""
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:transparent;}}
  #tts-wrap{{
    display:flex;align-items:center;gap:10px;
    padding:6px 0;
  }}
  #speak-btn{{
    display:flex;align-items:center;gap:6px;
    background:linear-gradient(135deg,rgba(124,107,255,.18),rgba(59,139,255,.12));
    border:1px solid rgba(124,107,255,.4);
    border-radius:8px;padding:7px 14px;
    color:#b4a9ff;font-size:.75rem;font-weight:700;
    cursor:pointer;font-family:inherit;
    transition:all .18s;letter-spacing:.04em;
    text-transform:uppercase;
  }}
  #speak-btn:hover{{
    background:linear-gradient(135deg,rgba(124,107,255,.3),rgba(59,139,255,.2));
    border-color:rgba(124,107,255,.7);color:#d0caff;
    transform:translateY(-1px);
  }}
  #speak-btn.speaking{{
    background:linear-gradient(135deg,rgba(34,216,122,.15),rgba(0,212,200,.1));
    border-color:rgba(34,216,122,.5);color:#22d87a;
    animation:pulse .9s ease-in-out infinite;
  }}
  #speak-btn.done{{
    border-color:rgba(90,112,152,.3);color:#5a7098;
    background:rgba(255,255,255,.03);
  }}
  #stop-btn{{
    display:none;align-items:center;
    background:rgba(255,92,92,.1);border:1px solid rgba(255,92,92,.3);
    border-radius:8px;padding:7px 12px;
    color:#ff7070;font-size:.72rem;font-weight:700;
    cursor:pointer;font-family:inherit;transition:all .15s;
    text-transform:uppercase;
  }}
  #stop-btn.visible{{display:flex;}}
  #stop-btn:hover{{background:rgba(255,92,92,.2);}}
  #tts-status{{
    font-size:.6rem;color:#5a7098;font-family:Share Tech Mono,monospace;
    letter-spacing:.05em;min-width:60px;
  }}
  @keyframes pulse{{0%,100%{{opacity:1;}}50%{{opacity:.6;}}}}
  .dot{{
    width:7px;height:7px;border-radius:50%;background:currentColor;
    flex-shrink:0;
  }}
</style>

<div id="tts-wrap">
  <button id="speak-btn" onclick="speakNow()">
    <span class="dot"></span> Speak Feedback
  </button>
  <button id="stop-btn" onclick="stopNow()">
    ■ Stop
  </button>
  <span id="tts-status"></span>
</div>

<script>
(function(){{
  const TEXT  = "{safe_text}";
  const RATE  = {rate};
  const PITCH = {pitch};
  const AUTO  = {auto};
  const btn   = document.getElementById('speak-btn');
  const stop  = document.getElementById('stop-btn');
  const stat  = document.getElementById('tts-status');
  let   utt   = null;

  function speakNow(){{
    if(!window.speechSynthesis){{
      stat.textContent = "TTS not supported";
      return;
    }}
    window.speechSynthesis.cancel();
    utt = new SpeechSynthesisUtterance(TEXT);
    utt.rate  = RATE;
    utt.pitch = PITCH;
    utt.lang  = 'en-US';

    // Prefer a natural English voice if available
    const voices = window.speechSynthesis.getVoices();
    const pref   = voices.find(v =>
      (v.name.includes('Google') || v.name.includes('Natural') || v.name.includes('Samantha'))
      && v.lang.startsWith('en')
    ) || voices.find(v => v.lang.startsWith('en'));
    if(pref) utt.voice = pref;

    utt.onstart = ()=>{{
      btn.className  = 'speaking';
      btn.innerHTML  = '<span class="dot"></span> Speaking…';
      stop.className = 'visible';
      stat.textContent = '';
    }};
    utt.onend = ()=>{{
      btn.className  = 'done';
      btn.innerHTML  = '<span class="dot"></span> Spoken';
      stop.className = '';
      stat.textContent = '✓ done';
    }};
    utt.onerror = (e)=>{{
      btn.className  = '';
      btn.innerHTML  = '<span class="dot"></span> Speak Feedback';
      stop.className = '';
      stat.textContent = 'error';
    }};

    window.speechSynthesis.speak(utt);
  }}

  function stopNow(){{
    window.speechSynthesis.cancel();
    btn.className  = '';
    btn.innerHTML  = '<span class="dot"></span> Speak Feedback';
    stop.className = '';
    stat.textContent = '';
  }}

  // Voices load asynchronously in Chrome
  if(window.speechSynthesis.onvoiceschanged !== undefined){{
    window.speechSynthesis.onvoiceschanged = ()=>{{
      if(AUTO && btn.className !== 'speaking'){{ speakNow(); }}
    }};
  }}

  if(AUTO){{
    setTimeout(()=>{{ speakNow(); }}, 400);
  }}
}})();
</script>
"""
    components.html(html, height=48)


# ══════════════════════════════════════════════════════════════════════════════
#  COACH CARD RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def render_coach_card(
    coaching_tip: str,
    ev:           Dict,
    score:        float,
    q_number:     int  = 0,
) -> None:
    """
    Render the live coaching card below the evaluation results.

    Shows:
      • Animated coach avatar + "Live Coach" label
      • Colour-coded score pill
      • The coaching message (2–3 sentences)
      • Spoken feedback button (Web Speech API)
      • Key improvement signals as mini chips

    Args:
        coaching_tip : The coaching message from generate_coaching_tip()
        ev           : The full AnswerEvaluator result dict
        score        : Raw score 0–5
        q_number     : Current question number (used to namespace the TTS key)
    """
    if not coaching_tip:
        # Generate a rule-based tip on the fly instead of showing nothing
        coaching_tip = _rule_based_coaching_tip(
            q_type        = ev.get("question_type", "technical"),
            score         = score,
            star_scores   = ev.get("star_scores", {}),
            keyword_hits  = ev.get("keyword_hits", []),
            missing_kws   = [],
            filler_count  = ev.get("filler_count", 0),
            word_count    = ev.get("word_count", 0),
            feedback      = ev.get("feedback", ""),
            grammar_score = ev.get("grammar_score", 80.0),
        )
        if not coaching_tip:
            return

    score_10 = round(score * 2, 1)

    # Score colour
    if score >= 4.0:
        sc_col, sc_label, sc_bg = "#22d87a", "Strong",    "rgba(34,216,122,.12)"
    elif score >= 3.0:
        sc_col, sc_label, sc_bg = "#ffb840", "Good",      "rgba(255,184,64,.1)"
    elif score >= 2.0:
        sc_col, sc_label, sc_bg = "#ff9040", "Fair",      "rgba(255,144,64,.1)"
    else:
        sc_col, sc_label, sc_bg = "#ff5c5c", "Needs Work","rgba(255,92,92,.1)"

    # Build improvement chips from evaluator signals
    chips_html = ""
    chip_items: List[str] = []

    star_scores  = ev.get("star_scores", {})
    q_type_key   = ev.get("question_type", "").lower()
    # v9.2: never flag missing STAR for Technical — it is not a story format
    _star_chip_ok = q_type_key in ("behavioural", "behavioral", "hr", "")
    missing_star = [k for k, v in star_scores.items() if not v]
    if missing_star and _star_chip_ok:
        chip_items.append(f"Missing STAR: {', '.join(missing_star)}")

    filler = ev.get("filler_count", 0)
    if filler >= 4:
        chip_items.append(f"{filler} filler words")

    wc = ev.get("word_count", 0)
    if wc < 60:
        chip_items.append("Too brief")
    elif wc > 350:
        chip_items.append("Too long")

    grammar = ev.get("grammar_score", 100)
    if grammar < 65:
        chip_items.append("Grammar issues")

    for chip in chip_items[:3]:
        chips_html += (
            f'<span style="background:rgba(255,184,64,.08);'
            f'border:1px solid rgba(255,184,64,.25);'
            f'color:#ffb840;border-radius:4px;'
            f'padding:2px 8px;font-size:.6rem;'
            f'font-family:Share Tech Mono,monospace;'
            f'white-space:nowrap;">'
            f'{chip}</span>'
        )

    st.markdown(
        f"""
<div style="
  background:linear-gradient(135deg,rgba(12,20,45,.95),rgba(4,8,22,.98));
  border:1px solid rgba(124,107,255,.3);
  border-left:3px solid #7c6bff;
  border-radius:0 14px 14px 0;
  padding:1rem 1.2rem;
  margin:.6rem 0;
  position:relative;
  overflow:hidden;
">
  <!-- Subtle background glow -->
  <div style="
    position:absolute;top:-30px;right:-30px;
    width:120px;height:120px;border-radius:50%;
    background:radial-gradient(circle,rgba(124,107,255,.08),transparent 70%);
    pointer-events:none;
  "></div>

  <!-- Header row -->
  <div style="display:flex;align-items:center;justify-content:space-between;
              flex-wrap:wrap;gap:.5rem;margin-bottom:.7rem;">
    <div style="display:flex;align-items:center;gap:.55rem;">
      <!-- Coach avatar -->
      <div style="
        width:32px;height:32px;border-radius:8px;flex-shrink:0;
        background:linear-gradient(135deg,#7c6bff,#3b8bff);
        display:flex;align-items:center;justify-content:center;
        font-size:.95rem;
      ">◈</div>
      <div>
        <div style="font-size:.65rem;font-weight:700;color:#7c6bff;
                    font-family:Share Tech Mono,monospace;letter-spacing:.1em;">
          LIVE COACH
        </div>
        <div style="font-size:.55rem;color:#5a7098;
                    font-family:Share Tech Mono,monospace;">
          Q{q_number} · Improvement Feedback
        </div>
      </div>
    </div>
    <!-- Score badge -->
    <div style="
      background:{sc_bg};border:1px solid {sc_col}55;
      border-radius:8px;padding:4px 12px;text-align:center;
    ">
      <div style="font-size:1.1rem;font-weight:900;color:{sc_col};
                  font-family:Orbitron,monospace;line-height:1.1;">
        {score_10}<span style="font-size:.55rem;color:{sc_col}99;">/10</span>
      </div>
      <div style="font-size:.55rem;color:{sc_col}99;
                  font-family:Share Tech Mono,monospace;">{sc_label}</div>
    </div>
  </div>

  <!-- Coaching message -->
  <div style="
    font-size:.84rem;color:#c8d8f0;line-height:1.72;
    margin-bottom:.75rem;
    font-style:italic;
  ">"{coaching_tip}"</div>

  <!-- Improvement chips -->
  {f'<div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:.6rem;">{chips_html}</div>' if chips_html else ''}

</div>""",
        unsafe_allow_html=True,
    )

    # TTS button rendered as a separate HTML component below the card
    auto_speak = st.session_state.get("coach_auto_speak", False)
    render_tts_button(coaching_tip, key=f"tts_{q_number}", auto_speak=auto_speak)


# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS PANEL (call from page_settings)
# ══════════════════════════════════════════════════════════════════════════════

def render_coach_settings() -> None:
    """
    Render coaching settings controls.
    Call this inside page_settings() to let users configure TTS behaviour.
    """
    st.markdown(
        '<h4 style="margin-top:0;font-size:.85rem;">◈ LIVE COACH SETTINGS</h4>',
        unsafe_allow_html=True,
    )

    st.toggle(
        "Auto-speak feedback after each answer",
        value   = st.session_state.get("coach_auto_speak", False),
        key     = "coach_auto_speak",
        help    = "Spoken feedback plays automatically when you submit an answer.",
    )

    st.slider(
        "Speech rate",
        min_value = 0.6,
        max_value = 1.4,
        value     = st.session_state.get("coach_voice_rate", 1.0),
        step      = 0.1,
        key       = "coach_voice_rate",
        help      = "0.6 = slow and clear · 1.0 = normal · 1.4 = fast",
    )

    st.slider(
        "Speech pitch",
        min_value = 0.7,
        max_value = 1.3,
        value     = st.session_state.get("coach_voice_pitch", 1.0),
        step      = 0.1,
        key       = "coach_voice_pitch",
        help      = "Lower = deeper voice · Higher = lighter voice",
    )

    st.markdown(
        '<div style="font-size:.72rem;color:#5a7098;margin:.8rem 0 .3rem;">'
        '⏱ MID-ANSWER NUDGE THRESHOLDS</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "The coach speaks a short audio nudge while you're still recording "
        "if you cross one of these thresholds.  Requires Auto-speak to be on."
    )

    st.slider(
        "Running-long nudge (seconds)",
        min_value = 45,
        max_value = 150,
        value     = st.session_state.get("nudge_long_sec", 90),
        step      = 15,
        key       = "nudge_long_sec",
        help      = "Nudge fires if you're still talking after this many seconds.",
    )

    st.slider(
        "Very-long nudge (seconds)",
        min_value = 90,
        max_value = 240,
        value     = st.session_state.get("nudge_very_long_sec", 150),
        step      = 15,
        key       = "nudge_very_long_sec",
        help      = "Second, more urgent nudge to wrap up.",
    )

    st.slider(
        "Filler-word nudge threshold (%)",
        min_value = 5,
        max_value = 25,
        value     = int(st.session_state.get("nudge_filler_pct", 10)),
        step      = 1,
        key       = "nudge_filler_pct",
        help      = "Nudge fires when filler words exceed this % of total words.",
    )

    st.slider(
        "Nudge cooldown (seconds)",
        min_value = 15,
        max_value = 60,
        value     = st.session_state.get("nudge_cooldown_sec", 30),
        step      = 5,
        key       = "nudge_cooldown_sec",
        help      = "Minimum gap between any two consecutive nudges.",
    )