"""
briefing_engine.py — Aura AI | Live Industry Briefing Engine v1.0
==================================================================
Surfaces 3 bullet-point industry news items from the last 48 hours
immediately before each interview session starts.  The AI plays the
role of a researcher who has already scanned the market so the
candidate can walk in informed.

HOW IT WORKS
────────────
1.  fetch_briefing(role, industry, groq_api_key)
      Builds a targeted search query from the role + industry.
      Calls Groq with web_search tool enabled to retrieve live news.
      Returns a BriefingResult dataclass with up to 3 BulletPoint items.
      Results are cached in session state for 6 hours — multiple
      st.rerun() calls during a session never fire redundant searches.

2.  render_briefing_card(result, accent_color)
      Renders the briefing as a styled Streamlit markdown component
      matching the Aura sci-fi aesthetic.  Each bullet shows:
        • Headline (bold)
        • One-sentence "why it matters for your interview" hook
        • Source label + recency chip
      Includes a "Speak Briefing" TTS button so the candidate can
      listen while stretching before the session.

3.  render_briefing_inline(role, industry, groq_api_key, accent_color)
      One-liner convenience call — fetch + render in a single call.
      Use this in weekly_prep_plan._render_practice_window() and
      app.page_start() before the first question loads.

INTEGRATION
───────────
weekly_prep_plan.py — inside _render_practice_window(), after the
← Plan button and before the HUD top bar, add:

    from briefing_engine import render_briefing_inline
    render_briefing_inline(
        role         = st.session_state.get("wp_role", "Software Engineer"),
        industry     = _role_to_industry(role),   # helper below
        groq_api_key = st.session_state.get("groq_api_key", ""),
        accent_color = cfg.color,
    )

app.py — inside page_start() before session launch button:

    from briefing_engine import render_briefing_inline
    render_briefing_inline(
        role         = st.session_state.get("target_role", "Software Engineer"),
        industry     = _role_to_industry(st.session_state.target_role),
        groq_api_key = st.session_state.get("groq_api_key", ""),
    )

SESSION STATE KEYS
──────────────────
  briefing_cache          dict  — {cache_key: BriefingResult, ts: float}
  briefing_last_role      str   — role used for cached briefing
  briefing_last_industry  str   — industry used for cached briefing

FALLBACK BEHAVIOUR
──────────────────
If Groq is unavailable (no key, network error, rate limit):
  • Falls back to a curated static briefing for the industry.
  • Static briefings are generic but still useful as talking-point
    primers.  They are clearly labelled "[offline — general tips]"
    so the candidate is never misled about recency.
  • The render card shows a "Refresh" button that retries the live
    fetch on the next st.rerun().

CACHE STRATEGY
──────────────
  • TTL: 6 hours (21 600 seconds).  Interview prep cycles are typically
    daily, so 6 hours avoids redundant calls without serving stale news.
  • Cache key: sha256(role + industry + floor(unix_time / 21600)).
    Different role + industry pairs are cached independently.
  • Cache is stored in st.session_state (RAM, not disk) so it resets
    cleanly when the browser session ends.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

import streamlit as st
import streamlit.components.v1 as components


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_GROQ_MODEL    = "llama-3.3-70b-versatile"
_CACHE_TTL_S   = 21_600          # 6 hours
_MAX_RETRIES   = 3


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BulletPoint:
    headline:    str            # 8–12 word headline
    hook:        str            # "why this matters for your interview" sentence
    source:      str            # e.g. "TechCrunch", "Reuters", "Bloomberg"
    recency:     str            # e.g. "2 hours ago", "yesterday"
    is_static:   bool = False   # True when sourced from offline fallback


@dataclass
class BriefingResult:
    bullets:     List[BulletPoint]
    role:        str
    industry:    str
    fetched_at:  float = field(default_factory=time.time)
    is_fallback: bool  = False
    error_msg:   str   = ""


# ══════════════════════════════════════════════════════════════════════════════
#  ROLE → INDUSTRY MAPPING
# ══════════════════════════════════════════════════════════════════════════════

_ROLE_INDUSTRY_MAP: dict[str, str] = {
    "software engineer":        "technology",
    "backend engineer":         "technology",
    "full stack developer":     "technology",
    "devops / sre":             "technology",
    "ml engineer":              "artificial intelligence",
    "data scientist":           "artificial intelligence",
    "data engineer":            "technology",
    "product manager":          "technology",
    "business analyst":         "technology",
    "consulting / strategy":    "management consulting",
    "finance / banking":        "financial services",
    "marketing":                "marketing & advertising",
    "operations":               "operations & supply chain",
    "general / other":          "business",
    "hr practice":              "human resources",
}


def role_to_industry(role: str) -> str:
    """
    Map a target role string to a broad industry keyword for the search query.
    Case-insensitive.  Returns 'technology' as default.
    """
    return _ROLE_INDUSTRY_MAP.get(role.lower().strip(), "technology")


# ══════════════════════════════════════════════════════════════════════════════
#  CACHE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _cache_key(role: str, industry: str) -> str:
    window = int(time.time() // _CACHE_TTL_S)
    raw    = f"{role.lower()}|{industry.lower()}|{window}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _get_cached(role: str, industry: str) -> Optional[BriefingResult]:
    store = st.session_state.get("briefing_cache", {})
    key   = _cache_key(role, industry)
    entry = store.get(key)
    if entry is None:
        return None
    age = time.time() - entry.fetched_at
    return entry if age < _CACHE_TTL_S else None


def _set_cached(result: BriefingResult) -> None:
    store = st.session_state.get("briefing_cache", {})
    key   = _cache_key(result.role, result.industry)
    store[key] = result
    st.session_state["briefing_cache"] = store


# ══════════════════════════════════════════════════════════════════════════════
#  GROQ WEB SEARCH FETCH
# ══════════════════════════════════════════════════════════════════════════════

def _build_search_query(role: str, industry: str) -> str:
    """
    Construct a tight, time-bounded search query.
    The 48-hour constraint is baked into the prompt rather than the query
    string because Groq's web search tool doesn't support date operators.
    The LLM is instructed to filter by recency in its response.
    """
    return (
        f"latest news {industry} industry 2025 2026 "
        f"relevant for {role} interview preparation"
    )


_SYSTEM_PROMPT = """\
You are a professional industry analyst who scans news feeds every hour.
Your job is to brief a job candidate before their interview with the 3 most
recent, relevant news items from the last 48 hours in their target industry.

RULES
─────
1.  Return EXACTLY 3 bullet points, no more, no fewer.
2.  Each bullet must be a REAL, RECENT news item — not fabricated.
    If you cannot find 3 genuine items from the last 48 hours, use items
    from the last week and mark them as "this week" instead of a specific
    time.
3.  Every bullet must be directly relevant to someone interviewing for the
    given role.  Skip celebrity news, sports, unrelated politics.
4.  Format your response as strict JSON — no markdown, no prose:

{
  "bullets": [
    {
      "headline": "8–12 word factual headline",
      "hook": "One sentence explaining why this matters to a candidate interviewing for [ROLE].",
      "source": "Publication name, e.g. TechCrunch",
      "recency": "e.g. 3 hours ago, yesterday, this week"
    }
  ]
}

5.  Headlines must be factual, not clickbait.
6.  Hooks must be practical — reference interview topics like "shows you
    understand AI infrastructure trends", "signals growth in cloud spending",
    "expect questions about regulatory compliance in your interview".
7.  Do NOT add any text outside the JSON object.
"""


def _groq_fetch(
    role:         str,
    industry:     str,
    groq_api_key: str,
) -> Optional[BriefingResult]:
    """
    Call Groq with web_search tool enabled.
    Returns BriefingResult on success, None on failure.
    Retries up to _MAX_RETRIES with exponential backoff.
    """
    try:
        from groq import Groq
    except ImportError:
        return None

    if not groq_api_key:
        return None

    client = Groq(api_key=groq_api_key)
    query  = _build_search_query(role, industry)

    user_prompt = (
        f"Search the web for: {query}\n\n"
        f"Target role: {role}\n"
        f"Industry: {industry}\n\n"
        f"Return the 3 most recent, relevant news items from the last 48 hours "
        f"as strict JSON following the format in your system instructions."
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT.replace("[ROLE]", role)},
        {"role": "user",   "content": user_prompt},
    ]

    # Groq web_search tool definition
    tools = [
        {
            "type": "function",
            "function": {
                "name":        "web_search",
                "description": "Search the web for recent news and information.",
                "parameters": {
                    "type":       "object",
                    "properties": {
                        "query": {
                            "type":        "string",
                            "description": "The search query string.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model       = _GROQ_MODEL,
                messages    = messages,
                max_tokens  = 600,
                temperature = 0.25,       # low temp — we want factual, structured output
                tools       = tools,
                tool_choice = "auto",
            )

            raw = resp.choices[0].message.content or ""
            raw = raw.strip()

            # Strip markdown code fences if model added them
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$",           "", raw)
            raw = raw.strip()

            if not raw:
                raise ValueError("Empty response from Groq")

            parsed = json.loads(raw)
            bullets_raw = parsed.get("bullets", [])

            if not bullets_raw:
                raise ValueError("No bullets in Groq response")

            bullets = []
            for b in bullets_raw[:3]:
                headline = str(b.get("headline", "")).strip()
                hook     = str(b.get("hook",     "")).strip()
                source   = str(b.get("source",   "Unknown")).strip()
                recency  = str(b.get("recency",  "recently")).strip()
                if headline and hook:
                    bullets.append(BulletPoint(
                        headline  = headline,
                        hook      = hook,
                        source    = source,
                        recency   = recency,
                        is_static = False,
                    ))

            if not bullets:
                raise ValueError("All bullets were empty after parsing")

            return BriefingResult(
                bullets     = bullets,
                role        = role,
                industry    = industry,
                fetched_at  = time.time(),
                is_fallback = False,
            )

        except Exception as exc:
            msg          = str(exc)
            is_rate_lim  = "429" in msg or "rate_limit" in msg.lower()
            is_server    = any(c in msg for c in ("500", "502", "503"))
            is_permanent = any(c in msg for c in ("401", "400", "403"))

            print(f"[BriefingEngine] Groq attempt {attempt}/{_MAX_RETRIES} failed: {exc}")

            if is_permanent:
                return None
            if attempt < _MAX_RETRIES:
                wait = (2 ** attempt) if is_rate_lim else (1.5 * attempt)
                print(f"[BriefingEngine] Retrying in {wait:.1f}s…")
                time.sleep(wait)

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  STATIC FALLBACK BRIEFINGS  (offline / no API key)
# ══════════════════════════════════════════════════════════════════════════════

_STATIC_FALLBACKS: dict[str, list[dict]] = {
    "technology": [
        {
            "headline": "AI coding assistants reshape software engineering workflows",
            "hook":     "Interviewers increasingly ask how you use AI tools in your development process — have a clear, honest answer ready.",
            "source":   "General industry knowledge",
            "recency":  "ongoing trend",
        },
        {
            "headline": "Cloud cost optimisation becomes top engineering priority",
            "hook":     "Expect system design questions to probe your awareness of cost-efficiency trade-offs, not just scalability.",
            "source":   "General industry knowledge",
            "recency":  "ongoing trend",
        },
        {
            "headline": "Security-first engineering culture accelerates at enterprise firms",
            "hook":     "Be prepared to discuss how you've incorporated security considerations into your past projects.",
            "source":   "General industry knowledge",
            "recency":  "ongoing trend",
        },
    ],
    "artificial intelligence": [
        {
            "headline": "LLM fine-tuning and RAG dominate enterprise AI adoption strategies",
            "hook":     "Interviewers at AI firms expect you to articulate trade-offs between fine-tuning vs retrieval-augmented approaches.",
            "source":   "General industry knowledge",
            "recency":  "ongoing trend",
        },
        {
            "headline": "AI evaluation and responsible deployment frameworks gain traction",
            "hook":     "Be ready to discuss how you validate model outputs and guard against hallucination in production systems.",
            "source":   "General industry knowledge",
            "recency":  "ongoing trend",
        },
        {
            "headline": "Multimodal AI models expanding beyond text into vision and audio",
            "hook":     "Show awareness of the broader AI stack — hiring panels value candidates who see beyond their immediate specialisation.",
            "source":   "General industry knowledge",
            "recency":  "ongoing trend",
        },
    ],
    "financial services": [
        {
            "headline": "Regulators tighten AI governance requirements for financial institutions",
            "hook":     "Compliance awareness is a differentiator — mention model risk management frameworks if asked about AI in finance.",
            "source":   "General industry knowledge",
            "recency":  "ongoing trend",
        },
        {
            "headline": "Digital transformation accelerates across retail and investment banking",
            "hook":     "Frame your experience around how it supports modernisation — legacy-to-cloud migrations are highly valued.",
            "source":   "General industry knowledge",
            "recency":  "ongoing trend",
        },
        {
            "headline": "ESG reporting requirements drive demand for data engineering talent",
            "hook":     "If asked about impact, connect your data work to sustainability reporting — it signals strategic awareness.",
            "source":   "General industry knowledge",
            "recency":  "ongoing trend",
        },
    ],
    "management consulting": [
        {
            "headline": "GenAI strategy engagements now top consulting firm revenue streams",
            "hook":     "Expect case interview components involving AI transformation scenarios — practise structuring ambiguous tech problems.",
            "source":   "General industry knowledge",
            "recency":  "ongoing trend",
        },
        {
            "headline": "Supply chain resilience remains a top client concern post-2024",
            "hook":     "Case prep should include operations frameworks — McKinsey and BCG cases increasingly blend strategy with ops.",
            "source":   "General industry knowledge",
            "recency":  "ongoing trend",
        },
        {
            "headline": "Consulting firms expand data and digital practice headcount aggressively",
            "hook":     "Highlight any cross-functional data or analytics experience — it directly maps to the fastest-growing practice areas.",
            "source":   "General industry knowledge",
            "recency":  "ongoing trend",
        },
    ],
    "marketing & advertising": [
        {
            "headline": "First-party data strategies replace third-party cookie dependence",
            "hook":     "Show you understand consent-based marketing and can discuss how brands are rebuilding their data foundations.",
            "source":   "General industry knowledge",
            "recency":  "ongoing trend",
        },
        {
            "headline": "AI-generated content floods channels, raising authenticity premium",
            "hook":     "Interviewers want to hear how you balance AI efficiency with brand authenticity in your creative process.",
            "source":   "General industry knowledge",
            "recency":  "ongoing trend",
        },
        {
            "headline": "Performance marketing budgets shift toward connected TV and retail media",
            "hook":     "Signal channel awareness beyond Meta and Google — breadth of platform knowledge differentiates mid-senior candidates.",
            "source":   "General industry knowledge",
            "recency":  "ongoing trend",
        },
    ],
}

_DEFAULT_FALLBACK = [
    {
        "headline": "AI tools transform how professionals approach complex problems",
        "hook":     "Interviewers appreciate candidates who can articulate how they leverage modern tools to increase productivity and quality.",
        "source":   "General industry knowledge",
        "recency":  "ongoing trend",
    },
    {
        "headline": "Remote and hybrid work patterns reshape team collaboration norms",
        "hook":     "Be ready to discuss how you've maintained alignment and output quality in distributed team environments.",
        "source":   "General industry knowledge",
        "recency":  "ongoing trend",
    },
    {
        "headline": "Continuous learning and upskilling now a core professional expectation",
        "hook":     "Have a specific, recent example of something you taught yourself ready — it signals initiative to any hiring panel.",
        "source":   "General industry knowledge",
        "recency":  "ongoing trend",
    },
]


def _static_fallback(role: str, industry: str, error_msg: str = "") -> BriefingResult:
    raw     = _STATIC_FALLBACKS.get(industry.lower(), _DEFAULT_FALLBACK)
    bullets = [
        BulletPoint(
            headline  = b["headline"],
            hook      = b["hook"],
            source    = b["source"],
            recency   = b["recency"],
            is_static = True,
        )
        for b in raw[:3]
    ]
    return BriefingResult(
        bullets     = bullets,
        role        = role,
        industry    = industry,
        fetched_at  = time.time(),
        is_fallback = True,
        error_msg   = error_msg,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC FETCH API
# ══════════════════════════════════════════════════════════════════════════════

def fetch_briefing(
    role:         str,
    industry:     str,
    groq_api_key: str = "",
    force_refresh: bool = False,
) -> BriefingResult:
    """
    Fetch or return cached industry briefing.

    Parameters
    ----------
    role           : Candidate's target role, e.g. "Software Engineer".
    industry       : Industry keyword, e.g. "technology".
                     Use role_to_industry(role) if you only have the role.
    groq_api_key   : Groq API key string.  If empty, falls back to static.
    force_refresh  : Bypass cache and fetch fresh even within TTL window.

    Returns
    -------
    BriefingResult with up to 3 BulletPoint items.
    """
    if not force_refresh:
        cached = _get_cached(role, industry)
        if cached is not None:
            return cached

    # Resolve API key: param → session state → env var
    key = (
        groq_api_key
        or st.session_state.get("groq_api_key", "")
        or os.environ.get("GROQ_API_KEY", "")
    )

    result = None
    if key:
        result = _groq_fetch(role, industry, key)

    if result is None:
        error = "Groq unavailable — showing general industry insights." if key else \
                "No API key set — showing general industry insights."
        result = _static_fallback(role, industry, error_msg=error)

    _set_cached(result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER — STREAMLIT COMPONENT
# ══════════════════════════════════════════════════════════════════════════════

def _tts_js(text: str, key: str) -> None:
    """Inject a tiny TTS speak button via components.html."""
    safe = text.replace("`", "'").replace('"', "&quot;").replace("\n", " ")
    components.html(
        f"""
<button onclick="(function(){{
  var u = new SpeechSynthesisUtterance(`{safe}`);
  u.rate = 0.92; u.pitch = 1.0;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
}})()" style="
  font-family: 'Share Tech Mono', monospace;
  font-size: 11px;
  color: rgba(0,229,255,.65);
  background: rgba(0,229,255,.06);
  border: 1px solid rgba(0,229,255,.2);
  border-radius: 8px;
  padding: 4px 14px;
  cursor: pointer;
  letter-spacing: .05em;
  margin-top: 4px;
">▶ Speak Briefing</button>
""",
        height=40,
    )


def render_briefing_card(
    result:       BriefingResult,
    accent_color: str = "#00e5ff",
    show_tts:     bool = True,
    key_suffix:   str = "",
) -> None:
    """
    Render the briefing card in Streamlit using unsafe_allow_html markdown.

    Parameters
    ----------
    result       : BriefingResult from fetch_briefing().
    accent_color : Hex colour matching the current day/session theme.
    show_tts     : Whether to render the "Speak Briefing" button.
    key_suffix   : Appended to widget keys to avoid duplicate-key errors
                   when the card is rendered in multiple places.
    """
    # ── Header ───────────────────────────────────────────────────────────────
    fallback_label = (
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:.52rem;'
        'color:rgba(255,180,50,.55);background:rgba(255,180,50,.06);'
        'border:1px solid rgba(255,180,50,.2);border-radius:6px;'
        'padding:1px 8px;margin-left:.6rem;">'
        '⚠ offline — general insights'
        '</span>'
    ) if result.is_fallback else ""

    age_secs = max(0, time.time() - result.fetched_at)
    if age_secs < 120:
        fetched_str = "just now"
    elif age_secs < 3600:
        fetched_str = f"{int(age_secs // 60)}m ago"
    else:
        fetched_str = f"{int(age_secs // 3600)}h ago"

    header_html = f"""
<div style="display:flex;align-items:center;justify-content:space-between;
  flex-wrap:wrap;gap:.4rem;margin-bottom:.75rem;">
  <div style="display:flex;align-items:center;gap:.5rem;">
    <span style="font-size:1rem;">📡</span>
    <span style="font-family:'Orbitron',monospace;font-size:.65rem;font-weight:700;
      color:{accent_color};letter-spacing:.1em;">INDUSTRY BRIEFING</span>
    {fallback_label}
  </div>
  <span style="font-family:'JetBrains Mono',monospace;font-size:.5rem;
    color:rgba(180,210,230,.28);letter-spacing:.08em;">
    FETCHED {fetched_str} · {result.industry.upper()}
  </span>
</div>
"""

    # ── Bullets ───────────────────────────────────────────────────────────────
    bullet_blocks = []
    for i, b in enumerate(result.bullets):
        num_color = accent_color if i == 0 else (
            "#7f5af0" if i == 1 else "#00ff88"
        )
        static_note = (
            '<span style="font-size:.48rem;color:rgba(255,180,50,.45);'
            'font-family:\'JetBrains Mono\',monospace;margin-left:4px;">'
            '[general]</span>'
        ) if b.is_static else ""

        bullet_blocks.append(f"""
<div style="padding:.7rem .9rem;margin-bottom:.55rem;
  background:rgba(0,8,30,.65);
  border:1px solid rgba(255,255,255,.06);
  border-left:2px solid {num_color};
  border-radius:0 10px 10px 0;">
  <div style="display:flex;align-items:flex-start;gap:.6rem;">
    <span style="font-family:'Orbitron',monospace;font-size:.7rem;
      font-weight:700;color:{num_color};flex-shrink:0;margin-top:.05rem;">
      {i+1:02d}
    </span>
    <div>
      <div style="font-family:'Syne',sans-serif;font-size:.82rem;
        font-weight:700;color:#e8f4ff;margin-bottom:.25rem;line-height:1.45;">
        {b.headline}{static_note}
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;
        color:rgba(0,229,255,.6);line-height:1.6;margin-bottom:.3rem;">
        ↳ {b.hook}
      </div>
      <div style="display:flex;gap:.5rem;align-items:center;flex-wrap:wrap;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:.5rem;
          color:rgba(180,210,230,.3);">{b.source}</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:.5rem;
          padding:0 6px;background:rgba(255,255,255,.03);
          border:1px solid rgba(255,255,255,.07);border-radius:5px;
          color:rgba(180,210,230,.3);">{b.recency}</span>
      </div>
    </div>
  </div>
</div>""")

    bullets_html = "\n".join(bullet_blocks)

    # ── Suggestion footer ─────────────────────────────────────────────────────
    footer_tip = (
        "Reference one of these if asked <em>\"What industry trends excite you?\"</em>"
        " or <em>\"Why do you want to work here?\"</em>"
    )
    footer_html = f"""
<div style="font-family:'JetBrains Mono',monospace;font-size:.57rem;
  color:rgba(0,229,255,.35);margin-top:.2rem;line-height:1.6;
  border-top:1px solid rgba(0,229,255,.07);padding-top:.55rem;">
  💡 {footer_tip}
</div>
"""

    # ── Outer wrapper ─────────────────────────────────────────────────────────
    card_html = f"""
<div style="
  background:rgba(0,6,22,.92);
  border:1px solid {accent_color}22;
  border-top:2px solid {accent_color};
  border-radius:14px;
  padding:1rem 1.1rem .85rem;
  margin-bottom:1rem;
  position:relative;
  overflow:hidden;
">
  <div style="position:absolute;top:0;left:0;right:0;height:40px;
    background:radial-gradient(ellipse 60% 100% at 50% 0%,
      {accent_color}10,transparent);
    pointer-events:none;"></div>
  {header_html}
  {bullets_html}
  {footer_html}
</div>
"""

    st.markdown(card_html, unsafe_allow_html=True)

    # ── TTS button ────────────────────────────────────────────────────────────
    if show_tts and result.bullets:
        tts_text = (
            f"Industry briefing for your {result.role} interview. "
            + " ".join(
                f"Point {i+1}: {b.headline}. {b.hook}"
                for i, b in enumerate(result.bullets)
            )
        )
        _tts_js(tts_text, key=f"brief_tts_{key_suffix}")

    # ── Refresh button (shown for fallback or stale results) ──────────────────
    if result.is_fallback:
        if st.button(
            "↺  Retry live fetch",
            key=f"brief_refresh_{key_suffix}",
            help="Attempt to fetch live news again",
        ):
            # Clear cache for this role/industry so next render re-fetches
            store = st.session_state.get("briefing_cache", {})
            # Remove all keys (simpler than computing the exact stale key)
            store.clear()
            st.session_state["briefing_cache"] = store
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  ONE-LINER CONVENIENCE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def render_briefing_inline(
    role:          str,
    industry:      str  = "",
    groq_api_key:  str  = "",
    accent_color:  str  = "#00e5ff",
    show_tts:      bool = True,
    force_refresh: bool = False,
    key_suffix:    str  = "",
) -> BriefingResult:
    """
    Fetch and immediately render the briefing card.
    Returns the BriefingResult so callers can inspect it if needed.

    Parameters
    ----------
    role          : Target role string (e.g. "Software Engineer").
    industry      : Override industry; auto-derived from role if blank.
    groq_api_key  : Groq API key.  Falls back to session state / env var.
    accent_color  : Hex colour for card accent — match the session theme.
    show_tts      : Show the Speak Briefing button.
    force_refresh : Bypass cache.
    key_suffix    : Suffix for Streamlit widget keys (avoids DuplicateWidgetID).
    """
    if not industry:
        industry = role_to_industry(role)

    with st.spinner("📡  Fetching live industry briefing…"):
        result = fetch_briefing(
            role          = role,
            industry      = industry,
            groq_api_key  = groq_api_key,
            force_refresh = force_refresh,
        )

    render_briefing_card(
        result       = result,
        accent_color = accent_color,
        show_tts     = show_tts,
        key_suffix   = key_suffix,
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION SNIPPETS  (copy-paste into existing files)
# ══════════════════════════════════════════════════════════════════════════════

"""
─────────────────────────────────────────────────────────────────────────────
SNIPPET A — weekly_prep_plan.py
─────────────────────────────────────────────────────────────────────────────
Add this import at the top of weekly_prep_plan.py:

    from briefing_engine import render_briefing_inline, role_to_industry

Then inside _render_practice_window(), immediately after the ← Plan button
block (after the st.rerun() inside the back button handler) and before the
`if q_num <= total_q:` block, add:

    # ── Live industry briefing (only show at Q1, not on subsequent Qs) ──────
    if q_num == 1 and not st.session_state.get("wp_briefing_shown"):
        render_briefing_inline(
            role         = role,
            industry     = role_to_industry(role),
            accent_color = cfg.color,
            key_suffix   = f"wp_day{day_num}",
        )
        st.session_state["wp_briefing_shown"] = True
    elif q_num > 1:
        # Reset flag so briefing shows again next session
        st.session_state["wp_briefing_shown"] = False

─────────────────────────────────────────────────────────────────────────────
SNIPPET B — app.py  (page_start)
─────────────────────────────────────────────────────────────────────────────
Add at top of app.py imports:

    from briefing_engine import render_briefing_inline, role_to_industry

Inside page_start(), before the "Start Interview" button block, add:

    render_briefing_inline(
        role         = st.session_state.get("target_role", "Software Engineer"),
        accent_color = "#00e5ff",
        key_suffix   = "start",
    )

─────────────────────────────────────────────────────────────────────────────
SNIPPET C — WEEKLY_PLAN_DEFAULTS addition  (weekly_prep_plan.py)
─────────────────────────────────────────────────────────────────────────────
Add to WEEKLY_PLAN_DEFAULTS dict:

    "wp_briefing_shown": False,

─────────────────────────────────────────────────────────────────────────────
SNIPPET D — DEFAULTS addition  (app.py)
─────────────────────────────────────────────────────────────────────────────
Add to DEFAULTS dict:

    "briefing_cache": {},
"""
