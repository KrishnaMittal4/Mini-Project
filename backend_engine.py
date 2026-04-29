"""
backend_engine.py  —  Aura AI  |  Multimodal Interview Backend  (v8.0)
=======================================================================
v8.0 — RL ADAPTIVE QUESTION SEQUENCER (replaces static heuristic)
==================================================================
Research basis:
  Patel et al. (2023, Springer AI Review) — reinforcement learning for
  adaptive question sequencing in mock interview simulators dynamically
  adjusts difficulty based on real-time candidate performance signals,
  improving engagement and skill-gap coverage versus static ladders.

  Srivastava & Bhatt (2022, IEEE ICCCIS) — Q-learning for adaptive
  assessment: ε-greedy exploration converges to optimal difficulty in
  5–8 questions, superior to threshold-based heuristics.

  Liu et al. (2021, ACM ITS) — contextual bandit with reward shaping
  (score delta + nervousness + STAR + timing) outperforms score-only
  adaptive systems by 18% on skill coverage.

What changed in v8.0:
  • RLAdaptiveSequencer imported from adaptive_sequencer.py replaces
    the two-line QuestionBank.next_difficulty() heuristic.
  • Sequencer warm-starts from saved Q-table across sessions (cross-session
    learning — agent improves every time the candidate practices).
  • InterviewEngine.evaluate_answer() now calls sequencer.record_and_select()
    after scoring and stores the recommended next action.
  • InterviewEngine.get_next_question() passes the RL hint (type, difficulty)
    to QuestionBank so the Groq API generates the right question type.
  • InterviewEngine exposes get_rl_report() for the Final Report page.
  • QuestionBank.next_difficulty() kept for backwards-compat but no longer
    called by the main evaluation path.

v7.0 carried forward: Unified voice pipeline (CREMA-D + TESS, 108-dim)
v7.0 carried forward: NervousnessFusion — v10.0: 100% voice nervousness
v7.0 carried forward: VoiceQualityIndex session-level VQI
v8.0 (prior): posture + confidence formula updates (kept)
"""

from __future__ import annotations

import json
import os
from dotenv import load_dotenv
load_dotenv()
import re
import random
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import logging
import numpy as np

log = logging.getLogger("BackendEngine")

warnings.filterwarnings("ignore")

# ── FER pipeline (facial emotion + posture) ───────────────────────────────────
try:
    from dataset_loader import (
        FERPipeline, EMOTION_LABELS, _dummy_result,
        bytes_to_bgr, pil_to_bgr, MediaPipePostureAnalyser,
    )
    FER_AVAILABLE = True
except ImportError as _e:
    FER_AVAILABLE = False
    FERPipeline              = None
    MediaPipePostureAnalyser = None
    print(f"[backend_engine] dataset_loader import warning: {_e}")

# ── Unified Voice Pipeline (CREMA-D + TESS, 108-dim) ─────────────────────────
try:
    from unified_voice_pipeline import (
        UnifiedVoicePipeline,
        compute_nervousness_score,
        UNIFIED_EMOTIONS,
        FEATURE_SIZE as VOICE_FEATURE_SIZE,
    )
    VOICE_AVAILABLE = True

    # Backwards-compat aliases so old code that imported from voice_pipeline still works
    VoicePipeline = UnifiedVoicePipeline

    def _dummy_voice() -> Dict:
        return {
            "dominant":    "Neutral",
            "emotions":    {e: round(100 / len(UNIFIED_EMOTIONS), 1)
                            for e in UNIFIED_EMOTIONS},
            "confidence":  50.0,
            "nervousness": 0.2,
        }

except ImportError as _e:
    VOICE_AVAILABLE      = False
    UnifiedVoicePipeline = None
    VoicePipeline        = None
    VOICE_FEATURE_SIZE   = 108
    print(f"[backend_engine] unified_voice_pipeline import warning: {_e}")

    def _dummy_voice() -> Dict:
        return {"dominant": "Neutral", "emotions": {}, "confidence": 50.0, "nervousness": 0.2}

    def compute_nervousness_score(emotion_probs: Dict) -> float:
        return 0.2

# ── NLP dependencies ──────────────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_NLP_AVAILABLE = True
except ImportError:
    SKLEARN_NLP_AVAILABLE = False

# ── AnswerEvaluator (v8.0) — uses ideal_answer for TF-IDF relevance ──────────
try:
    from answer_evaluator import AnswerEvaluator
    ANSWER_EVALUATOR_AVAILABLE = True
except ImportError as _e:
    ANSWER_EVALUATOR_AVAILABLE = False
    print(f"[backend_engine] answer_evaluator import warning: {_e}")

# ── RL Adaptive Sequencer (v8.0) — replaces static next_difficulty() ─────────
# Research: Patel et al. (2023, Springer AI Review) — RL adaptive question
# sequencing improves interview coverage and engagement vs static heuristics.
try:
    from adaptive_sequencer import RLAdaptiveSequencer, ACTIONS as RL_ACTIONS
    RL_SEQUENCER_AVAILABLE = True
except ImportError as _e:
    RL_SEQUENCER_AVAILABLE = False
    RLAdaptiveSequencer = None  # type: ignore
    print(f"[backend_engine] adaptive_sequencer import warning: {_e}")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

STAR_PATTERNS: Dict[str, str] = {
    "Situation": r"\b(situation|context|background|when|once|there was|faced|encountered|during|at the time)\b",
    "Task"     : r"\b(task|goal|objective|responsible|needed to|had to|assigned|my role|challenge|was asked)\b",
    "Action"   : r"\b(i did|i took|i used|implemented|developed|created|decided|solved|built|designed|led|coordinated)\b",
    "Result"   : r"\b(result|outcome|achieved|improved|reduced|increased|success|impact|as a result|completed|delivered)\b",
}

DISC_KEYWORDS: Dict[str, List[str]] = {
    "Dominance"        : ["lead","decided","took charge","goal","direct","challenge","result","win","fast","control","drove","pushed"],
    "Influence"        : ["team","collaborate","communicate","inspire","enthusiasm","motivated","people","fun","support","engaged","presented"],
    "Steadiness"       : ["consistent","reliable","patient","support","stable","process","listen","careful","methodical","thorough","steady"],
    "Conscientiousness": ["accurate","detail","quality","process","data","systematic","standard","precise","analysis","metrics","documented"],
}

# ── Multimodal fusion weights (v8.0 — knowledge depth promoted) ───────────────
# posture + confidence removed from final score per design decision.
# knowledge depth is already baked into `knowledge` via depth_fluency sub-score,
# but we now track avg_depth explicitly as a separate report metric.
#   knowledge (includes depth_fluency sub-score) — 70%
#   emotion   (voice nervousness inverse)         — 15%
#   voice     (VQI + voice nervousness inverse)   — 15%
WEIGHTS: Dict[str, float] = dict(
    knowledge=0.70, emotion=0.15, voice=0.15
)

# ── Confidence formula weights ────────────────────────────────────────────────
# Confidence formula v8.1: 0.25×Eye + 0.25×Fluency + 0.35×Voice + 0.15×Facial
# Research basis:
#   Voice (0.35)  — acoustic biomarkers most reliable for remote confidence
#                   (Low et al. 2020, Riad et al. 2024)
#   Eye   (0.25)  — gaze contact is a primary interviewer perception cue
#                   (Burgoon et al. 1990, Albadawi et al. 2024)
#   Fluency(0.25) — filler words + pacing directly correlate with perceived
#                   confidence (Mehrabian 1971, Knapp & Hall 2014)
#   Facial (0.15) — lowest weight: webcam lighting degrades stability
CONFIDENCE_WEIGHTS = dict(eye=0.25, fluency=0.25, voice=0.35, facial=0.15)

# ── Nervousness fusion weights (facial vs voice) ──────────────────────────────
# v10.0: 100% voice-only nervousness — facial signal removed entirely.
# Voice acoustic biomarkers (pitch variance, pause ratio, ZCR) are the
# sole nervousness signal. Facial expressions are excluded because candidates
# can consciously suppress them and webcam lighting degrades reliability.
# Research: Schuller et al. (IEEE Trans. Affect. Comput. 2011) — voice
# biomarkers reach 82-91% stress accuracy vs 64-73% for face-only.
# Liao et al. (IEEE FG 2020) — acoustic features outperform visual by
# 11.4% F1 in naturalistic video-call conditions.
NERVOUSNESS_FUSION = dict(facial=0.0, voice=1.0)

# High-nervousness emotions set (from unified_voice_pipeline research basis)
HIGH_NERVOUSNESS_EMOTIONS = {"Fear", "Angry", "Sad", "Disgust"}


# ══════════════════════════════════════════════════════════════════════════════
#  QUESTION BANK
# ══════════════════════════════════════════════════════════════════════════════

class QuestionBank:
    """
    API-only question bank — generates fresh, role-specific interview
    questions via the Groq API on every new session.

    Every question set is unique: the LLM generates them randomly each
    time for the selected role and difficulty level.

    No hardcoded or fallback questions exist. If the API call fails the
    engine raises a RuntimeError so the UI can show a clear message.

    JSON schema the API must return:
        [
          {
            "role":         "Data Scientist",
            "difficulty":   "medium",
            "type":         "Technical",
            "question":     "Explain ...",
            "keywords":     ["bias", "variance"],
            "ideal_answer": "High bias = ..."
          },
          ...
        ]
    """

    _GROQ_MODEL = "llama-3.3-70b-versatile"

    # Supported roles shown in the UI dropdown
    SUPPORTED_ROLES: List[str] = [
        "Software Engineer", "Frontend Developer", "Backend Developer",
        "Full Stack Developer", "Data Scientist", "Product Manager",
        "DevOps Engineer", "Mobile Developer", "QA Engineer", "System Designer",
        "Machine Learning Engineer", "Cloud Architect", "Cybersecurity Analyst",
        "Data Engineer", "Scrum Master",
    ]

    # Balanced question type mix per difficulty
    _TYPE_MIX = {
        "easy":   {"Technical": 2, "Behavioural": 1, "HR": 1},
        "medium": {"Technical": 3, "Behavioural": 2, "HR": 1},
        "hard":   {"Technical": 4, "Behavioural": 2, "HR": 1},
        # v9.2: "all" mode — equal spread across difficulties, RL picks each level
        # The mix uses medium proportions as base; actual difficulty per question
        # is determined by the RL sequencer dynamically.
        "all":    {"Technical": 3, "Behavioural": 2, "HR": 1},
    }

    def __init__(self, groq_api_key: str = "") -> None:
        self._api_key: str = (
            groq_api_key
            or os.environ.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
        )
        # Cache key = (role, difficulty, n, session_id)
        # session_id rotates on every invalidate_cache() so even identical
        # (role, difficulty, n) always trigger a fresh API call for a new session.
        self._cache: Dict[tuple, List[dict]] = {}
        self._session_id: str = self._new_session_id()
        # Running list of question stems passed as AVOID list to the LLM
        self._asked_questions: List[str] = []
        self.metadata: dict = {"roles_supported": self.SUPPORTED_ROLES}

    # ── HR dataset helpers ────────────────────────────────────────────────────

    # Candidate paths for the HR dataset JSON — checked in order.
    # The first path that exists and loads cleanly wins.
    _HR_DATASET_PATHS: List[str] = [
        "hr_interview_dataset.json",                          # CWD / project root
        r"C:\Users\ACER\Downloads\Miniproject\hr_interview_dataset.json",  # dev machine
        os.path.join(os.path.dirname(__file__), "hr_interview_dataset.json"),
    ]

    # Difficulty labels in the dataset → normalised key used in backend_engine
    _HR_DIFF_MAP: Dict[str, str] = {
        "easy":   "easy",
        "medium": "medium",
        "hard":   "hard",
        "Easy":   "easy",
        "Medium": "medium",
        "Hard":   "hard",
    }

    # Category labels the dataset uses that map to Behavioural question type
    _HR_BEHAVIOURAL_CATEGORIES: set = {
        "Behavioural", "Behavioral",
        "Leadership", "Communication",
        "Problem Solving", "Adaptability",
        "Collaboration", "Self-Awareness",
        "Career Goals", "Teamwork",
    }

    def _load_hr_dataset(self) -> List[dict]:
        """
        Load hr_interview_dataset.json from the first path that exists.
        Returns a list of raw dicts from the file, or [] on any failure.
        Supports top-level array, wrapped object, and JSONL formats.
        """
        for path in self._HR_DATASET_PATHS:
            try:
                if not os.path.isfile(path):
                    continue
                with open(path, encoding="utf-8") as fh:
                    raw = fh.read().strip()
                if not raw:
                    continue
                # Try top-level array
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        log.info(f"[QuestionBank] HR dataset loaded: {len(parsed)} records from {path}")
                        return parsed
                    if isinstance(parsed, dict):
                        # Wrapped: {"data": [...]} or {"questions": [...]}
                        for key in ("data", "questions", "records", "items"):
                            if isinstance(parsed.get(key), list):
                                log.info(f"[QuestionBank] HR dataset loaded (wrapped): "
                                         f"{len(parsed[key])} records from {path}")
                                return parsed[key]
                except json.JSONDecodeError:
                    pass
                # Try JSONL
                records = []
                for line in raw.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                if records:
                    log.info(f"[QuestionBank] HR dataset loaded (JSONL): "
                             f"{len(records)} records from {path}")
                    return records
            except Exception as exc:
                log.debug(f"[QuestionBank] HR dataset load failed ({path}): {exc}")
                continue
        log.warning("[QuestionBank] hr_interview_dataset.json not found — "
                    "Q1 behavioural will be Groq-generated instead.")
        return []

    def get_hr_dataset_q1(self, difficulty: str, role: str) -> Optional[dict]:
        """
        Pick one random Behavioural question from the HR dataset that matches
        the requested difficulty level and return it as a backend_engine
        question dict (same shape as Groq-generated questions).

        Parameters
        ----------
        difficulty : "easy" | "medium" | "hard"
        role       : session role (injected into the returned dict)

        Returns
        -------
        A question dict ready to prepend to self.questions, or None if the
        dataset cannot be loaded or has no matching records.

        Dict shape (matches Groq schema):
          {
            "role":         str,
            "difficulty":   str,
            "type":         "Behavioural",
            "question":     str,
            "keywords":     [],          # HR behavioural Qs rarely need keywords
            "ideal_answer": str,
          }
        """
        records = self._load_hr_dataset()
        if not records:
            return None

        diff_key = self._HR_DIFF_MAP.get(difficulty, "medium")

        # Filter: must be a behavioural/leadership category AND matching difficulty
        candidates = []
        for rec in records:
            rec_cat  = rec.get("category", rec.get("Category", ""))
            rec_diff = self._HR_DIFF_MAP.get(
                rec.get("difficulty", rec.get("Difficulty", "")), "")
            rec_q    = (rec.get("question") or rec.get("Question") or "").strip()
            rec_ideal= (rec.get("ideal")    or rec.get("ideal_answer") or
                        rec.get("Ideal")    or "").strip()

            if (rec_cat in self._HR_BEHAVIOURAL_CATEGORIES and
                    rec_diff == diff_key and
                    rec_q):
                candidates.append((rec_q, rec_ideal))

        if not candidates:
            # Relax difficulty constraint — take any behavioural question
            for rec in records:
                rec_cat = rec.get("category", rec.get("Category", ""))
                rec_q   = (rec.get("question") or rec.get("Question") or "").strip()
                rec_ideal=(rec.get("ideal")    or rec.get("ideal_answer") or
                           rec.get("Ideal")    or "").strip()
                if rec_cat in self._HR_BEHAVIOURAL_CATEGORIES and rec_q:
                    candidates.append((rec_q, rec_ideal))

        if not candidates:
            log.warning(f"[QuestionBank] No behavioural records found in HR dataset "
                        f"for difficulty='{diff_key}'.")
            return None

        # Pick a random record — different every session
        q_text, ideal = random.choice(candidates)

        log.info(f"[QuestionBank] HR dataset Q1: [{diff_key}] {q_text[:70]}")

        return {
            "role":         role,
            "difficulty":   diff_key,
            "type":         "Behavioural",
            "question":     q_text,
            "keywords":     [],
            "ideal_answer": ideal,
            "source":       "hr_dataset",   # extra tag — visible in session_answers
        }

    @staticmethod
    def _new_session_id() -> str:
        """Generate a short random session token (rotated on each invalidate)."""
        import uuid as _uuid
        return _uuid.uuid4().hex[:12]

    def _pick_topics(self, role: str, n_topics: int = 5) -> "List[str]":
        """
        Sample topic domains randomly — different topics = different questions.
        Each role pool includes:
          • Original domain-specific topics
          • B.Tech CSE Basic concepts (DS&A, OOP, OS, Networks, DBMS, Discrete Maths,
            Computer Architecture, Software Engineering, Theory of Computation)
          • B.Tech CSE Advanced concepts (Compiler Design, Distributed Systems,
            Parallel Computing, Cryptography, Advanced Algorithms, Formal Methods)
        """
        pool = {

            # ══════════════════════════════════════════════════════════════════
            # SOFTWARE ENGINEER
            # ══════════════════════════════════════════════════════════════════
            "Software Engineer": [
                # ── Original domain topics ────────────────────────────────────
                "concurrency and threading",
                "memory management and GC",
                "data structures internals",
                "algorithm complexity analysis",
                "system design at scale",
                "REST vs gRPC API design",
                "caching strategies (LRU/LFU/write-through)",
                "database indexing and query plans",
                "microservices decomposition patterns",
                "event-driven and CQRS architecture",
                "unit integration and contract testing",
                "CI/CD pipeline optimisation",
                "distributed consensus (Raft/Paxos)",
                "message queues (Kafka/RabbitMQ)",
                "OAuth2 and OIDC authentication",
                "observability (metrics/logs/traces)",
                "performance profiling and flamegraphs",
                "containerisation and k8s scheduling",
                # ── B.Tech CSE Basic ──────────────────────────────────────────
                "arrays, linked lists, stacks and queues internals",
                "binary trees, BST, AVL and red-black tree rotations",
                "graph representations (adjacency list vs matrix) and traversals",
                "sorting algorithms — merge sort, quick sort, heap sort complexity",
                "searching algorithms — binary search, interpolation search",
                "dynamic programming — memoization vs tabulation trade-offs",
                "greedy algorithms and proof of correctness",
                "recursion, backtracking and branch-and-bound",
                "time and space complexity analysis (Big-O, Theta, Omega)",
                "object-oriented principles — SOLID, DRY, YAGNI in practice",
                "design patterns — creational, structural, behavioural (GoF)",
                "process vs thread, context switching and scheduling algorithms",
                "virtual memory, paging, segmentation and TLB",
                "deadlock detection, prevention and Banker's algorithm",
                "file system internals — inode, journaling, VFS layer",
                "TCP/IP stack internals — handshake, flow control, congestion",
                "OSI model layers and how HTTP/2 maps to them",
                "SQL normalisation (1NF–BCNF), joins and query optimisation",
                "ACID properties, transaction isolation levels and lock granularity",
                "ER modelling and relational algebra fundamentals",
                "computer organisation — CPU pipeline, hazards and forwarding",
                "cache hierarchy (L1/L2/L3), cache coherence and false sharing",
                "bit manipulation techniques and low-level optimisation",
                # ── B.Tech CSE Advanced ───────────────────────────────────────
                "compiler phases — lexing, parsing, AST construction, IR emission",
                "code optimisation passes — constant folding, loop unrolling, inlining",
                "garbage collection algorithms — mark-sweep, generational, tri-colour",
                "NP-completeness, reduction proofs and approximation algorithms",
                "randomised algorithms — Las Vegas vs Monte Carlo, hash functions",
                "formal language theory — regular expressions to finite automata",
                "distributed systems consistency models (linearisability, serializability)",
                "vector clocks, logical clocks and distributed snapshot algorithms",
                "lock-free and wait-free data structures (CAS, ABA problem)",
                "NUMA architecture, memory-access patterns and performance",
                "SIMD and vectorisation for throughput-critical code paths",
            ],

            # ══════════════════════════════════════════════════════════════════
            # FRONTEND DEVELOPER
            # ══════════════════════════════════════════════════════════════════
            "Frontend Developer": [
                # ── Original domain topics ────────────────────────────────────
                "browser rendering pipeline (parse → style → layout → paint → composite)",
                "React reconciliation, virtual DOM and fibre architecture",
                "state management patterns — Redux, Zustand, Context API trade-offs",
                "CSS cascade, specificity, stacking context and BFC",
                "web performance — Core Web Vitals, LCP, CLS, INP optimisation",
                "code splitting, lazy loading and bundle size analysis",
                "progressive web apps — service workers, cache strategies, offline",
                "WebSockets and Server-Sent Events for real-time UI",
                "accessibility (WCAG 2.2) and ARIA patterns",
                "micro-frontend architecture and module federation",
                "TypeScript advanced types — generics, conditional, mapped types",
                "testing — React Testing Library, Playwright, visual regression",
                "security — XSS, CSRF, CSP headers and sanitisation",
                "design systems — token architecture, component contracts",
                # ── B.Tech CSE Basic ──────────────────────────────────────────
                "data structures relevant to UI — queues for event loops, trees for DOM",
                "algorithm complexity in rendering — why O(n²) layout triggers hurt",
                "object-oriented design in component hierarchies (composition vs inheritance)",
                "HTTP request-response cycle, DNS resolution and TLS handshake",
                "TCP vs UDP and why browsers use TCP for page loads",
                "browser JS engine — call stack, heap, event loop, task vs microtask queue",
                "process isolation — browser process model and site isolation",
                "memory leaks in JS — reference cycles, closure retention, detached nodes",
                "hashing in practice — content-addressed caching, cache busting strategies",
                "graph algorithms for dependency resolution in bundlers",
                # ── B.Tech CSE Advanced ───────────────────────────────────────
                "compiler theory behind transpilers (Babel, SWC, TypeScript)",
                "parsing theory — recursive descent, Pratt parser for template literals",
                "WebAssembly — linear memory model, compilation pipeline, use cases",
                "HTTP/3 and QUIC — why UDP, stream multiplexing, 0-RTT",
                "formal language theory applied to type-safe UI frameworks",
                "parallel rendering — React concurrent mode, scheduler and priority lanes",
                "GPU compositing layer creation and paint invalidation minimisation",
                "cryptography fundamentals — HTTPS, certificate chains, subresource integrity",
            ],

            # ══════════════════════════════════════════════════════════════════
            # BACKEND DEVELOPER
            # ══════════════════════════════════════════════════════════════════
            "Backend Developer": [
                # ── Original domain topics ────────────────────────────────────
                "API design — REST maturity, versioning, idempotency, pagination",
                "database connection pooling, prepared statements and N+1 prevention",
                "background job queues — worker concurrency, retries, dead-letter queues",
                "rate limiting strategies — token bucket, leaky bucket, sliding window",
                "caching layers — L1 in-process, L2 Redis, cache stampede prevention",
                "authentication patterns — JWT, session cookies, OAuth2 flows",
                "event sourcing and outbox pattern for distributed consistency",
                "database migrations — zero-downtime schema changes",
                "observability — structured logging, distributed tracing, RED metrics",
                "serverless vs containers trade-offs for backend workloads",
                "GraphQL — resolver N+1, DataLoader batching, schema stitching",
                "gRPC — proto3 IDL, streaming modes, interceptors",
                # ── B.Tech CSE Basic ──────────────────────────────────────────
                "hash maps — internal chaining, open addressing, load factor resizing",
                "heap data structure and priority queue applications in task scheduling",
                "B-tree and B+tree structure explaining why databases use them",
                "sorting algorithm selection for in-memory vs disk-based data",
                "transaction isolation levels — dirty read, phantom, lost update",
                "ACID vs BASE and when to choose eventual consistency",
                "process scheduling in Linux — CFS, priority inversion and priority inheritance",
                "inter-process communication — pipes, sockets, shared memory trade-offs",
                "network byte order, TCP Nagle algorithm and delayed ACK impact on APIs",
                "SQL execution plan analysis — seq scan vs index scan decisions",
                "normalisation vs denormalisation trade-offs for read vs write workloads",
                "recursion in tree-structured data (nested sets, adjacency lists, CTEs)",
                # ── B.Tech CSE Advanced ───────────────────────────────────────
                "distributed transactions — 2PC, saga pattern and compensating transactions",
                "consistent hashing for cache and database sharding",
                "MVCC internals in PostgreSQL and MySQL — undo logs, visibility rules",
                "lock-free queues and their use in high-throughput message brokers",
                "formal consistency models — linearisability vs sequential consistency",
                "WAL (Write-Ahead Logging) internals and crash recovery",
                "query optimiser internals — cost-based optimiser, statistics, join ordering",
                "TLS 1.3 handshake, 0-RTT risks and certificate transparency",
                "JVM / CPython memory model and how GC pauses affect API latency",
                "epoll / kqueue / io_uring — async I/O models and backend throughput",
            ],

            # ══════════════════════════════════════════════════════════════════
            # FULL STACK DEVELOPER
            # ══════════════════════════════════════════════════════════════════
            "Full Stack Developer": [
                # ── Original domain topics ────────────────────────────────────
                "full-stack architecture — BFF pattern, API gateway, edge functions",
                "monorepo tooling — Turborepo, Nx, shared library management",
                "SSR vs SSG vs ISR vs CSR trade-offs in Next.js / Nuxt",
                "end-to-end type safety — tRPC, Zod, OpenAPI code-gen",
                "real-time sync — optimistic updates, conflict resolution, CRDTs",
                "authentication across frontend and backend — session vs token trade-offs",
                "database migrations in CI/CD for zero-downtime full-stack deploys",
                "full-stack testing strategy — unit, integration, E2E pyramids",
                "Web Workers and offloading CPU work from the main thread",
                "WebRTC — STUN/TURN, SDP negotiation, peer-to-peer data channels",
                # ── B.Tech CSE Basic ──────────────────────────────────────────
                "data structures for UI state (immutable trees, trie for autocomplete)",
                "algorithmic thinking for full-stack features — pagination cursor design",
                "OOP design in layered architecture (controller, service, repository)",
                "network fundamentals — how cookies, sessions and JWT cross boundaries",
                "CORS, preflight requests and same-origin policy explained",
                "SQL vs NoSQL selection criteria based on access patterns",
                "event-driven programming — browser events vs server-side event emitters",
                "file system operations for file upload, streaming and static serving",
                "memory management across layers — frontend heap vs backend heap",
                "graph traversal for social feature implementation (followers, feed)",
                # ── B.Tech CSE Advanced ───────────────────────────────────────
                "HTTP/2 server push, multiplexing and head-of-line blocking removal",
                "distributed session management and sticky sessions in k8s",
                "compiler-level optimisation in build tools (tree shaking, DCE)",
                "formal API contracts — OpenAPI spec, contract testing with Pact",
                "CRDT theory for collaborative real-time editing features",
                "service worker caching strategies — stale-while-revalidate, network-first",
                "edge computing — V8 isolates, cold start, global state limitations",
                "WebAssembly for performance-critical full-stack computation",
            ],

            # ══════════════════════════════════════════════════════════════════
            # DATA SCIENTIST
            # ══════════════════════════════════════════════════════════════════
            "Data Scientist": [
                # ── Original domain topics ────────────────────────────────────
                "feature engineering and selection",
                "model evaluation and calibration",
                "cross-validation strategies",
                "regularisation and overfitting",
                "gradient boosting (XGBoost/LightGBM)",
                "deep learning architectures",
                "NLP preprocessing pipelines",
                "time series forecasting methods",
                "A/B testing and power analysis",
                "causal inference methods",
                "class imbalance handling",
                "hyperparameter optimisation (Bayesian)",
                "model interpretability (SHAP/LIME)",
                "recommendation system architectures",
                "data pipeline and ETL design",
                # ── B.Tech CSE Basic ──────────────────────────────────────────
                "probability fundamentals — Bayes theorem, conditional probability, prior/posterior",
                "probability distributions — Gaussian, Bernoulli, Poisson and their ML roles",
                "linear algebra for ML — matrix multiplication, eigendecomposition, SVD",
                "calculus for ML — partial derivatives, chain rule and gradient intuition",
                "hypothesis testing — t-test, chi-square, p-values and multiple comparisons",
                "data structures for analytics — hash maps for group-by, heaps for top-K",
                "sorting and searching algorithms applied to data preprocessing pipelines",
                "SQL and relational algebra for data extraction and transformation",
                "database normalisation vs denormalisation for analytical workloads",
                "ER modelling and relational algebra fundamentals",
                "algorithm complexity applied to ML training cost (O(n) vs O(n²) loops)",
                "graph theory basics — connected components, shortest paths for network analysis",
                "set theory operations — union, intersection, difference in feature engineering",
                "file I/O and data formats — CSV, Parquet, Avro and binary encoding trade-offs",
                # ── B.Tech CSE Advanced ───────────────────────────────────────
                "information theory — entropy, KL divergence, mutual information for feature selection",
                "numerical computation — floating-point precision, condition number, stable algorithms",
                "randomised algorithms — reservoir sampling, Monte Carlo approximations",
                "distributed computing models (MapReduce, Spark RDD/DAG execution)",
                "approximate nearest-neighbour search (LSH, FAISS) for embedding retrieval",
                "formal complexity theory applied to ML — PAC learning bounds, VC dimension",
                "graph neural network architectures and message passing formalism",
                "Markov decision processes, value iteration and Q-learning foundations",
                "algebraic structures (groups, rings) underpinning optimisation theory",
                "streaming algorithms — count-min sketch, HyperLogLog for large-scale stats",
                "compiler-level optimisations in NumPy / pandas (vectorisation, BLAS kernels)",
            ],

            # ══════════════════════════════════════════════════════════════════
            # PRODUCT MANAGER
            # ══════════════════════════════════════════════════════════════════
            "Product Manager": [
                # ── Original domain topics ────────────────────────────────────
                "prioritisation frameworks (RICE/ICE)",
                "OKR definition and alignment",
                "user story mapping",
                "roadmap communication to executives",
                "technical debt negotiation",
                "go-to-market execution",
                "pricing and packaging decisions",
                "customer discovery interviews",
                "MVP scoping and iteration",
                "metrics and KPI framework design",
                # ── B.Tech CSE Basic ──────────────────────────────────────────
                "software development lifecycle — waterfall, Agile, Kanban trade-offs for PMs",
                "SDLC phases and how each affects release planning and scope negotiation",
                "API concepts for PMs — REST, webhooks, rate limits and what they mean for roadmap",
                "database basics — SQL vs NoSQL choice impact on feature design decisions",
                "client-server model and how latency affects UX and retention metrics",
                "version control concepts — trunk-based development, feature flags, rollback risk",
                "software testing types (unit, integration, E2E) and their release-risk implications",
                "UML use-case and sequence diagrams for requirements communication with engineers",
                "TCP/IP and CDN basics for understanding global performance trade-offs",
                "basic algorithm complexity — why O(n²) features need scale consideration",
                "data structures intuition for PMs — why search requires indexing investment",
                "security fundamentals for PMs — authentication, authorisation, GDPR implications",
                "cloud computing basics — VMs vs containers vs serverless and cost implications",
                "caching concepts for PMs — why cache invalidation is hard and affects data freshness",
                # ── B.Tech CSE Advanced ───────────────────────────────────────
                "CAP theorem trade-offs for PMs — consistency vs availability in product decisions",
                "distributed system failure modes and how they inform SLA commitments",
                "event-driven architectures — asynchronous product flows and their UX challenges",
                "ML model lifecycle for PMs — training data, drift, retraining cadence",
                "A/B testing statistical foundations — sample size, significance, novelty effects",
                "privacy-enhancing technologies — differential privacy, k-anonymity for PM decisions",
                "system design concepts for PMs — load balancing, sharding impact on feature scope",
                "formal methods intuition — specification, invariants and correctness in product requirements",
            ],

            # ══════════════════════════════════════════════════════════════════
            # DEVOPS ENGINEER
            # ══════════════════════════════════════════════════════════════════
            "DevOps Engineer": [
                # ── Original domain topics ────────────────────────────────────
                "infrastructure as code (Terraform)",
                "Kubernetes operators and scheduling",
                "GitOps with ArgoCD/Flux",
                "blue-green and canary deployments",
                "chaos engineering practices",
                "log aggregation and SIEM",
                "distributed tracing (OpenTelemetry)",
                "SLO and error budget management",
                "zero-trust network architecture",
                "cost optimisation strategies",
                # ── B.Tech CSE Basic ──────────────────────────────────────────
                "Linux process model — fork, exec, zombie, orphan processes",
                "Linux namespaces and cgroups — the kernel primitives behind containers",
                "file system hierarchy — inodes, hard links, soft links, mount namespaces",
                "shell scripting fundamentals — pipes, redirects, signal handling, trap",
                "TCP/IP networking — subnetting, CIDR, routing tables and iptables rules",
                "DNS resolution — recursive vs iterative, TTL, negative caching",
                "SSH internals — key exchange, agent forwarding and port tunnelling",
                "UDP and ICMP in network diagnostics — ping, traceroute, MTU discovery",
                "process scheduling in Linux — CFS, nice values, real-time scheduling classes",
                "memory management — virtual memory, OOM killer, huge pages, NUMA effects",
                "socket programming concepts behind service health checks and probes",
                "POSIX IPC — message queues, semaphores and shared memory in system services",
                "database replication concepts — primary-replica, WAL shipping, PITR",
                "symmetric vs asymmetric cryptography, certificate chains and TLS internals",
                # ── B.Tech CSE Advanced ───────────────────────────────────────
                "eBPF — kernel programmability, tracing, networking and security use cases",
                "Linux kernel networking stack — netfilter, conntrack, XDP data path",
                "virtualisation — type 1 vs type 2 hypervisors, KVM, hardware-assisted virt",
                "distributed consensus — Raft leader election applied to etcd and consul",
                "consistent hashing in load balancers and distributed caches",
                "formal verification of infrastructure — TLA+ specs for distributed config",
                "NUMA-aware scheduling in Kubernetes CPU management and topology manager",
                "hardware security modules, TPM and measured boot in secure infrastructure",
                "network function virtualisation — DPDK, RDMA and SR-IOV for high-throughput",
                "compiler toolchain and build system internals for faster CI pipelines",
            ],

            # ══════════════════════════════════════════════════════════════════
            # MACHINE LEARNING ENGINEER
            # ══════════════════════════════════════════════════════════════════
            "Machine Learning Engineer": [
                # ── Original domain topics ────────────────────────────────────
                "model serving and inference optimisation",
                "feature store design",
                "data drift and concept drift detection",
                "MLOps and experiment tracking",
                "distributed training (DDP/FSDP)",
                "model quantisation and pruning",
                "transformer fine-tuning strategies",
                "vector databases and ANN search",
                "edge deployment constraints",
                "LLM serving infrastructure",
                # ── B.Tech CSE Basic ──────────────────────────────────────────
                "computer architecture for ML — CPU vs GPU memory hierarchy and bandwidth",
                "cache locality and its impact on matrix multiplication performance",
                "OS process scheduling impact on multi-GPU training job co-ordination",
                "data structures for ML pipelines — circular buffers, priority queues for dataloaders",
                "algorithm complexity of training — O(n·d·L) transformer cost breakdown",
                "object-oriented design in ML systems — model, trainer, evaluator separation",
                "file I/O optimisation — memory-mapped files, prefetching for training data",
                "network programming fundamentals for distributed training (NCCL, gRPC)",
                "serialisation formats — protobuf, safetensors, ONNX for model exchange",
                "hash maps for feature stores — O(1) lookup for online feature serving",
                "graph algorithms for computational graph optimisation (topological sort)",
                "POSIX threads and synchronisation primitives for data loader workers",
                "TCP flow control impact on parameter server gradient communication",
                "database indexing strategies for feature retrieval at low latency",
                # ── B.Tech CSE Advanced ───────────────────────────────────────
                "compiler theory behind ML compilers (XLA, TorchScript, TVM Relay IR)",
                "polyhedral model for loop nest optimisation in tensor computation",
                "SIMD and vectorisation — AVX-512 for CPU inference optimisation",
                "CUDA programming model — warps, thread blocks, shared memory, bank conflicts",
                "mixed-precision arithmetic — FP16/BF16 numerical stability and loss scaling",
                "gradient checkpointing — recomputation vs memory trade-off formalised",
                "distributed consensus applied to ML — parameter consistency in async SGD",
                "formal type systems for tensor shapes — Einops notation and shape checking",
                "register file and instruction-level parallelism in GPU kernel design",
                "NP-hard subproblems in AutoML — neural architecture search complexity",
                "algebraic structure of automatic differentiation — reverse-mode AD theory",
                "lock-free ring buffers for zero-copy inference request batching",
            ],

            # ══════════════════════════════════════════════════════════════════
            # QA ENGINEER
            # ══════════════════════════════════════════════════════════════════
            "QA Engineer": [
                # ── Original domain topics ────────────────────────────────────
                "test pyramid — unit, integration, E2E ratio and the ice-cream anti-pattern",
                "property-based testing — QuickCheck, Hypothesis and invariant design",
                "mutation testing — PIT, Stryker and escaping mutant strategies",
                "contract testing — Pact, schema versioning and consumer-driven contracts",
                "performance testing — load, stress, soak and spike testing methodologies",
                "chaos engineering for QA — fault injection, latency simulation",
                "API testing — idempotency verification, boundary value analysis",
                "visual regression testing — screenshot diffing and perceptual hashing",
                "test data management — factories, fixtures, database seeding strategies",
                "shift-left security testing — SAST, DAST integration in CI pipelines",
                "behaviour-driven development — Gherkin, living documentation",
                "test environment management — ephemeral environments, Docker Compose",
                # ── B.Tech CSE Basic ──────────────────────────────────────────
                "graph theory for test coverage — control flow graph, cyclomatic complexity",
                "finite automata for modelling system states in test case design",
                "equivalence partitioning, boundary value analysis — formal derivation",
                "sorting and searching algorithm correctness testing — invariant verification",
                "data structures for test frameworks — trees for test hierarchies, queues for runners",
                "algorithmic complexity — why O(n²) test suites hurt CI pipelines",
                "software engineering SDLC phases and when testing activities align",
                "OOP principles in test code — Page Object Model and DRY test helpers",
                "database testing — transaction rollback for isolation, referential integrity checks",
                "network basics for API testing — status codes, headers, idempotency semantics",
                "operating system process model for understanding flaky test isolation issues",
                "discrete math — combinatorial test design (pairwise testing, covering arrays)",
                "hash functions in test deduplication and flaky test fingerprinting",
                # ── B.Tech CSE Advanced ───────────────────────────────────────
                "formal verification and model checking (TLA+, Alloy) for protocol correctness",
                "program analysis — static analysis, taint analysis, symbolic execution",
                "concurrency testing — race detection, Happens-Before analysis, ThreadSanitizer",
                "fuzzing theory — coverage-guided, grammar-based and mutation-based fuzzers",
                "compiler-level instrumentation — sanitisers (ASan, UBSan, MSan)",
                "distributed system testing — linearisability checking with Jepsen",
                "type theory application — dependent types and contracts as specifications",
                "information flow analysis — taint tracking for security test coverage",
                "abstract interpretation for static bug detection without false positives",
                "automata-based test generation — model-based testing from state machines",
            ],

            # ══════════════════════════════════════════════════════════════════
            # SYSTEM DESIGNER
            # ══════════════════════════════════════════════════════════════════
            "System Designer": [
                # ── Original domain topics ────────────────────────────────────
                "capacity estimation — QPS, storage, bandwidth back-of-envelope approach",
                "load balancing algorithms — round-robin, least connections, consistent hashing",
                "database selection — SQL vs NoSQL vs NewSQL for given access patterns",
                "sharding strategies — range, hash, directory-based and re-sharding",
                "replication — sync vs async, read replicas, replica lag handling",
                "caching topologies — CDN, edge cache, read-through, write-behind",
                "message queue design — Kafka topic partitioning, consumer groups, lag",
                "rate limiting at scale — token bucket in distributed settings",
                "search system design — inverted index, Lucene segments, ranking pipeline",
                "URL shortener, pastebin, notification system classic design problems",
                "design for global distribution — multi-region active-active trade-offs",
                "API gateway patterns — auth, routing, aggregation, circuit breaker",
                # ── B.Tech CSE Basic ──────────────────────────────────────────
                "data structures underlying system components — LSM tree for write-heavy DBs",
                "B-tree and B+tree — why storage engines use them for range queries",
                "hash table design in distributed caches — load factor and rehashing cost",
                "graph theory for network topology design — minimum spanning tree, shortest paths",
                "sorting algorithms applied to log compaction and SSTable merging",
                "algorithm complexity — how O(log n) vs O(1) lookup changes system behaviour",
                "OS process model — how servers handle concurrent connections (C10K problem)",
                "virtual memory and memory-mapped files in database buffer pool design",
                "TCP connection lifecycle — keep-alive, TIME_WAIT storm and its system impact",
                "DNS hierarchy and how it influences global system topology decisions",
                "file system concepts — append-only logs, sequential vs random I/O implications",
                "relational model — normalisation, denormalisation and OLTP vs OLAP trade-offs",
                "computer architecture — memory hierarchy informing cache tier decisions",
                "discrete mathematics — pigeonhole principle in consistent hashing proofs",
                # ── B.Tech CSE Advanced ───────────────────────────────────────
                "distributed consensus — Paxos, Raft, Multi-Paxos for leader election",
                "CRDT theory — convergent and commutative replicated data types",
                "vector clocks and version vectors for conflict detection",
                "CAP theorem formal statement and PACELC extension",
                "consistent hashing mathematical properties — load distribution proofs",
                "Bloom filter and count-min sketch for approximate membership queries",
                "write-ahead logging, ARIES recovery algorithm and checkpoint protocols",
                "lock-free and wait-free data structures for concurrent system components",
                "formal specification — TLA+ for distributed system correctness proofs",
                "network flow algorithms — max-flow min-cut for capacity planning",
            ],

            # ══════════════════════════════════════════════════════════════════
            # CLOUD ARCHITECT
            # ══════════════════════════════════════════════════════════════════
            "Cloud Architect": [
                # ── Original domain topics ────────────────────────────────────
                "multi-cloud and hybrid-cloud connectivity patterns",
                "landing zone design — account/project structure, guardrails, automation",
                "identity federation — IAM roles, workload identity, OIDC trust chains",
                "cloud networking — VPC peering, transit gateway, private link",
                "cost engineering — rightsizing, committed use, savings plans, spot strategy",
                "disaster recovery — RTO/RPO targets, warm standby vs multi-site active-active",
                "data sovereignty and compliance — GDPR, HIPAA, FedRAMP cloud controls",
                "cloud-native security — CSPM, CWPP, zero-trust posture management",
                "Kubernetes on cloud — managed control planes, node auto-provisioner, Karpenter",
                "platform engineering — internal developer platform, golden paths, self-service",
                "FinOps maturity model — unit economics, chargeback and showback",
                "well-architected framework pillars applied to real workloads",
                # ── B.Tech CSE Basic ──────────────────────────────────────────
                "virtualisation fundamentals — hypervisors, VMs, para-virtualisation for cloud",
                "OS containers vs VMs — namespaces, cgroups, seccomp in cloud context",
                "networking fundamentals — BGP in cloud interconnects, MPLS, overlay networks",
                "TCP/IP for cloud architects — VPC routing, NAT, elastic IPs and latency",
                "distributed file systems concepts — NFS, NAS, SAN and cloud block/object storage",
                "database replication fundamentals — primary-replica for cloud RDS/Cloud SQL design",
                "algorithm complexity applied to cloud auto-scaling — reaction time analysis",
                "cryptography fundamentals — KMS, envelope encryption, key rotation lifecycle",
                "RAID levels and erasure coding concepts behind cloud storage durability",
                "DNS design — Route 53 routing policies, health checks, GeoDNS trade-offs",
                "load balancer types — L4 vs L7 and when to use each in cloud architecture",
                "software engineering patterns in IaC — DRY modules, state management, drift",
                # ── B.Tech CSE Advanced ───────────────────────────────────────
                "distributed consensus in managed cloud services — DynamoDB, Spanner, Cosmos",
                "hardware security — TPM, confidential computing, AMD SEV for cloud tenants",
                "network function virtualisation — SR-IOV, ENA, DPDK in cloud networking",
                "formal infrastructure modelling — CDK Aspects, OPA policies as specification",
                "eBPF for cloud-native networking and security observability",
                "Byzantine fault tolerance and its relevance to multi-cloud design",
                "formal capacity planning — Little's Law, queueing theory for cloud sizing",
                "NUMA effects in cloud virtualised environments and workload implications",
            ],

            # ══════════════════════════════════════════════════════════════════
            # CYBERSECURITY ANALYST
            # ══════════════════════════════════════════════════════════════════
            "Cybersecurity Analyst": [
                # ── Original domain topics ────────────────────────────────────
                "threat modelling — STRIDE, PASTA, attack trees and threat actor profiling",
                "MITRE ATT&CK — technique mapping, TTP chaining and detection engineering",
                "SIEM rules — sigma rules, correlation, alert fatigue reduction",
                "vulnerability management — CVSS scoring, exploit prediction, patch SLA",
                "incident response lifecycle — containment, eradication, forensics, lessons",
                "network detection — IDS vs IPS, Zeek/Suricata rule writing",
                "endpoint detection — EDR telemetry, LOLBIN abuse, hollowing techniques",
                "identity and access management — PAM, just-in-time access, SSO risks",
                "cloud security posture — misconfiguration risk, CSPM alert triage",
                "red vs blue vs purple team methodology",
                "OSINT techniques — passive recon, WHOIS, certificate transparency",
                # ── B.Tech CSE Basic ──────────────────────────────────────────
                "cryptography fundamentals — symmetric, asymmetric, hashing and MACs",
                "public key infrastructure — X.509, certificate chains, OCSP, CRL",
                "TLS handshake internals — cipher suite negotiation, HSTS, certificate pinning",
                "TCP/IP networking — how attackers exploit protocol weaknesses",
                "operating system internals — system calls, ring levels, privilege escalation paths",
                "process and memory model — stack overflow, heap spray, buffer overflow root causes",
                "file system concepts — NTFS ADS, Linux /proc as attacker information source",
                "discrete mathematics — Boolean algebra for firewall rule logic",
                "graph theory — attack path analysis, lateral movement graph traversal",
                "algorithm complexity in cryptanalysis — why brute force is infeasible",
                "formal automata theory — regex for log parsing, IDS signature matching",
                "database security — SQL injection mechanics, parameterised query defences",
                "software engineering — secure SDLC, threat modelling in design phase",
                # ── B.Tech CSE Advanced ───────────────────────────────────────
                "compiler-level security — stack canaries, PIE/ASLR/DEP and their bypasses",
                "formal verification applied to cryptographic protocol analysis (ProVerif, Tamarin)",
                "static and dynamic program analysis — taint analysis for vulnerability discovery",
                "reverse engineering — disassembly, decompilation, anti-analysis techniques",
                "symbolic execution for automatic exploit generation (angr, KLEE)",
                "side-channel attack theory — timing, cache, power analysis fundamentals",
                "homomorphic encryption and zero-knowledge proofs — privacy-preserving security",
                "hardware security — TPM, secure enclave, physically unclonable functions",
                "fuzzing theory — AFL, libFuzzer, grammar-based fuzzing for vuln discovery",
                "binary exploitation advanced — ROP chains, heap grooming, kernel exploitation basics",
            ],

            # ══════════════════════════════════════════════════════════════════
            # DATA ENGINEER
            # ══════════════════════════════════════════════════════════════════
            "Data Engineer": [
                # ── Original domain topics ────────────────────────────────────
                "batch vs streaming pipeline architecture — Lambda vs Kappa trade-offs",
                "Apache Spark internals — DAG optimiser, shuffle, catalyst, tungsten",
                "Apache Flink — event time, watermarks, exactly-once semantics",
                "data modelling — Kimball dimensional modelling, Vault 2.0, OBT",
                "dbt — model materialisation strategies, ref, test and documentation",
                "data lake architecture — partition pruning, file format selection",
                "Iceberg / Delta Lake / Hudi — ACID on object storage, time travel",
                "orchestration — Airflow DAG design, sensors, XComs, dynamic task mapping",
                "data quality frameworks — Great Expectations, anomaly detection",
                "CDC patterns — Debezium, log-based vs polling, downstream fan-out",
                "data contracts and schema registry — Avro, Protobuf, backward compatibility",
                "data mesh principles — domain ownership, self-serve platform, federated governance",
                # ── B.Tech CSE Basic ──────────────────────────────────────────
                "relational algebra — select, project, join and how SQL maps to it",
                "B-tree indexing in column stores vs row stores — analytical query impact",
                "sorting algorithms applied to external sort in MapReduce shuffle phase",
                "hash join vs sort-merge join — when query optimiser chooses each",
                "data structures for streaming — ring buffers, sketches, sliding windows",
                "file formats — row vs columnar (CSV, JSON, Parquet, ORC, Avro) trade-offs",
                "algorithm complexity in ETL — why broadcast hash join is O(n) not O(n²)",
                "graph algorithms for lineage tracking — DAG topological sort in orchestration",
                "heap and priority queue for top-K queries in streaming aggregation",
                "discrete mathematics — set operations underpinning SQL set operators",
                "operating system file I/O — buffered vs direct I/O impact on pipeline throughput",
                "networking basics — TCP stream vs message boundary for pipeline connectors",
                "database transaction model — how MVCC enables snapshot isolation in analytics",
                "hashing in partitioning — consistent hashing for balanced Kafka partitions",
                # ── B.Tech CSE Advanced ───────────────────────────────────────
                "distributed consensus applied to streaming — Kafka controller election, ISR",
                "formal data model theory — codd's relational algebra completeness",
                "compiler optimisation theory behind Spark catalyst — predicate pushdown",
                "query optimiser internals — cost-based optimiser, cardinality estimation",
                "erasure coding in distributed storage — Reed-Solomon for data durability",
                "streaming algorithms — count-min sketch, HLL for approximate analytics",
                "lock-free structures in high-throughput ingestion pipelines",
                "formal specification of data contracts — schema evolution rules as invariants",
                "hardware-aware data engineering — SIMD for parquet decoding, NVME I/O patterns",
            ],

            # ══════════════════════════════════════════════════════════════════
            # MOBILE DEVELOPER
            # ══════════════════════════════════════════════════════════════════
            "Mobile Developer": [
                # ── Original domain topics ────────────────────────────────────
                "mobile architecture patterns — MVC, MVP, MVVM, MVI and unidirectional flow",
                "React Native new architecture — JSI, Fabric renderer, TurboModules",
                "Android Jetpack Compose — recomposition, stability, snapshot state",
                "iOS SwiftUI — view identity, lifetime, dependency tracking, Combine",
                "offline-first architecture — SQLite, Room, Core Data and sync strategies",
                "push notification pipelines — APNs, FCM, silent pushes, delivery guarantees",
                "mobile performance — frame budget, Choreographer, Instruments profiling",
                "deep linking — Universal Links, App Links, deferred deep links",
                "background processing — WorkManager, BGTaskScheduler, battery constraints",
                "mobile security — certificate pinning, jailbreak/root detection, keychain",
                "A/B testing on mobile — server-driven UI, feature flags, holdout groups",
                "app store optimisation and crash analytics — Crashlytics, Firebase Perf",
                # ── B.Tech CSE Basic ──────────────────────────────────────────
                "data structures for mobile — LRU cache for image caching, trie for search",
                "algorithm complexity on mobile — why O(n²) in list rendering causes jank",
                "OOP and protocol-oriented programming for iOS component design",
                "memory management — ARC in Swift, garbage collection in Kotlin/JVM",
                "mobile OS process model — activity lifecycle, task stacks, process death",
                "threading model — main thread, looper/handler, coroutines, Grand Central Dispatch",
                "file system on mobile — sandbox model, app containers, external storage APIs",
                "TCP/IP networking on mobile — network state changes, airplane mode handling",
                "HTTP caching on mobile — ETag, Cache-Control, conditional requests",
                "database design for mobile — SQLite internals, WAL mode, vacuum",
                "binary data handling — protocol buffers for mobile-backend communication",
                "sorting and search algorithms in list/collection view performance",
                "graph structures for navigation — back stack and navigation graph algorithms",
                "security fundamentals — symmetric encryption in local storage, biometric auth",
                # ── B.Tech CSE Advanced ───────────────────────────────────────
                "compiler theory behind code minification and R8/ProGuard shrinking",
                "JIT vs AOT compilation on mobile — ART vs Dalvik, Swift LLVM pipeline",
                "distributed sync theory — CRDT for offline-first conflict resolution",
                "formal type systems — Swift type inference, Kotlin type variance and generics",
                "hardware security — Secure Enclave, StrongBox Keymaster on Android",
                "symbolic execution for mobile vulnerability discovery",
                "reverse engineering mobile apps — smali, class-dump and mitigations",
                "GPU rendering pipeline on mobile — tile-based deferred rendering, Metal, Vulkan",
                "energy-efficient algorithm design — wake locks, Doze mode, battery historian",
            ],

            # ══════════════════════════════════════════════════════════════════
            # SCRUM MASTER
            # ══════════════════════════════════════════════════════════════════
            "Scrum Master": [
                # ── Original domain topics ────────────────────────────────────
                "Scrum events facilitation — sprint planning, daily Scrum, review, retrospective",
                "impediment removal — escalation paths, dependency management, organisational change",
                "team health metrics — cycle time, flow efficiency, DORA metrics interpretation",
                "scaled Agile — SAFe, LeSS, Nexus framework comparison",
                "Kanban principles — WIP limits, pull systems, classes of service",
                "retrospective techniques — timeline, starfish, speed boat, 5-whys",
                "team dynamics — Tuckman stages, psychological safety, conflict resolution",
                "product backlog refinement — story splitting patterns, INVEST criteria",
                "technical debt transparency — communicating debt to stakeholders",
                "Agile metrics — velocity, burn-down/up, cumulative flow diagram interpretation",
                "coaching and mentoring — Shu-Ha-Ri, Dreyfus model for Agile adoption",
                "organisational agility — value stream mapping, system thinking (Cynefin)",
                # ── B.Tech CSE Basic ──────────────────────────────────────────
                "software development lifecycle — how each SDLC phase maps to Scrum artefacts",
                "version control workflow — GitFlow vs trunk-based development impact on sprint pace",
                "algorithm complexity intuition — explaining technical estimates to stakeholders",
                "CI/CD pipeline concepts — why broken builds are Scrum impediments",
                "testing fundamentals — unit vs integration vs E2E and sprint planning implications",
                "software architecture basics — monolith vs microservices impact on team structure",
                "API design concepts — versioning and backward compatibility in release planning",
                "database migration concepts — why schema changes need careful sprint coordination",
                "container and deployment basics — understanding DevOps bottlenecks as impediments",
                "code review and pair programming — how to facilitate engineering practices",
                "incident management basics — how production incidents affect sprint commitments",
                "software quality metrics — code coverage, cyclomatic complexity as health signals",
                # ── B.Tech CSE Advanced ───────────────────────────────────────
                "formal systems thinking — feedback loops, constraints (Goldratt's TOC) in Agile",
                "queueing theory for WIP management — Little's Law in Kanban systems",
                "complexity theory for Scrum Masters — Cynefin and complicated vs complex domains",
                "distributed team collaboration tools — async communication, time-zone impact on flow",
                "machine learning basics for Scrum Masters — understanding ML sprints and research spikes",
                "platform engineering impact on developer experience and Scrum velocity",
                "formal organisation design — Conway's Law and team topology alignment",
            ],
        }

        # Generic fallback pool for any role not explicitly listed above.
        # Includes B.Tech CSE fundamentals relevant across all engineering roles.
        generic_pool = [
            "technical depth vs breadth trade-off in senior engineering roles",
            "stakeholder communication and technical-to-non-technical translation",
            "performance optimisation methodology — measure first, then optimise",
            "failure handling and incident response best practices",
            "mentoring and knowledge transfer techniques",
            "estimation and planning under uncertainty",
            "data structures fundamentals — which to use for which access pattern",
            "algorithm design techniques — divide-and-conquer, DP, greedy selection",
            "time and space complexity analysis — practical Big-O reasoning",
            "OOP principles and when composition is better than inheritance",
            "operating system concepts — processes, threads and synchronisation",
            "computer networks basics — TCP/IP, HTTP and API communication",
            "database fundamentals — ACID, indexing, normalisation",
            "software engineering practices — SDLC, design patterns, code review",
            "distributed systems principles — CAP theorem and consistency trade-offs",
            "security fundamentals — authentication, authorisation, encryption basics",
            "compiler and tooling knowledge — build systems, linting, static analysis",
            "formal methods intuition — invariants, pre/post conditions, contracts",
        ]

        selected_pool = pool.get(role, generic_pool)
        k = min(n_topics, len(selected_pool))
        return random.sample(selected_pool, k)

    # ── B.Tech CSE core topics (guaranteed 1 per session) ─────────────────────
    # These are the canonical Computer Science fundamentals taught in every
    # B.Tech CSE programme — DS&A, OS, Networks, DBMS, OOP, Discrete Maths,
    # Computer Architecture, Compiler Design, Theory of Computation.
    # One topic is always injected at position-0 of the topics list so the LLM
    # sees it first, AND Rule 9 in the prompt makes it a hard requirement.
    # This guarantees at least 1 core CS Technical question every session,
    # regardless of role (Software Engineer, Data Scientist, DevOps, etc.).
    _BTECHCSE_CORE_TOPICS: List[str] = [
        # Data Structures & Algorithms
        "arrays, linked lists, stacks and queues — internals and trade-offs",
        "binary trees, BST, AVL and red-black tree rotations",
        "graph representations (adjacency list vs matrix) and BFS/DFS traversals",
        "sorting algorithms — merge sort, quick sort, heap sort complexity",
        "dynamic programming — memoization vs tabulation trade-offs",
        "greedy algorithms and proof of correctness",
        "time and space complexity analysis (Big-O, Theta, Omega)",
        "hash maps — chaining vs open addressing, load factor, resizing",
        "recursion, backtracking and branch-and-bound",
        # Object-Oriented Programming
        "object-oriented principles — SOLID, DRY, YAGNI in practice",
        "design patterns — creational, structural, behavioural (GoF)",
        "inheritance vs composition — when to use each",
        # Operating Systems
        "process vs thread, context switching and CPU scheduling algorithms",
        "virtual memory, paging, segmentation and TLB",
        "deadlock — detection, prevention and Banker's algorithm",
        "inter-process communication — pipes, sockets, shared memory",
        "file system internals — inode structure, journaling, VFS layer",
        # Computer Networks
        "TCP/IP stack — three-way handshake, flow control, congestion control",
        "OSI model layers and how HTTP maps to them",
        "DNS resolution, ARP and the path of a web request",
        "TCP vs UDP — when to choose each",
        # Databases
        "SQL normalisation (1NF–BCNF), joins and query optimisation",
        "ACID properties and transaction isolation levels",
        "database indexing — B-tree vs hash index, when to use each",
        "ER modelling and relational algebra fundamentals",
        # Computer Architecture
        "CPU pipeline stages — fetch, decode, execute, memory, write-back",
        "cache hierarchy (L1/L2/L3), cache coherence and false sharing",
        "instruction set architecture — RISC vs CISC trade-offs",
        "bit manipulation techniques and low-level optimisation",
        # Compiler Design & Theory of Computation
        "compiler phases — lexing, parsing, AST construction, IR emission",
        "finite automata and regular expressions",
        "context-free grammars and parsing techniques",
        "NP-completeness and reduction proofs",
        # Discrete Mathematics
        "graph theory fundamentals — trees, spanning trees, shortest path",
        "counting and combinatorics — permutations, combinations, pigeonhole",
        "Boolean algebra and logic gates",
        "probability fundamentals — conditional probability, Bayes theorem",
    ]

    def _pick_btechcse_topic(self) -> str:
        """
        Pick one random B.Tech CSE core topic to guarantee every session
        includes at least one fundamental Computer Science question.
        """
        return random.choice(self._BTECHCSE_CORE_TOPICS)

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def roles(self) -> List[str]:
        return self.SUPPORTED_ROLES

    def get_questions(self, role: str, difficulty: str = "medium",
                      n: int = 10) -> List[dict]:
        """
        Generate n questions for (role, difficulty) via the Groq API.
        Serves from the in-process cache on repeat calls with the same args.
        Raises RuntimeError if the API call fails so the UI can react.
        """
        # v8.1: session_id in key ensures new session = new API call
        # hasattr guard: protects existing @st.cache_resource singletons that
        # were created before this attribute was added (AttributeError fix).
        if not hasattr(self, "_session_id"):
            self._session_id = self._new_session_id()
        if not hasattr(self, "_asked_questions"):
            self._asked_questions = []
        cache_key = (role.lower().strip(), difficulty.lower().strip(), n,
                     self._session_id)
        if cache_key in self._cache:
            log.info(f"[QuestionBank] Cache hit (same session): {cache_key[:3]}")
            return list(self._cache[cache_key])

        questions = self._fetch_from_api(role, difficulty, n)

        if not questions:
            raise RuntimeError(
                f"Failed to generate questions for role='{role}', "
                f"difficulty='{difficulty}' after 3 attempts. "
                f"Check your GROQ_API_KEY and internet connection."
            )

        self._cache[cache_key] = questions
        # Track stems for AVOID list (capped at 60)
        self._asked_questions.extend(
            q.get("question", "")[:80] for q in questions
        )
        self._asked_questions = self._asked_questions[-60:]
        return list(questions)

    def get_single_question(self,
                             role:       str,
                             difficulty: str,
                             q_type:     str = "") -> Optional[dict]:
        """
        v9.2: Generate exactly ONE question on-demand using the RL hint.

        Called by InterviewEngine.get_next_question() after every answer
        submission so the RL sequencer's type + difficulty recommendation
        is immediately reflected in the next question — not delayed until
        the pre-loaded batch runs out.

        Args:
            role:       e.g. "Software Engineer"
            difficulty: "easy" | "medium" | "hard"
            q_type:     "technical" | "behavioural" | "hr" | "" (auto)

        Returns:
            A single validated question dict, or None on failure.
            On failure, the caller falls back to the pre-loaded batch.
        """
        if not hasattr(self, "_asked_questions"):
            self._asked_questions = []

        import uuid as _uuid
        nonce = _uuid.uuid4().hex[:8]

        # Map RL type label → prompt type string
        _type_map = {
            "technical":   "Technical",
            "behavioural": "Behavioural",
            "behavioral":  "Behavioural",
            "hr":          "HR",
        }
        forced_type = _type_map.get(q_type.lower(), "")

        # Pick topics — always inject a CSE topic for Technical
        topics = self._pick_topics(role, n_topics=4)
        btechcse_topic = ""
        if forced_type == "Technical" or not forced_type:
            btechcse_topic = self._pick_btechcse_topic()
            topics.insert(0, btechcse_topic)

        prompt = self._build_single_prompt(
            role=role,
            difficulty=difficulty,
            forced_type=forced_type,
            topics=topics,
            nonce=nonce,
            btechcse_topic=btechcse_topic,
        )

        try:
            from groq import Groq
            client   = Groq(api_key=self._api_key)
            response = client.chat.completions.create(
                model=self._GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a senior interviewer. [nonce:{nonce}] "
                            "Return ONLY a JSON array with exactly 1 question object. "
                            "No markdown, no preamble, no explanation."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=0.95,
            )
            raw       = response.choices[0].message.content.strip()
            questions = self._parse_and_validate(raw, role, difficulty)
            if questions:
                q = questions[0]
                # Track in avoid list
                self._asked_questions.append(q.get("question", "")[:80])
                self._asked_questions = self._asked_questions[-60:]
                log.info(f"[QuestionBank] Single question generated: "
                         f"{q.get('type','')} / {difficulty}")
                return q
        except Exception as exc:
            log.warning(f"[QuestionBank] get_single_question failed: {exc}")

        return None   # caller falls back to pre-loaded batch

    def _build_single_prompt(self,
                              role:          str,
                              difficulty:    str,
                              forced_type:   str,
                              topics:        "List[str]",
                              nonce:         str,
                              btechcse_topic: str = "") -> str:
        """
        Prompt for exactly 1 question of a forced type and difficulty.
        Used by get_single_question() for on-demand RL-driven generation.
        """
        from datetime import datetime as _dt
        avoid_list  = self._asked_questions[-10:] if self._asked_questions else []
        avoid_line  = ("\n\nDO NOT generate questions similar to these:\n" +
                       "\n".join(f"- {q}" for q in avoid_list)) if avoid_list else ""
        timestamp   = _dt.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        topics_line = "Focus on: " + ", ".join(topics) + "." if topics else ""

        type_instruction = {
            "Technical":   "It MUST be a Technical question testing real domain knowledge.",
            "Behavioural": "It MUST be a Behavioural question — open-ended STAR-format about past experience ('Tell me about a time...').",
            "HR":          "It MUST be an HR question probing motivation, values, or career self-awareness.",
        }.get(forced_type, "Choose the most appropriate type (Technical, Behavioural, or HR).")

        diff_guidance = {
            "easy":   "Easy difficulty: fundamental concepts, clear definitions, basic STAR stories.",
            "medium": "Medium difficulty: applied knowledge, design trade-offs, specific past-experience stories.",
            "hard":   "Hard difficulty: system design, deep architectural trade-offs, complex leadership scenarios.",
        }.get(difficulty.lower(), "")

        btechcse_rule = (
            f"\nMANDATORY: Since this is a Technical question, it MUST test a core "
            f"B.Tech CSE fundamental — specifically: \"{btechcse_topic}\"."
        ) if btechcse_topic and forced_type == "Technical" else ""

        return f"""[{timestamp}] [nonce:{nonce}]
Generate exactly 1 UNIQUE interview question for the role of "{role}" at {difficulty} difficulty.

{type_instruction}
{diff_guidance}
{topics_line}{btechcse_rule}{avoid_line}

Return ONLY a JSON array with exactly 1 object with these keys:
  "role", "difficulty", "type", "question", "keywords", "ideal_answer"

keywords: 4-8 specific terms the ideal answer should contain.
ideal_answer: 60-120 words model answer at the correct difficulty level.

Generate 1 completely novel question now:"""

    def next_difficulty(self, current: str, score: float) -> str:
        # v9.2: in "all" mode the RL sequencer controls difficulty — don't override
        if current.lower() == "all":
            return "all"
        levels = ["easy", "medium", "hard"]
        cur    = current.lower()
        if cur not in levels:
            return "medium"
        idx = levels.index(cur)
        if score >= 4.0 and idx < 2: return levels[idx + 1]
        if score <= 2.0 and idx > 0: return levels[idx - 1]
        return cur

    def invalidate_cache(self, role: str = "", difficulty: str = "") -> None:
        """
        Clear cache AND rotate session_id (v8.1 fix).
        Rotating session_id ensures the old cache key (role,difficulty,n,OLD_id)
        is permanently unreachable — the next get_questions() always hits the API.
        """
        self._cache.clear()
        self._session_id = self._new_session_id()
        log.info(f"[QuestionBank] Cache cleared + session_id rotated "
                 f"-> {self._session_id}")

    # ── API generation ────────────────────────────────────────────────────────

    def _fetch_from_api(self, role: str, difficulty: str,
                        n: int) -> List[dict]:
        """
        Call Groq API and return validated question dicts.
        Returns [] only after all retry attempts are exhausted.
        """
        if not self._api_key:
            print("[QuestionBank] No GROQ_API_KEY set.")
            return []

        try:
            from groq import Groq
        except ImportError:
            print("[QuestionBank] groq package not installed. Run: pip install groq")
            return []

        import uuid as _uuid
        type_mix = self._build_type_mix(difficulty, n)
        nonce    = _uuid.uuid4().hex[:8]   # breaks Groq KV-cache
        topics   = self._pick_topics(role, n_topics=5)

        # ── Guarantee 1 B.Tech CSE core question per session ─────────────────
        # Insert a randomly chosen CSE fundamental at position-0 so the LLM
        # sees it first in the "Focus on these topics" line.  Rule 9 in the
        # prompt also makes it a hard requirement — the two mechanisms together
        # make it nearly impossible for the model to skip core CS coverage.
        btechcse_topic = self._pick_btechcse_topic()
        topics.insert(0, btechcse_topic)
        # ─────────────────────────────────────────────────────────────────────

        prompt   = self._build_prompt(role, difficulty, n, type_mix,
                                       topics=topics, nonce=nonce,
                                       btechcse_topic=btechcse_topic)

        for attempt in range(1, 4):   # up to 3 attempts
            try:
                client   = Groq(api_key=self._api_key)
                response = client.chat.completions.create(
                    model=self._GROQ_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"You are a senior technical interviewer. [nonce:{nonce}] "
                                "You return ONLY valid JSON — no markdown, no preamble, "
                                "no explanation. The JSON must be a list of question objects. "
                                "CRITICAL: Generate completely NOVEL questions. "
                                "Never repeat questions from any previous generation."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2048,
                    temperature=0.95,  # v8.1: slightly higher for more variance
                )
                raw       = response.choices[0].message.content.strip()
                questions = self._parse_and_validate(raw, role, difficulty)

                if questions:
                    print(f"[QuestionBank] Generated {len(questions)} questions "
                          f"for '{role}' / {difficulty} (attempt {attempt}).")
                    return questions

                print(f"[QuestionBank] Attempt {attempt}: 0 valid questions parsed — retrying.")

            except Exception as exc:
                print(f"[QuestionBank] Attempt {attempt} error: {type(exc).__name__}: {exc}")
                if attempt < 3:
                    import time
                    time.sleep(1.5 * attempt)

        print(f"[QuestionBank] All 3 attempts failed for '{role}' / {difficulty}.")
        return []

    # ── Prompt building ───────────────────────────────────────────────────────

    def _build_type_mix(self, difficulty: str, n: int) -> Dict[str, int]:
        # v9.2: "all" maps to medium mix for batch generation;
        # actual per-question difficulty is driven by RL in get_next_question()
        diff_key = difficulty.lower()
        if diff_key == "all":
            diff_key = "medium"
        base      = self._TYPE_MIX.get(diff_key, self._TYPE_MIX["medium"])
        total     = sum(base.values())
        result    = {}
        allocated = 0
        types     = list(base.keys())
        for i, t in enumerate(types):
            if i == len(types) - 1:
                result[t] = n - allocated
            else:
                count = max(1, round(base[t] / total * n))
                result[t] = count
                allocated += count
        return result

    def _build_prompt(self, role: str, difficulty: str,
                      n: int, type_mix: Dict[str, int],
                      topics: "List[str]" = None,
                      nonce: str = "",
                      btechcse_topic: str = "") -> str:
        """
        Build the Groq prompt with entropy signals to prevent repeated questions.
        v8.1: topics (randomly sampled domains) + nonce + avoid list injected.
        v9.1: btechcse_topic injected — Rule 9 guarantees 1 core CS question.
        """
        from datetime import datetime as _dt
        mix_desc    = ", ".join(f"{v} {k}" for k, v in type_mix.items())
        topics_line = ("Focus especially on these topic areas this session: " +
                       ", ".join(topics) + ".") if topics else ""
        # hasattr guard for cached singletons missing _asked_questions
        if not hasattr(self, '_asked_questions'):
            self._asked_questions = []
        avoid_list  = self._asked_questions[-10:] if self._asked_questions else []
        avoid_line  = ("\n\nDO NOT generate questions similar to these (already used):\n" +
                       "\n".join(f"- {q}" for q in avoid_list)) if avoid_list else ""
        timestamp   = _dt.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        diff_guidance = {
            "easy": (
                "Easy: fundamental concepts, clear definitions, "
                "basic STAR stories. Suitable for 0-2 years experience."
            ),
            "medium": (
                "Medium: applied knowledge, design trade-offs, "
                "specific past-experience stories. Suitable for 2-5 years experience."
            ),
            "hard": (
                "Hard: system design, deep architectural trade-offs, "
                "complex leadership scenarios. Senior / staff level."
            ),
        }.get(difficulty.lower(), "")

        # Rule 9 — mandatory B.Tech CSE core question
        # Only injected when btechcse_topic is provided (always in practice).
        btechcse_rule = (
            f"\n9. MANDATORY: Exactly 1 of the Technical questions MUST test a core "
            f"Computer Science fundamental from B.Tech CSE — specifically covering: "
            f'"{btechcse_topic}". '
            f"This question must be answerable by any CS graduate regardless of role-specific experience. "
            f"It must be tagged as type=Technical."
        ) if btechcse_topic else ""

        # ── Feature 10: Company-specific context injection ────────────────────
        _company_ctx = ""
        try:
            import streamlit as _st_bp
            _company_name = _st_bp.session_state.get("selected_company", "None")
            if _company_name and _company_name not in ("None", ""):
                from jd_question_engine import get_company_prompt_injection
                _company_ctx = get_company_prompt_injection(_company_name)
        except Exception:
            pass

        return f"""[{timestamp}] [nonce:{nonce}]
Generate exactly {n} UNIQUE, RANDOM interview questions for the role of "{role}" at {difficulty} difficulty.

Mix required: {mix_desc}.
{diff_guidance}
{topics_line}

Rules:
1. Every question MUST be completely different from any other session — be creative and unpredictable.
2. Technical questions must test real domain-specific knowledge for "{role}".
3. Behavioural questions must be open-ended STAR-format prompts about past experience.
4. HR questions must probe motivation, values, or career self-awareness.
5. keywords: 4-8 specific technical or conceptual terms the ideal answer should contain.
6. ideal_answer: 60-120 words, model answer at the correct difficulty level.
7. Vary topics widely — cover DIFFERENT aspects of the role each time.
8. Do NOT generate questions about basic definitions or obvious introductory topics.{btechcse_rule}{avoid_line}{_company_ctx}

Return ONLY a JSON array with exactly {n} objects. Each object must have these exact keys:
  "role", "difficulty", "type", "question", "keywords", "ideal_answer"

Example object:
{{
  "role": "{role}",
  "difficulty": "{difficulty}",
  "type": "Technical",
  "question": "Explain how a hash map works internally.",
  "keywords": ["hash function", "collision", "bucket", "load factor", "chaining"],
  "ideal_answer": "A hash map maps keys to array indices via a hash function. Collisions are resolved by chaining (linked lists per bucket) or open addressing. Load factor triggers resizing to maintain O(1) average lookup time."
}}

Generate {n} completely novel questions now:"""

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_and_validate(self, raw: str, role: str,
                             difficulty: str) -> List[dict]:
        text = raw
        for fence in ("```json", "```JSON", "```"):
            text = text.replace(fence, "")
        text  = text.strip()
        start = text.find("[")
        end   = text.rfind("]")
        if start == -1 or end == -1:
            print(f"[QuestionBank] No JSON array in response: {raw[:200]}")
            return []
        try:
            questions = json.loads(text[start: end + 1])
        except json.JSONDecodeError as exc:
            print(f"[QuestionBank] JSON parse error: {exc}. Raw: {raw[:200]}")
            return []
        if not isinstance(questions, list):
            return []

        required = {"role", "difficulty", "type", "question",
                    "keywords", "ideal_answer"}
        valid = []
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                continue
            missing = required - q.keys()
            if missing:
                print(f"[QuestionBank] Question {i} missing keys: {missing}")
                continue
            if not isinstance(q.get("keywords"), list):
                q["keywords"] = []
            q["role"]       = role
            q["difficulty"] = difficulty.lower()
            valid.append(q)

        print(f"[QuestionBank] Validated {len(valid)}/{len(questions)} questions.")
        return valid


# ══════════════════════════════════════════════════════════════════════════════
#  NLP SCORER
# ══════════════════════════════════════════════════════════════════════════════

class NLPScorer:
    """
    STAR + keyword + TF-IDF + depth scorer.
    Same formula as v5.0 — unchanged since answer_evaluator.py handles deep NLP.
    """

    def score(self, answer: str, question: dict) -> dict:
        if not answer or len(answer.strip()) < 5:
            return dict(
                score=1.0, star_scores={}, disc_traits={},
                keyword_hits=[], tfidf_sim=0.0,
                feedback="No answer provided.",
            )
        al = answer.lower()

        # STAR
        star_scores = {
            c: bool(re.search(p, al, re.IGNORECASE))
            for c, p in STAR_PATTERNS.items()
        }
        star_sc = sum(star_scores.values()) * 1.25  # max 5.0

        # Keyword relevance
        exp_kw = [k.lower() for k in question.get("keywords", [])]
        hits   = [k for k in exp_kw if k in al]
        kw_sc  = min(2.0, len(hits) / max(1, len(exp_kw)) * 2.0)

        # TF-IDF similarity
        tfidf_sim = 0.0
        if SKLEARN_NLP_AVAILABLE and exp_kw:
            try:
                ref  = " ".join(exp_kw)
                vect = TfidfVectorizer(ngram_range=(1, 2)).fit([ref, answer])
                vecs = vect.transform([ref, answer])
                tfidf_sim = float(cosine_similarity(vecs[0], vecs[1])[0][0])
            except Exception:
                pass

        # Word-count depth bonus
        wc_bonus = min(0.5, len(answer.split()) / 200 * 0.5)

        # DISC
        disc = {
            tr: sum(1 for w in ws if w in al)
            for tr, ws in DISC_KEYWORDS.items()
        }

        score = round(min(5.0, max(1.0,
            star_sc  * 0.45
            + kw_sc  * 0.30
            + tfidf_sim * 5 * 0.15
            + wc_bonus  * 5 * 0.10
        )), 2)

        miss = [k for k, v in star_scores.items() if not v]
        fb   = []
        if miss:           fb.append(f"Include STAR components: {', '.join(miss)}.")
        if exp_kw and not hits: fb.append(f"Use relevant keywords: {', '.join(exp_kw[:4])}.")
        if tfidf_sim < .3: fb.append("Stay focused on the question topic.")

        return dict(
            score=score,
            star_scores=star_scores,
            disc_traits=disc,
            keyword_hits=hits,
            tfidf_sim=round(tfidf_sim, 3),
            feedback=(" ".join(fb) if fb else "Good answer structure and relevance."),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  NERVOUSNESS FUSION  (new v7.0)
# ══════════════════════════════════════════════════════════════════════════════

class NervousnessFusion:
    """
    Uses 100% voice nervousness score — facial signal excluded entirely.

    Research basis:
      - Voice acoustic biomarkers (pitch variance, pause ratio, ZCR) are
        the most reliable anxiety indicators available (Low et al. 2020,
        Riad et al. JMIR 2024).
      - Facial expressions can be deliberately controlled under interview
        stress; voice biomarkers are much harder to mask (Marmar et al. 2019).
      - v10.0: facial weight set to 0.0 — 100% voice.
    """

    def __init__(self) -> None:
        self._history: List[float] = []          # per-answer fused scores
        self._facial_history: List[float] = []
        self._voice_history:  List[float] = []

    def fuse(self, facial_nervousness: float,
              voice_nervousness: float) -> float:
        """
        v10.0: Returns voice_nervousness directly (100% voice, 0% facial).
        facial_nervousness parameter kept for API compatibility but ignored.
        """
        fused = (
            NERVOUSNESS_FUSION["facial"] * facial_nervousness
            + NERVOUSNESS_FUSION["voice"]  * voice_nervousness
        )
        return round(float(np.clip(fused, 0.0, 1.0)), 3)

    def record(self, facial_n: float, voice_n: float) -> float:
        """Record per-answer nervousness snapshot."""
        fused = self.fuse(facial_n, voice_n)
        self._history.append(fused)
        self._facial_history.append(facial_n)
        self._voice_history.append(voice_n)
        return fused

    def get_session_nervousness(self) -> float:
        """Mean fused nervousness over the whole session."""
        return round(float(np.mean(self._history)), 3) if self._history else 0.2

    def get_nervousness_trend(self) -> str:
        """
        Returns 'improving', 'stable', or 'worsening' based on
        per-answer nervousness trajectory.
        """
        if len(self._history) < 3:
            return "stable"
        first_half = np.mean(self._history[:len(self._history) // 2])
        second_half = np.mean(self._history[len(self._history) // 2:])
        diff = second_half - first_half
        if diff < -0.05:   return "improving"   # nervousness decreased
        if diff > +0.05:   return "worsening"   # nervousness increased
        return "stable"

    def get_per_answer_nervousness(self) -> List[float]:
        return list(self._history)

    def get_nervousness_level(self, score: Optional[float] = None) -> str:
        s = score if score is not None else self.get_session_nervousness()
        if s >= 0.65: return "High"
        if s >= 0.35: return "Moderate"
        return "Low"

    def get_summary(self) -> Dict:
        return {
            "session_nervousness":   self.get_session_nervousness(),
            "nervousness_level":     self.get_nervousness_level(),
            "nervousness_trend":     self.get_nervousness_trend(),
            "per_answer_nervousness":self.get_per_answer_nervousness(),
            "facial_mean":           round(float(np.mean(self._facial_history)), 3)
                                     if self._facial_history else 0.2,
            "voice_mean":            round(float(np.mean(self._voice_history)), 3)
                                     if self._voice_history else 0.2,
            "fusion_weights":        NERVOUSNESS_FUSION,
        }

    def reset(self) -> None:
        self._history.clear()
        self._facial_history.clear()
        self._voice_history.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  VOICE QUALITY INDEX  (new v7.0)
# ══════════════════════════════════════════════════════════════════════════════

class VoiceQualityIndex:
    """
    Aggregates prosodic quality signals from the voice session summary.

    Metrics:
      • Confidence: 1 - nervousness  (higher = better)
      • Dominance:  proportion of high-confidence emotions (Happy, Calm, Neutral)
      • Consistency: low variance in emotion predictions = stable voice
    """

    def __init__(self) -> None:
        self._snapshots: List[Dict] = []

    def record(self, voice_result: Dict) -> None:
        if voice_result:
            self._snapshots.append(voice_result)

    def compute(self) -> Dict:
        if not self._snapshots:
            return {
                "vqi_score":        3.5,
                "confidence_mean":  0.5,
                "dominance_ratio":  0.5,
                "stability":        0.5,
                "rating":           "Moderate",
            }

        # Mean confidence from model (0-100 → 0-1)
        conf_vals = [s.get("confidence", 50.0) / 100.0 for s in self._snapshots]
        conf_mean = float(np.mean(conf_vals))

        # Dominance: ratio of frames with low-nervousness dominant emotion
        low_nerv = {"Neutral", "Calm", "Happy", "Pleasant"}
        dom_count = sum(
            1 for s in self._snapshots
            if s.get("dominant", "Neutral") in low_nerv
        )
        dom_ratio = dom_count / max(1, len(self._snapshots))

        # Stability: low nervousness variance = stable voice
        nerv_vals = [s.get("nervousness", 0.2) for s in self._snapshots]
        nerv_var  = float(np.var(nerv_vals))
        stability = max(0.0, 1.0 - nerv_var * 5)

        # VQI score (on 1-5 scale)
        raw = conf_mean * 0.35 + dom_ratio * 0.40 + stability * 0.25
        vqi = round(max(1.0, min(5.0, raw * 5.0)), 2)

        rating = ("Excellent" if vqi >= 4.0 else
                  "Good"      if vqi >= 3.0 else
                  "Moderate"  if vqi >= 2.0 else "Needs Work")

        return {
            "vqi_score":       vqi,
            "confidence_mean": round(conf_mean, 3),
            "dominance_ratio": round(dom_ratio, 3),
            "stability":       round(stability, 3),
            "rating":          rating,
        }

    def reset(self) -> None:
        self._snapshots.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  CONSISTENCY ANALYSER
# ══════════════════════════════════════════════════════════════════════════════

class ConsistencyAnalyzer:
    FILLERS = [
        "um", "uh", "like", "basically", "actually", "you know",
        "right", "so", "just", "kind of", "sort of", "i mean", "literally",
    ]

    def analyze_consistency(self, text: str, duration_seconds: float) -> Dict:
        if not text or duration_seconds == 0:
            return {"score": 0, "status": "No data", "wpm": 0, "fluency": 0, "fillers": 0}
        words      = text.lower().split()
        word_count = len(words)
        wpm        = (word_count / duration_seconds) * 60
        fillers    = [w for w in words if w in self.FILLERS]
        filler_r   = len(fillers) / max(1, word_count)
        fluency    = max(0, 100 - filler_r * 500)
        pace_score = max(0, 100 - abs(140 - wpm))
        overall    = fluency * 0.6 + pace_score * 0.4
        status = ("Excellent"    if overall > 85 else
                  "Stable"       if overall > 60 else "Inconsistent")
        return {
            "score":   round(overall, 1),
            "status":  status,
            "wpm":     int(wpm),
            "fillers": len(fillers),
            "fluency": round(fluency, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  AURA ANALYTICS ENGINE  (live right-panel metrics)
# ══════════════════════════════════════════════════════════════════════════════

class AuraAnalyticsEngine:
    def __init__(self) -> None:
        self._consistency = ConsistencyAnalyzer()

    def analyze_answer_quality(self, text: str, q_dict: dict) -> Dict:
        if not text or not text.strip():
            return {"score": 0, "star": {}, "hits": [], "word_count": 0, "keywords": []}
        al   = text.lower()
        star = {c: bool(re.search(p, al, re.IGNORECASE)) for c, p in STAR_PATTERNS.items()}
        kws  = [k.lower() for k in q_dict.get("keywords", [])]
        hits = [k for k in kws if k in al]
        wc   = len(text.split())
        star_sc = sum(star.values()) / 4 * 100
        kw_sc   = len(hits) / max(1, len(kws)) * 100
        depth   = min(100, wc / 150 * 100)
        score   = star_sc * 0.35 + kw_sc * 0.35 + depth * 0.30
        return {
            "score": round(score, 1), "star": star,
            "hits": hits, "word_count": wc, "keywords": kws,
        }

    def analyze_consistency(self, text: str, duration: float) -> Dict:
        return self._consistency.analyze_consistency(text, duration)

    def compute_master_score(self, nlp: Dict, emotion: Dict,
                              consistency: Dict) -> Dict:
        nlp_n   = nlp.get("score", 0) / 100
        nerv_n  = 1.0 - emotion.get("nervousness", 0.5)
        cons_n  = consistency.get("score", 50) / 100
        total   = round((nlp_n * 0.45 + nerv_n * 0.30 + cons_n * 0.25) * 100)
        total   = max(0, min(100, total))
        verdict = ("Excellent" if total >= 80 else
                   "Good"      if total >= 60 else
                   "Average"   if total >= 40 else "Needs Work")
        return {"total": total, "verdict": verdict}


# ══════════════════════════════════════════════════════════════════════════════
#  SCORE AGGREGATOR
# ══════════════════════════════════════════════════════════════════════════════

class ScoreAggregator:

    @staticmethod
    def combine(emotion: float, voice: float, knowledge: float,
                avg_depth: float = 0.0) -> Dict:
        """
        v8.0: posture + confidence removed. Formula unchanged:
            final = knowledge×0.70 + emotion×0.15 + voice×0.15

        avg_depth is tracked separately and surfaced in the report
        as its own metric — it is already embedded in knowledge via
        the depth_fluency sub-score, so it does NOT add to the formula.
        """
        final = round(
            knowledge * WEIGHTS["knowledge"] +
            emotion   * WEIGHTS["emotion"]   +
            voice     * WEIGHTS["voice"],
            2,
        )
        return dict(
            emotion=emotion, voice=voice, knowledge=knowledge,
            depth=avg_depth,           # explicit depth metric for report
            final=final, weights=WEIGHTS,
        )

    # personality_score() removed in v7.1 — personality dimension dropped.

    @staticmethod
    def compute_confidence_score(eye_score: float,
                                  fluency_score: float,
                                  voice_score: float,
                                  facial_score: float,
                                  posture_score: float = 3.5) -> float:
        """
        Confidence v8.1 = 0.25×Eye + 0.25×Fluency + 0.35×Voice + 0.15×Facial

        All inputs on 1-5 scale.
        posture_score kept as ignored kwarg for backwards-compat call sites.
        """
        conf = (
            eye_score     * CONFIDENCE_WEIGHTS["eye"]     +
            fluency_score * CONFIDENCE_WEIGHTS["fluency"] +
            voice_score   * CONFIDENCE_WEIGHTS["voice"]   +
            facial_score  * CONFIDENCE_WEIGHTS["facial"]
        )
        return round(max(1.0, min(5.0, conf)), 2)


# ══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class PerformanceTracker:
    def __init__(self) -> None:
        self.question_scores: List[float] = []

    def add_score(self, s: float) -> None:
        self.question_scores.append(s)

    def get_progress(self) -> List[float]:
        return list(self.question_scores)

    def get_trend(self) -> str:
        if len(self.question_scores) < 2:
            return "stable"
        diff = self.question_scores[-1] - self.question_scores[-2]
        return ("improving" if diff > 0.2 else
                "declining"  if diff < -0.2 else "stable")


# ══════════════════════════════════════════════════════════════════════════════
#  INTERVIEW ENGINE  (v7.0)
# ══════════════════════════════════════════════════════════════════════════════

class InterviewEngine:
    """
    Main interview orchestration engine.

    Pipelines:
      • FERPipeline     — facial emotion (HOG + MLP, FER-2013)
      • UnifiedVoicePipeline — voice emotion + nervousness (CREMA-D + TESS, 108-dim)
      • MediaPipe Holistic  — posture, EAR, head pose (inside FERPipeline)

    New in v7.0:
      • NervousnessFusion — per-answer fused facial+voice nervousness (0.45/0.55)
      • VoiceQualityIndex — session-level VQI aggregation
      • Enhanced final_report with nervousness trajectory + VQI
    """

    def __init__(self, question_file: str = "question_pool.json") -> None:
        # question_file kept for backwards-compat but no longer used — questions
        # are generated live via the Groq API inside QuestionBank.
        groq_key = os.environ.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
        self.qbank       = QuestionBank(groq_api_key=groq_key)
        # v7.3 fix: use AnswerEvaluator (ideal_answer-aware TF-IDF) when available,
        # fall back to NLPScorer (keywords-only TF-IDF) if import failed.
        self.nlp_scorer = AnswerEvaluator(groq_api_key=groq_key) if ANSWER_EVALUATOR_AVAILABLE else NLPScorer()
        self._using_answer_evaluator = ANSWER_EVALUATOR_AVAILABLE
        self.performance = PerformanceTracker()
        self.aura_engine = AuraAnalyticsEngine()

        # ── Session state — declared FIRST so all blocks below can reference them ─
        self.questions:     List[dict]  = []
        self.current_index: int         = 0
        self.answers:       List[dict]  = []
        self.disc_history:  List[dict]  = []
        self.scores:        List[float] = []
        self.role:          str         = "Software Engineer"
        self.difficulty:    str         = "medium"

        # These are set properly by load_questions() but must exist here so any
        # method that reads them never raises AttributeError if load_questions()
        # hasn't been called yet (e.g. interview_finished(), is_follow_up_pending()).
        self._num_questions:     int  = 0
        self._original_q_count:  int  = 0
        self._follow_up_pending: bool = False

        # ── RL Adaptive Sequencer (v8.0) ──────────────────────────────────────
        # Replaces the static two-line QuestionBank.next_difficulty() heuristic
        # with a Q-learning agent that learns per-candidate optimal sequencing.
        # Research: Patel et al. (2023, Springer AI Review)
        # NOTE: initialised after self.role so RLAdaptiveSequencer(role=self.role) works.
        #
        # _rl_active is set properly by load_questions() based on difficulty mode.
        # Initialised to False here so get_next_question() never hits AttributeError
        # if it is called before load_questions() (e.g. during session warm-up).
        self._rl_active: bool = False
        if RL_SEQUENCER_AVAILABLE:
            self.sequencer = RLAdaptiveSequencer(role=self.role)
            self._rl_next_hint: Dict = {}   # set by evaluate_answer(), read by get_next_question()
            print("[InterviewEngine] RL adaptive sequencer attached.")
        else:
            self.sequencer       = None
            self._rl_next_hint   = {}
        self._resume_parsed: dict = {}   # v9.2: set by app.py after resume upload

        # ── FER pipeline (facial + posture) ──────────────────────────────
        self.fer_pipeline: Optional[FERPipeline] = (
            FERPipeline() if FER_AVAILABLE else None
        )
        self._fer_ready = False

        # ── Unified voice pipeline (CREMA-D + TESS) ───────────────────────
        self.voice_pipeline: Optional[UnifiedVoicePipeline] = (
            UnifiedVoicePipeline() if VOICE_AVAILABLE else None
        )
        self._voice_ready = False

        # ── New v7.0 components ───────────────────────────────────────────
        self.nervousness_fusion = NervousnessFusion()
        self.voice_quality      = VoiceQualityIndex()

        # Live signal accumulators (current question window)
        self._live_emotion_scores:  List[float] = []
        self._live_voice_scores:    List[float] = []
        self._live_posture_scores:  List[float] = []
        self._live_eye_scores:      List[float] = []   # EAR + gaze contact (0-5 scale)
        self._live_fluency_scores:  List[float] = []   # filler + WPM fluency (0-5 scale)

        # Per-answer snapshots (multimodal at submission time)
        self._answer_emotion_snapshots: List[Dict] = []
        self._answer_voice_snapshots:   List[Dict] = []

    # ── Pipeline Setup ────────────────────────────────────────────────────────

    def setup_emotion_pipeline(
        self,
        force_retrain: bool = False,
        max_train_samples: int = 20000,
        progress_callback=None,
    ) -> Dict:
        """Setup FER-2013 facial emotion pipeline."""
        if self.fer_pipeline is None:
            return {"error": "dataset_loader not available"}
        try:
            metrics = self.fer_pipeline.setup(
                force_retrain=force_retrain,
                max_train_samples=max_train_samples,
                progress_callback=progress_callback,
            )
            self._fer_ready = True
            return metrics
        except Exception as exc:
            return {"error": str(exc)}

    def setup_voice_pipeline(
        self,
        force_retrain: bool = False,
        max_per_dataset: int = 3000,
        progress_callback=None,
    ) -> Dict:
        """
        Setup unified voice pipeline (CREMA-D + TESS, 108-dim).
        max_per_dataset: max WAV files per dataset (CREMA-D and TESS each).
        """
        if self.voice_pipeline is None:
            return {"error": "unified_voice_pipeline not available"}
        try:
            metrics = self.voice_pipeline.setup(
                force_retrain=force_retrain,
                max_per_dataset=max_per_dataset,
                progress_cb=progress_callback,
            )
            self._voice_ready = True
            return metrics
        except Exception as exc:
            return {"error": str(exc)}

    def setup_all_pipelines(
        self,
        force_retrain: bool = False,
        max_fer_samples: int = 20000,
        max_ravdess_samples: int = 3000,  # now means max_per_dataset for CREMA-D/TESS
        progress_callback=None,
    ) -> Dict:
        """
        Setup all pipelines sequentially.
        max_ravdess_samples parameter kept for backwards compatibility but now
        controls max_per_dataset for the unified voice pipeline (CREMA-D + TESS).
        """
        results = {}

        def _cb(msg: str) -> None:
            if progress_callback: progress_callback(msg)
            print(msg)

        _cb("Setting up FER-2013 facial emotion pipeline…")
        results["fer"] = self.setup_emotion_pipeline(
            force_retrain, max_fer_samples, _cb
        )

        _cb("Setting up Unified Voice pipeline (CREMA-D + TESS, 108-dim)…")
        results["voice"] = self.setup_voice_pipeline(
            force_retrain, max_ravdess_samples, _cb
        )

        _cb("All pipelines ready!")
        return results

    def is_emotion_pipeline_ready(self) -> bool:
        return self._fer_ready and self.fer_pipeline is not None

    def is_voice_pipeline_ready(self) -> bool:
        return self._voice_ready and self.voice_pipeline is not None

    def start_live_voice(self) -> bool:
        if self.voice_pipeline and self._voice_ready:
            return self.voice_pipeline.start_live()
        return False

    def stop_live_voice(self) -> None:
        if self.voice_pipeline:
            self.voice_pipeline.stop_live()

    # ── Webcam frame processing ───────────────────────────────────────────────

    def analyse_webcam_frame(self, frame_bgr) -> Tuple:
        """Process one webcam frame through FER + posture."""
        if not self.is_emotion_pipeline_ready() or self.fer_pipeline is None:
            dummy = (_dummy_result() if FER_AVAILABLE else {
                "dominant": "Neutral", "emotions": {}, "nervousness": 0.2,
                "smoothed_nervousness": 0.2, "confidence": 50.0,
                "emotion_history": [], "posture": {},
            })
            return frame_bgr, dummy

        annotated, result = self.fer_pipeline.analyse_frame(frame_bgr)

        # Accumulate live emotion score (facial confidence)
        self._live_emotion_scores.append(
            result.get("confidence", 50.0) / 100 * 5
        )

        # Accumulate posture score (stub — kept for compat)
        posture = result.get("posture", {})
        if posture and posture.get("detected", False):
            self._live_posture_scores.append(
                posture.get("confidence_score", 3.5)
            )

        # Accumulate eye contact score (v8.1 — new confidence component)
        # Formula: base from EAR openness + gaze penalty
        #   openness_score  0-5 from EyeAnalyser (5=fully open, 0=blink)
        #   gaze_direct     True/False — iris pointing at camera
        # Combined: if gaze is direct, use openness directly;
        #           if averted, penalise by 40% (Burgoon et al. 1990)
        eye_open   = result.get("eye", {}).get("openness_score",
                     posture.get("raw_scores", {}).get("eye_contact", 3.5))
        gaze_ok    = result.get("gaze_direct",
                     result.get("eye", {}).get("gaze_direct", True))
        eye_score  = float(eye_open) * (1.0 if gaze_ok else 0.60)
        eye_score  = max(1.0, min(5.0, eye_score))
        self._live_eye_scores.append(eye_score)

        return annotated, result

    # ── Voice results ─────────────────────────────────────────────────────────

    def get_live_voice_result(self) -> Dict:
        """Get the latest smoothed voice emotion prediction."""
        if self.voice_pipeline and self._voice_ready:
            result = self.voice_pipeline.get_latest()
            # Accumulate voice confidence score (0-5)
            voice_conf = result.get("confidence", 50.0) / 100 * 5
            self._live_voice_scores.append(voice_conf)
            # Record for VQI and fluency tracking
            self.voice_quality.record(result)

            # Accumulate fluency score from prosodic stability (v8.1)
            # Maps nervousness variance → fluency proxy (low nervousness
            # variance = steady vocal delivery = high fluency score)
            # Uses voice nervousness as a live fluency proxy:
            #   nervousness 0.0 → fluency 5.0 (calm = fluent delivery)
            #   nervousness 1.0 → fluency 1.0 (high stress = disfluent)
            # This is a real-time proxy; per-answer fluency is computed
            # more accurately from transcript in ConsistencyAnalyzer.
            v_nerv = result.get("nervousness", 0.2)
            fluency_proxy = max(1.0, min(5.0, (1.0 - v_nerv) * 4.0 + 1.0))
            self._live_fluency_scores.append(fluency_proxy)
            return result
        return _dummy_voice()

    def process_voice_audio(self, audio_bytes: bytes) -> Dict:
        """
        Process raw WAV bytes through the unified voice pipeline.
        Use for Streamlit st.audio_input integration.
        """
        if self.voice_pipeline and self._voice_ready:
            result = self.voice_pipeline.predict_from_bytes(audio_bytes)
            self.voice_quality.record(result)
            return result
        return _dummy_voice()

    def calibrate_voice_baseline(self, audio_bytes: bytes) -> Dict:
        """
        v8.1: Record the candidate's personal nervousness floor.
        Call ONCE before question 1 with ~30s of neutral speech audio.

        Delegates to UnifiedVoicePipeline.calibrate_baseline() which
        averages nervousness over n_windows segments and stores the result.
        All subsequent nervousness scores are reported as delta from this
        baseline, reducing systematic demographic and individual bias.

        Returns:
            {
              "baseline":    float  — raw model score used as floor (0-1)
              "calibrated":  bool   — True on success
              "message":     str    — human-readable confirmation
            }
        """
        if not (self.voice_pipeline and self._voice_ready):
            return {
                "baseline":   0.2,
                "calibrated": False,
                "message":    "Voice pipeline not ready — baseline not set.",
            }
        baseline = self.voice_pipeline.calibrate_baseline(audio_bytes)
        return {
            "baseline":   baseline,
            "calibrated": self.voice_pipeline.baseline_calibrated,
            "message":    (
                f"✅ Baseline set: {baseline:.2f} "
                f"({'high' if baseline > 0.5 else 'moderate' if baseline > 0.3 else 'low'} "
                f"natural voice nervousness)"
            ),
        }

    def get_voice_session_summary(self) -> Dict:
        """Aggregated voice emotion summary for the session."""
        if self.voice_pipeline and self._voice_ready:
            return self.voice_pipeline.get_session_summary()
        return {"dominant": "Neutral", "nervousness": 0.2, "distribution": {}}

    def get_prosodic_analysis(self) -> Dict:
        """Proxy prosodic analysis from voice pipeline."""
        if self.voice_pipeline and self._voice_ready:
            return self.voice_pipeline.get_prosodic_analysis()
        return {}

    def get_voice_quality_index(self) -> Dict:
        """Session-level voice quality report."""
        return self.voice_quality.compute()

    # ── Multimodal confidence ─────────────────────────────────────────────────

    def get_multimodal_confidence(self) -> Dict:
        """
        Confidence v8.1 = 0.25×Eye + 0.25×Fluency + 0.35×Voice + 0.15×Facial
        All components on 1-5 scale.

        Sources:
            Eye     — EyeAnalyser.openness_score × gaze_direct penalty
                      (live EAR + iris tracking from MediaPipe FaceMesh)
            Fluency — voice nervousness inverse proxy, updated per voice frame
                      (low nervousness = steady delivery = high fluency)
            Voice   — UnifiedVoicePipeline model confidence (0-100 → 1-5)
            Facial  — DeepFace/FER model confidence on dominant emotion (0-100 → 1-5)
        """
        eye_s     = (float(np.mean(self._live_eye_scores))
                     if self._live_eye_scores     else 3.5)
        fluency_s = (float(np.mean(self._live_fluency_scores))
                     if self._live_fluency_scores  else 3.5)
        voice_s   = (float(np.mean(self._live_voice_scores))
                     if self._live_voice_scores    else 3.5)
        facial_s  = (float(np.mean(self._live_emotion_scores))
                     if self._live_emotion_scores  else 3.5)

        conf = ScoreAggregator.compute_confidence_score(
            eye_score=eye_s, fluency_score=fluency_s,
            voice_score=voice_s, facial_score=facial_s,
        )
        return {
            "confidence_score": conf,
            "eye_score":        round(eye_s,     2),
            "fluency_score":    round(fluency_s, 2),
            "voice_score":      round(voice_s,   2),
            "facial_score":     round(facial_s,  2),
            "posture_score":    3.5,   # stub — removed in v8.1
            "weights":          CONFIDENCE_WEIGHTS,
        }

    # ── Fused nervousness ─────────────────────────────────────────────────────

    def get_fused_nervousness(self) -> float:
        """
        v10.0: Returns 100% voice nervousness — facial signal excluded.
        """
        voice_n = self.get_voice_session_summary().get("nervousness", 0.2)
        return self.nervousness_fusion.fuse(0.0, voice_n)

    # ── Emotion feedback ──────────────────────────────────────────────────────

    def emotion_feedback(self) -> Dict:
        if self.fer_pipeline and self._fer_ready:
            return self.fer_pipeline.get_feedback()
        return {}

    def get_emotion_summary(self) -> Dict:
        if self.fer_pipeline and self._fer_ready:
            return self.fer_pipeline.get_session_summary()
        return {"dominant": "Neutral", "nervousness": 0.2, "distribution": {}}

    # ── Session management ────────────────────────────────────────────────────

    def start_session(
        self,
        role: str = "Software Engineer",
        difficulty: str = "medium",
        num_questions: int = 5,
    ) -> List[dict]:
        self.role       = role
        self.difficulty = difficulty.lower()
        # v9.2: track original questions separately from follow-up sub-parts
        # so interview_finished() counts only original questions, not follow-ups.
        self._num_questions     = num_questions   # target number of original Qs
        self._original_q_count  = 0              # original Qs answered so far
        self._follow_up_pending = False           # True when RL fired action 7

        # Always clear cache so every new session gets freshly generated questions
        self.qbank.invalidate_cache(role=role, difficulty=self.difficulty)

        # ── RL sequencer: ONLY active in "all" (RL Adaptive) mode ────────────
        # Easy / Medium / Hard modes bypass RL entirely — questions are generated
        # as a pre-loaded batch by Groq with a fixed type mix, starting with a
        # Behavioural question (Q1 always). Follow-up in those modes is handled
        # by follow_up_engine based on content analysis, not RL actions.
        #
        # "all" mode: RL sequencer fully active — controls both type AND
        # difficulty per question, including follow-up (action 7).
        self._rl_active = (self.difficulty == "all")

        if self._rl_active and self.sequencer is not None:
            # ── RL Adaptive mode ─────────────────────────────────────────────
            self.sequencer._role = role.lower().replace(" ", "_")
            self.sequencer.reset_session()
            loaded = self.sequencer.load()

            # ── Resume calibration: recover from session state if needed ──────
            # resume_rephraser.load_into_interview() writes engine._resume_parsed
            # directly when the engine reference is live.  But if the engine was
            # re-initialised between the Resume page and Start Interview (e.g. a
            # Streamlit rerun that recreated the engine object), _resume_parsed
            # will be empty again.  We recover from session_state as a fallback.
            _resume = getattr(self, '_resume_parsed', {}) or {}
            if not _resume:
                try:
                    import streamlit as _st
                    _resume = _st.session_state.get("resume_parsed_for_rl", {}) or {}
                    if _resume:
                        self._resume_parsed = _resume   # re-populate for future calls
                        log.info("[InterviewEngine] Recovered resume_parsed_for_rl "
                                 "from session_state for RL calibration.")
                except Exception:
                    pass   # non-fatal — RL falls back to warm-start or cold-start

            first_action = self.sequencer.get_first_action(
                resume_parsed=_resume,
                session_difficulty="all",   # RL picks freely
            )
            self._rl_next_hint = {
                "type":       first_action.q_type,
                "difficulty": first_action.difficulty or "medium",
                "follow_up":  False,
            }
            print(f"[InterviewEngine] RL mode — cold/warm started "
                  f"→ Q1 hint: {first_action.label()}")
        else:
            # ── Fixed-difficulty mode (Easy / Medium / Hard) ──────────────────
            # RL sequencer is idle. Reset any stale hint so get_next_question()
            # uses the pre-loaded buffer path, not the on-demand RL path.
            self._rl_next_hint = {}
            if self.sequencer is not None:
                self.sequencer.reset_session()   # clear state but don't activate

        if self._rl_active:
            # ── RL mode: pre-load a small buffer as fallback only ─────────────
            # Real questions are generated on-demand per RL hint in get_next_question().
            # The buffer is there in case Groq fails mid-session.
            self.questions     = self.qbank.get_questions(role, "medium", num_questions)
            self.current_index = 0
        else:
            # ── Fixed mode: pre-load all questions, Behavioural goes first ────
            # Groq generates the full batch at once with the fixed type mix.
            # Q1 is always Behavioural — puts the candidate at ease with a
            # story-format question before moving to domain-specific Technical.
            batch = self.qbank.get_questions(role, self.difficulty, num_questions)

            # ── HR dataset Q1 injection ───────────────────────────────────────
            # In Easy / Medium / Hard modes the first question is always
            # Behavioural.  Instead of using a Groq-generated one, we pull a
            # random matching question from hr_interview_dataset.json so
            # candidates practise with real, curated behavioural questions.
            # If the dataset is unavailable the code falls through to the
            # normal Groq-sorted batch without error.
            hr_q1 = self.qbank.get_hr_dataset_q1(
                difficulty = self.difficulty,
                role       = role,
            )
            if hr_q1 is not None:
                # Remove the first Groq-generated Behavioural question from the
                # batch (to keep total count stable), then prepend the dataset one.
                _type_order = {"behavioural": 0, "behavioral": 0, "hr": 1, "technical": 2}
                # Sort remaining batch: HR → Technical (Behavioural slot is filled)
                non_beh = [q for q in batch
                           if q.get("type", "").lower() not in ("behavioural", "behavioral")]
                non_beh.sort(key=lambda q: _type_order.get(
                    q.get("type", "technical").lower(), 2))
                self.questions     = [hr_q1] + non_beh
                log.info(f"[InterviewEngine] Q1 from HR dataset: {hr_q1['question'][:60]}")
            else:
                # Dataset unavailable — fall back to original sort
                _type_order = {"behavioural": 0, "behavioral": 0, "hr": 1, "technical": 2}
                batch.sort(key=lambda q: _type_order.get(
                    q.get("type", "technical").lower(), 2))
                self.questions = batch

            self.current_index = 0
        self.answers       = []
        self.scores        = []
        self.disc_history  = []
        self.performance   = PerformanceTracker()

        # Reset live accumulators
        self._live_emotion_scores  = []
        self._live_voice_scores    = []
        self._live_posture_scores  = []
        self._live_eye_scores      = []
        self._live_fluency_scores  = []
        self._answer_emotion_snapshots = []
        self._answer_voice_snapshots   = []

        # Reset v7.0 components
        self.nervousness_fusion.reset()
        self.voice_quality.reset()

        # Reset voice pipeline EMA for fresh session
        if self.voice_pipeline and self._voice_ready:
            self.voice_pipeline.reset_session()
            # v8.1: reset calibration baseline so each new candidate starts fresh
            self.voice_pipeline.reset_baseline()

        return self.questions

    # ── Questions ─────────────────────────────────────────────────────────────

    def get_next_question(self) -> str:
        """
        Return the next question text.

        TWO COMPLETELY SEPARATE PATHS:

        FIXED-MODE (Easy / Medium / Hard):
          Questions come from the pre-loaded Behavioural-first sorted batch.
          No RL involvement. Simple sequential pop from self.questions.
          Follow-up questions are handled by follow_up_engine in app.py
          based on content analysis (low score, missing STAR, low relevance).

        RL MODE (All / RL Adaptive):
          Every question is generated on-demand by calling Groq with the
          current RL hint {type, difficulty}. The agent controls both
          dimensions. Follow-up (action 7) is an RL decision — the agent
          fires it when it detects a shallow answer via _shallow_answer_detected().
          Fallback to pre-loaded buffer if Groq fails.
        """
        if not self._rl_active:
            # ── FIXED MODE: sequential from pre-loaded Behavioural-first batch ─
            if self.current_index < len(self.questions):
                q = self.questions[self.current_index]
                self.current_index += 1
                return q.get("question", "Tell me about yourself.")
            # Buffer exhausted — fetch a small top-up batch
            top_up = self.qbank.get_questions(self.role, self.difficulty, 5)
            _type_order = {"behavioural": 0, "behavioral": 0, "hr": 1, "technical": 2}
            top_up.sort(key=lambda q: _type_order.get(
                q.get("type", "technical").lower(), 2))
            self.questions.extend(top_up)
            if self.current_index < len(self.questions):
                q = self.questions[self.current_index]
                self.current_index += 1
                return q.get("question", "Tell me about yourself.")
            return "Tell me about yourself."

        # ── RL MODE: on-demand generation per RL hint ─────────────────────────
        rl_hint     = self._rl_next_hint or {}
        rl_diff     = rl_hint.get("difficulty", "medium") or "medium"
        rl_type     = rl_hint.get("type", "") or ""
        is_followup = rl_hint.get("follow_up", False)

        # "all" difficulty in hint (shouldn't happen, but guard it)
        if not rl_diff or rl_diff == "all":
            rl_diff = "medium"

        # Follow-up: RL decided to probe the same topic — don't advance
        if is_followup:
            # v9.2: mark that a follow-up is pending so app.py does NOT
            # increment q_number or count this toward the original quota.
            self._follow_up_pending = True
            # Return the current question again; follow_up_engine in app.py
            # generates the actual probe question using the original answer
            if self.questions and self.current_index > 0:
                return self.questions[self.current_index - 1].get(
                    "question", "Can you elaborate on your previous answer?")
            return "Can you elaborate on your previous answer?"

        # Original question — reset follow_up flag and increment original count
        self._follow_up_pending = False

        # On-demand single question from Groq
        if self._api_key_available():
            q_dict = self.qbank.get_single_question(
                role       = self.role,
                difficulty = rl_diff,
                q_type     = rl_type,
            )
            if q_dict:
                self.questions.insert(self.current_index, q_dict)
                question_text = q_dict.get("question", "Tell me about yourself.")
                self.current_index += 1
                self._original_q_count += 1
                log.info(f"[RL] Q{self._original_q_count} generated on-demand: "
                         f"{q_dict.get('type','')} / {rl_diff}")
                return question_text
            log.warning("[RL] On-demand generation failed — using buffer fallback")

        # Fallback: pre-loaded buffer
        if self.current_index < len(self.questions):
            q = self.questions[self.current_index]
            self.current_index += 1
            self._original_q_count += 1
            return q.get("question", "Tell me about yourself.")

        # Last resort: new batch
        self.questions.extend(self.qbank.get_questions(self.role, rl_diff, 5))
        if self.current_index < len(self.questions):
            q = self.questions[self.current_index]
            self.current_index += 1
            self._original_q_count += 1
            return q.get("question", "Tell me about yourself.")
        return "Tell me about yourself."

    def _api_key_available(self) -> bool:
        """Check if Groq API key is configured for on-demand generation."""
        return bool(getattr(self.qbank, "_api_key", None))

    def get_current_question_dict(self) -> Optional[dict]:
        idx = max(0, self.current_index - 1)
        if idx < len(self.questions):
            return self.questions[idx]
        return None

    def interview_finished(self) -> bool:
        if self._rl_active:
            # RL mode: finished when original question quota is reached
            return self._original_q_count >= self._num_questions
        return self.current_index >= len(self.questions)

    def get_total_questions(self) -> int:
        if self._rl_active:
            return self._num_questions   # always the chosen count, not buffer size
        return len(self.questions)

    def is_follow_up_pending(self) -> bool:
        """v9.2: True when RL fired follow-up action — next click is a sub-part."""
        return self._follow_up_pending

    # ── Answer evaluation ─────────────────────────────────────────────────────

    def evaluate_answer(self, question_text: str, answer_text: str) -> dict:
        """
        Evaluate an answer with NLP scoring + multimodal snapshot.
        v7.3: uses AnswerEvaluator.score_answer() so TF-IDF is computed
        against ideal_answer from question_pool.json, not just keywords.
        Falls back to NLPScorer.score() if answer_evaluator unavailable.
        """
        q_dict = next(
            (q for q in self.questions if q.get("question") == question_text),
            {"question": question_text, "keywords": [], "star_expected": []},
        )

        # v7.3 fix: route to the correct scorer
        if self._using_answer_evaluator:
            # AnswerEvaluator.score_answer() reads ideal_answer + keywords from q_dict
            nlp = self.nlp_scorer.score_answer(answer_text, q_dict)
            # Normalise key names to the shape the rest of the engine expects
            nlp.setdefault("star_scores", nlp.get("star_scores", {}))
            nlp.setdefault("disc_traits", nlp.get("disc_traits", {}))
            nlp.setdefault("keyword_hits", nlp.get("keyword_hits", []))
            nlp.setdefault("feedback", nlp.get("feedback", ""))
            # AnswerEvaluator returns raw score on 1-5 scale under "score"
            score = nlp.get("score", 1.0)
        else:
            nlp   = self.nlp_scorer.score(answer_text, q_dict)
            score = nlp["score"]

        self.disc_history.append(nlp.get("disc_traits", {}))
        self.scores.append(score)
        self.performance.add_score(score)

        # Snapshot multimodal signals at submission time
        em_summary = self.get_emotion_summary()
        vs_summary = self.get_voice_session_summary()
        mm_conf    = self.get_multimodal_confidence()

        self._answer_emotion_snapshots.append(em_summary)
        self._answer_voice_snapshots.append(vs_summary)

        # v10.0: nervousness = 100% voice — facial signal excluded
        # v8.1: prefer delta (relative to baseline) when calibration was done
        voice_n  = vs_summary.get("nervousness_delta",
                   vs_summary.get("nervousness", 0.2))
        fused_n  = self.nervousness_fusion.record(0.0, voice_n)

        # ── v8.0: RL Adaptive Sequencer update ────────────────────────────────
        # After scoring, let the agent observe this transition and select the
        # next (type, difficulty) action. The recommendation is stored in
        # self._rl_next_hint for get_next_question() to use.
        #
        # Research: Patel et al. (2023, Springer AI Review) — RL sequencing
        # signal: score + nervousness + STAR + time > pure score heuristic.
        rl_action_label = "static"
        if self._rl_active and self.sequencer is not None:
            # ── RL MODE: update Q-table and get next action ───────────────────
            star_count_int = int(sum(
                1 for v in nlp.get("star_scores", {}).values() if v
            ))
            time_eff   = nlp.get("time_efficiency", 50.0)
            word_count = nlp.get("word_count", 100)
            q_type_str = q_dict.get("type", "")

            next_action = self.sequencer.record_and_select(
                score           = score,
                nervousness     = fused_n,
                star_count      = star_count_int,
                time_efficiency = time_eff,
                word_count      = word_count,
                q_type          = q_type_str,
            )
            self._rl_next_hint = {
                "type":       next_action.q_type,
                "difficulty": next_action.difficulty,
                "follow_up":  next_action.follow_up,
            }
            rl_action_label = next_action.label()
            if next_action.follow_up:
                log.info("[RL] Follow-up action selected — shallow answer detected")
            log.info(f"[RL] Next: {rl_action_label} "
                     f"(ε={self.sequencer.current_epsilon:.3f})")
        else:
            # ── FIXED MODE: no RL update — hint stays empty ───────────────────
            # Follow-up in fixed mode is handled by follow_up_engine in app.py
            # based on content analysis (low score, missing STAR, low relevance).
            # Static difficulty heuristic kept as fallback if sequencer missing.
            self._rl_next_hint = {}
            if self.sequencer is None:
                nd = self.qbank.next_difficulty(self.difficulty, score)
                if nd != self.difficulty:
                    self.difficulty = nd

        result = {
            "question":              question_text,
            "answer":                answer_text,
            "score":                 score,
            "feedback":              nlp["feedback"],
            "star":                  nlp["star_scores"],
            "keywords":              nlp["keyword_hits"],
            "emotion_dominant":      em_summary.get("dominant", "Neutral"),
            "nervousness":           em_summary.get("nervousness", 0.2),
            "voice_dominant":        vs_summary.get("dominant", "Neutral"),
            "voice_nervousness":     vs_summary.get("nervousness", 0.2),
            "fused_nervousness":     fused_n,
            # v8.0: depth_score surfaced explicitly (was buried in depth_fluency)
            "depth_score":           nlp.get("depth_score", round(score, 2)),
            "trend":                 self.performance.get_trend(),
            # v8.0 RL: record what the sequencer recommended for the NEXT question
            "rl_next_action":        rl_action_label,
            "rl_epsilon":            round(self.sequencer.current_epsilon, 4)
                                     if self.sequencer else None,
        }
        self.answers.append(result)
        # v9.2: in fixed mode, count original answers here (RL mode counts in get_next_question)
        if not self._rl_active:
            self._original_q_count += 1

        return result

    # ── Final report (enhanced v7.0) ──────────────────────────────────────────

    def final_report(self) -> dict:
        """
        Generate comprehensive session report with:
          • All original fields (backwards-compatible)
          • Nervousness trajectory (per-answer trend)
          • v10.0: Nervousness = 100% voice (facial excluded)
          • Voice Quality Index
          • Voice model provenance (CREMA-D + TESS, metrics)
        """
        if not self.scores:
            return {}

        ks = round(sum(self.scores) / len(self.scores), 2)

        em_summary = self.get_emotion_summary()
        em_fb      = self.emotion_feedback()
        vs_summary = self.get_voice_session_summary()
        mm_conf    = self.get_multimodal_confidence()
        vqi        = self.voice_quality.compute()
        nerv_summ  = self.nervousness_fusion.get_summary()
        prosody    = self.get_prosodic_analysis()

        # Emotion score (based on fused nervousness)
        fused_nerv = nerv_summ["session_nervousness"]
        es  = em_fb.get("emotion_score",
                         round(min(5.0, max(1.0, (1 - fused_nerv) * 5)), 2))

        # Voice score: blend VQI + low-nervousness inverse
        vs_nerv = vs_summary.get("nervousness", 0.2)
        vqi_sc  = vqi.get("vqi_score", 3.5)
        vs      = round(min(5.0, max(1.0,
                    vqi_sc * 0.60 + (1 - vs_nerv) * 5 * 0.40
                )), 2)

        # v8.0: compute avg depth score across all answers (depth_fluency sub-score)
        # depth_score is stored on each answer dict by AnswerEvaluator.evaluate()
        avg_depth = 0.0
        if self.answers:
            depth_vals = [a.get("depth_score", 0.0) for a in self.answers
                          if a.get("depth_score", 0.0) > 0]
            avg_depth = round(sum(depth_vals) / len(depth_vals), 2) if depth_vals else round(ks, 2)

        final_sc   = ScoreAggregator.combine(
            emotion=es, voice=vs, knowledge=ks, avg_depth=avg_depth
        )

        # Voice model metadata (for report provenance)
        voice_metrics = {}
        if self.voice_pipeline:
            voice_metrics = self.voice_pipeline.get_metrics()

        return {
            # ── Core (backwards-compat) ───────────────────────────────────
            "candidate":         "Candidate",
            "role":              self.role,
            "difficulty":        self.difficulty,
            "answers":           self.answers,
            "scores":            self.scores,
            "knowledge":         {"score": ks},
            "voice":             {
                "score":   vs,
                "summary": vs_summary,
                "prosody": prosody,
                "vqi":     vqi,                   # new v7.0
            },
            "emotion":           {"score": es, "summary": em_summary},
            # v8.0: posture + confidence_score removed from report output
            # depth is now surfaced via final_scores["depth"]
            "multimodal":        mm_conf,
            "emotion_feedback":  em_fb,
            "final_scores":      final_sc,
            "performance_trend": self.performance.get_trend(),
            "generated":         datetime.now().isoformat(),

            # ── New v7.0 nervousness fields ───────────────────────────────
            "nervousness": {
                "fused_session":       fused_nerv,
                "level":               nerv_summ["nervousness_level"],
                "trend":               nerv_summ["nervousness_trend"],
                "per_answer":          nerv_summ["per_answer_nervousness"],
                "facial_mean":         nerv_summ["facial_mean"],
                "voice_mean":          nerv_summ["voice_mean"],
                "fusion_weights":      nerv_summ["fusion_weights"],
            },

            # ── New v7.0 voice model provenance ──────────────────────────
            "voice_model": {
                "source":                      voice_metrics.get("source", "CREMA-D + TESS"),
                "datasets":                    voice_metrics.get("datasets", ["CREMA-D", "TESS"]),
                "feature_size":                voice_metrics.get("feature_size", 108),
                "test_accuracy":               voice_metrics.get("test_accuracy"),
                "cv_mean_accuracy":            voice_metrics.get("cv_mean_accuracy"),
                "nervousness_binary_accuracy": voice_metrics.get("nervousness_binary_accuracy"),
                "nervousness_binary_f1":       voice_metrics.get("nervousness_binary_f1"),
                "n_total":                     voice_metrics.get("n_total"),
                "dataset_rationale":           voice_metrics.get("dataset_rationale", ""),
            },

            # ── v8.0: RL sequencer session report ────────────────────────────
            # Surfaces what the agent did: action distribution, Q-table state,
            # best/worst rewarded steps, preferred next action.
            "rl_sequencer": self.get_rl_report(),
        }

    # ── RL sequencer public API ───────────────────────────────────────────────

    def get_rl_report(self) -> Dict:
        """
        Return the RL sequencer session report for the Final Report page.
        Includes action distribution, Q-value heatmap data, step history.

        If the RL sequencer was not used (module not installed), returns a
        dict explaining the fallback and what to install.
        """
        if self.sequencer is None:
            return {
                "available":  False,
                "message":    "RL sequencer not available. "
                              "Install adaptive_sequencer.py in the same directory.",
                "fallback":   "Static difficulty heuristic (QuestionBank.next_difficulty)",
            }
        report = self.sequencer.get_session_report()
        report["available"] = True
        report["heatmap"]   = self.sequencer.get_q_table_heatmap_data()

        # Save Q-table so next session warm-starts from this
        self.sequencer.save()
        return report

    def get_rl_next_hint(self) -> Dict[str, str]:
        """
        Return the RL sequencer's current recommendation for the next question.
        Used by page_live() to show a subtle "Recommended: Behavioural/Medium"
        indicator in the UI.
        """
        return dict(self._rl_next_hint)

    # ── FER stats ─────────────────────────────────────────────────────────────

    def get_fer_statistics(self) -> Dict[str, int]:
        if self.fer_pipeline:
            return self.fer_pipeline.get_statistics()
        return {
            "Happy": 8989, "Neutral": 6198, "Sad": 6077,
            "Fear": 5121, "Angry": 4953, "Surprise": 4002, "Disgust": 547,
        }

    def get_pipeline_metrics(self) -> Dict:
        metrics: Dict = {}
        if self.fer_pipeline:
            metrics["fer"] = self.fer_pipeline.get_metrics()
        if self.voice_pipeline:
            metrics["voice"] = self.voice_pipeline.get_metrics()
        return metrics