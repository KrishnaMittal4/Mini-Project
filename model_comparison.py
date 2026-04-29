"""
model_comparison.py — Aura AI | Model Comparison & Benchmark Engine (v3.0)
===========================================================================
v3.1 changes (JSON-first, Kaggle-free, robust multi-strategy loader):
  • load_dataset() completely rewritten — 3-strategy cascade:
      Strategy 1: local CSV file (same dir or CWD) — instant, no network
      Strategy 2: kagglehub with 20-second SIGALRM timeout — skipped on
                  Windows or when SIGALRM unavailable
      Strategy 3: expanded embedded dataset (100 entries) — always works
  • Proxy/network error detection — 403, ProxyError, ConnectionError caught
    explicitly with human-readable messages instead of cryptic tracebacks.
  • save_dataset_template() — new helper: writes the embedded dataset as a
    properly-formatted CSV so users can extend it locally.
  • _EMBEDDED expanded from 40 → 100 entries — 60 new high-quality QA pairs
    spanning all 8 categories (Behavioural, Leadership, Communication,
    Problem Solving, Adaptability, Collaboration, Self-Awareness, Career Goals).

v2.0 changes:
  • groq_ground_truth() — Groq LLM rates submitted answer on 0-100 scale.
  • run_benchmark() — 40-question cap removed; max_entries defaults to None.
  • load_dataset() — errors surfaced as returned message string.
  • _make_tiers() — true_quality_pct replaced by Groq-rated GT when key set.

Benchmarks Aura AI's answer evaluation pipeline against 4 standard
baselines on the HR Interview Questions & Ideal Answers dataset
(Kaggle: aryan208/hr-interview-questions-and-ideal-answers).

BASELINES COMPARED
------------------
  B1  Keyword Match Only  -- word-overlap count, no semantics
  B2  TF-IDF Cosine       -- classic IR baseline (sklearn)
  B3  BM25                -- Okapi BM25, pure Python
  B4  SBERT               -- sentence embeddings (all-MiniLM-L6-v2)
  A   Aura AI             -- Groq LLM relevance + STAR + DISC + depth

METRICS (full dataset, all scorers)
-------------------------------------
  Pearson r       -- correlation with Groq-rated dynamic ground truth
  MAE             -- mean absolute error vs ground truth (pct pts)
  Consistency     -- std dev (lower = more repeatable)
  Coverage        -- % of answers that receive a meaningful score
  Category bias   -- per-category average (reveals systematic gaps)

LOCAL JSON USAGE (recommended when Kaggle network is blocked)
------------------------------------------------------------
  Place the Kaggle JSON next to this file as:
      hr_interview_dataset.json
  load_dataset() finds it automatically (Strategy 1, no network).
  Supported formats: top-level array, wrapped object, JSONL.
  To generate a starter JSON from the embedded data:
      from model_comparison import save_dataset_template
      save_dataset_template()     # writes hr_interview_dataset.json
  CSV is also accepted as a fallback (hr_interview_dataset.csv).
"""

from __future__ import annotations

import glob
import math
import os
import platform
import re
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _sk_cos
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

try:
    import kagglehub
    KAGGLE_OK = True
except ImportError:
    KAGGLE_OK = False


# ---------------------------------------------------------------------------
# EMBEDDED DATASET  (40 pairs, 8 categories, representative of Kaggle dataset)
# ---------------------------------------------------------------------------
_EMBEDDED: List[Dict] = [
    {"question":"Tell me about a time you had to work under pressure.",
     "ideal":"In my previous role as project lead our client moved the deadline forward by two weeks. I immediately reprioritised the backlog held daily stand-ups and delegated non-critical tasks. We delivered on time with zero defects and the client extended the contract.",
     "category":"Behavioural","difficulty":"Medium","gt_pct":92,
     "keywords":["deadline","prioritise","team","deliver","result"]},

    {"question":"Describe a situation where you resolved a conflict within your team.",
     "ideal":"Two engineers disagreed on the architecture for a microservice. I facilitated a structured technical review where each presented their approach with data. We agreed on a hybrid solution that reduced latency by 30 percent. The conflict became a productive design discussion.",
     "category":"Behavioural","difficulty":"Medium","gt_pct":94,
     "keywords":["conflict","resolve","team","outcome","approach"]},

    {"question":"Give me an example of when you showed initiative.",
     "ideal":"I noticed our onboarding documentation was outdated and causing repeated support tickets. Without being asked I rewrote it added a searchable FAQ and set up a monthly review cycle. Support tickets from new hires dropped by 60 percent in the following quarter.",
     "category":"Behavioural","difficulty":"Easy","gt_pct":91,
     "keywords":["initiative","proactive","improvement","impact"]},

    {"question":"Tell me about a time you failed and what you learned.",
     "ideal":"I underestimated database migration complexity and gave an optimistic timeline. The migration took three weeks longer than planned. I learned to add a 30 percent buffer to infrastructure estimates build in a staging rehearsal and communicate risks upfront. I have not missed a migration deadline since.",
     "category":"Behavioural","difficulty":"Hard","gt_pct":95,
     "keywords":["failure","learn","improve","process","result"]},

    {"question":"Describe a time you influenced someone without formal authority.",
     "ideal":"I needed the security team to approve a new API design but had no authority. I prepared a detailed threat model and invited their lead to co-design the solution. By making them co-authors they championed it internally. Approval came in three days instead of six weeks.",
     "category":"Behavioural","difficulty":"Hard","gt_pct":96,
     "keywords":["influence","stakeholder","persuade","result","strategy"]},

    {"question":"Tell me about a time you exceeded expectations.",
     "ideal":"I was hired to maintain a reporting pipeline but I identified that the data model had a flaw causing silent data loss. I scoped and fixed the issue without being asked wrote comprehensive tests and documented the data contract. My manager said it was the highest-impact contribution any new hire had made in their first month.",
     "category":"Behavioural","difficulty":"Medium","gt_pct":93,
     "keywords":["exceed","initiative","impact","deliver","proactive"]},

    {"question":"How do you motivate a team that is losing morale?",
     "ideal":"I start by listening individually to understand the root cause whether it is workload unclear goals or lack of recognition. I address it directly by rescoping work reinstating weekly wins and ensuring everyone sees how their work connects to the company mission.",
     "category":"Leadership","difficulty":"Medium","gt_pct":90,
     "keywords":["motivate","listen","team","recognition","mission"]},

    {"question":"How do you handle an underperforming team member?",
     "ideal":"I schedule a private non-judgmental one-to-one to understand whether the issue is skills motivation or personal circumstances. I set clear measurable improvement goals with a 30-day check-in and provide targeted resources. If performance does not improve I escalate through HR with full documentation.",
     "category":"Leadership","difficulty":"Hard","gt_pct":93,
     "keywords":["performance","support","goal","process","fair"]},

    {"question":"Describe your leadership style.",
     "ideal":"I lead by outcome not method. I set clear goals and trust people to find their path. I adapt being more structured with juniors who need guidance and more autonomous with seniors who need space. I prioritise psychological safety because it is the strongest predictor of team performance.",
     "category":"Leadership","difficulty":"Easy","gt_pct":89,
     "keywords":["style","outcome","adapt","trust","safety"]},

    {"question":"How do you make decisions under uncertainty?",
     "ideal":"I use a reversibility test. Reversible decisions I make fast with the best available data. Irreversible ones I slow down to gather more signal. I write a one-page decision brief stating the problem options risks and recommendation. This forces clarity and creates a record for post-mortems.",
     "category":"Leadership","difficulty":"Hard","gt_pct":91,
     "keywords":["decision","uncertainty","data","risk","process"]},

    {"question":"Tell me about a time you led a team through change.",
     "ideal":"Our company was acquired and the new parent required us to migrate from AWS to Azure within 90 days. I created a detailed migration plan assigned clear ownership per service and ran weekly risk reviews. We completed the migration in 84 days with no production incidents.",
     "category":"Leadership","difficulty":"Hard","gt_pct":94,
     "keywords":["change","lead","plan","outcome","team"]},

    {"question":"How do you handle disagreement with your manager?",
     "ideal":"I share my view clearly and once with data behind it. Then I listen to understand their reasoning not to rebut it. If we still disagree and the decision is theirs to make I commit fully to the direction. I disagree and commit rather than disagree and undermine.",
     "category":"Leadership","difficulty":"Hard","gt_pct":93,
     "keywords":["disagree","manager","commit","honest","respect"]},

    {"question":"How do you explain a complex technical concept to a non-technical audience?",
     "ideal":"I anchor to something the audience already understands using an analogy from daily life. I avoid acronyms use one concrete example and check understanding by asking them to paraphrase back. When I explained database sharding to our CFO I compared it to splitting a filing cabinet by surname.",
     "category":"Communication","difficulty":"Medium","gt_pct":93,
     "keywords":["explain","analogy","audience","clear","example"]},

    {"question":"Tell me about a time you had to deliver difficult feedback.",
     "ideal":"I use the SBI model: Situation Behaviour Impact. I delivered feedback to a colleague whose code reviews were blocking the team. I was specific about the behaviour and its measurable impact on delivery time. I framed it as here is the pattern I observe rather than here is what you are doing wrong.",
     "category":"Communication","difficulty":"Hard","gt_pct":92,
     "keywords":["feedback","specific","behaviour","impact","constructive"]},

    {"question":"How do you ensure clear communication in a remote team?",
     "ideal":"Asynchronous-first: all decisions documented in writing never only discussed in a call. I use a decision log weekly written status updates and explicit RACI on every project. Synchronous time is reserved for creative work and relationship building. This reduces meeting load by 40 percent.",
     "category":"Communication","difficulty":"Medium","gt_pct":91,
     "keywords":["remote","async","document","clarity","process"]},

    {"question":"Describe a time when a miscommunication caused a problem and how you resolved it.",
     "ideal":"A client believed we had agreed to a feature that was not in scope because I used the word likely in an email. I called them immediately acknowledged the ambiguity was mine documented the actual scope in writing and added a weekly scope-confirmation touchpoint. The relationship strengthened as a result.",
     "category":"Communication","difficulty":"Medium","gt_pct":90,
     "keywords":["miscommunication","resolve","clarity","document","relationship"]},

    {"question":"Describe a time you had to say no to a stakeholder request.",
     "ideal":"A senior VP asked us to ship a feature in one week that our security review showed needed three. I said no to the timeline not the feature explained the specific risk in non-technical terms and offered a phased approach: a safe minimal version in one week and the full feature in three. They agreed.",
     "category":"Communication","difficulty":"Hard","gt_pct":93,
     "keywords":["no","stakeholder","risk","alternative","outcome"]},

    {"question":"Walk me through how you approach a problem you have never seen before.",
     "ideal":"I break it into understand decompose hypothesise test. First I make sure I am solving the right problem by restating it and confirming. Then I decompose into sub-problems rank by impact and tackle the highest-leverage piece first with a small experiment. I timebox exploration to avoid analysis paralysis.",
     "category":"Problem Solving","difficulty":"Medium","gt_pct":93,
     "keywords":["approach","decompose","hypothesis","test","systematic"]},

    {"question":"Describe a creative solution you developed to solve a business problem.",
     "ideal":"Our support queue was growing faster than we could hire. I proposed an LLM-powered triage tool that categorised and auto-drafted responses for the top 20 ticket types. We built it in six weeks. Ticket resolution time dropped by 45 percent and we avoided two hires costing 200 thousand dollars.",
     "category":"Problem Solving","difficulty":"Hard","gt_pct":95,
     "keywords":["creative","solution","impact","cost","result"]},

    {"question":"How do you prioritise when everything feels urgent?",
     "ideal":"I use a 2x2 impact-effort matrix and ask one clarifying question: what is the cost of not doing this today? That separates genuine urgency from noise. I then communicate the priority order explicitly to stakeholders so expectations are reset before I start.",
     "category":"Problem Solving","difficulty":"Easy","gt_pct":88,
     "keywords":["prioritise","impact","urgency","stakeholder","matrix"]},

    {"question":"Tell me about the most complex problem you have solved at work.",
     "ideal":"We had a memory leak in a distributed system that only manifested under production load. I instrumented every service with custom memory snapshots at five-minute intervals correlated them with traffic spikes and identified a connection pool that was not being released after timeouts. The fix was four lines of code after two weeks of investigation.",
     "category":"Problem Solving","difficulty":"Hard","gt_pct":96,
     "keywords":["complex","diagnose","systematic","root cause","resolve"]},

    {"question":"How do you deal with ambiguity at work?",
     "ideal":"I treat ambiguity as a question to be clarified rather than a problem to be suffered. I write down my current understanding highlight the specific unknowns and bring that document to whoever can resolve them. This converts a fuzzy situation into a precise set of decisions.",
     "category":"Problem Solving","difficulty":"Medium","gt_pct":91,
     "keywords":["ambiguity","clarify","structure","progress","communicate"]},

    {"question":"Tell me about a time you had to adapt quickly to a major change.",
     "ideal":"Our company was acquired mid-project. The new parent required us to switch technology stacks within 60 days. I ran an emergency skills audit paired engineers with complementary expertise and negotiated a phased delivery plan. We met the deadline and the new stack improved our deployment frequency by three times.",
     "category":"Adaptability","difficulty":"Hard","gt_pct":93,
     "keywords":["adapt","change","plan","outcome","fast"]},

    {"question":"How do you handle working on multiple projects simultaneously?",
     "ideal":"I use time-blocking with clear context-switching rules: deep work in the morning meetings and reviews in the afternoon. I maintain a single priority list reviewed every Monday with my manager so I am never surprised by conflicting deadlines. I also protect 10 percent of my week as unscheduled buffer.",
     "category":"Adaptability","difficulty":"Medium","gt_pct":89,
     "keywords":["multitask","prioritise","organise","time","manage"]},

    {"question":"Describe a time you had to learn a new skill quickly.",
     "ideal":"I was asked to lead a machine learning project with no prior ML experience. I structured my learning in sprints: first week theory second week toy projects third week applying it to real data. I was productive on the team within three weeks. I documented the learning path and it became the onboarding guide for the next two ML hires.",
     "category":"Adaptability","difficulty":"Medium","gt_pct":92,
     "keywords":["learn","fast","skill","apply","outcome"]},

    {"question":"What is your approach to continuous improvement?",
     "ideal":"I run blameless retrospectives after every major project with three questions: what worked what did not what will we change. I track the what-will-we-change items as action items with owners. I also keep a personal learning log where I note one thing I would do differently each week.",
     "category":"Adaptability","difficulty":"Medium","gt_pct":92,
     "keywords":["improve","retrospective","system","learn","action"]},

    {"question":"How do you work with people whose working style is very different from yours?",
     "ideal":"I start with curiosity rather than judgement. I ask them how they prefer to work and what they find frustrating. I find the shared goal and let each person contribute in their strongest mode. I had a colleague who hated synchronous meetings; I switched to async document review and our collaboration quality improved.",
     "category":"Collaboration","difficulty":"Medium","gt_pct":91,
     "keywords":["collaborate","style","adapt","respect","outcome"]},

    {"question":"Tell me about a successful cross-functional project you contributed to.",
     "ideal":"I led the product analytics integration between engineering data science and marketing. I created a shared language glossary so each team used identical metric definitions ran weekly cross-team syncs with rotating facilitators and maintained a public project board. We shipped on time and the data reduced customer acquisition cost by 18 percent.",
     "category":"Collaboration","difficulty":"Hard","gt_pct":94,
     "keywords":["cross-functional","team","alignment","deliver","impact"]},

    {"question":"How do you build trust with a new team?",
     "ideal":"I do three things in the first 30 days: listen more than I speak deliver on every commitment no matter how small and learn what each person cares about beyond the work. Trust is built through consistent reliability not charisma. I also share my own working-style preferences early so there are no surprises.",
     "category":"Collaboration","difficulty":"Easy","gt_pct":90,
     "keywords":["trust","listen","deliver","reliable","relationship"]},

    {"question":"How do you give credit to others?",
     "ideal":"I make it a practice to name people publicly when referencing their ideas or contributions in all-hands meetings in written retrospectives and in messages to the team. I also advocate for people to present their own work rather than presenting it on their behalf. Recognition given publicly and specifically is far more powerful than a private thank you.",
     "category":"Collaboration","difficulty":"Easy","gt_pct":87,
     "keywords":["credit","recognise","team","public","specific"]},

    {"question":"What is your greatest professional weakness?",
     "ideal":"I tend to over-engineer solutions. I want them to be perfect before I share them. I have been actively working on this by setting artificial deadlines to share work-in-progress and requesting early feedback. My pull requests are now smaller and more frequent which has improved review quality and my shipping cadence.",
     "category":"Self-Awareness","difficulty":"Medium","gt_pct":92,
     "keywords":["weakness","improve","action","result","honest"]},

    {"question":"How do you handle criticism of your work?",
     "ideal":"I separate the feedback from the delivery. My goal is to extract the signal regardless of how it was given. I clarify what specifically the person observed rather than accepting vague criticism. I then decide whether it is actionable and if so I act on it visibly so the person knows I took it seriously.",
     "category":"Self-Awareness","difficulty":"Medium","gt_pct":91,
     "keywords":["criticism","feedback","respond","action","growth"]},

    {"question":"What motivates you professionally?",
     "ideal":"I am most energised by hard problems that require both technical depth and human coordination. I am motivated by seeing my work used by real users and by helping the people around me grow. I lose energy in environments where quality is deprioritised for short-term speed.",
     "category":"Self-Awareness","difficulty":"Easy","gt_pct":88,
     "keywords":["motivate","passion","drive","value","energy"]},

    {"question":"How do you manage your own professional development?",
     "ideal":"I run a quarterly personal review: what skills did I improve what did I ship what do I want to learn next. I allocate 20 percent of my working week to deliberate practice reading papers contributing to open source or taking a course. I make these commitments public to my manager so there is accountability.",
     "category":"Self-Awareness","difficulty":"Medium","gt_pct":90,
     "keywords":["development","learn","plan","deliberate","growth"]},

    {"question":"What does good work look like to you?",
     "ideal":"Good work solves the right problem not just the stated problem. It is simple enough that someone else can understand and maintain it. It ships because perfect work that never ships is not good work. And it makes the people who use it feel that the maker cared about their experience.",
     "category":"Self-Awareness","difficulty":"Easy","gt_pct":88,
     "keywords":["quality","value","simple","ship","care"]},

    {"question":"Where do you see yourself in five years?",
     "ideal":"I want to be a technical leader who has successfully shipped a product used by a significant number of people and has grown at least two engineers into senior roles. I am less attached to a specific title than to the scope of impact and the quality of the people I am working with.",
     "category":"Career Goals","difficulty":"Easy","gt_pct":87,
     "keywords":["goal","growth","leadership","impact","vision"]},

    {"question":"Why do you want to work at this company?",
     "ideal":"I have spent time studying your product your engineering blog and your recent strategic moves. Your approach to developer experience aligns directly with how I think great platforms are built. I also want to work in a team that treats reliability as a first-class feature not an afterthought.",
     "category":"Career Goals","difficulty":"Medium","gt_pct":90,
     "keywords":["company","research","align","culture","reason"]},

    {"question":"What are you looking for in your next role?",
     "ideal":"I am looking for three things: technical depth meaning problems that require genuine engineering thought not just configuration; ownership meaning the ability to influence architecture decisions not just implement them; and growth meaning colleagues who are better than me in areas I want to develop.",
     "category":"Career Goals","difficulty":"Easy","gt_pct":88,
     "keywords":["role","seek","growth","ownership","fit"]},

    {"question":"Why are you leaving your current role?",
     "ideal":"I have grown a great deal in my current role and I am proud of what the team has built. I am looking for a larger scope of impact and technical challenge that my current organisation is not positioned to offer at this stage of their growth. This is a growth move not a push away.",
     "category":"Career Goals","difficulty":"Medium","gt_pct":89,
     "keywords":["leaving","growth","opportunity","positive","reason"]},

    {"question":"Tell me about a long-term goal you are actively working towards.",
     "ideal":"I am working towards being able to design and lead the engineering strategy for a product from zero to one million users. To get there I am deliberately seeking projects with scaling challenges studying distributed systems architecture and mentoring junior engineers. I review my progress quarterly.",
     "category":"Career Goals","difficulty":"Medium","gt_pct":91,
     "keywords":["goal","plan","action","progress","strategy"]},

    # ── 60 NEW ENTRIES (v3.0) ────────────────────────────────────────────────

    # BEHAVIOURAL (10 new)
    {"question":"Tell me about a time you had to meet a very tight deadline.",
     "ideal":"Our regulatory filing was moved forward by three weeks with no warning. I mapped every dependency created a 48-hour sprint cycle secured two additional engineers from another team and held daily reviews at 8 am. We submitted two days early. The regulator noted it as the most complete filing they had received that quarter.",
     "category":"Behavioural","difficulty":"Hard","gt_pct":94,
     "keywords":["deadline","plan","team","deliver","regulatory"]},

    {"question":"Describe a time you had to manage competing priorities from different managers.",
     "ideal":"Two VPs each believed my time was allocated to them. I called a joint meeting presented a written breakdown of all active tasks and asked them to co-prioritise. They agreed on a split. I documented the decision and sent a weekly status to both. The ambiguity was eliminated and stayed eliminated.",
     "category":"Behavioural","difficulty":"Hard","gt_pct":93,
     "keywords":["priorities","manage","stakeholder","communicate","outcome"]},

    {"question":"Tell me about a time you went above and beyond your job description.",
     "ideal":"My role was backend engineer but I noticed our data team had no automated alerting on pipeline failures. I built a lightweight monitoring script over two weekends integrated it with our Slack channel and presented it at the next all-hands. It caught three critical failures in its first month before anyone noticed manually.",
     "category":"Behavioural","difficulty":"Medium","gt_pct":92,
     "keywords":["initiative","beyond","impact","proactive","result"]},

    {"question":"Tell me about a time you had to work with someone you found difficult.",
     "ideal":"A colleague had a habit of dismissing ideas in group settings before they were fully explained. Instead of avoiding the issue I asked for a one-to-one to share how the pattern was affecting team brainstorming. They were unaware of the impact and adjusted. Our collaboration became one of the most productive on the team.",
     "category":"Behavioural","difficulty":"Hard","gt_pct":93,
     "keywords":["difficult","relationship","honest","outcome","team"]},

    {"question":"Describe a time you had to make a decision with incomplete information.",
     "ideal":"We had four hours to decide whether to roll back a production deployment with partial user impact data. I listed what we knew what we did not know and the cost of each option. Given that 8 percent of users were affected and growing I recommended rollback. It was the right call — the root cause took two more days to diagnose.",
     "category":"Behavioural","difficulty":"Hard","gt_pct":95,
     "keywords":["decision","incomplete","risk","fast","outcome"]},

    {"question":"Tell me about a time your attention to detail prevented a serious problem.",
     "ideal":"During a pre-launch review I noticed that a permissions field defaulted to world-readable in our cloud storage configuration. The rest of the team had missed it. I flagged it raised a P0 fix request and blocked the launch for four hours. The misconfiguration would have exposed 200 thousand user records.",
     "category":"Behavioural","difficulty":"Medium","gt_pct":94,
     "keywords":["detail","prevent","risk","catch","impact"]},

    {"question":"Describe a time you successfully managed stakeholder expectations.",
     "ideal":"A client expected a feature in three months that our engineers estimated at six. I did not just relay the bad news. I prepared three delivery options with trade-off tables presented them in person and recommended a phased approach delivering core value in three months and the full feature in five. The client chose that option and rated the project a 9 out of 10.",
     "category":"Behavioural","difficulty":"Medium","gt_pct":93,
     "keywords":["stakeholder","expectation","option","communicate","outcome"]},

    {"question":"Tell me about a time you had to deliver a project with limited resources.",
     "ideal":"We lost two engineers mid-project to a company-wide priority shift. I immediately re-scoped the deliverable to the minimum viable set of features negotiated a two-week extension and automated two manual testing processes to recover capacity. We shipped a product that covered 80 percent of the original spec and users could not tell the difference.",
     "category":"Behavioural","difficulty":"Hard","gt_pct":93,
     "keywords":["constraint","scope","resource","deliver","outcome"]},

    {"question":"Describe a time you had to persuade a sceptical audience.",
     "ideal":"I needed to convince our CFO to invest in developer tooling that had no immediate revenue impact. I built a financial model showing that slower deployments cost an estimated 400 thousand dollars per year in engineer time and incident recovery. I also brought two peer company case studies. The CFO approved the budget in the same meeting.",
     "category":"Behavioural","difficulty":"Hard","gt_pct":95,
     "keywords":["persuade","data","audience","financial","outcome"]},

    {"question":"Tell me about a time you supported a colleague who was struggling.",
     "ideal":"A junior developer was repeatedly failing code reviews and becoming visibly disengaged. I asked to pair-program with them for two afternoons. Within those sessions I identified that they had a gap in understanding async patterns not a motivation issue. I suggested a targeted course and offered daily 15-minute check-ins for a month. Their review pass rate went from 40 to 85 percent.",
     "category":"Behavioural","difficulty":"Medium","gt_pct":92,
     "keywords":["support","colleague","diagnose","help","outcome"]},

    # LEADERSHIP (10 new)
    {"question":"How do you set goals for your team?",
     "ideal":"I use OKRs: one objective that is aspirational and three to five measurable key results. I co-create them with the team rather than dictating them because ownership increases follow-through. I review progress bi-weekly and adjust key results when the environment changes while holding the objective constant.",
     "category":"Leadership","difficulty":"Medium","gt_pct":91,
     "keywords":["goal","OKR","team","measure","review"]},

    {"question":"Describe a time you turned around a failing team.",
     "ideal":"I inherited a team with 60 percent on-time delivery and high attrition. In the first month I ran individual listening sessions identified three root causes: unclear ownership unclear priorities and no psychological safety. I redesigned the RACI introduced weekly wins and created a blameless incident culture. Delivery improved to 88 percent within a quarter and attrition dropped to zero.",
     "category":"Leadership","difficulty":"Hard","gt_pct":95,
     "keywords":["turnaround","diagnose","culture","result","process"]},

    {"question":"How do you balance short-term delivery with long-term team health?",
     "ideal":"I use a sustainability tax: every sprint I reserve 15 percent of capacity for tech debt documentation and team learning. When business pressure spikes I negotiate to reduce that to 10 percent rather than zero. Teams that eliminate slack entirely accumulate hidden debt that shows up as incidents six months later.",
     "category":"Leadership","difficulty":"Hard","gt_pct":92,
     "keywords":["balance","sustainability","debt","delivery","long-term"]},

    {"question":"Tell me about a time you developed someone on your team.",
     "ideal":"A mid-level engineer wanted to move into architecture but had no system design experience. I gave them three progressively complex design problems as stretch assignments offered pre-review sessions before each stakeholder presentation and paired them with our principal engineer for two months. They passed the architecture interview internally within six months.",
     "category":"Leadership","difficulty":"Medium","gt_pct":93,
     "keywords":["develop","mentor","stretch","outcome","growth"]},

    {"question":"How do you handle a high performer who is a poor team player?",
     "ideal":"I separate the two conversations: I acknowledge the technical performance explicitly and then address the collaboration behaviour with specific examples and measurable impact. I set a clear expectation that both dimensions are required for continued growth. I have found that high performers often respond well when they understand that team health directly affects their own delivery.",
     "category":"Leadership","difficulty":"Hard","gt_pct":94,
     "keywords":["performance","behaviour","specific","expect","balance"]},

    {"question":"How do you create psychological safety on a team?",
     "ideal":"I model it first by sharing my own mistakes in public. I use blameless post-mortems where we ask what failed not who failed. I actively invite dissenting opinions in meetings by asking who sees this differently. Research shows that psychological safety is the strongest predictor of team performance and innovation.",
     "category":"Leadership","difficulty":"Medium","gt_pct":91,
     "keywords":["safety","trust","blameless","model","culture"]},

    {"question":"Describe a time you had to communicate a difficult decision to your team.",
     "ideal":"We had to cancel a product line that the team had spent eight months building due to a strategic pivot. I gathered the team immediately explained the business rationale in full acknowledged the emotional weight and gave everyone the rest of the day. The following day I focused all our energy on what would be preserved and what came next. No one left and the team respected the honesty.",
     "category":"Leadership","difficulty":"Hard","gt_pct":94,
     "keywords":["communicate","decision","honest","empathy","outcome"]},

    {"question":"How do you delegate effectively?",
     "ideal":"I match the task to the person's development goal not just their current capability. I hand off the outcome not the method. I set a clear deadline and one checkpoint so the person has autonomy but I catch issues early. I give credit publicly and provide feedback privately.",
     "category":"Leadership","difficulty":"Medium","gt_pct":90,
     "keywords":["delegate","outcome","autonomy","feedback","develop"]},

    {"question":"How do you manage remote or distributed teams?",
     "ideal":"I establish three things: shared rituals a weekly written status and explicit norms around response times. I make sure every team member has at least one meaningful one-to-one per week and that decisions are always documented before they are announced. I also fly out once a quarter because relationship-building is harder asynchronously.",
     "category":"Leadership","difficulty":"Medium","gt_pct":91,
     "keywords":["remote","async","ritual","communicate","relationship"]},

    {"question":"How do you prioritise the team's workload when demands exceed capacity?",
     "ideal":"I use a forced-ranking exercise with the team and the key stakeholders in the same room. Everything goes on the list and we rank by impact and strategic alignment. Then I draw a line at realistic capacity and everything below the line is deferred with an explicit date or dropped. No invisible commitments.",
     "category":"Leadership","difficulty":"Hard","gt_pct":92,
     "keywords":["prioritise","capacity","rank","stakeholder","transparent"]},

    # COMMUNICATION (10 new)
    {"question":"How do you communicate progress on a project to senior leadership?",
     "ideal":"I use a three-line executive summary: where we are versus plan what the top risk is and what decision if any I need from them. I send it weekly in writing before any verbal review. Leaders should never be surprised in a meeting by something I could have told them in advance.",
     "category":"Communication","difficulty":"Medium","gt_pct":91,
     "keywords":["executive","summary","risk","decision","proactive"]},

    {"question":"Tell me about a time your written communication made a significant difference.",
     "ideal":"Our engineering team was repeatedly missing context on product decisions. I introduced a lightweight Architecture Decision Record template: one page per decision covering context options and rationale. Within two months new engineers were onboarding 40 percent faster because the decision history was searchable and they could understand why not just what.",
     "category":"Communication","difficulty":"Medium","gt_pct":93,
     "keywords":["written","document","impact","onboard","process"]},

    {"question":"How do you tailor your communication style to different audiences?",
     "ideal":"I run a quick mental checklist: what does this person care about what is their technical level and what decision do I want them to make. With engineers I go deep and show my working. With product managers I anchor to user impact. With executives I lead with the business outcome and put the details in the appendix.",
     "category":"Communication","difficulty":"Medium","gt_pct":91,
     "keywords":["audience","tailor","style","outcome","adapt"]},

    {"question":"Describe a time you presented to a large group and what you learned.",
     "ideal":"I presented our annual engineering strategy to 200 people at a company all-hands. I over-prepared the slides and under-prepared for questions. A CFO asked a cost question I could not answer in detail. I learned to prepare a one-page FAQ for every major presentation covering the 10 most likely questions. I have never been caught off guard since.",
     "category":"Communication","difficulty":"Medium","gt_pct":92,
     "keywords":["present","prepare","learn","audience","improve"]},

    {"question":"How do you handle communication breakdowns in a project?",
     "ideal":"I treat a communication breakdown as a system failure not a people failure. I call the affected parties together immediately replay what each side understood and find the exact point of divergence. Then I add a structural fix such as a shared definition document or a confirmation step that prevents the same gap from recurring.",
     "category":"Communication","difficulty":"Hard","gt_pct":93,
     "keywords":["breakdown","fix","structural","clarity","process"]},

    {"question":"Tell me about a time you had to communicate under pressure.",
     "ideal":"We had a major production outage affecting enterprise clients at 2 am. I sent a status update every 20 minutes to the account team following a template: what we know what we are doing what we will do next. This meant the account team could give clients real answers rather than no answers. The clients cited the communication quality specifically in their post-incident feedback.",
     "category":"Communication","difficulty":"Hard","gt_pct":95,
     "keywords":["crisis","update","template","pressure","outcome"]},

    {"question":"How do you ensure important information reaches everyone who needs it?",
     "ideal":"I use a default of over-communication for the first month of any new project then calibrate down based on feedback. Every project has a single source of truth page linked from the team channel. I also run a weekly digest that summarises decisions made that week so people who missed meetings stay current without having to watch recordings.",
     "category":"Communication","difficulty":"Medium","gt_pct":90,
     "keywords":["inform","digest","source","channel","inclusive"]},

    {"question":"Describe how you handle situations where you do not have all the answers.",
     "ideal":"I say I do not know but here is how I will find out and by when. I never speculate as fact to avoid losing credibility. After I have the answer I follow up explicitly because people remember whether you came back or went silent. This honesty under uncertainty builds more trust than false confidence.",
     "category":"Communication","difficulty":"Easy","gt_pct":89,
     "keywords":["honest","unknown","follow-up","credibility","trust"]},

    {"question":"How do you use data to support your communication?",
     "ideal":"I visualise before I verbalise. If I can show a trend on a chart I do not describe it in words. I also cite the source and the sample size because stakeholders trust data more when they can interrogate it. I am careful to present the confidence interval not just the point estimate.",
     "category":"Communication","difficulty":"Medium","gt_pct":91,
     "keywords":["data","visualise","source","credible","evidence"]},

    {"question":"Tell me about a time you improved communication across teams.",
     "ideal":"Engineering and design were making decisions in silos and shipping features that conflicted with each other. I introduced a bi-weekly cross-team design review open to both functions where work-in-progress was shared before it was final. Revision cycles dropped by 35 percent in the next quarter and we shipped two features that neither team would have designed alone.",
     "category":"Communication","difficulty":"Hard","gt_pct":93,
     "keywords":["cross-team","silo","review","outcome","process"]},

    # PROBLEM SOLVING (10 new)
    {"question":"Tell me about a time you solved a problem others had given up on.",
     "ideal":"A data pipeline had been flapping for six months with no root cause found. I started from scratch refused to accept the existing hypotheses and built a timeline of every failure correlated against infra changes. I found that failures clustered within 30 minutes of a routine cloud snapshot job. A configuration change fixed it permanently in one day.",
     "category":"Problem Solving","difficulty":"Hard","gt_pct":95,
     "keywords":["root cause","systematic","persist","diagnose","result"]},

    {"question":"How do you validate that your solution actually solved the problem?",
     "ideal":"I define success criteria before I start building. After deployment I monitor the specific metric that was degraded for two weeks. If the metric is stable I run a controlled A/B comparison where possible. I also schedule a retrospective 30 days post-fix to check for recurrence or side effects.",
     "category":"Problem Solving","difficulty":"Medium","gt_pct":92,
     "keywords":["validate","metric","monitor","criteria","retrospective"]},

    {"question":"Describe a time you used data to solve a business problem.",
     "ideal":"Our churn rate was rising but no one knew why. I built a cohort analysis segmenting by signup channel plan type and tenure. Users acquired through paid ads who signed up for monthly plans churned at three times the rate of organic annual subscribers. We shifted ad spend and introduced annual incentives. Churn dropped 22 percent in two quarters.",
     "category":"Problem Solving","difficulty":"Hard","gt_pct":95,
     "keywords":["data","cohort","insight","action","result"]},

    {"question":"How do you approach root cause analysis?",
     "ideal":"I use the five-whys method starting from the observable symptom and asking why until I reach a systemic cause not a proximate one. I document the chain in writing so the team can challenge my logic. I also look for related failures that share the same root cause because systemic problems rarely manifest in only one place.",
     "category":"Problem Solving","difficulty":"Medium","gt_pct":92,
     "keywords":["root cause","five-whys","systemic","document","diagnose"]},

    {"question":"Tell me about a time you identified a problem before it became critical.",
     "ideal":"While reviewing query performance logs I noticed that a slow query was being called 40 percent more frequently each week. At that trajectory it would have breached our SLA in six weeks. I raised it as a low-severity issue added an index and rewrote the query. The performance improved by 20x with zero user impact.",
     "category":"Problem Solving","difficulty":"Medium","gt_pct":93,
     "keywords":["proactive","identify","prevent","performance","impact"]},

    {"question":"How do you know when to stop investigating and start solving?",
     "ideal":"I use a time-box rule. For most problems I allocate investigation time equal to no more than 20 percent of the estimated fix time. If I have not found the root cause I implement the best available mitigation and continue investigating in parallel. Perfect diagnosis is the enemy of recovery speed.",
     "category":"Problem Solving","difficulty":"Hard","gt_pct":91,
     "keywords":["timebox","mitigation","trade-off","speed","pragmatic"]},

    {"question":"Describe a time you had to solve a problem as a team.",
     "ideal":"Our API response times spiked across all endpoints simultaneously. I ran a war room with frontend backend and infrastructure engineers each running independent diagnostics in parallel with a five-minute sync loop. Infrastructure found elevated CPU on the load balancer within 12 minutes. We rolled back a firmware update and were green in under 30 minutes.",
     "category":"Problem Solving","difficulty":"Hard","gt_pct":94,
     "keywords":["team","parallel","diagnose","coordinate","outcome"]},

    {"question":"How do you handle a problem that is outside your area of expertise?",
     "ideal":"I identify who the domain expert is and involve them immediately rather than spending hours becoming one. I prepare a structured problem statement before that conversation so I do not waste their time with vague questions. I stay in the room to understand the solution so I can own it long-term and not create a dependency.",
     "category":"Problem Solving","difficulty":"Medium","gt_pct":91,
     "keywords":["expert","collaborate","structured","own","learn"]},

    {"question":"Tell me about a time you had to balance quality with speed in solving a problem.",
     "ideal":"We had a payment processing bug affecting 2 percent of transactions. I implemented a fast targeted fix within two hours that resolved the immediate issue and documented a technical debt card for the proper architectural fix scheduled for the next sprint. The hotfix held for three weeks while we did the proper work.",
     "category":"Problem Solving","difficulty":"Hard","gt_pct":93,
     "keywords":["quality","speed","hotfix","debt","balance"]},

    {"question":"How do you evaluate multiple solutions to a problem?",
     "ideal":"I score each option on three dimensions: effectiveness risk and reversibility. Effectiveness is the primary filter. For options that pass I prefer the one with lower risk and higher reversibility because it gives us the ability to course-correct. I write the evaluation down because the act of writing it often reveals gaps in my reasoning.",
     "category":"Problem Solving","difficulty":"Medium","gt_pct":91,
     "keywords":["evaluate","risk","reversible","criteria","process"]},

    # ADAPTABILITY (10 new)
    {"question":"Tell me about a time you had to change your approach mid-project.",
     "ideal":"We were building a batch processing system when user research midway through revealed that users needed near-real-time results. I halted the batch architecture scoped what could be reused and redesigned the processing layer as an event-driven pipeline. We delivered four weeks late but with a product users actually needed.",
     "category":"Adaptability","difficulty":"Hard","gt_pct":93,
     "keywords":["pivot","redesign","user","outcome","adapt"]},

    {"question":"How do you stay productive during periods of organisational uncertainty?",
     "ideal":"I focus on what is within my control: shipping quality work building relationships and documenting what I learn. I treat uncertainty as signal to invest in fundamentals not to slow down. I also reduce my planning horizon from quarterly to monthly so I can adjust faster without losing direction.",
     "category":"Adaptability","difficulty":"Medium","gt_pct":90,
     "keywords":["uncertainty","control","productive","focus","adapt"]},

    {"question":"Tell me about a time you had to work in a new or unfamiliar industry.",
     "ideal":"I joined a healthtech company with no medical domain knowledge. In the first month I shadowed three clinical workflows interviewed five nurses and read the relevant regulatory standards. By month two I was proposing UX improvements grounded in actual clinical practice that our medical advisors validated. Domain knowledge is acquirable if you are systematic about it.",
     "category":"Adaptability","difficulty":"Medium","gt_pct":92,
     "keywords":["domain","learn","fast","systematic","outcome"]},

    {"question":"How do you handle feedback that requires you to change your behaviour?",
     "ideal":"I listen fully without defending. I ask one clarifying question to make sure I understand the specific behaviour being flagged not just the general sentiment. Then I choose one concrete action to take within the next two weeks and tell the person what I plan to do. Behaviour change is visible when it is specific.",
     "category":"Adaptability","difficulty":"Medium","gt_pct":91,
     "keywords":["feedback","listen","action","specific","change"]},

    {"question":"Tell me about a time you had to adapt your style to work with a new culture.",
     "ideal":"I managed a team across London and Tokyo offices. UK team members preferred blunt direct feedback while Japanese colleagues found the same delivery demotivating. I adapted my written feedback to be more context-first for Tokyo and kept the direct style for London. Both teams rated me highly on the same 360 survey.",
     "category":"Adaptability","difficulty":"Hard","gt_pct":93,
     "keywords":["culture","adapt","style","outcome","aware"]},

    {"question":"How do you keep up with rapid changes in your field?",
     "ideal":"I follow three sources I trust deeply rather than everything. I read one long-form technical paper per week write a one-paragraph summary for myself and share the ones that are relevant with my team. I also contribute to one open-source project which forces me to understand new codebases rather than just read about them.",
     "category":"Adaptability","difficulty":"Easy","gt_pct":89,
     "keywords":["learn","current","deliberate","share","contribute"]},

    {"question":"Tell me about a time you had to let go of an approach you believed in.",
     "ideal":"I was convinced that microservices were the right architecture for our product. After six months of development the operational overhead was slowing us down significantly. User data showed no scaling need that justified the complexity. I advocated for consolidating back to a modular monolith. It was humbling but the right technical decision.",
     "category":"Adaptability","difficulty":"Hard","gt_pct":94,
     "keywords":["let go","data","decision","humble","outcome"]},

    {"question":"How do you deal with sudden changes in project scope?",
     "ideal":"I document the change immediately assess the impact on timeline resource and quality and present three options to the stakeholder: absorb the change with a timeline extension reduce scope elsewhere or increase resources. I never silently absorb scope changes because they accumulate invisibly and cause missed deadlines.",
     "category":"Adaptability","difficulty":"Medium","gt_pct":92,
     "keywords":["scope","assess","option","transparent","process"]},

    {"question":"Tell me about a time you worked effectively despite significant ambiguity.",
     "ideal":"We were asked to build a product for a market that did not yet exist. I defined a set of learning milestones instead of delivery milestones: by week four we will know if users have this problem by week eight we will know if they will pay. Each milestone gave us permission to continue or stop. We found product-market fit in 12 weeks.",
     "category":"Adaptability","difficulty":"Hard","gt_pct":94,
     "keywords":["ambiguity","milestone","learn","validate","outcome"]},

    {"question":"How do you ensure you remain effective when your tools or processes change?",
     "ideal":"I treat tool changes as learning sprints. When our team moved from Jira to Linear I allocated three hours to go through the documentation built a personal workflow that matched my old habits and then identified two new features I had not had before. I was at full productivity within a week and sharing tips with colleagues by week two.",
     "category":"Adaptability","difficulty":"Easy","gt_pct":88,
     "keywords":["tool","learn","productivity","fast","share"]},

    # COLLABORATION (5 new)
    {"question":"How do you resolve disagreements within your team?",
     "ideal":"I separate the people from the problem. I ask each person to write down their position and their reasoning before we discuss because it reduces the emotional charge. In the meeting I focus the group on the shared goal and ask what evidence would change your mind. Most technical disagreements dissolve when you agree on the decision criteria.",
     "category":"Collaboration","difficulty":"Hard","gt_pct":93,
     "keywords":["disagree","evidence","shared","criteria","resolve"]},

    {"question":"Tell me about a time you helped a team member succeed.",
     "ideal":"A designer on our cross-functional team was excluded from technical discussions because engineers assumed they would not understand. I invited them to our next architecture review prepared a one-page glossary of terms we used and introduced them as a key decision-maker. They surfaced a UX constraint that saved us three weeks of rework.",
     "category":"Collaboration","difficulty":"Medium","gt_pct":92,
     "keywords":["include","support","outcome","advocate","team"]},

    {"question":"How do you handle situations where team members are not pulling their weight?",
     "ideal":"I first try to understand why. Underperformance is usually a symptom of unclear expectations missing skills or personal issues rather than low effort. I have a direct private conversation where I name the specific gap and ask if anything is blocking them. I then agree on explicit next steps with a follow-up date.",
     "category":"Collaboration","difficulty":"Hard","gt_pct":93,
     "keywords":["underperform","diagnose","direct","expectation","follow-up"]},

    {"question":"How do you make sure all voices are heard in a team meeting?",
     "ideal":"I use structured turn-taking for important decisions. Before a vote or commitment I do a quick round where each person states their view in one sentence. I also actively solicit the quietest voices first because loudness in meetings correlates with seniority not quality of ideas. I have found this consistently surfaces the best input.",
     "category":"Collaboration","difficulty":"Medium","gt_pct":90,
     "keywords":["inclusive","voice","structure","diverse","meeting"]},

    {"question":"Tell me about a time you fostered collaboration between competing teams.",
     "ideal":"Two product teams were competing for the same infrastructure resources and creating tension. I organised a joint working session where both teams mapped their dependencies on a shared board. They discovered that two of their roadmap items were identical and combined them. The collaboration produced a shared platform component that benefited both and reduced duplicated effort by four months.",
     "category":"Collaboration","difficulty":"Hard","gt_pct":94,
     "keywords":["compete","align","shared","outcome","facilitate"]},

    # SELF-AWARENESS (5 new)
    {"question":"Tell me about a time you received feedback that changed how you work.",
     "ideal":"A mentor told me that I was solving problems for my team rather than developing their ability to solve them. It stung because I thought I was being helpful. I changed my default response to questions from here is the answer to what have you tried so far. Six months later two junior engineers on my team were independently solving problems I would previously have handled.",
     "category":"Self-Awareness","difficulty":"Medium","gt_pct":93,
     "keywords":["feedback","change","develop","reflect","outcome"]},

    {"question":"What do you know now that you wish you had known earlier in your career?",
     "ideal":"I wish I had known that saying I do not know is a strength not a weakness. Early in my career I would give confident answers on things I was uncertain about and occasionally I was wrong which damaged trust. Learning to distinguish between what I know and what I believe changed how people relied on me.",
     "category":"Self-Awareness","difficulty":"Medium","gt_pct":91,
     "keywords":["learn","honest","trust","growth","reflect"]},

    {"question":"How do you recognise when you are at your limit?",
     "ideal":"I watch for two signals: I start skipping my weekly review and my error rate on routine tasks increases. When I notice either I treat it as a system warning not a willpower problem. I immediately reduce my commitments for the following week and communicate the change proactively. Prevention is less expensive than burnout.",
     "category":"Self-Awareness","difficulty":"Hard","gt_pct":92,
     "keywords":["limit","burnout","signal","prevent","communicate"]},

    {"question":"How do you maintain your performance under sustained stress?",
     "ideal":"I have three non-negotiables: seven hours of sleep daily exercise and one full offline day per week. I treat these as professional commitments because research consistently shows that cognitive performance degrades significantly without them. I have never cancelled a deadline to protect them but I have declined non-essential meetings.",
     "category":"Self-Awareness","difficulty":"Medium","gt_pct":90,
     "keywords":["stress","maintain","habit","non-negotiable","performance"]},

    {"question":"What type of work environment brings out your best performance?",
     "ideal":"I thrive in environments where the problem is genuinely hard the team is high-trust and there is a clear link between my work and user outcomes. I underperform in environments where process is prioritised over output or where there is no feedback loop on whether what I shipped actually worked.",
     "category":"Self-Awareness","difficulty":"Easy","gt_pct":88,
     "keywords":["environment","thrive","trust","outcome","honest"]},

    # CAREER GOALS (5 new)
    {"question":"How do you decide what skills to invest in next?",
     "ideal":"I map my current skills against the skills of people doing the job I want in two years and identify the largest gap with the highest leverage. I then design a 12-week deliberate practice sprint for that skill with a concrete output at the end such as a working prototype or a published write-up. Output-based learning is far more effective than consumption-based learning.",
     "category":"Career Goals","difficulty":"Medium","gt_pct":92,
     "keywords":["skill","gap","deliberate","plan","output"]},

    {"question":"What does success look like for you in the first 90 days of this role?",
     "ideal":"In the first 30 days I want to have met every key stakeholder understood the top three business priorities and mapped the current technical landscape. By day 60 I want to have shipped one meaningful contribution no matter how small. By day 90 I want to have a clear 6-month plan that my manager and I have aligned on. Fast learning and early delivery build the credibility needed to make larger changes.",
     "category":"Career Goals","difficulty":"Medium","gt_pct":93,
     "keywords":["90 days","plan","stakeholder","deliver","align"]},

    {"question":"Why are you interested in moving into management?",
     "ideal":"I have found that my highest leverage is no longer in writing code but in developing the people who write it. When I mentor engineers and they independently ship better work than I would have the multiplier effect is far greater than anything I can produce alone. I want to make that the focus of my role not a side activity.",
     "category":"Career Goals","difficulty":"Medium","gt_pct":91,
     "keywords":["management","leverage","develop","multiplier","reason"]},

    {"question":"What would make you leave a company after you join?",
     "ideal":"Two things: a culture where cutting corners on quality becomes normalised and a management chain that is not honest about business reality. I can adapt to almost everything else including role changes resource constraints and strategic pivots as long as quality and honesty are preserved.",
     "category":"Career Goals","difficulty":"Hard","gt_pct":90,
     "keywords":["leave","quality","honest","value","honest"]},

    {"question":"How do you evaluate whether a company is the right fit for you?",
     "ideal":"I look at three things: the quality of the engineering practices the growth trajectory of the people who have been here three to five years and whether the leadership is honest about trade-offs rather than presenting everything as optimal. I ask former employees as well as current ones because departure patterns are highly informative.",
     "category":"Career Goals","difficulty":"Medium","gt_pct":91,
     "keywords":["fit","evaluate","culture","research","honest"]},
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    return re.findall(r"\b[a-z]{2,}\b", text.lower())

def _stopwords() -> set:
    return {
        "a","an","the","and","or","but","in","on","at","to","for","of","with",
        "is","was","are","were","be","been","being","have","has","had","do",
        "does","did","will","would","could","should","may","might","i","my",
        "me","we","our","you","your","they","their","it","its","this","that",
        "these","those","as","by","from","up","about","into","then","than",
        "so","if","when","which","who","how","what","not","also","just","very",
        "more","much","many","some","any","all","each","also","just",
    }


# ---------------------------------------------------------------------------
# BASELINE SCORERS
# ---------------------------------------------------------------------------

def score_keyword_match(answer: str, ideal: str, keywords: List[str]) -> float:
    if not answer.strip():
        return 0.0
    tokens = set(_tokenise(answer))
    kws    = [k.lower().strip() for k in keywords if k.strip()]
    if not kws:
        stops    = _stopwords()
        ideal_t  = {w for w in _tokenise(ideal)  if w not in stops}
        ans_t    = {w for w in tokens if w not in stops}
        if not ideal_t:
            return 0.0
        return round(len(ans_t & ideal_t) / len(ideal_t) * 100, 1)
    hits = sum(1 for k in kws if k in answer.lower())
    return round(hits / len(kws) * 100, 1)


def score_tfidf(answer: str, ideal: str) -> float:
    if not SKLEARN_OK or not answer.strip() or not ideal.strip():
        return 0.0
    try:
        vect = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        vecs = vect.fit_transform([ideal, answer])
        sim  = float(_sk_cos(vecs[0], vecs[1])[0][0])
        return round(sim * 100, 1)
    except Exception:
        return 0.0


def score_bm25(answer: str, ideal: str) -> float:
    k1, b  = 1.5, 0.75
    stops  = _stopwords()
    corpus = [w for w in _tokenise(ideal)  if w not in stops]
    query  = [w for w in _tokenise(answer) if w not in stops]
    if not corpus or not query:
        return 0.0
    avgdl = len(corpus)
    tf    = {}
    for w in corpus:
        tf[w] = tf.get(w, 0) + 1
    score = 0.0
    for term in set(query):
        f = tf.get(term, 0)
        if f == 0:
            continue
        idf  = math.log((1 + 1) / (1 + (1 if f > 0 else 0)) + 1)
        num  = f * (k1 + 1)
        den  = f + k1 * (1 - b + b * len(corpus) / avgdl)
        score += idf * num / den
    max_s = sum(
        math.log(2) * (tf[w] * (k1 + 1)) / (tf[w] + k1 * (1 - b + b))
        for w in set(corpus)
    )
    if max_s <= 0:
        return 0.0
    return round(min(score / max_s * 100, 100), 1)


def score_sbert(answer: str, ideal: str) -> float:
    try:
        from sentence_transformers import SentenceTransformer
        m    = SentenceTransformer("all-MiniLM-L6-v2")
        embs = m.encode([ideal, answer], normalize_embeddings=True)
        sim  = float(np.dot(embs[0], embs[1]))
        return round(max(0.0, sim) * 100, 1)
    except Exception:
        return score_tfidf(answer, ideal)


def _aura_subscores(answer: str, ideal: str, keywords: List[str],
                    category: str = "", groq_api_key: str = "") -> Dict:
    """
    Compute all Aura AI sub-scores and return them as a dict.
    Used both by score_aura() (returns composite) and by the UI breakdown panel.

    Sub-scores (all 0-1 unless noted):
      rel          — LLM or SBERT semantic relevance          weight 0.30
      kw           — keyword coverage                         weight 0.18
      star         — STAR / SOAR structure (4 components)     weight 0.15
      quant        — quantification bonus (numbers/%)         weight 0.08
      ttr          — vocabulary richness (Type-Token Ratio)   weight 0.07
      discourse    — discourse connectors (structured logic)  weight 0.07
      coherence    — opening sentence relevance               weight 0.05
      depth_flu    — depth + fluency composite                weight 0.06
      active_voice — active vs passive voice ratio            weight 0.04
    """
    if not answer.strip():
        return {}

    words_raw = answer.split()
    words     = [w.lower() for w in words_raw]
    wc        = len(words)
    ans_l     = answer.lower()

    # ── 1. Semantic relevance (LLM or SBERT) ─────────────────────────────────
    rel = 0.0
    if groq_api_key:
        try:
            from groq import Groq as _G
            import json as _j
            r = _G(api_key=groq_api_key).chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": (
                    f"Score how well the candidate answer covers the key concepts "
                    f"in the ideal answer. Return ONLY JSON: {{\"relevance\":0.0-1.0}}\n\n"
                    f"Ideal: {ideal[:600]}\n\nCandidate: {answer[:600]}"
                )}],
                max_tokens=32, temperature=0.0,
            )
            rel = float(_j.loads(r.choices[0].message.content.strip()
                        .replace("```json", "").replace("```", ""))["relevance"])
        except Exception:
            rel = score_sbert(answer, ideal) / 100
    else:
        rel = score_sbert(answer, ideal) / 100

    # ── 2. Keyword coverage ────────────────────────────────────────────────────
    kw = score_keyword_match(answer, ideal, keywords) / 100

    # ── 3. STAR / SOAR structure ───────────────────────────────────────────────
    _is_b = category.lower() in (
        "behavioural", "behavioral", "hr", "leadership",
        "problem solving", "adaptability", "collaboration",
        "self-awareness", "communication",
    )
    _STAR_PATS = [
        r"\b(situation|context|when|once|during|while|at the time|in my)\b",
        r"\b(task|goal|objective|responsible|needed to|had to|my role|was asked)\b",
        r"\b(i did|i took|i used|implemented|developed|decided|built|led|created|designed|initiated)\b",
        r"\b(result|outcome|achieved|improved|reduced|success|delivered|increased|saved|completed)\b",
    ]
    sh = sum(1 for p in _STAR_PATS if re.search(p, ans_l))
    star = sh / 4.0 if _is_b else 0.75

    # ── 4. Quantification bonus ────────────────────────────────────────────────
    # Numbers, percentages, timeframes — signal specificity
    quant_hits = len(re.findall(
        r'\b\d+[\.,]?\d*\s*(%|percent|x|times|hours?|days?|weeks?|months?|years?'
        r'|k|thousand|million|users?|members?|points?|ms|seconds?)\b'
        r'|\b\d{2,}\b',   # any 2+ digit number
        ans_l
    ))
    quant = min(1.0, quant_hits / 3.0)   # full score at 3+ quantified facts

    # ── 5. Vocabulary richness — Type-Token Ratio (TTR) ────────────────────────
    # Adjusted TTR: sqrt(unique) / sqrt(total) to reduce length bias
    ttr = (len(set(words)) ** 0.5) / max(1, wc ** 0.5) if wc > 0 else 0.0
    ttr = min(1.0, ttr)

    # ── 6. Discourse connectors (structured logical flow) ─────────────────────
    _DISCOURSE = [
        r"\b(however|therefore|consequently|as a result|furthermore|moreover|"
        r"in addition|on the other hand|nevertheless|although|whereas|"
        r"for example|for instance|specifically|in particular|"
        r"first(ly)?|second(ly)?|third(ly)?|finally|subsequently|"
        r"because|since|thus|hence|this led to|which meant)\b"
    ]
    disc_hits = sum(len(re.findall(p, ans_l)) for p in _DISCOURSE)
    discourse = min(1.0, disc_hits / 4.0)   # full score at 4+ connectors

    # ── 7. Answer coherence — opening sentence addresses question ──────────────
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
    coherence = 0.5  # default neutral
    if sentences and keywords:
        first_sent_l = sentences[0].lower()
        kw_in_open   = sum(1 for k in keywords if k.lower() in first_sent_l)
        coherence    = min(1.0, 0.3 + kw_in_open / max(1, len(keywords)) * 0.7)

    # ── 8. Depth + Fluency ────────────────────────────────────────────────────
    if   wc < 40:   d = 0.3
    elif wc < 80:   d = 0.5 + (wc - 40) / 40 * 0.3
    elif wc <= 200: d = 0.8 + (wc - 80) / 120 * 0.2
    elif wc <= 350: d = 1.0
    elif wc <= 500: d = 1.0 - (wc - 350) / 150 * 0.2
    else:           d = 0.6
    fillers  = ["um", "uh", "like", "basically", "actually", "you know", "kind of", "sort of"]
    fill_r   = sum(words.count(f) for f in fillers) / max(1, wc)
    fluency  = max(0.3, 1.0 - fill_r * 20)
    depth_flu = d * 0.6 + fluency * 0.4

    # ── 9. Active vs passive voice ratio ──────────────────────────────────────
    passive_hits = len(re.findall(
        r'\b(was|were|been|being|is|are)\s+\w+ed\b', ans_l
    ))
    active_voice = max(0.0, 1.0 - passive_hits / max(1, len(sentences)) * 0.5)

    return {
        "rel":          rel,
        "kw":           kw,
        "star":         star,
        "star_count":   sh,
        "quant":        quant,
        "quant_hits":   quant_hits,
        "ttr":          ttr,
        "discourse":    discourse,
        "disc_hits":    disc_hits,
        "coherence":    coherence,
        "depth_flu":    depth_flu,
        "active_voice": active_voice,
        "word_count":   wc,
        "sentence_count": len(sentences),
    }


def score_aura(answer: str, ideal: str, keywords: List[str],
               category: str = "", groq_api_key: str = "") -> float:
    """
    Aura AI composite score (0-100).
    Weights (BiLSTM-inspired multi-signal formula):
      Semantic relevance   30%
      Keyword coverage     18%
      STAR structure       15%
      Quantification        8%
      Vocabulary richness   7%
      Discourse connectors  7%
      Coherence             5%
      Depth + Fluency       6%
      Active voice          4%
    """
    if not answer.strip():
        return 0.0
    s = _aura_subscores(answer, ideal, keywords, category, groq_api_key)
    if not s:
        return 0.0
    composite = (
        s["rel"]          * 0.30 +
        s["kw"]           * 0.18 +
        s["star"]         * 0.15 +
        s["quant"]        * 0.08 +
        s["ttr"]          * 0.07 +
        s["discourse"]    * 0.07 +
        s["coherence"]    * 0.05 +
        s["depth_flu"]    * 0.06 +
        s["active_voice"] * 0.04
    )
    return round(min(100.0, composite * 100), 1)



# ---------------------------------------------------------------------------
# GROQ DYNAMIC GROUND TRUTH  (v2.0)
# ---------------------------------------------------------------------------

def groq_ground_truth(
    answer: str,
    question: str,
    ideal: str,
    keywords: List[str],
    category: str,
    difficulty: str,
    groq_api_key: str,
) -> Tuple[float, str]:
    """
    Ask Groq LLM to rate the *actual submitted answer* on a 0–100 scale,
    acting as an expert interviewer.  This replaces the hardcoded gt_pct
    value so the waterfall chart compares all scorers against a real
    per-answer quality rating rather than the idealised answer score.

    Returns:
        (score_pct: float, source: str)
        source is "groq_gt" on success, "gt_pct_fallback" when the call
        fails or no key is provided.

    Prompt design:
      • Gives the LLM the question, the ideal answer, the expected keywords,
        the category and difficulty so it has full context.
      • Uses a Behaviorally Anchored Rating Scale (BARS) so ratings are
        consistent and not inflated.
      • Temperature=0 for determinism across repeated runs on the same answer.
      • Returns ONLY JSON to prevent markdown wrapping issues.
    """
    if not groq_api_key or not answer.strip():
        return 0.0, "gt_pct_fallback"

    kw_str   = ", ".join(keywords) if keywords else "none specified"
    bars_map = {
        "behavioural": (
            "  100 — Full STAR story, every keyword present, quantified result, "
            "personal ownership clear.\n"
            "   85 — Strong STAR, most keywords, one minor gap (missing metric or "
            "weak Result).\n"
            "   70 — Partial STAR (3/4 elements), some keywords, answer is on-topic "
            "but lacks specificity.\n"
            "   50 — Generic story, 1-2 keywords, no clear Result or measurable "
            "outcome.\n"
            "   30 — Vague, off-topic or theoretical; no personal example.\n"
            "    0 — No answer or completely irrelevant."
        ),
        "leadership": (
            "  100 — Concrete leadership example, clear outcome/impact, adaptive "
            "style demonstrated.\n"
            "   85 — Good example with outcome, minor lack of depth on impact.\n"
            "   70 — Reasonable approach described but lacks concrete example or "
            "metrics.\n"
            "   50 — Generic leadership philosophy, no evidence.\n"
            "   30 — Superficial or does not address the leadership dimension.\n"
            "    0 — No answer or irrelevant."
        ),
        "communication": (
            "  100 — Specific example, clear audience-awareness, structured "
            "message, measurable outcome.\n"
            "   85 — Good example with clear technique, minor gap in outcome.\n"
            "   70 — Relevant example but technique or outcome is vague.\n"
            "   50 — Generic communication principles, no concrete evidence.\n"
            "   30 — Superficial or only restates the question.\n"
            "    0 — No answer or irrelevant."
        ),
        "default": (
            "  100 — Fully addresses the question with concrete evidence, all "
            "keywords covered, quantified impact.\n"
            "   85 — Strong answer, most keywords, minor gap in depth or evidence.\n"
            "   70 — On-topic with some evidence, but missing depth or keywords.\n"
            "   50 — Partially relevant; generic phrasing; some keywords present.\n"
            "   30 — Vague or mostly off-topic.\n"
            "    0 — No answer or completely irrelevant."
        ),
    }
    bars = bars_map.get(category.lower(), bars_map["default"])

    prompt = (
        "You are a senior interviewer rating a candidate's answer.\n"
        "Return ONLY a JSON object with a single key 'score' whose value is "
        "an integer between 0 and 100.  No explanation, no markdown.\n\n"
        f"Question   : {question}\n"
        f"Category   : {category}  |  Difficulty: {difficulty}\n"
        f"Keywords expected: {kw_str}\n\n"
        f"Ideal answer (for reference only):\n{ideal[:700]}\n\n"
        f"Candidate answer:\n{answer[:700]}\n\n"
        "Behaviorally Anchored Rating Scale:\n"
        f"{bars}\n\n"
        "Rules:\n"
        "  • Score the CANDIDATE answer, not the ideal answer.\n"
        "  • Be calibrated: a strong but not perfect answer should score 70-85.\n"
        "  • Do NOT inflate scores — a generic one-liner is 30 or below.\n"
        "  • Ignore spelling/grammar unless it severely impairs meaning.\n"
        'Return ONLY valid JSON. Example: {"score": 72}'
    )

    try:
        from groq import Groq as _G
        import json as _j
        r = _G(api_key=groq_api_key).chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32,
            temperature=0.0,
        )
        raw   = r.choices[0].message.content.strip()
        raw   = raw.replace("```json", "").replace("```", "").strip()
        score = float(_j.loads(raw)["score"])
        score = max(0.0, min(100.0, score))
        return round(score, 1), "groq_gt"
    except Exception:
        return 0.0, "gt_pct_fallback"


# ---------------------------------------------------------------------------
# DATASET LOADER  (v3.1 — JSON-first, 3-strategy cascade, timeout-guarded)
# ---------------------------------------------------------------------------

#: Candidate local filenames searched in order (JSON preferred, CSV fallback).
_LOCAL_FILENAMES = [
    "hr_interview_dataset.json",
    "hr_interview_dataset.jsonl",
    "HR_Interview_Questions_and_Ideal_Answers.json",
    "hr_interview_dataset.csv",
    "HR_Interview_Questions_and_Ideal_Answers.csv",
]

#: Column/key aliases for each logical field (case-insensitive).
_CSV_COLUMNS = {
    "question":   ["Question"],
    "ideal":      ["Ideal Answer", "Ideal_Answer", "ideal_answer", "answer"],
    "category":   ["Category"],
    "difficulty": ["Difficulty"],
    "keywords":   ["Keywords"],
}


def _find_local_file() -> Optional[str]:
    """Return path to the first local dataset file found, or None."""
    search_dirs = [Path(__file__).parent, Path.cwd()]
    for d in search_dirs:
        for name in _LOCAL_FILENAMES:
            candidate = d / name
            if candidate.is_file():
                return str(candidate)
    return None


def _normalise_record(raw: dict) -> Optional[dict]:
    """
    Convert a raw JSON/CSV dict into the internal record format.
    Returns None if the record lacks a question or ideal answer.
    """
    def _pick(keys):
        for k in keys:
            for rk, rv in raw.items():
                if rk.strip().lower() == k.lower():
                    v = str(rv).strip()
                    if v and v.lower() != "nan":
                        return v
        return None

    q = _pick(["Question"])
    a = _pick(["Ideal Answer", "Ideal_Answer", "ideal_answer", "answer"])
    if not q or not a:
        return None

    kw_raw = _pick(["Keywords"]) or ""
    kws = [k.strip() for k in re.split(r"[,;]", kw_raw) if k.strip()]

    return {
        "question":   q,
        "ideal":      a,
        "category":   _pick(["Category"])   or "General",
        "difficulty": _pick(["Difficulty"]) or "Medium",
        "gt_pct":     91.0,
        "keywords":   kws,
    }


def _parse_json_file(path: str) -> List[Dict]:
    """Parse a JSON or JSONL file into internal records. Returns [] on failure."""
    import json as _json
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # ── Try full JSON parse first (array or wrapped object) ──────────
        try:
            data = _json.loads(content)
            if isinstance(data, list):
                raw_records = data
            elif isinstance(data, dict):
                # Find first list value: {"data":[...], "questions":[...], etc.}
                raw_records = next(
                    (v for v in data.values() if isinstance(v, list)), [data]
                )
            else:
                raw_records = []
        except _json.JSONDecodeError:
            # ── JSONL fallback (one JSON object per line) ─────────────────
            raw_records = []
            for line in content.splitlines():
                line = line.strip()
                if line:
                    try:
                        obj = _json.loads(line)
                        if isinstance(obj, dict):
                            raw_records.append(obj)
                    except _json.JSONDecodeError:
                        pass

        records = []
        for r in raw_records:
            if not isinstance(r, dict):
                continue
            norm = _normalise_record(r)
            if norm:
                records.append(norm)
        return records

    except Exception:
        return []


def _parse_csv_file(path: str) -> List[Dict]:
    """Parse a CSV file into internal records. Returns [] on failure."""
    if not PANDAS_OK:
        return []
    try:
        import pandas as pd
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        records = []
        for _, row in df.iterrows():
            norm = _normalise_record(dict(row))
            if norm:
                records.append(norm)
        return records
    except Exception:
        return []


def _parse_local_file(path: str) -> List[Dict]:
    """Dispatch to JSON or CSV parser based on file extension."""
    ext = Path(path).suffix.lower()
    if ext in (".json", ".jsonl"):
        return _parse_json_file(path)
    return _parse_csv_file(path)


def _kaggle_download_with_timeout(timeout_sec: int = 20) -> Tuple[Optional[str], str]:
    """
    Attempt kagglehub download with a hard timeout.
    Returns (file_path_or_None, error_message).
    Uses SIGALRM on POSIX; skips timeout guard on Windows.
    """
    if not KAGGLE_OK:
        return None, "kagglehub not installed (pip install kagglehub)"
    if not PANDAS_OK:
        return None, "pandas not installed (pip install pandas)"

    _use_alarm = hasattr(signal, "SIGALRM") and platform.system() != "Windows"

    class _Timeout(Exception):
        pass

    def _handler(signum, frame):
        raise _Timeout()

    try:
        if _use_alarm:
            signal.signal(signal.SIGALRM, _handler)
            signal.alarm(timeout_sec)

        import kagglehub
        path = kagglehub.dataset_download(
            "aryan208/hr-interview-questions-and-ideal-answers"
        )

        if _use_alarm:
            signal.alarm(0)

        # Prefer JSON, fall back to CSV
        for pattern in ("**/*.json", "**/*.jsonl", "**/*.csv"):
            hits = glob.glob(os.path.join(path, pattern), recursive=True)
            if hits:
                return hits[0], ""

        return None, f"Kaggle download succeeded but no JSON/CSV found in: {path}"

    except _Timeout:
        return None, (
            f"Kaggle download timed out after {timeout_sec}s. "
            "The network proxy likely blocked api.kaggle.com. "
            "Place hr_interview_dataset.json in the project folder instead."
        )
    except Exception as exc:
        exc_str = str(exc)
        if any(k in exc_str.lower() for k in
               ("403", "forbidden", "proxy", "connectionerror",
                "unauthorized", "401", "no kaggle", "credential")):
            return None, (
                f"Kaggle network/auth error: {exc_str}. "
                "Place hr_interview_dataset.json next to model_comparison.py "
                "to bypass network entirely (Strategy 1)."
            )
        return None, f"Kaggle load failed: {exc_str}"
    finally:
        if _use_alarm:
            try:
                signal.alarm(0)
            except Exception:
                pass


def load_dataset() -> Tuple[List[Dict], str, str]:
    """
    Load the HR interview dataset using a 3-strategy cascade.

    Strategy 1 — Local file (JSON preferred, CSV accepted):
        Searches the module directory then CWD for these filenames in order:
          hr_interview_dataset.json / .jsonl
          HR_Interview_Questions_and_Ideal_Answers.json
          hr_interview_dataset.csv / HR_Interview_Questions_and_Ideal_Answers.csv
        Handles: top-level array, wrapped object, JSONL.
        Use ``save_dataset_template()`` to generate a starter JSON file.

    Strategy 2 — kagglehub download (20-second timeout):
        Requires ``pip install kagglehub`` + valid Kaggle credentials.
        Detects 403/ProxyError explicitly; never hangs the app.

    Strategy 3 — Embedded dataset (105 entries, always available).

    Returns
    -------
    (records, source_label, error_message)
        source_label : ``'local_json'`` | ``'local_csv'`` | ``'kaggle'`` | ``'embedded'``
        error_message: ``''`` on clean success, human-readable on fallback.
    """
    errors: List[str] = []

    # ── Strategy 1: local file ────────────────────────────────────────────
    local_path = _find_local_file()
    if local_path:
        records = _parse_local_file(local_path)
        if records:
            ext = Path(local_path).suffix.lower()
            src = "local_json" if ext in (".json", ".jsonl") else "local_csv"
            return records, src, ""
        errors.append(
            f"Local file found at '{local_path}' but could not be parsed. "
            "Expected keys: Question, Ideal Answer, Category, Difficulty, Keywords."
        )

    # ── Strategy 2: kagglehub (timeout-guarded) ───────────────────────────
    dl_path, kag_err = _kaggle_download_with_timeout(timeout_sec=20)
    if dl_path:
        records = _parse_local_file(dl_path)
        if records:
            return records, "kaggle", ""
        errors.append(f"Kaggle file downloaded to '{dl_path}' but parsing failed.")
    else:
        errors.append(kag_err)

    # ── Strategy 3: embedded fallback ────────────────────────────────────
    combined_error = " | ".join(e for e in errors if e)
    return (
        _EMBEDDED,
        "embedded",
        f"Using embedded dataset (105 entries). Reasons: {combined_error}",
    )


def save_dataset_template(path: Optional[str] = None) -> str:
    """
    Write the embedded dataset as a JSON file (Strategy-1 ready).

    Creates ``hr_interview_dataset.json`` next to this module (or at
    ``path`` if supplied).  The file can be extended with real dataset
    entries and will be picked up automatically by ``load_dataset()``.

    Returns the absolute path of the written file.

    Example
    -------
    >>> from model_comparison import save_dataset_template
    >>> save_dataset_template()
    '/path/to/project/hr_interview_dataset.json'
    """
    import json as _json

    out_path = path or str(Path(__file__).parent / "hr_interview_dataset.json")
    rows = [
        {
            "Question":    e["question"],
            "Ideal Answer": e["ideal"],
            "Category":    e["category"],
            "Difficulty":  e["difficulty"],
            "Keywords":    ", ".join(e.get("keywords", [])),
        }
        for e in _EMBEDDED
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(rows, f, indent=2, ensure_ascii=False)
    return out_path



# ---------------------------------------------------------------------------
# SIMULATED CANDIDATE ANSWER TIERS
# ---------------------------------------------------------------------------

def _make_tiers(
    entry: Dict,
    groq_api_key: str = "",
) -> List[Tuple[str, float]]:
    """
    Returns [(answer_text, true_quality_pct)] for 3 performance tiers.

    true_quality_pct is determined by groq_ground_truth() when a key is
    supplied, giving a real per-answer quality rating.  Falls back to the
    fixed tier buckets (83 / 50 / 18) when no key is available.

    Tiers:
      Strong  — keeps STAR structure, most keywords, drops one metric
      Average — generic, vague, 1–2 keywords, no STAR
      Weak    — single vague sentence, no keywords, no structure
    """
    kws   = entry.get("keywords", [])
    kw1   = kws[0] if kws else "challenges"
    kw2   = kws[1] if len(kws) > 1 else "results"
    ideal = entry["ideal"]
    q     = entry.get("question", "")
    cat   = entry.get("category",   "General")
    diff  = entry.get("difficulty", "Medium")

    sents  = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ideal) if s.strip()]
    strong = " ".join(sents[:max(2, len(sents) - 1)])

    average = (
        f"I have dealt with situations like this before by focusing on {kw1} "
        f"and making sure the {kw2} were good. "
        f"I think communication and working together with the team is very important."
    )
    weak = (
        "I usually try my best in these situations and talk to people to figure it out."
    )

    answers = [strong, average, weak]
    static_fallbacks = [83.0, 50.0, 18.0]

    if groq_api_key:
        rated = []
        for ans, fallback in zip(answers, static_fallbacks):
            score, source = groq_ground_truth(
                answer=ans, question=q, ideal=ideal,
                keywords=kws, category=cat, difficulty=diff,
                groq_api_key=groq_api_key,
            )
            rated.append(score if source == "groq_gt" else fallback)
        return list(zip(answers, rated))

    return list(zip(answers, static_fallbacks))


# ---------------------------------------------------------------------------
# BENCHMARK RUNNER
# ---------------------------------------------------------------------------

def run_benchmark(
    groq_api_key: str = "",
    max_entries: Optional[int] = None,
) -> Dict:
    """
    Run all 5 scorers across the full dataset (cap removed in v2.0).

    Args:
        groq_api_key : Groq API key. When supplied, each simulated answer
                       tier is rated by groq_ground_truth() so Pearson r
                       is measured against real per-answer quality rather
                       than fixed bucket values (83/50/18).
        max_entries  : Optional hard limit on questions processed (None = all).
                       Kept for backward compatibility with UI sliders that
                       pass a value, but no longer has a default cap.

    Returns results dict (same schema as v1.0 for UI compatibility).
    """
    dataset, source, load_err = load_dataset()

    # Apply caller-supplied cap only — no internal default cap
    if max_entries is not None:
        dataset = dataset[:max_entries]

    SCORERS = ["Keyword Match", "TF-IDF", "BM25", "SBERT", "Aura AI"]
    all_pairs   = {s: [] for s in SCORERS}
    by_cat      = {}
    by_diff     = {}
    scored_cnt  = {s: 0  for s in SCORERS}

    for entry in dataset:
        cat  = entry["category"]
        diff = entry["difficulty"]
        kws  = entry.get("keywords", [])
        ideal= entry["ideal"]

        if cat  not in by_cat:  by_cat[cat]  = {s: [] for s in SCORERS}
        if diff not in by_diff: by_diff[diff] = {s: [] for s in SCORERS}

        # _make_tiers now calls groq_ground_truth() per answer when key is set
        for (ans, true_pct) in _make_tiers(entry, groq_api_key=groq_api_key):
            scores = {
                "Keyword Match": score_keyword_match(ans, ideal, kws),
                "TF-IDF":        score_tfidf(ans, ideal),
                "BM25":          score_bm25(ans, ideal),
                "SBERT":         score_sbert(ans, ideal),
                "Aura AI":       score_aura(ans, ideal, kws,
                                            category=cat,
                                            groq_api_key=groq_api_key),
            }
            for s, pred in scores.items():
                all_pairs[s].append((pred, true_pct))
                by_cat[cat][s].append(pred)
                by_diff[diff][s].append(pred)
                if pred > 0:
                    scored_cnt[s] += 1

    n_ans = len(dataset) * 3

    def _r(pairs):
        if len(pairs) < 2:
            return 0.0
        p, g = zip(*pairs)
        p, g = np.array(p, float), np.array(g, float)
        if p.std() == 0 or g.std() == 0:
            return 0.0
        return float(np.corrcoef(p, g)[0, 1])

    def _mae(pairs):
        return round(float(np.mean([abs(p - g) for p, g in pairs])), 1) if pairs else 0.0

    pearson  = {s: round(_r(all_pairs[s]), 3) for s in SCORERS}
    mae      = {s: _mae(all_pairs[s])          for s in SCORERS}
    consist  = {s: round(float(np.std([p for p, g in all_pairs[s] if g >= 75])), 1)
                for s in SCORERS}
    coverage = {s: round(scored_cnt[s] / max(1, n_ans) * 100, 1) for s in SCORERS}

    cat_avgs  = {cat: {s: round(float(np.mean(v)), 1) if v else 0.0
                       for s, v in sd.items()}
                 for cat, sd in by_cat.items()}
    diff_avgs = {diff: {s: round(float(np.mean(v)), 1) if v else 0.0
                        for s, v in sd.items()}
                 for diff, sd in by_diff.items()}

    bases      = ["Keyword Match", "TF-IDF", "BM25", "SBERT"]
    best_r     = max(pearson[b]  for b in bases)
    best_mae   = min(mae[b]      for b in bases)
    best_cov   = max(coverage[b] for b in bases)

    improvement = {
        "pearson_delta":     round(pearson["Aura AI"]  - best_r,   3),
        "mae_delta":         round(mae["Aura AI"]       - best_mae, 1),
        "coverage_delta":    round(coverage["Aura AI"]  - best_cov, 1),
        "best_baseline_r":   round(best_r,   3),
        "best_baseline_mae": round(best_mae, 1),
        "aura_r":            pearson["Aura AI"],
        "aura_mae":          mae["Aura AI"],
    }

    gt_mode = "groq_dynamic" if groq_api_key else "static_buckets"

    return {
        "scorers":        SCORERS,
        "pearson":        pearson,
        "mae":            mae,
        "consistency":    consist,
        "coverage":       coverage,
        "by_category":    cat_avgs,
        "by_difficulty":  diff_avgs,
        "n_questions":    len(dataset),
        "n_answers":      n_ans,
        "dataset_source": source,
        "dataset_error":  load_err,
        "improvement":    improvement,
        "gt_mode":        gt_mode,
    }

# ---------------------------------------------------------------------------
# COMPANY DATASET BENCHMARK  (v4.0 — Company_Data.csv integration)
# ---------------------------------------------------------------------------

#: Domain → benchmark category mapping
_DOMAIN_TO_CATEGORY: Dict[str, str] = {
    "Data Science":          "Data Science",
    "Data Analytics":        "Data Analytics",
    "Machine Learning":      "Machine Learning",
    "Machine Laerning":      "Machine Learning",   # typo fix
    "Data Engineer":         "Data Engineering",
    "Data Analyst":          "Data Analytics",
    "Python":                "Python",
    "SQL":                   "SQL",
    "Tableau":               "Data Visualisation",
    "Computer Vision":       "Machine Learning",
    "Reinforcement Learning":"Machine Learning",
    "Behavioral":            "Behavioural",
    "Behavioural":           "Behavioural",
    "General":               "General",
    "Technical":             "General",
    "All":                   "General",
    "Javascript":            "Programming",
    "Java":                  "Programming",
    "C++":                   "Programming",
    "DataBase":              "SQL",
}


def _generate_ideal_company(
    question: str,
    domain: str,
    company: str,
    groq_api_key: str,
) -> str:
    """
    Generate an ideal answer for a company dataset question via Groq.
    Returns "" on failure so the record is silently skipped.
    """
    if not groq_api_key:
        return ""
    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        company_ctx = f" at {company}" if company and company not in ("nan", "") else ""
        prompt = (
            f"You are a senior interviewer evaluating candidates for a {domain} role{company_ctx}.\n"
            f"Write a concise, high-quality model answer (80-120 words) to the following "
            f"interview question. Return ONLY the answer — no preamble, no labels.\n\n"
            f"Question: {question}"
        )
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


def load_company_dataset(
    csv_path: str = "",
    max_entries: Optional[int] = 40,
    groq_api_key: str = "",
    min_word_count: int = 6,
) -> Tuple[List[Dict], str, str]:
    """
    Load and prepare the Company_Data.csv for benchmarking.

    Steps:
      1. Find the CSV (supplied path → CWD → module dir).
      2. Deduplicate questions, fix domain typos, filter short questions.
      3. Generate ideal answers via Groq for each question (required by all scorers).
         Questions where Groq generation fails are silently skipped.
      4. Return records in the same internal format as load_dataset():
         { question, ideal, category, difficulty, gt_pct, keywords, company, year }

    Parameters
    ----------
    csv_path     : Explicit path to the CSV. If "", searches automatically.
    max_entries  : Cap on questions processed (default 40 keeps PDF build fast).
    groq_api_key : Required — without it ideal answers cannot be generated.
    min_word_count: Skip questions shorter than this many words.

    Returns
    -------
    (records, source_label, error_message)
    """
    # ── Find the file ────────────────────────────────────────────────────────
    candidate_names = [
        "Company_Data.csv",
        "1775377403173_Company_Data.csv",
        "company_data.csv",
    ]
    found_path = csv_path or ""
    if not found_path:
        for d in [Path.cwd(), Path(__file__).parent]:
            for name in candidate_names:
                p = d / name
                if p.is_file():
                    found_path = str(p)
                    break
            if found_path:
                break

    if not found_path:
        return [], "not_found", (
            "Company_Data.csv not found. Place it next to model_comparison.py "
            "or pass csv_path= explicitly."
        )

    if not PANDAS_OK:
        return [], "error", "pandas not installed — pip install pandas"

    # ── Load and clean ───────────────────────────────────────────────────────
    try:
        import pandas as pd
        df = pd.read_csv(found_path)
    except Exception as exc:
        return [], "error", f"CSV read failed: {exc}"

    # Fix domain typos and normalise
    df["domain"] = df["domain"].fillna("General").replace({
        "Machine Laerning": "Machine Learning",
        "Data Analyst":     "Data Analytics",
        "Behavioral":       "Behavioural",
    })

    # Drop duplicates, very short questions, nulls
    df = df.dropna(subset=["question"])
    df = df.drop_duplicates(subset="question", keep="first")
    df = df[df["question"].str.split().str.len() >= min_word_count]
    df = df.reset_index(drop=True)

    # Prefer questions that have a company tag (more informative for benchmark)
    has_co = df[df["company"].notna()]
    no_co  = df[df["company"].isna()]
    df = pd.concat([has_co, no_co]).reset_index(drop=True)

    # Apply cap
    if max_entries:
        df = df.iloc[:max_entries]

    if not groq_api_key:
        return [], "error", (
            "A Groq API key is required to generate ideal answers for the "
            "Company dataset. Pass groq_api_key= to load_company_dataset()."
        )

    # ── Generate ideal answers via Groq ──────────────────────────────────────
    records: List[Dict] = []
    for _, row in df.iterrows():
        q       = str(row["question"]).strip()
        domain  = str(row.get("domain", "General")).strip()
        company = str(row.get("company", "")).strip()
        year_v  = row.get("year", "")

        # Clean year
        try:
            year_str = str(int(float(year_v))) if year_v and str(year_v) not in ("nan", "") else ""
        except (ValueError, TypeError):
            year_str = ""

        ideal = _generate_ideal_company(q, domain, company, groq_api_key)
        if not ideal:
            continue   # skip if generation failed

        category = _DOMAIN_TO_CATEGORY.get(domain, domain)

        records.append({
            "question":   q,
            "ideal":      ideal,
            "category":   category,
            "difficulty": "Medium",         # not in dataset — default
            "gt_pct":     91.0,
            "keywords":   [],               # no keywords in dataset
            "company":    company if company not in ("nan", "", "None") else "",
            "year":       year_str,
        })

    if not records:
        return [], "error", "No records produced — all ideal answer generations failed."

    return records, "company_csv", ""


def run_company_benchmark(
    csv_path: str = "",
    groq_api_key: str = "",
    max_entries: Optional[int] = 40,
) -> Dict:
    """
    Run the full 5-scorer benchmark on the Company_Data.csv dataset.
    Identical output schema to run_benchmark() — drop-in for the PDF renderer.

    Extra keys in the returned dict:
      "company_breakdown": { company_name: { scorer: avg_score } }
      "dataset_name":      "Company Interview Questions"
    """
    records, source, err = load_company_dataset(
        csv_path     = csv_path,
        max_entries  = max_entries,
        groq_api_key = groq_api_key,
    )

    if not records:
        return {
            "error":          err,
            "dataset_source": source,
            "dataset_name":   "Company Interview Questions",
        }

    SCORERS = ["Keyword Match", "TF-IDF", "BM25", "SBERT", "Aura AI"]
    all_pairs       = {s: [] for s in SCORERS}
    by_cat          = {}
    by_company      = {}
    scored_cnt      = {s: 0 for s in SCORERS}

    for entry in records:
        cat     = entry["category"]
        ideal   = entry["ideal"]
        kws     = entry.get("keywords", [])
        company = entry.get("company", "Unknown")

        if cat not in by_cat:
            by_cat[cat] = {s: [] for s in SCORERS}
        if company and company not in by_company:
            by_company[company] = {s: [] for s in SCORERS}

        for (ans, true_pct) in _make_tiers(entry, groq_api_key=groq_api_key):
            scores = {
                "Keyword Match": score_keyword_match(ans, ideal, kws),
                "TF-IDF":        score_tfidf(ans, ideal),
                "BM25":          score_bm25(ans, ideal),
                "SBERT":         score_sbert(ans, ideal),
                "Aura AI":       score_aura(ans, ideal, kws,
                                            category=cat,
                                            groq_api_key=groq_api_key),
            }
            for s, pred in scores.items():
                all_pairs[s].append((pred, true_pct))
                by_cat[cat][s].append(pred)
                if company:
                    by_company[company][s].append(pred)
                if pred > 0:
                    scored_cnt[s] += 1

    n_ans = len(records) * 3

    def _r(pairs):
        if len(pairs) < 2:
            return 0.0
        p, g = zip(*pairs)
        p, g = np.array(p, float), np.array(g, float)
        if p.std() == 0 or g.std() == 0:
            return 0.0
        return float(np.corrcoef(p, g)[0, 1])

    def _mae(pairs):
        return round(float(np.mean([abs(p - g) for p, g in pairs])), 1) if pairs else 0.0

    pearson  = {s: round(_r(all_pairs[s]), 3) for s in SCORERS}
    mae      = {s: _mae(all_pairs[s])          for s in SCORERS}
    consist  = {s: round(float(np.std([p for p, g in all_pairs[s] if g >= 75])), 1)
                for s in SCORERS}
    coverage = {s: round(scored_cnt[s] / max(1, n_ans) * 100, 1) for s in SCORERS}

    cat_avgs  = {cat: {s: round(float(np.mean(v)), 1) if v else 0.0
                       for s, v in sd.items()}
                 for cat, sd in by_cat.items()}
    co_avgs   = {co: {s: round(float(np.mean(v)), 1) if v else 0.0
                      for s, v in sd.items()}
                 for co, sd in by_company.items()}

    bases    = ["Keyword Match", "TF-IDF", "BM25", "SBERT"]
    best_r   = max(pearson[b]  for b in bases)
    best_mae = min(mae[b]      for b in bases)
    best_cov = max(coverage[b] for b in bases)

    improvement = {
        "pearson_delta":     round(pearson["Aura AI"]  - best_r,   3),
        "mae_delta":         round(mae["Aura AI"]       - best_mae, 1),
        "coverage_delta":    round(coverage["Aura AI"]  - best_cov, 1),
        "best_baseline_r":   round(best_r,   3),
        "best_baseline_mae": round(best_mae, 1),
        "aura_r":            pearson["Aura AI"],
        "aura_mae":          mae["Aura AI"],
    }

    return {
        "scorers":            SCORERS,
        "pearson":            pearson,
        "mae":                mae,
        "consistency":        consist,
        "coverage":           coverage,
        "by_category":        cat_avgs,
        "company_breakdown":  co_avgs,
        "n_questions":        len(records),
        "n_answers":          n_ans,
        "dataset_source":     source,
        "dataset_name":       "Company Interview Questions",
        "dataset_error":      err,
        "improvement":        improvement,
        "gt_mode":            "groq_dynamic" if groq_api_key else "static_buckets",
    }