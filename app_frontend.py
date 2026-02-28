import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import random
import time
from datetime import datetime

# ════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NexInterview AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════
#  GLOBAL CSS  — deep-space dark theme with neon accents
# ════════════════════════════════════════════════════════
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>

/* ── reset & root ───────────────────────────────────── */
:root {
  --bg:       #060b18;
  --bg2:      #0d1526;
  --bg3:      #111d35;
  --surface:  #162040;
  --border:   #1e2d50;
  --cyan:     #00e5ff;
  --violet:   #a855f7;
  --green:    #00ffa3;
  --orange:   #ff6b35;
  --pink:     #ff3d7f;
  --yellow:   #ffe566;
  --text:     #dde6f5;
  --muted:    #5a7099;
  --font-h:   'Syne', sans-serif;
  --font-b:   'DM Sans', sans-serif;
}

/* app shell */
[data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  font-family: var(--font-b);
  color: var(--text);
}
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stHeader"]  { background: transparent !important; }
footer { display: none !important; }

/* remove streamlit chrome */
#MainMenu, .stDeployButton { display: none !important; }

/* ── typography ─────────────────────────────────────── */
h1,h2,h3 { font-family: var(--font-h) !important; color: #fff !important; }

/* ── inputs ─────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextInput > div > div > input:focus,
.stSelectbox > div > div > div,
.stTextArea textarea {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: var(--font-b) !important;
}
.stTextArea textarea:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 2px rgba(0,229,255,.15) !important;
}

/* ── buttons ─────────────────────────────────────────── */
.stButton > button {
  font-family: var(--font-h) !important;
  font-weight: 700 !important;
  font-size: .95rem !important;
  border: none !important;
  border-radius: 12px !important;
  height: 3em !important;
  width: 100% !important;
  letter-spacing: .03em !important;
  transition: transform .15s, box-shadow .15s !important;
  cursor: pointer !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 24px rgba(0,0,0,.4) !important;
}

/* radio pills */
.stRadio > div { flex-direction: column !important; gap: 6px !important; }
.stRadio label {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 10px 16px !important;
  color: var(--muted) !important;
  font-family: var(--font-b) !important;
  font-size: .9rem !important;
  cursor: pointer !important;
  transition: all .2s !important;
}
.stRadio label:hover { border-color: var(--cyan) !important; color: var(--text) !important; }
[data-testid="stWidgetLabel"] { color: var(--muted) !important; font-size: .8rem !important; }

/* progress bar */
.stProgress > div > div > div { background: linear-gradient(90deg,var(--cyan),var(--violet)) !important; border-radius: 99px !important; }
.stProgress > div > div { background: var(--surface) !important; border-radius: 99px !important; }

/* select slider */
.stSlider > div > div > div > div { background: var(--cyan) !important; }

/* ── cards ───────────────────────────────────────────── */
.nx-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px 24px;
  margin-bottom: 12px;
  position: relative;
  overflow: hidden;
}
.nx-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
}
.nx-card.cyan::before  { background: linear-gradient(90deg,var(--cyan),transparent); }
.nx-card.violet::before{ background: linear-gradient(90deg,var(--violet),transparent); }
.nx-card.green::before { background: linear-gradient(90deg,var(--green),transparent); }
.nx-card.orange::before{ background: linear-gradient(90deg,var(--orange),transparent); }
.nx-card.pink::before  { background: linear-gradient(90deg,var(--pink),transparent); }

/* ── metric tiles ────────────────────────────────────── */
.nx-metric {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 18px 16px;
  text-align: center;
  transition: transform .2s;
}
.nx-metric:hover { transform: translateY(-4px); }
.nx-metric .val  { font-family: var(--font-h); font-size: 2.1rem; font-weight: 800; }
.nx-metric .lbl  { font-size: .78rem; color: var(--muted); margin-top: 4px; letter-spacing:.05em; text-transform:uppercase; }
.nx-metric .delta{ font-size: .78rem; margin-top: 3px; }
.delta-pos { color: var(--green); }
.delta-neg { color: var(--pink); }

/* ── question box ────────────────────────────────────── */
.nx-question {
  background: linear-gradient(135deg, #0d1f3c, #162040);
  border: 1px solid var(--cyan);
  border-radius: 14px;
  padding: 20px 24px;
  font-family: var(--font-h);
  font-size: 1.15rem;
  font-weight: 600;
  color: #fff;
  line-height: 1.5;
  box-shadow: 0 0 30px rgba(0,229,255,.08);
  margin-bottom: 16px;
}
.nx-q-badge {
  display: inline-block;
  border-radius: 6px;
  padding: 2px 12px;
  font-size: .75rem;
  font-weight: 700;
  letter-spacing: .06em;
  margin-right: 10px;
  vertical-align: middle;
}

/* ── transcript live ─────────────────────────────────── */
.nx-transcript {
  background: #030d1a;
  border: 1px solid #003d2a;
  border-radius: 12px;
  padding: 16px;
  min-height: 90px;
  font-family: 'Courier New', monospace;
  font-size: .88rem;
  color: var(--green);
  line-height: 1.7;
}
.nx-chunk {
  display: inline-block;
  background: rgba(0,255,163,.08);
  border: 1px solid rgba(0,255,163,.2);
  border-radius: 20px;
  padding: 3px 12px;
  margin: 3px;
  font-size: .8rem;
  color: var(--green);
}
.nx-cursor {
  display: inline-block;
  width: 8px; height: 16px;
  background: var(--green);
  animation: blink 1s infinite;
  vertical-align: middle;
  margin-left: 4px;
  border-radius: 2px;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

/* ── star badges ─────────────────────────────────────── */
.star-hit  { background:rgba(0,255,163,.15);color:var(--green);  border:1px solid rgba(0,255,163,.3); border-radius:8px; padding:4px 14px; font-size:.82rem; margin:3px; display:inline-block; }
.star-miss { background:rgba(255,61,127,.12);color:var(--pink);   border:1px solid rgba(255,61,127,.3); border-radius:8px; padding:4px 14px; font-size:.82rem; margin:3px; display:inline-block; }

/* ── alert ───────────────────────────────────────────── */
.nx-alert {
  background: rgba(255,107,53,.1);
  border-left: 3px solid var(--orange);
  border-radius: 8px;
  padding: 10px 14px;
  color: #ffc49b;
  font-size: .88rem;
  margin: 6px 0;
}
.nx-tip {
  background: rgba(0,229,255,.08);
  border-left: 3px solid var(--cyan);
  border-radius: 8px;
  padding: 10px 14px;
  color: #a0eeff;
  font-size: .88rem;
  margin: 6px 0;
}

/* ── recording pill ──────────────────────────────────── */
.rec-pill {
  display:inline-flex; align-items:center; gap:8px;
  background:rgba(255,61,127,.12);
  border:1px solid rgba(255,61,127,.3);
  border-radius:99px; padding:6px 18px;
  color:var(--pink); font-weight:600; font-size:.88rem;
}
.rec-dot { width:10px;height:10px;background:var(--pink);border-radius:50%;animation:blink .8s infinite; }

/* ── score bar ───────────────────────────────────────── */
.nx-bar-wrap  { background:var(--bg3); border-radius:99px; height:10px; width:100%; overflow:hidden; }
.nx-bar-fill  { height:10px; border-radius:99px; transition:width .5s; }
.nx-bar-row   { margin-bottom:12px; }
.nx-bar-label { display:flex; justify-content:space-between; margin-bottom:5px; font-size:.85rem; }

/* ── login page ──────────────────────────────────────── */
.login-wrap {
  max-width: 440px;
  margin: 6vh auto 0;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 48px 44px;
  text-align: center;
  box-shadow: 0 0 80px rgba(0,229,255,.06);
}
.login-logo { font-family:var(--font-h); font-size:2.8rem; font-weight:800; letter-spacing:-.03em; }
.login-sub  { color:var(--muted); font-size:.9rem; margin-bottom:32px; }
.login-btn  {
  background: linear-gradient(135deg,var(--cyan),var(--violet)) !important;
  color: #000 !important; font-weight: 800 !important;
}
.demo-tag {
  display:inline-block;
  background:rgba(168,85,247,.15);
  border:1px solid rgba(168,85,247,.3);
  color:var(--violet);
  border-radius:99px; padding:3px 14px;
  font-size:.75rem; letter-spacing:.06em;
  margin-bottom:24px;
}
/* ── sidebar brand ───────────────────────────────────── */
.nx-brand {
  font-family: var(--font-h);
  font-size: 1.3rem;
  font-weight: 800;
  letter-spacing: -.02em;
  background: linear-gradient(135deg,var(--cyan),var(--violet));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 4px;
}
.nx-brand-sub { font-size:.75rem; color:var(--muted); margin-bottom:16px; }

/* ── final score hero ────────────────────────────────── */
.nx-hero-score {
  text-align:center;
  padding:36px;
  background: radial-gradient(ellipse at 50% 0%, rgba(0,229,255,.12), transparent 70%),
              var(--surface);
  border: 1px solid var(--border);
  border-radius:20px;
  margin-bottom:24px;
}
.nx-hero-score .big { font-family:var(--font-h); font-size:4rem; font-weight:800; }

/* ── tabs ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { background:var(--bg2) !important; border-radius:12px !important; gap:4px !important; padding:4px !important; }
.stTabs [data-baseweb="tab"]      { border-radius:9px !important; font-family:var(--font-b) !important; color:var(--muted) !important; }
.stTabs [aria-selected="true"]    { background:var(--surface) !important; color:#fff !important; }

/* page transitions */
@keyframes fadein { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
.stVerticalBlock { animation: fadein .35s ease; }

/* scrollbar */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:99px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════

SAMPLE_QUESTIONS = {
    "Data Scientist":        [
        {"q":"Explain overfitting and how you prevent it.",          "type":"Technical",   "diff":"Medium", "star":[]},
        {"q":"Describe a time you cleaned a very messy dataset.",     "type":"Behavioural", "diff":"Medium", "star":["Situation","Task","Action","Result"]},
        {"q":"What metrics would you use for an imbalanced dataset?", "type":"Technical",   "diff":"Hard",   "star":[]},
        {"q":"Tell me about your biggest data science achievement.",  "type":"HR",          "diff":"Easy",   "star":["Situation","Result"]},
    ],
    "Software Engineer":     [
        {"q":"Explain the difference between TCP and UDP.",           "type":"Technical",   "diff":"Easy",   "star":[]},
        {"q":"Describe a challenging bug you debugged.",              "type":"Behavioural", "diff":"Medium", "star":["Situation","Task","Action","Result"]},
        {"q":"How would you design a URL shortener at scale?",        "type":"Technical",   "diff":"Hard",   "star":[]},
        {"q":"Why should we hire you over other candidates?",        "type":"HR",          "diff":"Easy",   "star":[]},
    ],
    "AI/ML Engineer":        [
        {"q":"What is the vanishing gradient problem?",              "type":"Technical",   "diff":"Hard",   "star":[]},
        {"q":"Describe an end-to-end ML project you built.",         "type":"Behavioural", "diff":"Medium", "star":["Situation","Task","Action","Result"]},
        {"q":"Explain transformer attention mechanisms.",             "type":"Technical",   "diff":"Hard",   "star":[]},
        {"q":"How do you stay updated with AI research?",            "type":"HR",          "diff":"Easy",   "star":[]},
    ],
    "DevOps Engineer":       [
        {"q":"What is the difference between CI and CD?",             "type":"Technical",   "diff":"Easy",   "star":[]},
        {"q":"Tell me about a production incident you handled.",      "type":"Behavioural", "diff":"Hard",   "star":["Situation","Task","Action","Result"]},
    ],
}

BADGE_COLORS = {"Technical":"#2563eb","Behavioural":"#7c3aed","HR":"#0891b2"}
DIFF_COLORS  = {"Easy":"#22c55e","Medium":"#eab308","Hard":"#ef4444"}
EMOJIS       = {5:"🌟",4:"✅",3:"⚠️",2:"😬",1:"❌"}
DISC_COLORS  = ["#00e5ff","#a855f7","#00ffa3","#ff6b35"]

def sc(v): # score color
    if v>=4.2: return "#00ffa3"
    if v>=3.4: return "#ffe566"
    if v>=2.5: return "#ff6b35"
    return "#ff3d7f"

def score_emoji(v):
    if v>=4.5: return "🌟"
    if v>=3.5: return "✅"
    if v>=2.5: return "⚠️"
    return "❌"

def render_bar(label, val, color=None, max_val=5.0):
    pct = int(val/max_val*100)
    col = color or sc(val)
    st.markdown(f"""
    <div class="nx-bar-row">
      <div class="nx-bar-label">
        <span>{label}</span>
        <span style="font-weight:700;color:{col}">{val:.2f}</span>
      </div>
      <div class="nx-bar-wrap">
        <div class="nx-bar-fill" style="width:{pct}%;background:{col}"></div>
      </div>
    </div>""", unsafe_allow_html=True)

def sim_score(): return round(random.uniform(2.8, 5.0), 2)


# ════════════════════════════════════════════════════════
#  SESSION DEFAULTS
# ════════════════════════════════════════════════════════
if "logged_in"      not in st.session_state: st.session_state.logged_in      = False
if "user_name"      not in st.session_state: st.session_state.user_name      = ""
if "page"           not in st.session_state: st.session_state.page           = "login"
if "interview_on"   not in st.session_state: st.session_state.interview_on   = False
if "q_index"        not in st.session_state: st.session_state.q_index        = 0
if "answers"        not in st.session_state: st.session_state.answers        = []
if "recording"      not in st.session_state: st.session_state.recording      = False
if "live_chunks"    not in st.session_state: st.session_state.live_chunks    = []
if "transcript"     not in st.session_state: st.session_state.transcript     = ""
if "last_result"    not in st.session_state: st.session_state.last_result    = None
if "role"           not in st.session_state: st.session_state.role           = "Software Engineer"
if "difficulty"     not in st.session_state: st.session_state.difficulty     = "Medium"
if "menu"           not in st.session_state: st.session_state.menu           = "Dashboard"


# ════════════════════════════════════════════════════════
#  LOGIN PAGE
# ════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    # hide sidebar on login
    st.markdown("<style>[data-testid='stSidebar']{display:none}</style>", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("""
        <div class="login-wrap">
          <div class="login-logo">
            Nex<span style="color:#00e5ff">Interview</span>
          </div>
          <div class="login-sub" style="margin-top:6px;margin-bottom:18px">
            AI-Powered Multimodal Interview Coach
          </div>
          <div class="demo-tag">✦ POWERED BY DEEPFACE · MEDIAPIPE · NLTK</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            uname = st.text_input("", placeholder="👤  Full Name", label_visibility="collapsed")
            email = st.text_input("", placeholder="✉️  Email Address", label_visibility="collapsed")
            pwd   = st.text_input("", placeholder="🔒  Password", type="password", label_visibility="collapsed")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            login = st.form_submit_button("Get Started →", use_container_width=True)

        if login:
            if uname.strip() and email.strip() and pwd:
                st.session_state.logged_in = True
                st.session_state.user_name = uname.strip()
                st.rerun()
            else:
                st.error("Please fill all fields.")

        st.markdown("""
        <div style="text-align:center;margin-top:16px;color:#5a7099;font-size:.8rem">
          🔐 Demo mode — no data stored
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════════
#  SIDEBAR  (post-login)
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f'<div class="nx-brand">NexInterview</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="nx-brand-sub">Welcome, {st.session_state.user_name} 👋</div>', unsafe_allow_html=True)

    st.markdown("---")
    menu = st.radio(
        "NAVIGATION",
        ["🏠  Dashboard", "🎙️  Interview", "📷  Live Monitor", "📊  Analytics", "📄  Report"],
        label_visibility="visible"
    )
    st.markdown("---")

    # quick stats in sidebar
    total_q = len(st.session_state.answers)
    avg_sc  = round(sum(a["final"] for a in st.session_state.answers)/total_q, 2) if total_q else 0
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;gap:8px">
      <div class="nx-card cyan" style="padding:14px 18px">
        <div style="font-family:var(--font-h);font-size:1.5rem;font-weight:800;color:#00e5ff">{total_q}</div>
        <div style="font-size:.75rem;color:#5a7099;text-transform:uppercase;letter-spacing:.05em">Questions Done</div>
      </div>
      <div class="nx-card violet" style="padding:14px 18px">
        <div style="font-family:var(--font-h);font-size:1.5rem;font-weight:800;color:#a855f7">{avg_sc if avg_sc else '—'}</div>
        <div style="font-size:.75rem;color:#5a7099;text-transform:uppercase;letter-spacing:.05em">Avg Score</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:.75rem;color:#5a7099;line-height:1.9">
      😊 Emotion · DeepFace<br>
      🧍 Posture · MediaPipe<br>
      🎙️ Voice · Librosa<br>
      🧠 NLP · STAR + TF-IDF<br>
      🎭 DISC Personality
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚪 Logout"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ════════════════════════════════════════════════════════
if menu == "🏠  Dashboard":
    st.markdown(f'<h1 style="margin-bottom:4px">Interview Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#5a7099;margin-bottom:28px">Your performance across all evaluation dimensions</p>', unsafe_allow_html=True)

    answers = st.session_state.answers
    if answers:
        e = round(sum(a["emotion"]     for a in answers)/len(answers),2)
        v = round(sum(a["voice"]       for a in answers)/len(answers),2)
        k = round(sum(a["knowledge"]   for a in answers)/len(answers),2)
        p = round(sum(a["posture"]      for a in answers)/len(answers),2)
        pe= round(sum(a["personality"] for a in answers)/len(answers),2)
    else:
        e, v, k, p, pe = 4.2, 3.8, 4.5, 3.9, 4.0

    metrics = [
        ("😊 Emotion",     e,  "+0.3", "cyan"),
        ("🎙️ Voice",       v,  "−0.2", "pink"),
        ("🧠 Knowledge",   k,  "+0.5", "green"),
        ("🧍 Posture",     p,  "+0.1", "orange"),
        ("🎭 Personality", pe, "+0.2", "violet"),
    ]
    cols = st.columns(5)
    colors_map = {"cyan":"#00e5ff","pink":"#ff3d7f","green":"#00ffa3","orange":"#ff6b35","violet":"#a855f7"}
    for col, (lbl, val, delta, accent) in zip(cols, metrics):
        sign_cls = "delta-pos" if delta.startswith("+") else "delta-neg"
        col.markdown(f"""
        <div class="nx-metric" style="border-top:3px solid {colors_map[accent]}">
          <div class="val" style="color:{colors_map[accent]}">{val}</div>
          <div class="lbl">{lbl}</div>
          <div class="delta {sign_cls}">{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3,2])

    with col_l:
        st.markdown("### 📈 Score Trend")
        if answers:
            trend_data = {"Q": [f"Q{i+1}" for i in range(len(answers))],
                          "Final": [a["final"] for a in answers],
                          "Knowledge": [a["knowledge"] for a in answers],
                          "Emotion":   [a["emotion"]   for a in answers]}
        else:
            trend_data = {"Q":["Q1","Q2","Q3","Q4","Q5"],
                          "Final":[3.2,3.8,4.0,4.3,4.5],
                          "Knowledge":[3.0,3.5,4.2,4.4,4.6],
                          "Emotion":[3.5,4.0,3.8,4.2,4.4]}
        fig = go.Figure()
        for key, color in [("Final","#00e5ff"),("Knowledge","#a855f7"),("Emotion","#ff3d7f")]:
            fig.add_trace(go.Scatter(
                x=trend_data["Q"], y=trend_data[key], name=key,
                mode="lines+markers", line=dict(color=color,width=2.5),
                marker=dict(size=8,symbol="circle"),
                fill="tozeroy" if key=="Final" else None,
                fillcolor="rgba(0,229,255,0.05)" if key=="Final" else None
            ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#111d35", plot_bgcolor="#111d35",
            yaxis=dict(range=[0,5.3],gridcolor="#1e2d50",title="Score"),
            xaxis=dict(gridcolor="#1e2d50"),
            legend=dict(bgcolor="#111d35",bordercolor="#1e2d50"),
            margin=dict(l=10,r=10,t=10,b=20), height=280,
            font=dict(family="DM Sans")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("### 🕸️ Radar Profile")
        cats = ["Emotion","Voice","Knowledge","Posture","Personality"]
        vals = [e, v, k, p, pe]
        fig2 = go.Figure(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]],
            fill="toself", line=dict(color="#00e5ff",width=2),
            fillcolor="rgba(0,229,255,0.12)"
        ))
        fig2.update_layout(
            polar=dict(
                bgcolor="#0d1526",
                radialaxis=dict(visible=True,range=[0,5],gridcolor="#1e2d50",color="#5a7099",tickfont=dict(size=9)),
                angularaxis=dict(color="#dde6f5",gridcolor="#1e2d50")
            ),
            paper_bgcolor="#111d35", showlegend=False,
            margin=dict(l=20,r=20,t=20,b=20), height=280
        )
        st.plotly_chart(fig2, use_container_width=True)

    # DISC distribution
    st.markdown("### 🎭 DISC Personality Distribution")
    disc_vals = [random.randint(8,25) for _ in range(4)] if not answers else [12,18,10,15]
    disc_names= ["Dominance","Influence","Steadiness","Conscientiousness"]
    fig3 = px.pie(values=disc_vals, names=disc_names,
                  color_discrete_sequence=DISC_COLORS, hole=0.5)
    fig3.update_layout(paper_bgcolor="#111d35", font=dict(color="#dde6f5",family="DM Sans"),
                       legend=dict(bgcolor="#111d35"), margin=dict(l=0,r=0,t=0,b=0), height=240)
    fig3.update_traces(textfont_size=12)
    st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════
#  PAGE: INTERVIEW
# ════════════════════════════════════════════════════════
elif menu == "🎙️  Interview":
    st.markdown('<h1>Mock Interview Session</h1>', unsafe_allow_html=True)

    if not st.session_state.interview_on:
        # ── setup ──────────────────────────────────────
        st.markdown('<p style="color:#5a7099">Configure your session and click Begin.</p>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="nx-card cyan">', unsafe_allow_html=True)
            role = st.selectbox("💼 Target Role",
                                list(SAMPLE_QUESTIONS.keys()),
                                index=list(SAMPLE_QUESTIONS.keys()).index(st.session_state.role)
                                if st.session_state.role in SAMPLE_QUESTIONS else 0)
            st.session_state.role = role
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            diff = st.select_slider("🎯 Difficulty", ["Easy","Medium","Hard"],
                                    value=st.session_state.difficulty)
            st.session_state.difficulty = diff
        with c3:
            q_type = st.radio("Question Type", ["All","Technical","Behavioural","HR"])

        st.markdown("<br>", unsafe_allow_html=True)

        # feature cards
        feat_cols = st.columns(4)
        feats = [
            ("😊","Emotion AI","DeepFace 8-emotion detection with nervousness scoring","cyan"),
            ("🎙️","Voice Analysis","WPM, pitch, pauses & MFCC via Librosa","violet"),
            ("🧠","STAR NLP","STAR framework + TF-IDF keyword matching","green"),
            ("🧍","Posture Guard","MediaPipe 33-point skeleton posture analysis","orange"),
        ]
        for col, (icon, title, desc, accent) in zip(feat_cols, feats):
            col.markdown(f"""
            <div class="nx-card {accent}" style="text-align:center;padding:18px">
              <div style="font-size:2rem;margin-bottom:8px">{icon}</div>
              <div style="font-family:var(--font-h);font-weight:700;color:#fff;margin-bottom:6px">{title}</div>
              <div style="font-size:.78rem;color:#5a7099;line-height:1.5">{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀  Begin Interview Session", use_container_width=True):
            pool = SAMPLE_QUESTIONS.get(role, SAMPLE_QUESTIONS["Software Engineer"])
            if q_type != "All":
                pool = [q for q in pool if q["type"]==q_type] or pool
            random.shuffle(pool)
            st.session_state.interview_on = True
            st.session_state.q_index = 0
            st.session_state.answers = []
            st.session_state.questions_pool = pool
            st.session_state.last_result = None
            st.rerun()

    else:
        # ── active interview ──────────────────────────────
        pool = st.session_state.get("questions_pool", SAMPLE_QUESTIONS["Software Engineer"])
        idx  = st.session_state.q_index
        tot  = len(pool)

        if idx >= tot:
            st.success("🎉 Interview complete! View your **Final Report**.")
            if st.button("🔄 New Session"):
                st.session_state.interview_on = False
                st.session_state.q_index = 0
                st.rerun()
            st.stop()

        q = pool[idx]

        # progress
        pct = idx/tot
        st.progress(pct, text=f"Question {idx+1} of {tot}  ·  {st.session_state.difficulty}  ·  {q['type']}")

        badge_c = BADGE_COLORS.get(q["type"],"#475569")
        diff_c  = DIFF_COLORS.get(q["diff"],"#64748b")
        st.markdown(f"""
        <div class="nx-question">
          <span class="nx-q-badge" style="background:{badge_c};color:#fff">{q['type']}</span>
          <span class="nx-q-badge" style="background:{diff_c};color:#000;font-size:.7rem">{q['diff']}</span>
          {q['q']}
        </div>""", unsafe_allow_html=True)

        if q.get("star"):
            st.markdown(
                f'<div class="nx-tip">💡 <b>STAR Tip:</b> Include — {" → ".join(q["star"])}</div>',
                unsafe_allow_html=True
            )

        # recording + text columns
        col_rec, col_txt = st.columns([1,2])

        with col_rec:
            st.markdown("#### 🎙️ Voice Answer")
            is_rec = st.session_state.recording

            if not is_rec:
                if st.button("🔴  Start Recording"):
                    st.session_state.recording    = True
                    st.session_state.live_chunks  = []
                    st.session_state.transcript   = ""
                    st.rerun()
            else:
                if st.button("⏹️  Stop Recording"):
                    st.session_state.recording = False
                    fake = ["I worked on", "a machine learning project", "where I implemented", "a neural network to predict"]
                    st.session_state.transcript = " ".join(st.session_state.live_chunks) or \
                                                  "I worked on a challenging project where I implemented several optimisation techniques and achieved a 30% performance improvement."
                    st.rerun()

            if is_rec:
                # simulate chunk arrival
                if len(st.session_state.live_chunks) < 5:
                    fake_chunks = [
                        "In my previous role",
                        "I worked on a project",
                        "involving large-scale data",
                        "where the main challenge was",
                        "handling real-time inference"
                    ]
                    chunk = fake_chunks[len(st.session_state.live_chunks) % len(fake_chunks)]
                    st.session_state.live_chunks.append(chunk)

                chunks_html = "".join(f'<span class="nx-chunk">{c}</span>'
                                      for c in st.session_state.live_chunks)
                st.markdown(f"""
                <div style="margin-bottom:8px">
                  <span class="rec-pill"><span class="rec-dot"></span>Recording…</span>
                </div>
                <div class="nx-transcript">{chunks_html}<span class="nx-cursor"></span></div>
                """, unsafe_allow_html=True)
                time.sleep(0.8)
                st.rerun()
            else:
                st.markdown('<div style="color:#5a7099;font-size:.82rem;margin-top:8px">Click to record. Each 5-second chunk appears as a badge.</div>',
                            unsafe_allow_html=True)

        with col_txt:
            st.markdown("#### ✏️ Type / Edit Answer")
            text = st.text_area(
                "Answer",
                value=st.session_state.transcript,
                height=160,
                label_visibility="collapsed",
                placeholder="Type your answer here, or use voice recording above…"
            )
            st.session_state.transcript = text

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✅  Submit & Analyse Answer", use_container_width=True):
            answer = st.session_state.transcript.strip()
            if not answer:
                st.warning("Please provide an answer first.")
                st.stop()

            with st.spinner("Running multimodal analysis…"):
                time.sleep(1.2)  # simulate processing
                # SIMULATED SCORES (replace with real engine)
                star = {k: bool(k.lower() in answer.lower() or random.random()>.4)
                        for k in (q["star"] or ["Situation","Task","Action","Result"])}
                result = {
                    "question":    q,
                    "transcript":  answer,
                    "emotion":     sim_score(),
                    "voice":       sim_score(),
                    "knowledge":   sim_score(),
                    "posture":     sim_score(),
                    "personality": sim_score(),
                    "final":       0,
                    "star":        star,
                    "wpm":         random.randint(110,170),
                    "pitch":       random.randint(110,200),
                    "feedback":    "Good use of technical terminology. Try adding a concrete outcome next time." if random.random()>.5 else "Excellent STAR structure! Quantify your results for more impact.",
                }
                result["final"] = round(
                    result["emotion"]*0.20 + result["voice"]*0.20 +
                    result["knowledge"]*0.30 + result["posture"]*0.15 +
                    result["personality"]*0.15, 2)

                st.session_state.answers.append(result)
                st.session_state.last_result = result
                st.session_state.q_index    += 1
                st.session_state.transcript  = ""
                st.session_state.live_chunks = []

            st.rerun()

        # last result display
        last = st.session_state.last_result
        if last and st.session_state.q_index > 0:
            st.markdown("---")
            st.markdown("#### 📊 Last Answer Analysis")
            c = last
            m_cols = st.columns(5)
            for col, (lbl, key, clr) in zip(m_cols,[
                ("😊 Emotion","emotion","#00e5ff"),("🎙️ Voice","voice","#ff3d7f"),
                ("🧠 Knowledge","knowledge","#a855f7"),("🧍 Posture","posture","#ff6b35"),
                ("🎭 Personality","personality","#ffe566")
            ]):
                col.markdown(f"""
                <div class="nx-metric">
                  <div class="val" style="color:{clr}">{c[key]}</div>
                  <div class="lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="text-align:center;margin:16px 0;padding:14px;
                        background:var(--surface);border-radius:12px;
                        border:2px solid {sc(c['final'])}">
              <span style="font-family:var(--font-h);font-size:1.6rem;font-weight:800;color:{sc(c['final'])}">
                {score_emoji(c['final'])}  Final: {c['final']} / 5.0
              </span>
            </div>""", unsafe_allow_html=True)

            if last.get("feedback"):
                st.markdown(f'<div class="nx-tip">💬 {last["feedback"]}</div>', unsafe_allow_html=True)

            # STAR badges
            star_html = "".join(
                f'<span class="star-hit">✓ {k}</span>' if v else f'<span class="star-miss">✗ {k}</span>'
                for k,v in last.get("star",{}).items()
            )
            st.markdown(f"<div>**STAR Framework:** {star_html}</div>", unsafe_allow_html=True)

            # voice stats
            st.markdown(f"""
            <div style="display:flex;gap:16px;margin-top:10px;flex-wrap:wrap">
              <div class="nx-card cyan" style="padding:10px 20px;flex:1">
                <div style="font-size:1.3rem;font-weight:800;color:#00e5ff">{last['wpm']}</div>
                <div style="font-size:.72rem;color:#5a7099">WPM</div>
              </div>
              <div class="nx-card violet" style="padding:10px 20px;flex:1">
                <div style="font-size:1.3rem;font-weight:800;color:#a855f7">{last['pitch']} Hz</div>
                <div style="font-size:.72rem;color:#5a7099">Avg Pitch</div>
              </div>
              <div class="nx-card green" style="padding:10px 20px;flex:1">
                <div style="font-size:1.3rem;font-weight:800;color:#00ffa3">{last['knowledge']}/5</div>
                <div style="font-size:.72rem;color:#5a7099">Knowledge</div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  PAGE: LIVE MONITOR
# ════════════════════════════════════════════════════════
elif menu == "📷  Live Monitor":
    st.markdown('<h1>Live Monitoring Panel</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#5a7099">Real-time emotion & posture detection during your interview.</p>', unsafe_allow_html=True)

    run = st.toggle("▶  Enable Webcam", value=False)

    col_cam, col_data = st.columns([3, 2])
    frame_slot  = col_cam.empty()
    emotion_slot= col_data.empty()
    posture_slot= col_data.empty()

    if run:
        import cv2
        cap = cv2.VideoCapture(0)
        stop = st.button("⏹  Stop")
        fc   = 0
        while not stop:
            ret, frame = cap.read()
            if not ret:
                col_cam.error("Webcam unavailable — check permissions.")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Simulated overlays
            cv2.putText(frame_rgb, "Emotion: Confident  Nerv:0.12",
                        (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,229,255), 2)
            cv2.putText(frame_rgb, "Posture: 4.2/5",
                        (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,163), 2)
            frame_slot.image(frame_rgb, channels="RGB", use_container_width=True)

            if fc % 15 == 0:
                emos = {"Happy":random.randint(25,60),"Neutral":random.randint(20,40),
                        "Sad":random.randint(0,10),"Fear":random.randint(0,8),
                        "Angry":random.randint(0,6)}
                emo_html = "".join(f"""
                <div class="nx-bar-row">
                  <div class="nx-bar-label"><span>{k}</span><span style="color:#00e5ff">{v}</span></div>
                  <div class="nx-bar-wrap"><div class="nx-bar-fill" style="width:{v}%;background:#00e5ff"></div></div>
                </div>""" for k,v in emos.items())

                with emotion_slot.container():
                    st.markdown("**😊 Emotion Breakdown**")
                    st.markdown(emo_html, unsafe_allow_html=True)

                with posture_slot.container():
                    st.markdown("**🧍 Posture Analysis**")
                    p_sc = round(random.uniform(3.5,5.0),2)
                    render_bar("Overall Posture", p_sc, "#00ffa3")
                    render_bar("Shoulder Alignment", round(random.uniform(3.8,5.0),2), "#a855f7")
                    render_bar("Head Position", round(random.uniform(3.5,5.0),2), "#ffe566")
                    if random.random() > 0.7:
                        st.markdown('<div class="nx-alert">⚠️ Slight forward lean detected — sit upright.</div>',
                                    unsafe_allow_html=True)
            fc += 1
            time.sleep(0.04)
        cap.release()
    else:
        col_cam.markdown("""
        <div style="height:300px;background:var(--surface);border:1px solid var(--border);
                    border-radius:14px;display:flex;align-items:center;justify-content:center;
                    color:#5a7099;font-size:1rem">
          📷  Enable webcam to start monitoring
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  PAGE: ANALYTICS
# ════════════════════════════════════════════════════════
elif menu == "📊  Analytics":
    st.markdown('<h1>Deep Analytics</h1>', unsafe_allow_html=True)

    answers = st.session_state.answers
    if not answers:
        st.info("No interview data yet. Complete at least one question to see analytics.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📈 Trends", "🎭 DISC Deep-Dive", "🗂️ Per-Question"])

    with tab1:
        df = pd.DataFrame(answers)
        df["Q"] = [f"Q{i+1}" for i in range(len(df))]
        fig = go.Figure()
        for col, color in [("emotion","#00e5ff"),("voice","#ff3d7f"),
                           ("knowledge","#a855f7"),("posture","#ff6b35"),
                           ("final","#ffe566")]:
            fig.add_trace(go.Scatter(x=df["Q"], y=df[col], name=col.capitalize(),
                                     mode="lines+markers", line=dict(color=color,width=2),
                                     marker=dict(size=7)))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#111d35", plot_bgcolor="#111d35",
                          yaxis=dict(range=[0,5.3]), height=320,
                          legend=dict(bgcolor="#111d35"), margin=dict(l=10,r=10,t=10,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        disc_vals  = [14, 21, 11, 18]
        disc_names = ["Dominance","Influence","Steadiness","Conscientiousness"]
        disc_desc  = {
            "Dominance":      "Results-oriented, decisive, direct. Natural leader.",
            "Influence":      "People-oriented, enthusiastic, motivating communicator.",
            "Steadiness":     "Reliable, patient, consistent. Great team stabiliser.",
            "Conscientiousness":"Detail-focused, accurate, systematic analytical mind.",
        }
        c_l, c_r = st.columns([1,2])
        with c_l:
            fig_d = px.pie(values=disc_vals, names=disc_names,
                           color_discrete_sequence=DISC_COLORS, hole=0.55)
            fig_d.update_layout(paper_bgcolor="#111d35", font=dict(color="#dde6f5"),
                                 legend=dict(bgcolor="#111d35"), margin=dict(l=0,r=0,t=0,b=0), height=260)
            st.plotly_chart(fig_d, use_container_width=True)
        with c_r:
            dom = disc_names[disc_vals.index(max(disc_vals))]
            st.markdown(f"""<div class="nx-card cyan">
              <div style="font-family:var(--font-h);font-size:1.1rem;font-weight:700">
                Dominant Trait: {dom}
              </div>
              <div style="color:#5a7099;font-size:.88rem;margin-top:8px">{disc_desc[dom]}</div>
            </div>""", unsafe_allow_html=True)
            for n, v, c in zip(disc_names, disc_vals, DISC_COLORS):
                total = sum(disc_vals)
                render_bar(f"{n} ({round(v/total*100)}%)", v/total*5, c)

    with tab3:
        for i, a in enumerate(answers):
            with st.expander(f"Q{i+1}  {score_emoji(a['final'])}  {a['question']['q'][:60]}…  — {a['final']}/5"):
                cc1, cc2 = st.columns(2)
                with cc1:
                    render_bar("Emotion",     a["emotion"])
                    render_bar("Voice",        a["voice"])
                    render_bar("Knowledge",    a["knowledge"])
                with cc2:
                    render_bar("Posture",      a["posture"])
                    render_bar("Personality",  a["personality"])
                st.markdown(f"**Transcript:** {a['transcript'][:300]}…")
                if a.get("feedback"):
                    st.markdown(f'<div class="nx-tip">💬 {a["feedback"]}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  PAGE: FINAL REPORT
# ════════════════════════════════════════════════════════
elif menu == "📄  Report":
    st.markdown('<h1>Final Interview Report</h1>', unsafe_allow_html=True)

    answers = st.session_state.answers
    if not answers:
        st.info("Complete your interview session first.")
        st.stop()

    e  = round(sum(a["emotion"]     for a in answers)/len(answers),2)
    v  = round(sum(a["voice"]       for a in answers)/len(answers),2)
    k  = round(sum(a["knowledge"]   for a in answers)/len(answers),2)
    p  = round(sum(a["posture"]      for a in answers)/len(answers),2)
    pe = round(sum(a["personality"] for a in answers)/len(answers),2)
    fs = round(e*.20+v*.20+k*.30+p*.15+pe*.15,2)

    # Hero score
    st.markdown(f"""
    <div class="nx-hero-score">
      <div style="font-size:2.5rem;margin-bottom:8px">{score_emoji(fs)}</div>
      <div class="big" style="color:{sc(fs)}">{fs} <span style="font-size:1.8rem;color:#5a7099">/ 5.0</span></div>
      <div style="color:#5a7099;margin-top:8px">Overall Interview Score</div>
      <div style="color:#2d3f60;font-size:.82rem;margin-top:4px">
        {st.session_state.user_name}  ·  {st.session_state.role}  ·  {len(answers)} questions  ·  {datetime.now().strftime("%d %b %Y")}
      </div>
    </div>""", unsafe_allow_html=True)

    # Dimension bars
    st.markdown("### 📊 Dimension Breakdown")
    for lbl, val, clr in [
        ("😊 Emotion (weight 20%)",     e,  "#00e5ff"),
        ("🎙️ Voice (weight 20%)",       v,  "#ff3d7f"),
        ("🧠 Knowledge (weight 30%)",   k,  "#a855f7"),
        ("🧍 Posture (weight 15%)",     p,  "#ff6b35"),
        ("🎭 Personality (weight 15%)", pe, "#ffe566"),
    ]:
        render_bar(lbl, val, clr)

    # Per-question table
    st.markdown("### 🗂️ Question Summary")
    rows = []
    for i, a in enumerate(answers):
        rows.append({
            "Q":  f"Q{i+1}",
            "Question": a["question"]["q"][:50]+"…",
            "Type": a["question"]["type"],
            "Emotion": a["emotion"], "Voice": a["voice"],
            "Knowledge": a["knowledge"], "Final": a["final"]
        })
    df_table = pd.DataFrame(rows).set_index("Q")
    st.dataframe(df_table, use_container_width=True)

    # Recommendations
    st.markdown("### 📚 Skill Gap & Recommendations")
    weak = [(lbl, val) for lbl, val in [
        ("Emotion Control",      e),("Voice Delivery",    v),
        ("Technical Knowledge",  k),("Body Language",     p),
        ("Personality Balance",  pe)
    ] if val < 3.5]

    recs = {
        "Emotion Control":     "Coursera — Emotional Intelligence at Work",
        "Voice Delivery":      "YouTube — TED Public Speaking Masterclass",
        "Technical Knowledge": "LeetCode / HackerRank daily practice",
        "Body Language":       "Udemy — Body Language & Nonverbal Communication",
        "Personality Balance": "LinkedIn Learning — DISC Personality Profile",
    }
    if weak:
        for lbl, val in weak:
            st.markdown(f"""
            <div class="nx-alert">
              ⚠️ <b>{lbl}</b> scored {val:.2f}/5 —
              Recommended: <i>{recs.get(lbl,'Practice regularly')}</i>
            </div>""", unsafe_allow_html=True)
    else:
        st.success("🌟 Outstanding performance across all dimensions!")

    # Radar
    cats = ["Emotion","Voice","Knowledge","Posture","Personality"]
    vals = [e, v, k, p, pe]
    fig_r = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself",
        line=dict(color="#00e5ff",width=2), fillcolor="rgba(0,229,255,0.1)"
    ))
    fig_r.update_layout(
        polar=dict(bgcolor="#0d1526",
                   radialaxis=dict(visible=True,range=[0,5],gridcolor="#1e2d50",color="#5a7099"),
                   angularaxis=dict(color="#dde6f5",gridcolor="#1e2d50")),
        paper_bgcolor="#111d35", showlegend=False,
        margin=dict(l=20,r=20,t=20,b=20), height=300
    )
    st.plotly_chart(fig_r, use_container_width=True)

    # Download
    import json
    report_data = {
        "candidate": st.session_state.user_name,
        "role": st.session_state.role,
        "date": datetime.now().isoformat(),
        "scores": {"emotion":e,"voice":v,"knowledge":k,"posture":p,"personality":pe,"final":fs},
        "answers": [{
            "question": a["question"]["q"],
            "transcript": a["transcript"],
            "scores": {k2:v2 for k2,v2 in a.items() if isinstance(v2,float)}
        } for a in answers]
    }
    st.download_button(
        label="⬇️  Download Full Report (JSON)",
        data=json.dumps(report_data, indent=2, default=str),
        file_name=f"interview_report_{st.session_state.user_name.replace(' ','_')}.json",
        mime="application/json",
        use_container_width=True
    )
    if st.button("🔄  Start New Interview", use_container_width=True):
        st.session_state.interview_on = False
        st.session_state.answers      = []
        st.session_state.q_index      = 0
        st.rerun()
