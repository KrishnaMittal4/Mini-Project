"""
avatar_interviewer.py — Aura AI | AI Interviewer Avatar (v3.0)
===============================================================
v3.0 changes:
  • Avatar face replaced with Holographic Orb (futuristic glowing sphere)
  • Orb features: latitude rings, equatorial orbit band, animated pulse glow
  • Eye-dot attention: small eyes on orb surface follow + blink naturally
  • Mouth arc on orb surface lip-syncs during speech
  • Orbit ring rotates continuously; glow pulses on speak
  • All other features preserved: TTS, speak bars, voice selector, tips

Usage:
    from avatar_interviewer import render_avatar_interviewer
    render_avatar_interviewer(
        question_text = "Tell me about yourself.",
        question_type = "Behavioral",
        auto_speak    = True,
        height        = 480,
    )
"""

from __future__ import annotations
import json
import streamlit.components.v1 as components


def render_avatar_interviewer(
    question_text: str  = "",
    question_type: str  = "Technical",
    q_number: int       = 1,
    total_qs: int       = 5,
    auto_speak: bool    = True,
    height: int         = 480,
) -> None:
    """
    Renders the AI interviewer avatar as a compact portrait card.
    Avatar is a Holographic Orb — glowing sphere with latitude rings & orbit band.
    No question text is shown inside — question is only spoken aloud via TTS.
    """
    q_text_js = json.dumps(question_text)
    auto_js   = "true" if auto_speak else "false"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:transparent;overflow:hidden;font-family:'Inter',sans-serif;}}

  .av-root{{
    background:linear-gradient(170deg,#060e1e 0%,#050b18 100%);
    border:1px solid rgba(0,212,255,0.16);
    border-radius:14px;
    width:100%;
    height:{height}px;
    display:flex;
    flex-direction:column;
    align-items:center;
    padding:20px 14px 16px;
    position:relative;
    overflow:hidden;
  }}

  .av-root::before{{
    content:'';position:absolute;top:-60px;left:-60px;
    width:220px;height:220px;
    background:radial-gradient(circle,rgba(0,255,136,0.06) 0%,transparent 68%);
    pointer-events:none;
  }}
  .av-root::after{{
    content:'';position:absolute;inset:0;
    background:repeating-linear-gradient(
      0deg,transparent,transparent 2px,
      rgba(0,212,255,0.014) 2px,rgba(0,212,255,0.014) 4px
    );
    pointer-events:none;border-radius:14px;
  }}

  /* FUTURISTIC badge */
  .av-badge-fut{{
    position:absolute;top:12px;right:12px;
    font-family:'Share Tech Mono',monospace;font-size:7px;letter-spacing:2px;
    color:rgba(0,212,255,0.55);border:1px solid rgba(0,212,255,0.18);
    padding:2px 7px;border-radius:3px;z-index:10;
  }}

  .av-label{{
    font-family:'Share Tech Mono',monospace;font-size:8px;
    color:rgba(0,212,255,0.4);letter-spacing:3px;text-transform:uppercase;
    margin-bottom:14px;z-index:1;
  }}

  /* ── ORB WRAPPER ── */
  .av-wrapper{{
    position:relative;width:170px;height:170px;
    margin-bottom:12px;flex-shrink:0;z-index:1;
  }}

  /* outer decorative rings (unchanged) */
  .av-ring{{
    position:absolute;inset:-7px;border-radius:50%;
    border:1.5px solid transparent;
    background:conic-gradient(from 0deg,transparent 0%,#00ff88 25%,#00d4ff 50%,transparent 75%) border-box;
    -webkit-mask:linear-gradient(#fff 0 0) padding-box,linear-gradient(#fff 0 0);
    -webkit-mask-composite:destination-out;mask-composite:exclude;
    animation:rspin 4s linear infinite;opacity:0.5;
  }}
  .av-ring2{{
    position:absolute;inset:-14px;border-radius:50%;
    border:1px solid rgba(0,212,255,0.09);
    animation:rspin 9s linear infinite reverse;
  }}
  @keyframes rspin{{from{{transform:rotate(0deg)}}to{{transform:rotate(360deg)}}}}

  .av-face{{
    width:170px;height:170px;border-radius:50%;overflow:visible;
    background:transparent;border:none;
  }}

  /* speak bars */
  .speak-bars{{
    display:flex;align-items:flex-end;gap:2px;height:20px;
    margin-bottom:10px;z-index:1;
  }}
  .speak-bars .b{{width:3px;background:#00ff88;border-radius:2px;height:3px;}}
  .speaking .b:nth-child(1){{animation:sb1 0.5s ease-in-out infinite;}}
  .speaking .b:nth-child(2){{animation:sb2 0.4s ease-in-out infinite 0.08s;}}
  .speaking .b:nth-child(3){{animation:sb3 0.6s ease-in-out infinite 0.04s;}}
  .speaking .b:nth-child(4){{animation:sb4 0.45s ease-in-out infinite 0.12s;}}
  .speaking .b:nth-child(5){{animation:sb5 0.55s ease-in-out infinite 0.06s;}}
  @keyframes sb1{{0%,100%{{height:3px}}50%{{height:14px}}}}
  @keyframes sb2{{0%,100%{{height:3px}}50%{{height:20px}}}}
  @keyframes sb3{{0%,100%{{height:3px}}50%{{height:10px}}}}
  @keyframes sb4{{0%,100%{{height:3px}}50%{{height:16px}}}}
  @keyframes sb5{{0%,100%{{height:3px}}50%{{height:8px}}}}

  .status-badge{{
    display:inline-flex;align-items:center;gap:5px;
    padding:4px 11px;border-radius:20px;z-index:1;
    font-family:'Share Tech Mono',monospace;font-size:9px;letter-spacing:1px;
    transition:all 0.3s;margin-bottom:16px;
  }}
  .status-badge.idle{{background:rgba(0,255,136,0.07);border:1px solid rgba(0,255,136,0.2);color:#00ff88;}}
  .status-badge.speaking{{background:rgba(0,212,255,0.09);border:1px solid rgba(0,212,255,0.28);color:#00d4ff;}}
  .status-badge.listening{{background:rgba(138,43,226,0.09);border:1px solid rgba(138,43,226,0.25);color:#bf7fff;}}
  .s-dot{{width:5px;height:5px;border-radius:50%;background:currentColor;
    animation:pdot 1.5s ease-in-out infinite;}}
  @keyframes pdot{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:0.4;transform:scale(0.6)}}}}

  .av-divider{{width:100%;height:1px;background:rgba(0,212,255,0.07);margin-bottom:14px;z-index:1;}}

  .vc-wrap{{width:100%;z-index:1;}}
  .vc-label{{
    font-family:'Share Tech Mono',monospace;font-size:8px;
    color:rgba(0,212,255,0.35);letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;
  }}
  .vc-select{{
    width:100%;background:rgba(0,212,255,0.04);
    border:1px solid rgba(0,212,255,0.1);color:#88bcd6;
    font-family:'Share Tech Mono',monospace;font-size:10px;
    padding:5px 7px;border-radius:6px;outline:none;cursor:pointer;
  }}
  .vc-select option{{background:#0a1628;}}

  .rate-row{{display:flex;align-items:center;gap:7px;margin-top:8px;}}
  .rate-lbl{{font-family:'Share Tech Mono',monospace;font-size:8px;color:rgba(0,212,255,0.35);white-space:nowrap;}}
  .rate-slider{{
    flex:1;-webkit-appearance:none;height:3px;
    background:rgba(0,212,255,0.12);border-radius:2px;outline:none;
  }}
  .rate-slider::-webkit-slider-thumb{{
    -webkit-appearance:none;width:11px;height:11px;
    border-radius:50%;background:#00d4ff;cursor:pointer;
  }}
  .rate-val{{font-family:'Share Tech Mono',monospace;font-size:9px;color:#00d4ff;min-width:28px;text-align:right;}}

  .av-btn-row{{display:flex;gap:7px;margin-top:12px;width:100%;z-index:1;}}
  .av-btn{{
    flex:1;padding:7px 8px;border-radius:6px;
    font-family:'Share Tech Mono',monospace;font-size:8px;letter-spacing:1px;
    cursor:pointer;border:1px solid;transition:all 0.2s;
    text-align:center;text-transform:uppercase;
  }}
  .av-btn-speak{{background:rgba(0,255,136,0.05);border-color:rgba(0,255,136,0.2);color:#00ff88;}}
  .av-btn-speak:hover{{background:rgba(0,255,136,0.1);border-color:rgba(0,255,136,0.4);}}
  .av-btn-stop{{background:rgba(255,51,102,0.05);border-color:rgba(255,51,102,0.16);color:#ff3366;display:none;}}
  .av-btn-stop.vis{{display:block;flex:0 0 auto;padding:7px 8px;}}
  .av-btn-stop:hover{{background:rgba(255,51,102,0.1);}}

  .tips{{display:flex;gap:5px;flex-wrap:wrap;margin-top:10px;width:100%;z-index:1;}}
  .tip{{
    font-family:'Share Tech Mono',monospace;font-size:8px;padding:3px 7px;
    border-radius:20px;background:rgba(138,43,226,0.07);
    border:1px solid rgba(138,43,226,0.15);color:rgba(191,127,255,0.7);
    cursor:pointer;transition:all 0.2s;white-space:nowrap;
  }}
  .tip:hover{{background:rgba(138,43,226,0.14);color:#bf7fff;}}

  /* ── ORB SVG ANIMATIONS ── */
  @keyframes orbPulse{{
    0%,100%{{opacity:0.55;r:76;}}
    50%{{opacity:0.85;r:80;}}
  }}
  @keyframes orbPulseSpeak{{
    0%,100%{{opacity:0.7;r:76;}}
    50%{{opacity:1.0;r:82;}}
  }}
  @keyframes orbitSpin{{
    from{{transform:rotateX(75deg) rotateZ(0deg);}}
    to{{transform:rotateX(75deg) rotateZ(360deg);}}
  }}
  @keyframes glowPulse{{
    0%,100%{{opacity:0.4;}}
    50%{{opacity:0.9;}}
  }}
  @keyframes scanLine{{
    0%{{transform:translateY(-80px);opacity:0;}}
    10%{{opacity:0.6;}}
    90%{{opacity:0.6;}}
    100%{{transform:translateY(80px);opacity:0;}}
  }}
</style>
</head>
<body>

<div class="av-root">

  <div class="av-badge-fut">FUTURISTIC</div>
  <div class="av-label">AI Interviewer</div>

  <div class="av-wrapper">
    <div class="av-ring2"></div>
    <div class="av-ring"></div>

    <!-- ══ HOLOGRAPHIC ORB SVG ══ -->
    <svg id="orbSvg" width="170" height="170" viewBox="-85 -85 170 170"
         xmlns="http://www.w3.org/2000/svg" style="border-radius:50%;overflow:visible;">
      <defs>
        <!-- Core orb radial gradient -->
        <radialGradient id="orbCore" cx="38%" cy="32%" r="62%">
          <stop offset="0%"   stop-color="#5af0ff" stop-opacity="0.95"/>
          <stop offset="30%"  stop-color="#00cfff" stop-opacity="0.85"/>
          <stop offset="65%"  stop-color="#0077cc" stop-opacity="0.7"/>
          <stop offset="100%" stop-color="#001a44" stop-opacity="0.95"/>
        </radialGradient>
        <!-- Atmosphere glow overlay -->
        <radialGradient id="orbAtmo" cx="50%" cy="50%" r="50%">
          <stop offset="60%"  stop-color="transparent"/>
          <stop offset="85%"  stop-color="rgba(0,200,255,0.18)"/>
          <stop offset="100%" stop-color="rgba(0,180,255,0.42)"/>
        </radialGradient>
        <!-- Top highlight -->
        <radialGradient id="orbHi" cx="38%" cy="28%" r="35%">
          <stop offset="0%"   stop-color="rgba(200,245,255,0.55)"/>
          <stop offset="100%" stop-color="transparent"/>
        </radialGradient>
        <!-- Outer glow filter -->
        <filter id="orbGlow" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="5" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <!-- Scan line filter -->
        <filter id="scanGlow">
          <feGaussianBlur stdDeviation="1.5"/>
        </filter>
        <!-- Clip to orb circle -->
        <clipPath id="orbClip">
          <circle cx="0" cy="0" r="74"/>
        </clipPath>
      </defs>

      <!-- ── Outer ambient glow ── -->
      <circle cx="0" cy="0" r="80" fill="rgba(0,180,255,0.09)" id="ambGlow"/>
      <circle cx="0" cy="0" r="76" fill="rgba(0,160,255,0.07)"/>

      <!-- ── Main orb sphere ── -->
      <circle cx="0" cy="0" r="74" fill="url(#orbCore)" filter="url(#orbGlow)" id="orbMain"/>

      <!-- ── Latitude rings (clipped inside orb) ── -->
      <g clip-path="url(#orbClip)">
        <!-- equator -->
        <ellipse cx="0" cy="0"  rx="74" ry="16" fill="none" stroke="rgba(100,220,255,0.35)" stroke-width="1.2"/>
        <!-- 30° N -->
        <ellipse cx="0" cy="-34" rx="64" ry="12" fill="none" stroke="rgba(80,200,255,0.22)" stroke-width="0.9"/>
        <!-- 60° N -->
        <ellipse cx="0" cy="-58" rx="36" ry="7"  fill="none" stroke="rgba(60,190,255,0.15)" stroke-width="0.7"/>
        <!-- 30° S -->
        <ellipse cx="0" cy="34"  rx="64" ry="12" fill="none" stroke="rgba(80,200,255,0.22)" stroke-width="0.9"/>
        <!-- 60° S -->
        <ellipse cx="0" cy="58"  rx="36" ry="7"  fill="none" stroke="rgba(60,190,255,0.15)" stroke-width="0.7"/>
        <!-- prime meridian faint -->
        <ellipse cx="0" cy="0"  rx="16" ry="74" fill="none" stroke="rgba(80,200,255,0.12)" stroke-width="0.7"/>

        <!-- Scan line that sweeps top-to-bottom while speaking -->
        <rect id="scanRect" x="-74" y="-80" width="148" height="3"
              fill="rgba(0,240,255,0.45)" rx="1"
              style="display:none;filter:url(#scanGlow);"/>

        <!-- Atmosphere overlay -->
        <circle cx="0" cy="0" r="74" fill="url(#orbAtmo)"/>
        <!-- Top highlight -->
        <circle cx="0" cy="0" r="74" fill="url(#orbHi)"/>

        <!-- ── Eyes on orb surface ── -->
        <!-- Left eye socket -->
        <ellipse cx="-20" cy="-8" rx="9" ry="7" fill="rgba(0,30,60,0.55)"/>
        <!-- Right eye socket -->
        <ellipse cx="20"  cy="-8" rx="9" ry="7" fill="rgba(0,30,60,0.55)"/>
        <!-- Left iris -->
        <ellipse cx="-20" cy="-8" rx="6.5" ry="5" fill="#00d4ff" id="lEye" opacity="0.92"/>
        <!-- Right iris -->
        <ellipse cx="20"  cy="-8" rx="6.5" ry="5" fill="#00d4ff" id="rEye" opacity="0.92"/>
        <!-- Pupils -->
        <circle cx="-20" cy="-8" r="2.8" fill="#001020" id="lPupil"/>
        <circle cx="20"  cy="-8" r="2.8" fill="#001020" id="rPupil"/>
        <!-- Pupil shine -->
        <circle cx="-18.5" cy="-9.5" r="1.1" fill="rgba(255,255,255,0.5)"/>
        <circle cx="21.5"  cy="-9.5" r="1.1" fill="rgba(255,255,255,0.5)"/>

        <!-- Mouth arc (lip-sync target) -->
        <path id="mouth" d="M-14 12 Q0 18 14 12"
              stroke="rgba(0,220,255,0.65)" stroke-width="1.8"
              fill="none" stroke-linecap="round"/>
      </g>

      <!-- ── Orbit band (rotates around equator) ── -->
      <g id="orbitBand" style="transform-origin:center;">
        <ellipse cx="0" cy="0" rx="86" ry="20"
                 fill="none"
                 stroke="rgba(0,220,255,0.55)" stroke-width="2.2"
                 stroke-dasharray="18 10"/>
        <!-- Small orbiting dot -->
        <circle cx="86" cy="0" r="4" fill="#00d4ff" opacity="0.9"/>
      </g>

      <!-- ── Corner accent dots (HUD feel) ── -->
      <circle cx="-68" cy="-68" r="2.5" fill="rgba(0,255,136,0.7)"/>
      <circle cx="68"  cy="-68" r="2.5" fill="rgba(0,255,136,0.7)"/>
    </svg>
    <!-- ── HOLOGRAPHIC ORB LABEL ── -->
    <div style="position:absolute;bottom:-18px;left:0;right:0;text-align:center;
      font-family:'Share Tech Mono',monospace;font-size:8px;
      color:rgba(0,212,255,0.4);letter-spacing:2px;">HOLOGRAPHIC ORB</div>
  </div>

  <div class="speak-bars" id="speakBars" style="margin-top:24px;">
    <div class="b"></div><div class="b"></div><div class="b"></div>
    <div class="b"></div><div class="b"></div>
  </div>

  <div class="status-badge idle" id="statusBadge">
    <div class="s-dot"></div>
    <span id="statusTxt">Active</span>
  </div>

  <div class="av-divider"></div>

  <div class="vc-wrap">
    <div class="vc-label">Voice Profile</div>
    <select class="vc-select" id="voiceSel"><option value="">Loading...</option></select>
    <div class="rate-row">
      <span class="rate-lbl">Speed</span>
      <input type="range" class="rate-slider" id="rateSlider" min="0.7" max="1.4" step="0.05" value="0.9">
      <span class="rate-val" id="rateVal">0.90x</span>
    </div>
  </div>

  <div class="av-btn-row">
    <button class="av-btn av-btn-speak" id="btnSpeak" onclick="speakQuestion()">&#9654; Read Aloud</button>
    <button class="av-btn av-btn-stop" id="btnStop" onclick="stopSpeaking()">&#9632; Stop</button>
  </div>

  <div class="tips">
    <div class="tip" onclick="speakHint('Use the STAR method: Situation, Task, Action, Result.')">&#128161; STAR</div>
    <div class="tip" onclick="speakHint('Give a specific, concrete example from your past experience.')">&#127919; Specific</div>
    <div class="tip" onclick="speakHint('You have about two minutes for this answer.')">&#8987; 2 min</div>
  </div>

</div>

<script>
var QUESTION_TEXT = {q_text_js};
var AUTO_SPEAK    = {auto_js};

var utterance   = null;
var isSpeaking  = false;
var lipInterval = null;
var voices      = [];

// ── Orbit band rotation ──
var orbitAngle = 0;
var orbitBand  = document.getElementById('orbitBand');
(function spinOrbit(){{
  orbitAngle = (orbitAngle + 0.6) % 360;
  // Perspective tilt: rotate in 3D around Y axis
  orbitBand.setAttribute('transform',
    'rotate(' + orbitAngle + ')');
  requestAnimationFrame(spinOrbit);
}})();

// ── Ambient glow pulse ──
var ambGlow = document.getElementById('ambGlow');
var glowDir = 1, glowVal = 0.09;
(function pulseGlow(){{
  glowVal += glowDir * 0.002;
  if(glowVal > 0.18) glowDir = -1;
  if(glowVal < 0.05) glowDir = 1;
  ambGlow.setAttribute('fill', 'rgba(0,180,255,' + glowVal.toFixed(3) + ')');
  requestAnimationFrame(pulseGlow);
}})();

function g(id){{ return document.getElementById(id); }}

function setStatus(mode){{
  var badge=g('statusBadge'), txt=g('statusTxt'), bars=g('speakBars');
  badge.className='status-badge '+mode;
  if(mode==='speaking'){{ txt.textContent='Speaking'; bars.classList.add('speaking'); }}
  else if(mode==='listening'){{ txt.textContent='Listening'; bars.classList.remove('speaking'); }}
  else {{ txt.textContent='Active'; bars.classList.remove('speaking'); }}
}}

// ── Mouth shapes for lip-sync (arc paths on orb) ──
var mouthShapes=[
  'M-14 12 Q0 18 14 12',
  'M-14 11 Q0 20 14 11',
  'M-14 12 Q0 19 14 12',
  'M-13 12 Q0 17 13 12',
  'M-14 13 Q0 16 14 13',
  'M-14 11 Q0 21 14 11',
];
function animateLips(active){{
  clearInterval(lipInterval);
  var m=g('mouth');
  var scan=g('scanRect');
  if(!active){{
    m.setAttribute('d','M-14 12 Q0 18 14 12');
    if(scan) scan.style.display='none';
    return;
  }}
  // Show scan line while speaking
  if(scan){{
    scan.style.display='block';
    var sy=-80, sdir=1;
    var scanAnim=setInterval(function(){{
      sy += sdir*3.5;
      if(sy>80){{ sy=-80; }}
      scan.setAttribute('y', sy.toString());
    }}, 30);
    // store for cleanup
    window._scanAnim = scanAnim;
  }}
  var i=0;
  lipInterval=setInterval(function(){{
    m.setAttribute('d',mouthShapes[i%mouthShapes.length]);
    i++;
  }},90+Math.random()*50);
}}

// ── Blink animation (squish eye ry to ~0) ──
function blink(){{
  var lE=g('lEye'),rE=g('rEye'),lP=g('lPupil'),rP=g('rPupil');
  lE.setAttribute('ry','0.5'); rE.setAttribute('ry','0.5');
  lP.setAttribute('ry','0.5'); rP.setAttribute('ry','0.5');
  setTimeout(function(){{
    lE.setAttribute('ry','5'); rE.setAttribute('ry','5');
    lP.setAttribute('ry','2.8'); rP.setAttribute('ry','2.8');
  }},120);
}}
(function sb(){{ setTimeout(function(){{blink();sb();}},2800+Math.random()*3200); }})();

// ── Eye follow: subtle pupils drift toward centre on idle ──
var eyeDrift=0;
(function driftEyes(){{
  eyeDrift += 0.02;
  var dx = Math.sin(eyeDrift)*1.8;
  var dy = Math.cos(eyeDrift*0.7)*0.9;
  var lP=g('lPupil'), rP=g('rPupil');
  if(lP) lP.setAttribute('cx', (-20+dx).toFixed(2));
  if(lP) lP.setAttribute('cy', (-8+dy).toFixed(2));
  if(rP) rP.setAttribute('cx', (20+dx).toFixed(2));
  if(rP) rP.setAttribute('cy', (-8+dy).toFixed(2));
  requestAnimationFrame(driftEyes);
}})();

function loadVoices(){{
  voices=window.speechSynthesis.getVoices().filter(function(v){{return v.lang.startsWith('en');}});
  var sel=g('voiceSel'); sel.innerHTML='';
  if(!voices.length){{sel.innerHTML='<option value="">No voices</option>';return;}}
  var pref=0;
  voices.forEach(function(v,i){{
    var opt=document.createElement('option'); opt.value=i;
    var n=v.name.toLowerCase();
    var gd=(n.includes('female')||n.includes('samantha')||n.includes('karen')||n.includes('victoria')||n.includes('moira'))?'\u2640':'\u2642';
    opt.textContent=gd+' '+v.name.split(' ').slice(0,3).join(' ');
    sel.appendChild(opt);
    if(n.includes('daniel')||n.includes('alex')||n.includes('google uk english male')) pref=i;
  }});
  sel.value=pref;
}}
if(speechSynthesis.onvoiceschanged!==undefined) speechSynthesis.onvoiceschanged=loadVoices;
setTimeout(loadVoices,250);

g('rateSlider').addEventListener('input',function(){{
  g('rateVal').textContent=parseFloat(this.value).toFixed(2)+'x';
}});

function speakText(text,onDone){{
  stopSpeaking(false);
  if(!window.speechSynthesis||!text.trim()) return;
  utterance=new SpeechSynthesisUtterance(text);
  var idx=parseInt(g('voiceSel').value);
  utterance.voice=isNaN(idx)?null:(voices[idx]||null);
  utterance.rate=parseFloat(g('rateSlider').value);
  utterance.pitch=0.95; utterance.volume=1.0;
  utterance.onstart=function(){{
    isSpeaking=true; setStatus('speaking'); animateLips(true);
    g('btnStop').classList.add('vis');
  }};
  utterance.onend=utterance.onerror=function(){{
    isSpeaking=false; setStatus('idle'); animateLips(false);
    clearInterval(window._scanAnim);
    g('btnStop').classList.remove('vis');
    if(onDone) onDone();
  }};
  speechSynthesis.speak(utterance);
}}

function speakQuestion(){{ speakText(QUESTION_TEXT); }}
function speakHint(h){{ speakText(h); }}
function stopSpeaking(rs){{
  if(rs===undefined) rs=true;
  speechSynthesis.cancel(); isSpeaking=false; animateLips(false);
  clearInterval(window._scanAnim);
  g('btnStop').classList.remove('vis');
  if(rs) setStatus('idle');
}}

if(AUTO_SPEAK){{ setTimeout(speakQuestion,600); }}
</script>
</body>
</html>"""
    components.html(html, height=height, scrolling=False)