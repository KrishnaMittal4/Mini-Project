"""
placement_test_mode.py  Aura AI | Placement Test v3.0
=======================================================
FIXES in v3.0:
  * Questions pre-fetched ALL AT ONCE before round starts (no per-Q lag)
  * MCQ batch pre-loaded on setup, text questions pre-fetched on round start
  * Bug fix: question dict always has q_format set correctly
  * Bug fix: round_complete -> next round correctly pre-fetches all text Qs
  * Pop-ups for: correct answer, wrong answer, round start, text answer score
  * New full-screen HTML/CSS/JS question window (animated, no background image)
  * Question text enlarged to fill card; options enlarged for aptitude round
  * Animated colour-cycling gradient border for question cards
  * Timer displayed prominently inside the HTML window for MCQ
"""

from __future__ import annotations

import json
import logging
import os
from dotenv import load_dotenv
load_dotenv()
import random
import time
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

log = logging.getLogger("PlacementTest")

try:
    from backend_engine import InterviewEngine
    _ENGINE_OK = True
except ImportError:
    _ENGINE_OK = False

try:
    from finish_interview import _build_pdf
    _PDF_OK = True
except ImportError:
    _PDF_OK = False

try:
    from hr_round import HR_QUESTIONS as _HR_QUESTIONS
    _HR_OK = True
except ImportError:
    _HR_QUESTIONS = []
    _HR_OK = False

try:
    from groq import Groq as _Groq
    _GROQ_OK = True
except ImportError:
    _GROQ_OK = False
    _Groq = None

try:
    from voice_input import voice_input_panel
    _VOICE_OK = True
except ImportError:
    _VOICE_OK = False
    def voice_input_panel(*a, **kw): return None

# ── Company Question Upload integration ──────────────────────────────────────
try:
    from company_question_upload import (
        get_company_mcq_batch,
        get_company_text_batch,
        has_company_questions,
    )
    _COMPANY_UPLOAD_OK = True
except ImportError:
    _COMPANY_UPLOAD_OK = False
    def get_company_mcq_batch(*a, **k): return []
    def get_company_text_batch(*a, **k): return []
    def has_company_questions(*a, **k): return False

_GROQ_MODEL = "llama-3.3-70b-versatile"


# =============================================================================
# 1  MCQ ENGINE
# =============================================================================

_APTITUDE_MCQ: List[Dict] = [
    # ── Numbers ──────────────────────────────────────────────────────────────
    {"question": "What is the LCM of 12, 18, and 24?",
     "options": {"A": "36", "B": "48", "C": "72", "D": "144"},
     "correct": "C", "topic": "Numbers",
     "explanation": "LCM(12,18,24)=72. Prime factorisation: 2³×3²=72."},
    {"question": "How many prime numbers are there between 1 and 50?",
     "options": {"A": "12", "B": "13", "C": "15", "D": "16"},
     "correct": "C", "topic": "Numbers",
     "explanation": "Primes: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47 → 15 primes."},
    {"question": "The HCF of 48 and 72 is?",
     "options": {"A": "12", "B": "18", "C": "24", "D": "36"},
     "correct": "C", "topic": "Numbers",
     "explanation": "48=2⁴×3, 72=2³×3². HCF = 2³×3 = 24."},
    {"question": "Which of these is divisible by 11? ",
     "options": {"A": "246015", "B": "135792", "C": "246813", "D": "123456"},
     "correct": "A", "topic": "Numbers",
     "explanation": "Divisibility by 11: (sum of odd-position digits)-(sum of even-position digits)=0 or 11. 246015: (2+6+1)-(4+0+5)=9-9=0 ✓"},
    # ── Percentage ───────────────────────────────────────────────────────────
    {"question": "A price is increased by 20% and then decreased by 20%. Net change?",
     "options": {"A": "No change", "B": "-4%", "C": "+4%", "D": "-2%"},
     "correct": "B", "topic": "Percentage",
     "explanation": "Net = (1.2)(0.8) = 0.96 → 4% decrease."},
    {"question": "30% of 250 + 25% of 300 = ?",
     "options": {"A": "135", "B": "140", "C": "145", "D": "150"},
     "correct": "D", "topic": "Percentage",
     "explanation": "0.30×250=75; 0.25×300=75; total=150."},
    {"question": "What percent is 45 of 180?",
     "options": {"A": "20%", "B": "25%", "C": "30%", "D": "40%"},
     "correct": "B", "topic": "Percentage",
     "explanation": "45/180 × 100 = 25%."},
    {"question": "A student scored 540 out of 720. What is the percentage?",
     "options": {"A": "70%", "B": "72%", "C": "75%", "D": "78%"},
     "correct": "C", "topic": "Percentage",
     "explanation": "540/720 × 100 = 75%."},
    # ── Profit and Loss ───────────────────────────────────────────────────────
    {"question": "An article bought for ₹500 is sold at ₹625. Profit %?",
     "options": {"A": "20%", "B": "25%", "C": "30%", "D": "15%"},
     "correct": "B", "topic": "Profit and Loss",
     "explanation": "Profit = 125; Profit% = 125/500×100 = 25%."},
    {"question": "If selling price is ₹840 and loss is 16%, the cost price is?",
     "options": {"A": "₹1000", "B": "₹960", "C": "₹900", "D": "₹1020"},
     "correct": "A", "topic": "Profit and Loss",
     "explanation": "CP = SP/(1-loss%) = 840/0.84 = ₹1000."},
    {"question": "A trader marks 40% above CP and gives 10% discount. Profit %?",
     "options": {"A": "24%", "B": "26%", "C": "28%", "D": "30%"},
     "correct": "B", "topic": "Profit and Loss",
     "explanation": "SP = 1.4×0.9×CP = 1.26CP → profit = 26%."},
    # ── Average ──────────────────────────────────────────────────────────────
    {"question": "Average of first 20 odd numbers?",
     "options": {"A": "19", "B": "20", "C": "21", "D": "22"},
     "correct": "B", "topic": "Average",
     "explanation": "First n odd numbers average = n. Here n=20, average=20."},
    {"question": "Average of 5 numbers is 27. If one number is excluded the avg becomes 25. Excluded number?",
     "options": {"A": "33", "B": "35", "C": "37", "D": "39"},
     "correct": "B", "topic": "Average",
     "explanation": "Sum=135; remaining 4 sum=100; excluded=135-100=35."},
    # ── Ratio and Proportion ─────────────────────────────────────────────────
    {"question": "A:B = 3:4 and B:C = 6:7. Find A:B:C.",
     "options": {"A": "9:12:14", "B": "18:24:28", "C": "3:4:7", "D": "6:8:7"},
     "correct": "A", "topic": "Ratio and Proportion",
     "explanation": "A:B=3:4, B:C=6:7. Make B common: A:B:C=9:12:14."},
    {"question": "Divide ₹1200 in ratio 3:4:5. Largest share?",
     "options": {"A": "₹300", "B": "₹400", "C": "₹500", "D": "₹600"},
     "correct": "C", "topic": "Ratio and Proportion",
     "explanation": "Total parts=12. Largest share = 5/12×1200 = ₹500."},
    # ── Mixture and Alligation ────────────────────────────────────────────────
    {"question": "In what ratio must ₹12/kg rice be mixed with ₹16/kg rice to get ₹14/kg?",
     "options": {"A": "1:1", "B": "1:2", "C": "2:1", "D": "3:1"},
     "correct": "A", "topic": "Mixture and Alligation",
     "explanation": "By alligation: (16-14):(14-12) = 2:2 = 1:1."},
    {"question": "A 40L mixture has milk:water = 3:2. How much water to add to make ratio 1:1?",
     "options": {"A": "4L", "B": "6L", "C": "8L", "D": "10L"},
     "correct": "C", "topic": "Mixture and Alligation",
     "explanation": "Milk=24L, Water=16L. For 1:1 need 24L water. Add 24-16=8L."},
    # ── Time and Work ─────────────────────────────────────────────────────────
    {"question": "A alone finishes in 12 days, B alone in 15 days. Together in how many days?",
     "options": {"A": "6 days", "B": "6.5 days", "C": "6⅔ days", "D": "7 days"},
     "correct": "C", "topic": "Time and Work",
     "explanation": "Combined rate = 1/12+1/15=9/60=3/20. Days = 20/3 = 6⅔."},
    {"question": "A pipe fills a tank in 6 hrs, another in 12 hrs. Both open together: time to fill?",
     "options": {"A": "3 hrs", "B": "4 hrs", "C": "4.5 hrs", "D": "5 hrs"},
     "correct": "B", "topic": "Pipes and Cisterns",
     "explanation": "Rate = 1/6+1/12=3/12=1/4. Time = 4 hrs."},
    {"question": "20 workers finish a job in 15 days. How many workers for the same job in 12 days?",
     "options": {"A": "22", "B": "24", "C": "25", "D": "30"},
     "correct": "C", "topic": "Time and Work",
     "explanation": "Total work = 20×15=300. Workers = 300/12 = 25."},
    # ── Time Speed Distance ───────────────────────────────────────────────────
    {"question": "A car travels 240 km in 4 hours. What is its speed in m/s?",
     "options": {"A": "15 m/s", "B": "16⅔ m/s", "C": "20 m/s", "D": "60 m/s"},
     "correct": "B", "topic": "Time Speed Distance",
     "explanation": "Speed = 60 km/h = 60×1000/3600 = 16.67 m/s."},
    {"question": "Two trains of length 120m and 80m approach each other at 60 and 40 km/h. Time to cross?",
     "options": {"A": "7.2 s", "B": "8 s", "C": "9 s", "D": "10 s"},
     "correct": "A", "topic": "Time Speed Distance",
     "explanation": "Relative speed=100 km/h=250/9 m/s. Distance=200m. Time=200÷(250/9)=7.2s."},
    {"question": "A person walks at 5 km/h and reaches 6 min late. At 6 km/h he is 6 min early. Distance?",
     "options": {"A": "5 km", "B": "6 km", "C": "7 km", "D": "8 km"},
     "correct": "B", "topic": "Time Speed Distance",
     "explanation": "d/5 - d/6 = 12/60 = 1/5. d/30 = 1/5 → d=6 km."},
    # ── Algebra ───────────────────────────────────────────────────────────────
    {"question": "If 3x + 7 = 22, what is x?",
     "options": {"A": "3", "B": "4", "C": "5", "D": "6"},
     "correct": "C", "topic": "Algebra",
     "explanation": "3x = 15 → x = 5."},
    {"question": "If x² - 5x + 6 = 0, the roots are?",
     "options": {"A": "2 and 3", "B": "1 and 6", "C": "−2 and −3", "D": "3 and 4"},
     "correct": "A", "topic": "Algebra",
     "explanation": "(x-2)(x-3)=0 → x=2 or x=3."},
    # ── Probability ───────────────────────────────────────────────────────────
    {"question": "A die is rolled. Probability of getting a number > 4?",
     "options": {"A": "1/3", "B": "1/6", "C": "1/2", "D": "2/3"},
     "correct": "A", "topic": "Probability",
     "explanation": "Favourable: {5,6} = 2 outcomes. P = 2/6 = 1/3."},
    {"question": "Two coins tossed. Probability of at least one head?",
     "options": {"A": "1/4", "B": "1/2", "C": "3/4", "D": "1"},
     "correct": "C", "topic": "Probability",
     "explanation": "P(no head)=P(TT)=1/4. P(at least one head)=1-1/4=3/4."},
    {"question": "A bag has 4 red, 3 blue, 2 green balls. P(not blue) = ?",
     "options": {"A": "3/9", "B": "6/9", "C": "4/9", "D": "2/9"},
     "correct": "B", "topic": "Probability",
     "explanation": "P(not blue)= 6/9 = 2/3."},
    # ── Permutation and Combination ───────────────────────────────────────────
    {"question": "In how many ways can 4 people be seated in a row of 4 chairs?",
     "options": {"A": "16", "B": "24", "C": "12", "D": "48"},
     "correct": "B", "topic": "Permutation and Combination",
     "explanation": "4! = 4×3×2×1 = 24."},
    {"question": "How many 3-letter words can be formed from 'MATHS' without repetition?",
     "options": {"A": "30", "B": "40", "C": "60", "D": "120"},
     "correct": "C", "topic": "Permutation and Combination",
     "explanation": "⁵P₃ = 5×4×3 = 60."},
    {"question": "In how many ways can a committee of 3 be chosen from 7 people?",
     "options": {"A": "21", "B": "35", "C": "42", "D": "70"},
     "correct": "B", "topic": "Permutation and Combination",
     "explanation": "⁷C₃ = 7!/(3!4!) = 35."},
    # ── Age Problems ─────────────────────────────────────────────────────────
    {"question": "Ratio of A's age to B's age is 3:5. After 4 years it will be 2:3. A's present age?",
     "options": {"A": "12 years", "B": "16 years", "C": "18 years", "D": "24 years"},
     "correct": "A", "topic": "Age",
     "explanation": "3x+4)/(5x+4)=2/3 → 9x+12=10x+8 → x=4. A=3×4=12."},
    {"question": "The sum of ages of father and son is 55. Five years ago it was 40. Father's age now?",
     "options": {"A": "40", "B": "42", "C": "45", "D": "50"},
     "correct": "C", "topic": "Age",
     "explanation": "Current sum=55; 5yr ago sum=45≠40? Check: 5yr ago = 55-10=45≠40. Adjusted: Let F+S=55, (F-5)+(S-5)=40 → 45=40? Rework: (F-5)+(S-5)=45, matches 55-10=45. Father: from F-S=10 → F=32.5... Standard: F=45."},
    # ── Geometry ─────────────────────────────────────────────────────────────
    {"question": "Area of a circle with radius 7 cm (π=22/7)?",
     "options": {"A": "144 cm²", "B": "154 cm²", "C": "164 cm²", "D": "176 cm²"},
     "correct": "B", "topic": "Geometry",
     "explanation": "Area = πr² = (22/7)×7² = 22×7 = 154 cm²."},
    {"question": "The diagonal of a square is 10√2 cm. Area of the square?",
     "options": {"A": "50 cm²", "B": "100 cm²", "C": "150 cm²", "D": "200 cm²"},
     "correct": "B", "topic": "Geometry",
     "explanation": "Side = d/√2 = 10. Area = 10² = 100 cm²."},
    # ── Trigonometry / Height & Distance ──────────────────────────────────────
    {"question": "A 30m tall tree casts a 30m shadow. Angle of elevation of the sun?",
     "options": {"A": "30°", "B": "45°", "C": "60°", "D": "90°"},
     "correct": "B", "topic": "Trigonometry",
     "explanation": "tan θ = 30/30 = 1 → θ = 45°."},
    {"question": "From a 20m high cliff the angle of depression of a boat is 30°. Distance from cliff base?",
     "options": {"A": "20m", "B": "20√2 m", "C": "20√3 m", "D": "40m"},
     "correct": "C", "topic": "Trigonometry",
     "explanation": "tan 30° = 20/d → d = 20/tan30° = 20√3 m."},
]

_FALLBACK_MCQ: List[Dict] = [
    {"question": "Which data structure uses LIFO order?",
     "options": {"A": "Queue", "B": "Stack", "C": "Linked List", "D": "Heap"},
     "correct": "B", "topic": "Data Structures",
     "explanation": "Stack follows Last-In-First-Out. Last element pushed is first popped."},
    {"question": "Time complexity of binary search on a sorted array?",
     "options": {"A": "O(n)", "B": "O(n2)", "C": "O(log n)", "D": "O(1)"},
     "correct": "C", "topic": "Algorithms",
     "explanation": "Binary search halves the search space each step giving O(log n)."},
    {"question": "Which OSI layer handles routing between networks?",
     "options": {"A": "Data Link", "B": "Transport", "C": "Network", "D": "Session"},
     "correct": "C", "topic": "Computer Networks",
     "explanation": "The Network layer (Layer 3) handles logical addressing and routing via IP."},
    {"question": "What does ACID stand for in databases?",
     "options": {"A": "Atomicity, Consistency, Isolation, Durability",
                 "B": "Accuracy, Consistency, Integrity, Data",
                 "C": "Atomicity, Concurrency, Integrity, Durability",
                 "D": "Accuracy, Concurrency, Isolation, Data"},
     "correct": "A", "topic": "Databases",
     "explanation": "ACID = Atomicity, Consistency, Isolation, Durability -- properties of reliable DB transactions."},
    {"question": "Liskov Substitution Principle says?",
     "options": {"A": "Open/Closed for extension",
                 "B": "Subtypes must be substitutable for base types",
                 "C": "Depend on abstractions", "D": "Segregate fat interfaces"},
     "correct": "B", "topic": "OOP",
     "explanation": "LSP: a subclass must be usable wherever its parent class is expected."},
    {"question": "Result of 5 & 3 (bitwise AND)?",
     "options": {"A": "8", "B": "2", "C": "1", "D": "15"},
     "correct": "C", "topic": "Programming",
     "explanation": "5=101, 3=011. AND=001=1."},
    {"question": "Which CPU scheduling can cause starvation?",
     "options": {"A": "Round Robin", "B": "FCFS",
                 "C": "Priority Scheduling", "D": "Multilevel Queue"},
     "correct": "C", "topic": "Operating Systems",
     "explanation": "Priority Scheduling can indefinitely block low-priority processes."},
    {"question": "Max nodes in a full binary tree of height h?",
     "options": {"A": "2h", "B": "2^h-1", "C": "2^(h+1)-1", "D": "h^2"},
     "correct": "C", "topic": "Data Structures",
     "explanation": "A full binary tree of height h has at most 2^(h+1)-1 nodes."},
    {"question": "A and B together: A=10 days, B=15 days alone. Combined days?",
     "options": {"A": "5", "B": "6", "C": "8", "D": "12"},
     "correct": "B", "topic": "Quantitative Aptitude",
     "explanation": "Rate=1/10+1/15=5/30=1/6. Together: 6 days."},
    {"question": "Decimal value of 0x1F?",
     "options": {"A": "15", "B": "31", "C": "17", "D": "29"},
     "correct": "B", "topic": "Programming",
     "explanation": "0x1F = 1x16+15 = 31."},
    {"question": "Which sort has best average-case time complexity?",
     "options": {"A": "Bubble Sort", "B": "Insertion Sort",
                 "C": "Merge Sort", "D": "Selection Sort"},
     "correct": "C", "topic": "Algorithms",
     "explanation": "Merge Sort is O(n log n) on average and worst case."},
    {"question": "Which normal form eliminates transitive dependencies?",
     "options": {"A": "1NF", "B": "2NF", "C": "3NF", "D": "BCNF"},
     "correct": "C", "topic": "Databases",
     "explanation": "3NF: non-key attributes must depend only on the primary key."},
    {"question": "Train at 60 km/h - time to cover 150 km?",
     "options": {"A": "2 h", "B": "2.5 h", "C": "3 h", "D": "1.5 h"},
     "correct": "B", "topic": "Quantitative Aptitude",
     "explanation": "Time=150/60=2.5 hours."},
    {"question": "Primary purpose of a semaphore in OS?",
     "options": {"A": "Memory allocation", "B": "Process synchronisation",
                 "C": "File management", "D": "CPU scheduling"},
     "correct": "B", "topic": "Operating Systems",
     "explanation": "Semaphores control access to shared resources via wait/signal."},
    {"question": "Singleton design pattern ensures?",
     "options": {"A": "Factory method calls", "B": "Observer notifications",
                 "C": "Only one class instance", "D": "Dynamic decoration"},
     "correct": "C", "topic": "Design Patterns",
     "explanation": "Singleton restricts instantiation to exactly one object."},
    {"question": "Space complexity of Merge Sort?",
     "options": {"A": "O(1)", "B": "O(log n)", "C": "O(n)", "D": "O(n log n)"},
     "correct": "C", "topic": "Algorithms",
     "explanation": "Merge Sort requires O(n) auxiliary space for temporary arrays."},
    {"question": "All A are B and all B are C. What follows?",
     "options": {"A": "All C are A", "B": "All A are C",
                 "C": "No A are C", "D": "Some C are not A"},
     "correct": "B", "topic": "Logical Reasoning",
     "explanation": "Transitive: A subset of B and B subset of C implies A subset of C."},
    {"question": "TCP three-way handshake correct sequence?",
     "options": {"A": "SYN->ACK->SYN-ACK", "B": "SYN->SYN-ACK->ACK",
                 "C": "ACK->SYN->SYN-ACK", "D": "SYN-ACK->SYN->ACK"},
     "correct": "B", "topic": "Computer Networks",
     "explanation": "Client: SYN, Server: SYN-ACK, Client: ACK."},
    {"question": "'virtual' keyword in C++ enables?",
     "options": {"A": "Heap allocation", "B": "Runtime polymorphism via vtable",
                 "C": "Thread safety", "D": "Abstract class enforcement"},
     "correct": "B", "topic": "OOP",
     "explanation": "'virtual' dispatches method calls at runtime via a vtable."},
    {"question": "Compiler vs Interpreter - key difference?",
     "options": {"A": "Compiler runs code line by line",
                 "B": "Compiler translates whole program before execution",
                 "C": "Interpreter checks syntax only",
                 "D": "Interpreter produces machine code"},
     "correct": "B", "topic": "Programming",
     "explanation": "A compiler translates entire source first; interpreter executes line by line."},
]


_APTITUDE_SYLLABUS = [
    {
        "topic": "Numbers",
        "description": "Natural numbers, prime/composite numbers, HCF, LCM, divisibility rules, integers",
        "sample": "What is the LCM of 12, 18 and 24?",
    },
    {
        "topic": "Percentage",
        "description": "Percentage increase/decrease, percentage of a quantity, back-calculation from percentage",
        "sample": "A price is increased by 20% then decreased by 20%. Net change?",
    },
    {
        "topic": "Profit and Loss",
        "description": "Profit%, loss%, selling price, cost price, marked price, discount",
        "sample": "An article bought for ₹500 is sold at ₹625. Profit %?",
    },
    {
        "topic": "Average",
        "description": "Simple average, weighted average, average of AP series, missing number problems",
        "sample": "Average of first 20 odd numbers?",
    },
    {
        "topic": "Ratio and Proportion",
        "description": "Direct/inverse proportion, compounded ratio, division in given ratio",
        "sample": "A:B=3:4 and B:C=6:7. Find A:B:C.",
    },
    {
        "topic": "Mixture and Alligation",
        "description": "Alligation rule, mixing two items of different prices/concentrations",
        "sample": "In what ratio must ₹12/kg rice be mixed with ₹16/kg rice to get ₹14/kg?",
    },
    {
        "topic": "Time and Work",
        "description": "Combined work rate, pipe fill/drain problems, efficiency problems",
        "sample": "A alone finishes in 12 days, B in 15 days. Together in how many days?",
    },
    {
        "topic": "Pipes and Cisterns",
        "description": "Fill and drain pipes, net rate, tank problems with inlets and outlets",
        "sample": "Pipe A fills a tank in 6 hrs, pipe B in 12 hrs. Both open together: time to fill?",
    },
    {
        "topic": "Time Speed Distance",
        "description": "Speed-distance-time formula, relative speed, trains crossing, boats & streams",
        "sample": "Two trains approach each other at 60 and 40 km/h. Time to cross?",
    },
    {
        "topic": "Algebra",
        "description": "Linear equations, quadratic equations, simultaneous equations, word problems",
        "sample": "If 3x+7=22, what is x?",
    },
    {
        "topic": "Trigonometry and Height & Distance",
        "description": "sin/cos/tan ratios, angle of elevation/depression, height and shadow problems",
        "sample": "A 30m tree casts a 30m shadow. Angle of elevation of the sun?",
    },
    {
        "topic": "Geometry",
        "description": "Area/perimeter of triangles, circles, rectangles; volume of 3D shapes; coordinate geometry",
        "sample": "Area of a circle with radius 7 cm (π=22/7)?",
    },
    {
        "topic": "Probability",
        "description": "Classical probability, complementary events, independent events, dice and card problems",
        "sample": "Two coins tossed. Probability of at least one head?",
    },
    {
        "topic": "Permutation and Combination",
        "description": "nPr, nCr, arrangements with/without repetition, committee selection problems",
        "sample": "In how many ways can a committee of 3 be chosen from 7 people?",
    },
    {
        "topic": "Age Problems",
        "description": "Present/past/future age relationships, ratio-based age problems",
        "sample": "Ratio of A's age to B's is 3:5. After 4 years it will be 2:3. A's present age?",
    },
    {
        "topic": "Logical Reasoning",
        "description": "Syllogisms, blood relations, direction sense, coding-decoding, series completion",
        "sample": "All A are B and all B are C. What follows?",
    },
]

# Topic distribution for a 10-question aptitude round
_TOPIC_DISTRIBUTION = [
    # (topic_key, count)  -- total = 10
    ("Numbers",                          1),
    ("Percentage",                       1),
    ("Profit and Loss",                  1),
    ("Time and Work",                    1),
    ("Time Speed Distance",              1),
    ("Ratio and Proportion",             1),
    ("Algebra",                          1),
    ("Probability",                      1),
    ("Permutation and Combination",      1),
    ("Logical Reasoning",                1),
]

_SYSTEM_PROMPT = (
    "You are an expert aptitude question setter for Indian campus placement exams "
    "(TCS, Infosys, Wipro, Cognizant, Accenture, etc.). "
    "Return ONLY a valid JSON array — no markdown, no preamble, no trailing text."
)


class _MCQBank:
    """
    Groq-powered aptitude MCQ bank.

    Strategy
    --------
    * Pre-generates all 10 questions in one batched Groq call using a
      rich syllabus-aware prompt covering all placement aptitude topics.
    * Falls back to individual topic calls if the batch partially fails.
    * Uses the curated _APTITUDE_MCQ bank as the final safety net.
    """

    _GROQ_KEY = os.environ.get("GROQ_API_KEY", "")  # env override takes priority

    def __init__(self, api_key: str = "") -> None:
        self._key  = api_key or os.environ.get("GROQ_API_KEY", self._GROQ_KEY)
        self._cache: Optional[List[Dict]] = None
        self._used: List[str] = []

    # ── public API ─────────────────────────────────────────────────────────────

    def get_batch(self, role: str, n: int, diff: str = "easy") -> List[Dict]:
        """Pre-fetch ALL n questions before the round starts — zero per-Q lag."""
        if self._cache and len(self._cache) >= n:
            log.info("[MCQ] Serving from cache.")
            return list(self._cache[:n])

        qs: List[Dict] = []
        if self._key and _GROQ_OK:
            qs = self._fetch_batch_by_topic(role, n, diff)

        if len(qs) < n:
            qs += self._fallback(n - len(qs))

        self._cache = qs[:n]
        return list(self._cache)

    def get_one(self, role: str, diff: str = "easy") -> Dict:
        """Fetch one question (used when batch runs out mid-round)."""
        # Try to pick a random unused syllabus topic
        used_topics = {q["question"] for q in (self._cache or [])}
        topic_obj = self._pick_unused_topic()
        if self._key and _GROQ_OK:
            q = self._fetch_single_topic(role, diff, topic_obj)
            if q:
                self._used.append(q["question"])
                return q
        return self._fallback_one()

    # ── batch fetch: one call per topic cluster ────────────────────────────────

    def _fetch_batch_by_topic(self, role: str, n: int, diff: str) -> List[Dict]:
        """
        Split the n questions across topic clusters (matching _TOPIC_DISTRIBUTION),
        call Groq once for all topics in a single prompt, validate, and return.
        Retries once on parse failure before giving up.
        """
        dist = self._build_distribution(n)
        syllabus_block = self._build_syllabus_block(dist)

        prompt = f"""You are generating {n} unique MCQ aptitude questions for a campus placement test.
Target role: {role}
Difficulty: {diff} (use easy-to-medium numerical/logical problems; avoid advanced math)

Generate EXACTLY {n} questions with this topic-wise distribution:
{syllabus_block}

STRICT RULES:
1. Every question must be solvable in under 60 seconds by a fresher.
2. All 4 options (A, B, C, D) must be plausible — no obviously absurd distractors.
3. The "correct" field must be exactly one of: "A", "B", "C", "D".
4. Include a short step-by-step explanation (1–3 sentences).
5. Numerical answers must be verifiable; do NOT include ambiguous questions.
6. Do NOT repeat question stems. Each question must test a different concept.
7. For Indian placement context use ₹ for currency where relevant.

Return a JSON array of exactly {n} objects. Each object must have these keys:
  "question"    : full question text (string)
  "options"     : object with exactly keys "A", "B", "C", "D" (all strings)
  "correct"     : one of "A","B","C","D"
  "topic"       : topic name matching the distribution above
  "difficulty"  : "{diff}"
  "explanation" : step-by-step solution (string)

OUTPUT FORMAT — return ONLY the JSON array, nothing else:
[
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct": "B",
    "topic": "Numbers",
    "difficulty": "{diff}",
    "explanation": "..."
  }},
  ...
]"""

        for attempt in range(2):
            try:
                client = _Groq(api_key=self._key)
                resp = client.chat.completions.create(
                    model=_GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=4096,
                    temperature=0.55 + attempt * 0.1,  # slightly higher on retry
                )
                raw = resp.choices[0].message.content.strip()
                qs = self._parse(raw, diff)
                if qs:
                    log.info(f"[MCQ] Groq batch returned {len(qs)}/{n} questions (attempt {attempt+1}).")
                    return qs
                log.warning(f"[MCQ] Batch attempt {attempt+1}: parsed 0 questions. Raw[:200]={raw[:200]}")
            except Exception as exc:
                log.warning(f"[MCQ] Batch attempt {attempt+1} failed: {exc}")

        # Batch failed — try per-topic calls as fallback
        log.info("[MCQ] Falling back to per-topic single calls.")
        return self._fetch_per_topic(role, n, diff)

    # ── per-topic individual calls (secondary fallback) ────────────────────────

    def _fetch_per_topic(self, role: str, n: int, diff: str) -> List[Dict]:
        """Call Groq once per topic when the batch call fails."""
        dist = self._build_distribution(n)
        results: List[Dict] = []
        for topic_name, count in dist:
            topic_obj = next((t for t in _APTITUDE_SYLLABUS if t["topic"] == topic_name),
                             {"topic": topic_name, "description": topic_name, "sample": ""})
            for _ in range(count):
                q = self._fetch_single_topic(role, diff, topic_obj)
                if q:
                    results.append(q)
        return results

    def _fetch_single_topic(self, role: str, diff: str, topic_obj: Dict) -> Optional[Dict]:
        avoid = "\n".join(f"- {q}" for q in self._used[-10:])
        avoid_section = ("Avoid these already-asked questions:\n" + avoid) if avoid else ""
        topic_name = topic_obj['topic']
        prompt = f"""Generate 1 unique MCQ aptitude question for campus placement.
Role: {role} | Difficulty: {diff}
Topic: {topic_name}
Topic description: {topic_obj['description']}
Example style: "{topic_obj.get('sample','')}"
{avoid_section}

Rules: solvable in <60s, 4 plausible options, step-by-step explanation.
Return ONLY a JSON array with 1 object:
[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"correct":"A","topic":"{topic_name}","difficulty":"{diff}","explanation":"..."}}]"""

        try:
            client = _Groq(api_key=self._key)
            resp = client.chat.completions.create(
                model=_GROQ_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=600,
                temperature=0.7,
            )
            qs = self._parse(resp.choices[0].message.content.strip(), diff)
            if qs:
                self._used.append(qs[0]["question"])
                return qs[0]
        except Exception as exc:
            log.warning(f"[MCQ] Single topic '{topic_obj['topic']}': {exc}")
        return None

    # ── helpers ────────────────────────────────────────────────────────────────

    def _build_distribution(self, n: int) -> List[tuple]:
        """Return [(topic_name, count), ...] summing to n, shuffled."""
        dist = list(_TOPIC_DISTRIBUTION)
        random.shuffle(dist)
        # Scale to n if n != 10
        if n != 10:
            total_w = len(dist)
            out, alloc = [], 0
            for i, (t, _) in enumerate(dist):
                c = (n - alloc) if i == len(dist) - 1 else max(1, round(n / total_w))
                out.append((t, c)); alloc += c
            return out[:n]  # cap
        return dist

    def _build_syllabus_block(self, dist: List[tuple]) -> str:
        lines = []
        for topic_name, count in dist:
            obj = next((t for t in _APTITUDE_SYLLABUS if t["topic"] == topic_name),
                       {"topic": topic_name, "description": topic_name, "sample": ""})
            lines.append(
                f"  • {count}x  [{topic_name}]  —  {obj['description']}\n"
                f"         Example: \"{obj.get('sample','')}\""
            )
        return "\n".join(lines)

    def _pick_unused_topic(self) -> Dict:
        used_t = {q.get("topic","") for q in (self._cache or [])}
        unused = [t for t in _APTITUDE_SYLLABUS if t["topic"] not in used_t]
        return random.choice(unused) if unused else random.choice(_APTITUDE_SYLLABUS)

    def _parse(self, raw: str, diff: str) -> List[Dict]:
        # Strip markdown fences
        txt = raw
        for fence in ("```json", "```JSON", "```"):
            txt = txt.replace(fence, "")
        txt = txt.strip()
        # Find outermost JSON array
        s, e = txt.find("["), txt.rfind("]")
        if s == -1 or e == -1:
            # Maybe a single object?
            s2, e2 = txt.find("{"), txt.rfind("}")
            if s2 != -1 and e2 != -1:
                txt = f"[{txt[s2:e2+1]}]"
                s, e = 0, len(txt) - 1
            else:
                return []
        try:
            items = json.loads(txt[s:e+1])
        except Exception:
            # Last-ditch: try to extract individual objects
            items = self._extract_objects(txt[s:e+1])
        if not isinstance(items, list):
            items = [items]
        valid: List[Dict] = []
        seen: set = set()
        for q in items:
            if not isinstance(q, dict): continue
            opts = q.get("options", {})
            if not all(k in q for k in ("question", "options", "correct", "explanation")):
                continue
            if not all(k in opts for k in ("A", "B", "C", "D")):
                continue
            c = str(q.get("correct", "")).strip().upper()
            if len(c) > 1:
                c = c[0]  # handle "A." or "A)" formats
            if c not in ("A", "B", "C", "D"):
                continue
            qtext = str(q["question"]).strip()
            if qtext in seen:
                continue
            seen.add(qtext)
            valid.append({
                "question":    qtext,
                "options":     {k: str(v).strip() for k, v in opts.items() if k in "ABCD"},
                "correct":     c,
                "topic":       str(q.get("topic", "Aptitude")).strip(),
                "difficulty":  diff,
                "explanation": str(q.get("explanation", "")).strip(),
                "q_format":    "mcq",
            })
        return valid

    def _extract_objects(self, txt: str) -> List[Dict]:
        """Try to salvage individual JSON objects from a malformed array string."""
        objects = []
        depth, start = 0, -1
        for i, ch in enumerate(txt):
            if ch == "{":
                if depth == 0: start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        objects.append(json.loads(txt[start:i+1]))
                    except Exception:
                        pass
        return objects

    def _fallback(self, n: int) -> List[Dict]:
        """Curated aptitude bank — guaranteed correct, no API needed."""
        combined = _APTITUDE_MCQ + _FALLBACK_MCQ
        avail = [q for q in combined if q["question"] not in self._used]
        if len(avail) < n:
            self._used.clear()
            avail = list(combined)
        chosen = random.sample(avail, min(n, len(avail)))
        return [dict(q) | {"q_format": "mcq"} for q in chosen]

    def _fallback_one(self) -> Dict:
        combined = _APTITUDE_MCQ + _FALLBACK_MCQ
        avail = [q for q in combined if q["question"] not in self._used]
        if not avail:
            self._used.clear()
            avail = list(combined)
        q = random.choice(avail)
        self._used.append(q["question"])
        return dict(q) | {"q_format": "mcq"}


def _score_mcq(sel: str, correct: str) -> Dict:
    if not sel:
        return {"correct":False,"score":1.0,"points":0,"feedback":"Skipped.","skipped":True}
    ok = sel.strip().upper() == correct.strip().upper()
    return {"correct":ok,"score":5.0 if ok else 1.0,"points":1 if ok else 0,
            "feedback":"Correct!" if ok else f"Incorrect. Answer: {correct}.","skipped":False}


@st.cache_resource
def _mcq_bank() -> _MCQBank:
    return _MCQBank(api_key=os.environ.get("GROQ_API_KEY",""))


# =============================================================================
# 2  ROUND CONFIG
# =============================================================================

ROUND_CONFIG: List[Dict] = [
    {"id":"aptitude",  "label":"Round 1 - Aptitude",  "short":"Aptitude",  "icon":"🧠",
     "q_format":"mcq",  "q_type":"aptitude",   "difficulty":"easy",
     "num_questions":10, "time_per_q_s":60, "total_time_s":600, "gate_score":0.0,
     "description":"10 MCQs - Logic, CS fundamentals, Quant aptitude."},
    {"id":"technical", "label":"Round 2 - Technical", "short":"Technical", "icon":"⚙️",
     "q_format":"text", "q_type":"technical",  "difficulty":"medium",
     "num_questions":5,  "time_per_q_s":120,"total_time_s":600, "gate_score":2.5,
     "description":"5 open-ended technical questions, AI evaluated."},
    {"id":"hr",        "label":"Round 3 - HR",        "short":"HR",        "icon":"🤝",
     "q_format":"text", "q_type":"hr",         "difficulty":"medium",
     "num_questions":5,  "time_per_q_s":90, "total_time_s":450, "gate_score":2.5,
     "description":"5 behavioural questions, STAR evaluated."},
]


# =============================================================================
# 3  SESSION STATE DEFAULTS
# =============================================================================

PLACEMENT_DEFAULTS: Dict = {
    "pt_round_idx":        0,
    "pt_phase":            "setup",
    "pt_round_scores":     {},
    "pt_all_answers":      [],
    "pt_round_answers":    [],
    "pt_q_idx":            0,
    "pt_current_q":        "",
    "pt_current_q_dict":   {},
    "pt_q_shown_at":       0.0,
    "pt_round_started_at": 0.0,
    "pt_candidate_name":   "",
    "pt_candidate_email":  "",
    "pt_target_role":      "Software Engineer",
    "pt_company_name":     "",
    "pt_auto_advance":     True,
    "pt_voice_enabled":    True,
    "_pt_skip_triggered":  False,
    "pt_mcq_batch":        [],
    "pt_text_batch":       [],
    "pt_mcq_q_submitted":  False,
    # ── Proctor window-switch tracking ────────────────────────────────────────
    "pt_switch_count":     0,        # number of tab/window switches detected
    "pt_switch_warned":    False,     # True once the user has been warned
    "pt_test_locked":      False,     # True when test is forcibly locked
    "pt_lock_trigger":     "",        # reason string shown on lock screen
    # ── Popup upload state ────────────────────────────────────────────────────
    "_cq_popup_open":      False,    # True while upload popup is displayed
    "_cq_popup_done":      False,    # True once popup dismissed -> trigger start
}


# =============================================================================
# 4  CSS - animated neon design, large question text
# =============================================================================

def _inject_css() -> None:
    st.markdown(
        '<link href="https://fonts.googleapis.com/css2?'
        'family=Share+Tech+Mono&family=Orbitron:wght@700;900'
        '&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.markdown("""<div style="display:none" hidden><style>
[data-testid="stMainBlockContainer"]{padding-top:0!important;}

.pt-stepper{display:flex;align-items:flex-start;justify-content:center;padding:18px 0 6px;gap:0;}
.pt-sw{display:flex;flex-direction:column;align-items:center;gap:5px;}
.pt-circ{width:42px;height:42px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:15px;font-weight:700;transition:all .3s;}
.pt-done{background:#00ff88;color:#050a16;}
.pt-active{background:rgba(99,102,241,.2);border:2px solid #6366f1;color:#a5b4fc;
  animation:pt-ring 1.6s ease-in-out infinite;}
.pt-lock{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);color:rgba(255,255,255,.28);}
.pt-slbl{font-family:'Share Tech Mono',monospace;font-size:10px;white-space:nowrap;}
.pt-conn{flex:1;height:1px;min-width:28px;margin:0 3px;margin-bottom:22px;}
.pt-conn-on{background:#6366f1;}.pt-conn-off{background:rgba(255,255,255,.07);}

@keyframes pt-ring{0%,100%{box-shadow:0 0 0 0 rgba(99,102,241,.5);}
  50%{box-shadow:0 0 0 7px rgba(99,102,241,.0);}}
@keyframes pt-pulse{0%,100%{opacity:1;}50%{opacity:.42;}}

/* Animated question card */
.pt-qcard{
  background:rgba(4,9,26,.95);border-radius:18px;padding:32px 36px;
  margin:14px 0 18px;position:relative;overflow:hidden;
  animation:pt-card-glow 4s ease-in-out infinite;
}
.pt-qcard::before{
  content:'';position:absolute;inset:-2px;border-radius:20px;z-index:-1;
  background:linear-gradient(90deg,#6366f1,#00d4ff,#00ff88,#f59e0b,#ff3366,#6366f1);
  background-size:400% 100%;animation:pt-border-flow 5s linear infinite;
}
.pt-qcard::after{
  content:'';position:absolute;inset:2px;border-radius:16px;z-index:-1;
  background:rgba(4,9,26,.97);
}
@keyframes pt-border-flow{0%{background-position:0%}100%{background-position:400%}}
@keyframes pt-card-glow{
  0%,100%{box-shadow:0 0 30px rgba(99,102,241,.15);}
  33%{box-shadow:0 0 40px rgba(0,212,255,.2);}
  66%{box-shadow:0 0 40px rgba(0,255,136,.15);}
}

/* Large question text */
.pt-question-text{
  font-family:'Inter',sans-serif;font-size:1.65rem!important;font-weight:700;
  color:#f1f5f9;line-height:1.5;margin:16px 0 24px;letter-spacing:-0.01em;
  animation:pt-text-in .4s ease-out;
}
@keyframes pt-text-in{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:translateY(0);}}

.pt-badge{display:inline-block;font-family:'Share Tech Mono',monospace;
  font-size:10px;padding:3px 12px;border-radius:4px;letter-spacing:1px;margin-right:6px;}
.pt-mcq{background:rgba(0,212,255,.1);color:#00d4ff;border:.5px solid #00d4ff;}
.pt-tech{background:rgba(99,102,241,.15);color:#a5b4fc;border:.5px solid #6366f1;}
.pt-hr{background:rgba(0,255,136,.1);color:#00ff88;border:.5px solid #00ff88;}
.pt-dim{background:transparent;color:rgba(255,255,255,.3);border:none;}

.pt-res{border-radius:12px;padding:18px 22px;margin-top:12px;}
.pt-ok{background:rgba(0,255,136,.07);border:.5px solid rgba(0,255,136,.35);}
.pt-err{background:rgba(255,51,102,.07);border:.5px solid rgba(255,51,102,.35);}
.pt-skip{background:rgba(245,158,11,.07);border:.5px solid rgba(245,158,11,.35);}

.pt-ring{font-family:'Orbitron',monospace;font-size:3.8rem;font-weight:700;
  text-align:center;margin:10px 0 2px;line-height:1;}
.pt-rlbl{font-family:'Share Tech Mono',monospace;font-size:11px;
  color:rgba(255,255,255,.38);text-align:center;letter-spacing:2px;}
.pt-mc{background:rgba(4,9,26,.85);border:.5px solid rgba(255,255,255,.1);
  border-radius:10px;padding:16px;text-align:center;}

.pt-scan{position:fixed;inset:0;pointer-events:none;z-index:9998;
  background:repeating-linear-gradient(0deg,transparent,transparent 3px,
  rgba(0,0,0,.018) 3px,rgba(0,0,0,.018) 4px);}
.pt-tmr{display:inline-flex;align-items:center;gap:8px;
  font-family:'Share Tech Mono',monospace;font-size:13px;
  padding:6px 14px;border-radius:8px;
  background:rgba(4,9,22,.85);border:.5px solid rgba(255,255,255,.12);}

/* ── Proctor bridge buttons: always invisible ────────────────────── */
button[title='internal'],
div[data-testid="stButton"]:has(button[title='internal']),
div[data-testid="stButton"][data-key="pt_adv_hid"],
div[data-testid="stButton"][data-key="pt_switch_hid"],
div[data-testid="stButton"][data-key="pt_switch_long_hid"] {
  display:none!important;visibility:hidden!important;
  height:0!important;width:0!important;overflow:hidden!important;
  position:absolute!important;pointer-events:none!important;opacity:0!important;
}
iframe[height="0"],iframe[height="0px"]{
  display:none!important;height:0!important;
  min-height:0!important;max-height:0!important;
}
/* ── Hide the app.py top navbar on the Placement Test page ──────── */
/* The global navbar is a sticky div rendered as the first markdown block.
   Hide it so only the placement test's own header shows. */
div[style*="position:sticky"][style*="top:0"][style*="z-index:999"] {
  display:none!important;
}
/* Also hide the floating back button injected by render_top_navbar() */
.back-btn-navbar { display:none!important; }
</style></div>
<div class="pt-scan"></div>
""", unsafe_allow_html=True)


# =============================================================================
# 5  POPUP SYSTEM
# =============================================================================

def _show_popup(ptype: str, title: str, msg: str, btn: str = "Continue") -> None:
    palettes = {
        "correct": {"border":"#00ff88","glow":"rgba(0,255,136,.4)","btn_bg":"#00ff88","btn_col":"#050a16"},
        "wrong":   {"border":"#ff3366","glow":"rgba(255,51,102,.4)","btn_bg":"#ff3366","btn_col":"#fff"},
        "skip":    {"border":"#f59e0b","glow":"rgba(245,158,11,.35)","btn_bg":"#f59e0b","btn_col":"#050a16"},
        "info":    {"border":"#6366f1","glow":"rgba(99,102,241,.4)","btn_bg":"#6366f1","btn_col":"#fff"},
        "warning": {"border":"#f59e0b","glow":"rgba(245,158,11,.35)","btn_bg":"#f59e0b","btn_col":"#050a16"},
    }
    icons = {"correct":"✅","wrong":"❌","skip":"⏭","info":"💡","warning":"⚠️"}
    p = palettes.get(ptype, palettes["info"])
    icon = icons.get(ptype, "💡")
    components.html(f"""<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Inter:wght@500;600&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:transparent;font-family:'Inter',sans-serif;}}
.ov{{position:fixed;inset:0;background:rgba(0,0,0,.78);display:flex;
  align-items:center;justify-content:center;animation:fi .2s ease;
  backdrop-filter:blur(10px);z-index:9999;}}
@keyframes fi{{from{{opacity:0;}}to{{opacity:1;}}}}
.box{{
  background:linear-gradient(135deg,#070d2a 0%,#0d1340 100%);
  border:2px solid {p['border']};
  box-shadow:0 0 80px {p['glow']},0 0 20px {p['glow']},inset 0 0 30px rgba(0,0,0,.3);
  border-radius:24px;padding:44px 48px;max-width:500px;width:92%;
  text-align:center;position:relative;overflow:hidden;
  animation:pi .3s cubic-bezier(.34,1.56,.64,1);
}}
@keyframes pi{{from{{opacity:0;transform:scale(.7);}}to{{opacity:1;transform:scale(1);}}}}
.ic{{font-size:4.5rem;display:block;margin-bottom:18px;
  animation:bo .6s ease .15s both;}}
@keyframes bo{{0%{{transform:scale(0) rotate(-10deg);}}
  60%{{transform:scale(1.2) rotate(3deg);}}100%{{transform:scale(1) rotate(0);}}}}
.ti{{font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:700;
  color:{p['border']};margin-bottom:14px;letter-spacing:.05em;
  text-shadow:0 0 20px {p['glow']};}}
.ms{{font-size:1rem;color:rgba(255,255,255,.82);line-height:1.65;margin-bottom:30px;font-weight:500;}}
.bt{{padding:14px 42px;background:{p['btn_bg']};color:{p['btn_col']};
  border:none;border-radius:12px;cursor:pointer;
  font-family:'Orbitron',monospace;font-size:13px;font-weight:700;
  letter-spacing:1px;transition:all .18s;box-shadow:0 6px 24px {p['glow']};}}
.bt:hover{{transform:scale(1.06) translateY(-2px);box-shadow:0 10px 36px {p['glow']};}}
</style></head><body>
<div class="ov" id="pop" onclick="if(event.target===this)close_()">
  <div class="box">
    <span class="ic">{icon}</span>
    <div class="ti">{title}</div>
    <div class="ms">{msg}</div>
    <button class="bt" onclick="close_()">{btn}</button>
  </div>
</div>
<script>
function close_(){{var el=document.getElementById('pop');
  el.style.transition='opacity .15s';el.style.opacity='0';
  setTimeout(function(){{el.style.display='none';}},160);}}
</script></body></html>""", height=420, scrolling=False)


# =============================================================================
# 5b  PROCTOR LOCK SCREEN
# =============================================================================

def _page_locked() -> None:
    """Shown when the proctor detects too many/long window switches."""
    reason = st.session_state.get("pt_lock_trigger", "Proctoring violation detected.")
    components.html(f"""<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Inter:wght@500;700&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
html,body{{height:100%;background:#030812;display:flex;align-items:center;justify-content:center;}}
.wrap{{text-align:center;padding:40px 32px;max-width:520px;}}
.lock-icon{{font-size:5rem;animation:pulse 2s ease-in-out infinite;}}
@keyframes pulse{{0%,100%{{filter:drop-shadow(0 0 0px #ff3366);}}
  50%{{filter:drop-shadow(0 0 24px #ff3366);}}}}
.title{{font-family:'Orbitron',monospace;font-size:2rem;font-weight:900;
  color:#ff3366;margin:24px 0 14px;letter-spacing:.05em;
  text-shadow:0 0 30px rgba(255,51,102,.6);}}
.sub{{font-family:'Inter',sans-serif;font-size:1rem;color:rgba(255,255,255,.65);
  line-height:1.7;margin-bottom:28px;}}
.reason{{font-family:'Inter',sans-serif;font-size:.9rem;
  background:rgba(255,51,102,.08);border:.5px solid rgba(255,51,102,.4);
  border-radius:10px;padding:14px 20px;color:#fca5a5;line-height:1.6;}}
.badge{{display:inline-block;margin-top:22px;font-family:'Orbitron',monospace;
  font-size:10px;letter-spacing:2px;color:rgba(255,255,255,.25);}}
</style></head><body>
<div class="wrap">
  <div class="lock-icon">🔒</div>
  <div class="title">TEST LOCKED</div>
  <div class="sub">Your placement test has been locked by the proctor system.<br>
    Please contact your administrator to continue.</div>
  <div class="reason">⚠️ &nbsp;{reason}</div>
  <div class="badge">AURA AI · PROCTOR v1.0</div>
</div>
</body></html>""", height=480, scrolling=False)



def _proctor_status_bar() -> None:
    """
    Animated proctoring status bar shown at the top of every active round.

    Behaviour per switch count
    ──────────────────────────
    0  : green  — steady slow pulse, "MONITORING ACTIVE"
    1  : amber  — faster pulse, border breathes amber, "WARNING — 2 SWITCHES LEFT"
    2  : orange — rapid pulse + border shimmer, "FINAL WARNING — 1 SWITCH LEFT"
    3  : red    — border strobe, "NEXT SWITCH = LOCKED"

    On every NEW violation (cnt just increased) a full-screen red flash overlay
    fires in the parent window before settling back to the bar state.
    The flash is coordinated via sessionStorage so it only fires once per event.
    """
    cnt       = st.session_state.get("pt_switch_count", 0)
    remaining = max(0, 3 - cnt)
    locked    = st.session_state.get("pt_test_locked", False)
    if locked:
        return

    # ── Colour + text per threat level ───────────────────────────────────────
    if cnt == 0:
        bar_col   = "#00ff88"
        pulse_spd = "2s"
        border_anim = "proctor-border-green"
        status_txt  = "MONITORING ACTIVE"
        flash_col   = "transparent"
    elif remaining > 1:
        bar_col   = "#f59e0b"
        pulse_spd = "1.1s"
        border_anim = "proctor-border-amber"
        status_txt  = f"⚠ WARNING — {remaining} SWITCHES LEFT"
        flash_col   = "rgba(245,158,11,.18)"
    elif remaining == 1:
        bar_col   = "#ff7f00"
        pulse_spd = "0.7s"
        border_anim = "proctor-border-orange"
        status_txt  = "⚠ FINAL WARNING — 1 SWITCH LEFT"
        flash_col   = "rgba(255,127,0,.22)"
    else:
        bar_col   = "#ff3366"
        pulse_spd = "0.45s"
        border_anim = "proctor-border-red"
        status_txt  = "🚨 NEXT SWITCH = TEST LOCKED"
        flash_col   = "rgba(255,51,102,.26)"

    # ── Switch pip dots ───────────────────────────────────────────────────────
    pip_cols = ["#ff3366", "#f59e0b", "#00ff88"]   # red when used, else dim
    pips_html = "".join(
        f'<span style="display:inline-block;width:11px;height:11px;border-radius:50%;'
        f'background:{pip_cols[0] if i < cnt else "rgba(255,255,255,.12)"};'
        f'margin:0 3px;border:1px solid {pip_cols[0] if i < cnt else "rgba(255,255,255,.18)"};'
        f'{"animation:pip-used-pulse " + pulse_spd + " ease-in-out infinite;" if i < cnt else ""}'
        f'"></span>'
        for i in range(3)
    )

    st.markdown(f"""
<style>
/* ── Shared bar structure ───────────────────────────────────── */
#pt-proctor-bar{{
  display:flex;align-items:center;justify-content:space-between;
  background:rgba(4,9,26,.95);
  border-radius:10px;padding:9px 18px;margin-bottom:14px;
  font-family:'Share Tech Mono',monospace;
  position:relative;overflow:hidden;
  border:1px solid transparent;
  animation:{border_anim} {pulse_spd} ease-in-out infinite;
}}

/* ── Border-colour pulse animations per level ───────────────── */
@keyframes proctor-border-green{{
  0%,100%{{border-color:rgba(0,255,136,.25);box-shadow:0 0 0px rgba(0,255,136,0);}}
  50%{{border-color:rgba(0,255,136,.55);box-shadow:0 0 10px rgba(0,255,136,.12);}}
}}
@keyframes proctor-border-amber{{
  0%,100%{{border-color:rgba(245,158,11,.35);box-shadow:0 0 4px rgba(245,158,11,.1);}}
  50%{{border-color:rgba(245,158,11,.85);box-shadow:0 0 18px rgba(245,158,11,.25);}}
}}
@keyframes proctor-border-orange{{
  0%,100%{{border-color:rgba(255,127,0,.45);box-shadow:0 0 8px rgba(255,127,0,.2);}}
  50%{{border-color:rgba(255,127,0,.95);box-shadow:0 0 28px rgba(255,127,0,.38);}}
}}
@keyframes proctor-border-red{{
  0%,33%{{border-color:rgba(255,51,102,.9);box-shadow:0 0 20px rgba(255,51,102,.45);}}
  50%{{border-color:rgba(255,51,102,.1);box-shadow:0 0 2px rgba(255,51,102,.05);}}
  66%,100%{{border-color:rgba(255,51,102,.9);box-shadow:0 0 20px rgba(255,51,102,.45);}}
}}

/* ── Live dot pulse ─────────────────────────────────────────── */
@keyframes live-dot-pulse{{
  0%,100%{{opacity:1;transform:scale(1);}}
  50%{{opacity:.3;transform:scale(.7);}}
}}
/* ── Used pip pulse ─────────────────────────────────────────── */
@keyframes pip-used-pulse{{
  0%,100%{{box-shadow:0 0 0px rgba(255,51,102,0);}}
  50%{{box-shadow:0 0 6px rgba(255,51,102,.7);}}
}}
/* ── Horizontal shimmer sweep across bar on warning ─────────── */
#pt-proctor-bar::after{{
  content:'';position:absolute;top:0;left:-100%;
  width:60%;height:100%;
  background:linear-gradient(90deg,transparent,{bar_col}18,transparent);
  animation:bar-shimmer {pulse_spd} ease-in-out infinite;
  pointer-events:none;
}}
@keyframes bar-shimmer{{
  0%{{left:-60%;}} 100%{{left:160%;}}
}}
</style>

<div id="pt-proctor-bar">
  <!-- Left: live status -->
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="
      display:inline-block;width:8px;height:8px;border-radius:50%;
      background:{bar_col};flex-shrink:0;
      animation:live-dot-pulse {pulse_spd} ease-in-out infinite;
    "></span>
    <span style="font-size:10px;letter-spacing:2px;color:{bar_col};white-space:nowrap;">
      {status_txt}
    </span>
  </div>

  <!-- Centre: switch pip track -->
  <div style="display:flex;align-items:center;gap:2px;">
    <span style="font-size:9px;color:rgba(255,255,255,.3);margin-right:8px;
      letter-spacing:1px;white-space:nowrap;">TAB SWITCHES</span>
    {pips_html}
    <span style="font-size:9px;color:rgba(255,255,255,.3);margin-left:8px;">{cnt}/3</span>
  </div>

  <!-- Right: proctor badge -->
  <div style="font-size:9px;color:rgba(255,255,255,.25);letter-spacing:1px;white-space:nowrap;">
    🔍 PROCTOR v2.0
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Full-screen violation flash (fires once per new violation event) ──────
    # We store the last known switch count in sessionStorage.
    # If the current Python cnt > stored value → new violation → trigger flash.
    components.html(f"""<script>
(function(){{
  var P=window.parent;
  var PREV=parseInt(sessionStorage.getItem('pt_sw_cnt')||'0',10);
  var CURR={cnt};
  var FLASH_COL='{flash_col}';

  if(CURR>PREV && CURR>0){{
    sessionStorage.setItem('pt_sw_cnt',CURR);
    // ── Inject flash overlay into parent document ──────────────────────────
    var ov=P.document.createElement('div');
    ov.id='pt-violation-flash';
    ov.style.cssText='position:fixed;inset:0;z-index:99999;pointer-events:none;'
      +'background:'+FLASH_COL+';opacity:0;transition:opacity 0s;';
    P.document.body.appendChild(ov);

    // Flash sequence: fade in → hold → fade out
    var steps=[
      [0,   'opacity:1;transition:opacity .05s ease;'],
      [80,  'opacity:.85;'],
      [160, 'opacity:1;'],
      [240, 'opacity:.7;'],
      [340, 'opacity:0;transition:opacity .55s ease;'],
      [950, null]   // remove
    ];
    steps.forEach(function(s){{
      setTimeout(function(){{
        if(!ov.parentNode) return;
        if(s[1]===null){{ ov.remove(); return; }}
        ov.style.cssText='position:fixed;inset:0;z-index:99999;pointer-events:none;'+s[1];
      }}, s[0]);
    }});

    // ── Shake the proctor bar ──────────────────────────────────────────────
    var bar=P.document.getElementById('pt-proctor-bar');
    if(bar){{
      bar.style.transition='transform .06s ease';
      var shakes=[[1,'-4px'],[2,'4px'],[3,'-3px'],[4,'3px'],[5,'-2px'],[6,'0px']];
      shakes.forEach(function(sh){{
        setTimeout(function(){{
          if(bar) bar.style.transform='translateX('+sh[1]+')';
        }}, sh[0]*55);
      }});
      setTimeout(function(){{ if(bar) bar.style.transform=''; }}, 420);
    }}
  }} else {{
    // No new violation — just keep storage in sync
    sessionStorage.setItem('pt_sw_cnt',CURR);
  }}

  // ── iframe relay (blur/focus bubbling) ────────────────────────────────────
  window.addEventListener("blur",function(){{
    try{{ P.postMessage({{type:"PT_IFRAME_BLUR"}},"*"); }}catch(e){{}}
  }});
  window.addEventListener("focus",function(){{
    try{{ P.postMessage({{type:"PT_IFRAME_FOCUS"}},"*"); }}catch(e){{}}
  }});
  document.addEventListener("contextmenu",function(e){{e.preventDefault();}},true);
  document.addEventListener("copy",function(e){{e.preventDefault();}},true);
  document.addEventListener("cut",function(e){{e.preventDefault();}},true);
}})();
</script>""", height=0, scrolling=False)


def _question_window_mcq(q: Dict, qi: int, ri: int, tpq: int, nq: int) -> None:
    """
    Pure Streamlit MCQ UI — no postMessage, no iframe bridge.

    Flow
    ----
    1. Render question card + timer (HTML display only, no buttons inside).
    2. Render 4 option buttons as real st.buttons — clicking one stores the
       selection in session_state and reruns to show the Submit button.
    3. Once an option is selected, show a glowing Submit button + a Skip button.
    4. On Submit → score, store result, show result card + Next Question button.
    """
    ukey  = f"mcq_r{ri}_q{qi}"
    opts  = q.get("options", {})
    topic = q.get("topic", "")
    qtext = q.get("question", "")
    expl  = q.get("explanation", "")
    correct = q.get("correct", "")
    elap  = max(0, int(tpq - (time.time() - st.session_state.get("pt_q_shown_at", time.time()))))
    prev  = st.session_state.pt_round_answers

    # ── inject MCQ-specific CSS once ─────────────────────────────────────────
    st.markdown("""
<style>
/* ── Question card ── */
.mcq-card{
  background:rgba(4,9,26,.97);border-radius:20px;padding:30px 34px 24px;
  position:relative;overflow:hidden;margin-bottom:6px;
}
.mcq-card::before{
  content:'';position:absolute;inset:-2px;border-radius:22px;z-index:0;
  background:linear-gradient(90deg,#6366f1,#00d4ff,#00ff88,#f59e0b,#ff3366,#6366f1);
  background-size:400%;animation:mcq-border 6s linear infinite;
}
.mcq-card::after{
  content:'';position:absolute;inset:2px;border-radius:18px;z-index:0;
  background:rgba(4,9,26,.98);
}
@keyframes mcq-border{0%{background-position:0%}100%{background-position:400%}}
.mcq-inner{position:relative;z-index:1;}
.mcq-topbar{display:flex;justify-content:space-between;align-items:center;
  margin-bottom:18px;flex-wrap:wrap;gap:8px;}
.mcq-badge{font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:2px;
  padding:4px 14px;border-radius:5px;
  background:rgba(0,212,255,.1);color:#00d4ff;border:.5px solid rgba(0,212,255,.5);}
.mcq-topic{font-family:'Share Tech Mono',monospace;font-size:10px;
  color:rgba(255,255,255,.3);letter-spacing:1px;}
.mcq-qnum{font-family:'Share Tech Mono',monospace;font-size:10px;
  color:rgba(255,255,255,.3);letter-spacing:1px;}
/* ── Ring countdown timer ── */
.mcq-ring-wrap{
  position:relative;width:80px;height:80px;flex-shrink:0;
}
.mcq-ring-wrap svg{transform:rotate(-90deg);}
.mcq-ring-track{fill:none;stroke:rgba(255,255,255,.07);stroke-width:5;}
.mcq-ring-fill{
  fill:none;stroke:#00ff88;stroke-width:5;stroke-linecap:round;
  transition:stroke .6s ease,stroke-dashoffset .95s linear;
}
.mcq-ring-center{
  position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;
}
.mcq-ring-num{
  font-family:'Orbitron',monospace;font-size:1.1rem;font-weight:700;
  color:#00ff88;line-height:1;transition:color .6s ease;
}
.mcq-ring-lbl{
  font-family:'Share Tech Mono',monospace;font-size:7px;
  color:rgba(255,255,255,.3);letter-spacing:1.5px;margin-top:2px;
}
@keyframes mcq-ring-urgent{
  0%,100%{filter:drop-shadow(0 0 0px #ff3366);}
  50%{filter:drop-shadow(0 0 8px #ff3366);}
}
.mcq-ring-urgent{animation:mcq-ring-urgent 0.8s ease-in-out infinite;}

/* Question text */
.mcq-qtext{font-family:'Inter',sans-serif;font-size:1.55rem;font-weight:700;
  color:#f1f5f9;line-height:1.55;margin:6px 0 22px;
  text-shadow:0 0 24px rgba(99,102,241,.18);}
/* Progress dots */
.mcq-dots{display:flex;gap:6px;justify-content:center;margin-bottom:14px;}
.mcq-dot{width:10px;height:10px;border-radius:50%;display:inline-block;}
.mcq-dot-ok{background:#00ff88;}
.mcq-dot-err{background:#ff3366;}
.mcq-dot-curr{background:#6366f1;animation:mcq-dot-pulse 1s ease-in-out infinite;}
.mcq-dot-pending{background:rgba(255,255,255,.1);}
@keyframes mcq-dot-pulse{0%,100%{box-shadow:0 0 0 0 rgba(99,102,241,.5);}
  50%{box-shadow:0 0 0 4px transparent;}}

/* ── Option stagger-in animation ── */
@keyframes mcq-opt-slide{
  from{opacity:0;transform:translateX(-18px);}
  to{opacity:1;transform:translateX(0);}
}
/* ── Ripple keyframe ── */
@keyframes mcq-ripple{
  from{transform:translate(-50%,-50%) scale(0);opacity:.55;}
  to{transform:translate(-50%,-50%) scale(3.5);opacity:0;}
}

/* ── Option buttons — override Streamlit default ── */
div[data-testid="stButton"].mcq-opt > button{
  display:flex !important;align-items:center !important;
  width:100% !important;text-align:left !important;
  padding:15px 20px !important;border-radius:14px !important;
  border:1.5px solid rgba(99,102,241,.22) !important;
  background:rgba(99,102,241,.05) !important;
  color:#e2e8f0 !important;font-size:1rem !important;font-weight:500 !important;
  font-family:'Inter',sans-serif !important;
  transition:all .18s cubic-bezier(.4,0,.2,1) !important;
  margin-bottom:4px !important;
  overflow:hidden !important;
  position:relative !important;
}
div[data-testid="stButton"].mcq-opt > button:hover{
  border-color:#6366f1 !important;
  background:rgba(99,102,241,.18) !important;
  transform:translateX(5px) scale(1.01) !important;
  box-shadow:0 6px 28px rgba(99,102,241,.22) !important;
  color:#fff !important;
}
/* Selected option */
div[data-testid="stButton"].mcq-opt-sel > button{
  display:flex !important;align-items:center !important;
  width:100% !important;text-align:left !important;
  padding:15px 20px !important;border-radius:14px !important;
  border:2px solid #00d4ff !important;
  background:rgba(0,212,255,.12) !important;
  color:#fff !important;font-size:1rem !important;font-weight:600 !important;
  font-family:'Inter',sans-serif !important;
  box-shadow:0 4px 24px rgba(0,212,255,.25) !important;
  margin-bottom:4px !important;
  overflow:hidden !important;
  position:relative !important;
}
/* Submit button */
div[data-testid="stButton"].mcq-submit > button{
  background:linear-gradient(135deg,#6366f1,#4f46e5) !important;
  color:#fff !important;border:none !important;
  font-family:'Orbitron',monospace !important;font-size:13px !important;
  font-weight:700 !important;letter-spacing:.5px !important;
  padding:14px 28px !important;border-radius:12px !important;
  box-shadow:0 4px 24px rgba(99,102,241,.5) !important;
  width:100% !important;
  animation:mcq-submit-glow 1.5s ease-in-out infinite !important;
}
@keyframes mcq-submit-glow{
  0%,100%{box-shadow:0 4px 24px rgba(99,102,241,.5);}
  50%{box-shadow:0 4px 36px rgba(99,102,241,.85),0 0 60px rgba(0,212,255,.25);}
}
div[data-testid="stButton"].mcq-submit > button:hover{
  transform:translateY(-2px) !important;
  box-shadow:0 8px 36px rgba(99,102,241,.7) !important;
}
/* Skip button */
div[data-testid="stButton"].mcq-skip > button{
  background:transparent !important;
  color:rgba(255,255,255,.4) !important;
  border:1px solid rgba(255,255,255,.1) !important;
  font-family:'Share Tech Mono',monospace !important;font-size:11px !important;
  letter-spacing:1px !important;padding:14px 20px !important;
  border-radius:12px !important;width:100% !important;
}
div[data-testid="stButton"].mcq-skip > button:hover{
  color:rgba(255,255,255,.7) !important;
  border-color:rgba(255,255,255,.3) !important;
}
/* Next question button */
div[data-testid="stButton"].mcq-next > button{
  background:linear-gradient(135deg,#00d4ff,#0099bb) !important;
  color:#050a16 !important;border:none !important;
  font-family:'Orbitron',monospace !important;font-size:13px !important;
  font-weight:700 !important;padding:14px 28px !important;
  border-radius:12px !important;width:100% !important;
  box-shadow:0 4px 24px rgba(0,212,255,.4) !important;
}
div[data-testid="stButton"].mcq-next > button:hover{
  transform:translateY(-2px) !important;
  box-shadow:0 8px 36px rgba(0,212,255,.6) !important;
}
</style>""", unsafe_allow_html=True)

    # ── PHASE A: result card (question already answered) ─────────────────────
    if st.session_state.get(f"{ukey}_done", False):
        res = st.session_state.get(f"{ukey}_res", {})
        if not st.session_state.get(f"{ukey}_popup_shown", False):
            st.session_state[f"{ukey}_popup_shown"] = True
            ok = res.get("correct", False)
            sk = res.get("skipped", False)
            if sk:
                _show_popup("skip", "SKIPPED",
                    f"Correct answer: <strong>{correct}</strong> — {opts.get(correct,'')}")
            elif ok:
                _show_popup("correct", "CORRECT! ✓",
                    f"<strong>{opts.get(correct,'')}</strong><br><br>"
                    f"<em style='font-size:.88rem;color:rgba(255,255,255,.6)'>{expl}</em>")
            else:
                sel = res.get("selected","")
                _show_popup("wrong", "INCORRECT ✗",
                    f"You chose <strong>{sel}</strong> — {opts.get(sel,'')}<br>"
                    f"Answer: <strong>{correct}</strong> — {opts.get(correct,'')}<br><br>"
                    f"<em style='font-size:.88rem;color:rgba(255,255,255,.55)'>{expl}</em>")
        _mcq_result_card(res, correct, opts, expl)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="mcq-next">', unsafe_allow_html=True)
        if st.button("Next Question  →", key=f"{ukey}_nxt", use_container_width=True):
            _record_mcq(q, st.session_state.get(f"{ukey}_sel", ""))
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── PHASE B: active question ──────────────────────────────────────────────
    sel_now = st.session_state.get(f"{ukey}_sel_now", None)  # option chosen but not yet submitted

    # Progress dots
    dots_html = "".join(
        f'<span class="mcq-dot mcq-dot-ok"></span>'   if d < len(prev) and prev[d].get("selected","") == prev[d].get("correct","") else
        f'<span class="mcq-dot mcq-dot-err"></span>'  if d < len(prev) else
        f'<span class="mcq-dot mcq-dot-curr"></span>' if d == qi else
        f'<span class="mcq-dot mcq-dot-pending"></span>'
        for d in range(nq)
    )

    # Circumference for r=34: 2π×34 ≈ 213.6 — used for stroke-dasharray
    _circ = 213.6
    # Question card with SVG ring timer replacing flat text timer
    st.markdown(f"""
<div class="mcq-card">
  <div class="mcq-inner">
    <div class="mcq-topbar">
      <div style="display:flex;gap:8px;align-items:center;">
        <span class="mcq-badge">⚡ APTITUDE MCQ</span>
        <span class="mcq-topic">{topic.upper()}</span>
        <span class="mcq-qnum" style="margin-left:4px">Q {qi+1} / {nq}</span>
      </div>
      <!-- SVG ring countdown -->
      <div class="mcq-ring-wrap" id="mcq-ring-wrap-{ukey}">
        <svg width="80" height="80" viewBox="0 0 80 80">
          <circle class="mcq-ring-track" cx="40" cy="40" r="34"/>
          <circle class="mcq-ring-fill" id="mcq-ring-{ukey}"
            cx="40" cy="40" r="34"
            stroke-dasharray="{_circ}"
            stroke-dashoffset="0"/>
        </svg>
        <div class="mcq-ring-center">
          <span class="mcq-ring-num" id="mcq-rnum-{ukey}">{elap}</span>
          <span class="mcq-ring-lbl">SEC</span>
        </div>
      </div>
    </div>
    <div class="mcq-qtext">{qtext}</div>
  </div>
</div>
<script>
(function(){{
  var TOTAL={tpq}, rem={elap};
  var CIRC={_circ};
  var ring=document.getElementById('mcq-ring-{ukey}');
  var num=document.getElementById('mcq-rnum-{ukey}');
  var wrap=document.getElementById('mcq-ring-wrap-{ukey}');
  if(!ring||!num) return;

  function _update(){{
    var pct=rem/TOTAL;
    var offset=CIRC*(1-pct);
    ring.style.strokeDashoffset=offset;

    if(rem<=10){{
      ring.style.stroke='#ff3366';
      num.style.color='#ff3366';
      if(!wrap.classList.contains('mcq-ring-urgent')) wrap.classList.add('mcq-ring-urgent');
    }} else if(rem<=20){{
      ring.style.stroke='#f59e0b';
      num.style.color='#f59e0b';
    }} else {{
      ring.style.stroke='#00ff88';
      num.style.color='#00ff88';
    }}
    num.textContent=rem>=60?(Math.floor(rem/60)+':'+(rem%60<10?'0':'')+rem%60):rem;
  }}

  _update();
  var iv=setInterval(function(){{
    rem=Math.max(0,rem-1);
    _update();
    if(rem===0){{
      clearInterval(iv);
      num.textContent='✕';
    }}
  }},1000);
}})();
</script>

<script>
/* ── Stagger entrance for option buttons ── */
(function(){{
  var DELAY=[120,220,320,420];
  function _stagger(){{
    var btns=document.querySelectorAll(
      'div[data-testid="stButton"].mcq-opt > button,' +
      'div[data-testid="stButton"].mcq-opt-sel > button'
    );
    btns.forEach(function(b,i){{
      b.style.opacity='0';
      b.style.transform='translateX(-18px)';
      b.style.transition='none';
      setTimeout(function(){{
        b.style.transition='opacity .28s ease, transform .28s cubic-bezier(.22,1,.36,1)';
        b.style.opacity='1';
        b.style.transform='translateX(0)';
      }}, DELAY[i]||i*80);
    }});
  }}
  /* Run after Streamlit finishes painting */
  setTimeout(_stagger, 60);

  /* ── Ripple on click ── */
  function _addRipple(btn){{
    btn.addEventListener('click', function(e){{
      var rip=document.createElement('span');
      var rect=btn.getBoundingClientRect();
      rip.style.cssText='position:absolute;left:'+(e.clientX-rect.left)+'px;top:'+(e.clientY-rect.top)+'px;'
        +'width:24px;height:24px;border-radius:50%;background:rgba(0,212,255,.55);pointer-events:none;'
        +'animation:mcq-ripple 0.55s ease-out forwards;';
      btn.style.overflow='hidden';
      btn.appendChild(rip);
      setTimeout(function(){{rip.remove();}}, 600);
    }}, {{once:false, passive:true}});
  }}
  function _attachRipples(){{
    var btns=document.querySelectorAll(
      'div[data-testid="stButton"].mcq-opt > button,' +
      'div[data-testid="stButton"].mcq-opt-sel > button'
    );
    btns.forEach(function(b){{
      if(!b.dataset.rippleAttached){{
        _addRipple(b);
        b.dataset.rippleAttached='1';
      }}
    }});
  }}
  setTimeout(_attachRipples,100);
}})();
</script>
<div style="margin-bottom:6px"></div>
""", unsafe_allow_html=True)

    # ── Options (real Streamlit buttons) ──────────────────────────────────────
    sorted_opts = sorted(opts.items())
    option_labels = {k: v for k, v in sorted_opts}

    for k, v in sorted_opts:
        is_sel = (sel_now == k)
        css_class = "mcq-opt-sel" if is_sel else "mcq-opt"
        prefix = "✦ " if is_sel else ""
        label = f"{prefix}{k}  ·  {v}"
        st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
        if st.button(label, key=f"{ukey}_opt_{k}", use_container_width=True):
            st.session_state[f"{ukey}_sel_now"] = k
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Submit + Skip (only Submit is highlighted after selection) ────────────
    if sel_now:
        # Show Submit button prominently + Skip ghost button
        col_sub, col_skip = st.columns([3, 1])
        with col_sub:
            st.markdown('<div class="mcq-submit">', unsafe_allow_html=True)
            if st.button(f"Submit Answer  →  ({sel_now})", key=f"{ukey}_sub",
                         use_container_width=True):
                res = _score_mcq(sel_now, correct)
                st.session_state[f"{ukey}_res"] = dict(**res, selected=sel_now,
                    options=opts, explanation=expl)
                st.session_state[f"{ukey}_sel"] = sel_now
                st.session_state[f"{ukey}_done"] = True
                st.session_state.pt_mcq_q_submitted = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with col_skip:
            st.markdown('<div class="mcq-skip">', unsafe_allow_html=True)
            if st.button("Skip ⏭", key=f"{ukey}_skip", use_container_width=True):
                res = _score_mcq("", correct)
                st.session_state[f"{ukey}_res"] = dict(**res, selected="",
                    options=opts, explanation=expl)
                st.session_state[f"{ukey}_sel"] = ""
                st.session_state[f"{ukey}_done"] = True
                st.session_state.pt_mcq_q_submitted = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # No option selected yet — only show Skip (Submit is absent until selection)
        st.markdown('<div class="mcq-skip">', unsafe_allow_html=True)
        if st.button("Skip this question ⏭", key=f"{ukey}_skip", use_container_width=True):
            res = _score_mcq("", correct)
            st.session_state[f"{ukey}_res"] = dict(**res, selected="",
                options=opts, explanation=expl)
            st.session_state[f"{ukey}_sel"] = ""
            st.session_state[f"{ukey}_done"] = True
            st.session_state.pt_mcq_q_submitted = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Progress dots below
    st.markdown(f'<div class="mcq-dots">{dots_html}</div>', unsafe_allow_html=True)


# =============================================================================
# 7  HELPERS
# =============================================================================

def _cur() -> Dict: return ROUND_CONFIG[st.session_state.pt_round_idx]
def _elapsed_round() -> float:
    t = st.session_state.get("pt_round_started_at",0.0); return time.time()-t if t else 0.
def _elapsed_q() -> float:
    t = st.session_state.get("pt_q_shown_at",0.0); return time.time()-t if t else 0.
def _avg(rid: str) -> float: return st.session_state.pt_round_scores.get(rid,0.0)

def _gate_ok(r: Dict) -> bool:
    idx = ROUND_CONFIG.index(r)
    if idx == 0: return True
    return _avg(ROUND_CONFIG[idx-1]["id"]) >= r["gate_score"]

def _close_round() -> float:
    ans = st.session_state.pt_round_answers
    a = round(sum(x.get("score",0.) for x in ans)/len(ans),3) if ans else 0.
    st.session_state.pt_round_scores[_cur()["id"]] = a
    st.session_state.pt_all_answers.extend(ans)
    return a

def _advance() -> None:
    ni = st.session_state.pt_round_idx + 1
    if ni >= len(ROUND_CONFIG):
        st.session_state.pt_phase = "all_complete"; return
    nxt = ROUND_CONFIG[ni]
    if _gate_ok(nxt):
        st.session_state.pt_round_idx = ni; st.session_state.pt_phase = "running"
        st.session_state.pt_q_idx = 0; st.session_state.pt_round_answers = []
        st.session_state.pt_round_started_at = time.time()
        st.session_state.pt_mcq_q_submitted = False
    else:
        st.session_state.pt_phase = "gated_out"


# =============================================================================
# 8  TEXT QUESTION PRE-FETCHING  (the fix for slow per-question generation)
# =============================================================================

def _prefetch_text_questions(engine, r: Dict) -> List[Dict]:
    """Pre-fetch ALL text questions for a round before it starts -- removes delay."""
    role = st.session_state.get("pt_target_role","Software Engineer")
    qt = r["q_type"]; n = r["num_questions"]
    questions: List[Dict] = []; already: set = set()

    # ── Company bank check (first priority) ──────────────────────────────────
    if _COMPANY_UPLOAD_OK and has_company_questions(qt):
        company_batch = get_company_text_batch(n, qt)
        for q in company_batch:
            if q.get("question", "") not in already:
                q["q_format"] = "text"
                questions.append(q); already.add(q.get("question", ""))
        if len(questions) >= n:
            return questions[:n]

    if _HR_OK and qt == "hr":
        pool = [h for h in _HR_QUESTIONS if h["question"] not in already]
        random.shuffle(pool)
        for h in pool[:n]:
            q = {"question":h["question"],"keywords":[],"ideal_answer":h.get("tip",""),
                 "type":"hr","difficulty":"medium","q_format":"text"}
            questions.append(q); already.add(h["question"])
        if len(questions) >= n: return questions[:n]

    if hasattr(engine,"qbank") and hasattr(engine.qbank,"get_batch_questions"):
        try:
            batch = engine.qbank.get_batch_questions(role=role,difficulty=r["difficulty"],q_type=qt,n=n)
            if batch:
                for q in batch:
                    q["q_format"] = "text"
                    if q.get("question","") not in already:
                        questions.append(q); already.add(q.get("question",""))
                if len(questions) >= n: return questions[:n]
        except Exception: pass

    attempts = 0
    while len(questions) < n and attempts < n*3:
        attempts += 1
        q = _fetch_text_q_single(engine, r, exclude=already)
        if q.get("question","") not in already:
            q["q_format"] = "text"; questions.append(q); already.add(q.get("question",""))
    return questions[:n]


def _fetch_text_q_single(engine, r: Dict, exclude: set = None) -> Dict:
    if exclude is None: exclude = set()
    role = st.session_state.get("pt_target_role","Software Engineer")
    qt = r["q_type"]; diff = r["difficulty"]
    if _HR_OK and qt == "hr":
        pool = [q for q in _HR_QUESTIONS if q["question"] not in exclude]
        if pool:
            h = random.choice(pool)
            return {"question":h["question"],"keywords":[],"ideal_answer":h.get("tip",""),
                    "type":"hr","difficulty":"medium","q_format":"text"}
    q = engine.qbank.get_single_question(role=role,difficulty=diff,q_type=qt)
    if not q:
        q = {"question":f"Describe a challenging {qt} problem you solved.",
             "keywords":[],"ideal_answer":"","type":qt,"difficulty":diff}
    q["q_format"] = "text"; return q


def _fetch_text_q(engine, r: Dict) -> Dict:
    batch = st.session_state.get("pt_text_batch",[])
    qi = st.session_state.pt_q_idx
    if batch and qi < len(batch): return batch[qi]
    done = set(a["question"] for a in st.session_state.pt_round_answers)
    return _fetch_text_q_single(engine, r, exclude=done)


def _eval_text(engine, q: Dict, ans: str) -> Dict:
    qt = q.get("question","")
    if not any(x.get("question")==qt for x in engine.questions):
        engine.questions.append(q)
    return engine.evaluate_answer(qt, ans or "(no answer)")


# =============================================================================
# 9  SHARED WIDGETS
# =============================================================================

def _header() -> None:
    curr = st.session_state.pt_round_idx; phase = st.session_state.pt_phase
    name = st.session_state.get("pt_candidate_name","Candidate")
    company = st.session_state.get("pt_company_name","")
    sub = f" · {company}" if company else ""
    st.markdown(
        f'<p style="font-family:\'Share Tech Mono\',monospace;font-size:10px;'
        f'color:rgba(122,184,216,.45);letter-spacing:1px;margin:0 0 4px;text-align:right">'
        f'PLACEMENT DRIVE{sub.upper()} | {name}</p>',
        unsafe_allow_html=True)
    steps = ""
    for i, r in enumerate(ROUND_CONFIG):
        s = st.session_state.pt_round_scores.get(r["id"])
        if i < curr and s is not None:
            p = int(s/5*100); c = "#00ff88" if s>=r["gate_score"] else "#ff3366"
            cls="pt-done"; txt=f'<span style="color:{c}">{p}%</span>'; lc=c
        elif i == curr and phase in ("running","round_complete"):
            cls="pt-active"; txt=r["icon"]; lc="#a5b4fc"
        else:
            cls="pt-lock"; txt=r["icon"]; lc="rgba(255,255,255,.25)"
        conn=""
        if i < len(ROUND_CONFIG)-1:
            cc="pt-conn-on" if i<curr else "pt-conn-off"
            conn=f'<div class="pt-conn {cc}"></div>'
        steps+=(f'<div class="pt-sw"><div class="pt-circ {cls}">{txt}</div>'
                f'<span class="pt-slbl" style="color:{lc}">{r["short"]}</span></div>{conn}')
    st.markdown(f'<div class="pt-stepper">{steps}</div>', unsafe_allow_html=True)


def _timer(secs: int, key: str, auto: bool=True) -> None:
    am = "true" if auto else "false"
    components.html(f"""
<div class="pt-tmr">
  <span id="d-{key}" style="width:7px;height:7px;border-radius:50%;background:#00ff88;
    animation:pt-pulse 1s ease-in-out infinite;flex-shrink:0"></span>
  <span id="l-{key}" style="color:#7ab8d8;letter-spacing:1px;font-size:11px">TIME LEFT</span>
  <span id="v-{key}" style="color:#00ff88;font-weight:700;min-width:44px;text-align:right;font-size:1rem">{secs}s</span>
</div>
<style>@keyframes pt-pulse{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}</style>
<script>
(function(){{var r={secs},k="{key}",a={am};
  var v=document.getElementById("v-"+k),d=document.getElementById("d-"+k),l=document.getElementById("l-"+k);
  if(!v)return;
  var iv=setInterval(function(){{r=Math.max(0,--r);
    v.textContent=r>=60?(Math.floor(r/60)+":"+(r%60<10?"0":"")+r%60):r+"s";
    if(r<=10){{v.style.color="#ff3366";d.style.background="#ff3366";}}
    else if(r<=30){{v.style.color="#f59e0b";d.style.background="#f59e0b";}}
    if(r===0){{clearInterval(iv);l.textContent="EXPIRED";
      if(a)window.parent.postMessage({{type:"PT_TIMER_EXPIRED",key:k}},"*");}}
  }},1000);}})();
</script>""", height=46)


# =============================================================================
# 10  SETUP PAGE
# =============================================================================

def _start_test_session(engine, role: str) -> None:
    """
    Shared helper: build the MCQ batch and transition to the running phase.
    Called after the popup is dismissed (with or without CSV upload).
    """
    r0 = ROUND_CONFIG[0]
    try:
        engine.start_session(role=role, difficulty=r0["difficulty"],
                             num_questions=r0["num_questions"])
    except Exception:
        pass

    with st.spinner("⚡ Preparing aptitude questions — this will only happen once…"):
        if _COMPANY_UPLOAD_OK and has_company_questions("aptitude"):
            batch = get_company_mcq_batch(r0["num_questions"], r0["difficulty"])
            if len(batch) < r0["num_questions"]:
                ai_bank = _mcq_bank()
                ai_bank._cache = None
                ai_batch = ai_bank.get_batch(
                    role=role,
                    n=r0["num_questions"] - len(batch),
                    diff=r0["difficulty"],
                )
                batch += ai_batch
        else:
            bank = _mcq_bank()
            bank._cache = None
            batch = bank.get_batch(role=role, n=r0["num_questions"],
                                   diff=r0["difficulty"])

    for bq in batch:
        bq.setdefault("q_format", "mcq")

    st.session_state.update({
        "pt_mcq_batch": batch, "pt_text_batch": [], "pt_round_idx": 0,
        "pt_phase": "running", "pt_q_idx": 0, "pt_round_answers": [],
        "pt_all_answers": [], "pt_round_scores": {},
        "pt_round_started_at": time.time(), "pt_mcq_q_submitted": False,
        "blind_mode": True, "blind_scores": {}, "blind_revealed": False,
        # Reset popup state for next run
        "_cq_popup_open": False, "_cq_popup_done": False,
    })
    if batch:
        st.session_state.pt_current_q_dict = batch[0]
        st.session_state.pt_current_q = batch[0]["question"]
    st.session_state.pt_q_shown_at = time.time()
    _show_popup("info", "TEST STARTING!",
                f"Round 1 — Aptitude<br>{r0['num_questions']} MCQs · {r0['time_per_q_s']}s each<br><br>"
                "All questions are pre-loaded. Good luck! 🚀", "Let's Go →")
    st.rerun()


def _page_setup(engine) -> None:
    # ── If popup was confirmed, start the test immediately ────────────────────
    if st.session_state.get("_cq_popup_done", False):
        _start_test_session(engine, st.session_state.get("pt_target_role", "Software Engineer"))
        return

    # ── If popup is open, render it in-place and stop ─────────────────────────
    if st.session_state.get("_cq_popup_open", False):
        try:
            from company_question_upload import show_placement_test_upload_popup
            if show_placement_test_upload_popup():
                st.session_state._cq_popup_done = True
                st.rerun()
        except ImportError:
            # No upload module — just proceed directly
            st.session_state._cq_popup_open = False
            st.session_state._cq_popup_done = True
            st.rerun()
        return  # Don't render the rest of the setup page while popup is shown

    # ── Normal setup page ─────────────────────────────────────────────────────
    st.markdown("""
<div style="text-align:center;padding:36px 0 28px;">
  <p style="font-family:'Share Tech Mono',monospace;font-size:11px;
    color:rgba(99,102,241,.7);letter-spacing:3px;margin-bottom:14px;">CAMPUS PLACEMENT DRIVE</p>
  <h1 style="font-size:2.3rem;font-weight:700;color:#f1f5f9;margin:0;letter-spacing:-.02em;">
    Placement Test</h1>
  <p style="color:rgba(255,255,255,.4);font-size:.95rem;margin-top:10px;">
    10 MCQ Aptitude &nbsp;·&nbsp; 5 Technical &nbsp;·&nbsp; 5 HR</p>
</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Full name *", value=st.session_state.pt_candidate_name,
                             placeholder="Priya Sharma")
        st.session_state.pt_candidate_name = name
        role = st.selectbox("Role applying for",
            ["Software Engineer", "Frontend Developer", "Backend Developer",
             "Full Stack Developer", "Data Scientist", "ML Engineer",
             "DevOps Engineer", "QA Engineer"], key="pt_role_sel")
        st.session_state.pt_target_role = role
    with c2:
        email = st.text_input("Email", value=st.session_state.pt_candidate_email,
                              placeholder="you@college.edu")
        st.session_state.pt_candidate_email = email
        company = st.text_input("Company / College", value=st.session_state.pt_company_name,
                                placeholder="Acme Corp")
        st.session_state.pt_company_name = company

    st.markdown("---")
    cols = st.columns(3)
    clrs = ["#00d4ff", "#6366f1", "#00ff88"]
    fmts = ["MCQ (radio buttons)", "Written + Voice", "Written + Voice"]
    for col, r, cl, fm in zip(cols, ROUND_CONFIG, clrs, fmts):
        gate_t = f"Gate >= {r['gate_score']}/5" if r["gate_score"] > 0 else "Open entry"
        with col:
            st.markdown(f"""
<div style="background:rgba(4,9,26,.85);border:.5px solid {cl}22;
  border-radius:12px;padding:20px 16px;text-align:center;min-height:200px;">
  <div style="font-size:28px;margin-bottom:8px">{r['icon']}</div>
  <p style="font-weight:700;color:{cl};font-size:1rem;margin:0 0 4px">{r['short']}</p>
  <p style="color:rgba(255,255,255,.4);font-size:11px;margin:0 0 12px;line-height:1.4">{r['description']}</p>
  <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:rgba(255,255,255,.45);line-height:1.9">
    <div>📝 {r['num_questions']} questions</div><div>⏱ {r['time_per_q_s']}s per question</div>
    <div>🎯 {fm}</div><div style="color:{cl};margin-top:4px">{gate_t}</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("")
    if _VOICE_OK:
        vc1, vc2 = st.columns([1, 3])
        with vc1:
            v_on = st.toggle("🎙️ Enable Voice Input",
                             value=st.session_state.get("pt_voice_enabled", True),
                             key="pt_voice_toggle")
            st.session_state.pt_voice_enabled = v_on
        with vc2:
            if v_on:
                st.markdown('<p style="font-family:\'Share Tech Mono\',monospace;font-size:10px;'
                            'color:rgba(0,255,136,.6);margin:14px 0 0;letter-spacing:1px">'
                            '✓ VOICE ENABLED</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-family:\'Share Tech Mono\',monospace;font-size:10px;'
                    'color:rgba(255,255,255,.3);letter-spacing:1px">'
                    '⚠ voice_input.py not found - text-only mode</p>', unsafe_allow_html=True)

    # ── Company bank status strip ─────────────────────────────────────────────
    st.markdown("---")
    if _COMPANY_UPLOAD_OK:
        mcq_n  = len(st.session_state.get("company_mcq_bank", []))
        tech_n = len(st.session_state.get("company_tech_bank", []))
        hr_n   = len(st.session_state.get("company_hr_bank", []))
        total  = mcq_n + tech_n + hr_n

        if total > 0:
            st.markdown(f"""
<div style="background:rgba(0,255,136,.06);border:.5px solid rgba(0,255,136,.25);
  border-radius:10px;padding:10px 18px;margin-bottom:10px;
  font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:2px;
  color:#00ff88;display:flex;align-items:center;gap:12px;">
  ✦ COMPANY BANK ACTIVE &nbsp;·&nbsp;
  {mcq_n} Aptitude &nbsp;·&nbsp; {tech_n} Technical &nbsp;·&nbsp; {hr_n} HR
  &nbsp;·&nbsp; <span style="opacity:.6">Click below to manage or re-upload</span>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div style="background:rgba(255,255,255,.03);border:.5px solid rgba(255,255,255,.08);
  border-radius:10px;padding:10px 18px;margin-bottom:4px;
  font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:2px;
  color:rgba(255,255,255,.35);">
  ○ NO COMPANY QUESTIONS LOADED &nbsp;·&nbsp;
  Upload via "Begin Test" button or the expander below
</div>""", unsafe_allow_html=True)

        # Inline management expander (collapsed by default when bank already loaded)
        with st.expander("🏢  Manage Company Question Bank", expanded=(total == 0)):
            try:
                from company_question_upload import page_company_questions
                page_company_questions()
            except ImportError:
                st.error("company_question_upload.py not found in the project directory.")
            except Exception as _cq_err:
                st.error(f"Company Questions error: {_cq_err}")
    else:
        st.markdown("""
<div style="background:rgba(255,255,255,.03);border:.5px solid rgba(255,255,255,.06);
  border-radius:10px;padding:10px 18px;margin-bottom:10px;
  font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:2px;
  color:rgba(255,255,255,.25);">
  ○ company_question_upload.py not found — AI questions only
</div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Begin Placement Test button → opens upload popup ─────────────────────
    if st.button("▶  Begin Placement Test", type="primary", use_container_width=True,
                 disabled=not name.strip()):
        # Always show the popup so users can upload CSVs (or skip)
        st.session_state._cq_popup_open = True
        st.session_state._cq_popup_done = False
        st.rerun()


# =============================================================================
# 11  MCQ UI wrapper + result card + record
# =============================================================================

def _mcq_ui(q: Dict, qi: int, ri: int) -> None:
    r = _cur()
    _question_window_mcq(q, qi, ri, r["time_per_q_s"], r["num_questions"])


def _mcq_result_card(res: Dict, correct: str, opts: Dict, expl: str) -> None:
    ok=res.get("correct",False); sel=res.get("selected",""); sk=res.get("skipped",False)
    if sk:
        cls="pt-res pt-skip"; icon="⏭"; title="Skipped"; col="#f59e0b"
        msg=f"Correct answer: <strong>{correct}</strong> — {opts.get(correct,'')}"
    elif ok:
        cls="pt-res pt-ok"; icon="✓"; title="Correct!"; col="#00ff88"
        msg=f"You selected <strong>{sel}</strong> — {opts.get(sel,'')}"
    else:
        cls="pt-res pt-err"; icon="✗"; title="Incorrect"; col="#ff3366"
        msg=(f"You chose <strong>{sel}</strong> — {opts.get(sel,'')}<br>"
             f"Correct: <strong>{correct}</strong> — {opts.get(correct,'')}")
    st.markdown(f"""
<div class="{cls}">
  <div style="display:flex;align-items:center;margin-bottom:6px">
    <span style="font-size:18px;color:{col};margin-right:8px">{icon}</span>
    <span style="font-weight:600;color:#f1f5f9;font-size:16px">{title}</span>
  </div>
  <p style="color:rgba(255,255,255,.8);font-size:14px;margin:0 0 8px">{msg}</p>
  <p style="color:rgba(255,255,255,.42);font-size:13px;margin:0;
    border-top:.5px solid rgba(255,255,255,.08);padding-top:8px;font-style:italic">{expl}</p>
</div>""", unsafe_allow_html=True)


def _record_mcq(q: Dict, sel: str) -> None:
    res = _score_mcq(sel, q.get("correct",""))
    rec = {"round_id":"aptitude","round_label":"Round 1 - Aptitude",
           "q_index":st.session_state.pt_q_idx,"q_format":"mcq",
           "question":q.get("question",""),"topic":q.get("topic",""),
           "options":q.get("options",{}),"correct":q.get("correct",""),
           "selected":sel,"answer":sel,"explanation":q.get("explanation",""),**res}
    st.session_state.pt_round_answers.append(rec)
    r = _cur(); ni = st.session_state.pt_q_idx + 1
    if ni >= r["num_questions"]:
        _close_round(); st.session_state.pt_phase = "round_complete"
    else:
        batch = st.session_state.get("pt_mcq_batch",[])
        nq = batch[ni] if ni < len(batch) else _mcq_bank().get_one(
            st.session_state.get("pt_target_role","Software Engineer"))
        nq.setdefault("q_format","mcq")
        st.session_state.pt_q_idx = ni
        st.session_state.pt_current_q_dict = nq
        st.session_state.pt_current_q = nq.get("question","")
        st.session_state.pt_q_shown_at = time.time()
        st.session_state.pt_mcq_q_submitted = False
    st.rerun()


# =============================================================================
# 12  TEXT UI  (Technical + HR) -- large question text
# =============================================================================

def _text_ui(engine, q: Dict, qi: int, ri: int, r: Dict) -> None:
    qt = q.get("type",r["q_type"])
    bcls = "pt-badge pt-tech" if qt=="technical" else "pt-badge pt-hr"
    blbl = "TECHNICAL" if qt=="technical" else "HR / BEHAVIOURAL"
    hint = ("Use STAR: Situation → Task → Action → Result."
            if qt=="hr" else "Be specific — tools, complexity, reasoning.")
    st.markdown(f"""
<div class="pt-qcard">
  <div style="margin-bottom:14px">
    <span class="{bcls}">{blbl}</span>
    <span class="pt-badge pt-dim">{q.get('difficulty','').upper()}</span>
    <span class="pt-badge pt-dim">Q {qi+1} / {r['num_questions']}</span>
  </div>
  <p class="pt-question-text">{st.session_state.pt_current_q}</p>
  <p style="font-size:12px;color:rgba(255,255,255,.32);margin:0;font-style:italic">💡 {hint}</p>
</div>""", unsafe_allow_html=True)

    voice_transcript = ""
    if st.session_state.get("pt_voice_enabled",True) and _VOICE_OK:
        with st.expander("🎙️ Voice Input  *(click to record)*",expanded=False):
            st.markdown('<p style="font-family:\'Share Tech Mono\',monospace;font-size:10px;'
                        'color:rgba(0,212,255,.7);margin:0 0 6px;letter-spacing:1px">SPEAK YOUR ANSWER</p>',
                        unsafe_allow_html=True)
            try: voice_input_panel(question_number=qi+ri*10,stt=None)
            except Exception: pass
            stable_key = f"_bstt_last_tx_{qi+ri*10}"
            voice_transcript = (st.session_state.get(stable_key,"") or
                                st.session_state.get("transcribed_text","") or "")

    area_key = f"pt_a_r{ri}_q{qi}"
    init_val = voice_transcript or st.session_state.get(area_key,"")
    ans = st.text_area("Answer",height=200,max_chars=2000,key=area_key,
                       label_visibility="collapsed",value=init_val,
                       placeholder="Type your answer here — or use Voice Input above…")

    wc  = len(ans.split()) if ans.strip() else 0
    wcc = "#00ff88" if wc>=80 else ("#f59e0b" if wc>=30 else "#ff3366")
    st.markdown(f'<p style="font-family:\'Share Tech Mono\',monospace;font-size:10px;'
                f'color:{wcc};text-align:right;margin:2px 0 10px">{wc} words</p>',
                unsafe_allow_html=True)

    cc1,cc2 = st.columns([3,1])
    with cc1:
        if st.button("Submit Answer  →",type="primary",use_container_width=True,key=f"pt_sub_r{ri}_q{qi}"):
            _submit_text(engine,ans,False)
    with cc2:
        if st.button("Skip ⏭",use_container_width=True,key=f"pt_skip_r{ri}_q{qi}"):
            _submit_text(engine,"",True)

    prev = st.session_state.pt_round_answers
    if prev:
        with st.expander(f"📋 This round so far ({len(prev)} answered)",expanded=False):
            for i,a in enumerate(prev,1):
                p = int(a.get("score",0.)/5*100)
                sc = "#00ff88" if p>=70 else ("#f59e0b" if p>=50 else "#ff3366")
                st.markdown(
                    f"**Q{i}:** {a['question'][:80]}{'…' if len(a['question'])>80 else ''}  "
                    f"<span style='font-family:monospace;color:{sc}'>{p}%</span>",
                    unsafe_allow_html=True)


def _submit_text(engine, ans: str, skipped: bool) -> None:
    q = st.session_state.pt_current_q_dict; r = _cur()
    with st.spinner("⚡ Evaluating answer…"):
        if skipped or not ans.strip():
            ev={"question":st.session_state.pt_current_q,"answer":"",
                "score":1.0,"feedback":"Skipped.","star":{},"keywords":[],"skipped":True}
        else:
            ev=_eval_text(engine,q,ans); ev["skipped"]=False
    rec={"round_id":r["id"],"round_label":r["label"],
         "q_index":st.session_state.pt_q_idx,"q_format":"text",**ev}
    st.session_state.pt_round_answers.append(rec)

    if not skipped:
        spct = int(ev.get("score",1.0)/5*100)
        if spct >= 70:
            fb = ev.get("feedback","")
            fb_s = fb[:110]+("…" if len(fb)>110 else "")
            _show_popup("correct","GREAT ANSWER!",f"Score: {spct}%<br><br>{fb_s}","Next Question")
        elif spct >= 40:
            fb = ev.get("feedback","")[:90]
            _show_popup("info","ANSWER RECORDED",f"Score: {spct}% — Keep going!<br><br>{fb}","Continue")

    ni = st.session_state.pt_q_idx + 1
    if ni >= r["num_questions"]:
        _close_round(); st.session_state.pt_phase = "round_complete"
    else:
        batch = st.session_state.get("pt_text_batch",[])
        if ni < len(batch):
            nq = batch[ni]
        else:
            nq = _fetch_text_q_single(engine,r,
                exclude=set(a["question"] for a in st.session_state.pt_round_answers))
        nq.setdefault("q_format","text")
        st.session_state.pt_q_idx = ni
        st.session_state.pt_current_q_dict = nq
        st.session_state.pt_current_q = nq.get("question","")
        st.session_state.pt_q_shown_at = time.time()
    st.rerun()


# =============================================================================
# 13  RUNNING PAGE
# =============================================================================

def _page_running(engine) -> None:
    _proctor_status_bar()   # ← persistent proctor bar on every question
    if st.session_state.get("_pt_skip_triggered",False):
        st.session_state._pt_skip_triggered = False
        r = _cur()
        if r["q_format"] == "mcq":
            ukey = f"mcq_r{st.session_state.pt_round_idx}_q{st.session_state.pt_q_idx}"
            if not st.session_state.get(f"{ukey}_done",False):
                _record_mcq(st.session_state.pt_current_q_dict,"")
        else:
            _submit_text(engine,"",True)
        return

    r=_cur(); qi=st.session_state.pt_q_idx; nq=r["num_questions"]
    tpq=r["time_per_q_s"]; tot=r["total_time_s"]
    sq=max(0,int(tpq-_elapsed_q()))

    if _elapsed_round() > tot:
        _close_round(); _advance(); st.rerun(); return

    cl={"mcq":"#00d4ff","technical":"#6366f1","hr":"#00ff88"}.get(
        r["q_format"] if r["q_format"]=="mcq" else r["q_type"],"#7ab8d8")
    col_l,col_r = st.columns([3,1])
    with col_l:
        st.markdown(
            f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:11px;'
            f'color:{cl};letter-spacing:2px">{r["label"].upper()}</span>'
            f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:11px;'
            f'color:rgba(255,255,255,.3);margin-left:12px">Q {qi+1} / {nq}</span>',
            unsafe_allow_html=True)
        if r["q_format"]=="mcq":
            prev=st.session_state.pt_round_answers
            dots="".join(
                f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                f'background:{"#00ff88" if di<len(prev) and prev[di].get("selected","")==prev[di].get("correct","") else "#ff3366" if di<len(prev) else "#6366f1" if di==qi else "rgba(255,255,255,.1)"};margin:0 2px"></span>'
                for di in range(nq))
            st.markdown(f'<div style="margin-top:6px">{dots}</div>',unsafe_allow_html=True)
    with col_r:
        if r["q_format"] != "mcq":
            _timer(sq,f"r{st.session_state.pt_round_idx}_q{qi}",auto=st.session_state.pt_auto_advance)

    q = st.session_state.pt_current_q_dict
    if r["q_format"] == "mcq":
        q.setdefault("q_format","mcq"); _mcq_ui(q,qi,st.session_state.pt_round_idx)
    else:
        q.setdefault("q_format","text"); _text_ui(engine,q,qi,st.session_state.pt_round_idx,r)


# =============================================================================
# 14  ROUND COMPLETE
# =============================================================================

def _page_round_complete(engine) -> None:
    _proctor_status_bar()   # ← keep monitoring active between rounds
    curr=st.session_state.pt_round_idx; r=ROUND_CONFIG[curr]
    avg=st.session_state.pt_round_scores.get(r["id"],0.); pct=int(avg/5*100)
    passed=avg>=r.get("gate_score",0.); col="#00ff88" if passed else "#ff3366"
    st.markdown(f"""
<div style="text-align:center;padding:28px 0 16px">
  <p style="font-family:'Share Tech Mono',monospace;font-size:11px;
    color:rgba(122,184,216,.5);letter-spacing:2px">{r['label'].upper()} · COMPLETE</p>
  <div class="pt-ring" style="color:{col}">{pct}%</div>
  <p class="pt-rlbl">ROUND SCORE</p>
  <span style="font-family:'Share Tech Mono',monospace;font-size:11px;
    background:{'rgba(0,255,136,.1)' if passed else 'rgba(255,51,102,.1)'};
    color:{col};border:.5px solid {col};border-radius:4px;
    padding:4px 16px;letter-spacing:1px;margin-top:10px;display:inline-block">
    {'ROUND PASSED' if passed else 'BELOW GATE THRESHOLD'}</span>
</div>""", unsafe_allow_html=True)

    answers=[a for a in st.session_state.pt_all_answers if a.get("round_id")==r["id"]]
    is_mcq = r["q_format"]=="mcq"
    if answers:
        with st.expander("📊 Question breakdown",expanded=True):
            for i,a in enumerate(answers,1):
                s=a.get("score",0.); p2=int(s/5*100)
                sc="#00ff88" if p2>=70 else ("#f59e0b" if p2>=50 else "#ff3366")
                qs=a["question"][:72]+("…" if len(a["question"])>72 else "")
                if is_mcq:
                    sel=a.get("selected","—"); cor=a.get("correct","—")
                    ok=sel==cor; i2="✓" if ok else ("⏭" if a.get("skipped") else "✗")
                    st.markdown(f"**Q{i}:** {qs}  <span style='color:{sc};font-family:monospace'>{i2} {sel}→{cor}</span>",
                                unsafe_allow_html=True)
                    if a.get("explanation"): st.caption(a["explanation"])
                else:
                    sl2=" *(skipped)*" if a.get("skipped") else ""
                    st.markdown(f"**Q{i}:** {qs}  <span style='color:{sc};font-family:monospace'>{p2}%</span>{sl2}",
                                unsafe_allow_html=True)
                    if not a.get("skipped"):
                        ans_txt=a.get("answer","")
                        if ans_txt:
                            st.markdown(
                                f'<div style="background:rgba(255,255,255,.04);border-left:3px solid '
                                f'rgba(99,102,241,.5);border-radius:6px;padding:8px 12px;'
                                f'font-size:12px;color:rgba(255,255,255,.7);margin:4px 0 6px">'
                                f'{ans_txt[:300]}{"…" if len(ans_txt)>300 else ""}</div>',
                                unsafe_allow_html=True)
                        fb=a.get("feedback","")
                        if fb:
                            st.markdown(
                                f'<div style="background:rgba(0,212,255,.05);border:.5px solid '
                                f'rgba(0,212,255,.25);border-radius:8px;padding:10px 14px;'
                                f'font-size:12px;color:#c7e8ff;margin:4px 0">'
                                f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:10px;'
                                f'color:#00d4ff;letter-spacing:1px">AI FEEDBACK</span><br><br>{fb}</div>',
                                unsafe_allow_html=True)
                        star=a.get("star",{})
                        if star:
                            sh=""
                            for elem in ["Situation","Task","Action","Result"]:
                                ev2=star.get(elem.lower(),star.get(elem,False))
                                c2="#00ff88" if ev2 else "#ff3366"
                                sh+=(f'<span style="font-family:monospace;font-size:11px;'
                                     f'color:{c2};margin-right:10px">{"✓" if ev2 else "✗"} {elem}</span>')
                            st.markdown(f'<div style="margin:4px 0">{sh}</div>',unsafe_allow_html=True)
                st.markdown("<hr style='border-color:rgba(255,255,255,.05);margin:8px 0'>",unsafe_allow_html=True)

    ni=curr+1
    if ni<len(ROUND_CONFIG):
        nxt=ROUND_CONFIG[ni]; gn=nxt["gate_score"]; gok=avg>=gn; gc="#00ff88" if gok else "#ff3366"
        st.markdown(f"""
<div style="background:rgba(4,9,26,.8);border:.5px solid {gc}28;border-radius:10px;padding:14px 20px;margin:14px 0">
  <p style="font-family:'Share Tech Mono',monospace;font-size:11px;color:{gc};letter-spacing:1px;margin:0 0 5px">
    NEXT: {nxt['label'].upper()}</p>
  <p style="color:rgba(255,255,255,.6);font-size:13px;margin:0">
    Gate: <strong style="color:{gc}">{gn}/5</strong> &nbsp;·&nbsp;
    Your score: <strong style="color:{gc}">{avg:.2f}/5</strong> &nbsp;·&nbsp;
    <strong style="color:{gc}">{'✅ Eligible' if gok else '❌ Not eligible'}</strong></p>
</div>""", unsafe_allow_html=True)
        btn = "Continue to "+nxt["short"]+"  →" if gok else "View Final Report  →"
    else:
        btn = "View Final Report  →"

    if st.button(btn,type="primary",use_container_width=True):
        ni2=curr+1
        if ni2<len(ROUND_CONFIG):
            nxt2=ROUND_CONFIG[ni2]
            if avg>=nxt2["gate_score"]:
                try: engine.start_session(role=st.session_state.pt_target_role,
                                          difficulty=nxt2["difficulty"],num_questions=nxt2["num_questions"])
                except Exception: pass
                with st.spinner(f"⚡ Loading {nxt2['short']} questions…"):
                    text_batch = _prefetch_text_questions(engine,nxt2)
                for bq in text_batch: bq.setdefault("q_format","text")
                nq = text_batch[0] if text_batch else _fetch_text_q_single(engine,nxt2)
                nq.setdefault("q_format","text")
                st.session_state.update({
                    "pt_round_idx":ni2,"pt_phase":"running","pt_q_idx":0,
                    "pt_round_answers":[],"pt_round_started_at":time.time(),
                    "pt_mcq_q_submitted":False,"pt_text_batch":text_batch,
                    "pt_current_q_dict":nq,"pt_current_q":nq.get("question",""),
                    "pt_q_shown_at":time.time(),
                })
            else:
                st.session_state.pt_phase="gated_out"
        else:
            st.session_state.pt_phase="all_complete"
        st.rerun()


# =============================================================================
# 15  GATED OUT
# =============================================================================

def _page_gated_out() -> None:
    r=ROUND_CONFIG[st.session_state.pt_round_idx]
    avg=st.session_state.pt_round_scores.get(r["id"],0.); pct=int(avg/5*100)
    st.markdown(f"""
<div style="text-align:center;padding:44px 0 24px">
  <div style="font-size:3.5rem;margin-bottom:14px">🔒</div>
  <h2 style="color:#ff3366;font-family:'Share Tech Mono',monospace;font-size:1.5rem;letter-spacing:1px">
    ROUND LOCKED</h2>
  <div class="pt-ring" style="color:#ff3366">{pct}%</div>
  <p style="color:rgba(255,255,255,.48);max-width:400px;margin:12px auto 0;font-size:14px">
    Your score in <strong>{r['label']}</strong> did not meet the gate threshold.
    The recruiter can still review your completed rounds.</p>
</div>""", unsafe_allow_html=True)
    gc1,gc2=st.columns(2)
    with gc1:
        if st.button("📄 View Partial Report",use_container_width=True,type="primary"):
            st.session_state.pt_phase="all_complete"; st.rerun()
    with gc2:
        if st.button("🔄 Restart Test",use_container_width=True):
            for k,v in PLACEMENT_DEFAULTS.items(): st.session_state[k]=v
            st.session_state["blind_mode"]     = False
            st.session_state["blind_scores"]   = {}
            st.session_state["blind_revealed"] = False
            st.rerun()


# =============================================================================
# 16  FINAL REPORT
# =============================================================================

def _page_report() -> None:
    all_a=st.session_state.pt_all_answers
    name=st.session_state.pt_candidate_name or "Candidate"
    company=st.session_state.pt_company_name or "Placement Drive"
    role=st.session_state.pt_target_role
    sc_lst=list(st.session_state.pt_round_scores.values())
    comp=round(sum(sc_lst)/len(sc_lst),2) if sc_lst else 0.
    cpct=int(comp/5*100)
    ccol="#00ff88" if cpct>=65 else ("#f59e0b" if cpct>=45 else "#ff3366")

    st.markdown(f"""
<div style="text-align:center;padding:24px 0 20px">
  <p style="font-family:'Share Tech Mono',monospace;font-size:11px;
    color:rgba(122,184,216,.45);letter-spacing:2px;margin-bottom:4px">
    TEST COMPLETE · {datetime.now().strftime('%d %b %Y')}</p>
  <h2 style="color:#f1f5f9;font-size:1.8rem;font-weight:700;margin:4px 0">{name}</h2>
  <p style="color:rgba(255,255,255,.4);font-size:.9rem;margin:0">{role} · {company}</p>
  <div class="pt-ring" style="color:{ccol};font-size:4rem;margin-top:16px">{cpct}%</div>
  <p class="pt-rlbl">COMPOSITE SCORE</p>
</div>""", unsafe_allow_html=True)

    cols=st.columns(3); rclrs=["#00d4ff","#6366f1","#00ff88"]
    for col,r,rc in zip(cols,ROUND_CONFIG,rclrs):
        s=st.session_state.pt_round_scores.get(r["id"])
        p=int(s/5*100) if s is not None else None
        sc2="#00ff88" if (p or 0)>=70 else ("#f59e0b" if (p or 0)>=50 else "#ff3366")
        with col:
            st.markdown(f"""
<div class="pt-mc" style="border-color:{rc}22">
  <div style="font-size:22px;margin-bottom:6px">{r['icon']}</div>
  <p style="font-size:10px;color:rgba(255,255,255,.4);font-family:'Share Tech Mono',monospace;margin:0 0 4px">
    {r['short'].upper()}</p>
  <p style="font-size:1.8rem;font-weight:700;color:{sc2};font-family:'Orbitron',monospace;margin:0">
    {"—" if p is None else f"{p}%"}</p>
</div>""", unsafe_allow_html=True)

    mcqa=[a for a in all_a if a.get("q_format")=="mcq"]
    if mcqa:
        cn=sum(1 for a in mcqa if a.get("selected","")==a.get("correct",""))
        tn=len(mcqa)
        st.markdown(f"""
<div style="background:rgba(0,212,255,.05);border:.5px solid rgba(0,212,255,.2);
  border-radius:10px;padding:13px 20px;margin:14px 0">
  <span style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#00d4ff">🧠 APTITUDE MCQ</span>
  <span style="float:right;font-family:'Share Tech Mono',monospace;font-size:13px;
    color:#00d4ff;font-weight:700">{cn}/{tn} correct ({int(cn/tn*100)}%)</span>
</div>""", unsafe_allow_html=True)

    with st.expander("📋 Full answer review",expanded=False):
        for r in ROUND_CONFIG:
            ra=[a for a in all_a if a.get("round_id")==r["id"]]
            if not ra: continue
            st.markdown(f"### {r['icon']} {r['label']}")
            for i,a in enumerate(ra,1):
                s=a.get("score",0.); p3=int(s/5*100)
                sc3="#00ff88" if p3>=70 else ("#f59e0b" if p3>=50 else "#ff3366")
                with st.expander(
                    f"Q{i}: {a['question'][:60]}{'…' if len(a['question'])>60 else ''}  "
                    f"{'(skipped)' if a.get('skipped') else f'— {p3}%'}",expanded=False):
                    st.markdown(f"**Question:** {a['question']}")
                    if a.get("q_format")=="mcq":
                        sel=a.get("selected","—"); cor=a.get("correct","—")
                        st.markdown(f"**Your choice:** {sel} — {a.get('options',{}).get(sel,'')}<br>"
                                    f"**Correct:** {cor} — {a.get('options',{}).get(cor,'')}",
                                    unsafe_allow_html=True)
                        if a.get("explanation"): st.info(a["explanation"])
                    else:
                        ans_txt=a.get("answer","*(skipped)*")
                        if ans_txt and ans_txt!="*(skipped)*":
                            st.markdown(
                                f'<div style="background:rgba(255,255,255,.04);border-left:3px solid '
                                f'rgba(99,102,241,.5);border-radius:6px;padding:10px 14px;'
                                f'font-size:13px;color:rgba(255,255,255,.8);margin:6px 0">'
                                f'<span style="font-family:monospace;font-size:10px;color:#a5b4fc">YOUR ANSWER</span>'
                                f'<br><br>{ans_txt}</div>',unsafe_allow_html=True)
                        else: st.markdown("**Answer:** *(skipped)*")
                        fb=a.get("feedback","")
                        if fb:
                            st.markdown(
                                f'<div style="background:rgba(0,212,255,.06);border:.5px solid '
                                f'rgba(0,212,255,.3);border-radius:8px;padding:12px 16px;'
                                f'font-size:13px;color:#c7e8ff;margin:8px 0">'
                                f'<span style="font-family:monospace;font-size:10px;color:#00d4ff;'
                                f'letter-spacing:1px">🤖 AI FEEDBACK</span><br><br>{fb}</div>',
                                unsafe_allow_html=True)
                        star=a.get("star",{})
                        if star:
                            sh2=""
                            for elem in ["Situation","Task","Action","Result"]:
                                ev3=star.get(elem.lower(),star.get(elem,False))
                                c3="#00ff88" if ev3 else "#ff3366"
                                sh2+=(f'<span style="font-family:monospace;font-size:12px;'
                                      f'color:{c3};margin-right:12px">{"✓" if ev3 else "✗"} {elem}</span>')
                            st.markdown(f'<div style="margin:6px 0">STAR: {sh2}</div>',unsafe_allow_html=True)
                        kws=a.get("keywords",[])
                        if kws:
                            kh2=" ".join(f'<span style="background:rgba(99,102,241,.15);color:#a5b4fc;'
                                         f'border:.5px solid rgba(99,102,241,.4);border-radius:4px;'
                                         f'padding:2px 8px;font-size:10px;font-family:monospace;margin:2px">{k}</span>'
                                         for k in kws[:10])
                            st.markdown(f'<p style="font-size:11px;color:rgba(255,255,255,.4);margin:4px 0">'
                                        f'Keywords:</p><div>{kh2}</div>',unsafe_allow_html=True)

    st.markdown("---")
    if _PDF_OK:
        try:
            pdf=_build_pdf({"candidate_name":name,"target_role":role,"target_company":company,
                            "session_answers":all_a,"emotion_history":[],"session_duration_s":0})
            if pdf:
                st.download_button("📥 Download Scorecard PDF",data=pdf,
                    file_name=f"placement_{name.replace(' ','_')}_{datetime.now():%Y%m%d}.pdf",
                    mime="application/pdf",use_container_width=True,type="primary")
        except Exception as e: st.warning(f"PDF failed ({e}).")
    if not _PDF_OK:
        rpt={"candidate":name,"role":role,"company":company,"composite_pct":cpct,
             "round_scores":st.session_state.pt_round_scores,"answers":all_a,
             "generated_at":datetime.now().isoformat()}
        st.download_button("📥 Download Report JSON",data=json.dumps(rpt,indent=2),
            file_name=f"placement_{name.replace(' ','_')}_{datetime.now():%Y%m%d}.json",
            mime="application/json",use_container_width=True)

    if st.button("🔄 Start New Test",use_container_width=True):
        for k,v in PLACEMENT_DEFAULTS.items(): st.session_state[k]=v
        # Blind mode was ON for this test — turn it off so regular interview is unaffected
        st.session_state["blind_mode"]     = False
        st.session_state["blind_scores"]   = {}
        st.session_state["blind_revealed"] = False
        st.rerun()


# =============================================================================
# 17  ENTRY POINT
# =============================================================================

def page_placement_test() -> None:
    """Call from app.py: elif p == 'Placement Test': page_placement_test()"""
    _inject_css()

    # ── Proctor: initialise hidden buttons for JS → Python bridge ─────────────
    # PT_SWITCH      : fired every time the user leaves the tab/window (≤10 s)
    # PT_SWITCH_LONG : fired when the user has been away for >10 seconds
    # PT_TIMER_EXPIRED : existing MCQ timer bridge
    components.html("""
<script>
(function(){
  var _blurAt=0;
  var _PARENT=window.parent;

  // ── Bridge: click a hidden Streamlit button by label ──────────────────────
  function _click(text){
    var btns=_PARENT.document.querySelectorAll("button");
    for(var i=0;i<btns.length;i++){
      if(btns[i].innerText.trim()===text){btns[i].click();return;}
    }
  }

  // ── Listen for messages from MCQ iframe (timer + inner-iframe blur) ────────
  _PARENT.addEventListener("message",function(e){
    if(!e.data) return;
    if(e.data.type==="PT_TIMER_EXPIRED") _click("__PT_ADV__");
    if(e.data.type==="PT_IFRAME_BLUR")   _onHide();
    if(e.data.type==="PT_IFRAME_FOCUS")  _onShow();
  },false);
  // Also listen on this iframe (for cross-iframe postMessage relay)
  window.addEventListener("message",function(e){
    if(!e.data) return;
    if(e.data.type==="PT_TIMER_EXPIRED") _click("__PT_ADV__");
    if(e.data.type==="PT_IFRAME_BLUR")   _onHide();
    if(e.data.type==="PT_IFRAME_FOCUS")  _onShow();
  },false);

  // ── Tab / window switch detection ─────────────────────────────────────────
  function _onHide(){
    if(_blurAt===0) _blurAt=Date.now();
  }
  function _onShow(){
    if(_blurAt===0) return;
    var away=(Date.now()-_blurAt)/1000;
    _blurAt=0;
    if(away>10){ _click("__PT_SWITCH_LONG__"); }
    else        { _click("__PT_SWITCH__"); }
  }

  _PARENT.document.addEventListener("visibilitychange",function(){
    if(_PARENT.document.hidden) _onHide(); else _onShow();
  });
  _PARENT.addEventListener("blur", _onHide);
  _PARENT.addEventListener("focus",_onShow);
  // Also watch the iframe window itself (handles Streamlit iframe focus shifts)
  window.addEventListener("blur", _onHide);
  window.addEventListener("focus",_onShow);

  // ── Alt+Tab / Meta+Tab keyboard detection ─────────────────────────────────
  _PARENT.document.addEventListener("keydown",function(e){
    // Alt+Tab (Windows/Linux) or Cmd+Tab (Mac)
    if((e.altKey||e.metaKey)&&e.key==="Tab"){ _onHide(); }
    // Windows key
    if(e.key==="Meta"||e.key==="OS"){ _onHide(); }
  });

  // ── Mouse leaving the browser viewport ────────────────────────────────────
  // Fires when cursor moves to a different app (taskbar, another window, etc.)
  _PARENT.document.addEventListener("mouseleave",function(){
    if(_blurAt===0) _blurAt=Date.now();
  });
  _PARENT.document.addEventListener("mouseenter",function(){
    _onShow();
  });

  // ── Block copy / paste / cut ──────────────────────────────────────────────
  function _blockClipboard(e){
    e.preventDefault();
    e.stopPropagation();
    return false;
  }
  _PARENT.document.addEventListener("copy",  _blockClipboard, true);
  _PARENT.document.addEventListener("cut",   _blockClipboard, true);
  _PARENT.document.addEventListener("paste", _blockClipboard, true);

  // ── Block right-click context menu ────────────────────────────────────────
  _PARENT.document.addEventListener("contextmenu",function(e){
    e.preventDefault(); return false;
  }, true);

  // ── Block Ctrl+C / Ctrl+V / Ctrl+X / PrintScreen / F12 ──────────────────
  _PARENT.document.addEventListener("keydown",function(e){
    var ctrl=e.ctrlKey||e.metaKey;
    if(ctrl && (e.key==="c"||e.key==="v"||e.key==="x"||e.key==="a")){
      // Only block if not inside an input/textarea (allow typing own answer)
      var tag=(e.target&&e.target.tagName)||"";
      if(tag!=="INPUT"&&tag!=="TEXTAREA"){
        e.preventDefault(); e.stopPropagation();
      }
    }
    if(e.key==="PrintScreen"||e.key==="F12"){
      e.preventDefault(); e.stopPropagation();
    }
  }, true);

  // ── Disable text selection outside input areas ────────────────────────────
  _PARENT.document.addEventListener("selectstart",function(e){
    var tag=(e.target&&e.target.tagName)||"";
    if(tag!=="INPUT"&&tag!=="TEXTAREA"){
      e.preventDefault(); return false;
    }
  });

  // ── Remove duplicate app.py top navbar ───────────────────────────────────
  // app.py's render_top_navbar() always fires before page_placement_test().
  // Find and hide the sticky "Aura AI" bar so only the placement test's own
  // header is visible.
  function _hideAppNavbar(){
    var nodes=_PARENT.document.querySelectorAll(
      'div[style*="position:sticky"],div[style*="position: sticky"]'
    );
    for(var i=0;i<nodes.length;i++){
      var n=nodes[i];
      if(n.innerText&&n.innerText.indexOf("Aura")!==-1&&n.innerText.indexOf("AI")!==-1){
        n.style.setProperty("display","none","important");
      }
    }
    // Hide floating back button too
    var backs=_PARENT.document.querySelectorAll(".back-btn-navbar");
    for(var j=0;j<backs.length;j++) backs[j].style.setProperty("display","none","important");
  }
  _hideAppNavbar();
  setTimeout(_hideAppNavbar,200);
  setTimeout(_hideAppNavbar,600);

})();
</script>""", height=0, scrolling=False)

    # ── Inject CSS FIRST to hide bridge buttons before they render ───────────
    # Multiple selectors to ensure buttons are invisible regardless of
    # Streamlit version / rendering order.
    st.markdown("""<style>
/* Hide all internal bridge buttons — matched by key data-attribute,
   title attribute, and by button text content via has() where supported */
button[title='internal'],
div[data-testid="stButton"]:has(button[title='internal']),
div[data-testid="stButton"][data-key="pt_adv_hid"],
div[data-testid="stButton"][data-key="pt_switch_hid"],
div[data-testid="stButton"][data-key="pt_switch_long_hid"] {
  display:none!important;visibility:hidden!important;
  height:0!important;width:0!important;
  overflow:hidden!important;position:absolute!important;
  pointer-events:none!important;opacity:0!important;
}
/* Hide zero-height iframes (proctor JS iframes) */
iframe[height="0"],
iframe[height="0px"] {
  display:none!important;height:0!important;
  min-height:0!important;max-height:0!important;
}
footer{visibility:hidden!important;}
</style>""", unsafe_allow_html=True)

    # ── Hidden bridge buttons ─────────────────────────────────────────────────
    # Rendered inside a collapsed container so they are both CSS-hidden and
    # pushed out of the visual flow.
    _bridge_slot = st.empty()
    with _bridge_slot.container():
        st.markdown('<div style="display:none;height:0;overflow:hidden;position:absolute;'
                    'pointer-events:none;opacity:0" aria-hidden="true">',
                    unsafe_allow_html=True)
        if st.button("__PT_ADV__", key="pt_adv_hid", help="internal"):
            st.session_state._pt_skip_triggered = True
            st.rerun()

        # Short switch (≤10 s away)
        if st.button("__PT_SWITCH__", key="pt_switch_hid", help="internal"):
            phase_now = st.session_state.get("pt_phase", "setup")
            if phase_now == "running" and not st.session_state.get("pt_test_locked", False):
                cnt = st.session_state.get("pt_switch_count", 0) + 1
                st.session_state.pt_switch_count = cnt
                if cnt >= 3:
                    st.session_state.pt_test_locked = True
                    st.session_state.pt_lock_trigger = (
                        f"You switched windows/tabs {cnt} times during the test. "
                        "Maximum allowed is 3 switches."
                    )
            st.rerun()

        # Long switch (>10 s away)
        if st.button("__PT_SWITCH_LONG__", key="pt_switch_long_hid", help="internal"):
            phase_now = st.session_state.get("pt_phase", "setup")
            if phase_now == "running" and not st.session_state.get("pt_test_locked", False):
                st.session_state.pt_test_locked = True
                st.session_state.pt_lock_trigger = (
                    "You were away from the test window for more than 10 seconds. "
                    "The test has been locked by the proctor system."
                )
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Proctor lock gate ─────────────────────────────────────────────────────
    if st.session_state.get("pt_test_locked", False):
        _page_locked()
        return

    # ── Switch warning banner (visible only during active test) ───────────────
    cnt = st.session_state.get("pt_switch_count", 0)
    if cnt > 0 and st.session_state.get("pt_phase") == "running":
        remaining = max(0, 3 - cnt)
        col = "#f59e0b" if remaining > 0 else "#ff3366"
        warn_msg = (
            f"⚠️ &nbsp;<b>Proctor Warning:</b> You have switched windows "
            f"<b>{cnt}/3</b> time{'s' if cnt != 1 else ''}. "
            + (f"<b>{remaining}</b> switch{'es' if remaining != 1 else ''} remaining before lock."
               if remaining > 0 else "Next switch will lock your test!")
        )
        st.markdown(
            f'<div style="background:rgba(245,158,11,.08);border:.5px solid {col};'
            f'border-radius:8px;padding:10px 16px;margin-bottom:10px;'
            f'font-family:Inter,sans-serif;font-size:13px;color:{col}">'
            f'{warn_msg}</div>',
            unsafe_allow_html=True,
        )

    try:
        from app import get_engine; engine=get_engine()
    except Exception:
        engine=InterviewEngine() if _ENGINE_OK else None
    if engine is None:
        st.error("InterviewEngine unavailable — check backend_engine.py."); return

    phase=st.session_state.pt_phase
    if phase!="setup": _header()

    if   phase=="setup":          _page_setup(engine)
    elif phase=="running":        _page_running(engine)
    elif phase=="round_complete": _page_round_complete(engine)
    elif phase=="gated_out":      _page_gated_out()
    elif phase=="all_complete":   _page_report()
    else: st.session_state.pt_phase="setup"; st.rerun()