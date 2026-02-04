import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import streamlit as st
from supabase import create_client, Client

# =============================================================================
# CONFIG / SECRETS
# =============================================================================
def get_secret(name: str, default: str = "") -> str:
    if name in st.secrets:
        v = st.secrets.get(name)
        return (str(v) if v is not None else "").strip()
    return os.getenv(name, default).strip()

SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_ANON_KEY = get_secret("SUPABASE_ANON_KEY")

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-4o-mini")

APP_DIR = Path(__file__).parent
QUESTIONS_PATH = APP_DIR / "questions.json"

# Optional (future): show figures from a rendered image folder
FIGURES_DIR = APP_DIR / "figures"  # e.g. figures/fig_46.png

# =============================================================================
# REQUIRED CLUSTERING (totals)
# =============================================================================
# IMPORTANT: This is the *display/reference structure* for progress.
# Matching is done by (category, subchapter) from questions.json.
REQUIRED: Dict[str, Dict[str, int]] = {
    "Luftrecht": {
        "Kollisionsvermeidung": 14,
        "Rechtsvorschriften": 18,
        "Luftraumvorschriften allgemein": 11,
        "Luftraum G/E (Sichtflug 1)": 11,
        "Luftraum G/E (Sichtflug 2)": 13,
        "CTR/Tiefflug": 11,
        "Beschränkungsgebiete": 14,
        "ICAO-Grundlagen": 8,
        "ICAO-Aufgaben": 19,
    },
    "Meteorologie": {
        "Atmosphäre/Druck": 11,
        "Temperatur/Feuchte": 13,
        "Wolken/Nebel": 11,
        "Wettervorhersage/Karten": 22,
        "Wolkeninterpretation": 17,
        "Föhn": 11,
        "Thermik/Adiabatik (Block 1)": 13,
        "Thermik/Adiabatik (Block 2)": 7,
        "Labilität/Stabilität": 6,
        "Wind/Luv/Lee": 12,
        "Gewitter": 11,
        "Alpines Wetter": 9,
        "Fronten/Hoch/Tief (Block 1)": 10,
        "Fronten/Hoch/Tief (Block 2)": 10,
    },
    "Navigation": {
        "Grundlagen": 9,
        "Koordinatensystem (Block 1)": 4,
        "Koordinatensystem (Block 2)": 13,
        "Umrechnungen (Block 1)": 8,
        "Umrechnungen (Block 2)": 6,
        "Maßeinheiten": 6,
        "ICAO-Bestimmungen": 14,
        "ICAO-Begriffe": 4,
        "Kompass/Kurs": 11,
        "Satellitennavigation": 14,
        "Kartenlehre": 5,
        "Terrestrische Nav.": 5,
        "Planung/Sicherheit": 7,
    },
}

# =============================================================================
# SUPABASE CLIENT
# =============================================================================
@st.cache_resource(show_spinner=False)
def supa() -> Client:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("Supabase Secrets fehlen: SUPABASE_URL / SUPABASE_ANON_KEY")
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# =============================================================================
# AUTH
# =============================================================================
def _restore_session_from_state():
    """Ensure Supabase client has a session if we already logged in earlier in this Streamlit session."""
    if "sb_access" in st.session_state and "sb_refresh" in st.session_state:
        try:
            supa().auth.set_session(st.session_state.sb_access, st.session_state.sb_refresh)
        except Exception:
            # Ignore; user will need to login again
            pass

def auth_ui():
    st.sidebar.markdown("## Login")
    email = st.sidebar.text_input("E-Mail", key="login_email")
    pw = st.sidebar.text_input("Passwort", type="password", key="login_pw")

    colA, colB = st.sidebar.columns(2)
    if colA.button("Login"):
        try:
            res = supa().auth.sign_in_with_password({"email": email, "password": pw})
            st.session_state.user = res.user
            st.session_state.sb_access = res.session.access_token
            st.session_state.sb_refresh = res.session.refresh_token
            supa().auth.set_session(res.session.access_token, res.session.refresh_token)
            _reset_learning_state(hard=True)
            st.rerun()
        except Exception as e:
            st.sidebar.error(str(e))

    if colB.button("Logout"):
        try:
            supa().auth.sign_out()
        except Exception:
            pass
        st.session_state.clear()
        st.rerun()

def require_login() -> str:
    _restore_session_from_state()
    if "user" not in st.session_state:
        auth_ui()
        st.stop()
    return str(st.session_state.user.id)

# =============================================================================
# QUESTIONS
# =============================================================================
@st.cache_data(show_spinner=False)
def load_questions() -> List[Dict[str, Any]]:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"questions.json fehlt unter: {QUESTIONS_PATH}")
    return json.loads(QUESTIONS_PATH.read_text("utf-8"))

def index_questions(questions: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    idx: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for q in questions:
        cat = (q.get("category") or "").strip()
        sub = (q.get("subchapter") or "").strip()
        idx.setdefault((cat, sub), []).append(q)
    return idx

# =============================================================================
# DB (progress table)
# =============================================================================
def db_load_progress(uid: str) -> Dict[str, Dict[str, Any]]:
    r = supa().table("progress").select("*").eq("user_id", uid).execute()
    # expected row: user_id, question_id, seen, correct, wrong
    return {x["question_id"]: x for x in (r.data or [])}

def db_upsert(uid: str, qid: str, ok: bool):
    s = supa()
    r = s.table("progress").select("*").eq("user_id", uid).eq("question_id", qid).limit(1).execute()
    if r.data:
        row = r.data[0]
        s.table("progress").update({
            "seen": int(row.get("seen", 0)) + 1,
            "correct": int(row.get("correct", 0)) + (1 if ok else 0),
            "wrong": int(row.get("wrong", 0)) + (0 if ok else 1),
        }).eq("user_id", uid).eq("question_id", qid).execute()
    else:
        s.table("progress").insert({
            "user_id": uid,
            "question_id": qid,
            "seen": 1,
            "correct": 1 if ok else 0,
            "wrong": 0 if ok else 1,
        }).execute()

# =============================================================================
# OPTIONAL OPENAI
# =============================================================================
def ai_explain(question: str, correct_answer: str) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = (
            "Erkläre kurz und fachlich korrekt (Gleitschirm B-Lizenz Prüfung), "
            "warum diese Antwort richtig ist.\n\n"
            f"Frage: {question}\n"
            f"Richtige Antwort: {correct_answer}\n"
        )
        txt = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        ).choices[0].message.content
        return (txt or "").strip() or None
    except Exception:
        return None

# =============================================================================
# LEARNING STATE
# =============================================================================
def _reset_learning_state(hard: bool = False):
    # hard: also clears mode/filter selections
    st.session_state.pop("queue", None)
    st.session_state.pop("idx", None)
    st.session_state.pop("answered", None)
    st.session_state.pop("last_ok", None)
    st.session_state.pop("last_exp", None)
    st.session_state.pop("last_correct_index", None)
    st.session_state.pop("last_selected_index", None)
    if hard:
        st.session_state.pop("mode", None)
        st.session_state.pop("sel_category", None)
        st.session_state.pop("sel_subchapter", None)
        st.session_state.pop("only_unseen", None)
        st.session_state.pop("only_wrong", None)

def build_queue(
    questions: List[Dict[str, Any]],
    progress: Dict[str, Dict[str, Any]],
    mode: str,
    category: str,
    subchapter: str,
    only_unseen: bool,
    only_wrong: bool,
) -> List[Dict[str, Any]]:
    # Filter base set
    qset = questions
    if mode == "Lernen":
        if category != "Alle":
            qset = [q for q in qset if (q.get("category") or "") == category]
        if subchapter != "Alle":
            qset = [q for q in qset if (q.get("subchapter") or "") == subchapter]

        if only_unseen:
            qset = [q for q in qset if q.get("id") not in progress or int(progress[q["id"]].get("seen", 0)) == 0]
        if only_wrong:
            qset = [q for q in qset if q.get("id") in progress and int(progress[q["id"]].get("wrong", 0)) > 0]

        random.shuffle(qset)
        return qset

    if mode == "Prüfungsmodus (40)":
        base = list(questions)
        random.shuffle(base)
        return base[:40]

    if mode == "Wiederholen falsch":
        wrong_only = [q for q in questions if q.get("id") in progress and int(progress[q["id"]].get("wrong", 0)) > 0]
        random.shuffle(wrong_only)
        return wrong_only if wrong_only else []

    # fallback
    base = list(questions)
    random.shuffle(base)
    return base

# =============================================================================
# PROGRESS COMPUTATION
# =============================================================================
def compute_progress_by_cluster(
    questions_by_cluster: Dict[Tuple[str, str], List[Dict[str, Any]]],
    progress: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Returns:
      out[category][subchapter] = {
        total, learned, correct_total, wrong_total
      }
    learned = count of questions with seen>0
    correct_total = sum(correct) across questions
    wrong_total = sum(wrong) across questions
    """
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for (cat, sub), qs in questions_by_cluster.items():
        total = len(qs)
        learned = 0
        correct_total = 0
        wrong_total = 0
        for q in qs:
            qid = q.get("id")
            row = progress.get(qid)
            if row:
                if int(row.get("seen", 0)) > 0:
                    learned += 1
                correct_total += int(row.get("correct", 0))
                wrong_total += int(row.get("wrong", 0))
        out.setdefault(cat, {})[sub] = {
            "total": total,
            "learned": learned,
            "correct_total": correct_total,
            "wrong_total": wrong_total,
        }
    return out

def render_progress_sidebar(questions: List[Dict[str, Any]], progress: Dict[str, Dict[str, Any]]):
    st.sidebar.markdown("## Fortschritt")

    qidx = index_questions(questions)
    stats = compute_progress_by_cluster(qidx, progress)

    # Display in required order (structure first), fallback to stats for unknown keys
    for cat, subs in REQUIRED.items():
        st.sidebar.markdown(f"### {cat}")
        for sub, expected_total in subs.items():
            s = stats.get(cat, {}).get(sub, {"total": 0, "learned": 0, "correct_total": 0, "wrong_total": 0})
            total = expected_total if expected_total else s["total"]
            learned = s["learned"]
            correct_total = s["correct_total"]

            learned_pct = int(round((learned / total) * 100)) if total else 0
            correct_pct = int(round((correct_total / total) * 100)) if total else 0

            # compact line, mobile-friendly
            if learned == 0:
                st.sidebar.caption(f"{sub} ({total}) — nicht begonnen")
            else:
                st.sidebar.caption(f"{sub} ({total}) — gelernt: {learned_pct}% ({learned}), richtig: {correct_pct}% ({correct_total})")

# =============================================================================
# UI
# =============================================================================
st.set_page_config(page_title="B-Lizenz Lernapp", layout="wide")

if not QUESTIONS_PATH.exists():
    st.error(f"questions.json fehlt: {QUESTIONS_PATH}")
    st.stop()

uid = require_login()
questions = load_questions()

# load progress once per session; refresh after each answer
if "progress" not in st.session_state:
    st.session_state.progress = db_load_progress(uid)

progress = st.session_state.progress

# Sidebar: mode + filters + progress
st.sidebar.markdown("## Modus")

mode = st.sidebar.selectbox(
    "Modus",
    ["Lernen", "Prüfungsmodus (40)", "Wiederholen falsch"],
    key="mode",
)

# dynamic category/subchapter options based on questions.json
cats = sorted(set((q.get("category") or "").strip() for q in questions if q.get("category")))
subs = sorted(set((q.get("subchapter") or "").strip() for q in questions if q.get("subchapter")))

sel_category = "Alle"
sel_subchapter = "Alle"
only_unseen = False
only_wrong = False

if mode == "Lernen":
    sel_category = st.sidebar.selectbox("Kategorie", ["Alle"] + cats, key="sel_category")
    filtered_subs = subs
    if sel_category != "Alle":
        filtered_subs = sorted(set((q.get("subchapter") or "").strip() for q in questions if (q.get("category") or "") == sel_category))
    sel_subchapter = st.sidebar.selectbox("Unterkapitel", ["Alle"] + filtered_subs, key="sel_subchapter")
    only_unseen = st.sidebar.checkbox("Nur ungelernt", value=st.session_state.get("only_unseen", False), key="only_unseen")
    only_wrong = st.sidebar.checkbox("Nur falsch beantwortete", value=st.session_state.get("only_wrong", False), key="only_wrong")

st.sidebar.divider()

# session start/reset
start = st.sidebar.button("Session starten / neu mischen", use_container_width=True)
if start or "queue" not in st.session_state:
    queue = build_queue(
        questions=questions,
        progress=progress,
        mode=mode,
        category=sel_category,
        subchapter=sel_subchapter,
        only_unseen=only_unseen,
        only_wrong=only_wrong,
    )
    st.session_state.queue = queue
    st.session_state.idx = 0
    st.session_state.answered = False
    st.session_state.last_ok = None
    st.session_state.last_exp = None
    st.session_state.last_correct_index = None
    st.session_state.last_selected_index = None

render_progress_sidebar(questions, progress)

# =============================================================================
# MAIN VIEW
# =============================================================================
st.title("B-Lizenz Lernapp")

queue: List[Dict[str, Any]] = st.session_state.get("queue", [])
idx: int = int(st.session_state.get("idx", 0))

if not queue:
    st.warning("Keine Fragen für diese Auswahl (Filter/Modus).")
    st.stop()

if idx >= len(queue):
    st.session_state.idx = 0
    idx = 0

q = queue[idx]
qid = str(q.get("id"))
question = (q.get("question") or "").strip()
options = q.get("options") or []
correct_index = int(q.get("correctIndex", -1))

# Header
left, right = st.columns([3, 1])
with left:
    st.markdown(f"### {question}")
    st.caption(f"{q.get('category','')} · {q.get('subchapter','')} · {idx+1}/{len(queue)}")
with right:
    st.metric("Session", f"{idx+1}/{len(queue)}")

# Figures (optional)
figs = q.get("figures") or []
if figs:
    # show first figure mapping; if you render images later, place them in FIGURES_DIR and name fig_<nr>.png
    for f in figs:
        fig_no = f.get("figure")
        page = f.get("bilder_page")
        st.info(f"Abbildung {fig_no} (Bilder.pdf Seite {page})")
        img_path = FIGURES_DIR / f"fig_{fig_no}.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)

# Answer UI
labels = ["A", "B", "C", "D"]
st.session_state.setdefault("answered", False)

if not st.session_state.answered:
    for i, opt in enumerate(options[:4]):
        if st.button(f"{labels[i]}) {opt}", key=f"opt_{qid}_{i}", use_container_width=True):
            ok = i == correct_index

            db_upsert(uid, qid, ok)
            # refresh progress in session (authoritative)
            st.session_state.progress = db_load_progress(uid)
            progress = st.session_state.progress

            st.session_state.answered = True
            st.session_state.last_ok = ok
            st.session_state.last_selected_index = i
            st.session_state.last_correct_index = correct_index

            # Optional AI explanation on wrong answer (or always if key set)
            exp = None
            if OPENAI_API_KEY:
                correct_text = options[correct_index] if 0 <= correct_index < len(options) else ""
                exp = ai_explain(question, correct_text)
            st.session_state.last_exp = exp
            st.rerun()
else:
    ok = bool(st.session_state.get("last_ok"))
    sel_i = st.session_state.get("last_selected_index")
    corr_i = st.session_state.get("last_correct_index")
    if ok:
        st.success("Richtig")
    else:
        st.error("Falsch")
        if corr_i is not None and 0 <= int(corr_i) < len(options):
            st.info(f"Richtig ist: {labels[int(corr_i)]}) {options[int(corr_i)]}")

    if st.session_state.get("last_exp"):
        st.info(st.session_state["last_exp"])

    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("Nächste Frage", use_container_width=True):
        st.session_state.idx = (idx + 1) % len(queue)
        st.session_state.answered = False
        st.session_state.last_ok = None
        st.session_state.last_exp = None
        st.session_state.last_correct_index = None
        st.session_state.last_selected_index = None
        st.rerun()

    if c2.button("Session neu mischen", use_container_width=True):
        _reset_learning_state()
        st.rerun()

    # quick stats for current cluster
    with c3:
        row = progress.get(qid, {})
        seen = int(row.get("seen", 0)) if row else 0
        correct = int(row.get("correct", 0)) if row else 0
        wrong = int(row.get("wrong", 0)) if row else 0
        st.caption(f"Dein Verlauf für {qid}: gesehen {seen}, richtig {correct}, falsch {wrong}")
