import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import streamlit as st
from supabase import create_client, Client
import streamlit.components.v1 as components

# =============================================================================
# CONFIG / FILES
# =============================================================================
APP_DIR = Path(__file__).parent
QUESTIONS_PATH = APP_DIR / "questions.json"
BILDER_PDF = APP_DIR / "Bilder_v2.pdf"
WIKI_PATH = APP_DIR / "wiki_content.json"  # static, shared by all users (you maintain this)

def get_secret(name: str, default: str = "") -> str:
    if name in st.secrets:
        v = st.secrets.get(name)
        return (str(v) if v is not None else "").strip()
    return os.getenv(name, default).strip()

SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_ANON_KEY = get_secret("SUPABASE_ANON_KEY")

# =============================================================================
# REQUIRED CLUSTERING (display/progress structure)
# =============================================================================
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
# OPTIONAL COOKIE MANAGER (for "Angemeldet bleiben")
# =============================================================================
def cookie_mgr():
    try:
        from streamlit_cookies_manager import EncryptedCookieManager
        cookies = EncryptedCookieManager(prefix="bliz_", password=get_secret("COOKIE_PASSWORD", "change-me"))
        if not cookies.ready():
            st.stop()
        return cookies
    except Exception:
        return None

# =============================================================================
# QUESTIONS / WIKI (STATIC)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_questions() -> List[Dict[str, Any]]:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"questions.json fehlt: {QUESTIONS_PATH}")
    return json.loads(QUESTIONS_PATH.read_text("utf-8"))

@st.cache_data(show_spinner=False)
def load_wiki() -> Dict[str, Any]:
    # Static content, shared by all users, maintained manually for factual correctness
    if not WIKI_PATH.exists():
        return {}
    try:
        return json.loads(WIKI_PATH.read_text("utf-8"))
    except Exception:
        return {}

def index_questions(questions: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    idx: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for q in questions:
        cat = (q.get("category") or "").strip()
        sub = (q.get("subchapter") or "").strip()
        idx.setdefault((cat, sub), []).append(q)
    return idx

# =============================================================================
# PDF IMAGE RENDER (Bilder_v2.pdf)
# =============================================================================
@st.cache_data(show_spinner=False)
def render_pdf_page_png(pdf_path: str, page_1based: int, zoom: float = 2.0) -> Optional[bytes]:
    """
    Returns PNG bytes for a full page render.
    Requires PyMuPDF (fitz). If not installed, returns None.
    """
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None

    p = Path(pdf_path)
    if not p.exists():
        return None

    try:
        doc = fitz.open(str(p))
        page = doc.load_page(max(0, min(page_1based - 1, doc.page_count - 1)))
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    except Exception:
        return None

# =============================================================================
# DB: progress + notes
#   progress table: user_id, question_id, seen, correct, wrong
#   notes table:    user_id, question_id, note_text, updated_at
# =============================================================================
def db_load_progress(uid: str) -> Dict[str, Dict[str, Any]]:
    r = supa().table("progress").select("*").eq("user_id", uid).execute()
    return {x["question_id"]: x for x in (r.data or [])}

def db_upsert_progress(uid: str, qid: str, ok: bool):
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

def db_get_note(uid: str, qid: str) -> str:
    s = supa()
    r = s.table("notes").select("note_text").eq("user_id", uid).eq("question_id", qid).limit(1).execute()
    if r.data:
        return (r.data[0].get("note_text") or "").strip()
    return ""

def db_upsert_note(uid: str, qid: str, note_text: str):
    s = supa()
    note_text = (note_text or "").strip()
    r = s.table("notes").select("*").eq("user_id", uid).eq("question_id", qid).limit(1).execute()
    if r.data:
        s.table("notes").update({"note_text": note_text}).eq("user_id", uid).eq("question_id", qid).execute()
    else:
        s.table("notes").insert({"user_id": uid, "question_id": qid, "note_text": note_text}).execute()

# =============================================================================
# AUTH (Login + Registration + Remember)
# =============================================================================
def _set_session_tokens(access: str, refresh: str):
    st.session_state.sb_access = access
    st.session_state.sb_refresh = refresh
    try:
        supa().auth.set_session(access, refresh)
    except Exception:
        pass

def _restore_session_from_cookie():
    cookies = cookie_mgr()
    if not cookies:
        return
    access = cookies.get("access", "")
    refresh = cookies.get("refresh", "")
    if access and refresh and "user" not in st.session_state:
        try:
            supa().auth.set_session(access, refresh)
            user = supa().auth.get_user().user
            if user:
                st.session_state.user = user
                _set_session_tokens(access, refresh)
        except Exception:
            pass

def auth_ui():
    st.sidebar.markdown("## Account")

    tab_login, tab_register = st.sidebar.tabs(["Login", "Registrieren"])

    cookies = cookie_mgr()

    with tab_login:
        saved_email = ""
        if cookies:
            saved_email = cookies.get("email", "") or ""
        email = st.text_input("E-Mail", value=saved_email, key="login_email")
        pw = st.text_input("Passwort", type="password", key="login_pw")
        remember = st.checkbox("Angemeldet bleiben", value=True, key="remember_me")

        col1, col2 = st.columns(2)
        if col1.button("Login", use_container_width=True):
            try:
                res = supa().auth.sign_in_with_password({"email": email, "password": pw})
                st.session_state.user = res.user
                _set_session_tokens(res.session.access_token, res.session.refresh_token)

                if cookies:
                    cookies["email"] = email
                    if remember:
                        cookies["access"] = res.session.access_token
                        cookies["refresh"] = res.session.refresh_token
                    else:
                        cookies["access"] = ""
                        cookies["refresh"] = ""
                    cookies.save()

                _reset_learning_state(hard=True)
                st.rerun()
            except Exception as e:
                st.error(str(e))

        if col2.button("Logout", use_container_width=True):
            try:
                supa().auth.sign_out()
            except Exception:
                pass
            if cookies:
                cookies["access"] = ""
                cookies["refresh"] = ""
                cookies.save()
            st.session_state.clear()
            st.rerun()

    with tab_register:
        email_r = st.text_input("E-Mail", key="reg_email")
        pw_r = st.text_input("Passwort", type="password", key="reg_pw")
        pw_r2 = st.text_input("Passwort wiederholen", type="password", key="reg_pw2")

        if st.button("Registrieren", use_container_width=True):
            if not email_r or not pw_r or (pw_r != pw_r2):
                st.error("Bitte E-Mail + Passwort korrekt eingeben.")
            else:
                try:
                    supa().auth.sign_up({"email": email_r, "password": pw_r})
                    st.success("Registrierung erstellt. Bitte einloggen.")
                except Exception as e:
                    st.error(str(e))

def require_login() -> str:
    _restore_session_from_cookie()
    if "user" not in st.session_state:
        auth_ui()
        st.stop()
    return str(st.session_state.user.id)

# =============================================================================
# LEARNING STATE / QUEUES
# =============================================================================
def _reset_learning_state(hard: bool = False):
    for k in ["queue", "idx", "answered", "last_ok", "last_correct_index", "last_selected_index"]:
        st.session_state.pop(k, None)
    if hard:
        for k in ["mode", "sel_category", "sel_subchapter", "only_unseen", "only_wrong"]:
            st.session_state.pop(k, None)

def build_queue(
    questions: List[Dict[str, Any]],
    progress: Dict[str, Dict[str, Any]],
    mode: str,
    category: str,
    subchapter: str,
    only_unseen: bool,
    only_wrong: bool,
) -> List[Dict[str, Any]]:
    if mode == "Prüfungsmodus (40)":
        base = list(questions)
        random.shuffle(base)
        return base[:40]

    if mode == "Wiederholen falsch":
        wrong_only = [q for q in questions if q.get("id") in progress and int(progress[q["id"]].get("wrong", 0)) > 0]
        random.shuffle(wrong_only)
        return wrong_only

    # Lernen
    qset = list(questions)
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

# =============================================================================
# PROGRESS SIDEBAR
# =============================================================================
def compute_progress_by_cluster(
    questions_by_cluster: Dict[Tuple[str, str], List[Dict[str, Any]]],
    progress: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for (cat, sub), qs in questions_by_cluster.items():
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
            "total": len(qs),
            "learned": learned,
            "correct_total": correct_total,
            "wrong_total": wrong_total,
        }
    return out

def render_progress_sidebar(questions: List[Dict[str, Any]], progress: Dict[str, Dict[str, Any]]):
    st.sidebar.markdown("## Fortschritt")
    qidx = index_questions(questions)
    stats = compute_progress_by_cluster(qidx, progress)

    for cat, subs in REQUIRED.items():
        st.sidebar.markdown(f"### {cat}")
        for sub, expected_total in subs.items():
            s = stats.get(cat, {}).get(sub, {"total": 0, "learned": 0, "correct_total": 0})
            total = expected_total if expected_total else s["total"]
            learned = s["learned"]
            correct_total = s["correct_total"]
            if learned == 0:
                st.sidebar.caption(f"{sub} ({total}) — nicht begonnen")
            else:
                learned_pct = int(round((learned / total) * 100)) if total else 0
                correct_pct = int(round((correct_total / total) * 100)) if total else 0
                st.sidebar.caption(f"{sub} ({total}) — gelernt: {learned_pct}% ({learned}), richtig: {correct_pct}% ({correct_total})")

# =============================================================================
# UI STYLES
# =============================================================================
def inject_css():
    st.markdown(
        """
<style>
/* Bigger tap targets + cleaner cards */
div.stButton > button {
  width: 100%;
  padding: 0.9rem 1rem;
  border-radius: 12px;
  font-size: 1rem;
}
.block-container { padding-top: 1.5rem; }
.pp-card {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  background: rgba(255,255,255,0.03);
}
.pp-muted { opacity: 0.85; font-size: 0.95rem; }
.pp-divider { height: 1px; background: rgba(255,255,255,0.08); margin: 1rem 0; }
</style>
""",
        unsafe_allow_html=True,
    )

# =============================================================================
# MAIN APP
# =============================================================================
st.set_page_config(page_title="B-Lizenz Lernapp", layout="wide")
inject_css()

if not QUESTIONS_PATH.exists():
    st.error("questions.json fehlt")
    st.stop()

uid = require_login()

questions = load_questions()
wiki = load_wiki()

if "progress" not in st.session_state:
    st.session_state.progress = db_load_progress(uid)
progress = st.session_state.progress

# Sidebar controls
st.sidebar.markdown("## Modus")
mode = st.sidebar.selectbox("Modus", ["Lernen", "Prüfungsmodus (40)", "Wiederholen falsch"], key="mode")

cats = sorted(set((q.get("category") or "").strip() for q in questions if q.get("category")))
sel_category = "Alle"
sel_subchapter = "Alle"
only_unseen = False
only_wrong = False

if mode == "Lernen":
    sel_category = st.sidebar.selectbox("Kategorie", ["Alle"] + cats, key="sel_category")
    filtered_subs = sorted(set((q.get("subchapter") or "").strip() for q in questions if q.get("subchapter")))
    if sel_category != "Alle":
        filtered_subs = sorted(set((q.get("subchapter") or "").strip() for q in questions if (q.get("category") or "") == sel_category))
    sel_subchapter = st.sidebar.selectbox("Unterkapitel", ["Alle"] + filtered_subs, key="sel_subchapter")
    only_unseen = st.sidebar.checkbox("Nur ungelernt", value=st.session_state.get("only_unseen", False), key="only_unseen")
    only_wrong = st.sidebar.checkbox("Nur falsch beantwortete", value=st.session_state.get("only_wrong", False), key="only_wrong")

st.sidebar.divider()
start = st.sidebar.button("Session starten / neu mischen", use_container_width=True)

if start or "queue" not in st.session_state:
    st.session_state.queue = build_queue(
        questions=questions,
        progress=progress,
        mode=mode,
        category=sel_category,
        subchapter=sel_subchapter,
        only_unseen=only_unseen,
        only_wrong=only_wrong,
    )
    st.session_state.idx = 0
    st.session_state.answered = False
    st.session_state.last_ok = None
    st.session_state.last_correct_index = None
    st.session_state.last_selected_index = None

render_progress_sidebar(questions, progress)

# Main
st.title("B-Lizenz Lernapp")

queue: List[Dict[str, Any]] = st.session_state.get("queue", [])
idx: int = int(st.session_state.get("idx", 0))

if not queue:
    st.warning("Keine Fragen für diese Auswahl.")
    st.stop()

if idx >= len(queue):
    st.session_state.idx = 0
    idx = 0

q = queue[idx]
qid = str(q.get("id"))
question = (q.get("question") or "").strip()
options = q.get("options") or []
correct_index = int(q.get("correctIndex", -1))

# Header card
st.markdown(
    f"""
<div class="pp-card">
  <div><b>{question}</b></div>
  <div class="pp-muted">{q.get("category","")} · {q.get("subchapter","")} · {idx+1}/{len(queue)}</div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# Image from Bilder_v2.pdf (full page)
figs = q.get("figures") or []
if figs:
    # show first referenced figure/page
    f0 = figs[0]
    fig_no = f0.get("figure")
    page_1based = int(f0.get("bilder_page") or 0)
    if page_1based > 0:
        png = render_pdf_page_png(str(BILDER_PDF), page_1based=page_1based, zoom=2.0)
        if png:
            st.image(png, caption=f"Abbildung {fig_no} (Bilder.pdf Seite {page_1based})", use_container_width=True)
        else:
            st.info(f"Abbildung {fig_no} (Bilder.pdf Seite {page_1based})")

# Answer buttons
labels = ["A", "B", "C", "D"]

if not st.session_state.get("answered", False):
    for i, opt in enumerate(options[:4]):
        if st.button(f"{labels[i]}) {opt}", key=f"opt_{qid}_{i}"):
            ok = i == correct_index
            db_upsert_progress(uid, qid, ok)

            # refresh progress
            st.session_state.progress = db_load_progress(uid)
            progress = st.session_state.progress

            st.session_state.answered = True
            st.session_state.last_ok = ok
            st.session_state.last_selected_index = i
            st.session_state.last_correct_index = correct_index
            st.rerun()
else:
    ok = bool(st.session_state.get("last_ok"))
    corr_i = st.session_state.get("last_correct_index")

    if ok:
        st.success("Richtig")
    else:
        st.error("Falsch")
        if corr_i is not None and 0 <= int(corr_i) < len(options):
            st.info(f"Richtig ist: {labels[int(corr_i)]}) {options[int(corr_i)]}")

    # Static wiki (shared)
    # lookup order: question_id -> wiki_key -> category/subchapter
    wiki_key = (q.get("wiki_key") or "").strip()
    w = None
    if qid and qid in wiki:
        w = wiki.get(qid)
    elif wiki_key and wiki_key in wiki:
        w = wiki.get(wiki_key)
    else:
        fallback_key = f"{q.get('category','')}::{q.get('subchapter','')}"
        if fallback_key in wiki:
            w = wiki.get(fallback_key)

    with st.expander("Wiki (faktisch, zentral gepflegt)", expanded=True):
        if isinstance(w, dict):
            explanation = (w.get("explanation") or "").strip()
            merksatz = (w.get("merksatz") or "").strip()
            source_url = (w.get("source_url") or "").strip()

            if explanation:
                st.markdown(explanation)
            if merksatz:
                st.markdown(f"**Merksatz:** {merksatz}")
            if source_url:
                st.markdown(f"**Quelle:** {source_url}")
        else:
            st.info("Kein Wiki-Eintrag vorhanden (wiki_content.json ergänzen).")

    # Per-user note
    existing_note = db_get_note(uid, qid)
    with st.expander("Deine Bemerkung (nur für dich)", expanded=False):
        note_text = st.text_area("Notiz", value=existing_note, key=f"note_{qid}", height=120)
        if st.button("Notiz speichern", key=f"save_note_{qid}"):
            db_upsert_note(uid, qid, note_text)
            st.success("Gespeichert")

    # Controls
    c1, c2 = st.columns([1, 1])
    if c1.button("Nächste Frage"):
        st.session_state.idx = (idx + 1) % len(queue)
        st.session_state.answered = False
        st.session_state.last_ok = None
        st.session_state.last_correct_index = None
        st.session_state.last_selected_index = None
        st.rerun()

    if c2.button("Session neu mischen"):
        _reset_learning_state()
        st.rerun()
