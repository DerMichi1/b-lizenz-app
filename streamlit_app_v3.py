import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import streamlit as st
from supabase import create_client, Client

# =============================================================================
# CONFIG / FILES
# =============================================================================
APP_DIR = Path(__file__).parent
QUESTIONS_PATH = APP_DIR / "questions.json"
BILDER_PDF = APP_DIR / "Bilder_v2.pdf"
WIKI_PATH = APP_DIR / "wiki_content.json"  # static shared content (maintain manually)

PASS_PCT = float(os.getenv("PASS_PCT", "75"))  # exam pass threshold in percent (default 75%)

def get_secret(name: str, default: str = "") -> str:
    if name in st.secrets:
        v = st.secrets.get(name)
        return (str(v) if v is not None else "").strip()
    return os.getenv(name, default).strip()

SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_ANON_KEY = get_secret("SUPABASE_ANON_KEY")
COOKIE_PASSWORD = get_secret("COOKIE_PASSWORD", "")  # REQUIRED for remember-me

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
# COOKIE MANAGER (required for remember-me)
# =============================================================================
def cookie_mgr():
    from streamlit_cookies_manager import EncryptedCookieManager
    if not COOKIE_PASSWORD:
        raise RuntimeError("COOKIE_PASSWORD fehlt in Streamlit Secrets.")
    cookies = EncryptedCookieManager(prefix="bliz_", password=COOKIE_PASSWORD)
    if not cookies.ready():
        # component handshake requires at least one rerun
        st.stop()
    return cookies

# =============================================================================
# QUESTIONS / WIKI
# =============================================================================
@st.cache_data(show_spinner=False)
def load_questions() -> List[Dict[str, Any]]:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"questions.json fehlt: {QUESTIONS_PATH}")
    return json.loads(QUESTIONS_PATH.read_text("utf-8"))

@st.cache_data(show_spinner=False)
def load_wiki() -> Dict[str, Any]:
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

def by_id(questions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(q.get("id")): q for q in questions if q.get("id") is not None}

# =============================================================================
# PDF IMAGE RENDER (Bilder_v2.pdf)
# =============================================================================
@st.cache_data(show_spinner=False)
def render_pdf_page_png(pdf_path: str, page_1based: int, zoom: float = 2.0) -> Optional[bytes]:
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
# DB: progress + notes + exam_runs
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
    try:
        r = supa().table("notes").select("note_text").eq("user_id", uid).eq("question_id", qid).limit(1).execute()
        if r.data:
            return (r.data[0].get("note_text") or "").strip()
    except Exception:
        return ""
    return ""

def db_upsert_note(uid: str, qid: str, note_text: str) -> bool:
    try:
        s = supa()
        note_text = (note_text or "").strip()
        r = s.table("notes").select("*").eq("user_id", uid).eq("question_id", qid).limit(1).execute()
        if r.data:
            s.table("notes").update({"note_text": note_text}).eq("user_id", uid).eq("question_id", qid).execute()
        else:
            s.table("notes").insert({"user_id": uid, "question_id": qid, "note_text": note_text}).execute()
        return True
    except Exception:
        return False

def db_insert_exam_run(uid: str, total: int, correct: int, passed: bool) -> None:
    try:
        supa().table("exam_runs").insert({
            "user_id": uid,
            "total": total,
            "correct": correct,
            "passed": passed,
        }).execute()
    except Exception:
        # non-fatal if table missing
        pass

def db_list_exam_runs(uid: str, limit: int = 50) -> List[Dict[str, Any]]:
    try:
        r = supa().table("exam_runs").select("*").eq("user_id", uid).order("created_at", desc=True).limit(limit).execute()
        return list(r.data or [])
    except Exception:
        return []

# =============================================================================
# AUTH (Login + Registration + Remember)
# =============================================================================
def _set_session_tokens(access: str, refresh: str):
    st.session_state.sb_access = access
    st.session_state.sb_refresh = refresh
    supa().auth.set_session(access, refresh)

def _restore_session_from_cookie():
    cookies = cookie_mgr()
    access = cookies.get("access", "") or ""
    refresh = cookies.get("refresh", "") or ""
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

    cookies = cookie_mgr()

    tab_login, tab_register = st.sidebar.tabs(["Login", "Registrieren"])

    with tab_login:
        saved_email = (cookies.get("email", "") or "")
        email = st.text_input("E-Mail", value=saved_email, key="login_email")
        pw = st.text_input("Passwort", type="password", key="login_pw")
        remember = st.checkbox("Angemeldet bleiben", value=True, key="remember_me")

        col1, col2 = st.columns(2)
        if col1.button("Login", use_container_width=True):
            res = supa().auth.sign_in_with_password({"email": email, "password": pw})
            st.session_state.user = res.user
            _set_session_tokens(res.session.access_token, res.session.refresh_token)

            cookies["email"] = email
            if remember:
                cookies["access"] = res.session.access_token
                cookies["refresh"] = res.session.refresh_token
            else:
                cookies["access"] = ""
                cookies["refresh"] = ""
            cookies.save()

            _reset_app_state(hard=True)
            st.session_state.page = "dashboard"
            st.rerun()

        if col2.button("Logout", use_container_width=True):
            try:
                supa().auth.sign_out()
            except Exception:
                pass
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
                supa().auth.sign_up({"email": email_r, "password": pw_r})
                st.success("Registrierung erstellt. Bitte einloggen.")

def require_login() -> str:
    _restore_session_from_cookie()
    if "user" not in st.session_state:
        auth_ui()
        st.stop()

    # hard check: ensure supabase sees user as authenticated for this run
    try:
        u = supa().auth.get_user().user
        if u:
            st.session_state.user = u
    except Exception:
        # if session is invalid, force login UI
        st.session_state.pop("user", None)
        auth_ui()
        st.stop()

    return str(st.session_state.user.id)

# =============================================================================
# APP STATE
# =============================================================================
def _reset_learning_state():
    for k in ["queue", "idx", "answered", "last_ok", "last_correct_index", "last_selected_index"]:
        st.session_state.pop(k, None)

def _reset_exam_state():
    for k in ["exam_queue", "exam_idx", "exam_correct", "exam_done", "exam_answered", "exam_last_ok"]:
        st.session_state.pop(k, None)

def _reset_app_state(hard: bool = False):
    _reset_learning_state()
    _reset_exam_state()
    if hard:
        for k in ["mode", "sel_category", "sel_subchapter", "only_unseen", "only_wrong"]:
            st.session_state.pop(k, None)

# =============================================================================
# QUEUES
# =============================================================================
def build_learning_queue(
    questions: List[Dict[str, Any]],
    progress: Dict[str, Dict[str, Any]],
    category: str,
    subchapter: str,
    only_unseen: bool,
    only_wrong: bool,
) -> List[Dict[str, Any]]:
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

def build_exam_queue(questions: List[Dict[str, Any]], n: int = 40) -> List[Dict[str, Any]]:
    base = list(questions)
    random.shuffle(base)
    return base[:n]

# =============================================================================
# PROGRESS / STATS
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
            qid = str(q.get("id"))
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

def overall_progress_pct(questions: List[Dict[str, Any]], progress: Dict[str, Dict[str, Any]]) -> int:
    total = len(questions)
    learned = 0
    for q in questions:
        qid = str(q.get("id"))
        row = progress.get(qid)
        if row and int(row.get("seen", 0)) > 0:
            learned += 1
    return int(round((learned / total) * 100)) if total else 0

# =============================================================================
# UI / STYLES
# =============================================================================
def inject_css():
    st.markdown(
        """
<style>
.block-container { padding-top: 1.2rem; max-width: 1100px; }
div.stButton > button { width:100%; padding:0.9rem 1rem; border-radius:12px; font-size:1rem; }
.pp-card { border:1px solid rgba(255,255,255,0.10); border-radius:14px; padding:1rem 1.1rem; background: rgba(255,255,255,0.03); }
.pp-muted { opacity:0.85; font-size:0.95rem; }
.pp-kpi { border:1px solid rgba(255,255,255,0.10); border-radius:14px; padding:0.9rem 1rem; background: rgba(255,255,255,0.03); }
.pp-grid { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:0.8rem; }
@media (max-width: 900px){ .pp-grid{ grid-template-columns: 1fr; } }
hr { border:none; height:1px; background: rgba(255,255,255,0.10); margin: 1rem 0; }
</style>
""",
        unsafe_allow_html=True,
    )

def nav_sidebar():
    st.sidebar.markdown("## Navigation")
    c1, c2, c3 = st.sidebar.columns(3)
    if c1.button("Übersicht", use_container_width=True):
        st.session_state.page = "dashboard"
        _reset_learning_state()
        _reset_exam_state()
        st.rerun()
    if c2.button("Lernen", use_container_width=True):
        st.session_state.page = "learn"
        _reset_exam_state()
        st.rerun()
    if c3.button("Prüfung", use_container_width=True):
        st.session_state.page = "exam"
        _reset_learning_state()
        st.rerun()

# =============================================================================
# PAGES
# =============================================================================
def page_dashboard(uid: str, questions: List[Dict[str, Any]], progress: Dict[str, Dict[str, Any]]):
    st.title("Fortschritt Übersicht")

    qidx = index_questions(questions)
    stats = compute_progress_by_cluster(qidx, progress)
    overall = overall_progress_pct(questions, progress)

    runs = db_list_exam_runs(uid, limit=200)
    attempts = len(runs)
    passed = sum(1 for r in runs if bool(r.get("passed")))
    pass_rate = int(round((passed / attempts) * 100)) if attempts else 0
    best = 0
    if runs:
        for r in runs:
            total = int(r.get("total") or 0)
            corr = int(r.get("correct") or 0)
            if total:
                best = max(best, int(round((corr / total) * 100)))

    st.markdown(
        f"""
<div class="pp-grid">
  <div class="pp-kpi"><b>Gesamt gelernt</b><br>{overall}%</div>
  <div class="pp-kpi"><b>Prüfungen</b><br>{attempts} Versuche</div>
  <div class="pp-kpi"><b>Beste Prüfung</b><br>{best}% (Passrate {pass_rate}%)</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    colA, colB = st.columns([1, 1])
    if colA.button("Lernsession starten"):
        st.session_state.page = "learn"
        _reset_learning_state()
        st.rerun()
    if colB.button("Prüfungssimulation starten (40)"):
        st.session_state.page = "exam"
        _reset_exam_state()
        st.rerun()

    st.markdown("## Kapitel / Unterkapitel")
    for cat, subs in REQUIRED.items():
        st.markdown(f"### {cat}")
        for sub, expected_total in subs.items():
            s = stats.get(cat, {}).get(sub, {"total": 0, "learned": 0, "correct_total": 0})
            total = expected_total if expected_total else s["total"]
            learned = s["learned"]
            correct_total = s["correct_total"]
            learned_pct = int(round((learned / total) * 100)) if total else 0
            correct_pct = int(round((correct_total / total) * 100)) if total else 0

            if learned == 0:
                st.caption(f"{sub} ({total}) — nicht begonnen")
            else:
                st.caption(f"{sub} ({total}) — gelernt: {learned_pct}% ({learned}), richtig: {correct_pct}% ({correct_total})")

    if runs:
        st.markdown("## Letzte Prüfungen")
        for r in runs[:10]:
            total = int(r.get("total") or 0)
            corr = int(r.get("correct") or 0)
            pct = int(round((corr / total) * 100)) if total else 0
            ok = "BESTANDEN" if bool(r.get("passed")) else "NICHT bestanden"
            st.caption(f"{pct}% ({corr}/{total}) — {ok}")

def page_learn(uid: str, questions: List[Dict[str, Any]], progress: Dict[str, Dict[str, Any]], wiki: Dict[str, Any]):
    st.title("Lernen")

    cats = sorted(set((q.get("category") or "").strip() for q in questions if q.get("category")))
    sel_category = st.selectbox("Kategorie", ["Alle"] + cats, key="sel_category")

    subs = sorted(set((q.get("subchapter") or "").strip() for q in questions if q.get("subchapter")))
    if sel_category != "Alle":
        subs = sorted(set((q.get("subchapter") or "").strip() for q in questions if (q.get("category") or "") == sel_category))
    sel_subchapter = st.selectbox("Unterkapitel", ["Alle"] + subs, key="sel_subchapter")

    only_unseen = st.checkbox("Nur ungelernt", value=st.session_state.get("only_unseen", False), key="only_unseen")
    only_wrong = st.checkbox("Nur falsch beantwortete", value=st.session_state.get("only_wrong", False), key="only_wrong")

    col1, col2 = st.columns([1, 1])
    start = col1.button("Session starten / neu mischen")
    if col2.button("Zur Übersicht"):
        st.session_state.page = "dashboard"
        _reset_learning_state()
        st.rerun()

    if start or "queue" not in st.session_state:
        st.session_state.queue = build_learning_queue(
            questions=questions,
            progress=progress,
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

    st.markdown(
        f"""<div class="pp-card"><div><b>{question}</b></div>
<div class="pp-muted">{q.get("category","")} · {q.get("subchapter","")} · {idx+1}/{len(queue)}</div></div>""",
        unsafe_allow_html=True,
    )
    st.write("")

    # Image (first matched figure)
    figs = q.get("figures") or []
    if figs:
        f0 = figs[0]
        fig_no = f0.get("figure")
        page_1based = int(f0.get("bilder_page") or 0)
        if page_1based > 0:
            png = render_pdf_page_png(str(BILDER_PDF), page_1based=page_1based, zoom=2.0)
            if png:
                st.image(png, caption=f"Abbildung {fig_no} (Bilder.pdf Seite {page_1based})", use_container_width=True)

    labels = ["A", "B", "C", "D"]

    if not st.session_state.get("answered", False):
        for i, opt in enumerate(options[:4]):
            if st.button(f"{labels[i]}) {opt}", key=f"learn_{qid}_{i}"):
                ok = (i == correct_index)
                db_upsert_progress(uid, qid, ok)

                # refresh progress in state
                st.session_state.progress = db_load_progress(uid)
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

        # Wiki (static)
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

        with st.expander("Wiki (zentral, statisch)", expanded=True):
            if isinstance(w, dict):
                explanation = (w.get("explanation") or "").strip()
                merksatz = (w.get("merksatz") or "").strip()
                source = (w.get("source") or "").strip()
                if explanation:
                    st.markdown(explanation)
                if merksatz:
                    st.markdown(f"**Merksatz:** {merksatz}")
                if source:
                    st.markdown(f"**Quelle:** {source}")
            else:
                st.info("Kein Wiki-Eintrag vorhanden (wiki_content.json ergänzen).")

        # Per-user note
        existing_note = db_get_note(uid, qid)
        with st.expander("Deine Bemerkung (nur für dich)", expanded=False):
            note_text = st.text_area("Notiz", value=existing_note, key=f"note_{qid}", height=120)
            if st.button("Notiz speichern", key=f"save_note_{qid}"):
                if db_upsert_note(uid, qid, note_text):
                    st.success("Gespeichert")
                else:
                    st.error("Speichern fehlgeschlagen (notes Tabelle/RLS prüfen).")

        c1, c2, c3 = st.columns([1, 1, 1])
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

        if c3.button("Zur Übersicht"):
            st.session_state.page = "dashboard"
            _reset_learning_state()
            st.rerun()

def page_exam(uid: str, questions: List[Dict[str, Any]], wiki: Dict[str, Any]):
    st.title("Prüfungssimulation (40)")

    col1, col2 = st.columns([1, 1])
    if col1.button("Neue Prüfung starten"):
        _reset_exam_state()
        st.rerun()
    if col2.button("Zur Übersicht"):
        st.session_state.page = "dashboard"
        _reset_exam_state()
        st.rerun()

    if "exam_queue" not in st.session_state:
        st.session_state.exam_queue = build_exam_queue(questions, n=40)
        st.session_state.exam_idx = 0
        st.session_state.exam_correct = 0
        st.session_state.exam_done = False
        st.session_state.exam_answered = False
        st.session_state.exam_last_ok = None

    qlist: List[Dict[str, Any]] = st.session_state.exam_queue
    i = int(st.session_state.exam_idx)
    total = len(qlist)

    if st.session_state.exam_done:
        correct = int(st.session_state.exam_correct)
        pct = int(round((correct / total) * 100)) if total else 0
        passed = pct >= PASS_PCT

        st.markdown(
            f"""<div class="pp-card"><div><b>Ergebnis</b></div>
<div class="pp-muted">{pct}% ({correct}/{total}) — {'BESTANDEN' if passed else 'NICHT bestanden'} (Schwelle {int(PASS_PCT)}%)</div></div>""",
            unsafe_allow_html=True,
        )

        if st.button("Ergebnis speichern"):
            db_insert_exam_run(uid, total=total, correct=correct, passed=passed)
            st.success("Gespeichert (falls exam_runs Tabelle vorhanden).")

        st.write("")
        st.markdown("## Verlauf (letzte 10)")
        runs = db_list_exam_runs(uid, limit=10)
        if not runs:
            st.info("Kein Verlauf (oder exam_runs Tabelle fehlt).")
        else:
            for r in runs:
                t = int(r.get("total") or 0)
                c = int(r.get("correct") or 0)
                p = int(round((c / t) * 100)) if t else 0
                ok = "BESTANDEN" if bool(r.get("passed")) else "NICHT bestanden"
                st.caption(f"{p}% ({c}/{t}) — {ok}")
        return

    if i >= total:
        st.session_state.exam_done = True
        st.rerun()

    q = qlist[i]
    qid = str(q.get("id"))
    question = (q.get("question") or "").strip()
    options = q.get("options") or []
    correct_index = int(q.get("correctIndex", -1))

    st.markdown(
        f"""<div class="pp-card"><div><b>{question}</b></div>
<div class="pp-muted">Frage {i+1}/{total}</div></div>""",
        unsafe_allow_html=True,
    )
    st.write("")

    # image if matched
    figs = q.get("figures") or []
    if figs:
        f0 = figs[0]
        fig_no = f0.get("figure")
        page_1based = int(f0.get("bilder_page") or 0)
        if page_1based > 0:
            png = render_pdf_page_png(str(BILDER_PDF), page_1based=page_1based, zoom=2.0)
            if png:
                st.image(png, caption=f"Abbildung {fig_no} (Bilder.pdf Seite {page_1based})", use_container_width=True)

    labels = ["A", "B", "C", "D"]

    if not st.session_state.exam_answered:
        for idx_opt, opt in enumerate(options[:4]):
            if st.button(f"{labels[idx_opt]}) {opt}", key=f"exam_{qid}_{idx_opt}"):
                ok = (idx_opt == correct_index)
                if ok:
                    st.session_state.exam_correct = int(st.session_state.exam_correct) + 1
                st.session_state.exam_answered = True
                st.session_state.exam_last_ok = ok
                st.session_state.exam_last_selected = idx_opt
                st.session_state.exam_last_correct = correct_index
                st.rerun()
    else:
        ok = bool(st.session_state.exam_last_ok)
        if ok:
            st.success("Richtig")
        else:
            st.error("Falsch")
            ci = int(st.session_state.exam_last_correct)
            if 0 <= ci < len(options):
                st.info(f"Richtig ist: {labels[ci]}) {options[ci]}")

        # optional: show static wiki (does not affect score)
        wiki_key = (q.get("wiki_key") or "").strip()
        w = None
        if qid and qid in wiki:
            w = wiki.get(qid)
        elif wiki_key and wiki_key in wiki:
            w = wiki.get(wiki_key)

        with st.expander("Wiki (zentral, statisch)", expanded=False):
            if isinstance(w, dict):
                explanation = (w.get("explanation") or "").strip()
                merksatz = (w.get("merksatz") or "").strip()
                source = (w.get("source") or "").strip()
                if explanation:
                    st.markdown(explanation)
                if merksatz:
                    st.markdown(f"**Merksatz:** {merksatz}")
                if source:
                    st.markdown(f"**Quelle:** {source}")
            else:
                st.caption("Kein Wiki-Eintrag vorhanden.")

        if st.button("Nächste Frage"):
            st.session_state.exam_idx = i + 1
            st.session_state.exam_answered = False
            st.session_state.exam_last_ok = None
            st.rerun()

# =============================================================================
# MAIN
# =============================================================================
st.set_page_config(page_title="B-Lizenz Lernapp", layout="wide")
inject_css()

if not QUESTIONS_PATH.exists():
    st.error("questions.json fehlt")
    st.stop()

# login first (restores cookies before any DB access)
uid = require_login()

questions = load_questions()
wiki = load_wiki()

# refresh progress per run to always show correct dashboard values
progress = db_load_progress(uid)
st.session_state.progress = progress

# navigation
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

nav_sidebar()

# route
page = st.session_state.page
if page == "learn":
    page_learn(uid, questions, progress, wiki)
elif page == "exam":
    page_exam(uid, questions, wiki)
else:
    page_dashboard(uid, questions, progress)
