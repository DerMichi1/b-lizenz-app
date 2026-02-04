import json
import random
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import streamlit as st
from supabase import create_client, Client


# =============================================================================
# CONFIG / FILES
# =============================================================================
APP_DIR = Path(__file__).parent
QUESTIONS_PATH = APP_DIR / "questions.json"
BILDER_PDF = APP_DIR / "Bilder.pdf"          # <-- updated
WIKI_PATH = APP_DIR / "wiki_content.json"    # must contain entry per question id


def cfg(path: str, default: str = "") -> str:
    parts = path.split(".")
    cur: Any = st.secrets
    for p in parts:
        if p not in cur:
            return default
        cur = cur[p]
    return (str(cur) if cur is not None else "").strip()


PASS_PCT = float(cfg("PASS_PCT", "75"))

SUPABASE_URL = cfg("supabase.url")
SUPABASE_SERVICE_ROLE_KEY = cfg("supabase.service_role_key")
SUPABASE_ANON_KEY = cfg("supabase.anon_key")

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
# SUPABASE CLIENT (DB only)
# =============================================================================
@st.cache_resource(show_spinner=False)
def supa() -> Client:
    if not SUPABASE_URL:
        raise RuntimeError("Supabase secret fehlt: [supabase].url")

    key = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY
    if not key:
        raise RuntimeError("Supabase secret fehlt: [supabase].service_role_key (empfohlen) oder [supabase].anon_key")

    return create_client(SUPABASE_URL, key)


# =============================================================================
# QUESTIONS / WIKI
# =============================================================================
@st.cache_data(show_spinner=False)
def load_questions() -> List[Dict[str, Any]]:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"questions.json fehlt: {QUESTIONS_PATH}")
    return json.loads(QUESTIONS_PATH.read_text("utf-8"))


def build_wiki_map_from_questions(questions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Primary source: questions.json -> question["wiki"].
    Returns a map keyed by question id (e.g., "Q081").
    """
    out: Dict[str, Dict[str, Any]] = {}
    for q in questions:
        qid = str(q.get("id") or "").strip()
        if not qid:
            continue
        w = q.get("wiki")
        if isinstance(w, dict):
            out[qid] = w
        else:
            out[qid] = {"explanation": "", "merksatz": "", "links": []}
    return out


def index_questions(questions: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    idx: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for q in questions:
        cat = (q.get("category") or "").strip()
        sub = (q.get("subchapter") or "").strip()
        idx.setdefault((cat, sub), []).append(q)
    return idx


# =============================================================================
# PDF IMAGE RENDER (Bilder.pdf)
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
# AUTH (Streamlit built-in OIDC: Google)
# =============================================================================
def require_login() -> None:
    if not getattr(st.user, "is_logged_in", False):
        st.title("B-Lizenz Lernapp")
        st.caption("Bitte mit Google anmelden.")
        st.button("Mit Google anmelden", on_click=st.login, use_container_width=True)
        st.stop()


def user_claims() -> Dict[str, str]:
    return {
        "email": (getattr(st.user, "email", "") or "").strip(),
        "name": (getattr(st.user, "name", "") or "").strip(),
        "sub": (getattr(st.user, "sub", "") or "").strip(),
    }


def stable_user_id(claims: Dict[str, str]) -> str:
    basis = claims.get("sub") or claims.get("email") or claims.get("name") or "anonymous"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"bliz:{basis}"))


def ensure_user_registered(claims: Dict[str, str]) -> None:
    provider = "google"
    sub = claims.get("sub") or ""
    if not sub:
        return

    s = supa()
    existing = (
        s.table("app_users")
        .select("id")
        .eq("provider", provider)
        .eq("provider_sub", sub)
        .limit(1)
        .execute()
    )

    if existing.data:
        return

    s.table("app_users").insert(
        {
            "provider": provider,
            "provider_sub": sub,
            "email": claims.get("email") or None,
            "name": claims.get("name") or None,
        }
    ).execute()


# =============================================================================
# DB: progress + notes + exam_runs
# =============================================================================
def db_load_progress(uid: str) -> Dict[str, Dict[str, Any]]:
    r = supa().table("progress").select("*").eq("user_id", uid).execute()
    return {str(x["question_id"]): x for x in (r.data or [])}


def db_upsert_progress(uid: str, qid: str, ok: bool):
    s = supa()
    r = s.table("progress").select("*").eq("user_id", uid).eq("question_id", qid).limit(1).execute()
    if r.data:
        row = r.data[0]
        s.table("progress").update(
            {
                "seen": int(row.get("seen", 0)) + 1,
                "correct": int(row.get("correct", 0)) + (1 if ok else 0),
                "wrong": int(row.get("wrong", 0)) + (0 if ok else 1),
            }
        ).eq("user_id", uid).eq("question_id", qid).execute()
    else:
        s.table("progress").insert(
            {
                "user_id": uid,
                "question_id": qid,
                "seen": 1,
                "correct": 1 if ok else 0,
                "wrong": 0 if ok else 1,
            }
        ).execute()


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
        supa().table("exam_runs").insert(
            {"user_id": uid, "total": total, "correct": correct, "passed": passed}
        ).execute()
    except Exception:
        pass


def db_list_exam_runs(uid: str, limit: int = 50) -> List[Dict[str, Any]]:
    try:
        r = (
            supa()
            .table("exam_runs")
            .select("*")
            .eq("user_id", uid)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return list(r.data or [])
    except Exception:
        return []


# =============================================================================
# APP STATE
# =============================================================================
def _reset_learning_state():
    for k in [
        "queue", "idx", "answered", "last_ok", "last_correct_index", "last_selected_index",
        "learn_started", "learn_plan"
    ]:
        st.session_state.pop(k, None)


def _reset_exam_state():
    for k in [
        "exam_queue", "exam_idx", "exam_correct", "exam_done", "exam_answered",
        "exam_last_ok", "exam_last_selected", "exam_last_correct", "exam_started", "exam_plan"
    ]:
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
        qset = [
            q
            for q in qset
            if (str(q.get("id")) not in progress) or int(progress[str(q.get("id"))].get("seen", 0)) == 0
        ]
    if only_wrong:
        qset = [
            q
            for q in qset
            if (str(q.get("id")) in progress) and int(progress[str(q.get("id"))].get("wrong", 0)) > 0
        ]

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


def overall_correct_wrong(progress: Dict[str, Dict[str, Any]]) -> Tuple[int, int]:
    c = 0
    w = 0
    for row in progress.values():
        c += int(row.get("correct", 0))
        w += int(row.get("wrong", 0))
    return c, w


def weakest_subchapters(
    stats: Dict[str, Dict[str, Dict[str, int]]],
    min_seen: int = 6,
    topn: int = 8
) -> List[Dict[str, Any]]:
    """
    Rank subchapters by low accuracy. We require a minimum number of answered attempts
    (correct_total + wrong_total) to avoid nonsense on fresh accounts.
    """
    rows: List[Dict[str, Any]] = []
    for cat, subs in stats.items():
        for sub, s in subs.items():
            attempts = int(s.get("correct_total", 0)) + int(s.get("wrong_total", 0))
            if attempts < min_seen:
                continue
            acc = (int(s.get("correct_total", 0)) / attempts) if attempts else 0.0
            rows.append({
                "category": cat,
                "subchapter": sub,
                "attempts": attempts,
                "accuracy": acc,
                "wrong": int(s.get("wrong_total", 0)),
            })
    rows.sort(key=lambda r: (r["accuracy"], -r["attempts"]))
    return rows[:topn]


# =============================================================================
# UI / STYLES
# =============================================================================
def inject_css():
    st.markdown(
        """
<style>
:root{
  --pp-border: rgba(255,255,255,0.12);
  --pp-bg: rgba(255,255,255,0.04);
  --pp-bg2: rgba(255,255,255,0.06);
  --pp-text-muted: rgba(255,255,255,0.78);
}
.block-container { padding-top: 1.2rem; max-width: 1180px; }
div.stButton > button { width:100%; padding:0.85rem 1rem; border-radius:14px; font-size:1rem; }
.pp-card { border:1px solid var(--pp-border); border-radius:16px; padding:1rem 1.1rem; background: var(--pp-bg); }
.pp-card2 { border:1px solid var(--pp-border); border-radius:16px; padding:1rem 1.1rem; background: var(--pp-bg2); }
.pp-muted { color: var(--pp-text-muted); font-size:0.95rem; }
.pp-kpi { border:1px solid var(--pp-border); border-radius:16px; padding:0.9rem 1rem; background: var(--pp-bg); }
.pp-grid { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:0.8rem; }
@media (max-width: 1100px){ .pp-grid{ grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 700px){ .pp-grid{ grid-template-columns: 1fr; } }
hr { border:none; height:1px; background: var(--pp-border); margin: 1rem 0; }
.small { font-size: 0.9rem; opacity: 0.85; }
.pp-pill { display:inline-block; padding: 0.25rem 0.55rem; border:1px solid var(--pp-border); border-radius:999px; background: rgba(255,255,255,0.03); font-size:0.85rem; margin-right:0.4rem;}
</style>
""",
        unsafe_allow_html=True,
    )


def nav_sidebar(claims: Dict[str, str]):
    st.sidebar.markdown("## Account")
    st.sidebar.write(claims.get("email") or claims.get("name") or "User")
    st.sidebar.button("Logout", on_click=st.logout, use_container_width=True)

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
# DASHBOARD (more stats + clickable jump-to-learn)
# =============================================================================
def page_dashboard(uid: str, questions: List[Dict[str, Any]], progress: Dict[str, Dict[str, Any]]):
    st.title("Fortschritt Übersicht")

    qidx = index_questions(questions)
    stats = compute_progress_by_cluster(qidx, progress)

    overall = overall_progress_pct(questions, progress)
    c_total, w_total = overall_correct_wrong(progress)
    attempts_total = c_total + w_total
    accuracy_total = int(round((c_total / attempts_total) * 100)) if attempts_total else 0

    runs = db_list_exam_runs(uid, limit=200)
    exam_attempts = len(runs)
    passed = sum(1 for r in runs if bool(r.get("passed")))
    pass_rate = int(round((passed / exam_attempts) * 100)) if exam_attempts else 0
    best = 0
    last_pct = None
    if runs:
        r0 = runs[0]
        t0 = int(r0.get("total") or 0)
        c0 = int(r0.get("correct") or 0)
        last_pct = int(round((c0 / t0) * 100)) if t0 else 0
    for r in runs:
        total = int(r.get("total") or 0)
        corr = int(r.get("correct") or 0)
        if total:
            best = max(best, int(round((corr / total) * 100)))

    # KPI row
    st.markdown(
        f"""
<div class="pp-grid">
  <div class="pp-kpi"><b>Abdeckung</b><br>{overall}%<div class="pp-muted">Fragen mindestens 1× gesehen</div></div>
  <div class="pp-kpi"><b>Trefferquote</b><br>{accuracy_total}%<div class="pp-muted">{c_total} richtig · {w_total} falsch</div></div>
  <div class="pp-kpi"><b>Prüfungen</b><br>{exam_attempts} Versuche<div class="pp-muted">Passrate {pass_rate}%</div></div>
  <div class="pp-kpi"><b>Beste Prüfung</b><br>{best}%<div class="pp-muted">Letzte: {('-' if last_pct is None else str(last_pct)+'%')}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    cA, cB, cC = st.columns([1, 1, 1])
    if cA.button("Lernsession starten"):
        st.session_state.page = "learn"
        st.session_state.learn_plan = {"category": "Alle", "subchapter": "Alle", "only_unseen": False, "only_wrong": False}
        _reset_learning_state()
        st.session_state.page = "learn"
        st.session_state.learn_plan = {"category": "Alle", "subchapter": "Alle", "only_unseen": False, "only_wrong": False}
        st.rerun()
    if cB.button("Falsche wiederholen"):
        _reset_learning_state()
        st.session_state.page = "learn"
        st.session_state.learn_plan = {"category": "Alle", "subchapter": "Alle", "only_unseen": False, "only_wrong": True}
        st.rerun()
    if cC.button("Prüfungssimulation starten (40)"):
        st.session_state.page = "exam"
        _reset_exam_state()
        st.rerun()

    st.write("")
    # Visuals: progress + accuracy
    v1, v2 = st.columns([1, 1])
    with v1:
        st.markdown('<div class="pp-card2"><b>Abdeckung</b><div class="pp-muted">Wie viel vom Fragenkatalog du schon gesehen hast</div>', unsafe_allow_html=True)
        st.progress(overall / 100.0)
        st.markdown("</div>", unsafe_allow_html=True)
    with v2:
        st.markdown('<div class="pp-card2"><b>Trefferquote</b><div class="pp-muted">Richtig vs. Falsch (alle Versuche)</div>', unsafe_allow_html=True)
        if attempts_total:
            st.bar_chart({"Richtig": c_total, "Falsch": w_total})
        else:
            st.caption("Noch keine beantworteten Fragen.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Weak areas (clickable)
    st.write("")
    st.markdown("## Fokus: Schwache Bereiche")
    weak = weakest_subchapters(stats, min_seen=6, topn=8)
    if not weak:
        st.caption("Noch zu wenig Daten. Beantworte erst ein paar Fragen (mind. 6 pro Unterkapitel), dann wird hier priorisiert.")
    else:
        for row in weak:
            acc = int(round(row["accuracy"] * 100))
            wrong = row["wrong"]
            attempts = row["attempts"]
            cc1, cc2, cc3, cc4 = st.columns([3, 1, 1, 2])
            cc1.markdown(f"**{row['category']} · {row['subchapter']}**  \n<span class='pp-muted'>{attempts} Versuche · {wrong} falsch</span>", unsafe_allow_html=True)
            cc2.markdown(f"<span class='pp-pill'>Acc {acc}%</span>", unsafe_allow_html=True)
            cc3.markdown(f"<span class='pp-pill'>Wrong {wrong}</span>", unsafe_allow_html=True)
            if cc4.button("Gezielt üben", key=f"weak_{row['category']}::{row['subchapter']}"):
                _reset_learning_state()
                st.session_state.page = "learn"
                st.session_state.learn_plan = {
                    "category": row["category"],
                    "subchapter": row["subchapter"],
                    "only_unseen": False,
                    "only_wrong": True,
                }
                st.rerun()

    st.write("")
    st.markdown("## Kapitel / Unterkapitel")
    for cat, subs in REQUIRED.items():
        st.markdown(f"### {cat}")
        for sub, expected_total in subs.items():
            s = stats.get(cat, {}).get(sub, {"total": 0, "learned": 0, "correct_total": 0, "wrong_total": 0})
            total = expected_total if expected_total else s["total"]
            learned = int(s["learned"])
            attempts = int(s["correct_total"]) + int(s["wrong_total"])
            acc = int(round((int(s["correct_total"]) / attempts) * 100)) if attempts else 0
            learned_pct = int(round((learned / total) * 100)) if total else 0

            line = f"{sub} ({total}) — Abdeckung {learned_pct}% · Acc {acc}% · Versuche {attempts}"
            c1, c2 = st.columns([4, 1])
            c1.caption(line)
            if c2.button("Üben", key=f"sub_{cat}::{sub}"):
                _reset_learning_state()
                st.session_state.page = "learn"
                st.session_state.learn_plan = {
                    "category": cat,
                    "subchapter": sub,
                    "only_unseen": False,
                    "only_wrong": False,
                }
                st.rerun()

    if runs:
        st.write("")
        st.markdown("## Letzte Prüfungen")
        for r in runs[:10]:
            total = int(r.get("total") or 0)
            corr = int(r.get("correct") or 0)
            pct = int(round((corr / total) * 100)) if total else 0
            ok = "BESTANDEN" if bool(r.get("passed")) else "NICHT bestanden"
            st.caption(f"{pct}% ({corr}/{total}) — {ok}")


# =============================================================================
# LEARN (questions only after session start; filters hidden during session)
# =============================================================================
def page_learn(uid: str, questions: List[Dict[str, Any]], progress: Dict[str, Dict[str, Any]], wiki: Dict[str, Any]):
    st.title("Lernen")

    # Plan (from dashboard shortcuts) or defaults
    if "learn_plan" not in st.session_state:
        st.session_state.learn_plan = {"category": "Alle", "subchapter": "Alle", "only_unseen": False, "only_wrong": False}

    if "learn_started" not in st.session_state:
        st.session_state.learn_started = False

    # --- PRE-START: show filters + start
    if not st.session_state.learn_started:
        plan = st.session_state.learn_plan

        cats = sorted(set((q.get("category") or "").strip() for q in questions if q.get("category")))
        sel_category = st.selectbox("Kategorie", ["Alle"] + cats, index=(["Alle"] + cats).index(plan["category"]) if plan["category"] in (["Alle"] + cats) else 0, key="sel_category")

        subs = sorted(set((q.get("subchapter") or "").strip() for q in questions if q.get("subchapter")))
        if sel_category != "Alle":
            subs = sorted(set((q.get("subchapter") or "").strip() for q in questions if (q.get("category") or "") == sel_category))
        sel_subchapter = st.selectbox("Unterkapitel", ["Alle"] + subs, index=(["Alle"] + subs).index(plan["subchapter"]) if plan["subchapter"] in (["Alle"] + subs) else 0, key="sel_subchapter")

        only_unseen = st.checkbox("Nur ungelernt", value=bool(plan.get("only_unseen", False)), key="only_unseen")
        only_wrong = st.checkbox("Nur falsch beantwortete", value=bool(plan.get("only_wrong", False)), key="only_wrong")

        c1, c2 = st.columns([1, 1])
        start = c1.button("Session starten", type="primary")
        if c2.button("Zur Übersicht"):
            st.session_state.page = "dashboard"
            _reset_learning_state()
            st.rerun()

        if start:
            st.session_state.learn_plan = {"category": sel_category, "subchapter": sel_subchapter, "only_unseen": only_unseen, "only_wrong": only_wrong}
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
            st.session_state.learn_started = True
            st.rerun()

        st.stop()

    # --- IN SESSION: hide filters, show compact session header + actions
    plan = st.session_state.learn_plan
    queue: List[Dict[str, Any]] = st.session_state.get("queue", [])
    idx: int = int(st.session_state.get("idx", 0))

    if not queue:
        st.warning("Keine Fragen für diese Auswahl.")
        st.session_state.learn_started = False
        st.stop()

    if idx >= len(queue):
        st.session_state.idx = 0
        idx = 0

    header = f"{plan['category']} · {plan['subchapter']}  |  {'nur ungelernt' if plan['only_unseen'] else ''}{' ' if (plan['only_unseen'] and plan['only_wrong']) else ''}{'nur falsch' if plan['only_wrong'] else ''}"
    st.caption(header.strip(" |"))

    h1, h2, h3 = st.columns([1, 1, 1])
    if h1.button("Session beenden"):
        st.session_state.learn_started = False
        _reset_learning_state()
        st.rerun()
    if h2.button("Neu mischen"):
        _reset_learning_state()
        st.session_state.learn_started = False
        st.rerun()
    if h3.button("Zur Übersicht"):
        st.session_state.page = "dashboard"
        _reset_learning_state()
        st.rerun()

    st.write("")

    q = queue[idx]
    qid = str(q.get("id"))
    question = (q.get("question") or "").strip()
    options = q.get("options") or []
    correct_index = int(q.get("correctIndex", -1))

    st.markdown(
        f"""<div class="pp-card"><div><b>{question}</b></div>
<div class="pp-muted">{q.get("category","")} · {q.get("subchapter","")} · {idx+1}/{len(queue)} · ID {qid}</div></div>""",
        unsafe_allow_html=True,
    )
    st.write("")

    # Render all linked figures (if present)
    figs = q.get("figures") or []
    if figs:
        for f in figs[:3]:
            fig_no = f.get("figure")
            page_1based = int(f.get("bilder_page") or 0)
            if page_1based > 0:
                png = render_pdf_page_png(str(BILDER_PDF), page_1based=page_1based, zoom=2.0)
                if png:
                    st.image(png, caption=f"Abbildung {fig_no} (Bilder.pdf Seite {page_1based})", use_container_width=True)

    labels = ["A", "B", "C", "D"]

    if not st.session_state.get("answered", False):
        for i_opt, opt in enumerate(options[:4]):
            if st.button(f"{labels[i_opt]}) {opt}", key=f"learn_{qid}_{i_opt}"):
                ok = (i_opt == correct_index)
                db_upsert_progress(uid, qid, ok)

                st.session_state.progress = db_load_progress(uid)
                st.session_state.answered = True
                st.session_state.last_ok = ok
                st.session_state.last_selected_index = i_opt
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

# --- Wiki (from questions.json) ---
w = wiki.get(qid, {})
with st.expander("Wiki (kurz + Merksatz + Links)", expanded=True):
    explanation = (w.get("explanation") or "").strip()
    merksatz = (w.get("merksatz") or "").strip()
    links = w.get("links") or []

    if explanation:
        st.markdown(explanation)
    else:
        st.warning("Wiki-Inhalt ist leer. (In questions.json -> wiki.explanation befüllen)")

    if merksatz:
        st.markdown(f"**Merksatz:** {merksatz}")

    if links:
        st.markdown("**Weiterlesen (offizielle Quellen):**")
        for li in links:
            title = (li.get("title") or "Link").strip()
            url = (li.get("url") or "").strip()
            if url:
                st.markdown(f"- [{title}]({url})")
    else:
        st.caption("Keine Links hinterlegt.")

# --- KI-Nachfrage / Chatbot zur aktuellen Frage ---
ai_cfg = q.get("ai") if isinstance(q.get("ai"), dict) else {}
ai_allowed = bool(ai_cfg.get("allowed", True))

with st.expander("KI-Nachfrage zur Frage", expanded=False):
    if not ai_allowed:
        st.caption("KI-Nachfragen sind für diese Frage deaktiviert.")
    else:
        # Lazy init chat storage
        if "ai_chat" not in st.session_state:
            st.session_state.ai_chat = {}
        st.session_state.ai_chat.setdefault(qid, [])
        history = st.session_state.ai_chat[qid]

        # Render history
        for msg in history:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            with st.chat_message(role):
                st.markdown(content)

        user_text = st.chat_input("Frage zur aktuellen Aufgabe eingeben…", key=f"chat_in_{qid}")
        if user_text:
            history.append({"role": "user", "content": user_text})

            # Build grounded prompt: question + options + correct + wiki + figures hint
            system_hint = (ai_cfg.get("system_hint") or "Antworte strikt faktenbasiert. Wenn unsicher, sag es. Verweise auf offizielle Quellen/Links.")
            context = (ai_cfg.get("context") or "").strip()

            prompt = f"""
SYSTEM:
{system_hint}

KONTEXT:
{context}

FRAGE:
{question}

OPTIONEN:
A) {options[0] if len(options)>0 else ""}
B) {options[1] if len(options)>1 else ""}
C) {options[2] if len(options)>2 else ""}
D) {options[3] if len(options)>3 else ""}

RICHTIGE OPTION (Index):
{correct_index}

WIKI-KURZ:
{(w.get("explanation") or "").strip()}

MERKSATZ:
{(w.get("merksatz") or "").strip()}

USER-FRAGE:
{user_text}
""".strip()

            # Call OpenAI (requires secret: openai.api_key)
            api_key = ""
            try:
                api_key = (st.secrets.get("openai", {}).get("api_key", "") or "").strip()
            except Exception:
                api_key = ""

            if not api_key:
                history.append({"role": "assistant", "content": "OpenAI API Key fehlt in st.secrets: [openai].api_key"})
                st.rerun()

            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)

                resp = client.chat.completions.create(
                    model=(st.secrets.get("openai", {}).get("model", "gpt-4.1-mini")),
                    messages=[
                        {"role": "system", "content": system_hint},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                answer = resp.choices[0].message.content or ""
                history.append({"role": "assistant", "content": answer})
            except Exception as e:
                history.append({"role": "assistant", "content": f"KI-Fehler: {e}"})

            st.rerun()

        existing_note = db_get_note(uid, qid)
        with st.expander("Deine Bemerkung (nur für dich)", expanded=False):
            note_text = st.text_area("Notiz", value=existing_note, key=f"note_{qid}", height=120)
            if st.button("Notiz speichern", key=f"save_note_{qid}"):
                if db_upsert_note(uid, qid, note_text):
                    st.success("Gespeichert")
                else:
                    st.error("Speichern fehlgeschlagen (notes Tabelle/RLS prüfen).")

        c1, c2 = st.columns([1, 1])
        if c1.button("Nächste Frage", type="primary"):
            st.session_state.idx = (idx + 1) % len(queue)
            st.session_state.answered = False
            st.session_state.last_ok = None
            st.session_state.last_correct_index = None
            st.session_state.last_selected_index = None
            st.rerun()
        if c2.button("Gezielt: nur falsch in diesem Set"):
            # Rebuild queue with only_wrong=True, keep same cat/sub
            plan2 = dict(plan)
            plan2["only_wrong"] = True
            st.session_state.learn_plan = plan2
            st.session_state.queue = build_learning_queue(
                questions=questions,
                progress=st.session_state.progress,
                category=plan2["category"],
                subchapter=plan2["subchapter"],
                only_unseen=bool(plan2["only_unseen"]),
                only_wrong=True,
            )
            st.session_state.idx = 0
            st.session_state.answered = False
            st.rerun()


# =============================================================================
# EXAM (questions only after start; no filters bar)
# =============================================================================
def page_exam(uid: str, questions: List[Dict[str, Any]], wiki: Dict[str, Any]):
    st.title("Prüfungssimulation (40)")

    if "exam_started" not in st.session_state:
        st.session_state.exam_started = False

    # Pre-start screen
    if not st.session_state.exam_started:
        st.markdown(
            """<div class="pp-card2"><b>Regeln</b>
<div class="pp-muted">40 zufällige Fragen. Ergebnis am Ende, optional speichern.</div></div>""",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([1, 1])
        if c1.button("Prüfung starten", type="primary"):
            _reset_exam_state()
            st.session_state.exam_queue = build_exam_queue(questions, n=40)
            st.session_state.exam_idx = 0
            st.session_state.exam_correct = 0
            st.session_state.exam_done = False
            st.session_state.exam_answered = False
            st.session_state.exam_last_ok = None
            st.session_state.exam_started = True
            st.rerun()
        if c2.button("Zur Übersicht"):
            st.session_state.page = "dashboard"
            _reset_exam_state()
            st.rerun()
        st.stop()

    # In exam session
    top1, top2, top3 = st.columns([1, 1, 1])
    if top1.button("Prüfung abbrechen"):
        st.session_state.exam_started = False
        _reset_exam_state()
        st.rerun()
    if top2.button("Neu starten"):
        st.session_state.exam_started = False
        _reset_exam_state()
        st.rerun()
    if top3.button("Zur Übersicht"):
        st.session_state.page = "dashboard"
        st.session_state.exam_started = False
        _reset_exam_state()
        st.rerun()

    qlist: List[Dict[str, Any]] = st.session_state.exam_queue
    i = int(st.session_state.exam_idx)
    total = len(qlist)

    # progress indicator
    if total:
        st.progress(min(1.0, i / total))

    if st.session_state.exam_done:
        correct = int(st.session_state.exam_correct)
        pct = int(round((correct / total) * 100)) if total else 0
        passed = pct >= PASS_PCT

        st.markdown(
            f"""<div class="pp-card"><div><b>Ergebnis</b></div>
<div class="pp-muted">{pct}% ({correct}/{total}) — {'BESTANDEN' if passed else 'NICHT bestanden'} (Schwelle {int(PASS_PCT)}%)</div></div>""",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([1, 1])
        if c1.button("Ergebnis speichern", type="primary"):
            db_insert_exam_run(uid, total=total, correct=correct, passed=passed)
            st.success("Gespeichert")
        if c2.button("Zur Übersicht"):
            st.session_state.page = "dashboard"
            st.session_state.exam_started = False
            _reset_exam_state()
            st.rerun()

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
<div class="pp-muted">Frage {i+1}/{total} · ID {qid}</div></div>""",
        unsafe_allow_html=True,
    )
    st.write("")

    figs = q.get("figures") or []
    if figs:
        for f in figs[:2]:
            fig_no = f.get("figure")
            page_1based = int(f.get("bilder_page") or 0)
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

        if st.button("Nächste Frage", type="primary"):
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

require_login()
claims = user_claims()

ensure_user_registered(claims)

uid = stable_user_id(claims)
questions = load_questions()
wiki = build_wiki_map_from_questions(questions)

progress = db_load_progress(uid)
st.session_state.progress = progress

if "page" not in st.session_state:
    st.session_state.page = "dashboard"

nav_sidebar(claims)

page = st.session_state.page
if page == "learn":
    page_learn(uid, questions, progress, wiki)
elif page == "exam":
    page_exam(uid, questions, wiki)
else:
    page_dashboard(uid, questions, progress)
