import json
import random
import re
import uuid
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from supabase import Client, create_client

# =============================================================================
# CONFIG / FILES
# =============================================================================
APP_DIR = Path(__file__).parent
QUESTIONS_PATH = APP_DIR / "questions.json"
BILDER_PDF = APP_DIR / "Bilder.pdf"  # PDF with figures (page numbers referenced by questions[*].figures[*].bilder_page)
FIGURE_MAP_PATH = APP_DIR / "figure_map.json"  # supports {"47": 14} OR {"47": {"page":14,"clip":[...] }}


def cfg(path: str, default: str = "") -> str:
    """
    Read from Streamlit secrets only.
    """
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

OPENAI_API_KEY = cfg("openai.api_key")
OPENAI_MODEL = cfg("openai.model", "gpt-4.1-mini")

# Exam timer
EXAM_DURATION_SEC = int(float(cfg("exam.duration_minutes", "60")) * 60)  # default 60 min


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
        raise RuntimeError(
            "Supabase secret fehlt: [supabase].service_role_key (empfohlen) oder [supabase].anon_key"
        )

    return create_client(SUPABASE_URL, key)


# =============================================================================
# QUESTIONS / WIKI / AI (single source of truth: questions.json)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_questions() -> List[Dict[str, Any]]:
    """Load questions from the bundled questions.json.

    Important: The app can override this file at runtime via the sidebar uploader.
    The override is stored in st.session_state (not on disk).
    """
    override = st.session_state.get("questions_override")
    if isinstance(override, list) and override:
        return override

    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"questions.json fehlt: {QUESTIONS_PATH}")

    data = json.loads(QUESTIONS_PATH.read_text("utf-8"))
    if not isinstance(data, list):
        raise ValueError("questions.json muss eine Liste sein.")
    return data


def get_wiki(q: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports both keys:
      wiki.links  (old)
      wiki.nachlesen (new)
    """
    w = q.get("wiki")
    if isinstance(w, dict):
        links = w.get("links")
        if not isinstance(links, list):
            links = w.get("nachlesen") or []
        return {
            "explanation": (w.get("explanation") or "").strip(),
            "merksatz": (w.get("merksatz") or "").strip(),
            "links": links if isinstance(links, list) else [],
            "reliability_note": (w.get("reliability_note") or "").strip(),
        }
    return {"explanation": "", "merksatz": "", "links": [], "reliability_note": ""}


def get_ai_cfg(q: Dict[str, Any]) -> Dict[str, Any]:
    a = q.get("ai")
    if isinstance(a, dict):
        return {
            "allowed": bool(a.get("allowed", True)),
            "context": (a.get("context") or "").strip(),
            "system_hint": (a.get("system_hint") or "").strip()
            or "Antworte strikt faktenbasiert. Wenn unsicher, sag es.",
        }
    return {"allowed": True, "context": "", "system_hint": "Antworte strikt faktenbasiert. Wenn unsicher, sag es."}


def validate_questions(questions: List[Dict[str, Any]]) -> Dict[str, int]:
    """Hard validation to avoid silent errors in production."""
    missing_id = 0
    bad_correct = 0
    bad_opts = 0
    missing_wiki = 0
    for q in questions:
        qid = str(q.get("id") or "").strip()
        if not qid:
            missing_id += 1

        opts = q.get("options") or []
        if not isinstance(opts, list) or len(opts) < 2:
            bad_opts += 1

        try:
            ci = int(q.get("correctIndex", -1))
        except Exception:
            ci = -1
        if ci < 0 or ci > 3:
            bad_correct += 1

        w = q.get("wiki")
        if not isinstance(w, dict):
            missing_wiki += 1

    return {
        "missing_id": missing_id,
        "bad_correctIndex": bad_correct,
        "bad_options": bad_opts,
        "missing_wiki_obj": missing_wiki,
    }


def index_questions(questions: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    idx: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for q in questions:
        cat = (q.get("category") or "").strip()
        sub = (q.get("subchapter") or "").strip()
        idx.setdefault((cat, sub), []).append(q)
    return idx


# =============================================================================
# PDF IMAGE RENDER (Bilder.pdf) with CLIP support
# =============================================================================
@st.cache_data(show_spinner=False)
def render_pdf_page_png(
    pdf_path: str,
    page_1based: int,
    zoom: float = 2.0,
    clip: Optional[List[float]] = None,
) -> Optional[bytes]:
    """
    Render a PDF page (or clipped region) to PNG bytes using PyMuPDF.
    clip = [x0, y0, x1, y1] in PDF points (PyMuPDF).
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
        if doc.page_count <= 0:
            return None

        page_index = max(0, min(int(page_1based) - 1, doc.page_count - 1))
        page = doc.load_page(page_index)

        mat = fitz.Matrix(float(zoom), float(zoom))

        clip_rect = None
        if isinstance(clip, list) and len(clip) == 4:
            clip_rect = fitz.Rect(float(clip[0]), float(clip[1]), float(clip[2]), float(clip[3]))

        pix = page.get_pixmap(matrix=mat, alpha=False, clip=clip_rect)
        return pix.tobytes("png")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_figure_map() -> Dict[str, Any]:
    """
    Supports BOTH formats:

    Old:
      { "47": 14 }

    New:
      {
        "47": { "page": 14, "clip": [x0,y0,x1,y1] }
      }
    """
    if not FIGURE_MAP_PATH.exists():
        return {}
    try:
        data = json.loads(FIGURE_MAP_PATH.read_text("utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _infer_figures_from_text(q: Dict[str, Any]) -> List[Dict[str, Any]]:
    figs = q.get("figures")
    if isinstance(figs, list) and figs:
        return [f for f in figs if isinstance(f, dict)]

    text = f"{q.get('title','')} {q.get('question','')}".strip()
    m = re.search(r"\bAbbildung\s*(\d+)\b", text, flags=re.IGNORECASE)
    if not m:
        return []
    return [{"figure": int(m.group(1))}]


def render_figures(q: Dict[str, Any], max_n: int = 3):
    figs = _infer_figures_from_text(q)
    if not figs:
        return

    fig_map = load_figure_map()
    shown = 0

    for f in figs:
        if shown >= max_n:
            break
        if not isinstance(f, dict):
            continue

        try:
            fig_no_int = int(f.get("figure"))
        except Exception:
            continue

        # page priority: explicit bilder_page > figure_map
        try:
            page_1based = int(f.get("bilder_page") or 0)
        except Exception:
            page_1based = 0

        clip = None

        if page_1based <= 0:
            entry = fig_map.get(str(fig_no_int))
            if isinstance(entry, int):
                page_1based = int(entry)
            elif isinstance(entry, dict):
                try:
                    page_1based = int(entry.get("page") or 0)
                except Exception:
                    page_1based = 0
                clip = entry.get("clip")

        if page_1based <= 0:
            continue

        png = render_pdf_page_png(str(BILDER_PDF), page_1based=page_1based, zoom=2.0, clip=clip)
        if png:
            st.image(
                png,
                caption=f"Abbildung {fig_no_int} (Bilder.pdf Seite {page_1based})",
                use_container_width=True,
            )
            shown += 1


# =============================================================================
# AUTH (Streamlit built-in OIDC: Google)
# =============================================================================
def require_login() -> None:
    # st.user exists only when Streamlit auth is configured; guard with getattr
    if not getattr(getattr(st, "user", None), "is_logged_in", False):
        st.title("B-Lizenz Lernapp")
        st.caption("Bitte mit Google anmelden.")
        st.button("Mit Google anmelden", on_click=st.login, use_container_width=True)
        st.stop()


def user_claims() -> Dict[str, str]:
    u = getattr(st, "user", None)
    return {
        "email": (getattr(u, "email", "") or "").strip(),
        "name": (getattr(u, "name", "") or "").strip(),
        "sub": (getattr(u, "sub", "") or "").strip(),
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
        r = (
            supa()
            .table("notes")
            .select("note_text")
            .eq("user_id", uid)
            .eq("question_id", qid)
            .limit(1)
            .execute()
        )
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
# AI (OpenAI) - ONLY FOR LEARNING
# =============================================================================
def ai_available() -> bool:
    if not OPENAI_API_KEY:
        return False
    try:
        from openai import OpenAI  # noqa: F401
        return True
    except Exception:
        return False


def ai_ask_question(q: Dict[str, Any], user_text: str) -> str:
    """
    Grounded assistant: includes question, options, correctIndex, wiki.
    Note: this is for "Rückfragen" AFTER answering; it can reveal correct answer.
    """
    ai_cfg = get_ai_cfg(q)
    system_hint = ai_cfg["system_hint"]
    context = ai_cfg["context"]

    opts = q.get("options") or []
    while len(opts) < 4:
        opts.append("")

    w = get_wiki(q)

    prompt = f"""
KONTEXT:
{context}

FRAGE:
{(q.get("question") or "").strip()}

OPTIONEN:
A) {opts[0]}
B) {opts[1]}
C) {opts[2]}
D) {opts[3]}

RICHTIGE OPTION (Index):
{int(q.get("correctIndex", -1))}

WIKI-KURZ:
{w.get("explanation","")}

MERKSATZ:
{w.get("merksatz","")}

USER-FRAGE:
{user_text}
""".strip()

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_hint},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"KI-Fehler: {e}"


def render_ai_chat(q: Dict[str, Any], qid: str):
    ai_cfg = get_ai_cfg(q)
    if not ai_cfg["allowed"]:
        st.caption("KI-Nachfragen sind für diese Frage deaktiviert.")
        return

    if "ai_chat" not in st.session_state:
        st.session_state.ai_chat = {}
    st.session_state.ai_chat.setdefault(qid, [])
    history = st.session_state.ai_chat[qid]

    if not ai_available():
        st.warning("OpenAI nicht verfügbar (fehlender Key oder openai-Paket nicht installiert).")
        st.caption("Setze in Streamlit secrets: [openai].api_key und optional [openai].model")
        return

    for msg in history:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        with st.chat_message(role):
            st.markdown(content)

    user_text = st.text_area(
        "Deine Rückfrage",
        key=f"ai_draft_{qid}",
        height=90,
        placeholder="Warum ist Antwort D korrekt? Bitte kurz erklären.",
    )
    c1, c2 = st.columns([1, 1])
    send = c1.button("Senden", key=f"ai_send_{qid}", type="primary")
    clear = c2.button("Chat leeren", key=f"ai_clear_{qid}")

    if clear:
        st.session_state.ai_chat[qid] = []
        st.rerun()

    if send:
        user_text = (user_text or "").strip()
        if not user_text:
            st.warning("Bitte eine Frage eingeben.")
            return
        history.append({"role": "user", "content": user_text})
        answer = ai_ask_question(q, user_text)
        history.append({"role": "assistant", "content": answer})
        st.rerun()


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
.pp-timer { border:1px solid var(--pp-border); border-radius:14px; padding:0.55rem 0.75rem; background: rgba(255,255,255,0.03); display:inline-block;}
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

    with st.sidebar.expander("Daten", expanded=False):
        st.caption("Optional: questions.json zur Laufzeit überschreiben (ohne Speicherung).")
        up = st.file_uploader("questions.json hochladen", type=["json"], key="upload_questions")
        if up is not None:
            try:
                data = json.loads(up.getvalue().decode("utf-8"))
                if not isinstance(data, list):
                    raise ValueError("JSON muss eine Liste von Fragen sein.")
                st.session_state.questions_override = data
                _reset_learning_state()
                _reset_exam_state()
                st.success(f"Geladen: {len(data)} Fragen")
                st.rerun()
            except Exception as e:
                st.error(f"Upload-Fehler: {e}")

        if st.button("Override zurücksetzen", use_container_width=True):
            st.session_state.pop("questions_override", None)
            _reset_learning_state()
            _reset_exam_state()
            st.rerun()


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


def weakest_subchapters(stats: Dict[str, Dict[str, Dict[str, int]]], min_seen: int = 6, topn: int = 8) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cat, subs in stats.items():
        for sub, s in subs.items():
            attempts = int(s.get("correct_total", 0)) + int(s.get("wrong_total", 0))
            if attempts < min_seen:
                continue
            acc = (int(s.get("correct_total", 0)) / attempts) if attempts else 0.0
            rows.append(
                {
                    "category": cat,
                    "subchapter": sub,
                    "attempts": attempts,
                    "accuracy": acc,
                    "wrong": int(s.get("wrong_total", 0)),
                }
            )
    rows.sort(key=lambda r: (r["accuracy"], -r["attempts"]))
    return rows[:topn]


# =============================================================================
# APP STATE
# =============================================================================
def _reset_learning_state():
    for k in [
        "queue",
        "idx",
        "answered",
        "last_ok",
        "last_correct_index",
        "last_selected_index",
        "learn_started",
        "learn_plan",
        "ai_chat",
        "ai_draft",
    ]:
        st.session_state.pop(k, None)


def _reset_exam_state():
    for k in [
        "exam_queue",
        "exam_idx",
        "exam_started",
        "exam_phase",          # "taking" | "review" | "result"
        "exam_start_ts",
        "exam_deadline_ts",
        "exam_answers",        # {qid: int|None}
        "exam_submitted",
        "exam_score",
        "exam_correct_count",
        "exam_passed",
        "exam_auto_submit",
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
# DASHBOARD
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
    v1, v2 = st.columns([1, 1])
    with v1:
        st.markdown(
            '<div class="pp-card2"><b>Abdeckung</b><div class="pp-muted">Wie viel vom Fragenkatalog du schon gesehen hast</div>',
            unsafe_allow_html=True,
        )
        st.progress(overall / 100.0)
        st.markdown("</div>", unsafe_allow_html=True)
    with v2:
        st.markdown(
            '<div class="pp-card2"><b>Trefferquote</b><div class="pp-muted">Richtig vs. Falsch (alle Versuche)</div>',
            unsafe_allow_html=True,
        )
        if attempts_total:
            st.bar_chart({"Richtig": c_total, "Falsch": w_total})
        else:
            st.caption("Noch keine beantworteten Fragen.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("## Fokus: Schwache Bereiche")
    weak = weakest_subchapters(stats, min_seen=6, topn=8)
    if not weak:
        st.caption("Noch zu wenig Daten. (mind. 6 Antworten pro Unterkapitel)")
    else:
        for row in weak:
            acc = int(round(row["accuracy"] * 100))
            wrong = row["wrong"]
            attempts = row["attempts"]
            cc1, cc2, cc3, cc4 = st.columns([3, 1, 1, 2])
            cc1.markdown(
                f"**{row['category']} · {row['subchapter']}**  \n<span class='pp-muted'>{attempts} Versuche · {wrong} falsch</span>",
                unsafe_allow_html=True,
            )
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
# LEARN
# =============================================================================
def page_learn(uid: str, questions: List[Dict[str, Any]], progress: Dict[str, Dict[str, Any]]):
    st.title("Lernen")

    if "learn_plan" not in st.session_state:
        st.session_state.learn_plan = {"category": "Alle", "subchapter": "Alle", "only_unseen": False, "only_wrong": False}
    if "learn_started" not in st.session_state:
        st.session_state.learn_started = False

    if not st.session_state.learn_started:
        plan = st.session_state.learn_plan

        cats = sorted(set((q.get("category") or "").strip() for q in questions if q.get("category")))
        sel_category = st.selectbox(
            "Kategorie",
            ["Alle"] + cats,
            index=(["Alle"] + cats).index(plan["category"]) if plan["category"] in (["Alle"] + cats) else 0,
            key="sel_category",
        )

        subs = sorted(set((q.get("subchapter") or "").strip() for q in questions if q.get("subchapter")))
        if sel_category != "Alle":
            subs = sorted(
                set(
                    (q.get("subchapter") or "").strip()
                    for q in questions
                    if (q.get("category") or "") == sel_category
                )
            )
        sel_subchapter = st.selectbox(
            "Unterkapitel",
            ["Alle"] + subs,
            index=(["Alle"] + subs).index(plan["subchapter"]) if plan["subchapter"] in (["Alle"] + subs) else 0,
            key="sel_subchapter",
        )

        only_unseen = st.checkbox("Nur ungelernt", value=bool(plan.get("only_unseen", False)), key="only_unseen")
        only_wrong = st.checkbox("Nur falsch beantwortete", value=bool(plan.get("only_wrong", False)), key="only_wrong")

        c1, c2 = st.columns([1, 1])
        start = c1.button("Session starten", type="primary")
        if c2.button("Zur Übersicht"):
            st.session_state.page = "dashboard"
            _reset_learning_state()
            st.rerun()

        if start:
            st.session_state.learn_plan = {
                "category": sel_category,
                "subchapter": sel_subchapter,
                "only_unseen": only_unseen,
                "only_wrong": only_wrong,
            }
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

    render_figures(q, max_n=3)

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

        w = get_wiki(q)
        with st.expander("Wiki (kurz + Merksatz + Links)", expanded=True):
            if w["explanation"]:
                st.markdown(w["explanation"])
            else:
                src = "Upload-Override" if st.session_state.get("questions_override") else "questions.json (App-Verzeichnis)"
                st.error(
                    f"Wiki-Inhalt ist leer für {qid}. Quelle: {src}. "
                    f"Prüfe: question['wiki']['explanation'] ist befüllt und die App lädt wirklich die erwartete Datei. "
                    f"(Tipp: Sidebar → Daten → questions.json hochladen.)"
                )

            if w["merksatz"]:
                st.markdown(f"**Merksatz:** {w['merksatz']}")

            links = w.get("links") or []
            if links:
                st.markdown("**Weiterlesen (offizielle Quellen):**")
                for li in links:
                    if not isinstance(li, dict):
                        continue
                    title = (li.get("title") or "Link").strip()
                    url = (li.get("url") or "").strip()
                    locator = (li.get("locator") or "").strip()
                    if url:
                        extra = f" — {locator}" if locator else ""
                        st.markdown(f"- [{title}]({url}){extra}")
            else:
                st.caption("Keine Links hinterlegt.")

            if w.get("reliability_note"):
                st.caption(w["reliability_note"])

        with st.expander("KI-Nachfrage zur Frage", expanded=False):
            render_ai_chat(q, qid)

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
# EXAM (Timer + Navigation + Review + Submit + Result with explanations)
# =============================================================================
def _fmt_mmss(seconds_left: int) -> str:
    seconds_left = max(0, int(seconds_left))
    mm = seconds_left // 60
    ss = seconds_left % 60
    hh = mm // 60
    mm2 = mm % 60
    if hh > 0:
        return f"{hh:02d}:{mm2:02d}:{ss:02d}"
    return f"{mm2:02d}:{ss:02d}"


def _exam_time_left() -> int:
    deadline = float(st.session_state.get("exam_deadline_ts") or 0)
    if deadline <= 0:
        return EXAM_DURATION_SEC
    return int(round(deadline - time.time()))


def _exam_auto_submit_if_needed():
    left = _exam_time_left()
    if left <= 0 and not bool(st.session_state.get("exam_submitted")):
        st.session_state.exam_auto_submit = True
        _exam_submit(final_reason="time")


def _exam_submit(final_reason: str = "manual"):
    qlist: List[Dict[str, Any]] = st.session_state.get("exam_queue", [])
    answers: Dict[str, Optional[int]] = st.session_state.get("exam_answers", {}) or {}

    total = len(qlist)
    correct = 0
    for q in qlist:
        qid = str(q.get("id"))
        try:
            ci = int(q.get("correctIndex", -1))
        except Exception:
            ci = -1
        sel = answers.get(qid)
        if sel is not None and int(sel) == ci:
            correct += 1

    pct = int(round((correct / total) * 100)) if total else 0
    passed = pct >= int(PASS_PCT)

    st.session_state.exam_submitted = True
    st.session_state.exam_phase = "result"
    st.session_state.exam_correct_count = correct
    st.session_state.exam_score = pct
    st.session_state.exam_passed = passed
    # do NOT show correct answers before submission (handled by result screen)

    # Optional: persist exam run
    try:
        uid = str(st.session_state.get("uid") or "")
        if uid:
            db_insert_exam_run(uid, total=total, correct=correct, passed=passed)
    except Exception:
        pass


def page_exam(uid: str, questions: List[Dict[str, Any]]):
    st.title("Prüfungssimulation (40)")

    st.session_state.uid = uid  # for submit helper

    if "exam_started" not in st.session_state:
        st.session_state.exam_started = False

    # Start screen
    if not st.session_state.exam_started:
        st.markdown(
            f"""<div class="pp-card2"><b>Regeln</b>
<div class="pp-muted">40 zufällige Fragen · Zeitlimit {int(EXAM_DURATION_SEC/60)} Minuten · Keine Lösungen während der Prüfung · Am Ende prüfen & abschicken.</div></div>""",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([1, 1])
        if c1.button("Prüfung starten", type="primary"):
            _reset_exam_state()
            st.session_state.exam_queue = build_exam_queue(questions, n=40)
            st.session_state.exam_idx = 0
            st.session_state.exam_started = True
            st.session_state.exam_phase = "taking"
            st.session_state.exam_submitted = False
            st.session_state.exam_answers = {}
            st.session_state.exam_start_ts = time.time()
            st.session_state.exam_deadline_ts = float(st.session_state.exam_start_ts) + float(EXAM_DURATION_SEC)
            st.session_state.exam_auto_submit = False
            st.rerun()
        if c2.button("Zur Übersicht"):
            st.session_state.page = "dashboard"
            _reset_exam_state()
            st.rerun()
        st.stop()

    # Global top controls
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

    # Timer line
    left = _exam_time_left()
    st.markdown(f"<div class='pp-timer'><b>Restzeit:</b> {_fmt_mmss(left)}</div>", unsafe_allow_html=True)

    # Auto-submit if time is up
    _exam_auto_submit_if_needed()

    phase = str(st.session_state.get("exam_phase") or "taking")
    qlist: List[Dict[str, Any]] = st.session_state.get("exam_queue", [])
    total = len(qlist)

    if not total:
        st.error("Keine Prüfungsfragen verfügbar.")
        st.session_state.exam_started = False
        return

    answers: Dict[str, Optional[int]] = st.session_state.get("exam_answers", {}) or {}

    # Helper metrics
    answered_count = sum(1 for v in answers.values() if v is not None)
    st.caption(f"Beantwortet: {answered_count}/{total}")

    # RESULT SCREEN
    if phase == "result" or bool(st.session_state.get("exam_submitted")):
        pct = int(st.session_state.get("exam_score") or 0)
        correct = int(st.session_state.get("exam_correct_count") or 0)
        passed = bool(st.session_state.get("exam_passed"))
        auto = bool(st.session_state.get("exam_auto_submit"))

        st.markdown(
            f"""<div class="pp-card"><div><b>Ergebnis</b></div>
<div class="pp-muted">{pct}% ({correct}/{total}) — {'BESTANDEN' if passed else 'NICHT bestanden'} (Schwelle {int(PASS_PCT)}%)</div>
<div class="pp-muted">{'Automatisch abgegeben (Zeit abgelaufen).' if auto else ''}</div>
</div>""",
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("## Auswertung: Richtige Lösungen + Erklärungen")
        labels = ["A", "B", "C", "D"]

        for idx_q, q in enumerate(qlist, start=1):
            qid = str(q.get("id"))
            question = (q.get("question") or "").strip()
            opts = q.get("options") or []
            while len(opts) < 4:
                opts.append("")
            try:
                ci = int(q.get("correctIndex", -1))
            except Exception:
                ci = -1

            sel = answers.get(qid)
            sel_txt = "-" if sel is None else f"{labels[int(sel)]}) {opts[int(sel)]}"
            corr_txt = "-" if ci < 0 or ci > 3 else f"{labels[int(ci)]}) {opts[int(ci)]}"
            ok = (sel is not None and ci == int(sel))

            title = f"{idx_q}/{total} · {qid} · {'✅' if ok else '❌'}"
            with st.expander(title, expanded=False):
                st.markdown(f"**Frage:** {question}")
                st.caption(f"{q.get('category','')} · {q.get('subchapter','')}")
                render_figures(q, max_n=2)

                st.markdown(f"**Deine Antwort:** {sel_txt}")
                st.markdown(f"**Richtig:** {corr_txt}")

                # Explanation from Learn (Wiki block)
                w = get_wiki(q)
                st.write("")
                st.markdown("**Erklärung (Wiki):**")
                if w["explanation"]:
                    st.markdown(w["explanation"])
                else:
                    st.caption("Keine Erklärung hinterlegt.")

                if w.get("merksatz"):
                    st.markdown(f"**Merksatz:** {w['merksatz']}")

                links = w.get("links") or []
                if links:
                    st.markdown("**Weiterlesen (offizielle Quellen):**")
                    for li in links:
                        if not isinstance(li, dict):
                            continue
                        title2 = (li.get("title") or "Link").strip()
                        url = (li.get("url") or "").strip()
                        locator = (li.get("locator") or "").strip()
                        if url:
                            extra = f" — {locator}" if locator else ""
                            st.markdown(f"- [{title2}]({url}){extra}")
                if w.get("reliability_note"):
                    st.caption(w["reliability_note"])

        return

    # REVIEW SCREEN (no correct answers)
    if phase == "review":
        st.markdown(
            """<div class="pp-card2"><b>Antworten prüfen</b>
<div class="pp-muted">Hier kannst du deine Antworten kontrollieren und zu Fragen springen. Lösungen werden erst nach dem Abschicken angezeigt.</div></div>""",
            unsafe_allow_html=True,
        )

        # jump to question
        ids = [str(q.get("id")) for q in qlist]
        labels = ["A", "B", "C", "D"]

        jump = st.selectbox(
            "Zu Frage springen",
            options=list(range(1, total + 1)),
            format_func=lambda n: f"{n}/{total} · {ids[n-1]} · {'✓' if answers.get(ids[n-1]) is not None else '—'}",
        )
        if st.button("Öffnen"):
            st.session_state.exam_idx = int(jump) - 1
            st.session_state.exam_phase = "taking"
            st.rerun()

        st.write("")
        st.markdown("### Übersicht")
        for n in range(1, total + 1):
            qid = ids[n-1]
            sel = answers.get(qid)
            sel_txt = "—"
            if sel is not None:
                try:
                    sel_txt = f"{labels[int(sel)]}"
                except Exception:
                    sel_txt = "—"
            st.caption(f"{n:02d}/{total} · {qid} · Antwort: {sel_txt}")

        st.write("")
        c1, c2, c3 = st.columns([1, 1, 2])
        if c1.button("Zurück zur Prüfung"):
            st.session_state.exam_phase = "taking"
            st.rerun()

        confirm = c2.checkbox("Ich möchte jetzt abschicken", value=False, key="exam_confirm_submit")
        if c3.button("Abschicken", type="primary", disabled=not confirm):
            _exam_submit(final_reason="manual")
            st.rerun()

        st.stop()

    # TAKING SCREEN
    i = int(st.session_state.get("exam_idx") or 0)
    i = max(0, min(i, total - 1))
    st.session_state.exam_idx = i

    st.progress(min(1.0, (i) / total))

    q = qlist[i]
    qid = str(q.get("id"))
    question = (q.get("question") or "").strip()
    options = q.get("options") or []
    while len(options) < 4:
        options.append("")

    st.markdown(
        f"""<div class="pp-card"><div><b>{question}</b></div>
<div class="pp-muted">Frage {i+1}/{total} · ID {qid}</div></div>""",
        unsafe_allow_html=True,
    )
    st.write("")

    render_figures(q, max_n=2)

    # Answer input (allows "no selection")
    labels = ["A", "B", "C", "D"]
    choice_labels = ["— keine Auswahl —"] + [f"{labels[idx]} ) {options[idx]}" for idx in range(4)]
    prev = answers.get(qid)
    prev_idx = 0 if prev is None else int(prev) + 1
    prev_idx = max(0, min(prev_idx, len(choice_labels) - 1))

    picked = st.radio(
        "Deine Antwort",
        options=list(range(len(choice_labels))),
        format_func=lambda k: choice_labels[int(k)],
        index=prev_idx,
        key=f"exam_pick_{qid}",
    )
    picked = int(picked)
    if picked == 0:
        answers[qid] = None
    else:
        answers[qid] = picked - 1
    st.session_state.exam_answers = answers

    st.write("")
    nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 2])
    if nav1.button("Zurück", disabled=(i <= 0)):
        st.session_state.exam_idx = max(0, i - 1)
        st.rerun()
    if nav2.button("Weiter", disabled=(i >= total - 1)):
        st.session_state.exam_idx = min(total - 1, i + 1)
        st.rerun()
    if nav3.button("Antworten prüfen"):
        st.session_state.exam_phase = "review"
        st.rerun()

    # quick jump by number
    jump_to = nav4.number_input("Springen zu Nr.", min_value=1, max_value=total, value=i + 1, step=1, key="exam_jump_to")
    if nav4.button("Springen"):
        st.session_state.exam_idx = int(jump_to) - 1
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
val = validate_questions(questions)
if any(v > 0 for v in val.values()):
    st.sidebar.markdown("## Daten-Checks")
    st.sidebar.warning(
        f"questions.json hat Probleme: "
        f"missing_id={val['missing_id']}, "
        f"bad_correctIndex={val['bad_correctIndex']}, "
        f"bad_options={val['bad_options']}, "
        f"missing_wiki_obj={val['missing_wiki_obj']}"
    )

progress = db_load_progress(uid)
st.session_state.progress = progress

if "page" not in st.session_state:
    st.session_state.page = "dashboard"

nav_sidebar(claims)

page = st.session_state.page
if page == "learn":
    page_learn(uid, questions, progress)
elif page == "exam":
    page_exam(uid, questions)
else:
    page_dashboard(uid, questions, progress)
