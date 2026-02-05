# streamlit_app_v3.py (AFTER)
# Requirements (per README):
#   streamlit>=1.32
#   supabase
#   pymupdf==1.24.9
#   Authlib>=1.3.2
#   openai>=1.40.0

import json
import random
import re
import uuid
import time
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from supabase import Client, create_client

# =============================================================================
# CONFIG / FILES
# =============================================================================
APP_DIR = Path(__file__).parent
QUESTIONS_PATH = APP_DIR / "questions.json"
BILDER_PDF = APP_DIR / "Bilder.pdf"  # PDF with figures
FIGURE_MAP_PATH = APP_DIR / "figure_map.json"  # {"47": {"page":14,"clip":[...]}}


def cfg(path: str, default: str = "") -> str:
    """Read config from Streamlit secrets only.

    Works with Streamlit's secrets object (mapping-like), not assuming plain dict.
    Path format: "section.key".
    """
    parts = path.split(".")
    cur: Any = st.secrets
    for p in parts:
        try:
            cur = cur[p]
        except Exception:
            return default
    return (str(cur) if cur is not None else "").strip()


PASS_PCT = float(cfg("PASS_PCT", "75"))

SUPABASE_URL = cfg("supabase.url")
SUPABASE_SERVICE_ROLE_KEY = cfg("supabase.service_role_key")
SUPABASE_ANON_KEY = cfg("supabase.anon_key")

OPENAI_API_KEY = cfg("openai.api_key")
OPENAI_MODEL = cfg("openai.model", "gpt-4.1-mini")

# Exam timer (default 60 min)
EXAM_DURATION_SEC = int(float(cfg("exam.duration_minutes", "60")) * 60)

# Optional dev flags
DEV_SELFTEST = cfg("dev.selftest", "0") in ("1", "true", "True", "yes", "YES")


# =============================================================================
# DEBUG (toggleable, no secrets)
# =============================================================================
def debug_enabled() -> bool:
    return bool(st.session_state.get("debug_on", False))


def dlog(event: str, **fields: Any) -> None:
    """Stores lightweight debug events in session_state. Never log secrets."""
    if not debug_enabled():
        return
    safe = {}
    for k, v in fields.items():
        if v is None:
            safe[k] = None
        else:
            s = str(v)
            # crude protection: don't dump long strings
            safe[k] = s if len(s) <= 200 else (s[:200] + "…")
    st.session_state.setdefault("_debug_events", [])
    st.session_state["_debug_events"].append({"ts": time.time(), "event": event, **safe})


def render_debug_panel() -> None:
    if not debug_enabled():
        return
    with st.sidebar.expander("Debug", expanded=False):
        st.caption("Leichte Debug-Logs (keine Secrets).")
        events = list(st.session_state.get("_debug_events", [])[-80:])
        if not events:
            st.caption("Keine Logs.")
            return
        for e in reversed(events):
            ts = time.strftime("%H:%M:%S", time.localtime(float(e.get("ts") or 0)))
            evt = e.get("event", "")
            rest = {k: v for k, v in e.items() if k not in ("ts", "event")}
            st.code(f"[{ts}] {evt} | {rest}", language="text")


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
        raise RuntimeError("Supabase secret fehlt: [supabase].service_role_key oder [supabase].anon_key")

    return create_client(SUPABASE_URL, key)


# =============================================================================
# QUESTIONS / WIKI / AI (single source of truth: questions.json)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_questions_file() -> List[Dict[str, Any]]:
    """Load questions from bundled questions.json ONLY (cached)."""
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"questions.json fehlt: {QUESTIONS_PATH}")

    data = json.loads(QUESTIONS_PATH.read_text("utf-8"))
    if not isinstance(data, list):
        raise ValueError("questions.json muss eine Liste sein.")
    return data


def load_questions() -> List[Dict[str, Any]]:
    """Runtime questions: prefer override (session_state) else file."""
    override = st.session_state.get("questions_override")
    if isinstance(override, list) and override:
        return override
    return load_questions_file()


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
            "system_hint": (a.get("system_hint") or "").strip() or "Antworte strikt faktenbasiert. Wenn unsicher, sag es.",
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
# PDF IMAGE RENDER (Bilder.pdf) with CLIP support + safe fallback crop
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
        with fitz.open(str(p)) as doc:
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
def autocrop_png(png_bytes: bytes, margin: int = 14) -> bytes:
    """
    Crops white margins from a PNG (fallback if no clip is provided).
    Keeps behavior stable: if crop fails, returns original.
    """
    try:
        from PIL import Image, ImageChops
        import io

        im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        bg = Image.new("RGB", im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, bg)
        bbox = diff.getbbox()
        if not bbox:
            return png_bytes

        x0, y0, x1, y1 = bbox
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(im.size[0], x1 + margin)
        y1 = min(im.size[1], y1 + margin)

        cropped = im.crop((x0, y0, x1, y1))
        out = io.BytesIO()
        cropped.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception:
        return png_bytes


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


@st.cache_data(show_spinner=False)
def infer_clip_from_pdf_by_figure(pdf_path: str, page_1based: int, figure_no: int) -> Optional[List[float]]:
    """Best-effort clip inference when figure_map has no clip."""
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None

    p = Path(pdf_path)
    if not p.exists():
        return None

    try:
        with fitz.open(str(p)) as doc:
            if doc.page_count <= 0:
                return None
            page_index = max(0, min(int(page_1based) - 1, doc.page_count - 1))
            page = doc.load_page(page_index)

            label = f"Abbildung {int(figure_no)}"
            hits = page.search_for(label)
            if not hits:
                return None

            cur_rect = sorted(hits, key=lambda r: (r.y0, r.x0))[-1]

            next_hits = page.search_for("Abbildung")
            below = [r for r in next_hits if r.y0 > cur_rect.y0 + 1]
            next_rect = sorted(below, key=lambda r: (r.y0, r.x0))[0] if below else None

            page_rect = page.rect
            margin_x = 18.0
            pad_y = 12.0

            x0 = float(page_rect.x0 + margin_x)
            x1 = float(page_rect.x1 - margin_x)

            y0 = float(cur_rect.y1 + pad_y)
            y1 = float((next_rect.y0 - pad_y) if next_rect else (page_rect.y1 - pad_y))

            if y1 <= y0 + 20:
                return None

            y0 = max(float(page_rect.y0), y0)
            y1 = min(float(page_rect.y1), y1)

            return [x0, y0, x1, y1]
    except Exception:
        return None


def _infer_figures_from_text(q: Dict[str, Any]) -> List[Dict[str, Any]]:
    figs = q.get("figures")
    if isinstance(figs, list) and figs:
        return [f for f in figs if isinstance(f, dict)]

    text = f"{q.get('title','')} {q.get('question','')}".strip()
    m = re.search(r"\bAbbildung\s*(\d+)\b", text, flags=re.IGNORECASE)
    if not m:
        return []
    return [{"figure": int(m.group(1))}]


def render_figures(q: Dict[str, Any], max_n: int = 3) -> None:
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

        try:
            page_1based = int(f.get("bilder_page") or 0)
        except Exception:
            page_1based = 0

        clip = f.get("clip") if isinstance(f.get("clip"), list) else None
        entry = fig_map.get(str(fig_no_int))

        if page_1based <= 0:
            if isinstance(entry, int):
                page_1based = int(entry)
            elif isinstance(entry, dict):
                try:
                    page_1based = int(entry.get("page") or 0)
                except Exception:
                    page_1based = 0

        if clip is None and isinstance(entry, dict):
            c = entry.get("clip")
            if isinstance(c, list) and len(c) == 4:
                clip = c

        if page_1based <= 0:
            continue

        png = render_pdf_page_png(str(BILDER_PDF), page_1based=page_1based, zoom=2.0, clip=clip)
        if not png:
            continue

        inferred = None
        if not clip:
            inferred = infer_clip_from_pdf_by_figure(str(BILDER_PDF), page_1based, fig_no_int)
            if inferred:
                png2 = render_pdf_page_png(str(BILDER_PDF), page_1based=page_1based, zoom=2.0, clip=inferred)
                if png2:
                    png = png2
                    clip = inferred

        if not clip:
            png = autocrop_png(png, margin=14)

        cap = f"Abbildung {fig_no_int} (Bilder.pdf Seite {page_1based})"
        cap += " · Ausschnitt" if clip else " · Auto-Crop"
        st.image(png, caption=cap, width="stretch")
        shown += 1


# =============================================================================
# AUTH (Streamlit built-in OIDC: Google)
# =============================================================================
def require_login() -> None:
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
    dlog("db_load_progress", uid=uid)
    r = supa().table("progress").select("question_id,seen,correct,wrong").eq("user_id", uid).execute()
    return {str(x["question_id"]): x for x in (r.data or [])}


def db_upsert_progress(uid: str, qid: str, ok: bool) -> Dict[str, int]:
    """
    Returns the new counters for this (uid,qid) so caller can update local session_state
    without reloading whole progress table.
    """
    s = supa()
    dlog("db_upsert_progress.begin", uid=uid, qid=qid, ok=ok)

    r = (
        s.table("progress")
        .select("seen,correct,wrong")
        .eq("user_id", uid)
        .eq("question_id", qid)
        .limit(1)
        .execute()
    )

    if r.data:
        row = r.data[0]
        new_seen = int(row.get("seen", 0)) + 1
        new_correct = int(row.get("correct", 0)) + (1 if ok else 0)
        new_wrong = int(row.get("wrong", 0)) + (0 if ok else 1)

        s.table("progress").update(
            {"seen": new_seen, "correct": new_correct, "wrong": new_wrong}
        ).eq("user_id", uid).eq("question_id", qid).execute()

        dlog("db_upsert_progress.update", seen=new_seen, correct=new_correct, wrong=new_wrong)
        return {"seen": new_seen, "correct": new_correct, "wrong": new_wrong}

    else:
        new_seen = 1
        new_correct = 1 if ok else 0
        new_wrong = 0 if ok else 1
        s.table("progress").insert(
            {"user_id": uid, "question_id": qid, "seen": new_seen, "correct": new_correct, "wrong": new_wrong}
        ).execute()
        dlog("db_upsert_progress.insert", seen=new_seen, correct=new_correct, wrong=new_wrong)
        return {"seen": new_seen, "correct": new_correct, "wrong": new_wrong}


def apply_progress_delta_local(uid: str, qid: str, counters: Dict[str, int]) -> None:
    """Update st.session_state.progress in-place to reflect the DB write (Zero Feature Loss)."""
    p = st.session_state.get("progress")
    if not isinstance(p, dict):
        p = {}
    p[str(qid)] = {"user_id": uid, "question_id": str(qid), **counters}
    st.session_state.progress = p


def db_get_note(uid: str, qid: str) -> str:
    try:
        dlog("db_get_note", uid=uid, qid=qid)
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
    except Exception as e:
        dlog("db_get_note.err", err=str(e))
        return ""
    return ""


def db_upsert_note(uid: str, qid: str, note_text: str) -> bool:
    try:
        dlog("db_upsert_note.begin", uid=uid, qid=qid)
        s = supa()
        note_text = (note_text or "").strip()
        r = s.table("notes").select("user_id").eq("user_id", uid).eq("question_id", qid).limit(1).execute()
        if r.data:
            s.table("notes").update({"note_text": note_text}).eq("user_id", uid).eq("question_id", qid).execute()
        else:
            s.table("notes").insert({"user_id": uid, "question_id": qid, "note_text": note_text}).execute()
        dlog("db_upsert_note.ok")
        return True
    except Exception as e:
        dlog("db_upsert_note.err", err=str(e))
        return False


def db_insert_exam_run(uid: str, total: int, correct: int, passed: bool) -> Tuple[bool, str]:
    """Insert exam run. Returns (ok, error_message)."""
    try:
        dlog("db_insert_exam_run", uid=uid, total=total, correct=correct, passed=passed)
        supa().table("exam_runs").insert(
            {"user_id": uid, "total": int(total), "correct": int(correct), "passed": bool(passed)}
        ).execute()
        return True, ""
    except Exception as e:
        dlog("db_insert_exam_run.err", err=str(e))
        return False, str(e)


def db_list_exam_runs(uid: str, limit: int = 50) -> List[Dict[str, Any]]:
    try:
        dlog("db_list_exam_runs", uid=uid, limit=limit)
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
    except Exception as e:
        dlog("db_list_exam_runs.err", err=str(e))
        return []


def db_reset_user_data(uid: str) -> Tuple[bool, str]:
    """Delete all user-owned learning data (progress, notes, exam_runs).
    Keeps app_users entry intact (login mapping).
    Returns (ok, error_message).
    """
    try:
        dlog("db_reset_user_data.begin", uid=uid)
        s = supa()
        s.table("notes").delete().eq("user_id", uid).execute()
        s.table("progress").delete().eq("user_id", uid).execute()
        s.table("exam_runs").delete().eq("user_id", uid).execute()
        dlog("db_reset_user_data.ok")
        return True, ""
    except Exception as e:
        dlog("db_reset_user_data.err", err=str(e))
        return False, str(e)


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

        dlog("ai_call", model=OPENAI_MODEL)
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
        dlog("ai_err", err=str(e))
        return f"KI-Fehler: {e}"


def render_ai_chat(q: Dict[str, Any], qid: str) -> None:
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
def inject_css() -> None:
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


def _reset_learning_state() -> None:
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
    ]:
        st.session_state.pop(k, None)


def _reset_exam_state() -> None:
    for k in [
        "exam_queue",
        "exam_idx",
        "exam_started",
        "exam_done",
        "exam_submitted",
        "exam_answers",
        "exam_result",
        "exam_deadline_ts",
        "exam_radio_seed",
        "exam_save_ok",
        "exam_save_err",
    ]:
        st.session_state.pop(k, None)
    for k in list(st.session_state.keys()):
        if str(k).startswith("exam_radio_"):
            st.session_state.pop(k, None)


def nav_sidebar(claims: Dict[str, str]) -> None:
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

    st.sidebar.markdown("## Tools")
    st.sidebar.checkbox("Debug logs", key="debug_on", value=bool(st.session_state.get("debug_on", False)))

@st.cache_data(show_spinner=False)
def load_questions() -> List[Dict[str, Any]]:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"questions.json fehlt: {QUESTIONS_PATH}")

    data = json.loads(QUESTIONS_PATH.read_text("utf-8"))
    if not isinstance(data, list):
        raise ValueError("questions.json muss eine Liste sein.")
    return data

    # Wartung: Fortschritt zurücksetzen (nur userbezogene Daten)
    st.sidebar.markdown("## Wartung")
    with st.sidebar.expander("Fortschritt zurücksetzen", expanded=False):
        st.caption("Löscht: Lernfortschritt, Notizen, Prüfungs-Historie (nur dein User). Fragen/Wiki bleiben unverändert.")
        confirm = st.checkbox("Ich verstehe, dass das nicht rückgängig gemacht werden kann.", key="reset_confirm")
        token = st.text_input("Tippe RESET zur Bestätigung", value="", key="reset_token")
        do_reset = st.button(
            "Jetzt zurücksetzen",
            type="primary",
            use_container_width=True,
            disabled=not (confirm and (token or "").strip().upper() == "RESET"),
            key="reset_do",
        )
        if do_reset:
            uid = str(st.session_state.get("uid") or "")
            ok, err = db_reset_user_data(uid)
            if ok:
                st.session_state.progress = {}
                _reset_learning_state()
                _reset_exam_state()
                st.session_state.page = "dashboard"
                st.success("Zurückgesetzt.")
                st.rerun()
            else:
                st.error("Reset fehlgeschlagen (Supabase/RLS).")
                if err:
                    st.caption(f"DB-Fehler: {err}")

    render_debug_panel()


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
                {"category": cat, "subchapter": sub, "attempts": attempts, "accuracy": acc, "wrong": int(s.get("wrong_total", 0))}
            )
    rows.sort(key=lambda r: (r["accuracy"], -r["attempts"]))
    return rows[:topn]


def top_wrong_questions(questions: List[Dict[str, Any]], progress: Dict[str, Dict[str, Any]], topn: int = 10) -> List[Dict[str, Any]]:
    by_id = {str(q.get("id")): q for q in questions}
    rows: List[Dict[str, Any]] = []
    for qid, row in progress.items():
        w = int(row.get("wrong", 0) or 0)
        if w <= 0:
            continue
        q = by_id.get(str(qid))
        if not q:
            continue
        rows.append(
            {
                "qid": str(qid),
                "wrong": w,
                "seen": int(row.get("seen", 0) or 0),
                "category": (q.get("category") or "").strip(),
                "subchapter": (q.get("subchapter") or "").strip(),
                "question": (q.get("question") or "").strip(),
            }
        )
    rows.sort(key=lambda r: (-r["wrong"], -r["seen"]))
    return rows[:topn]


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
            q for q in qset
            if (str(q.get("id")) not in progress) or int(progress[str(q.get("id"))].get("seen", 0)) == 0
        ]
    if only_wrong:
        qset = [
            q for q in qset
            if (str(q.get("id")) in progress) and int(progress[str(q.get("id"))].get("wrong", 0)) > 0
        ]

    random.shuffle(qset)
    return qset


def build_exam_queue(questions: List[Dict[str, Any]], n: int = 40) -> List[Dict[str, Any]]:
    base = list(questions)
    random.shuffle(base)
    return base[:n]


def _alloc_counts(total: int, keys: List[str]) -> Dict[str, int]:
    if total <= 0 or not keys:
        return {k: 0 for k in keys}
    base = total // len(keys)
    rem = total % len(keys)
    out = {k: base for k in keys}
    for k in keys[:rem]:
        out[k] += 1
    return out


def build_exam_queue_balanced(questions: List[Dict[str, Any]], n: int = 40, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    by_cat: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for q in questions:
        cat = (q.get("category") or "").strip() or "Unbekannt"
        sub = (q.get("subchapter") or "").strip() or "Allgemein"
        by_cat.setdefault(cat, {}).setdefault(sub, []).append(q)

    cats = sorted(by_cat.keys())
    if not cats:
        return []

    cat_target = _alloc_counts(n, cats)
    selected: List[Dict[str, Any]] = []

    for cat in cats:
        subs = sorted(by_cat[cat].keys())
        if not subs or cat_target[cat] <= 0:
            continue

        avail = {s: len(by_cat[cat][s]) for s in subs}
        total_avail = sum(avail.values())

        raw = {}
        for s in subs:
            raw[s] = (cat_target[cat] * avail[s] / total_avail) if total_avail else 0.0

        sub_target = {s: int(math.floor(raw[s])) for s in subs}
        rem = cat_target[cat] - sum(sub_target.values())
        order = sorted(subs, key=lambda s: (raw[s] - math.floor(raw[s]), avail[s]), reverse=True)
        for s in order:
            if rem <= 0:
                break
            sub_target[s] += 1
            rem -= 1

        if cat_target[cat] > 1 and len(subs) > 1:
            zeros = [s for s in subs if sub_target[s] == 0 and avail[s] > 0]
            for s in zeros:
                donor = max(subs, key=lambda d: sub_target[d])
                if sub_target[donor] <= 1:
                    break
                sub_target[donor] -= 1
                sub_target[s] += 1

        for s in subs:
            bucket = list(by_cat[cat][s])
            rng.shuffle(bucket)
            take = min(sub_target[s], len(bucket))
            selected.extend(bucket[:take])

    if len(selected) < n:
        selected_ids = {str(q.get("id")) for q in selected}
        remaining = [q for q in questions if str(q.get("id")) not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: max(0, n - len(selected))])

    if len(selected) > n:
        rng.shuffle(selected)
        selected = selected[:n]

    rng.shuffle(selected)
    return selected


# =============================================================================
# DASHBOARD
# =============================================================================
def page_dashboard(uid: str, questions: List[Dict[str, Any]], progress: Dict[str, Dict[str, Any]]) -> None:
    st.title("Übersicht")

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

    last7 = runs[:7]
    if last7:
        pcts7 = []
        for r in last7:
            t = int(r.get("total") or 0)
            c = int(r.get("correct") or 0)
            pcts7.append(int(round((c / t) * 100)) if t else 0)
        avg7 = int(round(sum(pcts7) / len(pcts7)))
        trend7 = (pcts7[0] - pcts7[-1]) if len(pcts7) >= 2 else 0
    else:
        avg7 = 0
        trend7 = 0

    st.markdown(
        f"""
<div class="pp-grid">
  <div class="pp-kpi"><b>Abdeckung</b><br>{overall}%<div class="pp-muted">Fragen mindestens 1× gesehen</div></div>
  <div class="pp-kpi"><b>Trefferquote</b><br>{accuracy_total}%<div class="pp-muted">{c_total} richtig · {w_total} falsch</div></div>
  <div class="pp-kpi"><b>Prüfungen</b><br>{exam_attempts} Versuche<div class="pp-muted">Passrate {pass_rate}% · Ø7 {avg7}% · Trend {('+' if trend7>0 else '')}{trend7}</div></div>
  <div class="pp-kpi"><b>Beste Prüfung</b><br>{best}%<div class="pp-muted">Letzte: {('-' if last_pct is None else str(last_pct)+'%')}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    cA, cB, cC = st.columns([1, 1, 1])
    if cA.button("Weiterlernen", type="primary"):
        _reset_learning_state()
        st.session_state.page = "learn"
        st.session_state.learn_plan = {"category": "Alle", "subchapter": "Alle", "only_unseen": False, "only_wrong": False}
        st.rerun()
    if cB.button("Falsche wiederholen"):
        _reset_learning_state()
        st.session_state.page = "learn"
        st.session_state.learn_plan = {"category": "Alle", "subchapter": "Alle", "only_unseen": False, "only_wrong": True}
        st.rerun()
    if cC.button("Prüfung starten (40)"):
        st.session_state.page = "exam"
        _reset_exam_state()
        st.rerun()

    st.write("")
    weak = weakest_subchapters(stats, min_seen=6, topn=8)
    target = weak[0] if weak else None
    st.markdown("## Nächster sinnvoller Schritt")
    if target:
        acc = int(round(target["accuracy"] * 100))
        st.markdown(
            f"""<div class="pp-card2"><b>Empfehlung</b>
<div class="pp-muted">Übe als nächstes: <b>{target['category']} · {target['subchapter']}</b>
(Acc {acc}% bei {target['attempts']} Versuchen)</div></div>""",
            unsafe_allow_html=True,
        )
        if st.button("Jetzt starten (nur falsch)", use_container_width=True):
            _reset_learning_state()
            st.session_state.page = "learn"
            st.session_state.learn_plan = {
                "category": target["category"],
                "subchapter": target["subchapter"],
                "only_unseen": False,
                "only_wrong": True,
            }
            st.rerun()
    else:
        st.caption("Noch nicht genug Daten für eine Empfehlung (mind. 6 Antworten pro Unterkapitel).")

    st.write("")
    st.markdown("## Deine häufigsten Fehler")
    wrong_rows = top_wrong_questions(questions, progress, topn=10)
    if not wrong_rows:
        st.caption("Noch keine falsch beantworteten Fragen gespeichert.")
    else:
        for r in wrong_rows:
            q_short = r["question"][:140] + ("…" if len(r["question"]) > 140 else "")
            a1, a2 = st.columns([4, 1])
            a1.caption(f"{r['wrong']}× falsch · {r['category']} · {r['subchapter']}  |  {q_short}")
            if a2.button("Üben", key=f"wrong_{r['qid']}"):
                _reset_learning_state()
                st.session_state.page = "learn"
                st.session_state.learn_plan = {
                    "category": r["category"] or "Alle",
                    "subchapter": r["subchapter"] or "Alle",
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
                st.session_state.learn_plan = {"category": cat, "subchapter": sub, "only_unseen": False, "only_wrong": False}
                st.rerun()

    st.write("")
    st.markdown("## Letzte Prüfungen")
    if runs:
        for r in runs[:10]:
            total = int(r.get("total") or 0)
            corr = int(r.get("correct") or 0)
            pct = int(round((corr / total) * 100)) if total else 0
            ok = "BESTANDEN" if bool(r.get("passed")) else "NICHT bestanden"
            st.caption(f"{pct}% ({corr}/{total}) — {ok}")
    else:
        st.caption("Keine gespeicherten Prüfungen gefunden. Wenn du gerade Prüfungen gemacht hast, werden sie wahrscheinlich nicht gespeichert (Supabase RLS/Key).")


# =============================================================================
# LEARN
# =============================================================================
def page_learn(uid: str, questions: List[Dict[str, Any]], progress: Dict[str, Dict[str, Any]]) -> None:
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
            subs = sorted(set((q.get("subchapter") or "").strip() for q in questions if (q.get("category") or "") == sel_category))
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
        if c2.button("Zur Übersicht", key="learn_to_dashboard"):
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
    while len(options) < 4:
        options.append("")
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
        for i_opt in range(4):
            opt = options[i_opt]
            if st.button(f"{labels[i_opt]}) {opt}", key=f"learn_{qid}_{i_opt}"):
                ok = (i_opt == correct_index)

                counters = db_upsert_progress(uid, qid, ok)
                apply_progress_delta_local(uid, qid, counters)

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
                st.error(f"Wiki-Inhalt ist leer für {qid}. Prüfe questions.json → wiki.explanation.")

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
# EXAM
# =============================================================================
def _fmt_hhmmss(seconds_left: int) -> str:
    seconds_left = max(0, int(seconds_left))
    hh = seconds_left // 3600
    mm = (seconds_left % 3600) // 60
    ss = seconds_left % 60
    if hh > 0:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"


def _exam_compute_result(qlist: List[Dict[str, Any]], answers: Dict[str, Optional[int]]) -> Dict[str, Any]:
    total = len(qlist)
    correct = 0
    details = []
    for q in qlist:
        qid = str(q.get("id"))
        try:
            ci = int(q.get("correctIndex", -1))
        except Exception:
            ci = -1
        sel = answers.get(qid, None)
        is_ok = (sel is not None and int(sel) == ci)
        if is_ok:
            correct += 1
        details.append({"qid": qid, "selected": sel, "correct": ci, "ok": is_ok, "q": q})

    pct = int(round((correct / total) * 100)) if total else 0
    passed = pct >= int(PASS_PCT)
    return {"correct": correct, "total": total, "pct": pct, "passed": passed, "details": details}


def _exam_submit(uid: str, reason: str = "manual") -> None:
    st.session_state.exam_submitted = True
    st.session_state.exam_done = True

    qlist: List[Dict[str, Any]] = st.session_state.get("exam_queue", [])
    answers: Dict[str, Optional[int]] = st.session_state.get("exam_answers", {}) or {}
    result = _exam_compute_result(qlist, answers)
    st.session_state.exam_result = result

    ok_db, err_db = db_insert_exam_run(
        uid,
        total=int(result["total"]),
        correct=int(result["correct"]),
        passed=bool(result["passed"]),
    )
    st.session_state.exam_save_ok = ok_db
    st.session_state.exam_save_err = err_db
    dlog("exam_submit", reason=reason, pct=result.get("pct"))


def page_exam(uid: str, questions: List[Dict[str, Any]]) -> None:
    st.title("Prüfungssimulation (40)")

    if "exam_started" not in st.session_state:
        st.session_state.exam_started = False

    if not st.session_state.exam_started:
        st.markdown(
            f"""<div class="pp-card2"><b>Regeln</b>
<div class="pp-muted">40 zufällige Fragen · {int(EXAM_DURATION_SEC/60)} Minuten Gesamtzeit · Antworten frei ändern · Abgabe am Ende.</div></div>""",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([1, 1])
        if c1.button("Prüfung starten", type="primary"):
            _reset_exam_state()
            st.session_state.exam_queue = build_exam_queue_balanced(questions, n=40, seed=int(time.time()))
            st.session_state.exam_idx = 0
            st.session_state.exam_started = True
            st.session_state.exam_done = False
            st.session_state.exam_submitted = False
            st.session_state.exam_answers = {}
            st.session_state.exam_deadline_ts = float(time.time()) + float(EXAM_DURATION_SEC)
            st.rerun()
        if c2.button("Zur Übersicht", key="exam_start_to_dashboard"):
            st.session_state.page = "dashboard"
            _reset_exam_state()
            st.rerun()
        st.stop()

    if not st.session_state.get("exam_done", False):
        if hasattr(st, "autorefresh"):
            st.autorefresh(interval=1000, limit=None, key="exam_timer_refresh")

    deadline = float(st.session_state.get("exam_deadline_ts") or (time.time() + EXAM_DURATION_SEC))
    remaining = int(round(deadline - time.time()))

    if remaining <= 0 and not st.session_state.get("exam_done", False):
        _exam_submit(uid, reason="time")
        st.rerun()

    top1, top2, top3, top4 = st.columns([1, 1, 1, 1])
    with top1:
        st.markdown(f"<div class='pp-pill'>⏱️ Restzeit {_fmt_hhmmss(remaining)}</div>", unsafe_allow_html=True)
    if top2.button("Prüfung abbrechen", key="exam_abort"):
        st.session_state.exam_started = False
        _reset_exam_state()
        st.rerun()
    if top3.button("Neu starten", key="exam_restart"):
        st.session_state.exam_started = False
        _reset_exam_state()
        st.rerun()
    if top4.button("Zur Übersicht", key="exam_top_to_dashboard"):
        st.session_state.page = "dashboard"
        st.session_state.exam_started = False
        _reset_exam_state()
        st.rerun()

    qlist: List[Dict[str, Any]] = st.session_state.get("exam_queue", [])
    total = len(qlist)
    i = int(st.session_state.get("exam_idx", 0))
    i = max(0, min(i, max(0, total - 1)))
    st.session_state.exam_idx = i

    if st.session_state.get("exam_done", False):
        result = st.session_state.get("exam_result")
        if not isinstance(result, dict):
            answers: Dict[str, Optional[int]] = st.session_state.get("exam_answers", {}) or {}
            result = _exam_compute_result(qlist, answers)
            st.session_state.exam_result = result

        correct = int(result["correct"])
        pct = int(result["pct"])
        passed = bool(result["passed"])

        st.markdown(
            f"""<div class="pp-card"><div><b>Ergebnis</b></div>
<div class="pp-muted">{pct}% ({correct}/{total}) — {'BESTANDEN' if passed else 'NICHT bestanden'} (Schwelle {int(PASS_PCT)}%)</div></div>""",
            unsafe_allow_html=True,
        )

        if st.session_state.get("exam_save_ok") is False:
            st.warning("Prüfungsergebnis konnte nicht gespeichert werden (Supabase/RLS).")
            err = (st.session_state.get("exam_save_err") or "").strip()
            if err:
                st.caption(f"DB-Fehler: {err}")
        elif st.session_state.get("exam_save_ok") is True:
            st.caption("Prüfungsergebnis gespeichert.")

        c1, c2 = st.columns([1, 1])
        if c1.button("Neue Prüfung starten", type="primary"):
            st.session_state.exam_started = False
            _reset_exam_state()
            st.rerun()
        if c2.button("Zur Übersicht"):
            st.session_state.page = "dashboard"
            st.session_state.exam_started = False
            _reset_exam_state()
            st.rerun()

        st.write("")
        st.markdown("## Lösungen & Erklärungen")
        st.caption("Aufklappen, um richtige Lösung + Wiki-Erklärung zu sehen.")

        labels = ["A", "B", "C", "D"]
        for d in result["details"]:
            q = d["q"]
            qid = d["qid"]
            sel = d["selected"]
            ci = int(d["correct"])
            opts = q.get("options") or []
            while len(opts) < 4:
                opts.append("")

            title = f"{qid} · {'✅' if d['ok'] else '❌'}"
            with st.expander(title, expanded=False):
                st.markdown(f"**Frage:** {(q.get('question') or '').strip()}")
                render_figures(q, max_n=2)

                your = "-" if sel is None else f"{labels[int(sel)]}) {opts[int(sel)]}"
                corr = "-" if ci < 0 else f"{labels[ci]}) {opts[ci]}"
                st.markdown(f"**Deine Antwort:** {your}")
                st.markdown(f"**Richtig:** {corr}")

                w = get_wiki(q)
                if w.get("explanation"):
                    st.markdown("---")
                    st.markdown(w["explanation"])
                if w.get("merksatz"):
                    st.markdown(f"**Merksatz:** {w['merksatz']}")
                links = w.get("links") or []
                if links:
                    st.markdown("**Weiterlesen:**")
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

    if total:
        st.progress(min(1.0, i / total))

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

    answers: Dict[str, Optional[int]] = st.session_state.get("exam_answers") or {}
    current = answers.get(qid, None)

    labels = ["A", "B", "C", "D"]
    radio_vals = [-1, 0, 1, 2, 3]

    def _fmt_choice(v: int) -> str:
        if v == -1:
            return "— keine Auswahl —"
        return f"{labels[v]}) {options[v]}"

    default_val = int(current) if current is not None else -1
    sel_val = st.radio(
        "Antwort wählen",
        radio_vals,
        index=radio_vals.index(default_val),
        key=f"exam_radio_{qid}",
        format_func=_fmt_choice,
    )

    st.session_state.exam_answers[qid] = (None if int(sel_val) == -1 else int(sel_val))

    cA, cB, cC = st.columns([1, 1, 1])
    if cA.button("◀ Zurück", use_container_width=True, disabled=(i == 0)):
        st.session_state.exam_idx = max(0, i - 1)
        st.rerun()
    if cB.button("Weiter ▶", use_container_width=True, disabled=(i >= total - 1)):
        st.session_state.exam_idx = min(total - 1, i + 1)
        st.rerun()
    if cC.button("Antwort löschen", use_container_width=True):
        st.session_state.exam_answers[qid] = None
        st.session_state[f"exam_radio_{qid}"] = -1
        st.rerun()

    st.write("")
    answered_cnt = sum(1 for v in (st.session_state.exam_answers or {}).values() if v is not None)
    st.caption(f"Beantwortet: {answered_cnt}/{total}")

    if st.button("Abschicken & auswerten", type="primary", use_container_width=True):
        _exam_submit(uid, reason="manual")
        st.rerun()


# =============================================================================
# SELFTEST (optional)
# =============================================================================
def run_selftest(questions: List[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []

    v = validate_questions(questions)
    if any(vv > 0 for vv in v.values()):
        issues.append(f"questions.json Validation: {v}")

    fmap = load_figure_map()
    if fmap:
        k = sorted(fmap.keys(), key=lambda x: int(re.sub(r"\\D", "", x) or 0))[0]
        entry = fmap.get(k)
        page = None
        clip = None
        if isinstance(entry, int):
            page = int(entry)
        elif isinstance(entry, dict):
            page = int(entry.get("page") or 0) if str(entry.get("page") or "").strip() else 0
            c = entry.get("clip")
            if isinstance(c, list) and len(c) == 4:
                clip = c
        if page and BILDER_PDF.exists():
            png = render_pdf_page_png(str(BILDER_PDF), page_1based=page, zoom=1.5, clip=clip)
            if not png:
                issues.append(f"Figure render failed for figure {k} page={page} clip={clip}")
        elif page and not BILDER_PDF.exists():
            issues.append("Bilder.pdf fehlt im App-Verzeichnis.")
    else:
        issues.append("figure_map.json fehlt oder leer (ok, wenn keine Abbildungen genutzt werden).")

    if SUPABASE_URL and (SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY):
        try:
            _ = supa()
        except Exception as e:
            issues.append(f"Supabase init failed: {e}")

    return issues


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
st.session_state.uid = uid

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

try:
    qp = getattr(st, "query_params", {})
    wants_selftest = DEV_SELFTEST or (isinstance(qp, dict) and str(qp.get("selftest", "")).strip() == "1")
except Exception:
    wants_selftest = DEV_SELFTEST

if wants_selftest:
    st.sidebar.markdown("## Selftest")
    issues = run_selftest(questions)
    if issues:
        for it in issues:
            st.sidebar.error(it)
    else:
        st.sidebar.success("OK")

progress = db_load_progress(uid)
st.session_state.progress = progress

if "page" not in st.session_state:
    st.session_state.page = "dashboard"

nav_sidebar(claims)

page = st.session_state.page
if page == "learn":
    page_learn(uid, questions, st.session_state.progress)
elif page == "exam":
    page_exam(uid, questions)
else:
    page_dashboard(uid, questions, st.session_state.progress)
