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
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from supabase import Client, create_client

# FSRS (optional - if not installed, FSRS mode is disabled gracefully)
try:
    from fsrs import Scheduler as FsrsScheduler, Card as FsrsCard, Rating as FsrsRating  # type: ignore
    _FSRS_AVAILABLE = True
except Exception:
    _FSRS_AVAILABLE = False

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



def db_get_teacher_state(uid: str) -> Dict[str, Any]:
    """Load persistent teacher-path state for a user. Returns dict (may be empty)."""
    try:
        resp = supa().table("learn_state").select("teacher_state").eq("user_id", uid).limit(1).execute()
        rows = getattr(resp, "data", None) or []
        if rows:
            return rows[0].get("teacher_state") or {}
    except Exception:
        pass
    return {}


def db_upsert_teacher_state(uid: str, teacher_state: Dict[str, Any]) -> None:
    """Persist teacher-path state for a user. Best-effort; app works without it."""
    try:
        supa().table("learn_state").upsert(
            {"user_id": uid, "teacher_state": teacher_state},
            on_conflict="user_id",
        ).execute()
    except Exception:
        # ignore hard failures; UI still works without persistence
        return


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



# =============================================================================
# NEW: Teacher-path persistence (Supabase) - cursor + per-question correctness
# Tables expected (per user):
# - public.user_question_progress (PK: user_id, question_id)
# - public.user_teacherpath_cursor (PK: user_id, chapter)
# =============================================================================

def db_upsert_user_question_progress(uid: str, q: Dict[str, Any], ok: bool) -> None:
    """Persist per-question progress (seen/correct/wrong + is_correct_once). Best-effort."""
    qid = str(q.get("id"))
    chapter = str(_learn_meta(q)[0]) if q else None
    subchapter = str(q.get("subchapter") or "") if q else None

    try:
        # load existing counters (to increment reliably without RPC)
        resp = supa().table("user_question_progress").select(
            "seen_count,correct_count,wrong_count,is_correct_once"
        ).eq("user_id", uid).eq("question_id", qid).limit(1).execute()
        rows = getattr(resp, "data", None) or []
        if rows:
            row = rows[0]
            seen = int(row.get("seen_count") or 0) + 1
            correct = int(row.get("correct_count") or 0) + (1 if ok else 0)
            wrong = int(row.get("wrong_count") or 0) + (0 if ok else 1)
            once = bool(row.get("is_correct_once") or False) or bool(ok)
        else:
            seen = 1
            correct = 1 if ok else 0
            wrong = 0 if ok else 1
            once = bool(ok)

        supa().table("user_question_progress").upsert(
            {
                "user_id": uid,
                "question_id": qid,
                "chapter": chapter,
                "subchapter": subchapter,
                "seen_count": seen,
                "correct_count": correct,
                "wrong_count": wrong,
                "is_correct_once": once,
                "last_answer_at": datetime.utcnow().isoformat(),
            },
            on_conflict="user_id,question_id",
        ).execute()
    except Exception:
        return


def db_get_not_correct_once_question_ids(uid: str, question_ids: List[str]) -> List[str]:
    """Return question_ids where user has not yet answered correctly at least once."""
    if not question_ids:
        return []
    try:
        resp = supa().table("user_question_progress").select("question_id,is_correct_once") \
            .eq("user_id", uid).in_("question_id", list(map(str, question_ids))).execute()
        rows = getattr(resp, "data", None) or []
        correct_once = {str(r.get("question_id")) for r in rows if bool(r.get("is_correct_once"))}
        return [str(qid) for qid in question_ids if str(qid) not in correct_once]
    except Exception:
        # if query fails, be conservative: do not block unlock
        return []


def db_upsert_teacher_cursor(uid: str, chapter: str, last_qid: str, last_idx: int) -> None:
    """Persist the last position in a teacher-path chapter (refresh-safe). Best-effort."""
    try:
        supa().table("user_teacherpath_cursor").upsert(
            {
                "user_id": uid,
                "chapter": str(chapter),
                "last_question_id": str(last_qid),
                "last_question_idx": int(last_idx),
                "updated_at": datetime.utcnow().isoformat(),
            },
            on_conflict="user_id,chapter",
        ).execute()
    except Exception:
        return


def db_get_latest_teacher_cursor(uid: str) -> Optional[Dict[str, Any]]:
    """Return the most recently updated teacher-path cursor row for a user."""
    try:
        resp = supa().table("user_teacherpath_cursor").select(
            "chapter,last_question_id,last_question_idx,updated_at"
        ).eq("user_id", uid).order("updated_at", desc=True).limit(1).execute()
        rows = getattr(resp, "data", None) or []
        return rows[0] if rows else None
    except Exception:
        return None


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
        # FSRS-related (best-effort, tables may not exist on older DBs)
        try:
            s.table("card_state").delete().eq("user_id", uid).execute()
        except Exception:
            pass
        try:
            s.table("review_logs").delete().eq("user_id", uid).execute()
        except Exception:
            pass
        # Per-question progress + teacher cursor (best-effort)
        try:
            s.table("user_question_progress").delete().eq("user_id", uid).execute()
        except Exception:
            pass
        try:
            s.table("user_teacherpath_cursor").delete().eq("user_id", uid).execute()
        except Exception:
            pass
        try:
            s.table("learn_state").delete().eq("user_id", uid).execute()
        except Exception:
            pass
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
# FSRS (Free Spaced Repetition Scheduler) - learning mode "Wiederholung"
# =============================================================================
# Rating values follow py-fsrs:
#   1 = Again  (forgot)
#   2 = Hard
#   3 = Good
#   4 = Easy
# State values: 1=Learning, 2=Review, 3=Relearning

def fsrs_available() -> bool:
    return bool(_FSRS_AVAILABLE)


@st.cache_resource(show_spinner=False)
def fsrs_scheduler() -> Any:
    """Singleton FSRS scheduler with default parameters (FSRS-6 weights)."""
    if not _FSRS_AVAILABLE:
        return None
    return FsrsScheduler()


def _card_to_row(card: Any, uid: str, qid: str) -> Dict[str, Any]:
    """Serialize a py-fsrs Card to a row for the card_state table."""
    d = card.to_dict()
    return {
        "user_id": uid,
        "question_id": qid,
        "state": int(d.get("state") or 1),
        "step": int(d.get("step") or 0),
        "stability": d.get("stability"),
        "difficulty": d.get("difficulty"),
        "due": d.get("due"),
        "last_review": d.get("last_review"),
        "card_json": d,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _row_to_card(row: Dict[str, Any]) -> Any:
    """Restore a py-fsrs Card from a card_state row (preferring card_json)."""
    if not _FSRS_AVAILABLE:
        return None
    cj = row.get("card_json") or {}
    if isinstance(cj, dict) and cj:
        try:
            return FsrsCard.from_dict(cj)
        except Exception:
            pass
    # Fallback: build a fresh card (treat as new)
    return FsrsCard()


def db_get_card_state(uid: str, qid: str) -> Optional[Dict[str, Any]]:
    try:
        r = (
            supa()
            .table("card_state")
            .select("*")
            .eq("user_id", uid)
            .eq("question_id", qid)
            .limit(1)
            .execute()
        )
        rows = getattr(r, "data", None) or []
        return rows[0] if rows else None
    except Exception as e:
        dlog("db_get_card_state.err", err=str(e))
        return None


def db_upsert_card_state(uid: str, qid: str, card: Any, *,
                         response_ms: Optional[int],
                         rating: int,
                         is_lapse: bool) -> None:
    """Persist updated FSRS card. Best-effort."""
    try:
        row = _card_to_row(card, uid, qid)
        # Increment counters via RPC-less round-trip
        existing = db_get_card_state(uid, qid)
        review_count = int((existing or {}).get("review_count") or 0) + 1
        lapse_count = int((existing or {}).get("lapse_count") or 0) + (1 if is_lapse else 0)
        row.update({
            "review_count": review_count,
            "lapse_count": lapse_count,
            "last_rating": int(rating),
            "last_response_ms": int(response_ms) if response_ms is not None else None,
        })
        supa().table("card_state").upsert(row, on_conflict="user_id,question_id").execute()
    except Exception as e:
        dlog("db_upsert_card_state.err", err=str(e))


def db_insert_review_log(uid: str, qid: str, rating: int,
                         response_ms: Optional[int],
                         state_before: Optional[int],
                         state_after: Optional[int]) -> None:
    """Log a single review event. Best-effort."""
    try:
        supa().table("review_logs").insert({
            "user_id": uid,
            "question_id": qid,
            "rating": int(rating),
            "response_ms": int(response_ms) if response_ms is not None else None,
            "state_before": int(state_before) if state_before is not None else None,
            "state_after": int(state_after) if state_after is not None else None,
        }).execute()
    except Exception as e:
        dlog("db_insert_review_log.err", err=str(e))


def db_count_due_cards(uid: str, until: Optional[datetime] = None) -> int:
    """Count cards in card_state with due <= until (default: now)."""
    if until is None:
        until = datetime.now(timezone.utc)
    try:
        r = (
            supa()
            .table("card_state")
            .select("question_id", count="exact")
            .eq("user_id", uid)
            .lte("due", until.isoformat())
            .execute()
        )
        return int(getattr(r, "count", None) or 0)
    except Exception as e:
        dlog("db_count_due_cards.err", err=str(e))
        return 0


def db_load_due_question_ids(uid: str, until: Optional[datetime] = None) -> List[str]:
    """Return all question_ids whose card is due (<= until)."""
    if until is None:
        until = datetime.now(timezone.utc)
    try:
        r = (
            supa()
            .table("card_state")
            .select("question_id,due")
            .eq("user_id", uid)
            .lte("due", until.isoformat())
            .order("due", desc=False)
            .execute()
        )
        return [str(row["question_id"]) for row in (r.data or [])]
    except Exception as e:
        dlog("db_load_due_question_ids.err", err=str(e))
        return []


def db_load_existing_card_ids(uid: str) -> set:
    """Return set of question_ids that already have a card_state row for this user."""
    try:
        r = supa().table("card_state").select("question_id").eq("user_id", uid).execute()
        return {str(row["question_id"]) for row in (r.data or [])}
    except Exception as e:
        dlog("db_load_existing_card_ids.err", err=str(e))
        return set()


def build_fsrs_queue(
    questions: List[Dict[str, Any]],
    uid: str,
    *,
    new_per_session: int = 10,
    max_per_session: int = 50,
) -> List[Dict[str, Any]]:
    """Build a review queue: due cards first, then up to N new cards."""
    if not _FSRS_AVAILABLE:
        return []

    by_id = {str(q.get("id")): q for q in questions}

    # Due cards (already in card_state, due <= now)
    due_ids = db_load_due_question_ids(uid)
    due_qs = [by_id[qid] for qid in due_ids if qid in by_id]

    # New cards (no card_state row yet)
    seen_ids = db_load_existing_card_ids(uid)
    new_qs = [q for q in questions if str(q.get("id")) not in seen_ids]
    random.shuffle(new_qs)
    new_qs = new_qs[:max(0, int(new_per_session))]

    queue = list(due_qs) + list(new_qs)
    return queue[:max(1, int(max_per_session))]


def fsrs_rating_from_correctness(ok: bool) -> int:
    """Default rating mapping when user only chose A/B/C/D without explicit rating."""
    return int(FsrsRating.Good.value) if ok else int(FsrsRating.Again.value)


def fsrs_review(uid: str, qid: str, rating: int, response_ms: Optional[int]) -> None:
    """Run FSRS review_card and persist new state + review log."""
    if not _FSRS_AVAILABLE:
        return
    sched = fsrs_scheduler()
    if sched is None:
        return

    existing = db_get_card_state(uid, qid)
    card = _row_to_card(existing) if existing else FsrsCard()
    state_before = int(getattr(card, "state", 1) or 1)

    # py-fsrs Rating enum from int
    try:
        r = FsrsRating(int(rating))
    except Exception:
        r = FsrsRating.Good

    new_card, _log = sched.review_card(card, r)
    state_after = int(getattr(new_card, "state", 1) or 1)

    # A "lapse" = card was in Review (2) and got rated Again (1) -> moved to Relearning
    is_lapse = (state_before == 2 and int(rating) == 1)

    db_upsert_card_state(uid, qid, new_card,
                         response_ms=response_ms,
                         rating=int(rating),
                         is_lapse=is_lapse)
    db_insert_review_log(uid, qid, rating=int(rating),
                         response_ms=response_ms,
                         state_before=state_before,
                         state_after=state_after)


# =============================================================================
# Streak & Heatmap (uses review_logs + falls back to user_question_progress)
# =============================================================================
def db_load_review_dates(uid: str, since_days: int = 120) -> List[str]:
    """Return list of YYYY-MM-DD strings (one per review event) in the last N days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=int(since_days))
    out: List[str] = []
    # Primary source: review_logs (FSRS era)
    try:
        r = (
            supa()
            .table("review_logs")
            .select("review_datetime")
            .eq("user_id", uid)
            .gte("review_datetime", cutoff.isoformat())
            .execute()
        )
        for row in (r.data or []):
            ts = row.get("review_datetime") or ""
            if isinstance(ts, str) and len(ts) >= 10:
                out.append(ts[:10])
    except Exception as e:
        dlog("db_load_review_dates.review_logs.err", err=str(e))

    # Fallback / complement: user_question_progress.last_answer_at (pre-FSRS data)
    try:
        r = (
            supa()
            .table("user_question_progress")
            .select("last_answer_at")
            .eq("user_id", uid)
            .gte("last_answer_at", cutoff.isoformat())
            .execute()
        )
        for row in (r.data or []):
            ts = row.get("last_answer_at") or ""
            if isinstance(ts, str) and len(ts) >= 10:
                out.append(ts[:10])
    except Exception as e:
        dlog("db_load_review_dates.uqp.err", err=str(e))

    return out


def compute_streak(dates_iso: List[str]) -> int:
    """Compute consecutive-day streak ending today (or yesterday if not yet today).

    A streak counts days where the user did at least one review.
    """
    days = set(dates_iso)
    if not days:
        return 0
    today = datetime.now(timezone.utc).date()
    # Allow grace: if today not in days but yesterday is, streak still counts
    cur = today if today.isoformat() in days else (today - timedelta(days=1))
    if cur.isoformat() not in days:
        return 0
    streak = 0
    while cur.isoformat() in days:
        streak += 1
        cur = cur - timedelta(days=1)
    return streak


# =============================================================================
# UI / STYLES
# =============================================================================
def inject_css() -> None:
    st.markdown(
        """
<style>
:root{
  --pp-border: rgba(255,255,255,0.14);
  --pp-bg: rgba(255,255,255,0.055);
  --pp-bg2: rgba(255,255,255,0.075);
  --pp-text-muted: rgba(255,255,255,0.78);
  --pp-good: #2ecc71;
  --pp-bad:  #e74c3c;
  --pp-warn: #f39c12;
}
.block-container { padding-top: 1.2rem; max-width: 1180px; }
div.stButton > button { width:100%; padding:0.95rem 1rem; border-radius:14px; font-size:1rem; min-height:48px; }
div.stButton > button:hover { border-color: rgba(255,255,255,0.25); transform: translateY(-1px); }
div.stButton > button:active { transform: translateY(0px); }

.pp-card { box-shadow: 0 10px 30px rgba(0,0,0,0.28); border:1px solid var(--pp-border); border-radius:16px; padding:1rem 1.1rem; background: var(--pp-bg); }
.pp-card2 { box-shadow: 0 10px 30px rgba(0,0,0,0.22); border:1px solid var(--pp-border); border-radius:16px; padding:1rem 1.1rem; background: var(--pp-bg2); }
.pp-muted { color: var(--pp-text-muted); font-size:0.95rem; }
.pp-kpi { box-shadow: 0 10px 30px rgba(0,0,0,0.22); border:1px solid var(--pp-border); border-radius:16px; padding:0.9rem 1rem; background: var(--pp-bg); }
.pp-grid { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:0.8rem; }
@media (max-width: 1100px){ .pp-grid{ grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 700px){ .pp-grid{ grid-template-columns: 1fr; } }
hr { border:none; height:1px; background: var(--pp-border); margin: 1rem 0; }
.small { font-size: 0.9rem; opacity: 0.85; }
.pp-pill { display:inline-block; padding: 0.25rem 0.55rem; border:1px solid var(--pp-border); border-radius:999px; background: rgba(255,255,255,0.03); font-size:0.85rem; margin-right:0.4rem;}
h1{margin-bottom:0.6rem !important;}
h2{margin-top:1.2rem !important;}
.stMarkdown{margin-top:0.25rem;}

/* Streak / FSRS pills */
.pp-streak { display:inline-flex; align-items:center; gap:0.4rem; padding:0.3rem 0.7rem; border-radius:999px;
  border:1px solid var(--pp-border); background: rgba(243,156,18,0.10); color:#ffd27a; font-weight:700; }
.pp-due    { display:inline-flex; align-items:center; gap:0.4rem; padding:0.3rem 0.7rem; border-radius:999px;
  border:1px solid var(--pp-border); background: rgba(46,204,113,0.10); color:#a8f0c1; font-weight:700; }

/* Heatmap (90 days) */
.pp-heatmap { display:grid; grid-auto-flow: column; grid-template-rows: repeat(7, 14px); gap:3px; }
.pp-heatmap div { width:14px; height:14px; border-radius:3px; background: rgba(255,255,255,0.06); }
.pp-heatmap div.l1 { background:#1f4032; }
.pp-heatmap div.l2 { background:#2f6b4f; }
.pp-heatmap div.l3 { background:#3fa471; }
.pp-heatmap div.l4 { background:#4fd292; }

/* Mobile: compact spacing + larger touch targets */
@media (max-width: 640px){
  .block-container { padding-top: 0.6rem; padding-left:0.6rem; padding-right:0.6rem; }
  div.stButton > button { padding:1.05rem 0.8rem; font-size:1.02rem; min-height:54px; }
  .pp-card, .pp-card2, .pp-kpi { padding: 0.8rem 0.85rem; }
  h1{ font-size: 1.5rem; }
  h2{ font-size: 1.2rem; }
  .pp-heatmap { grid-template-rows: repeat(7, 11px); gap:2px; }
  .pp-heatmap div { width:11px; height:11px; }
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_heatmap_html(dates_iso: List[str], days: int = 90) -> str:
    """Render a 90-day GitHub-style heatmap from a list of YYYY-MM-DD strings."""
    from collections import Counter
    counts = Counter(dates_iso)
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=int(days) - 1)
    # Align start to Monday for nicer columns
    start = start - timedelta(days=start.weekday())
    cells = []
    cur = start
    while cur <= today:
        n = counts.get(cur.isoformat(), 0)
        if n == 0:
            cls = ""
        elif n <= 2:
            cls = "l1"
        elif n <= 5:
            cls = "l2"
        elif n <= 10:
            cls = "l3"
        else:
            cls = "l4"
        cells.append(
            f'<div class="{cls}" title="{cur.isoformat()}: {n} Antworten"></div>'
        )
        cur = cur + timedelta(days=1)
    return f'<div class="pp-heatmap">{"".join(cells)}</div>'


def inject_keyboard_shortcuts(*, mode: str = "answer") -> None:
    """Bind 1/2/3/4 + N keys to Streamlit buttons via parent-DOM lookup.

    mode = "answer"  -> 1..4 click answer buttons "A) ..", "B) ..", etc.; "n" -> "Weiter →"
    mode = "rate"    -> 1..4 click FSRS rating buttons (Nochmal/Schwer/Gut/Einfach); "n" -> "Weiter →"

    Implementation note: Streamlit's components.html() runs in an iframe but on
    the same origin as the parent app on Streamlit Cloud, so we can reach into
    window.parent.document and click buttons by their text content. If the
    browser blocks parent access, key shortcuts simply do nothing (graceful).
    """
    if mode == "rate":
        # FSRS rating prefixes (emojis + space)
        targets = {
            "1": "🔁 Nochmal",
            "2": "😬 Schwer",
            "3": "🙂 Gut",
            "4": "😎 Einfach",
        }
    else:
        # Answer A/B/C/D
        targets = {
            "1": "A) ",
            "2": "B) ",
            "3": "C) ",
            "4": "D) ",
        }
    next_label = "Weiter →"

    import json as _json
    targets_js = _json.dumps(targets, ensure_ascii=False)
    next_js = _json.dumps(next_label, ensure_ascii=False)

    components.html(
        f"""
<script>
(function() {{
  const TARGETS = {targets_js};
  const NEXT_LABEL = {next_js};

  function getParentDoc() {{
    try {{ return window.parent.document; }} catch(e) {{ return null; }}
  }}

  function clickByText(prefix) {{
    const doc = getParentDoc();
    if (!doc) return false;
    const buttons = doc.querySelectorAll('button');
    for (const b of buttons) {{
      const t = (b.innerText || '').trim();
      if (t.startsWith(prefix)) {{
        b.click();
        return true;
      }}
    }}
    return false;
  }}

  function onKey(e) {{
    // ignore typing in inputs
    const ae = (getParentDoc() || document).activeElement;
    const tag = ae && ae.tagName ? ae.tagName.toUpperCase() : '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || (ae && ae.isContentEditable)) return;

    const k = e.key;
    if (TARGETS[k]) {{
      if (clickByText(TARGETS[k])) e.preventDefault();
    }} else if (k === 'n' || k === 'N' || k === 'Enter') {{
      if (clickByText(NEXT_LABEL)) e.preventDefault();
    }}
  }}

  // Bind to parent doc so keys work even when iframe isn't focused
  const doc = getParentDoc();
  if (doc) {{
    if (window._kbShortcutsBound) {{
      doc.removeEventListener('keydown', window._kbShortcutsBound, true);
    }}
    doc.addEventListener('keydown', onKey, true);
    window._kbShortcutsBound = onKey;
  }}
}})();
</script>
""",
        height=0,
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
        "q_shown_at",
        "fsrs_rated",
    ]:
        st.session_state.pop(k, None)


def _go_to_dashboard() -> None:
    """Sauberer Wechsel zurück zur Übersicht (Dashboard).

    Setzt skip_teacher_autoresume=True, damit die Auto-Resume-Logik den User
    nicht wieder ins zuletzt aktive Kapitel zurückzieht.
    """
    st.session_state.page = "dashboard"
    st.session_state.skip_teacher_autoresume = True
    _reset_learning_state()
    st.rerun()


def _go_to_learn_plan() -> None:
    """Bleibt auf der Lernen-Seite, zeigt aber wieder die Modus-Auswahl an."""
    st.session_state.page = "learn"
    st.session_state.skip_teacher_autoresume = True
    _reset_learning_state()
    st.rerun()


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
        st.session_state.skip_teacher_autoresume = True
        _reset_learning_state()
        _reset_exam_state()
        st.rerun()
    if c2.button("Lernen", use_container_width=True):
        st.session_state.page = "learn"
        st.session_state.skip_teacher_autoresume = True
        _reset_exam_state()
        _reset_learning_state()
        st.rerun()
    if c3.button("Prüfung", use_container_width=True):
        st.session_state.page = "exam"
        st.session_state.skip_teacher_autoresume = True
        _reset_learning_state()
        st.rerun()

    st.sidebar.markdown("## Tools")
    st.sidebar.checkbox("Debug logs", key="debug_on", value=bool(st.session_state.get("debug_on", False)))

    # Wartung: Fortschritt zurücksetzen (nur userbezogene Daten)
    st.sidebar.markdown("## Wartung")
    with st.sidebar.expander("Fortschritt zurücksetzen", expanded=False):
        st.caption("Löscht deinen kompletten Lernfortschritt: Antworten, Notizen, Prüfungs-Historie und Wiederholungs-Plan. Die Fragen selbst bleiben erhalten.")
        confirm = st.checkbox("Ich weiß, dass das nicht rückgängig gemacht werden kann.", key="reset_confirm")
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

def _fmt_hhmmss(seconds: int) -> str:
    """Format seconds as HH:MM:SS (never negative)."""
    try:
        s = int(seconds)
    except Exception:
        s = 0
    if s < 0:
        s = 0
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _render_exam_countdown(deadline_ts: float, *, label: str = "Restzeit") -> None:
    """Client-side countdown (no rerun needed)."""
    try:
        dl = float(deadline_ts)
    except Exception:
        dl = time.time()

    components.html(
        f"""
<div id="pp-timer" style="display:inline-flex;align-items:center;gap:10px;padding:10px 14px;border-radius:999px;border:1px solid rgba(255,255,255,.22);background:rgba(16,18,22,.78);backdrop-filter: blur(10px);font-weight:900;color:#ffffff;box-shadow:0 10px 30px rgba(0,0,0,.35);">
  <span style="opacity:1;color:#ffffff;">⏱️ {label}</span>
  <span id="pp-timer-val" style="font-variant-numeric:tabular-nums;min-width:86px;text-align:right;color:#ffffff;letter-spacing:.5px;">--:--:--</span>
</div>

<script>
(function () {{
  const deadline = {dl} * 1000;
  const el = document.getElementById("pp-timer-val");
  function pad(n) {{ return String(n).padStart(2, "0"); }}
  function tick() {{
    const now = Date.now();
    let s = Math.floor((deadline - now) / 1000);
    if (s < 0) s = 0;
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (el) el.textContent = `${{pad(h)}}:${{pad(m)}}:${{pad(sec)}}`;
  }}
  tick();
  setInterval(tick, 1000);
}})();
</script>
""",
        height=46,
    )


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



# =============================================================================
# LEARNING PATH (Teacher Path via q["learn"])
# =============================================================================
LEARN_BLOCK_LABELS: Dict[int, str] = {
    1: "Grundlagen (Sprache/Zahlen)",
    2: "Karte/Navigation/Planung",
    3: "System & Grundregeln",
    4: "Lufträume & Verfahren",
    5: "Meteorologie",
}

def _init_learn_runtime_state() -> None:
    if "learn_answers" not in st.session_state:
        st.session_state.learn_answers = {}  # qid -> {"selected": int, "ok": bool}


def _subchapter_sequence(questions: List[Dict[str, Any]], category: str) -> List[str]:
    seen = set()
    out: List[str] = []
    for q in questions:
        if category != "Alle" and (q.get("category") or "") != category:
            continue
        sub = (q.get("subchapter") or "").strip()
        if not sub:
            continue
        if sub not in seen:
            seen.add(sub)
            out.append(sub)
    return out


def _next_subchapter(questions: List[Dict[str, Any]], category: str, current_sub: str) -> Optional[str]:
    seq = _subchapter_sequence(questions, category)
    if not seq:
        return None
    try:
        i = seq.index(current_sub)
    except ValueError:
        return seq[0]
    return seq[i + 1] if i + 1 < len(seq) else None




def _learn_meta(q: Dict[str, Any]) -> Tuple[int, int, int]:
    """Return (block, stage, difficulty) with safe defaults.

    If the JSON has no learn field, it is placed late.
    """
    l = q.get("learn")
    if isinstance(l, dict):
        try:
            b = int(l.get("block", 999) or 999)
        except Exception:
            b = 999
        try:
            s = int(l.get("stage", 999) or 999)
        except Exception:
            s = 999
        try:
            d = int(l.get("difficulty", 999) or 999)
        except Exception:
            d = 999
        return b, s, d
    return 999, 999, 999



def build_teacher_block_queue(
    questions: List[Dict[str, Any]],
    progress: Dict[str, Dict[str, Any]],
    block: int,
    only_unseen: bool,
    only_wrong: bool,
) -> List[Dict[str, Any]]:
    """Subset of teacher-path questions for a given block, sorted didactically."""
    qset = [q for q in questions if _learn_meta(q)[0] == int(block)]

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

    qset.sort(
        key=lambda q: (
            _learn_meta(q),
            int(q.get("local_number") or 0),
            str(q.get("id") or ""),
        )
    )
    return qset


def build_learning_queue_teacher_path(
    questions: List[Dict[str, Any]],
    progress: Dict[str, Dict[str, Any]],
    only_unseen: bool,
    only_wrong: bool,
) -> List[Dict[str, Any]]:
    """Teacher path ordering (block -> stage -> difficulty), with optional filters."""
    qset = list(questions)

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

    # Stable didactic sort; local_number as tie-breaker (keeps chapter-internal order when equal)
    qset.sort(
        key=lambda q: (
            _learn_meta(q),
            (q.get("category") or ""),
            (q.get("subchapter") or ""),
            int(q.get("local_number") or 0),
            str(q.get("id") or ""),
        )
    )
    return qset


def _teacher_path_stats(queue: List[Dict[str, Any]]) -> Dict[int, int]:
    """Count items per block in a queue."""
    out: Dict[int, int] = {k: 0 for k in LEARN_BLOCK_LABELS.keys()}
    for q in queue:
        b, _, _ = _learn_meta(q)
        if b in out:
            out[b] += 1
    return out

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
def _exam_compute_result(qlist: List[Dict[str, Any]], answers: Dict[str, Optional[int]]) -> Dict[str, Any]:
    """Compute exam result from the queued questions and the recorded answers.

    IMPORTANT: The UI expects result['details'] to exist and to contain items shaped like:
      {"qid": str, "q": question_dict, "selected": Optional[int], "correct": int, "ok": bool}
    """
    total = int(len(qlist) or 0)
    correct = 0
    details: List[Dict[str, Any]] = []

    for q in qlist:
        qid = str(q.get("id") or "")
        if not qid:
            continue

        sel = answers.get(qid, None)
        try:
            sel_i = int(sel) if sel is not None else None
        except Exception:
            sel_i = None

        try:
            ci = int(q.get("correctIndex"))
        except Exception:
            ci = -1

        ok = bool(sel_i is not None and ci >= 0 and sel_i == ci)
        if ok:
            correct += 1

        details.append({"qid": qid, "q": q, "selected": sel_i, "correct": ci, "ok": ok})

    pct = int(round((correct / total) * 100)) if total > 0 else 0
    passed = bool(pct >= int(PASS_PCT))
    return {"total": total, "correct": int(correct), "pct": int(pct), "passed": passed, "details": details}
def _exam_submit(uid: str, reason: str = "manual") -> None:
    """Finalize an exam attempt, compute result and (best-effort) persist to DB."""
    if st.session_state.get("exam_submitted", False):
        return

    qlist: List[Dict[str, Any]] = st.session_state.get("exam_queue", []) or []
    answers: Dict[str, Optional[int]] = st.session_state.get("exam_answers", {}) or {}

    result = _exam_compute_result(qlist, answers)
    st.session_state.exam_result = result
    st.session_state.exam_done = True
    st.session_state.exam_submitted = True

    ok, err = db_insert_exam_run(uid, total=int(result["total"]), correct=int(result["correct"]), passed=bool(result["passed"]))
    st.session_state.exam_save_ok = bool(ok)
    st.session_state.exam_save_err = str(err or "")
    dlog("exam.submit", uid=uid, reason=reason, **result, save_ok=ok)


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

    # ----------------------------
    # Streak + FSRS Due + 90-Tage-Heatmap
    # ----------------------------
    review_dates = db_load_review_dates(uid, since_days=120)
    streak = compute_streak(review_dates)
    due_n = db_count_due_cards(uid) if fsrs_available() else 0

    sc1, sc2 = st.columns([1, 1])
    with sc1:
        flame = "🔥" if streak > 0 else "·"
        st.markdown(
            f'<div class="pp-card2"><b>Lern-Serie</b><br>'
            f'<span class="pp-streak">{flame} {streak} Tag{"e" if streak != 1 else ""} in Folge</span>'
            f'<div class="pp-muted" style="margin-top:0.4rem">Antworte täglich mindestens eine Frage, um die Serie zu halten.</div></div>',
            unsafe_allow_html=True,
        )
    with sc2:
        if fsrs_available():
            st.markdown(
                f'<div class="pp-card2"><b>Smart wiederholen</b><br>'
                f'<span class="pp-due">📚 {due_n} fällig</span>'
                f'<div class="pp-muted" style="margin-top:0.4rem">Fragen, die du heute am ehesten vergessen würdest.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="pp-card2"><b>Smart wiederholen</b><br>'
                '<span class="pp-muted">Modul nicht installiert (<code>pip install fsrs</code>).</span></div>',
                unsafe_allow_html=True,
            )

    # 90-Tage-Heatmap
    with st.container():
        st.markdown('<div class="pp-card" style="margin-top:0.6rem"><b>Letzte 90 Tage</b>', unsafe_allow_html=True)
        st.markdown(render_heatmap_html(review_dates, days=90), unsafe_allow_html=True)
        st.markdown('<div class="pp-muted" style="margin-top:0.4rem">Jede Box = ein Tag. Dunkler = mehr Antworten.</div></div>', unsafe_allow_html=True)
    st.write("")

    # ----------------------------
    # Visualisierung: Fortschritt / Trefferquote pro Kategorie
    # ----------------------------
    try:
        rows = []
        for cat, subs in stats.items():
            total_q = 0
            learned_q = 0
            corr = 0
            wrong = 0
            for sub, s in subs.items():
                total_q += int(s.get("total") or 0)
                learned_q += int(s.get("learned") or 0)
                corr += int(s.get("correct_total") or 0)
                wrong += int(s.get("wrong_total") or 0)
            attempts = corr + wrong
            coverage = (learned_q / total_q) if total_q else 0.0
            acc = (corr / attempts) if attempts else 0.0
            rows.append(
                {
                    "Kategorie": cat,
                    "Abdeckung_%": int(round(coverage * 100)),
                    "Trefferquote_%": int(round(acc * 100)),
                }
            )
        df = pd.DataFrame(rows).sort_values("Kategorie")
        left, right = st.columns(2)
        with left:
            st.markdown("### Abdeckung nach Kategorie")
            st.bar_chart(df.set_index("Kategorie")[["Abdeckung_%"]], height=240)
        with right:
            st.markdown("### Trefferquote nach Kategorie")
            st.bar_chart(df.set_index("Kategorie")[["Trefferquote_%"]], height=240)
    except Exception:
        # Charts sind nice-to-have; UI darf nicht crashen, falls Daten fehlen.
        pass

    st.write("")
    # Smart-Wiederholen-Button als prominentester CTA, wenn fällige Fragen existieren
    if fsrs_available() and due_n > 0:
        if st.button(f"📚 {due_n} Frage{'n' if due_n != 1 else ''} jetzt smart wiederholen", type="primary", use_container_width=True):
            _reset_learning_state()
            st.session_state.page = "learn"
            st.session_state.skip_teacher_autoresume = True
            st.session_state.learn_plan = {
                "mode": "FSRS",
                "category": "Alle",
                "subchapter": "Alle",
                "only_unseen": False,
                "only_wrong": False,
            }
            st.session_state.queue = build_fsrs_queue(questions, uid, new_per_session=10, max_per_session=60)
            st.session_state.idx = 0
            st.session_state.answered = False
            st.session_state.learn_answers = {}
            st.session_state.learn_started = True
            st.rerun()

    cA, cB = st.columns([1, 1])
    if cA.button("Weiterlernen"):
        _reset_learning_state()
        st.session_state.page = "learn"
        st.session_state.skip_teacher_autoresume = True
        st.session_state.learn_plan = {
            "mode": "Zufällig",
            "category": "Alle",
            "subchapter": "Alle",
            "only_unseen": False,
            "only_wrong": False,
        }
        st.session_state.queue = build_learning_queue(
            questions=questions,
            progress=progress,
            category="Alle",
            subchapter="Alle",
            only_unseen=False,
            only_wrong=False,
        )
        st.session_state.idx = 0
        st.session_state.answered = False
        st.session_state.learn_answers = {}
        st.session_state.learn_started = True
        st.rerun()
    if cB.button("Falsche wiederholen"):
        _reset_learning_state()
        st.session_state.page = "learn"
        st.session_state.skip_teacher_autoresume = True
        st.session_state.learn_plan = {
            "mode": "Zufällig",
            "category": "Alle",
            "subchapter": "Alle",
            "only_unseen": False,
            "only_wrong": True,
        }
        st.session_state.queue = build_learning_queue(
            questions=questions,
            progress=progress,
            category="Alle",
            subchapter="Alle",
            only_unseen=False,
            only_wrong=True,
        )
        st.session_state.idx = 0
        st.session_state.answered = False
        st.session_state.learn_answers = {}
        st.session_state.learn_started = True
        st.rerun()
    if st.button("Prüfung starten (40)"):
        st.session_state.page = "exam"
        _reset_exam_state()
        st.rerun()


    st.write("")
    weak = weakest_subchapters(stats, min_seen=6, topn=8)
    target = weak[0] if weak else None
    st.markdown("## Wo solltest du als Nächstes ansetzen?")
    if target:
        acc = int(round(target["accuracy"] * 100))
        st.markdown(
            f"""<div class="pp-card2"><b>Vorschlag</b>
<div class="pp-muted">Hier hast du am meisten Luft nach oben: <b>{target['category']} · {target['subchapter']}</b> — aktuell {acc}% richtig bei {target['attempts']} Versuchen.</div></div>""",
            unsafe_allow_html=True,
        )
        if st.button("Übung starten (nur falsche Fragen)", use_container_width=True):
            _reset_learning_state()
            st.session_state.page = "learn"
            st.session_state.skip_teacher_autoresume = True
            st.session_state.learn_plan = {
                "mode": "Zufällig",
                "category": target["category"],
                "subchapter": target["subchapter"],
                "only_unseen": False,
                "only_wrong": True,
            }
            st.session_state.queue = build_learning_queue(
                questions=questions,
                progress=progress,
                category=target["category"],
                subchapter=target["subchapter"],
                only_unseen=False,
                only_wrong=True,
            )
            st.session_state.idx = 0
            st.session_state.answered = False
            st.session_state.learn_answers = {}
            st.session_state.learn_started = True
            st.rerun()
    else:
        st.caption("Noch nicht genug Daten für einen Vorschlag (mindestens 6 Antworten pro Unterkapitel nötig).")

    st.write("")
    st.markdown("## Deine häufigsten Fehler")
    wrong_rows = top_wrong_questions(questions, progress, topn=10)
    if not wrong_rows:
        st.caption("Noch keine falsch beantworteten Fragen — sehr gut.")
    else:
        for r in wrong_rows:
            q_short = r["question"][:140] + ("…" if len(r["question"]) > 140 else "")
            a1, a2 = st.columns([4, 1])
            a1.caption(f"{r['wrong']}× falsch · {r['category']} · {r['subchapter']}  |  {q_short}")
            if a2.button("Üben", key=f"wrong_{r['qid']}"):
                _reset_learning_state()
                st.session_state.page = "learn"
                st.session_state.skip_teacher_autoresume = True
                _cat_w = r["category"] or "Alle"
                _sub_w = r["subchapter"] or "Alle"
                st.session_state.learn_plan = {
                    "mode": "Zufällig",
                    "category": _cat_w,
                    "subchapter": _sub_w,
                    "only_unseen": False,
                    "only_wrong": True,
                }
                st.session_state.queue = build_learning_queue(
                    questions=questions,
                    progress=progress,
                    category=_cat_w,
                    subchapter=_sub_w,
                    only_unseen=False,
                    only_wrong=True,
                )
                st.session_state.idx = 0
                st.session_state.answered = False
                st.session_state.learn_answers = {}
                st.session_state.learn_started = True
                st.rerun()

    st.write("")
    st.markdown("## Themen im Überblick")
    for cat, subs in REQUIRED.items():
        st.markdown(f"### {cat}")
        for sub, expected_total in subs.items():
            s = stats.get(cat, {}).get(sub, {"total": 0, "learned": 0, "correct_total": 0, "wrong_total": 0})
            total = expected_total if expected_total else s["total"]
            learned = int(s["learned"])
            attempts = int(s["correct_total"]) + int(s["wrong_total"])
            acc = int(round((int(s["correct_total"]) / attempts) * 100)) if attempts else 0
            learned_pct = int(round((learned / total) * 100)) if total else 0

            line = f"{sub} ({total} Fragen) — bisher gesehen {learned_pct}% · Trefferquote {acc}% · {attempts} Versuche"
            c1, c2 = st.columns([4, 1])
            c1.caption(line)
            if c2.button("Üben", key=f"sub_{cat}::{sub}"):
                _reset_learning_state()
                st.session_state.page = "learn"
                st.session_state.skip_teacher_autoresume = True
                st.session_state.learn_plan = {
                    "mode": "Zufällig",
                    "category": cat,
                    "subchapter": sub,
                    "only_unseen": False,
                    "only_wrong": False,
                }
                st.session_state.queue = build_learning_queue(
                    questions=questions,
                    progress=progress,
                    category=cat,
                    subchapter=sub,
                    only_unseen=False,
                    only_wrong=False,
                )
                st.session_state.idx = 0
                st.session_state.answered = False
                st.session_state.learn_answers = {}
                st.session_state.learn_started = True
                st.rerun()

    st.write("")
    st.markdown("## Letzte Prüfungen")
    if runs:
        for r in runs[:10]:
            total = int(r.get("total") or 0)
            corr = int(r.get("correct") or 0)
            pct = int(round((corr / total) * 100)) if total else 0
            ok = "✓ bestanden" if bool(r.get("passed")) else "✗ nicht bestanden"
            st.caption(f"{pct}% ({corr}/{total}) — {ok}")
    else:
        st.caption("Noch keine Prüfungen abgelegt.")


# =============================================================================
# LEARN
# =============================================================================
def page_learn(uid: str, questions: List[Dict[str, Any]], progress: Dict[str, Dict[str, Any]]) -> None:
    st.title("Lernen")
    _init_learn_runtime_state()

    # ----------------------------
    # Persistent teacher state (best-effort)
    # ----------------------------
    if "teacher_state" not in st.session_state:
        st.session_state.teacher_state = db_get_teacher_state(uid) or {}

    def _teacher_defaults() -> Dict[str, Any]:
        # teacher_state schema (jsonb):
        # {
        #   "pathVersion": "teacherPath_v2",
        #   "unlockedBlock": 1,
        #   "checkpoints": {"1": 0, "2": 0, ...},
        #   "lastBlock": 1
        # }
        ts = st.session_state.teacher_state or {}
        if not isinstance(ts, dict):
            ts = {}
        if "pathVersion" not in ts:
            ts["pathVersion"] = "teacherPath_v2"
        if "unlockedBlock" not in ts or not isinstance(ts.get("unlockedBlock"), int):
            ts["unlockedBlock"] = 1
        if "checkpoints" not in ts or not isinstance(ts.get("checkpoints"), dict):
            ts["checkpoints"] = {str(b): 0 for b in sorted(LEARN_BLOCK_LABELS.keys())}
        else:
            for b in sorted(LEARN_BLOCK_LABELS.keys()):
                ts["checkpoints"].setdefault(str(b), 0)
        if "lastBlock" not in ts or not isinstance(ts.get("lastBlock"), int):
            ts["lastBlock"] = 1
        return ts

    st.session_state.teacher_state = _teacher_defaults()

    # Auto-resume teacher path after refresh (if a cursor exists)
    # This avoids losing progress mid-chapter when Streamlit session is restarted.
    # WICHTIG: Auto-Resume darf NIEMALS einen explizit gesetzten Plan überschreiben
    # (z.B. wenn der User vom Dashboard mit konkreter Kategorie/Unterkapitel kommt).
    _existing_plan = st.session_state.get("learn_plan")
    _has_explicit_plan = (
        isinstance(_existing_plan, dict)
        and (
            _existing_plan.get("mode") in ("Zufällig", "Lehrerpfad", "FSRS")
            or _existing_plan.get("category") not in (None, "", "Alle")
            or _existing_plan.get("subchapter") not in (None, "", "Alle")
            or bool(_existing_plan.get("only_unseen"))
            or bool(_existing_plan.get("only_wrong"))
            or _existing_plan.get("teacher_block") is not None
        )
    )
    if (
        (not st.session_state.get("skip_teacher_autoresume", False))
        and (not st.session_state.get("learn_started", False))
        and (not st.session_state.get("queue"))
        and (not _has_explicit_plan)
    ):
        latest = db_get_latest_teacher_cursor(uid)
        if latest:
            try:
                b = int(latest.get("chapter") or int(st.session_state.teacher_state.get("lastBlock", 1)))
            except Exception:
                b = int(st.session_state.teacher_state.get("lastBlock", 1))
            q = build_teacher_block_queue(
                questions=questions,
                progress=progress,
                block=b,
                only_unseen=False,
                only_wrong=False,
            )
            if q:
                st.session_state.learn_plan = {
                    "mode": "Lehrerpfad",
                    "teacher_block": b,
                    "only_unseen": False,
                    "only_wrong": False,
                    "category": "Alle",
                    "subchapter": "Alle",
                }
                st.session_state.queue = q
                try:
                    li = int(latest.get("last_question_idx") or 0)
                except Exception:
                    li = 0
                                # Resume on the next unanswered question (cursor stores last answered idx)
                st.session_state.idx = max(0, min(li + 1, len(q) - 1))
                st.session_state.answered = False
                st.session_state.learn_answers = {}
                st.session_state.learn_started = True

    # One-shot flag set by navigation/session end to prevent auto-resume
    st.session_state.pop("skip_teacher_autoresume", None)

    if "learn_plan" not in st.session_state:
        st.session_state.learn_plan = {"mode": "Zufällig", "category": "Alle", "subchapter": "Alle", "only_unseen": False, "only_wrong": False}
    if "learn_started" not in st.session_state:
        st.session_state.learn_started = False
    if "queue" not in st.session_state:
        st.session_state.queue = []
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "answered" not in st.session_state:
        st.session_state.answered = False

    # ----------------------------
    # Sidebar: global controls + (teacher) block navigation
    # ----------------------------
    with st.sidebar:
        st.subheader("Lernen")
        if st.session_state.learn_started:
            if st.button("Lernrunde beenden", use_container_width=True):
                _reset_learning_state()
                st.session_state.skip_teacher_autoresume = True
                st.session_state.page = "learn"
                st.rerun()

        st.divider()

        # Teacher block navigation (only if unlocked)
        ts = st.session_state.teacher_state
        unlocked = int(ts.get("unlockedBlock", 1))

        st.markdown("**Lehrerpfad – Kapitel**")
        for b in sorted(LEARN_BLOCK_LABELS.keys()):
            label = LEARN_BLOCK_LABELS[b]
            cp = int((ts.get("checkpoints", {}) or {}).get(str(b), 0))
            disabled = b > unlocked
            btn = st.button(
                f"{b}. {label}",
                disabled=disabled,
                use_container_width=True,
                help=("Noch gesperrt — schließe erst das vorherige Kapitel ab." if disabled else f"Fortsetzen bei Frage {cp+1}"),
                key=f"tp_jump_{b}",
            )
            if btn:
                # Start/continue teacher block session at checkpoint
                st.session_state.learn_plan = {
                    "mode": "Lehrerpfad",
                    "teacher_block": b,
                    "only_unseen": False,
                    "only_wrong": False,
                    "category": "Alle",
                    "subchapter": "Alle",
                }
                st.session_state.queue = build_teacher_block_queue(
                    questions=questions,
                    progress=progress,
                    block=b,
                    only_unseen=False,
                    only_wrong=False,
                )
                st.session_state.idx = max(0, min(cp, max(0, len(st.session_state.queue) - 1)))
                st.session_state.answered = False
                st.session_state.learn_answers = {}
                st.session_state.learn_started = True
                ts["lastBlock"] = b
                st.session_state.teacher_state = ts
                db_upsert_teacher_state(uid, ts)
                st.rerun()

        st.caption("Hinweis: Kapitel werden erst freigeschaltet, wenn du das vorherige Kapitel bis zum Ende durchgearbeitet hast.")

    # ----------------------------
    # Start / Plan screen (main)
    # ----------------------------
    if not st.session_state.learn_started:
        plan = st.session_state.learn_plan

        # UI-Labels (User-facing) <-> interne mode-Werte (technisch unverändert)
        UI_LABELS = {
            "Zufällig": "Frei lernen",
            "Lehrerpfad": "Lehrerpfad",
            "FSRS": "Smart wiederholen",
        }
        ui_to_mode = {v: k for k, v in UI_LABELS.items()}

        ui_modes = ["Frei lernen", "Lehrerpfad"]
        if fsrs_available():
            ui_modes.append("Smart wiederholen")

        cur_mode = plan.get("mode", "Zufällig")
        cur_label = UI_LABELS.get(cur_mode, "Frei lernen")
        idx_mode = ui_modes.index(cur_label) if cur_label in ui_modes else 0

        mode_ui = st.radio(
            "Wie möchtest du lernen?",
            ui_modes,
            horizontal=True,
            index=idx_mode,
            captions=[
                "Du wählst Kapitel und Themen selbst",
                "Schritt für Schritt durch alle Themen",
                "Wiederholt automatisch das, was du am ehesten vergisst",
            ][:len(ui_modes)],
        )

        if mode_ui == "Frei lernen":
            st.markdown("### Frei lernen")
            cats = sorted(set((q.get("category") or "").strip() for q in questions if q.get("category")))
            sel_category = st.selectbox(
                "Kategorie",
                ["Alle"] + cats,
                index=(["Alle"] + cats).index(plan.get("category", "Alle")) if plan.get("category", "Alle") in (["Alle"] + cats) else 0,
            )

            subs = sorted(set((q.get("subchapter") or "").strip() for q in questions if q.get("subchapter")))
            if sel_category != "Alle":
                subs = sorted(set((q.get("subchapter") or "").strip() for q in questions if (q.get("category") or "") == sel_category))
            sel_subchapter = st.selectbox(
                "Unterkapitel",
                ["Alle"] + subs,
                index=(["Alle"] + subs).index(plan.get("subchapter", "Alle")) if plan.get("subchapter", "Alle") in (["Alle"] + subs) else 0,
            )

            only_unseen = st.checkbox("Nur Fragen, die ich noch nicht gesehen habe", value=bool(plan.get("only_unseen", False)))
            only_wrong = st.checkbox("Nur Fragen, die ich falsch beantwortet habe", value=bool(plan.get("only_wrong", False)))

            c1, c2 = st.columns([1, 1])
            if c1.button("Loslegen", type="primary", use_container_width=True):
                st.session_state.learn_plan = {
                    "mode": "Zufällig",
                    "category": sel_category,
                    "subchapter": sel_subchapter,
                    "only_unseen": bool(only_unseen),
                    "only_wrong": bool(only_wrong),
                }
                st.session_state.queue = build_learning_queue(
                    questions=questions,
                    progress=progress,
                    category=sel_category,
                    subchapter=sel_subchapter,
                    only_unseen=bool(only_unseen),
                    only_wrong=bool(only_wrong),
                )
                st.session_state.idx = 0
                st.session_state.answered = False
                st.session_state.learn_answers = {}
                st.session_state.learn_started = True
                st.rerun()
            if c2.button("Zur Übersicht", use_container_width=True):
                _go_to_dashboard()

        elif mode_ui == "Smart wiederholen":
            st.markdown("### Smart wiederholen")
            due_n = db_count_due_cards(uid)
            seen_n = len(db_load_existing_card_ids(uid))
            new_avail = max(0, len(questions) - seen_n)
            st.markdown(
                f'<div class="pp-card2"><b>Stand</b>'
                f'<div class="pp-muted" style="margin-top:0.3rem">'
                f'Heute zu wiederholen: <b>{due_n}</b> · Bereits gelernt: <b>{seen_n}</b> · Noch nie gesehen: <b>{new_avail}</b>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            new_per = st.slider("Wie viele neue Fragen sollen dazukommen?", min_value=0, max_value=30, value=10, step=1)
            max_per = st.slider("Wie viele Fragen maximal in dieser Runde?", min_value=10, max_value=120, value=60, step=10)

            c1, c2 = st.columns([1, 1])
            start_disabled = (due_n == 0 and new_avail == 0)
            if c1.button("Loslegen", type="primary", use_container_width=True, disabled=start_disabled):
                st.session_state.learn_plan = {
                    "mode": "FSRS",
                    "category": "Alle",
                    "subchapter": "Alle",
                    "only_unseen": False,
                    "only_wrong": False,
                }
                st.session_state.queue = build_fsrs_queue(
                    questions, uid,
                    new_per_session=int(new_per),
                    max_per_session=int(max_per),
                )
                st.session_state.idx = 0
                st.session_state.answered = False
                st.session_state.learn_answers = {}
                st.session_state.learn_started = True
                st.rerun()
            if c2.button("Zur Übersicht", use_container_width=True, key="smart_to_dashboard"):
                _go_to_dashboard()

            if start_disabled:
                st.caption("Aktuell ist nichts zu wiederholen. Lerne erst ein paar Fragen über „Frei lernen“ oder „Lehrerpfad“ — sie werden dann automatisch hier eingeplant.")
            else:
                st.caption("So funktioniert's: Beantworte die Fragen wie gewohnt. Wenn du eine Frage richtig hattest, kannst du sie zusätzlich als „Schwer“ oder „Einfach“ markieren — dann sehen wir sie früher oder später wieder.")

        else:
            st.markdown("### Lehrerpfad")
            st.caption("Du arbeitest die Themen Kapitel für Kapitel durch — vom Einfachen zum Komplexen. Ein Kapitel wird erst freigeschaltet, wenn du jede Frage darin mindestens einmal richtig beantwortet hast.")
            ts = st.session_state.teacher_state
            unlocked = int(ts.get("unlockedBlock", 1))
            last_block = int(ts.get("lastBlock", 1))
            cp = int((ts.get("checkpoints", {}) or {}).get(str(last_block), 0))
            last_label = LEARN_BLOCK_LABELS.get(last_block, f"Kapitel {last_block}")
            st.markdown(
                f'<div class="pp-card2"><b>Dein Stand</b>'
                f'<div class="pp-muted" style="margin-top:0.3rem">'
                f'Freigeschaltet bis Kapitel <b>{unlocked}</b> · Zuletzt aktiv: <b>{last_label}</b> (bei Frage {cp+1})'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns([1, 1, 1])
            if c1.button(f"Weiter mit „{last_label}“", type="primary", use_container_width=True):
                b = last_block
                st.session_state.learn_plan = {"mode": "Lehrerpfad", "teacher_block": b, "only_unseen": False, "only_wrong": False, "category":"Alle", "subchapter":"Alle"}
                st.session_state.queue = build_teacher_block_queue(questions, progress, b, False, False)
                st.session_state.idx = max(0, min(cp, max(0, len(st.session_state.queue) - 1)))
                st.session_state.answered = False
                st.session_state.learn_answers = {}
                st.session_state.learn_started = True
                st.rerun()

            if c2.button("Von vorn beginnen", use_container_width=True):
                ts["unlockedBlock"] = max(1, int(ts.get("unlockedBlock", 1)))
                ts["lastBlock"] = 1
                ts["checkpoints"][str(1)] = 0
                st.session_state.teacher_state = ts
                db_upsert_teacher_state(uid, ts)

                st.session_state.learn_plan = {"mode": "Lehrerpfad", "teacher_block": 1, "only_unseen": False, "only_wrong": False, "category":"Alle", "subchapter":"Alle"}
                st.session_state.queue = build_teacher_block_queue(questions, progress, 1, False, False)
                st.session_state.idx = 0
                st.session_state.answered = False
                st.session_state.learn_answers = {}
                st.session_state.learn_started = True
                st.rerun()

            if c3.button("Kapitel wechseln", use_container_width=True):
                st.info("Tipp: Du kannst links in der Seitenleiste direkt zu jedem freigeschalteten Kapitel springen.")

            if st.button("Zur Übersicht", use_container_width=True):
                _go_to_dashboard()

        st.stop()

    # ----------------------------
    # Active session
    # ----------------------------
    plan = st.session_state.learn_plan
    queue: List[Dict[str, Any]] = st.session_state.get("queue", [])
    idx: int = int(st.session_state.get("idx", 0))

    if not queue:
        st.warning("Für diese Auswahl gibt es keine passenden Fragen.")
        if st.button("Andere Auswahl treffen", type="primary", use_container_width=True):
            _go_to_learn_plan()
        if st.button("Zur Übersicht", use_container_width=True):
            _go_to_dashboard()
        st.stop()

    # Session end handling
    if idx >= len(queue):
        # Stats für die Lernrunde berechnen
        answers_local = st.session_state.get("learn_answers") or {}
        ans_total = len(answers_local)
        ans_correct = sum(1 for v in answers_local.values() if isinstance(v, dict) and v.get("ok"))
        ans_pct = int(round((ans_correct / ans_total) * 100)) if ans_total else 0

        st.subheader("Lernrunde beendet 🎯")
        if ans_total > 0:
            st.markdown(
                f'<div class="pp-card"><b>Dein Ergebnis</b>'
                f'<div class="pp-muted" style="margin-top:0.3rem">{ans_correct} von {ans_total} richtig · {ans_pct}%</div></div>',
                unsafe_allow_html=True,
            )

        if plan.get("mode") == "Zufällig":
            sel_cat = plan.get("category", "Alle")
            sel_sub = plan.get("subchapter", "Alle")

            if sel_sub != "Alle" and sel_sub:
                nxt = _next_subchapter(questions, sel_cat, sel_sub)
                if nxt:
                    st.markdown(
                        f'<div class="pp-card2" style="margin-top:0.6rem"><b>Nächster Abschnitt</b>'
                        f'<div class="pp-muted" style="margin-top:0.3rem">{nxt}</div></div>',
                        unsafe_allow_html=True,
                    )
                    a1, a2 = st.columns([1, 1])
                    with a1:
                        if st.button(f"Weiter mit „{nxt}“", type="primary", use_container_width=True):
                            st.session_state.learn_plan = dict(plan)
                            st.session_state.learn_plan["subchapter"] = nxt
                            st.session_state.queue = build_learning_queue(
                                questions=questions,
                                progress=progress,
                                category=sel_cat,
                                subchapter=nxt,
                                only_unseen=bool(plan.get("only_unseen", False)),
                                only_wrong=bool(plan.get("only_wrong", False)),
                            )
                            st.session_state.idx = 0
                            st.session_state.answered = False
                            st.session_state.learn_answers = {}
                            st.rerun()
                    with a2:
                        if st.button("Zur Übersicht", use_container_width=True):
                            _go_to_dashboard()
                else:
                    st.info("Das war der letzte Abschnitt dieser Auswahl.")
                    if st.button("Zur Übersicht", type="primary", use_container_width=True):
                        _go_to_dashboard()
            else:
                if st.button("Zur Übersicht", type="primary", use_container_width=True):
                    _go_to_dashboard()

        elif plan.get("mode") == "FSRS":
            more_due = db_count_due_cards(uid)
            st.markdown(
                f'<div class="pp-card2" style="margin-top:0.6rem"><b>Wiederholungs-Status</b>'
                f'<div class="pp-muted" style="margin-top:0.3rem">Noch <b>{more_due}</b> fällige Frage{"n" if more_due != 1 else ""}.</div></div>',
                unsafe_allow_html=True,
            )
            a1, a2 = st.columns([1, 1])
            with a1:
                if st.button(
                    f"Weiter wiederholen ({more_due} fällig)" if more_due > 0 else "Keine Fragen mehr fällig",
                    type="primary",
                    disabled=(more_due == 0),
                    use_container_width=True,
                ):
                    st.session_state.queue = build_fsrs_queue(questions, uid, new_per_session=10, max_per_session=60)
                    st.session_state.idx = 0
                    st.session_state.answered = False
                    st.session_state.learn_answers = {}
                    st.rerun()
            with a2:
                if st.button("Zur Übersicht", use_container_width=True):
                    _go_to_dashboard()

        else:
            # Lehrerpfad: Nächstes Kapitel nur freischalten, wenn alle Fragen
            # mindestens einmal korrekt beantwortet wurden.
            ts = st.session_state.teacher_state
            b = int(plan.get("teacher_block", int(ts.get("lastBlock", 1))))

            block_qs = [str(qq.get("id")) for qq in questions if _learn_meta(qq)[0] == int(b)]
            not_ok = db_get_not_correct_once_question_ids(uid, block_qs)

            if not_ok:
                st.warning(
                    f"Du hast {len(not_ok)} Frage{'n' if len(not_ok) != 1 else ''} in diesem Kapitel noch nicht "
                    "richtig beantwortet. Wir starten jetzt eine kurze Wiederholungsrunde nur mit diesen Fragen — "
                    "danach ist das Kapitel freigeschaltet."
                )
                st.session_state.learn_plan = {
                    "mode": "Lehrerpfad",
                    "teacher_block": b,
                    "only_unseen": False,
                    "only_wrong": False,
                    "category": "Alle",
                    "subchapter": "Alle",
                }
                block_queue = build_teacher_block_queue(
                    questions=[qq for qq in questions if str(qq.get("id")) in set(not_ok)],
                    progress=progress,
                    block=b,
                    only_unseen=False,
                    only_wrong=False,
                )
                st.session_state.queue = block_queue
                st.session_state.idx = 0
                st.session_state.answered = False
                st.session_state.learn_answers = {}
                st.session_state.learn_started = True

                ts["checkpoints"][str(b)] = 0
                ts["lastBlock"] = b
                st.session_state.teacher_state = ts
                db_upsert_teacher_state(uid, ts)
                db_upsert_teacher_cursor(
                    uid, str(b),
                    str(block_queue[0].get("id")) if block_queue else "",
                    0,
                )
                st.rerun()
            else:
                # Alle Fragen mind. 1× richtig: Kapitel als abgeschlossen markieren
                # und ggf. das nächste freischalten.
                ts["checkpoints"][str(b)] = len(queue)
                ts["lastBlock"] = b
                if int(ts.get("unlockedBlock", 1)) <= b and b < max(LEARN_BLOCK_LABELS.keys()):
                    ts["unlockedBlock"] = b + 1
                    ts["checkpoints"].setdefault(str(b + 1), 0)
                st.session_state.teacher_state = ts
                db_upsert_teacher_state(uid, ts)

                cur_label = LEARN_BLOCK_LABELS.get(b, f"Kapitel {b}")
                st.success(f"🎉 Du hast das Kapitel **{cur_label}** abgeschlossen!")

                nxt_b = b + 1 if b < max(LEARN_BLOCK_LABELS.keys()) else None
                if nxt_b and nxt_b <= int(ts.get("unlockedBlock", 1)):
                    nxt_label = LEARN_BLOCK_LABELS.get(nxt_b, f"Kapitel {nxt_b}")
                    st.markdown(
                        f'<div class="pp-card2" style="margin-top:0.6rem"><b>Freigeschaltet: {nxt_label}</b>'
                        f'<div class="pp-muted" style="margin-top:0.3rem">Du kannst direkt weitermachen oder erst eine Pause einlegen.</div></div>',
                        unsafe_allow_html=True,
                    )
                    a1, a2 = st.columns([1, 1])
                    with a1:
                        if st.button(f"Weiter mit {nxt_label}", type="primary", use_container_width=True):
                            st.session_state.learn_plan = {
                                "mode": "Lehrerpfad",
                                "teacher_block": nxt_b,
                                "only_unseen": False,
                                "only_wrong": False,
                                "category": "Alle",
                                "subchapter": "Alle",
                            }
                            st.session_state.queue = build_teacher_block_queue(
                                questions, progress, nxt_b, False, False
                            )
                            st.session_state.idx = int(ts["checkpoints"].get(str(nxt_b), 0) or 0)
                            st.session_state.answered = False
                            st.session_state.learn_answers = {}
                            st.session_state.learn_started = True
                            ts["lastBlock"] = nxt_b
                            st.session_state.teacher_state = ts
                            db_upsert_teacher_state(uid, ts)
                            st.rerun()
                    with a2:
                        if st.button("Zur Übersicht", use_container_width=True):
                            _go_to_dashboard()
                else:
                    st.info("Du hast den gesamten Lehrerpfad durchgearbeitet. Glückwunsch! 🎉")
                    if st.button("Zur Übersicht", type="primary", use_container_width=True):
                        _go_to_dashboard()

        st.stop()

    q = queue[idx]
    # persist teacher checkpoint (handles refresh without losing position)
    if plan.get('mode') == 'Lehrerpfad':
        ts = st.session_state.teacher_state
        b = int(plan.get('teacher_block', _learn_meta(q)[0]))
        cur = ts.get('checkpoints', {}).get(str(b))
        if cur != int(idx):
            ts.setdefault('checkpoints', {})[str(b)] = int(idx)
            ts['lastBlock'] = b
            st.session_state.teacher_state = ts
            db_upsert_teacher_state(uid, ts)
    qid = str(q.get("id"))
    options = q.get("options") or []
    while len(options) < 4:
        options.append("")
    correct_index = int(q.get("correctIndex", -1))

    # Header
    if plan.get("mode") == "Lehrerpfad":
        b, s, d = _learn_meta(q)
        block_label = LEARN_BLOCK_LABELS.get(b, f"Kapitel {b}")
        st.caption(f"Lehrerpfad · {block_label} · Frage {idx+1} von {len(queue)}")
    elif plan.get("mode") == "FSRS":
        cs = db_get_card_state(uid, qid) if fsrs_available() else None
        if cs is None:
            tag = "🆕 neu"
        else:
            try:
                state_n = int(cs.get("state") or 1)
            except Exception:
                state_n = 1
            tag = {1: "🌱 frisch", 2: "🔁 wiederholen", 3: "♻️ auffrischen"}.get(state_n, "🔁")
        st.caption(f"Smart wiederholen · {tag} · Frage {idx+1} von {len(queue)}")
    else:
        cat_lbl = plan.get('category', 'Alle')
        sub_lbl = plan.get('subchapter', 'Alle')
        st.caption(f"Frei lernen · {cat_lbl} · {sub_lbl} · Frage {idx+1} von {len(queue)}")

    # Question card
    is_answered = bool(st.session_state.get("answered", False))
    kbd_hint = "⌨ Tastatur: 1–4 für Antwort · N für Weiter"
    st.markdown(
        f"""<div class="pp-card"><div><b>{(q.get("question") or "").strip()}</b></div>
<div class="pp-muted">{q.get("category","")} · {q.get("subchapter","")} · ID {qid} <span style="opacity:0.6">· {kbd_hint}</span></div></div>""",
        unsafe_allow_html=True,
    )

    # Navigation bar (cleaner, more compact)
    st.write("")
    total = len(queue)
    st.progress(min(1.0, (idx + 1) / max(1, total)))

    nav1, nav2, nav3, nav4 = st.columns([1, 1, 2, 1])
    with nav1:
        if st.button("← Zurück", disabled=(idx <= 0), use_container_width=True, key=f"nav_back_{qid}"):
            st.session_state.idx = max(0, idx - 1)
            # persist teacher checkpoint on move
            if plan.get('mode') == 'Lehrerpfad':
                ts = st.session_state.teacher_state
                b = int(plan.get('teacher_block', _learn_meta(q)[0]))
                if ts.get('checkpoints', {}).get(str(b)) != int(st.session_state.idx):
                    ts.setdefault('checkpoints', {})[str(b)] = int(st.session_state.idx)
                    ts['lastBlock'] = b
                    st.session_state.teacher_state = ts
                    db_upsert_teacher_state(uid, ts)
            prev_q = queue[st.session_state.idx]
            prev_id = str(prev_q.get("id"))
            prev = st.session_state.learn_answers.get(prev_id)
            if prev and isinstance(prev, dict) and prev.get("selected") is not None:
                st.session_state.answered = True
                st.session_state.last_selected_index = int(prev["selected"])
                st.session_state.last_ok = bool(prev.get("ok"))
                st.session_state.last_correct_index = int(prev_q.get("correctIndex", -1))
            else:
                st.session_state.answered = False
                st.session_state.last_ok = None
                st.session_state.last_correct_index = None
                st.session_state.last_selected_index = None
            st.rerun()

    with nav2:
        if st.button("Weiter →", disabled=(not st.session_state.get("answered", False)), use_container_width=True, key=f"nav_next_{qid}"):
            st.session_state.idx = idx + 1
            st.session_state.answered = False
            st.session_state.last_ok = None
            st.session_state.last_correct_index = None
            st.session_state.last_selected_index = None

            # persist teacher checkpoint on move
            if plan.get("mode") == "Lehrerpfad":
                ts = st.session_state.teacher_state
                b = int(plan.get("teacher_block", _learn_meta(q)[0]))
                ts["checkpoints"][str(b)] = int(st.session_state.idx)
                ts["lastBlock"] = b
                st.session_state.teacher_state = ts
                db_upsert_teacher_state(uid, ts)

            st.rerun()

    with nav3:
        jump = st.number_input("Zur Frage springen", min_value=1, max_value=total, value=idx + 1, step=1, key=f"jump_{qid}")
        if int(jump) != (idx + 1) and st.button("Springen", use_container_width=True, key=f"jump_btn_{qid}"):
            st.session_state.idx = int(jump) - 1
            # persist teacher checkpoint on jump
            if plan.get('mode') == 'Lehrerpfad':
                ts = st.session_state.teacher_state
                b = int(plan.get('teacher_block', _learn_meta(q)[0]))
                ts.setdefault('checkpoints', {})[str(b)] = int(st.session_state.idx)
                ts['lastBlock'] = b
                st.session_state.teacher_state = ts
                db_upsert_teacher_state(uid, ts)
            j_q = queue[st.session_state.idx]
            j_id = str(j_q.get("id"))
            prev = st.session_state.learn_answers.get(j_id)
            if prev and isinstance(prev, dict) and prev.get("selected") is not None:
                st.session_state.answered = True
                st.session_state.last_selected_index = int(prev["selected"])
                st.session_state.last_ok = bool(prev.get("ok"))
                st.session_state.last_correct_index = int(j_q.get("correctIndex", -1))
            else:
                st.session_state.answered = False
                st.session_state.last_ok = None
                st.session_state.last_correct_index = None
                st.session_state.last_selected_index = None

            if plan.get("mode") == "Lehrerpfad":
                ts = st.session_state.teacher_state
                b = int(plan.get("teacher_block", _learn_meta(j_q)[0]))
                ts["checkpoints"][str(b)] = int(st.session_state.idx)
                ts["lastBlock"] = b
                st.session_state.teacher_state = ts
                db_upsert_teacher_state(uid, ts)

            st.rerun()

    with nav4:
        if plan.get("mode") == "Lehrerpfad":
            ts = st.session_state.teacher_state
            b = int(plan.get("teacher_block", _learn_meta(q)[0]))
            if st.button("Kapitel neu starten", use_container_width=True, key=f"restart_block_{qid}",
                         help="Setzt den Fortschritt in diesem Kapitel zurück und beginnt bei Frage 1."):
                ts["checkpoints"][str(b)] = 0
                ts["lastBlock"] = b
                st.session_state.teacher_state = ts
                db_upsert_teacher_state(uid, ts)
                st.session_state.idx = 0
                st.session_state.answered = False
                st.session_state.learn_answers = {}
                st.rerun()

    st.write("")
    render_figures(q, max_n=3)

    labels = ["A", "B", "C", "D"]

    # Track when this question was shown (for response-latency)
    if not st.session_state.get("answered", False):
        # Reset timestamp the first time we render this specific question unanswered
        ts_key = f"q_shown_at__{qid}__{idx}"
        if st.session_state.get("q_shown_at_key") != ts_key:
            st.session_state.q_shown_at = time.time()
            st.session_state.q_shown_at_key = ts_key

    # Answer buttons
    if not st.session_state.get("answered", False):
        # Keyboard shortcuts: 1/2/3/4 = A/B/C/D
        inject_keyboard_shortcuts(mode="answer")
        for i_opt in range(4):
            opt = options[i_opt]
            if st.button(f"{labels[i_opt]}) {opt}", key=f"learn_{qid}_{i_opt}", use_container_width=True):
                ok = (i_opt == correct_index)

                # Compute response latency in ms
                shown_at = float(st.session_state.get("q_shown_at") or time.time())
                response_ms = max(0, int((time.time() - shown_at) * 1000))
                st.session_state.last_response_ms = response_ms

                counters = db_upsert_progress(uid, qid, ok)
                apply_progress_delta_local(uid, qid, counters)

                # NEW: persist per-question correctness for gating + teacher cursor (refresh-safe)
                try:
                    db_upsert_user_question_progress(uid, q, ok)
                    if plan.get('mode') == 'Lehrerpfad':
                        b_cur = int(plan.get('teacher_block', _learn_meta(q)[0]))
                        db_upsert_teacher_cursor(uid, str(b_cur), qid, int(st.session_state.idx))
                except Exception:
                    pass

                # FSRS: jede Antwort (egal welcher Modus) updated den Card-State.
                # So baut sich der Wiederholungs-Stapel automatisch auf, sobald
                # Michi überhaupt Fragen beantwortet — auch im Lehrerpfad/Zufalls-Modus.
                # Im FSRS-Modus selbst zeigen wir zusätzlich die manuelle Override-UI.
                if fsrs_available():
                    default_rating = fsrs_rating_from_correctness(ok)
                    fsrs_review(uid, qid, rating=default_rating, response_ms=response_ms)
                    st.session_state.fsrs_rated = int(default_rating)

                st.session_state.answered = True
                st.session_state.last_ok = ok
                st.session_state.last_selected_index = i_opt
                st.session_state.last_correct_index = correct_index
                st.session_state.learn_answers[qid] = {"selected": int(i_opt), "ok": bool(ok)}

                # Persist teacher checkpoint immediately (best-effort)
                if plan.get("mode") == "Lehrerpfad":
                    ts = st.session_state.teacher_state
                    b = int(plan.get("teacher_block", _learn_meta(q)[0]))
                    ts["checkpoints"][str(b)] = int(st.session_state.idx)
                    ts["lastBlock"] = b
                    st.session_state.teacher_state = ts
                    db_upsert_teacher_state(uid, ts)

                st.rerun()

    else:
        is_ok = bool(st.session_state.get("last_ok") or False)
        corr_i = st.session_state.get("last_correct_index")
        sel_i = st.session_state.get("last_selected_index")

        if is_ok:
            st.success(f"Richtig: {labels[int(sel_i)]}) {options[int(sel_i)]}" if sel_i is not None else "Richtig")
        else:
            st.error("Falsch")
            if corr_i is not None and 0 <= int(corr_i) < len(options):
                st.info(f"Richtig ist: {labels[int(corr_i)]}) {options[int(corr_i)]}")

        # Manuelles Rating (nur im "Smart wiederholen"-Modus)
        if plan.get("mode") == "FSRS" and fsrs_available():
            inject_keyboard_shortcuts(mode="rate")
            applied = int(st.session_state.get("fsrs_rated") or 0)
            applied_label = {1: "Nochmal", 2: "Schwer", 3: "Gut", 4: "Einfach"}.get(applied, "—")
            st.markdown(
                f'<div class="pp-card2"><b>Wie schwer fiel dir die Frage?</b>'
                f'<div class="pp-muted" style="margin-top:0.3rem">Aktuell gewertet als: <b>{applied_label}</b>. '
                f'Du kannst es unten überschreiben — wir planen die nächste Wiederholung dann entsprechend.</div></div>',
                unsafe_allow_html=True,
            )
            r1, r2, r3, r4 = st.columns([1, 1, 1, 1])
            if r1.button("🔁 Nochmal", key=f"fsrs_again_{qid}", use_container_width=True):
                fsrs_review(uid, qid, rating=1, response_ms=int(st.session_state.get("last_response_ms") or 0))
                st.session_state.fsrs_rated = 1
                st.toast("Bewertung: Nochmal")
            if r2.button("😬 Schwer", key=f"fsrs_hard_{qid}", use_container_width=True):
                fsrs_review(uid, qid, rating=2, response_ms=int(st.session_state.get("last_response_ms") or 0))
                st.session_state.fsrs_rated = 2
                st.toast("Bewertung: Schwer")
            if r3.button("🙂 Gut", key=f"fsrs_good_{qid}", use_container_width=True):
                fsrs_review(uid, qid, rating=3, response_ms=int(st.session_state.get("last_response_ms") or 0))
                st.session_state.fsrs_rated = 3
                st.toast("Bewertung: Gut")
            if r4.button("😎 Einfach", key=f"fsrs_easy_{qid}", use_container_width=True):
                fsrs_review(uid, qid, rating=4, response_ms=int(st.session_state.get("last_response_ms") or 0))
                st.session_state.fsrs_rated = 4
                st.toast("Bewertung: Einfach")

            # Show response latency (small)
            ms = int(st.session_state.get("last_response_ms") or 0)
            if ms > 0:
                st.caption(f"Antwortzeit: {ms/1000:.1f}s")
        else:
            # Non-FSRS: still allow N/Enter to advance
            inject_keyboard_shortcuts(mode="answer")

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

        # ----------------------------
        # Untere Navigation (damit man nach Wiki/KI/Notiz nicht hochscrollen muss)
        # ----------------------------
        st.markdown('<hr style="margin-top:1.2rem;margin-bottom:0.6rem">', unsafe_allow_html=True)
        bnav1, bnav2 = st.columns([1, 1])
        with bnav1:
            if st.button("← Zurück", disabled=(idx <= 0), use_container_width=True, key=f"bottom_back_{qid}"):
                st.session_state.idx = max(0, idx - 1)
                if plan.get('mode') == 'Lehrerpfad':
                    ts = st.session_state.teacher_state
                    b = int(plan.get('teacher_block', _learn_meta(q)[0]))
                    if ts.get('checkpoints', {}).get(str(b)) != int(st.session_state.idx):
                        ts.setdefault('checkpoints', {})[str(b)] = int(st.session_state.idx)
                        ts['lastBlock'] = b
                        st.session_state.teacher_state = ts
                        db_upsert_teacher_state(uid, ts)
                prev_q = queue[st.session_state.idx]
                prev_id = str(prev_q.get("id"))
                prev = st.session_state.learn_answers.get(prev_id)
                if prev and isinstance(prev, dict) and prev.get("selected") is not None:
                    st.session_state.answered = True
                    st.session_state.last_selected_index = int(prev["selected"])
                    st.session_state.last_ok = bool(prev.get("ok"))
                    st.session_state.last_correct_index = int(prev_q.get("correctIndex", -1))
                else:
                    st.session_state.answered = False
                    st.session_state.last_ok = None
                    st.session_state.last_correct_index = None
                    st.session_state.last_selected_index = None
                st.rerun()
        with bnav2:
            if st.button("Weiter →", type="primary", use_container_width=True, key=f"bottom_next_{qid}"):
                st.session_state.idx = idx + 1
                st.session_state.answered = False
                st.session_state.last_ok = None
                st.session_state.last_correct_index = None
                st.session_state.last_selected_index = None
                if plan.get("mode") == "Lehrerpfad":
                    ts = st.session_state.teacher_state
                    b = int(plan.get("teacher_block", _learn_meta(q)[0]))
                    ts["checkpoints"][str(b)] = int(st.session_state.idx)
                    ts["lastBlock"] = b
                    st.session_state.teacher_state = ts
                    db_upsert_teacher_state(uid, ts)
                st.rerun()


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
    deadline = float(st.session_state.get("exam_deadline_ts") or (time.time() + EXAM_DURATION_SEC))
    remaining = int(round(deadline - time.time()))

    if remaining <= 0 and not st.session_state.get("exam_done", False):
        _exam_submit(uid, reason="time")
        st.rerun()

    top1, top2, top3, top4 = st.columns([1, 1, 1, 1])
    with top1:
        _render_exam_countdown(deadline, label='Restzeit')
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
        for d in (result.get("details") or []):
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

    cA, cB = st.columns([1, 1])
    if cA.button("◀ Zurück", use_container_width=True, disabled=(i == 0)):
        st.session_state.exam_idx = max(0, i - 1)
        st.rerun()
    if cB.button("Weiter ▶", use_container_width=True, disabled=(i >= total - 1)):
        st.session_state.exam_idx = min(total - 1, i + 1)
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
