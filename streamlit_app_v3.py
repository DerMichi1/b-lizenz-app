import os
import json
import random
from pathlib import Path

import streamlit as st
from supabase import create_client, Client

# -----------------------------
# CONFIG
# -----------------------------
APP_DIR = Path(__file__).parent
QUESTIONS_PATH = APP_DIR / "questions.json"
BILDER_PDF = APP_DIR / "Bilder.pdf"

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()

# Optional: KI-Erklärungen/Merksatz/Referenzen
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()


# -----------------------------
# SUPABASE
# -----------------------------
@st.cache_resource(show_spinner=False)
def supa() -> Client:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_ANON_KEY fehlen (Secrets setzen).")
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def auth_ui():
    st.sidebar.markdown("## Login")
    tab_login, tab_signup = st.sidebar.tabs(["Login", "Registrieren"])

    with tab_login:
        email = st.text_input("E-Mail", key="login_email")
        pw = st.text_input("Passwort", type="password", key="login_pw")
        if st.button("Login", use_container_width=True):
            try:
                res = supa().auth.sign_in_with_password({"email": email, "password": pw})
                st.session_state["sb_user"] = res.user
                st.success("Login ok.")
                st.rerun()
            except Exception as e:
                st.error(f"Login fehlgeschlagen: {e}")

    with tab_signup:
        email2 = st.text_input("E-Mail ", key="signup_email")
        pw2 = st.text_input("Passwort ", type="password", key="signup_pw")
        if st.button("Registrieren", use_container_width=True):
            try:
                supa().auth.sign_up({"email": email2, "password": pw2})
                st.info("Registrierung ausgelöst. Prüfe ggf. deine E-Mail (Bestätigung).")
            except Exception as e:
                st.error(f"Registrierung fehlgeschlagen: {e}")

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True):
        try:
            supa().auth.sign_out()
        except Exception:
            pass
        st.session_state.pop("sb_user", None)
        st.session_state.pop("progress", None)
        st.rerun()

def require_login() -> str:
    if "sb_user" not in st.session_state or not st.session_state["sb_user"]:
        auth_ui()
        st.info("Bitte einloggen, um fortzufahren.")
        st.stop()

    u = st.session_state["sb_user"]

    # supabase-py liefert i.d.R. ein User-Objekt mit .id
    if hasattr(u, "id") and u.id:
        return u.id

    # Fallback, falls doch mal dict/json drin liegt
    if isinstance(u, dict) and u.get("id"):
        return u["id"]

    raise RuntimeError("Supabase-User hat keine id (Session-State inkonsistent).")

def db_load_progress(user_id: str) -> dict:
    resp = supa().table("progress").select("*").eq("user_id", user_id).execute()
    p = {}
    for r in resp.data:
        p[r["question_id"]] = {"seen": r["seen"], "correct": r["correct"], "wrong": r["wrong"]}
    return p

def db_upsert_progress(user_id: str, question_id: str, ok: bool):
    resp = supa().table("progress").select("*").eq("user_id", user_id).eq("question_id", question_id).execute()
    if resp.data:
        row = resp.data[0]
        seen = int(row.get("seen", 0)) + 1
        correct = int(row.get("correct", 0)) + (1 if ok else 0)
        wrong = int(row.get("wrong", 0)) + (0 if ok else 1)
        supa().table("progress").update({
            "seen": seen,
            "correct": correct,
            "wrong": wrong,
            "updated_at": "now()"
        }).eq("user_id", user_id).eq("question_id", question_id).execute()
    else:
        supa().table("progress").insert({
            "user_id": user_id,
            "question_id": question_id,
            "seen": 1,
            "correct": 1 if ok else 0,
            "wrong": 0 if ok else 1
        }).execute()

def db_reset_progress(user_id: str, scope_category: str | None = None, scope_sub: str | None = None, questions: list[dict] | None = None):
    if not scope_category and not scope_sub:
        supa().table("progress").delete().eq("user_id", user_id).execute()
        return

    if not questions:
        return
    ids = [q["id"] for q in questions if (not scope_category or q["category"] == scope_category) and (not scope_sub or q["subchapter"] == scope_sub)]
    if not ids:
        return
    supa().table("progress").delete().eq("user_id", user_id).in_("question_id", ids).execute()


# -----------------------------
# OPTIONAL: KI Auto-Wiki
# -----------------------------
def try_auto_wiki(q: dict) -> dict | None:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        labels = ["A", "B", "C", "D"]
        opts = "\n".join([f"{labels[i]}) {q['options'][i]}" for i in range(4)])
        correct = labels[q["correctIndex"]] if q.get("correctIndex") is not None else "?"

        prompt = f"""Du bist Fluglehrer (Gleitschirm B-Lizenz). Antworte NUR als JSON.

Schema:
{{
  "explain": "max 3 Sätze, sachlich",
  "merksatz": "1 Satz",
  "refs": [{{"title":"...", "url":"..."}}]
}}

Frage:
{q["question"]}

Antworten:
{opts}

Richtig: {correct}

Kategorie: {q["category"]} / {q["subchapter"]}

Regeln:
- keine Floskeln
- refs thematisch passend (SERA/ICAO/AIP/Meteo/Navigation)
"""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content.strip()

        data = json.loads(resp)
        return {
            "auto_explain": (data.get("explain") or "").strip(),
            "auto_merksatz": (data.get("merksatz") or "").strip(),
            "auto_refs": data.get("refs") or [],
        }
    except Exception:
        return None


# -----------------------------
# QUESTIONS / UI
# -----------------------------
@st.cache_data(show_spinner=False)
def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

@st.cache_resource(show_spinner=False)
def get_bilder_doc():
    import fitz
    return fitz.open(BILDER_PDF)

def render_bilder_page(page_1_based: int, zoom: float = 2.0) -> bytes:
    import fitz
    doc = get_bilder_doc()
    page = doc.load_page(page_1_based - 1)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")

def build_queue(questions: list[dict], mode: str, cat: str | None, sub: str | None, hard_only: bool, progress: dict) -> list[str]:
    pool = questions[:]
    if mode == "Lernmodus":
        if cat:
            pool = [q for q in pool if q["category"] == cat]
        if sub:
            pool = [q for q in pool if q["subchapter"] == sub]
    else:
        random.shuffle(pool)
        pool = pool[:40]

    if hard_only:
        def is_hard(q):
            stq = progress.get(q["id"], {})
            return stq.get("wrong", 0) > 0 and stq.get("correct", 0) == 0
        pool = [q for q in pool if is_hard(q)]

    random.shuffle(pool)
    return [q["id"] for q in pool]

def q_by_id(questions: list[dict], qid: str) -> dict | None:
    for q in questions:
        if q["id"] == qid:
            return q
    return None

def subchapter_percent(questions: list[dict], cat: str, sub: str, progress: dict) -> int:
    ids = [q["id"] for q in questions if q["category"] == cat and q["subchapter"] == sub]
    if not ids:
        return 0
    correct_once = sum(1 for qid in ids if progress.get(qid, {}).get("correct", 0) > 0)
    return int(round(100 * correct_once / len(ids)))

def category_percent(questions: list[dict], cat: str, progress: dict) -> int:
    subs = sorted({q["subchapter"] for q in questions if q["category"] == cat})
    if not subs:
        return 0
    return int(round(sum(subchapter_percent(questions, cat, s, progress) for s in subs) / len(subs)))

def total_percent(questions: list[dict], progress: dict) -> int:
    ids = [q["id"] for q in questions]
    if not ids:
        return 0
    correct_once = sum(1 for qid in ids if progress.get(qid, {}).get("correct", 0) > 0)
    return int(round(100 * correct_once / len(ids)))

def ui_css():
    st.markdown("""
    <style>
      .stButton>button { width: 100%; padding: 0.9rem 1rem; font-size: 1.05rem; border-radius: 0.9rem; }
      .pill { display:inline-block; padding: 0.25rem 0.6rem; border-radius: 999px;
              border: 1px solid rgba(255,255,255,0.18); font-size: 0.85rem; margin-right: 0.4rem; }
      .muted { color: rgba(255,255,255,0.70); }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# APP
# -----------------------------
st.set_page_config(page_title="B-Lizenz Gleitschirm – Lernapp", layout="wide")
ui_css()

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("SUPABASE_URL / SUPABASE_ANON_KEY fehlen. Setze diese als Secrets/Env-Variablen im Hosting.")
    st.stop()

user_id = require_login()
questions = load_questions()

if "progress" not in st.session_state:
    with st.spinner("Lade Fortschritt..."):
        st.session_state.progress = db_load_progress(user_id)

progress = st.session_state.progress

with st.sidebar:
    st.markdown("## Modus & Filter")
    mode = st.selectbox("Modus", ["Lernmodus", "Prüfungsmodus (40)"], index=0)
    cats = ["—"] + sorted({q["category"] for q in questions})
    cat = st.selectbox("Kategorie", cats, index=0)
    cat = None if cat == "—" else cat

    subs = ["—"]
    if cat:
        subs += sorted({q["subchapter"] for q in questions if q["category"] == cat})
    sub = st.selectbox("Unterkapitel", subs, index=0)
    sub = None if sub == "—" else sub

    hard_only = st.toggle("Nur schwierige (noch nie korrekt)", value=False)

    st.markdown("---")
    st.markdown("## Fortschritt")
    if cat and sub:
        st.metric("Unterkapitel", f"{subchapter_percent(questions, cat, sub, progress)}%")
    elif cat:
        st.metric("Kategorie", f"{category_percent(questions, cat, progress)}%")
    else:
        st.metric("Gesamt", f"{total_percent(questions, progress)}%")

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Session starten"):
            st.session_state.queue = build_queue(questions, mode, cat, sub, hard_only, progress)
            st.session_state.idx = 0
            st.session_state.answered = False
            st.session_state.sel = None
            st.session_state.auto = None
            st.rerun()
    with colB:
        if st.button("Reset (Scope)"):
            db_reset_progress(user_id, cat, sub, questions=questions)
            st.session_state.progress = db_load_progress(user_id)
            st.success("Reset ok.")
            st.rerun()

st.title("Gleitschirm B-Lizenz – Lernapp (Multi-User)")

if "queue" not in st.session_state:
    st.session_state.queue = build_queue(questions, "Lernmodus", None, None, False, progress)
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "answered" not in st.session_state:
    st.session_state.answered = False
if "sel" not in st.session_state:
    st.session_state.sel = None
if "auto" not in st.session_state:
    st.session_state.auto = None

qid = st.session_state.queue[st.session_state.idx] if st.session_state.queue else None
q = q_by_id(questions, qid) if qid else None
if not q:
    st.info("Keine Fragen im aktuellen Pool.")
    st.stop()

left, right = st.columns([3, 1])
with left:
    st.markdown(
        f"<span class='pill'>{q['category']}</span><span class='pill'>{q['subchapter']}</span><span class='pill'>ID {q['id']}</span>",
        unsafe_allow_html=True
    )
with right:
    st.markdown(f"<div class='muted' style='text-align:right'>Frage {st.session_state.idx+1}/{len(st.session_state.queue)}</div>", unsafe_allow_html=True)

st.markdown(f"### {q['question']}")

if q.get("figures"):
    with st.expander("Abbildungen"):
        for fi in q["figures"]:
            fig_n = fi["figure"]
            page = fi.get("bilder_page")
            st.markdown(f"**Abbildung {fig_n}** (Bilder.pdf Seite {page})")
            if page:
                st.image(render_bilder_page(page, zoom=2.0), use_container_width=True)

labels = ["A", "B", "C", "D"]
cols = st.columns(2)
opts = q["options"]

def choose(i: int):
    st.session_state.sel = i
    st.session_state.answered = True

st.markdown("#### Antworten")
for i in range(4):
    with cols[i % 2]:
        if st.button(f"{labels[i]}) {opts[i]}", key=f"opt_{qid}_{i}", disabled=st.session_state.answered):
            choose(i)

if st.session_state.answered:
    correct = q.get("correctIndex", None)
    sel = st.session_state.sel

    ok = (sel == correct) if correct is not None else False
    if correct is None:
        st.error("Keine Lösung vorhanden.")
    elif ok:
        st.success(f"Richtig: {labels[correct]}")
    else:
        st.error(f"Falsch. Richtig wäre: {labels[correct]}")

    if correct is not None:
        db_upsert_progress(user_id, qid, ok)
        stq = progress.get(qid, {"seen": 0, "correct": 0, "wrong": 0})
        stq["seen"] += 1
        if ok: stq["correct"] += 1
        else: stq["wrong"] += 1
        progress[qid] = stq
        st.session_state.progress = progress

    st.markdown("#### Erklärung & Merksatz")
    if st.session_state.auto is None:
        st.session_state.auto = try_auto_wiki(q)

    if st.session_state.auto:
        st.write(st.session_state.auto.get("auto_explain", ""))
        ms = st.session_state.auto.get("auto_merksatz", "")
        if ms:
            st.info(ms)
        refs = st.session_state.auto.get("auto_refs", [])
        if refs:
            st.markdown("#### Offizielle Referenzen")
            for r in refs[:6]:
                title = r.get("title", "Quelle")
                url = r.get("url", "")
                if url:
                    st.markdown(f"- [{title}]({url})")
    else:
        st.caption("Optional: OPENAI_API_KEY setzen, um Erklärung/Merksatz/Referenzen automatisch zu erzeugen.")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Nächste Frage"):
            st.session_state.idx = (st.session_state.idx + 1) % len(st.session_state.queue)
            st.session_state.answered = False
            st.session_state.sel = None
            st.session_state.auto = None
            st.rerun()
    with c2:
        if st.button("Session neu mischen"):
            st.session_state.queue = build_queue(questions, mode, cat, sub, hard_only, progress)
            st.session_state.idx = 0
            st.session_state.answered = False
            st.session_state.sel = None
            st.session_state.auto = None
            st.rerun()
    with c3:
        if st.button("Zurücksetzen (nur diese Frage)"):
            supa().table("progress").delete().eq("user_id", user_id).eq("question_id", qid).execute()
            progress.pop(qid, None)
            st.session_state.progress = progress
            st.success("Zurückgesetzt.")
            st.rerun()
else:
    st.caption("Antwort wählen → sofortige Korrektur + Erklärung (optional).")
