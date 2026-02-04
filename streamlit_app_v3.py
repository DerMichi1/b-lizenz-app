import os
import json
import random
from pathlib import Path

import streamlit as st
from supabase import create_client, Client

# -----------------------------
# SECRETS HELPER
# -----------------------------
def get_secret(name: str, default: str = "") -> str:
    if name in st.secrets:
        return str(st.secrets[name]).strip()
    return os.getenv(name, default).strip()

SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_ANON_KEY = get_secret("SUPABASE_ANON_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-4o-mini")

APP_DIR = Path(__file__).parent
QUESTIONS_PATH = APP_DIR / "questions.json"

# -----------------------------
# SUPABASE CLIENT (CACHED)
# -----------------------------
@st.cache_resource(show_spinner=False)
def supa() -> Client:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("Supabase Secrets fehlen.")
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# -----------------------------
# AUTH
# -----------------------------
def auth_ui():
    st.sidebar.markdown("## Login")

    email = st.sidebar.text_input("E-Mail")
    pw = st.sidebar.text_input("Passwort", type="password")

    if st.sidebar.button("Login"):
        try:
            res = supa().auth.sign_in_with_password({"email": email, "password": pw})
            st.session_state.user = res.user
            st.session_state.session = res.session
            supa().auth.set_session(res.session.access_token, res.session.refresh_token)
            st.rerun()
        except Exception as e:
            st.sidebar.error(str(e))

    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

def require_login():
    if "user" not in st.session_state:
        auth_ui()
        st.stop()
    return st.session_state.user.id

# -----------------------------
# DB
# -----------------------------
def db_load_progress(uid: str):
    r = supa().table("progress").select("*").eq("user_id", uid).execute()
    return {x["question_id"]: x for x in r.data}

def db_upsert(uid, qid, ok):
    s = supa()
    r = s.table("progress").select("*").eq("user_id", uid).eq("question_id", qid).execute()

    if r.data:
        row = r.data[0]
        s.table("progress").update({
            "seen": row["seen"] + 1,
            "correct": row["correct"] + (1 if ok else 0),
            "wrong": row["wrong"] + (0 if ok else 1),
        }).eq("user_id", uid).eq("question_id", qid).execute()
    else:
        s.table("progress").insert({
            "user_id": uid,
            "question_id": qid,
            "seen": 1,
            "correct": 1 if ok else 0,
            "wrong": 0 if ok else 1
        }).execute()

# -----------------------------
# QUESTIONS
# -----------------------------
@st.cache_data
def load_questions():
    return json.loads(QUESTIONS_PATH.read_text("utf-8"))

# -----------------------------
# OPTIONAL OPENAI
# -----------------------------
def ai_explain(q):
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        txt = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": q}],
            temperature=0.2
        ).choices[0].message.content
        return txt
    except Exception:
        return None

# -----------------------------
# APP
# -----------------------------
st.set_page_config("Lernapp", layout="wide")

if not QUESTIONS_PATH.exists():
    st.error("questions.json fehlt")
    st.stop()

uid = require_login()
questions = load_questions()

if "progress" not in st.session_state:
    st.session_state.progress = db_load_progress(uid)

progress = st.session_state.progress

if "queue" not in st.session_state:
    st.session_state.queue = random.sample(questions, len(questions))
    st.session_state.idx = 0

q = st.session_state.queue[st.session_state.idx]

st.title("Lernapp")

st.markdown("### " + q["question"])

labels = ["A","B","C","D"]

for i,opt in enumerate(q["options"]):
    if st.button(f"{labels[i]}) {opt}", key=f"opt{i}"):
        ok = i == q["correctIndex"]
        db_upsert(uid, q["id"], ok)
        st.success("Richtig" if ok else "Falsch")

        if OPENAI_API_KEY:
            with st.spinner("Erklärung..."):
                exp = ai_explain(q["question"])
                if exp:
                    st.info(exp)

        st.session_state.idx = (st.session_state.idx + 1) % len(st.session_state.queue)
        st.rerun()
