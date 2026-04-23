# ==========================================================
# 🎙️ AUDIORAG FINAL VERSION (FIXED - April 2026)
# Changes Made:
# ✅ gemini-2.5-flash kept
# ✅ embedding-001 REMOVED (causing 404)
# ✅ Local HuggingFace embeddings added (FREE forever)
# ✅ No Gemini quota used for embeddings
# ==========================================================

import streamlit as st
import os
from io import BytesIO
from dotenv import load_dotenv
from gtts import gTTS

import google.generativeai as genai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ LOCAL EMBEDDINGS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================================
# CONFIG
# ==========================================================
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file")
    st.stop()

genai.configure(api_key=API_KEY)

FAISS_PATH = "faiss_index"

SUPPORTED_FORMATS = [
    "mp3", "wav", "m4a", "ogg", "flac", "aac", "wma", "webm"
]

MIME_MAP = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "m4a": "audio/mp4",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "wma": "audio/x-ms-wma",
    "webm": "audio/webm",
}

# ==========================================================
# STAGE 1 → 2  | TRANSCRIBE AUDIO
# ==========================================================
def transcribe_audio(audio_file):

    audio_bytes = audio_file.read()
    ext = audio_file.name.split(".")[-1].lower()
    mime = MIME_MAP.get(ext, "audio/mpeg")

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")

        response = model.generate_content([
            {
                "mime_type": mime,
                "data": audio_bytes
            },
            "Transcribe this audio exactly. Return only transcript text."
        ])

        return response.text.strip()

    except Exception as e:
        raise Exception(f"Transcription Error: {str(e)}")


# ==========================================================
# STAGE 3 | CHUNKING
# ==========================================================
def split_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=400
    )

    return splitter.split_text(text)


# ==========================================================
# LOCAL EMBEDDING MODEL
# ==========================================================
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ==========================================================
# STAGE 4 | BUILD VECTOR STORE
# ==========================================================
def build_vector_store(chunks):

    embeddings = get_embeddings()

    db = FAISS.from_texts(chunks, embedding=embeddings)
    db.save_local(FAISS_PATH)


# ==========================================================
# LOAD VECTOR STORE
# ==========================================================
def load_vector_store():

    embeddings = get_embeddings()

    db = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


# ==========================================================
# STAGE 5 | RETRIEVE DOCS
# ==========================================================
def retrieve_docs(question):

    db = load_vector_store()
    docs = db.similarity_search(question, k=5)

    return docs


# ==========================================================
# STAGE 6 | GENERATE ANSWER
# ==========================================================
PROMPT = """
Answer only from given context.

Context:
{context}

Question:
{question}

Answer:
"""


def generate_answer(question, docs):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=API_KEY
    )

    prompt = PromptTemplate(
        template=PROMPT,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(
        llm,
        chain_type="stuff",
        prompt=prompt
    )

    result = chain(
        {
            "input_documents": docs,
            "question": question
        },
        return_only_outputs=True
    )

    return result["output_text"]


# ==========================================================
# STAGE 7 | TEXT TO AUDIO
# ==========================================================
def text_to_audio(text):

    buf = BytesIO()
    gTTS(text=text, lang="en").write_to_fp(buf)
    buf.seek(0)

    return buf


# ==========================================================
# UI  ←  ONLY THIS SECTION CHANGED
# ==========================================================
def main():

    st.set_page_config(
        page_title="AudioRAG",
        page_icon="🎙️",
        layout="wide"
    )

    # ── session state init ─────────────────────────────────
    if "ready" not in st.session_state:
        st.session_state.ready = False
    if "transcript" not in st.session_state:
        st.session_state.transcript = None
    if "history" not in st.session_state:
        st.session_state.history = []

    # ── global CSS ─────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Outfit:wght@300;400;500;600&display=swap');

    /* ── reset & base ── */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #080C14;
        color: #E8EAF0;
    }

    /* ── hide default streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

    /* ── hero banner ── */
    .hero {
        background: linear-gradient(135deg, #0D1B2A 0%, #0A1628 50%, #0D1421 100%);
        border: 1px solid #1E2D42;
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 260px; height: 260px;
        background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 30%;
        width: 180px; height: 180px;
        background: radial-gradient(circle, rgba(167,139,250,0.06) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #63B3ED, #A78BFA, #F687B3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.4rem 0;
        line-height: 1.1;
    }
    .hero-sub {
        color: #64748B;
        font-size: 0.95rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    .hero-sub span {
        color: #94A3B8;
        margin: 0 6px;
    }

    /* ── pipeline steps ── */
    .pipeline-row {
        display: flex;
        align-items: center;
        gap: 0;
        margin-bottom: 2rem;
        flex-wrap: nowrap;
        overflow-x: auto;
        padding-bottom: 4px;
    }
    .pipe-step {
        display: flex;
        align-items: center;
        gap: 6px;
        background: #0F1923;
        border: 1px solid #1E2D42;
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.75rem;
        font-weight: 500;
        color: #475569;
        white-space: nowrap;
        transition: all 0.3s ease;
        flex-shrink: 0;
    }
    .pipe-step.active {
        border-color: #63B3ED;
        color: #63B3ED;
        background: rgba(99,179,237,0.07);
    }
    .pipe-step.done {
        border-color: #34D399;
        color: #34D399;
        background: rgba(52,211,153,0.07);
    }
    .pipe-arrow {
        color: #1E2D42;
        font-size: 1rem;
        padding: 0 4px;
        flex-shrink: 0;
    }

    /* ── section cards ── */
    .card {
        background: #0C1520;
        border: 1px solid #1A2638;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }
    .card-title {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 1rem;
    }

    /* ── file uploader override ── */
    [data-testid="stFileUploader"] {
        background: #0C1520;
        border: 1.5px dashed #1E2D42;
        border-radius: 12px;
        padding: 1rem;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #2D4A6A;
    }

    /* ── buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #1D4ED8, #7C3AED) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.6rem !important;
        letter-spacing: 0.3px !important;
        transition: opacity 0.2s, transform 0.15s !important;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
        transform: translateY(-1px) !important;
    }

    /* ── transcript box ── */
    .transcript-box {
        background: #080C14;
        border: 1px solid #1A2638;
        border-left: 3px solid #63B3ED;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
        line-height: 1.8;
        color: #94A3B8;
        max-height: 260px;
        overflow-y: auto;
    }
    .transcript-box::-webkit-scrollbar { width: 4px; }
    .transcript-box::-webkit-scrollbar-track { background: transparent; }
    .transcript-box::-webkit-scrollbar-thumb { background: #1E2D42; border-radius: 4px; }

    /* ── stat chips ── */
    .stat-row {
        display: flex;
        gap: 10px;
        margin-top: 10px;
    }
    .stat-chip {
        background: #0F1923;
        border: 1px solid #1E2D42;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        color: #64748B;
    }
    .stat-chip b { color: #94A3B8; }

    /* ── Q&A chat bubbles ── */
    .bubble-q {
        background: linear-gradient(135deg, #1E3A5F, #1E2D4A);
        border-radius: 12px 12px 12px 4px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0 0.2rem;
        font-size: 0.9rem;
        color: #BAD4F0;
        border-left: 3px solid #63B3ED;
    }
    .bubble-a {
        background: linear-gradient(135deg, #1A3040, #14273A);
        border-radius: 12px 12px 4px 12px;
        padding: 0.75rem 1rem;
        margin: 0.2rem 0 0.5rem;
        font-size: 0.9rem;
        color: #A7C5A0;
        line-height: 1.7;
        border-left: 3px solid #34D399;
    }
    .bubble-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .lbl-q { color: #3B82F6; }
    .lbl-a { color: #34D399; }

    /* ── format badges ── */
    .fmt-badge {
        display: inline-block;
        background: #0F1923;
        border: 1px solid #1E2D42;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.7rem;
        color: #475569;
        margin: 2px;
        font-family: monospace;
    }

    /* ── text input ── */
    .stTextInput > div > div > input {
        background: #0C1520 !important;
        border: 1.5px solid #1A2638 !important;
        border-radius: 10px !important;
        color: #E2E8F0 !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.92rem !important;
        padding: 0.6rem 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #63B3ED !important;
        box-shadow: 0 0 0 2px rgba(99,179,237,0.1) !important;
    }

    /* ── success / error overrides ── */
    .stAlert {
        border-radius: 10px !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* ── audio player ── */
    audio {
        width: 100%;
        border-radius: 8px;
        margin-top: 4px;
    }

    /* ── divider ── */
    hr { border-color: #1A2638 !important; margin: 1.2rem 0 !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── hero ───────────────────────────────────────────────
    ready = st.session_state.ready

    st.markdown("""
    <div class="hero">
        <div class="hero-title">AudioRAG</div>
        <div class="hero-sub">
            Upload audio <span>→</span> Transcribe <span>→</span>
            Ask anything <span>→</span> Hear the answer
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── pipeline tracker ───────────────────────────────────
    steps = [
        ("🎙", "Upload"),
        ("📝", "Transcribe"),
        ("✂️", "Chunk"),
        ("🗄", "Embed"),
        ("🔍", "Retrieve"),
        ("🤖", "Generate"),
        ("🔊", "Speak"),
    ]
    html_steps = ""
    for i, (icon, label) in enumerate(steps):
        css = "done" if (ready and i <= 3) else "pipe-step"
        html_steps += f'<div class="pipe-step {css}">{icon} {label}</div>'
        if i < len(steps) - 1:
            html_steps += '<div class="pipe-arrow">›</div>'

    st.markdown(f'<div class="pipeline-row">{html_steps}</div>',
                unsafe_allow_html=True)

    # ── two-column layout ──────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    # ════════════════════════════════════════
    # LEFT  —  upload + transcript
    # ════════════════════════════════════════
    with col_left:

        st.markdown('<div class="card-title">01 — Audio Input</div>',
                    unsafe_allow_html=True)

        # format badges
        badges = "".join(
            f'<span class="fmt-badge">.{f}</span>'
            for f in SUPPORTED_FORMATS
        )
        st.markdown(badges, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop your audio file here",
            type=SUPPORTED_FORMATS,
            label_visibility="collapsed"
        )

        if uploaded:
            st.audio(uploaded)
            st.markdown(
                f'<div class="stat-row">'
                f'<span class="stat-chip">📄 <b>{uploaded.name}</b></span>'
                f'<span class="stat-chip">💾 <b>{round(uploaded.size/1024, 1)} KB</b></span>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("⚡  Run Pipeline  (Stages 1 – 4)"):
                prog = st.progress(0, text="Starting pipeline…")
                try:
                    prog.progress(15, text="🎙️  Stage 2 · Transcribing with Gemini 2.5 Flash…")
                    transcript = transcribe_audio(uploaded)

                    prog.progress(50, text="✂️  Stage 3 · Splitting into chunks…")
                    chunks = split_text(transcript)

                    prog.progress(75, text="🗄  Stage 4 · Building FAISS vector store…")
                    build_vector_store(chunks)

                    prog.progress(100, text="✅  Pipeline complete!")

                    st.session_state.ready = True
                    st.session_state.transcript = transcript
                    st.session_state.history = []

                    st.success(
                        f"Pipeline complete — {len(chunks)} chunks indexed."
                    )

                except Exception as e:
                    prog.empty()
                    st.error(str(e))

        # transcript display
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-title">02 — Transcript</div>',
                    unsafe_allow_html=True)

        if st.session_state.transcript:
            word_count = len(st.session_state.transcript.split())
            st.markdown(
                f'<div class="transcript-box">{st.session_state.transcript}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="stat-row">'
                f'<span class="stat-chip">🔤 <b>{word_count}</b> words</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="transcript-box" style="color:#2D3F55;font-style:italic">'
                'Transcript will appear here after processing…'
                '</div>',
                unsafe_allow_html=True
            )

        # reset
        if ready:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑  Reset Session"):
                st.session_state.ready = False
                st.session_state.transcript = None
                st.session_state.history = []
                st.rerun()

    # ════════════════════════════════════════
    # RIGHT  —  Q&A
    # ════════════════════════════════════════
    with col_right:

        st.markdown('<div class="card-title">03 — Ask a Question</div>',
                    unsafe_allow_html=True)

        if not ready:
            st.markdown(
                '<div style="background:#0C1520;border:1px dashed #1A2638;'
                'border-radius:12px;padding:2rem;text-align:center;color:#2D3F55;'
                'font-size:0.9rem">Process an audio file first to unlock Q&A</div>',
                unsafe_allow_html=True
            )
        else:
            with st.form("qa_form", clear_on_submit=True):
                question = st.text_input(
                    "question",
                    placeholder="What was discussed about…?",
                    label_visibility="collapsed"
                )
                submitted = st.form_submit_button("Ask  →  (Stages 5 – 7)")

            if submitted and question.strip():
                with st.spinner("🔍 Retrieving · Generating · Converting to speech…"):
                    try:
                        docs   = retrieve_docs(question)
                        answer = generate_answer(question, docs)
                        audio  = text_to_audio(answer)
                        st.session_state.history.append({
                            "q": question,
                            "a": answer,
                            "audio": audio
                        })
                    except Exception as e:
                        st.error(str(e))

            # chat history
            if st.session_state.history:
                st.markdown('<div class="card-title" style="margin-top:1.2rem">04 — Conversation</div>',
                            unsafe_allow_html=True)

                for item in reversed(st.session_state.history):
                    st.markdown(
                        f'<div class="bubble-label lbl-q">You</div>'
                        f'<div class="bubble-q">{item["q"]}</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<div class="bubble-label lbl-a">AudioRAG</div>'
                        f'<div class="bubble-a">{item["a"]}</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown("🔊 **Voice answer**")
                    st.audio(item["audio"])
                    st.markdown("<hr>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
