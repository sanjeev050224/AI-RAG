# AudioRAG — Listen & Ask

End-to-end RAG pipeline: audio in → voice answer out.

## 7-Stage Pipeline

| Stage | Step | What happens |
|-------|------|-------------|
| 1 | Audio upload | User uploads an audio file (mp3, wav, m4a, ogg, flac, aac, wma, webm) |
| 2 | Transcription | Gemini 1.5 Flash transcribes audio natively — no external STT needed |
| 3 | Chunking | RecursiveCharacterTextSplitter splits text into 3000-char overlapping chunks |
| 4 | Embedding + Index | Google Embedding-001 vectorises chunks; FAISS stores them on disk |
| 5 | Retrieval | User question is embedded; FAISS returns top-5 semantically similar chunks |
| 6 | Generation | Gemini 1.5 Pro generates a grounded answer from context + question |
| 7 | Speech | gTTS converts the answer to MP3; Streamlit plays it inline |

## Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Create .env
echo "GOOGLE_API_KEY=your_key_here" > .env

# 3. Run
streamlit run app.py
```

## Supported Audio Formats

| Format | MIME type | Notes |
|--------|-----------|-------|
| .mp3 | audio/mpeg | Most common |
| .wav | audio/wav | Uncompressed, best quality |
| .m4a | audio/mp4 | Apple default (voice memos) |
| .ogg | audio/ogg | Open source |
| .flac | audio/flac | Lossless |
| .aac | audio/aac | Compressed, good quality |
| .wma | audio/x-ms-wma | Windows Media |
| .webm | audio/webm | Browser-recorded audio |

## Project Structure

```
audiorag/
├── app.py             # Main Streamlit app (all 7 pipeline stages)
├── requirements.txt   # Python dependencies
├── .env               # GOOGLE_API_KEY (not committed)
└── faiss_index/       # Auto-generated vector store (not committed)
```

## Key Design Decisions

- **Gemini 1.5 Flash** for transcription — multimodal, handles all 8 formats natively
- **Gemini 1.5 Pro** for generation — lower temperature (0.2) for factual grounding
- **chunk_size=3000** — avoids Gemini token limit errors vs the common 10000 default  
- **allow_dangerous_deserialization=True** — required by newer LangChain for FAISS load
- **Session state** — chat history persists; no re-upload needed between questions
- **BytesIO audio** — TTS stored in memory, no temp files on disk
