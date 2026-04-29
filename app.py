"""
shortener/app.py
────────────────
Comprime un articolo web in 3 step prima di inviarlo a un LLM a pagamento:
  Step 1 — TF-IDF  : rimuove i paragrafi meno rilevanti
  Step 2 — TextRank: estrae le frasi chiave con ranking estrattivo
  Step 3 — BART    : sintesi astrattiva con map-reduce su chunk
"""

import re
from urllib.parse import urlparse

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Shortener", page_icon="✂️", layout="centered")
st.title("✂️ Shortener")
st.caption("Comprime un articolo web in 3 step prima di inviarlo all'LLM.")


# ── Caricamento modelli (cached) ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Caricamento modelli NLP (solo al primo avvio)…")
def load_models():
    import spacy
    import pytextrank  # noqa: F401 — registra la pipe "textrank"
    from transformers import pipeline as hf_pipeline

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")
    summarizer = hf_pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=-1,
    )
    return nlp, summarizer


# ── Helpers ───────────────────────────────────────────────────────────────────
def _slug_from_url(url: str) -> str:
    """Restituisce il nome della pagina (ultimo segmento del path) senza dominio."""
    path = urlparse(url).path.rstrip("/")
    slug = path.split("/")[-1] if path else "output"
    slug = re.sub(r"[^\w-]", "", slug)          # rimuove caratteri speciali
    return slug or "output"


def _filename_from_url(url: str) -> str:
    """Primi 6 caratteri della slug + .txt"""
    slug = _slug_from_url(url)
    return slug[:6].rstrip("-") + ".txt"


def _chunk_text(text: str, max_chars: int = 3000) -> list[str]:
    """Divide il testo in chunk rispettando i confini delle frasi."""
    sentences = text.split(". ")
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) < max_chars:
            current += sent + ". "
        else:
            if current:
                chunks.append(current.strip())
            current = sent + ". "
    if current:
        chunks.append(current.strip())
    return chunks


# ── Pipeline ──────────────────────────────────────────────────────────────────
def pipeline_compressione(url: str) -> tuple[str, str]:
    """
    Ritorna (testo_compresso, nome_file).
    """
    nlp, summarizer = load_models()

    # ── STEP 0: scraping ──────────────────────────────────────────────────────
    with st.status("Step 0 — Scraping…"):
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p") if len(p.get_text()) > 40]
        if not paragraphs:
            st.error("Nessun paragrafo trovato nella pagina.")
            st.stop()
        full_text = "\n".join(paragraphs)
        st.write(f"Testo originale: **{len(full_text):,}** caratteri — **{len(paragraphs)}** paragrafi")

    # ── STEP 1: TF-IDF ───────────────────────────────────────────────────────
    with st.status("Step 1 — TF-IDF (rimozione paragrafi irrilevanti)…"):
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(paragraphs)
        sums = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
        threshold = np.percentile(sums, 10)
        filtered = [paragraphs[i] for i, s in enumerate(sums) if s >= threshold]
        text_filtered = " ".join(filtered)
        reduction_1 = (1 - len(text_filtered) / len(full_text)) * 100
        st.write(f"{len(filtered)} paragrafi mantenuti — riduzione **{reduction_1:.0f}%**")

    # ── STEP 2: TextRank ─────────────────────────────────────────────────────
    with st.status("Step 2 — TextRank (estrazione frasi chiave)…"):
        doc = nlp(text_filtered[:100_000])  # limite spacy
        ranked = [sent.text for sent in doc._.textrank.summary(limit_sentences=30)]
        text_ranked = " ".join(ranked)
        reduction_2 = (1 - len(text_ranked) / len(full_text)) * 100
        st.write(f"{len(ranked)} frasi estratte — riduzione **{reduction_2:.0f}%**")

    # ── STEP 3: BART map-reduce ───────────────────────────────────────────────
    with st.status("Step 3 — Sintesi astrattiva BART…"):
        chunks = _chunk_text(text_ranked, max_chars=3000)
        st.write(f"Testo diviso in **{len(chunks)}** chunk…")

        partials = []
        prog = st.progress(0)
        for i, chunk in enumerate(chunks):
            result = summarizer(chunk, max_length=200, min_length=80, do_sample=False)
            partials.append(result[0]["summary_text"])
            prog.progress((i + 1) / len(chunks))

        combined = " ".join(partials)
        if len(combined) > 3000:
            st.write("Sintesi delle sintesi parziali…")
            result = summarizer(combined[:3000], max_length=300, min_length=100, do_sample=False)
            final = result[0]["summary_text"]
        else:
            final = combined

        reduction_final = (1 - len(final) / len(full_text)) * 100
        st.write(f"Testo finale: **{len(final):,}** caratteri — riduzione totale **{reduction_final:.0f}%**")

    return final, _filename_from_url(url)


# ── UI ────────────────────────────────────────────────────────────────────────
url = st.text_input(
    "URL articolo da comprimere",
    placeholder="https://example.com/articolo-interessante/",
)

if st.button("✂️ Comprimi", type="primary", disabled=not url):
    try:
        testo, fname = pipeline_compressione(url.strip())

        st.divider()
        st.subheader("📄 Testo compresso (pronto per l'LLM)")
        st.text_area("", value=testo, height=300, label_visibility="collapsed")

        st.download_button(
            label=f"⬇️ Scarica {fname}",
            data=testo,
            file_name=fname,
            mime="text/plain",
        )

    except requests.exceptions.RequestException as e:
        st.error(f"Errore di rete: {e}")
    except Exception as e:
        st.exception(e)
