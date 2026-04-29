"""
shortener/app.py
────────────────
Comprime un articolo web in 3 step prima di inviarlo a un LLM a pagamento:
  Step 1 — TF-IDF   : rimuove i paragrafi meno rilevanti (locale, gratuito)
  Step 2 — TextRank : estrae le frasi chiave con ranking estrattivo (locale, gratuito)
  Step 3 — LLM      : sintesi astrattiva finale via llm_client
"""

import re
from urllib.parse import urlparse

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from llm_client import render_provider_selector, generate

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Shortener", page_icon="✂️", layout="centered")
render_provider_selector()
st.title("✂️ Shortener")
st.caption("Comprime un articolo web in 3 step prima di inviarlo all'LLM.")


# ── Caricamento modelli NLP (cached) ──────────────────────────────────────────
@st.cache_resource(show_spinner="Caricamento modelli NLP (solo al primo avvio)…")
def load_nlp():
    import spacy
    import pytextrank  # noqa: F401 — registra la pipe "textrank"
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")
    return nlp


# ── Helpers ───────────────────────────────────────────────────────────────────
def _slug_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    slug = path.split("/")[-1] if path else "output"
    slug = re.sub(r"[^\w-]", "", slug)
    return slug or "output"


def _filename_from_url(url: str) -> str:
    return _slug_from_url(url)[:6].rstrip("-") + ".txt"


# ── Pipeline ──────────────────────────────────────────────────────────────────
def pipeline_compressione(url: str) -> tuple[str, str]:
    nlp = load_nlp()

    # ── STEP 0: scraping ──────────────────────────────────────────────────────
    with st.status("Step 0 — Scraping…", expanded=True):
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
    with st.status("Step 1 — TF-IDF (rimozione paragrafi irrilevanti)…", expanded=True):
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(paragraphs)
        sums = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
        threshold = np.percentile(sums, 10)
        filtered = [paragraphs[i] for i, s in enumerate(sums) if s >= threshold]
        text_filtered = " ".join(filtered)
        reduction_1 = (1 - len(text_filtered) / len(full_text)) * 100
        st.write(f"{len(filtered)} paragrafi mantenuti — riduzione **{reduction_1:.0f}%**")

    # ── STEP 2: TextRank ─────────────────────────────────────────────────────
    with st.status("Step 2 — TextRank (estrazione frasi chiave)…", expanded=True):
        doc = nlp(text_filtered[:100_000])
        ranked = [sent.text for sent in doc._.textrank.summary(limit_sentences=30)]
        text_ranked = " ".join(ranked)
        reduction_2 = (1 - len(text_ranked) / len(full_text)) * 100
        st.write(f"{len(ranked)} frasi estratte — riduzione **{reduction_2:.0f}%** — "
                 f"**{len(text_ranked):,}** caratteri inviati all'LLM")

    # ── STEP 3: sintesi astrattiva via LLM ───────────────────────────────────
    with st.status("Step 3 — Sintesi astrattiva (LLM)…", expanded=True):
        final = generate(
            prompt=f"Riassumi questo testo in modo chiaro e completo:\n\n{text_ranked}",
            system="Sei un assistente esperto nella sintesi di articoli. Rispondi in italiano.",
            max_tokens=2048,
            temperature=0.3,
        )
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
        st.session_state["risultato"] = testo
        st.session_state["fname"] = fname
    except requests.exceptions.RequestException as e:
        st.error(f"Errore di rete: {e}")
    except Exception as e:
        st.exception(e)

# Mostra il risultato persistente (sopravvive ai rerun)
if "risultato" in st.session_state:
    st.divider()
    st.subheader("📄 Testo compresso (pronto per l'LLM)")
    st.text_area("Testo compresso", value=st.session_state["risultato"], height=300, label_visibility="collapsed")
    st.download_button(
        label=f"⬇️ Scarica {st.session_state['fname']}",
        data=st.session_state["risultato"],
        file_name=st.session_state["fname"],
        mime="text/plain",
    )
