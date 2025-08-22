import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import json
import gdown
import os
import google.generativeai as genai

# ------------------- Config -------------------
TOP_K_RETRIEVE = 50
FINAL_RESULTS = 10
SYNONYMS = {
    "wireless": ["inductive", "contactless"],
    "charging": ["power transfer", "energy transfer"],
}

# Gemini API key setup
genai.configure(api_key="AIzaSyB1LiX61P3nlDYPidUYyUM3yPx6Tvjtd_M")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Google Drive file IDs
FILE_IDS = {
    "csv": "1Asg94OHDh7iuqT58pqJWMwasTjL1OeK9",
    "json":"1DYpxrBlIPzv90R83JU5EWO7GH_mbiL3r",
    "npy": "1OXEsD8JAP83VAM9p3tUX4MiY6Lhu5UBu"
}

@st.cache_data(show_spinner=True)
def download_file(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)
    return output

@st.cache_data(show_spinner=True)
def load_data():
    csv_path = download_file(FILE_IDS["csv"], "patent_data.csv")
    json_path = download_file(FILE_IDS["json"], "combined_texts.json")
    npy_path = download_file(FILE_IDS["npy"], "patent_embeddings_mpnet.npy")

    df = pd.read_csv(csv_path)
    with open(json_path, "r", encoding="utf-8") as f:
        combined_texts = json.load(f)
    embeddings = np.load(npy_path, allow_pickle=True)
    return df, combined_texts, embeddings

df, combined_texts, embeddings = load_data()

# FAISS setup
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)

# Models
embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device="cpu")

# ------------------- Helper functions -------------------
def clean_text(text):
    return re.sub('<.*?>', '', str(text))

def expand_query_with_synonyms(query):
    tokens = query.lower().split()
    expanded_tokens = []
    for token in tokens:
        expanded_tokens.append(token)
        if token in SYNONYMS:
            expanded_tokens.extend(SYNONYMS[token])
    return " ".join(expanded_tokens)

def search(query, top_k=TOP_K_RETRIEVE):
    expanded_query = expand_query_with_synonyms(query)
    query_embedding = embedder.encode([expanded_query], normalize_embeddings=True)
    D, I = index.search(query_embedding, top_k)

    candidates = []
    for i, idx in enumerate(I[0]):
        row = df.iloc[idx]
        combined_text = combined_texts[idx]
        candidates.append({
            "idx": idx,
            "text": combined_text,
            "metadata": row,
            "faiss_score": D[0][i]
        })

    cross_inp = [(query, c['text']) for c in candidates]
    rerank_scores = reranker.predict(cross_inp)
    for c, score in zip(candidates, rerank_scores):
        c['rerank_score'] = score

    candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

    results = []
    for i, c in enumerate(candidates[:FINAL_RESULTS]):
        abstract = clean_text(c['metadata'].get('abstract', ''))
        sentences = re.split(r'(?<=[.!?]) +', abstract)
        if sentences:
            sentence_embeddings = embedder.encode(sentences, normalize_embeddings=True)
            q_emb = embedder.encode([query], normalize_embeddings=True)
            sims = util.cos_sim(sentence_embeddings, q_emb).squeeze(1)
            best_idx = int(sims.argmax())
            best_sentence = sentences[best_idx]
        else:
            best_sentence = ""

        results.append({
            "index": i + 1,
            "similarity": c['faiss_score'] * 100,
            "rerank_score": c['rerank_score'],
            "most_similar_sentence": best_sentence,
            "title": clean_text(c['metadata'].get('title', '')),
            "abstract": abstract,
            "patent_number": c['metadata'].get('patent_number', ''),
            "publication_date": c['metadata'].get('publication_date', ''),
            "application_number": c['metadata'].get('application_number', ''),
            "inventors": c['metadata'].get('inventors', ''),
            "assignee": c['metadata'].get('assignee', '')
        })
    return results

# ------------------- Gemini RAG: Best Solution -------------------
def generate_best_solution(query, results):
    # Use abstracts + patent numbers to let Gemini select best matching solution
    context_text = "\n\n".join([
        f"Patent {r['patent_number']}: {r['abstract']}" for r in results
    ])
    prompt = f"""
    You are a patent research assistant.
    Based on the following patents, identify the best solution that answers the user's question.
    Include the supporting patent number(s) in your answer.
    If no patent matches, say "I don't know".

    Context:
    {context_text}

    Question:
    {query}

    Best Solution:
    """
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# ------------------- Streamlit UI -------------------
st.set_page_config(layout="wide")
st.title("🔍 Semantic Patent Search + Gemini Best Solution")

query_col, icon_col = st.columns([9, 1])
with query_col:
    query = st.text_input("Search", placeholder="Enter your search query here...", label_visibility="collapsed")
with icon_col:
    st.write("")
    search_triggered = st.button("🔍")

if query or search_triggered:
    with st.spinner("Searching..."):
        all_results = search(query)

    with st.spinner("Generating best solution using Gemini..."):
        best_solution = generate_best_solution(query, all_results)

    st.subheader("💡 Best Solution (AI)")
    st.markdown(best_solution)

    st.markdown(f"**Top {FINAL_RESULTS} results shown (from {TOP_K_RETRIEVE} retrieved)**")
    for result in all_results:
        st.markdown(f"**Why this result?**\n"
                    f"→ FAISS Similarity Score: {result['similarity']:.2f}%\n"
                    f"→ Rerank Score: {result['rerank_score']:.4f}\n"
                    f"→ Most Relevant Sentence: “{result['most_similar_sentence']}”")
        st.markdown(f"### {result['index']}. {result['title']}")
        st.markdown(f"**Abstract:** {result['abstract']}")
        st.markdown(f"""
        - **Patent Number**: {result['patent_number']}  
        - **Publication Date**: {result['publication_date']}  
        - **Application Number**: {result['application_number']}  
        - **Inventors**: {result['inventors']}  
        - **Assignee**: {result['assignee']}
        """)
        st.markdown("---")
