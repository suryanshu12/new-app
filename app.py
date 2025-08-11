import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import json

# Config
TOP_K_RETRIEVE = 200  # Number of candidates retrieved from FAISS
MAX_DISPLAY_RESULTS = 10  # Show only top 10 results
SYNONYMS = {
    "wireless": ["inductive", "contactless"],
    "charging": ["power transfer", "energy transfer"],
    # Add more synonyms as needed
}

# Load data and models
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv("patent_data.csv")
    with open("combined_texts.json", "r", encoding="utf-8") as f:
        combined_texts = json.load(f)
    embeddings = np.load("patent_embeddings.npy")
    return df, combined_texts, embeddings

df, combined_texts, embeddings = load_data()

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)

embedder = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

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

    # Sort by rerank score descending
    candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

    # Take only top MAX_DISPLAY_RESULTS
    candidates = candidates[:MAX_DISPLAY_RESULTS]

    results = []
    for i, c in enumerate(candidates):
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

st.set_page_config(layout="wide")
st.title("🔍 Semantic Patent Search - Top Results Only")

query_col, icon_col = st.columns([9, 1])
with query_col:
    query = st.text_input("", placeholder="Enter your search query here...")
with icon_col:
    st.write("")
    search_triggered = st.button("🔍")

if query or search_triggered:
    with st.spinner("Searching..."):
        all_results = search(query)

    total_results = len(all_results)
    st.markdown(f"**Showing top {total_results} most relevant results**")

    for result in all_results:
        st.markdown(f"**Why this result?**  \n"
                    f"→ FAISS Similarity Score: `{result['similarity']:.2f}%`  \n"
                    f"→ Rerank Score: `{result['rerank_score']:.4f}`  \n"
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
