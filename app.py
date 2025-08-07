import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re

# ------------------- Load data and model -------------------
df = pd.read_csv("patent_data.csv")
embeddings = np.load("embeddings.npy")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------- Helper Functions -------------------
def clean_text(text):
    return re.sub('<.*?>', '', str(text))

def search(query, top_k=50):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, top_k)
    results = []

    for i, idx in enumerate(I[0]):
        row = df.iloc[idx]
        similarity_score = (1 - D[0][i] / 4) * 100
        similarity_score = max(0, min(100, similarity_score))

        sentences = re.split(r'(?<=[.!?]) +', clean_text(row['abstract']))
        most_similar = ""
        max_sim = -1
        for sent in sentences:
            sent_vec = model.encode([sent])
            sim = np.dot(sent_vec, query_embedding.T).flatten()[0]
            if sim > max_sim:
                max_sim = sim
                most_similar = sent

        results.append({
            "index": i + 1,
            "similarity": similarity_score,
            "most_similar_sentence": most_similar,
            "title": clean_text(row.get('title', '')),
            "abstract": clean_text(row.get('abstract', '')),
            "patent_number": row.get('patent_number', ''),
            "publication_date": row.get('publication_date', ''),
            "application_number": row.get('application_number', ''),
            "inventors": row.get('inventors', ''),
            "assignee": row.get('assignee', '')
        })

    return results

# ------------------- Streamlit UI -------------------
st.set_page_config(layout="wide")
st.title("🔍 Semantic Patent Search")

# Clean single-line query input with icon
query_col, icon_col = st.columns([9, 1])
with query_col:
    query = st.text_input("", placeholder="Enter your search query here...")
with icon_col:
    st.write("")  # vertical space
    search_triggered = st.button("🔍")

# ------------------- Search and Display -------------------
if query or search_triggered:
    with st.spinner("Searching..."):
        all_results = search(query)

    # Pagination Setup
    items_per_page = 10
    total_pages = (len(all_results) + items_per_page - 1) // items_per_page
    page = st.session_state.get("page", 1)

    # Paginated results
    start = (page - 1) * items_per_page
    end = start + items_per_page
    paginated_results = all_results[start:end]

    for result in paginated_results:
        # Why this result FIRST
        st.markdown(f"**Why this result?**  \n"
                    f"→ Similarity Score: `{result['similarity']:.2f}%`  \n"
                    f"→ Most Relevant Sentence: “{result['most_similar_sentence']}”")

        # Title
        st.markdown(f"### {result['index']}. {result['title']}")

        # Abstract
        st.markdown(f"**Abstract:** {result['abstract']}")

        # Metadata block
        st.markdown(f"""
        - **Patent Number**: {result['patent_number']}  
        - **Publication Date**: {result['publication_date']}  
        - **Application Number**: {result['application_number']}  
        - **Inventors**: {result['inventors']}  
        - **Assignee**: {result['assignee']}
        """)

        st.markdown("---")

    # ------------------- Pagination Controls at Bottom -------------------
    st.markdown("### 📄 Page Navigation")
    page = st.number_input("Select Page", min_value=1, max_value=total_pages, value=page, step=1, key="page")
