import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# ------------------- NLTK Setup -------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ------------------- Load Data -------------------
df = pd.read_csv("patent_data.csv")
embeddings = np.load("embeddings.npy")

# ------------------- FAISS Index -------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ------------------- Load SentenceTransformer Model -------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ------------------- Keyword Extraction -------------------
def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return set([word for word in words if word not in stop_words and len(word) > 2])

# ------------------- Streamlit UI -------------------
st.title("🔎 Patent Semantic Search Engine")
st.write("Search patents by meaning using AI-powered semantic search.")

sort_option = st.selectbox(
    "Sort results by:",
    ["Semantic Score", "Publication Date (newest)", "Abstract Length"],
    key="sort_selectbox"
)

query = st.text_input("Enter your search query:")
top_k = st.slider("Number of results", 1, 10, 3)

# ------------------- Search Logic -------------------
if st.button("Search") and query:
    clean_query = query.strip().lower()
    query_embedding = model.encode([clean_query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    query_keywords = extract_keywords(clean_query)
    results = []

    for idx, dist in zip(indices[0], distances[0]):
        row = df.iloc[idx].copy()
        row['semantic_score'] = dist
        row['abstract_length'] = len(str(row.get("abstract", "")))

        matched_keywords = extract_keywords(
            str(row.get("title", "")) + " " + str(row.get("abstract", ""))
        ).intersection(query_keywords)

        row['match_keywords'] = list(matched_keywords)
        results.append(row)

    result_df = pd.DataFrame(results)

    # ------------------- Sorting Logic -------------------
    if sort_option == "Semantic Score":
        result_df = result_df.sort_values("semantic_score", ascending=True)
    elif sort_option == "Publication Date (newest)":
        result_df['publication_date'] = pd.to_datetime(result_df['publication_date'], errors='coerce')
        result_df = result_df.sort_values("publication_date", ascending=False)
    elif sort_option == "Abstract Length":
        result_df = result_df.sort_values("abstract_length", ascending=False)

    # ------------------- Display Results -------------------
    st.markdown(f"### 🔍 Top {top_k} Results for: `{query}`")
    for _, row in result_df.iterrows():
        st.markdown("----")

        # ------------------- Explanation FIRST -------------------
        explanation = f"""
        **Why this result?**  
        → Semantic similarity score: **{row['semantic_score']:.4f}**  
        → Matched keywords: _{', '.join(row['match_keywords']) if row['match_keywords'] else 'None'}_  
        → This result was selected because it shares key concepts with your query.
        """
        st.markdown(explanation)

        # ------------------- Then Title and Patent Info -------------------
        st.markdown(f"### {row.get('title', '')}")
        st.write(f"**Patent Number:** {row.get('patent_number', 'N/A')}")
        st.write(f"**Publication Date:** {row.get('publication_date', 'N/A')}")
        st.write(f"**Inventors:** {row.get('inventors', 'N/A')}")
        st.write(f"**Assignees:** {row.get('assignees', 'N/A')}")

        st.markdown("**Abstract:**")
        st.markdown(row.get("abstract", "N/A"))
