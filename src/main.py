import streamlit as st
from query import ask
import time
import os

model_name = os.getenv("MODEL_NAME", "codellama:7b")
index_model_name = os.getenv("INDEX_MODEL_NAME", "all-minilm:l6-v2")

st.set_page_config(page_title="AI Code Assistant", layout="wide")
st.title("🤖 Local AI code assistant")

query = st.text_area("Request:", height=200)

if st.button("🔍 Ask") and query.strip():
    with st.spinner("💡 Processing..."):
        st.markdown(f"### ⛑️  AI: `{model_name}` - `{index_model_name}`")

        start_time = time.time();

        result = ask(query)

        duration = time.time() - start_time;

        st.markdown("### ✅ Answer:")
        st.write(result)
        st.markdown(f"⏱️  **Time taken:** `{duration:.2f}` seconds")