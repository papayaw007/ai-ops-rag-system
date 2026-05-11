import streamlit as st
import requests

st.set_page_config(page_title="AI Ops RAG System")

st.title("📄 AI Ops RAG Assistant")

# ---------- Upload PDF ----------
uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/pdf"
        )
    }

    with st.spinner("Processing PDF..."):

        response = requests.post(
            "http://127.0.0.1:8000/upload",
            files=files
        )

    if response.status_code == 200:

        data = response.json()

        st.success(
            f"Processed {data['chunks_created']} chunks"
        )

    else:
        st.error("Failed to process PDF")


# ---------- Ask Question ----------
question = st.text_input("Ask a question")

if question:

    with st.spinner("Thinking..."):

        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"question": question}
        )

    if response.status_code == 200:

        data = response.json()

        st.subheader("Answer")
        st.write(data["answer"])

        st.subheader("Retrieved Sources")

        for source in data["sources"]:
            st.code(source)

    else:
        st.error("Error getting answer")