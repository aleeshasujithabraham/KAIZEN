
import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from document_manager import DocumentManager
from tutor import TutorAgent


st.set_page_config(
    page_title="KAIZEN Tutor",
    page_icon="📚",
    layout="centered"
)

st.title("📚 KAIZEN")
st.subheader("AI-Powered Personalized Learning")
st.caption("Learn from your own study notes with Gemini AI.")

# Store information between Streamlit reruns
if "tutor" not in st.session_state:
    st.session_state.tutor = None

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "filename" not in st.session_state:
    st.session_state.filename = None


st.sidebar.header("Study Material")

uploaded_file = st.sidebar.file_uploader(
    "Upload your PDF notes",
    type=["pdf"],
    max_upload_size=20
)

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if file_hash != st.session_state.document_hash:

        if st.sidebar.button(
            "Process Document",
            type="primary"
        ):
            temp_path = None

            try:
                with st.spinner(
                    "Processing notes and generating embeddings..."
                ):
                    # Save uploaded PDF temporarily
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf",
                        delete=False
                    ) as temp_file:
                        temp_file.write(pdf_bytes)
                        temp_path = Path(temp_file.name)

                    manager = DocumentManager()

                    result = manager.ingest_document(
                        temp_path
                    )

                    tutor = TutorAgent(
                        document_id=result["document_id"]
                    )

                    st.session_state.tutor = tutor
                    st.session_state.document_hash = file_hash
                    st.session_state.filename = uploaded_file.name
                    st.session_state.messages = []

                st.sidebar.success(
                    "Document processed successfully!"
                )

                st.rerun()

            except Exception as error:
                st.error(
                    "Unable to process the PDF. "
                    f"Error type: {type(error).__name__}"
                )

            finally:
                if temp_path and temp_path.exists():
                    temp_path.unlink()

    else:
        st.sidebar.success("Document ready!")

if st.session_state.tutor is not None:

    st.success(
        f"Studying: {st.session_state.filename}"
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask something about your notes..."
    )

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):
                try:
                    response = (
                        st.session_state.tutor.ask(question)
                    )

                    answer = response["answer"]

                    if response["sources"]:
                        answer += "\n\n**Sources:**\n"

                        for source in response["sources"]:
                            answer += (
                                f"- Page {source['page']}, "
                                f"chunk {source['chunk_id']}\n"
                            )

                except Exception:
                    answer = (
                        "Something went wrong while generating "
                        "the response. Please try again."
                    )

            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:
    st.info(
        "Upload and process a PDF to start learning."
    )
