import streamlit as st
import dspy
from DSPyPipeline import RAG, load_gemini_model, PineconeRM
import json

# Load the compiled RAG model
def load_compiled_rag(complie_version_path='./v1.json'):
    compiled_rag = RAG()
    compiled_rag.load(complie_version_path)
    return compiled_rag

# Initialize models
gemini_flash = load_gemini_model()
pinecone_retriever = PineconeRM
dspy.settings.configure(lm=gemini_flash, rm=pinecone_retriever)

original_rag = RAG()
compiled_rag_v1 = load_compiled_rag()
compiled_rag_v2 = load_compiled_rag("./v2.json")

def comparison_page_logic():
    st.title("Pipeline Comparison")

    # Query input form
    query = st.text_input("Enter your query:")
    submit_button = st.button("Submit")

    if submit_button and query:
        # Original RAG Pipeline
        with st.expander("Original Pipeline", expanded=True):
            original_prediction = original_rag(query)
            st.write("Answer:", original_prediction.answer)
            st.write("Gemini History:")
            st.code(gemini_flash.inspect_history(n=1), language="json")

        # Compiled RAG Pipeline 1
        with st.expander("Compiled Pipeline V1", expanded=False):
            compiled_prediction = compiled_rag_v1(query)
            st.write("Answer:", compiled_prediction.answer)
            st.write("Gemini History:")
            st.code(gemini_flash.inspect_history(n=1), language="json")

        # Compiled RAG Pipeline 2
        with st.expander("Compiled Pipeline V2", expanded=False):
            compiled_prediction = compiled_rag_v2(query)
            st.write("Answer:", compiled_prediction.answer)
            st.write("Gemini History:")
            st.code(gemini_flash.inspect_history(n=1), language="json")
