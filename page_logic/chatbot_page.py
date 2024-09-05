import streamlit as st
from DSPyPipeline import RAG
import time

@st.cache_resource
def load_compiled_rag(compile_version_path='./v1.json'):
    compiled_rag = RAG()
    compiled_rag.load(compile_version_path)
    return compiled_rag

def chatbot_page_logic():
    # Initialize RAG models
    original_rag = RAG()
    compiled_rag_v1 = load_compiled_rag()
    compiled_rag_v2 = load_compiled_rag("./v2.json")
    
    pipelines = {
        "Original": original_rag,
        "V1": compiled_rag_v1,
        "V2": compiled_rag_v2
    }

    pipeline_icons = {
        "Original": "👋",
        "V1": "🧑‍💻",
        "V2": "🦖"
    }

    # Main chat interface
    st.title("Multi-Pipeline Chat Interface")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"Welcome! I'm the {version} pipeline. How can I assist you today?", "version": version, "avatar": pipeline_icons[version]}
            for version in pipelines.keys()
        ]

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Enter your query:")

    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": None})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate responses from all pipeline versions
        context = None
        for version, rag in pipelines.items():
            with st.chat_message("assistant", avatar=pipeline_icons[version]):
                message_placeholder = st.empty()
                full_response = ""
                
                # Get prediction from RAG model
                prediction = rag(question=prompt)
                
                for chunk in prediction.answer.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                
                # Display final response
                message_placeholder.markdown(full_response)
                
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "version": version,
                "avatar": pipeline_icons[version]
            })

            # Store context (it's the same for all pipelines)
            if context is None:
                context = prediction.context

        # Update context in sidebar (only once, as it's the same for all pipelines)
        st.sidebar.empty()
        st.sidebar.subheader("Retrieved Context")
        for i, chunk in enumerate(context, 1):
            with st.sidebar.expander(f"Context Chunk {i}"):
                st.write(chunk)





