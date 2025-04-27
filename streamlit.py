import streamlit as st
from src.app_core import TechnicalDocAssistant, init_session_state  # Assume we modularized your code

# Initialize session state
init_session_state()

# Set page config
st.set_page_config(
    page_title="Technical Documentation Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("Generate High-Quality Technical Documentation with AI")

# Initialize the assistant
if 'assistant' not in st.session_state:
    with st.spinner("Initializing Documentation Assistant..."):
        st.session_state.assistant = TechnicalDocAssistant()

assistant = st.session_state.assistant

# Create the four main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Documentation Generator", 
    "About Project", 
    "Documentation", 
    "GitHub & Setup"
])

# --- TAB 1: Documentation Generator ---
with tab1:
    st.header("Documentation Generator")

    auto_classify = st.checkbox("Auto-classify document type", value=False, disabled=st.session_state.is_fallback_mode)

    if auto_classify and not st.session_state.is_fallback_mode:
        input_text = st.text_area("Enter text to classify document type", height=150)
        if st.button("Classify") and input_text:
            with st.spinner("Classifying..."):
                doc_type, confidence = assistant.classify_document_type(input_text)
                st.success(f"Classified as: {doc_type} ({confidence:.2f}% confidence)")
                st.session_state.classified_type = doc_type

    available_prompts = assistant.prompts.list_available_prompts()

    selected_doc_type = st.session_state.classified_type if auto_classify else None

    doc_type = selected_doc_type or st.selectbox("Select document type", available_prompts, index=2)

    # Inputs depending on document type
    kwargs = {}
    if doc_type == "api_reference":
        kwargs['language'] = st.selectbox("Programming Language", ["python", "javascript", "java", "c++", "go"])
        kwargs['code'] = st.text_area("Paste your code")
        kwargs['code_type'] = st.selectbox("Code Type", ["function", "class", "method"])
        kwargs['num_examples'] = st.slider("Number of Examples", 1, 5, 2)
    elif doc_type == "tutorial":
        kwargs['topic'] = st.text_input("Tutorial Topic")
        kwargs['audience'] = st.selectbox("Audience", ["beginners", "intermediate users", "advanced users"])
        kwargs['num_steps'] = st.slider("Number of Steps", 3, 10, 5)
        kwargs['num_challenges'] = st.slider("Number of Challenges", 1, 5, 3)
    elif doc_type == "concept_explanation":
        kwargs['concept'] = st.text_input("Concept to Explain")
        kwargs['expertise_level'] = st.selectbox("Expertise Level", ["beginner", "intermediate", "advanced"])
        kwargs['num_use_cases'] = st.slider("Number of Use Cases", 1, 5, 3)
    elif doc_type == "troubleshooting":
        kwargs['technology'] = st.text_input("Technology")
        kwargs['issue'] = st.text_input("Issue")
        kwargs['num_causes'] = st.slider("Number of Causes", 1, 5, 3)
        kwargs['num_solutions'] = st.slider("Number of Solutions", 1, 5, 3)

    if st.button("Generate Documentation"):
        if not kwargs:
            st.error("Please fill in the required fields.")
        else:
            with st.spinner("Generating documentation..."):
                documentation = assistant.generate_documentation(doc_type, **kwargs)
                st.session_state.last_generated_doc = documentation

    if st.session_state.last_generated_doc:
        st.subheader("Generated Documentation")
        st.markdown(st.session_state.last_generated_doc)

# --- TAB 2: About Project ---
with tab2:
    st.header("About Technical Documentation Assistant")
    st.markdown("""
    - **Problem**: Creating consistent and high-quality documentation is tedious.
    - **Solution**: AI-powered generator for API references, tutorials, explanations, and troubleshooting.
    - **Tech Stack**: Streamlit, Transformers, PyTorch, Hugging Face Datasets.
    - **Local Models**: Privacy-friendly, no external API calls.
    """)

# --- TAB 3: Documentation ---
with tab3:
    st.header("Project Technical Documentation")
    st.subheader("System Architecture")
    st.code("""
    ┌───────────────────┐    ┌─────────────────────┐    ┌──────────────────┐
    │                   │    │                     │    │                  │
    │ User Interface    │◄───►  Core Processing    │◄───►  Model Layer     │
    │ (Streamlit)       │    │  (Template Engine)  │    │  (Transformers)  │
    │                   │    │                     │    │                  │
    └───────────────────┘    └─────────────────────┘    └──────────────────┘
    """, language="text")

    st.subheader("Performance Metrics")
    st.markdown("""
    - **Generation Speed**: ~5-8 seconds
    - **Classification Accuracy**: ~90%
    - **Content Quality**: Avg 4.2/5 rating by reviewers
    """)

# --- TAB 4: GitHub & Setup ---
with tab4:
    st.header("Setup Instructions")
    st.markdown("""
    **GitHub Repo**: [Technical Documentation Assistant](https://github.com/GopalAnil/Technical_doc_ai)
    
    **Setup:**
    ```bash
    git clone https://github.com/GopalAnil/Technical_doc_ai
    cd Technical_doc_ai
    python -m venv venv
    source venv/bin/activate  # On Windows use venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run src/app.py
    ```
    """)

    st.subheader("Requirements")
    st.code("""
    streamlit>=1.18.0
    torch>=1.13.0
    transformers>=4.27.0
    datasets>=2.9.0
    pandas, scikit-learn, nltk
    """, language="text")

    st.subheader("Deployment")
    st.markdown("""
    - Streamlit Cloud
    - Heroku (with Procfile)
    - Local server + Ngrok
    """)

if __name__ == "__main__":
    pass
