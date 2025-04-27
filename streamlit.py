# Optimized Streamlit App for Streamlit Cloud - Lightweight Version

import streamlit as st
import os
import sys
import logging

# Basic Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dynamically add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Safe imports
try:
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Fallback minimal prompt template
DEFAULT_PROMPT = """
# Concept Explanation: {concept}

## Definition
Explain briefly.

## Key Components
List main parts.

## Applications
List a few real-world uses.
"""

# Helper to load a SMALL model
@st.cache_resource(show_spinner=False)
def load_small_model():
    if TRANSFORMERS_AVAILABLE:
        model_name = "google/flan-t5-small"  # Very lightweight
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        return tokenizer, model
    else:
        return None, None

# Generate text
def generate_text(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit App
st.set_page_config(page_title="Tech Doc Assistant (Cloud Optimized)", page_icon="📚", layout="wide")
st.title("📚 Cloud-Optimized Technical Documentation Assistant")

st.write("Lightweight AI for fast documentation generation on Streamlit Cloud.")

# Load models
with st.spinner("Loading lightweight model..."):
    tokenizer, model = load_small_model()

if not tokenizer or not model:
    st.error("Transformers or Torch not installed. App running in limited mode.")

# Input Section
concept = st.text_input("Enter a technical concept you want explained:", "Encapsulation")

if st.button("Generate Documentation"):
    if not concept.strip():
        st.warning("Please enter a valid concept.")
    else:
        prompt = DEFAULT_PROMPT.format(concept=concept)
        with st.spinner("Generating documentation..."):
            if tokenizer and model:
                doc = generate_text(prompt, tokenizer, model)
            else:
                doc = f"**{concept.capitalize()}** is an important topic.\n\n(Install transformers and torch to enable AI generation.)"
        st.markdown("---")
        st.subheader("Generated Documentation")
        st.markdown(doc)
        st.markdown("---")

st.info("\n✅ Uses google/flan-t5-small for faster load\n✅ Designed for 1 CPU 1 GB RAM\n✅ No heavy local models needed\n✅ Perfect for Streamlit Cloud deployments")
