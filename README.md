# 📚 Technical Documentation Assistant

An AI-powered application that generates comprehensive technical documentation using fine-tuned generative models.

Built with:
- Fine-tuned **FLAN-T5** for text generation
- Fine-tuned **DistilBERT** for document classification
- Interactive **Streamlit** web application

---



## 🛠️ Features
- **Document Classification**: Automatically classifies input text into API Reference, Concept Explanation, Tutorial, or Troubleshooting Guide.
- **Prompt Engineering**: Specialized prompts dynamically structured based on document type.
- **Fine-Tuned Generation**: Uses a fine-tuned FLAN-T5 model for generating high-quality, domain-specific content.
- **Fallback Content**: Provides meaningful outputs even if generation fails, using a curated knowledge base.
- **Interactive Interface**: User-friendly web app with live generation, editable parameters, and model status display.

---

## 🧐 System Architecture

```
🔌 Streamlit Frontend (UI)
    🔻 Core Processing (Prompting + Context Management)
        🔻 Fine-tuned Models (Classification + Generation)
            🔻 Fallback Knowledgebase (Optional)
                🔻 Final Documentation Output
```

---

## 📦 Project Structure

```
technical_doc_assistant/
├── data/
├── models/
├── src/
│   ├── app.py
│   ├── model/
│   ├── data_processing/
│   └── prompt_engineering/
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🛠️ Installation and Setup

1. **Clone the repository**
```bash
git clone https://github.com/GopalAnil/Technical_doc_ai.git
cd Technical_doc_ai
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate    # For Linux/macOS
venv\Scripts\activate       # For Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run src/app.py
```

---

## 🧪 Tech Stack

- **Frontend**: Streamlit
- **Backend Models**:
  - Fine-tuned **DistilBERT** (Classification)
  - Fine-tuned **FLAN-T5** (Generation)
- **Libraries**: Hugging Face Transformers, PyTorch, Scikit-learn, Pandas, Numpy, Datasets, Streamlit

---

## 📊 Performance Metrics

| Metric                        | Result                |
| ------------------------------ | --------------------- |
| Classification Accuracy       | ~90% on test data      |
| Generation Quality (Rating)    | 4.2/5 (internal eval)  |
| Average Response Time          | 5-15 seconds           |

---

## 🧑‍💻 Challenges Faced

- Fine-tuning large models with limited hardware (solved via LoRA).
- Designing effective prompts for consistent structured outputs.
- Handling model fallback cases for reliability.

---

## 🔮 Future Work

- Add **Retrieval-Augmented Generation (RAG)** for knowledge-grounded outputs.
- Deploy app via **Streamlit Cloud** or **AWS**.
- Extend multi-lingual documentation generation.

---

## ⚖️ Ethical Considerations

- All models run locally ensuring **user data privacy**.
- Generated content is **original** — no plagiarism.
- Encourages **manual review** before publishing generated documentation.

---

## ✨ Contributors

- **Gopal Anil**  
*(Developer, Fine-tuning Engineer, Prompt Designer)*

---

