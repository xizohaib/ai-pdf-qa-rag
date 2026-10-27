# AI PDF Question-Answering System — Advanced RAG

An AI-powered PDF Question-Answering System that extracts information from PDF documents, retrieves relevant passages using hybrid search, and generates evidence-grounded answers with a locally running Large Language Model (LLM).

## Features

**Complete project journey:** PDF text extraction and chunking, TF-IDF retrieval, semantic search with Sentence Transformers, extractive question answering, local LLM-powered RAG, hybrid retrieval with Reciprocal Rank Fusion (RRF), optional cross-encoder reranking, citation-format validation, and retrieval evaluation.

- **Hybrid retrieval:** Combines keyword-based TF-IDF search with semantic embeddings.
- **Reciprocal Rank Fusion:** Combines retrieval rankings without directly comparing incompatible raw scores.
- **Optional reranking:** Uses a cross-encoder to improve the ordering of retrieved passages.
- **Grounded answer generation:** Uses Ollama to generate answers based on retrieved PDF evidence.
- **Source references:** Displays retrieved passages, PDF page numbers, and citation-format checks.
- **Abstention handling:** Instructs the model to acknowledge when the available evidence is insufficient.
- **Retrieval evaluation:** Measures Hit@K, Mean Reciprocal Rank (MRR), and retrieval time using labeled questions.
- **Interactive interface:** Built with Streamlit for document upload, question answering, evidence inspection, and evaluation.

## Technology Stack

- **Language:** Python
- **Interface:** Streamlit
- **PDF processing:** PyMuPDF
- **Keyword retrieval:** scikit-learn TF-IDF
- **Semantic search:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Reranking:** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
- **LLM inference:** Ollama
- **Default LLM:** `qwen2.5:3b`
- **Data analysis:** Pandas and NumPy

## Requirements

- Windows 10/11
- Python 3.10 or newer
- Ollama installed
- An Ollama model downloaded locally

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/xizohaib/ai-pdf-qa-rag.git
cd ai-pdf-qa-rag
```

### 2. Create a virtual environment

On Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Prepare Ollama

Install Ollama from [ollama.com](https://ollama.com/), then run:

```bash
ollama pull qwen2.5:3b
ollama list
```

Ensure Ollama is running before asking questions in the application.

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Usage

1. Upload a selectable-text PDF.
2. Enter a question about the document.
3. Review the generated answer and its source references.
4. Inspect retrieved passages and their retrieval scores.
5. Optionally enable cross-encoder reranking.
6. Upload a labeled evaluation CSV to assess retrieval performance.

Scanned PDFs without selectable text are not supported because OCR is not included.

## Retrieval Evaluation

The evaluation CSV must contain these columns:

```csv
question,expected_pages
"What is the main conclusion?","3;4"
"Which method was used?","2"
```

Use actual PDF page numbers and separate multiple relevant pages with semicolons.

**Metrics:**

- **Hit@K:** Percentage of questions for which at least one expected page appears among the top K retrieved passages.
- **Mean Reciprocal Rank (MRR):** Measures how highly the first relevant result ranks.
- **Average retrieval time:** Measures retrieval latency, excluding LLM generation.

For a fair comparison, keep the PDF, chunk settings, test questions, and expected page labels consistent.

## Example Evaluation Results

An initial evaluation on an Operating Systems notes PDF produced the following results:

| Metric                 |        Result |
| ---------------------- | ------------: |
| Hit@5                  |         93.3% |
| Mean Reciprocal Rank   |         0.822 |
| Average retrieval time | 0.049 seconds |

These results apply to the current labeled test set. They measure retrieval performance, not overall answer accuracy or factual correctness.

## Limitations

- Retrieval scores are ranking signals, not probabilities of correctness.
- Citation checks identify certain source-ID and page-number mismatches but do not prove that a citation supports a claim.
- Evaluation quality depends on accurate expected-page labels and representative test questions.
- Cross-encoder reranking can increase latency and memory usage.
- The first use of embedding and reranking models may require downloading model files.
- Scanned PDFs require OCR support, which is not currently included.
- The application runs locally and requires Ollama; it is not a hosted web service.

## Future Improvements

Potential extensions include multi-PDF question answering, OCR for scanned documents, conversation history, MCQ answer-accuracy evaluation, and deployment.

---

**Author:** Zohaib Ahmed

**GitHub:** https://github.com/xizohaib/ai-pdf-qa-rag
