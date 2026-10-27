# AI PDF Q&A — Phase 5: Advanced RAG

Phase 5 builds on the Phase 4 custom Python RAG app. It adds:

- **Hybrid retrieval:** TF-IDF keyword retrieval + MiniLM semantic retrieval.
- **Reciprocal-rank fusion (RRF):** combines both rankings without assuming raw scores share the same scale.
- **Optional cross-encoder reranking:** reorders a small candidate set for relevance. It is off by default because it downloads another model and uses additional memory.
- **Citation format checks:** flags unknown source IDs and source/page mismatches. This does not prove that a claim is factually supported.
- **Retrieval evaluation:** upload labeled questions and relevant PDF page numbers; calculate Hit@K, Mean Reciprocal Rank (MRR), and retrieval time.
- **Grounded prompting and abstention:** the model is instructed to answer only from the supplied evidence and abstain when evidence is insufficient.
- **Evidence display:** keyword, semantic, fusion, and optional reranking scores are shown alongside source passages.

The app is designed for local use with Ollama and does not require a paid LLM API.

## 1. Requirements

- Windows 10/11
- Python 3.10 or newer (Python 3.11 is a sensible choice)
- Ollama installed and running
- At least one Ollama model, e.g. `qwen2.5:3b`

## 2. Open the folder

Extract `AI_PDF_QA_Phase5.zip`, open the `AI_PDF_QA_Phase5` folder in VS Code, and open a terminal in that folder.

Keep Phase 4 as a separate backup. Do not overwrite your working Phase 4 folder.

## 3. Create and activate a virtual environment (CMD)

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you use PowerShell instead:

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked or the `streamlit` command is not recognized, try:

```bat
python -m streamlit run app.py
```

## 4. Prepare Ollama

In a separate terminal:

```bat
ollama pull qwen2.5:3b
ollama list
```

Make sure Ollama is running. The app sends requests to `http://localhost:11434` by default.

## 5. Run the app

From the Phase 5 project folder, with `.venv` active:

```bat
python -m streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`.

## 6. Ask questions

1. Upload a selectable-text PDF.
2. Ask a question.
3. Hybrid retrieval combines keyword and semantic rankings.
4. Optionally enable cross-encoder reranking. The first use downloads the reranker model.
5. Inspect the answer, citation-format warning (if any), and evidence passages.

Scanned PDFs without selectable text are not supported because OCR is not included.

## 7. Evaluate retrieval

Prepare a CSV with two required columns:

```csv
question,expected_pages
"What is the main conclusion?","3;4"
"Which method was used?","2"
```

Use page numbers from the PDF itself. If more than one page can answer the question, separate page numbers with semicolons.

In the **Evaluate retrieval** tab:
1. Download the CSV template or create your own.
2. Upload the CSV.
3. Select top-K and run evaluation.
4. Review/download the per-question results.

Metrics:
- **Hit@K:** proportion of labeled questions for which at least one expected page appears among the top K retrieved passages.
- **MRR:** reciprocal rank of the first retrieved passage on an expected page, averaged across labeled questions. Higher means relevant evidence tends to appear earlier.
- **Average retrieval time:** time to retrieve the evidence for one question; it excludes LLM generation.

For a fair comparison, keep the PDF, chunk settings, test questions, and expected page labels unchanged when changing retrieval settings.

## 8. Notes and limitations

- Retrieval scores are ranking signals, not probabilities of correctness.
- The citation checker validates citation IDs/page numbers against retrieved sources. It does not prove that each claim is entailed by its cited text.
- Evaluation is only as reliable as your labeled expected pages. A page can be relevant even if it is not listed; include all acceptable pages when possible.
- Cross-encoder reranking can improve ordering, but may be slow or memory-intensive on CPU. Keep it disabled if your laptop struggles.
- MiniLM model files download on first use and are then cached by the local Python environment.
- PDF text stays local in this pipeline. Ollama is called through its local HTTP endpoint.

## 9. Suggested experiment

Run the same labeled question CSV twice:
1. Hybrid retrieval with reranking disabled.
2. Hybrid retrieval with reranking enabled.

Compare Hit@K, MRR, and retrieval latency. Keep the option that gives a useful quality improvement for its extra cost.
