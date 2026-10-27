import hashlib
import io
import re
import time

import pandas as pd
import streamlit as st

from src.pdf_processor import extract_pages
from src.chunker import create_chunks
from src.retriever import HybridRetriever, rerank_passages
from src.rag_engine import generate_grounded_answer, validate_citations

st.set_page_config(page_title="AI PDF Q&A — Phase 5", page_icon="📚", layout="wide")
st.title("📚 AI Research & Document Intelligence")
st.caption("Phase 5 · Advanced RAG: hybrid retrieval, optional reranking, citation checks, and evaluation")

with st.sidebar:
    st.header("RAG settings")
    chunk_words = st.slider("Words per chunk", 80, 400, 180, 20)
    overlap_words = st.slider("Chunk overlap", 0, 80, 30, 10)
    top_k = st.slider("Evidence passages", 1, 10, 5)
    candidate_k = st.slider("Candidates before reranking", 3, 20, 10)
    use_reranker = st.checkbox("Enable cross-encoder reranking (slower)", value=False)
    st.caption("Reranking downloads an additional model the first time it is used. Keep it off on low-memory laptops.")
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
    model_name = st.text_input("Ollama model", value="qwen2.5:3b")
    temperature = st.slider("Generation temperature", 0.0, 0.5, 0.1, 0.05)
    st.caption("Generation runs locally through Ollama.")

uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])

if uploaded:
    pdf_bytes = uploaded.getvalue()
    fingerprint = hashlib.sha256(pdf_bytes).hexdigest()
    try:
        pages = [p for p in extract_pages(pdf_bytes) if p["text"].strip()]
        if not pages:
            st.error("No selectable text found. This may be a scanned PDF; OCR is not included.")
            st.stop()

        chunks = create_chunks(pages, chunk_words, overlap_words)
        if not chunks:
            st.error("Could not create chunks from this PDF.")
            st.stop()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pages with text", len(pages))
        c2.metric("Text chunks", len(chunks))
        c3.metric("Words extracted", sum(len(p["text"].split()) for p in pages))
        c4.metric("PDF size", f"{len(pdf_bytes) / 1024:.0f} KB")

        with st.expander("Preview extracted text"):
            page_no = st.selectbox("Choose page", [p["page"] for p in pages])
            st.text_area("Page text", next(p["text"] for p in pages if p["page"] == page_no), height=200)

        @st.cache_resource(show_spinner="Building keyword and semantic indexes...")
        def get_hybrid_retriever(pdf_fingerprint, _chunks, words_per_chunk, overlap, model_name):
            # Fingerprint and chunk settings key the cache; _chunks is deliberately excluded from hashing.
            return HybridRetriever(_chunks)

        retriever = get_hybrid_retriever(
            fingerprint, chunks, chunk_words, overlap_words, HybridRetriever.SEMANTIC_MODEL
        )

        chat_tab, eval_tab, guide_tab = st.tabs(["💬 Ask the PDF", "📊 Evaluate retrieval", "🧠 How it works"])

        with chat_tab:
            question = st.text_input(
                "Ask a question about this PDF",
                placeholder="e.g., What are the main findings and which page supports them?"
            )
            answer_button = st.button("Generate grounded answer", type="primary", disabled=not question.strip())

            if answer_button:
                try:
                    with st.spinner("Searching by keywords and meaning..."):
                        start = time.perf_counter()
                        candidates = retriever.search(question, top_k=max(candidate_k, top_k))
                        if use_reranker and candidates:
                            candidates = rerank_passages(question, candidates)
                        retrieved = candidates[:top_k]
                        retrieval_seconds = time.perf_counter() - start

                    st.caption(
                        f"Hybrid retrieval completed in {retrieval_seconds:.2f}s "
                        f"using reciprocal-rank fusion{ ' + cross-encoder reranking' if use_reranker else ''}."
                    )

                    if not retrieved or max(
                        retrieved[0].get("lexical_score", 0.0),
                        retrieved[0].get("semantic_score", 0.0)
                    ) <= 0:
                        st.warning("No meaningful match was found. Try rephrasing the question.")
                    else:
                        with st.spinner(f"Generating an evidence-grounded answer with {model_name}..."):
                            answer = generate_grounded_answer(
                                question=question,
                                passages=retrieved,
                                model=model_name.strip(),
                                base_url=ollama_url.strip(),
                                temperature=temperature
                            )

                        st.markdown("### Answer")
                        if answer["abstained"]:
                            st.warning(answer["answer"])
                        else:
                            st.markdown(answer["answer"])
                            st.caption("Verify important claims against the evidence below; generated answers can still be wrong.")

                        citation_check = validate_citations(answer["answer"], answer["sources"])
                        if citation_check["unknown_source_ids"] or citation_check["page_mismatches"]:
                            st.warning("Citation check found a possible issue. Review the cited pages before trusting the answer.")
                            if citation_check["unknown_source_ids"]:
                                st.write("Unknown source IDs:", ", ".join(citation_check["unknown_source_ids"]))
                            if citation_check["page_mismatches"]:
                                st.write("Source/page mismatches:", ", ".join(citation_check["page_mismatches"]))
                        elif citation_check["citation_count"]:
                            st.success(f"Citation format check passed for {citation_check['citation_count']} citation(s). This checks format, not factual support.")

                        st.markdown("### Retrieved evidence")
                        for rank, source in enumerate(answer["sources"], 1):
                            with st.container(border=True):
                                st.markdown(f"**{rank}. Source {source['source_id']} · PDF page {source['page']}**")
                                left, right, third = st.columns(3)
                                left.metric("Hybrid score", f"{source.get('score', 0.0):.4f}")
                                right.metric("Keyword score", f"{source.get('lexical_score', 0.0):.3f}")
                                third.metric("Semantic score", f"{source.get('semantic_score', 0.0):.3f}")
                                st.write(source["text"])
                                if "rerank_score" in source:
                                    st.caption(f"Cross-encoder rerank score: {source['rerank_score']:.3f}")
                except Exception as err:
                    st.error("Could not complete the RAG request. Check the Ollama service and selected model.")
                    st.code(str(err))

        with eval_tab:
            st.markdown(
                "Upload a CSV with test questions and known relevant PDF pages. "
                "This evaluates retrieval without calling Ollama, so it is faster and helps compare retrieval changes."
            )
            template = pd.DataFrame([
                {"question": "What is the main conclusion?", "expected_pages": "3;4"},
                {"question": "Which method was used?", "expected_pages": "2"}
            ])
            st.download_button(
                "Download evaluation CSV template",
                data=template.to_csv(index=False).encode("utf-8"),
                file_name="rag_evaluation_template.csv",
                mime="text/csv"
            )
            st.caption("Use the actual page numbers from your uploaded PDF. Separate multiple relevant pages with semicolons, e.g. 2;5;8.")
            eval_file = st.file_uploader("Upload evaluation CSV", type=["csv"], key="eval_csv")
            if eval_file:
                try:
                    eval_df = pd.read_csv(eval_file)
                    if "question" not in eval_df.columns or "expected_pages" not in eval_df.columns:
                        st.error("CSV must contain 'question' and 'expected_pages' columns.")
                    else:
                        max_eval_k = st.slider("Evaluate top-k", 1, 10, min(5, top_k), key="eval_k")
                        if st.button("Run retrieval evaluation", type="primary"):
                            rows = []
                            progress = st.progress(0)
                            valid_rows = eval_df.dropna(subset=["question"])
                            for i, row in enumerate(valid_rows.itertuples(index=False), 1):
                                row_dict = row._asdict()
                                q = str(row_dict.get("question", "")).strip()
                                raw_pages = str(row_dict.get("expected_pages", "")).strip()
                                if not q:
                                    continue
                                expected = set()
                                for value in re.split(r"[;,|]", raw_pages):
                                    value = value.strip()
                                    if value.isdigit():
                                        expected.add(int(value))
                                t0 = time.perf_counter()
                                found = retriever.search(q, top_k=max_eval_k)
                                if use_reranker and found:
                                    found = rerank_passages(q, found)
                                found = found[:max_eval_k]
                                elapsed = time.perf_counter() - t0
                                result_pages = [int(x["page"]) for x in found]
                                hit = any(p in expected for p in result_pages) if expected else None
                                reciprocal_rank = 0.0
                                for rank, page in enumerate(result_pages, 1):
                                    if page in expected:
                                        reciprocal_rank = 1.0 / rank
                                        break
                                rows.append({
                                    "question": q,
                                    "expected_pages": ";".join(map(str, sorted(expected))),
                                    "retrieved_pages": ";".join(map(str, result_pages)),
                                    f"hit@{max_eval_k}": hit,
                                    "reciprocal_rank": reciprocal_rank if expected else None,
                                    "retrieval_seconds": round(elapsed, 4)
                                })
                                progress.progress(min(i / max(1, len(valid_rows)), 1.0))
                            results_df = pd.DataFrame(rows)
                            if results_df.empty:
                                st.warning("No usable questions were found in the CSV.")
                            else:
                                measurable = results_df[results_df["expected_pages"].astype(bool)]
                                m1, m2, m3 = st.columns(3)
                                if len(measurable):
                                    m1.metric(f"Hit rate @ {max_eval_k}", f"{measurable[f'hit@{max_eval_k}'].mean():.1%}")
                                    m2.metric("Mean reciprocal rank", f"{measurable['reciprocal_rank'].mean():.3f}")
                                else:
                                    m1.metric(f"Hit rate @ {max_eval_k}", "N/A")
                                    m2.metric("Mean reciprocal rank", "N/A")
                                m3.metric("Average retrieval time", f"{results_df['retrieval_seconds'].mean():.3f}s")
                                st.dataframe(results_df, use_container_width=True)
                                st.download_button(
                                    "Download evaluation results",
                                    data=results_df.to_csv(index=False).encode("utf-8"),
                                    file_name="rag_evaluation_results.csv",
                                    mime="text/csv"
                                )
                                if not len(measurable):
                                    st.info("Add expected page numbers to the CSV to calculate hit rate and reciprocal rank.")
                except Exception as err:
                    st.error(f"Could not read/evaluate the CSV: {err}")

        with guide_tab:
            st.markdown("""
            **Phase 5 pipeline**

            1. **Chunk:** split PDF text into overlapping chunks while preserving original page numbers.
            2. **Hybrid retrieve:** run TF-IDF keyword retrieval and MiniLM semantic retrieval.
            3. **Fuse rankings:** combine both result lists with reciprocal-rank fusion (RRF), which does not assume their raw scores are comparable.
            4. **Optional rerank:** use a cross-encoder to score the question against retrieved candidates more directly.
            5. **Generate:** send the best passages to Ollama with strict grounding and citation instructions.
            6. **Check citations:** flag source IDs or page numbers that do not match the retrieved evidence.
            7. **Evaluate:** use questions with known relevant pages to measure Hit@K and Mean Reciprocal Rank (MRR).

            **Metric meanings**
            - **Hit@K:** fraction of test questions for which at least one known relevant page appears in the top K retrieved results.
            - **MRR:** rewards relevant evidence appearing near the top of the ranking.
            - **Citation format check:** catches some malformed or mismatched citations; it does not prove that a claim is supported.

            **Privacy:** PDF processing and generation run locally. Sentence Transformer models may be downloaded on first use. No hosted LLM API is required.
            """)
    except Exception as err:
        st.error(f"Could not process the PDF: {err}")
        st.exception(err)
else:
    st.markdown("### Get started")
    st.write("Upload a text-based PDF, then ask a question or evaluate retrieval with a labeled CSV.")
    st.markdown("**Required once:** install Ollama, download the model, and leave Ollama running. See the README.")
