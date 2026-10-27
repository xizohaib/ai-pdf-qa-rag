import re

def create_chunks(pages: list[dict], chunk_words: int = 180, overlap_words: int = 30) -> list[dict]:
    """Create overlapping word windows and preserve original page metadata."""
    if chunk_words < 1 or overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("Require chunk_words >= 1 and 0 <= overlap_words < chunk_words.")
    chunks = []
    step = chunk_words - overlap_words
    for page in pages:
        words = re.findall(r"\S+", page["text"])
        for start in range(0, len(words), step):
            window = words[start:start + chunk_words]
            if not window:
                break
            chunks.append({
                "page": int(page["page"]),
                "text": " ".join(window),
                "chunk_id": len(chunks),
                "start_word": start,
                "end_word": start + len(window)
            })
            if start + chunk_words >= len(words):
                break
    return chunks
