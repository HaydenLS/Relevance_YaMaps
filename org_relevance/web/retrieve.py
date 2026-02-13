import numpy as np
import re
import html
from typing import TypedDict, Literal, Optional, Dict, Any, List, Sequence, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from org_relevance.common.types import RetrievedChunk

### Вспомогательные методы
### ----------------------
def clean_text(raw: str) -> str:
    """
    Метод предназначен для небольшой очистки текста от мусора в найденных документах.
    """
    s = html.unescape(raw or "")
    # убрать теги
    s = re.sub(r"<script.*?</script>", " ", s, flags=re.S|re.I)
    s = re.sub(r"<style.*?</style>", " ", s, flags=re.S|re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    # нормализовать пробелы
    s = re.sub(r"[\t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ ]{2,}", " ", s)
    s = s.strip()

    return s


def tokenize(text):
    """
    Метод для токенизации
    """
    text = text.lower()
    text = re.sub(r"[^a-zа-я0-9\s]+", " ", text)
    return text.split()

### ----------------
### Конец блока вспомогательных методов


def retrieve_top_chunks_on_the_fly(
    query: str,
    web_results: Sequence[Dict[str, Any]],
    embedding_model,
    top_k: int = 5,
    min_chunk_len: int = 60,
    chunk_size: int = 380,
    chunk_overlap: int = 60,
    batch_size: int = 64,
    max_total_chunks: int = 2000,
    lexical_prefilter_topn: int = 500) -> List[RetrievedChunk]:


    query = query.strip()
    if not query or not web_results:

        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[str] = []
    metas: List[Tuple[str, str]] = []

    # for logging
    skipeed_min_len_chunks_count = 0

    # 1) чанкование
    for d in web_results:
        content = str(d.get("content", "") or "").strip()
        content = clean_text(content)
        if not content:
            continue

        url = str(d.get("url", "") or "")
        title = str(d.get("title", "") or "")

        parts = splitter.split_text(content)
        for p in parts:
            p = p.strip()
            if len(p) < min_chunk_len:
                skipeed_min_len_chunks_count+=1
                continue
            chunks.append(p)
            metas.append((url, title))
            if len(chunks) >= max_total_chunks:
                break
        if len(chunks) >= max_total_chunks:
            break

    if not chunks:
        return []

    # 2) фильтр bm-25
    if (lexical_prefilter_topn != 0) and (len(chunks) > lexical_prefilter_topn):
        # токенизация
        tokenized_chunks = [tokenize(c) for c in chunks]
        query_tokens = tokenize(query)

        # создаем объект bm-25
        bm25 = BM25Okapi(tokenized_chunks)
        # получаем скоры
        scores = bm25.get_scores(query_tokens)

        topn = min(lexical_prefilter_topn, len(chunks))
        idx = np.argpartition(-scores, kth=topn-1)[:topn]

        cand_idx = idx[np.argsort(-scores[idx])]

        # оставить только эти чанки
        chunks = [chunks[i] for i in cand_idx]
        metas = [metas[i] for i in cand_idx]

    # 3) embeddings (E5: префиксы query:/passage:)

    q_text = "query: " + query
    p_texts = ["passage: " + c for c in chunks]

    qv = embedding_model.encode(
        [q_text],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)  # (1, dim)

    X = embedding_model.encode(
        p_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)  # (n, dim)

    # 4) cosine similarity = dot
    sims = (X @ qv.T).reshape(-1)

    k = min(top_k, len(chunks))
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    out: List[RetrievedChunk] = []
    for j in idx:
        url, title = metas[int(j)]
        out.append({
            "text": chunks[int(j)],
            "score": float(sims[int(j)]),
            "url": url,
            "title": title,
        })

    return out


