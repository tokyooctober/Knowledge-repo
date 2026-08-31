# `ingestion/chunker.py` — Text Chunker

---
```
module:     ingestion/chunker.py
spec:       ingestion/SPEC_chunker.md
layer:      Ingestion — chunking
depends_on: config.py · logger.py
            models.py  (Article, ImageTranscription → Chunk)
used_by:    scheduler/monthly_job.py
input:      Article  (from ingestion/extractor.py)
            list[ImageTranscription]  (from ingestion/image_transcriber.py)
output:     list[Chunk]  →  passed to ingestion/embedder.py
services:   none  (pure text splitting, no network calls)
```
---

## Purpose
Split a clean `Article` body — including extracted HTML tables and vision-transcribed image content — into overlapping text chunks suitable for embedding and retrieval. Each chunk carries metadata linking it back to its source article and content type.

---

## Responsibilities
- Split article body text into overlapping chunks
- Emit each Markdown table as its own dedicated chunk (tables are not split mid-row)
- Emit each image transcription as its own dedicated chunk
- Attach source metadata and `content_type` to every chunk
- Skip stub articles
- Return a list of `Chunk` objects ready for the embedder

---

## Inputs

| Input | Source | Description |
|---|---|---|
| `article` | `Article` from extractor | Clean article with `body_text`, `is_stub`, metadata |

---

## Outputs

| Output | Type | Description |
|---|---|---|
| `List[Chunk]` | Python list | Ordered chunks with text and source metadata |

### `Chunk` dataclass

Defined once in `models.py` (see [SPEC.md](../SPEC.md)); shown here for reference.

```python
@dataclass
class Chunk:
    chunk_id:      str       # f"{url_hash}_{content_type[0]}_{chunk_index}" — stable, unique
    article_url:   str
    article_title: str
    published_at:  datetime | None
    tags:          list[str]
    text:          str
    content_type:  str       # "body" | "table" | "image_transcription"
    chunk_index:   int       # 0-based position within this content_type group
    total_chunks:  int       # total chunks for this article (all types combined)
    word_count:    int
```

`content_type` is stored in the Qdrant payload, enabling filtered retrieval. A query like *"show me the table data about Bitcoin"* can filter to `content_type="table"` only.

---

## Chunking Strategy

### Primary: Recursive Character Splitting
Use `langchain_text_splitters.RecursiveCharacterTextSplitter` with separators in priority order:

```python
separators = ["\n\n", "\n", ". ", " ", ""]
```

This respects paragraph boundaries first, then sentence boundaries, then word boundaries. A chunk will never be split mid-sentence unless a single sentence exceeds `CHUNK_SIZE`.

### Secondary: Header-aware splitting (optional)
If the article body contains Markdown headers (`## Section`), apply `MarkdownHeaderTextSplitter` first to create sections, then recursively split each section. This keeps section context intact and improves retrieval precision.

Enable via `USE_HEADER_SPLITTING = True` in config.

---

## Configuration Constants
```python
CHUNK_SIZE            = 512    # target chunk size in tokens
CHUNK_OVERLAP         = 64     # overlap in tokens between consecutive chunks
TOKENIZER             = "cl100k_base"  # tiktoken encoding (matches most embedding models)
USE_HEADER_SPLITTING  = False  # enable section-aware splitting
MIN_CHUNK_WORDS       = 30     # discard chunks shorter than this (e.g. orphaned headings)
```

### Why 512 tokens with 64 overlap?
- 512 tokens (~380 words) fits within the 512-token limit of `BAAI/bge-large-en-v1.5`
- 64-token overlap (~50 words) ensures key sentences near chunk boundaries appear in two chunks, so retrieval doesn't miss them
- If switching to OpenAI embeddings (8192-token limit), increase `CHUNK_SIZE` to 1024–2048

---

## Core Logic

```
1. If article.is_stub → return []

chunks = []
url_hash = sha256(article.url.encode()).hexdigest()[:8]

2. BODY TEXT CHUNKS  (content_type = "body")
   texts = RecursiveCharacterTextSplitter.split_text(article.body_text)
   Filter texts where word_count < MIN_CHUNK_WORDS
   For i, text in enumerate(texts):
     chunk_id = f"{url_hash}_b_{i:04d}"
     chunks.append(Chunk(..., content_type="body", chunk_index=i))

3. TABLE CHUNKS  (content_type = "table")
   Tables are NOT split mid-row. Each table is a single chunk,
   even if it exceeds CHUNK_SIZE. Oversized tables log a WARNING.
   For i, table_md in enumerate(article.tables_md):
     if word_count(table_md) < MIN_CHUNK_WORDS: skip
     chunk_id = f"{url_hash}_t_{i:04d}"
     chunks.append(Chunk(..., content_type="table", chunk_index=i))

4. IMAGE TRANSCRIPTION CHUNKS  (content_type = "image_transcription")
   For i, transcription in enumerate(image_transcriptions):
     if transcription.skipped: continue
     if word_count(transcription.transcription) < MIN_CHUNK_WORDS: continue
     chunk_id = f"{url_hash}_i_{i:04d}"
     text = transcription.transcription   # already prefixed with [Chart N: "title"]
     chunks.append(Chunk(..., content_type="image_transcription", chunk_index=i))

5. SET total_chunks on all chunks
   n = len(chunks)
   for chunk in chunks: chunk.total_chunks = n

6. RETURN chunks
```

### Why tables and transcriptions are never split
Tables are Markdown pipe-format — splitting mid-row produces malformed Markdown that embedding models cannot reliably parse. Each table is a semantic unit: a reader would never answer a table-based question from half a table.

Image transcriptions are already bounded to `VISION_MAX_TOKENS` (~300 words) by the vision prompt — they will always fit within `CHUNK_SIZE`.

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| `article.is_stub == True` | Return `[]` immediately; log at DEBUG |
| Single chunk > CHUNK_SIZE after all separators exhausted | Allow it (hard truncation is worse than a slightly large chunk); log at WARNING |
| All chunks filtered by MIN_CHUNK_WORDS | Return `[]`; log at WARNING with article URL |

---

## Key Dependencies
- `langchain-text-splitters` — `RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter`
- `tiktoken` — token counting consistent with embedding models
- `hashlib` — stable chunk ID generation

---

## Public Interface
```python
def chunk_article(
    article: Article,
    image_transcriptions: list[ImageTranscription] | None = None,
) -> list[Chunk]:
    """Split article into body, table, and image transcription chunks.

    Body text: split with overlap (RecursiveCharacterTextSplitter).
    Tables: one chunk per table, never split mid-row.
    Image transcriptions: one chunk per non-skipped transcription.
    All chunks share total_chunks count and carry content_type for filtered retrieval.
    Returns empty list if article is a stub.
    """
```

---

## Logging

```python
log = get_logger(__name__)   # "knowledge_repo.ingestion.chunker"
```

| Event | Level | Extra fields |
|---|---|---|
| Skipping stub article | DEBUG | `url`, `word_count` |
| Body text chunking started | DEBUG | `url`, `body_length_chars` |
| Body chunk too short — discarded | WARNING | `url`, `chunk_index`, `word_count`, `min_chunk_words` |
| Body chunk exceeds CHUNK_SIZE | WARNING | `url`, `chunk_index`, `token_count` |
| Table chunk added | DEBUG | `url`, `table_index`, `word_count` |
| Table chunk too short — skipped | WARNING | `url`, `table_index`, `word_count` |
| Table chunk oversized | WARNING | `url`, `table_index`, `token_count` |
| Image transcription chunk added | DEBUG | `url`, `image_index`, `image_type`, `word_count` |
| Image transcription skipped | DEBUG | `url`, `image_index`, `skip_reason` |
| Chunking complete | INFO | `url`, `total_chunks`, `body_chunks`, `table_chunks`, `image_chunks` |

---

## Testing Notes
- Assert body chunks have `content_type="body"` and correct overlap
- Assert each table in `article.tables_md` produces exactly one chunk with `content_type="table"`
- Assert tables are never split (chunk text == full table Markdown)
- Assert oversized tables log WARNING but still produce a chunk
- Assert each non-skipped `ImageTranscription` produces one chunk with `content_type="image_transcription"`
- Assert skipped transcriptions produce no chunk
- Assert `chunk_id` encodes content type: `_b_`, `_t_`, `_i_` prefixes
- Assert `total_chunks` equals body + table + image chunks combined
- Assert stub articles return `[]`
- Assert `chunk_article(article, image_transcriptions=None)` produces only body + table chunks
