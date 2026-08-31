# `query/answerer.py` — Context Builder + LLM Answerer

---
```
module:     query/answerer.py
spec:       query/SPEC_answerer.md
layer:      Query
depends_on: config.py · logger.py
            llm_provider.py  (TextProvider)
            models.py  (SearchResult → Answer, Source)
used_by:    app.py
input:      query str + list[SearchResult]  (from query/retriever.py)
output:     Answer  (response text + cited sources)  →  rendered by app.py
services:   text LLM  (via llm_provider.py)
```
---

## Purpose
Take a ranked list of retrieved `SearchResult` chunks, assemble a grounded prompt, and call the Claude API to produce a cited, faithful answer. This is the final stage of the query pipeline visible to the user.

---

## Responsibilities
- Format retrieved chunks into a structured context block for the prompt
- Craft a system prompt that enforces grounded, citation-bearing responses
- Call the Claude API with the assembled prompt
- Parse the response and extract inline citations
- Return a structured `Answer` object with the response text and source list
- Handle the case where no relevant chunks were found

---

## Inputs

| Input | Source | Description |
|---|---|---|
| `query` | User | Original natural language question |
| `results` | `retriever.retrieve()` | Ranked list of `SearchResult` objects |

---

## Outputs

| Output | Type | Description |
|---|---|---|
| `Answer` | dataclass | Response text + list of cited sources |

### `Answer` dataclass
```python
@dataclass
class Answer:
    query:       str
    response:    str            # Claude's answer text, may include [1], [2] citations
    sources:     list[Source]   # ordered list of cited sources
    model:       str            # e.g. "claude-sonnet-4-20250514"
    input_tokens:  int
    output_tokens: int

@dataclass
class Source:
    index:       int            # 1-based citation number
    title:       str
    url:         str
    published_at: datetime | None
    score:       float          # retrieval similarity score
```

---

## Prompt Design

### System prompt
```
You are an expert research assistant with deep knowledge of [AUTHOR_NAME]'s writing and ideas.

You will be given a question and a set of numbered excerpts from [AUTHOR_NAME]'s articles.

Rules:
1. Answer using ONLY information present in the provided excerpts.
2. Cite sources inline using [N] notation, where N is the excerpt number.
3. If the excerpts do not contain enough information to answer, say so clearly — do not speculate.
4. If multiple excerpts support the same point, cite all of them: [1][3].
5. Preserve [AUTHOR_NAME]'s voice and terminology when summarising.
6. Keep your answer concise: 2–4 paragraphs unless detail is explicitly requested.
```

### User message (context block)
```
EXCERPTS:

[1] "{chunk_1_text}"
Source: {title_1} ({published_at_1})
URL: {url_1}

[2] "{chunk_2_text}"
Source: {title_2} ({published_at_2})
URL: {url_2}

... (up to top_k excerpts)

QUESTION: {query}
```

---

## Core Logic

```
1. HANDLE EMPTY RESULTS
   If results == []:
     Return Answer(
       response="I couldn't find relevant content in the knowledge base for this question.",
       sources=[]
     )

2. CAP AND TRUNCATE
   results = results[:MAX_CONTEXT_CHUNKS]        # retriever may return more if the
                                                # caller passed a large --top-k
   For each result: if len(result.text) > MAX_CHUNK_CHARS, truncate to MAX_CHUNK_CHARS
     on a word boundary and append " …"; log DEBUG {chunk_id, original_chars}.
   The context-window budget math below assumes both caps are applied here.

3. BUILD CONTEXT BLOCK
   For i, result in enumerate(results, start=1):
     Format excerpt block with index, (truncated) text, source title, date, URL
   Join all blocks with double newline

4. BUILD MESSAGES
   messages = [
     {"role": "system", "content": SYSTEM_PROMPT},
     {"role": "user",   "content": context_block + "\n\nQUESTION: " + query}
   ]

5. CALL LLM VIA PROVIDER
   provider = llm_provider.get_text_provider()
   text_response = provider.complete(messages, max_tokens=MAX_OUTPUT_TOKENS)
   response_text = text_response.content

6. EXTRACT SOURCES
   Parse [N] citation markers from response_text
   cited_indices = set of unique N values found
   # Map SearchResult → Source explicitly. `Source(index=i, **results[i-1])` does
   # not work: a SearchResult is a dataclass, not a mapping, and its field names
   # (article_url / article_title) differ from Source's (url / title).
   sources = [
     Source(
       index        = i,
       title        = results[i-1].article_title,
       url          = results[i-1].article_url,
       published_at = results[i-1].published_at,
       score        = results[i-1].score,
     )
     for i in sorted(cited_indices) if 1 <= i <= len(results)
   ]
   # A model that cites [9] against 6 results is dropped by the bounds check rather
   # than raising. Log it at WARNING: a run with many dropped citations means the
   # prompt's numbering instructions are not landing.

7. RETURN Answer(
     query=query,
     response=response_text,
     sources=sources,
     model=text_response.model,
     input_tokens=text_response.input_tokens,
     output_tokens=text_response.output_tokens,
   )
```

---

## Configuration Constants
```python
# Model selection is fully delegated to llm_provider.py.
# Set LLM_BACKEND, LLM_MODEL, and LLM_BASE_URL in config.py to choose the provider.
# No model name is referenced here — answerer.py is provider-agnostic.
MAX_OUTPUT_TOKENS  = 1024
AUTHOR_NAME        = "..."              # personalises system prompt
MAX_CONTEXT_CHUNKS = 6                  # max excerpts passed to LLM
MAX_CHUNK_CHARS    = 1200               # truncate individual excerpts at this length
```

### Context window budget
At 6 excerpts × 1200 chars ≈ 7200 chars ≈ ~1800 tokens of context. System prompt ≈ 150 tokens. Question ≈ 50 tokens. Total input ≈ 2000 tokens — fits comfortably within every model in the supported reference table (minimum context window across supported open-weights models is 8k tokens for Llama 3.3 70B; Qwen 2.5 72B and Claude Sonnet support 128k+).

> **Open-weights citation reliability note**: smaller models (< 30B parameters) may not follow the `[N]` citation instruction as reliably as larger ones or Claude. If `sources=[]` after parsing, the answer is still returned — the quality difference is in citation precision, not in the grounded answer itself. Models known to follow citation instructions well: `llama3.3:70b`, `qwen2.5:72b`, `mistral-small3.1`, `claude-sonnet-*`.

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| `results` is empty | Return graceful "not found" answer — no LLM call made |
| Provider rate limit (any backend) | Retry with exponential backoff via `tenacity` (max 3); handled by `llm_provider.py` |
| Provider auth error (any backend) | Propagate `AuthenticationError`; hint references the relevant env var |
| Provider endpoint unreachable | Propagate `ProviderConnectionError` from `llm_provider.py` |
| Response contains no `[N]` citations | Return answer with `sources=[]`; log at WARNING |
| Citation index out of range | Skip that citation; log at WARNING |

---

## Key Dependencies
- `llm_provider.py` — provider abstraction (no direct SDK imports in answerer.py)
- `re` — citation index parsing (stdlib)

---

## Public Interface
```python
def answer(query: str, results: list[SearchResult]) -> Answer:
    """Build a grounded, cited answer from retrieved chunks.

    Delegates LLM call to llm_provider.get_text_provider().
    Provider (Claude, Ollama, vLLM, etc.) is determined entirely by config.
    Returns a graceful 'not found' Answer if results is empty.
    Citations in response text are [N] numbered, matching Answer.sources list.
    """
```

---

## Logging

```python
log = get_logger(__name__)   # "knowledge_repo.query.answerer"
```

| Event | Level | Extra fields |
|---|---|---|
| No results — returning graceful answer | INFO | `query` |
| Context block assembled | DEBUG | `query`, `excerpt_count`, `total_context_chars` |
| Chunk truncated to MAX_CHUNK_CHARS | DEBUG | `chunk_id`, `original_chars`, `truncated_chars` |
| LLM call started | DEBUG | `backend`, `model`, `input_tokens_estimate` |
| LLM call complete | INFO | `backend`, `model`, `input_tokens`, `output_tokens`, `elapsed_ms` |
| Response contains no citation markers | WARNING | `query`, `model` |
| Out-of-range citation index skipped | WARNING | `citation_index`, `max_valid_index` |
| Answer complete | INFO | `query`, `source_count`, `input_tokens`, `output_tokens` |

---

## Testing Notes
- Mock `llm_provider.get_text_provider()` to return a `MockTextProvider` — no real API calls
- Assert `answer()` never imports `anthropic` or `openai` directly (provider-agnostic)
- Assert `Answer.sources` only contains indices cited in `response_text`
- Assert out-of-range citation indices are skipped gracefully
- Assert empty `results` returns the "not found" message without calling the provider
- Assert 20 input results are capped to `MAX_CONTEXT_CHUNKS` before the prompt is built
- Assert an excerpt longer than `MAX_CHUNK_CHARS` is truncated in the context block and the
  `Source` list still numbers from the capped set
- Assert `Answer.model` is populated from `TextResponse.model` returned by the provider
- Assert `input_tokens` and `output_tokens` are passed through from `TextResponse`
- Integration test: swap `MockTextProvider` for a real provider and confirm grounded answer
