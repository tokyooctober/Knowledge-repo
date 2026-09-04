"""Every configuration constant the system reads, plus the two Phase-2 validators.

Every module imports `config`, so this file must import cleanly on a machine that only
ever runs Phase 1 — no `os.environ["X"]`, no network, no heavy deps. Phase 2 secrets and
the author-/site-specific values use `os.environ.get(..., "")` and are checked at point of
use by `require_phase2_config()`.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from models import ConfigError  # models.py imports nothing first-party — no cycle

load_dotenv()

# ── Phase 1: markdown corpus ─────────────────────────────────────────────────
MD_CORPUS_DIR = os.environ.get("MD_CORPUS_DIR", "corpus")  # *.md + images/
MD_IMAGES_SUBDIR = "images"  # one subfolder per article inside
PIPELINE_VERSION = 1  # bump when loader/chunker/transcriber semantics change;
#                       stored per article, and a mismatch forces a re-ingest

# ── Phase 2: email reader ────────────────────────────────────────────────────
EMAIL_BACKEND = "gmail"
TRUSTED_SENDER = os.environ.get("TRUSTED_SENDER", "")  # author@example.com
EMAIL_SUBJECT_PATTERN = os.environ.get("EMAIL_SUBJECT_PATTERN", "premium report")
#   matched case-insensitively; see email_reader for how
MAILBOX_FOLDER = "INBOX"
SINCE_DAYS = 40
SITE_DOMAIN = os.environ.get("SITE_DOMAIN", "")  # required for Phase 2, no default:
#                                                  it is the cookie-attach allowlist
PROCESSED_LABEL = "knowledge-repo/processed"
GMAIL_TOKEN_FILE = "session/gmail_token.json"
GMAIL_CREDENTIALS_FILE = "session/gmail_credentials.json"
IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993
IMAP_USERNAME = os.environ.get("IMAP_USERNAME", "")
IMAP_APP_PASSWORD = os.environ.get("IMAP_APP_PASSWORD", "")

# ── Phase 2: auth (Playwright, human-in-the-loop) ────────────────────────────
# No credentials live here or anywhere else: the human logs in, in a visible window.
LOGIN_URL = os.environ.get("LOGIN_URL", "")
STATE_FILE = "session/state.json"
SUCCESS_SELECTOR = os.environ.get("SUCCESS_SELECTOR", ".member-content")  # only when logged in
HEALTH_CHECK_URL = os.environ.get("HEALTH_CHECK_URL", "")  # required for Phase 2, no default:
#                                                            a network target hit with the session
LOGIN_URL_FRAGMENT = "/login"  # login-wall signal, not a target
LOGIN_FORM_SELECTOR = "input[name='username'], input[type='email'], input[name='log']"
#   RECOGNISES a login wall; never filled
INTERACTIVE_LOGIN = os.environ.get("INTERACTIVE_LOGIN", "auto")  # auto|always|never
MANUAL_LOGIN_TIMEOUT_MS = 300_000  # how long to wait for the human (5 min)
MANUAL_LOGIN_POLL_MS = 1_000  # how often to re-check for SUCCESS_SELECTOR
BROWSER_TYPE = "chromium"
DEBUG_SCREENSHOT_DIR = "debug/"
# headless is DERIVED, not configured: headless = not login._human_available()

# ── Phase 2: crawl ───────────────────────────────────────────────────────────
CRAWL_DELAY_MS = 2_000
PAGE_TIMEOUT_MS = 30_000

# ── Extraction (both phases) ─────────────────────────────────────────────────
MIN_WORD_COUNT = 100
ARTICLE_BODY_SELECTOR = "article, .post-content, .entry-content, main"  # Phase 2 only

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
USE_HEADER_SPLITTING = False
MIN_CHUNK_WORDS = 30  # discard shorter chunks (orphaned headings)
TOKENIZER = "cl100k_base"  # tiktoken encoding

# ── Image transcription ──────────────────────────────────────────────────────
IMAGE_CACHE_DIR = "images/"  # normalised image copies
IMAGE_CACHE_DB = "data/image_cache.db"
VISION_MAX_TOKENS = 400
MIN_IMAGE_BYTES = 5_000  # skips tracking pixels and icons
MIN_IMAGE_PIXELS = 100  # min width and height
TRANSCRIBE_TYPES = {"chart", "table", "diagram"}  # "photo" is skipped
MAX_IMAGE_MB = 5
DOWNLOAD_TIMEOUT_S = 30  # Phase 2 only
MAX_IMAGES_PER_ARTICLE = 20

# ── Text LLM (via llm_provider.py) ───────────────────────────────────────────
LLM_BACKEND = "openai_compat"  # "anthropic" | "openai" | "openai_compat"
LLM_MODEL = "qwen2.5:14b-instruct"
LLM_BASE_URL = "http://localhost:11434/v1"  # set URL for openai_compat

# ── Vision LLM (via llm_provider.py) ─────────────────────────────────────────
VISION_BACKEND = "openai_compat"  # "anthropic" | "openai" | "openai_compat"
VISION_MODEL = "qwen2.5vl:7b"
VISION_BASE_URL = "http://localhost:11434/v1"

# ── Embedding (via llm_provider.py) ──────────────────────────────────────────
EMBEDDING_BACKEND = "local"  # "local" | "openai" | "openai_compat"
LOCAL_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BASE_URL = None
EMBEDDING_DIM = 1024  # must match active model
EMBEDDING_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
#   nomic-embed-text: "search_query: " | OpenAI-style: ""
BATCH_SIZE = 64
NORMALIZE_EMBEDDINGS = True

# ── Storage ──────────────────────────────────────────────────────────────────
# Qdrant runs in one of three modes, checked in this order by storage/vector_store.py:
#   QDRANT_IN_MEMORY=true  → ephemeral, lost on exit (tests, throwaway runs)
#   QDRANT_PATH=<dir>      → embedded on-disk store, no server needed (single-user default)
#   otherwise             → connect to a Qdrant server at QDRANT_HOST:QDRANT_PORT
QDRANT_IN_MEMORY = os.environ.get("QDRANT_IN_MEMORY", "").strip().lower() in ("1", "true", "yes")
QDRANT_PATH = os.environ.get("QDRANT_PATH", "").strip() or None
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION_NAME = "knowledge_repo"
DB_PATH = "data/metadata.db"
UPSERT_BATCH_SIZE = 100  # points per upsert call

# ── Query ────────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 6
MAX_CHUNKS_PER_ARTICLE = 3
MIN_SCORE_THRESHOLD = 0.35
ENABLE_QUERY_REWRITING = False
ENABLE_HYBRID_SEARCH = False

# ── Answer generation ────────────────────────────────────────────────────────
MAX_OUTPUT_TOKENS = 1024
MAX_CONTEXT_CHUNKS = 6  # max excerpts passed to the LLM
MAX_CHUNK_CHARS = 1200  # truncate each excerpt at this length
AUTHOR_NAME = os.environ.get("AUTHOR_NAME", "the author")  # UI header, answer system
#   prompt, frontmatter author fallback

# ── App ──────────────────────────────────────────────────────────────────────
STREAMLIT_PORT = 8501
SHOW_SCORES = True  # show retrieval scores in source cards

# ── Scheduler ────────────────────────────────────────────────────────────────
SCHEDULE_DAY = 1
SCHEDULE_HOUR = 3
EMAIL_POLL_INTERVAL = 12  # hours

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE = "logs/knowledge_repo.log"
LOG_LEVEL = "INFO"


# ── Phase 2 validation (called at point of use, never at import) ─────────────

_PHASE2_REQUIRED = ("LOGIN_URL", "TRUSTED_SENDER", "SITE_DOMAIN", "HEALTH_CHECK_URL")


def require_phase2_config() -> None:
    """Raise ConfigError naming every missing or inconsistent Phase 2 setting.

    Called by scraper/login.py and inbox/email_reader.py on entry — never at import time.

    SITE_DOMAIN and HEALTH_CHECK_URL have no default because they are a cookie target and
    a network target: an install that fills in LOGIN_URL and TRUSTED_SENDER but leaves
    these blank would fail obscurely — or, if they carried a default, health-check the
    wrong host with a live session. The domain cross-check catches the subtler version —
    everything filled in, but not all for the same site.
    """
    missing = [name for name in _PHASE2_REQUIRED if not globals().get(name)]
    if missing:
        raise ConfigError(f"Phase 2 requires these .env settings: {', '.join(missing)}")
    for name in ("LOGIN_URL", "HEALTH_CHECK_URL"):
        if SITE_DOMAIN not in globals()[name]:
            raise ConfigError(
                f"{name} is not on SITE_DOMAIN ({SITE_DOMAIN}) — "
                "check every Phase 2 URL points at the same site"
            )


def phase2_configured() -> bool:
    """True when Phase 2 can run.

    app.py uses this to disable the 'Check email' button with a tooltip instead of
    offering an action that can only fail.
    """
    try:
        require_phase2_config()
    except ConfigError:
        return False
    return True
