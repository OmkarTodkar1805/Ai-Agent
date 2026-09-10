"""Document Q&A agent: token-aware chunking, vector retrieval, and a search loop."""

from pathlib import Path

from dotenv import load_dotenv

__version__ = "0.1.0"

# Load .env at package import, before any submodule reads the environment:
# provider model names are resolved when providers.py is first imported.
# load_dotenv does not overwrite variables that are already set, so secrets
# injected by a host such as Hugging Face Spaces take precedence over a file.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

if _ENV_FILE.is_file():
    load_dotenv(_ENV_FILE)
else:
    load_dotenv()
