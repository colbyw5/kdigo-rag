"""Load .env before any test module runs, so integration tests relying on
ANTHROPIC_API_KEY / VOYAGE_API_KEY see them the same way the application does.
"""

from dotenv import load_dotenv

load_dotenv()
