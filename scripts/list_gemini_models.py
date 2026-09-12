"""
list_gemini_models.py
----------------------
Diagnostic: lists the Gemini models actually available to your API key, so
you can pick one with a workable free-tier quota instead of guessing model
names. Run:

    python scripts/list_gemini_models.py
"""
import os
import sys
sys.path.insert(0, ".")
from dotenv import load_dotenv

load_dotenv()

from google import genai

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("GEMINI_API_KEY not set in .env")
    sys.exit(1)

client = genai.Client(api_key=api_key)
print("Models available to your key:\n")
for m in client.models.list():
    name = getattr(m, "name", str(m))
    print(f"- {name}")
