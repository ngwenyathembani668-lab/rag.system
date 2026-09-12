import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError(
        "No API key found. Add GEMINI_API_KEY or GOOGLE_API_KEY to your .env file."
    )

print("Gemini API key loaded successfully into memory.")

# Example usage:
# from google import genai
# client = genai.Client(api_key=api_key)
