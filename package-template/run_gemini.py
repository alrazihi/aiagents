import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()  # Loads .env in current folder
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in your .env file")

# Configure API
genai.configure(api_key=api_key)

# Create model
model = genai.GenerativeModel("gemini-2.5-flash")

# Request content
response = model.generate_content("Explain how AI works in a few words")

print(response.text)  # Should work in most versions
