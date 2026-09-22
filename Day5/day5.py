from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
oa = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = oa.chat.completions.create(
    model="gpt-5-mini",
    messages=[
        {
            "role": "user",
            "content": "Hello, world!"
        }
    ]
)
print(response)

r = oa.responses.create(
    model="gpt-5-mini",
    input="Hello, world!"
)

print(r)

from anthropic import Anthropic
an = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
response_anthropic = an.messages.create(
    model="claude-v1",
    messages=[
        {
            "role": "user",
            "content": "Hello, world!"
        }
    ]
)
print(response_anthropic)

from google import generativeai as google_ai
import os
google_ai.configure(api_key=os.getenv("GEMINI_API_KEY"))
response_google = google_ai.chat.completions.create(
    model="chat-bison-001",
    messages=[
        {
            "role": "user",
            "content": "Hello, world!"
        }
    ]
)
print(response_google)