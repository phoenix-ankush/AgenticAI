from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Replace

class EmailAnalysis(BaseModel):
    senderIntent: str
    urgency: str
    actionItems: list[str]
    summary: str

class Query(BaseModel):
    emailText: str

@app.post("/analyze_email")
def ask_question(query: Query):
    stream = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract structured info from the email."},
            {"role": "user", "content": query.emailText}
            ],
        response_format=EmailAnalysis
        )
    result = stream.choices[0].message.parsed
    if result is None:
        raise ValueError("Failed to parse the response into EmailAnalysis format.")
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)