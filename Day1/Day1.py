from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Replace

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    stream = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": query.question}],
        stream=True
        )
    def gen():
        for chunk in stream:
            if chunk.type == "response.output_text.delta":
                yield chunk.delta

    return StreamingResponse(gen(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)