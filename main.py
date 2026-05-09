from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import pandas as pd
import faiss
import os

load_dotenv()

app = FastAPI()

df = pd.read_csv("scraper/cleaned_shl_data.csv")
index = faiss.read_index("scraper/shl_index.faiss")
model = SentenceTransformer("all-MiniLM-L6-v2")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class ChatRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    query = request.query

    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, 5)

    recommendations = []
    context = ""

    for i in indices[0]:
        item = {
            "name": df.iloc[i]["title"],
            "url": df.iloc[i]["url"],
            "description": df.iloc[i]["description"]
        }

        recommendations.append(item)

        context += f"""
Name: {item['name']}
Description: {item['description']}
URL: {item['url']}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are an SHL assessment recommendation assistant. Only recommend SHL assessments from the provided context."
            },
            {
                "role": "user",
                "content": f"""
User query:
{query}

Available SHL assessments:
{context}

Give a short recommendation explanation.
"""
            }
        ]
    )

    answer = response.choices[0].message.content

    return {
        "reply": answer,
        "recommendations": recommendations,
        "end_of_conversation": False
    }