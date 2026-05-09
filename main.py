from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import re

app = FastAPI()

df = pd.read_csv("scraper/cleaned_shl_data.csv")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


@app.get("/health")
def health():
    return {"status": "ok"}


def score_row(row, query):
    text = f"{row.get('title', '')} {row.get('description', '')}".lower()
    words = re.findall(r"\w+", query.lower())
    return sum(1 for word in words if word in text)


@app.post("/chat")
def chat(request: ChatRequest):
    full_query = " ".join(
        [m.content for m in request.messages if m.role == "user"]
    )

    if len(full_query.strip()) < 10:
        return {
            "reply": "Please share the hiring role, required skills, and seniority level.",
            "recommendations": [],
            "end_of_conversation": False
        }

    temp = df.copy()
    temp["score"] = temp.apply(
        lambda row: score_row(row, full_query),
        axis=1
    )

    temp = temp.sort_values(
        "score",
        ascending=False
    ).head(5)

    recommendations = []

    for _, row in temp.iterrows():
        recommendations.append({
            "name": row["title"],
            "url": row["url"],
            "test_type": "General"
        })

    return {
        "reply": "Here are the most relevant SHL assessments based on your requirement.",
        "recommendations": recommendations,
        "end_of_conversation": False
    }
