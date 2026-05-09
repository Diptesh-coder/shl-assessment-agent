import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from dotenv import load_dotenv
import os

# -----------------------------------
# Load Data
# -----------------------------------

df = pd.read_csv(
    "scraper/cleaned_shl_data.csv"
)

index = faiss.read_index(
    "scraper/shl_index.faiss"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------------
# Initialize Groq
# -----------------------------------

load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# -----------------------------------
# Streamlit Page Config
# -----------------------------------

st.set_page_config(
    page_title="SHL Assessment Recommendation System",
    layout="wide"
)

# -----------------------------------
# Title
# -----------------------------------

st.title("SHL Assessment Recommendation System")

st.write(
    "Enter job requirements or skills to get matching SHL assessments."
)

# -----------------------------------
# User Input
# -----------------------------------

query = st.text_input(
    "Enter assessment requirement"
)

# -----------------------------------
# Search Button
# -----------------------------------

if st.button("Search"):

    if query:

        # -----------------------------------
        # Convert Query to Embedding
        # -----------------------------------

        query_embedding = model.encode([query])

        # -----------------------------------
        # Search FAISS Index
        # -----------------------------------

        distances, indices = index.search(
            query_embedding,
            5
        )

        # -----------------------------------
        # Build Context for LLM
        # -----------------------------------

        results_text = ""

        for i in indices[0]:

            results_text += f"""

            Title: {df.iloc[i]['title']}

            Description: {df.iloc[i]['description']}

            URL: {df.iloc[i]['url']}

            """

        # -----------------------------------
        # Generate AI Recommendation
        # -----------------------------------

        response = client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=[

                {
                    "role": "system",

                    "content": (
                        "You are an SHL assessment "
                        "recommendation assistant."
                    )
                },

                {
                    "role": "user",

                    "content": f"""

                    User Query:
                    {query}

                    Matching Assessments:
                    {results_text}

                    Recommend the best assessments
                    with explanation.

                    """
                }
            ]
        )

        answer = response.choices[0].message.content

        # -----------------------------------
        # Display AI Recommendation
        # -----------------------------------

        st.subheader("AI Recommendation")

        st.write(answer)

        # -----------------------------------
        # Display Retrieved Assessments
        # -----------------------------------

        st.subheader("Top Matching Assessments")

        for i in indices[0]:

            st.markdown("---")

            st.subheader(
                df.iloc[i]["title"]
            )

            st.write(
                df.iloc[i]["description"]
            )

            st.markdown(
                f"[Open Assessment]({df.iloc[i]['url']})"
            )

    else:

        st.warning(
            "Please enter a query."
        )
