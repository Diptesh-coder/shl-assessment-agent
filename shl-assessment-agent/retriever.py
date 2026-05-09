import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# -----------------------------------
# STEP 1 — Load CSV Data
# -----------------------------------

df = pd.read_csv("scraper/cleaned_shl_data.csv")

print("CSV Loaded Successfully")

# -----------------------------------
# STEP 2 — Load FAISS Index
# -----------------------------------

index = faiss.read_index("scraper/shl_index.faiss")

print("FAISS Index Loaded Successfully")

# -----------------------------------
# STEP 3 — Load Embedding Model
# -----------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding Model Loaded")

# -----------------------------------
# STEP 4 — User Query Input
# -----------------------------------

query = input("\nEnter your assessment requirement: ")

# -----------------------------------
# STEP 5 — Convert Query to Embedding
# -----------------------------------

query_embedding = model.encode([query])

# -----------------------------------
# STEP 6 — Search Similar Assessments
# -----------------------------------

distances, indices = index.search(query_embedding, 5)

# -----------------------------------
# STEP 7 — Display Results
# -----------------------------------

print("\nTop Matching Assessments:\n")

for i in indices[0]:

    print("Title:")
    print(df.iloc[i]["title"])

    print("\nDescription:")
    print(df.iloc[i]["description"])

    print("\nURL:")
    print(df.iloc[i]["url"])

    print("\n" + "-" * 60 + "\n")