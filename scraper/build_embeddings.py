import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -----------------------------------
# STEP 1 — Load CSV File
# -----------------------------------

df = pd.read_csv("shl_assessments.csv")

print("CSV Loaded Successfully\n")

print("Columns Found:")

print(df.columns)

# -----------------------------------
# STEP 2 — Fill Missing Values
# -----------------------------------

df["title"] = df["title"].fillna("")

df["description"] = df["description"].fillna("")

# -----------------------------------
# STEP 3 — Combine Text
# -----------------------------------

df["combined_text"] = (
    df["title"] + " " +
    df["description"]
)

print("\nSample Combined Text:\n")

print(df["combined_text"].head())

# -----------------------------------
# STEP 4 — Load Embedding Model
# -----------------------------------

print("\nLoading Embedding Model...\n")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------------
# STEP 5 — Generate Embeddings
# -----------------------------------

print("Generating Embeddings...\n")

embeddings = model.encode(
    df["combined_text"].tolist(),
    show_progress_bar=True
)

# Convert embeddings to numpy array
embeddings = np.array(embeddings)

print("\nEmbeddings Shape:")

print(embeddings.shape)

# -----------------------------------
# STEP 6 — Create FAISS Index
# -----------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("\nFAISS Index Created Successfully")

# -----------------------------------
# STEP 7 — Save FAISS Index
# -----------------------------------

faiss.write_index(
    index,
    "shl_index.faiss"
)

print("\nFAISS Index Saved")

# -----------------------------------
# STEP 8 — Save Cleaned CSV
# -----------------------------------

df.to_csv(
    "cleaned_shl_data.csv",
    index=False
)

print("\nCleaned CSV Saved")

# -----------------------------------
# STEP 9 — Finished
# -----------------------------------

print("\nEmbeddings Built Successfully")