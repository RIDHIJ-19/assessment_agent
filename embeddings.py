import faiss
import pickle
import numpy as np

from load_catalog import assessments
from onnx_embedder import ONNXEmbedder


model = ONNXEmbedder("onnx_model")


def create_text(a):
    return f"""
Assessment Name:
{a['name']}

Description:
{a['description']}

Job Levels:
{', '.join(a['job_levels'])}

Skills:
{', '.join(a['keys'])}
""".strip()


documents = [create_text(a) for a in assessments]

print("Generating embeddings...")

embeddings = model.encode(documents)

# ✅ CRITICAL: FAISS requires float32
embeddings = np.array(embeddings, dtype="float32")

print("Embeddings shape:", embeddings.shape)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "faiss_index.bin")

with open("metadata.pkl", "wb") as f:
    pickle.dump(assessments, f)

print("FAISS index created")