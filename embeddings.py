from sentence_transformers import SentenceTransformer
import faiss
import pickle

from load_catalog import assessments


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


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
    """



documents = [
    create_text(a)
    for a in assessments
]


print("Generating embeddings...")


embeddings = model.encode(
    documents,
    show_progress_bar=True
)


dimension = embeddings.shape[1]


index = faiss.IndexFlatL2(
    dimension
)


index.add(
    embeddings
)


faiss.write_index(
    index,
    "faiss_index.bin"
)


with open(
    "metadata.pkl",
    "wb"
) as f:

    pickle.dump(
        assessments,
        f
    )


print("FAISS index created")