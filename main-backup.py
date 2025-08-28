from data_loaders.s3_data_loader import S3BatchDownloader
from pypdf import PdfReader
import tiktoken
import numpy as np
from openai import OpenAI
import os
from sentence_transformers import SentenceTransformer
import psycopg
from pgvector.psycopg import register_vector

S3_BUCKET = "pinpointmigration2"

postgres_client = psycopg.connect("postgresql://postgres:postgres@localhost:5432/hcm")
register_vector(postgres_client)  # enables passing Python lists/np arrays as vector
data_loader = S3BatchDownloader(bucket=S3_BUCKET, dest_dir="data/", prefix=None, chunk_size=5, state_path=".s3_progress.json")


# OPENAI_EMBED_MODEL = "text-embedding-3-small"  # cost-efficient (1536 dims). Use -3-large for higher quality (3072 dims).
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  
# openai_client = OpenAI(api_key=OPENAI_API_KEY)  


#TODO: Increase chunk size so that most resumes are processed in one go.

def run_pipeline():
    print("Running the Vector RAG Pipeline...")

    while not data_loader.done():
        pdfs_pulled = data_loader.get_next_chunk()
        print(f"Downloaded {len(pdfs_pulled)} files from S3 bucket '{S3_BUCKET}' to 'data/' directory.")

        for pdf_pulled in pdfs_pulled:
            pdf_path = pdf_pulled[0]
            s3url = pdf_pulled[1]

            pdf_chunks = get_pdf_chunks(pdf_path)
            for chunk in pdf_chunks:
                # print(f"Processing chunk: {chunk[:100]}...")  # Print first 100 characters for brevity
                vector_embedding = get_vector_embeddings_of_chunks_local(chunk)
                # print(f"Vector embedding shape: {vector_embedding.shape}")
                # print(f"Vector embedding sample: {vector_embedding[:5]}")

                postgres_client.execute(
                    "INSERT INTO resumes (content, source, embedding) VALUES (%s, %s, %s)", 
                    (chunk, s3url, vector_embedding)
                )
                postgres_client.commit()


def get_pdf_chunks(pdf_path):
    full_text = pdf_to_text(pdf_path)
    # print(f"PDF Text: {full_text[:100]}...")  # Print first 100 characters for brevity
    chunks = chunk_text_by_tokens(full_text, max_tokens=30000, overlap=100)
    # print(f"Chunked into {len(chunks)} pieces.")
    return chunks

# def get_vector_embeddings(pdf_path):
#     full_text = pdf_to_text(pdf_path)
#     print(f"PDF Text: {full_text[:100]}...")  # Print first 100 characters for brevity
#     chunks = chunk_text_by_tokens(full_text, max_tokens=800, overlap=100)

#     print(f"Chunked into {len(chunks)} pieces.")
#     vector_embeddings = get_vector_embeddings_of_chunks_local(chunks)
#     print(f"Vector embeddings: {vector_embeddings}")
#     return vector_embeddings

def pdf_to_text(path):
    reader = PdfReader(path)
    pages = []
    for i, p in enumerate(reader.pages):
        txt = p.extract_text() or ""
        pages.append(f"[page {i+1}]\n{txt.strip()}")
    return "\n\n".join(pages)

def chunk_text_by_tokens(text, max_tokens=800, overlap=100, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    toks = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(toks):
        end = min(start + max_tokens, len(toks))
        chunk = encoding.decode(toks[start:end])
        chunks.append(chunk)
        if end == len(toks): break
        start = end - overlap  # slide with overlap
    return chunks

# def get_vector_embeddings_of_chunks_openai(texts):
#     resp = openai_client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
#     return np.array([d.embedding for d in resp.data], dtype="float32")

def get_vector_embeddings_of_chunks_local(text):
    embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    embedding = embedding_model.encode(text)
    print(f"embeddings: {embedding}")
    print(f"embedding shape: {embedding.shape}")
    return embedding


def run_query():
    while True:
        query = input("Enter your search query: ")

        query_embedding = get_vector_embeddings_of_chunks_local(query)
        print(f"Query embedding: {query_embedding}")

        results = postgres_client.execute(
            # Maybe changing how we search can help?
            "SELECT id, content, source, embedding <=> %s AS cosine_dist FROM resumes ORDER BY embedding <-> %s LIMIT 5",
            (query_embedding, query_embedding)
        ).fetchall()

        for result in results:
            print(f"Source: {result[2]}, Distance: {result[3]}")
            # print(f"ID: {result[0]}, Content: {result[1][:100]}..., Source: {result[2]}, Distance: {result[3]}")


if __name__ == "__main__":
    # run_pipeline()
    run_query()

