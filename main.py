from data_loaders.s3_data_loader import S3BatchDownloader
from pypdf import PdfReader
import tiktoken
import numpy as np
from openai import OpenAI
import os

S3_BUCKET = "pinpointmigration2"
OPENAI_EMBED_MODEL = "text-embedding-3-small"  # cost-efficient (1536 dims). Use -3-large for higher quality (3072 dims).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  

openai_client = OpenAI(api_key=OPENAI_API_KEY)  

def run_pipeline():
    # print("Running the Vector RAG Pipeline...")
    # data_loader = S3BatchDownloader(bucket=S3_BUCKET, dest_dir="data/", prefix=None, chunk_size=5, state_path=".s3_progress.json")
    # pdf_paths = data_loader.get_next_chunk()
    # print(f"Downloaded {file_paths} files from S3 bucket '{S3_BUCKET}' to 'data/' directory.")

    pdf_paths = ['/Users/clowman/Documents/Code/VectorRAGPipeline/data/(Sunday) Olubunmi Ogundipe_9807831.zip.d/converted_cv.pdf']
    for pdf_path in pdf_paths:
        vector_embeddings = get_vector_embeddings(pdf_path)

def get_vector_embeddings(pdf_path):
    full_text = pdf_to_text(pdf_path)
    print(f"PDF Text: {full_text[:100]}...")  # Print first 100 characters for brevity
    chunks = chunk_text_by_tokens(full_text, max_tokens=800, overlap=100)

    print(f"Chunked into {len(chunks)} pieces.")
    vector_embeddings = get_vector_embeddings_of_chunks(chunks)
    print(f"Vector embeddings: {vector_embeddings}")
    return vector_embeddings

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

def get_vector_embeddings_of_chunks(texts):
    resp = openai_client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype="float32")

if __name__ == "__main__":
    run_pipeline()

